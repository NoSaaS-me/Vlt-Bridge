"""Agent SDK Session Manager — manages Claude Agent SDK sessions."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

logger = logging.getLogger("vlt.daemon.sdk_manager")


@dataclass
class AgentSDKSession:
    """Active Agent SDK session state."""
    session_id: str
    cwd: str
    client: ClaudeSDKClient
    status: str = "idle"  # idle/thinking/executing/dead
    model: str | None = None
    cost_usd: float = 0.0
    turn_count: int = 0
    _reader_task: asyncio.Task | None = field(default=None, repr=False)


class SDKSessionManager:
    """Manages Claude Agent SDK sessions with environment isolation.

    Each session gets:
    - Separate ANTHROPIC_API_KEY (from daemon settings)
    - setting_sources=[] (no config inheritance)
    - Explicit MCP server config
    - bypassPermissions mode
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        mcp_config: dict[str, Any] | None = None,
        on_status_change: Any = None,  # callback(session_id, status)
        on_message: Any = None,  # callback(session_id, message_dict)
        on_result: Any = None,  # callback(session_id, result_message)
    ):
        self.api_key = api_key or os.environ.get("VLT_SDK_API_KEY", "")
        self.default_model = default_model
        self.mcp_config = mcp_config or {}
        self.sessions: dict[str, AgentSDKSession] = {}
        self._on_status_change = on_status_change
        self._on_message = on_message
        self._on_result = on_result

    def _build_options(
        self,
        cwd: str,
        session_id: str | None = None,
        model: str | None = None,
        resume: bool = False,
    ) -> ClaudeAgentOptions:
        """Build isolated ClaudeAgentOptions for a session."""
        env = {}
        if self.api_key:
            env["ANTHROPIC_API_KEY"] = self.api_key
        # Strip vars that cause nested session errors
        for var in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
            env.pop(var, None)

        opts = ClaudeAgentOptions(
            env=env,
            setting_sources=[],  # Full isolation
            cwd=cwd,
            permission_mode="bypassPermissions",
            model=model or self.default_model,
        )

        # MCP config — pass existing vlt MCP server
        if self.mcp_config:
            opts.mcp_servers = self.mcp_config

        # Resume existing session
        if resume and session_id:
            opts.resume = session_id

        return opts

    async def spawn(
        self,
        session_id: str,
        cwd: str,
        prompt: str,
        model: str | None = None,
    ) -> AgentSDKSession:
        """Spawn a new Agent SDK session.

        Creates a ClaudeSDKClient, connects it, sends the initial prompt,
        and starts a background reader task to process responses.
        """
        # Check if already running
        existing = self.sessions.get(session_id)
        if existing and existing.status != "dead":
            logger.info(f"Agent SDK session already active: {session_id}")
            return existing

        if not os.path.isdir(cwd):
            raise ValueError(f"CWD does not exist: {cwd}")

        opts = self._build_options(cwd, session_id, model, resume=False)
        client = ClaudeSDKClient(options=opts)

        session = AgentSDKSession(
            session_id=session_id,
            cwd=cwd,
            client=client,
            model=model or self.default_model,
        )
        self.sessions[session_id] = session

        try:
            await client.connect()
            logger.info(f"Agent SDK session connected: {session_id}")

            # Start background reader
            session._reader_task = asyncio.create_task(
                self._read_responses(session)
            )

            # Send initial prompt
            await client.query(prompt)
            session.status = "thinking"
            await self._notify_status(session_id, "thinking")

        except Exception as e:
            logger.error(f"Failed to spawn Agent SDK session {session_id}: {e}")
            session.status = "dead"
            self.sessions.pop(session_id, None)
            raise

        return session

    async def resume(
        self,
        session_id: str,
        cwd: str,
        prompt: str,
        model: str | None = None,
    ) -> AgentSDKSession:
        """Resume an existing Agent SDK session by ID."""
        existing = self.sessions.get(session_id)
        if existing and existing.status != "dead":
            # Already alive — just send a new message
            await self.send_message(session_id, prompt)
            return existing

        opts = self._build_options(cwd, session_id, model, resume=True)
        client = ClaudeSDKClient(options=opts)

        session = AgentSDKSession(
            session_id=session_id,
            cwd=cwd,
            client=client,
            model=model or self.default_model,
        )
        self.sessions[session_id] = session

        try:
            await client.connect()
            session._reader_task = asyncio.create_task(
                self._read_responses(session)
            )
            await client.query(prompt)
            session.status = "thinking"
            await self._notify_status(session_id, "thinking")
        except Exception as e:
            logger.error(f"Failed to resume Agent SDK session {session_id}: {e}")
            session.status = "dead"
            self.sessions.pop(session_id, None)
            raise

        return session

    async def send_message(self, session_id: str, text: str) -> bool:
        """Send a user message to a running session."""
        session = self.sessions.get(session_id)
        if not session or session.status == "dead":
            logger.warning(f"Cannot send to dead/missing session: {session_id}")
            return False

        try:
            await session.client.query(text)
            session.status = "thinking"
            await self._notify_status(session_id, "thinking")
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {session_id}: {e}")
            return False

    async def interrupt(self, session_id: str) -> bool:
        """Interrupt a running session."""
        session = self.sessions.get(session_id)
        if not session or session.status == "dead":
            return False
        try:
            await session.client.interrupt()
            return True
        except Exception as e:
            logger.error(f"Failed to interrupt {session_id}: {e}")
            return False

    async def set_model(self, session_id: str, model: str) -> bool:
        """Change model for a running session."""
        session = self.sessions.get(session_id)
        if not session or session.status == "dead":
            return False
        try:
            await session.client.set_model(model)
            session.model = model
            return True
        except Exception as e:
            logger.error(f"Failed to set model for {session_id}: {e}")
            return False

    async def dismiss(self, session_id: str) -> None:
        """Clean shutdown of a session."""
        session = self.sessions.pop(session_id, None)
        if not session:
            return

        session.status = "dead"
        await self._notify_status(session_id, "dead")

        if session._reader_task and not session._reader_task.done():
            session._reader_task.cancel()
            try:
                await session._reader_task
            except asyncio.CancelledError:
                pass

        try:
            await session.client.disconnect()
        except Exception as e:
            logger.debug(f"Error disconnecting {session_id}: {e}")

    async def shutdown_all(self) -> None:
        """Shut down all active sessions. Called during daemon shutdown."""
        session_ids = list(self.sessions.keys())
        for sid in session_ids:
            await self.dismiss(sid)

    async def _read_responses(self, session: AgentSDKSession) -> None:
        """Background task that reads Agent SDK responses and forwards to callbacks."""
        try:
            async for message in session.client.receive_messages():
                try:
                    await self._process_message(session, message)
                except Exception as e:
                    logger.error(f"Error processing message for {session.session_id}: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Reader task died for {session.session_id}: {e}")
        finally:
            if session.session_id in self.sessions:
                session.status = "idle"
                await self._notify_status(session.session_id, "idle")

    async def _process_message(self, session: AgentSDKSession, message: Any) -> None:
        """Process a single message from the Agent SDK stream."""
        if isinstance(message, AssistantMessage):
            session.status = "thinking"
            # Forward content blocks
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    session.status = "executing"
                    await self._notify_status(session.session_id, "executing")

            if self._on_message:
                await self._on_message(session.session_id, message)

        elif isinstance(message, ResultMessage):
            session.status = "idle"
            session.cost_usd += message.total_cost_usd or 0.0
            session.turn_count += message.num_turns
            await self._notify_status(session.session_id, "idle")
            if self._on_result:
                await self._on_result(session.session_id, message)

        elif isinstance(message, SystemMessage):
            if self._on_message:
                await self._on_message(session.session_id, message)

        elif isinstance(message, UserMessage):
            if self._on_message:
                await self._on_message(session.session_id, message)

    async def _notify_status(self, session_id: str, status: str) -> None:
        """Notify the daemon of a status change."""
        if self._on_status_change:
            try:
                await self._on_status_change(session_id, status)
            except Exception as e:
                logger.debug(f"Status callback error for {session_id}: {e}")
