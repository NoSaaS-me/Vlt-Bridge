"""Event Bridge — translates Agent SDK messages to daemon live-streaming format.

The daemon's existing infrastructure (WebSocket protocol, _live_stream_queues,
JSONL transcripts) expects messages in a specific format. This bridge converts
Agent SDK message types into that format so the frontend sees identical messages
regardless of whether a session is relay, subprocess-sdk, or agent-sdk.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

logger = logging.getLogger("vlt.daemon.sdk_event_bridge")


class SDKEventBridge:
    """Bridges Agent SDK events to the daemon's live streaming infrastructure.

    Responsibilities:
    1. Convert SDK messages → JSONL transcript entries (for persistence)
    2. Push entries to _live_stream_queues (for WebSocket clients)
    3. Push status updates (thinking/executing/idle/dead)
    4. Update AgentSession DB records (cost, turns, ctx_pct)
    """

    def __init__(
        self,
        live_stream_queues: dict[str, set],  # session_id → set of asyncio.Queue
        push_status_fn: Any,  # async fn(session_id, status, event_name)
        update_session_fn: Any,  # async fn(session_id, **kwargs) — updates DB
        transcript_dir: str | Path | None = None,
    ):
        self.live_stream_queues = live_stream_queues
        self._push_status = push_status_fn
        self._update_session = update_session_fn
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
        self._transcript_files: dict[str, Any] = {}  # session_id → file handle

    def _make_jsonl_entry(self, entry_type: str, **kwargs) -> dict:
        """Create a JSONL transcript entry matching Claude Code's format."""
        entry = {
            "type": entry_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        return entry

    async def on_assistant_message(self, session_id: str, message: AssistantMessage) -> None:
        """Handle AssistantMessage from Agent SDK."""
        # Build content blocks in Claude Code transcript format
        content_blocks = []
        for block in message.content:
            if isinstance(block, TextBlock):
                content_blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ThinkingBlock):
                content_blocks.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                })
            elif isinstance(block, ToolUseBlock):
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif isinstance(block, ToolResultBlock):
                content_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": block.content,
                    "is_error": block.is_error,
                })

        entry = self._make_jsonl_entry(
            "assistant",
            message={"role": "assistant", "content": content_blocks},
            model=message.model,
        )

        await self._push_entry(session_id, entry)
        await self._write_transcript(session_id, entry)

        # Determine status from content
        has_tool_use = any(isinstance(b, ToolUseBlock) for b in message.content)
        if has_tool_use:
            await self._push_status(session_id, "executing", "agent-sdk:tool_use")
        else:
            await self._push_status(session_id, "thinking", "agent-sdk:assistant")

    async def on_user_message(self, session_id: str, message: UserMessage) -> None:
        """Handle UserMessage from Agent SDK (echoed back input)."""
        content = message.content if isinstance(message.content, str) else str(message.content)
        entry = self._make_jsonl_entry(
            "user",
            message={"role": "user", "content": content},
        )
        await self._push_entry(session_id, entry)
        await self._write_transcript(session_id, entry)

    async def on_result(self, session_id: str, result: ResultMessage) -> None:
        """Handle ResultMessage — session turn complete."""
        entry = self._make_jsonl_entry(
            "result",
            subtype=result.subtype,
            is_error=result.is_error,
            duration_ms=result.duration_ms,
            num_turns=result.num_turns,
            cost_usd=result.total_cost_usd,
            session_id=result.session_id,
        )
        await self._push_entry(session_id, entry)
        await self._write_transcript(session_id, entry)
        await self._push_status(session_id, "idle", "agent-sdk:result")

        # Update DB
        if self._update_session:
            try:
                await self._update_session(
                    session_id,
                    cost_usd=result.total_cost_usd,
                    turn_count=result.num_turns,
                    status="idle",
                )
            except Exception as e:
                logger.debug(f"Failed to update session {session_id}: {e}")

    async def on_system_message(self, session_id: str, message: SystemMessage) -> None:
        """Handle SystemMessage from Agent SDK."""
        entry = self._make_jsonl_entry(
            "system",
            subtype=message.subtype,
            data=message.data,
        )
        await self._push_entry(session_id, entry)
        await self._write_transcript(session_id, entry)

    async def on_status_change(self, session_id: str, status: str) -> None:
        """Direct status change notification."""
        await self._push_status(session_id, status, f"agent-sdk:{status}")

    async def _push_entry(self, session_id: str, entry: dict) -> None:
        """Push a JSONL entry to all connected WebSocket clients."""
        queues = self.live_stream_queues.get(session_id, set())
        msg = {"type": "entry", "entry": entry}
        for q in queues:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                logger.debug(f"Queue full for {session_id}, dropping entry")

    async def _write_transcript(self, session_id: str, entry: dict) -> None:
        """Append entry to the session's JSONL transcript file."""
        if not self.transcript_dir:
            return

        transcript_path = self.transcript_dir / f"{session_id}.jsonl"
        try:
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            with open(transcript_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write transcript for {session_id}: {e}")

    async def cleanup(self, session_id: str) -> None:
        """Clean up resources for a session."""
        self._transcript_files.pop(session_id, None)
