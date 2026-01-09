"""
OracleBTWrapper - Bridges BT-based Oracle to SSE streaming interface.

This wrapper implements the same interface as the original OracleAgent
but uses the behavior tree runtime internally. It provides:

1. process_query() that returns an async generator
2. Streaming bridge to yield SSE chunks
3. Context persistence bridge
4. Shadow mode support for parallel operation

Migration from: backend/src/services/oracle_agent.py
Target: Drop-in replacement that uses BT runtime

Part of the BT Universal Runtime (spec 019).
Tasks covered: 5.2.1-5.2.4 from tasks.md
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)

from pydantic import BaseModel

from ..core.context import TickContext
from ..state.blackboard import TypedBlackboard
from ..state.base import RunStatus
from ..lua.loader import TreeLoader
from ..lua.builder import TreeBuilder
from ..lua.registry import TreeRegistry
from ..services.openrouter_client import OpenRouterClient, BTServices

if TYPE_CHECKING:
    from ..core.tree import BehaviorTree

logger = logging.getLogger(__name__)


# =============================================================================
# Chunk Models (matching oracle_agent.py OracleStreamChunk)
# =============================================================================


class OracleStreamChunk(BaseModel):
    """SSE chunk model matching existing OracleAgent interface."""

    type: str  # content, reasoning, tool_call, tool_result, error, done, etc.
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    context_id: Optional[str] = None
    accumulated_content: Optional[str] = None
    context_tokens: Optional[int] = None
    max_context_tokens: Optional[int] = None
    warning: Optional[str] = None
    severity: Optional[str] = None


# =============================================================================
# OracleBTWrapper
# =============================================================================


class OracleBTWrapper:
    """Wrapper that runs Oracle via behavior tree runtime.

    Provides the same async generator interface as the original OracleAgent
    but executes via the BT runtime. Supports:

    - Streaming responses via async generator
    - Context management (tree + legacy)
    - Tool execution
    - Budget tracking
    - ANS event emission
    - Shadow mode for comparison testing

    Example:
        >>> wrapper = OracleBTWrapper(user_id="user1")
        >>> async for chunk in wrapper.process_query("Hello"):
        ...     print(chunk.type, chunk.content)
    """

    # Path to oracle-agent.lua tree definition
    TREE_PATH = Path(__file__).parent.parent / "trees" / "oracle-agent.lua"

    def __init__(
        self,
        user_id: str,
        api_key: str,
        project_id: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize the Oracle BT wrapper.

        Args:
            user_id: User ID for context scoping.
            api_key: OpenRouter API key for LLM calls.
            project_id: Optional project ID for tool scoping.
            model: LLM model identifier.
            max_tokens: Maximum tokens for LLM response.
        """
        self._user_id = user_id
        self._api_key = api_key
        self._project_id = project_id
        self._model = model or "deepseek/deepseek-chat"
        self._max_tokens = max_tokens

        # Create LLM client for BT nodes
        self._llm_client = OpenRouterClient(api_key=api_key)
        # Note: BTServices is created after tree loading so we can include tree_registry
        self._services: Optional[BTServices] = None

        # Tree runtime components
        self._registry: Optional[TreeRegistry] = None
        self._tree: Optional["BehaviorTree"] = None
        self._blackboard: Optional[TypedBlackboard] = None
        self._ctx: Optional[TickContext] = None

        # Cancellation
        self._cancelled = False

        # Chunk collection for streaming
        self._collected_chunks: List[OracleStreamChunk] = []

    # =========================================================================
    # Public Interface
    # =========================================================================

    async def process_query(
        self,
        query: str,
        context_id: Optional[str] = None,
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """Process a query and yield streaming chunks.

        This is the main entry point, matching OracleAgent.query() interface.

        Args:
            query: User question/query text.
            context_id: Optional context ID to resume conversation.

        Yields:
            OracleStreamChunk objects for SSE streaming.
        """
        try:
            # Load tree if not already loaded
            await self._ensure_tree_loaded()

            # Initialize blackboard with query parameters
            self._init_blackboard(query, context_id)

            # Create tick context with services for LLM calls
            self._ctx = TickContext(
                blackboard=self._blackboard,
                services=self._services,  # Provides llm_client for BT.llm_call
                tick_budget=1000,  # High budget for full execution
                trace_enabled=logger.isEnabledFor(logging.DEBUG),
            )

            # Run tree until completion
            async for chunk in self._run_tree():
                yield chunk

                # Collect chunks for potential inspection
                self._collected_chunks.append(chunk)

        except asyncio.CancelledError:
            logger.info("Oracle BT query cancelled")
            yield OracleStreamChunk(
                type="error",
                error="cancelled",
                content="Request was cancelled"
            )
        except Exception as e:
            logger.exception(f"Oracle BT query failed: {e}")
            yield OracleStreamChunk(
                type="error",
                error="internal_error",
                content=str(e)
            )

    def cancel(self) -> None:
        """Cancel the current query.

        Matches OracleAgent.cancel() interface.
        """
        logger.info(f"Cancelling Oracle BT for user {self._user_id}")
        self._cancelled = True

        if self._ctx:
            self._ctx.request_cancellation("user_request")

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled

    def reset_cancellation(self) -> None:
        """Reset cancellation flag for new query."""
        self._cancelled = False

        if self._ctx:
            self._ctx.clear_cancellation()

    # =========================================================================
    # Chunk Access Interface
    # =========================================================================

    def get_collected_chunks(self) -> List[OracleStreamChunk]:
        """Get collected chunks from the last query.

        Returns:
            List of chunks from the BT execution.
        """
        return self._collected_chunks.copy()

    def clear_collected_chunks(self) -> None:
        """Clear collected chunks."""
        self._collected_chunks = []

    async def compare_with_legacy(
        self,
        legacy_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare BT execution with legacy Oracle output.

        Args:
            legacy_chunks: Chunks from legacy OracleAgent.

        Returns:
            Comparison report with discrepancies.
        """
        bt_chunks = self._collected_chunks

        report = {
            "bt_chunk_count": len(bt_chunks),
            "legacy_chunk_count": len(legacy_chunks),
            "discrepancies": [],
            "match_rate": 0.0,
        }

        # Compare chunk types
        bt_types = [c.type for c in bt_chunks]
        legacy_types = [c.get("type") for c in legacy_chunks]

        if bt_types != legacy_types:
            report["discrepancies"].append({
                "field": "chunk_types",
                "bt": bt_types,
                "legacy": legacy_types,
            })

        # Compare final content
        bt_content = ""
        legacy_content = ""

        for c in bt_chunks:
            if c.type == "done" and c.accumulated_content:
                bt_content = c.accumulated_content
                break

        for c in legacy_chunks:
            if c.get("type") == "done" and c.get("accumulated_content"):
                legacy_content = c.get("accumulated_content", "")
                break

        if bt_content != legacy_content:
            report["discrepancies"].append({
                "field": "accumulated_content",
                "bt_length": len(bt_content),
                "legacy_length": len(legacy_content),
                "bt_preview": bt_content[:200],
                "legacy_preview": legacy_content[:200],
            })

        # Calculate match rate
        matches = 0
        total = max(len(bt_chunks), len(legacy_chunks))

        if total > 0:
            for i, bt_chunk in enumerate(bt_chunks):
                if i < len(legacy_chunks):
                    legacy = legacy_chunks[i]
                    if bt_chunk.type == legacy.get("type"):
                        matches += 1

            report["match_rate"] = matches / total

        return report

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _ensure_tree_loaded(self) -> None:
        """Load the oracle-agent tree if not already loaded."""
        if self._tree is not None:
            return

        # Initialize registry with tree directory and load all trees
        # (including subtrees like agent-turn.lua, execute-tools.lua)
        # Note: validate_on_load=False because function paths are resolved at runtime
        logger.debug(f"Creating TreeRegistry for: {self.TREE_PATH.parent}")
        self._registry = TreeRegistry(
            tree_dir=self.TREE_PATH.parent,
            validate_on_load=False,
        )
        self._registry.load_all()
        logger.debug(f"Loaded trees: {list(self._registry._trees.keys()) if hasattr(self._registry, '_trees') else 'unknown'}")

        # Get the oracle-agent tree from the registry
        if not self.TREE_PATH.exists():
            raise FileNotFoundError(
                f"Oracle tree definition not found: {self.TREE_PATH}"
            )

        # The tree should now be loaded by registry.load_all()
        self._tree = self._registry.get("oracle-agent")
        if self._tree is None:
            # Fallback: load directly if not in registry
            loader = TreeLoader()
            tree_def = loader.load(self.TREE_PATH)
            builder = TreeBuilder(registry=self._registry)
            self._tree = builder.build(tree_def)

        # Create BTServices with tree_registry for lazy subtree resolution
        self._services = BTServices(
            llm_client=self._llm_client,
            tree_registry=self._registry,
        )

        logger.info(f"Loaded oracle-agent tree: {self._tree.id}")

    def _init_blackboard(
        self,
        query: str,
        context_id: Optional[str],
    ) -> None:
        """Initialize blackboard with query parameters.

        Uses a simple dict-based wrapper since TypedBlackboard requires
        schema registration which is complex for this use case.
        """
        # Create a simple dict-based blackboard wrapper
        # Initialize messages for LLMCallNode (required by messages_key validation)
        initial_messages = [
            {"role": "user", "content": query}
        ]

        self._bb_data: Dict[str, Any] = {
            "query": query,
            "user_id": self._user_id,
            "project_id": self._project_id,
            "context_id": context_id,
            "model": self._model,
            "max_tokens": self._max_tokens,
            # LLMCallNode requires messages to be present
            "messages": initial_messages,
            # Initialize other required keys
            "tool_calls": [],
            "tools": None,
            "accumulated_content": "",
            "turn": 0,
            "thinking_enabled": False,
            "_pending_chunks": [],
            # Track streaming progress to avoid duplicate content
            "_last_streamed_len": 0,
        }

        # Create a wrapper object that behaves like a blackboard
        class SimpleBlackboard:
            """Simple dict-based blackboard for BT runtime."""

            def __init__(bb_self, data: Dict[str, Any]):
                bb_self._data = data
                bb_self._reads: set = set()
                bb_self._writes: set = set()

            def get(bb_self, key: str, default: Any = None) -> Any:
                bb_self._reads.add(key)
                return bb_self._data.get(key, default)

            def set(bb_self, key: str, value: Any) -> None:
                bb_self._writes.add(key)
                bb_self._data[key] = value

            def has(bb_self, key: str) -> bool:
                return key in bb_self._data

            def delete(bb_self, key: str) -> None:
                bb_self._data.pop(key, None)

            def keys(bb_self) -> List[str]:
                return list(bb_self._data.keys())

            def clear(bb_self) -> None:
                bb_self._data.clear()

            def clear_access_tracking(bb_self) -> None:
                bb_self._reads = set()
                bb_self._writes = set()

            def get_reads(bb_self) -> set:
                return bb_self._reads

            def get_writes(bb_self) -> set:
                return bb_self._writes

            def snapshot(bb_self) -> Dict[str, Any]:
                return bb_self._data.copy()

            # Methods needed by BT actions that bypass schema validation
            def _lookup(bb_self, key: str) -> Any:
                """Get value without schema validation (for action helpers)."""
                return bb_self._data.get(key)

            def _store(bb_self, key: str, value: Any) -> None:
                """Set value without schema validation (for action helpers)."""
                bb_self._writes.add(key)
                bb_self._data[key] = value

            def create_child_scope(bb_self, scope_id: str = "") -> "SimpleBlackboard":
                """Create a child scope that inherits from this blackboard.

                For SimpleBlackboard, we just return self since we're not
                tracking scopes. Writes go to the same data dict.
                """
                # For simple blackboard, return self - no scoping needed
                return bb_self

            def register(bb_self, key: str, schema: Any = None) -> None:
                """Register a key in the blackboard (no-op for SimpleBlackboard).

                TypedBlackboard uses this for schema validation, but SimpleBlackboard
                doesn't validate schemas.
                """
                # No-op: SimpleBlackboard doesn't track schemas
                pass

            def has(bb_self, key: str) -> bool:
                """Check if a key exists in the blackboard."""
                return key in bb_self._data

        self._blackboard = SimpleBlackboard(self._bb_data)

        # Initialize pending chunks list (on wrapper, not blackboard)
        self._pending_chunks: List[Dict[str, Any]] = []

    async def _run_tree(self) -> AsyncGenerator[OracleStreamChunk, None]:
        """Run the behavior tree and yield chunks.

        This method ticks the tree repeatedly until completion,
        yielding any pending chunks between ticks.
        """
        if self._tree is None or self._ctx is None:
            logger.error("Tree or context not initialized")
            return

        status = RunStatus.RUNNING
        tick_count = 0
        max_ticks = 50000  # Safety limit (50000 ticks @ 100ms = 83 min max)

        while status == RunStatus.RUNNING and tick_count < max_ticks:
            # Check cancellation
            if self._cancelled:
                self._ctx.request_cancellation("user_request")
                yield OracleStreamChunk(
                    type="error",
                    error="cancelled"
                )
                return

            # Tick the tree
            try:
                status = self._tree.tick(self._ctx)
                tick_count += 1
                self._ctx.increment_tick()
            except Exception as e:
                logger.exception(f"Tree tick failed: {e}")
                yield OracleStreamChunk(
                    type="error",
                    error="tree_error",
                    content=str(e)
                )
                return

            # Yield any pending chunks
            async for chunk in self._drain_pending_chunks():
                yield chunk

            # Delay to allow async operations (LLM calls can take 30+ seconds)
            # Use longer sleep to reduce tick consumption while waiting
            if status == RunStatus.RUNNING:
                await asyncio.sleep(0.1)  # 100ms between ticks

        # Final status handling
        if tick_count >= max_ticks:
            logger.error("Tree execution exceeded max ticks")
            yield OracleStreamChunk(
                type="error",
                error="max_ticks_exceeded"
            )
        elif status == RunStatus.FAILURE:
            logger.warning("Tree completed with FAILURE status")
            # Drain any remaining chunks
            async for chunk in self._drain_pending_chunks():
                logger.info(f"Draining final chunk (failure): {chunk.type}")
                yield chunk
        else:
            logger.info(f"Tree completed successfully with status: {status}, ticks: {tick_count}")
            # Drain any remaining chunks
            chunks_drained = 0
            async for chunk in self._drain_pending_chunks():
                chunks_drained += 1
                logger.info(f"Draining final chunk ({chunks_drained}): {chunk.type}")
                yield chunk
            logger.info(f"Final drain complete: {chunks_drained} chunks")

    async def _drain_pending_chunks(self) -> AsyncGenerator[OracleStreamChunk, None]:
        """Drain pending chunks from blackboard and yield them.

        Actions add chunks to bb._data["_pending_chunks"] via _add_pending_chunk().
        """
        # Get chunks from blackboard, not wrapper attribute
        if self._blackboard is None:
            return

        chunks = self._blackboard._data.get("_pending_chunks", [])
        if not chunks:
            return

        # Clear the chunks list in blackboard
        self._blackboard._data["_pending_chunks"] = []

        for raw_chunk in chunks:
            chunk = self._convert_chunk(raw_chunk)
            if chunk:
                yield chunk

    def _convert_chunk(self, raw: Dict[str, Any]) -> Optional[OracleStreamChunk]:
        """Convert raw chunk dict to OracleStreamChunk model."""
        try:
            chunk_type = raw.get("type", "unknown")

            if chunk_type == "content":
                return OracleStreamChunk(
                    type="content",
                    content=raw.get("content"),
                )
            elif chunk_type == "reasoning":
                return OracleStreamChunk(
                    type="reasoning",
                    reasoning=raw.get("content"),
                )
            elif chunk_type == "tool_call":
                return OracleStreamChunk(
                    type="tool_call",
                    tool_call={
                        "call_id": raw.get("call_id"),
                        "name": raw.get("name"),
                        "status": raw.get("status"),
                    }
                )
            elif chunk_type == "tool_result":
                return OracleStreamChunk(
                    type="tool_result",
                    tool_result={
                        "call_id": raw.get("call_id"),
                        "name": raw.get("name"),
                        "status": raw.get("status"),
                        "result": raw.get("result"),
                        "error": raw.get("error"),
                    }
                )
            elif chunk_type == "error":
                return OracleStreamChunk(
                    type="error",
                    error=raw.get("error"),
                    content=raw.get("message"),
                )
            elif chunk_type == "done":
                return OracleStreamChunk(
                    type="done",
                    accumulated_content=raw.get("accumulated_content"),
                    context_id=self._blackboard.get("current_node_id") if self._blackboard else None,
                    warning=raw.get("warning"),
                )
            elif chunk_type == "context_update":
                return OracleStreamChunk(
                    type="context_update",
                    context_tokens=raw.get("context_tokens"),
                    max_context_tokens=raw.get("max_context_tokens"),
                )
            elif chunk_type == "sources":
                return OracleStreamChunk(
                    type="sources",
                    sources=raw.get("sources"),
                )
            elif chunk_type == "system":
                return OracleStreamChunk(
                    type="system",
                    content=raw.get("content"),
                    severity=raw.get("severity"),
                )
            else:
                logger.warning(f"Unknown chunk type: {chunk_type}")
                return OracleStreamChunk(
                    type=chunk_type,
                    content=str(raw),
                )

        except Exception as e:
            logger.error(f"Failed to convert chunk: {e}")
            return None

    # =========================================================================
    # Context Persistence Bridge
    # =========================================================================

    def get_context_id(self) -> Optional[str]:
        """Get current context ID for persistence.

        Returns:
            Current context node ID or None.
        """
        if self._blackboard:
            return self._blackboard._lookup("current_node_id")
        return None

    def get_tree_root_id(self) -> Optional[str]:
        """Get tree root ID for context navigation.

        Returns:
            Tree root ID or None.
        """
        if self._blackboard:
            return self._blackboard._lookup("tree_root_id")
        return None

    def get_accumulated_content(self) -> str:
        """Get accumulated response content.

        Returns:
            Full response content accumulated so far.
        """
        if self._blackboard:
            return self._blackboard._lookup("accumulated_content") or ""
        return ""

    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics.

        Returns:
            Dict with tokens_used, context_tokens, max_context_tokens.
        """
        if self._blackboard:
            return {
                "tokens_used": self._blackboard._lookup("tokens_used") or 0,
                "context_tokens": self._blackboard._lookup("context_tokens") or 0,
                "max_context_tokens": self._blackboard._lookup("max_context_tokens") or 0,
            }
        return {"tokens_used": 0, "context_tokens": 0, "max_context_tokens": 0}

    # =========================================================================
    # Signal Parsing Methods (T022)
    # =========================================================================

    def _should_parse_signal(self) -> bool:
        """Check if LLM response was received this tick and needs parsing.

        Returns True if:
        1. Blackboard exists
        2. accumulated_content has content
        3. Signal hasn't been parsed this turn yet

        Returns:
            True if signal should be parsed, False otherwise.
        """
        if not self._blackboard:
            return False

        accumulated = self._blackboard._lookup("accumulated_content")
        if not accumulated:
            return False

        already_parsed = self._blackboard._lookup("_signal_parsed_this_turn")
        return not already_parsed

    def _parse_signal_from_response(self) -> None:
        """Parse signal from LLM response and update blackboard state.

        This method is called after an LLM response is received and
        processes the signal using the signal_actions module.

        Signal processing sequence:
        1. parse_response_signal - Extract XML signal
        2. update_signal_state - Track consecutive reasons
        3. log_signal - Emit to ANS event bus
        4. strip_signal_from_response - Remove XML from content

        Part of T022: Signal parsing integration.
        """
        if not self._ctx or not self._blackboard:
            logger.debug("_parse_signal_from_response: No context/blackboard")
            return

        try:
            from ..actions.signal_actions import (
                parse_response_signal,
                update_signal_state,
                log_signal,
                strip_signal_from_response,
            )

            # Process signal in sequence
            parse_response_signal(self._ctx)
            update_signal_state(self._ctx)
            log_signal(self._ctx)
            strip_signal_from_response(self._ctx)

            # Mark as parsed to prevent double-parsing
            self._blackboard._data["_signal_parsed_this_turn"] = True
            self._blackboard._writes.add("_signal_parsed_this_turn")

            logger.debug("_parse_signal_from_response: Signal processing complete")

        except ImportError as e:
            logger.debug(f"_parse_signal_from_response: signal_actions not available: {e}")
        except Exception as e:
            logger.warning(f"_parse_signal_from_response: Error: {e}")

    def _reset_signal_parse_flag(self) -> None:
        """Reset the signal parse flag at the start of each tick.

        Called before each tree tick to allow signal parsing
        for new LLM responses.
        """
        if self._blackboard:
            self._blackboard._data["_signal_parsed_this_turn"] = False
            self._blackboard._writes.add("_signal_parsed_this_turn")

    def get_last_signal(self) -> Optional[Dict[str, Any]]:
        """Get the last parsed signal from blackboard.

        Returns:
            Signal dict or None if no signal was parsed.
        """
        if self._blackboard:
            return self._blackboard._lookup("last_signal")
        return None

    def get_signal_state(self) -> Dict[str, Any]:
        """Get the current signal state for debugging.

        Returns:
            Dict with signal tracking state.
        """
        if not self._blackboard:
            return {}

        return {
            "last_signal": self._blackboard._lookup("last_signal"),
            "signals_emitted": self._blackboard._lookup("signals_emitted") or [],
            "consecutive_same_reason": self._blackboard._lookup("consecutive_same_reason") or 0,
            "turns_without_signal": self._blackboard._lookup("turns_without_signal") or 0,
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_oracle_bt_wrapper(
    user_id: str,
    project_id: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    enable_shadow_mode: bool = False,
) -> OracleBTWrapper:
    """Create an OracleBTWrapper instance.

    Factory function for dependency injection and testing.

    Args:
        user_id: User ID for context scoping.
        project_id: Optional project ID.
        model: LLM model identifier.
        max_tokens: Maximum tokens for response.
        enable_shadow_mode: Enable parallel comparison mode.

    Returns:
        Configured OracleBTWrapper instance.
    """
    return OracleBTWrapper(
        user_id=user_id,
        project_id=project_id,
        model=model,
        max_tokens=max_tokens,
        enable_shadow_mode=enable_shadow_mode,
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "OracleBTWrapper",
    "OracleStreamChunk",
    "create_oracle_bt_wrapper",
]
