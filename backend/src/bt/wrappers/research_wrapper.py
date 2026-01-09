"""
ResearchBTWrapper - Bridges BT-based Research to SSE streaming interface.

This wrapper implements the same interface as the original ResearchOrchestrator
but uses the behavior tree runtime internally. It provides:

1. run_research() that returns research state
2. run_research_streaming() that yields progress updates
3. Streaming bridge to yield SSE chunks
4. Vault persistence support

Migration from: backend/src/services/research/orchestrator.py
Target: Drop-in replacement that uses BT runtime

Part of the BT Universal Runtime (spec 019).
Tasks covered: Phase 6.2 Subtree Integration
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

if TYPE_CHECKING:
    from ..core.tree import BehaviorTree

logger = logging.getLogger(__name__)


# =============================================================================
# Progress Model (matching ResearchProgress)
# =============================================================================


class ResearchProgressChunk(BaseModel):
    """Progress update for research streaming."""

    research_id: str
    status: str
    phase: str
    progress_pct: float = 0
    sources_found: int = 0
    current_subtopic: Optional[str] = None
    message: Optional[str] = None


class ResearchCompleteChunk(BaseModel):
    """Completion chunk with full results."""

    research_id: str
    status: str
    report: Optional[Dict[str, Any]] = None
    vault_path: Optional[str] = None
    sources_found: int = 0
    error: Optional[str] = None


# =============================================================================
# ResearchBTWrapper
# =============================================================================


class ResearchBTWrapper:
    """Wrapper that runs Research via behavior tree runtime.

    Provides the same async generator interface as ResearchOrchestrator
    but executes via the BT runtime. Supports:

    - Blocking research via run_research()
    - Streaming research via run_research_streaming()
    - Context management
    - Vault persistence
    - Progress tracking

    Example:
        >>> wrapper = ResearchBTWrapper(user_id="user1", vault_path="/path/to/vault")
        >>> async for progress in wrapper.run_research_streaming(query="What is AI?"):
        ...     print(progress.phase, progress.progress_pct)
    """

    # Path to research.lua tree definition
    TREE_PATH = Path(__file__).parent.parent / "trees" / "research.lua"

    def __init__(
        self,
        user_id: str,
        vault_path: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        """Initialize the Research BT wrapper.

        Args:
            user_id: User ID for context scoping.
            vault_path: Optional path to user vault for persistence.
            project_id: Optional project ID for context.
        """
        self._user_id = user_id
        self._vault_path = vault_path
        self._project_id = project_id

        # Tree runtime components
        self._registry: Optional[TreeRegistry] = None
        self._tree: Optional["BehaviorTree"] = None
        self._blackboard: Optional[TypedBlackboard] = None
        self._ctx: Optional[TickContext] = None

        # Cancellation
        self._cancelled = False

    # =========================================================================
    # Public Interface
    # =========================================================================

    async def run_research(
        self,
        query: str,
        depth: str = "standard",
        save_to_vault: bool = True,
        max_sources: int = 10,
    ) -> Dict[str, Any]:
        """Run complete research workflow and return final state.

        Args:
            query: Research question/query text.
            depth: Research depth (quick/standard/thorough).
            save_to_vault: Whether to persist to vault.
            max_sources: Maximum sources to gather.

        Returns:
            Dict with research results including report.
        """
        result = None

        async for chunk in self.run_research_streaming(
            query=query,
            depth=depth,
            save_to_vault=save_to_vault,
            max_sources=max_sources,
        ):
            # Keep track of last complete chunk
            if isinstance(chunk, ResearchCompleteChunk):
                result = {
                    "research_id": chunk.research_id,
                    "status": chunk.status,
                    "report": chunk.report,
                    "vault_path": chunk.vault_path,
                    "sources_found": chunk.sources_found,
                    "error": chunk.error,
                }

        if result is None:
            # Return blackboard state if no complete chunk
            result = self._get_final_state()

        return result

    async def run_research_streaming(
        self,
        query: str,
        depth: str = "standard",
        save_to_vault: bool = True,
        max_sources: int = 10,
    ) -> AsyncGenerator[Union[ResearchProgressChunk, ResearchCompleteChunk], None]:
        """Run research with progress streaming.

        Args:
            query: Research question/query text.
            depth: Research depth (quick/standard/thorough).
            save_to_vault: Whether to persist to vault.
            max_sources: Maximum sources to gather.

        Yields:
            Progress and completion chunks.
        """
        try:
            # Load tree if not already loaded
            await self._ensure_tree_loaded()

            # Initialize blackboard with research parameters
            self._init_blackboard(query, depth, save_to_vault, max_sources)

            # Create tick context
            self._ctx = TickContext(
                blackboard=self._blackboard,
                tick_budget=10000,  # High budget for full research execution
                trace_enabled=logger.isEnabledFor(logging.DEBUG),
            )

            # Yield initial progress
            yield ResearchProgressChunk(
                research_id=self._get_research_id() or "pending",
                status="starting",
                phase="initializing",
                progress_pct=0,
                message="Starting research...",
            )

            # Run tree until completion
            async for chunk in self._run_tree():
                yield chunk

        except asyncio.CancelledError:
            logger.info("Research BT cancelled")
            yield ResearchCompleteChunk(
                research_id=self._get_research_id() or "unknown",
                status="cancelled",
                error="Research was cancelled",
            )
        except Exception as e:
            logger.exception(f"Research BT failed: {e}")
            yield ResearchCompleteChunk(
                research_id=self._get_research_id() or "unknown",
                status="failed",
                error=str(e),
            )

    def cancel(self) -> None:
        """Cancel the current research.

        Matches ResearchOrchestrator interface for cancellation.
        """
        logger.info(f"Cancelling Research BT for user {self._user_id}")
        self._cancelled = True

        if self._ctx:
            self._ctx.request_cancellation("user_request")

    # =========================================================================
    # Subtree Invocation from Oracle
    # =========================================================================

    @classmethod
    async def invoke_from_oracle(
        cls,
        ctx: TickContext,
        query: str,
        depth: str = "standard",
        save_to_vault: bool = True,
    ) -> RunStatus:
        """Invoke research subtree from Oracle agent.

        This is the entry point when Oracle calls research as a tool/subtree.

        Args:
            ctx: The Oracle's tick context.
            query: Research query.
            depth: Research depth.
            save_to_vault: Whether to save to vault.

        Returns:
            RunStatus indicating completion.
        """
        bb = ctx.blackboard
        if bb is None:
            return RunStatus.FAILURE

        user_id = bb._lookup("user_id") or ""
        vault_path = bb._lookup("vault_path")
        project_id = bb._lookup("project_id")

        wrapper = cls(
            user_id=user_id,
            vault_path=vault_path,
            project_id=project_id,
        )

        try:
            result = await wrapper.run_research(
                query=query,
                depth=depth,
                save_to_vault=save_to_vault,
            )

            # Store result in parent blackboard
            bb._data["research_result"] = result
            bb._writes.add("research_result")

            if result.get("status") == "completed":
                return RunStatus.SUCCESS
            else:
                return RunStatus.FAILURE

        except Exception as e:
            logger.error(f"Research subtree failed: {e}")
            bb._data["research_error"] = str(e)
            bb._writes.add("research_error")
            return RunStatus.FAILURE

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _ensure_tree_loaded(self) -> None:
        """Load the research tree if not already loaded."""
        if self._tree is not None:
            return

        # Initialize registry with tree directory
        self._registry = TreeRegistry(tree_dir=self.TREE_PATH.parent)

        # Load tree definition from Lua file
        if not self.TREE_PATH.exists():
            raise FileNotFoundError(
                f"Research tree definition not found: {self.TREE_PATH}"
            )

        loader = TreeLoader()
        tree_def = loader.load(self.TREE_PATH)

        # Build executable tree from definition
        builder = TreeBuilder(registry=self._registry)
        self._tree = builder.build(tree_def)

        logger.info(f"Loaded research tree: {self._tree.id}")

    def _init_blackboard(
        self,
        query: str,
        depth: str,
        save_to_vault: bool,
        max_sources: int,
    ) -> None:
        """Initialize blackboard with research parameters."""
        self._blackboard = TypedBlackboard(scope_name="research")

        # Set input parameters
        self._blackboard.set("query", query)
        self._blackboard.set("user_id", self._user_id)
        self._blackboard.set("depth", depth)
        self._blackboard.set("save_to_vault", save_to_vault)
        self._blackboard.set("max_sources", max_sources)

        if self._vault_path:
            self._blackboard.set("vault_path", self._vault_path)
        if self._project_id:
            self._blackboard.set("project_id", self._project_id)

        # Initialize pending chunks list
        self._blackboard.set("_pending_chunks", [])

    async def _run_tree(self) -> AsyncGenerator[Union[ResearchProgressChunk, ResearchCompleteChunk], None]:
        """Run the behavior tree and yield chunks.

        This method ticks the tree repeatedly until completion,
        yielding any pending chunks between ticks.
        """
        if self._tree is None or self._ctx is None:
            logger.error("Tree or context not initialized")
            return

        status = RunStatus.RUNNING
        tick_count = 0
        max_ticks = 50000  # Safety limit for research (higher than Oracle)

        while status == RunStatus.RUNNING and tick_count < max_ticks:
            # Check cancellation
            if self._cancelled:
                self._ctx.request_cancellation("user_request")
                yield ResearchCompleteChunk(
                    research_id=self._get_research_id() or "unknown",
                    status="cancelled",
                    error="Research was cancelled",
                )
                return

            # Tick the tree
            try:
                status = self._tree.tick(self._ctx)
                tick_count += 1
                self._ctx.increment_tick()
            except Exception as e:
                logger.exception(f"Tree tick failed: {e}")
                yield ResearchCompleteChunk(
                    research_id=self._get_research_id() or "unknown",
                    status="failed",
                    error=f"Tree error: {e}",
                )
                return

            # Yield any pending chunks
            async for chunk in self._drain_pending_chunks():
                yield chunk

            # Small delay to allow async operations
            if status == RunStatus.RUNNING:
                await asyncio.sleep(0.01)

        # Final status handling
        if tick_count >= max_ticks:
            logger.error("Tree execution exceeded max ticks")
            yield ResearchCompleteChunk(
                research_id=self._get_research_id() or "unknown",
                status="failed",
                error="Research timed out (max ticks exceeded)",
            )
        elif status == RunStatus.FAILURE:
            logger.warning("Tree completed with FAILURE status")
            # Drain any remaining chunks
            async for chunk in self._drain_pending_chunks():
                yield chunk

            # Yield failure completion
            yield ResearchCompleteChunk(
                research_id=self._get_research_id() or "unknown",
                status="failed",
                error=self._get_error() or "Research failed",
            )
        else:
            logger.debug(f"Tree completed with status: {status}")
            # Drain any remaining chunks
            async for chunk in self._drain_pending_chunks():
                yield chunk

            # Yield success completion if not already yielded
            yield ResearchCompleteChunk(
                research_id=self._get_research_id() or "unknown",
                status="completed",
                report=self._get_report(),
                vault_path=self._get_vault_path(),
                sources_found=self._get_sources_found(),
            )

    async def _drain_pending_chunks(
        self,
    ) -> AsyncGenerator[Union[ResearchProgressChunk, ResearchCompleteChunk], None]:
        """Drain pending chunks from blackboard and yield them."""
        if self._blackboard is None:
            return

        chunks = self._blackboard.get("_pending_chunks") or []

        if not chunks:
            return

        # Clear the list
        self._blackboard.set("_pending_chunks", [])

        for raw_chunk in chunks:
            chunk = self._convert_chunk(raw_chunk)
            if chunk:
                yield chunk

    def _convert_chunk(
        self,
        raw: Dict[str, Any],
    ) -> Optional[Union[ResearchProgressChunk, ResearchCompleteChunk]]:
        """Convert raw chunk dict to typed chunk model."""
        try:
            chunk_type = raw.get("type", "unknown")

            if chunk_type == "research_progress":
                return ResearchProgressChunk(
                    research_id=raw.get("research_id", "unknown"),
                    status="in_progress",
                    phase=raw.get("phase", "unknown"),
                    progress_pct=raw.get("pct", 0),
                    sources_found=raw.get("sources_found", 0),
                    message=raw.get("message"),
                )
            elif chunk_type == "research_complete":
                return ResearchCompleteChunk(
                    research_id=raw.get("research_id", "unknown"),
                    status=raw.get("status", "completed"),
                    vault_path=raw.get("vault_path"),
                    sources_found=raw.get("sources_found", 0),
                )
            else:
                # Unknown type - skip
                logger.debug(f"Unknown research chunk type: {chunk_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to convert chunk: {e}")
            return None

    def _get_research_id(self) -> Optional[str]:
        """Get research ID from blackboard."""
        if self._blackboard:
            return self._blackboard._lookup("research_id")
        return None

    def _get_report(self) -> Optional[Dict[str, Any]]:
        """Get report from blackboard."""
        if self._blackboard:
            return self._blackboard._lookup("report")
        return None

    def _get_vault_path(self) -> Optional[str]:
        """Get vault path from blackboard."""
        if self._blackboard:
            return self._blackboard._lookup("output_vault_path")
        return None

    def _get_sources_found(self) -> int:
        """Get sources count from blackboard."""
        if self._blackboard:
            return self._blackboard._lookup("sources_found") or 0
        return 0

    def _get_error(self) -> Optional[str]:
        """Get error from blackboard."""
        if self._blackboard:
            return self._blackboard._lookup("error")
        return None

    def _get_final_state(self) -> Dict[str, Any]:
        """Get final state from blackboard."""
        if self._blackboard:
            return {
                "research_id": self._get_research_id(),
                "status": self._blackboard._lookup("status") or "unknown",
                "report": self._get_report(),
                "vault_path": self._get_vault_path(),
                "sources_found": self._get_sources_found(),
                "error": self._get_error(),
            }
        return {
            "research_id": None,
            "status": "failed",
            "error": "No blackboard available",
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_research_bt_wrapper(
    user_id: str,
    vault_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> ResearchBTWrapper:
    """Create a ResearchBTWrapper instance.

    Factory function for dependency injection and testing.

    Args:
        user_id: User ID for context scoping.
        vault_path: Optional path to user vault.
        project_id: Optional project ID.

    Returns:
        Configured ResearchBTWrapper instance.
    """
    return ResearchBTWrapper(
        user_id=user_id,
        vault_path=vault_path,
        project_id=project_id,
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "ResearchBTWrapper",
    "ResearchProgressChunk",
    "ResearchCompleteChunk",
    "create_research_bt_wrapper",
]
