"""
TickContext - Execution context for behavior tree ticks.

Implements the TickContext from data-model.md and addresses footgun A.1
(progress tracking for stuck detection).

Tasks covered: 1.2.1-1.2.7 from tasks.md

Part of the BT Universal Runtime (spec 019).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state.blackboard import TypedBlackboard


@dataclass
class TickContext:
    """Execution context for tree ticks.

    Passed to every node on tick, providing access to:
    - State (blackboard)
    - Budget tracking
    - Path/debugging information
    - Async operation coordination
    - Cancellation support
    - Progress tracking (footgun A.1)

    From data-model.md TickContext specification:

    Attributes:
        event: The event that triggered this tick (optional, for event-driven ticks).
        blackboard: Current scope blackboard for state access.
        services: Dependency injection container (future, optional for now).
        tick_count: Number of ticks in current execution.
        tick_budget: Maximum ticks before yielding (default 1000).
        start_time: When execution started (for elapsed time tracking).
        parent_path: List of parent node IDs for debugging.
        trace_enabled: Whether to log detailed trace information.
        async_pending: Set of pending async operation IDs.
        cancellation_requested: Flag indicating cancellation was requested.
        cancellation_reason: Optional reason for cancellation.

    Progress tracking (footgun A.1):
        _last_progress_at: Timestamp of last progress mark.

    Methods:
        elapsed_ms(): Time since start in milliseconds.
        budget_remaining(): Ticks remaining in budget.
        budget_exceeded(): Check if budget is exhausted.
        push_path(node_id): Add node to path (entering node).
        pop_path(): Remove last node from path (exiting node).
        add_async(op_id): Register async operation.
        complete_async(op_id): Mark async operation complete.
        has_pending_async(): Check if any async operations pending.
        mark_progress(): Explicitly mark progress (A.1).
        request_cancellation(reason): Request cancellation.
        with_blackboard(bb): Create copy with different blackboard.
    """

    # Event that triggered this tick (optional for now)
    event: Optional[Any] = None

    # State access - required, but using default to allow creation without it
    # In practice, should always be provided
    blackboard: Optional["TypedBlackboard"] = None

    # Dependency injection container (future enhancement)
    services: Optional[Any] = None

    # Tick tracking
    tick_count: int = 0
    tick_budget: int = 1000
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Debugging
    parent_path: List[str] = field(default_factory=list)
    trace_enabled: bool = False

    # Async coordination
    async_pending: Set[str] = field(default_factory=set)

    # Cancellation
    cancellation_requested: bool = False
    cancellation_reason: Optional[str] = None

    # Progress tracking (footgun A.1)
    _last_progress_at: Optional[datetime] = field(default=None, repr=False)

    # Trace log for debugging
    _trace_log: List[dict] = field(default_factory=list, repr=False)

    # =========================================================================
    # Time/Budget Methods
    # =========================================================================

    def elapsed_ms(self) -> float:
        """Get elapsed time since execution started.

        Returns:
            Milliseconds since start_time.
        """
        delta = datetime.now(timezone.utc) - self.start_time
        return delta.total_seconds() * 1000

    def budget_remaining(self) -> int:
        """Get remaining tick budget.

        Returns:
            Number of ticks remaining (tick_budget - tick_count).
            Will be negative if budget exceeded.
        """
        return self.tick_budget - self.tick_count

    def budget_exceeded(self) -> bool:
        """Check if tick budget has been exhausted.

        Returns:
            True if tick_count >= tick_budget.
        """
        return self.tick_count >= self.tick_budget

    # =========================================================================
    # Path Management (for debugging)
    # =========================================================================

    def push_path(self, node_id: str) -> None:
        """Add node ID to parent path.

        Called when entering a node during traversal.

        Args:
            node_id: The ID of the node being entered.
        """
        self.parent_path.append(node_id)

    def pop_path(self) -> Optional[str]:
        """Remove and return last node ID from path.

        Called when exiting a node during traversal.

        Returns:
            The ID of the node being exited, or None if path is empty.
        """
        if self.parent_path:
            return self.parent_path.pop()
        return None

    def get_current_path(self) -> str:
        """Get the current path as a string.

        Returns:
            Path as "node1 > node2 > node3" format.
        """
        return " > ".join(self.parent_path)

    # =========================================================================
    # Async Coordination
    # =========================================================================

    def add_async(self, op_id: str) -> None:
        """Register a pending async operation.

        Called when starting an async operation (LLM call, tool call, etc.)
        that will complete in a future tick.

        Args:
            op_id: Unique identifier for the async operation.
        """
        self.async_pending.add(op_id)

    def complete_async(self, op_id: str) -> None:
        """Mark an async operation as complete.

        Called when an async operation finishes. Also marks progress
        per footgun A.1 (async completion is progress).

        Args:
            op_id: The identifier of the completed operation.
        """
        self.async_pending.discard(op_id)
        # Per footgun A.1: async completion marks progress
        self._last_progress_at = datetime.now(timezone.utc)

    def has_pending_async(self) -> bool:
        """Check if any async operations are pending.

        Returns:
            True if async_pending set is non-empty.
        """
        return len(self.async_pending) > 0

    # =========================================================================
    # Progress Tracking (footgun A.1)
    # =========================================================================

    def mark_progress(self) -> None:
        """Explicitly mark progress for watchdog.

        Per footgun A.1 addendum, progress is defined as ANY of:
        1. Blackboard write (handled by blackboard hook)
        2. Async operation completion (handled by complete_async)
        3. Explicit progress mark (this method)

        Call this when the node is making progress but not writing
        to blackboard or completing async operations.
        """
        self._last_progress_at = datetime.now(timezone.utc)

    @property
    def last_progress_at(self) -> Optional[datetime]:
        """Get timestamp of last progress mark.

        Returns:
            Datetime of last progress, or None if no progress marked.
        """
        return self._last_progress_at

    def get_last_progress_at(self) -> Optional[datetime]:
        """Get timestamp of last progress mark.

        Used by TreeWatchdog for stuck detection.

        Returns:
            Timestamp of last progress, or None if never marked.
        """
        return self._last_progress_at

    def time_since_progress_ms(self) -> Optional[float]:
        """Get time since last progress was marked.

        Returns:
            Milliseconds since last progress, or None if no progress marked.
        """
        if self._last_progress_at is None:
            return None
        delta = datetime.now(timezone.utc) - self._last_progress_at
        return delta.total_seconds() * 1000

    # =========================================================================
    # Cancellation
    # =========================================================================

    def request_cancellation(self, reason: Optional[str] = None) -> None:
        """Request cancellation of the current execution.

        Sets the cancellation flag which nodes should check at tick start.

        Args:
            reason: Optional human-readable reason for cancellation.
        """
        self.cancellation_requested = True
        self.cancellation_reason = reason

    def clear_cancellation(self) -> None:
        """Clear the cancellation request.

        Should be called when starting a new execution.
        """
        self.cancellation_requested = False
        self.cancellation_reason = None

    # =========================================================================
    # Context Copying
    # =========================================================================

    def with_blackboard(self, bb: "TypedBlackboard") -> "TickContext":
        """Create a copy of this context with a different blackboard.

        Useful for parallel children that need isolated scopes.

        Args:
            bb: The new blackboard to use.

        Returns:
            New TickContext with the new blackboard and copied state.
        """
        new_ctx = TickContext(
            event=self.event,
            blackboard=bb,
            services=self.services,
            tick_count=self.tick_count,
            tick_budget=self.tick_budget,
            start_time=self.start_time,
            parent_path=list(self.parent_path),  # Copy the list
            trace_enabled=self.trace_enabled,
            async_pending=set(self.async_pending),  # Copy the set
            cancellation_requested=self.cancellation_requested,
            cancellation_reason=self.cancellation_reason,
            _last_progress_at=self._last_progress_at,
        )
        # Share trace log reference (don't copy - want unified trace)
        new_ctx._trace_log = self._trace_log
        return new_ctx

    def increment_tick(self) -> None:
        """Increment the tick count.

        Called by the runtime after each tree tick.
        """
        self.tick_count += 1

    # =========================================================================
    # Debug Methods
    # =========================================================================

    def debug_info(self) -> dict:
        """Get debug information about the context.

        Returns:
            Dictionary with context state.
        """
        return {
            "tick_count": self.tick_count,
            "tick_budget": self.tick_budget,
            "budget_remaining": self.budget_remaining(),
            "elapsed_ms": self.elapsed_ms(),
            "parent_path": list(self.parent_path),
            "async_pending": list(self.async_pending),
            "cancellation_requested": self.cancellation_requested,
            "cancellation_reason": self.cancellation_reason,
            "trace_enabled": self.trace_enabled,
            "last_progress_at": (
                self._last_progress_at.isoformat() if self._last_progress_at else None
            ),
            "time_since_progress_ms": self.time_since_progress_ms(),
        }

    # =========================================================================
    # Tracing
    # =========================================================================

    def trace(
        self,
        node_id: str,
        event: str,
        **details: Any,
    ) -> None:
        """Log a trace event if tracing is enabled.

        Args:
            node_id: ID of the node generating the trace.
            event: Type of event (e.g., "tick_start", "tick_end").
            **details: Additional event details.
        """
        if self.trace_enabled:
            self._trace_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tick": self.tick_count,
                "node_id": node_id,
                "path": list(self.parent_path),
                "event": event,
                **details,
            })

    def get_trace_log(self) -> List[dict]:
        """Get the trace log."""
        return list(self._trace_log)

    def clear_trace_log(self) -> None:
        """Clear the trace log."""
        self._trace_log.clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TickContext",
]
