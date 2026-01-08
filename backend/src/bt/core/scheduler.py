"""
TickScheduler - Manages tick execution with budget and event buffering.

Part of the BT Universal Runtime (spec 019).
Implements A.2 Event Buffering During Tick from footgun-addendum.md.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

from ..state import RunStatus

if TYPE_CHECKING:
    from .context import TickContext
    from .tree import BehaviorTree, TreeStatus
    from .watchdog import TreeWatchdog, StuckNodeInfo

# Import Event and Severity from ANS
from ...services.ans.event import Event, Severity
from ...services.ans.bus import EventBus


logger = logging.getLogger(__name__)


@dataclass
class TickResult:
    """
    Result of a tick execution.

    From footgun-addendum.md A.2 - returned after tick completes.

    Attributes:
        status: The RunStatus returned by the tree tick.
        tree_status: Current TreeStatus after tick.
        tick_count: Total ticks executed (in this execution).
        duration_ms: Duration of this tick in milliseconds.
        nodes_ticked: Number of nodes that were ticked.
        budget_exceeded: True if tick budget was exceeded.
        stuck_node: StuckNodeInfo if a node was detected stuck.
        events_emitted: Number of events emitted after tick.
    """

    status: RunStatus
    tree_status: "TreeStatus"
    tick_count: int
    duration_ms: float
    nodes_ticked: int = 0
    budget_exceeded: bool = False
    stuck_node: Optional["StuckNodeInfo"] = None
    events_emitted: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.name,
            "tree_status": self.tree_status.value,
            "tick_count": self.tick_count,
            "duration_ms": self.duration_ms,
            "nodes_ticked": self.nodes_ticked,
            "budget_exceeded": self.budget_exceeded,
            "stuck_node": (
                self.stuck_node.to_dict() if self.stuck_node else None
            ),
            "events_emitted": self.events_emitted,
        }


@dataclass
class TickScheduler:
    """
    Manages tick execution with budget and event buffering.

    From footgun-addendum.md A.2:
    - Events are buffered during tick to prevent re-entrant ticks
    - Buffer is flushed after tick completes in FIFO order
    - Buffer has max size with oldest-drop policy

    This scheduler:
    1. Enters tick scope (buffers events)
    2. Runs tree.tick(ctx)
    3. Checks watchdog for stuck nodes
    4. Exits tick scope (flushes events)
    5. Returns TickResult

    Attributes:
        event_bus: ANS EventBus for event emission (optional).
        max_buffer_size: Maximum events to buffer (default 1000).
        watchdog: Optional TreeWatchdog for stuck detection.
        emit_tick_events: Whether to emit tree.tick.* events.

    Example:
        >>> scheduler = TickScheduler(event_bus=get_event_bus())
        >>> ctx = TickContext(blackboard=tree.blackboard)
        >>> result = scheduler.run_tick(tree, ctx)
        >>> print(f"Tick completed: {result.status.name}")
    """

    event_bus: Optional[EventBus] = None
    max_buffer_size: int = 1000
    watchdog: Optional["TreeWatchdog"] = None
    emit_tick_events: bool = True

    # Internal state
    _buffer: List[Event] = field(default_factory=list, repr=False)
    _tick_in_progress: bool = field(default=False, repr=False)
    _nodes_ticked: int = field(default=0, repr=False)

    def run_tick(
        self,
        tree: "BehaviorTree",
        ctx: "TickContext",
    ) -> TickResult:
        """
        Execute tree tick with event buffering.

        From footgun-addendum.md A.2:
        1. Enter tick scope (buffer events)
        2. Run tree.tick(ctx)
        3. Check watchdog
        4. Exit tick scope (flush events)
        5. Return result

        Args:
            tree: The behavior tree to tick.
            ctx: Tick context with blackboard and state.

        Returns:
            TickResult with status, duration, and stuck info.
        """
        tick_start = datetime.now(timezone.utc)
        self._nodes_ticked = 0
        stuck_node: Optional["StuckNodeInfo"] = None

        # Emit tick start event
        if self.emit_tick_events:
            self._emit_tick_start(tree, ctx)

        # Execute tick with event buffering
        with self.tick_scope():
            try:
                # Run the actual tick
                status = tree.tick(ctx)

                # Check watchdog for stuck nodes
                if self.watchdog is not None:
                    stuck_node = self.watchdog.check_stuck(tree, ctx)
                    if stuck_node is not None:
                        self._handle_stuck_node(tree, stuck_node)

            except Exception as e:
                logger.error(f"Tick execution failed: {e}")
                status = RunStatus.FAILURE

        # Calculate duration
        tick_end = datetime.now(timezone.utc)
        duration_ms = (tick_end - tick_start).total_seconds() * 1000

        # Count events emitted
        events_emitted = len(self._buffer)

        # Build result
        result = TickResult(
            status=status,
            tree_status=tree.status,
            tick_count=ctx.tick_count,
            duration_ms=duration_ms,
            nodes_ticked=self._nodes_ticked,
            budget_exceeded=ctx.budget_exceeded(),
            stuck_node=stuck_node,
            events_emitted=events_emitted,
        )

        # Emit tick complete event
        if self.emit_tick_events:
            self._emit_tick_complete(tree, ctx, result)

        return result

    @contextmanager
    def tick_scope(self) -> Generator[None, None, None]:
        """
        Context manager for tree tick. Buffers events.

        From footgun-addendum.md A.2:
        - Sets _tick_in_progress = True
        - Events are buffered during this scope
        - On exit, flushes buffer in FIFO order

        Example:
            >>> with scheduler.tick_scope():
            ...     tree.tick(ctx)
            ...     # Events buffered during this block
            >>> # Events flushed here
        """
        self._tick_in_progress = True
        try:
            yield
        finally:
            self._tick_in_progress = False
            self._flush_buffer()

    def emit(self, event: Event) -> None:
        """
        Emit an event, buffering if tick is in progress.

        From footgun-addendum.md A.2:
        - During tick: buffer event (with overflow protection)
        - Outside tick: dispatch immediately

        Args:
            event: The event to emit.
        """
        if self._tick_in_progress:
            # Buffer during tick
            if len(self._buffer) >= self.max_buffer_size:
                logger.warning(
                    f"Event buffer full ({self.max_buffer_size}), "
                    f"dropping oldest: {self._buffer[0].type}"
                )
                self._buffer.pop(0)
            self._buffer.append(event)
        else:
            # Dispatch immediately
            self._dispatch(event)

    def _flush_buffer(self) -> None:
        """
        Dispatch all buffered events in FIFO order.

        From footgun-addendum.md A.2:
        - Processes events in order they were emitted
        - Clears buffer after dispatch
        """
        while self._buffer:
            event = self._buffer.pop(0)
            self._dispatch(event)

    def _dispatch(self, event: Event) -> None:
        """
        Dispatch a single event to the event bus.

        Args:
            event: The event to dispatch.
        """
        if self.event_bus is not None:
            try:
                self.event_bus.emit(event)
            except Exception as e:
                logger.error(f"Failed to dispatch event {event.type}: {e}")

    def _handle_stuck_node(
        self,
        tree: "BehaviorTree",
        stuck_info: "StuckNodeInfo",
    ) -> None:
        """
        Handle a stuck node detection.

        Args:
            tree: The behavior tree.
            stuck_info: Information about the stuck node.
        """
        if self.watchdog is None:
            return

        if stuck_info.is_warning:
            # Emit warning event
            event = self.watchdog.create_warning_event(tree, stuck_info)
            self.emit(event)
        else:
            # Emit timeout event
            event = self.watchdog.create_timeout_event(tree, stuck_info)
            self.emit(event)

    def _emit_tick_start(
        self,
        tree: "BehaviorTree",
        ctx: "TickContext",
    ) -> None:
        """
        Emit tree.tick.start event.

        From events.yaml tree.tick.start:
        - severity: debug
        - payload: tree_id, tick_number, trigger_event
        """
        event = Event(
            type="tree.tick.start",
            source="tick_scheduler",
            severity=Severity.DEBUG,
            payload={
                "tree_id": tree.id,
                "tick_number": ctx.tick_count + 1,  # Will be incremented
                "trigger_event": ctx.event.type if ctx.event else None,
            },
        )
        self._dispatch(event)  # Start event goes out immediately

    def _emit_tick_complete(
        self,
        tree: "BehaviorTree",
        ctx: "TickContext",
        result: TickResult,
    ) -> None:
        """
        Emit tree.tick.complete event.

        From events.yaml tree.tick.complete:
        - severity: info
        - payload: tree_id, tick_number, status, duration_ms, nodes_ticked
        """
        event = Event(
            type="tree.tick.complete",
            source="tick_scheduler",
            severity=Severity.INFO,
            payload={
                "tree_id": tree.id,
                "tick_number": result.tick_count,
                "status": result.status.name.lower(),
                "duration_ms": result.duration_ms,
                "nodes_ticked": result.nodes_ticked,
            },
        )
        self._dispatch(event)  # Complete event goes out immediately

    def increment_nodes_ticked(self) -> None:
        """Increment the count of nodes ticked."""
        self._nodes_ticked += 1

    def get_nodes_ticked(self) -> int:
        """Get the count of nodes ticked."""
        return self._nodes_ticked

    @property
    def is_tick_in_progress(self) -> bool:
        """Check if a tick is currently in progress."""
        return self._tick_in_progress

    @property
    def buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)

    def clear_buffer(self) -> int:
        """
        Clear the event buffer without dispatching.

        Returns:
            Number of events that were cleared.
        """
        count = len(self._buffer)
        self._buffer.clear()
        return count

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information about scheduler state."""
        return {
            "tick_in_progress": self._tick_in_progress,
            "buffer_size": len(self._buffer),
            "max_buffer_size": self.max_buffer_size,
            "nodes_ticked": self._nodes_ticked,
            "emit_tick_events": self.emit_tick_events,
            "has_event_bus": self.event_bus is not None,
            "has_watchdog": self.watchdog is not None,
        }


__all__ = ["TickScheduler", "TickResult"]
