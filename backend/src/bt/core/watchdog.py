"""
TreeWatchdog - Detects stuck nodes based on progress tracking.

Part of the BT Universal Runtime (spec 019).
Implements A.1 Progress Tracking for Stuck Detection from footgun-addendum.md.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import TickContext
    from .tree import BehaviorTree

# Import Event and Severity from ANS
from ...services.ans.event import Event, Severity


logger = logging.getLogger(__name__)


@dataclass
class StuckNodeInfo:
    """
    Information about a stuck node.

    From footgun-addendum.md A.1 - returned when stuck detection triggers.

    Attributes:
        node_id: ID of the stuck node.
        node_path: Path from root to stuck node.
        running_duration_ms: How long the node has been RUNNING.
        last_progress_at: When progress was last marked (if ever).
        is_warning: True if warning threshold, False if timeout.
    """

    node_id: str
    node_path: List[str]
    running_duration_ms: float
    last_progress_at: Optional[datetime]
    is_warning: bool  # True = warning threshold, False = hard timeout

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_path": self.node_path,
            "running_duration_ms": self.running_duration_ms,
            "last_progress_at": (
                self.last_progress_at.isoformat()
                if self.last_progress_at else None
            ),
            "is_warning": self.is_warning,
        }


@dataclass
class TreeWatchdog:
    """
    Detects stuck nodes based on progress tracking.

    From footgun-addendum.md A.1:
    - Progress is defined as ANY of:
      1. Blackboard write (bb.set() succeeds)
      2. Async operation completion (ctx.complete_async() called)
      3. Explicit progress mark (ctx.mark_progress() called)

    - Stuck detection triggers when a node has been RUNNING longer than
      the timeout AND has not made progress since entering RUNNING.

    Configuration:
        warning_threshold_ms: Time before emitting warning (default 30s).
        timeout_ms: Time before hard timeout / forced failure (default 60s).
        check_interval_ms: Minimum time between checks (default 1000ms).

    Example:
        >>> watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        >>> stuck = watchdog.check_stuck(tree, ctx)
        >>> if stuck:
        ...     if stuck.is_warning:
        ...         logger.warning(f"Node {stuck.node_id} may be stuck")
        ...     else:
        ...         logger.error(f"Node {stuck.node_id} timed out")
    """

    # Thresholds from tree-loader.yaml and footgun-addendum.md
    warning_threshold_ms: int = 30000  # 30 seconds - emit warning
    timeout_ms: int = 60000  # 60 seconds - hard timeout

    # Internal tracking
    _last_check_at: Optional[datetime] = field(default=None, repr=False)
    _check_interval_ms: int = field(default=1000, repr=False)  # Min 1s between checks
    _warned_nodes: Dict[str, datetime] = field(default_factory=dict, repr=False)

    def check_stuck(
        self,
        tree: "BehaviorTree",
        ctx: "TickContext",
    ) -> Optional[StuckNodeInfo]:
        """
        Check all RUNNING nodes for stuck condition.

        From footgun-addendum.md A.1:
        - Iterates through nodes with status == RUNNING
        - Checks if running_ms > timeout threshold
        - Considers progress: if last_progress_at is before running_since, stuck

        Args:
            tree: The behavior tree to check.
            ctx: Current tick context.

        Returns:
            StuckNodeInfo if any node is stuck, None if all OK.
            Returns the FIRST stuck node found (most critical).
        """
        now = datetime.now(timezone.utc)

        # Throttle check frequency
        if self._last_check_at is not None:
            elapsed = (now - self._last_check_at).total_seconds() * 1000
            if elapsed < self._check_interval_ms:
                return None

        self._last_check_at = now

        # Get all running nodes
        running_nodes = tree.get_running_nodes()
        if not running_nodes:
            return None

        # Check each running node
        for node in running_nodes:
            stuck_info = self._check_node(node, ctx, tree, now)
            if stuck_info is not None:
                return stuck_info

        return None

    def _check_node(
        self,
        node: Any,  # BehaviorNodeProtocol
        ctx: "TickContext",
        tree: "BehaviorTree",
        now: datetime,
    ) -> Optional[StuckNodeInfo]:
        """
        Check if a specific node is stuck.

        Args:
            node: The node to check.
            ctx: Current tick context.
            tree: The behavior tree.
            now: Current timestamp.

        Returns:
            StuckNodeInfo if stuck, None otherwise.
        """
        # Skip if node has no running_since (shouldn't happen but be safe)
        running_since = getattr(node, "running_since", None)
        if running_since is None:
            return None

        # Calculate how long node has been running
        running_ms = (now - running_since).total_seconds() * 1000

        # Check if we should consider it stuck
        last_progress = ctx.get_last_progress_at()

        # Node is stuck if:
        # 1. Running longer than threshold
        # 2. No progress since running_since
        is_stuck_candidate = (
            last_progress is None or last_progress < running_since
        )

        if not is_stuck_candidate:
            # Node made progress - clear warning if any
            if node.id in self._warned_nodes:
                del self._warned_nodes[node.id]
            return None

        # Get path to node for reporting
        node_path = tree.get_node_path(node.id)

        # Check for hard timeout first
        if running_ms >= self.timeout_ms:
            logger.error(
                f"Node {node.id} stuck: running {running_ms:.0f}ms without progress "
                f"(timeout: {self.timeout_ms}ms)"
            )
            return StuckNodeInfo(
                node_id=node.id,
                node_path=node_path,
                running_duration_ms=running_ms,
                last_progress_at=last_progress,
                is_warning=False,
            )

        # Check for warning threshold
        if running_ms >= self.warning_threshold_ms:
            # Only warn once per node
            if node.id not in self._warned_nodes:
                self._warned_nodes[node.id] = now
                logger.warning(
                    f"Node {node.id} may be stuck: running {running_ms:.0f}ms without progress "
                    f"(warning: {self.warning_threshold_ms}ms, timeout: {self.timeout_ms}ms)"
                )
                return StuckNodeInfo(
                    node_id=node.id,
                    node_path=node_path,
                    running_duration_ms=running_ms,
                    last_progress_at=last_progress,
                    is_warning=True,
                )

        return None

    def create_warning_event(
        self,
        tree: "BehaviorTree",
        stuck_info: StuckNodeInfo,
    ) -> Event:
        """
        Create ANS event for watchdog warning.

        From events.yaml tree.watchdog.warning:
        - severity: warning
        - payload: tree_id, node_id, node_path, running_duration_ms, thresholds

        Args:
            tree: The behavior tree.
            stuck_info: Information about the stuck node.

        Returns:
            ANS Event for emission.
        """
        return Event(
            type="tree.watchdog.warning",
            source="tree_watchdog",
            severity=Severity.WARNING,
            payload={
                "tree_id": tree.id,
                "node_id": stuck_info.node_id,
                "node_path": stuck_info.node_path,
                "running_duration_ms": int(stuck_info.running_duration_ms),
                "warning_threshold_ms": self.warning_threshold_ms,
                "timeout_ms": self.timeout_ms,
            },
        )

    def create_timeout_event(
        self,
        tree: "BehaviorTree",
        stuck_info: StuckNodeInfo,
    ) -> Event:
        """
        Create ANS event for watchdog timeout.

        From events.yaml tree.watchdog.timeout:
        - severity: critical
        - payload: tree_id, node_id, node_path, running_duration_ms, timeout_ms, last_progress_at

        Args:
            tree: The behavior tree.
            stuck_info: Information about the stuck node.

        Returns:
            ANS Event for emission.
        """
        return Event(
            type="tree.watchdog.timeout",
            source="tree_watchdog",
            severity=Severity.CRITICAL,
            payload={
                "tree_id": tree.id,
                "node_id": stuck_info.node_id,
                "node_path": stuck_info.node_path,
                "running_duration_ms": int(stuck_info.running_duration_ms),
                "timeout_ms": self.timeout_ms,
                "last_progress_at": (
                    stuck_info.last_progress_at.isoformat()
                    if stuck_info.last_progress_at else None
                ),
            },
        )

    def clear_warnings(self) -> None:
        """Clear all warned node tracking."""
        self._warned_nodes.clear()

    def clear_warning(self, node_id: str) -> None:
        """Clear warning for a specific node."""
        if node_id in self._warned_nodes:
            del self._warned_nodes[node_id]

    def get_warned_nodes(self) -> List[str]:
        """Get list of node IDs that have been warned."""
        return list(self._warned_nodes.keys())

    def reset(self) -> None:
        """Reset watchdog state for new execution."""
        self._warned_nodes.clear()
        self._last_check_at = None

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information about watchdog state."""
        return {
            "warning_threshold_ms": self.warning_threshold_ms,
            "timeout_ms": self.timeout_ms,
            "check_interval_ms": self._check_interval_ms,
            "warned_nodes": list(self._warned_nodes.keys()),
            "last_check_at": (
                self._last_check_at.isoformat()
                if self._last_check_at else None
            ),
        }


__all__ = ["TreeWatchdog", "StuckNodeInfo"]
