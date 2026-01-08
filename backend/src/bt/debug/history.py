"""
Tick History Tracker - Records tick history for debugging.

Part of the BT Universal Runtime (spec 019).
Implements FR-8 from spec.md:
- Last N ticks history with timestamps
- Per-node timing (average tick duration, total time in RUNNING)
- Error counts and last error per node
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Deque, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import RunStatus


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TickEntry:
    """Single tick history entry."""

    tick_number: int
    timestamp: datetime
    tree_id: str
    node_id: str
    node_path: List[str]
    status: str  # RunStatus name
    duration_ms: float
    event_type: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeStats:
    """Statistics for a single node."""

    node_id: str
    tick_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    running_count: int = 0
    total_duration_ms: float = 0.0
    total_running_time_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None

    @property
    def avg_tick_duration_ms(self) -> float:
        """Average tick duration in milliseconds."""
        if self.tick_count == 0:
            return 0.0
        return self.total_duration_ms / self.tick_count


# =============================================================================
# Tick History Tracker
# =============================================================================


class TickHistoryTracker:
    """
    Tracks tick history for behavior trees.

    From spec.md FR-8:
    - Last N ticks history with timestamps
    - Per-node timing
    - Error counts and last error per node

    Acceptance criteria:
    - History retains at least 100 ticks
    - Timing data accurate to 1ms
    """

    DEFAULT_MAX_ENTRIES = 100

    def __init__(
        self,
        tree_id: str,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        """
        Initialize tick history tracker.

        Args:
            tree_id: ID of the tree being tracked.
            max_entries: Maximum history entries to retain.
        """
        self._tree_id = tree_id
        self._max_entries = max_entries
        self._entries: Deque[TickEntry] = deque(maxlen=max_entries)
        self._node_stats: Dict[str, NodeStats] = {}
        self._tick_number = 0

    @property
    def tree_id(self) -> str:
        """Tree being tracked."""
        return self._tree_id

    @property
    def entry_count(self) -> int:
        """Number of entries in history."""
        return len(self._entries)

    @property
    def tick_number(self) -> int:
        """Current tick number."""
        return self._tick_number

    def record_tick(
        self,
        node_id: str,
        node_path: List[str],
        status: "RunStatus",
        duration_ms: float,
        event_type: Optional[str] = None,
        error_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> TickEntry:
        """
        Record a tick in history.

        Args:
            node_id: ID of the node that was ticked.
            node_path: Path from root to node.
            status: Result status of the tick.
            duration_ms: Duration of the tick in milliseconds.
            event_type: Optional event type that triggered tick.
            error_message: Error message if status is FAILURE.
            details: Additional details to record.

        Returns:
            The created TickEntry.
        """
        self._tick_number += 1

        # Create entry
        entry = TickEntry(
            tick_number=self._tick_number,
            timestamp=datetime.now(timezone.utc),
            tree_id=self._tree_id,
            node_id=node_id,
            node_path=list(node_path),
            status=status.name,
            duration_ms=duration_ms,
            event_type=event_type,
            details=details or {},
        )

        # Add to history
        self._entries.append(entry)

        # Update node stats
        self._update_node_stats(node_id, status, duration_ms, error_message)

        return entry

    def _update_node_stats(
        self,
        node_id: str,
        status: "RunStatus",
        duration_ms: float,
        error_message: Optional[str] = None,
    ) -> None:
        """Update statistics for a node."""
        if node_id not in self._node_stats:
            self._node_stats[node_id] = NodeStats(node_id=node_id)

        stats = self._node_stats[node_id]
        stats.tick_count += 1
        stats.total_duration_ms += duration_ms

        # Update status counts
        status_name = status.name
        if status_name == "SUCCESS":
            stats.success_count += 1
        elif status_name == "FAILURE":
            stats.failure_count += 1
            if error_message:
                stats.last_error = error_message
                stats.last_error_at = datetime.now(timezone.utc)
        elif status_name == "RUNNING":
            stats.running_count += 1
            stats.total_running_time_ms += duration_ms

    def get_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        node_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[TickEntry]:
        """
        Get tick history entries.

        Args:
            limit: Maximum entries to return.
            offset: Number of entries to skip.
            node_id: Filter by node ID.
            status: Filter by status name.

        Returns:
            List of TickEntry (newest first).
        """
        # Start with all entries (reversed for newest first)
        entries = list(reversed(self._entries))

        # Apply filters
        if node_id:
            entries = [e for e in entries if e.node_id == node_id]

        if status:
            entries = [e for e in entries if e.status == status]

        # Apply pagination
        return entries[offset : offset + limit]

    def get_node_stats(self, node_id: str) -> Optional[NodeStats]:
        """
        Get statistics for a specific node.

        Args:
            node_id: Node to get stats for.

        Returns:
            NodeStats or None if node never ticked.
        """
        return self._node_stats.get(node_id)

    def get_all_node_stats(self) -> Dict[str, NodeStats]:
        """Get statistics for all tracked nodes."""
        return dict(self._node_stats)

    def get_recent_errors(self, limit: int = 10) -> List[TickEntry]:
        """
        Get recent tick entries with FAILURE status.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of TickEntry with FAILURE status (newest first).
        """
        return self.get_entries(limit=limit, status="FAILURE")

    def clear(self) -> None:
        """Clear all history and stats."""
        self._entries.clear()
        self._node_stats.clear()
        self._tick_number = 0

    def reset_tick_number(self) -> None:
        """Reset tick number (typically on tree reset)."""
        self._tick_number = 0

    def to_dict(self) -> Dict[str, Any]:
        """Export history as dictionary."""
        return {
            "tree_id": self._tree_id,
            "tick_number": self._tick_number,
            "entry_count": len(self._entries),
            "max_entries": self._max_entries,
            "entries": [
                {
                    "tick_number": e.tick_number,
                    "timestamp": e.timestamp.isoformat(),
                    "node_id": e.node_id,
                    "node_path": e.node_path,
                    "status": e.status,
                    "duration_ms": e.duration_ms,
                    "event_type": e.event_type,
                    "details": e.details,
                }
                for e in self._entries
            ],
            "node_stats": {
                node_id: {
                    "tick_count": stats.tick_count,
                    "success_count": stats.success_count,
                    "failure_count": stats.failure_count,
                    "running_count": stats.running_count,
                    "avg_tick_duration_ms": stats.avg_tick_duration_ms,
                    "total_running_time_ms": stats.total_running_time_ms,
                    "last_error": stats.last_error,
                    "last_error_at": (
                        stats.last_error_at.isoformat() if stats.last_error_at else None
                    ),
                }
                for node_id, stats in self._node_stats.items()
            },
        }


# =============================================================================
# Global History Registry
# =============================================================================


class HistoryRegistry:
    """
    Registry of tick history trackers for all trees.

    Allows looking up history by tree ID.
    """

    def __init__(self, default_max_entries: int = 100) -> None:
        """
        Initialize history registry.

        Args:
            default_max_entries: Default max entries for new trackers.
        """
        self._trackers: Dict[str, TickHistoryTracker] = {}
        self._default_max_entries = default_max_entries

    def get_or_create(self, tree_id: str) -> TickHistoryTracker:
        """
        Get or create history tracker for a tree.

        Args:
            tree_id: Tree identifier.

        Returns:
            TickHistoryTracker for the tree.
        """
        if tree_id not in self._trackers:
            self._trackers[tree_id] = TickHistoryTracker(
                tree_id=tree_id,
                max_entries=self._default_max_entries,
            )
        return self._trackers[tree_id]

    def get(self, tree_id: str) -> Optional[TickHistoryTracker]:
        """
        Get history tracker for a tree if exists.

        Args:
            tree_id: Tree identifier.

        Returns:
            TickHistoryTracker or None.
        """
        return self._trackers.get(tree_id)

    def remove(self, tree_id: str) -> bool:
        """
        Remove history tracker for a tree.

        Args:
            tree_id: Tree to remove history for.

        Returns:
            True if removed, False if not found.
        """
        if tree_id in self._trackers:
            del self._trackers[tree_id]
            return True
        return False

    def list_trees(self) -> List[str]:
        """List all trees with history."""
        return list(self._trackers.keys())

    def clear_all(self) -> None:
        """Clear all history trackers."""
        self._trackers.clear()


# Global singleton instance
_history_registry: Optional[HistoryRegistry] = None


def get_history_registry() -> HistoryRegistry:
    """Get the global history registry singleton."""
    global _history_registry
    if _history_registry is None:
        _history_registry = HistoryRegistry()
    return _history_registry


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "TickEntry",
    "NodeStats",
    "TickHistoryTracker",
    "HistoryRegistry",
    "get_history_registry",
]
