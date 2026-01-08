"""
Breakpoint Manager - Debug breakpoints for behavior trees.

Part of the BT Universal Runtime (spec 019).
Implements FR-8 from spec.md and task 7.1.5 from tasks.md:
- Set breakpoints on nodes
- Conditional breakpoints
- Enable/disable breakpoints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import TypedBlackboard


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Breakpoint:
    """A breakpoint on a behavior tree node."""

    node_id: str
    tree_id: str
    enabled: bool = True
    condition: Optional[str] = None
    hit_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_hit_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "node_id": self.node_id,
            "tree_id": self.tree_id,
            "enabled": self.enabled,
            "condition": self.condition,
            "hit_count": self.hit_count,
            "created_at": self.created_at.isoformat(),
            "last_hit_at": self.last_hit_at.isoformat() if self.last_hit_at else None,
        }


@dataclass
class BreakpointHit:
    """Information about a breakpoint being hit."""

    breakpoint: Breakpoint
    node_id: str
    tree_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    blackboard_snapshot: Optional[Dict[str, Any]] = None


# =============================================================================
# Breakpoint Manager
# =============================================================================


class BreakpointManager:
    """
    Manages breakpoints for behavior trees.

    Breakpoints can be:
    - Unconditional: Always trigger
    - Conditional: Only trigger when condition evaluates to True

    Conditions are evaluated against the blackboard state using
    a safe expression evaluator.

    From spec.md FR-8 acceptance criteria:
    - Breakpoints on any node
    - Conditional breakpoints supported
    """

    def __init__(self, tree_id: str) -> None:
        """
        Initialize breakpoint manager.

        Args:
            tree_id: ID of the tree to manage breakpoints for.
        """
        self._tree_id = tree_id
        self._breakpoints: Dict[str, Breakpoint] = {}  # node_id -> Breakpoint
        self._on_hit_callbacks: List[Callable[[BreakpointHit], None]] = []
        self._paused: bool = False
        self._last_hit: Optional[BreakpointHit] = None

    @property
    def tree_id(self) -> str:
        """Tree this manager is for."""
        return self._tree_id

    @property
    def is_paused(self) -> bool:
        """Whether execution is paused at a breakpoint."""
        return self._paused

    @property
    def last_hit(self) -> Optional[BreakpointHit]:
        """Information about last breakpoint hit."""
        return self._last_hit

    def set_breakpoint(
        self,
        node_id: str,
        enabled: bool = True,
        condition: Optional[str] = None,
    ) -> Breakpoint:
        """
        Set or update a breakpoint on a node.

        Args:
            node_id: Node to set breakpoint on.
            enabled: Whether breakpoint is enabled.
            condition: Optional condition expression.

        Returns:
            The created or updated Breakpoint.
        """
        if node_id in self._breakpoints:
            # Update existing
            bp = self._breakpoints[node_id]
            bp.enabled = enabled
            bp.condition = condition
        else:
            # Create new
            bp = Breakpoint(
                node_id=node_id,
                tree_id=self._tree_id,
                enabled=enabled,
                condition=condition,
            )
            self._breakpoints[node_id] = bp

        return bp

    def remove_breakpoint(self, node_id: str) -> bool:
        """
        Remove a breakpoint from a node.

        Args:
            node_id: Node to remove breakpoint from.

        Returns:
            True if removed, False if not found.
        """
        if node_id in self._breakpoints:
            del self._breakpoints[node_id]
            return True
        return False

    def get_breakpoint(self, node_id: str) -> Optional[Breakpoint]:
        """
        Get breakpoint for a node.

        Args:
            node_id: Node to get breakpoint for.

        Returns:
            Breakpoint or None if no breakpoint set.
        """
        return self._breakpoints.get(node_id)

    def get_all_breakpoints(self) -> List[Breakpoint]:
        """Get all breakpoints."""
        return list(self._breakpoints.values())

    def get_enabled_breakpoints(self) -> List[Breakpoint]:
        """Get all enabled breakpoints."""
        return [bp for bp in self._breakpoints.values() if bp.enabled]

    def enable_breakpoint(self, node_id: str) -> bool:
        """
        Enable a breakpoint.

        Args:
            node_id: Node whose breakpoint to enable.

        Returns:
            True if enabled, False if breakpoint not found.
        """
        bp = self._breakpoints.get(node_id)
        if bp:
            bp.enabled = True
            return True
        return False

    def disable_breakpoint(self, node_id: str) -> bool:
        """
        Disable a breakpoint.

        Args:
            node_id: Node whose breakpoint to disable.

        Returns:
            True if disabled, False if breakpoint not found.
        """
        bp = self._breakpoints.get(node_id)
        if bp:
            bp.enabled = False
            return True
        return False

    def clear_all(self) -> int:
        """
        Remove all breakpoints.

        Returns:
            Number of breakpoints removed.
        """
        count = len(self._breakpoints)
        self._breakpoints.clear()
        self._paused = False
        self._last_hit = None
        return count

    def check_breakpoint(
        self,
        node_id: str,
        blackboard: Optional["TypedBlackboard"] = None,
    ) -> Optional[BreakpointHit]:
        """
        Check if a breakpoint should trigger.

        Called before ticking a node to check if we should pause.

        Args:
            node_id: Node about to be ticked.
            blackboard: Current blackboard state for condition evaluation.

        Returns:
            BreakpointHit if breakpoint triggered, None otherwise.
        """
        bp = self._breakpoints.get(node_id)
        if not bp or not bp.enabled:
            return None

        # Check condition if present
        if bp.condition:
            if not self._evaluate_condition(bp.condition, blackboard):
                return None

        # Breakpoint hit
        bp.hit_count += 1
        bp.last_hit_at = datetime.now(timezone.utc)

        # Create hit record
        hit = BreakpointHit(
            breakpoint=bp,
            node_id=node_id,
            tree_id=self._tree_id,
            blackboard_snapshot=blackboard.snapshot() if blackboard else None,
        )

        self._paused = True
        self._last_hit = hit

        # Notify callbacks
        for callback in self._on_hit_callbacks:
            try:
                callback(hit)
            except Exception:
                pass  # Don't let callback errors break execution

        return hit

    def resume(self) -> None:
        """Resume execution after breakpoint pause."""
        self._paused = False

    def on_hit(self, callback: Callable[[BreakpointHit], None]) -> None:
        """
        Register callback for breakpoint hits.

        Args:
            callback: Function to call when breakpoint is hit.
        """
        self._on_hit_callbacks.append(callback)

    def _evaluate_condition(
        self,
        condition: str,
        blackboard: Optional["TypedBlackboard"],
    ) -> bool:
        """
        Evaluate a breakpoint condition.

        Safe expression evaluation against blackboard state.
        Supports simple comparisons like:
        - key == value
        - key > value
        - key != value
        - key exists
        - key is None

        Args:
            condition: Condition expression string.
            blackboard: Blackboard to evaluate against.

        Returns:
            True if condition passes, False otherwise.
        """
        if not condition or not blackboard:
            return True  # No condition = always trigger

        try:
            # Get blackboard snapshot for evaluation
            snapshot = blackboard.snapshot()

            # Very simple expression evaluation (safe subset)
            condition = condition.strip()

            # Handle "key exists" pattern
            if condition.endswith(" exists"):
                key = condition[:-7].strip()
                return key in snapshot

            # Handle "key is None" pattern
            if " is None" in condition:
                key = condition.replace(" is None", "").strip()
                return key not in snapshot or snapshot.get(key) is None

            # Handle comparison operators
            for op in ["==", "!=", ">=", "<=", ">", "<"]:
                if op in condition:
                    parts = condition.split(op, 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value_str = parts[1].strip()

                        # Get blackboard value
                        if key not in snapshot:
                            return False
                        bb_value = snapshot[key]

                        # Parse comparison value
                        try:
                            # Try as int
                            cmp_value: Any = int(value_str)
                        except ValueError:
                            try:
                                # Try as float
                                cmp_value = float(value_str)
                            except ValueError:
                                # Try as boolean
                                if value_str.lower() == "true":
                                    cmp_value = True
                                elif value_str.lower() == "false":
                                    cmp_value = False
                                else:
                                    # Treat as string (remove quotes if present)
                                    cmp_value = value_str.strip("\"'")

                        # Evaluate comparison
                        if op == "==":
                            return bb_value == cmp_value
                        elif op == "!=":
                            return bb_value != cmp_value
                        elif op == ">":
                            return bb_value > cmp_value
                        elif op == "<":
                            return bb_value < cmp_value
                        elif op == ">=":
                            return bb_value >= cmp_value
                        elif op == "<=":
                            return bb_value <= cmp_value

            # Unknown format - treat as truthy check of key
            return bool(snapshot.get(condition))

        except Exception:
            # On any error, don't trigger breakpoint
            return False


# =============================================================================
# Global Breakpoint Registry
# =============================================================================


class BreakpointRegistry:
    """
    Registry of breakpoint managers for all trees.

    Allows looking up breakpoints by tree ID.
    """

    def __init__(self) -> None:
        """Initialize breakpoint registry."""
        self._managers: Dict[str, BreakpointManager] = {}

    def get_or_create(self, tree_id: str) -> BreakpointManager:
        """
        Get or create breakpoint manager for a tree.

        Args:
            tree_id: Tree identifier.

        Returns:
            BreakpointManager for the tree.
        """
        if tree_id not in self._managers:
            self._managers[tree_id] = BreakpointManager(tree_id=tree_id)
        return self._managers[tree_id]

    def get(self, tree_id: str) -> Optional[BreakpointManager]:
        """
        Get breakpoint manager for a tree if exists.

        Args:
            tree_id: Tree identifier.

        Returns:
            BreakpointManager or None.
        """
        return self._managers.get(tree_id)

    def remove(self, tree_id: str) -> bool:
        """
        Remove breakpoint manager for a tree.

        Args:
            tree_id: Tree to remove breakpoints for.

        Returns:
            True if removed, False if not found.
        """
        if tree_id in self._managers:
            del self._managers[tree_id]
            return True
        return False

    def list_trees(self) -> List[str]:
        """List all trees with breakpoint managers."""
        return list(self._managers.keys())

    def clear_all(self) -> None:
        """Clear all breakpoint managers."""
        self._managers.clear()


# Global singleton instance
_breakpoint_registry: Optional[BreakpointRegistry] = None


def get_breakpoint_registry() -> BreakpointRegistry:
    """Get the global breakpoint registry singleton."""
    global _breakpoint_registry
    if _breakpoint_registry is None:
        _breakpoint_registry = BreakpointRegistry()
    return _breakpoint_registry


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "Breakpoint",
    "BreakpointHit",
    "BreakpointManager",
    "BreakpointRegistry",
    "get_breakpoint_registry",
]
