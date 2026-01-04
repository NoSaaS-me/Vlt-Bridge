"""Core types for the behavior tree system.

This module defines the fundamental types used throughout the behavior tree:
- RunStatus: Execution result states (Success, Failure, Running)
- TickContext: Per-tick evaluation context
- Blackboard: Shared state between nodes

Design based on Honorbuddy patterns (see research.md Section 7.2):
- RunStatus enables coroutine-like multi-tick operations
- Blackboard provides isolated per-tree shared state
- TickContext carries frame-local data for optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import RuleContext


class RunStatus(Enum):
    """Execution result status for behavior tree nodes.

    Based on Honorbuddy's RunStatus pattern:
    - SUCCESS: Task completed successfully, move to next
    - FAILURE: Task failed, try alternative (in selector) or abort (in sequence)
    - RUNNING: Task in progress, resume on next tick (coroutine-like)

    This tri-state enables:
    - Short-circuit evaluation in selectors/sequences
    - Multi-tick operations (e.g., waiting, async actions)
    - Frame locking optimization (cache running node)
    """

    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()

    def __bool__(self) -> bool:
        """Allow boolean conversion for convenience.

        Returns:
            True if SUCCESS, False otherwise.
        """
        return self == RunStatus.SUCCESS

    @classmethod
    def from_bool(cls, value: bool) -> "RunStatus":
        """Convert a boolean to RunStatus.

        Args:
            value: Boolean value.

        Returns:
            SUCCESS if True, FAILURE if False.
        """
        return cls.SUCCESS if value else cls.FAILURE


@dataclass
class TickContext:
    """Context passed to nodes during tree evaluation.

    Contains all data needed for node evaluation:
    - rule_context: The RuleContext from the plugin system
    - frame_id: Monotonic frame counter for cache invalidation
    - cache: Per-tick cache for computed values
    - blackboard: Reference to the tree's shared blackboard

    The frame_id enables frame locking optimization:
    - If frame_id unchanged, cached running node is valid
    - If frame_id changed, re-evaluate from root

    Attributes:
        rule_context: The RuleContext containing agent state.
        frame_id: Monotonic counter incremented each evaluation cycle.
        cache: Per-tick cache dictionary (cleared each tick).
        blackboard: Reference to the tree's Blackboard.
        delta_time_ms: Time since last tick in milliseconds (for time-based nodes).
    """

    rule_context: "RuleContext"
    frame_id: int = 0
    cache: Dict[str, Any] = field(default_factory=dict)
    blackboard: Optional["Blackboard"] = None
    delta_time_ms: float = 0.0

    def get_cached(self, key: str, default: Any = None) -> Any:
        """Get a value from the per-tick cache.

        Args:
            key: Cache key.
            default: Default value if not found.

        Returns:
            Cached value or default.
        """
        return self.cache.get(key, default)

    def set_cached(self, key: str, value: Any) -> None:
        """Set a value in the per-tick cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        self.cache[key] = value

    def clear_cache(self) -> None:
        """Clear the per-tick cache."""
        self.cache.clear()


class Blackboard:
    """Shared state container for behavior tree nodes.

    The Blackboard provides a key-value store that all nodes in a tree
    can read from and write to. This enables:
    - Data passing between nodes (e.g., selector stores choice for sequence)
    - State accumulation across ticks
    - Conditional behavior based on prior node results

    Features:
    - Namespaced keys for plugin isolation (optional prefix)
    - Type-safe getters with defaults
    - Scoped state that persists across ticks

    Example:
        >>> bb = Blackboard()
        >>> bb.set("target", "file.md")
        >>> bb.get("target")
        'file.md'
        >>> bb.get("missing", default="none")
        'none'
    """

    def __init__(self, namespace: str = "") -> None:
        """Initialize the blackboard.

        Args:
            namespace: Optional prefix for all keys (for isolation).
        """
        self._data: Dict[str, Any] = {}
        self._namespace = namespace

    def _make_key(self, key: str) -> str:
        """Create a namespaced key.

        Args:
            key: The base key.

        Returns:
            Namespaced key string.
        """
        if self._namespace:
            return f"{self._namespace}:{key}"
        return key

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the blackboard.

        Args:
            key: Key to retrieve.
            default: Default value if key not found.

        Returns:
            The stored value or default.
        """
        return self._data.get(self._make_key(key), default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the blackboard.

        Args:
            key: Key to set.
            value: Value to store.
        """
        self._data[self._make_key(key)] = value

    def has(self, key: str) -> bool:
        """Check if a key exists in the blackboard.

        Args:
            key: Key to check.

        Returns:
            True if key exists.
        """
        return self._make_key(key) in self._data

    def delete(self, key: str) -> bool:
        """Delete a key from the blackboard.

        Args:
            key: Key to delete.

        Returns:
            True if key was deleted, False if not found.
        """
        full_key = self._make_key(key)
        if full_key in self._data:
            del self._data[full_key]
            return True
        return False

    def clear(self) -> None:
        """Clear all values from the blackboard.

        If namespaced, only clears keys with this namespace.
        """
        if self._namespace:
            # Only clear namespaced keys
            prefix = f"{self._namespace}:"
            self._data = {
                k: v for k, v in self._data.items()
                if not k.startswith(prefix)
            }
        else:
            self._data.clear()

    def keys(self) -> list[str]:
        """Get all keys in the blackboard.

        If namespaced, returns keys without the namespace prefix.

        Returns:
            List of keys.
        """
        if self._namespace:
            prefix = f"{self._namespace}:"
            return [
                k[len(prefix):] for k in self._data.keys()
                if k.startswith(prefix)
            ]
        return list(self._data.keys())

    def items(self) -> list[tuple[str, Any]]:
        """Get all key-value pairs.

        If namespaced, returns keys without the namespace prefix.

        Returns:
            List of (key, value) tuples.
        """
        if self._namespace:
            prefix = f"{self._namespace}:"
            return [
                (k[len(prefix):], v) for k, v in self._data.items()
                if k.startswith(prefix)
            ]
        return list(self._data.items())

    def copy(self) -> "Blackboard":
        """Create a copy of this blackboard.

        Returns:
            New Blackboard with copied data.
        """
        new_bb = Blackboard(namespace=self._namespace)
        new_bb._data = self._data.copy()
        return new_bb

    def __len__(self) -> int:
        """Return the number of entries.

        Returns:
            Number of keys in the blackboard (respecting namespace).
        """
        if self._namespace:
            prefix = f"{self._namespace}:"
            return sum(1 for k in self._data.keys() if k.startswith(prefix))
        return len(self._data)

    def __repr__(self) -> str:
        """String representation for debugging."""
        ns_str = f"namespace='{self._namespace}', " if self._namespace else ""
        return f"Blackboard({ns_str}keys={self.keys()})"


__all__ = [
    "RunStatus",
    "TickContext",
    "Blackboard",
]
