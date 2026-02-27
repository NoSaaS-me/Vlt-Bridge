"""Base class for behavior tree nodes.

This module defines the abstract BehaviorNode class that all nodes
(composites, decorators, and leaves) inherit from.

Design principles:
- Minimal interface: tick(), reset(), status
- No child management in base (handled by Composite/Decorator)
- Support for debugging (name, debug_info)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import logging

from .types import RunStatus, TickContext


logger = logging.getLogger(__name__)


class BehaviorNode(ABC):
    """Abstract base class for all behavior tree nodes.

    Every node in the behavior tree inherits from this class.
    Nodes can be:
    - Composites: Manage multiple children (Selector, Sequence, Parallel)
    - Decorators: Wrap a single child with modified behavior
    - Leaves: Terminal nodes that perform actual work (Condition, Action)

    The tick() method is called each evaluation cycle. Nodes return:
    - SUCCESS: Task completed successfully
    - FAILURE: Task failed
    - RUNNING: Task in progress, will resume next tick

    Attributes:
        name: Human-readable node name for debugging.
        _status: Current status (from last tick).
        _tick_count: Number of times tick() has been called.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the behavior node.

        Args:
            name: Optional name for debugging. Defaults to class name.
        """
        self._name = name or self.__class__.__name__
        self._status: RunStatus = RunStatus.FAILURE
        self._tick_count: int = 0

    @property
    def name(self) -> str:
        """Get the node's name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the node's name."""
        self._name = value

    @property
    def status(self) -> RunStatus:
        """Get the last execution status."""
        return self._status

    @property
    def tick_count(self) -> int:
        """Get the number of times this node has been ticked."""
        return self._tick_count

    def tick(self, context: TickContext) -> RunStatus:
        """Execute the node and return its status.

        This is the main entry point for node evaluation. It:
        1. Increments the tick counter
        2. Calls the subclass implementation (_tick)
        3. Stores and returns the result

        Args:
            context: The tick context with evaluation data.

        Returns:
            The execution status (SUCCESS, FAILURE, or RUNNING).
        """
        self._tick_count += 1
        self._status = self._tick(context)
        return self._status

    @abstractmethod
    def _tick(self, context: TickContext) -> RunStatus:
        """Subclass implementation of tick behavior.

        This method must be implemented by all node types to define
        their specific behavior.

        Args:
            context: The tick context with evaluation data.

        Returns:
            The execution status.
        """
        pass

    def reset(self) -> None:
        """Reset the node to its initial state.

        Called when the tree needs to restart evaluation, typically:
        - When a parent sequence/selector restarts
        - When explicitly resetting the tree
        - When frame locking cache is invalidated

        Subclasses should override to reset any internal state.
        """
        self._status = RunStatus.FAILURE
        # Don't reset tick_count - it's useful for debugging

    def debug_info(self) -> dict:
        """Get debugging information about this node.

        Returns:
            Dictionary with node debug state.
        """
        return {
            "type": self.__class__.__name__,
            "name": self._name,
            "status": self._status.name,
            "tick_count": self._tick_count,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(name='{self._name}', status={self._status.name})"


class Composite(BehaviorNode):
    """Base class for composite nodes that manage multiple children.

    Composites orchestrate the execution of their children in different ways:
    - Selector: First success wins (OR logic)
    - Sequence: All must succeed (AND logic)
    - Parallel: Run all with configurable success policy

    Attributes:
        children: List of child nodes.
    """

    def __init__(
        self,
        children: Optional[list[BehaviorNode]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the composite node.

        Args:
            children: Initial list of child nodes.
            name: Optional name for debugging.
        """
        super().__init__(name=name)
        self._children: list[BehaviorNode] = children or []

    @property
    def children(self) -> list[BehaviorNode]:
        """Get the list of child nodes."""
        return self._children

    def add_child(self, child: BehaviorNode) -> "Composite":
        """Add a child node.

        Args:
            child: The node to add.

        Returns:
            Self for method chaining.
        """
        self._children.append(child)
        return self

    def add_children(self, children: list[BehaviorNode]) -> "Composite":
        """Add multiple child nodes.

        Args:
            children: List of nodes to add.

        Returns:
            Self for method chaining.
        """
        self._children.extend(children)
        return self

    def remove_child(self, child: BehaviorNode) -> bool:
        """Remove a child node.

        Args:
            child: The node to remove.

        Returns:
            True if removed, False if not found.
        """
        try:
            self._children.remove(child)
            return True
        except ValueError:
            return False

    def clear_children(self) -> None:
        """Remove all child nodes."""
        self._children.clear()

    def reset(self) -> None:
        """Reset this node and all children."""
        super().reset()
        for child in self._children:
            child.reset()

    def debug_info(self) -> dict:
        """Get debugging information including children."""
        info = super().debug_info()
        info["children"] = [child.debug_info() for child in self._children]
        return info

    def __len__(self) -> int:
        """Return the number of children."""
        return len(self._children)

    def __iter__(self):
        """Iterate over children."""
        return iter(self._children)


class Decorator(BehaviorNode):
    """Base class for decorator nodes that wrap a single child.

    Decorators modify the behavior of their child in various ways:
    - Inverter: Flip SUCCESS/FAILURE
    - Succeeder: Always return SUCCESS
    - Guard: Conditional execution
    - Cooldown: Rate limiting

    Attributes:
        child: The wrapped child node.
    """

    def __init__(
        self,
        child: Optional[BehaviorNode] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the decorator node.

        Args:
            child: The node to wrap.
            name: Optional name for debugging.
        """
        super().__init__(name=name)
        self._child = child

    @property
    def child(self) -> Optional[BehaviorNode]:
        """Get the wrapped child node."""
        return self._child

    @child.setter
    def child(self, value: BehaviorNode) -> None:
        """Set the wrapped child node."""
        self._child = value

    def reset(self) -> None:
        """Reset this node and the child."""
        super().reset()
        if self._child:
            self._child.reset()

    def debug_info(self) -> dict:
        """Get debugging information including child."""
        info = super().debug_info()
        if self._child:
            info["child"] = self._child.debug_info()
        return info


class Leaf(BehaviorNode):
    """Base class for leaf nodes that perform actual work.

    Leaf nodes are terminal nodes with no children. They:
    - Evaluate conditions
    - Execute actions
    - Wait for time/events
    - Run scripts

    Leaf nodes typically interact with the RuleContext in TickContext
    to make decisions or trigger effects.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the leaf node.

        Args:
            name: Optional name for debugging.
        """
        super().__init__(name=name)


__all__ = [
    "BehaviorNode",
    "Composite",
    "Decorator",
    "Leaf",
]
