"""Composite nodes for behavior tree orchestration.

This module implements the core composite nodes that control child execution:
- PrioritySelector: First success wins (priority-ordered OR)
- Sequence: All must succeed (AND)
- Parallel: Run all with configurable success policy

Based on Honorbuddy patterns (research.md Section 7.2):
- Priority-based selection reduces conditional nesting
- Short-circuit evaluation for performance
- Running state support for multi-tick operations
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional
import logging

from .types import RunStatus, TickContext
from .node import Composite, BehaviorNode


logger = logging.getLogger(__name__)


class PrioritySelector(Composite):
    """Selector that evaluates children in priority order.

    Evaluates children from first to last (highest to lowest priority).
    Returns SUCCESS on first child that succeeds.
    Returns RUNNING if a child is still running.
    Returns FAILURE only if ALL children fail.

    This is the core decision node for rule evaluation:
    - Rules are sorted by priority (highest first)
    - First matching rule's action is executed
    - Lower priority rules are skipped

    Honorbuddy pattern: PrioritySelector with cached running child.

    Example:
        >>> selector = PrioritySelector([
        ...     Guard(condition="token_usage > 0.9", child=ActionNode(action)),  # P0
        ...     Guard(condition="token_usage > 0.7", child=ActionNode(action)),  # P1
        ...     ActionNode(default_action),  # P2 fallback
        ... ])
        >>> status = selector.tick(context)
    """

    def __init__(
        self,
        children: Optional[list[BehaviorNode]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the priority selector.

        Args:
            children: Child nodes in priority order (highest first).
            name: Optional name for debugging.
        """
        super().__init__(children=children, name=name or "PrioritySelector")
        self._running_child_index: int = -1

    def _tick(self, context: TickContext) -> RunStatus:
        """Evaluate children in order, return first success.

        Optimization: If a child was RUNNING on previous tick, resume from there
        instead of re-evaluating higher-priority children.

        Args:
            context: The tick context.

        Returns:
            SUCCESS if any child succeeds.
            RUNNING if a child is running.
            FAILURE if all children fail.
        """
        if not self._children:
            logger.debug(f"{self._name}: No children, returning FAILURE")
            return RunStatus.FAILURE

        # Frame locking: resume from running child if valid
        start_index = 0
        if self._running_child_index >= 0:
            start_index = self._running_child_index
            logger.debug(
                f"{self._name}: Resuming from child {start_index} "
                f"({self._children[start_index].name})"
            )

        for i in range(start_index, len(self._children)):
            child = self._children[i]
            status = child.tick(context)

            if status == RunStatus.SUCCESS:
                self._running_child_index = -1
                logger.debug(
                    f"{self._name}: Child {i} ({child.name}) succeeded"
                )
                return RunStatus.SUCCESS

            if status == RunStatus.RUNNING:
                self._running_child_index = i
                logger.debug(
                    f"{self._name}: Child {i} ({child.name}) running"
                )
                return RunStatus.RUNNING

            # Child failed, try next
            logger.debug(f"{self._name}: Child {i} ({child.name}) failed")

        # All children failed
        self._running_child_index = -1
        return RunStatus.FAILURE

    def reset(self) -> None:
        """Reset selector and clear running child cache."""
        super().reset()
        self._running_child_index = -1

    def debug_info(self) -> dict:
        """Include running child index in debug info."""
        info = super().debug_info()
        info["running_child_index"] = self._running_child_index
        return info


class Sequence(Composite):
    """Sequence that requires all children to succeed.

    Evaluates children from first to last.
    Returns FAILURE immediately if any child fails (fail-fast).
    Returns RUNNING if a child is still running.
    Returns SUCCESS only if ALL children succeed.

    Use for chained actions that must all complete:
    - Check preconditions, then execute action
    - Multi-step workflows

    Example:
        >>> sequence = Sequence([
        ...     ConditionNode("has_permission"),
        ...     ConditionNode("file_exists"),
        ...     ActionNode(write_action),
        ... ])
        >>> status = sequence.tick(context)
    """

    def __init__(
        self,
        children: Optional[list[BehaviorNode]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the sequence.

        Args:
            children: Child nodes in execution order.
            name: Optional name for debugging.
        """
        super().__init__(children=children, name=name or "Sequence")
        self._current_child_index: int = 0

    def _tick(self, context: TickContext) -> RunStatus:
        """Evaluate children in order, all must succeed.

        Args:
            context: The tick context.

        Returns:
            SUCCESS if all children succeed.
            RUNNING if a child is running.
            FAILURE if any child fails.
        """
        if not self._children:
            logger.debug(f"{self._name}: No children, returning SUCCESS")
            return RunStatus.SUCCESS

        # Continue from current position
        for i in range(self._current_child_index, len(self._children)):
            child = self._children[i]
            status = child.tick(context)

            if status == RunStatus.FAILURE:
                self._current_child_index = 0  # Reset for next run
                logger.debug(
                    f"{self._name}: Child {i} ({child.name}) failed"
                )
                return RunStatus.FAILURE

            if status == RunStatus.RUNNING:
                self._current_child_index = i
                logger.debug(
                    f"{self._name}: Child {i} ({child.name}) running"
                )
                return RunStatus.RUNNING

            # Child succeeded, continue to next
            logger.debug(f"{self._name}: Child {i} ({child.name}) succeeded")

        # All children succeeded
        self._current_child_index = 0  # Reset for next run
        return RunStatus.SUCCESS

    def reset(self) -> None:
        """Reset sequence and restart from first child."""
        super().reset()
        self._current_child_index = 0

    def debug_info(self) -> dict:
        """Include current child index in debug info."""
        info = super().debug_info()
        info["current_child_index"] = self._current_child_index
        return info


class ParallelPolicy(Enum):
    """Policy for determining Parallel node success/failure.

    - REQUIRE_ONE: Success if at least one child succeeds
    - REQUIRE_ALL: Success only if all children succeed
    """

    REQUIRE_ONE = auto()
    REQUIRE_ALL = auto()


class Parallel(Composite):
    """Parallel node that executes all children each tick.

    Unlike Selector/Sequence, Parallel ticks ALL children every time.
    The success/failure policy determines when to return:

    REQUIRE_ONE (default):
        - Returns SUCCESS if any child succeeds
        - Returns FAILURE only if all children fail

    REQUIRE_ALL:
        - Returns SUCCESS only if all children succeed
        - Returns FAILURE if any child fails

    Both policies:
        - Returns RUNNING if any child is RUNNING and policy not yet met

    Example:
        >>> parallel = Parallel(
        ...     [
        ...         ActionNode(action1),
        ...         ActionNode(action2),
        ...     ],
        ...     policy=ParallelPolicy.REQUIRE_ALL,
        ... )
    """

    def __init__(
        self,
        children: Optional[list[BehaviorNode]] = None,
        name: Optional[str] = None,
        policy: ParallelPolicy = ParallelPolicy.REQUIRE_ONE,
    ) -> None:
        """Initialize the parallel node.

        Args:
            children: Child nodes to execute in parallel.
            name: Optional name for debugging.
            policy: Success/failure policy.
        """
        super().__init__(children=children, name=name or "Parallel")
        self._policy = policy
        self._child_statuses: list[RunStatus] = []

    @property
    def policy(self) -> ParallelPolicy:
        """Get the parallel policy."""
        return self._policy

    def _tick(self, context: TickContext) -> RunStatus:
        """Tick all children and evaluate based on policy.

        Args:
            context: The tick context.

        Returns:
            Status based on policy evaluation.
        """
        if not self._children:
            return RunStatus.SUCCESS

        # Tick all children
        self._child_statuses = []
        for child in self._children:
            status = child.tick(context)
            self._child_statuses.append(status)

        # Count results
        success_count = sum(1 for s in self._child_statuses if s == RunStatus.SUCCESS)
        failure_count = sum(1 for s in self._child_statuses if s == RunStatus.FAILURE)
        running_count = sum(1 for s in self._child_statuses if s == RunStatus.RUNNING)

        logger.debug(
            f"{self._name}: success={success_count}, "
            f"failure={failure_count}, running={running_count}"
        )

        if self._policy == ParallelPolicy.REQUIRE_ONE:
            # Success if any succeeded
            if success_count > 0:
                return RunStatus.SUCCESS
            # All failed = failure
            if failure_count == len(self._children):
                return RunStatus.FAILURE
            # Some running, none succeeded yet
            return RunStatus.RUNNING

        else:  # REQUIRE_ALL
            # Any failure = immediate failure
            if failure_count > 0:
                return RunStatus.FAILURE
            # All succeeded
            if success_count == len(self._children):
                return RunStatus.SUCCESS
            # Some running, none failed yet
            return RunStatus.RUNNING

    def reset(self) -> None:
        """Reset parallel and clear status tracking."""
        super().reset()
        self._child_statuses.clear()

    def debug_info(self) -> dict:
        """Include policy and child statuses in debug info."""
        info = super().debug_info()
        info["policy"] = self._policy.name
        info["child_statuses"] = [s.name for s in self._child_statuses]
        return info


class MemorySelector(Composite):
    """Selector that remembers which child was running.

    Unlike PrioritySelector which may re-evaluate higher priority children
    when a child completes, MemorySelector continues from where it left off
    until success or all children exhausted.

    Use when you want to ensure each child gets a fair chance without
    higher priority children preempting.
    """

    def __init__(
        self,
        children: Optional[list[BehaviorNode]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the memory selector.

        Args:
            children: Child nodes.
            name: Optional name for debugging.
        """
        super().__init__(children=children, name=name or "MemorySelector")
        self._last_child_index: int = 0

    def _tick(self, context: TickContext) -> RunStatus:
        """Evaluate from last position, remembering progress.

        Args:
            context: The tick context.

        Returns:
            SUCCESS if any child succeeds.
            RUNNING if a child is running.
            FAILURE if all children from current position fail.
        """
        if not self._children:
            return RunStatus.FAILURE

        for i in range(self._last_child_index, len(self._children)):
            child = self._children[i]
            status = child.tick(context)

            if status == RunStatus.SUCCESS:
                self._last_child_index = 0  # Reset on success
                return RunStatus.SUCCESS

            if status == RunStatus.RUNNING:
                self._last_child_index = i
                return RunStatus.RUNNING

            # Child failed, remember position and continue
            self._last_child_index = i + 1

        # All children from position failed
        self._last_child_index = 0  # Reset for next run
        return RunStatus.FAILURE

    def reset(self) -> None:
        """Reset and restart from first child."""
        super().reset()
        self._last_child_index = 0


class MemorySequence(Composite):
    """Sequence that remembers progress across ticks.

    Standard Sequence resets on success, but MemorySequence
    maintains progress even when a child fails, allowing retry
    from the failed child on next invocation.
    """

    def __init__(
        self,
        children: Optional[list[BehaviorNode]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the memory sequence.

        Args:
            children: Child nodes.
            name: Optional name for debugging.
        """
        super().__init__(children=children, name=name or "MemorySequence")
        self._last_child_index: int = 0

    def _tick(self, context: TickContext) -> RunStatus:
        """Evaluate from last position, remembering progress.

        Args:
            context: The tick context.

        Returns:
            SUCCESS if all children succeed.
            RUNNING if a child is running.
            FAILURE if current child fails (but position remembered).
        """
        if not self._children:
            return RunStatus.SUCCESS

        for i in range(self._last_child_index, len(self._children)):
            child = self._children[i]
            status = child.tick(context)

            if status == RunStatus.FAILURE:
                # Remember position for retry
                self._last_child_index = i
                return RunStatus.FAILURE

            if status == RunStatus.RUNNING:
                self._last_child_index = i
                return RunStatus.RUNNING

            # Child succeeded, continue

        # All children succeeded
        self._last_child_index = 0  # Reset on full success
        return RunStatus.SUCCESS

    def reset(self) -> None:
        """Reset and restart from first child."""
        super().reset()
        self._last_child_index = 0


__all__ = [
    "PrioritySelector",
    "Sequence",
    "Parallel",
    "ParallelPolicy",
    "MemorySelector",
    "MemorySequence",
]
