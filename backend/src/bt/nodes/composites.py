"""
BT Composite Nodes - Nodes that orchestrate multiple children.

Implements Sequence, Selector, Parallel, and ForEach composite nodes
per contracts/nodes.yaml specification.

Tasks covered: 2.1.1-2.1.5 from tasks.md

Error codes:
- E2001: Missing required input (from base.py)
- E8001: Parallel merge conflict (from merge.py)

Part of the BT Universal Runtime (spec 019).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from ..state.base import NodeType, RunStatus
from ..state.contracts import NodeContract
from ..state.merge import (
    MergeStrategy,
    MergeResult,
    ParallelMerger,
    apply_merge_result_to_parent,
)
from .base import BehaviorNode

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard

logger = logging.getLogger(__name__)


class CompositeNode(BehaviorNode):
    """Base class for nodes with multiple children.

    Composites orchestrate the execution of their children. Different
    composite types implement different orchestration strategies:
    - Sequence: Run children until one fails
    - Selector: Run children until one succeeds
    - Parallel: Run all children concurrently
    - ForEach: Run children for each item in a collection

    From nodes.yaml:
    - child_count: "1+"
    - node_type: COMPOSITE

    Subclasses must implement _tick() to define their specific behavior.
    """

    def __init__(
        self,
        id: str,
        children: List[BehaviorNode],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a composite node with children.

        Args:
            id: Unique identifier within the tree.
            children: List of child nodes to manage.
            name: Human-readable name (defaults to id).
            metadata: Arbitrary metadata for debugging.

        Raises:
            ValueError: If children list is empty (composites need at least 1 child).
        """
        super().__init__(id=id, name=name, metadata=metadata)

        if not children:
            raise ValueError(
                f"CompositeNode '{id}' requires at least one child node"
            )

        # Set parent reference on each child
        for child in children:
            self._add_child(child)

    @property
    def node_type(self) -> NodeType:
        """Composites have 1+ children."""
        return NodeType.COMPOSITE

    def reset(self) -> None:
        """Reset composite and all children to initial state."""
        super().reset()
        # Subclasses should also reset their internal state


class Sequence(CompositeNode):
    """Execute children in order until one fails or all succeed.

    From nodes.yaml Sequence specification:
    - Tick children left-to-right
    - If child returns FAILURE: return FAILURE immediately
    - If child returns RUNNING: return RUNNING, resume from this child next tick
    - If all children return SUCCESS: return SUCCESS

    Short-circuits on first FAILURE.

    State:
        _current_child_index: Index of child currently executing (reset_to: 0)

    Example:
        >>> seq = Sequence("seq", [action1, action2, action3])
        >>> status = seq.tick(ctx)  # Runs action1
        >>> # If action1 returns RUNNING, next tick resumes there
        >>> # If action1 returns SUCCESS, next tick runs action2
        >>> # If action1 returns FAILURE, sequence returns FAILURE
    """

    def __init__(
        self,
        id: str,
        children: List[BehaviorNode],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize sequence node."""
        super().__init__(id=id, children=children, name=name, metadata=metadata)
        self._current_child_index: int = 0

    @classmethod
    def contract(cls) -> NodeContract:
        """Sequence has no specific inputs/outputs."""
        return NodeContract(
            description="Executes children sequentially until one fails"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Execute children left-to-right until one fails.

        Args:
            ctx: Execution context.

        Returns:
            - FAILURE: If any child returns FAILURE
            - RUNNING: If a child returns RUNNING (will resume there)
            - SUCCESS: If all children return SUCCESS
        """
        while self._current_child_index < len(self._children):
            child = self._children[self._current_child_index]

            # Add to path for debugging
            ctx.push_path(child.id)
            try:
                status = child.tick(ctx)
            finally:
                ctx.pop_path()

            if status == RunStatus.FAILURE:
                # Short-circuit on failure
                return RunStatus.FAILURE
            elif status == RunStatus.RUNNING:
                # Pause at this child, resume next tick
                return RunStatus.RUNNING
            else:
                # SUCCESS - continue to next child
                self._current_child_index += 1

        # All children succeeded
        return RunStatus.SUCCESS

    def reset(self) -> None:
        """Reset sequence to start from first child."""
        super().reset()
        self._current_child_index = 0


class Selector(CompositeNode):
    """Execute children in order until one succeeds or all fail.

    From nodes.yaml Selector specification:
    - Tick children left-to-right
    - If child returns SUCCESS: return SUCCESS immediately
    - If child returns RUNNING: return RUNNING, resume from this child next tick
    - If all children return FAILURE: return FAILURE

    Short-circuits on first SUCCESS.

    State:
        _current_child_index: Index of child currently executing (reset_to: 0)

    Example:
        >>> sel = Selector("sel", [fallback1, fallback2, fallback3])
        >>> status = sel.tick(ctx)
        >>> # Tries fallback1 first, if it fails, tries fallback2, etc.
    """

    def __init__(
        self,
        id: str,
        children: List[BehaviorNode],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize selector node."""
        super().__init__(id=id, children=children, name=name, metadata=metadata)
        self._current_child_index: int = 0

    @classmethod
    def contract(cls) -> NodeContract:
        """Selector has no specific inputs/outputs."""
        return NodeContract(
            description="Tries children until one succeeds"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Execute children left-to-right until one succeeds.

        Args:
            ctx: Execution context.

        Returns:
            - SUCCESS: If any child returns SUCCESS
            - RUNNING: If a child returns RUNNING (will resume there)
            - FAILURE: If all children return FAILURE
        """
        while self._current_child_index < len(self._children):
            child = self._children[self._current_child_index]

            # Add to path for debugging
            ctx.push_path(child.id)
            try:
                status = child.tick(ctx)
            finally:
                ctx.pop_path()

            if status == RunStatus.SUCCESS:
                # Short-circuit on success
                return RunStatus.SUCCESS
            elif status == RunStatus.RUNNING:
                # Pause at this child, resume next tick
                return RunStatus.RUNNING
            else:
                # FAILURE - continue to next child
                self._current_child_index += 1

        # All children failed
        return RunStatus.FAILURE

    def reset(self) -> None:
        """Reset selector to start from first child."""
        super().reset()
        self._current_child_index = 0


class ParallelPolicy(str, Enum):
    """Policy for determining Parallel node success/failure.

    From nodes.yaml Parallel.policies:
    - REQUIRE_ALL: SUCCESS if all succeed, FAILURE if any fail
    - REQUIRE_ONE: SUCCESS if any succeeds, FAILURE if all fail
    - REQUIRE_N: SUCCESS if N succeed, FAILURE if too many fail
    """

    REQUIRE_ALL = "require_all"
    REQUIRE_ONE = "require_one"
    REQUIRE_N = "require_n"


class Parallel(CompositeNode):
    """Execute all children concurrently with configurable success policy.

    From nodes.yaml Parallel specification:
    - Tick ALL children every tick (not truly concurrent, sequential in tick)
    - Each child gets isolated scope (no cross-child visibility mid-tick)
    - After all children tick, merge scopes per merge_strategy
    - Return based on policy

    Per footgun-addendum.md A.3:
    - Each parallel child gets an isolated child scope
    - Writes are merged after all children complete
    - Merge conflicts handled per merge_strategy

    State:
        _child_statuses: List[RunStatus] - Last status of each child
        _merge_strategy: MergeStrategy - How to merge conflicting writes

    Outputs:
        _parallel_conflicts: List[Dict] - Conflict info if FAIL_ON_CONFLICT triggered
        _parallel_child_errors: List[Dict] - Errors from failed children

    Example:
        >>> parallel = Parallel("parallel-search", [
        ...     search_code,
        ...     search_vault,
        ...     search_threads,
        ... ], policy=ParallelPolicy.REQUIRE_ALL)
        >>> status = parallel.tick(ctx)
    """

    def __init__(
        self,
        id: str,
        children: List[BehaviorNode],
        policy: ParallelPolicy = ParallelPolicy.REQUIRE_ALL,
        required_successes: int = 1,
        merge_strategy: MergeStrategy = MergeStrategy.LAST_WINS,
        per_key_strategies: Optional[Dict[str, MergeStrategy]] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize parallel node.

        Args:
            id: Unique identifier.
            children: Child nodes to execute in parallel.
            policy: Success/failure policy (REQUIRE_ALL, REQUIRE_ONE, REQUIRE_N).
            required_successes: N for REQUIRE_N policy.
            merge_strategy: Default strategy for merging child writes.
            per_key_strategies: Per-key merge strategy overrides.
            name: Human-readable name.
            metadata: Debugging metadata.
        """
        super().__init__(id=id, children=children, name=name, metadata=metadata)

        self._policy = policy
        self._required_successes = required_successes
        self._merge_strategy = merge_strategy
        self._merger = ParallelMerger(
            default_strategy=merge_strategy,
            per_key_strategies=per_key_strategies or {},
        )

        # State tracking
        self._child_statuses: List[RunStatus] = [
            RunStatus.FRESH for _ in children
        ]

    @classmethod
    def contract(cls) -> NodeContract:
        """Parallel may output conflict info."""
        return NodeContract(
            description="Executes all children concurrently",
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Execute all children with isolated scopes, then merge.

        Per footgun-addendum.md A.3:
        1. Create isolated scope per child
        2. Tick each child with its isolated scope
        3. Merge scopes after all children complete
        4. Return based on policy

        Args:
            ctx: Execution context.

        Returns:
            Status based on policy evaluation.
        """
        child_scopes: List["TypedBlackboard"] = []
        child_statuses: List[RunStatus] = []

        # Step 1 & 2: Create isolated scope per child and tick
        for i, child in enumerate(self._children):
            # Create isolated child scope (A.3 scope isolation)
            child_scope = ctx.blackboard.create_child_scope(
                f"parallel_{self._id}_{i}"
            )
            child_ctx = ctx.with_blackboard(child_scope)

            # Add to path for debugging
            ctx.push_path(child.id)
            try:
                status = child.tick(child_ctx)
            finally:
                ctx.pop_path()

            child_scopes.append(child_scope)
            child_statuses.append(status)
            self._child_statuses[i] = status

        # Step 3: Merge scopes if we should (any completed children)
        if self._should_merge(child_statuses):
            merge_result = self._merger.merge(ctx.blackboard, child_scopes)

            if not merge_result.success:
                # Merge failed - likely FAIL_ON_CONFLICT
                ctx.blackboard.set_internal(
                    "_parallel_conflicts",
                    [c.to_dict() for c in merge_result.conflicts]
                )
                logger.warning(
                    f"Parallel '{self._id}' merge failed: {len(merge_result.conflicts)} conflicts"
                )
                return RunStatus.FAILURE

            # Apply merged data to parent blackboard
            apply_result = apply_merge_result_to_parent(ctx.blackboard, merge_result)
            if apply_result.is_error:
                logger.warning(
                    f"Parallel '{self._id}' failed to apply merge: {apply_result.error}"
                )

        # Step 4: Evaluate policy
        return self._evaluate_policy(child_statuses)

    def _should_merge(self, statuses: List[RunStatus]) -> bool:
        """Check if we should attempt to merge child scopes.

        Returns True if at least one child completed (not RUNNING or FRESH).
        """
        return any(s.is_complete() for s in statuses)

    def _evaluate_policy(self, statuses: List[RunStatus]) -> RunStatus:
        """Evaluate success/failure based on policy.

        Args:
            statuses: List of child statuses.

        Returns:
            Aggregate status based on policy.
        """
        success_count = sum(1 for s in statuses if s == RunStatus.SUCCESS)
        failure_count = sum(1 for s in statuses if s == RunStatus.FAILURE)
        running_count = sum(1 for s in statuses if s == RunStatus.RUNNING)

        if self._policy == ParallelPolicy.REQUIRE_ALL:
            # SUCCESS if all succeed, FAILURE if any fail
            if failure_count > 0:
                return RunStatus.FAILURE
            elif running_count > 0:
                return RunStatus.RUNNING
            else:
                return RunStatus.SUCCESS

        elif self._policy == ParallelPolicy.REQUIRE_ONE:
            # SUCCESS if any succeeds, FAILURE if all fail
            if success_count > 0:
                return RunStatus.SUCCESS
            elif failure_count == len(statuses):
                return RunStatus.FAILURE
            else:
                return RunStatus.RUNNING

        elif self._policy == ParallelPolicy.REQUIRE_N:
            # SUCCESS if N succeed, FAILURE if too many fail
            max_possible_successes = success_count + running_count
            if success_count >= self._required_successes:
                return RunStatus.SUCCESS
            elif max_possible_successes < self._required_successes:
                # Can't possibly reach N successes anymore
                return RunStatus.FAILURE
            else:
                return RunStatus.RUNNING

        # Fallback
        return RunStatus.RUNNING

    def reset(self) -> None:
        """Reset parallel node and all children."""
        super().reset()
        self._child_statuses = [RunStatus.FRESH for _ in self._children]


class ForEach(CompositeNode):
    """Iterate over collection, executing children for each item.

    From nodes.yaml ForEach specification:
    - Read collection from blackboard key
    - For each item, set iteration variable and tick children
    - If empty collection: return SUCCESS (not FAILURE) unless min_items not met
    - If child returns FAILURE: depends on continue_on_failure option

    Contract:
        inputs: collection_key (List[Any]) - Key containing iterable
        outputs: _for_each_results (List[Any]) - Results from each iteration

    Options:
        continue_on_failure: bool (default: False) - Continue to next item if child fails
        min_items: int (default: 0) - Minimum items required (0 = empty OK)

    Example:
        >>> foreach = ForEach(
        ...     "process-results",
        ...     collection_key="search_results",
        ...     item_key="current_result",
        ...     children=[process_result],
        ...     continue_on_failure=True,
        ... )
    """

    def __init__(
        self,
        id: str,
        children: List[BehaviorNode],
        collection_key: str,
        item_key: str,
        continue_on_failure: bool = False,
        min_items: int = 0,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ForEach node.

        Args:
            id: Unique identifier.
            children: Child nodes to execute for each item (typically Sequence).
            collection_key: Blackboard key containing the collection to iterate.
            item_key: Blackboard key to write current item to.
            continue_on_failure: Continue to next item if child fails.
            min_items: Minimum items required in collection.
            name: Human-readable name.
            metadata: Debugging metadata.
        """
        super().__init__(id=id, children=children, name=name, metadata=metadata)

        self._collection_key = collection_key
        self._item_key = item_key
        self._continue_on_failure = continue_on_failure
        self._min_items = min_items

        # Iteration state
        self._current_index: int = 0
        self._iteration_results: List[Any] = []
        self._collection: Optional[List[Any]] = None

    @classmethod
    def contract(cls) -> NodeContract:
        """ForEach requires collection input and outputs results."""
        return NodeContract(
            description="Iterates over collection, executing children for each item"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Execute children for each item in collection.

        Args:
            ctx: Execution context.

        Returns:
            - SUCCESS: All items processed successfully (or empty with min_items=0)
            - FAILURE: Child failed (if continue_on_failure=False) or min_items not met
            - RUNNING: Still processing items
        """
        # First tick: fetch collection
        if self._collection is None:
            self._collection = self._get_collection(ctx)

            if self._collection is None:
                logger.warning(
                    f"ForEach '{self._id}': collection key '{self._collection_key}' not found"
                )
                return RunStatus.FAILURE

            # Check min_items
            if len(self._collection) < self._min_items:
                logger.warning(
                    f"ForEach '{self._id}': collection has {len(self._collection)} items, "
                    f"minimum required is {self._min_items}"
                )
                return RunStatus.FAILURE

            # Empty collection with min_items=0 is SUCCESS
            if len(self._collection) == 0:
                return RunStatus.SUCCESS

        # Process current item
        while self._current_index < len(self._collection):
            item = self._collection[self._current_index]

            # Set current item in blackboard
            self._set_current_item(ctx, item)

            # Execute children (typically a sequence)
            child_status = self._tick_children(ctx)

            if child_status == RunStatus.RUNNING:
                # Child still running, will continue next tick
                return RunStatus.RUNNING

            elif child_status == RunStatus.FAILURE:
                if self._continue_on_failure:
                    # Track failure but continue
                    self._iteration_results.append({
                        "index": self._current_index,
                        "status": "failure",
                        "item": item,
                    })
                    self._current_index += 1
                else:
                    # Stop on failure
                    return RunStatus.FAILURE

            else:
                # SUCCESS - track and continue
                self._iteration_results.append({
                    "index": self._current_index,
                    "status": "success",
                    "item": item,
                })
                self._current_index += 1

        # All items processed
        ctx.blackboard.set_internal("_for_each_results", self._iteration_results)
        return RunStatus.SUCCESS

    def _get_collection(self, ctx: "TickContext") -> Optional[List[Any]]:
        """Get collection from blackboard.

        Attempts to get the collection value. Supports various types
        that can be iterated.

        Returns:
            List of items or None if not found.
        """
        # Direct lookup without schema (for flexibility)
        # Check both regular keys and internal keys (starting with _)
        value = None
        if ctx.blackboard.has(self._collection_key):
            value = ctx.blackboard._lookup(self._collection_key)
        elif self._collection_key in ctx.blackboard._data:
            # Also check internal keys directly
            value = ctx.blackboard._data[self._collection_key]

        if value is not None:
            # Convert to list if needed
            if isinstance(value, list):
                return value
            elif isinstance(value, (tuple, set)):
                return list(value)
            elif hasattr(value, "__iter__") and not isinstance(value, (str, dict)):
                return list(value)
            elif isinstance(value, dict):
                # Dict iteration returns items
                return list(value.items())

        return None

    def _set_current_item(self, ctx: "TickContext", item: Any) -> None:
        """Set current item in blackboard.

        Uses set_internal since item_key may not be registered.
        """
        # Use internal key with underscore prefix for iteration variable
        ctx.blackboard.set_internal(f"_{self._item_key}", item)

    def _tick_children(self, ctx: "TickContext") -> RunStatus:
        """Tick all children as a sequence.

        If there's only one child, ticks it directly.
        If multiple children, executes them as a sequence.

        Returns:
            Combined status from children.
        """
        if len(self._children) == 1:
            child = self._children[0]
            ctx.push_path(child.id)
            try:
                return child.tick(ctx)
            finally:
                ctx.pop_path()

        # Multiple children - run as sequence
        for child in self._children:
            ctx.push_path(child.id)
            try:
                status = child.tick(ctx)
            finally:
                ctx.pop_path()

            if status != RunStatus.SUCCESS:
                return status

        return RunStatus.SUCCESS

    def reset(self) -> None:
        """Reset ForEach to start from beginning."""
        super().reset()
        self._current_index = 0
        self._iteration_results = []
        self._collection = None


__all__ = [
    "CompositeNode",
    "Sequence",
    "Selector",
    "Parallel",
    "ParallelPolicy",
    "ForEach",
]
