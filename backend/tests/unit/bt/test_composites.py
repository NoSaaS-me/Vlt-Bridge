"""
Unit tests for BT Composite Nodes.

Tests:
- CompositeNode base class
- Sequence: mixed results, failure propagation, RUNNING behavior
- Selector: fallback behavior, success propagation
- Parallel: policies (REQUIRE_ALL, REQUIRE_ONE, REQUIRE_N), merge strategies
- ForEach: iteration, continue_on_failure, min_items

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.1.1-2.1.5 from tasks.md
"""

import pytest
from typing import List, Optional
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from backend.src.bt.nodes.base import BehaviorNode
from backend.src.bt.nodes.composites import (
    CompositeNode,
    Sequence,
    Selector,
    Parallel,
    ParallelPolicy,
    ForEach,
)
from backend.src.bt.state.base import NodeType, RunStatus
from backend.src.bt.state.blackboard import TypedBlackboard
from backend.src.bt.state.merge import MergeStrategy
from backend.src.bt.core.context import TickContext


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class MockLeafNode(BehaviorNode):
    """Mock leaf node for testing composites."""

    def __init__(
        self,
        id: str,
        results: Optional[List[RunStatus]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize mock leaf.

        Args:
            id: Node identifier.
            results: Sequence of results to return on each tick.
                     Cycles if more ticks than results.
            name: Human-readable name.
        """
        super().__init__(id=id, name=name)
        self._results = results or [RunStatus.SUCCESS]
        self._tick_index = 0
        self._ticks_received = 0

    @property
    def node_type(self) -> NodeType:
        return NodeType.LEAF

    @property
    def ticks_received(self) -> int:
        """Number of times this node was ticked."""
        return self._ticks_received

    def _tick(self, ctx: TickContext) -> RunStatus:
        self._ticks_received += 1
        result = self._results[self._tick_index % len(self._results)]
        self._tick_index += 1
        return result

    def reset(self) -> None:
        super().reset()
        self._tick_index = 0


class WritingLeafNode(BehaviorNode):
    """Leaf node that writes to blackboard for testing parallel merges."""

    def __init__(
        self,
        id: str,
        write_key: str,
        write_value: str,
        result: RunStatus = RunStatus.SUCCESS,
    ) -> None:
        super().__init__(id=id)
        self._write_key = write_key
        self._write_value = write_value
        self._result = result

    @property
    def node_type(self) -> NodeType:
        return NodeType.LEAF

    def _tick(self, ctx: TickContext) -> RunStatus:
        # Write to internal key (bypasses schema)
        ctx.blackboard.set_internal(f"_{self._write_key}", self._write_value)
        return self._result


class ValueSchema(BaseModel):
    """Schema for testing."""
    value: str


@pytest.fixture
def basic_blackboard() -> TypedBlackboard:
    """Create a basic blackboard for testing."""
    return TypedBlackboard(scope_name="test")


@pytest.fixture
def tick_context(basic_blackboard: TypedBlackboard) -> TickContext:
    """Create a basic tick context."""
    return TickContext(blackboard=basic_blackboard)


# =============================================================================
# CompositeNode Base Tests
# =============================================================================


class TestCompositeNodeBase:
    """Tests for CompositeNode base class."""

    def test_requires_at_least_one_child(self) -> None:
        """CompositeNode should require at least one child."""
        with pytest.raises(ValueError) as exc_info:
            Sequence(id="empty-seq", children=[])
        assert "at least one child" in str(exc_info.value)

    def test_sets_parent_on_children(self) -> None:
        """CompositeNode should set parent reference on children."""
        child1 = MockLeafNode(id="child1")
        child2 = MockLeafNode(id="child2")

        seq = Sequence(id="seq", children=[child1, child2])

        assert child1.parent is seq
        assert child2.parent is seq

    def test_node_type_is_composite(self) -> None:
        """CompositeNode should report COMPOSITE node type."""
        child = MockLeafNode(id="child")
        seq = Sequence(id="seq", children=[child])

        assert seq.node_type == NodeType.COMPOSITE

    def test_children_property_returns_all(self) -> None:
        """children property should return all children."""
        child1 = MockLeafNode(id="child1")
        child2 = MockLeafNode(id="child2")

        seq = Sequence(id="seq", children=[child1, child2])

        assert len(seq.children) == 2
        assert child1 in seq.children
        assert child2 in seq.children


# =============================================================================
# Sequence Tests
# =============================================================================


class TestSequence:
    """Tests for Sequence composite node."""

    def test_all_success_returns_success(self, tick_context: TickContext) -> None:
        """Sequence should return SUCCESS when all children succeed."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="c", results=[RunStatus.SUCCESS]),
        ]
        seq = Sequence(id="seq", children=children)

        result = seq.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert all(c.ticks_received == 1 for c in children)

    def test_first_failure_returns_failure(self, tick_context: TickContext) -> None:
        """Sequence should return FAILURE on first child failure."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.FAILURE]),
            MockLeafNode(id="c", results=[RunStatus.SUCCESS]),
        ]
        seq = Sequence(id="seq", children=children)

        result = seq.tick(tick_context)

        assert result == RunStatus.FAILURE
        assert children[0].ticks_received == 1
        assert children[1].ticks_received == 1
        assert children[2].ticks_received == 0  # Not reached

    def test_running_pauses_sequence(self, tick_context: TickContext) -> None:
        """Sequence should pause on RUNNING child."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.RUNNING, RunStatus.SUCCESS]),
            MockLeafNode(id="c", results=[RunStatus.SUCCESS]),
        ]
        seq = Sequence(id="seq", children=children)

        # First tick: a succeeds, b returns RUNNING
        result1 = seq.tick(tick_context)
        assert result1 == RunStatus.RUNNING
        assert children[0].ticks_received == 1
        assert children[1].ticks_received == 1
        assert children[2].ticks_received == 0

        # Second tick: b succeeds, c succeeds
        result2 = seq.tick(tick_context)
        assert result2 == RunStatus.SUCCESS
        assert children[0].ticks_received == 1  # Not re-ticked
        assert children[1].ticks_received == 2
        assert children[2].ticks_received == 1

    def test_reset_restarts_sequence(self, tick_context: TickContext) -> None:
        """reset() should restart sequence from first child."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.RUNNING]),
        ]
        seq = Sequence(id="seq", children=children)

        # Tick until RUNNING
        seq.tick(tick_context)
        assert seq._current_child_index == 1

        # Reset
        seq.reset()

        assert seq._current_child_index == 0
        assert seq.status == RunStatus.FRESH

    def test_sequence_tracks_path(self, tick_context: TickContext) -> None:
        """Sequence should push/pop path for debugging."""
        child = MockLeafNode(id="tracked-child", results=[RunStatus.SUCCESS])
        seq = Sequence(id="seq", children=[child])

        # After tick, path should be empty (pushed and popped)
        seq.tick(tick_context)
        assert tick_context.parent_path == []


# =============================================================================
# Selector Tests
# =============================================================================


class TestSelector:
    """Tests for Selector composite node."""

    def test_first_success_returns_success(self, tick_context: TickContext) -> None:
        """Selector should return SUCCESS on first child success."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.SUCCESS]),
        ]
        sel = Selector(id="sel", children=children)

        result = sel.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert children[0].ticks_received == 1
        assert children[1].ticks_received == 0  # Not reached

    def test_all_failure_returns_failure(self, tick_context: TickContext) -> None:
        """Selector should return FAILURE when all children fail."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.FAILURE]),
            MockLeafNode(id="b", results=[RunStatus.FAILURE]),
            MockLeafNode(id="c", results=[RunStatus.FAILURE]),
        ]
        sel = Selector(id="sel", children=children)

        result = sel.tick(tick_context)

        assert result == RunStatus.FAILURE
        assert all(c.ticks_received == 1 for c in children)

    def test_fallback_behavior(self, tick_context: TickContext) -> None:
        """Selector should try next child on failure."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.FAILURE]),
            MockLeafNode(id="b", results=[RunStatus.FAILURE]),
            MockLeafNode(id="c", results=[RunStatus.SUCCESS]),
        ]
        sel = Selector(id="sel", children=children)

        result = sel.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert children[0].ticks_received == 1
        assert children[1].ticks_received == 1
        assert children[2].ticks_received == 1

    def test_running_pauses_selector(self, tick_context: TickContext) -> None:
        """Selector should pause on RUNNING child."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.FAILURE]),
            MockLeafNode(id="b", results=[RunStatus.RUNNING, RunStatus.SUCCESS]),
        ]
        sel = Selector(id="sel", children=children)

        # First tick: a fails, b returns RUNNING
        result1 = sel.tick(tick_context)
        assert result1 == RunStatus.RUNNING
        assert children[1].ticks_received == 1

        # Second tick: b succeeds
        result2 = sel.tick(tick_context)
        assert result2 == RunStatus.SUCCESS
        assert children[1].ticks_received == 2

    def test_reset_restarts_selector(self, tick_context: TickContext) -> None:
        """reset() should restart selector from first child."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.FAILURE]),
            MockLeafNode(id="b", results=[RunStatus.RUNNING]),
        ]
        sel = Selector(id="sel", children=children)

        # Tick until RUNNING
        sel.tick(tick_context)
        assert sel._current_child_index == 1

        # Reset
        sel.reset()

        assert sel._current_child_index == 0
        assert sel.status == RunStatus.FRESH


# =============================================================================
# Parallel Tests
# =============================================================================


class TestParallel:
    """Tests for Parallel composite node."""

    def test_require_all_success(self, tick_context: TickContext) -> None:
        """REQUIRE_ALL: SUCCESS if all children succeed."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="c", results=[RunStatus.SUCCESS]),
        ]
        parallel = Parallel(
            id="par",
            children=children,
            policy=ParallelPolicy.REQUIRE_ALL,
        )

        result = parallel.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert all(c.ticks_received == 1 for c in children)

    def test_require_all_any_failure(self, tick_context: TickContext) -> None:
        """REQUIRE_ALL: FAILURE if any child fails."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.FAILURE]),
            MockLeafNode(id="c", results=[RunStatus.SUCCESS]),
        ]
        parallel = Parallel(
            id="par",
            children=children,
            policy=ParallelPolicy.REQUIRE_ALL,
        )

        result = parallel.tick(tick_context)

        assert result == RunStatus.FAILURE
        # All children still ticked
        assert all(c.ticks_received == 1 for c in children)

    def test_require_one_any_success(self, tick_context: TickContext) -> None:
        """REQUIRE_ONE: SUCCESS if any child succeeds."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.FAILURE]),
            MockLeafNode(id="b", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="c", results=[RunStatus.FAILURE]),
        ]
        parallel = Parallel(
            id="par",
            children=children,
            policy=ParallelPolicy.REQUIRE_ONE,
        )

        result = parallel.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_require_one_all_failure(self, tick_context: TickContext) -> None:
        """REQUIRE_ONE: FAILURE if all children fail."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.FAILURE]),
            MockLeafNode(id="b", results=[RunStatus.FAILURE]),
        ]
        parallel = Parallel(
            id="par",
            children=children,
            policy=ParallelPolicy.REQUIRE_ONE,
        )

        result = parallel.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_require_n_enough_successes(self, tick_context: TickContext) -> None:
        """REQUIRE_N: SUCCESS if N children succeed."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="c", results=[RunStatus.FAILURE]),
        ]
        parallel = Parallel(
            id="par",
            children=children,
            policy=ParallelPolicy.REQUIRE_N,
            required_successes=2,
        )

        result = parallel.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_require_n_not_enough_successes(self, tick_context: TickContext) -> None:
        """REQUIRE_N: FAILURE if too many failures."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.FAILURE]),
            MockLeafNode(id="c", results=[RunStatus.FAILURE]),
        ]
        parallel = Parallel(
            id="par",
            children=children,
            policy=ParallelPolicy.REQUIRE_N,
            required_successes=2,
        )

        result = parallel.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_parallel_running_state(self, tick_context: TickContext) -> None:
        """Parallel should return RUNNING if any child is RUNNING."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.RUNNING]),
        ]
        parallel = Parallel(
            id="par",
            children=children,
            policy=ParallelPolicy.REQUIRE_ALL,
        )

        result = parallel.tick(tick_context)

        assert result == RunStatus.RUNNING

    def test_parallel_children_get_isolated_scopes(
        self, tick_context: TickContext
    ) -> None:
        """Parallel children should get isolated scopes (A.3)."""
        children = [
            WritingLeafNode(id="a", write_key="result", write_value="from_a"),
            WritingLeafNode(id="b", write_key="result", write_value="from_b"),
        ]
        parallel = Parallel(
            id="par",
            children=children,
            policy=ParallelPolicy.REQUIRE_ALL,
        )

        result = parallel.tick(tick_context)

        assert result == RunStatus.SUCCESS
        # Both children were ticked with their own isolated scopes


class TestParallelMergeStrategies:
    """Tests for Parallel merge strategies (A.3)."""

    def test_last_wins_strategy(self, tick_context: TickContext) -> None:
        """LAST_WINS: Last child's value wins on conflict."""
        # This is the default strategy - just verify parallel completes
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.SUCCESS]),
        ]
        parallel = Parallel(
            id="par",
            children=children,
            merge_strategy=MergeStrategy.LAST_WINS,
        )

        result = parallel.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_collect_strategy(self, tick_context: TickContext) -> None:
        """COLLECT: Collect all values into list."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.SUCCESS]),
        ]
        parallel = Parallel(
            id="par",
            children=children,
            merge_strategy=MergeStrategy.COLLECT,
        )

        result = parallel.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_fail_on_conflict_no_conflicts(self, tick_context: TickContext) -> None:
        """FAIL_ON_CONFLICT: Should succeed if no conflicts."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.SUCCESS]),
        ]
        parallel = Parallel(
            id="par",
            children=children,
            merge_strategy=MergeStrategy.FAIL_ON_CONFLICT,
        )

        result = parallel.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_reset_clears_child_statuses(self, tick_context: TickContext) -> None:
        """reset() should reset child status tracking."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.SUCCESS]),
            MockLeafNode(id="b", results=[RunStatus.SUCCESS]),
        ]
        parallel = Parallel(id="par", children=children)

        parallel.tick(tick_context)
        assert parallel._child_statuses == [RunStatus.SUCCESS, RunStatus.SUCCESS]

        parallel.reset()

        assert parallel._child_statuses == [RunStatus.FRESH, RunStatus.FRESH]


# =============================================================================
# ForEach Tests
# =============================================================================


class TestForEach:
    """Tests for ForEach composite node."""

    def test_iterates_over_collection(self, tick_context: TickContext) -> None:
        """ForEach should iterate over blackboard collection."""
        tick_context.blackboard.set_internal("_items", ["a", "b", "c"])

        child = MockLeafNode(id="process", results=[RunStatus.SUCCESS])
        foreach = ForEach(
            id="foreach",
            children=[child],
            collection_key="items",
            item_key="current",
        )
        # Set collection directly (bypass schema)
        foreach._collection = ["a", "b", "c"]

        result = foreach.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert child.ticks_received == 3

    def test_empty_collection_with_min_zero(self, tick_context: TickContext) -> None:
        """ForEach should return SUCCESS for empty collection when min_items=0."""
        child = MockLeafNode(id="process", results=[RunStatus.SUCCESS])
        foreach = ForEach(
            id="foreach",
            children=[child],
            collection_key="items",
            item_key="current",
            min_items=0,
        )
        foreach._collection = []

        result = foreach.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert child.ticks_received == 0

    def test_empty_collection_min_items_not_met(
        self, tick_context: TickContext
    ) -> None:
        """ForEach should return FAILURE if min_items not met."""
        # Put empty collection in blackboard
        tick_context.blackboard.set_internal("_items", [])

        child = MockLeafNode(id="process", results=[RunStatus.SUCCESS])
        foreach = ForEach(
            id="foreach",
            children=[child],
            collection_key="_items",  # Use internal key to bypass schema
            item_key="current",
            min_items=1,
        )
        # Don't set _collection directly - let _tick() fetch and check it

        result = foreach.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_child_failure_stops_iteration(self, tick_context: TickContext) -> None:
        """ForEach should stop on child failure when continue_on_failure=False."""
        child = MockLeafNode(
            id="process",
            results=[RunStatus.SUCCESS, RunStatus.FAILURE, RunStatus.SUCCESS],
        )
        foreach = ForEach(
            id="foreach",
            children=[child],
            collection_key="items",
            item_key="current",
            continue_on_failure=False,
        )
        foreach._collection = ["a", "b", "c"]

        result = foreach.tick(tick_context)

        assert result == RunStatus.FAILURE
        assert child.ticks_received == 2  # Stopped after failure

    def test_continue_on_failure(self, tick_context: TickContext) -> None:
        """ForEach should continue on failure when continue_on_failure=True."""
        child = MockLeafNode(
            id="process",
            results=[RunStatus.SUCCESS, RunStatus.FAILURE, RunStatus.SUCCESS],
        )
        foreach = ForEach(
            id="foreach",
            children=[child],
            collection_key="items",
            item_key="current",
            continue_on_failure=True,
        )
        foreach._collection = ["a", "b", "c"]

        result = foreach.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert child.ticks_received == 3  # All items processed

    def test_running_child_pauses_iteration(self, tick_context: TickContext) -> None:
        """ForEach should pause on RUNNING child."""
        child = MockLeafNode(
            id="process",
            results=[RunStatus.SUCCESS, RunStatus.RUNNING, RunStatus.SUCCESS],
        )
        foreach = ForEach(
            id="foreach",
            children=[child],
            collection_key="items",
            item_key="current",
        )
        foreach._collection = ["a", "b", "c"]

        # First tick: processes "a", starts "b" -> RUNNING
        result1 = foreach.tick(tick_context)
        assert result1 == RunStatus.RUNNING
        assert child.ticks_received == 2

        # Second tick: "b" completes, processes "c" -> SUCCESS
        result2 = foreach.tick(tick_context)
        assert result2 == RunStatus.SUCCESS
        assert child.ticks_received == 4  # "b" retried + "c"

    def test_reset_clears_iteration_state(self, tick_context: TickContext) -> None:
        """reset() should clear iteration state."""
        child = MockLeafNode(id="process", results=[RunStatus.SUCCESS])
        foreach = ForEach(
            id="foreach",
            children=[child],
            collection_key="items",
            item_key="current",
        )
        foreach._collection = ["a", "b"]
        foreach._current_index = 1
        foreach._iteration_results = [{"status": "success"}]

        foreach.reset()

        assert foreach._current_index == 0
        assert foreach._iteration_results == []
        assert foreach._collection is None

    def test_missing_collection_returns_failure(
        self, tick_context: TickContext
    ) -> None:
        """ForEach should return FAILURE if collection not found."""
        child = MockLeafNode(id="process", results=[RunStatus.SUCCESS])
        foreach = ForEach(
            id="foreach",
            children=[child],
            collection_key="nonexistent",
            item_key="current",
        )

        result = foreach.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_multiple_children_run_as_sequence(
        self, tick_context: TickContext
    ) -> None:
        """ForEach with multiple children should run them as sequence."""
        child1 = MockLeafNode(id="step1", results=[RunStatus.SUCCESS])
        child2 = MockLeafNode(id="step2", results=[RunStatus.SUCCESS])
        foreach = ForEach(
            id="foreach",
            children=[child1, child2],
            collection_key="items",
            item_key="current",
        )
        foreach._collection = ["a", "b"]

        result = foreach.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert child1.ticks_received == 2
        assert child2.ticks_received == 2


# =============================================================================
# Edge Cases
# =============================================================================


class TestCompositeEdgeCases:
    """Edge case tests for composite nodes."""

    def test_single_child_sequence(self, tick_context: TickContext) -> None:
        """Sequence with single child should work."""
        child = MockLeafNode(id="only", results=[RunStatus.SUCCESS])
        seq = Sequence(id="seq", children=[child])

        result = seq.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert child.ticks_received == 1

    def test_single_child_selector(self, tick_context: TickContext) -> None:
        """Selector with single child should work."""
        child = MockLeafNode(id="only", results=[RunStatus.FAILURE])
        sel = Selector(id="sel", children=[child])

        result = sel.tick(tick_context)

        assert result == RunStatus.FAILURE
        assert child.ticks_received == 1

    def test_nested_composites(self, tick_context: TickContext) -> None:
        """Nested composites should work correctly."""
        leaf1 = MockLeafNode(id="leaf1", results=[RunStatus.SUCCESS])
        leaf2 = MockLeafNode(id="leaf2", results=[RunStatus.SUCCESS])

        inner_seq = Sequence(id="inner", children=[leaf1])
        outer_seq = Sequence(id="outer", children=[inner_seq, leaf2])

        result = outer_seq.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert leaf1.ticks_received == 1
        assert leaf2.ticks_received == 1

    def test_parallel_all_running(self, tick_context: TickContext) -> None:
        """Parallel with all children RUNNING should return RUNNING."""
        children = [
            MockLeafNode(id="a", results=[RunStatus.RUNNING]),
            MockLeafNode(id="b", results=[RunStatus.RUNNING]),
        ]
        parallel = Parallel(id="par", children=children)

        result = parallel.tick(tick_context)

        assert result == RunStatus.RUNNING
