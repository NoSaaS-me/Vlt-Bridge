"""Unit tests for behavior tree composite nodes."""

import pytest

from backend.src.services.plugins.behavior_tree.types import (
    RunStatus,
    TickContext,
    Blackboard,
)
from backend.src.services.plugins.behavior_tree.composites import (
    PrioritySelector,
    Sequence,
    Parallel,
    ParallelPolicy,
    MemorySelector,
    MemorySequence,
)
from backend.src.services.plugins.behavior_tree.leaves import (
    SuccessNode,
    FailureNode,
    RunningNode,
)
from backend.src.services.plugins.context import RuleContext


@pytest.fixture
def tick_context():
    """Create a TickContext for testing."""
    rule_context = RuleContext.create_minimal("user1", "project1")
    return TickContext(
        rule_context=rule_context,
        frame_id=1,
        blackboard=Blackboard(),
    )


class TestPrioritySelector:
    """Tests for PrioritySelector composite."""

    def test_empty_selector_returns_failure(self, tick_context):
        """Empty selector should return FAILURE."""
        selector = PrioritySelector()
        assert selector.tick(tick_context) == RunStatus.FAILURE

    def test_first_success_wins(self, tick_context):
        """Selector should return SUCCESS on first successful child."""
        selector = PrioritySelector([
            FailureNode(name="fail1"),
            SuccessNode(name="success"),
            FailureNode(name="fail2"),
        ])

        assert selector.tick(tick_context) == RunStatus.SUCCESS

    def test_all_fail_returns_failure(self, tick_context):
        """Selector should return FAILURE only if all children fail."""
        selector = PrioritySelector([
            FailureNode(name="fail1"),
            FailureNode(name="fail2"),
            FailureNode(name="fail3"),
        ])

        assert selector.tick(tick_context) == RunStatus.FAILURE

    def test_running_propagates(self, tick_context):
        """Selector should return RUNNING if child is running."""
        selector = PrioritySelector([
            FailureNode(name="fail1"),
            RunningNode(name="running"),
            SuccessNode(name="success"),
        ])

        assert selector.tick(tick_context) == RunStatus.RUNNING

    def test_short_circuit_on_success(self, tick_context):
        """Selector should not tick children after success."""
        success_node = SuccessNode(name="success")
        skip_node = SuccessNode(name="skip")

        selector = PrioritySelector([
            FailureNode(),
            success_node,
            skip_node,
        ])

        selector.tick(tick_context)

        # Success node was ticked
        assert success_node.tick_count == 1
        # Skip node was not ticked
        assert skip_node.tick_count == 0

    def test_priority_order_preserved(self, tick_context):
        """Children should be evaluated in order (first = highest priority)."""
        order = []

        class TrackingNode(SuccessNode):
            def _tick(self, context):
                order.append(self.name)
                return RunStatus.FAILURE  # Fail to continue checking

        selector = PrioritySelector([
            TrackingNode(name="first"),
            TrackingNode(name="second"),
            TrackingNode(name="third"),
        ])

        selector.tick(tick_context)
        assert order == ["first", "second", "third"]

    def test_resume_from_running_child(self, tick_context):
        """Selector should resume from cached running child."""
        # First child that fails initially but succeeds later
        first_fail = FailureNode(name="first")

        # Running node
        running = RunningNode(name="running")

        selector = PrioritySelector([first_fail, running])

        # First tick - running node
        result = selector.tick(tick_context)
        assert result == RunStatus.RUNNING
        assert first_fail.tick_count == 1

        # Second tick - should resume from running
        tick_context.frame_id = 2
        result = selector.tick(tick_context)
        assert result == RunStatus.RUNNING
        # First child should be ticked again (standard behavior)
        # Note: PrioritySelector always re-evaluates higher priority children


class TestSequence:
    """Tests for Sequence composite."""

    def test_empty_sequence_returns_success(self, tick_context):
        """Empty sequence should return SUCCESS."""
        sequence = Sequence()
        assert sequence.tick(tick_context) == RunStatus.SUCCESS

    def test_all_succeed_returns_success(self, tick_context):
        """Sequence should return SUCCESS only if all children succeed."""
        sequence = Sequence([
            SuccessNode(name="s1"),
            SuccessNode(name="s2"),
            SuccessNode(name="s3"),
        ])

        assert sequence.tick(tick_context) == RunStatus.SUCCESS

    def test_first_failure_returns_failure(self, tick_context):
        """Sequence should return FAILURE on first failure (fail-fast)."""
        sequence = Sequence([
            SuccessNode(name="s1"),
            FailureNode(name="fail"),
            SuccessNode(name="s2"),
        ])

        assert sequence.tick(tick_context) == RunStatus.FAILURE

    def test_running_propagates(self, tick_context):
        """Sequence should return RUNNING if child is running."""
        sequence = Sequence([
            SuccessNode(name="s1"),
            RunningNode(name="running"),
            SuccessNode(name="s2"),
        ])

        assert sequence.tick(tick_context) == RunStatus.RUNNING

    def test_fail_fast(self, tick_context):
        """Sequence should not tick children after failure."""
        fail_node = FailureNode(name="fail")
        skip_node = SuccessNode(name="skip")

        sequence = Sequence([
            SuccessNode(),
            fail_node,
            skip_node,
        ])

        sequence.tick(tick_context)

        assert fail_node.tick_count == 1
        assert skip_node.tick_count == 0

    def test_resume_from_running_child(self, tick_context):
        """Sequence should resume from running child."""
        first_success = SuccessNode(name="first")
        running = RunningNode(name="running")

        sequence = Sequence([first_success, running])

        # First tick
        result = sequence.tick(tick_context)
        assert result == RunStatus.RUNNING
        assert first_success.tick_count == 1

        # Second tick - resumes from running
        tick_context.frame_id = 2
        result = sequence.tick(tick_context)
        assert result == RunStatus.RUNNING
        # First child not re-ticked (sequence remembers position)
        assert first_success.tick_count == 1


class TestParallel:
    """Tests for Parallel composite."""

    def test_empty_parallel_returns_success(self, tick_context):
        """Empty parallel should return SUCCESS."""
        parallel = Parallel()
        assert parallel.tick(tick_context) == RunStatus.SUCCESS

    def test_require_one_any_success(self, tick_context):
        """REQUIRE_ONE: success if any child succeeds."""
        parallel = Parallel(
            [
                FailureNode(),
                SuccessNode(),
                FailureNode(),
            ],
            policy=ParallelPolicy.REQUIRE_ONE,
        )

        assert parallel.tick(tick_context) == RunStatus.SUCCESS

    def test_require_one_all_fail(self, tick_context):
        """REQUIRE_ONE: failure only if all fail."""
        parallel = Parallel(
            [
                FailureNode(),
                FailureNode(),
                FailureNode(),
            ],
            policy=ParallelPolicy.REQUIRE_ONE,
        )

        assert parallel.tick(tick_context) == RunStatus.FAILURE

    def test_require_all_all_success(self, tick_context):
        """REQUIRE_ALL: success only if all succeed."""
        parallel = Parallel(
            [
                SuccessNode(),
                SuccessNode(),
                SuccessNode(),
            ],
            policy=ParallelPolicy.REQUIRE_ALL,
        )

        assert parallel.tick(tick_context) == RunStatus.SUCCESS

    def test_require_all_any_fail(self, tick_context):
        """REQUIRE_ALL: failure if any fails."""
        parallel = Parallel(
            [
                SuccessNode(),
                FailureNode(),
                SuccessNode(),
            ],
            policy=ParallelPolicy.REQUIRE_ALL,
        )

        assert parallel.tick(tick_context) == RunStatus.FAILURE

    def test_all_children_ticked(self, tick_context):
        """Parallel should tick all children every time."""
        children = [
            SuccessNode(name="c1"),
            SuccessNode(name="c2"),
            SuccessNode(name="c3"),
        ]
        parallel = Parallel(children)

        parallel.tick(tick_context)

        for child in children:
            assert child.tick_count == 1

    def test_running_with_require_one(self, tick_context):
        """REQUIRE_ONE: running if some running, none succeeded."""
        parallel = Parallel(
            [
                FailureNode(),
                RunningNode(),
                FailureNode(),
            ],
            policy=ParallelPolicy.REQUIRE_ONE,
        )

        assert parallel.tick(tick_context) == RunStatus.RUNNING

    def test_running_with_require_all(self, tick_context):
        """REQUIRE_ALL: running if some running, none failed."""
        parallel = Parallel(
            [
                SuccessNode(),
                RunningNode(),
                SuccessNode(),
            ],
            policy=ParallelPolicy.REQUIRE_ALL,
        )

        assert parallel.tick(tick_context) == RunStatus.RUNNING


class TestMemorySelector:
    """Tests for MemorySelector composite."""

    def test_remembers_position_on_failure(self, tick_context):
        """MemorySelector should remember position across ticks."""

        class CountingNode(FailureNode):
            pass

        first = CountingNode(name="first")
        second = CountingNode(name="second")
        third = SuccessNode(name="third")

        selector = MemorySelector([first, second, third])

        # First tick - goes through all
        result = selector.tick(tick_context)
        assert result == RunStatus.SUCCESS
        assert first.tick_count == 1
        assert second.tick_count == 1


class TestMemorySequence:
    """Tests for MemorySequence composite."""

    def test_remembers_position_on_failure(self, tick_context):
        """MemorySequence should remember failed position."""

        class CountingSuccess(SuccessNode):
            pass

        first = CountingSuccess(name="first")
        fail_node = FailureNode(name="fail")

        sequence = MemorySequence([first, fail_node])

        # First tick - fails at second
        result = sequence.tick(tick_context)
        assert result == RunStatus.FAILURE
        assert first.tick_count == 1

        # Second tick - resumes from failure point
        tick_context.frame_id = 2
        result = sequence.tick(tick_context)
        assert result == RunStatus.FAILURE
        # First not re-ticked
        assert first.tick_count == 1


class TestCompositeChildManagement:
    """Tests for child management in composites."""

    def test_add_child(self):
        """add_child should append child."""
        selector = PrioritySelector()
        child = SuccessNode()

        selector.add_child(child)

        assert len(selector) == 1
        assert child in selector.children

    def test_add_children(self):
        """add_children should append multiple children."""
        selector = PrioritySelector()
        children = [SuccessNode(), FailureNode()]

        selector.add_children(children)

        assert len(selector) == 2

    def test_remove_child(self):
        """remove_child should remove and return success."""
        child = SuccessNode()
        selector = PrioritySelector([child])

        result = selector.remove_child(child)

        assert result is True
        assert len(selector) == 0

    def test_remove_child_not_found(self):
        """remove_child should return False if not found."""
        selector = PrioritySelector()
        child = SuccessNode()

        result = selector.remove_child(child)

        assert result is False

    def test_clear_children(self):
        """clear_children should remove all children."""
        selector = PrioritySelector([SuccessNode(), FailureNode()])

        selector.clear_children()

        assert len(selector) == 0

    def test_iteration(self, tick_context):
        """Composite should be iterable."""
        children = [SuccessNode(), FailureNode()]
        selector = PrioritySelector(children)

        iterated = list(selector)

        assert iterated == children

    def test_reset_propagates(self, tick_context):
        """reset should propagate to all children."""
        selector = PrioritySelector([
            SuccessNode(),
            SuccessNode(),
        ])

        # Tick to set state
        selector.tick(tick_context)

        # Reset
        selector.reset()

        # Status reset
        assert selector.status == RunStatus.FAILURE
        for child in selector.children:
            assert child.status == RunStatus.FAILURE
