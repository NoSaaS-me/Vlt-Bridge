"""Unit tests for behavior tree leaf nodes."""

import pytest

from backend.src.services.plugins.behavior_tree.types import (
    RunStatus,
    TickContext,
    Blackboard,
)
from backend.src.services.plugins.behavior_tree.leaves import (
    SuccessNode,
    FailureNode,
    RunningNode,
    ConditionNode,
    ActionNode,
    WaitNode,
    BlackboardCondition,
    BlackboardSet,
    LogNode,
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


class TestSuccessNode:
    """Tests for SuccessNode leaf."""

    def test_always_returns_success(self, tick_context):
        """SuccessNode should always return SUCCESS."""
        node = SuccessNode()
        assert node.tick(tick_context) == RunStatus.SUCCESS

    def test_multiple_ticks(self, tick_context):
        """SuccessNode should return SUCCESS on every tick."""
        node = SuccessNode()
        for _ in range(5):
            assert node.tick(tick_context) == RunStatus.SUCCESS


class TestFailureNode:
    """Tests for FailureNode leaf."""

    def test_always_returns_failure(self, tick_context):
        """FailureNode should always return FAILURE."""
        node = FailureNode()
        assert node.tick(tick_context) == RunStatus.FAILURE


class TestRunningNode:
    """Tests for RunningNode leaf."""

    def test_always_returns_running(self, tick_context):
        """RunningNode should always return RUNNING."""
        node = RunningNode()
        assert node.tick(tick_context) == RunStatus.RUNNING


class TestConditionNode:
    """Tests for ConditionNode leaf."""

    def test_callable_condition_true(self, tick_context):
        """ConditionNode should return SUCCESS when callable returns True."""
        node = ConditionNode(condition=lambda ctx: True)
        assert node.tick(tick_context) == RunStatus.SUCCESS

    def test_callable_condition_false(self, tick_context):
        """ConditionNode should return FAILURE when callable returns False."""
        node = ConditionNode(condition=lambda ctx: False)
        assert node.tick(tick_context) == RunStatus.FAILURE

    def test_expression_condition_true(self, tick_context):
        """ConditionNode should evaluate expression returning True."""
        # Set context values for expression
        tick_context.rule_context.turn = tick_context.rule_context.turn.__class__(
            number=1,
            token_usage=0.9,
            context_usage=0.0,
            iteration_count=0,
        )

        node = ConditionNode(expression="context.turn.token_usage > 0.5")
        assert node.tick(tick_context) == RunStatus.SUCCESS

    def test_expression_condition_false(self, tick_context):
        """ConditionNode should evaluate expression returning False."""
        tick_context.rule_context.turn = tick_context.rule_context.turn.__class__(
            number=1,
            token_usage=0.1,
            context_usage=0.0,
            iteration_count=0,
        )

        node = ConditionNode(expression="context.turn.token_usage > 0.5")
        assert node.tick(tick_context) == RunStatus.FAILURE

    def test_no_condition_returns_success(self, tick_context):
        """ConditionNode with no condition should return SUCCESS."""
        node = ConditionNode()
        assert node.tick(tick_context) == RunStatus.SUCCESS

    def test_condition_exception_returns_failure(self, tick_context):
        """ConditionNode should return FAILURE if condition raises."""
        node = ConditionNode(condition=lambda ctx: 1 / 0)
        assert node.tick(tick_context) == RunStatus.FAILURE

    def test_expression_property(self):
        """ConditionNode should expose expression property."""
        node = ConditionNode(expression="context.turn.number > 5")
        assert node.expression == "context.turn.number > 5"


class TestActionNode:
    """Tests for ActionNode leaf."""

    def test_callable_action_success(self, tick_context):
        """ActionNode should return SUCCESS when action returns True."""
        executed = []

        def action(ctx):
            executed.append(True)
            return True

        node = ActionNode(action=action)
        assert node.tick(tick_context) == RunStatus.SUCCESS
        assert executed == [True]

    def test_callable_action_failure(self, tick_context):
        """ActionNode should return FAILURE when action returns False."""
        node = ActionNode(action=lambda ctx: False)
        assert node.tick(tick_context) == RunStatus.FAILURE

    def test_no_action_returns_success(self, tick_context):
        """ActionNode with no action should return SUCCESS."""
        node = ActionNode()
        assert node.tick(tick_context) == RunStatus.SUCCESS

    def test_action_exception_returns_failure(self, tick_context):
        """ActionNode should return FAILURE if action raises."""
        node = ActionNode(action=lambda ctx: 1 / 0)
        assert node.tick(tick_context) == RunStatus.FAILURE

    def test_no_dispatcher_for_rule_action(self, tick_context):
        """ActionNode should return FAILURE if rule_action without dispatcher."""
        from backend.src.services.plugins.rule import RuleAction, ActionType

        action = RuleAction(type=ActionType.LOG, message="test")
        node = ActionNode(rule_action=action)  # No dispatcher

        assert node.tick(tick_context) == RunStatus.FAILURE


class TestWaitNode:
    """Tests for WaitNode leaf."""

    def test_immediate_completion_no_wait(self, tick_context):
        """WaitNode with no wait should complete immediately."""
        node = WaitNode()
        assert node.tick(tick_context) == RunStatus.SUCCESS

    def test_tick_based_wait(self, tick_context):
        """WaitNode should wait for N ticks."""
        node = WaitNode(wait_ticks=3)

        # First tick starts waiting
        assert node.tick(tick_context) == RunStatus.RUNNING

        # Second tick still waiting
        tick_context.frame_id = 2
        assert node.tick(tick_context) == RunStatus.RUNNING

        # Third tick still waiting
        tick_context.frame_id = 3
        assert node.tick(tick_context) == RunStatus.RUNNING

        # Fourth tick completes
        tick_context.frame_id = 4
        assert node.tick(tick_context) == RunStatus.SUCCESS

    def test_reset_clears_wait(self, tick_context):
        """reset should clear wait state."""
        node = WaitNode(wait_ticks=3)

        # Start waiting
        node.tick(tick_context)
        assert node.tick(tick_context) == RunStatus.RUNNING

        # Reset
        node.reset()

        # Starts fresh
        tick_context.frame_id = 100
        assert node.tick(tick_context) == RunStatus.RUNNING


class TestBlackboardCondition:
    """Tests for BlackboardCondition leaf."""

    def test_key_equals_expected(self, tick_context):
        """BlackboardCondition should check equality."""
        tick_context.blackboard.set("target", "found")

        node = BlackboardCondition(
            key="target",
            expected="found",
            comparison="eq",
        )

        assert node.tick(tick_context) == RunStatus.SUCCESS

    def test_key_not_equals_expected(self, tick_context):
        """BlackboardCondition should detect inequality."""
        tick_context.blackboard.set("target", "found")

        node = BlackboardCondition(
            key="target",
            expected="missing",
            comparison="eq",
        )

        assert node.tick(tick_context) == RunStatus.FAILURE

    def test_comparison_operators(self, tick_context):
        """BlackboardCondition should support various comparisons."""
        tick_context.blackboard.set("count", 5)

        # Greater than
        node = BlackboardCondition(key="count", expected=3, comparison="gt")
        assert node.tick(tick_context) == RunStatus.SUCCESS

        # Less than
        node = BlackboardCondition(key="count", expected=10, comparison="lt")
        assert node.tick(tick_context) == RunStatus.SUCCESS

        # Greater or equal
        node = BlackboardCondition(key="count", expected=5, comparison="ge")
        assert node.tick(tick_context) == RunStatus.SUCCESS

        # Less or equal
        node = BlackboardCondition(key="count", expected=5, comparison="le")
        assert node.tick(tick_context) == RunStatus.SUCCESS

        # Not equal
        node = BlackboardCondition(key="count", expected=10, comparison="ne")
        assert node.tick(tick_context) == RunStatus.SUCCESS

    def test_missing_key_returns_failure(self, tick_context):
        """BlackboardCondition should fail for missing key."""
        node = BlackboardCondition(key="missing", expected=True)
        # Missing key returns None, which != True
        assert node.tick(tick_context) == RunStatus.FAILURE

    def test_no_blackboard_returns_failure(self, tick_context):
        """BlackboardCondition should fail if no blackboard."""
        tick_context.blackboard = None
        node = BlackboardCondition(key="target", expected=True)
        assert node.tick(tick_context) == RunStatus.FAILURE


class TestBlackboardSet:
    """Tests for BlackboardSet leaf."""

    def test_sets_value(self, tick_context):
        """BlackboardSet should set value in blackboard."""
        node = BlackboardSet(key="target", value="found")

        assert node.tick(tick_context) == RunStatus.SUCCESS
        assert tick_context.blackboard.get("target") == "found"

    def test_overwrites_existing(self, tick_context):
        """BlackboardSet should overwrite existing value."""
        tick_context.blackboard.set("target", "old")

        node = BlackboardSet(key="target", value="new")
        node.tick(tick_context)

        assert tick_context.blackboard.get("target") == "new"

    def test_no_key_returns_failure(self, tick_context):
        """BlackboardSet should fail with no key."""
        node = BlackboardSet(key="", value="test")
        assert node.tick(tick_context) == RunStatus.FAILURE

    def test_no_blackboard_returns_failure(self, tick_context):
        """BlackboardSet should fail if no blackboard."""
        tick_context.blackboard = None
        node = BlackboardSet(key="target", value=True)
        assert node.tick(tick_context) == RunStatus.FAILURE


class TestLogNode:
    """Tests for LogNode leaf."""

    def test_logs_and_returns_success(self, tick_context, caplog):
        """LogNode should log message and return SUCCESS."""
        import logging

        with caplog.at_level(logging.DEBUG):
            node = LogNode(message="Test message", level="debug")
            assert node.tick(tick_context) == RunStatus.SUCCESS

        assert "Test message" in caplog.text

    def test_log_levels(self, tick_context, caplog):
        """LogNode should respect log level."""
        import logging

        # Info level
        with caplog.at_level(logging.INFO):
            node = LogNode(message="Info message", level="info")
            node.tick(tick_context)

        assert "Info message" in caplog.text


class TestLeafNodeBase:
    """Tests for common leaf node behavior."""

    def test_tick_count_incremented(self, tick_context):
        """Leaf nodes should increment tick count."""
        node = SuccessNode()

        assert node.tick_count == 0

        node.tick(tick_context)
        assert node.tick_count == 1

        node.tick(tick_context)
        assert node.tick_count == 2

    def test_status_updated(self, tick_context):
        """Leaf nodes should update status after tick."""
        node = SuccessNode()

        assert node.status == RunStatus.FAILURE  # Initial

        node.tick(tick_context)
        assert node.status == RunStatus.SUCCESS

    def test_name_property(self):
        """Leaf nodes should have customizable name."""
        node = SuccessNode(name="CustomName")
        assert node.name == "CustomName"

    def test_debug_info(self, tick_context):
        """debug_info should include relevant fields."""
        node = SuccessNode(name="TestNode")
        node.tick(tick_context)

        info = node.debug_info()

        assert info["type"] == "SuccessNode"
        assert info["name"] == "TestNode"
        assert info["status"] == "SUCCESS"
        assert info["tick_count"] == 1

    def test_repr(self):
        """Leaf nodes should have useful repr."""
        node = SuccessNode(name="TestNode")
        repr_str = repr(node)

        assert "SuccessNode" in repr_str
        assert "TestNode" in repr_str
