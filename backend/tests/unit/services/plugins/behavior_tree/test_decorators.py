"""Unit tests for behavior tree decorator nodes."""

import pytest
import time

from backend.src.services.plugins.behavior_tree.types import (
    RunStatus,
    TickContext,
    Blackboard,
)
from backend.src.services.plugins.behavior_tree.decorators import (
    Inverter,
    Succeeder,
    Failer,
    UntilFail,
    UntilSuccess,
    Cooldown,
    Guard,
    Retry,
    Timeout,
    Repeat,
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


class TestInverter:
    """Tests for Inverter decorator."""

    def test_inverts_success_to_failure(self, tick_context):
        """Inverter should convert SUCCESS to FAILURE."""
        inverter = Inverter(child=SuccessNode())
        assert inverter.tick(tick_context) == RunStatus.FAILURE

    def test_inverts_failure_to_success(self, tick_context):
        """Inverter should convert FAILURE to SUCCESS."""
        inverter = Inverter(child=FailureNode())
        assert inverter.tick(tick_context) == RunStatus.SUCCESS

    def test_running_passes_through(self, tick_context):
        """Inverter should not affect RUNNING status."""
        inverter = Inverter(child=RunningNode())
        assert inverter.tick(tick_context) == RunStatus.RUNNING

    def test_no_child_returns_failure(self, tick_context):
        """Inverter with no child should return FAILURE."""
        inverter = Inverter()
        assert inverter.tick(tick_context) == RunStatus.FAILURE


class TestSucceeder:
    """Tests for Succeeder decorator."""

    def test_success_stays_success(self, tick_context):
        """Succeeder should keep SUCCESS as SUCCESS."""
        succeeder = Succeeder(child=SuccessNode())
        assert succeeder.tick(tick_context) == RunStatus.SUCCESS

    def test_failure_becomes_success(self, tick_context):
        """Succeeder should convert FAILURE to SUCCESS."""
        succeeder = Succeeder(child=FailureNode())
        assert succeeder.tick(tick_context) == RunStatus.SUCCESS

    def test_running_passes_through(self, tick_context):
        """Succeeder should not affect RUNNING status."""
        succeeder = Succeeder(child=RunningNode())
        assert succeeder.tick(tick_context) == RunStatus.RUNNING

    def test_no_child_returns_success(self, tick_context):
        """Succeeder with no child should return SUCCESS."""
        succeeder = Succeeder()
        assert succeeder.tick(tick_context) == RunStatus.SUCCESS


class TestFailer:
    """Tests for Failer decorator."""

    def test_success_becomes_failure(self, tick_context):
        """Failer should convert SUCCESS to FAILURE."""
        failer = Failer(child=SuccessNode())
        assert failer.tick(tick_context) == RunStatus.FAILURE

    def test_failure_stays_failure(self, tick_context):
        """Failer should keep FAILURE as FAILURE."""
        failer = Failer(child=FailureNode())
        assert failer.tick(tick_context) == RunStatus.FAILURE

    def test_running_passes_through(self, tick_context):
        """Failer should not affect RUNNING status."""
        failer = Failer(child=RunningNode())
        assert failer.tick(tick_context) == RunStatus.RUNNING


class TestUntilFail:
    """Tests for UntilFail decorator."""

    def test_repeats_until_failure(self, tick_context):
        """UntilFail should repeat child until it fails."""
        # Child that succeeds first, then fails
        call_count = 0

        class CountingNode(SuccessNode):
            def _tick(self, context):
                nonlocal call_count
                call_count += 1
                if call_count >= 3:
                    return RunStatus.FAILURE
                return RunStatus.SUCCESS

        until_fail = UntilFail(child=CountingNode())

        # Keep ticking until success (child failed)
        max_ticks = 10
        for _ in range(max_ticks):
            result = until_fail.tick(tick_context)
            if result == RunStatus.SUCCESS:
                break

        assert result == RunStatus.SUCCESS
        assert call_count == 3

    def test_max_iterations_limit(self, tick_context):
        """UntilFail should respect max_iterations."""
        until_fail = UntilFail(child=SuccessNode(), max_iterations=5)

        # Tick until done
        for _ in range(10):
            result = until_fail.tick(tick_context)
            if result == RunStatus.SUCCESS:
                break

        assert result == RunStatus.SUCCESS

    def test_running_passes_through(self, tick_context):
        """UntilFail should propagate RUNNING."""
        until_fail = UntilFail(child=RunningNode())
        assert until_fail.tick(tick_context) == RunStatus.RUNNING


class TestUntilSuccess:
    """Tests for UntilSuccess decorator."""

    def test_repeats_until_success(self, tick_context):
        """UntilSuccess should repeat child until it succeeds."""
        call_count = 0

        class CountingNode(FailureNode):
            def _tick(self, context):
                nonlocal call_count
                call_count += 1
                if call_count >= 3:
                    return RunStatus.SUCCESS
                return RunStatus.FAILURE

        until_success = UntilSuccess(child=CountingNode())

        # Keep ticking until success
        max_ticks = 10
        for _ in range(max_ticks):
            result = until_success.tick(tick_context)
            if result == RunStatus.SUCCESS:
                break

        assert result == RunStatus.SUCCESS
        assert call_count == 3

    def test_max_iterations_returns_failure(self, tick_context):
        """UntilSuccess should return FAILURE after max iterations."""
        until_success = UntilSuccess(child=FailureNode(), max_iterations=3)

        # Tick until done
        for _ in range(10):
            result = until_success.tick(tick_context)
            if result != RunStatus.RUNNING:
                break

        assert result == RunStatus.FAILURE


class TestCooldown:
    """Tests for Cooldown decorator."""

    def test_tick_based_cooldown(self, tick_context):
        """Cooldown should block execution for N ticks."""
        cooldown = Cooldown(child=SuccessNode(), cooldown_ticks=3)

        # First tick succeeds
        assert cooldown.tick(tick_context) == RunStatus.SUCCESS

        # Next ticks blocked
        tick_context.frame_id = 2
        assert cooldown.tick(tick_context) == RunStatus.FAILURE

        tick_context.frame_id = 3
        assert cooldown.tick(tick_context) == RunStatus.FAILURE

        # After cooldown, succeeds again
        tick_context.frame_id = 4
        assert cooldown.tick(tick_context) == RunStatus.SUCCESS

    def test_time_based_cooldown(self, tick_context):
        """Cooldown should block execution for N milliseconds."""
        cooldown = Cooldown(child=SuccessNode(), cooldown_ms=50)

        # First tick succeeds
        assert cooldown.tick(tick_context) == RunStatus.SUCCESS

        # Immediate re-tick blocked
        assert cooldown.tick(tick_context) == RunStatus.FAILURE

        # Wait for cooldown
        time.sleep(0.06)

        # Now succeeds
        assert cooldown.tick(tick_context) == RunStatus.SUCCESS

    def test_force_reset_cooldown(self, tick_context):
        """force_reset_cooldown should clear cooldown state."""
        cooldown = Cooldown(child=SuccessNode(), cooldown_ticks=10)

        # First tick succeeds, starts cooldown
        cooldown.tick(tick_context)

        # Force reset
        cooldown.force_reset_cooldown()

        # Can execute again immediately
        assert cooldown.tick(tick_context) == RunStatus.SUCCESS


class TestGuard:
    """Tests for Guard decorator."""

    def test_condition_true_executes_child(self, tick_context):
        """Guard should execute child when condition is true."""
        guard = Guard(
            child=SuccessNode(),
            condition=lambda ctx: True,
        )

        assert guard.tick(tick_context) == RunStatus.SUCCESS

    def test_condition_false_returns_failure(self, tick_context):
        """Guard should return FAILURE when condition is false."""
        child = SuccessNode()
        guard = Guard(
            child=child,
            condition=lambda ctx: False,
        )

        assert guard.tick(tick_context) == RunStatus.FAILURE
        # Child not ticked
        assert child.tick_count == 0

    def test_expression_evaluation(self, tick_context):
        """Guard should evaluate expression strings."""
        # Set token usage for testing
        tick_context.rule_context.turn = tick_context.rule_context.turn.__class__(
            number=1,
            token_usage=0.9,
            context_usage=0.0,
            iteration_count=0,
        )

        guard = Guard(
            child=SuccessNode(),
            expression="context.turn.token_usage > 0.8",
        )

        assert guard.tick(tick_context) == RunStatus.SUCCESS

    def test_no_condition_passes(self, tick_context):
        """Guard with no condition should pass."""
        guard = Guard(child=SuccessNode())
        assert guard.tick(tick_context) == RunStatus.SUCCESS

    def test_condition_exception_returns_failure(self, tick_context):
        """Guard should return FAILURE if condition raises."""
        guard = Guard(
            child=SuccessNode(),
            condition=lambda ctx: 1 / 0,  # Raises ZeroDivisionError
        )

        assert guard.tick(tick_context) == RunStatus.FAILURE


class TestRetry:
    """Tests for Retry decorator."""

    def test_success_returns_immediately(self, tick_context):
        """Retry should return SUCCESS immediately if child succeeds."""
        retry = Retry(child=SuccessNode(), max_attempts=3)

        assert retry.tick(tick_context) == RunStatus.SUCCESS

    def test_retries_on_failure(self, tick_context):
        """Retry should retry child on failure."""
        attempt_count = 0

        class FailThenSucceed(SuccessNode):
            def _tick(self, context):
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    return RunStatus.FAILURE
                return RunStatus.SUCCESS

        retry = Retry(child=FailThenSucceed(), max_attempts=5)

        # Keep ticking until success
        for _ in range(10):
            result = retry.tick(tick_context)
            if result == RunStatus.SUCCESS:
                break

        assert result == RunStatus.SUCCESS
        assert attempt_count == 3

    def test_max_attempts_returns_failure(self, tick_context):
        """Retry should return FAILURE after max attempts."""
        retry = Retry(child=FailureNode(), max_attempts=3)

        # Tick through all attempts
        for _ in range(10):
            result = retry.tick(tick_context)
            if result != RunStatus.RUNNING:
                break

        assert result == RunStatus.FAILURE


class TestTimeout:
    """Tests for Timeout decorator."""

    def test_success_within_timeout(self, tick_context):
        """Timeout should allow success within limit."""
        timeout = Timeout(child=SuccessNode(), timeout_ticks=10)
        assert timeout.tick(tick_context) == RunStatus.SUCCESS

    def test_failure_within_timeout(self, tick_context):
        """Timeout should allow failure within limit."""
        timeout = Timeout(child=FailureNode(), timeout_ticks=10)
        assert timeout.tick(tick_context) == RunStatus.FAILURE

    def test_tick_timeout(self, tick_context):
        """Timeout should fail after too many running ticks."""
        timeout = Timeout(child=RunningNode(), timeout_ticks=3)

        # First tick - running
        assert timeout.tick(tick_context) == RunStatus.RUNNING

        # Continue running
        tick_context.frame_id = 2
        assert timeout.tick(tick_context) == RunStatus.RUNNING

        tick_context.frame_id = 3
        assert timeout.tick(tick_context) == RunStatus.RUNNING

        # Timeout exceeded
        tick_context.frame_id = 4
        assert timeout.tick(tick_context) == RunStatus.FAILURE


class TestRepeat:
    """Tests for Repeat decorator."""

    def test_repeats_n_times(self, tick_context):
        """Repeat should execute child N times."""
        child = SuccessNode()
        repeat = Repeat(child=child, count=3)

        # Tick until done
        for _ in range(10):
            result = repeat.tick(tick_context)
            if result == RunStatus.SUCCESS:
                break

        assert result == RunStatus.SUCCESS
        # Child was ticked 3 times
        assert child.tick_count == 3

    def test_failure_stops_repeat(self, tick_context):
        """Repeat should stop on child failure."""
        call_count = 0

        class FailOnSecond(SuccessNode):
            def _tick(self, context):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    return RunStatus.FAILURE
                return RunStatus.SUCCESS

        repeat = Repeat(child=FailOnSecond(), count=5)

        # Tick until failure
        for _ in range(10):
            result = repeat.tick(tick_context)
            if result == RunStatus.FAILURE:
                break

        assert result == RunStatus.FAILURE
        assert call_count == 2

    def test_count_zero_returns_success(self, tick_context):
        """Repeat with count=0 should return SUCCESS immediately."""
        repeat = Repeat(child=SuccessNode(), count=0)
        # Empty repeat is immediate success (nothing to do)
        assert repeat.tick(tick_context) == RunStatus.SUCCESS


class TestDecoratorChildManagement:
    """Tests for child management in decorators."""

    def test_set_child(self, tick_context):
        """child setter should update child."""
        inverter = Inverter()
        child = SuccessNode()

        inverter.child = child

        assert inverter.child is child

    def test_reset_propagates(self, tick_context):
        """reset should propagate to child."""
        child = SuccessNode()
        inverter = Inverter(child=child)

        # Tick to set state
        inverter.tick(tick_context)

        # Reset
        inverter.reset()

        assert inverter.status == RunStatus.FAILURE
        assert child.status == RunStatus.FAILURE

    def test_debug_info_includes_child(self, tick_context):
        """debug_info should include child info."""
        child = SuccessNode(name="TestChild")
        inverter = Inverter(child=child, name="TestInverter")

        info = inverter.debug_info()

        assert info["name"] == "TestInverter"
        assert "child" in info
        assert info["child"]["name"] == "TestChild"
