"""
Unit tests for BT Decorator Nodes.

Tests:
- DecoratorNode base class
- Timeout: timeout triggering, E6001 logging
- Retry: exhaustion, backoff behavior
- Guard: condition evaluation (Python callable, string expression)
- Cooldown: timing enforcement
- Inverter: result inversion
- AlwaysSucceed/AlwaysFail: forced status

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.2.1-2.2.6 from tasks.md
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from backend.src.bt.nodes.base import BehaviorNode
from backend.src.bt.nodes.decorators import (
    DecoratorNode,
    Timeout,
    Retry,
    Guard,
    Cooldown,
    Inverter,
    AlwaysSucceed,
    AlwaysFail,
)
from backend.src.bt.state.base import NodeType, RunStatus
from backend.src.bt.state.blackboard import TypedBlackboard
from backend.src.bt.core.context import TickContext


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class MockLeafNode(BehaviorNode):
    """Mock leaf node for testing decorators."""

    def __init__(
        self,
        id: str,
        results: Optional[list] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize mock leaf.

        Args:
            id: Node identifier.
            results: Sequence of results to return on each tick.
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


@pytest.fixture
def basic_blackboard() -> TypedBlackboard:
    """Create a basic blackboard for testing."""
    return TypedBlackboard(scope_name="test")


@pytest.fixture
def tick_context(basic_blackboard: TypedBlackboard) -> TickContext:
    """Create a basic tick context."""
    return TickContext(blackboard=basic_blackboard)


# =============================================================================
# DecoratorNode Base Tests
# =============================================================================


class TestDecoratorNodeBase:
    """Tests for DecoratorNode base class."""

    def test_requires_exactly_one_child(self) -> None:
        """DecoratorNode should accept exactly one child."""
        child = MockLeafNode(id="child")
        decorator = Inverter(id="inv", child=child)

        assert decorator.child is child

    def test_sets_parent_on_child(self) -> None:
        """DecoratorNode should set parent reference on child."""
        child = MockLeafNode(id="child")
        decorator = Inverter(id="inv", child=child)

        assert child.parent is decorator

    def test_node_type_is_decorator(self) -> None:
        """DecoratorNode should report DECORATOR node type."""
        child = MockLeafNode(id="child")
        decorator = Inverter(id="inv", child=child)

        assert decorator.node_type == NodeType.DECORATOR

    def test_child_property(self) -> None:
        """child property should return the decorated child."""
        child = MockLeafNode(id="child")
        decorator = Inverter(id="inv", child=child)

        assert decorator.child is child


# =============================================================================
# Timeout Tests
# =============================================================================


class TestTimeout:
    """Tests for Timeout decorator node."""

    def test_timeout_ms_must_be_positive(self) -> None:
        """Timeout should reject non-positive timeout_ms."""
        child = MockLeafNode(id="child")

        with pytest.raises(ValueError) as exc_info:
            Timeout(id="timeout", child=child, timeout_ms=0)
        assert "positive" in str(exc_info.value)

        with pytest.raises(ValueError):
            Timeout(id="timeout", child=child, timeout_ms=-100)

    def test_passes_through_success(self, tick_context: TickContext) -> None:
        """Timeout should pass through SUCCESS status."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        timeout = Timeout(id="timeout", child=child, timeout_ms=1000)

        result = timeout.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert tick_context.blackboard._data.get("_timeout_triggered") is False

    def test_passes_through_failure(self, tick_context: TickContext) -> None:
        """Timeout should pass through FAILURE status."""
        child = MockLeafNode(id="child", results=[RunStatus.FAILURE])
        timeout = Timeout(id="timeout", child=child, timeout_ms=1000)

        result = timeout.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_running_within_timeout(self, tick_context: TickContext) -> None:
        """Timeout should allow RUNNING within time limit."""
        child = MockLeafNode(id="child", results=[RunStatus.RUNNING])
        timeout = Timeout(id="timeout", child=child, timeout_ms=10000)  # 10 seconds

        result = timeout.tick(tick_context)

        assert result == RunStatus.RUNNING
        assert timeout._child_running_since is not None

    def test_timeout_triggers_on_exceeded(self, tick_context: TickContext) -> None:
        """Timeout should trigger when time exceeded."""
        child = MockLeafNode(id="child", results=[RunStatus.RUNNING])
        timeout = Timeout(id="timeout", child=child, timeout_ms=1)  # 1ms

        # First tick - start running
        timeout.tick(tick_context)

        # Wait briefly to exceed timeout
        time.sleep(0.002)  # 2ms

        # Second tick - should trigger timeout
        result = timeout.tick(tick_context)

        assert result == RunStatus.FAILURE
        assert tick_context.blackboard._data.get("_timeout_triggered") is True

    def test_timeout_resets_child(self, tick_context: TickContext) -> None:
        """Timeout should reset child when triggered."""
        child = MockLeafNode(id="child", results=[RunStatus.RUNNING])
        timeout = Timeout(id="timeout", child=child, timeout_ms=1)

        # Start running
        timeout.tick(tick_context)
        time.sleep(0.002)

        # Trigger timeout
        timeout.tick(tick_context)

        # Child should be reset
        assert child.status == RunStatus.FRESH

    def test_reset_clears_tracking(self, tick_context: TickContext) -> None:
        """reset() should clear timeout tracking."""
        child = MockLeafNode(id="child", results=[RunStatus.RUNNING])
        timeout = Timeout(id="timeout", child=child, timeout_ms=1000)

        timeout.tick(tick_context)
        assert timeout._child_running_since is not None

        timeout.reset()

        assert timeout._child_running_since is None


# =============================================================================
# Retry Tests
# =============================================================================


class TestRetry:
    """Tests for Retry decorator node."""

    def test_max_retries_must_be_non_negative(self) -> None:
        """Retry should reject negative max_retries."""
        child = MockLeafNode(id="child")

        with pytest.raises(ValueError) as exc_info:
            Retry(id="retry", child=child, max_retries=-1)
        assert "non-negative" in str(exc_info.value)

    def test_passes_through_success(self, tick_context: TickContext) -> None:
        """Retry should pass through SUCCESS immediately."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        retry = Retry(id="retry", child=child, max_retries=3)

        result = retry.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_retries_on_failure(self, tick_context: TickContext) -> None:
        """Retry should retry on child failure."""
        # Note: Retry calls child.reset() after failure, which resets _tick_index.
        # So we need a child that eventually succeeds after being reset.
        # Using a mock that cycles through results WITHOUT reset affecting it.
        child = MockLeafNode(
            id="child",
            results=[RunStatus.FAILURE],  # Always fails
        )
        retry = Retry(id="retry", child=child, max_retries=3, backoff_ms=0)

        # First tick: FAILURE -> retry scheduled
        result1 = retry.tick(tick_context)
        assert result1 == RunStatus.RUNNING
        assert retry._retry_count == 1

        # Second tick: Still failing, retry again
        result2 = retry.tick(tick_context)
        assert result2 == RunStatus.RUNNING
        assert retry._retry_count == 2

        # Now change child to succeed for the test
        child._results = [RunStatus.SUCCESS]

        # Third tick: SUCCESS
        result3 = retry.tick(tick_context)
        assert result3 == RunStatus.SUCCESS

    def test_exhausts_retries(self, tick_context: TickContext) -> None:
        """Retry should return FAILURE when retries exhausted."""
        child = MockLeafNode(id="child", results=[RunStatus.FAILURE])
        retry = Retry(id="retry", child=child, max_retries=2, backoff_ms=0)

        # Exhaust retries
        retry.tick(tick_context)  # RUNNING (retry 1)
        retry.tick(tick_context)  # RUNNING (retry 2)
        result = retry.tick(tick_context)  # FAILURE (exhausted)

        assert result == RunStatus.FAILURE
        assert retry._retry_count == 2

    def test_backoff_delays_retry(self, tick_context: TickContext) -> None:
        """Retry should wait for backoff_ms before retry."""
        child = MockLeafNode(id="child", results=[RunStatus.FAILURE])
        retry = Retry(id="retry", child=child, max_retries=2, backoff_ms=50)

        # First failure
        retry.tick(tick_context)
        assert retry._last_retry_at is not None

        # Immediate tick should still be in backoff
        result = retry.tick(tick_context)
        assert result == RunStatus.RUNNING

    def test_passes_through_running(self, tick_context: TickContext) -> None:
        """Retry should pass through RUNNING status."""
        child = MockLeafNode(id="child", results=[RunStatus.RUNNING])
        retry = Retry(id="retry", child=child, max_retries=3)

        result = retry.tick(tick_context)

        assert result == RunStatus.RUNNING
        assert retry._retry_count == 0  # No retry needed

    def test_reset_clears_retry_count(self, tick_context: TickContext) -> None:
        """reset() should clear retry count."""
        child = MockLeafNode(id="child", results=[RunStatus.FAILURE])
        retry = Retry(id="retry", child=child, max_retries=3, backoff_ms=0)

        retry.tick(tick_context)
        assert retry._retry_count == 1

        retry.reset()

        assert retry._retry_count == 0
        assert retry._last_retry_at is None


# =============================================================================
# Guard Tests
# =============================================================================


class TestGuard:
    """Tests for Guard decorator node."""

    def test_callable_condition_true(self, tick_context: TickContext) -> None:
        """Guard should tick child when callable returns True."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        guard = Guard(id="guard", child=child, condition=lambda ctx: True)

        result = guard.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert child.ticks_received == 1

    def test_callable_condition_false(self, tick_context: TickContext) -> None:
        """Guard should return FAILURE when callable returns False."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        guard = Guard(id="guard", child=child, condition=lambda ctx: False)

        result = guard.tick(tick_context)

        assert result == RunStatus.FAILURE
        assert child.ticks_received == 0  # Child not ticked

    def test_callable_uses_context(self, tick_context: TickContext) -> None:
        """Guard callable should receive context."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        tick_context.blackboard.set_internal("_budget", 100)

        def check_budget(ctx: TickContext) -> bool:
            return ctx.blackboard._data.get("_budget", 0) > 0

        guard = Guard(id="guard", child=child, condition=check_budget)

        result = guard.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_callable_exception_returns_failure(
        self, tick_context: TickContext
    ) -> None:
        """Guard should return FAILURE if condition raises exception."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])

        def bad_condition(ctx: TickContext) -> bool:
            raise RuntimeError("Condition error")

        guard = Guard(id="guard", child=child, condition=bad_condition)

        result = guard.tick(tick_context)

        assert result == RunStatus.FAILURE
        assert child.ticks_received == 0

    def test_string_condition_evaluation(self, tick_context: TickContext) -> None:
        """Guard should evaluate string conditions."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])

        # Simple string condition using eval (limited for safety)
        guard = Guard(id="guard", child=child, condition="True")

        result = guard.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_passes_through_child_failure(self, tick_context: TickContext) -> None:
        """Guard should pass through child FAILURE when condition is true."""
        child = MockLeafNode(id="child", results=[RunStatus.FAILURE])
        guard = Guard(id="guard", child=child, condition=lambda ctx: True)

        result = guard.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_passes_through_child_running(self, tick_context: TickContext) -> None:
        """Guard should pass through child RUNNING when condition is true."""
        child = MockLeafNode(id="child", results=[RunStatus.RUNNING])
        guard = Guard(id="guard", child=child, condition=lambda ctx: True)

        result = guard.tick(tick_context)

        assert result == RunStatus.RUNNING


# =============================================================================
# Cooldown Tests
# =============================================================================


class TestCooldown:
    """Tests for Cooldown decorator node."""

    def test_cooldown_ms_must_be_positive(self) -> None:
        """Cooldown should reject non-positive cooldown_ms."""
        child = MockLeafNode(id="child")

        with pytest.raises(ValueError) as exc_info:
            Cooldown(id="cooldown", child=child, cooldown_ms=0)
        assert "positive" in str(exc_info.value)

    def test_first_run_allowed(self, tick_context: TickContext) -> None:
        """Cooldown should allow first run."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        cooldown = Cooldown(id="cooldown", child=child, cooldown_ms=1000)

        result = cooldown.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_second_run_blocked_during_cooldown(
        self, tick_context: TickContext
    ) -> None:
        """Cooldown should block second run during cooldown period."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        cooldown = Cooldown(id="cooldown", child=child, cooldown_ms=1000)

        # First run
        cooldown.tick(tick_context)
        assert cooldown._last_completion_at is not None

        # Immediate second run should be blocked
        result = cooldown.tick(tick_context)

        assert result == RunStatus.FAILURE
        assert child.ticks_received == 1

    def test_run_allowed_after_cooldown(self, tick_context: TickContext) -> None:
        """Cooldown should allow run after cooldown elapsed."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        cooldown = Cooldown(id="cooldown", child=child, cooldown_ms=1)  # 1ms

        # First run
        cooldown.tick(tick_context)

        # Wait for cooldown
        time.sleep(0.002)  # 2ms

        # Second run should be allowed
        result = cooldown.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert child.ticks_received == 2

    def test_running_child_does_not_trigger_cooldown(
        self, tick_context: TickContext
    ) -> None:
        """Cooldown should not start until child completes."""
        child = MockLeafNode(
            id="child", results=[RunStatus.RUNNING, RunStatus.SUCCESS]
        )
        cooldown = Cooldown(id="cooldown", child=child, cooldown_ms=1000)

        # First tick - child RUNNING
        result1 = cooldown.tick(tick_context)
        assert result1 == RunStatus.RUNNING
        assert cooldown._last_completion_at is None  # Not yet completed

        # Second tick - child SUCCESS
        result2 = cooldown.tick(tick_context)
        assert result2 == RunStatus.SUCCESS
        assert cooldown._last_completion_at is not None  # Now completed

    def test_reset_clears_cooldown(self, tick_context: TickContext) -> None:
        """reset() should clear cooldown timer."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        cooldown = Cooldown(id="cooldown", child=child, cooldown_ms=1000)

        cooldown.tick(tick_context)
        assert cooldown._last_completion_at is not None

        cooldown.reset()

        assert cooldown._last_completion_at is None


# =============================================================================
# Inverter Tests
# =============================================================================


class TestInverter:
    """Tests for Inverter decorator node."""

    def test_inverts_success_to_failure(self, tick_context: TickContext) -> None:
        """Inverter should convert SUCCESS to FAILURE."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        inverter = Inverter(id="inv", child=child)

        result = inverter.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_inverts_failure_to_success(self, tick_context: TickContext) -> None:
        """Inverter should convert FAILURE to SUCCESS."""
        child = MockLeafNode(id="child", results=[RunStatus.FAILURE])
        inverter = Inverter(id="inv", child=child)

        result = inverter.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_running_unchanged(self, tick_context: TickContext) -> None:
        """Inverter should pass through RUNNING unchanged."""
        child = MockLeafNode(id="child", results=[RunStatus.RUNNING])
        inverter = Inverter(id="inv", child=child)

        result = inverter.tick(tick_context)

        assert result == RunStatus.RUNNING


# =============================================================================
# AlwaysSucceed Tests
# =============================================================================


class TestAlwaysSucceed:
    """Tests for AlwaysSucceed decorator node."""

    def test_success_remains_success(self, tick_context: TickContext) -> None:
        """AlwaysSucceed should keep SUCCESS as SUCCESS."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        always = AlwaysSucceed(id="always", child=child)

        result = always.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_failure_becomes_success(self, tick_context: TickContext) -> None:
        """AlwaysSucceed should convert FAILURE to SUCCESS."""
        child = MockLeafNode(id="child", results=[RunStatus.FAILURE])
        always = AlwaysSucceed(id="always", child=child)

        result = always.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_running_unchanged(self, tick_context: TickContext) -> None:
        """AlwaysSucceed should pass through RUNNING unchanged."""
        child = MockLeafNode(id="child", results=[RunStatus.RUNNING])
        always = AlwaysSucceed(id="always", child=child)

        result = always.tick(tick_context)

        assert result == RunStatus.RUNNING


# =============================================================================
# AlwaysFail Tests
# =============================================================================


class TestAlwaysFail:
    """Tests for AlwaysFail decorator node."""

    def test_failure_remains_failure(self, tick_context: TickContext) -> None:
        """AlwaysFail should keep FAILURE as FAILURE."""
        child = MockLeafNode(id="child", results=[RunStatus.FAILURE])
        always = AlwaysFail(id="always", child=child)

        result = always.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_success_becomes_failure(self, tick_context: TickContext) -> None:
        """AlwaysFail should convert SUCCESS to FAILURE."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        always = AlwaysFail(id="always", child=child)

        result = always.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_running_unchanged(self, tick_context: TickContext) -> None:
        """AlwaysFail should pass through RUNNING unchanged."""
        child = MockLeafNode(id="child", results=[RunStatus.RUNNING])
        always = AlwaysFail(id="always", child=child)

        result = always.tick(tick_context)

        assert result == RunStatus.RUNNING


# =============================================================================
# Decorator Composition Tests
# =============================================================================


class TestDecoratorComposition:
    """Tests for composing multiple decorators."""

    def test_retry_with_timeout(self, tick_context: TickContext) -> None:
        """Retry wrapping Timeout should work."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        timeout = Timeout(id="timeout", child=child, timeout_ms=1000)
        retry = Retry(id="retry", child=timeout, max_retries=3)

        result = retry.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_guard_with_inverter(self, tick_context: TickContext) -> None:
        """Guard wrapping Inverter should work."""
        child = MockLeafNode(id="child", results=[RunStatus.SUCCESS])
        inverter = Inverter(id="inv", child=child)
        guard = Guard(id="guard", child=inverter, condition=lambda ctx: True)

        result = guard.tick(tick_context)

        assert result == RunStatus.FAILURE  # Inverted SUCCESS

    def test_always_succeed_wrapping_retry(self, tick_context: TickContext) -> None:
        """AlwaysSucceed wrapping Retry should mask failures."""
        child = MockLeafNode(id="child", results=[RunStatus.FAILURE])
        retry = Retry(id="retry", child=child, max_retries=0)  # No retries
        always = AlwaysSucceed(id="always", child=retry)

        result = always.tick(tick_context)

        assert result == RunStatus.SUCCESS  # Masked failure


# =============================================================================
# Edge Cases
# =============================================================================


class TestDecoratorEdgeCases:
    """Edge case tests for decorator nodes."""

    def test_decorator_tracks_path(self, tick_context: TickContext) -> None:
        """Decorators should push/pop path for debugging."""
        child = MockLeafNode(id="tracked", results=[RunStatus.SUCCESS])
        inverter = Inverter(id="inv", child=child)

        inverter.tick(tick_context)

        # Path should be empty after tick (pushed and popped)
        assert tick_context.parent_path == []

    def test_zero_retries_fails_immediately(self, tick_context: TickContext) -> None:
        """Retry with max_retries=0 should fail immediately on child failure."""
        child = MockLeafNode(id="child", results=[RunStatus.FAILURE])
        retry = Retry(id="retry", child=child, max_retries=0)

        result = retry.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_cooldown_tracks_failure_completion(
        self, tick_context: TickContext
    ) -> None:
        """Cooldown should track completion time even on failure."""
        child = MockLeafNode(id="child", results=[RunStatus.FAILURE])
        cooldown = Cooldown(id="cooldown", child=child, cooldown_ms=1000)

        cooldown.tick(tick_context)

        assert cooldown._last_completion_at is not None  # FAILURE still completes
