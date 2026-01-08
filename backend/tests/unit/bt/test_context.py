"""
Unit tests for TickContext.

Tests:
- TickContext creation
- Budget tracking
- Path management
- Async tracking
- Progress tracking (footgun A.1)
- Cancellation
- Context copying

Part of the BT Universal Runtime (spec 019).
"""

import pytest
import time
from datetime import datetime, timedelta, timezone

from backend.src.bt.core.context import TickContext
from backend.src.bt.state.blackboard import TypedBlackboard


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_blackboard() -> TypedBlackboard:
    """Create a basic blackboard for testing."""
    return TypedBlackboard(scope_name="test")


@pytest.fixture
def basic_context(basic_blackboard: TypedBlackboard) -> TickContext:
    """Create a basic tick context."""
    return TickContext(blackboard=basic_blackboard)


# =============================================================================
# Creation Tests
# =============================================================================


class TestTickContextCreation:
    """Tests for TickContext creation."""

    def test_default_creation(self) -> None:
        """TickContext should have sensible defaults."""
        ctx = TickContext()

        assert ctx.event is None
        assert ctx.blackboard is None
        assert ctx.services is None
        assert ctx.tick_count == 0
        assert ctx.tick_budget == 1000
        assert ctx.parent_path == []
        assert ctx.trace_enabled is False
        assert ctx.async_pending == set()
        assert ctx.cancellation_requested is False
        assert ctx.cancellation_reason is None
        assert ctx.last_progress_at is None

    def test_creation_with_blackboard(self, basic_blackboard: TypedBlackboard) -> None:
        """TickContext should accept a blackboard."""
        ctx = TickContext(blackboard=basic_blackboard)

        assert ctx.blackboard is basic_blackboard

    def test_creation_with_custom_budget(self) -> None:
        """TickContext should accept custom tick budget."""
        ctx = TickContext(tick_budget=500)

        assert ctx.tick_budget == 500

    def test_creation_with_start_time(self) -> None:
        """TickContext should have start_time set automatically."""
        before = datetime.now(timezone.utc)
        ctx = TickContext()
        after = datetime.now(timezone.utc)

        assert before <= ctx.start_time <= after

    def test_creation_with_trace_enabled(self) -> None:
        """TickContext should accept trace_enabled flag."""
        ctx = TickContext(trace_enabled=True)

        assert ctx.trace_enabled is True


# =============================================================================
# Budget Tracking Tests
# =============================================================================


class TestBudgetTracking:
    """Tests for tick budget tracking."""

    def test_budget_remaining_initial(self) -> None:
        """budget_remaining should return full budget initially."""
        ctx = TickContext(tick_budget=100)

        assert ctx.budget_remaining() == 100

    def test_budget_remaining_after_ticks(self) -> None:
        """budget_remaining should decrease after ticks."""
        ctx = TickContext(tick_budget=100, tick_count=30)

        assert ctx.budget_remaining() == 70

    def test_budget_remaining_negative(self) -> None:
        """budget_remaining can be negative if exceeded."""
        ctx = TickContext(tick_budget=100, tick_count=110)

        assert ctx.budget_remaining() == -10

    def test_budget_exceeded_false(self) -> None:
        """budget_exceeded should be False when budget remains."""
        ctx = TickContext(tick_budget=100, tick_count=50)

        assert ctx.budget_exceeded() is False

    def test_budget_exceeded_true(self) -> None:
        """budget_exceeded should be True when budget exhausted."""
        ctx = TickContext(tick_budget=100, tick_count=100)

        assert ctx.budget_exceeded() is True

    def test_budget_exceeded_over(self) -> None:
        """budget_exceeded should be True when over budget."""
        ctx = TickContext(tick_budget=100, tick_count=150)

        assert ctx.budget_exceeded() is True

    def test_increment_tick(self) -> None:
        """increment_tick should increase tick_count."""
        ctx = TickContext(tick_count=0)

        ctx.increment_tick()
        assert ctx.tick_count == 1

        ctx.increment_tick()
        assert ctx.tick_count == 2


# =============================================================================
# Elapsed Time Tests
# =============================================================================


class TestElapsedTime:
    """Tests for elapsed time tracking."""

    def test_elapsed_ms_initial(self) -> None:
        """elapsed_ms should be small initially."""
        ctx = TickContext()

        elapsed = ctx.elapsed_ms()

        # Should be very small (less than 100ms for sure)
        assert elapsed >= 0
        assert elapsed < 100

    def test_elapsed_ms_increases(self) -> None:
        """elapsed_ms should increase over time."""
        ctx = TickContext()

        elapsed1 = ctx.elapsed_ms()
        time.sleep(0.01)  # Sleep 10ms
        elapsed2 = ctx.elapsed_ms()

        assert elapsed2 > elapsed1

    def test_elapsed_ms_with_custom_start(self) -> None:
        """elapsed_ms should work with custom start time."""
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        ctx = TickContext(start_time=past)

        elapsed = ctx.elapsed_ms()

        # Should be at least 1000ms (1 second)
        assert elapsed >= 1000


# =============================================================================
# Path Management Tests
# =============================================================================


class TestPathManagement:
    """Tests for parent path tracking."""

    def test_push_path(self) -> None:
        """push_path should add node to path."""
        ctx = TickContext()

        ctx.push_path("node1")

        assert ctx.parent_path == ["node1"]

    def test_push_path_multiple(self) -> None:
        """push_path should build up path."""
        ctx = TickContext()

        ctx.push_path("root")
        ctx.push_path("child")
        ctx.push_path("grandchild")

        assert ctx.parent_path == ["root", "child", "grandchild"]

    def test_pop_path(self) -> None:
        """pop_path should remove and return last node."""
        ctx = TickContext()
        ctx.push_path("node1")
        ctx.push_path("node2")

        result = ctx.pop_path()

        assert result == "node2"
        assert ctx.parent_path == ["node1"]

    def test_pop_path_empty(self) -> None:
        """pop_path should return None when empty."""
        ctx = TickContext()

        result = ctx.pop_path()

        assert result is None

    def test_get_current_path(self) -> None:
        """get_current_path should return formatted string."""
        ctx = TickContext()
        ctx.push_path("root")
        ctx.push_path("child")

        path = ctx.get_current_path()

        assert path == "root > child"

    def test_get_current_path_empty(self) -> None:
        """get_current_path should return empty string when empty."""
        ctx = TickContext()

        path = ctx.get_current_path()

        assert path == ""


# =============================================================================
# Async Tracking Tests
# =============================================================================


class TestAsyncTracking:
    """Tests for async operation tracking."""

    def test_add_async(self) -> None:
        """add_async should register operation."""
        ctx = TickContext()

        ctx.add_async("op1")

        assert "op1" in ctx.async_pending

    def test_add_async_multiple(self) -> None:
        """add_async should handle multiple operations."""
        ctx = TickContext()

        ctx.add_async("op1")
        ctx.add_async("op2")
        ctx.add_async("op3")

        assert ctx.async_pending == {"op1", "op2", "op3"}

    def test_complete_async(self) -> None:
        """complete_async should remove operation."""
        ctx = TickContext()
        ctx.add_async("op1")
        ctx.add_async("op2")

        ctx.complete_async("op1")

        assert "op1" not in ctx.async_pending
        assert "op2" in ctx.async_pending

    def test_complete_async_nonexistent(self) -> None:
        """complete_async should be safe for non-existent operations."""
        ctx = TickContext()

        # Should not raise
        ctx.complete_async("nonexistent")

    def test_has_pending_async_true(self) -> None:
        """has_pending_async should be True when operations pending."""
        ctx = TickContext()
        ctx.add_async("op1")

        assert ctx.has_pending_async() is True

    def test_has_pending_async_false(self) -> None:
        """has_pending_async should be False when empty."""
        ctx = TickContext()

        assert ctx.has_pending_async() is False

    def test_complete_async_marks_progress(self) -> None:
        """complete_async should mark progress (footgun A.1)."""
        ctx = TickContext()
        ctx.add_async("op1")

        assert ctx.last_progress_at is None

        ctx.complete_async("op1")

        assert ctx.last_progress_at is not None


# =============================================================================
# Progress Tracking Tests (footgun A.1)
# =============================================================================


class TestProgressTracking:
    """Tests for progress tracking per footgun A.1."""

    def test_mark_progress(self) -> None:
        """mark_progress should set last_progress_at."""
        ctx = TickContext()

        assert ctx.last_progress_at is None

        before = datetime.now(timezone.utc)
        ctx.mark_progress()
        after = datetime.now(timezone.utc)

        assert ctx.last_progress_at is not None
        assert before <= ctx.last_progress_at <= after

    def test_mark_progress_updates(self) -> None:
        """mark_progress should update timestamp on each call."""
        ctx = TickContext()

        ctx.mark_progress()
        first = ctx.last_progress_at

        time.sleep(0.01)  # Small delay
        ctx.mark_progress()
        second = ctx.last_progress_at

        assert second > first

    def test_time_since_progress_ms_none(self) -> None:
        """time_since_progress_ms should return None if no progress."""
        ctx = TickContext()

        result = ctx.time_since_progress_ms()

        assert result is None

    def test_time_since_progress_ms_after_mark(self) -> None:
        """time_since_progress_ms should return elapsed time."""
        ctx = TickContext()
        ctx.mark_progress()

        time.sleep(0.01)  # Sleep 10ms

        result = ctx.time_since_progress_ms()

        assert result is not None
        assert result >= 10  # At least 10ms

    def test_last_progress_at_property(self) -> None:
        """last_progress_at property should work."""
        ctx = TickContext()

        assert ctx.last_progress_at is None

        ctx.mark_progress()

        assert ctx.last_progress_at is not None


# =============================================================================
# Cancellation Tests
# =============================================================================


class TestCancellation:
    """Tests for cancellation handling."""

    def test_request_cancellation(self) -> None:
        """request_cancellation should set flag."""
        ctx = TickContext()

        ctx.request_cancellation()

        assert ctx.cancellation_requested is True

    def test_request_cancellation_with_reason(self) -> None:
        """request_cancellation should set reason."""
        ctx = TickContext()

        ctx.request_cancellation("User pressed stop")

        assert ctx.cancellation_requested is True
        assert ctx.cancellation_reason == "User pressed stop"

    def test_clear_cancellation(self) -> None:
        """clear_cancellation should reset flags."""
        ctx = TickContext(
            cancellation_requested=True,
            cancellation_reason="Test",
        )

        ctx.clear_cancellation()

        assert ctx.cancellation_requested is False
        assert ctx.cancellation_reason is None


# =============================================================================
# Context Copying Tests
# =============================================================================


class TestContextCopying:
    """Tests for with_blackboard context copying."""

    def test_with_blackboard_creates_copy(
        self, basic_blackboard: TypedBlackboard
    ) -> None:
        """with_blackboard should create a new context."""
        original = TickContext(
            blackboard=basic_blackboard,
            tick_count=5,
            tick_budget=100,
            trace_enabled=True,
        )

        new_bb = TypedBlackboard(scope_name="new")
        copy = original.with_blackboard(new_bb)

        assert copy is not original
        assert copy.blackboard is new_bb

    def test_with_blackboard_copies_state(
        self, basic_blackboard: TypedBlackboard
    ) -> None:
        """with_blackboard should copy state."""
        original = TickContext(
            blackboard=basic_blackboard,
            tick_count=5,
            tick_budget=100,
            trace_enabled=True,
        )
        original.push_path("node1")
        original.add_async("op1")
        original.mark_progress()

        new_bb = TypedBlackboard(scope_name="new")
        copy = original.with_blackboard(new_bb)

        assert copy.tick_count == 5
        assert copy.tick_budget == 100
        assert copy.trace_enabled is True
        assert copy.parent_path == ["node1"]
        assert copy.async_pending == {"op1"}
        assert copy.last_progress_at == original.last_progress_at

    def test_with_blackboard_isolates_path(
        self, basic_blackboard: TypedBlackboard
    ) -> None:
        """with_blackboard should isolate path list."""
        original = TickContext(blackboard=basic_blackboard)
        original.push_path("node1")

        new_bb = TypedBlackboard(scope_name="new")
        copy = original.with_blackboard(new_bb)

        # Modify copy's path
        copy.push_path("node2")

        # Original should be unchanged
        assert original.parent_path == ["node1"]
        assert copy.parent_path == ["node1", "node2"]

    def test_with_blackboard_isolates_async(
        self, basic_blackboard: TypedBlackboard
    ) -> None:
        """with_blackboard should isolate async set."""
        original = TickContext(blackboard=basic_blackboard)
        original.add_async("op1")

        new_bb = TypedBlackboard(scope_name="new")
        copy = original.with_blackboard(new_bb)

        # Modify copy's async
        copy.add_async("op2")

        # Original should be unchanged
        assert original.async_pending == {"op1"}
        assert copy.async_pending == {"op1", "op2"}


# =============================================================================
# Debug Info Tests
# =============================================================================


class TestDebugInfo:
    """Tests for debug_info method."""

    def test_debug_info_contains_fields(self) -> None:
        """debug_info should contain all relevant fields."""
        ctx = TickContext(tick_count=5, tick_budget=100)
        ctx.push_path("root")
        ctx.add_async("op1")
        ctx.mark_progress()

        info = ctx.debug_info()

        assert info["tick_count"] == 5
        assert info["tick_budget"] == 100
        assert info["budget_remaining"] == 95
        assert info["elapsed_ms"] >= 0
        assert info["parent_path"] == ["root"]
        assert info["async_pending"] == ["op1"]
        assert info["cancellation_requested"] is False
        assert info["cancellation_reason"] is None
        assert info["trace_enabled"] is False
        assert info["last_progress_at"] is not None
        assert info["time_since_progress_ms"] >= 0

    def test_debug_info_handles_no_progress(self) -> None:
        """debug_info should handle no progress gracefully."""
        ctx = TickContext()

        info = ctx.debug_info()

        assert info["last_progress_at"] is None
        assert info["time_since_progress_ms"] is None
