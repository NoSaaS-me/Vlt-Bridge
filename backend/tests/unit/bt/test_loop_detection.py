"""Unit tests for loop detection conditions.

Tests for loop detection conditions in backend/src/bt/conditions/loop_detection.py.

Part of feature 020-bt-oracle-agent.
Tasks covered: T033 from tasks-expanded-us3.md

Acceptance Criteria Mapping:
- US3-AC3: 3 consecutive same reason triggers stuck -> test_consecutive_same_reason_exceeds_threshold
- FR-008: Detect loop patterns -> test_tool_loop_detection
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.src.bt.state.base import RunStatus


# =============================================================================
# Fixtures
# =============================================================================


class MockBlackboard:
    """Mock blackboard for testing."""

    def __init__(self, data: dict = None):
        self._data = data or {}
        self._writes = set()

    def _lookup(self, key: str):
        return self._data.get(key)


class MockContext:
    """Mock TickContext for testing."""

    def __init__(self, blackboard_data: dict = None):
        self.blackboard = MockBlackboard(blackboard_data)

    def mark_progress(self):
        pass


@pytest.fixture
def mock_config():
    """Mock oracle config with loop_threshold=3."""
    config = MagicMock()
    config.max_turns = 30
    config.iteration_warning_threshold = 0.70
    config.loop_threshold = 3
    return config


# =============================================================================
# TestIsStuckLoop
# =============================================================================


class TestIsStuckLoop:
    """Tests for is_stuck_loop condition."""

    def test_returns_success_when_consecutive_same_reason_exceeds_threshold(self, mock_config):
        """Should return SUCCESS when consecutive_same_reason >= 3 (US3-AC3)."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            ctx = MockContext({
                "consecutive_same_reason": 3,
                "loop_detected": False
            })
            result = is_stuck_loop(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_success_when_consecutive_exceeds_threshold(self, mock_config):
        """Should return SUCCESS when consecutive_same_reason > 3."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            ctx = MockContext({
                "consecutive_same_reason": 5,
                "loop_detected": False
            })
            result = is_stuck_loop(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_success_when_tool_loop_detected(self, mock_config):
        """Should return SUCCESS when loop_detected is True."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            ctx = MockContext({
                "consecutive_same_reason": 0,
                "loop_detected": True
            })
            result = is_stuck_loop(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_failure_when_no_loop(self, mock_config):
        """Should return FAILURE when no loop indicators."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            ctx = MockContext({
                "consecutive_same_reason": 2,
                "loop_detected": False
            })
            result = is_stuck_loop(ctx)
            assert result == RunStatus.FAILURE

    def test_returns_failure_with_empty_blackboard(self, mock_config):
        """Should return FAILURE with empty blackboard (defaults)."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            ctx = MockContext({})
            result = is_stuck_loop(ctx)
            assert result == RunStatus.FAILURE

    def test_returns_failure_when_no_blackboard(self, mock_config):
        """Should return FAILURE when blackboard is None."""
        from backend.src.bt.conditions.loop_detection import is_stuck_loop

        ctx = MockContext()
        ctx.blackboard = None
        result = is_stuck_loop(ctx)
        assert result == RunStatus.FAILURE

    def test_threshold_constant_is_3(self):
        """Default threshold constant should be 3."""
        from backend.src.bt.conditions.loop_detection import (
            CONSECUTIVE_SAME_REASON_THRESHOLD
        )
        assert CONSECUTIVE_SAME_REASON_THRESHOLD == 3


# =============================================================================
# TestHasRepeatedSignal
# =============================================================================


class TestHasRepeatedSignal:
    """Tests for has_repeated_signal condition."""

    def test_returns_success_when_reasons_match(self):
        """Should return SUCCESS when last two need_turn signals have same reason."""
        from backend.src.bt.conditions.loop_detection import has_repeated_signal

        signals = [
            {"type": "need_turn", "fields": {"reason": "waiting for tool"}},
            {"type": "need_turn", "fields": {"reason": "waiting for tool"}},
        ]
        ctx = MockContext({"signals_emitted": signals})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.SUCCESS

    def test_returns_failure_when_reasons_differ(self):
        """Should return FAILURE when reasons are different."""
        from backend.src.bt.conditions.loop_detection import has_repeated_signal

        signals = [
            {"type": "need_turn", "fields": {"reason": "waiting for tool"}},
            {"type": "need_turn", "fields": {"reason": "analyzing results"}},
        ]
        ctx = MockContext({"signals_emitted": signals})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.FAILURE

    def test_returns_failure_when_not_need_turn(self):
        """Should return FAILURE when signals are not need_turn type."""
        from backend.src.bt.conditions.loop_detection import has_repeated_signal

        signals = [
            {"type": "context_sufficient", "fields": {"sources_found": 3}},
            {"type": "context_sufficient", "fields": {"sources_found": 3}},
        ]
        ctx = MockContext({"signals_emitted": signals})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.FAILURE

    def test_returns_failure_when_first_not_need_turn(self):
        """Should return FAILURE when first signal is not need_turn."""
        from backend.src.bt.conditions.loop_detection import has_repeated_signal

        signals = [
            {"type": "context_sufficient", "fields": {}},
            {"type": "need_turn", "fields": {"reason": "waiting"}},
        ]
        ctx = MockContext({"signals_emitted": signals})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.FAILURE

    def test_returns_failure_when_less_than_two_signals(self):
        """Should return FAILURE when fewer than 2 signals."""
        from backend.src.bt.conditions.loop_detection import has_repeated_signal

        ctx = MockContext({"signals_emitted": [
            {"type": "need_turn", "fields": {"reason": "test"}}
        ]})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.FAILURE

    def test_returns_failure_when_no_signals(self):
        """Should return FAILURE when no signals emitted."""
        from backend.src.bt.conditions.loop_detection import has_repeated_signal

        ctx = MockContext({"signals_emitted": []})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.FAILURE

    def test_returns_failure_when_no_blackboard(self):
        """Should return FAILURE when blackboard is None."""
        from backend.src.bt.conditions.loop_detection import has_repeated_signal

        ctx = MockContext()
        ctx.blackboard = None
        result = has_repeated_signal(ctx)
        assert result == RunStatus.FAILURE

    def test_handles_enum_type_values(self):
        """Should handle signal type as enum string."""
        from backend.src.bt.conditions.loop_detection import has_repeated_signal

        signals = [
            {"type": "SignalType.NEED_TURN", "fields": {"reason": "test"}},
            {"type": "SignalType.NEED_TURN", "fields": {"reason": "test"}},
        ]
        ctx = MockContext({"signals_emitted": signals})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.SUCCESS

    def test_handles_missing_fields(self):
        """Should handle signals without fields dict."""
        from backend.src.bt.conditions.loop_detection import has_repeated_signal

        signals = [
            {"type": "need_turn"},  # No fields
            {"type": "need_turn"},
        ]
        ctx = MockContext({"signals_emitted": signals})
        result = has_repeated_signal(ctx)
        # Both have no reason, so they match (None == None is falsy but comparison is True)
        assert result == RunStatus.FAILURE  # None reason doesn't match


# =============================================================================
# TestHasRepeatedToolPattern
# =============================================================================


class TestHasRepeatedToolPattern:
    """Tests for has_repeated_tool_pattern condition."""

    def test_returns_success_when_pattern_repeats(self):
        """Should return SUCCESS when tool pattern repeats."""
        from backend.src.bt.conditions.loop_detection import has_repeated_tool_pattern

        # Pattern: [a, b, c, a, b, c] - first half equals second half
        patterns = ["search", "fetch", "parse", "search", "fetch", "parse"]
        ctx = MockContext({"recent_tool_patterns": patterns})
        result = has_repeated_tool_pattern(ctx, window=6)
        assert result == RunStatus.SUCCESS

    def test_returns_success_when_same_tool_three_times(self):
        """Should return SUCCESS when same tool called 3+ times consecutively."""
        from backend.src.bt.conditions.loop_detection import has_repeated_tool_pattern

        # Need 6 patterns for window=6, with last 3 being the same
        patterns = ["a", "b", "c", "search", "search", "search"]
        ctx = MockContext({"recent_tool_patterns": patterns})
        result = has_repeated_tool_pattern(ctx, window=6)
        assert result == RunStatus.SUCCESS

    def test_returns_failure_when_no_pattern(self):
        """Should return FAILURE when no repeated pattern."""
        from backend.src.bt.conditions.loop_detection import has_repeated_tool_pattern

        patterns = ["search", "fetch", "parse", "analyze", "store", "done"]
        ctx = MockContext({"recent_tool_patterns": patterns})
        result = has_repeated_tool_pattern(ctx, window=6)
        assert result == RunStatus.FAILURE

    def test_returns_failure_when_not_enough_patterns(self):
        """Should return FAILURE when fewer patterns than window."""
        from backend.src.bt.conditions.loop_detection import has_repeated_tool_pattern

        patterns = ["search", "fetch"]
        ctx = MockContext({"recent_tool_patterns": patterns})
        result = has_repeated_tool_pattern(ctx, window=6)
        assert result == RunStatus.FAILURE

    def test_returns_failure_when_no_blackboard(self):
        """Should return FAILURE when blackboard is None."""
        from backend.src.bt.conditions.loop_detection import has_repeated_tool_pattern

        ctx = MockContext()
        ctx.blackboard = None
        result = has_repeated_tool_pattern(ctx)
        assert result == RunStatus.FAILURE


# =============================================================================
# TestLoopDetectionIntegration
# =============================================================================


class TestLoopDetectionIntegration:
    """Integration tests for loop detection."""

    def test_three_consecutive_same_reason_triggers_stuck(self, mock_config):
        """3 consecutive same reason signals should trigger stuck (US3-AC3)."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            signals = [
                {"type": "need_turn", "fields": {"reason": "retrying failed tool"}},
                {"type": "need_turn", "fields": {"reason": "retrying failed tool"}},
                {"type": "need_turn", "fields": {"reason": "retrying failed tool"}},
            ]
            ctx = MockContext({
                "signals_emitted": signals,
                "consecutive_same_reason": 3,
                "loop_detected": False
            })
            assert is_stuck_loop(ctx) == RunStatus.SUCCESS

    def test_two_consecutive_does_not_trigger(self, mock_config):
        """2 consecutive same reason signals should NOT trigger stuck."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            signals = [
                {"type": "need_turn", "fields": {"reason": "retrying failed tool"}},
                {"type": "need_turn", "fields": {"reason": "retrying failed tool"}},
            ]
            ctx = MockContext({
                "signals_emitted": signals,
                "consecutive_same_reason": 2,
                "loop_detected": False
            })
            assert is_stuck_loop(ctx) == RunStatus.FAILURE

    def test_different_reason_resets_detection(self, mock_config):
        """Different reason between signals should not trigger."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            # Third signal has different reason
            signals = [
                {"type": "need_turn", "fields": {"reason": "reason A"}},
                {"type": "need_turn", "fields": {"reason": "reason A"}},
                {"type": "need_turn", "fields": {"reason": "reason B"}},
            ]
            ctx = MockContext({
                "signals_emitted": signals,
                "consecutive_same_reason": 1,  # Reset due to different reason
                "loop_detected": False
            })
            assert is_stuck_loop(ctx) == RunStatus.FAILURE


# =============================================================================
# TestConsecutiveReasonCount
# =============================================================================


class TestConsecutiveReasonCount:
    """Tests for consecutive_reason_count helper."""

    def test_returns_correct_count(self):
        """Should return the consecutive_same_reason value."""
        from backend.src.bt.conditions.loop_detection import consecutive_reason_count

        ctx = MockContext({"consecutive_same_reason": 5})
        count = consecutive_reason_count(ctx)
        assert count == 5

    def test_returns_zero_when_not_set(self):
        """Should return 0 when consecutive_same_reason not set."""
        from backend.src.bt.conditions.loop_detection import consecutive_reason_count

        ctx = MockContext({})
        count = consecutive_reason_count(ctx)
        assert count == 0

    def test_returns_zero_when_no_blackboard(self):
        """Should return 0 when blackboard is None."""
        from backend.src.bt.conditions.loop_detection import consecutive_reason_count

        ctx = MockContext()
        ctx.blackboard = None
        count = consecutive_reason_count(ctx)
        assert count == 0


# =============================================================================
# TestCustomThreshold
# =============================================================================


class TestCustomThreshold:
    """Tests with custom loop_threshold configuration."""

    def test_respects_custom_threshold_5(self):
        """Should use loop_threshold from config (threshold=5)."""
        config = MagicMock()
        config.loop_threshold = 5

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            # 4 consecutive should NOT trigger with threshold=5
            ctx = MockContext({
                "consecutive_same_reason": 4,
                "loop_detected": False
            })
            assert is_stuck_loop(ctx) == RunStatus.FAILURE

            # 5 consecutive SHOULD trigger with threshold=5
            ctx = MockContext({
                "consecutive_same_reason": 5,
                "loop_detected": False
            })
            assert is_stuck_loop(ctx) == RunStatus.SUCCESS

    def test_respects_custom_threshold_1(self):
        """Should use loop_threshold from config (threshold=1)."""
        config = MagicMock()
        config.loop_threshold = 1

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            # Even 1 consecutive should trigger with threshold=1
            ctx = MockContext({
                "consecutive_same_reason": 1,
                "loop_detected": False
            })
            assert is_stuck_loop(ctx) == RunStatus.SUCCESS


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for loop detection."""

    def test_handles_string_consecutive_value(self, mock_config):
        """Should handle consecutive_same_reason as string."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            ctx = MockContext({
                "consecutive_same_reason": "3",  # String
                "loop_detected": False
            })
            result = is_stuck_loop(ctx)
            assert result == RunStatus.SUCCESS

    def test_handles_invalid_consecutive_value(self, mock_config):
        """Should handle invalid consecutive_same_reason value."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.loop_detection import is_stuck_loop

            ctx = MockContext({
                "consecutive_same_reason": "invalid",  # Invalid
                "loop_detected": False
            })
            result = is_stuck_loop(ctx)
            assert result == RunStatus.FAILURE  # Defaults to 0

    def test_handles_signals_not_list(self):
        """Should handle signals_emitted not being a list."""
        from backend.src.bt.conditions.loop_detection import has_repeated_signal

        ctx = MockContext({"signals_emitted": "not a list"})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.FAILURE

    def test_handles_tool_patterns_not_list(self):
        """Should handle recent_tool_patterns not being a list."""
        from backend.src.bt.conditions.loop_detection import has_repeated_tool_pattern

        ctx = MockContext({"recent_tool_patterns": "not a list"})
        result = has_repeated_tool_pattern(ctx)
        assert result == RunStatus.FAILURE
