"""Unit tests for signal condition functions.

Tests the BT condition functions for signal checking in signals.py.

Part of feature 020-bt-oracle-agent.
Tasks covered: T025 from tasks-expanded-us2.md

Acceptance Criteria Mapping:
- AC-4a: BT parses signal -> check_signal() returns SUCCESS after parsing
- AC-4b: BT acts on signal -> Conditions drive BT routing
- US3-AC-3: 3x same reason = stuck -> consecutive_same_reason_gte(ctx, 3)
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from backend.src.bt.state.base import RunStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ctx():
    """Create mock TickContext with blackboard.

    The blackboard uses a dict-based _lookup method for simplicity.
    """
    ctx = MagicMock()
    ctx.blackboard = MagicMock()
    ctx.blackboard._data = {}
    ctx.blackboard._lookup = lambda key: ctx.blackboard._data.get(key)
    return ctx


@pytest.fixture
def need_turn_signal():
    """Sample need_turn signal dict."""
    return {
        "type": "need_turn",
        "confidence": 0.85,
        "fields": {
            "reason": "Need to verify the API response format",
            "expected_turns": 2,
        },
        "raw_xml": '<signal type="need_turn">...</signal>',
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def context_sufficient_signal():
    """Sample context_sufficient signal dict."""
    return {
        "type": "context_sufficient",
        "confidence": 0.92,
        "fields": {
            "sources_found": 3,
            "source_types": ["code", "docs"],
        },
        "raw_xml": '<signal type="context_sufficient">...</signal>',
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def stuck_signal():
    """Sample stuck signal dict."""
    return {
        "type": "stuck",
        "confidence": 0.78,
        "fields": {
            "attempted": ["search_code", "search_vault"],
            "blocker": "No results found for the query",
            "suggestions": ["Try web search"],
        },
        "raw_xml": '<signal type="stuck">...</signal>',
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# TestCheckSignal
# =============================================================================


class TestCheckSignal:
    """Tests for check_signal condition."""

    def test_returns_success_when_signal_parsed_this_turn(self, mock_ctx):
        """Should return SUCCESS when _signal_parsed_this_turn is True."""
        mock_ctx.blackboard._data["_signal_parsed_this_turn"] = True

        from backend.src.bt.conditions.signals import check_signal
        result = check_signal(mock_ctx)

        assert result == RunStatus.SUCCESS

    def test_returns_failure_when_no_signal_parsed(self, mock_ctx):
        """Should return FAILURE when _signal_parsed_this_turn is False."""
        mock_ctx.blackboard._data["_signal_parsed_this_turn"] = False

        from backend.src.bt.conditions.signals import check_signal
        result = check_signal(mock_ctx)

        assert result == RunStatus.FAILURE

    def test_returns_failure_when_flag_not_set(self, mock_ctx):
        """Should return FAILURE when flag is not set at all."""
        # _signal_parsed_this_turn not in _data

        from backend.src.bt.conditions.signals import check_signal
        result = check_signal(mock_ctx)

        assert result == RunStatus.FAILURE

    def test_returns_failure_when_blackboard_none(self, mock_ctx):
        """Should return FAILURE when blackboard is None."""
        mock_ctx.blackboard = None

        from backend.src.bt.conditions.signals import check_signal
        result = check_signal(mock_ctx)

        assert result == RunStatus.FAILURE


# =============================================================================
# TestHasSignal
# =============================================================================


class TestHasSignal:
    """Tests for has_signal condition."""

    def test_returns_success_with_any_signal(self, mock_ctx, need_turn_signal):
        """Should return SUCCESS when any signal exists."""
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal

        from backend.src.bt.conditions.signals import has_signal
        result = has_signal(mock_ctx)

        assert result == RunStatus.SUCCESS

    def test_returns_success_with_matching_type(self, mock_ctx, need_turn_signal):
        """Should return SUCCESS when signal type matches."""
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal

        from backend.src.bt.conditions.signals import has_signal
        result = has_signal(mock_ctx, signal_type="need_turn")

        assert result == RunStatus.SUCCESS

    def test_returns_failure_with_wrong_type(self, mock_ctx, need_turn_signal):
        """Should return FAILURE when signal type doesn't match."""
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal

        from backend.src.bt.conditions.signals import has_signal
        result = has_signal(mock_ctx, signal_type="stuck")

        assert result == RunStatus.FAILURE

    def test_returns_failure_when_no_signal(self, mock_ctx):
        """Should return FAILURE when no signal in blackboard."""
        mock_ctx.blackboard._data["last_signal"] = None

        from backend.src.bt.conditions.signals import has_signal
        result = has_signal(mock_ctx)

        assert result == RunStatus.FAILURE

    def test_returns_failure_when_blackboard_none(self, mock_ctx):
        """Should return FAILURE when blackboard is None."""
        mock_ctx.blackboard = None

        from backend.src.bt.conditions.signals import has_signal
        result = has_signal(mock_ctx)

        assert result == RunStatus.FAILURE

    def test_handles_enum_type_value(self, mock_ctx):
        """Should handle signal type as enum value string."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "SignalType.NEED_TURN",  # Enum string representation
            "confidence": 0.8,
            "fields": {"reason": "test"},
        }

        from backend.src.bt.conditions.signals import has_signal
        result = has_signal(mock_ctx, signal_type="need_turn")

        assert result == RunStatus.SUCCESS


# =============================================================================
# TestSignalTypeIs
# =============================================================================


class TestSignalTypeIs:
    """Tests for signal_type_is condition."""

    def test_need_turn_match(self, mock_ctx, need_turn_signal):
        """Should return SUCCESS for need_turn type match."""
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal

        from backend.src.bt.conditions.signals import signal_type_is
        result = signal_type_is(mock_ctx, "need_turn")

        assert result == RunStatus.SUCCESS

    def test_context_sufficient_match(self, mock_ctx, context_sufficient_signal):
        """Should return SUCCESS for context_sufficient type match."""
        mock_ctx.blackboard._data["last_signal"] = context_sufficient_signal

        from backend.src.bt.conditions.signals import signal_type_is
        result = signal_type_is(mock_ctx, "context_sufficient")

        assert result == RunStatus.SUCCESS

    def test_stuck_match(self, mock_ctx, stuck_signal):
        """Should return SUCCESS for stuck type match."""
        mock_ctx.blackboard._data["last_signal"] = stuck_signal

        from backend.src.bt.conditions.signals import signal_type_is
        result = signal_type_is(mock_ctx, "stuck")

        assert result == RunStatus.SUCCESS

    def test_mismatch_returns_failure(self, mock_ctx, need_turn_signal):
        """Should return FAILURE when types don't match."""
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal

        from backend.src.bt.conditions.signals import signal_type_is
        result = signal_type_is(mock_ctx, "stuck")

        assert result == RunStatus.FAILURE

    def test_no_signal_returns_failure(self, mock_ctx):
        """Should return FAILURE when no signal exists."""
        mock_ctx.blackboard._data["last_signal"] = None

        from backend.src.bt.conditions.signals import signal_type_is
        result = signal_type_is(mock_ctx, "need_turn")

        assert result == RunStatus.FAILURE

    def test_case_insensitive_match(self, mock_ctx):
        """Should match signal types case-insensitively."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "NEED_TURN",  # Uppercase
            "confidence": 0.8,
            "fields": {},
        }

        from backend.src.bt.conditions.signals import signal_type_is
        result = signal_type_is(mock_ctx, "need_turn")

        assert result == RunStatus.SUCCESS


# =============================================================================
# TestSignalConfidenceAbove
# =============================================================================


class TestSignalConfidenceAbove:
    """Tests for signal_confidence_above condition."""

    def test_confidence_above_threshold(self, mock_ctx, need_turn_signal):
        """Should return SUCCESS when confidence is above threshold."""
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal  # confidence = 0.85

        from backend.src.bt.conditions.signals import signal_confidence_above
        result = signal_confidence_above(mock_ctx, threshold=0.7)

        assert result == RunStatus.SUCCESS

    def test_confidence_at_threshold(self, mock_ctx):
        """Should return SUCCESS when confidence equals threshold."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "confidence": 0.5,
            "fields": {},
        }

        from backend.src.bt.conditions.signals import signal_confidence_above
        result = signal_confidence_above(mock_ctx, threshold=0.5)

        assert result == RunStatus.SUCCESS

    def test_confidence_below_threshold(self, mock_ctx):
        """Should return FAILURE when confidence is below threshold."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "confidence": 0.3,
            "fields": {},
        }

        from backend.src.bt.conditions.signals import signal_confidence_above
        result = signal_confidence_above(mock_ctx, threshold=0.5)

        assert result == RunStatus.FAILURE

    def test_default_threshold(self, mock_ctx):
        """Should use default threshold of 0.5."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "confidence": 0.6,
            "fields": {},
        }

        from backend.src.bt.conditions.signals import signal_confidence_above
        result = signal_confidence_above(mock_ctx)  # Default 0.5

        assert result == RunStatus.SUCCESS

    def test_no_signal_returns_failure(self, mock_ctx):
        """Should return FAILURE when no signal exists."""
        mock_ctx.blackboard._data["last_signal"] = None

        from backend.src.bt.conditions.signals import signal_confidence_above
        result = signal_confidence_above(mock_ctx)

        assert result == RunStatus.FAILURE

    def test_missing_confidence_returns_failure(self, mock_ctx):
        """Should return FAILURE when signal has no confidence field."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "fields": {},
            # No confidence field
        }

        from backend.src.bt.conditions.signals import signal_confidence_above
        result = signal_confidence_above(mock_ctx)

        assert result == RunStatus.FAILURE


# =============================================================================
# TestConsecutiveSameReasonGte
# =============================================================================


class TestConsecutiveSameReasonGte:
    """Tests for consecutive_same_reason_gte condition (loop detection)."""

    def test_at_threshold_returns_success(self, mock_ctx):
        """Should return SUCCESS when count equals threshold."""
        mock_ctx.blackboard._data["consecutive_same_reason"] = 3

        from backend.src.bt.conditions.signals import consecutive_same_reason_gte
        result = consecutive_same_reason_gte(mock_ctx, count=3)

        assert result == RunStatus.SUCCESS

    def test_above_threshold_returns_success(self, mock_ctx):
        """Should return SUCCESS when count exceeds threshold."""
        mock_ctx.blackboard._data["consecutive_same_reason"] = 5

        from backend.src.bt.conditions.signals import consecutive_same_reason_gte
        result = consecutive_same_reason_gte(mock_ctx, count=3)

        assert result == RunStatus.SUCCESS

    def test_below_threshold_returns_failure(self, mock_ctx):
        """Should return FAILURE when count is below threshold."""
        mock_ctx.blackboard._data["consecutive_same_reason"] = 2

        from backend.src.bt.conditions.signals import consecutive_same_reason_gte
        result = consecutive_same_reason_gte(mock_ctx, count=3)

        assert result == RunStatus.FAILURE

    def test_default_threshold(self, mock_ctx):
        """Should use default threshold of 3."""
        mock_ctx.blackboard._data["consecutive_same_reason"] = 3

        from backend.src.bt.conditions.signals import consecutive_same_reason_gte
        result = consecutive_same_reason_gte(mock_ctx)  # Default 3

        assert result == RunStatus.SUCCESS

    def test_zero_count_returns_failure(self, mock_ctx):
        """Should return FAILURE when count is 0."""
        mock_ctx.blackboard._data["consecutive_same_reason"] = 0

        from backend.src.bt.conditions.signals import consecutive_same_reason_gte
        result = consecutive_same_reason_gte(mock_ctx)

        assert result == RunStatus.FAILURE

    def test_missing_counter_returns_failure(self, mock_ctx):
        """Should return FAILURE when counter is not set."""
        # consecutive_same_reason not in _data

        from backend.src.bt.conditions.signals import consecutive_same_reason_gte
        result = consecutive_same_reason_gte(mock_ctx)

        assert result == RunStatus.FAILURE


# =============================================================================
# TestTurnsWithoutSignalGte
# =============================================================================


class TestTurnsWithoutSignalGte:
    """Tests for turns_without_signal_gte condition (fallback trigger)."""

    def test_at_threshold_returns_success(self, mock_ctx):
        """Should return SUCCESS when turns equals threshold."""
        mock_ctx.blackboard._data["turns_without_signal"] = 3

        from backend.src.bt.conditions.signals import turns_without_signal_gte
        result = turns_without_signal_gte(mock_ctx, count=3)

        assert result == RunStatus.SUCCESS

    def test_above_threshold_returns_success(self, mock_ctx):
        """Should return SUCCESS when turns exceeds threshold."""
        mock_ctx.blackboard._data["turns_without_signal"] = 5

        from backend.src.bt.conditions.signals import turns_without_signal_gte
        result = turns_without_signal_gte(mock_ctx, count=3)

        assert result == RunStatus.SUCCESS

    def test_below_threshold_returns_failure(self, mock_ctx):
        """Should return FAILURE when turns is below threshold."""
        mock_ctx.blackboard._data["turns_without_signal"] = 2

        from backend.src.bt.conditions.signals import turns_without_signal_gte
        result = turns_without_signal_gte(mock_ctx, count=3)

        assert result == RunStatus.FAILURE


# =============================================================================
# TestSignalIsTerminal
# =============================================================================


class TestSignalIsTerminal:
    """Tests for signal_is_terminal condition."""

    def test_context_sufficient_is_terminal(self, mock_ctx, context_sufficient_signal):
        """context_sufficient should be terminal."""
        mock_ctx.blackboard._data["last_signal"] = context_sufficient_signal

        from backend.src.bt.conditions.signals import signal_is_terminal
        result = signal_is_terminal(mock_ctx)

        assert result == RunStatus.SUCCESS

    def test_stuck_is_terminal(self, mock_ctx, stuck_signal):
        """stuck should be terminal."""
        mock_ctx.blackboard._data["last_signal"] = stuck_signal

        from backend.src.bt.conditions.signals import signal_is_terminal
        result = signal_is_terminal(mock_ctx)

        assert result == RunStatus.SUCCESS

    def test_partial_answer_is_terminal(self, mock_ctx):
        """partial_answer should be terminal."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "partial_answer",
            "confidence": 0.7,
            "fields": {"missing": "some data"},
        }

        from backend.src.bt.conditions.signals import signal_is_terminal
        result = signal_is_terminal(mock_ctx)

        assert result == RunStatus.SUCCESS

    def test_need_turn_is_not_terminal(self, mock_ctx, need_turn_signal):
        """need_turn should NOT be terminal."""
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal

        from backend.src.bt.conditions.signals import signal_is_terminal
        result = signal_is_terminal(mock_ctx)

        assert result == RunStatus.FAILURE


# =============================================================================
# TestSignalIsContinuation
# =============================================================================


class TestSignalIsContinuation:
    """Tests for signal_is_continuation condition."""

    def test_need_turn_is_continuation(self, mock_ctx, need_turn_signal):
        """need_turn should be continuation."""
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal

        from backend.src.bt.conditions.signals import signal_is_continuation
        result = signal_is_continuation(mock_ctx)

        assert result == RunStatus.SUCCESS

    def test_need_capability_is_continuation(self, mock_ctx):
        """need_capability should be continuation."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_capability",
            "confidence": 0.8,
            "fields": {"capability": "file_write", "reason": "Need to save results"},
        }

        from backend.src.bt.conditions.signals import signal_is_continuation
        result = signal_is_continuation(mock_ctx)

        assert result == RunStatus.SUCCESS

    def test_delegation_recommended_is_continuation(self, mock_ctx):
        """delegation_recommended should be continuation."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "delegation_recommended",
            "confidence": 0.9,
            "fields": {"reason": "Complex task", "scope": "research"},
        }

        from backend.src.bt.conditions.signals import signal_is_continuation
        result = signal_is_continuation(mock_ctx)

        assert result == RunStatus.SUCCESS

    def test_context_sufficient_is_not_continuation(self, mock_ctx, context_sufficient_signal):
        """context_sufficient should NOT be continuation."""
        mock_ctx.blackboard._data["last_signal"] = context_sufficient_signal

        from backend.src.bt.conditions.signals import signal_is_continuation
        result = signal_is_continuation(mock_ctx)

        assert result == RunStatus.FAILURE


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_signal_object_instead_of_dict(self, mock_ctx):
        """Should handle Signal model object (not just dict)."""
        # Create mock Signal object
        signal_obj = MagicMock()
        signal_obj.type = "need_turn"
        signal_obj.confidence = 0.85
        signal_obj.fields = {"reason": "test"}

        mock_ctx.blackboard._data["last_signal"] = signal_obj

        from backend.src.bt.conditions.signals import has_signal
        result = has_signal(mock_ctx, signal_type="need_turn")

        assert result == RunStatus.SUCCESS

    def test_handles_none_blackboard_gracefully(self):
        """All conditions should handle None blackboard without crashing."""
        ctx = MagicMock()
        ctx.blackboard = None

        from backend.src.bt.conditions.signals import (
            check_signal,
            has_signal,
            signal_type_is,
            signal_confidence_above,
            consecutive_same_reason_gte,
        )

        # All should return FAILURE without raising
        assert check_signal(ctx) == RunStatus.FAILURE
        assert has_signal(ctx) == RunStatus.FAILURE
        assert signal_type_is(ctx, "need_turn") == RunStatus.FAILURE
        assert signal_confidence_above(ctx) == RunStatus.FAILURE
        assert consecutive_same_reason_gte(ctx) == RunStatus.FAILURE

    def test_handles_empty_signal_dict(self, mock_ctx):
        """Should handle empty signal dict gracefully."""
        mock_ctx.blackboard._data["last_signal"] = {}

        from backend.src.bt.conditions.signals import signal_type_is
        result = signal_type_is(mock_ctx, "need_turn")

        assert result == RunStatus.FAILURE
