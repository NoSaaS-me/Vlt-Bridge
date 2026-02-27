"""Unit tests for fallback condition functions.

Tests the BT condition functions for fallback triggering in fallback.py.

Part of feature 020-bt-oracle-agent.
Tasks covered: T048 from tasks-expanded-us5.md

Acceptance Criteria Mapping:
- FR-019: BERT fallback activates when no signal for 3+ turns
- FR-020: BERT fallback activates when signal confidence < 0.3
- FR-021: BERT fallback activates on explicit `stuck` signal
- US5-AC-1: Given agent response with no signal, when 3 turns pass, BERT fallback activates
- US5-AC-2: Given signal with confidence < 0.3, BERT fallback is consulted
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from backend.src.bt.state.base import RunStatus
from backend.src.bt.conditions.fallback import (
    needs_fallback,
    needs_fallback_condition,
    no_signal_for_n_turns,
    signal_confidence_below,
    is_stuck_signal,
    TURNS_WITHOUT_SIGNAL_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)


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
    """Sample need_turn signal dict with high confidence."""
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
def low_confidence_signal():
    """Sample signal with low confidence."""
    return {
        "type": "need_turn",
        "confidence": 0.25,  # Below 0.3 threshold
        "fields": {"reason": "Not sure what to do"},
        "raw_xml": '<signal type="need_turn">...</signal>',
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def stuck_signal():
    """Sample stuck signal dict."""
    return {
        "type": "stuck",
        "confidence": 0.9,
        "fields": {
            "attempted": ["search_code", "search_vault"],
            "blocker": "No results found for the query",
        },
        "raw_xml": '<signal type="stuck">...</signal>',
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# TestConstants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_turns_threshold(self):
        """Turns without signal threshold should be 3."""
        assert TURNS_WITHOUT_SIGNAL_THRESHOLD == 3

    def test_confidence_threshold(self):
        """Low confidence threshold should be 0.3."""
        assert LOW_CONFIDENCE_THRESHOLD == 0.3


# =============================================================================
# TestNeedsFallback (Composite Condition)
# =============================================================================


class TestNeedsFallback:
    """Tests for the composite needs_fallback condition."""

    def test_triggers_on_turns_without_signal(self, mock_ctx):
        """Should trigger when turns_without_signal >= 3 (FR-019, US5-AC-1)."""
        mock_ctx.blackboard._data["turns_without_signal"] = 3
        mock_ctx.blackboard._data["last_signal"] = None

        assert needs_fallback(mock_ctx) is True

    def test_triggers_on_many_turns_without_signal(self, mock_ctx):
        """Should trigger when turns_without_signal > 3."""
        mock_ctx.blackboard._data["turns_without_signal"] = 5
        mock_ctx.blackboard._data["last_signal"] = None

        assert needs_fallback(mock_ctx) is True

    def test_triggers_on_low_confidence(self, mock_ctx, low_confidence_signal):
        """Should trigger when signal confidence < 0.3 (FR-020, US5-AC-2)."""
        mock_ctx.blackboard._data["turns_without_signal"] = 0
        mock_ctx.blackboard._data["last_signal"] = low_confidence_signal

        assert needs_fallback(mock_ctx) is True

    def test_triggers_on_stuck_signal(self, mock_ctx, stuck_signal):
        """Should trigger when last signal is 'stuck' (FR-021)."""
        mock_ctx.blackboard._data["turns_without_signal"] = 0
        mock_ctx.blackboard._data["last_signal"] = stuck_signal

        assert needs_fallback(mock_ctx) is True

    def test_no_trigger_on_normal_state(self, mock_ctx, need_turn_signal):
        """Should not trigger on normal state."""
        mock_ctx.blackboard._data["turns_without_signal"] = 1
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal  # confidence 0.85

        assert needs_fallback(mock_ctx) is False

    def test_no_trigger_below_turn_threshold(self, mock_ctx):
        """Should not trigger when turns_without_signal < 3."""
        mock_ctx.blackboard._data["turns_without_signal"] = 2
        mock_ctx.blackboard._data["last_signal"] = None

        assert needs_fallback(mock_ctx) is False

    def test_no_trigger_above_confidence_threshold(self, mock_ctx):
        """Should not trigger when signal confidence >= 0.3."""
        mock_ctx.blackboard._data["turns_without_signal"] = 0
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "confidence": 0.5,  # Above threshold
            "fields": {},
        }

        assert needs_fallback(mock_ctx) is False

    def test_no_trigger_without_blackboard(self):
        """Should return False if no blackboard."""
        ctx = MagicMock()
        ctx.blackboard = None

        assert needs_fallback(ctx) is False

    def test_handles_missing_signal(self, mock_ctx):
        """Should handle missing last_signal gracefully."""
        mock_ctx.blackboard._data["turns_without_signal"] = 1
        # No last_signal key

        assert needs_fallback(mock_ctx) is False


# =============================================================================
# TestNeedsFallbackCondition (RunStatus Wrapper)
# =============================================================================


class TestNeedsFallbackCondition:
    """Tests for the BT RunStatus wrapper."""

    def test_returns_success_when_fallback_needed(self, mock_ctx):
        """Should return SUCCESS when fallback should trigger."""
        mock_ctx.blackboard._data["turns_without_signal"] = 3
        mock_ctx.blackboard._data["last_signal"] = None

        result = needs_fallback_condition(mock_ctx)

        assert result == RunStatus.SUCCESS

    def test_returns_failure_when_no_fallback_needed(self, mock_ctx, need_turn_signal):
        """Should return FAILURE when no fallback needed."""
        mock_ctx.blackboard._data["turns_without_signal"] = 1
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal

        result = needs_fallback_condition(mock_ctx)

        assert result == RunStatus.FAILURE


# =============================================================================
# TestNoSignalForNTurns
# =============================================================================


class TestNoSignalForNTurns:
    """Tests for the no_signal_for_n_turns condition."""

    def test_true_when_at_threshold(self, mock_ctx):
        """Returns True when exactly at threshold."""
        mock_ctx.blackboard._data["turns_without_signal"] = 3

        assert no_signal_for_n_turns(mock_ctx, n=3) is True

    def test_true_when_above_threshold(self, mock_ctx):
        """Returns True when above threshold."""
        mock_ctx.blackboard._data["turns_without_signal"] = 5

        assert no_signal_for_n_turns(mock_ctx, n=3) is True

    def test_false_when_below_threshold(self, mock_ctx):
        """Returns False when below threshold."""
        mock_ctx.blackboard._data["turns_without_signal"] = 2

        assert no_signal_for_n_turns(mock_ctx, n=3) is False

    def test_false_when_zero(self, mock_ctx):
        """Returns False when turns is 0."""
        mock_ctx.blackboard._data["turns_without_signal"] = 0

        assert no_signal_for_n_turns(mock_ctx, n=3) is False

    def test_uses_default_threshold(self, mock_ctx):
        """Uses default threshold of 3."""
        mock_ctx.blackboard._data["turns_without_signal"] = 3

        assert no_signal_for_n_turns(mock_ctx) is True

    def test_custom_threshold(self, mock_ctx):
        """Supports custom threshold."""
        mock_ctx.blackboard._data["turns_without_signal"] = 5

        assert no_signal_for_n_turns(mock_ctx, n=5) is True
        assert no_signal_for_n_turns(mock_ctx, n=6) is False

    def test_false_without_blackboard(self):
        """Returns False if no blackboard."""
        ctx = MagicMock()
        ctx.blackboard = None

        assert no_signal_for_n_turns(ctx) is False

    def test_handles_missing_key(self, mock_ctx):
        """Returns False when key is not set (defaults to 0)."""
        # No turns_without_signal key

        assert no_signal_for_n_turns(mock_ctx) is False


# =============================================================================
# TestSignalConfidenceBelow
# =============================================================================


class TestSignalConfidenceBelow:
    """Tests for the signal_confidence_below condition."""

    def test_true_when_below(self, mock_ctx, low_confidence_signal):
        """Returns True when confidence below threshold."""
        mock_ctx.blackboard._data["last_signal"] = low_confidence_signal  # 0.25

        assert signal_confidence_below(mock_ctx, threshold=0.3) is True

    def test_false_when_above(self, mock_ctx, need_turn_signal):
        """Returns False when confidence above threshold."""
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal  # 0.85

        assert signal_confidence_below(mock_ctx, threshold=0.3) is False

    def test_false_when_at_threshold(self, mock_ctx):
        """Returns False when confidence equals threshold."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "confidence": 0.3,  # Exactly at threshold
            "fields": {},
        }

        assert signal_confidence_below(mock_ctx, threshold=0.3) is False

    def test_false_when_no_signal(self, mock_ctx):
        """Returns False when no last signal."""
        mock_ctx.blackboard._data["last_signal"] = None

        assert signal_confidence_below(mock_ctx) is False

    def test_false_when_signal_missing_confidence(self, mock_ctx):
        """Returns False when signal has no confidence field."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "fields": {},
            # No confidence
        }

        assert signal_confidence_below(mock_ctx) is False

    def test_uses_default_threshold(self, mock_ctx):
        """Uses default threshold of 0.3."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "confidence": 0.25,
            "fields": {},
        }

        assert signal_confidence_below(mock_ctx) is True

    def test_custom_threshold(self, mock_ctx):
        """Supports custom threshold."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "confidence": 0.5,
            "fields": {},
        }

        assert signal_confidence_below(mock_ctx, threshold=0.6) is True
        assert signal_confidence_below(mock_ctx, threshold=0.4) is False

    def test_false_without_blackboard(self):
        """Returns False if no blackboard."""
        ctx = MagicMock()
        ctx.blackboard = None

        assert signal_confidence_below(ctx) is False


# =============================================================================
# TestIsStuckSignal
# =============================================================================


class TestIsStuckSignal:
    """Tests for the is_stuck_signal condition."""

    def test_true_when_stuck(self, mock_ctx, stuck_signal):
        """Returns True when last signal is stuck."""
        mock_ctx.blackboard._data["last_signal"] = stuck_signal

        assert is_stuck_signal(mock_ctx) is True

    def test_false_when_other_type(self, mock_ctx, need_turn_signal):
        """Returns False for other signal types."""
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal

        assert is_stuck_signal(mock_ctx) is False

    def test_false_when_no_signal(self, mock_ctx):
        """Returns False when no last signal."""
        mock_ctx.blackboard._data["last_signal"] = None

        assert is_stuck_signal(mock_ctx) is False

    def test_handles_enum_type_value(self, mock_ctx):
        """Handles signal type as enum value string."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "SignalType.STUCK",  # Enum string representation
            "confidence": 0.8,
            "fields": {},
        }

        assert is_stuck_signal(mock_ctx) is True

    def test_case_insensitive(self, mock_ctx):
        """Matches stuck type case-insensitively."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "STUCK",
            "confidence": 0.8,
            "fields": {},
        }

        assert is_stuck_signal(mock_ctx) is True

    def test_false_when_signal_missing_type(self, mock_ctx):
        """Returns False when signal has no type field."""
        mock_ctx.blackboard._data["last_signal"] = {
            "confidence": 0.8,
            "fields": {},
            # No type
        }

        assert is_stuck_signal(mock_ctx) is False

    def test_false_without_blackboard(self):
        """Returns False if no blackboard."""
        ctx = MagicMock()
        ctx.blackboard = None

        assert is_stuck_signal(ctx) is False


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_signal_object_instead_of_dict(self, mock_ctx):
        """Should handle Signal model object (not just dict)."""
        signal_obj = MagicMock()
        signal_obj.type = "stuck"
        signal_obj.confidence = 0.9
        signal_obj.fields = {"blocker": "test"}

        mock_ctx.blackboard._data["last_signal"] = signal_obj
        mock_ctx.blackboard._data["turns_without_signal"] = 0

        assert is_stuck_signal(mock_ctx) is True
        assert needs_fallback(mock_ctx) is True

    def test_handles_none_blackboard_gracefully_all_conditions(self):
        """All conditions should handle None blackboard without crashing."""
        ctx = MagicMock()
        ctx.blackboard = None

        # All should return False without raising
        assert needs_fallback(ctx) is False
        assert no_signal_for_n_turns(ctx) is False
        assert signal_confidence_below(ctx) is False
        assert is_stuck_signal(ctx) is False

    def test_handles_empty_signal_dict(self, mock_ctx):
        """Should handle empty signal dict gracefully."""
        mock_ctx.blackboard._data["last_signal"] = {}
        mock_ctx.blackboard._data["turns_without_signal"] = 0

        # Should not crash, should return False for type-based checks
        assert is_stuck_signal(mock_ctx) is False
        assert signal_confidence_below(mock_ctx) is False
        assert needs_fallback(mock_ctx) is False

    def test_priority_order_turns_first(self, mock_ctx, need_turn_signal):
        """Turns without signal check should trigger before signal checks."""
        mock_ctx.blackboard._data["turns_without_signal"] = 3
        mock_ctx.blackboard._data["last_signal"] = need_turn_signal  # High confidence

        # Should trigger on turns, not signal content
        assert needs_fallback(mock_ctx) is True

    def test_confidence_check_with_object_attribute(self, mock_ctx):
        """Should extract confidence from object with attribute."""
        signal_obj = MagicMock()
        signal_obj.type = "need_turn"
        signal_obj.confidence = 0.2  # Low confidence
        signal_obj.fields = {}

        mock_ctx.blackboard._data["last_signal"] = signal_obj
        mock_ctx.blackboard._data["turns_without_signal"] = 0

        assert signal_confidence_below(mock_ctx) is True
        assert needs_fallback(mock_ctx) is True

    def test_invalid_confidence_type_handled(self, mock_ctx):
        """Should handle non-numeric confidence gracefully."""
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "confidence": "invalid",  # Not a number
            "fields": {},
        }
        mock_ctx.blackboard._data["turns_without_signal"] = 0

        # Should return False for confidence check
        assert signal_confidence_below(mock_ctx) is False
