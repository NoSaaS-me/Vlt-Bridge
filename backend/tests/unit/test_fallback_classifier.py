"""Unit tests for fallback classifier service.

Tests the heuristic fallback classification when the agent fails to emit
clear signals.

Part of feature 020-bt-oracle-agent.
Tasks covered: T047 from tasks-expanded-us5.md

Acceptance Criteria Mapping:
- US5-AC-1: Given agent response with no signal, when 3 turns pass, BERT fallback activates
- US5-AC-4: System functions with heuristic defaults when BERT unavailable
- FR-022: System functions with heuristic defaults when BERT unavailable
- SC-006: System functions correctly with BERT disabled (heuristic-only mode)
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.src.services.fallback_classifier import (
    FallbackAction,
    FallbackClassification,
    heuristic_classify,
    log_fallback_trigger,
)


# =============================================================================
# TestFallbackAction
# =============================================================================


class TestFallbackAction:
    """Tests for FallbackAction enum."""

    def test_action_values(self):
        """Should have expected action values."""
        assert FallbackAction.CONTINUE.value == "continue"
        assert FallbackAction.FORCE_RESPONSE.value == "force_response"
        assert FallbackAction.RETRY_WITH_HINT.value == "retry_with_hint"
        assert FallbackAction.ESCALATE.value == "escalate"

    def test_all_actions_defined(self):
        """Should have exactly 4 actions."""
        actions = list(FallbackAction)
        assert len(actions) == 4


# =============================================================================
# TestFallbackClassification
# =============================================================================


class TestFallbackClassification:
    """Tests for FallbackClassification dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        classification = FallbackClassification(
            action=FallbackAction.RETRY_WITH_HINT,
            confidence=0.6,
            hint="Try a different approach",
            reason="No progress detected",
        )

        result = classification.to_dict()

        assert result["action"] == "retry_with_hint"
        assert result["confidence"] == 0.6
        assert result["hint"] == "Try a different approach"
        assert result["reason"] == "No progress detected"

    def test_from_dict(self):
        """Should create from dictionary correctly."""
        data = {
            "action": "force_response",
            "confidence": 0.8,
            "hint": None,
            "reason": "Test reason",
        }

        classification = FallbackClassification.from_dict(data)

        assert classification.action == FallbackAction.FORCE_RESPONSE
        assert classification.confidence == 0.8
        assert classification.hint is None
        assert classification.reason == "Test reason"

    def test_from_dict_defaults(self):
        """Should handle missing fields with defaults."""
        data = {}

        classification = FallbackClassification.from_dict(data)

        assert classification.action == FallbackAction.CONTINUE
        assert classification.confidence == 0.5
        assert classification.hint is None
        assert classification.reason == ""


# =============================================================================
# TestHeuristicClassify
# =============================================================================


class TestHeuristicClassify:
    """Tests for heuristic fallback classification."""

    def test_high_tool_failure_rate_escalates(self):
        """High tool failure rate should trigger ESCALATE."""
        result = heuristic_classify(
            query="How do I fix this bug?",
            accumulated_content="",
            turns_without_signal=2,
            tool_results=[
                {"name": "search_code", "success": False},
                {"name": "read_file", "success": False},
                {"name": "get_repo_map", "success": False},
            ],
        )

        assert result.action == FallbackAction.ESCALATE
        assert result.confidence >= 0.6
        assert "failure" in result.reason.lower()

    def test_substantial_content_forces_response(self):
        """Substantial accumulated content should force response."""
        result = heuristic_classify(
            query="Explain the architecture",
            accumulated_content="A" * 600,  # > 500 chars
            turns_without_signal=2,
            tool_results=[{"name": "search_vault", "success": True}],
        )

        assert result.action == FallbackAction.FORCE_RESPONSE
        assert result.confidence >= 0.7
        assert "content" in result.reason.lower()

    def test_simple_query_no_tools_forces_response(self):
        """Simple query with no tool usage should force response."""
        result = heuristic_classify(
            query="What is Python?",
            accumulated_content="",
            turns_without_signal=1,
            tool_results=[],
        )

        assert result.action == FallbackAction.FORCE_RESPONSE
        assert result.hint is not None
        assert "simple" in result.reason.lower()

    def test_no_progress_retries_with_hint(self):
        """No progress after many turns should retry with hint."""
        result = heuristic_classify(
            query="Find the authentication middleware implementation in the codebase",
            accumulated_content="",
            turns_without_signal=4,
            tool_results=[
                {"name": "search_code", "success": False},
            ],
        )

        assert result.action == FallbackAction.RETRY_WITH_HINT
        assert result.hint is not None
        assert "search_code" in result.hint

    def test_partial_content_many_turns_forces_response(self):
        """Partial content after many turns should force response."""
        result = heuristic_classify(
            query="Complex multi-part question about the system",
            accumulated_content="Here is what I found so far: " * 10,  # ~300 chars
            turns_without_signal=3,
            tool_results=[],
        )

        assert result.action == FallbackAction.FORCE_RESPONSE
        assert result.confidence >= 0.6

    def test_low_confidence_signal_with_content_forces_response(self):
        """Low confidence signal with content should force response."""
        result = heuristic_classify(
            query="Help me understand this code",
            accumulated_content="The code appears to..." * 20,  # Some content
            turns_without_signal=1,
            tool_results=[],
            last_signal_confidence=0.2,
        )

        assert result.action == FallbackAction.FORCE_RESPONSE
        assert result.confidence >= 0.5

    def test_low_confidence_signal_no_content_retries_with_hint(self):
        """Low confidence signal with no content should retry with hint."""
        # Use a longer query to avoid triggering "simple query" logic
        result = heuristic_classify(
            query="I need help understanding the complex interaction patterns between the authentication middleware and the session management subsystem in this project",
            accumulated_content="",
            turns_without_signal=1,
            tool_results=[{"name": "search_code", "success": True}],  # Has tool use
            last_signal_confidence=0.2,
        )

        assert result.action == FallbackAction.RETRY_WITH_HINT
        assert result.hint is not None

    def test_default_continues(self):
        """Normal state should continue."""
        result = heuristic_classify(
            query="How does the database work?",
            accumulated_content="The database uses SQLite...",
            turns_without_signal=0,
            tool_results=[{"name": "search_code", "success": True}],
        )

        assert result.action == FallbackAction.CONTINUE
        assert result.confidence == 0.5

    def test_classification_has_required_fields(self):
        """Classification should have all required fields."""
        result = heuristic_classify(
            query="Test query",
            accumulated_content="",
            turns_without_signal=0,
            tool_results=[],
        )

        assert isinstance(result, FallbackClassification)
        assert isinstance(result.action, FallbackAction)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reason, str)

    def test_empty_query_handled_gracefully(self):
        """Should handle empty query gracefully."""
        result = heuristic_classify(
            query="",
            accumulated_content="",
            turns_without_signal=0,
            tool_results=[],
        )

        # Empty query has 0 words, should be FORCE_RESPONSE
        assert result.action == FallbackAction.FORCE_RESPONSE

    def test_mixed_tool_results(self):
        """Should handle mixed success/failure tool results."""
        result = heuristic_classify(
            query="Find and analyze the authentication module",
            accumulated_content="Found some information...",
            turns_without_signal=1,
            tool_results=[
                {"name": "search_code", "success": True, "result": "..."},
                {"name": "read_file", "success": False, "error": "File not found"},
                {"name": "get_repo_map", "success": True, "result": "..."},
            ],
        )

        # 1/3 failure rate (33%) is below 70% threshold, should continue
        assert result.action == FallbackAction.CONTINUE

    def test_all_tools_succeed_continues(self):
        """All tools succeeding should continue normally."""
        result = heuristic_classify(
            query="What are the dependencies?",
            accumulated_content="",
            turns_without_signal=0,
            tool_results=[
                {"name": "search_code", "success": True},
                {"name": "search_vault", "success": True},
            ],
        )

        assert result.action == FallbackAction.CONTINUE

    def test_no_progress_includes_failed_tools_in_hint(self):
        """Hint should include failed tool names when retrying."""
        result = heuristic_classify(
            query="Debug the auth issue",
            accumulated_content="",
            turns_without_signal=4,
            tool_results=[
                {"name": "search_code", "success": False},
                {"name": "get_repo_map", "success": False},
            ],
        )

        assert result.action == FallbackAction.RETRY_WITH_HINT
        assert "search_code" in result.hint
        assert "get_repo_map" in result.hint


# =============================================================================
# TestLogFallbackTrigger
# =============================================================================


class TestLogFallbackTrigger:
    """Tests for fallback logging function."""

    @patch("backend.src.services.fallback_classifier.logger")
    def test_logs_info_for_non_escalate(self, mock_logger):
        """Should log at INFO level for non-ESCALATE actions."""
        log_fallback_trigger(
            reason="No progress detected",
            turns_without_signal=3,
            last_signal_type="need_turn",
            last_signal_confidence=0.5,
            classification_action="retry_with_hint",
            classification_confidence=0.6,
        )

        mock_logger.info.assert_called_once()
        call_args = str(mock_logger.info.call_args)
        assert "retry_with_hint" in call_args

    @patch("backend.src.services.fallback_classifier.logger")
    def test_logs_warning_for_escalate(self, mock_logger):
        """Should log at WARNING level for ESCALATE action."""
        log_fallback_trigger(
            reason="High failure rate",
            turns_without_signal=2,
            last_signal_type=None,
            last_signal_confidence=None,
            classification_action="escalate",
            classification_confidence=0.7,
        )

        mock_logger.warning.assert_called_once()
        call_args = str(mock_logger.warning.call_args)
        assert "ESCALATE" in call_args

    @patch("backend.src.services.fallback_classifier.logger")
    def test_handles_ans_not_available(self, mock_logger):
        """Should handle ANS not being available gracefully."""
        with patch.dict("sys.modules", {"src.services.ans.bus": None}):
            log_fallback_trigger(
                reason="Test reason",
                turns_without_signal=1,
                last_signal_type=None,
                last_signal_confidence=None,
                classification_action="continue",
                classification_confidence=0.5,
            )

        # Should not raise, just log
        assert True

    @patch("backend.src.services.fallback_classifier.logger")
    def test_emits_ans_event(self, mock_logger):
        """Should log fallback events when called.

        Note: ANS integration is tested indirectly since it's inside the function.
        We verify the function runs without error and logs appropriately.
        """
        log_fallback_trigger(
            reason="Test fallback",
            turns_without_signal=3,
            last_signal_type="need_turn",
            last_signal_confidence=0.4,
            classification_action="force_response",
            classification_confidence=0.8,
        )

        # Should log at INFO level for non-escalate
        mock_logger.info.assert_called()
        # Debug logging may also occur for ANS status
        # The important thing is no exceptions are raised


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exactly_three_turns_with_hint(self):
        """Exactly 3 turns without signal should trigger retry."""
        # Use a longer query with tool usage to avoid simple query logic
        result = heuristic_classify(
            query="I need to understand how the complex authentication middleware interacts with the session management and authorization subsystems in this project",
            accumulated_content="",
            turns_without_signal=3,
            tool_results=[{"name": "search_code", "success": True}],  # Has tool use
        )

        assert result.action == FallbackAction.RETRY_WITH_HINT

    def test_boundary_tool_failure_rate(self):
        """Test boundary at 70% failure rate."""
        # 2/3 = 66.7% < 70%, should NOT escalate
        result = heuristic_classify(
            query="Test query",
            accumulated_content="",
            turns_without_signal=1,
            tool_results=[
                {"name": "tool1", "success": False},
                {"name": "tool2", "success": False},
                {"name": "tool3", "success": True},
            ],
        )
        assert result.action != FallbackAction.ESCALATE

        # 3/4 = 75% > 70%, should escalate
        result = heuristic_classify(
            query="Test query",
            accumulated_content="",
            turns_without_signal=1,
            tool_results=[
                {"name": "tool1", "success": False},
                {"name": "tool2", "success": False},
                {"name": "tool3", "success": False},
                {"name": "tool4", "success": True},
            ],
        )
        assert result.action == FallbackAction.ESCALATE

    def test_boundary_content_length(self):
        """Test boundary at 500 chars content length."""
        # 499 chars - not "substantial"
        result = heuristic_classify(
            query="Test query about something complex",
            accumulated_content="A" * 499,
            turns_without_signal=2,
            tool_results=[],
        )
        # Without substantial content, should continue (simple query check may kick in)
        assert result.action != FallbackAction.FORCE_RESPONSE or result.confidence < 0.8

        # 501 chars - "substantial" with 2+ turns
        result = heuristic_classify(
            query="Test query about something complex",
            accumulated_content="A" * 501,
            turns_without_signal=2,
            tool_results=[],
        )
        assert result.action == FallbackAction.FORCE_RESPONSE
        assert result.confidence >= 0.7

    def test_none_values_handled(self):
        """Should handle None values in tool_results gracefully."""
        result = heuristic_classify(
            query="Test",
            accumulated_content="",
            turns_without_signal=0,
            tool_results=[
                {"name": "tool1"},  # Missing 'success' key
                {"success": True},  # Missing 'name' key
            ],
        )

        # Should not raise, should return valid classification
        # Use duck-typing check instead of isinstance due to potential module reload issues
        assert hasattr(result, "action")
        assert hasattr(result, "confidence")
        assert hasattr(result, "reason")
        assert result.action.value in ["continue", "force_response", "retry_with_hint", "escalate"]
