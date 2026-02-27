"""Unit tests for budget conditions.

Tests for budget checking conditions in backend/src/bt/conditions/budget.py.

Part of feature 020-bt-oracle-agent.
Tasks covered: T032 from tasks-expanded-us3.md

Acceptance Criteria Mapping:
- US3-AC1: Agent at 29/30 turns gets one more -> test_returns_success_at_last_turn
- US3-AC2: Agent at 30/30 turns forced completion -> test_returns_success_when_at_max
- FR-007: Configurable max turn limits -> test_respects_custom_max_turns
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
    """Mock oracle config with max_turns=30."""
    config = MagicMock()
    config.max_turns = 30
    config.iteration_warning_threshold = 0.70
    config.token_warning_threshold = 0.80
    config.context_warning_threshold = 0.70
    config.loop_threshold = 3
    return config


# =============================================================================
# TestTurnsRemaining
# =============================================================================


class TestTurnsRemaining:
    """Tests for turns_remaining condition."""

    def test_returns_success_when_turns_available(self, mock_config):
        """Should return SUCCESS when turn < max_turns."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import turns_remaining

            ctx = MockContext({"turn": 10})
            result = turns_remaining(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_failure_when_no_turns_available(self, mock_config):
        """Should return FAILURE when turn >= max_turns."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import turns_remaining

            ctx = MockContext({"turn": 30})
            result = turns_remaining(ctx)
            assert result == RunStatus.FAILURE

    def test_returns_failure_when_over_max_turns(self, mock_config):
        """Should return FAILURE when turn > max_turns."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import turns_remaining

            ctx = MockContext({"turn": 35})
            result = turns_remaining(ctx)
            assert result == RunStatus.FAILURE

    def test_defaults_turn_to_zero(self, mock_config):
        """Should default turn to 0 if not set."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import turns_remaining

            ctx = MockContext({})
            result = turns_remaining(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_failure_when_no_blackboard(self, mock_config):
        """Should return FAILURE when blackboard is None."""
        from backend.src.bt.conditions.budget import turns_remaining

        ctx = MockContext()
        ctx.blackboard = None
        result = turns_remaining(ctx)
        assert result == RunStatus.FAILURE

    def test_handles_string_turn_value(self, mock_config):
        """Should handle turn value as string."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import turns_remaining

            ctx = MockContext({"turn": "10"})
            result = turns_remaining(ctx)
            assert result == RunStatus.SUCCESS

    def test_handles_invalid_turn_value(self, mock_config):
        """Should default to 0 for invalid turn value."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import turns_remaining

            ctx = MockContext({"turn": "invalid"})
            result = turns_remaining(ctx)
            assert result == RunStatus.SUCCESS  # Defaults to 0, so success


# =============================================================================
# TestIsAtBudgetLimit
# =============================================================================


class TestIsAtBudgetLimit:
    """Tests for is_at_budget_limit condition."""

    def test_returns_success_at_last_turn(self, mock_config):
        """Should return SUCCESS when turn == max_turns - 1 (US3-AC1)."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import is_at_budget_limit

            ctx = MockContext({"turn": 29})
            result = is_at_budget_limit(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_failure_before_last_turn(self, mock_config):
        """Should return FAILURE when turn < max_turns - 1."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import is_at_budget_limit

            ctx = MockContext({"turn": 28})
            result = is_at_budget_limit(ctx)
            assert result == RunStatus.FAILURE

    def test_returns_failure_after_last_turn(self, mock_config):
        """Should return FAILURE when turn >= max_turns."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import is_at_budget_limit

            ctx = MockContext({"turn": 30})
            result = is_at_budget_limit(ctx)
            assert result == RunStatus.FAILURE

    def test_returns_failure_when_no_blackboard(self, mock_config):
        """Should return FAILURE when blackboard is None."""
        from backend.src.bt.conditions.budget import is_at_budget_limit

        ctx = MockContext()
        ctx.blackboard = None
        result = is_at_budget_limit(ctx)
        assert result == RunStatus.FAILURE


# =============================================================================
# TestIsOverBudget
# =============================================================================


class TestIsOverBudget:
    """Tests for is_over_budget condition."""

    def test_returns_success_when_at_max(self, mock_config):
        """Should return SUCCESS when turn == max_turns (US3-AC2)."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import is_over_budget

            ctx = MockContext({"turn": 30})
            result = is_over_budget(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_success_when_over_max(self, mock_config):
        """Should return SUCCESS when turn > max_turns."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import is_over_budget

            ctx = MockContext({"turn": 35})
            result = is_over_budget(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_failure_when_under_max(self, mock_config):
        """Should return FAILURE when turn < max_turns."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import is_over_budget

            ctx = MockContext({"turn": 29})
            result = is_over_budget(ctx)
            assert result == RunStatus.FAILURE

    def test_returns_failure_when_no_blackboard(self, mock_config):
        """Should return FAILURE when blackboard is None."""
        from backend.src.bt.conditions.budget import is_over_budget

        ctx = MockContext()
        ctx.blackboard = None
        result = is_over_budget(ctx)
        assert result == RunStatus.FAILURE


# =============================================================================
# TestBudgetWarningNeeded
# =============================================================================


class TestBudgetWarningNeeded:
    """Tests for budget_warning_needed condition."""

    def test_returns_success_at_warning_threshold(self, mock_config):
        """Should return SUCCESS at 70% of max_turns."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import budget_warning_needed

            # 70% of 30 = 21
            ctx = MockContext({"turn": 21, "iteration_warning_emitted": False})
            result = budget_warning_needed(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_failure_before_threshold(self, mock_config):
        """Should return FAILURE before 70% of max_turns."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import budget_warning_needed

            ctx = MockContext({"turn": 10, "iteration_warning_emitted": False})
            result = budget_warning_needed(ctx)
            assert result == RunStatus.FAILURE

    def test_returns_failure_if_already_warned(self, mock_config):
        """Should return FAILURE if warning already emitted."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import budget_warning_needed

            ctx = MockContext({"turn": 25, "iteration_warning_emitted": True})
            result = budget_warning_needed(ctx)
            assert result == RunStatus.FAILURE

    def test_returns_failure_when_no_blackboard(self, mock_config):
        """Should return FAILURE when blackboard is None."""
        from backend.src.bt.conditions.budget import budget_warning_needed

        ctx = MockContext()
        ctx.blackboard = None
        result = budget_warning_needed(ctx)
        assert result == RunStatus.FAILURE


# =============================================================================
# TestBudgetWithCustomMaxTurns
# =============================================================================


class TestBudgetWithCustomMaxTurns:
    """Tests with custom max_turns configuration (FR-007)."""

    def test_respects_custom_max_turns_5(self):
        """Should use max_turns from config (max=5)."""
        config = MagicMock()
        config.max_turns = 5
        config.iteration_warning_threshold = 0.70

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ):
            from backend.src.bt.conditions.budget import (
                is_at_budget_limit,
                is_over_budget,
                turns_remaining,
            )

            # Turn 4 is last turn when max=5
            ctx = MockContext({"turn": 4})
            assert is_at_budget_limit(ctx) == RunStatus.SUCCESS
            assert is_over_budget(ctx) == RunStatus.FAILURE
            assert turns_remaining(ctx) == RunStatus.SUCCESS

            # Turn 5 is over budget when max=5
            ctx = MockContext({"turn": 5})
            assert is_at_budget_limit(ctx) == RunStatus.FAILURE
            assert is_over_budget(ctx) == RunStatus.SUCCESS
            assert turns_remaining(ctx) == RunStatus.FAILURE

    def test_respects_custom_max_turns_10(self):
        """Should use max_turns from config (max=10)."""
        config = MagicMock()
        config.max_turns = 10
        config.iteration_warning_threshold = 0.70

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ):
            from backend.src.bt.conditions.budget import (
                is_at_budget_limit,
                is_over_budget,
            )

            # Turn 9 is last turn when max=10
            ctx = MockContext({"turn": 9})
            assert is_at_budget_limit(ctx) == RunStatus.SUCCESS

            # Turn 10 is over budget when max=10
            ctx = MockContext({"turn": 10})
            assert is_over_budget(ctx) == RunStatus.SUCCESS


# =============================================================================
# TestGetTurnsRemainingCount
# =============================================================================


class TestGetTurnsRemainingCount:
    """Tests for get_turns_remaining_count helper."""

    def test_returns_correct_count(self, mock_config):
        """Should return correct remaining turn count."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import get_turns_remaining_count

            ctx = MockContext({"turn": 10})
            remaining = get_turns_remaining_count(ctx)
            assert remaining == 20  # 30 - 10

    def test_returns_zero_when_over_budget(self, mock_config):
        """Should return 0 when over budget."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import get_turns_remaining_count

            ctx = MockContext({"turn": 35})
            remaining = get_turns_remaining_count(ctx)
            assert remaining == 0

    def test_returns_zero_when_no_blackboard(self, mock_config):
        """Should return 0 when no blackboard."""
        from backend.src.bt.conditions.budget import get_turns_remaining_count

        ctx = MockContext()
        ctx.blackboard = None
        remaining = get_turns_remaining_count(ctx)
        assert remaining == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for budget conditions."""

    def test_max_turns_one(self):
        """Should handle max_turns=1 correctly."""
        config = MagicMock()
        config.max_turns = 1
        config.iteration_warning_threshold = 0.70

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ):
            from backend.src.bt.conditions.budget import (
                is_at_budget_limit,
                is_over_budget,
                turns_remaining,
            )

            # Turn 0 is last turn when max=1
            ctx = MockContext({"turn": 0})
            assert is_at_budget_limit(ctx) == RunStatus.SUCCESS
            assert is_over_budget(ctx) == RunStatus.FAILURE
            assert turns_remaining(ctx) == RunStatus.SUCCESS

            # Turn 1 is over budget when max=1
            ctx = MockContext({"turn": 1})
            assert is_over_budget(ctx) == RunStatus.SUCCESS
            assert turns_remaining(ctx) == RunStatus.FAILURE

    def test_handles_float_turn_value(self, mock_config):
        """Should handle turn as float (converts to int)."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import turns_remaining

            ctx = MockContext({"turn": 10.5})
            result = turns_remaining(ctx)
            assert result == RunStatus.SUCCESS

    def test_negative_turn_treated_as_zero(self, mock_config):
        """Should treat negative turn as valid (weird but allowed)."""
        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_config
        ):
            from backend.src.bt.conditions.budget import turns_remaining

            ctx = MockContext({"turn": -5})
            result = turns_remaining(ctx)
            # -5 < 30, so SUCCESS (35 turns remaining!)
            assert result == RunStatus.SUCCESS
