"""Unit tests for AgentState dataclass."""

import pytest
import time
from dataclasses import FrozenInstanceError

from backend.src.models.settings import AgentConfig
from backend.src.models.agent_state import AgentState


class TestAgentStateImmutability:
    """Test that AgentState is immutable."""

    def test_frozen_prevents_attribute_change(self):
        """Verify frozen=True prevents attribute modification."""
        config = AgentConfig()
        state = AgentState(user_id="test", project_id="proj", config=config)

        with pytest.raises(FrozenInstanceError):
            state.turn = 5

    def test_frozen_prevents_new_attributes(self):
        """Verify frozen=True prevents adding new attributes."""
        config = AgentConfig()
        state = AgentState(user_id="test", project_id="proj", config=config)

        with pytest.raises(FrozenInstanceError):
            state.new_attr = "value"


class TestAgentStateDerivedProperties:
    """Test derived properties of AgentState."""

    def test_is_terminal_false_when_no_reason(self):
        """Test is_terminal is False when termination_reason is None."""
        config = AgentConfig()
        state = AgentState(user_id="test", project_id="proj", config=config)
        assert state.is_terminal is False

    def test_is_terminal_true_when_reason_set(self):
        """Test is_terminal is True when termination_reason is set."""
        config = AgentConfig()
        state = AgentState(
            user_id="test",
            project_id="proj",
            config=config,
            termination_reason="max_iterations"
        )
        assert state.is_terminal is True

    def test_elapsed_seconds(self):
        """Test elapsed_seconds calculation."""
        config = AgentConfig()
        start = time.time() - 5.0  # 5 seconds ago
        state = AgentState(
            user_id="test",
            project_id="proj",
            config=config,
            start_time=start
        )
        # Should be approximately 5 seconds
        assert 4.9 <= state.elapsed_seconds <= 5.2

    def test_iteration_percent(self):
        """Test iteration_percent calculation."""
        config = AgentConfig(max_iterations=10)
        state = AgentState(
            user_id="test",
            project_id="proj",
            config=config,
            turn=7
        )
        assert state.iteration_percent == 70.0

    def test_token_percent(self):
        """Test token_percent calculation."""
        config = AgentConfig(token_budget=50000)
        state = AgentState(
            user_id="test",
            project_id="proj",
            config=config,
            tokens_used=40000
        )
        assert state.token_percent == 80.0


class TestAgentStateDefaults:
    """Test default values of AgentState."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = AgentConfig()
        state = AgentState(user_id="test", project_id="proj", config=config)

        assert state.turn == 0
        assert state.tokens_used == 0
        assert state.recent_actions == ()
        assert state.termination_reason is None
        assert state.extensions == {}
        assert isinstance(state.start_time, float)
