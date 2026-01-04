"""Tests for settings models, AgentConfig validation bounds, and ReasoningEffort."""

import pytest
from pydantic import ValidationError

from backend.src.models.settings import (
    AgentConfig,
    AgentConfigUpdate,
    ModelSettings,
    ModelSettingsUpdateRequest,
    ReasoningEffort,
)


class TestAgentConfigDefaults:
    """Test that AgentConfig has correct default values."""

    def test_defaults(self):
        """Verify all default values match specification."""
        config = AgentConfig()

        assert config.max_iterations == 15
        assert config.soft_warning_percent == 70
        assert config.token_budget == 50000
        assert config.token_warning_percent == 80
        assert config.timeout_seconds == 120
        assert config.max_tool_calls_per_turn == 100
        assert config.max_parallel_tools == 3
        assert config.loop_detection_window_seconds == 300


class TestAgentConfigValidBounds:
    """Test that AgentConfig accepts valid values at boundaries."""

    def test_valid_at_min_bounds(self):
        """All fields accept their minimum valid values."""
        config = AgentConfig(
            max_iterations=1,
            soft_warning_percent=50,
            token_budget=1000,
            token_warning_percent=50,
            timeout_seconds=10,
            max_tool_calls_per_turn=1,
            max_parallel_tools=1,
            loop_detection_window_seconds=60,
        )

        assert config.max_iterations == 1
        assert config.soft_warning_percent == 50
        assert config.token_budget == 1000
        assert config.token_warning_percent == 50
        assert config.timeout_seconds == 10
        assert config.max_tool_calls_per_turn == 1
        assert config.max_parallel_tools == 1
        assert config.loop_detection_window_seconds == 60

    def test_valid_at_max_bounds(self):
        """All fields accept their maximum valid values."""
        config = AgentConfig(
            max_iterations=50,
            soft_warning_percent=90,
            token_budget=200000,
            token_warning_percent=95,
            timeout_seconds=600,
            max_tool_calls_per_turn=200,
            max_parallel_tools=10,
            loop_detection_window_seconds=600,
        )

        assert config.max_iterations == 50
        assert config.soft_warning_percent == 90
        assert config.token_budget == 200000
        assert config.token_warning_percent == 95
        assert config.timeout_seconds == 600
        assert config.max_tool_calls_per_turn == 200
        assert config.max_parallel_tools == 10
        assert config.loop_detection_window_seconds == 600

    def test_valid_mid_range_values(self):
        """Fields accept values within their valid range."""
        config = AgentConfig(
            max_iterations=25,
            soft_warning_percent=75,
            token_budget=100000,
            token_warning_percent=85,
            timeout_seconds=300,
            max_tool_calls_per_turn=10,
            max_parallel_tools=5,
        )

        assert config.max_iterations == 25
        assert config.soft_warning_percent == 75
        assert config.token_budget == 100000
        assert config.token_warning_percent == 85
        assert config.timeout_seconds == 300
        assert config.max_tool_calls_per_turn == 10
        assert config.max_parallel_tools == 5


class TestAgentConfigInvalidBelowMin:
    """Test that AgentConfig rejects values below minimum."""

    def test_max_iterations_below_min(self):
        """max_iterations must be >= 1."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(max_iterations=0)
        assert "max_iterations" in str(excinfo.value)

    def test_soft_warning_percent_below_min(self):
        """soft_warning_percent must be >= 50."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(soft_warning_percent=49)
        assert "soft_warning_percent" in str(excinfo.value)

    def test_token_budget_below_min(self):
        """token_budget must be >= 1000."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(token_budget=999)
        assert "token_budget" in str(excinfo.value)

    def test_token_warning_percent_below_min(self):
        """token_warning_percent must be >= 50."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(token_warning_percent=49)
        assert "token_warning_percent" in str(excinfo.value)

    def test_timeout_seconds_below_min(self):
        """timeout_seconds must be >= 10."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(timeout_seconds=9)
        assert "timeout_seconds" in str(excinfo.value)

    def test_max_tool_calls_per_turn_below_min(self):
        """max_tool_calls_per_turn must be >= 1."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(max_tool_calls_per_turn=0)
        assert "max_tool_calls_per_turn" in str(excinfo.value)

    def test_max_parallel_tools_below_min(self):
        """max_parallel_tools must be >= 1."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(max_parallel_tools=0)
        assert "max_parallel_tools" in str(excinfo.value)

    def test_loop_detection_window_seconds_below_min(self):
        """loop_detection_window_seconds must be >= 60."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(loop_detection_window_seconds=59)
        assert "loop_detection_window_seconds" in str(excinfo.value)


class TestAgentConfigInvalidAboveMax:
    """Test that AgentConfig rejects values above maximum."""

    def test_max_iterations_above_max(self):
        """max_iterations must be <= 50."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(max_iterations=51)
        assert "max_iterations" in str(excinfo.value)

    def test_soft_warning_percent_above_max(self):
        """soft_warning_percent must be <= 90."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(soft_warning_percent=91)
        assert "soft_warning_percent" in str(excinfo.value)

    def test_token_budget_above_max(self):
        """token_budget must be <= 200000."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(token_budget=200001)
        assert "token_budget" in str(excinfo.value)

    def test_token_warning_percent_above_max(self):
        """token_warning_percent must be <= 95."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(token_warning_percent=96)
        assert "token_warning_percent" in str(excinfo.value)

    def test_timeout_seconds_above_max(self):
        """timeout_seconds must be <= 600."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(timeout_seconds=601)
        assert "timeout_seconds" in str(excinfo.value)

    def test_max_tool_calls_per_turn_above_max(self):
        """max_tool_calls_per_turn must be <= 200."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(max_tool_calls_per_turn=201)
        assert "max_tool_calls_per_turn" in str(excinfo.value)

    def test_loop_detection_window_seconds_above_max(self):
        """loop_detection_window_seconds must be <= 600."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(loop_detection_window_seconds=601)
        assert "loop_detection_window_seconds" in str(excinfo.value)

    def test_max_parallel_tools_above_max(self):
        """max_parallel_tools must be <= 10."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(max_parallel_tools=11)
        assert "max_parallel_tools" in str(excinfo.value)


class TestAgentConfigUpdatePartial:
    """Test that AgentConfigUpdate accepts partial updates."""

    def test_all_fields_optional(self):
        """AgentConfigUpdate can be created with no fields."""
        update = AgentConfigUpdate()

        assert update.max_iterations is None
        assert update.soft_warning_percent is None
        assert update.token_budget is None
        assert update.token_warning_percent is None
        assert update.timeout_seconds is None
        assert update.max_tool_calls_per_turn is None
        assert update.max_parallel_tools is None

    def test_single_field_update(self):
        """AgentConfigUpdate accepts single field updates."""
        update = AgentConfigUpdate(max_iterations=10)

        assert update.max_iterations == 10
        assert update.soft_warning_percent is None
        assert update.token_budget is None
        assert update.token_warning_percent is None
        assert update.timeout_seconds is None
        assert update.max_tool_calls_per_turn is None
        assert update.max_parallel_tools is None

    def test_multiple_fields_update(self):
        """AgentConfigUpdate accepts multiple field updates."""
        update = AgentConfigUpdate(
            max_iterations=20,
            token_budget=75000,
            timeout_seconds=180,
        )

        assert update.max_iterations == 20
        assert update.soft_warning_percent is None
        assert update.token_budget == 75000
        assert update.token_warning_percent is None
        assert update.timeout_seconds == 180
        assert update.max_tool_calls_per_turn is None
        assert update.max_parallel_tools is None

    def test_all_fields_update(self):
        """AgentConfigUpdate accepts all field updates."""
        update = AgentConfigUpdate(
            max_iterations=25,
            soft_warning_percent=75,
            token_budget=100000,
            token_warning_percent=85,
            timeout_seconds=300,
            max_tool_calls_per_turn=10,
            max_parallel_tools=5,
        )

        assert update.max_iterations == 25
        assert update.soft_warning_percent == 75
        assert update.token_budget == 100000
        assert update.token_warning_percent == 85
        assert update.timeout_seconds == 300
        assert update.max_tool_calls_per_turn == 10
        assert update.max_parallel_tools == 5


class TestAgentConfigUpdateValidation:
    """Test that AgentConfigUpdate validates bounds when values are provided."""

    def test_update_rejects_invalid_max_iterations(self):
        """AgentConfigUpdate rejects invalid max_iterations."""
        with pytest.raises(ValidationError):
            AgentConfigUpdate(max_iterations=0)
        with pytest.raises(ValidationError):
            AgentConfigUpdate(max_iterations=51)

    def test_update_rejects_invalid_soft_warning_percent(self):
        """AgentConfigUpdate rejects invalid soft_warning_percent."""
        with pytest.raises(ValidationError):
            AgentConfigUpdate(soft_warning_percent=49)
        with pytest.raises(ValidationError):
            AgentConfigUpdate(soft_warning_percent=91)

    def test_update_rejects_invalid_token_budget(self):
        """AgentConfigUpdate rejects invalid token_budget."""
        with pytest.raises(ValidationError):
            AgentConfigUpdate(token_budget=999)
        with pytest.raises(ValidationError):
            AgentConfigUpdate(token_budget=200001)

    def test_update_rejects_invalid_token_warning_percent(self):
        """AgentConfigUpdate rejects invalid token_warning_percent."""
        with pytest.raises(ValidationError):
            AgentConfigUpdate(token_warning_percent=49)
        with pytest.raises(ValidationError):
            AgentConfigUpdate(token_warning_percent=96)

    def test_update_rejects_invalid_timeout_seconds(self):
        """AgentConfigUpdate rejects invalid timeout_seconds."""
        with pytest.raises(ValidationError):
            AgentConfigUpdate(timeout_seconds=9)
        with pytest.raises(ValidationError):
            AgentConfigUpdate(timeout_seconds=601)

    def test_update_rejects_invalid_max_tool_calls_per_turn(self):
        """AgentConfigUpdate rejects invalid max_tool_calls_per_turn."""
        with pytest.raises(ValidationError):
            AgentConfigUpdate(max_tool_calls_per_turn=0)
        with pytest.raises(ValidationError):
            AgentConfigUpdate(max_tool_calls_per_turn=201)

    def test_update_rejects_invalid_max_parallel_tools(self):
        """AgentConfigUpdate rejects invalid max_parallel_tools."""
        with pytest.raises(ValidationError):
            AgentConfigUpdate(max_parallel_tools=0)
        with pytest.raises(ValidationError):
            AgentConfigUpdate(max_parallel_tools=11)


class TestAgentConfigUpdateBoundsAccepted:
    """Test that AgentConfigUpdate accepts values at boundaries."""

    def test_update_accepts_min_bounds(self):
        """AgentConfigUpdate accepts minimum valid values."""
        update = AgentConfigUpdate(
            max_iterations=1,
            soft_warning_percent=50,
            token_budget=1000,
            token_warning_percent=50,
            timeout_seconds=10,
            max_tool_calls_per_turn=1,
            max_parallel_tools=1,
        )

        assert update.max_iterations == 1
        assert update.soft_warning_percent == 50
        assert update.token_budget == 1000
        assert update.token_warning_percent == 50
        assert update.timeout_seconds == 10
        assert update.max_tool_calls_per_turn == 1
        assert update.max_parallel_tools == 1

    def test_update_accepts_max_bounds(self):
        """AgentConfigUpdate accepts maximum valid values."""
        update = AgentConfigUpdate(
            max_iterations=50,
            soft_warning_percent=90,
            token_budget=200000,
            token_warning_percent=95,
            timeout_seconds=600,
            max_tool_calls_per_turn=200,
            max_parallel_tools=10,
        )

        assert update.max_iterations == 50
        assert update.soft_warning_percent == 90
        assert update.token_budget == 200000
        assert update.token_warning_percent == 95
        assert update.timeout_seconds == 600
        assert update.max_tool_calls_per_turn == 200
        assert update.max_parallel_tools == 10


# =============================================================================
# ReasoningEffort Tests
# =============================================================================


class TestReasoningEffortEnum:
    """Test the ReasoningEffort enum."""

    def test_enum_values(self):
        """All expected values exist."""
        assert ReasoningEffort.LOW.value == "low"
        assert ReasoningEffort.MEDIUM.value == "medium"
        assert ReasoningEffort.HIGH.value == "high"

    def test_enum_is_string(self):
        """ReasoningEffort values should work as strings."""
        assert ReasoningEffort.LOW == "low"
        assert ReasoningEffort.MEDIUM == "medium"
        assert ReasoningEffort.HIGH == "high"

    def test_enum_from_string(self):
        """Can create ReasoningEffort from string."""
        assert ReasoningEffort("low") == ReasoningEffort.LOW
        assert ReasoningEffort("medium") == ReasoningEffort.MEDIUM
        assert ReasoningEffort("high") == ReasoningEffort.HIGH

    def test_invalid_value_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            ReasoningEffort("invalid")
        with pytest.raises(ValueError):
            ReasoningEffort("MEDIUM")  # Case sensitive


class TestModelSettingsReasoningEffort:
    """Test reasoning_effort in ModelSettings."""

    def test_default_is_medium(self):
        """Default reasoning effort should be medium."""
        settings = ModelSettings()
        assert settings.reasoning_effort == ReasoningEffort.MEDIUM

    def test_can_set_low(self):
        """Can set to low."""
        settings = ModelSettings(reasoning_effort=ReasoningEffort.LOW)
        assert settings.reasoning_effort == ReasoningEffort.LOW

    def test_can_set_high(self):
        """Can set to high."""
        settings = ModelSettings(reasoning_effort=ReasoningEffort.HIGH)
        assert settings.reasoning_effort == ReasoningEffort.HIGH

    def test_can_set_from_string(self):
        """Can set reasoning_effort from string value."""
        settings = ModelSettings(reasoning_effort="low")
        assert settings.reasoning_effort == ReasoningEffort.LOW

        settings = ModelSettings(reasoning_effort="medium")
        assert settings.reasoning_effort == ReasoningEffort.MEDIUM

        settings = ModelSettings(reasoning_effort="high")
        assert settings.reasoning_effort == ReasoningEffort.HIGH

    def test_invalid_value_raises(self):
        """Invalid reasoning_effort raises validation error."""
        with pytest.raises(ValidationError):
            ModelSettings(reasoning_effort="invalid")

    def test_other_defaults_preserved(self):
        """Setting reasoning_effort doesn't affect other defaults."""
        settings = ModelSettings(reasoning_effort=ReasoningEffort.HIGH)
        assert settings.thinking_enabled is False
        assert settings.max_iterations == 15
        assert settings.token_budget == 50000


class TestModelSettingsUpdateRequestReasoningEffort:
    """Test reasoning_effort in update request."""

    def test_optional_none_by_default(self):
        """reasoning_effort should be None by default."""
        update = ModelSettingsUpdateRequest()
        assert update.reasoning_effort is None

    def test_can_update_reasoning_effort(self):
        """Can include reasoning_effort in update."""
        update = ModelSettingsUpdateRequest(reasoning_effort=ReasoningEffort.HIGH)
        assert update.reasoning_effort == ReasoningEffort.HIGH

    def test_can_update_to_low(self):
        """Can update reasoning_effort to low."""
        update = ModelSettingsUpdateRequest(reasoning_effort=ReasoningEffort.LOW)
        assert update.reasoning_effort == ReasoningEffort.LOW

    def test_can_update_to_medium(self):
        """Can update reasoning_effort to medium."""
        update = ModelSettingsUpdateRequest(reasoning_effort=ReasoningEffort.MEDIUM)
        assert update.reasoning_effort == ReasoningEffort.MEDIUM

    def test_can_update_from_string(self):
        """Can update reasoning_effort from string value."""
        update = ModelSettingsUpdateRequest(reasoning_effort="high")
        assert update.reasoning_effort == ReasoningEffort.HIGH

    def test_invalid_value_raises(self):
        """Invalid reasoning_effort in update raises validation error."""
        with pytest.raises(ValidationError):
            ModelSettingsUpdateRequest(reasoning_effort="invalid")

    def test_partial_update_only_reasoning_effort(self):
        """Can update only reasoning_effort, leaving other fields None."""
        update = ModelSettingsUpdateRequest(reasoning_effort=ReasoningEffort.LOW)

        assert update.reasoning_effort == ReasoningEffort.LOW
        assert update.oracle_model is None
        assert update.thinking_enabled is None
        assert update.max_iterations is None

    def test_reasoning_effort_with_other_fields(self):
        """Can update reasoning_effort along with other fields."""
        update = ModelSettingsUpdateRequest(
            reasoning_effort=ReasoningEffort.HIGH,
            thinking_enabled=True,
            max_iterations=25,
        )

        assert update.reasoning_effort == ReasoningEffort.HIGH
        assert update.thinking_enabled is True
        assert update.max_iterations == 25


class TestModelSettingsDefaults:
    """Test ModelSettings default values."""

    def test_all_defaults(self):
        """Verify all default values in ModelSettings."""
        settings = ModelSettings()

        assert settings.oracle_model == "gemini-2.0-flash-exp"
        assert settings.subagent_model == "gemini-2.0-flash-exp"
        assert settings.thinking_enabled is False
        assert settings.reasoning_effort == ReasoningEffort.MEDIUM
        assert settings.chat_center_mode is False
        assert settings.librarian_timeout == 1200
        assert settings.max_context_nodes == 30
        assert settings.openrouter_api_key is None
        assert settings.openrouter_api_key_set is False
        assert settings.max_iterations == 15
        assert settings.soft_warning_percent == 70
        assert settings.token_budget == 50000
        assert settings.token_warning_percent == 80
        assert settings.timeout_seconds == 120
        assert settings.max_tool_calls_per_turn == 100
        assert settings.max_parallel_tools == 3
        assert settings.tool_timeout_seconds == 30
        assert settings.loop_detection_window_seconds == 300


class TestModelSettingsThinkingIntegration:
    """Test integration between thinking_enabled and reasoning_effort."""

    def test_thinking_with_low_effort(self):
        """Can combine thinking enabled with low effort."""
        settings = ModelSettings(
            thinking_enabled=True,
            reasoning_effort=ReasoningEffort.LOW,
        )
        assert settings.thinking_enabled is True
        assert settings.reasoning_effort == ReasoningEffort.LOW

    def test_thinking_with_high_effort(self):
        """Can combine thinking enabled with high effort."""
        settings = ModelSettings(
            thinking_enabled=True,
            reasoning_effort=ReasoningEffort.HIGH,
        )
        assert settings.thinking_enabled is True
        assert settings.reasoning_effort == ReasoningEffort.HIGH

    def test_thinking_disabled_with_reasoning_effort(self):
        """reasoning_effort can be set even when thinking is disabled."""
        settings = ModelSettings(
            thinking_enabled=False,
            reasoning_effort=ReasoningEffort.HIGH,
        )
        assert settings.thinking_enabled is False
        assert settings.reasoning_effort == ReasoningEffort.HIGH
