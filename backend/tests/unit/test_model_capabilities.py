"""Unit tests for the model capability registry.

Tests cover:
- Model family identification via prefix matching
- Unknown model fallback to default capabilities
- Suffix stripping (model:thinking -> model)
- Helper function correctness
"""

import pytest

from backend.src.services.model_capabilities import (
    DEFAULT_CAPABILITY,
    MODEL_CAPABILITIES,
    ModelCapability,
    ReasoningApproach,
    get_model_capability,
    requires_reasoning_passback,
    should_use_reasoning_param,
    should_use_thinking_suffix,
    supports_native_function_calling,
)


class TestReasoningApproach:
    """Test the ReasoningApproach enum values."""

    def test_enum_values(self):
        """Verify all expected enum values exist."""
        assert ReasoningApproach.REASONING_PARAM == "reasoning_param"
        assert ReasoningApproach.THINKING_SUFFIX == "thinking_suffix"
        assert ReasoningApproach.NATIVE_DEEPSEEK == "native_deepseek"
        assert ReasoningApproach.PROMPT_BASED == "prompt_based"

    def test_enum_string_conversion(self):
        """Enum values should be usable as strings."""
        # str(Enum) on a StrEnum returns the value directly in comparisons
        assert ReasoningApproach.REASONING_PARAM == "reasoning_param"
        assert ReasoningApproach.THINKING_SUFFIX == "thinking_suffix"
        # Direct value access
        assert ReasoningApproach.REASONING_PARAM.value == "reasoning_param"
        assert ReasoningApproach.THINKING_SUFFIX.value == "thinking_suffix"


class TestModelCapabilityDataclass:
    """Test the ModelCapability dataclass."""

    def test_default_values(self):
        """Verify default values are applied correctly."""
        cap = ModelCapability(reasoning_approach=ReasoningApproach.PROMPT_BASED)
        assert cap.supports_parallel_tools is True
        assert cap.supports_function_calling is True
        assert cap.requires_reasoning_passback is False
        assert cap.max_thinking_tokens is None

    def test_custom_values(self):
        """Verify custom values override defaults."""
        cap = ModelCapability(
            reasoning_approach=ReasoningApproach.NATIVE_DEEPSEEK,
            supports_parallel_tools=False,
            supports_function_calling=False,
            requires_reasoning_passback=True,
            max_thinking_tokens=8000,
        )
        assert cap.reasoning_approach == ReasoningApproach.NATIVE_DEEPSEEK
        assert cap.supports_parallel_tools is False
        assert cap.supports_function_calling is False
        assert cap.requires_reasoning_passback is True
        assert cap.max_thinking_tokens == 8000

    def test_frozen_dataclass(self):
        """ModelCapability should be immutable."""
        cap = ModelCapability(reasoning_approach=ReasoningApproach.PROMPT_BASED)
        with pytest.raises(Exception):  # FrozenInstanceError
            cap.reasoning_approach = ReasoningApproach.REASONING_PARAM


class TestGetModelCapability:
    """Test the get_model_capability function."""

    # Anthropic/Claude models
    @pytest.mark.parametrize(
        "model_id",
        [
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            "anthropic/claude-sonnet-4",
            "anthropic/claude-opus-4",
            "anthropic/claude-3.5-sonnet",
            "Anthropic/Claude-3-Opus",  # Case insensitivity
        ],
    )
    def test_claude_models_use_reasoning_param(self, model_id: str):
        """Claude models should use the reasoning parameter."""
        cap = get_model_capability(model_id)
        assert cap.reasoning_approach == ReasoningApproach.REASONING_PARAM
        assert cap.supports_parallel_tools is True
        assert cap.supports_function_calling is True
        assert cap.max_thinking_tokens == 16000

    # DeepSeek models
    def test_deepseek_reasoner_native(self):
        """DeepSeek Reasoner should use native reasoning with passback."""
        cap = get_model_capability("deepseek/deepseek-reasoner")
        assert cap.reasoning_approach == ReasoningApproach.NATIVE_DEEPSEEK
        assert cap.requires_reasoning_passback is True
        assert cap.supports_function_calling is False

    def test_deepseek_chat_prompt_based(self):
        """DeepSeek Chat should use prompt-based reasoning."""
        cap = get_model_capability("deepseek/deepseek-chat")
        assert cap.reasoning_approach == ReasoningApproach.PROMPT_BASED
        assert cap.requires_reasoning_passback is False
        assert cap.supports_function_calling is False

    # Google/Gemini models
    @pytest.mark.parametrize(
        "model_id",
        [
            "google/gemini-pro",
            "google/gemini-1.5-pro",
            "google/gemini-2.0-flash",
            "google/gemini-2.5-pro",
            "Google/Gemini-Pro",  # Case insensitivity
        ],
    )
    def test_gemini_models_use_thinking_suffix(self, model_id: str):
        """Gemini models should use :thinking suffix."""
        cap = get_model_capability(model_id)
        assert cap.reasoning_approach == ReasoningApproach.THINKING_SUFFIX
        assert cap.supports_parallel_tools is True
        assert cap.supports_function_calling is True

    # OpenAI models
    @pytest.mark.parametrize(
        "model_id",
        [
            "openai/gpt-4",
            "openai/gpt-4-turbo",
            "openai/gpt-4o",
            "openai/gpt-3.5-turbo",
            "openai/o1-preview",
            "OpenAI/GPT-4",  # Case insensitivity
        ],
    )
    def test_openai_models_prompt_based(self, model_id: str):
        """OpenAI models should use prompt-based reasoning."""
        cap = get_model_capability(model_id)
        assert cap.reasoning_approach == ReasoningApproach.PROMPT_BASED
        assert cap.supports_parallel_tools is True
        assert cap.supports_function_calling is True

    # Meta Llama models
    @pytest.mark.parametrize(
        "model_id",
        [
            "meta-llama/llama-3-70b",
            "meta-llama/llama-3.1-405b",
            "meta-llama/llama-2-70b-chat",
            "Meta-Llama/Llama-3-70B",  # Case insensitivity
        ],
    )
    def test_llama_models_prompt_based(self, model_id: str):
        """Llama models should use prompt-based reasoning."""
        cap = get_model_capability(model_id)
        assert cap.reasoning_approach == ReasoningApproach.PROMPT_BASED
        assert cap.supports_parallel_tools is False  # Llama doesn't support parallel tools
        assert cap.supports_function_calling is True

    # Mistral models
    @pytest.mark.parametrize(
        "model_id",
        [
            "mistralai/mistral-large",
            "mistralai/mistral-medium",
            "mistralai/mixtral-8x7b",
            "MistralAI/Mistral-Large",  # Case insensitivity
        ],
    )
    def test_mistral_models_prompt_based(self, model_id: str):
        """Mistral models should use prompt-based reasoning."""
        cap = get_model_capability(model_id)
        assert cap.reasoning_approach == ReasoningApproach.PROMPT_BASED
        assert cap.supports_parallel_tools is True
        assert cap.supports_function_calling is True


class TestSuffixStripping:
    """Test that model suffixes like :thinking are stripped before lookup."""

    def test_claude_with_thinking_suffix(self):
        """Claude model with :thinking suffix should still match."""
        cap = get_model_capability("anthropic/claude-3-sonnet:thinking")
        assert cap.reasoning_approach == ReasoningApproach.REASONING_PARAM

    def test_gemini_with_thinking_suffix(self):
        """Gemini model with :thinking suffix should still match."""
        cap = get_model_capability("google/gemini-1.5-pro:thinking")
        assert cap.reasoning_approach == ReasoningApproach.THINKING_SUFFIX

    def test_model_with_arbitrary_suffix(self):
        """Model with arbitrary suffix should still match."""
        cap = get_model_capability("openai/gpt-4:some-variant")
        assert cap.reasoning_approach == ReasoningApproach.PROMPT_BASED


class TestUnknownModels:
    """Test fallback behavior for unknown models."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "unknown/some-model",
            "custom/my-model",
            "local/llama-custom",
            "",
            "just-a-model-name",
        ],
    )
    def test_unknown_models_get_default_capability(self, model_id: str):
        """Unknown models should return default capability."""
        cap = get_model_capability(model_id)
        assert cap == DEFAULT_CAPABILITY
        assert cap.reasoning_approach == ReasoningApproach.PROMPT_BASED
        assert cap.supports_parallel_tools is True
        assert cap.supports_function_calling is True
        assert cap.requires_reasoning_passback is False


class TestHelperFunctions:
    """Test the convenience helper functions."""

    def test_should_use_reasoning_param_claude(self):
        """Claude models should return True for reasoning param check."""
        assert should_use_reasoning_param("anthropic/claude-3-opus") is True
        assert should_use_reasoning_param("anthropic/claude-sonnet-4") is True

    def test_should_use_reasoning_param_non_claude(self):
        """Non-Claude models should return False for reasoning param check."""
        assert should_use_reasoning_param("google/gemini-pro") is False
        assert should_use_reasoning_param("openai/gpt-4") is False
        assert should_use_reasoning_param("deepseek/deepseek-reasoner") is False

    def test_should_use_thinking_suffix_gemini(self):
        """Gemini models should return True for thinking suffix check."""
        assert should_use_thinking_suffix("google/gemini-pro") is True
        assert should_use_thinking_suffix("google/gemini-2.0-flash") is True

    def test_should_use_thinking_suffix_non_gemini(self):
        """Non-Gemini models should return False for thinking suffix check."""
        assert should_use_thinking_suffix("anthropic/claude-3-opus") is False
        assert should_use_thinking_suffix("openai/gpt-4") is False

    def test_requires_reasoning_passback_deepseek(self):
        """DeepSeek Reasoner should require reasoning passback."""
        assert requires_reasoning_passback("deepseek/deepseek-reasoner") is True

    def test_requires_reasoning_passback_others(self):
        """Other models should not require reasoning passback."""
        assert requires_reasoning_passback("anthropic/claude-3-opus") is False
        assert requires_reasoning_passback("google/gemini-pro") is False
        assert requires_reasoning_passback("deepseek/deepseek-chat") is False
        assert requires_reasoning_passback("openai/gpt-4") is False

    def test_supports_native_function_calling_true(self):
        """Models with native function calling should return True."""
        assert supports_native_function_calling("anthropic/claude-3-opus") is True
        assert supports_native_function_calling("google/gemini-pro") is True
        assert supports_native_function_calling("openai/gpt-4") is True

    def test_supports_native_function_calling_false(self):
        """DeepSeek models should not support native function calling."""
        assert supports_native_function_calling("deepseek/deepseek-reasoner") is False
        assert supports_native_function_calling("deepseek/deepseek-chat") is False


class TestModelCapabilitiesRegistry:
    """Test the MODEL_CAPABILITIES registry itself."""

    def test_registry_has_expected_entries(self):
        """Verify the registry contains expected model families."""
        expected_patterns = [
            "anthropic/claude-",
            "deepseek/deepseek-reasoner",
            "deepseek/deepseek-chat",
            "google/gemini-",
            "openai/",
            "meta-llama/",
            "mistralai/",
        ]
        for pattern in expected_patterns:
            assert pattern in MODEL_CAPABILITIES, f"Missing pattern: {pattern}"

    def test_default_capability_is_safe(self):
        """Default capability should be conservative and safe."""
        assert DEFAULT_CAPABILITY.reasoning_approach == ReasoningApproach.PROMPT_BASED
        assert DEFAULT_CAPABILITY.supports_parallel_tools is True
        assert DEFAULT_CAPABILITY.supports_function_calling is True
        assert DEFAULT_CAPABILITY.requires_reasoning_passback is False
        assert DEFAULT_CAPABILITY.max_thinking_tokens is None
