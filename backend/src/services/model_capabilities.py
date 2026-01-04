"""Model capability registry for reasoning approach detection.

Different LLM providers handle chain-of-thought reasoning differently:
- Anthropic/Claude: Use the `reasoning` parameter in the request body
- DeepSeek: Native reasoning that returns `reasoning_content` which MUST be passed back
- Google/Gemini: Use `:thinking` suffix on model name
- Others: Use prompt-based CoT with the think tool

This registry allows the Oracle agent to determine the correct approach at runtime.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class ReasoningApproach(str, Enum):
    """How to enable extended thinking for a model."""
    REASONING_PARAM = "reasoning_param"  # Use reasoning: {enabled: true}
    THINKING_SUFFIX = "thinking_suffix"  # Append :thinking to model name
    NATIVE_DEEPSEEK = "native_deepseek"  # DeepSeek native with reasoning_content passback
    PROMPT_BASED = "prompt_based"  # Use think tool only, no native support


@dataclass(frozen=True)
class ModelCapability:
    """Capabilities for a specific model or model family."""
    reasoning_approach: ReasoningApproach
    supports_parallel_tools: bool = True
    supports_function_calling: bool = True
    requires_reasoning_passback: bool = False  # DeepSeek requires reasoning_content passed back
    max_thinking_tokens: Optional[int] = None  # If model has a limit


# Model family patterns and their capabilities
MODEL_CAPABILITIES: Dict[str, ModelCapability] = {
    # Anthropic Claude models - use reasoning param
    "anthropic/claude-": ModelCapability(
        reasoning_approach=ReasoningApproach.REASONING_PARAM,
        supports_parallel_tools=True,
        supports_function_calling=True,
        max_thinking_tokens=16000,
    ),

    # DeepSeek models - native reasoning with passback requirement
    "deepseek/deepseek-reasoner": ModelCapability(
        reasoning_approach=ReasoningApproach.NATIVE_DEEPSEEK,
        supports_parallel_tools=True,
        supports_function_calling=False,  # Often uses XML-style calls
        requires_reasoning_passback=True,
    ),
    "deepseek/deepseek-chat": ModelCapability(
        reasoning_approach=ReasoningApproach.PROMPT_BASED,
        supports_parallel_tools=True,
        supports_function_calling=False,  # Often uses XML-style calls
    ),

    # Google Gemini models - use :thinking suffix
    "google/gemini-": ModelCapability(
        reasoning_approach=ReasoningApproach.THINKING_SUFFIX,
        supports_parallel_tools=True,
        supports_function_calling=True,
    ),

    # OpenAI models - prompt-based only
    "openai/": ModelCapability(
        reasoning_approach=ReasoningApproach.PROMPT_BASED,
        supports_parallel_tools=True,
        supports_function_calling=True,
    ),

    # Meta Llama models - prompt-based
    "meta-llama/": ModelCapability(
        reasoning_approach=ReasoningApproach.PROMPT_BASED,
        supports_parallel_tools=False,
        supports_function_calling=True,
    ),

    # Mistral models - prompt-based
    "mistralai/": ModelCapability(
        reasoning_approach=ReasoningApproach.PROMPT_BASED,
        supports_parallel_tools=True,
        supports_function_calling=True,
    ),
}

# Default capability for unknown models
DEFAULT_CAPABILITY = ModelCapability(
    reasoning_approach=ReasoningApproach.PROMPT_BASED,
    supports_parallel_tools=True,
    supports_function_calling=True,
)


def get_model_capability(model_id: str) -> ModelCapability:
    """Get the capability profile for a model.

    Args:
        model_id: Full model identifier (e.g., "anthropic/claude-sonnet-4")

    Returns:
        ModelCapability with reasoning approach and feature flags
    """
    # Normalize model ID
    model_lower = model_id.lower()

    # Remove any existing suffixes like :thinking
    if ":" in model_lower:
        model_lower = model_lower.split(":")[0]

    # Check each pattern (prefix matching)
    for pattern, capability in MODEL_CAPABILITIES.items():
        if model_lower.startswith(pattern.lower()):
            return capability

    return DEFAULT_CAPABILITY


def should_use_reasoning_param(model_id: str) -> bool:
    """Check if model should use the reasoning parameter."""
    cap = get_model_capability(model_id)
    return cap.reasoning_approach == ReasoningApproach.REASONING_PARAM


def should_use_thinking_suffix(model_id: str) -> bool:
    """Check if model should use :thinking suffix."""
    cap = get_model_capability(model_id)
    return cap.reasoning_approach == ReasoningApproach.THINKING_SUFFIX


def requires_reasoning_passback(model_id: str) -> bool:
    """Check if model requires reasoning_content to be passed back."""
    cap = get_model_capability(model_id)
    return cap.requires_reasoning_passback


def supports_native_function_calling(model_id: str) -> bool:
    """Check if model supports native function calling (vs XML-style)."""
    cap = get_model_capability(model_id)
    return cap.supports_function_calling
