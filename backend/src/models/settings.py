"""Pydantic models for user settings and model providers."""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """Available model providers."""
    OPENROUTER = "openrouter"
    GOOGLE = "google"


class ReasoningEffort(str, Enum):
    """Level of reasoning effort for models that support it."""
    LOW = "low"        # Quick reasoning, fewer tokens
    MEDIUM = "medium"  # Balanced (default)
    HIGH = "high"      # Thorough reasoning, more tokens


class ModelSettings(BaseModel):
    """User's model preferences for oracle and subagent."""
    oracle_model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Model to use for oracle queries"
    )
    oracle_provider: ModelProvider = Field(
        default=ModelProvider.GOOGLE,
        description="Provider for oracle model"
    )
    subagent_model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Model to use for subagent tasks"
    )
    subagent_provider: ModelProvider = Field(
        default=ModelProvider.GOOGLE,
        description="Provider for subagent model"
    )
    thinking_enabled: bool = Field(
        default=False,
        description="Enable extended thinking mode (adds :thinking suffix for supported models)"
    )
    reasoning_effort: ReasoningEffort = Field(
        default=ReasoningEffort.MEDIUM,
        description="Level of reasoning effort for extended thinking (low/medium/high)"
    )
    chat_center_mode: bool = Field(
        default=False,
        description="Show AI chat in center view instead of flyout panel"
    )
    librarian_timeout: int = Field(
        default=1200,
        ge=60,
        le=3600,
        description="Timeout in seconds for Librarian subagent tasks (default: 1200 = 20 minutes, max: 3600 = 1 hour)"
    )
    max_context_nodes: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Maximum context nodes to keep per conversation tree before pruning (default: 30)"
    )
    openrouter_api_key: Optional[str] = Field(
        default=None,
        description="User's OpenRouter API key for accessing paid models"
    )
    openrouter_api_key_set: bool = Field(
        default=False,
        description="Whether an OpenRouter API key has been configured (key itself is not returned)"
    )
    # AgentConfig fields for turn control
    max_iterations: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum agent turns per query"
    )
    soft_warning_percent: int = Field(
        default=70,
        ge=50,
        le=90,
        description="Percentage of max iterations to trigger warning"
    )
    token_budget: int = Field(
        default=50000,
        ge=1000,
        le=200000,
        description="Maximum tokens per session"
    )
    token_warning_percent: int = Field(
        default=80,
        ge=50,
        le=95,
        description="Percentage of token budget to trigger warning"
    )
    timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Overall query timeout in seconds"
    )
    max_tool_calls_per_turn: int = Field(
        default=100,
        ge=1,
        le=200,
        description="Maximum tool calls per agent turn"
    )
    max_parallel_tools: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent tool executions"
    )
    tool_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Per-tool execution timeout in seconds"
    )
    loop_detection_window_seconds: int = Field(
        default=300,
        ge=60,
        le=600,
        description="Time window in seconds for loop detection (5 min default)"
    )


class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str = Field(..., description="Model identifier (e.g., 'deepseek/deepseek-chat')")
    name: str = Field(..., description="Human-readable model name")
    provider: ModelProvider = Field(..., description="Model provider")
    is_free: bool = Field(default=False, description="Whether the model is free to use")
    supports_thinking: bool = Field(
        default=False,
        description="Whether model supports :thinking suffix for extended reasoning"
    )
    context_length: Optional[int] = Field(
        None,
        description="Maximum context length in tokens"
    )
    description: Optional[str] = Field(
        None,
        description="Model description"
    )


class ModelsListResponse(BaseModel):
    """Response containing available models."""
    models: List[ModelInfo] = Field(default_factory=list, description="List of available models")


class ModelSettingsUpdateRequest(BaseModel):
    """Request to update user model settings."""
    oracle_model: Optional[str] = None
    oracle_provider: Optional[ModelProvider] = None
    subagent_model: Optional[str] = None
    subagent_provider: Optional[ModelProvider] = None
    thinking_enabled: Optional[bool] = None
    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None,
        description="Level of reasoning effort (low/medium/high)"
    )
    chat_center_mode: Optional[bool] = None
    librarian_timeout: Optional[int] = Field(
        default=None,
        ge=60,
        le=3600,
        description="Timeout in seconds for Librarian subagent tasks (60-3600)"
    )
    max_context_nodes: Optional[int] = Field(
        default=None,
        ge=5,
        le=100,
        description="Maximum context nodes per conversation tree (5-100)"
    )
    openrouter_api_key: Optional[str] = Field(
        default=None,
        description="OpenRouter API key (set to empty string to clear)"
    )
    # AgentConfig fields for turn control
    max_iterations: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum agent turns per query (1-50)"
    )
    soft_warning_percent: Optional[int] = Field(
        default=None,
        ge=50,
        le=90,
        description="Percentage of max iterations to trigger warning (50-90)"
    )
    token_budget: Optional[int] = Field(
        default=None,
        ge=1000,
        le=200000,
        description="Maximum tokens per session (1000-200000)"
    )
    token_warning_percent: Optional[int] = Field(
        default=None,
        ge=50,
        le=95,
        description="Percentage of token budget to trigger warning (50-95)"
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        ge=10,
        le=600,
        description="Overall query timeout in seconds (10-600)"
    )
    max_tool_calls_per_turn: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Maximum tool calls per agent turn (1-200)"
    )
    max_parallel_tools: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Maximum concurrent tool executions (1-10)"
    )
    tool_timeout_seconds: Optional[int] = Field(
        default=None,
        ge=5,
        le=120,
        description="Per-tool execution timeout in seconds (5-120)"
    )
    loop_detection_window_seconds: Optional[int] = Field(
        default=None,
        ge=60,
        le=600,
        description="Time window in seconds for loop detection (60-600)"
    )


class AgentConfig(BaseModel):
    """Agent configuration for turn control."""
    max_iterations: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum agent turns per query"
    )
    loop_detection_window_seconds: int = Field(
        default=300,
        ge=60,
        le=600,
        description="Seconds of continuous identical actions before terminating (default: 300 = 5 minutes)"
    )
    soft_warning_percent: int = Field(
        default=70,
        ge=50,
        le=90,
        description="Percentage of max iterations to trigger warning"
    )
    token_budget: int = Field(
        default=50000,
        ge=1000,
        le=200000,
        description="Maximum tokens per session"
    )
    token_warning_percent: int = Field(
        default=80,
        ge=50,
        le=95,
        description="Percentage of token budget to trigger warning"
    )
    timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Overall query timeout in seconds"
    )
    max_tool_calls_per_turn: int = Field(
        default=100,
        ge=1,
        le=200,
        description="Maximum tool calls per agent turn"
    )
    max_parallel_tools: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent tool executions"
    )
    tool_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Per-tool execution timeout in seconds"
    )


class AgentConfigUpdate(BaseModel):
    """Partial update for agent configuration."""
    max_iterations: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum agent turns per query (1-50)"
    )
    soft_warning_percent: Optional[int] = Field(
        default=None,
        ge=50,
        le=90,
        description="Percentage of max iterations to trigger warning (50-90)"
    )
    token_budget: Optional[int] = Field(
        default=None,
        ge=1000,
        le=200000,
        description="Maximum tokens per session (1000-200000)"
    )
    token_warning_percent: Optional[int] = Field(
        default=None,
        ge=50,
        le=95,
        description="Percentage of token budget to trigger warning (50-95)"
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        ge=10,
        le=600,
        description="Overall query timeout in seconds (10-600)"
    )
    max_tool_calls_per_turn: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Maximum tool calls per agent turn (1-200)"
    )
    max_parallel_tools: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Maximum concurrent tool executions (1-10)"
    )
    tool_timeout_seconds: Optional[int] = Field(
        default=None,
        ge=5,
        le=120,
        description="Per-tool execution timeout in seconds (5-120)"
    )
    loop_detection_window_seconds: Optional[int] = Field(
        default=None,
        ge=60,
        le=600,
        description="Time window in seconds for loop detection (60-600)"
    )
