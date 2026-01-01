"""Pydantic models for user settings and model providers."""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """Available model providers."""
    OPENROUTER = "openrouter"
    GOOGLE = "google"


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
