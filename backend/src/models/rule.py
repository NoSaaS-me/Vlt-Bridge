"""Pydantic models for Rules API.

These models define the API contract for rule management endpoints,
matching the OpenAPI schema in rules-api.yaml.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class HookPointEnum(str, Enum):
    """Agent lifecycle hook points for rule triggers."""

    ON_QUERY_START = "on_query_start"
    ON_TURN_START = "on_turn_start"
    ON_TURN_END = "on_turn_end"
    ON_TOOL_CALL = "on_tool_call"
    ON_TOOL_COMPLETE = "on_tool_complete"
    ON_TOOL_FAILURE = "on_tool_failure"
    ON_SESSION_END = "on_session_end"


class ActionTypeEnum(str, Enum):
    """Types of actions a rule can execute."""

    NOTIFY_SELF = "notify_self"
    LOG = "log"
    SET_STATE = "set_state"
    EMIT_EVENT = "emit_event"


class PriorityEnum(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class InjectionPointEnum(str, Enum):
    """Where in the agent flow to inject notifications."""

    IMMEDIATE = "immediate"
    TURN_START = "turn_start"
    AFTER_TOOL = "after_tool"
    TURN_END = "turn_end"


class RuleAction(BaseModel):
    """Action configuration for a rule."""

    type: ActionTypeEnum = Field(..., description="Type of action to execute")
    message: Optional[str] = Field(None, description="Notification message (for notify_self)")
    category: Optional[str] = Field(None, description="Notification category")
    priority: PriorityEnum = Field(PriorityEnum.NORMAL, description="Priority level")
    deliver_at: InjectionPointEnum = Field(
        InjectionPointEnum.TURN_START, description="Injection point"
    )


class RuleInfo(BaseModel):
    """Summary information about a rule for list responses."""

    id: str = Field(..., description="Rule identifier (kebab-case)")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Rule description")
    trigger: HookPointEnum = Field(..., description="Hook point that triggers the rule")
    enabled: bool = Field(..., description="Whether the rule is currently enabled")
    core: bool = Field(..., description="Whether the rule is a core rule (cannot be disabled)")
    priority: int = Field(100, description="Rule priority (higher fires first)")
    plugin_id: Optional[str] = Field(None, description="Parent plugin ID if applicable")


class RuleDetail(RuleInfo):
    """Detailed information about a rule including condition and action."""

    version: str = Field("1.0.0", description="Semantic version string")
    condition: Optional[str] = Field(None, description="Expression condition (simpleeval)")
    script: Optional[str] = Field(None, description="Lua script path (alternative to condition)")
    action: Optional[RuleAction] = Field(None, description="Action configuration")
    source_path: str = Field("", description="File path where rule was loaded from")


class RuleListResponse(BaseModel):
    """Response for listing rules."""

    rules: list[RuleInfo] = Field(..., description="List of rules")
    total: int = Field(..., description="Total number of rules")


class RuleToggleRequest(BaseModel):
    """Request body for toggling a rule."""

    enabled: bool = Field(..., description="Whether to enable or disable the rule")


class RuleTestRequest(BaseModel):
    """Request body for testing a rule."""

    context_override: Optional[dict[str, Any]] = Field(
        None, description="Optional context values to override for testing"
    )


class RuleTestResponse(BaseModel):
    """Response from testing a rule."""

    condition_result: bool = Field(..., description="Whether the condition matched")
    action_would_execute: bool = Field(..., description="Whether the action would have executed")
    evaluation_time_ms: float = Field(..., description="Time taken to evaluate in milliseconds")
    error: Optional[str] = Field(None, description="Error message if evaluation failed")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    detail: Optional[str] = Field(None, description="Error details")


# Plugin models


class PluginSettingSchema(BaseModel):
    """Schema for a plugin setting."""

    name: str = Field(..., description="Human-readable setting name")
    type: str = Field(..., description="Data type (integer, float, string, boolean)")
    default: Any = Field(..., description="Default value")
    description: str = Field("", description="What this setting controls")
    min_value: Optional[float] = Field(None, description="Minimum value for numeric types")
    max_value: Optional[float] = Field(None, description="Maximum value for numeric types")
    options: Optional[list[str]] = Field(None, description="Valid options for enum/select types")


class PluginInfo(BaseModel):
    """Summary information about a plugin for list responses."""

    id: str = Field(..., description="Plugin identifier (kebab-case)")
    name: str = Field(..., description="Display name")
    version: str = Field(..., description="Semantic version string")
    description: str = Field("", description="What the plugin provides")
    rule_count: int = Field(0, description="Number of rules in this plugin")
    enabled: bool = Field(True, description="Whether the plugin is currently enabled")


class PluginDetail(PluginInfo):
    """Detailed information about a plugin."""

    rules: list[RuleInfo] = Field(default_factory=list, description="Rules in this plugin")
    requires: list[str] = Field(default_factory=list, description="Required capabilities")
    settings_schema: dict[str, PluginSettingSchema] = Field(
        default_factory=dict, description="User-configurable settings"
    )
    source_dir: str = Field("", description="Directory where plugin was loaded")


class PluginListResponse(BaseModel):
    """Response for listing plugins."""

    plugins: list[PluginInfo] = Field(..., description="List of plugins")


class PluginSettingsUpdateRequest(BaseModel):
    """Request body for updating plugin settings."""

    settings: dict[str, Any] = Field(..., description="Setting values to update")


__all__ = [
    "HookPointEnum",
    "ActionTypeEnum",
    "PriorityEnum",
    "InjectionPointEnum",
    "RuleAction",
    "RuleInfo",
    "RuleDetail",
    "RuleListResponse",
    "RuleToggleRequest",
    "RuleTestRequest",
    "RuleTestResponse",
    "ErrorResponse",
    # Plugin models
    "PluginSettingSchema",
    "PluginInfo",
    "PluginDetail",
    "PluginListResponse",
    "PluginSettingsUpdateRequest",
]
