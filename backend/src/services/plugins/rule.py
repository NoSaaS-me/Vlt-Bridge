"""Rule definitions and enums for the Oracle Plugin System.

This module defines the core rule types, enums, and validation logic
for the plugin rule engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class HookPoint(str, Enum):
    """Agent lifecycle hook points for rule triggers.

    These define the specific points in the agent's execution lifecycle
    where rules can be triggered and evaluated.
    """

    ON_QUERY_START = "on_query_start"      # New user query received
    ON_TURN_START = "on_turn_start"        # Before agent processes turn
    ON_TURN_END = "on_turn_end"            # After agent completes turn
    ON_TOOL_CALL = "on_tool_call"          # Before tool execution
    ON_TOOL_COMPLETE = "on_tool_complete"  # After tool returns
    ON_TOOL_FAILURE = "on_tool_failure"    # When tool fails/times out
    ON_SESSION_END = "on_session_end"      # Session closing


class ActionType(str, Enum):
    """Types of actions a rule can execute.

    Each action type has specific parameters and behaviors defined
    in the RuleAction dataclass.
    """

    NOTIFY_SELF = "notify_self"  # Inject notification into agent context
    LOG = "log"                  # Write to system log
    SET_STATE = "set_state"      # Store plugin-scoped state
    EMIT_EVENT = "emit_event"    # Emit ANS event


class Priority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class InjectionPoint(str, Enum):
    """Where in the agent flow to inject notifications."""

    TURN_START = "turn_start"    # Injected before agent receives next prompt
    AFTER_TOOL = "after_tool"    # Injected after tool execution
    IMMEDIATE = "immediate"       # Injected as soon as event occurs


@dataclass
class RuleAction:
    """Action to execute when rule condition is met.

    Attributes:
        type: The type of action to execute.
        message: Notification message (Jinja2 template) for notify_self.
        category: Notification category for notify_self.
        priority: Priority level for notify_self.
        deliver_at: Injection point for notify_self.
        level: Log level (debug, info, warning, error) for log action.
        key: State key to set for set_state action.
        value: State value (can be template) for set_state action.
        event_type: Event type to emit for emit_event action.
        payload: Event payload template for emit_event action.
    """

    type: ActionType

    # For notify_self
    message: Optional[str] = None
    category: Optional[str] = None
    priority: Priority = Priority.NORMAL
    deliver_at: InjectionPoint = InjectionPoint.TURN_START

    # For log
    level: str = "info"

    # For set_state
    key: Optional[str] = None
    value: Optional[Any] = None

    # For emit_event
    event_type: Optional[str] = None
    payload: Optional[dict[str, Any]] = None

    def validate(self) -> list[str]:
        """Validate the action configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        if self.type == ActionType.NOTIFY_SELF:
            if not self.message:
                errors.append("notify_self action requires a message")

        elif self.type == ActionType.LOG:
            valid_levels = {"debug", "info", "warning", "error"}
            if self.level not in valid_levels:
                errors.append(f"Invalid log level: {self.level}. Must be one of: {valid_levels}")

        elif self.type == ActionType.SET_STATE:
            if not self.key:
                errors.append("set_state action requires a key")

        elif self.type == ActionType.EMIT_EVENT:
            if not self.event_type:
                errors.append("emit_event action requires an event_type")

        return errors


@dataclass
class Rule:
    """A rule definition loaded from TOML configuration.

    Rules define conditional behaviors that trigger on specific agent
    lifecycle events. Each rule has a trigger hook point, a condition
    or script for evaluation, and an action to execute when the condition
    is met.

    Attributes:
        id: Unique identifier (kebab-case).
        name: Human-readable name.
        description: What the rule does.
        version: Semantic version string.
        trigger: Which lifecycle event activates this rule.
        condition: Expression string (simpleeval) for evaluation.
        script: Path to Lua script (alternative to condition).
        action: What happens when rule fires.
        priority: Higher values fire earlier (default: 100).
        enabled: Whether rule is active.
        core: If True, cannot be disabled by user.
        plugin_id: Parent plugin (None for standalone rules).
        source_path: File path where rule was loaded from.
    """

    # Identity
    id: str
    name: str
    description: str
    version: str = "1.0.0"

    # Trigger
    trigger: HookPoint = field(default=HookPoint.ON_TURN_START)

    # Condition (XOR with script)
    condition: Optional[str] = None
    script: Optional[str] = None

    # Action
    action: Optional[RuleAction] = None

    # Metadata
    priority: int = 100
    enabled: bool = True
    core: bool = False

    # Source
    plugin_id: Optional[str] = None
    source_path: str = ""

    def validate(self) -> list[str]:
        """Validate the rule configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Validate ID format (kebab-case)
        if not re.match(r"^[a-z0-9-]+$", self.id):
            errors.append(f"Rule ID must be kebab-case: {self.id}")

        # Validate condition XOR script
        if self.condition and self.script:
            errors.append("Rule cannot have both condition and script")
        if not self.condition and not self.script:
            errors.append("Rule must have either condition or script")

        # Validate priority range (recommended 1-1000)
        if self.priority < 1 or self.priority > 1000:
            errors.append(f"Rule priority should be 1-1000, got: {self.priority}")

        # Validate action
        if self.action is None:
            errors.append("Rule must have an action")
        else:
            action_errors = self.action.validate()
            errors.extend(action_errors)

        return errors

    @property
    def qualified_id(self) -> str:
        """Return the fully qualified rule ID including plugin prefix if applicable."""
        if self.plugin_id:
            return f"{self.plugin_id}:{self.id}"
        return self.id


__all__ = [
    "HookPoint",
    "ActionType",
    "Priority",
    "InjectionPoint",
    "RuleAction",
    "Rule",
]
