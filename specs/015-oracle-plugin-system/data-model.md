# Data Model: Oracle Plugin System

**Date**: 2026-01-04
**Branch**: `015-oracle-plugin-system`
**Spec**: [spec.md](./spec.md)

## Overview

This document defines the data entities, relationships, and validation rules for the Oracle Plugin System.

---

## Core Entities

### Rule

A rule defines a conditional behavior that triggers on specific agent lifecycle events.

```python
@dataclass
class Rule:
    """A rule definition loaded from TOML configuration."""

    # Identity
    id: str                     # Unique identifier (kebab-case)
    name: str                   # Human-readable name
    description: str            # What the rule does
    version: str = "1.0.0"      # Semantic version

    # Trigger
    trigger: HookPoint          # Which lifecycle event activates this rule

    # Condition
    condition: Optional[str]    # Expression string (simpleeval)
    script: Optional[str]       # Path to Lua script (alternative to condition)

    # Action
    action: RuleAction          # What happens when rule fires

    # Metadata
    priority: int = 100         # Higher = fires earlier (default: 100)
    enabled: bool = True        # Whether rule is active
    core: bool = False          # If True, cannot be disabled by user

    # Source
    plugin_id: Optional[str]    # Parent plugin (None for standalone rules)
    source_path: str            # File path where rule was loaded from

    def validate(self) -> list[str]:
        """Returns list of validation errors (empty if valid)."""
        errors = []
        if not re.match(r'^[a-z0-9-]+$', self.id):
            errors.append(f"Rule ID must be kebab-case: {self.id}")
        if self.condition and self.script:
            errors.append("Rule cannot have both condition and script")
        if not self.condition and not self.script:
            errors.append("Rule must have either condition or script")
        return errors
```

**Validation Rules**:
- `id`: kebab-case, unique within plugin scope
- `trigger`: Must be valid HookPoint enum value
- `condition` XOR `script`: Exactly one must be set
- `priority`: 1-1000 range recommended
- `action`: Must have valid action type

---

### HookPoint

Enum defining agent lifecycle points where rules can attach.

```python
class HookPoint(str, Enum):
    """Agent lifecycle hook points for rule triggers."""

    ON_QUERY_START = "on_query_start"      # New user query received
    ON_TURN_START = "on_turn_start"        # Before agent processes turn
    ON_TURN_END = "on_turn_end"            # After agent completes turn
    ON_TOOL_CALL = "on_tool_call"          # Before tool execution
    ON_TOOL_COMPLETE = "on_tool_complete"  # After tool returns
    ON_TOOL_FAILURE = "on_tool_failure"    # When tool fails/times out
    ON_SESSION_END = "on_session_end"      # Session closing
```

**Mapping to ANS Events**:
| HookPoint | EventType | Notes |
|-----------|-----------|-------|
| ON_QUERY_START | QUERY_START | New event type |
| ON_TURN_START | AGENT_TURN_START | Existing |
| ON_TURN_END | AGENT_TURN_END | Existing |
| ON_TOOL_CALL | TOOL_CALL_PENDING | Existing |
| ON_TOOL_COMPLETE | TOOL_CALL_SUCCESS | Existing |
| ON_TOOL_FAILURE | TOOL_CALL_FAILURE | Existing |
| ON_SESSION_END | SESSION_END | New event type |

---

### RuleAction

Defines what happens when a rule fires.

```python
@dataclass
class RuleAction:
    """Action to execute when rule condition is met."""

    type: ActionType            # Action type

    # For notify_self
    message: Optional[str]      # Notification message (Jinja2 template)
    category: Optional[str]     # Notification category
    priority: Priority = Priority.NORMAL
    deliver_at: InjectionPoint = InjectionPoint.TURN_START

    # For log
    level: str = "info"         # debug, info, warning, error

    # For set_state
    key: Optional[str]          # State key to set
    value: Optional[Any]        # State value (can be template)

    # For emit_event
    event_type: Optional[str]   # Event type to emit
    payload: Optional[dict]     # Event payload template


class ActionType(str, Enum):
    """Types of actions a rule can execute."""

    NOTIFY_SELF = "notify_self"  # Inject notification into agent context
    LOG = "log"                  # Write to system log
    SET_STATE = "set_state"      # Store plugin-scoped state
    EMIT_EVENT = "emit_event"    # Emit ANS event
```

---

### RuleContext

Read-only context API available to rule conditions and scripts.

```python
@dataclass
class RuleContext:
    """Context available to rule evaluation."""

    turn: TurnState
    history: HistoryState
    user: UserState
    project: ProjectState
    state: PluginState
    event: Optional[EventData]  # The triggering event (if applicable)
    result: Optional[ToolResult]  # Tool result (for ON_TOOL_COMPLETE)


@dataclass
class TurnState:
    """Current turn information."""

    number: int                 # Turn number (1-indexed)
    token_usage: float          # Token budget usage (0.0-1.0)
    context_usage: float        # Context window usage (0.0-1.0)
    iteration_count: int        # Current iteration in turn


@dataclass
class HistoryState:
    """Historical information."""

    messages: list[dict]        # Recent messages (role, content)
    tools: list[ToolCallRecord] # Recent tool calls
    failures: dict[str, int]    # Tool name → failure count


@dataclass
class ToolCallRecord:
    """Record of a tool call."""

    name: str
    arguments: dict
    result: Optional[str]
    success: bool
    timestamp: datetime


@dataclass
class UserState:
    """User information (read-only)."""

    id: str
    settings: dict              # User settings snapshot


@dataclass
class ProjectState:
    """Project information (read-only)."""

    id: str
    settings: dict              # Project settings snapshot


@dataclass
class PluginState:
    """Plugin-scoped persistent state."""

    _store: dict                # Internal storage

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from plugin state."""
        return self._store.get(key, default)

    # Note: set() is only available in actions, not conditions
```

---

### Plugin

A plugin packages multiple rules with shared configuration.

```python
@dataclass
class Plugin:
    """A plugin definition with multiple rules."""

    # Identity
    id: str                     # Unique identifier
    name: str                   # Display name
    version: str                # Semantic version
    description: str            # What the plugin provides

    # Rules
    rules: list[Rule]           # Rules this plugin provides

    # Dependencies
    requires: list[str]         # Required capabilities (e.g., "vault_search")

    # Configuration
    settings: dict[str, PluginSetting]  # User-configurable parameters

    # Source
    source_dir: str             # Directory where plugin was loaded


@dataclass
class PluginSetting:
    """A configurable plugin setting."""

    name: str
    type: str                   # "integer", "float", "string", "boolean"
    default: Any
    description: str
    min_value: Optional[float]  # For numeric types
    max_value: Optional[float]  # For numeric types
    options: Optional[list[str]]  # For enum/select types
```

---

## Database Schema

### Plugin State Table

```sql
-- Plugin-scoped persistent state
CREATE TABLE IF NOT EXISTS plugin_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    plugin_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(user_id, project_id, plugin_id, key)
);

CREATE INDEX idx_plugin_state_lookup
ON plugin_state(user_id, project_id, plugin_id);
```

### Rule Execution Log Table (Optional - for debugging)

```sql
-- Rule execution history for debugging
CREATE TABLE IF NOT EXISTS rule_execution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    condition_result INTEGER NOT NULL,  -- 0 = false, 1 = true
    action_executed INTEGER NOT NULL,   -- 0 = no, 1 = yes
    execution_time_ms REAL NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_rule_log_lookup
ON rule_execution_log(user_id, project_id, created_at DESC);
```

### User Settings Extension

Extend existing `user_settings` table:

```sql
-- Add column for disabled rules (parallel to disabled_subscribers)
ALTER TABLE user_settings
ADD COLUMN disabled_rules_json TEXT DEFAULT '[]';

-- Add column for plugin settings overrides
ALTER TABLE user_settings
ADD COLUMN plugin_settings_json TEXT DEFAULT '{}';
```

---

## TOML Schema

### Rule TOML Format

```toml
# rules/token-budget-warning.toml

[rule]
id = "token-budget-warning"
name = "Token Budget Warning"
description = "Warn when token usage exceeds 80%"
version = "1.0.0"
trigger = "on_turn_start"
priority = 100
enabled = true
core = true

[condition]
expression = "context.turn.token_usage > 0.8"
# OR: script = "scripts/check_token_budget.lua"

[action]
type = "notify_self"
message = "Token budget at {{ (context.turn.token_usage * 100) | int }}%. Consider wrapping up or summarizing."
category = "warning"
priority = "high"
deliver_at = "turn_start"
```

### Plugin Manifest Format

```toml
# plugins/research-assistant/manifest.toml

[plugin]
id = "research-assistant"
name = "Research Assistant"
version = "1.0.0"
description = "Enhanced research workflow with multi-step guidance"

[capabilities]
requires = ["vault_search", "web_search"]

[rules]
# Rules are loaded from ./rules/*.toml
include = ["rules/*.toml"]

[settings]
# User-configurable settings

[settings.max_sources]
type = "integer"
default = 10
min = 1
max = 50
description = "Maximum sources to gather before synthesizing"

[settings.auto_summarize]
type = "boolean"
default = true
description = "Automatically summarize large result sets"
```

---

## Entity Relationships

```
Plugin (1) ──────────────── (N) Rule
   │                              │
   │                              ├── HookPoint (trigger)
   │                              ├── RuleAction (action)
   │                              └── condition OR script
   │
   └── PluginSetting (N)

Rule ──── evaluates with ──── RuleContext
                                   │
                                   ├── TurnState
                                   ├── HistoryState
                                   ├── UserState
                                   ├── ProjectState
                                   ├── PluginState
                                   └── EventData (optional)
```

---

## State Transitions

### Rule Lifecycle

```
LOADED → ENABLED → TRIGGERED → EVALUATED → ACTION_EXECUTED
   │        │          │           │              │
   │        │          │           └── condition=false: NO_ACTION
   │        │          │
   │        │          └── event matches trigger
   │        │
   │        └── user has not disabled
   │
   └── TOML validation passed
```

### Plugin Lifecycle

```
DISCOVERED → VALIDATED → LOADED → ACTIVE
     │           │          │        │
     │           │          │        └── rules registered with engine
     │           │          │
     │           │          └── dependencies satisfied
     │           │
     │           └── manifest + rules valid
     │
     └── manifest.toml found in plugins directory
```

---

## Implementation Status Markers

For documentation purposes, mark each entity:

| Entity | Status | Notes |
|--------|--------|-------|
| Rule | 🟡 Planned | MVP scope |
| HookPoint | 🟡 Planned | MVP scope |
| RuleAction | 🟡 Planned | MVP scope |
| RuleContext | 🟡 Planned | MVP scope |
| Plugin | 🟡 Planned | MVP scope |
| PluginState | 🟡 Planned | MVP scope |
| TurnState | 🟢 Partial | Exists in oracle_agent |
| HistoryState | 🟢 Partial | Exists in oracle_agent |
| UserState | 🟢 Exists | user_settings.py |
| ProjectState | 🟢 Exists | project settings |

Legend: 🟢 Exists | 🟡 Planned | 🔴 Not Started | ⚪ Stretch Goal
