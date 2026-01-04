# TOML Rule Format

Rules are defined in TOML files with three main sections: `[rule]`, `[condition]`, and `[action]`.

## Complete Schema

```toml
# Rule identity and metadata
[rule]
id = "kebab-case-id"              # Required: Unique identifier
name = "Human Readable Name"       # Optional: Display name (defaults to id)
description = "What this rule does" # Optional: Explanation
version = "1.0.0"                  # Optional: Semantic version (default: "1.0.0")
trigger = "on_turn_start"          # Required: Hook point
priority = 100                     # Optional: Evaluation order (1-1000, default: 100)
enabled = true                     # Optional: Active state (default: true)
core = false                       # Optional: Cannot be disabled (default: false)
plugin_id = "my-plugin"            # Optional: Parent plugin ID

# Condition for rule firing (one of expression or script)
[condition]
expression = "context.turn.token_usage > 0.8"  # simpleeval expression
# OR
script = "scripts/my-check.lua"                 # Path to Lua script

# Action to execute when condition matches
[action]
type = "notify_self"               # Required: Action type
message = "Template {{ context.turn.number }}"  # For notify_self
category = "warning"               # For notify_self (default: "info")
priority = "high"                  # For notify_self: low|normal|high|critical
deliver_at = "turn_start"          # For notify_self: turn_start|after_tool|immediate
level = "info"                     # For log: debug|info|warning|error
key = "state_key"                  # For set_state
value = "state_value"              # For set_state
event_type = "custom.event"        # For emit_event
payload = { key = "value" }        # For emit_event
```

## Field Reference

### [rule] Section

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | string | Yes | - | Unique kebab-case identifier |
| `name` | string | No | `id` | Human-readable name |
| `description` | string | No | `""` | What the rule does |
| `version` | string | No | `"1.0.0"` | Semantic version |
| `trigger` | string | Yes | - | Hook point to trigger on |
| `priority` | int | No | `100` | Evaluation order (1-1000, higher first) |
| `enabled` | bool | No | `true` | Whether rule is active |
| `core` | bool | No | `false` | If true, cannot be disabled |
| `plugin_id` | string | No | `null` | Parent plugin identifier |

### Trigger Values

| Trigger | Description |
|---------|-------------|
| `on_query_start` | New user query received |
| `on_turn_start` | Before agent processes turn |
| `on_turn_end` | After agent completes turn |
| `on_tool_call` | Before tool execution |
| `on_tool_complete` | After tool returns successfully |
| `on_tool_failure` | When tool fails or times out |
| `on_session_end` | Session closing |

### [condition] Section

One of `expression` or `script` is required (mutually exclusive).

| Field | Type | Description |
|-------|------|-------------|
| `expression` | string | simpleeval expression returning bool |
| `script` | string | Path to Lua script (relative to rule file) |

### [action] Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Action type: `notify_self`, `log`, `set_state`, `emit_event` |

#### Action Type: `notify_self`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | Required | Jinja2 template for notification |
| `category` | string | `"info"` | Category: `info`, `warning`, `error`, `reminder` |
| `priority` | string | `"normal"` | Priority: `low`, `normal`, `high`, `critical` |
| `deliver_at` | string | `"turn_start"` | When to inject: `turn_start`, `after_tool`, `immediate` |

#### Action Type: `log`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | `"Rule triggered"` | Log message (supports Jinja2) |
| `level` | string | `"info"` | Log level: `debug`, `info`, `warning`, `error` |

#### Action Type: `set_state`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `key` | string | Yes | State key to set |
| `value` | any | No | Value to store (supports Jinja2 for strings) |

#### Action Type: `emit_event`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event_type` | string | Yes | Event type to emit |
| `payload` | table | No | Event payload (supports Jinja2 for string values) |

## Examples

### Simple Threshold Rule

```toml
[rule]
id = "token-warning"
name = "Token Usage Warning"
trigger = "on_turn_start"
priority = 100

[condition]
expression = "context.turn.token_usage > 0.8"

[action]
type = "notify_self"
message = "Token usage at {{ (context.turn.token_usage * 100) | int }}%"
category = "warning"
priority = "high"
```

### Rule with Boolean Composition

```toml
[rule]
id = "complex-check"
trigger = "on_turn_start"

[condition]
expression = "context.turn.token_usage > 0.7 and context.history.total_tool_calls > 5"

[action]
type = "notify_self"
message = "High activity detected"
```

### Rule with Lua Script

```toml
[rule]
id = "research-complete"
trigger = "on_tool_complete"

[condition]
script = "scripts/check_research.lua"

[action]
type = "notify_self"
message = "Research phase complete. Consider synthesis."
```

### State-Setting Rule

```toml
[rule]
id = "track-searches"
trigger = "on_tool_complete"

[condition]
expression = "context.result is not None and context.result.tool_name == 'vault_search'"

[action]
type = "set_state"
key = "last_search_turn"
value = "{{ context.turn.number }}"
```

### Event-Emitting Rule

```toml
[rule]
id = "milestone-reached"
trigger = "on_turn_end"

[condition]
expression = "context.turn.number == 10"

[action]
type = "emit_event"
event_type = "custom.milestone"
payload = { milestone = "turn_10", user = "{{ context.user.id }}" }
```

## Validation Rules

1. **ID format**: Must be kebab-case (`[a-z0-9-]+`)
2. **Condition XOR script**: Must have exactly one
3. **Priority range**: Must be 1-1000
4. **Action required**: Must have an `[action]` section
5. **Action-specific fields**: Must include required fields for action type

## File Naming Convention

- Use kebab-case: `my-rule-name.toml`
- Match rule ID: `token-budget.toml` for `id = "token-budget"`
- Place in `backend/src/services/plugins/rules/` for auto-discovery

## See Also

- [Condition Expressions](./conditions.md)
- [Action Types](./actions.md)
- [Examples](./examples.md)
