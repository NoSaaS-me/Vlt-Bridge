# Action Types

Actions define what happens when a rule condition matches. Four action types are supported.

## Action Type: `notify_self`

Injects a notification into the agent's context, making the agent aware of the condition.

### Configuration

```toml
[action]
type = "notify_self"
message = "Token usage at {{ (context.turn.token_usage * 100) | int }}%"
category = "warning"
priority = "high"
deliver_at = "turn_start"
```

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | Yes | - | Jinja2 template for notification text |
| `category` | string | No | `"info"` | Category for grouping |
| `priority` | string | No | `"normal"` | Priority level |
| `deliver_at` | string | No | `"turn_start"` | When to inject |

### Priority Levels

| Level | Use Case |
|-------|----------|
| `low` | Informational, can be ignored |
| `normal` | Standard notifications |
| `high` | Important, should be acknowledged |
| `critical` | Urgent, requires immediate attention |

### Delivery Points

| Point | When | Best For |
|-------|------|----------|
| `turn_start` | Before agent processes next turn | Budget warnings |
| `after_tool` | Immediately after tool execution | Tool result hints |
| `immediate` | As soon as event occurs | Critical alerts |

### Template Variables

The message supports Jinja2 templating with access to the full context:

```toml
message = """
Turn {{ context.turn.number }}:
Token usage: {{ (context.turn.token_usage * 100) | int }}%
Tools used: {{ context.history.total_tool_calls }}
"""
```

### Template Filters

| Filter | Description | Example |
|--------|-------------|---------|
| `int` | Convert to integer | `{{ value \| int }}` |
| `round(n)` | Round to n decimals | `{{ value \| round(2) }}` |
| `default(v)` | Default if undefined | `{{ x \| default('N/A') }}` |
| `length` | Get length | `{{ items \| length }}` |

## Action Type: `log`

Writes a message to the system log for debugging and auditing.

### Configuration

```toml
[action]
type = "log"
message = "Rule triggered at turn {{ context.turn.number }}"
level = "info"
```

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | No | `"Rule triggered"` | Log message |
| `level` | string | No | `"info"` | Log level |

### Log Levels

| Level | Use Case |
|-------|----------|
| `debug` | Development/troubleshooting |
| `info` | Normal operation events |
| `warning` | Potential issues |
| `error` | Problems that need attention |

### Example Output

```
2026-01-04 12:00:00 INFO [Rule] Token budget warning fired at turn 5
2026-01-04 12:00:01 WARNING [Rule] Multiple failures detected: 3
```

## Action Type: `set_state`

Stores a value in persistent plugin state for cross-turn and cross-session access.

### Configuration

```toml
[action]
type = "set_state"
key = "last_warning_turn"
value = "{{ context.turn.number }}"
```

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `key` | string | Yes | - | State key to set |
| `value` | any | No | `null` | Value to store |

### Value Types

```toml
# String (with template)
value = "{{ context.turn.number }}"

# Number
value = 42

# Boolean
value = true

# Object
value = { count = 5, timestamp = "{{ context.turn.number }}" }
```

### State Scope

State is scoped by:
- `user_id`: Per-user isolation
- `project_id`: Per-project isolation
- `plugin_id`: Per-plugin isolation (if rule belongs to a plugin)

### Accessing State

In conditions:
```toml
expression = "context.state.get('last_warning_turn', 0) < context.turn.number - 5"
```

In Lua scripts:
```lua
local last = context.state.last_warning_turn or 0
```

## Action Type: `emit_event`

Emits an ANS event that other subscribers can react to.

### Configuration

```toml
[action]
type = "emit_event"
event_type = "custom.milestone.reached"
payload = {
    milestone = "high_usage",
    turn = "{{ context.turn.number }}",
    usage = "{{ context.turn.token_usage }}"
}
```

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `event_type` | string | Yes | - | Event type identifier |
| `payload` | table | No | `{}` | Event payload data |

### Event Type Conventions

```
namespace.category.event
```

Examples:
- `custom.budget.warning`
- `custom.research.complete`
- `plugin.my-plugin.activated`

### Payload Templates

Payload values support Jinja2:

```toml
payload = {
    rule_id = "token-warning",
    user = "{{ context.user.id }}",
    project = "{{ context.project.id }}",
    details = { turn = "{{ context.turn.number }}" }
}
```

### Subscribing to Custom Events

Other rules or ANS subscribers can react to emitted events:

```toml
# Another rule
[rule]
id = "react-to-milestone"
trigger = "on_turn_start"  # Events arrive on next cycle

[condition]
# Check for recent custom event in event log
expression = "context.state.has('milestone_reached')"
```

## Multiple Actions

A rule can only have one action. For multiple effects, use `emit_event` to trigger additional rules:

```toml
# Primary rule
[rule]
id = "primary-action"
trigger = "on_turn_start"

[condition]
expression = "context.turn.token_usage > 0.9"

[action]
type = "emit_event"
event_type = "custom.critical_usage"
payload = { usage = "{{ context.turn.token_usage }}" }
```

Then create secondary rules that react to the custom event.

## Action Execution Order

1. Condition evaluated
2. If matched, action dispatched
3. For `notify_self`: notification queued for delivery
4. For `log`: written immediately
5. For `set_state`: persisted to database
6. For `emit_event`: emitted to EventBus

## Error Handling

Action failures are logged but do not stop rule evaluation:

```python
# In engine.py
try:
    success = self._dispatcher.dispatch(rule.action, context)
    if not success:
        logger.warning(f"Action failed for rule {rule.id}")
except Exception as e:
    logger.error(f"Action error for rule {rule.id}: {e}")
```

## See Also

- [TOML Format](./format.md)
- [Condition Expressions](./conditions.md)
- [Examples](./examples.md)
