# Quickstart: Oracle Plugin System

**Date**: 2026-01-04
**Branch**: `015-oracle-plugin-system`

## Overview

The Oracle Plugin System enables you to extend the Oracle agent with custom rules that react to agent lifecycle events. Rules can:

- **Notify** the agent about conditions (token budget, repeated failures)
- **Log** debugging information
- **Store state** across turns
- **Emit events** for other subscribers

---

## Creating Your First Rule

### 1. Create a Rule File

Rules are TOML files in `backend/src/services/plugins/rules/`:

```toml
# backend/src/services/plugins/rules/my-first-rule.toml

[rule]
id = "my-first-rule"
name = "My First Rule"
description = "Demonstrate the plugin system"
trigger = "on_turn_start"
priority = 100

[condition]
expression = "context.turn.number >= 3"

[action]
type = "notify_self"
message = "You're on turn {{ context.turn.number }}. Remember to check your progress!"
category = "reminder"
priority = "normal"
```

### 2. Restart the Backend

The rule is auto-discovered on startup:

```bash
cd backend
uv run uvicorn src.api.main:app --reload
```

### 3. Test the Rule

Use the Settings UI or API to verify:

```bash
curl -X POST http://localhost:8000/api/rules/my-first-rule/test \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context_override": {"turn": {"number": 5}}}'
```

---

## Rule Triggers (Hook Points)

| Trigger | When it fires | Common use cases |
|---------|---------------|------------------|
| `on_query_start` | New user query received | Initialize state, log start |
| `on_turn_start` | Before each agent turn | Budget warnings, reminders |
| `on_turn_end` | After each turn completes | Progress checkpoints |
| `on_tool_call` | Before tool execution | Validate tool usage |
| `on_tool_complete` | After tool returns | Process results |
| `on_tool_failure` | When tool fails/times out | Error handling |
| `on_session_end` | Session closing | Cleanup, save state |

---

## Condition Expressions

### Simple Comparisons

```toml
[condition]
expression = "context.turn.token_usage > 0.8"
```

### Boolean Composition

```toml
[condition]
expression = "context.turn.token_usage > 0.8 and context.history.tool_count > 5"
```

### Built-in Functions

```toml
[condition]
# Check if specific tool completed
expression = "tool_completed('vault_search')"

# Check context usage threshold
expression = "context_above_threshold(0.9)"

# Check message count
expression = "message_count_above(20)"
```

### Available Context Fields

| Field | Type | Description |
|-------|------|-------------|
| `context.turn.number` | int | Current turn number |
| `context.turn.token_usage` | float | Token budget (0.0-1.0) |
| `context.turn.context_usage` | float | Context window (0.0-1.0) |
| `context.history.tool_count` | int | Tools called this session |
| `context.history.failure_count` | int | Failed tool calls |
| `context.user.id` | str | Current user ID |
| `context.project.id` | str | Current project ID |

---

## Action Types

### notify_self

Inject a notification into the agent's context:

```toml
[action]
type = "notify_self"
message = "Token budget at {{ (context.turn.token_usage * 100) | int }}%"
category = "warning"
priority = "high"
deliver_at = "turn_start"
```

**Priority levels**: `low`, `normal`, `high`, `critical`
**Delivery points**: `immediate`, `turn_start`, `after_tool`, `turn_end`

### log

Write to the system log:

```toml
[action]
type = "log"
message = "Rule fired at turn {{ context.turn.number }}"
level = "info"
```

### set_state

Store persistent state:

```toml
[action]
type = "set_state"
key = "last_warning_turn"
value = "{{ context.turn.number }}"
```

### emit_event

Emit an ANS event:

```toml
[action]
type = "emit_event"
event_type = "custom.rule.fired"
payload = { rule_id = "my-rule", turn = "{{ context.turn.number }}" }
```

---

## Lua Scripts (Advanced)

For complex logic, use Lua scripts instead of expressions:

```toml
# rules/complex-research.toml
[rule]
id = "complex-research"
name = "Research Workflow"
trigger = "on_tool_complete"

[condition]
script = "scripts/check_research.lua"

[action]
type = "notify_self"
message = "Consider synthesizing your findings."
```

```lua
-- scripts/check_research.lua
local vault_searches = 0
local web_searches = 0

for _, tool in ipairs(context.history.tools) do
    if tool.name == "vault_search" then
        vault_searches = vault_searches + 1
    elseif tool.name == "web_search" then
        web_searches = web_searches + 1
    end
end

-- Fire when we have enough research
return vault_searches >= 3 and web_searches >= 2
```

---

## Built-in Rules

The system ships with these rules (all can be disabled except core):

| Rule | Trigger | Condition | Core |
|------|---------|-----------|------|
| `token-budget-warning` | on_turn_start | token_usage > 0.8 | Yes |
| `iteration-budget-warning` | on_turn_start | iteration > 0.7 * max | Yes |
| `large-result-hint` | on_tool_complete | result_count > 6 | No |
| `repeated-failure-warning` | on_tool_failure | same tool failed 3+ times | No |

---

## Managing Rules via API

### List All Rules

```bash
curl http://localhost:8000/api/rules \
  -H "Authorization: Bearer $TOKEN"
```

### Toggle a Rule

```bash
curl -X POST http://localhost:8000/api/rules/large-result-hint/toggle \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

### Test a Rule

```bash
curl -X POST http://localhost:8000/api/rules/token-budget-warning/test \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context_override": {"turn": {"token_usage": 0.85}}}'
```

---

## Creating a Plugin

For multiple related rules, create a plugin:

```
plugins/
└── my-plugin/
    ├── manifest.toml
    ├── rules/
    │   ├── rule-one.toml
    │   └── rule-two.toml
    └── scripts/
        └── helper.lua
```

```toml
# manifest.toml
[plugin]
id = "my-plugin"
name = "My Plugin"
version = "1.0.0"
description = "A collection of custom rules"

[capabilities]
requires = ["vault_search"]

[rules]
include = ["rules/*.toml"]

[settings.threshold]
type = "float"
default = 0.8
min = 0.0
max = 1.0
description = "Threshold for triggering warnings"
```

---

## Troubleshooting

### Rule Not Firing

1. Check if rule is enabled: `GET /api/rules/{id}`
2. Verify trigger matches the event
3. Test condition with override: `POST /api/rules/{id}/test`
4. Check logs for evaluation errors

### Script Timeout

Scripts have a 5-second execution limit. If exceeded:
- Simplify logic
- Move expensive operations to Python
- Use rule state to cache intermediate results

### Expression Errors

```
Invalid syntax: Expected '(' after function name
```

Check that function calls have parentheses: `context_above_threshold(0.8)`

---

## Next Steps

- Read the [Expression Language Reference](../docs/plugin-api/rules/conditions.md)
- Explore [Context API](../docs/plugin-api/context-api/reference.md)
- See [Lua Scripting Guide](../docs/plugin-api/scripting/lua-guide.md)
- Browse [Built-in Rules](../docs/plugin-api/built-ins/)
