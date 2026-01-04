# Oracle Plugin System

The Oracle Plugin System enables reactive and proactive agent behaviors through a rule engine built on the Agent Notification System (ANS). Rules define conditional actions that trigger on specific agent lifecycle events.

## Quick Overview

- **TOML Rules**: Simple threshold and conditional rules defined in TOML with `simpleeval` expressions
- **Lua Scripts**: Complex logic escape hatch via sandboxed Lua execution
- **Hook Points**: Integration with ANS EventBus for agent lifecycle events
- **Persistent State**: SQLite-backed per-plugin state storage

## Getting Started

### Create Your First Rule

Create a TOML file in `backend/src/services/plugins/rules/`:

```toml
# my-first-rule.toml
[rule]
id = "my-first-rule"
name = "My First Rule"
description = "Notify when token usage is high"
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

Rules are auto-discovered on backend startup.

### Test Your Rule

```bash
curl -X POST http://localhost:8000/api/rules/my-first-rule/test \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context_override": {"turn": {"token_usage": 0.85}}}'
```

## Documentation Structure

| Section | Description |
|---------|-------------|
| [Architecture](./architecture/overview.md) | System design and component interactions |
| [Rules](./rules/format.md) | TOML rule format and schema |
| [Conditions](./rules/conditions.md) | Expression language guide |
| [Actions](./rules/actions.md) | Available action types |
| [Context API](./context-api/reference.md) | RuleContext API reference |
| [Hooks](./hooks/lifecycle.md) | Hook points and when they fire |
| [Scripting](./scripting/lua-guide.md) | Lua scripting guide |
| [Built-ins](./built-ins/) | Documentation for built-in rules |

## Built-in Rules

| Rule | Trigger | Description | Core |
|------|---------|-------------|------|
| [token-budget-warning](./built-ins/token-budget.md) | `on_turn_start` | Warn when token usage > 80% | Yes |
| [iteration-budget-warning](./built-ins/iteration-budget.md) | `on_turn_start` | Warn when iterations > 70% of max | Yes |
| [large-result-hint](./built-ins/large-result.md) | `on_tool_complete` | Suggest summarization for large results | No |
| [repeated-failure-warning](./built-ins/repeated-failure.md) | `on_tool_failure` | Alert on repeated tool failures | No |

## Key Concepts

### Hook Points

Rules trigger on specific lifecycle events:

```
Query Start -> Turn Start -> Tool Call -> Tool Complete/Failure -> Turn End -> Session End
```

### Rule Priority

Rules are evaluated in priority order (highest first). Use priority to control evaluation order when multiple rules may fire.

### Core Rules

Rules marked `core = true` cannot be disabled by users. These are essential for agent safety (budget limits, loop detection).

## Performance Targets

- Rule evaluation: <50ms per rule
- Condition parsing: <1ms (simpleeval)
- Lua script execution: <5s timeout

## Related Documentation

- [Quickstart Guide](../../specs/015-oracle-plugin-system/quickstart.md)
- [Feature Specification](../../specs/015-oracle-plugin-system/spec.md)
- [Research Notes](../../specs/015-oracle-plugin-system/research.md)
