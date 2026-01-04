# Built-in Rules

This directory contains TOML rule definitions for the Oracle Plugin System.

## Rule File Format

Each `.toml` file defines a single rule with the following structure:

```toml
[rule]
id = "rule-id"                    # Unique kebab-case identifier
name = "Human Readable Name"      # Display name
description = "What this rule does"
version = "1.0.0"                 # Semantic version
trigger = "on_turn_start"         # Hook point (see below)
priority = 100                    # Higher = fires earlier (1-1000)
enabled = true                    # Whether rule is active by default
core = false                      # If true, user cannot disable

[condition]
expression = "context.turn.token_usage > 0.8"  # simpleeval expression
# OR
# script = "scripts/my_script.lua"             # Lua script path

[action]
type = "notify_self"              # Action type (see below)
message = "Your notification message"
category = "warning"              # info, warning, error
priority = "normal"               # low, normal, high, critical
deliver_at = "turn_start"         # Injection point
```

## Available Triggers (Hook Points)

| Trigger | When Fired |
|---------|-----------|
| `on_query_start` | New user query received |
| `on_turn_start` | Before agent processes each turn |
| `on_turn_end` | After agent completes each turn |
| `on_tool_call` | Before a tool is executed |
| `on_tool_complete` | After a tool returns successfully |
| `on_tool_failure` | When a tool fails or times out |
| `on_session_end` | Session closing |

## Available Action Types

| Type | Description |
|------|-------------|
| `notify_self` | Inject notification into agent context |
| `log` | Write to system log |
| `set_state` | Store plugin-scoped persistent state |
| `emit_event` | Emit an ANS event |

## Context API (for expressions)

Access these fields in condition expressions:

- `context.turn.number` - Current turn number
- `context.turn.token_usage` - Token budget usage (0.0-1.0)
- `context.turn.iteration_count` - Current iteration
- `context.history.tools` - Recent tool calls list
- `context.history.failures` - Dict of tool name to failure count
- `context.user.id` - User ID
- `context.project.id` - Project ID
- `context.state.get(key)` - Plugin-scoped state

## Expression Language

Conditions use simpleeval with support for:
- Comparisons: `>`, `<`, `>=`, `<=`, `==`, `!=`
- Boolean: `and`, `or`, `not`
- Field access: `context.turn.token_usage`
- List ops: `any()`, `all()`, `len()`
- Arithmetic: `+`, `-`, `*`, `/`

## Built-in Rules

- `token_budget.toml` - Warn at 80% token usage
- `iteration_budget.toml` - Warn at 70% iteration usage
- `large_result.toml` - Suggest summarization for 6+ results
- `repeated_failure.toml` - Alert on 3+ failures of same tool
