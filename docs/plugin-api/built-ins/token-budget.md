# Token Budget Warning Rule

**Rule ID**: `token-budget-warning`
**File**: `backend/src/services/plugins/rules/token_budget.toml`
**Core**: Yes (cannot be disabled)

## Purpose

Notifies the agent when token usage exceeds 80% of the available budget, prompting consideration of wrapping up or summarizing.

## Configuration

```toml
[rule]
id = "token-budget-warning"
name = "Token Budget Warning"
description = "Warn when token usage exceeds 80% of budget"
version = "1.0.0"
trigger = "on_turn_start"
priority = 100
enabled = true
core = true

[condition]
expression = "context.turn.token_usage > 0.8"

[action]
type = "notify_self"
message = "Token budget at {{ (context.turn.token_usage * 100) | int }}%. Consider wrapping up or summarizing."
category = "warning"
priority = "high"
deliver_at = "turn_start"
```

## Behavior

### When It Fires

The rule evaluates at the start of each turn (`on_turn_start`). It fires when:
- `context.turn.token_usage` exceeds 0.8 (80%)

### Notification

When triggered, the agent receives a high-priority warning message:

```
Token budget at 85%. Consider wrapping up or summarizing.
```

The message is injected at `turn_start`, ensuring the agent sees it before processing the turn.

## Context Used

| Field | Type | Description |
|-------|------|-------------|
| `context.turn.token_usage` | float | Token budget usage ratio (0.0-1.0) |

## Why Core?

This rule is marked as `core = true` because:
1. **Agent Safety**: Prevents agents from running out of context mid-task
2. **Cost Management**: Token usage directly impacts API costs
3. **User Experience**: Allows graceful completion rather than abrupt termination

Core rules cannot be disabled by users to ensure these safety guarantees.

## Customization

While this core rule cannot be disabled, you can create additional token-related rules:

### Critical Threshold (95%)

```toml
[rule]
id = "token-critical"
name = "Token Critical Warning"
trigger = "on_turn_start"
priority = 200

[condition]
expression = "context.turn.token_usage > 0.95"

[action]
type = "notify_self"
message = "CRITICAL: Token budget nearly exhausted. Complete current task immediately."
category = "error"
priority = "critical"
deliver_at = "immediate"
```

### Early Warning (70%)

```toml
[rule]
id = "token-early-warning"
name = "Token Early Warning"
trigger = "on_turn_start"
priority = 50

[condition]
expression = "context.turn.token_usage > 0.7 and context.turn.token_usage <= 0.8"

[action]
type = "notify_self"
message = "Token usage at {{ (context.turn.token_usage * 100) | int }}%. Plan for completion."
category = "info"
priority = "normal"
```

## Related Rules

- [iteration-budget-warning](./iteration-budget.md) - Iteration limit warnings
- [large-result-hint](./large-result.md) - Suggests summarization for large results

## See Also

- [Hook Points](../hooks/lifecycle.md#on_turn_start)
- [Context API](../context-api/reference.md#turnstate)
- [Action Types](../rules/actions.md#action-type-notify_self)
