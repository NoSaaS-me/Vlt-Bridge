# Iteration Budget Warning Rule

**Rule ID**: `iteration-budget-warning`
**File**: `backend/src/services/plugins/rules/iteration_budget.toml`
**Core**: Yes (cannot be disabled)

## Purpose

Notifies the agent when the iteration count exceeds 70% of the maximum allowed iterations (default: 10), prompting task completion or breakdown.

## Configuration

```toml
[rule]
id = "iteration-budget-warning"
name = "Iteration Budget Warning"
description = "Warn when iteration count exceeds 70% of maximum allowed"
version = "1.0.0"
trigger = "on_turn_start"
priority = 90
enabled = true
core = true

[condition]
# Check if iteration_count / max_iterations > 0.7
# Using a reasonable default of 10 iterations max if not configured
expression = "context.turn.iteration_count > 7"

[action]
type = "notify_self"
message = "Iteration {{ context.turn.iteration_count }} of 10. Consider completing current task or breaking it down."
category = "warning"
priority = "high"
deliver_at = "turn_start"
```

## Behavior

### When It Fires

The rule evaluates at the start of each turn (`on_turn_start`). It fires when:
- `context.turn.iteration_count` exceeds 7 (70% of default max 10)

### Notification

When triggered, the agent receives a high-priority warning message:

```
Iteration 8 of 10. Consider completing current task or breaking it down.
```

The message is injected at `turn_start`, ensuring the agent sees it before processing.

## Context Used

| Field | Type | Description |
|-------|------|-------------|
| `context.turn.iteration_count` | int | Current iteration within the turn |

## Why Core?

This rule is marked as `core = true` because:
1. **Loop Prevention**: Prevents agents from getting stuck in infinite loops
2. **Task Management**: Encourages breaking down complex tasks
3. **Resource Control**: Ensures bounded computation per query

Core rules cannot be disabled by users to ensure these safety guarantees.

## Customization

While this core rule cannot be disabled, you can create additional iteration-related rules:

### Critical Iteration Limit

```toml
[rule]
id = "iteration-critical"
name = "Iteration Critical"
trigger = "on_turn_start"
priority = 200

[condition]
expression = "context.turn.iteration_count >= 9"

[action]
type = "notify_self"
message = "FINAL ITERATION: Complete task now or summarize progress."
category = "error"
priority = "critical"
deliver_at = "immediate"
```

### Progress Checkpoint

```toml
[rule]
id = "iteration-checkpoint"
name = "Iteration Checkpoint"
trigger = "on_turn_start"
priority = 40

[condition]
expression = "context.turn.iteration_count % 5 == 0 and context.turn.iteration_count > 0"

[action]
type = "notify_self"
message = "Checkpoint: {{ context.turn.iteration_count }} iterations. Progress check recommended."
category = "info"
priority = "normal"
```

### Track Iteration History

```toml
[rule]
id = "track-iterations"
name = "Track Iteration Count"
trigger = "on_turn_end"
priority = 10

[condition]
expression = "True"

[action]
type = "set_state"
key = "last_iteration_count"
value = "{{ context.turn.iteration_count }}"
```

## Understanding Iterations vs Turns

| Concept | Description |
|---------|-------------|
| **Turn** | A complete user query -> agent response cycle |
| **Iteration** | A single processing step within a turn |

An agent turn may involve multiple iterations:
1. Parse query
2. Call tool
3. Process result
4. Call another tool
5. Generate response

This rule monitors iterations to prevent runaway processing.

## Related Rules

- [token-budget-warning](./token-budget.md) - Token usage warnings
- [repeated-failure-warning](./repeated-failure.md) - Repeated failure detection

## See Also

- [Hook Points](../hooks/lifecycle.md#on_turn_start)
- [Context API](../context-api/reference.md#turnstate)
- [Action Types](../rules/actions.md#action-type-notify_self)
