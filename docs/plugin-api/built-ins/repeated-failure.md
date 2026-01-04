# Repeated Failure Warning Rule

**Rule ID**: `repeated-failure-warning`
**File**: `backend/src/services/plugins/rules/repeated_failure.toml`
**Core**: No (can be disabled)

## Purpose

Alerts the agent when multiple tool failures have occurred in the session, prompting consideration of alternative approaches.

## Configuration

```toml
[rule]
id = "repeated-failure-warning"
name = "Repeated Failure Warning"
description = "Alert when a tool has failed 3 or more times in this session"
version = "1.0.0"
trigger = "on_tool_failure"
priority = 80
enabled = true
core = false

[condition]
# Check if the event contains tool info and has failed multiple times
# context.event.payload should contain the tool name for failure events
expression = "context.event is not None and context.history.total_failures >= 3"

[action]
type = "notify_self"
message = "Multiple tool failures detected ({{ context.history.total_failures }} total). Consider trying a different approach or checking tool availability."
category = "warning"
priority = "high"
deliver_at = "immediate"
```

## Behavior

### When It Fires

The rule evaluates after each tool failure (`on_tool_failure`). It fires when:
- `context.event` exists (failure event data)
- `context.history.total_failures` is >= 3

### Notification

When triggered, the agent receives a high-priority warning:

```
Multiple tool failures detected (3 total). Consider trying a different approach or checking tool availability.
```

The message is delivered `immediate`ly to ensure prompt attention to the failure pattern.

## Context Used

| Field | Type | Description |
|-------|------|-------------|
| `context.event` | EventData | Failure event data |
| `context.history.total_failures` | int | Sum of all tool failures |
| `context.history.failures` | dict | Per-tool failure counts |

## Why Not Core?

This rule is marked as `core = false` because:
1. **Flexibility**: Some workflows expect transient failures
2. **Retry Logic**: Users may have custom retry handling
3. **Tool-Specific**: Failure significance varies by tool

Users can disable this rule in environments where failures are expected (e.g., testing).

## Customization

### Disable the Rule

```bash
curl -X POST http://localhost:8000/api/rules/repeated-failure-warning/toggle \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

### Per-Tool Failure Tracking

```toml
[rule]
id = "specific-tool-failure"
name = "API Tool Failure Alert"
trigger = "on_tool_failure"
priority = 85

[condition]
expression = "context.event is not None and failure_count('api_call') >= 2"

[action]
type = "notify_self"
message = "API call has failed {{ failure_count('api_call') }} times. Check endpoint availability."
category = "warning"
priority = "high"
deliver_at = "immediate"
```

### Lower Threshold

```toml
[rule]
id = "early-failure-warning"
name = "Early Failure Warning"
trigger = "on_tool_failure"
priority = 75

[condition]
expression = "context.event is not None and context.history.total_failures >= 2"

[action]
type = "notify_self"
message = "{{ context.history.total_failures }} failures detected. Consider alternative approaches."
category = "info"
priority = "normal"
```

### Critical Failure Threshold

```toml
[rule]
id = "critical-failures"
name = "Critical Failure Alert"
trigger = "on_tool_failure"
priority = 200

[condition]
expression = "context.event is not None and context.history.total_failures >= 5"

[action]
type = "notify_self"
message = "CRITICAL: {{ context.history.total_failures }} failures. Stop and reassess approach."
category = "error"
priority = "critical"
deliver_at = "immediate"
```

### Using Lua for Complex Logic

```toml
[rule]
id = "failure-pattern-detection"
name = "Failure Pattern Detection"
trigger = "on_tool_failure"
priority = 70

[condition]
script = "scripts/check_failure_pattern.lua"

[action]
type = "notify_self"
message = "Failure pattern detected"
```

`scripts/check_failure_pattern.lua`:
```lua
-- Check for specific failure patterns
local search_failures = context.history.failures["vault_search"] or 0
local web_failures = context.history.failures["web_search"] or 0

-- Both search tools failing suggests connectivity issues
if search_failures >= 2 and web_failures >= 2 then
    return {
        type = "notify_self",
        message = "Multiple search tools failing. Check network connectivity.",
        priority = "high"
    }
end

-- Single tool repeated failures
for tool, count in pairs(context.history.failures) do
    if count >= 3 then
        return {
            type = "notify_self",
            message = string.format(
                "'%s' has failed %d times. Try an alternative approach.",
                tool, count
            ),
            priority = "high"
        }
    end
end

return nil
```

## Use Cases

### When This Rule Helps

1. **Network Issues**: API endpoints temporarily unavailable
2. **Rate Limiting**: Too many requests to external services
3. **Invalid Queries**: Malformed tool arguments
4. **Resource Limits**: Timeouts on heavy operations

### Example Scenario

```
Turn 1: vault_search fails (timeout)
Turn 2: vault_search fails (timeout)
Turn 3: vault_search fails (timeout)
Rule fires: "Multiple tool failures detected (3 total)..."
Agent: Switches to alternative approach or asks user for help
```

## Related Rules

- [token-budget-warning](./token-budget.md) - Failures may waste token budget
- [iteration-budget-warning](./iteration-budget.md) - Repeated retries consume iterations

## See Also

- [Hook Points](../hooks/lifecycle.md#on_tool_failure)
- [Context API](../context-api/reference.md#historystate)
- [Action Types](../rules/actions.md#action-type-notify_self)
