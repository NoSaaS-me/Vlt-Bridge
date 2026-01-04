# Rule Examples

Common patterns and examples for creating effective rules.

## Budget Management

### Token Budget Warning

```toml
[rule]
id = "token-budget-warning"
name = "Token Budget Warning"
description = "Warn when token usage exceeds 80%"
trigger = "on_turn_start"
priority = 100
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

### Critical Budget Alert

```toml
[rule]
id = "budget-critical"
name = "Critical Budget Alert"
trigger = "on_turn_start"
priority = 200

[condition]
expression = "context.turn.token_usage > 0.95 or context.turn.context_usage > 0.95"

[action]
type = "notify_self"
message = "CRITICAL: Budget nearly exhausted. Finish current task immediately."
category = "error"
priority = "critical"
deliver_at = "immediate"
```

### Iteration Limit Warning

```toml
[rule]
id = "iteration-limit"
name = "Iteration Limit Warning"
trigger = "on_turn_start"
priority = 90

[condition]
expression = "context.turn.iteration_count > 7"

[action]
type = "notify_self"
message = "Iteration {{ context.turn.iteration_count }} of 10. Consider breaking down the task."
category = "warning"
priority = "high"
```

## Tool Monitoring

### Large Result Hint

```toml
[rule]
id = "large-result-hint"
name = "Large Result Hint"
trigger = "on_tool_complete"
priority = 50

[condition]
expression = "context.result is not None and context.result.success and len(context.result.result or '') > 2000"

[action]
type = "notify_self"
message = "Large result from {{ context.result.tool_name }}. Consider summarizing key findings."
category = "info"
priority = "normal"
deliver_at = "after_tool"
```

### Tool Success Tracker

```toml
[rule]
id = "track-tool-success"
name = "Track Tool Success"
trigger = "on_tool_complete"
priority = 10

[condition]
expression = "context.result is not None and context.result.success"

[action]
type = "set_state"
key = "last_successful_tool"
value = "{{ context.result.tool_name }}"
```

### Repeated Failure Detection

```toml
[rule]
id = "repeated-failure"
name = "Repeated Failure Warning"
trigger = "on_tool_failure"
priority = 80

[condition]
expression = "context.history.total_failures >= 3"

[action]
type = "notify_self"
message = "Multiple tool failures ({{ context.history.total_failures }} total). Try a different approach."
category = "warning"
priority = "high"
deliver_at = "immediate"
```

### Specific Tool Failure

```toml
[rule]
id = "api-call-failed"
name = "API Call Failed"
trigger = "on_tool_failure"
priority = 70

[condition]
expression = "context.event is not None and 'api' in (context.event.payload.get('tool_name', '') or '')"

[action]
type = "notify_self"
message = "API call failed. Check network connectivity or try alternative endpoints."
category = "warning"
```

## Research Workflow

### Research Phase Complete

```toml
[rule]
id = "research-complete"
name = "Research Phase Complete"
trigger = "on_tool_complete"
priority = 60

[condition]
expression = "tool_completed('vault_search') and tool_completed('web_search') and context.history.total_tool_calls >= 5"

[action]
type = "notify_self"
message = "Research phase appears complete. Consider synthesizing findings."
category = "info"
```

### Search Without Results

```toml
[rule]
id = "empty-search"
name = "Empty Search Result"
trigger = "on_tool_complete"
priority = 55

[condition]
expression = "context.result is not None and 'search' in context.result.tool_name and context.result.success and len(context.result.result or '') < 100"

[action]
type = "notify_self"
message = "Search returned minimal results. Consider broadening search terms."
category = "info"
deliver_at = "after_tool"
```

## Session Management

### Long Session Warning

```toml
[rule]
id = "long-session"
name = "Long Session Warning"
trigger = "on_turn_start"
priority = 40

[condition]
expression = "context.turn.number >= 20"

[action]
type = "notify_self"
message = "Session has been running for {{ context.turn.number }} turns. Consider checkpointing progress."
category = "info"
priority = "normal"
```

### Session Milestone

```toml
[rule]
id = "session-milestone"
name = "Session Milestone"
trigger = "on_turn_end"
priority = 30

[condition]
expression = "context.turn.number % 10 == 0"

[action]
type = "emit_event"
event_type = "custom.session.milestone"
payload = { turn = "{{ context.turn.number }}", user = "{{ context.user.id }}" }
```

## State-Based Rules

### Cooldown Pattern

```toml
[rule]
id = "warning-cooldown"
name = "Warning with Cooldown"
trigger = "on_turn_start"
priority = 100

[condition]
# Only warn if 5 turns have passed since last warning
expression = "context.turn.token_usage > 0.8 and (not context.state.has('last_warning') or context.turn.number - int(context.state.get('last_warning', 0)) >= 5)"

[action]
type = "set_state"
key = "last_warning"
value = "{{ context.turn.number }}"
```

Then add a companion rule for the notification:

```toml
[rule]
id = "warning-notify"
name = "Warning Notification"
trigger = "on_turn_start"
priority = 99

[condition]
expression = "context.turn.token_usage > 0.8 and (not context.state.has('last_warning') or context.turn.number - int(context.state.get('last_warning', 0)) >= 5)"

[action]
type = "notify_self"
message = "Token budget reminder (cooldown active for next 5 turns)"
category = "warning"
```

### Counter Pattern

```toml
[rule]
id = "search-counter"
name = "Count Searches"
trigger = "on_tool_complete"
priority = 20

[condition]
expression = "context.result is not None and 'search' in context.result.tool_name"

[action]
type = "set_state"
key = "search_count"
value = "{{ int(context.state.get('search_count', 0)) + 1 }}"
```

## Debugging Rules

### Log All Tool Calls

```toml
[rule]
id = "log-tools"
name = "Log All Tool Calls"
trigger = "on_tool_complete"
priority = 1
enabled = false  # Enable for debugging

[condition]
expression = "context.result is not None"

[action]
type = "log"
message = "Tool: {{ context.result.tool_name }} | Success: {{ context.result.success }} | Duration: {{ context.result.duration_ms }}ms"
level = "debug"
```

### Log Turn Summary

```toml
[rule]
id = "log-turn-summary"
name = "Log Turn Summary"
trigger = "on_turn_end"
priority = 1
enabled = false

[condition]
expression = "True"

[action]
type = "log"
message = "Turn {{ context.turn.number }} complete. Tools: {{ context.history.total_tool_calls }}, Failures: {{ context.history.total_failures }}, Token usage: {{ (context.turn.token_usage * 100) | int }}%"
level = "info"
```

## Lua Script Examples

### Complex Research Check

```toml
[rule]
id = "complex-research"
name = "Complex Research Check"
trigger = "on_tool_complete"
priority = 65

[condition]
script = "scripts/check_research.lua"

[action]
type = "notify_self"
message = "Sufficient research collected. Ready for synthesis."
```

`scripts/check_research.lua`:
```lua
-- Check if we have enough research data
local vault_searches = 0
local web_searches = 0

for _, tool in ipairs(context.history.tools) do
    if tool.name == "vault_search" and tool.success then
        vault_searches = vault_searches + 1
    elseif tool.name == "web_search" and tool.success then
        web_searches = web_searches + 1
    end
end

-- Require at least 3 vault and 2 web searches
return vault_searches >= 3 and web_searches >= 2
```

### Dynamic Action from Script

```toml
[rule]
id = "dynamic-notification"
name = "Dynamic Notification"
trigger = "on_turn_start"

[condition]
script = "scripts/dynamic_check.lua"

[action]
type = "notify_self"
message = "Default message"  # Overridden by script
```

`scripts/dynamic_check.lua`:
```lua
-- Return different actions based on conditions
if context.turn.token_usage > 0.9 then
    return {
        type = "notify_self",
        message = "URGENT: Token budget critical!",
        priority = "critical"
    }
elseif context.turn.token_usage > 0.8 then
    return {
        type = "notify_self",
        message = "Token budget getting low",
        priority = "high"
    }
else
    return nil  -- Don't fire
end
```

## See Also

- [TOML Format](./format.md)
- [Condition Expressions](./conditions.md)
- [Action Types](./actions.md)
- [Lua Scripting](../scripting/lua-guide.md)
