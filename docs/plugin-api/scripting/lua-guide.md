# Lua Scripting Guide

For complex conditions that exceed expression capabilities, use Lua scripts.

## Overview

Lua scripts provide:
- Full programming language features (loops, conditionals, functions)
- Complex data manipulation
- Dynamic action generation
- Access to full context API

Scripts run in a sandboxed environment via `lupa` (LuaJIT).

## Basic Script Structure

```lua
-- Access context (read-only)
local usage = context.turn.token_usage
local tools = context.history.tools

-- Perform logic
local count = 0
for _, tool in ipairs(tools) do
    if tool.name == "vault_search" then
        count = count + 1
    end
end

-- Return result
if count >= 3 then
    return true  -- Use rule's defined action
else
    return nil   -- Don't fire rule
end
```

## Return Values

| Return | Behavior |
|--------|----------|
| `nil` | Rule does not match, no action |
| `true` | Rule matches, execute rule's action |
| `false` | Rule does not match, no action |
| `{type="..."}` | Rule matches, use returned action |

### Returning Custom Actions

```lua
-- Return a custom notification
return {
    type = "notify_self",
    message = "Custom message based on logic",
    priority = "high",
    category = "warning"
}
```

### Dynamic Action Selection

```lua
if context.turn.token_usage > 0.9 then
    return {
        type = "notify_self",
        message = "CRITICAL: Token budget nearly exhausted!",
        priority = "critical"
    }
elseif context.turn.token_usage > 0.8 then
    return {
        type = "notify_self",
        message = "Warning: Token budget getting low",
        priority = "high"
    }
else
    return nil  -- No notification needed
end
```

## Accessing Context

### Turn State

```lua
local turn_number = context.turn.number
local token_usage = context.turn.token_usage
local context_usage = context.turn.context_usage
local iterations = context.turn.iteration_count
```

### History

```lua
-- Messages
for _, msg in ipairs(context.history.messages) do
    print(msg.role, msg.content)
end

-- Tool calls
for _, tool in ipairs(context.history.tools) do
    print(tool.name, tool.success, tool.result)
end

-- Failures
local api_failures = context.history.failures["api_call"] or 0

-- Computed properties
local total_tools = context.history.total_tool_calls
local total_fails = context.history.total_failures
```

### User and Project

```lua
local user_id = context.user.id
local user_settings = context.user.settings

local project_id = context.project.id
local project_settings = context.project.settings
```

### Plugin State

```lua
-- Read state (state is a table)
local counter = context.state.counter or 0
local last_turn = context.state.last_warning_turn or 0

-- Check all state keys
for key, value in pairs(context.state) do
    print(key, value)
end
```

### Event Data

```lua
if context.event then
    local event_type = context.event.type
    local source = context.event.source
    local severity = context.event.severity

    -- Access payload
    for key, value in pairs(context.event.payload) do
        print(key, value)
    end
end
```

### Tool Result

```lua
if context.result then
    local tool_name = context.result.tool_name
    local success = context.result.success
    local result = context.result.result
    local error = context.result.error
    local duration = context.result.duration_ms
end
```

## Available Functions

### String Functions

```lua
string.byte(s)      -- Get byte value
string.char(...)    -- Create string from bytes
string.find(s, p)   -- Find pattern
string.format(...)  -- Format string
string.gmatch(s, p) -- Pattern iterator
string.gsub(s, p, r)-- Replace pattern
string.len(s)       -- String length
string.lower(s)     -- Lowercase
string.match(s, p)  -- Pattern match
string.rep(s, n)    -- Repeat string
string.reverse(s)   -- Reverse string
string.sub(s, i, j) -- Substring
string.upper(s)     -- Uppercase
```

### Table Functions

```lua
table.concat(t, sep) -- Join array
table.insert(t, v)   -- Insert element
table.maxn(t)        -- Max numeric key
table.remove(t, i)   -- Remove element
table.sort(t, comp)  -- Sort array
```

### Math Functions

```lua
math.abs(x)         -- Absolute value
math.ceil(x)        -- Round up
math.floor(x)       -- Round down
math.max(...)       -- Maximum
math.min(...)       -- Minimum
math.random(m, n)   -- Random number
math.sqrt(x)        -- Square root
-- Plus trigonometric functions
```

### Basic Functions

```lua
print(...)          -- Output (logged)
type(x)             -- Get type
tostring(x)         -- Convert to string
tonumber(x)         -- Convert to number
pairs(t)            -- Iterate table
ipairs(t)           -- Iterate array
next(t, k)          -- Next key-value
select(i, ...)      -- Select from varargs
unpack(t)           -- Unpack array
pcall(f, ...)       -- Protected call
error(msg)          -- Raise error
assert(cond, msg)   -- Assert condition
```

## Blocked Functions

For security, these are NOT available:

- `os.*` - Operating system access
- `io.*` - File I/O
- `debug.*` - Debug facilities
- `dofile`, `loadfile`, `load`, `loadstring` - Code loading
- `require`, `module`, `package` - Module system
- `rawget`, `rawset`, `rawequal` - Raw table access
- `setmetatable`, `getmetatable` - Metatable access
- `collectgarbage` - GC control
- `coroutine.*` - Coroutines

## Examples

### Count Specific Tool Usage

```lua
-- scripts/count_searches.lua
local vault_count = 0
local web_count = 0

for _, tool in ipairs(context.history.tools) do
    if tool.name == "vault_search" and tool.success then
        vault_count = vault_count + 1
    elseif tool.name == "web_search" and tool.success then
        web_count = web_count + 1
    end
end

-- Fire when enough research is done
return vault_count >= 3 and web_count >= 2
```

### Analyze Result Patterns

```lua
-- scripts/check_large_result.lua
if not context.result or not context.result.success then
    return nil
end

local result = context.result.result or ""
local length = string.len(result)

-- Check for patterns indicating large result
if length > 2000 then
    return {
        type = "notify_self",
        message = string.format(
            "Large result (%d chars) from %s. Consider summarizing.",
            length,
            context.result.tool_name
        ),
        priority = "normal"
    }
end

return nil
```

### Cooldown Logic

```lua
-- scripts/cooldown_warning.lua
local last_warning = context.state.last_warning_turn or 0
local current_turn = context.turn.number
local cooldown = 5  -- Turns between warnings

-- Check if enough turns have passed
if current_turn - last_warning < cooldown then
    return nil  -- Still in cooldown
end

-- Check condition
if context.turn.token_usage > 0.8 then
    return {
        type = "notify_self",
        message = string.format(
            "Token usage at %d%% (cooldown: %d turns)",
            math.floor(context.turn.token_usage * 100),
            cooldown
        ),
        priority = "high"
    }
end

return nil
```

### Complex Workflow Detection

```lua
-- scripts/detect_research_phase.lua

-- Track tool sequences
local search_tools = {"vault_search", "web_search", "grep_search"}
local search_count = 0
local last_search_index = 0

for i, tool in ipairs(context.history.tools) do
    for _, search_name in ipairs(search_tools) do
        if tool.name == search_name and tool.success then
            search_count = search_count + 1
            last_search_index = i
        end
    end
end

-- Check if in active research phase (recent searches)
local total_tools = context.history.total_tool_calls
local recent_search = (total_tools - last_search_index) < 3

-- Fire if sufficient research and still active
if search_count >= 5 and recent_search then
    return {
        type = "notify_self",
        message = string.format(
            "Research phase active (%d searches). Consider synthesis when ready.",
            search_count
        ),
        category = "info"
    }
end

return nil
```

## Rule Configuration

Reference scripts in rules:

```toml
[rule]
id = "complex-check"
name = "Complex Check"
trigger = "on_tool_complete"

[condition]
script = "scripts/my-check.lua"  # Relative to rule file

[action]
type = "notify_self"
message = "Default message"  # Used if script returns true
```

## Timeout and Limits

- **Timeout**: 5 seconds (configurable)
- **Memory**: 100MB limit

Scripts exceeding limits are terminated:

```python
# Raises LuaTimeoutError
raise LuaTimeoutError("Script execution exceeded 5 second timeout")
```

## Debugging

Use `print()` for debugging (output goes to logs):

```lua
print("Debug: turn =", context.turn.number)
print("Debug: tools =", context.history.total_tool_calls)

-- Enable debug logging
-- Set logging.getLogger("services.plugins.lua_sandbox").setLevel(DEBUG)
```

## Best Practices

1. **Keep scripts focused**: One purpose per script
2. **Return early**: Check conditions at start
3. **Cache computations**: Store in local variables
4. **Handle nil**: Always check for nil values
5. **Avoid infinite loops**: Use bounded iterations
6. **Log sparingly**: Use print() only for debugging

## See Also

- [Sandbox Security](./sandbox.md)
- [Rule Format](../rules/format.md)
- [Context API](../context-api/reference.md)
