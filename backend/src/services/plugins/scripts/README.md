# Lua Scripts

This directory contains Lua scripts for complex rule logic that exceeds
the capabilities of TOML expression conditions.

## When to Use Scripts

Use Lua scripts when you need:
- Complex temporal patterns (e.g., "X happened 3 times in last 5 turns")
- Aggregations over historical data
- Multi-step conditional logic
- Dynamic message generation based on complex state

For simple threshold checks, prefer TOML expressions.

## Script Format

Scripts are executed via lupa (Lua 5.4) in a sandboxed environment.

```lua
-- my_rule_script.lua

-- The context object is available globally
local token_usage = context.turn.token_usage
local tool_failures = context.history.failures

-- Check conditions
if token_usage > 0.8 and tool_failures["vault_search"] > 2 then
    -- Return action to execute
    return {
        type = "notify_self",
        message = "High resource usage with search failures",
        priority = "high"
    }
end

-- Return nil if rule should not fire
return nil
```

## Available APIs

### Context Object

```lua
-- Turn state
context.turn.number           -- int: Current turn (1-indexed)
context.turn.token_usage      -- float: 0.0-1.0
context.turn.context_usage    -- float: 0.0-1.0
context.turn.iteration_count  -- int: Current iteration

-- History
context.history.messages      -- list of {role, content}
context.history.tools         -- list of tool call records
context.history.failures      -- table: tool_name -> failure_count

-- User/Project
context.user.id               -- string
context.user.settings         -- table
context.project.id            -- string
context.project.settings      -- table

-- Plugin state (persistent)
context.state:get("key")      -- Get value (nil if not set)
```

### Action Functions

```lua
-- Available actions to return
return {
    type = "notify_self",
    message = "Message to agent",
    category = "warning",      -- info, warning, error
    priority = "normal"        -- low, normal, high, critical
}

return {
    type = "log",
    level = "info",           -- debug, info, warning, error
    message = "Debug message"
}

return {
    type = "set_state",
    key = "my_key",
    value = "my_value"
}

return {
    type = "emit_event",
    event_type = "custom.event.name",
    payload = { key = "value" }
}
```

## Sandbox Restrictions

Scripts run in a restricted environment:
- **No filesystem access** - Cannot read/write files
- **No network access** - Cannot make HTTP requests
- **No os module** - Cannot execute system commands
- **Execution timeout** - Default 5 seconds (configurable)
- **Memory limit** - Default 100MB

## Referencing Scripts from Rules

In your rule TOML:

```toml
[rule]
id = "my-complex-rule"
name = "My Complex Rule"
trigger = "on_turn_start"

[condition]
script = "scripts/my_rule_script.lua"  # Relative to plugins directory

[action]
# Action is defined in the script return value
```

## Best Practices

1. Keep scripts focused on a single concern
2. Use descriptive variable names
3. Add comments explaining complex logic
4. Test with various context states
5. Handle edge cases (nil values, empty lists)
6. Prefer returning structured actions over side effects

## Available Safe Lua Functions

The sandbox provides access to:

**Safe Globals:**
- `print`, `type`, `tostring`, `tonumber`
- `pairs`, `ipairs`, `next`, `select`, `unpack`
- `pcall`, `xpcall`, `error`, `assert`

**String Module:**
- `string.byte`, `string.char`, `string.find`, `string.format`
- `string.gmatch`, `string.gsub`, `string.len`, `string.lower`
- `string.match`, `string.rep`, `string.reverse`, `string.sub`, `string.upper`

**Table Module:**
- `table.concat`, `table.insert`, `table.remove`, `table.sort`, `table.unpack`

**Math Module:**
- All math functions (`math.abs`, `math.floor`, `math.ceil`, `math.max`, `math.min`, etc.)
- Constants: `math.pi`, `math.huge`

## Blocked Functions (Security)

The following are NOT available:
- `os.*` - System operations
- `io.*` - File I/O
- `debug.*` - Debug hooks
- `require`, `dofile`, `loadfile`, `loadstring` - Code loading
- `rawget`, `rawset`, `rawequal` - Raw access (sandbox escape)
- `setmetatable`, `getmetatable` - Metatable manipulation
- `collectgarbage` - GC control

## Example Scripts

See `repeated_failure_detector.lua` for a complete example of complex
temporal pattern matching with context analysis.
