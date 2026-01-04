# Lua Sandbox Security

The Lua sandbox provides a secure execution environment for scripts with multiple layers of protection.

## Security Model

### Defense in Depth

```
+----------------------------------+
| Layer 1: Environment Whitelisting |  <- Only safe functions exposed
+----------------------------------+
                |
                v
+----------------------------------+
| Layer 2: Timeout Enforcement      |  <- 5 second execution limit
+----------------------------------+
                |
                v
+----------------------------------+
| Layer 3: Memory Limits            |  <- 100MB per script
+----------------------------------+
                |
                v
+----------------------------------+
| Layer 4: Read-Only Context        |  <- Cannot modify agent state
+----------------------------------+
```

## Environment Whitelisting

### Allowed Globals

```lua
-- Basic functions
print       -- Output to log
type        -- Get type
tostring    -- Convert to string
tonumber    -- Convert to number
pairs       -- Table iteration
ipairs      -- Array iteration
next        -- Next key-value
select      -- Select from varargs
unpack      -- Unpack array
pcall       -- Protected call
xpcall      -- Extended protected call
error       -- Raise error
assert      -- Assert condition
```

### Allowed Modules

#### string (Safe subset)

```lua
string.byte
string.char
string.find
string.format
string.gmatch
string.gsub
string.len
string.lower
string.match
string.rep
string.reverse
string.sub
string.upper
```

#### table (Safe subset)

```lua
table.concat
table.insert
table.maxn
table.remove
table.sort
table.unpack
```

#### math (Full module)

```lua
math.abs, math.acos, math.asin, math.atan, math.atan2
math.ceil, math.cos, math.cosh, math.deg, math.exp
math.floor, math.fmod, math.frexp, math.huge, math.ldexp
math.log, math.log10, math.max, math.min, math.modf
math.pi, math.pow, math.rad, math.random, math.randomseed
math.sin, math.sinh, math.sqrt, math.tan, math.tanh
```

### Blocked Functions

These functions are explicitly removed from the environment:

| Function | Risk |
|----------|------|
| `os.*` | System command execution |
| `io.*` | File system access |
| `debug.*` | Sandbox escape, introspection |
| `dofile` | Load external files |
| `loadfile` | Load external files |
| `load` | Arbitrary code execution |
| `loadstring` | Arbitrary code execution |
| `require` | Module loading |
| `module` | Module system |
| `package` | Package loading |
| `rawget` | Bypass metatables |
| `rawset` | Bypass metatables |
| `rawequal` | Bypass comparisons |
| `rawlen` | Bypass length operator |
| `getmetatable` | Access metatables |
| `setmetatable` | Modify metatables |
| `collectgarbage` | GC manipulation |
| `setfenv` | Change function environment |
| `getfenv` | Access function environment |
| `coroutine.*` | Coroutine creation |

## Timeout Enforcement

Scripts are executed with a timeout to prevent infinite loops and resource exhaustion.

### Implementation

```python
# Threading-based timeout (cross-platform)
thread = threading.Thread(target=run_script, daemon=True)
thread.start()
thread.join(timeout=5.0)  # 5 second timeout

if thread.is_alive():
    raise LuaTimeoutError("Script execution exceeded 5 second timeout")
```

### Configuration

```python
sandbox = LuaSandbox(
    timeout_seconds=5.0,  # Default: 5 seconds
    max_memory_mb=100,    # Default: 100 MB
)
```

### Timeout Behavior

When a script times out:
1. Script thread is marked for termination
2. `LuaTimeoutError` is raised
3. Rule evaluation fails (no action executed)
4. Error is logged for debugging

## Memory Limits

### Limit Configuration

```python
# Memory limit passed to LuaRuntime
lua = LuaRuntime(max_memory=100 * 1024 * 1024)  # 100MB
```

### Enforcement

When memory limit is exceeded:
1. LuaRuntime raises memory error
2. `LuaMemoryError` is raised to Python
3. Rule evaluation fails
4. Resources are cleaned up

## Read-Only Context

The `context` object exposed to scripts is read-only:

### No Direct State Modification

```lua
-- These would NOT work (state is a snapshot)
context.state.counter = 5  -- Has no effect
context.turn.number = 10   -- Has no effect
```

### State Changes via Actions

To persist state changes, return an action:

```lua
-- Correct way to update state
return {
    type = "set_state",
    key = "counter",
    value = tostring(current_count + 1)
}
```

## Script Isolation

### Per-Script Runtime

Each script execution creates a fresh LuaRuntime:

```python
def _execute_in_sandbox(self, script: str, context: RuleContext) -> Any:
    # Create new runtime for isolation
    lua = LuaRuntime(unpack_returned_tuples=True)
    try:
        # Execute in sandbox
        ...
    finally:
        # Cleanup
        del lua
```

### No Cross-Script State

Scripts cannot access:
- Other scripts' variables
- Previous execution state
- Global Lua state

## Error Handling

### Exception Types

```python
class LuaSandboxError(Exception):
    """Base exception for Lua sandbox errors."""

class LuaExecutionError(LuaSandboxError):
    """Error during Lua script execution."""

class LuaTimeoutError(LuaSandboxError):
    """Lua script exceeded timeout limit."""

class LuaMemoryError(LuaSandboxError):
    """Lua script exceeded memory limit."""
```

### Error Propagation

```python
try:
    result = sandbox.execute(script, context)
except LuaTimeoutError as e:
    logger.warning(f"Script timeout: {e}")
    # Rule does not fire
except LuaExecutionError as e:
    logger.warning(f"Script error: {e}")
    # Rule does not fire
except LuaSandboxError as e:
    logger.error(f"Sandbox error: {e}")
    # Rule does not fire
```

## Security Considerations

### Potential Risks

| Risk | Mitigation |
|------|------------|
| Infinite loops | Timeout enforcement |
| Memory exhaustion | Memory limits |
| File system access | `io` module blocked |
| Network access | No network modules |
| Code injection | `load*` functions blocked |
| Sandbox escape | Metatable access blocked |

### What Scripts CAN Do

- Read context data
- Perform calculations
- String manipulation
- Table operations
- Return values to rule engine

### What Scripts CANNOT Do

- Access file system
- Make network requests
- Execute system commands
- Load external code
- Modify global state
- Escape the sandbox

## Best Practices

1. **Keep scripts simple**: Complex logic is harder to audit
2. **Validate script sources**: Only use trusted script files
3. **Monitor execution**: Enable debug logging for script execution
4. **Test thoroughly**: Test scripts with edge cases
5. **Limit script count**: Fewer scripts = smaller attack surface

## Auditing

Enable debug logging for script execution:

```python
import logging
logging.getLogger("services.plugins.lua_sandbox").setLevel(logging.DEBUG)
```

Logs include:
- Script execution start/end
- Timeout warnings
- Execution errors
- Return values

## See Also

- [Lua Scripting Guide](./lua-guide.md)
- [Performance](../architecture/performance.md)
- [Architecture](../architecture/overview.md)
