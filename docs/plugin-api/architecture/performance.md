# Performance Considerations

This document covers performance characteristics, optimization strategies, and benchmarks for the Oracle Plugin System.

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Rule evaluation | <50ms per rule | Including condition + action |
| Condition parsing | <1ms | simpleeval expression |
| Lua script execution | <5s timeout | Hard limit enforced |
| Rule loading | <100ms | All rules from disk |
| Context building | <10ms | Database lookups cached |

## Tiered Performance Architecture

### Tier 1: TOML Rules (simpleeval)

**Latency**: 0.1-1ms per evaluation

```python
# Fast path: compiled expression evaluation
evaluator.eval("context.turn.token_usage > 0.8")
```

**Optimization tips**:
- Keep expressions simple
- Avoid nested function calls
- Use built-in helper functions when possible

### Tier 2: Lua Scripts (lupa/LuaJIT)

**Latency**: 0.05-0.5ms for simple scripts, up to 5s timeout

```lua
-- LuaJIT provides 20-30x speedup over CPython
local count = 0
for _, tool in ipairs(context.history.tools) do
    if tool.name == "vault_search" then count = count + 1 end
end
return count >= 3
```

**Optimization tips**:
- Keep scripts focused and minimal
- Cache intermediate results in plugin state
- Avoid infinite loops (hard timeout at 5s)

### Tier 3: Future (Rust + PyO3)

**Target**: 0.01-0.1ms per evaluation

When requirements exceed 10k rules/sec, the rule engine core can be migrated to Rust for an additional 10x speedup.

## Expression Evaluation Benchmarks

| Expression Type | Latency | Example |
|----------------|---------|---------|
| Simple comparison | ~0.1ms | `context.turn.token_usage > 0.8` |
| Boolean composition | ~0.2ms | `a > 0.8 and b > 5` |
| Function call | ~0.3ms | `tool_completed('search')` |
| Complex nested | ~0.5ms | `any(t.name == 'x' for t in tools)` |

## Memory Constraints

| Resource | Limit | Notes |
|----------|-------|-------|
| Lua sandbox | 100MB | Per-script memory limit |
| Rule cache | ~1MB | For 100 rules |
| Context object | ~10KB | Per evaluation |

## Optimization Strategies

### 1. Rule Priority

Higher priority rules evaluate first. Place frequently-matching rules at higher priorities to short-circuit evaluation:

```toml
[rule]
id = "critical-check"
priority = 900  # Evaluates early
```

### 2. Condition Caching

For rules with static conditions, consider caching the compiled expression:

```python
# Internal optimization: expressions are compiled once
_compiled_expressions = {}
```

### 3. Lazy Context Building

Only build context components that are actually used:

```python
# Context fields loaded on-demand
@property
def user_settings(self):
    if self._user_settings is None:
        self._user_settings = self._load_user_settings()
    return self._user_settings
```

### 4. Batch State Writes

For rules that update state frequently, batch writes:

```python
# Instead of individual writes
state.set("key1", value1)
state.set("key2", value2)

# Use batch operation (if available)
state.set_batch({"key1": value1, "key2": value2})
```

## Profiling Rules

Enable debug logging to profile rule evaluation:

```python
import logging
logging.getLogger("services.plugins.engine").setLevel(logging.DEBUG)
```

Output includes:
- Time per rule evaluation
- Condition evaluation duration
- Action execution time

## Scale Considerations

### Rule Count Limits

| User Type | Recommended Max | Notes |
|-----------|-----------------|-------|
| Standard | 50 rules | Per user/project |
| Power user | 100 rules | With performance impact |
| Hard limit | 1000 rules | System safety limit |

### Evaluation Frequency

Rules evaluate at each hook point. High-frequency hooks:

| Hook | Frequency | Optimization |
|------|-----------|--------------|
| `on_turn_start` | Every turn | Keep rules fast |
| `on_tool_call` | Per tool | May fire 10+ times/turn |
| `on_tool_complete` | Per tool | Check `context.result` |

### Concurrent Sessions

Each session has isolated rule evaluation:
- No cross-session rule interference
- State isolation per user/project
- Thread-safe context building

## Troubleshooting Performance

### Slow Rule Evaluation

1. **Check condition complexity**: Simplify expressions
2. **Profile Lua scripts**: Add timing logs
3. **Review state access**: Cache frequently-accessed values
4. **Check rule count**: Disable unused rules

### High Memory Usage

1. **Lua script memory**: Check for large table creation
2. **Context size**: Review history retention
3. **State accumulation**: Implement state cleanup

### Timeout Issues

```toml
# Increase Lua timeout (not recommended beyond 10s)
lua_timeout_seconds = 10.0
```

Better: Refactor complex scripts into smaller, faster operations.

## See Also

- [Architecture Overview](./overview.md)
- [Lua Scripting Guide](../scripting/lua-guide.md)
- [Expression Language](../rules/conditions.md)
