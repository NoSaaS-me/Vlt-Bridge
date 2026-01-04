# Condition Expressions

Rule conditions use `simpleeval` for safe expression evaluation with boolean results.

## Expression Syntax

```
expression = or_expr
or_expr    = and_expr ("or" and_expr)*
and_expr   = not_expr ("and" not_expr)*
not_expr   = "not" not_expr | comparison
comparison = value (comp_op value)?
comp_op    = ">" | "<" | ">=" | "<=" | "==" | "!=" | "is" | "is not" | "in" | "not in"
value      = function_call | attribute_access | literal
```

## Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `>` | Greater than | `context.turn.number > 5` |
| `<` | Less than | `context.turn.token_usage < 0.5` |
| `>=` | Greater or equal | `context.turn.iteration_count >= 3` |
| `<=` | Less or equal | `context.history.total_failures <= 2` |
| `==` | Equal | `context.user.id == "admin"` |
| `!=` | Not equal | `context.project.id != ""` |
| `is` | Identity | `context.result is None` |
| `is not` | Not identity | `context.event is not None` |
| `in` | Membership | `"search" in context.result.tool_name` |
| `not in` | Not membership | `"error" not in message` |

## Boolean Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `and` | Both true | `a > 0.8 and b > 5` |
| `or` | Either true | `a > 0.9 or b > 10` |
| `not` | Negation | `not context.result.success` |

## Context Fields

### Turn State (`context.turn`)

| Field | Type | Description |
|-------|------|-------------|
| `number` | int | Current turn number (1-indexed) |
| `token_usage` | float | Token budget usage (0.0-1.0) |
| `context_usage` | float | Context window usage (0.0-1.0) |
| `iteration_count` | int | Current iteration in turn |

### History State (`context.history`)

| Field | Type | Description |
|-------|------|-------------|
| `messages` | list | Recent messages |
| `tools` | list | Tool call records |
| `failures` | dict | Tool name to failure count |
| `total_tool_calls` | int | Total tools called |
| `total_failures` | int | Sum of all failures |

### User State (`context.user`)

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | User identifier |
| `settings` | dict | User settings snapshot |

### Project State (`context.project`)

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Project identifier |
| `settings` | dict | Project settings snapshot |

### Plugin State (`context.state`)

| Method | Description |
|--------|-------------|
| `get(key, default)` | Get value or default |
| `has(key)` | Check if key exists |
| `keys()` | List all keys |

### Event Data (`context.event`)

Available when triggered by an event:

| Field | Type | Description |
|-------|------|-------------|
| `type` | str | Event type |
| `source` | str | Event source component |
| `severity` | str | Event severity |
| `payload` | dict | Event-specific data |

### Tool Result (`context.result`)

Available for `on_tool_complete` and `on_tool_failure`:

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | str | Name of the tool |
| `success` | bool | Whether tool succeeded |
| `result` | str | Tool output (may be None) |
| `error` | str | Error message (if failed) |
| `duration_ms` | float | Execution time |

## Built-in Functions

### Tool Helpers

```python
tool_completed(name: str) -> bool
```
Returns `True` if the named tool completed successfully in this session.

```toml
expression = "tool_completed('vault_search')"
```

```python
tool_failed(name: str) -> bool
```
Returns `True` if the named tool has failed at least once.

```toml
expression = "tool_failed('web_search')"
```

```python
failure_count(name: str) -> int
```
Returns the number of failures for a specific tool.

```toml
expression = "failure_count('api_call') >= 3"
```

### Context Helpers

```python
context_above_threshold(threshold: float) -> bool
```
Returns `True` if context usage exceeds the threshold.

```toml
expression = "context_above_threshold(0.9)"
```

```python
message_count_above(count: int) -> bool
```
Returns `True` if message count exceeds the count.

```toml
expression = "message_count_above(20)"
```

### Standard Functions

| Function | Description | Example |
|----------|-------------|---------|
| `len(x)` | Length | `len(context.history.tools) > 5` |
| `int(x)` | Convert to int | `int(context.turn.token_usage * 100)` |
| `float(x)` | Convert to float | `float(value) > 0.5` |
| `str(x)` | Convert to string | `str(context.turn.number)` |
| `bool(x)` | Convert to bool | `bool(context.result)` |
| `abs(x)` | Absolute value | `abs(delta) < 0.1` |
| `min(a,b)` | Minimum | `min(usage, 1.0)` |
| `max(a,b)` | Maximum | `max(count, 0)` |
| `round(x)` | Round | `round(context.turn.token_usage, 2)` |
| `sum(xs)` | Sum | `sum([1, 2, 3])` |
| `any(xs)` | Any true | `any([a, b, c])` |
| `all(xs)` | All true | `all([a, b, c])` |

## Expression Examples

### Simple Thresholds

```toml
# Token usage over 80%
expression = "context.turn.token_usage > 0.8"

# More than 10 tool calls
expression = "context.history.total_tool_calls > 10"

# Turn number is a multiple of 5
expression = "context.turn.number % 5 == 0"
```

### Boolean Composition

```toml
# High usage AND many tools
expression = "context.turn.token_usage > 0.8 and context.history.total_tool_calls > 5"

# Critical: either context full OR many failures
expression = "context.turn.context_usage > 0.95 or context.history.total_failures >= 5"

# Not the first turn
expression = "not context.turn.number == 1"
```

### Tool-Specific Checks

```toml
# Specific tool completed
expression = "tool_completed('vault_search')"

# Multiple search tools used
expression = "tool_completed('vault_search') and tool_completed('web_search')"

# Same tool failed multiple times
expression = "failure_count('api_call') >= 3"
```

### Result Inspection

```toml
# Large result (for on_tool_complete)
expression = "context.result is not None and len(context.result.result or '') > 2000"

# Specific tool with failure
expression = "context.result is not None and context.result.tool_name == 'compile' and not context.result.success"
```

### State-Based Conditions

```toml
# State key exists
expression = "context.state.has('last_warning_turn')"

# State value check
expression = "context.turn.number - int(context.state.get('last_warning_turn', 0)) >= 5"
```

## Security

The expression evaluator is sandboxed:

- No access to `__dunder__` attributes
- No `import` or `exec` capabilities
- Limited function whitelist
- No file or network operations
- Read-only context access

## Error Handling

Invalid expressions raise `ExpressionError`:

```
ExpressionError: Invalid expression syntax: unexpected token '{'
ExpressionError: Attribute error: 'TurnState' has no attribute 'invalid'
ExpressionError: Feature not available: lambda not supported
```

## See Also

- [TOML Format](./format.md)
- [Context API Reference](../context-api/reference.md)
- [Lua Scripting](../scripting/lua-guide.md) for complex logic
