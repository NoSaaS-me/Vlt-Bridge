# Context API Reference

The `RuleContext` provides read-only access to agent state during rule evaluation.

## RuleContext Structure

```python
@dataclass
class RuleContext:
    turn: TurnState           # Current turn information
    history: HistoryState     # Conversation and tool history
    user: UserState           # User information
    project: ProjectState     # Project information
    state: PluginState        # Plugin-scoped persistent state
    event: EventData | None   # Triggering event (if applicable)
    result: ToolResult | None # Tool result (for tool hooks)
```

## TurnState

Current turn information.

### Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `number` | int | Turn number (1-indexed) | `5` |
| `token_usage` | float | Token budget usage (0.0-1.0) | `0.75` |
| `context_usage` | float | Context window usage (0.0-1.0) | `0.45` |
| `iteration_count` | int | Current iteration in turn | `3` |

### Expression Access

```toml
expression = "context.turn.number > 5"
expression = "context.turn.token_usage > 0.8"
expression = "context.turn.context_usage > 0.9"
expression = "context.turn.iteration_count >= 3"
```

### Lua Access

```lua
local turn = context.turn.number
local usage = context.turn.token_usage
```

### Validation

- `number` must be >= 1
- `token_usage` must be 0.0-1.0
- `context_usage` must be 0.0-1.0
- `iteration_count` must be >= 0

## HistoryState

Historical conversation and tool information.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `messages` | list[dict] | Recent messages (role, content) |
| `tools` | list[ToolCallRecord] | Tool call records |
| `failures` | dict[str, int] | Tool name to failure count |

### Computed Properties

| Property | Type | Description |
|----------|------|-------------|
| `total_tool_calls` | int | Total number of tool calls |
| `total_failures` | int | Sum of all failure counts |

### Methods

```python
def get_failures_for_tool(self, tool_name: str) -> int:
    """Get failure count for a specific tool."""
```

### Expression Access

```toml
# Total counts
expression = "context.history.total_tool_calls > 10"
expression = "context.history.total_failures >= 3"

# Check message count
expression = "len(context.history.messages) > 20"

# Check tool list
expression = "len(context.history.tools) > 0"
```

### Lua Access

```lua
-- Iterate over tools
for _, tool in ipairs(context.history.tools) do
    if tool.name == "vault_search" then
        -- ...
    end
end

-- Check failures
local failures = context.history.failures["api_call"] or 0
```

## ToolCallRecord

Individual tool call record in history.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Tool name |
| `arguments` | dict | Arguments passed |
| `result` | str | Tool result (may be None) |
| `success` | bool | Whether tool succeeded |
| `timestamp` | datetime | When called |

### Expression Access

```toml
# In list comprehension style
expression = "any(t.name == 'search' for t in context.history.tools)"
```

### Lua Access

```lua
for _, tool in ipairs(context.history.tools) do
    print(tool.name, tool.success)
    for k, v in pairs(tool.arguments) do
        print("  arg:", k, "=", v)
    end
end
```

## UserState

User information (read-only).

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | User identifier |
| `settings` | dict | User settings snapshot |

### Settings Keys

| Key | Type | Description |
|-----|------|-------------|
| `oracle_model` | str | Selected Oracle model |
| `oracle_provider` | str | Model provider |
| `thinking_enabled` | bool | Thinking mode enabled |
| `librarian_timeout` | int | Timeout for librarian |

### Expression Access

```toml
expression = "context.user.id != ''"
expression = "context.user.settings.get('thinking_enabled', False)"
```

### Lua Access

```lua
local user_id = context.user.id
local model = context.user.settings.oracle_model
```

## ProjectState

Project information (read-only).

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Project identifier |
| `settings` | dict | Project settings snapshot |

### Expression Access

```toml
expression = "context.project.id != ''"
expression = "context.project.settings.get('name', '') == 'MyProject'"
```

## PluginState

Plugin-scoped persistent state.

### Methods

```python
def get(self, key: str, default: Any = None) -> Any:
    """Get value from state."""

def has(self, key: str) -> bool:
    """Check if key exists."""

def keys(self) -> list[str]:
    """Return all keys."""
```

### Expression Access

```toml
# Get with default
expression = "int(context.state.get('counter', 0)) > 5"

# Check existence
expression = "context.state.has('last_warning_turn')"

# Combine with turn
expression = "context.turn.number - int(context.state.get('last_turn', 0)) >= 5"
```

### Lua Access

```lua
-- Direct access (nil if not exists)
local counter = context.state.counter or 0

-- Keys iteration
for key, value in pairs(context.state) do
    print(key, value)
end
```

### Note on Writing

State is read-only during condition evaluation. Use `set_state` action to write:

```toml
[action]
type = "set_state"
key = "counter"
value = "{{ int(context.state.get('counter', 0)) + 1 }}"
```

## EventData

Data from the triggering event (available when applicable).

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | str | Event type (e.g., `TOOL_CALL_SUCCESS`) |
| `source` | str | Component that generated event |
| `severity` | str | Event severity |
| `payload` | dict | Event-specific data |
| `timestamp` | datetime | When event occurred |

### Expression Access

```toml
expression = "context.event is not None"
expression = "context.event is not None and context.event.type == 'TOOL_CALL_SUCCESS'"
expression = "context.event is not None and 'error' in context.event.payload"
```

### Lua Access

```lua
if context.event then
    print("Event type:", context.event.type)
    print("Source:", context.event.source)

    for k, v in pairs(context.event.payload) do
        print("Payload:", k, "=", v)
    end
end
```

## ToolResult

Result from tool execution (for `on_tool_complete` and `on_tool_failure`).

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | str | Name of the tool |
| `success` | bool | Whether succeeded |
| `result` | str | Tool output (may be None) |
| `error` | str | Error message (if failed) |
| `duration_ms` | float | Execution time |

### Expression Access

```toml
# Check if result exists
expression = "context.result is not None"

# Check specific tool
expression = "context.result is not None and context.result.tool_name == 'vault_search'"

# Check success
expression = "context.result is not None and context.result.success"

# Check result size
expression = "context.result is not None and len(context.result.result or '') > 2000"

# Check duration
expression = "context.result is not None and (context.result.duration_ms or 0) > 5000"
```

### Lua Access

```lua
if context.result then
    print("Tool:", context.result.tool_name)
    print("Success:", context.result.success)

    if context.result.success then
        print("Result length:", string.len(context.result.result or ""))
    else
        print("Error:", context.result.error)
    end
end
```

## Creating Minimal Context (Testing)

```python
from services.plugins.context import RuleContext

# For testing
context = RuleContext.create_minimal(
    user_id="test-user",
    project_id="test-project",
    turn_number=5,
)
```

## RuleContextBuilder

For building full context from agent state:

```python
from services.plugins.context import RuleContextBuilder
from services.database import DatabaseService

builder = RuleContextBuilder(
    database_service=DatabaseService(),
    plugin_id="my-plugin",
)

context = builder.build(
    turn_number=5,
    token_usage=0.75,
    context_usage=0.45,
    iteration_count=3,
    messages=[{"role": "user", "content": "..."}],
    tool_calls=[{"name": "search", "status": "success", "result": "..."}],
    user_id="user-123",
    project_id="project-456",
    event=some_event,
)
```

## See Also

- [Condition Expressions](../rules/conditions.md)
- [Action Types](../rules/actions.md)
- [Lua Scripting](../scripting/lua-guide.md)
