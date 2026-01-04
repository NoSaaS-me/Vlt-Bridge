# Hook Points Lifecycle

Hook points define when rules are triggered during the agent's execution lifecycle.

## Lifecycle Overview

```
User Query
    |
    v
+-------------------+
| on_query_start    |  <- New query received
+-------------------+
    |
    v
+-------------------+
| on_turn_start     |  <- Before processing
+-------------------+
    |
    +---> [Agent Processing]
    |         |
    |         v
    |    +-------------------+
    |    | on_tool_call      |  <- Before tool execution
    |    +-------------------+
    |         |
    |         v
    |    [Tool Execution]
    |         |
    |         +---> Success: on_tool_complete
    |         |
    |         +---> Failure: on_tool_failure
    |         |
    |         v
    |    [Continue Processing...]
    |
    v
+-------------------+
| on_turn_end       |  <- After turn complete
+-------------------+
    |
    v
[More turns or...]
    |
    v
+-------------------+
| on_session_end    |  <- Session closing
+-------------------+
```

## Hook Point Reference

### on_query_start

**When**: New user query received, before any processing.

**Event Type**: `EventType.QUERY_START`

**Use Cases**:
- Initialize session state
- Log query start
- Reset counters
- Validate user permissions

**Available Context**:
- `context.user` - Full user information
- `context.project` - Full project information
- `context.event.payload` - Contains `question`, `user_id`, `project_id`
- `context.state` - Previous session state

**Example**:
```toml
[rule]
id = "query-start-init"
trigger = "on_query_start"

[condition]
expression = "not context.state.has('session_start')"

[action]
type = "set_state"
key = "session_start"
value = "{{ context.turn.number }}"
```

### on_turn_start

**When**: Before agent processes each turn.

**Event Type**: `EventType.AGENT_TURN_START`

**Use Cases**:
- Budget warnings
- Progress reminders
- Iteration limits
- Periodic notifications

**Available Context**:
- `context.turn` - Current turn state
- `context.history` - Messages and tools from previous turns
- `context.state` - Plugin state
- All user/project info

**Example**:
```toml
[rule]
id = "turn-budget-check"
trigger = "on_turn_start"

[condition]
expression = "context.turn.token_usage > 0.8"

[action]
type = "notify_self"
message = "Token budget at {{ (context.turn.token_usage * 100) | int }}%"
priority = "high"
```

### on_turn_end

**When**: After agent completes a turn.

**Event Type**: `EventType.AGENT_TURN_END`

**Use Cases**:
- Progress checkpoints
- Summary triggers
- Milestone events
- Logging

**Available Context**:
- Full turn state (final)
- Complete tool history for turn
- Updated state

**Example**:
```toml
[rule]
id = "turn-milestone"
trigger = "on_turn_end"

[condition]
expression = "context.turn.number % 10 == 0"

[action]
type = "emit_event"
event_type = "custom.milestone"
payload = { turn = "{{ context.turn.number }}" }
```

### on_tool_call

**When**: Before a tool is executed.

**Event Type**: `EventType.TOOL_CALL_PENDING`

**Use Cases**:
- Tool validation
- Usage tracking
- Rate limiting
- Logging tool attempts

**Available Context**:
- `context.event.payload` - Contains `tool_name`, `arguments`
- Current turn state
- History up to this point

**Example**:
```toml
[rule]
id = "track-tool-call"
trigger = "on_tool_call"

[condition]
expression = "True"

[action]
type = "log"
message = "Tool call: {{ context.event.payload.get('tool_name', 'unknown') }}"
level = "debug"
```

### on_tool_complete

**When**: After a tool returns successfully.

**Event Type**: `EventType.TOOL_CALL_SUCCESS`

**Use Cases**:
- Result processing hints
- Research tracking
- Success counters
- Large result handling

**Available Context**:
- `context.result` - Full ToolResult with output
- `context.event` - Completion event
- Updated history with this tool

**Example**:
```toml
[rule]
id = "large-result-hint"
trigger = "on_tool_complete"

[condition]
expression = "context.result is not None and len(context.result.result or '') > 2000"

[action]
type = "notify_self"
message = "Large result from {{ context.result.tool_name }}. Consider summarizing."
deliver_at = "after_tool"
```

### on_tool_failure

**When**: When a tool fails or times out.

**Event Types**: `EventType.TOOL_CALL_FAILURE`, `EventType.TOOL_CALL_TIMEOUT`

**Use Cases**:
- Failure tracking
- Alternative suggestions
- Error logging
- Circuit breaker patterns

**Available Context**:
- `context.result` - ToolResult with error info
- `context.event` - Failure event
- `context.history.failures` - Failure counts

**Example**:
```toml
[rule]
id = "repeated-failure"
trigger = "on_tool_failure"

[condition]
expression = "context.history.total_failures >= 3"

[action]
type = "notify_self"
message = "Multiple failures ({{ context.history.total_failures }}). Try a different approach."
priority = "high"
deliver_at = "immediate"
```

### on_session_end

**When**: Session is closing.

**Event Type**: `EventType.SESSION_END`

**Use Cases**:
- Cleanup state
- Final logging
- Summary generation
- Analytics events

**Available Context**:
- Full session history
- Final state
- All accumulated data

**Example**:
```toml
[rule]
id = "session-cleanup"
trigger = "on_session_end"

[condition]
expression = "True"

[action]
type = "log"
message = "Session ended. Turns: {{ context.turn.number }}, Tools: {{ context.history.total_tool_calls }}"
level = "info"
```

## Event to Hook Mapping

| ANS EventType | Hook Point | Notes |
|---------------|------------|-------|
| `QUERY_START` | `on_query_start` | Added for plugin system |
| `AGENT_TURN_START` | `on_turn_start` | Existing ANS event |
| `AGENT_TURN_END` | `on_turn_end` | Existing ANS event |
| `TOOL_CALL_PENDING` | `on_tool_call` | Before execution |
| `TOOL_CALL_SUCCESS` | `on_tool_complete` | Successful completion |
| `TOOL_CALL_FAILURE` | `on_tool_failure` | Execution failed |
| `TOOL_CALL_TIMEOUT` | `on_tool_failure` | Timeout (maps to failure) |
| `SESSION_END` | `on_session_end` | Added for plugin system |

## Evaluation Order

1. Events fire in lifecycle order
2. For each hook, rules evaluated in priority order (highest first)
3. All matching rules execute (no short-circuit)
4. Actions dispatch immediately after match

## Frequency Considerations

| Hook | Frequency | Impact |
|------|-----------|--------|
| `on_query_start` | Once per query | Low |
| `on_turn_start` | Every turn | Medium |
| `on_turn_end` | Every turn | Medium |
| `on_tool_call` | Per tool | High (10+ per turn) |
| `on_tool_complete` | Per tool | High |
| `on_tool_failure` | On failures only | Low |
| `on_session_end` | Once per session | Low |

Keep rules on high-frequency hooks (tool events) fast and simple.

## See Also

- [Architecture Overview](../architecture/overview.md)
- [Rule Format](../rules/format.md)
- [Performance](../architecture/performance.md)
