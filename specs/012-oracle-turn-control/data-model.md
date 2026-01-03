# Data Model: Oracle Agent Turn Control

**Feature**: 012-oracle-turn-control
**Date**: 2026-01-02

## Entities

### AgentConfig

User-configurable limits and thresholds for agent behavior.

| Field | Type | Default | Bounds | Description |
|-------|------|---------|--------|-------------|
| `max_iterations` | int | 15 | 1-50 | Maximum agent turns per query |
| `soft_warning_percent` | int | 70 | 50-90 | Percentage of max to trigger iteration warning |
| `token_budget` | int | 50000 | 1000-200000 | Maximum tokens per session |
| `token_warning_percent` | int | 80 | 50-95 | Percentage of budget to trigger token warning |
| `timeout_seconds` | int | 120 | 10-600 | Overall query timeout |
| `max_tool_calls_per_turn` | int | 5 | 1-20 | Max tools per agent turn |
| `max_parallel_tools` | int | 3 | 1-10 | Concurrency limit for tool execution |

**Persistence**: Stored in `user_settings` table as JSON fields within ModelSettings.

**Validation Rules**:
- All fields must be within bounds
- `soft_warning_percent` < 100 (must warn before limit)
- `token_warning_percent` < 100 (must warn before limit)

---

### AgentState

Runtime state during query execution. Immutable - new instances created for state transitions.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | str | Yes | User executing the query |
| `project_id` | str | Yes | Project context |
| `turn` | int | No (0) | Current iteration number |
| `tokens_used` | int | No (0) | Accumulated token count (estimate) |
| `start_time` | float | No (now) | Query start timestamp (time.time()) |
| `recent_actions` | tuple[str, ...] | No (()) | Last 3 action signatures for no-progress detection |
| `termination_reason` | str | None | Why query terminated (if terminal) |
| `config` | AgentConfig | Yes | Active configuration |
| `extensions` | dict[str, Any] | No ({}) | Extension data for future modules |

**State Transitions**:
- `INIT` → `RUNNING` (query starts)
- `RUNNING` → `RUNNING` (tool executed, turn incremented)
- `RUNNING` → `TERMINATED` (limit reached or goal achieved)

**Derived Properties**:
- `is_terminal`: True if termination_reason is set
- `elapsed_seconds`: Current time - start_time
- `iteration_percent`: (turn / config.max_iterations) * 100
- `token_percent`: (tokens_used / config.token_budget) * 100

---

### DecisionTree (Protocol)

Interface for pluggable control flow behavior.

| Method | Signature | Description |
|--------|-----------|-------------|
| `should_continue` | `(state: AgentState) -> tuple[bool, str]` | Check if loop should continue; returns (continue?, reason) |
| `on_turn_start` | `(state: AgentState) -> AgentState` | Hook before each turn, can emit warnings |
| `on_tool_result` | `(state: AgentState, result: dict) -> AgentState` | Process tool result, update action history |
| `get_config` | `() -> AgentConfig` | Return configuration for this tree |

**Implementations**:
- `DefaultDecisionTree`: Standard termination logic per FR-004
- Future: `DeepResearcherTree`, skill-specific trees

---

### SystemMessage

Chat message for system notifications.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | Literal["system"] | Yes | Message role identifier |
| `content` | str | Yes | Human-readable notification text |
| `timestamp` | str | Yes | ISO 8601 timestamp |
| `system_type` | str | No | Category: "limit_warning", "limit_reached", "no_progress", "error_limit" |
| `metadata` | dict | No | Additional context (current_value, limit_value, etc.) |

**Display Rules**:
- Distinct styling (amber warning treatment)
- No reply/edit actions (read-only)
- Ephemeral (not persisted to conversation tree)

---

### OracleStreamChunk (Extended)

SSE chunk with new system type.

| Field | Type | Description |
|-------|------|-------------|
| `type` | Literal[..., "system"] | Chunk type (existing + new) |
| `system_type` | str | Optional: notification category |
| `system_message` | str | Optional: notification content |

**New Chunk Examples**:

```json
{
  "type": "system",
  "system_type": "limit_warning",
  "system_message": "Approaching iteration limit (7/10). Consider wrapping up."
}

{
  "type": "system",
  "system_type": "limit_reached",
  "system_message": "Maximum iterations reached. Saving partial response."
}

{
  "type": "system",
  "system_type": "no_progress",
  "system_message": "No progress detected - same action attempted 3 times."
}
```

---

## Relationships

```
┌─────────────────┐
│   User          │
└────────┬────────┘
         │ 1:1
         ▼
┌─────────────────┐       ┌─────────────────┐
│  ModelSettings  │◄──────│   AgentConfig   │
│  (persisted)    │       │   (embedded)    │
└────────┬────────┘       └─────────────────┘
         │                         │
         │                         │ used by
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│  OracleAgent    │──────▶│  DecisionTree   │
│  (runtime)      │       │  (protocol)     │
└────────┬────────┘       └────────┬────────┘
         │                         │
         │ creates                 │ operates on
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│   AgentState    │◄──────│DefaultDecision  │
│  (immutable)    │       │     Tree        │
└────────┬────────┘       └─────────────────┘
         │
         │ emits
         ▼
┌─────────────────┐
│ OracleStream    │
│    Chunk        │
└────────┬────────┘
         │ includes
         ▼
┌─────────────────┐
│ SystemMessage   │
│  (new type)     │
└─────────────────┘
```

---

## Database Changes

### user_settings Table

No schema changes - AgentConfig fields added to existing JSON ModelSettings column.

**Before**:
```json
{
  "oracle_model": "...",
  "subagent_model": "...",
  "thinking_enabled": true,
  "librarian_timeout": 1200,
  "max_context_nodes": 30
}
```

**After**:
```json
{
  "oracle_model": "...",
  "subagent_model": "...",
  "thinking_enabled": true,
  "librarian_timeout": 1200,
  "max_context_nodes": 30,
  "max_iterations": 15,
  "soft_warning_percent": 70,
  "token_budget": 50000,
  "token_warning_percent": 80,
  "timeout_seconds": 120,
  "max_tool_calls_per_turn": 5,
  "max_parallel_tools": 3
}
```

**Migration**: Default values applied on first access if fields missing (backward compatible).

---

## Validation

### AgentConfig Validation

```python
class AgentConfigBounds:
    max_iterations: tuple[int, int] = (1, 50)
    soft_warning_percent: tuple[int, int] = (50, 90)
    token_budget: tuple[int, int] = (1000, 200000)
    token_warning_percent: tuple[int, int] = (50, 95)
    timeout_seconds: tuple[int, int] = (10, 600)
    max_tool_calls_per_turn: tuple[int, int] = (1, 20)
    max_parallel_tools: tuple[int, int] = (1, 10)
```

### AgentState Invariants

- `turn >= 0`
- `tokens_used >= 0`
- `start_time <= current_time`
- `len(recent_actions) <= 3`
- `config is not None`
