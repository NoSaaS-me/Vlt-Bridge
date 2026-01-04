# Data Model: Agent Notification System

**Feature Branch**: `013-agent-notification-system`
**Date**: 2026-01-03
**Spec**: [spec.md](./spec.md)

## Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     Event       │──────▶│   Subscriber    │──────▶│  Notification   │
├─────────────────┤  1:N  ├─────────────────┤  1:N  ├─────────────────┤
│ id              │       │ id              │       │ id              │
│ type            │       │ name            │       │ subscriber_id   │
│ source          │       │ event_types[]   │       │ content (TOON)  │
│ severity        │       │ template        │       │ priority        │
│ timestamp       │       │ enabled         │       │ inject_at       │
│ payload         │       │ core            │       │ events[]        │
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                   │
                                   │ N:1
                                   ▼
                          ┌─────────────────┐
                          │ UserSettings    │
                          ├─────────────────┤
                          │ user_id         │
                          │ disabled_subs[] │
                          └─────────────────┘
```

---

## Core Entities

### Event

An occurrence in the system that may trigger a notification.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | UUID | Yes | Unique event identifier |
| `type` | EventType | Yes | Hierarchical event type (e.g., "tool.call.failure") |
| `source` | string | Yes | Component that generated the event |
| `severity` | Severity | Yes | Event severity level |
| `timestamp` | datetime | Yes | When the event occurred (UTC) |
| `payload` | object | Yes | Event-specific data |
| `dedupe_key` | string | No | Key for deduplication (derived if not set) |

**Event Types** (hierarchical):
```
tool.call.pending
tool.call.success
tool.call.failure
tool.call.timeout

budget.token.warning
budget.token.exceeded
budget.iteration.warning
budget.iteration.exceeded
budget.timeout.warning

agent.turn.start
agent.turn.end
agent.loop.detected

subagent.complete    # Future
subagent.failed      # Future
cli.event            # Future
```

**Severity Levels**:
```
debug    - Verbose diagnostics
info     - Normal operation
warning  - Potential issues
error    - Operation failures
critical - System-level issues
```

**Payload Examples**:

```python
# tool.call.failure
{
    "tool_name": "vault_search",
    "error_type": "timeout",
    "error_message": "Operation timed out after 5000ms",
    "retry_count": 2,
    "call_id": "tc_abc123"
}

# budget.token.warning
{
    "budget_type": "token",
    "current": 42500,
    "limit": 50000,
    "percentage": 85,
    "remaining": 7500
}

# agent.loop.detected
{
    "pattern": "vault_search → parse → vault_search",
    "repetitions": 3,
    "window_seconds": 60
}
```

---

### Subscriber

A configuration that listens for events and transforms them into notifications.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique subscriber identifier |
| `name` | string | Yes | Human-readable name |
| `description` | string | Yes | What this subscriber does |
| `version` | string | Yes | Subscriber config version |
| `event_types` | string[] | Yes | Event types to listen for |
| `severity_filter` | Severity | No | Minimum severity to process |
| `template` | string | Yes | Jinja2 template path for TOON output |
| `priority` | Priority | Yes | Notification priority level |
| `inject_at` | InjectionPoint | Yes | When to inject notification |
| `core` | boolean | Yes | If true, cannot be disabled |
| `enabled` | boolean | Yes | Current enabled state |
| `batch_window_ms` | int | No | Time window for batching (default: 2000) |
| `max_batch_size` | int | No | Max events per batch (default: 10) |
| `dedupe_window_ms` | int | No | Deduplication window (default: 5000) |

**Priority Levels**:
```
critical - Inject immediately, bypass batching
high     - Inject at next opportunity
normal   - Standard batching behavior
low      - Aggregate at turn end
```

**Injection Points**:
```
immediate  - Insert now (critical only)
turn_start - Before agent gets control
after_tool - Between tool result and next LLM call
turn_end   - Summary before yielding
```

**Subscriber Config Example** (`tool_failure.toml`):

```toml
[subscriber]
id = "tool_failure"
name = "Tool Failure Notifications"
description = "Notifies agent when tool calls fail or timeout"
version = "1.0.0"

[events]
types = ["tool.call.failure", "tool.call.timeout"]
severity_filter = "warning"

[batching]
window_ms = 2000
max_size = 10
dedupe_key = "type:payload.tool_name"
dedupe_window_ms = 5000

[output]
priority = "high"
inject_at = "after_tool"
template = "templates/tool_failure.toon.j2"
core = true
```

---

### Notification

A formatted message ready to be injected into the agent's conversation context.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | UUID | Yes | Unique notification identifier |
| `subscriber_id` | string | Yes | Subscriber that generated this |
| `content` | string | Yes | TOON-formatted notification content |
| `priority` | Priority | Yes | Delivery priority |
| `inject_at` | InjectionPoint | Yes | When to inject |
| `timestamp` | datetime | Yes | When notification was created |
| `events` | Event[] | Yes | Events that triggered this notification |

---

### SystemMessage

A notification persisted in the conversation, displayed in the chat UI.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | UUID | Yes | Message identifier |
| `role` | "system" | Yes | Always "system" for notifications |
| `content` | string | Yes | TOON content (or parsed human-readable) |
| `timestamp` | datetime | Yes | When message was created |
| `source` | string | Yes | Source identifier (e.g., "Tool Executor") |
| `notification_id` | UUID | No | Link to original notification |

---

### NotificationSettings (User Preferences)

Extends the existing `user_settings` table.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | User identifier (PK) |
| `disabled_subscribers` | string[] | Yes | List of subscriber IDs user has disabled |

**Storage**: JSON array in `disabled_subscribers_json` column.

**Behavior**:
- Core subscribers cannot appear in disabled list (enforced at API level)
- Default is all subscribers enabled (empty list)

---

## State Transitions

### Event Lifecycle

```
[Created] → [Queued] → [Dispatched] → [Filtered] → [Batched] → [Formatted] → [Injected]
              │           │              │            │           │
              │           │              │            │           └── Notification created
              │           │              │            └── Added to batch buffer
              │           │              └── Matched subscriber(s)
              │           └── Sent to event bus
              └── Added to queue
```

### Subscriber State

```
[Loaded] ←─────────────────────────────────────┐
    │                                           │
    ▼                                           │
[Active] ─────────► [Disabled] ─────────────────┘
    │                   │
    │                   │ (core subscribers cannot be disabled)
    │                   │
    └───────────────────┘
```

### Notification State

```
[Pending] → [Injected] → [Persisted]
    │           │
    │           └── Added to conversation context
    └── Waiting for injection point
```

---

## Validation Rules

### Event

1. `type` must be a valid hierarchical event type
2. `severity` must be one of: debug, info, warning, error, critical
3. `timestamp` must be valid UTC datetime
4. `payload` must be JSON-serializable

### Subscriber

1. `id` must be unique across all subscribers
2. `event_types` must contain at least one valid event type
3. `template` must reference an existing Jinja2 template file
4. `priority` must be one of: critical, high, normal, low
5. `inject_at` must be one of: immediate, turn_start, after_tool, turn_end
6. `batch_window_ms` must be 0-10000 (0 = no batching)
7. `dedupe_window_ms` must be 0-60000 (0 = no deduplication)

### Notification

1. `content` must be valid TOON format
2. `events` must contain at least one event
3. `priority` must match subscriber priority

### NotificationSettings

1. `disabled_subscribers` cannot contain core subscriber IDs
2. All IDs in list must reference valid loaded subscribers

---

## Database Schema Extensions

### user_settings table (extend existing)

```sql
-- Add column for disabled subscribers
ALTER TABLE user_settings
ADD COLUMN disabled_subscribers_json TEXT DEFAULT '[]';
```

### context_nodes table (extend existing)

```sql
-- Add column for system messages per node
ALTER TABLE context_nodes
ADD COLUMN system_messages_json TEXT DEFAULT '[]';
```

**System Message JSON Schema**:
```json
[
  {
    "id": "uuid",
    "source": "Tool Executor",
    "content": "tool_fail: vault_search timeout",
    "timestamp": "2026-01-03T15:30:00Z"
  }
]
```

---

## TOON Format Examples

### Single Tool Failure

```
tool_fail: vault_search timeout after 5000ms
```

### Batched Tool Failures

```
tool_fails[3]{tool,error,ts}:
  vault_search,timeout,15:30:01
  coderag_search,index_missing,15:30:02
  vault_write,permission_denied,15:30:05
```

### Budget Warning

```
budget_warn: 85% tokens consumed (42500/50000)
```

### Mixed Batch

```
notifications[4]{type,source,message}:
  tool_fail,vault_search,timeout after 5000ms
  tool_fail,coderag_search,index not found
  budget_warn,token_monitor,85% consumed (42500/50000)
  loop_detect,pattern_analyzer,repeated vault_search 3x
```

### With Severity Prefix (for UI styling)

```
!critical: budget exceeded - turn will terminate
!warning: loop detected - same pattern 3 times
info: tool completed successfully
```

---

## Index Recommendations

No additional database indexes required. The `disabled_subscribers_json` column is small (typically < 100 bytes) and queried only on settings load.

The `system_messages_json` column is per-node and accessed during tree traversal, which already uses the existing `context_nodes` indexes.
