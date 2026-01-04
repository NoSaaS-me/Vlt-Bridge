# Implementation Plan: ANS Enhancements (014-ans-enhancements)

**Feature Branch**: `014-ans-enhancements`
**Created**: 2026-01-04
**Status**: Planning
**Depends On**: 013-agent-notification-system (implemented)

## Executive Summary

This plan details four enhancements to the Agent Notification System (ANS):

1. **notify_self tool** - Allow the agent to emit notifications to its future self
2. **Proactive event types** - New events beyond failures/limits for context awareness
3. **Cross-session persistence** - Notifications that survive session restarts
4. **Scheduled/deferred delivery** - Time-based and condition-based notification delivery

---

## Current ANS Architecture Summary

The existing system has:

1. **Event Layer** (`event.py`): Event dataclass with type, source, severity, payload, and dedupe_key. EventType constants define hierarchical event types like `tool.call.failure`, `budget.token.warning`.

2. **Event Bus** (`bus.py`): Pub/sub system with wildcard subscription support, handler dispatch, and overflow protection.

3. **Subscriber System** (`subscriber.py`): TOML-based configuration files in `subscribers/` directory with validation, priority levels, injection points, and batching configs.

4. **Accumulator** (`accumulator.py`): Batches events per subscriber, handles deduplication, and queues notifications by injection point (immediate, turn_start, after_tool, turn_end).

5. **TOON Formatter** (`toon_formatter.py`): Jinja2 templates for compact notification formatting.

6. **Integration with Oracle** (`oracle_agent.py`): Events emitted from `_execute_tools()` and budget checks; notifications drained at turn_start, after_tool, and immediate injection points; system messages persisted in `context_nodes.system_messages_json`.

7. **Persistence**: System messages stored per context node in JSON array; loaded during context tree traversal.

---

## Feature 1: notify_self Tool

### Overview

The `notify_self` tool allows the Oracle agent to emit notifications that will appear in its future context. This enables the agent to leave "breadcrumbs" for itself - recording important discoveries, warnings, or context that should persist across tool calls and turns.

### Technical Approach

Create a new tool `notify_self` that:
1. Accepts a message, priority, and optional delivery timing
2. Creates an Event with type `agent.self.notify`
3. Routes through the existing ANS pipeline
4. Appears in the agent's context at the specified injection point

### Files to Modify/Create

#### 1. `backend/src/services/ans/event.py`

Add new event type constants:

```python
# Agent self-notification events
AGENT_SELF_NOTIFY = "agent.self.notify"
AGENT_SELF_REMIND = "agent.self.remind"
```

#### 2. `backend/prompts/tools.json`

Add new tool definition:

```json
{
  "type": "function",
  "function": {
    "name": "notify_self",
    "description": "Send a notification to your future self. Use this to record important discoveries, warnings, or context that should persist across turns.",
    "parameters": {
      "type": "object",
      "properties": {
        "message": {
          "type": "string",
          "description": "The notification message content."
        },
        "priority": {
          "type": "string",
          "enum": ["low", "normal", "high", "critical"],
          "default": "normal"
        },
        "category": {
          "type": "string",
          "enum": ["discovery", "warning", "checkpoint", "reminder", "context"],
          "default": "context"
        },
        "deliver_at": {
          "type": "string",
          "enum": ["immediate", "next_turn", "after_tool"],
          "default": "next_turn"
        },
        "persist_cross_session": {
          "type": "boolean",
          "default": false
        }
      },
      "required": ["message"]
    }
  },
  "agent_scope": ["oracle"],
  "category": "meta"
}
```

#### 3. `backend/src/services/tool_executor.py`

Add handler method `_notify_self` that emits the event through the event bus.

#### 4. `backend/src/services/ans/subscribers/self_notify.toml`

Create subscriber configuration for self-notifications.

#### 5. `backend/src/services/ans/templates/self_notify.toon.j2`

Create TOON template for formatting self-notifications.

---

## Feature 2: Proactive Event Types

### Overview

Extend ANS with new event types that proactively inform the agent about context changes, session state, and important discoveries.

### New Event Types

| Event Type | Description | Trigger |
|------------|-------------|---------|
| `context.approaching_limit` | Context usage hit 70% | During context token estimation |
| `session.resumed` | Context restored from previous session | When loading tree context |
| `source.stale` | Referenced file changed since last read | When file is accessed and mtime differs |
| `discovery.notable` | Agent flagged important finding | Via notify_self with category=discovery |
| `task.checkpoint` | Periodic progress update | Every N turns or on explicit checkpoint |

### Files to Modify/Create

1. `backend/src/services/ans/event.py` - Add new EventType constants
2. `backend/src/services/oracle_agent.py` - Add context limit detection, session resumed detection
3. `backend/src/services/tool_executor.py` - Add file mtime tracking for staleness detection
4. New subscriber TOML configs for each event type
5. New TOON templates for each event type

---

## Feature 3: Cross-Session Persistence

### Overview

Allow notifications to persist across session restarts. When the agent resumes a session, important notifications from the previous session can be re-injected into context.

### Technical Approach

1. Create a new database table for cross-session notifications
2. Store notifications marked with `persist_cross_session=True`
3. On session resume, load and inject pending notifications
4. Provide expiration/cleanup mechanism

### Database Schema

```sql
CREATE TABLE cross_session_notifications (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    tree_id TEXT,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    severity TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    formatted_content TEXT,
    priority TEXT NOT NULL DEFAULT 'normal',
    inject_at TEXT NOT NULL DEFAULT 'turn_start',
    created_at TEXT NOT NULL,
    expires_at TEXT,
    delivered_at TEXT,
    acknowledged_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    category TEXT,
    dedupe_key TEXT
);
```

### Files to Create/Modify

1. `backend/src/services/database.py` - Add table DDL
2. `backend/src/services/ans/persistence.py` - New service for persistence
3. `backend/src/services/oracle_agent.py` - Load pending on session resume
4. `backend/src/services/tool_executor.py` - Store persistent notifications

---

## Feature 4: Scheduled/Deferred Delivery

### Overview

Allow notifications to be scheduled for delivery at specific points:
- `next_turn` - Guaranteed delivery at start of next turn
- `after_n_turns(n)` - Deliver after N turns complete
- `after_tool(tool_name)` - Deliver after a specific tool completes
- `on_condition(predicate)` - Deliver when a condition is met

### Technical Approach

1. Extend the Accumulator with a deferred queue
2. Add delivery condition evaluation
3. Track turn counts and tool completions
4. Evaluate conditions at appropriate injection points

### Files to Create/Modify

1. `backend/src/services/ans/deferred.py` - New deferred delivery system
2. `backend/src/services/ans/accumulator.py` - Add deferred integration methods
3. `backend/src/services/oracle_agent.py` - Reset queue, drain at injection points
4. `backend/src/services/tool_executor.py` - Extend notify_self with deferred options

---

## Summary of All Files to Create/Modify

### New Files

1. `backend/src/services/ans/persistence.py`
2. `backend/src/services/ans/deferred.py`
3. `backend/src/services/ans/subscribers/self_notify.toml`
4. `backend/src/services/ans/subscribers/context_limit.toml`
5. `backend/src/services/ans/subscribers/session_resumed.toml`
6. `backend/src/services/ans/subscribers/source_stale.toml`
7. `backend/src/services/ans/subscribers/task_checkpoint.toml`
8. `backend/src/services/ans/templates/self_notify.toon.j2`
9. `backend/src/services/ans/templates/context_limit.toon.j2`
10. `backend/src/services/ans/templates/session_resumed.toon.j2`
11. `backend/src/services/ans/templates/source_stale.toon.j2`
12. `backend/src/services/ans/templates/task_checkpoint.toon.j2`

### Modified Files

1. `backend/src/services/ans/event.py` - Add new EventType constants
2. `backend/src/services/ans/accumulator.py` - Add deferred delivery methods
3. `backend/src/services/ans/__init__.py` - Export new modules
4. `backend/src/services/database.py` - Add cross_session_notifications table
5. `backend/src/services/tool_executor.py` - Add notify_self tool and file mtime tracking
6. `backend/src/services/oracle_agent.py` - Integrate proactive events and deferred delivery
7. `backend/prompts/tools.json` - Add notify_self tool definition

---

## Implementation Order

**Phase 1: Foundation (Features 1 & 2)**
1. Add new EventType constants to `event.py`
2. Create `notify_self` tool in `tool_executor.py`
3. Add tool definition to `tools.json`
4. Create self_notify subscriber and template
5. Add proactive event emissions to `oracle_agent.py`
6. Create remaining subscribers and templates

**Phase 2: Persistence (Feature 3)**
1. Add database schema for `cross_session_notifications`
2. Create `persistence.py` service
3. Integrate persistence into `notify_self` and session loading

**Phase 3: Scheduling (Feature 4)**
1. Create `deferred.py` with delivery queue
2. Add deferred methods to `accumulator.py`
3. Integrate deferred queue into agent loop
4. Extend `notify_self` with deferred parameters

**Phase 4: Testing & Polish**
1. Write unit tests for each component
2. Write integration tests for end-to-end flows
3. Update documentation and CLAUDE.md
