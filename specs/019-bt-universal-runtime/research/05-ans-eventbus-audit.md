# ANS EventBus Deep Audit for BT Runtime Integration

## Executive Summary

The Agent Notification System (ANS) EventBus is a mature, thread-safe pub/sub system designed for intra-agent communication. This audit examines how it can serve as the triggering mechanism for behavior tree ticks in the BT Universal Runtime (spec 019).

**Key Finding**: The EventBus is well-suited for triggering tree ticks, but requires careful design to avoid tick-event feedback loops and ensure proper event batching before tick initiation.

---

## 1. EventBus Pattern Analysis

### 1.1 Core Architecture

**Location**: `backend/src/services/ans/bus.py`

The EventBus implements a classic pub/sub pattern with these characteristics:

```
                    ┌─────────────────────────┐
                    │       EventBus          │
                    │  (Global Singleton)     │
                    ├─────────────────────────┤
   emit(Event) ────>│  _pending_events[]      │
                    │  _handlers{}            │──────> Handler Callbacks
                    │  _global_handlers[]     │
                    │  _lock (Threading)      │
                    └─────────────────────────┘
                              │
                              ▼
                       Immediate Dispatch
```

**Key Properties**:
- **Singleton Pattern**: `get_event_bus()` returns global instance
- **Thread-Safe**: Uses `threading.Lock` for all operations
- **Immediate Dispatch**: Events are dispatched synchronously on `emit()`
- **Overflow Protection**: Drops low-priority events when queue exceeds `max_queue_size` (default: 1000)

### 1.2 Subscribe/Emit Flow

```python
# Subscribe Pattern
bus = get_event_bus()
bus.subscribe("tool.*", handler_callback)      # Wildcard subscription
bus.subscribe("budget.token.warning", handler) # Exact subscription
bus.subscribe_all(global_handler)              # All events

# Emit Pattern
bus.emit(Event(
    type="tool.call.failure",
    source="oracle_agent",
    severity=Severity.ERROR,
    payload={"tool_name": "search", "error": "timeout"}
))
```

**Dispatch Order**:
1. Global handlers called first (in registration order)
2. Type-specific handlers called (in registration order)
3. Errors in handlers are logged but don't stop other handlers

### 1.3 Wildcard Matching Algorithm

**Location**: `bus.py:117-125`

```python
def _matches_type(self, actual_type: str, subscribed_type: str) -> bool:
    if actual_type == subscribed_type:
        return True
    # Support wildcard: "tool.*" matches "tool.call.failure"
    if subscribed_type.endswith(".*"):
        prefix = subscribed_type[:-2]
        return actual_type.startswith(prefix + ".")
    return False
```

**Pattern Support**:
- `tool.call.failure` - Exact match only
- `tool.*` - Matches `tool.call.failure`, `tool.call.success`, `tool.batch.complete`
- `*` or `**` - NOT supported (use `subscribe_all()` instead)

**Limitation**: No multi-level wildcards (e.g., `tool.*.failure`).

---

## 2. Event Types Inventory

### 2.1 Current EventType Constants

**Location**: `backend/src/services/ans/event.py`

```python
class EventType:
    # Tool events
    TOOL_CALL_PENDING = "tool.call.pending"
    TOOL_CALL_SUCCESS = "tool.call.success"
    TOOL_CALL_FAILURE = "tool.call.failure"
    TOOL_CALL_TIMEOUT = "tool.call.timeout"
    TOOL_BATCH_COMPLETE = "tool.batch.complete"

    # Budget events
    BUDGET_TOKEN_WARNING = "budget.token.warning"
    BUDGET_TOKEN_EXCEEDED = "budget.token.exceeded"
    BUDGET_ITERATION_WARNING = "budget.iteration.warning"
    BUDGET_ITERATION_EXCEEDED = "budget.iteration.exceeded"
    BUDGET_TIMEOUT_WARNING = "budget.timeout.warning"

    # Agent events
    AGENT_TURN_START = "agent.turn.start"
    AGENT_TURN_END = "agent.turn.end"
    AGENT_LOOP_DETECTED = "agent.loop.detected"
    AGENT_SELF_NOTIFY = "agent.self.notify"
    AGENT_SELF_REMIND = "agent.self.remind"

    # Proactive context events (014-ans-enhancements)
    CONTEXT_APPROACHING_LIMIT = "context.approaching_limit"
    SESSION_RESUMED = "session.resumed"
    SOURCE_STALE = "source.stale"
    TASK_CHECKPOINT = "task.checkpoint"

    # Session lifecycle events (015-oracle-plugin-system)
    QUERY_START = "query.start"
    SESSION_END = "session.end"

    # Future events (placeholders)
    SUBAGENT_COMPLETE = "subagent.complete"
    SUBAGENT_FAILED = "subagent.failed"
    CLI_EVENT = "cli.event"
```

### 2.2 Severity Levels

```python
class Severity(Enum):
    DEBUG = "debug"      # value_int: 0
    INFO = "info"        # value_int: 1
    WARNING = "warning"  # value_int: 2
    ERROR = "error"      # value_int: 3
    CRITICAL = "critical" # value_int: 4
```

**Overflow Protection**: CRITICAL and ERROR events are never dropped during queue overflow.

---

## 3. Batching System (NotificationAccumulator)

### 3.1 Architecture

**Location**: `backend/src/services/ans/accumulator.py`

```
     Event arrives
          │
          ▼
  ┌───────────────────┐
  │ Subscriber Match  │──── No match ────> Ignored
  └───────────────────┘
          │ Match
          ▼
  ┌───────────────────┐
  │ Dedup Check       │──── Duplicate ───> Ignored
  └───────────────────┘
          │ New
          ▼
  ┌───────────────────┐
  │ Priority Check    │──── CRITICAL ────> Immediate Notification
  └───────────────────┘
          │ Normal
          ▼
  ┌───────────────────┐
  │ Batch Accumulator │
  │  _pending[sub_id] │
  └───────────────────┘
          │ Batch full OR window expired
          ▼
  ┌───────────────────┐
  │ Create            │
  │ Notification      │
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │ Queue by          │
  │ InjectionPoint    │
  └───────────────────┘
```

### 3.2 Batching Configuration

From subscriber TOML files:

```toml
[batching]
window_ms = 2000          # Time window for batching
max_size = 10             # Max events before flush
dedupe_key = "type:payload.tool_name"  # Deduplication pattern
dedupe_window_ms = 5000   # Window for duplicate detection
```

**Deduplication Key Patterns**:
- `""` - No deduplication
- `"type:payload.tool_name"` - Dedupe by event type + tool_name payload field
- `"type:payload.budget_type"` - Dedupe by event type + budget_type

### 3.3 Drain Methods

```python
# Drain by injection point
accumulator.drain(InjectionPoint.IMMEDIATE)   # Critical notifications
accumulator.drain(InjectionPoint.TURN_START)  # Budget warnings
accumulator.drain(InjectionPoint.AFTER_TOOL)  # Tool failures
accumulator.drain(InjectionPoint.TURN_END)    # Summaries

# Convenience methods
accumulator.drain_immediate()
accumulator.drain_turn_start()
accumulator.drain_after_tool()
accumulator.drain_turn_end()
```

**Priority Sorting**: Notifications sorted CRITICAL > HIGH > NORMAL > LOW

---

## 4. Injection Points

### 4.1 Defined Injection Points

**Location**: `backend/src/services/ans/subscriber.py`

```python
class InjectionPoint(str, Enum):
    IMMEDIATE = "immediate"     # Insert now (critical only)
    TURN_START = "turn_start"   # Before agent gets control
    AFTER_TOOL = "after_tool"   # Between tool result and next LLM call
    TURN_END = "turn_end"       # Summary before yielding
```

### 4.2 Current Subscriber Configurations

| Subscriber | Events | Priority | Inject At | Core |
|------------|--------|----------|-----------|------|
| tool_failure | tool.call.failure, tool.call.timeout | high | after_tool | true |
| budget_warning | budget.token.warning, budget.iteration.warning | high | turn_start | true |
| budget_exceeded | budget.token.exceeded, budget.iteration.exceeded | critical | immediate | true |
| loop_detected | agent.loop.detected | critical | immediate | true |
| self_notify | agent.self.notify, agent.self.remind | normal | turn_start | false |
| context_limit | context.approaching_limit | high | turn_start | true |
| session_resumed | session.resumed | normal | turn_start | false |
| source_stale | source.stale | normal | turn_start | false |
| task_checkpoint | task.checkpoint | normal | turn_end | false |

---

## 5. Deferred Delivery System

### 5.1 Delivery Triggers

**Location**: `backend/src/services/ans/deferred.py`

```python
class DeliveryTrigger(str, Enum):
    NEXT_TURN = "next_turn"           # Deliver at start of next turn
    AFTER_N_TURNS = "after_n_turns"   # Deliver after N turns complete
    AFTER_TOOL = "after_tool"         # Deliver after specific tool completes
    ON_CONDITION = "on_condition"     # Deliver when condition predicate returns True
```

### 5.2 Condition Predicates

Built-in predicates for conditional delivery:

```python
context_above_threshold(0.8)    # Fire when context usage >= 80%
tool_completed("search_code")   # Fire when specific tool completes
message_count_above(10)         # Fire when message count >= 10
token_usage_above(0.7)          # Fire when token budget usage >= 70%
```

### 5.3 DeliveryContext

```python
@dataclass
class DeliveryContext:
    turn_number: int = 0
    total_tokens: int = 0
    max_tokens: int = 0
    context_tokens: int = 0
    max_context_tokens: int = 0
    last_tool_name: Optional[str] = None
    last_tool_result: Optional[str] = None
    message_count: int = 0
    custom_data: Dict[str, Any] = field(default_factory=dict)
```

---

## 6. Cross-Session Persistence

### 6.1 Architecture

**Location**: `backend/src/services/ans/persistence.py`

```python
class CrossSessionNotification:
    id: str
    user_id: str
    project_id: str
    tree_id: Optional[str]
    event_type: str
    source: str
    severity: str
    payload: dict
    formatted_content: Optional[str]
    priority: str
    inject_at: str
    created_at: datetime
    expires_at: Optional[datetime]  # Default: 24 hours
    delivered_at: Optional[datetime]
    acknowledged_at: Optional[datetime]
    status: NotificationStatus  # PENDING, DELIVERED, ACKNOWLEDGED, EXPIRED, CANCELLED
    category: Optional[str]
    dedupe_key: Optional[str]
```

### 6.2 Database Schema

Stored in SQLite table `cross_session_notifications` with columns matching the dataclass fields.

### 6.3 Lifecycle Methods

```python
service = get_persistence_service()

# Store notification
notification = service.store(notification, expiry_hours=24)

# Retrieve pending
pending = service.get_pending(user_id, project_id, tree_id=None)

# Update status
service.mark_delivered(notification_id)
service.mark_acknowledged(notification_id)
service.cancel(notification_id)

# Cleanup
service.cleanup_expired(user_id=None)  # Remove expired notifications
```

---

## 7. Integration with Rule Engine

### 7.1 Event-to-HookPoint Mapping

**Location**: `backend/src/services/plugins/engine.py`

```python
EVENT_TO_HOOK: Dict[str, HookPoint] = {
    EventType.QUERY_START: HookPoint.ON_QUERY_START,
    EventType.AGENT_TURN_START: HookPoint.ON_TURN_START,
    EventType.AGENT_TURN_END: HookPoint.ON_TURN_END,
    EventType.TOOL_CALL_PENDING: HookPoint.ON_TOOL_CALL,
    EventType.TOOL_CALL_SUCCESS: HookPoint.ON_TOOL_COMPLETE,
    EventType.TOOL_CALL_FAILURE: HookPoint.ON_TOOL_FAILURE,
    EventType.TOOL_CALL_TIMEOUT: HookPoint.ON_TOOL_FAILURE,
    EventType.SESSION_END: HookPoint.ON_SESSION_END,
}
```

This mapping shows how ANS events already drive the plugin system, which can be extended for BT integration.

---

## 8. Integration Design for BT Runtime

### 8.1 Events That Should Trigger Tree Ticks

Based on the spec requirements and existing event patterns:

| Event Type | Tick Trigger Behavior |
|------------|----------------------|
| `query.start` | Start new tree execution |
| `tool.call.success` | Resume tree after tool completion |
| `tool.call.failure` | Resume tree with failure context |
| `tool.call.timeout` | Resume tree with timeout context |
| `budget.*.exceeded` | Interrupt tree, force failure path |
| `agent.loop.detected` | Interrupt tree, activate recovery |
| `tree.reload.requested` | Hot reload signal |
| `tree.tick.resume` | Resume suspended tree |

### 8.2 New Event Types Needed for BT Runtime

```python
# Tree lifecycle events
TREE_TICK_START = "tree.tick.start"          # Tree tick cycle beginning
TREE_TICK_COMPLETE = "tree.tick.complete"    # Tree tick cycle finished
TREE_STATUS_CHANGED = "tree.status.changed"  # SUCCESS, FAILURE, RUNNING transition

# Node events
TREE_NODE_STARTED = "tree.node.started"      # Node began execution
TREE_NODE_COMPLETED = "tree.node.completed"  # Node finished execution
TREE_NODE_STUCK = "tree.node.stuck"          # Node exceeded timeout
TREE_NODE_CANCELLED = "tree.node.cancelled"  # Node was interrupted

# Tree management
TREE_RELOAD_REQUESTED = "tree.reload.requested"  # Hot reload request
TREE_RELOAD_COMPLETE = "tree.reload.complete"    # Hot reload finished
TREE_LOADED = "tree.loaded"                      # New tree loaded
TREE_UNLOADED = "tree.unloaded"                  # Tree removed

# Blackboard events
BLACKBOARD_KEY_CHANGED = "blackboard.key.changed"  # Key written
BLACKBOARD_SCOPE_CREATED = "blackboard.scope.created"  # New scope
BLACKBOARD_SCOPE_DESTROYED = "blackboard.scope.destroyed"  # Scope removed

# LLM node specific
LLM_STREAM_CHUNK = "llm.stream.chunk"        # Streaming chunk received
LLM_STREAM_COMPLETE = "llm.stream.complete"  # Streaming finished
LLM_BUDGET_EXCEEDED = "llm.budget.exceeded"  # LLM node budget exceeded
```

### 8.3 Tick-Event Loop Prevention

**Problem**: If tree tick emits events, and events trigger ticks, we get infinite loops.

**Solution: Event Batching Window**

```
┌─────────────────────────────────────────────────────────────────┐
│                     TICK CYCLE                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] EVENT COLLECTION PHASE (events buffered, not dispatched)    │
│      │                                                           │
│      ▼                                                           │
│  [2] TICK EXECUTION PHASE (tree.root.tick())                     │
│      │                                                           │
│      │  Events emitted during tick → stored in tick_events[]     │
│      │                                                           │
│      ▼                                                           │
│  [3] EVENT DISPATCH PHASE (after tick completes)                 │
│      │                                                           │
│      │  tick_events[] dispatched to handlers                     │
│      │  Handlers may trigger new events → next tick batch        │
│      │                                                           │
│      ▼                                                           │
│  [4] TICK DECISION PHASE                                         │
│      │                                                           │
│      │  If tree status == RUNNING and events pending:            │
│      │     → Schedule next tick (async)                          │
│      │  Else:                                                    │
│      │     → Tree complete or waiting for external event         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation Pattern**:

```python
class BTEventIntegration:
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._tick_in_progress = False
        self._tick_events: List[Event] = []

    def tick_tree(self, tree: BehaviorTree) -> RunStatus:
        """Execute one tick cycle with event batching."""
        self._tick_in_progress = True

        try:
            # [1] Drain pending events into blackboard
            pending_events = self._drain_pending_events()
            tree.blackboard.set("_pending_events", pending_events)

            # [2] Execute tick
            status = tree.root.tick(self._create_context(tree))

            # [3] Dispatch accumulated tick events
            for event in self._tick_events:
                self._event_bus.emit(event)
            self._tick_events.clear()

            return status

        finally:
            self._tick_in_progress = False

    def emit_during_tick(self, event: Event) -> None:
        """Buffer event if tick in progress, else emit immediately."""
        if self._tick_in_progress:
            self._tick_events.append(event)
        else:
            self._event_bus.emit(event)
```

### 8.4 Event-Driven Tick Triggering

```
                         External Event
                              │
                              ▼
                    ┌─────────────────┐
                    │  EventBus       │
                    │  Handler        │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ query.start  │   │ tool.success │   │ timer.expire │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                   │                   │
         ▼                   ▼                   ▼
  ┌──────────────────────────────────────────────────┐
  │              BT Tick Scheduler                    │
  │                                                   │
  │  - Coalesces multiple events into single tick     │
  │  - Respects tick budget (max ticks per second)    │
  │  - Queues events if tick in progress              │
  └──────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  tree.tick()    │
                    └─────────────────┘
```

### 8.5 Recommended Tick Trigger Events

```python
# Events that should ALWAYS trigger a tick
TICK_TRIGGER_EVENTS = {
    EventType.QUERY_START,           # New user input
    EventType.TOOL_CALL_SUCCESS,     # Tool completed
    EventType.TOOL_CALL_FAILURE,     # Tool failed
    EventType.TOOL_CALL_TIMEOUT,     # Tool timed out
    "tree.reload.requested",         # Hot reload
    "tree.tick.resume",              # Explicit resume
}

# Events that MAY trigger a tick (based on tree state)
CONDITIONAL_TICK_EVENTS = {
    EventType.BUDGET_TOKEN_EXCEEDED,     # Only if tree is RUNNING
    EventType.BUDGET_ITERATION_EXCEEDED, # Only if tree is RUNNING
    EventType.AGENT_LOOP_DETECTED,       # Only if tree is RUNNING
}

# Events that NEVER trigger a tick (informational only)
NO_TICK_EVENTS = {
    "tree.tick.start",       # Emitted BY tick, not trigger FOR tick
    "tree.tick.complete",    # Emitted BY tick
    "tree.node.started",     # Observability only
    "tree.node.completed",   # Observability only
    "llm.stream.chunk",      # High frequency, handled internally
}
```

---

## 9. Event Flow Diagram for Tree Tick Triggering

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          USER QUERY FLOW                                  │
│                                                                          │
│  User Query ──► API ──► emit(query.start) ──► BTScheduler ──► tick()     │
│                                                    │                      │
│                          ┌─────────────────────────┘                      │
│                          │                                                │
│                          ▼                                                │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │                    TICK EXECUTION                              │       │
│  │                                                                │       │
│  │  tree.tick()                                                   │       │
│  │    │                                                           │       │
│  │    ├──► Sequence Node                                          │       │
│  │    │      ├──► LoadContext (LEAF) ──► SUCCESS                  │       │
│  │    │      │                                                    │       │
│  │    │      ├──► LLMCall (LEAF) ──► RUNNING                      │       │
│  │    │      │      │                                             │       │
│  │    │      │      └──► Emit: llm.stream.chunk (buffered)        │       │
│  │    │      │      └──► Emit: llm.stream.complete (buffered)     │       │
│  │    │      │                                                    │       │
│  │    │      └──► HasToolCalls? (CONDITION)                       │       │
│  │    │             │                                             │       │
│  │    │             ├──► YES ──► ExecuteTool (LEAF) ──► RUNNING   │       │
│  │    │             │              │                              │       │
│  │    │             │              └──► [Tool execution async]    │       │
│  │    │             │                                             │       │
│  │    │             └──► NO ──► EmitResponse (LEAF) ──► SUCCESS   │       │
│  │    │                                                           │       │
│  │    └──► Emit: tree.tick.complete                               │       │
│  │                                                                │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                          │                                                │
│                          │ status == RUNNING                              │
│                          │                                                │
│                          ▼                                                │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │                  ASYNC TOOL EXECUTION                          │       │
│  │                                                                │       │
│  │  ToolExecutor.execute(tool_call)                               │       │
│  │    │                                                           │       │
│  │    ├──► Success ──► emit(tool.call.success)                    │       │
│  │    │                    │                                      │       │
│  │    │                    └──► BTScheduler ──► tick() [resume]   │       │
│  │    │                                                           │       │
│  │    └──► Failure ──► emit(tool.call.failure)                    │       │
│  │                         │                                      │       │
│  │                         └──► BTScheduler ──► tick() [handle]   │       │
│  │                                                                │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 10. E2E Test Scenarios

### 10.1 Event Emission and Subscription

```python
def test_event_emission_and_subscription():
    """Verify basic pub/sub functionality."""
    bus = EventBus()
    received = []

    bus.subscribe("tool.call.failure", lambda e: received.append(e))

    bus.emit(Event(
        type="tool.call.failure",
        source="test",
        severity=Severity.ERROR,
        payload={"tool_name": "search"}
    ))

    assert len(received) == 1
    assert received[0].type == "tool.call.failure"
```

### 10.2 Wildcard Matching

```python
def test_wildcard_matching():
    """Verify wildcard subscription patterns."""
    bus = EventBus()
    received = []

    bus.subscribe("tool.*", lambda e: received.append(e))

    bus.emit(Event(type="tool.call.failure", source="test", severity=Severity.ERROR))
    bus.emit(Event(type="tool.call.success", source="test", severity=Severity.INFO))
    bus.emit(Event(type="budget.token.warning", source="test", severity=Severity.WARNING))

    assert len(received) == 2  # Only tool.* events
```

### 10.3 Batching Within Window

```python
def test_batching_within_window():
    """Verify events are batched within time window."""
    accumulator = NotificationAccumulator()
    subscriber = create_test_subscriber(batching_window_ms=2000)
    accumulator.register_subscriber(subscriber)

    # Emit multiple events rapidly
    for i in range(5):
        event = Event(type="tool.call.failure", source="test", severity=Severity.ERROR)
        accumulator.accumulate(event, subscriber)

    # Should have single notification with 5 events (or batched)
    notifications = accumulator.drain_after_tool()
    # Behavior depends on whether batch max_size is reached
```

### 10.4 Deduplication

```python
def test_deduplication():
    """Verify duplicate events are filtered."""
    accumulator = NotificationAccumulator()
    subscriber = create_test_subscriber(dedupe_window_ms=5000)
    accumulator.register_subscriber(subscriber)

    event = Event(
        type="tool.call.failure",
        source="test",
        severity=Severity.ERROR,
        dedupe_key="tool.call.failure:search"
    )

    result1 = accumulator.accumulate(event, subscriber)
    result2 = accumulator.accumulate(event, subscriber)  # Same dedupe_key

    assert result1 is not None or result2 is None  # Second should be deduped
```

### 10.5 Persistence Across Restart

```python
def test_persistence_across_restart():
    """Verify cross-session notifications survive restart."""
    service = CrossSessionPersistenceService()

    # Store notification
    notification = CrossSessionNotification(
        user_id="user1",
        project_id="proj1",
        event_type="task.checkpoint",
        source="test",
        severity="info",
        payload={"checkpoint": "step_5"}
    )
    service.store(notification)

    # Simulate restart by creating new service instance
    service2 = CrossSessionPersistenceService()
    pending = service2.get_pending("user1", "proj1")

    assert len(pending) == 1
    assert pending[0].event_type == "task.checkpoint"
```

### 10.6 Integration with Oracle Agent

```python
def test_ans_oracle_integration():
    """Verify ANS events flow through Oracle agent correctly."""
    oracle = OracleAgent(user_id="test", project_id="test")
    received_events = []

    bus = get_event_bus()
    bus.subscribe_all(lambda e: received_events.append(e))

    # Trigger query processing (mocked)
    await oracle.process_query("Test question")

    # Verify expected events emitted
    event_types = [e.type for e in received_events]
    assert EventType.QUERY_START in event_types
    assert EventType.SESSION_END in event_types
```

### 10.7 BT Tick Triggering (Future)

```python
def test_bt_tick_triggering():
    """Verify events trigger tree ticks correctly."""
    scheduler = BTTickScheduler()
    tree = load_test_tree("oracle-agent.lisp")
    tick_count = [0]

    original_tick = tree.tick
    def counting_tick(*args, **kwargs):
        tick_count[0] += 1
        return original_tick(*args, **kwargs)
    tree.tick = counting_tick

    scheduler.register_tree(tree)

    # Emit tool success - should trigger tick
    get_event_bus().emit(Event(
        type=EventType.TOOL_CALL_SUCCESS,
        source="test",
        severity=Severity.INFO
    ))

    await asyncio.sleep(0.1)  # Allow scheduler to process
    assert tick_count[0] == 1
```

---

## 11. Recommendations

### 11.1 Immediate Changes Needed

1. **Add BT Event Types**: Extend `EventType` class with tree-specific events
2. **Create BTTickScheduler**: New component to manage event-to-tick mapping
3. **Implement Event Batching During Tick**: Prevent tick-event loops

### 11.2 Architecture Decisions

1. **Tick Trigger Strategy**: Use event-driven + polling hybrid
   - Events trigger initial tick
   - Polling at 100ms interval while tree is RUNNING

2. **Event Buffering**: Buffer events during tick, dispatch after
   - Prevents recursive tick triggering
   - Allows tree to see all events at once

3. **Priority Handling**: CRITICAL events should interrupt RUNNING trees
   - Budget exceeded -> immediate failure path
   - Loop detected -> recovery tree activation

### 11.3 Migration Path

1. Keep existing Oracle agent alongside new BT implementation
2. Run both in parallel during migration
3. Compare outputs to verify behavior equivalence
4. Gradually shift traffic to BT-based agent

---

## 12. References

- `backend/src/services/ans/bus.py` - EventBus implementation
- `backend/src/services/ans/event.py` - Event types and Severity
- `backend/src/services/ans/subscriber.py` - Subscriber configuration
- `backend/src/services/ans/accumulator.py` - Batching logic
- `backend/src/services/ans/deferred.py` - Deferred delivery
- `backend/src/services/ans/persistence.py` - Cross-session storage
- `backend/src/services/plugins/engine.py` - RuleEngine integration
- `backend/src/services/oracle_agent.py` - Current ANS usage
- `specs/019-bt-universal-runtime/spec.md` - BT Runtime specification
