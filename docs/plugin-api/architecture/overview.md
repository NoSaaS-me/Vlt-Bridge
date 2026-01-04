# Architecture Overview

The Oracle Plugin System is a rule engine that extends the Agent Notification System (ANS) to enable reactive and proactive agent behaviors.

## System Architecture

```
                              +------------------+
                              |   Oracle Agent   |
                              |  (oracle_agent.py)|
                              +--------+---------+
                                       |
                                       | emit(Event)
                                       v
+------------------+          +------------------+          +------------------+
|   Rule Loader    |          |    EventBus      |          |    ANS Core      |
|   (loader.py)    |          |    (bus.py)      |          |  (subscribers)   |
+--------+---------+          +--------+---------+          +------------------+
         |                             |
         | load rules                  | notify handlers
         v                             v
+------------------+          +------------------+
|   Rule Engine    |<---------|   Event Handler  |
|   (engine.py)    |          +------------------+
+--------+---------+
         |
         | evaluate conditions
         v
+------------------+          +------------------+
|   Expression     |          |   Lua Sandbox    |
|   Evaluator      |          |  (lua_sandbox.py)|
| (expression.py)  |          +------------------+
+--------+---------+                   ^
         |                             |
         | 80% TOML rules              | 20% complex logic
         |                             |
         +-------------+---------------+
                       |
                       | dispatch actions
                       v
              +------------------+
              | Action Dispatcher|
              |   (actions.py)   |
              +--------+---------+
                       |
         +-------------+-------------+
         |             |             |
         v             v             v
   +----------+  +----------+  +----------+
   | notify_  |  |   log    |  |set_state |
   |   self   |  |          |  |          |
   +----------+  +----------+  +----------+
```

## Component Interactions

### 1. Event Flow

The Oracle agent emits events at lifecycle points:

```python
# In oracle_agent.py
self._event_bus.emit(Event(
    type=EventType.QUERY_START,
    source="oracle_agent",
    severity=Severity.INFO,
    payload={"question": question, "user_id": user_id}
))
```

### 2. Event to Hook Mapping

Events map to hook points for rule triggering:

| ANS EventType | HookPoint |
|--------------|-----------|
| `QUERY_START` | `on_query_start` |
| `AGENT_TURN_START` | `on_turn_start` |
| `AGENT_TURN_END` | `on_turn_end` |
| `TOOL_CALL_PENDING` | `on_tool_call` |
| `TOOL_CALL_SUCCESS` | `on_tool_complete` |
| `TOOL_CALL_FAILURE` | `on_tool_failure` |
| `TOOL_CALL_TIMEOUT` | `on_tool_failure` |
| `SESSION_END` | `on_session_end` |

### 3. Rule Evaluation

```python
class RuleEngine:
    def evaluate_hook(self, hook: HookPoint, event: Event) -> HookEvaluationResult:
        rules = self.get_rules_for_hook(hook)  # Sorted by priority
        for rule in rules:
            if rule.script:
                result = self._lua_sandbox.execute(rule.script, context)
            else:
                result = self._evaluator.evaluate(rule.condition, context)
            if result:
                self._dispatcher.dispatch(rule.action, context)
```

### 4. Context Building

RuleContext is built from agent state for each evaluation:

```python
context = RuleContextBuilder(database_service).build(
    turn_number=5,
    token_usage=0.75,
    user_id="user-123",
    project_id="proj-456",
    event=event,
)
```

## Module Structure

```
backend/src/services/plugins/
+-- __init__.py           # Package exports
+-- rule.py               # Rule, RuleAction dataclasses, enums
+-- loader.py             # TOML discovery and parsing
+-- engine.py             # Event subscription, rule evaluation
+-- expression.py         # simpleeval wrapper for conditions
+-- lua_sandbox.py        # Sandboxed Lua execution via lupa
+-- actions.py            # Action dispatcher (notify, log, state, emit)
+-- context.py            # RuleContext and builder
+-- state.py              # SQLite state persistence
+-- rules/                # Built-in TOML rule files
    +-- token_budget.toml
    +-- iteration_budget.toml
    +-- large_result.toml
    +-- repeated_failure.toml
```

## Data Flow

### Rule Loading (Startup)

```
1. RuleLoader scans rules/ directory
2. Each .toml file is parsed and validated
3. Rules are organized by HookPoint in RuleEngine
4. Rules sorted by priority (highest first)
```

### Rule Evaluation (Runtime)

```
1. Agent emits Event via EventBus
2. RuleEngine receives event, maps to HookPoint
3. Build RuleContext from current agent state
4. For each rule in priority order:
   a. Evaluate condition (simpleeval or Lua)
   b. If matched, dispatch action
5. Return evaluation results
```

### State Persistence

```
1. set_state action triggered
2. ActionDispatcher calls PluginStateService
3. Value JSON-serialized and stored in SQLite
4. Next evaluation: RuleContextBuilder loads state
```

## Integration Points

### ANS Integration

The plugin system reuses ANS infrastructure:
- **EventBus**: Pub/sub for event distribution
- **Event types**: Extended with `QUERY_START`, `SESSION_END`
- **Notification format**: Compatible with ANS ToonFormatter

### Oracle Agent Integration

Hook points are emitted from specific locations in `oracle_agent.py`:
- Line ~759: `QUERY_START`
- Line ~1051: `AGENT_TURN_START`
- Line ~1080: `AGENT_TURN_END`
- Line ~1744: `TOOL_CALL_PENDING`
- Line ~1859: `TOOL_CALL_SUCCESS`
- Line ~1920: `TOOL_CALL_FAILURE`
- Line ~1131: `SESSION_END`

## See Also

- [Performance Considerations](./performance.md)
- [Roadmap](./roadmap.md)
- [Rule Format](../rules/format.md)
