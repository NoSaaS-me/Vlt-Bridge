# Plugin System Audit for BT Integration

**Date**: 2026-01-06
**Purpose**: Analyze the existing Plugin system to understand how rules and Lua integrate with the BT runtime per spec 019-bt-universal-runtime.

---

## 1. Executive Summary

The existing Plugin system provides a solid foundation for BT integration:

- **RuleEngine** already maps events to hook points and evaluates rules in priority order
- **Behavior Tree** implementation is complete with composites, decorators, leaves, and frame locking
- **Lua Sandbox** provides secure script execution with RuleContext exposure
- **State Management** offers SQLite-based persistence scoped by user/project/plugin

**Key Finding**: The BT implementation already exists and is well-designed. The main work for spec 019 is:
1. Adding LISP tree definition language (parser + loader)
2. Creating LLM-aware nodes (streaming, budget, interruptibility)
3. Implementing hierarchical blackboard with scoping
4. Hot reload infrastructure
5. Stuck detection and recovery

---

## 2. RuleEngine Analysis

**Location**: `backend/src/services/plugins/engine.py` (815 lines)

### How It Processes Events

```
Event (from ANS EventBus)
    |
    v
EVENT_TO_HOOK mapping
    |
    v
evaluate_hook(hook, event, context)
    |
    v
For each rule in priority order:
    |
    +--> Expression condition? --> ExpressionEvaluator.evaluate()
    |                                    |
    +--> Lua script? ----------> LuaSandbox.execute()
                                         |
                                         v
                              Action dispatch via ActionDispatcher
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `start()` | Subscribe to ANS events |
| `stop()` | Unsubscribe from events |
| `evaluate_hook()` | Evaluate all rules for a hook point |
| `_evaluate_rule()` | Evaluate single rule (expression or script) |
| `_evaluate_script_rule()` | Handle Lua script evaluation |
| `reload_rules()` | Reload rules from TOML files |

### Event-to-Hook Mapping

```python
EVENT_TO_HOOK = {
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

### BT Integration Potential

The RuleEngine currently uses a linear priority-based evaluation. The existing `TreeBuilder` class can convert rules into BT structures:

```python
# From builder.py
tree = builder.build_from_rules(rules, hook=HookPoint.ON_TURN_START)
```

**Recommendation**: Replace linear rule evaluation with BT tick in `evaluate_hook()`.

---

## 3. Rule Model Analysis

**Location**: `backend/src/services/plugins/rule.py` (222 lines)

### Rule Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique kebab-case identifier |
| `name` | `str` | Human-readable name |
| `description` | `str` | What the rule does |
| `version` | `str` | Semantic version (default: "1.0.0") |
| `trigger` | `HookPoint` | When rule fires |
| `condition` | `Optional[str]` | simpleeval expression |
| `script` | `Optional[str]` | Lua script path |
| `action` | `Optional[RuleAction]` | What happens when fired |
| `priority` | `int` | 1-1000, higher fires first |
| `enabled` | `bool` | Is rule active |
| `core` | `bool` | Cannot be disabled |
| `plugin_id` | `Optional[str]` | Parent plugin |
| `source_path` | `str` | TOML file location |

### Hook Points (Triggers)

```python
class HookPoint(str, Enum):
    ON_QUERY_START = "on_query_start"
    ON_TURN_START = "on_turn_start"
    ON_TURN_END = "on_turn_end"
    ON_TOOL_CALL = "on_tool_call"
    ON_TOOL_COMPLETE = "on_tool_complete"
    ON_TOOL_FAILURE = "on_tool_failure"
    ON_SESSION_END = "on_session_end"
```

### Action Types

```python
class ActionType(str, Enum):
    NOTIFY_SELF = "notify_self"  # Inject notification
    LOG = "log"                   # Write to log
    SET_STATE = "set_state"       # Plugin state
    EMIT_EVENT = "emit_event"     # ANS event
```

### BT Integration Potential

Rules map naturally to BT leaf nodes:
- `condition` -> `ConditionNode` or `Guard` decorator
- `script` -> `ScriptNode`
- `action` -> `ActionNode`

**Gap**: Need new action types for BT-specific operations (blackboard write, subtree invoke).

---

## 4. Expression Evaluation

**Location**: `backend/src/services/plugins/expression.py` (425 lines)

### Features

- Uses `simpleeval` library for safe expression evaluation
- Expression caching via `@lru_cache` (max 256 expressions)
- Timing instrumentation for performance debugging
- Context-specific helper functions

### Safe Functions Exposed

```python
# Type conversions
int, float, str, bool

# Math functions
abs, min, max, round, sum

# Collection functions
len, any, all, sorted, reversed

# Type checks
isinstance, type
```

### Context Helper Functions

```python
tool_completed(name) -> bool     # Has tool completed successfully?
tool_failed(name) -> bool        # Has tool failed?
failure_count(name) -> int       # How many failures?
context_above_threshold(t) -> bool  # Context usage > threshold?
message_count_above(n) -> bool   # Message count > n?
```

### Security

- No `__dunder__` attribute access
- No import/exec capabilities
- Limited function whitelist
- No file/network operations

### BT Integration Potential

The ExpressionEvaluator is already used by:
- `ConditionNode` in `leaves.py`
- `Guard` decorator in `decorators.py`

**Works as-is for BT leaf conditions.**

---

## 5. Lua Sandbox Analysis

**Location**: `backend/src/services/plugins/lua_sandbox.py` (593 lines)

### Security Model

**Blocked globals**:
```python
BLOCKED_GLOBALS = {
    "os", "io", "debug",           # System access
    "dofile", "loadfile", "load",  # Code loading
    "rawget", "rawset",            # Raw access
    "require", "module", "package", # Module system
    "collectgarbage",              # GC control
    "setmetatable", "getmetatable", # Metatable manipulation
    "coroutine",                   # Side effects
}
```

**Allowed modules** (filtered functions):
- `string`: byte, char, find, format, gmatch, gsub, len, lower, match, rep, reverse, sub, upper
- `table`: concat, insert, maxn, remove, sort, unpack
- `math`: abs, acos, asin, atan, ceil, cos, exp, floor, log, max, min, pow, sin, sqrt, tan, random, etc.

### Timeout Enforcement

```python
def execute(self, script: str, context: RuleContext) -> Any:
    # Uses threading with timeout
    thread = threading.Thread(target=run_script, daemon=True)
    thread.start()
    thread.join(timeout=self.timeout_seconds)  # Default: 5.0s

    if thread.is_alive():
        raise LuaTimeoutError(...)
```

### Context Exposure to Lua

RuleContext is converted to nested Lua tables:
```lua
context.turn.number
context.turn.token_usage
context.turn.context_usage
context.history.tools[i].name
context.history.failures[tool_name]
context.user.id
context.project.id
context.state[key]
context.event.type
context.event.payload[key]
```

### Script Return Values

| Return | Interpretation |
|--------|----------------|
| `nil` | Rule did not match |
| `true` | Use rule's defined action |
| `{type="notify_self", message="..."}` | Execute returned action |

### BT Integration Potential

The `ScriptNode` in `leaves.py` already integrates with LuaSandbox:
```python
class ScriptNode(Leaf):
    def _tick(self, context: TickContext) -> RunStatus:
        result = self._sandbox.execute(script, context.rule_context)
        return self._map_result(result)
```

**Works as-is for BT Lua leaves.**

**Gap**: For BT, need ability to:
1. Write to blackboard from Lua
2. Return RUNNING status for async operations
3. Access blackboard state (not just RuleContext)

---

## 6. State Management

**Location**: `backend/src/services/plugins/state.py` (295 lines)

### Scoping

State is scoped by:
- `user_id`: Per-user isolation
- `project_id`: Per-project isolation
- `plugin_id`: Per-plugin isolation
- `key`: Individual state keys

### Database Schema

```sql
CREATE TABLE plugin_state (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    plugin_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, project_id, plugin_id, key)
);
```

### API

```python
class PluginStateService:
    def get(user_id, project_id, plugin_id, key, default=None) -> Any
    def set(user_id, project_id, plugin_id, key, value) -> None
    def get_all(user_id, project_id, plugin_id) -> dict
    def clear(user_id, project_id, plugin_id) -> None
    def delete(user_id, project_id, plugin_id, key) -> bool
```

### BT Integration Potential

This provides the persistence layer for BT's **global blackboard**:
- Global blackboard = plugin_state with well-known keys
- Tree-local = in-memory only
- Subtree-local = in-memory with scope prefix

**Recommendation**: Use PluginStateService for global blackboard persistence.

---

## 7. Existing Behavior Tree Implementation

**Location**: `backend/src/services/plugins/behavior_tree/` (7 files, ~3,600 lines)

### Core Types (`types.py`)

```python
class RunStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()  # Enables multi-tick operations

@dataclass
class TickContext:
    rule_context: RuleContext
    frame_id: int
    cache: Dict[str, Any]
    blackboard: Optional[Blackboard]
    delta_time_ms: float

class Blackboard:
    # Namespaced key-value store
    def get(key, default=None) -> Any
    def set(key, value) -> None
    def has(key) -> bool
    def delete(key) -> bool
    def keys() -> list[str]
```

### Node Hierarchy (`node.py`)

```
BehaviorNode (abstract)
    |
    +-- Composite (manages multiple children)
    |
    +-- Decorator (wraps single child)
    |
    +-- Leaf (terminal, performs work)
```

### Composites (`composites.py`)

| Node | Behavior |
|------|----------|
| `PrioritySelector` | First success wins, priority order |
| `Sequence` | All must succeed (AND) |
| `Parallel` | Run all with REQUIRE_ONE/REQUIRE_ALL policy |
| `MemorySelector` | Selector with progress memory |
| `MemorySequence` | Sequence with progress memory |

### Decorators (`decorators.py`)

| Node | Behavior |
|------|----------|
| `Inverter` | Flip SUCCESS/FAILURE |
| `Succeeder` | Always return SUCCESS |
| `Failer` | Always return FAILURE |
| `UntilFail` | Repeat until child fails |
| `UntilSuccess` | Repeat until child succeeds |
| `Cooldown` | Rate limiting (time or ticks) |
| `Guard` | Conditional execution |
| `Retry` | Retry on failure |
| `Timeout` | Time limit |
| `Repeat` | Fixed repetitions |

### Leaves (`leaves.py`)

| Node | Purpose |
|------|---------|
| `SuccessNode` | Always SUCCESS |
| `FailureNode` | Always FAILURE |
| `RunningNode` | Always RUNNING |
| `ConditionNode` | Expression evaluation |
| `ActionNode` | Action execution |
| `WaitNode` | Time/tick delay |
| `ScriptNode` | Lua script execution |
| `BlackboardCondition` | Check blackboard value |
| `BlackboardSet` | Write blackboard value |
| `LogNode` | Debug logging |

### Frame Locking (`tree.py`)

```python
class BehaviorTree:
    def tick(self, rule_context) -> TickResult:
        # Frame locking optimization
        if self._can_use_cache(rule_context):
            status = self._cached_running_node.tick(context)
        else:
            status = self._root.tick(context)

        if status == RunStatus.RUNNING:
            self._update_cache(rule_context)
        else:
            self._cached_running_node = None
```

### Tree Builder (`builder.py`)

```python
class TreeBuilder:
    def build_from_rules(rules, hook, name) -> BehaviorTree
    def build_from_rule(rule, name) -> BehaviorTree
    def build_hook_trees(rules) -> dict[HookPoint, BehaviorTree]

class DeclarativeTreeBuilder:
    # Fluent API for programmatic tree construction
    (DeclarativeTreeBuilder("MyTree")
        .selector()
            .guard("context.turn.token_usage > 0.8")
                .action(notify_action)
            .end()
        .end()
        .build())
```

### What's Missing for Spec 019

| Requirement | Current State |
|-------------|---------------|
| LISP tree definition | Not implemented |
| Hierarchical blackboard (global/tree/subtree) | Single flat blackboard |
| LLM-aware nodes | Not implemented |
| Hot reload | Not implemented |
| Stuck detection | Not implemented |
| Async coordination | Basic RUNNING support only |

---

## 8. Hook Points: Current vs BT Needs

### Current Hook Points (7)

| Hook | When | Current Use |
|------|------|-------------|
| `ON_QUERY_START` | New user query | Initialize session |
| `ON_TURN_START` | Before agent turn | Budget checks |
| `ON_TURN_END` | After agent turn | State cleanup |
| `ON_TOOL_CALL` | Before tool | Validation |
| `ON_TOOL_COMPLETE` | After tool success | Result processing |
| `ON_TOOL_FAILURE` | Tool error | Error handling |
| `ON_SESSION_END` | Session close | Cleanup |

### Additional Hooks Needed for BT

| Hook | When | Use Case |
|------|------|----------|
| `ON_LLM_START` | Before LLM call | Inject system prompts |
| `ON_LLM_CHUNK` | Streaming chunk | Progress updates |
| `ON_LLM_COMPLETE` | After LLM | Response processing |
| `ON_BUDGET_WARNING` | Near limit | Wrap-up signals |
| `ON_STUCK_DETECTED` | Node timeout | Recovery triggers |

---

## 9. Integration Points: Rules as BT Nodes

### Rules as Leaf Conditions

**Current flow**:
```
Rule.condition (simpleeval) -> ExpressionEvaluator -> bool
```

**BT integration**:
```python
# ConditionNode already supports this
class ConditionNode(Leaf):
    def _tick(self, context: TickContext) -> RunStatus:
        result = self._evaluator.evaluate(self._expression, context.rule_context)
        return RunStatus.from_bool(result)
```

### Lua Scripts as Leaf Actions

**Current flow**:
```
Rule.script (path) -> LuaSandbox.execute() -> dict/bool/nil
```

**BT integration**:
```python
# ScriptNode already supports this
class ScriptNode(Leaf):
    def _tick(self, context: TickContext) -> RunStatus:
        result = self._sandbox.execute(script, context.rule_context)
        return self._map_result(result)
```

### Plugin State as Blackboard Layer

**Mapping**:
```
Global Blackboard    -> PluginStateService (persisted)
Tree-local Blackboard -> In-memory Blackboard (cleared on tree completion)
Subtree-local        -> In-memory with prefix (cleared on subtree completion)
```

---

## 10. LISP Parser Research

### Requirements from Spec 019

The LISP subset needed is minimal:
1. S-expressions for tree structure
2. Symbols (node names, keywords)
3. Strings (messages, expressions)
4. Numbers (priorities, timeouts)
5. Keyword arguments (`:key value`)
6. Lists (children)

Example from spec:
```lisp
(tree "oracle-agent"
  :description "Main chat agent workflow"
  :blackboard-schema {:context nil :response nil :tools []}

  (sequence
    (action load-context :fn "oracle.load_context")
    (llm-call :model "claude-sonnet-4" :stream-to [:partial-response])))
```

### Library Options

#### 1. sexpdata (Recommended)

**Pros**:
- Purpose-built for S-expressions
- Simple API: `sexpdata.load()`, `sexpdata.dump()`
- BSD license
- 5,333 weekly downloads
- No security vulnerabilities

**Cons**:
- Inactive maintenance (no releases in 12 months)
- Minimal feature set

**Usage**:
```python
import sexpdata
tree = sexpdata.loads('(sequence (action foo) (action bar))')
# Returns: [Symbol('sequence'), [Symbol('action'), Symbol('foo')], ...]
```

**Source**: [sexpdata PyPI](https://pypi.org/project/sexpdata/), [GitHub](https://github.com/jd-boyd/sexpdata)

#### 2. Lark (Alternative)

**Pros**:
- Actively maintained (v1.3.1, October 2025)
- Highly flexible (can define any grammar)
- Earley and LALR(1) parsers
- Better error messages

**Cons**:
- Requires writing S-expression grammar
- More complex setup
- Overkill for simple S-expressions

**Source**: [lark PyPI](https://pypi.org/project/lark/), [GitHub](https://github.com/lark-parser/lark)

#### 3. Hy (Not Recommended)

**Pros**:
- Full Lisp implementation
- Compiles to Python AST

**Cons**:
- Too heavyweight for our needs
- Brings in full Lisp semantics
- Dependency on Hy runtime

**Source**: [Hy Documentation](https://leanpub.com/hy-lisp-python/read)

#### 4. Custom Parser (Fallback)

**Pros**:
- Exactly what we need
- No external dependencies
- ~100-200 lines of code

**Cons**:
- Development/maintenance burden
- Need to handle edge cases

### Recommendation

**Primary choice: sexpdata**

Rationale:
1. Perfect fit for S-expression parsing
2. Simple, proven, stable
3. Inactive maintenance is acceptable (S-expression syntax is stable)
4. Easy to vendor if needed

**Fallback: Custom parser**

If sexpdata proves insufficient:
```python
def parse_sexp(s: str) -> Any:
    """Minimal S-expression parser."""
    # Tokenize: '(', ')', strings, symbols, numbers
    # Recursive descent parse
    # ~150 lines
```

### Validation Strategy

1. **Syntax validation**: Parse LISP at load time
2. **Reference validation**: Check `:fn` references exist
3. **Schema validation**: Verify node types and required properties
4. **Circular reference detection**: Track tree includes

---

## 11. E2E Test Scenarios

### Scenario 1: Rule Loading from TOML

```python
def test_rule_loading():
    loader = RuleLoader(Path("rules/"))
    rules = loader.load_all(skip_invalid=True)

    assert len(rules) > 0
    for rule in rules:
        assert rule.id
        assert rule.trigger in HookPoint
        assert rule.condition or rule.script
```

### Scenario 2: Condition Evaluation

```python
def test_condition_evaluation():
    evaluator = ExpressionEvaluator()
    context = RuleContext.create_minimal("user1", "project1")
    context.turn.token_usage = 0.85

    result = evaluator.evaluate("context.turn.token_usage > 0.8", context)
    assert result is True
```

### Scenario 3: Lua Script Execution

```python
def test_lua_execution():
    sandbox = LuaSandbox(timeout_seconds=5.0)
    context = RuleContext.create_minimal("user1", "project1")

    script = """
    if context.turn.token_usage > 0.8 then
        return {type = "notify_self", message = "High usage!"}
    end
    return nil
    """

    result = sandbox.execute(script, context)
    assert result is None  # token_usage is 0.0 by default
```

### Scenario 4: State Persistence

```python
def test_state_persistence():
    service = PluginStateService(database_service)

    service.set("user1", "proj1", "plugin1", "counter", 42)
    value = service.get("user1", "proj1", "plugin1", "counter")

    assert value == 42

    # Persistence across service instances
    service2 = PluginStateService(database_service)
    value2 = service2.get("user1", "proj1", "plugin1", "counter")
    assert value2 == 42
```

### Scenario 5: BT Tick with Hook Point

```python
def test_bt_hook_evaluation():
    builder = TreeBuilder(TreeBuilderConfig(
        evaluator=ExpressionEvaluator(),
        dispatcher=ActionDispatcher(),
    ))

    rules = [
        Rule(id="rule1", condition="context.turn.token_usage > 0.8", ...),
        Rule(id="rule2", condition="context.turn.token_usage > 0.5", ...),
    ]

    tree = builder.build_from_rules(rules, hook=HookPoint.ON_TURN_START)

    context = RuleContext.create_minimal("user1", "project1")
    context.turn.token_usage = 0.75

    result = tree.tick(context)
    assert result.status == RunStatus.SUCCESS  # rule2 matches
```

---

## 12. Recommendations for Spec 019 Implementation

### High Priority (Must Have)

1. **LISP Parser**: Use `sexpdata` with custom transformer
2. **LLM Nodes**: Create `LLMCallNode` with streaming, budget, timeout
3. **Hierarchical Blackboard**: Extend `Blackboard` with parent chain
4. **Hot Reload**: Watch files + swap trees safely

### Medium Priority (Should Have)

1. **Stuck Detection**: Watchdog thread with timeout tracking
2. **New Hook Points**: `ON_LLM_*`, `ON_BUDGET_*`
3. **Async Coordination**: Proper `RUNNING` state management

### Lower Priority (Nice to Have)

1. **Debugging Infrastructure**: Breakpoints, step mode
2. **Tree Visualization**: ASCII or JSON export
3. **Performance Metrics**: Per-node timing

### Integration Checklist

- [ ] Add sexpdata dependency
- [ ] Create LispTreeLoader class
- [ ] Create LLMCallNode class
- [ ] Extend Blackboard for hierarchical scoping
- [ ] Add global blackboard persistence via PluginStateService
- [ ] Implement hot reload in BehaviorTreeManager
- [ ] Add stuck detection watchdog
- [ ] Create new hook points for LLM events
- [ ] Migrate Oracle agent to BT structure
- [ ] Write comprehensive tests

---

## 13. File Reference

| File | Purpose | Lines |
|------|---------|-------|
| `engine.py` | RuleEngine orchestrator | 815 |
| `rule.py` | Rule model definitions | 222 |
| `context.py` | RuleContext and builder | 631 |
| `loader.py` | TOML rule loading | 244 |
| `expression.py` | simpleeval wrapper | 425 |
| `actions.py` | Action dispatcher | 303 |
| `lua_sandbox.py` | Secure Lua execution | 593 |
| `state.py` | Plugin state persistence | 295 |
| `behavior_tree/types.py` | RunStatus, TickContext, Blackboard | 288 |
| `behavior_tree/node.py` | BehaviorNode base classes | 312 |
| `behavior_tree/composites.py` | Selector, Sequence, Parallel | 471 |
| `behavior_tree/decorators.py` | Guard, Cooldown, Retry, etc. | ~400 |
| `behavior_tree/leaves.py` | Condition, Action, Script nodes | 708 |
| `behavior_tree/tree.py` | BehaviorTree with frame locking | 512 |
| `behavior_tree/builder.py` | TreeBuilder from rules | 538 |

**Total Plugin System**: ~5,200 lines of Python
