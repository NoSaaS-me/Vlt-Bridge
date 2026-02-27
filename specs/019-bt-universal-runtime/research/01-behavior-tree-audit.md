# Behavior Tree Implementation Audit

**Date:** 2026-01-06
**Spec:** 019-bt-universal-runtime
**Author:** Claude Opus 4.5
**Purpose:** Deep audit of existing behavior tree implementation as foundation for universal runtime

---

## Executive Summary

The existing behavior tree implementation at `backend/src/services/plugins/behavior_tree/` is a **solid, well-designed foundation** with 3,602 lines of clean, documented Python code. It implements core BT patterns based on Honorbuddy/game bot research. However, several gaps exist relative to the spec requirements, particularly around:

1. **Blackboard scoping** - Current implementation is flat, needs hierarchy
2. **Async support** - `tick()` is synchronous, needs async adaptation
3. **Stuck detection** - Timeout decorator exists but no meta-watchdog
4. **Parallel semantics** - Missing REQUIRE_N, child cancellation, memory mode persistence

The implementation is **production-ready for its original scope** but requires enhancement for the universal runtime vision.

---

## File-by-File Analysis

### 1. `types.py` (287 lines)

**Purpose:** Core types for behavior tree system

#### Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `RunStatus` | Tri-state execution result | `__bool__()`, `from_bool(value)` |
| `TickContext` | Per-tick evaluation context | `get_cached()`, `set_cached()`, `clear_cache()` |
| `Blackboard` | Shared state between nodes | `get()`, `set()`, `has()`, `delete()`, `clear()`, `keys()`, `items()`, `copy()` |

#### Current TickContext vs Spec FR-4

**Current TickContext:**
```python
@dataclass
class TickContext:
    rule_context: "RuleContext"  # Plugin system context
    frame_id: int = 0            # Monotonic frame counter
    cache: Dict[str, Any]        # Per-tick cache
    blackboard: Optional["Blackboard"] = None
    delta_time_ms: float = 0.0   # Time since last tick
```

**Spec FR-4 Requirements:**
```
TickContext:
  - event: The triggering event (query, tool result, timer, etc.)
  - blackboard: Hierarchical state
  - services: Injected dependencies (LLM clients, tools, database)
  - tick_count: Number of ticks in current execution
  - tick_budget: Maximum ticks before yielding
  - start_time: When this tick cycle started
  - parent_path: Path of parent nodes (for debugging)
```

**Gap Analysis:**

| Field | Current | Spec | Status |
|-------|---------|------|--------|
| event | Via `rule_context` | Direct `event` field | PARTIAL - event buried in RuleContext |
| blackboard | Optional[Blackboard] | Hierarchical | MISSING hierarchy |
| services | Via `rule_context` | Direct injection | PARTIAL |
| tick_count | `frame_id` | `tick_count` | PRESENT (different name) |
| tick_budget | N/A | Required | MISSING |
| start_time | N/A | Required | MISSING |
| parent_path | N/A | Required for debugging | MISSING |
| async_pending | N/A | Required | MISSING |

#### Current Blackboard

The Blackboard implementation is **flat with optional namespace isolation**:

```python
class Blackboard:
    def __init__(self, namespace: str = "") -> None:
        self._data: Dict[str, Any] = {}
        self._namespace = namespace
```

**Missing vs FR-1:**
- No scope hierarchy (GLOBAL / TREE / SUBTREE)
- No parent reference for scope chain lookup
- No `set_global()` for explicit global writes
- No `snapshot()` for debugging/recovery
- No thread-safety for parallel nodes (uses plain dict)

**Positive:**
- Namespace isolation works via key prefixing
- Clean API with sensible defaults
- Copy method for state preservation

---

### 2. `node.py` (311 lines)

**Purpose:** Base classes for all behavior tree nodes

#### Classes

| Class | Base | Purpose | Key Methods |
|-------|------|---------|-------------|
| `BehaviorNode` | ABC | Abstract base for all nodes | `tick()`, `reset()`, `debug_info()` |
| `Composite` | BehaviorNode | Manages multiple children | `add_child()`, `add_children()`, `remove_child()`, `clear_children()` |
| `Decorator` | BehaviorNode | Wraps single child | `child` property |
| `Leaf` | BehaviorNode | Terminal nodes (no children) | N/A |

#### Tick Interface

```python
def tick(self, context: TickContext) -> RunStatus:
    """Execute the node and return its status."""
    self._tick_count += 1
    self._status = self._tick(context)
    return self._status

@abstractmethod
def _tick(self, context: TickContext) -> RunStatus:
    """Subclass implementation."""
    pass
```

**Observation:** `tick()` is **synchronous**. For async operations (LLM calls, tool execution), nodes return `RUNNING` and must be re-ticked. This is correct BT semantics but requires adaptation for async Python.

#### Reset Behavior

```python
def reset(self) -> None:
    """Reset the node to its initial state."""
    self._status = RunStatus.FAILURE
    # Don't reset tick_count - useful for debugging
```

Composite/Decorator `reset()` propagates to children - this is correct.

---

### 3. `composites.py` (470 lines)

**Purpose:** Composite nodes for child orchestration

#### Classes

| Class | Purpose | Memory-Aware? |
|-------|---------|---------------|
| `PrioritySelector` | First success wins (OR) | Yes - caches running child |
| `Sequence` | All must succeed (AND) | Yes - remembers current index |
| `Parallel` | Run all with policy | No - re-ticks all each frame |
| `MemorySelector` | Selector that continues from position | Yes - explicit memory mode |
| `MemorySequence` | Sequence that retries from failure | Yes - explicit memory mode |

#### PrioritySelector Memory

```python
class PrioritySelector(Composite):
    def __init__(self, ...):
        self._running_child_index: int = -1

    def _tick(self, context: TickContext) -> RunStatus:
        # Frame locking: resume from running child if valid
        start_index = 0
        if self._running_child_index >= 0:
            start_index = self._running_child_index
```

**Note:** This resumes from cached running child but may re-evaluate higher priority children depending on implementation. Standard BT semantics typically re-evaluate from root to detect higher-priority interrupts.

#### Sequence Memory

```python
class Sequence(Composite):
    def __init__(self, ...):
        self._current_child_index: int = 0

    def _tick(self, context: TickContext) -> RunStatus:
        for i in range(self._current_child_index, len(self._children)):
            # ... tick child ...
            if status == RunStatus.RUNNING:
                self._current_child_index = i
                return RunStatus.RUNNING
```

Sequence **correctly** remembers position across ticks.

#### Parallel Node vs Spec FR-7

**Current Implementation:**

```python
class ParallelPolicy(Enum):
    REQUIRE_ONE = auto()  # Success if any succeeds
    REQUIRE_ALL = auto()  # Success only if all succeed
```

**Spec FR-7 Requirements:**

| Feature | Current | Spec | Status |
|---------|---------|------|--------|
| REQUIRE_ALL | Yes | Yes | PRESENT |
| REQUIRE_ONE | Yes | Yes | PRESENT |
| REQUIRE_N(n) | No | Yes | MISSING |
| `:on-child-fail :cancel-siblings` | No | Yes | MISSING |
| `:on-child-fail :continue` | Implicit (current behavior) | Yes | PRESENT |
| `:on-child-fail :retry` | No | Yes | MISSING |
| `:memory true` | No | Yes | MISSING |
| `:max-concurrent N` | No | Yes | MISSING |

**Current Parallel Behavior:**

```python
def _tick(self, context: TickContext) -> RunStatus:
    # Tick ALL children every time
    self._child_statuses = []
    for child in self._children:
        status = child.tick(context)
        self._child_statuses.append(status)
```

**Issues:**
1. No cancellation - all children always ticked even after policy decision
2. No memory - doesn't remember which children completed across ticks
3. No concurrency limit - all children ticked immediately
4. No retry logic for failed children

---

### 4. `decorators.py` (779 lines)

**Purpose:** Decorators that modify child behavior

#### Classes

| Decorator | Purpose | Has State |
|-----------|---------|-----------|
| `Inverter` | Flip SUCCESS/FAILURE | No |
| `Succeeder` | Always SUCCESS | No |
| `Failer` | Always FAILURE | No |
| `UntilFail` | Repeat until child fails | Yes - iteration count |
| `UntilSuccess` | Repeat until child succeeds | Yes - iteration count |
| `Cooldown` | Rate limiting | Yes - last complete time/tick |
| `Guard` | Conditional execution | No (condition is stateless) |
| `Retry` | Retry on failure | Yes - attempt count |
| `Timeout` | Fail if too long | Yes - running start time/tick |
| `Repeat` | Execute N times | Yes - iteration count |

#### Stuck Detection (FR-5)

**Current:** The `Timeout` decorator provides basic stuck detection:

```python
class Timeout(Decorator):
    def __init__(self, ..., timeout_ticks: int = 0, timeout_ms: float = 0.0):
        self._timeout_ticks = timeout_ticks
        self._timeout_ms = timeout_ms
        self._running_start_tick: int = -1
        self._running_start_time_ms: float = 0.0

    def _tick(self, context: TickContext) -> RunStatus:
        # ... tick child ...
        if self._timeout_ticks > 0:
            ticks_running = current_tick - self._running_start_tick
            if ticks_running >= self._timeout_ticks:
                logger.warning(f"{self._name}: Timeout after {ticks_running} ticks")
                self._child.reset()
                return RunStatus.FAILURE
```

**Missing vs FR-5:**

| Feature | Current | Spec | Status |
|---------|---------|------|--------|
| Node-level timeout | Timeout decorator | Yes | PRESENT |
| Progress detection | No | "RUNNING without progress" | MISSING |
| Repeated failure pattern detection | No | "3+ times in window" | MISSING |
| Recovery tree trigger | No | Yes | MISSING |
| ANS event emission | No | Yes | MISSING |
| Blackboard snapshot on stuck | No | Yes | MISSING |
| Escalation to different tree | No | Yes | MISSING |

**Needed:** A meta-level `Watchdog` or `StuckDetector` that monitors ALL nodes, not just individually decorated ones.

#### Guard Decorator

```python
class Guard(Decorator):
    def __init__(self, ..., condition: ConditionFn = None, expression: str = None):
        self._condition = condition
        self._expression = expression
        self._evaluator = None  # Lazy-loaded ExpressionEvaluator

    def _evaluate_condition(self, context: TickContext) -> bool:
        if self._condition:
            return self._condition(context.rule_context)
        if self._expression:
            return self._evaluator.evaluate(self._expression, context.rule_context)
        return True  # No condition = pass
```

**Positive:** Supports both callable conditions and expression strings.

---

### 5. `leaves.py` (707 lines)

**Purpose:** Terminal nodes that perform actual work

#### Classes

| Leaf | Purpose | Returns RUNNING? |
|------|---------|------------------|
| `SuccessNode` | Always SUCCESS | No |
| `FailureNode` | Always FAILURE | No |
| `RunningNode` | Always RUNNING | Yes (for testing) |
| `ConditionNode` | Evaluate condition | No |
| `ActionNode` | Execute action | No (current impl) |
| `WaitNode` | Wait ticks/time | Yes |
| `ScriptNode` | Execute Lua script | No (current impl) |
| `BlackboardCondition` | Check blackboard value | No |
| `BlackboardSet` | Set blackboard value | No |
| `LogNode` | Log message | No |

#### ActionNode

```python
class ActionNode(Leaf):
    def __init__(self, ..., action: ActionCallable = None,
                 rule_action: RuleAction = None, dispatcher: ActionDispatcher = None):
        self._action = action
        self._rule_action = rule_action
        self._dispatcher = dispatcher

    def _tick(self, context: TickContext) -> RunStatus:
        success = self._execute(context)
        return RunStatus.from_bool(success)  # Never RUNNING
```

**Issue:** ActionNode never returns RUNNING. For async actions (tool calls, API requests), this needs adaptation.

#### ScriptNode

```python
class ScriptNode(Leaf):
    def _tick(self, context: TickContext) -> RunStatus:
        result = self._sandbox.execute(script, context.rule_context)
        return self._map_result(result)  # Never RUNNING
```

**Issue:** Same as ActionNode - synchronous execution only.

---

### 6. `tree.py` (511 lines)

**Purpose:** BehaviorTree wrapper with frame locking optimization

#### Classes

| Class | Purpose |
|-------|---------|
| `TickResult` | Result dataclass with status, frame_id, elapsed_ms, cache info |
| `BehaviorTree` | Main tree executor with tick(), reset(), metrics |
| `BehaviorTreeManager` | Registry for multiple trees by name |

#### Frame Locking

```python
class BehaviorTree:
    def __init__(self, ...):
        self._cached_running_node: Optional[BehaviorNode] = None
        self._cache_frame_id: int = -1
        self._cache_state_hash: int = 0

    def tick(self, rule_context: "RuleContext") -> TickResult:
        if self._can_use_cache(rule_context):
            used_cache = True
            self._cache_hits += 1
        else:
            self._cache_misses += 1
            self._cached_running_node = None
```

**State Hash for Cache Invalidation:**

```python
def _compute_state_hash(self, rule_context: "RuleContext") -> int:
    state_tuple = (
        rule_context.turn.number,
        rule_context.turn.iteration_count,
        len(rule_context.history.tools),
        rule_context.history.total_failures,
    )
    return hash(state_tuple)
```

**Issue:** Cache invalidation is tied to `RuleContext` fields. For universal runtime, this needs to be configurable or based on blackboard state.

#### Tick Loop

**Current:** Single tick, returns immediately.

**Spec FR-4 Pseudocode:**

```
while True:
    ctx.tick_count += 1
    status = tree.root.tick(ctx)

    if status == SUCCESS: return SUCCESS
    if status == FAILURE: return FAILURE
    if status == RUNNING:
        if ctx.tick_count >= ctx.tick_budget: yield
        if ctx.elapsed() > tree.max_tick_duration: timeout
        if ctx.has_pending_async(): await
        else: continue
```

**Gap:** Current implementation does NOT have the tick loop - caller must repeatedly call `tick()`. This is by design for integration with existing event loops, but spec expects managed loop.

---

### 7. `builder.py` (537 lines)

**Purpose:** Build trees from TOML rules or fluent API

#### Classes

| Class | Purpose |
|-------|---------|
| `TreeBuilderConfig` | Config with evaluator, dispatcher, sandbox, cooldown |
| `TreeBuilder` | Converts Rule objects to BehaviorTree |
| `DeclarativeTreeBuilder` | Fluent API for programmatic tree construction |

#### TreeBuilder Flow

```
Rules (TOML) -> TreeBuilder -> BehaviorTree
                    |
                    v
            PrioritySelector (if multiple)
                /     |     \
            Guard   Guard   Guard
              |       |       |
           Action  Action  Action
```

#### DeclarativeTreeBuilder API

```python
tree = (DeclarativeTreeBuilder("MyTree")
    .selector()
        .guard("context.turn.token_usage > 0.8")
            .action(notify_action)
        .end()
        .guard("context.turn.token_usage > 0.5")
            .action(log_action)
        .end()
    .end()
    .build())
```

**Question:** Is this API sufficient for LISP-generated trees?

**Analysis:**
- API is method-chain based, not data-driven
- LISP parser would need to call methods dynamically
- Better approach: Add `from_dict()` or `from_ast()` that takes tree structure as data
- Current API good for hand-written Python, needs adaptation for LISP

---

### 8. `__init__.py` (163 lines)

**Purpose:** Package exports and documentation

**Exports all relevant classes with good documentation including:**
- Performance targets (from behavior-tree-tasks.md)
- Example usage patterns
- Module organization

---

## Existing Test Coverage

### Test Files

| File | Lines | Coverage Focus |
|------|-------|----------------|
| `test_types.py` | 242 | RunStatus, TickContext, Blackboard |
| `test_composites.py` | 433 | PrioritySelector, Sequence, Parallel, Memory variants |
| `test_decorators.py` | 463 | All decorators |
| `test_tree.py` | 421 | BehaviorTree, frame locking, manager |
| `test_leaves.py` | 373 | All leaf nodes |
| `test_builder.py` | 368 | TreeBuilder, DeclarativeTreeBuilder |

**Total:** ~2,300 lines of tests

### What Tests Exist

**Good Coverage:**
- RunStatus boolean conversion
- Blackboard get/set/delete/namespace
- TickContext cache operations
- PrioritySelector short-circuit, priority order, RUNNING handling
- Sequence fail-fast, resume from RUNNING
- Parallel REQUIRE_ONE and REQUIRE_ALL policies
- All decorator status transformations
- Frame locking cache hits/misses
- TreeBuilder rule sorting and filtering

**Partial Coverage:**
- Memory mode composites (MemorySelector, MemorySequence) - basic tests
- Timeout decorator - tick-based only, not time-based reliability

### What Tests Are Missing

**For Current Implementation:**
1. Time-based timeout reliability under load
2. Cooldown decorator edge cases (reset during cooldown)
3. Complex nested tree structures
4. Concurrent tick calls (thread safety)
5. Large tree performance benchmarks

**For Spec Requirements (to be added):**

| Test Category | Description |
|---------------|-------------|
| Hierarchical Blackboard | Scope chain lookup, shadowing, global writes |
| Parallel REQUIRE_N | N-of-M success policy |
| Parallel Cancellation | Clean child cancellation on policy decision |
| Parallel Memory Mode | Persistence of child status across ticks |
| Stuck Detection | Progress monitoring, repeated failure patterns |
| Async Operations | Nodes returning RUNNING for async work |
| Tick Budget | Yield after N ticks |
| Hot Reload | Tree swap with RUNNING state |
| LISP Integration | Tree construction from S-expressions |
| LLM Nodes | Streaming, budget, interruption |

---

## Specific Questions Answered

### 1. Is there blackboard support? If not, where would it be added?

**Yes, there is Blackboard support.** Located in `types.py`.

**Enhancement needed:** Add hierarchical scoping. Suggested approach:

```python
class Blackboard:
    def __init__(self, scope: Scope = Scope.TREE, parent: "Blackboard" = None):
        self._scope = scope
        self._parent = parent
        self._data: Dict[str, Any] = {}

    def get(self, key: str, default=None):
        if key in self._data:
            return self._data[key]
        if self._parent:
            return self._parent.get(key, default)
        return default

    def set_global(self, key: str, value):
        bb = self
        while bb._parent:
            bb = bb._parent
        bb._data[key] = value
```

### 2. Are composites (Sequence, Selector, Parallel) memory-aware?

**Partially:**

- `PrioritySelector`: **Yes** - caches `_running_child_index`
- `Sequence`: **Yes** - tracks `_current_child_index`
- `Parallel`: **No** - re-ticks all children every frame
- `MemorySelector`: **Yes** - explicit memory mode
- `MemorySequence`: **Yes** - explicit memory mode

**Note:** "Memory-aware" for Parallel specifically means remembering which children completed (SUCCESS/FAILURE) so they're not re-ticked. Current Parallel always re-ticks all.

### 3. What decorators exist vs what FR-5 needs (stuck detection)?

**Existing Decorators:**
- Inverter, Succeeder, Failer (status modification)
- UntilFail, UntilSuccess (loop until condition)
- Cooldown (rate limiting)
- Guard (conditional execution)
- Retry (retry on failure)
- Timeout (tick/time limit)
- Repeat (N iterations)

**FR-5 Stuck Detection Needs:**

| Need | Current Solution | Gap |
|------|-----------------|-----|
| Node timeout | `Timeout` decorator | OK for individual nodes |
| Progress detection | None | MISSING - need to detect "RUNNING without change" |
| Repeated failure | None | MISSING - need pattern detection |
| Recovery tree | None | MISSING - need escalation mechanism |
| ANS integration | None | MISSING - need event emission |
| Full context snapshot | `debug_info()` partial | MISSING - need blackboard snapshot |

**Suggested Addition:** `Watchdog` decorator or tree-level `StuckDetector` service.

### 4. Is the builder API sufficient for LISP-generated trees?

**Partially sufficient, needs enhancement.**

**Current DeclarativeTreeBuilder:**
- Good for hand-written Python
- Uses method chaining with stack for nesting
- Not easily driven by data/AST

**Needed for LISP:**

```python
# Data-driven construction
def from_sexp(sexp: list) -> BehaviorNode:
    """
    Convert S-expression to tree.

    Example:
    (selector
      (guard "token_usage > 0.8"
        (action notify))
      (sequence
        (condition "has_tools")
        (action execute)))
    """
    node_type = sexp[0]
    args = sexp[1:]

    if node_type == 'selector':
        children = [from_sexp(child) for child in args]
        return PrioritySelector(children=children)
    elif node_type == 'guard':
        expr, child = args
        return Guard(expression=expr, child=from_sexp(child))
    # ... etc
```

### 5. What's the current reset/cleanup behavior?

**Reset Propagation:**

```python
# BehaviorNode
def reset(self) -> None:
    self._status = RunStatus.FAILURE
    # tick_count NOT reset (intentional for debugging)

# Composite
def reset(self) -> None:
    super().reset()
    for child in self._children:
        child.reset()

# Decorator
def reset(self) -> None:
    super().reset()
    if self._child:
        self._child.reset()

# BehaviorTree
def reset(self) -> None:
    if self._root:
        self._root.reset()
    self._blackboard.clear()
    self.invalidate_cache()
    self._frame_id = 0
```

**Good:** Clean propagation from root to leaves.

**Issue:** No async cleanup. If a node has pending async operations (e.g., HTTP request), `reset()` doesn't cancel them.

---

## Recommendations Summary

### High Priority (Block Spec Implementation)

1. **Hierarchical Blackboard** - Add scope chain, parent references, `set_global()`
2. **Async Support** - Add async variants of `tick()` or async action nodes
3. **Parallel Memory Mode** - Track completed children, don't re-tick
4. **Parallel Cancellation** - Clean child termination on policy decision

### Medium Priority (Needed for Full Spec)

5. **Tick Context Enhancement** - Add `tick_budget`, `start_time`, `parent_path`, `async_pending`
6. **Stuck Detection Service** - Meta-level watchdog beyond Timeout decorator
7. **LISP Builder** - `from_sexp()` or `from_ast()` for data-driven construction
8. **Hot Reload Support** - State preservation, tree swap, in-flight handling

### Low Priority (Enhancement)

9. **Thread-safe Blackboard** - Concurrent access protection
10. **Performance Benchmarks** - Validate against spec targets (<1ms tick overhead)
11. **Debug Infrastructure** - Breakpoints, step mode, blackboard watch

---

## Code Examples

### Current Tree Construction

```python
from backend.src.services.plugins.behavior_tree import (
    BehaviorTree, PrioritySelector, Sequence, Guard, ActionNode, ConditionNode
)

tree = BehaviorTree(
    root=PrioritySelector([
        Guard(
            expression="context.turn.token_usage > 0.8",
            child=ActionNode(action=lambda ctx: notify_high_usage(ctx)),
        ),
        Sequence([
            ConditionNode(condition=lambda ctx: ctx.has_tools),
            ActionNode(action=lambda ctx: execute_tools(ctx)),
        ]),
    ]),
    name="OracleMain",
)

# Execute
result = tree.tick(rule_context)
while result.status == RunStatus.RUNNING:
    # Caller manages loop
    result = tree.tick(rule_context)
```

### Current Parallel Usage

```python
from backend.src.services.plugins.behavior_tree import Parallel, ParallelPolicy

parallel = Parallel(
    children=[
        ActionNode(action=search_web),
        ActionNode(action=search_code),
        ActionNode(action=search_docs),
    ],
    policy=ParallelPolicy.REQUIRE_ONE,  # Succeed on first result
)

# All children ticked every frame
result = parallel.tick(context)
```

### Current Decorator Stack

```python
from backend.src.services.plugins.behavior_tree import Guard, Cooldown, Retry, Timeout

# Retry up to 3 times, with 5 second cooldown between attempts,
# timeout after 10 ticks, only if condition passes
node = Guard(
    expression="context.turn.iteration_count < 10",
    child=Cooldown(
        cooldown_ms=5000,
        child=Retry(
            max_attempts=3,
            child=Timeout(
                timeout_ticks=10,
                child=ActionNode(action=call_llm),
            ),
        ),
    ),
)
```

---

## Conclusion

The existing behavior tree implementation is a **strong foundation** with:
- Clean architecture following BT best practices
- Good test coverage (~2,300 lines)
- Solid core node types and decorators
- Frame locking optimization

Key gaps for universal runtime:
1. **Blackboard hierarchy** - Current flat, needs scoping
2. **Parallel semantics** - Missing memory mode, cancellation, REQUIRE_N
3. **Stuck detection** - Node-level only, needs meta-watchdog
4. **Async support** - Synchronous tick, needs async adaptation
5. **LISP integration** - Method-chain API, needs data-driven builder

**Estimated enhancement effort:**
- Core enhancements (blackboard, parallel): 2-3 days
- Async support: 1-2 days
- LISP integration: 2-3 days
- Stuck detection service: 1-2 days
- Testing and validation: 2-3 days

**Total:** ~8-13 days of focused development to bring existing implementation to spec requirements.
