# Behavior Tree Implementation Tasks

> **Decision**: Custom Cython implementation (py_trees is pure Python, targets 50-200ms latency)
> **Pattern**: Honorbuddy-style with PrioritySelector, frame locking, RunStatus

## Phase BT-0: Design Foundations

**Purpose**: Establish core types and enums before any composite node implementation

- [ ] BT001 [P] Create RunStatus enum (Success, Failure, Running) in `backend/src/services/plugins/behavior_tree/types.pyx`
- [ ] BT002 [P] Create TickContext cdef class with agent_state, event_payload, cache, frame_id
- [ ] BT003 [P] Create Blackboard cdef class for shared state between nodes
- [ ] BT004 Create abstract BehaviorNode base class with tick(), reset(), status property

**Test**: Unit test for TickContext creation and Blackboard get/set

---

## Phase BT-1: Core Composites

**Purpose**: Implement fundamental composite nodes that orchestrate child evaluation

- [ ] BT005 Implement Composite base class (children management, reset propagation)
- [ ] BT006 Implement PrioritySelector (first success wins, short-circuit)
- [ ] BT007 Implement Sequence (all must succeed, fail-fast)
- [ ] BT008 Implement Parallel (run all children, configurable success policy)

**Tests**:
- [ ] BT009 [P] Unit test for PrioritySelector (first match wins, priority ordering)
- [ ] BT010 [P] Unit test for Sequence (all success, fail-fast on first failure)
- [ ] BT011 [P] Unit test for Parallel (configurable success threshold)

---

## Phase BT-2: Decorators

**Purpose**: Implement node wrappers that modify child behavior

- [ ] BT012 Create Decorator base class (single child wrapper)
- [ ] BT013 Implement Inverter decorator (flip Success/Failure)
- [ ] BT014 Implement Succeeder decorator (always return Success)
- [ ] BT015 Implement UntilFail decorator (repeat until Failure)
- [ ] BT016 Implement Cooldown decorator (suppress re-evaluation for N frames/ms)
- [ ] BT017 Implement Guard decorator (condition wrapper that gates child execution)

**Tests**:
- [ ] BT018 [P] Unit test for Inverter (Success->Failure, Failure->Success)
- [ ] BT019 [P] Unit test for Cooldown (rate limiting behavior)
- [ ] BT020 [P] Unit test for Guard (condition gating)

---

## Phase BT-3: Leaf Nodes

**Purpose**: Implement terminal nodes that perform actual work

- [ ] BT021 Create LeafNode base class
- [ ] BT022 Implement ConditionNode (evaluates expression, returns Success/Failure)
- [ ] BT023 Implement ActionNode (executes action, returns status)
- [ ] BT024 Implement WaitNode (returns Running for N ticks, then Success)
- [ ] BT025 Implement ScriptNode (executes Lua script, maps return to RunStatus)

**Tests**:
- [ ] BT026 [P] Unit test for ConditionNode with simpleeval expressions
- [ ] BT027 [P] Unit test for ActionNode execution and status mapping
- [ ] BT028 [P] Unit test for WaitNode tick counting

---

## Phase BT-4: Frame Locking Optimization

**Purpose**: Implement Honorbuddy-style optimization to avoid O(n) tree traversal each tick

- [ ] BT029 Add `_running_node` reference to tree root for cached running state
- [ ] BT030 Implement `tick_optimized()` that resumes from cached running node
- [ ] BT031 Implement `invalidate_cache()` for state change detection
- [ ] BT032 Add frame_id tracking to prevent stale cache issues

**Tests**:
- [ ] BT033 Unit test for frame locking (verify O(1) resume vs O(n) fresh tick)
- [ ] BT034 Unit test for cache invalidation on state change

---

## Phase BT-5: TOML Tree Builder

**Purpose**: Convert TOML rule definitions to behavior tree structures

- [ ] BT035 Define BehaviorTreeSchema (TOML structure for trees)
- [ ] BT036 Implement TreeBuilder class with `from_toml()` method
- [ ] BT037 Add rule-to-tree mapping: `condition` -> Guard, `actions` -> Sequence
- [ ] BT038 Add priority-based rule ordering into PrioritySelector
- [ ] BT039 Add script reference handling (Lua nodes) in tree builder

**Tests**:
- [ ] BT040 [P] Unit test for simple TOML rule -> tree conversion
- [ ] BT041 [P] Unit test for multi-rule priority ordering
- [ ] BT042 [P] Unit test for script reference tree building

---

## Phase BT-6: Oracle Agent Integration

**Purpose**: Wire behavior tree evaluation into oracle_agent.py hook points

- [ ] BT043 Create BehaviorTreeEngine class that wraps tree with hook point subscription
- [ ] BT044 Add `register_hook()` method to map hook points to tree ticks
- [ ] BT045 Implement `on_event()` handler that creates TickContext from ANS Event
- [ ] BT046 Add tree tick scheduling (sync vs async evaluation)
- [ ] BT047 Wire BehaviorTreeEngine into RuleEngine

**Tests**:
- [ ] BT048 Integration test for hook point -> tree tick -> action dispatch
- [ ] BT049 Integration test for budget warning rule firing via behavior tree

---

## Phase BT-7: Context Detection (Singular Pattern)

**Purpose**: Implement environment-aware behavior switching (from Honorbuddy's Singular)

- [ ] BT050 Create ContextDetector class for environment classification
- [ ] BT051 Implement detection rules (high-token-usage, research-mode, error-recovery)
- [ ] BT052 Add context-based tree selection to BehaviorTreeEngine

**Test**:
- [ ] BT053 Unit test for context detection and tree switching

---

## Injection Points (modifications to existing tasks.md)

### T033 (Phase 4: US2 - RuleEngine with EventBus subscription)
**Modification**: Delegate to BehaviorTreeEngine for rule evaluation instead of direct condition checking.

### T034 (Phase 4: US2 - Rule evaluation logic)
**Modification**: Replace direct expression evaluation with `tree.tick(context)` call.

### T035 (Phase 4: US2 - Priority-ordered execution)
**Modification**: Remove manual priority sorting; PrioritySelector handles this structurally.

### T022 (Phase 3: US1 - ExpressionEvaluator)
**Modification**: Ensure `evaluate(expr, context) -> bool` signature works for ConditionNode.

### T028-T030 (Phase 3: US1 - ActionDispatcher)
**Modification**: Add `execute() -> bool` return value consumed by ActionNode.

### T057 (Phase 6: US4 - Script execution)
**Modification**: Remove from RuleEngine; ScriptNode handles via LuaSandbox.

---

## Directory Structure (Cython)

```
backend/src/services/plugins/
├── behavior_tree/
│   ├── __init__.py           # Python interface exports
│   ├── types.pyx             # RunStatus, TickContext, Blackboard (Cython)
│   ├── types.pxd             # Cython declarations
│   ├── node.pyx              # BehaviorNode base class (Cython)
│   ├── node.pxd              # Cython declarations
│   ├── composites.pyx        # PrioritySelector, Sequence, Parallel
│   ├── decorators.pyx        # Inverter, Guard, Cooldown
│   ├── leaves.pyx            # ConditionNode, ActionNode, ScriptNode
│   ├── tree.pyx              # BehaviorTree with frame locking
│   ├── builder.py            # TreeBuilder (pure Python, TOML parsing)
│   ├── engine.py             # BehaviorTreeEngine (pure Python, orchestration)
│   └── context.py            # ContextDetector (pure Python)
├── setup.py                  # Cython build configuration
└── ...
```

---

## Cython Build Setup

```python
# backend/src/services/plugins/setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "behavior_tree/types.pyx",
        "behavior_tree/node.pyx",
        "behavior_tree/composites.pyx",
        "behavior_tree/decorators.pyx",
        "behavior_tree/leaves.pyx",
        "behavior_tree/tree.pyx",
    ], language_level="3"),
)
```

---

## Task Summary

| Phase | Tasks | Parallel | Dependencies |
|-------|-------|----------|--------------|
| BT-0 Design | 4 | 3 | None |
| BT-1 Composites | 7 | 3 | BT-0 |
| BT-2 Decorators | 9 | 3 | BT-0 |
| BT-3 Leaves | 8 | 3 | BT-0, T022, T028-30 |
| BT-4 Frame Lock | 6 | 2 | BT-1 |
| BT-5 Builder | 8 | 3 | BT-1,2,3 |
| BT-6 Integration | 7 | 2 | BT-5, T033 |
| BT-7 Context | 4 | 0 | BT-6 |
| **Total** | **53** | **19** | |

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Single node tick | <100ns | Cython cdef class |
| Tree traversal (100 nodes) | <10μs | Frame locking |
| Condition evaluation | <1μs | simpleeval cached |
| Full hook point cycle | <100μs | Budget for 7 hooks/turn |

Compare to py_trees: 50-200ms target latency (1000x slower than our goal)
