# Implementation Plan: Behavior Tree Universal Runtime

## Technical Context

### Existing Codebase Analysis

| Component | Location | Lines | Status |
|-----------|----------|-------|--------|
| Behavior Tree | `backend/src/services/plugins/behavior_tree/` | ~3,602 | Foundation - extend |
| Oracle Agent | `backend/src/services/oracle_agent.py` | ~2,765 | Migrate to tree |
| Research System | `backend/src/services/research/` | ~1,985 | Convert to subtree |
| Tool Executor | `backend/src/services/tool_executor.py` | ~2,539 | Convert to leaves |
| ANS EventBus | `backend/src/services/ans/` | ~1,600 | Tick trigger |
| Plugin System | `backend/src/services/plugins/` | ~2,238 | Integrate rules |
| MCP Server | `backend/src/mcp/server.py` | ~735 | Expose debug API |

**Total affected code: ~15,000+ lines**

### Technology Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Tree Definition | **Lua DSL** via `BT.*` API | LLMs are fluent in Lua. No new parser needed. Better error messages. Familiar table syntax. |
| Lua Engine | **lupa** (existing) | LuaJIT2 bindings already used by ScriptNode. Extend with `BT.*` namespace for tree definition. 593 lines in lua_sandbox.py. |
| Blackboard Storage | SQLite via **PluginStateService** | Global blackboard persists to existing DB using plugin_state table. Tree/subtree scopes are in-memory only. |
| Async Runtime | asyncio | Consistent with existing codebase |
| File Watching | watchdog | Standard Python hot reload library |
| Event Buffering | **Batch-during-tick** | Events emitted during tick are buffered and dispatched after tick completes to prevent tick-event loops. See research/05. |

### Integration Points

1. **EventBus → Tree Runtime**: Events trigger tree ticks
2. **Tree → Tool Executor**: Tools become leaf nodes
3. **Tree → LLM Service**: Special LLM nodes with streaming
4. **Tree → ANS**: Trees emit events for observability
5. **Lua Files → Tree Runtime**: Executed through `LuaSandbox` to produce tree definition

---

## Constitution Check

### Gate 1: Scope Validation
- [x] Feature aligns with project goals (universal agent orchestration)
- [x] No scope creep beyond spec (visual editor deferred)
- [x] Dependencies identified (existing BT, EventBus, SQLite)

### Gate 2: Technical Viability
- [x] All unknowns resolved (7 research documents completed)
- [x] Integration points feasible (existing patterns)
- [x] Performance targets achievable (<1ms tick overhead)

### Gate 3: Risk Assessment
- [x] Migration risk mitigated (parallel operation during transition)
- [x] Rollback possible (feature-flagged)
- [x] Testing strategy defined (E2E scenarios in spec)

---

## Phase 0: Research Summary

> **Status**: ✅ COMPLETE - All 7 research documents delivered

### Research Documents

| # | Topic | File | Status | Key Finding |
|---|-------|------|--------|-------------|
| 01 | Behavior Tree Audit | `research/01-behavior-tree-audit.md` | ✅ Complete | 3,602 lines existing BT is solid foundation. Gaps: hierarchical blackboard, async support, parallel memory mode. 8-13 days enhancement. |
| 02 | Oracle Agent Audit | `research/02-oracle-agent-audit.md` | ✅ Complete | 2,765 lines maps to ~300-400 lines Lua DSL. Key complexity: streaming LLM, context management. 10 E2E tests defined. |
| 03 | Research System Audit | `research/03-research-system-audit.md` | ✅ Complete | 2,301 lines custom behavior pattern. No tick semantics, no parallel policy control. Full subtree Lua design provided. |
| 04 | Tool Executor Audit | `research/04-tool-executor-audit.md` | ✅ Complete | 2,539 lines, 19 tools (17 implemented). Missing ANS events for tool.call.success/failure. Complete blackboard contracts per tool. |
| 05 | ANS EventBus Audit | `research/05-ans-eventbus-audit.md` | ✅ Complete | Mature pub/sub with batching, deferred delivery, cross-session persistence. 20+ new event types needed for BT. Tick-event loop prevention design. |
| 06 | Plugin System Audit | `research/06-plugin-system-audit.md` | ✅ Complete | ~5,200 lines total. BT already exists! RuleEngine, Lua sandbox, expression evaluator all work. Main work: `BT.*` Lua API, LLM nodes, hierarchical blackboard. |
| 07 | Lua DSL Design | `research/07-lua-dsl-design.md` | ✅ Complete | Lua DSL chosen over LISP. `lupa` already in codebase. LLMs fluent in Lua. No parser needed - just `BT.*` API design. |

### Key Decisions (from research)

1. **Blackboard Implementation**: Extend existing `Blackboard` class with parent chain for scope hierarchy. Use `PluginStateService` for global blackboard persistence. Tree/subtree scopes are in-memory only, cleared on completion.

2. **Lua DSL Choice**: **Lua via `lupa`** with `BT.*` fluent API. Rationale: (a) `lupa` already exists in codebase for `LuaSandbox`, (b) LLMs are significantly more fluent in Lua than LISP, (c) no new parser needed - just good API design, (d) better error messages out of the box, (e) familiar table syntax for configuration.

3. **Tick Trigger Strategy**: **Event-driven + polling hybrid**. Events (query.start, tool.success, tool.failure) trigger initial tick. Poll at 100ms while tree is RUNNING. Buffer events during tick to prevent loops.

4. **LLM Node Design**: New `LLMCallNode` class extending Leaf. Features: model selection, streaming with blackboard writes, token budget tracking, interruptible, timeout, retry policy. First tick initiates request (RUNNING), subsequent ticks check completion.

5. **Hot Reload Policy**: Default to `let-finish-then-swap` (safest). Support `cancel-and-restart` for urgent changes. Watch `.lua` files with watchdog library. Preserve global blackboard, optionally preserve tree-local if schema compatible.

6. **Lua Integration**: Existing `LuaSandbox` (593 lines, 5s timeout, secure) extended with `BT.*` namespace for tree definition. Features: (a) `BT.tree()`, `BT.sequence()`, `BT.selector()`, etc. for tree structure, (b) blackboard read/write from Lua, (c) RUNNING status return capability, (d) async operation support. Lua is both the tree definition language AND the escape hatch for complex conditions/actions.

7. **State Architecture**: **TypedBlackboard** with Pydantic models replaces `Dict[str, Any]`. Nodes declare **contracts** (inputs/outputs) validated at load time. State types form inheritance hierarchy: `IdentityState` → `ConversationState` → `OracleState`. Bridges connect to existing `RuleContext` and Lua. See `state-architecture.md` for full design.

### Consolidated Findings

#### Existing BT Implementation (research/01, research/06)

The existing behavior tree in `backend/src/services/plugins/behavior_tree/` (~3,602 lines) provides:
- **Composites**: Sequence, Selector, Parallel, MemorySelector, MemorySequence
- **Decorators**: Guard, Cooldown, Retry, Timeout, Repeat, Inverter, Succeeder, Failer, UntilFail, UntilSuccess
- **Leaves**: ConditionNode, ActionNode, ScriptNode (Lua), BlackboardCondition, BlackboardSet, WaitNode, LogNode
- **Frame Locking**: Optimization to cache RUNNING node and skip tree traversal
- **Tree Builder**: Fluent API and rule-based tree construction

**Gaps to Address**:
1. Flat blackboard → Need hierarchical scoping (global/tree/subtree)
2. Synchronous tick → Need async adaptation for LLM streaming
3. Parallel lacks REQUIRE_N, memory mode, clean cancellation
4. No hot reload infrastructure
5. No stuck detection/recovery

#### Oracle Agent Migration (research/02)

The Oracle agent (2,765 lines) maps to BT as:
```
Oracle.query()           → tree.tick(query.start event)
Agent loop               → BT.repeater({ until_failure = true })
Model selection          → BT.action() leaf + blackboard
LLM streaming           → BT.llm_call() with callbacks
Tool execution          → BT.parallel() with tool leaves
Context management      → Blackboard hierarchical state
Error handling          → BT.selector() fallback pattern
```

**Estimated Lua DSL**: 300-400 lines for oracle-agent.lua

#### Research System Subtree (research/03)

The research system (2,301 lines) has a custom behavior pattern that maps to:
```lua
return BT.subtree("deep-research", {
    root = BT.sequence {
        BT.llm_call { name = "generate-brief", ... },
        BT.action("plan-subtopics", { fn = "research.plan_subtopics" }),
        BT.parallel({ policy = "require-all", on_child_fail = "continue" }, {
            BT.for_each("researchers", {
                BT.subtree_ref("single-researcher")
            })
        }),
        BT.llm_call { name = "compress-findings", ... },
        BT.llm_call { name = "generate-report", ... },
        BT.selector {
            BT.sequence {
                BT.condition("should-persist?"),
                BT.action("persist-to-vault", { fn = "research.persist" })
            },
            BT.action("skip-persist", { fn = "research.noop" })
        }
    }
})
```

Full Lua DSL design provided in research/03.

#### ANS EventBus Integration (research/05)

**Events that trigger tree ticks**:
- `query.start` → Start new tree execution
- `tool.call.success/failure/timeout` → Resume tree after tool completion
- `budget.*.exceeded` → Interrupt tree, force failure path
- `agent.loop.detected` → Interrupt tree, activate recovery
- `tree.reload.requested` → Hot reload signal

**New event types needed** (20+):
- `tree.tick.start/complete`, `tree.status.changed`
- `tree.node.started/completed/stuck/cancelled`
- `tree.reload.requested/complete`, `tree.loaded/unloaded`
- `blackboard.key.changed`, `blackboard.scope.created/destroyed`
- `llm.stream.chunk/complete`, `llm.budget.exceeded`

**Tick-Event Loop Prevention**: Buffer events during tick, dispatch after completion.

#### Tool Executor Integration (research/04)

19 tools available (17 implemented). Each tool becomes an Action leaf:
- Input: Blackboard keys for tool parameters
- Output: Blackboard keys for results
- Events: Emit `tool.call.success/failure` (currently missing)

Complete blackboard contracts per tool documented in research/04.

#### Lua Sandbox Integration (research/06)

The existing Lua sandbox (`backend/src/services/plugins/lua_sandbox.py`, 593 lines) provides:
- **Secure execution**: Blocked globals (os, io, debug, require, etc.)
- **Allowed modules**: string, table, math (filtered functions only)
- **Timeout enforcement**: 5s default via threading
- **Context exposure**: RuleContext converted to Lua tables

**Current ScriptNode integration**:
```python
class ScriptNode(Leaf):
    def _tick(self, context: TickContext) -> RunStatus:
        result = self._sandbox.execute(script, context.rule_context)
        return self._map_result(result)  # nil→FAILURE, true→SUCCESS, table→action
```

**Gaps for BT**:
1. Cannot write to blackboard from Lua (only read via context)
2. Cannot return RUNNING status (only SUCCESS/FAILURE)
3. No async operation support
4. No blackboard access (only RuleContext)

**Enhancement plan**:
```lua
-- Lua API for BT (available in all scripts and tree definitions)
blackboard.get("key")           -- Read from blackboard
blackboard.set("key", value)    -- Write to blackboard
blackboard.has("key")           -- Check existence

-- Return RUNNING for async operations
return {status = "running", async_id = "my_operation"}

-- Return with blackboard writes
return {
    status = "success",
    blackboard = {["result"] = computed_value}
}
```

**Inline script in tree definition**:
```lua
BT.script("check-complex-condition", {
    lua = [[
        return context.turn.token_usage > 0.8 and #context.history.tools > 5
    ]]
})

-- External Lua file
BT.script("custom-logic", { file = "scripts/custom.lua" })
```

---

## Phase 1: Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Lua Tree Files                                 │
│  trees/oracle-agent.lua | trees/research.lua | trees/recovery.lua          │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │ execute via LuaSandbox
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Lua DSL Runtime                                   │
│  BT.* API → Validate → Build BehaviorTree                                  │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Tree Runtime (Core)                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Tick Loop   │  │ Blackboard  │  │ Watchdog    │  │ Hot Reload  │       │
│  │ (FR-4)      │  │ (FR-1)      │  │ (FR-5)      │  │ (FR-6)      │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                 │                                 │
        ▼                                 ▼                                 ▼
┌───────────────┐                ┌───────────────┐                ┌───────────────┐
│   Node Types  │                │  Leaf Types   │                │  Services     │
├───────────────┤                ├───────────────┤                ├───────────────┤
│ Sequence      │                │ Action        │                │ LLM Client    │
│ Selector      │                │ Condition     │                │ Tool Executor │
│ Parallel (FR-7)│               │ LLM Call (FR-3)│               │ EventBus      │
│ Decorators    │                │ Subtree       │                │ Database      │
└───────────────┘                └───────────────┘                └───────────────┘
```

### Module Structure

```
backend/src/
├── bt/                              # NEW: Behavior Tree Runtime
│   ├── __init__.py
│   ├── runtime.py                   # Tick loop, tree management
│   ├── blackboard.py                # Hierarchical state (FR-1)
│   ├── watchdog.py                  # Stuck detection (FR-5)
│   ├── hot_reload.py                # File watching, reload (FR-6)
│   ├── debug.py                     # Observability (FR-8)
│   ├── scheduler.py                 # Event-driven tick scheduling
│   ├── lua/
│   │   ├── __init__.py
│   │   ├── bt_api.py                # BT.* Lua API implementation
│   │   ├── loader.py                # Load .lua tree files via LuaSandbox
│   │   ├── validator.py             # Reference validation (fn, subtree_ref)
│   │   ├── builder.py               # Lua table → BehaviorTree
│   │   └── errors.py                # Custom error classes
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── base.py                  # Extend existing BehaviorNode
│   │   ├── llm.py                   # LLM-aware nodes (FR-3)
│   │   ├── tool.py                  # Tool leaf nodes
│   │   └── subtree.py               # Subtree composition
│   └── trees/                       # Lua tree definitions
│       ├── oracle-agent.lua
│       ├── research.lua
│       └── recovery.lua
├── services/
│   ├── plugins/behavior_tree/       # EXISTING: Keep, extend
│   ├── oracle_agent.py              # MIGRATE: Thin wrapper over tree
│   └── research/                    # MIGRATE: Becomes subtree
```

---

## Phase 2: Data Model

> See `data-model.md` for full entity specifications

### Core Entities

1. **Blackboard** - Hierarchical key-value state
2. **BehaviorTree** - Named tree composition
3. **BehaviorNode** - Node abstraction (extended)
4. **TickContext** - Execution context
5. **LLMCallNode** - Specialized LLM node
6. **TreeRegistry** - Loaded tree management

---

## Phase 3: Implementation Tasks

### Milestone 1: Core Runtime (Week 1-2)

| Task | Description | Dependencies | Estimate |
|------|-------------|--------------|----------|
| 1.1 | Implement Blackboard with scoping | None | 2 days |
| 1.2 | Extend TickContext from existing | Blackboard | 1 day |
| 1.3 | Implement tick loop with budget | TickContext | 2 days |
| 1.4 | Implement watchdog (stuck detection) | Tick loop | 1 day |
| 1.5 | Extend existing BT nodes | TickContext | 2 days |
| 1.6 | Unit tests for core runtime | All above | 2 days |

### Milestone 2: Lua DSL Integration (Week 2-3)

| Task | Description | Dependencies | Estimate |
|------|-------------|--------------|----------|
| 2.1 | Implement `BT.*` Lua API | LuaSandbox | 2 days |
| 2.2 | Implement tree loader (execute .lua via sandbox) | BT.* API | 1 day |
| 2.3 | Implement validator (fn refs, subtree refs) | Loader | 1 day |
| 2.4 | Implement Lua table → BehaviorTree builder | Validator | 2 days |
| 2.5 | Implement hot reload (watch .lua files) | Builder | 1 day |
| 2.6 | Unit tests for Lua DSL | All above | 1 day |

### Milestone 3: LLM Nodes (Week 3-4)

| Task | Description | Dependencies | Estimate |
|------|-------------|--------------|----------|
| 3.1 | Implement LLMCallNode base | Core runtime | 2 days |
| 3.2 | Add streaming support | LLMCallNode | 1 day |
| 3.3 | Add budget tracking | LLMCallNode | 1 day |
| 3.4 | Add interruption support | LLMCallNode | 1 day |
| 3.5 | Integration tests | All above | 2 days |

### Milestone 4: Tool Integration (Week 4-5)

| Task | Description | Dependencies | Estimate |
|------|-------------|--------------|----------|
| 4.1 | Create tool leaf wrapper | Core runtime | 2 days |
| 4.2 | Migrate tool executor | Tool leaf | 3 days |
| 4.3 | Integration tests | All above | 2 days |

### Milestone 5: Oracle Migration (Week 5-7)

| Task | Description | Dependencies | Estimate |
|------|-------------|--------------|----------|
| 5.1 | Write oracle-agent.lua | Lua DSL, LLM, Tools | 3 days |
| 5.2 | Create Oracle tree wrapper | Tree definition | 2 days |
| 5.3 | Parallel operation (old + new) | Wrapper | 2 days |
| 5.4 | E2E testing | All above | 3 days |
| 5.5 | Deprecate old Oracle loop | E2E pass | 1 day |

### Milestone 6: Research Migration (Week 7-8)

| Task | Description | Dependencies | Estimate |
|------|-------------|--------------|----------|
| 6.1 | Write research.lua | Oracle complete | 2 days |
| 6.2 | Integrate as subtree | Tree definition | 1 day |
| 6.3 | E2E testing | All above | 2 days |

### Milestone 7: Observability (Week 8)

| Task | Description | Dependencies | Estimate |
|------|-------------|--------------|----------|
| 7.1 | Implement debug API | Core runtime | 2 days |
| 7.2 | Add tree visualization | Debug API | 1 day |
| 7.3 | Frontend integration | Visualization | 2 days |

---

## E2E Test Plan

### Test Categories

1. **Unit Tests**: Per-node, per-module
2. **Integration Tests**: Cross-module interactions
3. **E2E Tests**: Full user flows
4. **Performance Tests**: Tick overhead, memory
5. **Regression Tests**: Existing functionality preserved

### Critical E2E Scenarios

| # | Scenario | Coverage |
|---|----------|----------|
| E1 | Basic query → response | Oracle migration |
| E2 | Tool calling (single, batch) | Tool integration |
| E3 | LLM streaming to frontend | LLM nodes |
| E4 | Deep research execution | Research migration |
| E5 | Stuck node recovery | Watchdog |
| E6 | Hot reload mid-execution | Hot reload |
| E7 | Parallel researcher failure | Parallel semantics |
| E8 | Budget exceeded | Budget tracking |
| E9 | Loop detection → ANS event | ANS integration |
| E10 | Debug visualization | Observability |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Migration breaks existing | Feature flag, parallel operation |
| Lua sandbox security | Reuse proven `LuaSandbox` (blocked globals, timeouts) |
| Hot reload state corruption | Conservative default policy |
| LLM node complexity | Extensive testing, fallback |
| Performance regression | Benchmark, optimize hot paths |

---

## Success Metrics

1. Oracle agent: 2,765 lines → ~500 lines Lua DSL + runtime
2. Research as subtree: Invokable from Oracle
3. Hot reload: <1 second
4. Tick overhead: <1ms
5. All E2E tests passing
6. No regressions in existing functionality

---

## Dependencies

### Python Packages (backend)

```toml
# Add to pyproject.toml [project.dependencies]
watchdog = "^4.0.0"      # File system watcher for hot reload
```

### Existing Dependencies (already present)

- `asyncio` - Async runtime (stdlib)
- `sqlite3` - Blackboard persistence (stdlib)
- `threading` - Watchdog thread (stdlib)
- `simpleeval` - Expression evaluation (already in plugins for ConditionNode)
- `lupa` - Lua sandbox (already in plugins for ScriptNode, LuaJIT2 bindings) - **Now also used for tree definition DSL**

---

## Appendices

- `research/` - Detailed component audits from subagents (7 documents)
- `data-model.md` - Full entity specifications
- `state-architecture.md` - Unified state management design (TypedBlackboard, contracts, bridges)
- `contracts/` - API contracts for debug endpoints
- `quickstart.md` - Developer onboarding guide
