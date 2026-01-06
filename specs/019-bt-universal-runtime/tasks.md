# Implementation Tasks: Behavior Tree Universal Runtime

## Overview

This document provides a dependency-ordered task checklist for implementing the BT Universal Runtime (spec 019). Tasks are organized by milestone with dependencies clearly marked.

**Total Estimated Effort**: 8 weeks (flexible based on team size)

---

## Prerequisites

- [ ] **P1**: Add `lark` dependency to pyproject.toml
- [ ] **P2**: Add `watchdog` dependency to pyproject.toml
- [ ] **P3**: Create `backend/src/bt/` package structure

---

## Milestone 1: Core Runtime

### 1.1 Hierarchical Blackboard (FR-1)
**Dependencies**: P3

- [ ] **1.1.1**: Extend existing `Blackboard` class with `parent` field
- [ ] **1.1.2**: Implement scope hierarchy lookup (`get()` searches up chain)
- [ ] **1.1.3**: Implement `set_global()` to write to root scope
- [ ] **1.1.4**: Implement `snapshot()` for debugging
- [ ] **1.1.5**: Add `BlackboardScope` enum (GLOBAL, TREE, SUBTREE)
- [ ] **1.1.6**: Integrate with `PluginStateService` for global persistence
- [ ] **1.1.7**: Unit tests for hierarchical blackboard

### 1.2 Extended TickContext (FR-4)
**Dependencies**: 1.1

- [ ] **1.2.1**: Extend `TickContext` with `tick_budget` field
- [ ] **1.2.2**: Add `async_pending` set for async operation tracking
- [ ] **1.2.3**: Add `parent_path` list for debugging
- [ ] **1.2.4**: Implement `budget_exceeded()` check
- [ ] **1.2.5**: Implement `elapsed_ms()` helper
- [ ] **1.2.6**: Unit tests for TickContext

### 1.3 Tick Loop Implementation (FR-4)
**Dependencies**: 1.2

- [ ] **1.3.1**: Create `BTRuntime` class with tick loop
- [ ] **1.3.2**: Implement tick budget enforcement
- [ ] **1.3.3**: Implement async wait logic (no busy-loop)
- [ ] **1.3.4**: Add `schedule_resume()` for RUNNING state
- [ ] **1.3.5**: Integrate with ANS EventBus for event-driven ticks
- [ ] **1.3.6**: Unit tests for tick loop

### 1.4 Watchdog / Stuck Detection (FR-5)
**Dependencies**: 1.3

- [ ] **1.4.1**: Create `Watchdog` class with timeout tracking per node
- [ ] **1.4.2**: Implement stuck detection (RUNNING > timeout)
- [ ] **1.4.3**: Implement forced FAILURE on stuck
- [ ] **1.4.4**: Emit ANS event on stuck detection
- [ ] **1.4.5**: Implement failure pattern tracking (3+ in window)
- [ ] **1.4.6**: Unit tests for watchdog

### 1.5 Event Buffering / Scheduler
**Dependencies**: 1.3

- [ ] **1.5.1**: Create `BTTickScheduler` class
- [ ] **1.5.2**: Implement event buffering during tick
- [ ] **1.5.3**: Implement event dispatch after tick completes
- [ ] **1.5.4**: Subscribe to tick-trigger events (query.start, tool.*)
- [ ] **1.5.5**: Implement polling while tree is RUNNING
- [ ] **1.5.6**: Unit tests for scheduler

---

## Milestone 2: LISP Integration

### 2.1 Lark Parser (FR-2)
**Dependencies**: P1, P3

- [ ] **2.1.1**: Create `grammar.lark` with BT LISP grammar
- [ ] **2.1.2**: Create `ast.py` with Symbol, Keyword, BTList, BTVector, BTMap
- [ ] **2.1.3**: Create `errors.py` with custom error classes
- [ ] **2.1.4**: Create `transformer.py` (parse tree → AST)
- [ ] **2.1.5**: Create `parser.py` with `BTLispParser` class
- [ ] **2.1.6**: Implement error pattern matching for helpful messages
- [ ] **2.1.7**: Unit tests for parser (valid trees, syntax errors, line numbers)

### 2.2 AST Node Classes
**Dependencies**: 2.1

- [ ] **2.2.1**: Create `TreeDef` AST node
- [ ] **2.2.2**: Create `SequenceNode`, `SelectorNode`, `ParallelNode`
- [ ] **2.2.3**: Create `ActionNode`, `ConditionNode`
- [ ] **2.2.4**: Create `LLMCallNode` AST (not BT node yet)
- [ ] **2.2.5**: Create `SubtreeNode` AST
- [ ] **2.2.6**: Create decorator AST nodes (Repeater, Timeout, Retry, etc.)

### 2.3 Validator (FR-2)
**Dependencies**: 2.2

- [ ] **2.3.1**: Create `BTValidator` class
- [ ] **2.3.2**: Implement `:fn` reference resolution checking
- [ ] **2.3.3**: Implement circular subtree detection
- [ ] **2.3.4**: Implement required property validation per node type
- [ ] **2.3.5**: Implement schema validation for blackboard
- [ ] **2.3.6**: Unit tests for validator

### 2.4 AST → BehaviorTree Builder
**Dependencies**: 2.3, 1.1

- [ ] **2.4.1**: Create `BTBuilder` class
- [ ] **2.4.2**: Implement AST → existing Composite nodes
- [ ] **2.4.3**: Implement AST → existing Decorator nodes
- [ ] **2.4.4**: Implement AST → Action/Condition leaves (fn resolution)
- [ ] **2.4.5**: Implement subtree reference linking
- [ ] **2.4.6**: Unit tests for builder

### 2.5 Hot Reload (FR-6)
**Dependencies**: P2, 2.4, 1.3

- [ ] **2.5.1**: Create `TreeRegistry` class
- [ ] **2.5.2**: Implement file watcher with watchdog
- [ ] **2.5.3**: Implement `let-finish-then-swap` policy
- [ ] **2.5.4**: Implement `cancel-and-restart` policy
- [ ] **2.5.5**: Implement reload logging with tree diff
- [ ] **2.5.6**: Unit tests for hot reload

---

## Milestone 3: LLM Nodes (FR-3)

### 3.1 LLMCallNode Base
**Dependencies**: 1.1, 1.2

- [ ] **3.1.1**: Create `LLMCallNode` extending Leaf
- [ ] **3.1.2**: Implement first tick → initiate request → RUNNING
- [ ] **3.1.3**: Implement subsequent tick → check completion
- [ ] **3.1.4**: Implement model selection from blackboard or props
- [ ] **3.1.5**: Implement prompt template loading

### 3.2 Streaming Support
**Dependencies**: 3.1

- [ ] **3.2.1**: Implement `stream_to` blackboard writes
- [ ] **3.2.2**: Implement `on_chunk` callback support
- [ ] **3.2.3**: Emit `llm.stream.chunk` events
- [ ] **3.2.4**: Implement partial response accumulation
- [ ] **3.2.5**: Integration test with mock LLM

### 3.3 Budget Tracking
**Dependencies**: 3.1

- [ ] **3.3.1**: Implement token counting per model
- [ ] **3.3.2**: Implement budget check before request
- [ ] **3.3.3**: Implement incremental budget check during streaming
- [ ] **3.3.4**: Emit `llm.budget.exceeded` event on failure
- [ ] **3.3.5**: Unit tests for budget tracking

### 3.4 Interruption Support
**Dependencies**: 3.2

- [ ] **3.4.1**: Implement `interruptible` flag check
- [ ] **3.4.2**: Implement HTTP request cancellation
- [ ] **3.4.3**: Implement cleanup on interruption
- [ ] **3.4.4**: Set FAILURE with interrupt reason in blackboard
- [ ] **3.4.5**: Integration tests for interruption

### 3.5 Retry Logic
**Dependencies**: 3.1

- [ ] **3.5.1**: Implement `retry_on` error type matching
- [ ] **3.5.2**: Implement exponential backoff
- [ ] **3.5.3**: Implement max retry tracking
- [ ] **3.5.4**: Unit tests for retry logic

---

## Milestone 4: Tool Integration

### 4.1 Tool Leaf Wrapper
**Dependencies**: 1.1

- [ ] **4.1.1**: Create `ToolLeaf` class extending Action
- [ ] **4.1.2**: Implement blackboard → tool input mapping
- [ ] **4.1.3**: Implement tool output → blackboard mapping
- [ ] **4.1.4**: Implement async tool execution with RUNNING
- [ ] **4.1.5**: Emit `tool.call.success/failure` events

### 4.2 Tool Executor Events
**Dependencies**: 4.1

- [ ] **4.2.1**: Add `tool.call.success` emission to tool executor
- [ ] **4.2.2**: Add `tool.call.failure` emission to tool executor
- [ ] **4.2.3**: Update existing tool calls to emit events
- [ ] **4.2.4**: Integration tests for tool events

### 4.3 Per-Tool Blackboard Contracts
**Dependencies**: 4.1

- [ ] **4.3.1**: Document blackboard keys for each tool (from research/04)
- [ ] **4.3.2**: Create type hints for tool inputs/outputs
- [ ] **4.3.3**: Integration tests for critical tools

---

## Milestone 5: Oracle Migration

### 5.1 Oracle LISP Tree
**Dependencies**: 2.4, 3.1, 4.1

- [ ] **5.1.1**: Write `oracle-agent.lisp` tree definition
- [ ] **5.1.2**: Define blackboard schema
- [ ] **5.1.3**: Implement context loading action
- [ ] **5.1.4**: Implement response emission action
- [ ] **5.1.5**: Implement tool execution parallel

### 5.2 Oracle Wrapper
**Dependencies**: 5.1

- [ ] **5.2.1**: Create `OracleBTWrapper` class
- [ ] **5.2.2**: Implement `process_query()` → tree tick
- [ ] **5.2.3**: Implement streaming bridge to SSE
- [ ] **5.2.4**: Implement context persistence bridge

### 5.3 Parallel Operation
**Dependencies**: 5.2

- [ ] **5.3.1**: Add feature flag for BT-based Oracle
- [ ] **5.3.2**: Run old and new Oracle in parallel (shadow mode)
- [ ] **5.3.3**: Compare outputs for regression detection
- [ ] **5.3.4**: Log discrepancies

### 5.4 E2E Testing
**Dependencies**: 5.3

- [ ] **5.4.1**: E2E: Basic query → response
- [ ] **5.4.2**: E2E: Single tool call
- [ ] **5.4.3**: E2E: Batch tool calls
- [ ] **5.4.4**: E2E: LLM streaming to frontend
- [ ] **5.4.5**: E2E: Budget exceeded handling
- [ ] **5.4.6**: E2E: Loop detection
- [ ] **5.4.7**: E2E: Context management
- [ ] **5.4.8**: E2E: Model selection
- [ ] **5.4.9**: E2E: Error recovery
- [ ] **5.4.10**: E2E: Interrupt handling

### 5.5 Deprecate Old Oracle
**Dependencies**: 5.4 (all passing)

- [ ] **5.5.1**: Remove parallel mode
- [ ] **5.5.2**: Update API routes to use BT-based Oracle
- [ ] **5.5.3**: Mark old Oracle as deprecated
- [ ] **5.5.4**: Update documentation

---

## Milestone 6: Research Migration

### 6.1 Research LISP Tree
**Dependencies**: 5.1 (Oracle working)

- [ ] **6.1.1**: Write `research.lisp` subtree definition
- [ ] **6.1.2**: Write `single-researcher.lisp` subtree
- [ ] **6.1.3**: Define blackboard schema for research
- [ ] **6.1.4**: Implement search actions (Tavily, OpenRouter fallback)
- [ ] **6.1.5**: Implement compression/report generation LLM calls

### 6.2 Subtree Integration
**Dependencies**: 6.1

- [ ] **6.2.1**: Register research subtree in TreeRegistry
- [ ] **6.2.2**: Invoke from Oracle via tool or direct call
- [ ] **6.2.3**: Implement progress streaming
- [ ] **6.2.4**: Implement vault persistence action

### 6.3 E2E Testing
**Dependencies**: 6.2

- [ ] **6.3.1**: E2E: Quick research (1 researcher)
- [ ] **6.3.2**: E2E: Standard research (3 researchers)
- [ ] **6.3.3**: E2E: Researcher failure handling
- [ ] **6.3.4**: E2E: Token budget exceeded
- [ ] **6.3.5**: E2E: Progress streaming to frontend
- [ ] **6.3.6**: E2E: Vault persistence

---

## Milestone 7: Observability (FR-8)

### 7.1 Debug API
**Dependencies**: 1.3

- [ ] **7.1.1**: Create `/api/bt/debug/trees` endpoint (list trees)
- [ ] **7.1.2**: Create `/api/bt/debug/tree/{id}` endpoint (tree state)
- [ ] **7.1.3**: Create `/api/bt/debug/tree/{id}/blackboard` endpoint
- [ ] **7.1.4**: Create `/api/bt/debug/tree/{id}/history` endpoint (tick history)
- [ ] **7.1.5**: Create `/api/bt/debug/tree/{id}/breakpoint` endpoint

### 7.2 Tree Visualization
**Dependencies**: 7.1

- [ ] **7.2.1**: Implement JSON export of tree structure
- [ ] **7.2.2**: Implement ASCII tree visualization
- [ ] **7.2.3**: Implement graphviz DOT export
- [ ] **7.2.4**: Add active node highlighting

### 7.3 Frontend Integration
**Dependencies**: 7.2

- [ ] **7.3.1**: Create BT debug panel component
- [ ] **7.3.2**: Display tree visualization
- [ ] **7.3.3**: Display blackboard state
- [ ] **7.3.4**: Display tick history
- [ ] **7.3.5**: Implement breakpoint UI

---

## ANS Event Types to Add

Throughout implementation, add these new event types:

- [ ] `tree.tick.start`
- [ ] `tree.tick.complete`
- [ ] `tree.status.changed`
- [ ] `tree.node.started`
- [ ] `tree.node.completed`
- [ ] `tree.node.stuck`
- [ ] `tree.node.cancelled`
- [ ] `tree.reload.requested`
- [ ] `tree.reload.complete`
- [ ] `tree.loaded`
- [ ] `tree.unloaded`
- [ ] `blackboard.key.changed`
- [ ] `blackboard.scope.created`
- [ ] `blackboard.scope.destroyed`
- [ ] `llm.stream.chunk`
- [ ] `llm.stream.complete`
- [ ] `llm.budget.exceeded`

---

## Success Criteria Verification

After all milestones complete, verify:

- [ ] Oracle agent LISP is ≤500 lines (spec says 300-400)
- [ ] Research invokable as Oracle subtree
- [ ] Hot reload completes in <1 second
- [ ] Tick overhead <1ms for non-LLM nodes
- [ ] All E2E tests passing
- [ ] No regressions in existing functionality

---

## References

- `spec.md` - Feature specification
- `plan.md` - Implementation plan with architecture
- `data-model.md` - Entity specifications
- `research/` - Component audits and research documents
