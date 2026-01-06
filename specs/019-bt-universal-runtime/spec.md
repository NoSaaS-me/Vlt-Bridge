# Behavior Tree Universal Runtime

## Overview

### Problem Statement

Vlt-Bridge has grown into a collection of disparate systems: a monolithic Oracle agent (2,765 lines), a separate research behavior pattern (1,985 lines), an unused plugin behavior tree (3,602 lines), and a disconnected event bus. These systems don't compose. Adding new agent types, research workflows, or tool orchestration requires understanding and modifying multiple subsystems.

The fundamental insight: **behavior trees are the universal composition pattern for agent systems**. This was proven at scale by game bot frameworks (HonorBuddy, OpenBot) that handled millions of users with complex decision-making through clean, composable trees.

### Solution Summary

Transform the existing behavior tree implementation into THE universal runtime for all agent orchestration. LISP defines tree structure (homoiconicity = trees define themselves). Python provides the core runtime, blackboard state, and leaf execution. All agents, workflows, and tools become nodes in composable trees.

### Scope

**In Scope:**
- Blackboard state management with proper scoping (global, tree-local, subtree-local)
- LISP tree definition language with Python interop
- LLM-aware nodes (streaming, budget, interruptibility)
- Runtime semantics (tick loop, budget, yield points, async coordination)
- Stuck detection and recovery heuristics
- Hot reload of trees at runtime
- Parallel node policies with precise semantics
- Observability and debugging infrastructure
- Migration of Oracle agent to tree-based execution
- Migration of Research workflow to subtree

**Out of Scope:**
- Visual tree editor (future enhancement)
- Distributed tree execution across machines (future enhancement)
- BERT/ML-based condition nodes (future enhancement)

---

## User Scenarios & Testing

### Scenario 1: Developer Defines Agent Behavior in LISP

**User**: A developer creating a new agent workflow

**Flow**:
1. Developer writes a `.lisp` file defining the agent's behavior tree
2. The LISP references Python functions for leaf execution
3. System loads and validates the tree at startup
4. Tree executes when events arrive

**Acceptance**:
- LISP syntax errors produce clear error messages with line numbers
- Undefined Python references are caught at load time, not runtime
- Tree can be tested in isolation with mock blackboard

### Scenario 2: LLM Call with Streaming and Budget

**User**: An agent making an LLM API call

**Flow**:
1. Tree ticks to an LLM node
2. Node begins streaming response, writing partial results to blackboard
3. Each chunk triggers configured callbacks (e.g., emit to SSE)
4. If token budget exceeded, node returns FAILURE with budget error
5. If interrupted (higher priority event), node cancels gracefully
6. On completion, full response available in blackboard

**Acceptance**:
- Partial responses visible in blackboard during streaming
- Token counting accurate within 5% of actual usage
- Interruption cleans up resources (no dangling connections)
- Budget exceeded is distinguishable from other failures

### Scenario 3: Hot Reload of Running Tree

**User**: A developer iterating on behavior

**Flow**:
1. Agent is running with tree in RUNNING state (e.g., waiting for LLM)
2. Developer modifies `.lisp` file and triggers reload
3. System loads new tree definition
4. For RUNNING nodes: option to cancel-and-restart or let-finish-then-swap
5. New tree takes effect for next tick cycle

**Acceptance**:
- Reload does not crash running system
- In-flight operations complete or cancel cleanly
- Blackboard state preserved across reload (unless schema changes)
- Reload logged with before/after tree diff

### Scenario 4: Stuck Detection and Recovery

**User**: An agent that gets into unexpected state

**Flow**:
1. Tree node returns RUNNING for longer than configured timeout
2. Watchdog detects stuck condition
3. Watchdog forces FAILURE on stuck node
4. Parent composite handles failure (retry, fallback, escalate)
5. If repeated failures detected, system escalates to recovery tree

**Acceptance**:
- Stuck detection triggers within 2x the configured timeout
- Forced FAILURE is logged with full context (node path, blackboard snapshot)
- Recovery tree can access failure context to make informed decisions
- Repeated failure patterns (3+ in 1 minute) trigger escalation

### Scenario 5: Parallel Researchers with Proper Semantics

**User**: Research workflow running multiple searchers concurrently

**Flow**:
1. Parallel node spawns 5 researcher subtrees
2. Each researcher searches, thinks, iterates
3. Researcher 3 fails (API error)
4. Based on policy (REQUIRE_ALL vs REQUIRE_ONE), parallel either:
   - Fails immediately and cancels other researchers, OR
   - Continues, succeeding when first researcher completes
5. Completed researcher results in blackboard; failed ones logged

**Acceptance**:
- Policy is configurable per parallel node
- Cancellation of in-flight children is clean (no orphaned tasks)
- Memory mode preserves success status across ticks
- Final blackboard contains results from all completed children

---

## Functional Requirements

### FR-1: Blackboard State Management

The system provides a hierarchical blackboard for inter-node communication.

**Scoping:**
- **Global blackboard**: Shared across all trees, persists across sessions
- **Tree-local blackboard**: Scoped to current tree execution, cleared on tree completion
- **Subtree-local blackboard**: Scoped to a subtree, useful for isolated workflows

**Operations:**
- `get(key)` - Read value (searches up scope hierarchy)
- `set(key, value)` - Write value at current scope
- `set_global(key, value)` - Write to global scope explicitly
- `has(key)` - Check existence
- `delete(key)` - Remove value
- `snapshot()` - Capture current state for debugging/recovery

**Acceptance Criteria:**
- Child scopes can read parent scope values
- Child scopes can shadow parent values without modifying them
- Global blackboard persists to database across sessions
- Blackboard access is thread-safe for parallel nodes

### FR-2: LISP Tree Definition Language

Trees are defined in LISP S-expressions that reference Python runtime components.

**Syntax:**
```lisp
(tree "oracle-agent"
  :description "Main chat agent workflow"
  :blackboard-schema {:context nil :response nil :tools []}

  (sequence
    (action load-context :fn "oracle.load_context")
    (repeater :until-failure
      (sequence
        (llm-call :model "claude-sonnet-4"
                  :stream-to [:partial-response]
                  :budget 4000)
        (selector
          (sequence
            (condition has-tool-calls?)
            (parallel :policy :wait-all
              (for-each [:tool-calls]
                (action execute-tool :fn "tools.execute"))))
          (action emit-response :fn "oracle.emit"))))))
```

**Python Interop:**
- `:fn "module.function"` - References Python function by dotted path
- Python functions receive `(ctx: TickContext, blackboard: Blackboard) -> RunStatus`
- Python functions can be async
- Type annotations on Python functions used for validation

**Acceptance Criteria:**
- Parser produces clear errors for syntax issues
- Undefined `:fn` references caught at load time
- Circular tree references detected and rejected
- Trees can import/reference other trees as subtrees

### FR-3: LLM-Aware Nodes

LLM calls are a special node type, not regular leaves, due to their unique characteristics.

**Properties:**
- `:stream-to [blackboard-key]` - Where to write streaming chunks
- `:budget tokens` - Maximum tokens for this call
- `:interruptible bool` - Can this call be cancelled by higher-priority events
- `:on-chunk callback` - LISP form or Python fn to call on each chunk
- `:timeout seconds` - Maximum wall-clock time
- `:retry-on [error-types]` - Which errors trigger automatic retry

**Tick Behavior:**
1. First tick: Initiate LLM call, return RUNNING
2. Subsequent ticks: Check for completion
   - If streaming, update blackboard with new content
   - If complete, return SUCCESS with full response in blackboard
   - If error, return FAILURE with error in blackboard
   - If interrupted, cleanup and return FAILURE with interrupt reason
3. Budget tracking: Count tokens incrementally, fail early if exceeded

**Acceptance Criteria:**
- Streaming updates visible in blackboard within 100ms of chunk arrival
- Token counting works for all supported models
- Interruption cancels the HTTP request (doesn't just ignore result)
- Retry logic respects exponential backoff

### FR-4: Runtime Semantics and Tick Loop

The core tick loop defines exactly how trees execute.

**Tick Context:**
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

**Tick Loop Pseudocode:**
```
def tick_tree(tree, event):
    ctx = TickContext(
        event=event,
        blackboard=tree.blackboard,
        services=get_services(),
        tick_count=0,
        tick_budget=1000,  # configurable
        start_time=now()
    )

    while True:
        ctx.tick_count += 1
        status = tree.root.tick(ctx)

        if status == SUCCESS:
            tree.on_complete(ctx)
            return SUCCESS

        if status == FAILURE:
            tree.on_failure(ctx)
            return FAILURE

        if status == RUNNING:
            # Check budgets
            if ctx.tick_count >= ctx.tick_budget:
                # Yield - will resume on next event
                tree.schedule_resume()
                return RUNNING

            if ctx.elapsed() > tree.max_tick_duration:
                # Timeout - force failure
                log_stuck(tree, ctx)
                return FAILURE

            # Check for pending async operations
            if ctx.has_pending_async():
                # Wait for async, don't busy-loop
                await ctx.wait_for_async(timeout=100ms)
            else:
                # Immediate retry (computation still in progress)
                continue
```

**Tick Trigger Modes:**
- **Event-driven**: Tick when event arrives (default)
- **Polling**: Tick every N ms while any tree is RUNNING
- **Hybrid**: Event triggers initial tick, poll until tree settles

**Acceptance Criteria:**
- Tick budget prevents runaway loops
- Async operations don't busy-loop
- Stuck trees detected and handled
- Tick history available for debugging

### FR-5: Stuck Detection and Recovery

Meta-level monitoring detects and handles unexpected states.

**Stuck Conditions:**
- Node returns RUNNING for longer than `:timeout` without progress
- Same failure pattern repeats 3+ times in configured window
- Tree exceeds total execution time budget

**Recovery Actions:**
- Force FAILURE on stuck node
- Log full context (node path, blackboard snapshot, tick history)
- Emit ANS event for visibility
- Optionally trigger recovery tree

**Escalation:**
- Repeated failures (configurable threshold) escalate to different tree
- Escalation tree receives failure context in blackboard
- Ultimate fallback: graceful degradation response to user

**Acceptance Criteria:**
- Stuck detection within 2x configured timeout
- Recovery tree can access full failure context
- Escalation prevents infinite failure loops
- User receives response even in degraded mode

### FR-6: Hot Reload

Trees can be reloaded at runtime without restarting the system.

**Reload Process:**
1. Watch for file changes (or explicit reload command)
2. Parse new tree definition
3. Validate against schema
4. For RUNNING trees, apply configured policy:
   - `cancel-and-restart`: Cancel in-flight operations, load new tree
   - `let-finish-then-swap`: Complete current execution, use new tree for next
   - `immediate`: Swap tree structure, preserve node state where compatible
5. Log reload with tree diff

**State Preservation:**
- Global blackboard always preserved
- Tree-local blackboard optionally preserved if schema compatible
- In-flight async operations handled per policy

**Acceptance Criteria:**
- Reload completes in under 1 second
- No crashes or hangs during reload
- Incompatible schema changes produce clear warnings
- Reload history logged for debugging

### FR-7: Parallel Node Semantics

Parallel nodes execute multiple children concurrently with configurable policies.

**Policies:**
- `REQUIRE_ALL`: Succeed only if all children succeed. Fail immediately if any fails.
- `REQUIRE_ONE`: Succeed when first child succeeds. Continue others or cancel.
- `REQUIRE_N(n)`: Succeed when N children succeed.

**Failure Handling:**
- `:on-child-fail :cancel-siblings` - Cancel running siblings on failure
- `:on-child-fail :continue` - Let siblings finish regardless
- `:on-child-fail :retry` - Retry failed child

**Memory Mode:**
- `:memory true` - Remember which children completed across ticks
- Useful for long-running parallel operations

**Concurrency Control:**
- `:max-concurrent N` - Limit concurrent children (for rate limiting)

**Acceptance Criteria:**
- Policy enforced correctly for all combinations
- Cancellation is clean (no orphaned tasks)
- Memory mode correctly resumes across ticks
- Concurrency limit prevents resource exhaustion

### FR-8: Observability and Debugging

Trees are inspectable at runtime for debugging and monitoring.

**Runtime Visibility:**
- Current active path (highlighted nodes)
- Last N ticks history with timestamps
- Blackboard state at each scope level
- Per-node timing (average tick duration, total time in RUNNING)
- Error counts and last error per node

**Debugging Tools:**
- Breakpoints: Pause execution at specific nodes
- Step mode: Tick one node at a time
- Blackboard watch: Notify on specific key changes
- Trace mode: Log every tick with full context

**Output Formats:**
- JSON for programmatic access
- Tree visualization (ASCII or exportable to graphviz)
- Integration with frontend for live view

**Acceptance Criteria:**
- Active path accurate within 1 tick
- History retains at least 100 ticks
- Timing data accurate to 1ms
- Debug mode does not affect production execution when disabled

---

## Success Criteria

1. **Migration completeness**: Oracle agent (2,765 lines) refactored to tree definition under 500 lines of LISP, with all current functionality preserved

2. **Composition achieved**: Research workflow runs as subtree of Oracle, can be invoked via tool or directly

3. **Hot reload works**: Developer can modify tree LISP and reload without restarting, with less than 1 second reload time

4. **LLM streaming works**: Partial responses visible in UI during generation, with chunks appearing within 100ms of receipt

5. **Stuck recovery works**: Artificially stuck node recovers within 2x timeout, user receives response

6. **Parallel semantics correct**: All policy combinations tested and working, no orphaned tasks

7. **Performance maintained**: Tick overhead under 1ms for non-LLM nodes, no regression in response time

8. **Debugging usable**: Tree visualization available, active path visible, breakpoints functional

---

## Key Entities

### Blackboard

Hierarchical key-value state shared between nodes.

**Attributes:**
- `scope`: GLOBAL | TREE | SUBTREE
- `data`: Dictionary of key-value pairs
- `parent`: Reference to parent scope (or null)
- `modified_at`: Timestamp of last modification
- `version`: Optimistic concurrency version

### BehaviorNode

Base abstraction for all tree nodes.

**Attributes:**
- `id`: Unique identifier within tree
- `name`: Human-readable name
- `type`: COMPOSITE | DECORATOR | LEAF
- `status`: SUCCESS | FAILURE | RUNNING | FRESH
- `tick_count`: Number of times ticked
- `last_tick_duration`: Timing of last tick
- `metadata`: Arbitrary node metadata

### BehaviorTree

A named, reusable composition of nodes.

**Attributes:**
- `id`: Unique tree identifier
- `name`: Human-readable name
- `root`: Root BehaviorNode
- `blackboard`: Tree-scoped blackboard
- `schema`: Expected blackboard schema
- `source_path`: Path to LISP definition
- `loaded_at`: When tree was loaded
- `tick_count`: Total ticks executed

### TickContext

Context passed to every node on tick.

**Attributes:**
- `event`: Triggering event
- `blackboard`: Current blackboard scope
- `services`: Dependency injection container
- `tick_count`: Ticks in current execution
- `tick_budget`: Max ticks before yield
- `start_time`: Execution start time
- `parent_path`: List of parent node IDs
- `async_pending`: Set of pending async operations

### LLMCallNode

Specialized node for LLM API calls.

**Attributes:**
- `model`: Model identifier
- `prompt_template`: LISP form or string template
- `stream_to`: Blackboard key for streaming
- `budget`: Token budget
- `interruptible`: Whether can be cancelled
- `timeout`: Wall-clock timeout
- `retry_policy`: Error types to retry

---

## Dependencies & Assumptions

### Dependencies

- Existing behavior tree implementation (3,602 lines in `plugins/behavior_tree/`)
- ANS EventBus for event-driven tick triggering
- OpenRouter/Anthropic API for LLM calls
- Existing tool executor for tool leaf execution

### Assumptions

- LISP parsing can use existing Python libraries (e.g., `hy` or custom s-expression parser)
- Hot reload is acceptable with sub-second pause, not zero-downtime
- Parallel execution uses asyncio, not multiprocessing
- Global blackboard persistence uses existing SQLite database

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LISP parsing complexity | Medium | High | Start with minimal subset, expand incrementally |
| Hot reload state corruption | Medium | High | Conservative default policy (let-finish-then-swap) |
| LLM node complexity | High | Medium | Extensive testing, fallback to simple retry |
| Migration breaks existing behavior | Medium | High | Run old and new in parallel during migration |
| Performance regression | Low | Medium | Benchmark before/after, optimize hot paths |

---

## Out of Scope (Future Enhancements)

- Visual tree editor (drag-and-drop node composition)
- Distributed tree execution (trees spanning multiple processes/machines)
- ML-based condition nodes (BERT for semantic matching)
- Record/replay for deterministic testing
- Tree versioning and rollback
