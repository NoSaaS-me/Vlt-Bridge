# Footgun Addendum: Behavior Tree Universal Runtime

**Version:** 1.0.0
**Date:** 2026-01-07
**Purpose:** Document remaining footguns not fully covered by contracts and specify implementation requirements.

---

## Coverage Summary

The following contracts now cover most error conditions:

| Contract | Coverage |
|----------|----------|
| `contracts/errors.yaml` | 28 error codes with recovery actions |
| `contracts/blackboard.yaml` | TypedBlackboard interface + type coercion |
| `contracts/nodes.yaml` | All node types + error propagation rules |
| `contracts/tree-loader.yaml` | TreeRegistry, hot reload, validation |
| `contracts/events.yaml` | 40+ ANS events for observability |
| `contracts/integrations.yaml` | MCP tools + CodeRAG as leaf nodes |

This addendum covers items requiring **architectural decisions**, **implementation patterns**, or **developer experience tooling** beyond contract specifications.

---

## A. Remaining Architectural Decisions

### A.1 Progress Tracking for Stuck Detection

**Issue:** UB-01 from footgun audit - "progress" is undefined for stuck detection.

**Decision:** Progress is defined as ANY of:
1. Blackboard write (`bb.set()` succeeds)
2. Async operation completion (`ctx.complete_async()` called)
3. Explicit progress mark (`ctx.mark_progress()` called)

**Implementation Pattern:**
```python
class TickContext:
    _last_progress_at: Optional[datetime] = None

    def mark_progress(self) -> None:
        """Explicitly mark progress for watchdog."""
        self._last_progress_at = datetime.utcnow()

    def _on_blackboard_write(self, key: str) -> None:
        """Internal: called by blackboard after successful write."""
        self._last_progress_at = datetime.utcnow()

    def complete_async(self, op_id: str) -> None:
        self.async_pending.discard(op_id)
        self._last_progress_at = datetime.utcnow()

class TreeWatchdog:
    def check_stuck(self, tree: BehaviorTree, ctx: TickContext) -> Optional[str]:
        """Returns node_id if stuck, None if OK."""
        for node in tree.get_running_nodes():
            running_ms = (datetime.utcnow() - node.running_since).total_seconds() * 1000
            if running_ms > tree.max_tick_duration_ms:
                # Check progress
                if ctx._last_progress_at is None or \
                   ctx._last_progress_at < node.running_since:
                    return node.id
        return None
```

**Test Requirement:** `test_watchdog_detects_stuck_node_without_progress`

---

### A.2 Event Buffering During Tick

**Issue:** RC-02 - Events emitted during tick could cause re-entrant ticks.

**Decision:** Global event buffer with FIFO dispatch after tick completion.

**Implementation Pattern:**
```python
class EventBus:
    _buffer: List[Event] = []
    _tick_in_progress: bool = False
    MAX_BUFFER_SIZE: int = 1000

    def emit(self, event: Event) -> None:
        if self._tick_in_progress:
            if len(self._buffer) >= self.MAX_BUFFER_SIZE:
                logger.warning(f"Event buffer full, dropping oldest: {self._buffer[0].type}")
                self._buffer.pop(0)
            self._buffer.append(event)
        else:
            self._dispatch(event)

    @contextmanager
    def tick_scope(self):
        """Context manager for tree tick. Buffers events."""
        self._tick_in_progress = True
        try:
            yield
        finally:
            self._tick_in_progress = False
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Dispatch all buffered events in FIFO order."""
        while self._buffer:
            event = self._buffer.pop(0)
            self._dispatch(event)
```

**Usage:**
```python
def tick_tree(tree: BehaviorTree, ctx: TickContext) -> RunStatus:
    with event_bus.tick_scope():
        return tree.tick(ctx)
```

**Test Requirements:**
- `test_events_buffered_during_tick`
- `test_buffer_overflow_drops_oldest`
- `test_buffer_flushed_after_tick`

---

### A.3 Parallel Child Scope Isolation

**Issue:** RC-01 - Parallel children sharing blackboard is a bug waiting to happen.

**Decision:** Each parallel child gets an isolated child scope. Writes are merged after all children complete.

**Implementation Pattern:**
```python
class ParallelNode(CompositeNode):
    def _tick(self, ctx: TickContext) -> RunStatus:
        child_scopes = []
        child_statuses = []

        # Create isolated scope per child
        for i, child in enumerate(self._children):
            child_scope = ctx.blackboard.create_child_scope(f"parallel_{self.id}_{i}")
            child_ctx = ctx.with_blackboard(child_scope)
            status = child.tick(child_ctx)
            child_scopes.append(child_scope)
            child_statuses.append(status)

        # Merge after all complete (or policy met)
        if self._should_merge(child_statuses):
            merge_result = self._merger.merge(ctx.blackboard, child_scopes)
            if merge_result.has_conflicts and self._merge_strategy == FAIL_ON_CONFLICT:
                ctx.blackboard.set_internal("_parallel_conflicts", merge_result.conflicts)
                return RunStatus.FAILURE

        return self._evaluate_policy(child_statuses)
```

**Test Requirements:**
- `test_parallel_children_cannot_see_sibling_writes`
- `test_parallel_merge_after_all_complete`
- `test_fail_on_conflict_returns_failure`

---

### A.4 Cancellation Propagation

**Issue:** E3006 - How does cancellation reach async operations?

**Decision:** Cooperative cancellation via `ctx.cancellation_requested` flag + async task cancellation.

**Implementation Pattern:**
```python
class LLMCallNode(LeafNode):
    _task: Optional[asyncio.Task] = None

    def _tick(self, ctx: TickContext) -> RunStatus:
        # Check cancellation at tick start
        if ctx.cancellation_requested:
            self._cancel_in_flight()
            return RunStatus.FAILURE

        if self._task is None:
            # Start request
            self._task = asyncio.create_task(self._make_request())
            ctx.add_async(self._request_id)
            return RunStatus.RUNNING

        if self._task.done():
            # Handle result
            ctx.complete_async(self._request_id)
            return self._handle_result(self._task.result())

        return RunStatus.RUNNING

    def _cancel_in_flight(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
```

**Test Requirements:**
- `test_cancellation_cancels_llm_task`
- `test_cancelled_node_returns_failure`
- `test_multiple_async_ops_all_cancelled`

---

### A.5 Lazy Subtree Resolution

**Issue:** UB-02 - What about dynamically loaded subtrees?

**Decision:** Support optional `lazy=True` for runtime resolution.

**Implementation Pattern:**
```lua
-- Eager (validated at load time)
BT.subtree_ref("research-runner")

-- Lazy (resolved at first tick)
BT.subtree_ref("dynamic-subtree", { lazy = true })
```

```python
class SubtreeRefNode(LeafNode):
    def __init__(self, tree_name: str, lazy: bool = False, ...):
        self._tree_name = tree_name
        self._lazy = lazy
        self._resolved_tree: Optional[BehaviorTree] = None

    def _tick(self, ctx: TickContext) -> RunStatus:
        if self._resolved_tree is None:
            tree = ctx.services.tree_registry.get(self._tree_name)
            if tree is None:
                if self._lazy:
                    # Try to load dynamically
                    tree = self._try_load(ctx)
                if tree is None:
                    raise TreeNotFoundError(E3001, tree_id=self._tree_name)
            self._resolved_tree = tree

        return self._resolved_tree.tick(ctx)
```

**Test Requirements:**
- `test_eager_subtree_validated_at_load`
- `test_lazy_subtree_resolved_at_tick`
- `test_lazy_subtree_not_found_error`

---

## B. Implementation Checklist Additions

These tests MUST be implemented before the runtime is considered complete.

### B.1 Contract Enforcement Tests

- [ ] `test_unregistered_key_raises_E1001`
- [ ] `test_validation_error_raises_E1002`
- [ ] `test_reserved_key_raises_E1003`
- [ ] `test_size_limit_raises_E1004`
- [ ] `test_missing_input_raises_E2001`
- [ ] `test_undeclared_read_logs_E2002_warning`
- [ ] `test_undeclared_write_logs_E2003_warning`

### B.2 Error Propagation Tests

- [ ] `test_sequence_fails_on_first_child_failure`
- [ ] `test_selector_succeeds_on_first_child_success`
- [ ] `test_parallel_require_all_fails_on_any_failure`
- [ ] `test_parallel_require_one_succeeds_on_any_success`
- [ ] `test_decorator_timeout_returns_failure`
- [ ] `test_decorator_retry_exhausts_attempts`

### B.3 Security Tests

- [ ] `test_sandbox_blocks_os_execute`
- [ ] `test_sandbox_blocks_io_open`
- [ ] `test_sandbox_blocks_loadfile`
- [ ] `test_sandbox_blocks_dofile`
- [ ] `test_sandbox_blocks_require`
- [ ] `test_sandbox_blocks_metatable_access`
- [ ] `test_path_traversal_blocked_in_subtree_ref`
- [ ] `test_path_traversal_blocked_in_tree_load`

### B.4 Hot Reload Tests

- [ ] `test_let_finish_then_swap_queues_reload`
- [ ] `test_cancel_and_restart_cancels_running`
- [ ] `test_reload_debounce_coalesces_changes`
- [ ] `test_invalid_reload_keeps_old_tree`
- [ ] `test_reload_emits_tree_reloaded_event`

### B.5 Async Operation Tests

- [ ] `test_llm_timeout_returns_failure`
- [ ] `test_llm_cancellation_cancels_request`
- [ ] `test_async_progress_updates_watchdog`
- [ ] `test_orphaned_chunks_discarded`

### B.6 Merge Strategy Tests

- [ ] `test_last_wins_uses_last_value`
- [ ] `test_first_wins_uses_first_value`
- [ ] `test_collect_creates_list`
- [ ] `test_merge_dict_deep_merges`
- [ ] `test_fail_on_conflict_detects_conflict`
- [ ] `test_merge_result_contains_conflict_info`

---

## C. Type Coercion Edge Cases

These edge cases from TS-01 require explicit handling:

### C.1 Lua Number to Python Int

```python
# Problem: Pydantic model expects int, Lua returns float
class TurnNumber(BaseModel):
    value: int  # Will fail with 1.0

# Solution: Use validator or float field
class TurnNumber(BaseModel):
    value: float  # Accepts Lua numbers

    @property
    def as_int(self) -> int:
        return int(self.value)
```

### C.2 Lua Empty Table Ambiguity

```python
# Problem: {} could be empty list or empty dict
# Lua doesn't distinguish

# Solution: Check context or use explicit type
def _lua_to_python(value: Any) -> Any:
    if isinstance(value, lua.Table):
        if len(value) == 0:
            # Ambiguous - check schema context
            return {}  # Default to dict
        # ... rest of conversion
```

### C.3 Nested Table Conversion

```python
# Problem: Deeply nested tables need recursive conversion
def _lua_to_python(value: Any, depth: int = 0) -> Any:
    MAX_DEPTH = 20
    if depth > MAX_DEPTH:
        raise ValueError(f"Lua table nesting too deep (>{MAX_DEPTH})")

    if isinstance(value, lua.Table):
        if _is_array_like(value):
            return [_lua_to_python(v, depth + 1) for v in value.values()]
        else:
            return {str(k): _lua_to_python(v, depth + 1) for k, v in value.items()}
    # ... primitives
```

---

## D. Performance Considerations

### D.1 Blackboard Size Tracking

```python
class TypedBlackboard:
    _size_bytes: int = 0

    def set(self, key: str, value: BaseModel) -> ErrorResult:
        # Estimate size before write
        new_size = len(value.model_dump_json().encode('utf-8'))
        old_size = self._get_key_size(key)
        delta = new_size - old_size

        if self._size_bytes + delta > self._max_size_bytes:
            return ErrorResult.failure(E1004, ...)

        # ... actual write
        self._size_bytes += delta
```

### D.2 Contract Validation Caching

```python
class BehaviorNode:
    _contract_cache: Optional[NodeContract] = None

    @classmethod
    def contract(cls) -> NodeContract:
        if cls._contract_cache is None:
            cls._contract_cache = cls._build_contract()
        return cls._contract_cache
```

### D.3 Event Batching for Performance

```python
class EventBus:
    BATCH_THRESHOLD = 100  # Events per second

    def _should_batch(self, event: Event) -> bool:
        if event.severity == Severity.CRITICAL:
            return False  # Never batch critical
        return self._recent_event_count > self.BATCH_THRESHOLD
```

---

## E. Developer Experience for Graph Editing

**Goal:** A human can step into the behavior tree code and modify agent logic without wading through thousands of lines of Python. The graph IS the logic.

### E.1 Tree Structure Visualization (CLI)

**Requirement:** `vlt bt show <tree-name>` prints human-readable tree structure.

```bash
$ vlt bt show oracle-agent

oracle-agent (Sequence)
├── load-context (Action: oracle.load_context)
│   └── inputs: session_id
│   └── outputs: context, turn_number
├── check-budget (Guard: ctx.budget > 0)
│   └── process-query (Sequence)
│       ├── search-code (CodeSearch)
│       │   └── query: ${bb.user_query}
│       ├── search-vault (VaultSearch)
│       │   └── query: ${bb.user_query}
│       └── generate-response (Oracle)
│           └── stream_to: partial_response
└── save-context (Action: oracle.save_context)
```

**Implementation:**
```python
def print_tree(tree: BehaviorTree, indent: int = 0) -> str:
    """Recursively print tree structure with contracts."""
    lines = []
    prefix = "│   " * indent
    lines.append(f"{prefix}├── {node.id} ({node.__class__.__name__})")

    # Show contract summary
    contract = node.contract()
    if contract.inputs:
        lines.append(f"{prefix}│   └── inputs: {', '.join(contract.inputs.keys())}")
    if contract.outputs:
        lines.append(f"{prefix}│   └── outputs: {', '.join(contract.outputs.keys())}")

    # Recurse for children
    for child in node.children:
        lines.extend(print_tree(child, indent + 1))

    return "\n".join(lines)
```

### E.2 Execution Trace (Debug Mode)

**Requirement:** When `trace_enabled=True`, every tick prints human-readable trace.

```bash
$ vlt bt run oracle-agent --trace

[TICK 1] oracle-agent
  → load-context: SUCCESS (12ms)
    ✓ wrote: context, turn_number
  → check-budget: evaluating guard...
    ✓ condition: ctx.budget > 0 = True
  → process-query
    → search-code: RUNNING (started async)

[TICK 2] oracle-agent
  → process-query
    → search-code: SUCCESS (342ms)
      ✓ wrote: code_results (5 items)
    → search-vault: SUCCESS (89ms)
      ✓ wrote: vault_results (3 items)
    → generate-response: RUNNING (streaming...)
      ⋯ partial: "Based on the code..."

[TICK 3] oracle-agent
  → process-query
    → generate-response: SUCCESS (1,203ms)
      ✓ wrote: final_answer (847 tokens)
  → save-context: SUCCESS (8ms)

COMPLETE: SUCCESS in 3 ticks (1,654ms total)
Blackboard final state:
  context: ConversationContext(session_id="abc", turn=5)
  code_results: [5 items]
  vault_results: [3 items]
  final_answer: "Based on the code..."
```

**Implementation:**
```python
class TraceFormatter:
    """Human-readable trace output for debugging."""

    SYMBOLS = {
        RunStatus.SUCCESS: "✓",
        RunStatus.FAILURE: "✗",
        RunStatus.RUNNING: "⋯",
    }

    def format_tick(self, tick_num: int, node: BehaviorNode,
                    status: RunStatus, duration_ms: float,
                    bb_writes: Set[str]) -> str:
        symbol = self.SYMBOLS[status]
        line = f"  → {node.id}: {status.name} ({duration_ms:.0f}ms)"
        if bb_writes:
            line += f"\n    {symbol} wrote: {', '.join(bb_writes)}"
        return line
```

### E.3 Subtree Testing in Isolation

**Requirement:** Run any subtree with mock blackboard inputs.

```bash
$ vlt bt test research-runner \
    --input user_query="How does auth work?" \
    --input session_id="test-123" \
    --mock oracle_bridge  # Use mock instead of real API

Running research-runner with mock inputs...

[RESULT] SUCCESS in 2 ticks
Outputs:
  research_results: [...3 items...]
  summary: "Authentication uses JWT tokens..."

Mocked calls:
  oracle_bridge.search_code("How does auth work?") → [mocked 3 results]
```

**Implementation:**
```python
class SubtreeTestRunner:
    """Run subtrees in isolation with mock inputs."""

    def __init__(self, tree: BehaviorTree, mocks: Dict[str, Any] = None):
        self.tree = tree
        self.mocks = mocks or {}

    def run(self, inputs: Dict[str, Any]) -> TestResult:
        # Create isolated blackboard with inputs
        bb = TypedBlackboard(scope_name="test")
        for key, value in inputs.items():
            bb.register(key, type(value))
            bb.set(key, value)

        # Create mock services
        services = self._create_mock_services()

        # Run tree
        ctx = TickContext(blackboard=bb, services=services, trace_enabled=True)
        status = self.tree.tick(ctx)

        return TestResult(
            status=status,
            outputs=bb.snapshot(),
            trace=ctx.trace_log,
            mock_calls=self._get_mock_calls()
        )
```

### E.4 Error Messages for Tree Authors

**Requirement:** Errors point to exact Lua line with context and suggestions.

```
ERROR [E4004]: Subtree 'reserch-runner' not found

  Location: trees/oracle-agent.lua:42

  40 │     BT.sequence {
  41 │         BT.code_search { ... },
  42 │         BT.subtree_ref("reserch-runner"),  ← HERE
  43 │     }

  Did you mean: 'research-runner'?

  Available subtrees:
    - research-runner
    - tool-executor
    - context-loader
```

**Implementation:**
```python
class TreeAuthorError(Exception):
    """Error formatted for tree authors, not programmers."""

    def __init__(self, code: str, message: str,
                 source_file: str, line: int,
                 suggestion: str = None,
                 available: List[str] = None):
        self.code = code
        self.message = message
        self.source_file = source_file
        self.line = line
        self.suggestion = suggestion
        self.available = available

    def format(self) -> str:
        lines = [f"ERROR [{self.code}]: {self.message}", ""]

        # Show source context
        lines.append(f"  Location: {self.source_file}:{self.line}")
        lines.append("")
        lines.extend(self._format_source_context())

        # Suggestions
        if self.suggestion:
            lines.append(f"  Did you mean: '{self.suggestion}'?")

        if self.available:
            lines.append("")
            lines.append("  Available options:")
            for item in self.available[:5]:
                lines.append(f"    - {item}")

        return "\n".join(lines)
```

### E.5 Lua DSL Validation (Pre-Run)

**Requirement:** `vlt bt validate <file>` checks tree before running.

```bash
$ vlt bt validate trees/oracle-agent.lua

Validating oracle-agent.lua...

⚠ WARNING: Node 'search-code' reads 'user_query' but no parent writes it
  Suggestion: Add to tree's blackboard schema or ensure upstream node writes it

⚠ WARNING: Node 'generate-response' has no timeout decorator
  Long-running LLM calls should have explicit timeout

✗ ERROR: Circular reference detected
  oracle-agent → research-runner → oracle-agent

✓ Contracts valid (12 nodes checked)
✓ Subtree references valid (2 refs checked)

Result: 1 error, 2 warnings
```

**Checks performed:**
1. All subtree_ref targets exist
2. No circular references
3. Contract inputs are satisfied by parent outputs
4. Async nodes have timeout decorators
5. Lua syntax valid
6. BT.* API usage correct

### E.6 Hot Reload Feedback

**Requirement:** When editing a .lua file, immediate feedback on save.

```
[HOT RELOAD] Detected change: trees/oracle-agent.lua

Validating... ✓
Reloading... ✓ (policy: let-finish-then-swap)

Tree 'oracle-agent' updated:
  - Modified: generate-response (changed model to claude-sonnet-4)
  - Added: new node 'validate-input'
  - Removed: deprecated node 'old-fallback'

Ready for next execution.
```

---

## E.7 Documentation Requirements

### E.7.1 Error Code Reference

Create `docs/error-codes.md` with:
- Every E1xxx-E8xxx code
- Example message with source context
- Common causes
- Resolution steps
- Example fix

### E.7.2 Contract Examples

Create `docs/contracts.md` with:
- How to declare NodeContract
- Common contract patterns
- Validation example outputs
- Debugging contract violations

### E.7.3 Type Coercion Guide

Create `docs/lua-python-types.md` with:
- Complete type mapping table
- Edge case examples
- Schema design recommendations

### E.7.4 Tree Authoring Guide

Create `docs/authoring-trees.md` with:
- BT.* API reference
- Common patterns (retry, fallback, parallel research)
- Debugging workflow
- Testing subtrees

---

## F. Deprecations

### F.1 Immediate Reload Policy

**Status:** REMOVED
**Reason:** Too dangerous per UB-08 analysis
**Migration:** Use `LET_FINISH_THEN_SWAP` or `CANCEL_AND_RESTART`

### F.2 Untyped Blackboard Access

**Status:** DEPRECATED
**Timeline:** Warning in 1.0, Error in 2.0
**Reason:** Type safety violations (TS-02)
**Migration:** Always use `bb.get(key, Schema)` not `bb._data[key]`

---

## G. Future Considerations

These items are OUT OF SCOPE for MVP but documented for future:

1. **Distributed BT execution** - Would require consensus/quorum
2. **Persistent RUNNING state** - Resume tree after process restart
3. **Visual debugger** - Real-time tree visualization
4. **BERT semantic conditions** - ML-based condition evaluation
5. **Tree versioning** - A/B testing different tree versions

---

## Appendix: Cross-Reference to Contracts

| Footgun ID | Contract Coverage | Addendum Section |
|------------|-------------------|------------------|
| UB-01 | `errors.yaml:E3003` | A.1 Progress Tracking |
| UB-02 | `errors.yaml:E4004`, `tree-loader.yaml` | A.5 Lazy Subtree |
| UB-03 | `errors.yaml:E5001-E5004` | - |
| UB-04 | `errors.yaml:E1002`, `blackboard.yaml` | - |
| UB-05 | `errors.yaml:E8001`, `nodes.yaml:Parallel` | A.3 Scope Isolation |
| UB-06 | `events.yaml:tree.budget.yielded` | - |
| UB-07 | `errors.yaml:E6001` | - |
| UB-08 | `tree-loader.yaml:ReloadPolicy` | - |
| UB-09 | - | A.2 Event Buffering |
| UB-10 | `nodes.yaml:ForEach` | - |
| SEC-01 | `errors.yaml:E7001` | B.3 Security Tests |
| SEC-02 | `errors.yaml:E5003` | - |
| SEC-03 | `errors.yaml:E1004` | D.1 Size Tracking |
| SEC-04 | `errors.yaml:E7002` | B.3 Security Tests |
| SEC-05 | `errors.yaml:E1003`, `blackboard.yaml` | - |
| RC-01 | `nodes.yaml:Parallel` | A.3 Scope Isolation |
| RC-02 | - | A.2 Event Buffering |
| RC-04 | `nodes.yaml:LLMCall` | - |
| TS-01 | `blackboard.yaml:type_coercion` | C. Type Coercion |
| TS-02 | `blackboard.yaml:set` | F.2 Deprecation |
| EP-01 | `nodes.yaml:error_propagation_rules` | B.2 Tests |
| EP-02 | `nodes.yaml:Parallel` | - |
| EP-03 | `errors.yaml:failure_trace_schema` | - |
