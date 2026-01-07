# Footgun Audit & Interface Specifications

**Date:** 2026-01-07
**Spec:** 019-bt-universal-runtime
**Author:** Claude Opus 4.5
**Purpose:** Identify undefined behaviors, security gaps, race conditions, and specify precise class interfaces

---

## Part 1: Footgun Inventory

### 1.1 Undefined Behaviors

#### UB-01: Node Returns RUNNING Forever (CRITICAL)

**Scenario:** A node enters RUNNING state and never completes (e.g., network hang, Lua infinite loop, orphaned async task).

**What spec says:** FR-5 mentions stuck detection ("RUNNING for longer than `:timeout` without progress") but:
- "Progress" is undefined - is it blackboard writes? Tick count? External callback?
- No default timeout specified - what happens if node has no timeout decorator?
- `Timeout` decorator exists but only resets child, doesn't propagate cancellation upstream

**Impact:** Tree hangs forever, blocks all downstream processing, memory leak from accumulated state.

**Resolution:**
1. Define "progress" as: blackboard write, async operation completion, or explicit `context.mark_progress()`
2. Require tree-level `max_running_duration_ms` (default 60000) that applies to ALL nodes without explicit timeout
3. RUNNING nodes MUST track `running_since: datetime` (already in data-model.md but not enforced)
4. Add `TreeWatchdog` that runs every tick checking all RUNNING nodes against tree-level timeout

---

#### UB-02: Subtree Reference to Non-Existent Tree (HIGH)

**Scenario:** Lua DSL uses `BT.subtree_ref("nonexistent")` and tree doesn't exist in registry.

**What spec says:** FR-2 says "Circular tree references detected and rejected" but says nothing about missing references.

**Current state:** `BTValidator` task exists (2.3.3) but behavior on missing ref is unspecified.

**Impact:** Runtime crash or silent FAILURE when subtree node ticks.

**Resolution:**
1. `BTValidator.validate_subtree_refs()` MUST run at load time, before tree registration
2. Missing subtree ref = load error with clear message: `SubtreeNotFoundError: Tree 'foo' referenced by 'bar.lua:line 42' not found. Available trees: [...]`
3. Option: Allow lazy resolution with explicit `BT.subtree_ref("name", { lazy = true })` for dynamic tree loading

---

#### UB-03: Lua Script Throws Error (HIGH)

**Scenario:** User Lua code throws: `error("something went wrong")` or causes nil dereference.

**What spec says:** FR-2 says "Lua syntax errors produce clear error messages with line numbers" but nothing about runtime errors.

**Current impl:** `LuaSandbox.execute()` catches `LuaError` and raises `LuaExecutionError`. But:
- Error message doesn't include Lua stack trace
- `ScriptNode._tick()` catches exception, logs warning, returns FAILURE
- No distinction between "script bug" and "intended FAILURE"

**Impact:** Debugging nightmare - can't tell if FAILURE is intentional or script bug.

**Resolution:**
1. `LuaExecutionError` MUST include Lua traceback: `LuaExecutionError(message, lua_traceback=str, line_number=int)`
2. `ScriptNode` MUST write error details to blackboard: `bb.set("_last_script_error", { message, traceback, line })`
3. Add explicit failure mechanism: `return { status = "failure", reason = "intended failure" }` vs thrown error
4. Emit ANS event `script.error` with severity HIGH for thrown errors (not intended failures)

---

#### UB-04: Blackboard Schema Validation Fails Mid-Tick (MEDIUM)

**Scenario:** Node calls `bb.set("key", value)` but value doesn't match registered Pydantic schema.

**What spec says:** state-architecture.md shows `BlackboardValidationError` raised, but:
- What happens to the tick?
- Does partial state from earlier writes persist?
- Is this recoverable?

**Impact:** Inconsistent blackboard state, hard to debug, partial writes may corrupt system.

**Resolution:**
1. Schema validation errors are ALWAYS recoverable - they should NOT crash the tick
2. On validation error:
   a. DO NOT modify blackboard state
   b. Return `ValidationFailure` (new RunStatus variant? or wrap in result?)
   c. Log with full context: key, expected schema, actual value, node path
3. Alternative: Make `bb.set()` return `Result[None, ValidationError]` instead of raising
4. Add `bb.set_unsafe(key, value)` for emergency unvalidated writes (with warning log)

**Recommendation:** Keep it simple - validation error = node returns FAILURE with error in blackboard. Don't add new RunStatus.

---

#### UB-05: MergeStrategy FAIL_ON_CONFLICT Actually Conflicts (HIGH)

**Scenario:** Parallel node with `merge_strategy = "fail"` has two children write same key.

**What spec says:** state-architecture.md shows: `conflicts.append(...)` and "Don't write - leave parent unchanged"

**Problems:**
1. What status does Parallel return? SUCCESS (because policy met) or FAILURE (because merge failed)?
2. If children succeeded but merge fails, is that a Parallel failure?
3. Conflicts list is returned but not persisted - caller can't inspect it

**Impact:** Ambiguous success/failure semantics, lost conflict information.

**Resolution:**
1. FAIL_ON_CONFLICT conflict = Parallel returns FAILURE regardless of child policy
2. Conflicts written to blackboard: `bb.set("_parallel_conflicts", conflicts_list)`
3. Emit ANS event `tree.parallel.conflict` with severity WARNING
4. Document clearly: "FAIL_ON_CONFLICT is for critical sections where any conflict is unacceptable"

---

#### UB-06: Tick Budget Exceeded (MEDIUM)

**Scenario:** `tick_budget = 1000` and tree ticks 1001 times in single event handling.

**What spec says:** FR-4 pseudocode shows `if ctx.tick_count >= ctx.tick_budget: tree.schedule_resume(); return RUNNING`

**Problems:**
1. What does `schedule_resume()` do? Event emission? Polling? Manual re-tick?
2. If budget exceeded in middle of Sequence, what's the sequence state?
3. How does caller distinguish "budget RUNNING" from "async RUNNING"?

**Impact:** Caller can't properly handle budget exhaustion vs normal async wait.

**Resolution:**
1. Define `schedule_resume()`: Emit `tree.budget.yield` event with tree_id and tick_count
2. Tree status becomes `YIELDED` (new TreeStatus variant) distinct from `RUNNING`
3. Caller receives `TickResult(status=YIELDED, resume_token=str)`
4. Resume via `tree.resume(resume_token)` or automatic on next event
5. Sequence/Composite state MUST be preserved across yield (already is, but document it)

---

#### UB-07: Async Operation Timeout (MEDIUM)

**Scenario:** LLM node initiates HTTP request, request hangs for 5 minutes, no timeout configured.

**What spec says:** FR-3 has `timeout = seconds` as optional property.

**Problems:**
1. No default timeout - what happens with `timeout = None`?
2. HTTP client may have its own timeout, but no coordination
3. `asyncio.wait_for()` timeout vs HTTP timeout vs node timeout - which wins?

**Impact:** Resource exhaustion, blocked tree, user waits forever.

**Resolution:**
1. Default timeout MUST exist: `DEFAULT_LLM_TIMEOUT = 120` seconds
2. Timeout hierarchy (outer cancels inner):
   a. Tree-level `max_tick_duration` (outermost)
   b. Node-level `timeout` property
   c. HTTP client timeout (should be slightly longer than node timeout)
3. On timeout: Cancel HTTP request (not just ignore response), emit `llm.timeout` event, return FAILURE with reason

---

#### UB-08: Hot Reload Mid-Tick with RUNNING Nodes (MEDIUM)

**Scenario:** Tree is mid-tick, LLM node is RUNNING, developer saves .lua file triggering reload.

**What spec says:** FR-6 has policies: `cancel-and-restart`, `let-finish-then-swap`, `immediate`

**Problems:**
1. `let-finish-then-swap`: What if current execution takes 10 minutes? User waits?
2. `immediate`: How does state transfer work? Node IDs may have changed.
3. `cancel-and-restart`: What about in-flight HTTP requests? Do they get cancelled?

**Impact:** State corruption, orphaned resources, unpredictable behavior.

**Resolution:**
1. `let-finish-then-swap` (default): Queue reload, apply after current tick completes. Max queue depth = 1 (latest wins).
2. `cancel-and-restart`:
   a. Set `tree.cancellation_requested = True`
   b. Each RUNNING node checks flag at start of tick and returns FAILURE with `CancellationReason.RELOAD`
   c. LLM nodes MUST cancel HTTP clients via `asyncio.Task.cancel()`
   d. Wait for all async operations to complete or timeout (5s)
   e. Apply new tree
3. `immediate`: DEPRECATED - too dangerous. Remove from spec or mark as "development only"
4. Add `tree.reload_pending: bool` observable property

---

#### UB-09: Event Handler Modifies State Another Handler Reads (MEDIUM)

**Scenario:** Two ANS event handlers subscribed to `tool.call.success`. Handler A writes `bb.set("result", ...)`, Handler B reads `bb.get("result")`. Order matters.

**What spec says:** Nothing explicitly about handler ordering.

**Current impl:** EventBus uses list, handlers called in subscription order.

**Impact:** Non-deterministic behavior depending on subscription order.

**Resolution:**
1. Document clearly: Handlers execute sequentially in subscription order
2. Add `priority` to handler registration: `bus.subscribe("event", handler, priority=10)`
3. Within same priority, FIFO order preserved
4. Handlers SHOULD NOT read blackboard keys written by other handlers in same event dispatch
5. Emit ANS event `handler.conflict.warning` if detected (detect via read/write tracking)

---

#### UB-10: ForEach Over Empty Collection (LOW)

**Scenario:** `BT.for_each("researchers", { ... })` and `bb.get("researchers")` returns empty list.

**What spec says:** No explicit semantics for empty iteration.

**Impact:** Ambiguous - should this be SUCCESS (did nothing) or FAILURE (nothing to do)?

**Resolution:**
1. Empty collection = immediate SUCCESS (consistent with functional map over empty)
2. Add optional `BT.for_each("key", { min_items = 1, ... })` to require non-empty
3. Document: "ForEach over empty is SUCCESS. Use Guard to check non-empty first if FAILURE desired."

---

### 1.2 Security Gaps

#### SEC-01: Lua Sandbox Escape via Metatable Injection (HIGH)

**Current mitigation:** `setmetatable` is in BLOCKED_GLOBALS.

**Attack vector:** If ANY object passed to Lua has a metatable with `__index` pointing to blocked functions, script can access them.

**Example:**
```lua
-- If some object 'obj' has metatable with __index = _G
local os = getmetatable(obj).__index.os
os.execute("rm -rf /")
```

**Current impl:** `_context_to_lua()` creates fresh tables via `lua.table()` which have no metatables.

**Gap:** If future code passes Python objects directly (not via `_python_to_lua`), metatables may leak.

**Resolution:**
1. Add test case that verifies metatable access is blocked
2. `_python_to_lua()` MUST be used for ALL values passed to Lua - no direct object passing
3. Audit all code paths that call `sandbox_env[key] = value`
4. Consider: Strip metatables from all returned values: `setmetatable(t, nil)` before returning

---

#### SEC-02: Infinite Loop in Lua Script (MEDIUM)

**Current mitigation:** `timeout_seconds = 5.0` default, enforced via threading.

**Gap:** Timeout relies on thread joining. Python's GIL may delay detection. lupa doesn't support instruction counting.

**Attack:**
```lua
while true do end  -- Tight loop, GIL contention may delay interrupt
```

**Impact:** 5 seconds of 100% CPU per script. If many scripts, DoS.

**Resolution:**
1. Current timeout is acceptable for moderate usage
2. Add rate limiting: Max 10 script executions per second per user
3. Add metric tracking: Log scripts that reach timeout
4. Future: Consider LuaJIT hooks for instruction counting (complex)

---

#### SEC-03: Memory Exhaustion via Blackboard (MEDIUM)

**Attack:** Node writes massive string to blackboard: `bb.set("data", "A" * 10_000_000_000)`

**Current mitigation:** None explicit.

**Impact:** OOM, system crash.

**Resolution:**
1. Add `max_blackboard_size_bytes` config (default 100MB)
2. `bb.set()` checks total blackboard size after write, raises `BlackboardSizeExceededError` if over limit
3. Alternative: Per-key limit `bb.set(key, value, max_size=1_000_000)`
4. Global blackboard persistence to SQLite naturally limits via database size

---

#### SEC-04: Path Traversal in Subtree Ref (MEDIUM)

**Attack:** `BT.subtree_ref("../../../etc/passwd")` or `BT.subtree_ref("/etc/passwd")`

**Current impl:** No path sanitization visible in spec.

**Impact:** Could load arbitrary Lua files outside tree directory.

**Resolution:**
1. Subtree refs are NAMES not PATHS: `BT.subtree_ref("research")` maps to `trees/research.lua`
2. Names MUST match `^[a-zA-Z][a-zA-Z0-9_-]*$` - no slashes, dots, or special chars
3. `TreeRegistry.get(name)` validates name format before ANY file operations
4. Tree directory MUST be explicitly configured, not derived from refs

---

#### SEC-05: Event Injection via Blackboard Keys (LOW)

**Attack:** Node writes to internal key: `bb.set("_parallel_conflicts", malicious_data)`

**Gap:** Internal keys (`_` prefix) not protected.

**Impact:** Could corrupt internal state or inject false conflict data.

**Resolution:**
1. Reserve `_` prefix for system use
2. `bb.set()` rejects keys starting with `_` unless `internal=True` flag passed
3. Only runtime code (not user nodes) may pass `internal=True`
4. Alternative: Use separate namespace for internal state

---

### 1.3 Race Conditions & Ordering Issues

#### RC-01: Parallel Children Tick Order (MEDIUM)

**Scenario:** Parallel with 3 children. What's the tick order? Does it matter?

**What spec says:** Nothing explicit.

**Current impl:** `for child in self._children: status = child.tick(context)`

**Issue:** If child A writes to blackboard, child B reads it, behavior depends on order.

**Resolution:**
1. Document: Children tick in list order (definition order in Lua)
2. state-architecture.md already specifies child scopes - enforce this:
   ```python
   for child in self._children:
       child_scope = ctx.blackboard.create_child_scope(f"parallel_{child.id}")
       child.tick(ctx.with_blackboard(child_scope))
   ```
3. Children CANNOT see each other's writes until merge phase
4. This is already designed but not yet implemented - add to task 0.6.5

---

#### RC-02: Event Buffer During Tick (MEDIUM)

**Scenario:** During tick, node emits event. Event handler triggers another tree tick.

**What spec says:** plan.md mentions "Buffer events during tick, dispatch after completion"

**Gap:** Implementation details unclear:
- Is buffer per-tree or global?
- What if buffered event would modify same tree?
- What's the max buffer size?

**Resolution:**
1. Global event buffer (not per-tree)
2. During ANY tree tick, ALL events buffered
3. After tick completes, process buffered events in FIFO order
4. If buffered event targets same tree and tree is RUNNING, queue for next poll interval
5. Max buffer size: 1000 events. If exceeded, oldest dropped with warning log.

---

#### RC-03: Hot Reload vs Active Tick (LOW)

**Scenario:** File watcher triggers reload callback while tick is in progress.

**Resolution:** Already covered in UB-08. Reload is queued, not immediate.

---

#### RC-04: AsyncIO Task Ordering in LLM Node (LOW)

**Scenario:** Multiple LLM nodes streaming. Chunk arrival order may not match tick order.

**Impact:** Chunks could be written to wrong blackboard keys if not careful.

**Resolution:**
1. Each LLM node has unique `request_id`
2. Chunk callbacks MUST verify `request_id` matches before writing
3. Orphaned chunks (request_id not found) are logged and discarded

---

### 1.4 Type Safety Holes

#### TS-01: Lua to Python Type Coercion (MEDIUM)

**Gap areas:**
1. Lua number → Python: Float or int? (`1.0` vs `1`)
2. Lua nil → Python: `None` or missing key?
3. Lua array vs dict: Table with `{1: "a", 2: "b"}` vs `{"a": 1, "b": 2}`

**Current impl:** `_lua_to_python()` converts tables to dicts, numbers stay as-is.

**Issue:** Pydantic schema expects `int`, Lua returns `1.0`, validation fails.

**Resolution:**
1. Document Lua → Python type mapping explicitly:
   - `nil` → `None`
   - `boolean` → `bool`
   - `number` → `float` (always, even if integral)
   - `string` → `str`
   - `table` with sequential integer keys starting at 1 → `list`
   - `table` otherwise → `dict`
2. Add helper: `lua_to_int(value)` that converts `1.0` → `1`
3. TypedBlackboard schemas should use `float` for numeric fields from Lua

---

#### TS-02: Blackboard Keys Bypassing Schema (MEDIUM)

**Scenario:** Code uses `bb._data["key"] = value` directly, bypassing `set()` validation.

**Current impl:** `_data` is "private" but Python doesn't enforce.

**Resolution:**
1. Consider `__slots__` to prevent attribute addition
2. Add runtime check: `bb.get()` verifies returned value matches schema (expensive but safe)
3. Alternative: Trust internal code, rely on code review. Document "never access `_data` directly"

---

#### TS-03: Tool Results with Unexpected Shapes (MEDIUM)

**Scenario:** Tool executor returns `{"result": {"nested": {"deep": ...}}}` but schema expects `{"result": str}`.

**Gap:** Tool contracts exist in research/04 but no runtime enforcement.

**Resolution:**
1. Add `ToolResultSchema` Pydantic model per tool
2. Tool executor validates result against schema before returning
3. Validation failure = tool failure (not silent corruption)

---

### 1.5 Error Propagation Ambiguity

#### EP-01: Failed Child Causes Parent Failure? (HIGH)

**Question:** Does FAILURE propagate up? Is it configurable?

**Current behavior by node type:**
- `Sequence`: Any child FAILURE → immediate parent FAILURE (correct)
- `Selector`: All children FAILURE → parent FAILURE (correct)
- `Parallel(REQUIRE_ALL)`: Any FAILURE → parent FAILURE (correct)
- `Parallel(REQUIRE_ONE)`: All FAILURE → parent FAILURE (correct)
- `Decorator`: Varies - some transform, some propagate

**Issue:** For new decorators/nodes, rule isn't clear.

**Resolution:** Document explicitly:
1. FAILURE from leaf = leaf's decision
2. FAILURE from composite = based on policy
3. FAILURE from decorator = decorator's decision (transform or propagate)
4. Add `@propagates_failure` decorator annotation for documentation

---

#### EP-02: Partial Failures in Parallel (HIGH)

**Scenario:** Parallel(REQUIRE_ONE) with 5 children. Child 1 succeeds, child 2 fails, children 3-5 RUNNING.

**Questions:**
1. What happens to children 3-5? Keep running? Cancel?
2. Is the failed child's error preserved anywhere?
3. Does child 2's failure affect anything if policy already met?

**Current impl:** Parallel ticks ALL children every time, doesn't track individual failures.

**Resolution:**
1. On policy satisfaction (e.g., first SUCCESS for REQUIRE_ONE):
   - If `on_child_fail = "cancel-siblings"`: Cancel remaining (set cancel flag)
   - If `on_child_fail = "continue"`: Let them finish
2. Failed children's errors written to blackboard: `_parallel_child_errors: [{child_id, error}]`
3. Final Parallel result includes summary: `{successes: 2, failures: 1, cancelled: 2}`

---

#### EP-03: Error Surfacing to User (MEDIUM)

**Question:** When tree fails, how does user know why?

**Current:** Tree returns FAILURE status. Error might be in logs. Maybe in blackboard.

**Resolution:**
1. Every FAILURE MUST have reason in blackboard: `_failure_reason: str`
2. Tree maintains `failure_trace: List[{node_id, reason, timestamp}]`
3. On tree FAILURE, emit ANS event `tree.failed` with failure_trace
4. Frontend displays failure trace in debug panel

---

## Part 2: Interface Specifications

### 2.1 TypedBlackboard

```python
from typing import TypeVar, Type, Optional, Dict, Set, Any, Union
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class TypedBlackboard:
    """Type-safe hierarchical blackboard with schema enforcement.

    Invariants:
    - All keys in _data have corresponding entry in _schemas
    - All values in _data are valid instances of their schema
    - Parent chain is acyclic (no circular references)
    - _scope_name is non-empty

    Thread Safety:
    - NOT thread-safe (asyncio single-threaded assumption)
    - Create child scopes for parallel isolation
    """

    # === Constructor ===

    def __init__(
        self,
        parent: Optional["TypedBlackboard"] = None,
        scope_name: str = "root"
    ) -> None:
        """Create a new blackboard scope.

        Preconditions:
        - scope_name is non-empty string
        - parent is None or valid TypedBlackboard
        - No circular parent references

        Postconditions:
        - Instance has empty _data
        - Instance inherits parent's _schemas
        - _reads and _writes are empty sets
        """
        ...

    # === Schema Registration ===

    def register(self, key: str, schema: Type[T]) -> None:
        """Register expected type for a key.

        Preconditions:
        - key is non-empty string
        - key does not start with '_' (reserved for system)
        - schema is subclass of BaseModel

        Postconditions:
        - _schemas[key] = schema
        - Does NOT affect _data
        """
        ...

    def register_many(self, schemas: Dict[str, Type[BaseModel]]) -> None:
        """Register multiple schemas atomically.

        Preconditions:
        - All keys/schemas valid per register()

        Postconditions:
        - All schemas registered
        - Atomic: all succeed or none registered
        """
        ...

    # === Data Access ===

    def get(
        self,
        key: str,
        schema: Type[T],
        default: Optional[T] = None
    ) -> Optional[T]:
        """Get typed value with scope chain lookup.

        Preconditions:
        - key is non-empty string
        - schema matches registered schema for key (if registered)

        Postconditions:
        - key added to _reads
        - Returns value from this scope or nearest parent
        - Returns default if not found and default provided
        - Returns None if not found and no default

        Raises:
        - BlackboardKeyError: key not registered and no default
        """
        ...

    def set(self, key: str, value: Union[BaseModel, Dict[str, Any]]) -> None:
        """Set validated value in current scope.

        Preconditions:
        - key is non-empty string
        - key is registered in _schemas
        - value matches schema (will be validated)

        Postconditions:
        - _data[key] = validated_value
        - key added to _writes

        Raises:
        - BlackboardKeyError: key not registered
        - BlackboardValidationError: value doesn't match schema
        """
        ...

    def set_global(self, key: str, value: Union[BaseModel, Dict[str, Any]]) -> None:
        """Set value in root (global) scope.

        Preconditions:
        - Same as set()

        Postconditions:
        - Value set in root scope (follows parent chain)
        - key added to this scope's _writes
        """
        ...

    def has(self, key: str) -> bool:
        """Check if key exists in this scope or parents.

        Postconditions:
        - Does NOT add to _reads (peek only)
        - Returns True if value exists anywhere in scope chain
        """
        ...

    def delete(self, key: str) -> bool:
        """Delete key from this scope only.

        Postconditions:
        - Key removed from _data (not parent scopes)
        - Returns True if key was present
        - Does NOT affect _schemas
        """
        ...

    # === Scope Management ===

    def create_child_scope(self, scope_name: str) -> "TypedBlackboard":
        """Create isolated child scope.

        Preconditions:
        - scope_name is non-empty

        Postconditions:
        - New blackboard with self as parent
        - Inherits all schemas
        - Empty _data (no value inheritance)
        """
        ...

    # === Access Tracking ===

    def get_reads(self) -> Set[str]:
        """Get keys read since last clear. Used for contract validation."""
        ...

    def get_writes(self) -> Set[str]:
        """Get keys written since last clear. Used for contract validation."""
        ...

    def clear_access_tracking(self) -> None:
        """Reset read/write tracking. Called at tick start."""
        ...

    # === Debugging ===

    def snapshot(self) -> Dict[str, Any]:
        """Create serializable snapshot of all data including parents."""
        ...
```

### 2.2 NodeContract

```python
from dataclasses import dataclass, field
from typing import Dict, Type, Set, List
from pydantic import BaseModel


@dataclass(frozen=True)
class NodeContract:
    """Declares a node's state requirements and outputs.

    Invariants:
    - inputs, optional_inputs, and outputs have disjoint keys
    - All schema types are BaseModel subclasses
    """

    inputs: Dict[str, Type[BaseModel]] = field(default_factory=dict)
    """Required inputs. Node will fail if missing."""

    optional_inputs: Dict[str, Type[BaseModel]] = field(default_factory=dict)
    """Optional inputs. Node works without them."""

    outputs: Dict[str, Type[BaseModel]] = field(default_factory=dict)
    """State this node may produce."""

    description: str = ""
    """Human-readable description for documentation."""

    def validate_inputs(self, blackboard: "TypedBlackboard") -> List[str]:
        """Check all required inputs exist.

        Returns:
        - Empty list if all inputs present
        - List of missing key names otherwise
        """
        ...

    def validate_access(
        self,
        reads: Set[str],
        writes: Set[str]
    ) -> List[str]:
        """Check actual access matches declared contract.

        Returns:
        - Empty list if compliant
        - List of violation descriptions otherwise

        Violations:
        - Read undeclared key (not in inputs or optional_inputs)
        - Wrote undeclared key (not in outputs)
        """
        ...

    @property
    def all_input_keys(self) -> Set[str]:
        """Union of required and optional input keys."""
        return set(self.inputs.keys()) | set(self.optional_inputs.keys())
```

### 2.3 BehaviorNode (Base)

```python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum, auto


class RunStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()
    FRESH = auto()  # Not yet ticked

    @classmethod
    def from_bool(cls, value: bool) -> "RunStatus":
        """Convert boolean to SUCCESS/FAILURE."""
        ...


class BehaviorNode(ABC):
    """Base class for all behavior tree nodes.

    Invariants:
    - _id is unique within tree
    - _status reflects result of most recent tick
    - _tick_count >= 0
    - If _status == RUNNING, _running_since is not None
    """

    # === Constructor ===

    def __init__(
        self,
        id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create behavior node.

        Preconditions:
        - id is non-empty unique identifier

        Postconditions:
        - _status = FRESH
        - _tick_count = 0
        - _running_since = None
        """
        ...

    # === Properties (read-only) ===

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def status(self) -> RunStatus: ...

    @property
    def tick_count(self) -> int: ...

    @property
    def running_since(self) -> Optional[datetime]: ...

    @property
    def last_tick_duration_ms(self) -> float: ...

    # === Core Interface ===

    def tick(self, ctx: "TickContext") -> RunStatus:
        """Execute node for one tick.

        Preconditions:
        - ctx is valid TickContext
        - ctx.blackboard has schemas registered for contract inputs

        Postconditions:
        - _tick_count incremented
        - _status set to return value
        - If RUNNING: _running_since set (if was FRESH/SUCCESS/FAILURE)
        - If not RUNNING: _running_since cleared
        - _last_tick_duration_ms updated

        Implementation:
        1. Record start time
        2. Call _tick(ctx)
        3. Update state
        4. Return status
        """
        ...

    @abstractmethod
    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Subclass implementation. Override this, not tick()."""
        ...

    def reset(self) -> None:
        """Reset node to initial state.

        Postconditions:
        - _status = FRESH
        - _running_since = None
        - Does NOT reset _tick_count (intentional for debugging)
        - Subclass state reset
        """
        ...

    # === Contract Support ===

    @classmethod
    def contract(cls) -> "NodeContract":
        """Declare state requirements. Override in subclasses.

        Default: Empty contract (no requirements).
        """
        return NodeContract()

    # === Debugging ===

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information dictionary.

        Includes: id, name, status, tick_count, running_since,
                  last_tick_duration_ms, and subclass-specific info.
        """
        ...
```

### 2.4 TickContext

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Set


@dataclass
class TickContext:
    """Context passed to nodes during tick execution.

    Invariants:
    - tick_count >= 0
    - tick_budget > 0
    - start_time <= now()
    - parent_path has no duplicates
    """

    # === Required Fields ===

    blackboard: "TypedBlackboard"
    """Current scope blackboard."""

    services: "Services"
    """Dependency injection container."""

    # === Tick Tracking ===

    tick_count: int = 0
    """Ticks in current execution cycle."""

    tick_budget: int = 1000
    """Maximum ticks before yield."""

    start_time: datetime = field(default_factory=datetime.utcnow)
    """When this tick cycle started."""

    # === Debugging ===

    parent_path: List[str] = field(default_factory=list)
    """Path of parent node IDs (for debugging)."""

    trace_enabled: bool = False
    """Whether to log every tick."""

    # === Async Coordination ===

    async_pending: Set[str] = field(default_factory=set)
    """IDs of pending async operations."""

    # === Cancellation ===

    cancellation_requested: bool = False
    """Set True to request graceful cancellation."""

    # === Methods ===

    def elapsed_ms(self) -> float:
        """Milliseconds since start_time."""
        ...

    def budget_remaining(self) -> int:
        """Ticks remaining before yield."""
        return self.tick_budget - self.tick_count

    def budget_exceeded(self) -> bool:
        """True if tick_count >= tick_budget."""
        return self.tick_count >= self.tick_budget

    def push_path(self, node_id: str) -> None:
        """Add node to parent path. Called before child tick."""
        ...

    def pop_path(self) -> None:
        """Remove last node from parent path. Called after child tick."""
        ...

    def with_blackboard(self, bb: "TypedBlackboard") -> "TickContext":
        """Create copy with different blackboard (for child scopes)."""
        ...

    # === Async Operations ===

    def add_async(self, op_id: str) -> None:
        """Register pending async operation."""
        ...

    def complete_async(self, op_id: str) -> None:
        """Mark async operation complete."""
        ...

    def has_pending_async(self) -> bool:
        """True if any async operations pending."""
        return len(self.async_pending) > 0
```

### 2.5 TreeRegistry

```python
from typing import Dict, Optional, List, Callable
from pathlib import Path


class TreeRegistry:
    """Manages loaded behavior trees.

    Invariants:
    - All tree IDs in _trees are unique
    - All loaded trees have valid root nodes
    - Hot reload queue depth <= 1 (latest wins)
    """

    def __init__(
        self,
        tree_dir: Path,
        default_reload_policy: "ReloadPolicy" = ReloadPolicy.LET_FINISH_THEN_SWAP
    ) -> None:
        """Create registry.

        Preconditions:
        - tree_dir exists and is directory

        Postconditions:
        - _trees is empty
        - File watcher NOT started (call start_watching)
        """
        ...

    # === Tree Management ===

    def load(self, path: Path) -> "BehaviorTree":
        """Load tree from Lua file.

        Preconditions:
        - path exists and is .lua file
        - path is within tree_dir (no traversal)

        Postconditions:
        - Tree validated (syntax, refs, schema)
        - Tree registered in _trees
        - Returns loaded tree

        Raises:
        - TreeLoadError: Parse/validation failed
        - SubtreeNotFoundError: Missing subtree reference
        - FileNotFoundError: Path doesn't exist
        """
        ...

    def get(self, tree_id: str) -> Optional["BehaviorTree"]:
        """Get tree by ID.

        Returns:
        - Tree if found, None otherwise
        """
        ...

    def reload(self, tree_id: str, policy: Optional["ReloadPolicy"] = None) -> None:
        """Reload tree from source file.

        Preconditions:
        - tree_id exists in registry

        Postconditions:
        - If tree is RUNNING and policy is LET_FINISH_THEN_SWAP:
            - Reload queued (applied after completion)
        - If tree is IDLE or policy is CANCEL_AND_RESTART:
            - Tree reloaded immediately
        - Reload logged with diff

        Raises:
        - TreeNotFoundError: tree_id not in registry
        - TreeLoadError: New definition invalid
        """
        ...

    def unload(self, tree_id: str) -> None:
        """Remove tree from registry.

        Preconditions:
        - Tree is not RUNNING

        Postconditions:
        - Tree removed from _trees

        Raises:
        - TreeInUseError: Tree is currently RUNNING
        """
        ...

    def list_trees(self) -> List[str]:
        """Return all registered tree IDs."""
        ...

    # === Hot Reload ===

    def start_watching(self) -> None:
        """Start file watcher for tree_dir.

        Postconditions:
        - File changes trigger reload
        - Multiple rapid changes batched (debounce 500ms)
        """
        ...

    def stop_watching(self) -> None:
        """Stop file watcher."""
        ...

    def on_file_changed(self, path: Path) -> None:
        """Handle file change (internal, called by watcher).

        Implementation:
        1. Debounce rapid changes
        2. Find tree using this file
        3. Call reload with default policy
        """
        ...
```

### 2.6 TreeLoader (Lua DSL)

```python
from pathlib import Path
from typing import Dict, Any


class TreeLoader:
    """Loads behavior trees from Lua DSL files.

    Invariants:
    - Uses sandboxed Lua execution (LuaSandbox)
    - BT.* namespace injected into environment
    """

    def __init__(self, sandbox_timeout: float = 5.0) -> None:
        """Create loader.

        Preconditions:
        - lupa library available

        Postconditions:
        - LuaSandbox configured
        - BT.* functions registered
        """
        ...

    def load(self, path: Path) -> "TreeDefinition":
        """Load tree definition from Lua file.

        Preconditions:
        - path exists and is readable
        - path is .lua file

        Postconditions:
        - Returns TreeDefinition (not built tree)

        Raises:
        - LuaSyntaxError: Invalid Lua syntax (includes line number)
        - LuaTimeoutError: Script exceeded timeout
        - TreeDefinitionError: Invalid tree structure
        """
        ...

    def load_string(self, lua_code: str, source_name: str = "<string>") -> "TreeDefinition":
        """Load tree from Lua string (for testing)."""
        ...

    def _inject_bt_api(self, env: Dict[str, Any]) -> None:
        """Inject BT.* functions into Lua environment.

        Registers:
        - BT.tree(name, config) -> TreeDef
        - BT.sequence(children) -> SequenceDef
        - BT.selector(children) -> SelectorDef
        - BT.parallel(config, children) -> ParallelDef
        - BT.action(name, config) -> ActionDef
        - BT.condition(name, config) -> ConditionDef
        - BT.llm_call(config) -> LLMCallDef
        - BT.subtree_ref(name) -> SubtreeRefDef
        - BT.for_each(key, config) -> ForEachDef
        - BT.script(name, config) -> ScriptDef
        - BT decorators: timeout, retry, guard, cooldown, etc.
        """
        ...


class TreeBuilder:
    """Builds BehaviorTree from TreeDefinition.

    Invariants:
    - All fn references resolved before build completes
    - All subtree refs validated
    """

    def __init__(self, registry: "TreeRegistry") -> None:
        """Create builder with registry for subtree resolution."""
        ...

    def build(self, definition: "TreeDefinition") -> "BehaviorTree":
        """Build executable tree from definition.

        Preconditions:
        - definition is valid TreeDefinition
        - All subtree refs exist in registry
        - All fn paths resolve to callable

        Postconditions:
        - Returns BehaviorTree with all nodes instantiated
        - Tree blackboard has schema registered

        Raises:
        - SubtreeNotFoundError: subtree_ref target missing
        - FunctionNotFoundError: fn path doesn't resolve
        - SchemaValidationError: blackboard schema invalid
        """
        ...


class TreeValidator:
    """Validates tree definitions before building.

    Run at load time to catch errors early.
    """

    def validate(
        self,
        definition: "TreeDefinition",
        registry: "TreeRegistry"
    ) -> List["ValidationError"]:
        """Validate tree definition.

        Checks:
        - All fn paths resolve to callables
        - All subtree_refs exist or are marked lazy
        - No circular subtree references
        - Required node properties present
        - Blackboard schema valid

        Returns:
        - Empty list if valid
        - List of ValidationError otherwise
        """
        ...
```

### 2.7 MergeStrategy Implementations

```python
from enum import Enum
from typing import List, Dict, Tuple, Any
from abc import ABC, abstractmethod


class MergeStrategyType(Enum):
    LAST_WINS = "last_wins"
    FIRST_WINS = "first_wins"
    COLLECT = "collect"
    MERGE_DICT = "merge_dict"
    FAIL_ON_CONFLICT = "fail"


class MergeStrategy(ABC):
    """Base class for parallel child result merge strategies.

    Used when parallel children write to same blackboard keys.
    """

    @abstractmethod
    def merge(
        self,
        key: str,
        values: List[Tuple[int, Any]]  # (child_index, value) pairs
    ) -> Tuple[Any, List[str]]:
        """Merge multiple values for same key.

        Args:
            key: Blackboard key being merged
            values: List of (child_index, value) pairs in tick order

        Returns:
            (merged_value, warnings) tuple
            - merged_value: Result to write (or None if no write)
            - warnings: List of conflict/issue messages
        """
        ...


class LastWinsStrategy(MergeStrategy):
    """Last child's value wins. No conflicts."""

    def merge(self, key: str, values: List[Tuple[int, Any]]) -> Tuple[Any, List[str]]:
        return values[-1][1], []


class FirstWinsStrategy(MergeStrategy):
    """First child's value wins. No conflicts."""

    def merge(self, key: str, values: List[Tuple[int, Any]]) -> Tuple[Any, List[str]]:
        return values[0][1], []


class CollectStrategy(MergeStrategy):
    """Collect all values into list."""

    def merge(self, key: str, values: List[Tuple[int, Any]]) -> Tuple[Any, List[str]]:
        return [v[1] for v in values], []


class MergeDictStrategy(MergeStrategy):
    """Deep merge dictionaries. Later values override."""

    def merge(self, key: str, values: List[Tuple[int, Any]]) -> Tuple[Any, List[str]]:
        result = {}
        for _, v in values:
            if hasattr(v, 'model_dump'):
                result.update(v.model_dump())
            elif isinstance(v, dict):
                result.update(v)
        return result, []


class FailOnConflictStrategy(MergeStrategy):
    """Raise conflict error. Does not write."""

    def merge(self, key: str, values: List[Tuple[int, Any]]) -> Tuple[Any, List[str]]:
        # Return None (no write) with conflict warning
        return None, [f"Conflict on key '{key}': {len(values)} writers"]


class ParallelMerger:
    """Orchestrates merge across all keys.

    Invariants:
    - default_strategy is never None
    - key_strategies overrides are applied
    """

    def __init__(self, default: MergeStrategyType = MergeStrategyType.COLLECT) -> None:
        """Create merger with default strategy."""
        ...

    def configure(self, key: str, strategy: MergeStrategyType) -> None:
        """Set strategy for specific key."""
        ...

    def merge(
        self,
        parent: "TypedBlackboard",
        children: List["TypedBlackboard"]
    ) -> "MergeResult":
        """Merge all child scopes into parent.

        Preconditions:
        - children are child scopes of parent

        Postconditions:
        - Non-conflicting values written to parent
        - Conflicts handled per strategy

        Returns:
            MergeResult with merged_keys, conflicts, and per-key status
        """
        ...


@dataclass
class MergeResult:
    """Result of parallel merge operation."""

    merged_keys: List[str]
    """Keys successfully merged."""

    conflicts: List[str]
    """Conflict descriptions (for FAIL_ON_CONFLICT)."""

    has_conflicts: bool
    """True if any unresolved conflicts."""
```

---

## Part 3: Recommendations

### Critical Changes to Spec

1. **Add default timeouts everywhere**
   - Tree-level: `max_tick_duration_ms = 60000` (required)
   - LLM node: `timeout = 120` (default)
   - All async operations: explicit timeout

2. **Define "progress" for stuck detection**
   - Blackboard write = progress
   - Async completion = progress
   - Add explicit `ctx.mark_progress()` for custom cases

3. **Make FAIL_ON_CONFLICT semantics explicit**
   - Conflict = Parallel returns FAILURE
   - Write conflicts to `_parallel_conflicts` key
   - Document when to use vs other strategies

4. **Remove or deprecate `immediate` reload policy**
   - Too dangerous for production
   - Keep only `let-finish-then-swap` and `cancel-and-restart`

5. **Add blackboard size limits**
   - Default 100MB total
   - Per-key limits configurable

### Implementation Checklist Additions

Add these tests:

- [ ] **UB-01**: Test that RUNNING node with no timeout eventually fails via tree watchdog
- [ ] **UB-02**: Test load failure when subtree_ref points to non-existent tree
- [ ] **UB-03**: Test Lua runtime error produces traceback in blackboard
- [ ] **UB-04**: Test blackboard validation error doesn't corrupt state
- [ ] **UB-05**: Test FAIL_ON_CONFLICT returns FAILURE on conflict
- [ ] **UB-06**: Test tick budget exceeded produces YIELDED status
- [ ] **SEC-01**: Test metatable escape is blocked
- [ ] **SEC-03**: Test blackboard size limit is enforced
- [ ] **SEC-04**: Test path traversal in subtree_ref is blocked
- [ ] **RC-01**: Test parallel children cannot see each other's writes mid-tick
- [ ] **RC-02**: Test events emitted during tick are buffered
- [ ] **EP-02**: Test partial failure tracking in parallel

### Quick Wins

1. **Document type coercion rules** (TS-01) - Just documentation, no code
2. **Reserve `_` prefix** (SEC-05) - Simple check in `bb.set()`
3. **Add name validation** (SEC-04) - Regex check: `^[a-zA-Z][a-zA-Z0-9_-]*$`
4. **Log slow scripts** (SEC-02) - Add metric in existing timeout handler

---

## Appendix: Summary Table

| ID | Category | Severity | Resolution Status |
|----|----------|----------|-------------------|
| UB-01 | Undefined | CRITICAL | Needs tree watchdog |
| UB-02 | Undefined | HIGH | Add validation |
| UB-03 | Undefined | HIGH | Add traceback |
| UB-04 | Undefined | MEDIUM | Document behavior |
| UB-05 | Undefined | HIGH | Clarify semantics |
| UB-06 | Undefined | MEDIUM | Add YIELDED status |
| UB-07 | Undefined | MEDIUM | Add default timeout |
| UB-08 | Undefined | MEDIUM | Deprecate immediate |
| UB-09 | Undefined | MEDIUM | Document ordering |
| UB-10 | Undefined | LOW | Document empty=SUCCESS |
| SEC-01 | Security | HIGH | Add test, audit |
| SEC-02 | Security | MEDIUM | Rate limit |
| SEC-03 | Security | MEDIUM | Add size limit |
| SEC-04 | Security | MEDIUM | Validate names |
| SEC-05 | Security | LOW | Reserve prefix |
| RC-01 | Race | MEDIUM | Use child scopes |
| RC-02 | Race | MEDIUM | Document buffering |
| RC-03 | Race | LOW | Covered by UB-08 |
| RC-04 | Race | LOW | Use request_id |
| TS-01 | Type | MEDIUM | Document coercion |
| TS-02 | Type | MEDIUM | Trust & document |
| TS-03 | Type | MEDIUM | Add tool schemas |
| EP-01 | Error | HIGH | Document rules |
| EP-02 | Error | HIGH | Track partial fails |
| EP-03 | Error | MEDIUM | Add failure trace |
