# Research: Oracle Plugin System

**Date**: 2026-01-04
**Branch**: `015-oracle-plugin-system`
**Spec**: [spec.md](./spec.md)

## Executive Summary

This document consolidates research findings for implementing the Oracle Plugin System. Key decisions:

1. **Performance Architecture**: Hybrid Python + Lua (via lupa) for MVP; Rust + PyO3 as future optimization path
2. **Expression Language**: `simpleeval` for TOML conditions; escape to Lua for complex logic
3. **Hook Points**: Integrate with existing ANS event bus architecture
4. **Sandboxing**: Environment whitelisting for Lua + instruction counting (Lua 5.2+)

---

## 1. Performance Architecture Decision

### Decision: Python + Lua (lupa) for MVP

**Rationale**: LuaJIT provides 20-30x speedup over CPython with minimal integration overhead via lupa. This covers the 80% TOML / 20% Lua split in the spec.

**Alternatives Considered**:

| Approach | Per-Op Latency | Throughput | Setup Time | Decision |
|----------|---------------|------------|------------|----------|
| Pure Python (CPython) | 1-10 ms | 100-1000 rules/sec | None | **Baseline** |
| Lupa (LuaJIT) | 0.05-0.5 ms | 2k-20k rules/sec | 1-2 days | **MVP Choice** |
| Rust (PyO3) | 0.01-0.1 ms | 10k-100k rules/sec | 1-2 weeks | **Future Path** |
| PyPy3 | 0.5-2 ms | 500-2000 rules/sec | Deploy change | Not viable (FastAPI) |

**Why Lua over Rust for MVP**:
- Faster development cycle (Lua is simpler than Rust)
- Hot-reloadable scripts (no compilation)
- Already in spec as scripting escape hatch
- Sufficient performance for <1000 rules/sec target
- Binary overhead: ~800KB (acceptable)

**Rust Migration Path**:
When performance requirements exceed 10k rules/sec, migrate rule engine core to Rust:
1. Create Rust crate with `maturin init --bindings pyo3`
2. Implement Rete-style pattern matching in Rust
3. Expose as Python module via `#[pyfunction]` decorators
4. Python continues orchestration; Rust handles hot path

### Implementation Pattern

```
Oracle Agent (Python)
    ↓ emit(Event)
EventBus (Python - existing ANS)
    ↓ notify handlers
RuleEngine (Python orchestration)
    ├── TOML Rules → simpleeval (Python)
    └── Lua Scripts → lupa (LuaJIT in-process)
        ↓
Actions (Python)
    ├── notify_self → ANS
    ├── log → logging
    ├── set_state → SQLite
    └── emit_event → EventBus
```

---

## 2. Lua Embedding via lupa

### Decision: Use lupa with LuaJIT

**Key Capabilities**:
- In-process execution (no subprocess overhead)
- ~800KB binary overhead
- GIL release for true threading
- Direct Python ↔ Lua object passing

### Sandboxing Strategy

**Environment Whitelisting** (primary protection):
```python
restricted_env = lua.eval('''
{
    print = print,
    type = type,
    tostring = tostring,
    tonumber = tonumber,
    table = { insert = table.insert, concat = table.concat },
    string = { sub = string.sub, len = string.len },
    math = math,
    -- NO: os, io, debug, dofile, loadfile, require
}
''')
```

**Unsafe Functions to Remove**:
- `debug.*` — Can break sandbox
- `dofile`, `load`, `loadstring` — Arbitrary code
- `os.execute`, `os.getenv` — System access
- `io.*` — File I/O
- `require`, `module` — Can load unsafe code

### Timeout Enforcement

**Approach**: Python threading with timeout (cross-platform):
```python
def execute_lua_with_timeout(code, timeout_sec=5):
    result_queue = Queue()

    def run():
        lua = LuaRuntime(max_memory=50*1024*1024)
        result_queue.put(lua.eval(code))

    thread = Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        raise TimeoutError("Script exceeded 5 second limit")
    return result_queue.get()
```

**Note**: LuaJIT does NOT support instruction counting. For CPU-bound infinite loop protection, use signal.SIGALRM (Unix) or threading timeout.

### State Sharing

Python objects exposed to Lua via `lua.globals()`:
```python
lua.globals().context = RuleContext(
    turn=TurnState(number=5, token_usage=0.85),
    history=HistoryState(tools=[...], failures=[...]),
    user=UserState(id="user_123", settings={...}),
    state=PluginState(get=getter, set=setter),
)
```

---

## 3. Expression Language for TOML Rules

### Decision: simpleeval for TOML conditions

**Current ANS Approach**: Regex-based function parsing (`func(arg)`)
**Recommended Upgrade**: `simpleeval` for boolean composition

**Grammar Supported**:
```
expression = or_expr
or_expr = and_expr ("or" and_expr)*
and_expr = comparison_expr ("and" comparison_expr)*
comparison_expr = call_expr (comp_op call_expr)?
comp_op = ">" | "<" | ">=" | "<=" | "==" | "!="
call_expr = function_call | field_access | literal
function_call = identifier "(" [argument_list] ")"
field_access = "context." identifier ("." identifier)*
```

**Example Conditions**:
```toml
# Simple threshold
condition = "context.turn.token_usage > 0.8"

# Boolean composition
condition = "context.turn.token_usage > 0.8 and context.history.tool_count > 5"

# Function-based
condition = "context_above_threshold(0.8) or tool_completed('vault_search')"
```

### Implementation

```python
from simpleeval import EvalWithCompoundTypes

def evaluate_condition(expr: str, ctx: RuleContext) -> bool:
    evaluator = EvalWithCompoundTypes(
        names={
            'context': dataclasses.asdict(ctx),
            # Flatten common fields for convenience
            'turn_number': ctx.turn.number,
            'token_usage': ctx.turn.token_usage,
        },
        functions={
            'context_above_threshold': lambda t: ctx.turn.token_usage >= t,
            'tool_completed': lambda name: ctx.event.tool_name == name if ctx.event else False,
            'any': any,
            'all': all,
            'len': len,
        }
    )
    return evaluator.eval(expr)
```

### Performance

| Evaluator | Speed | Notes |
|-----------|-------|-------|
| Manual regex (current) | <0.1ms | Limited to `func(arg)` |
| simpleeval | 0.1-1ms | Good for boolean composition |
| Lua (via lupa) | 0.05-0.5ms | For complex scripts |

---

## 4. Hook Point Integration

### Decision: Extend existing ANS EventBus

The ANS already has hook point infrastructure. Plugin rules attach as subscribers:

| Hook Point | ANS Event Type | Integration Location |
|------------|---------------|---------------------|
| `on_query_start` | `EventType.QUERY_START` (new) | oracle_agent.py:759 |
| `on_turn_start` | `EventType.AGENT_TURN_START` | oracle_agent.py:1051 |
| `on_turn_end` | `EventType.AGENT_TURN_END` | oracle_agent.py:1080 |
| `on_tool_call` | `EventType.TOOL_CALL_PENDING` | oracle_agent.py:1744 |
| `on_tool_complete` | `EventType.TOOL_CALL_SUCCESS` | oracle_agent.py:1859 |
| `on_tool_failure` | `EventType.TOOL_CALL_FAILURE` | oracle_agent.py:1920 |
| `on_session_end` | `EventType.SESSION_END` (new) | oracle_agent.py:1131 |

### Pattern to Follow

```python
# In oracle_agent.py, add new event emissions:
self._event_bus.emit(Event(
    type=EventType.QUERY_START,
    source="oracle_agent",
    severity=Severity.INFO,
    payload={
        "question": question,
        "user_id": user_id,
        "project_id": project_id,
    }
))
```

Rules subscribe via ANS pattern:
```toml
# rules/token_warning.toml
[rule]
id = "token-budget-warning"
trigger = "on_turn_start"

[condition]
expression = "context.turn.token_usage > 0.8"

[action]
type = "notify_self"
message = "Token budget at {{ context.turn.token_usage | percent }}. Consider wrapping up."
priority = "high"
```

---

## 5. ANS Extension Points

### Current ANS Architecture

```
Event → EventBus → Subscribers → Accumulator → ToonFormatter → Notifications
```

**Files** (2,795 lines total):
- `bus.py` (239 lines): Pub/sub, queue management
- `event.py` (164 lines): Event types, severity
- `subscriber.py` (382 lines): TOML loading, schema validation
- `accumulator.py` (635 lines): Batching, dedup, injection points
- `toon_formatter.py` (209 lines): Jinja2 + TOON rendering
- `persistence.py` (585 lines): Cross-session storage
- `deferred.py` (523 lines): Deferred delivery triggers

### Plugin System Extension

The Plugin System layers on top of ANS:

1. **RuleLoader** (parallel to SubscriberLoader): Load TOML rule definitions
2. **RuleEngine**: Evaluate conditions and dispatch actions
3. **LuaSandbox**: Execute Lua scripts safely
4. **ActionDispatcher**: Execute rule actions (notify_self, log, set_state, emit_event)

**New Files**:
```
backend/src/services/plugins/
├── __init__.py
├── rule.py          # Rule dataclass and RuleConfig
├── loader.py        # RuleLoader (TOML discovery)
├── engine.py        # RuleEngine (condition eval, action dispatch)
├── lua_sandbox.py   # LuaSandbox (lupa integration)
├── actions.py       # ActionDispatcher
├── context.py       # RuleContext (API for rules)
├── rules/           # Built-in rule definitions
│   ├── token_budget.toml
│   ├── iteration_budget.toml
│   ├── large_result.toml
│   └── repeated_failure.toml
└── scripts/         # Lua script examples
    └── complex_research.lua
```

---

## 6. Stretch Goals Architecture

### BERT Semantic Conditions (Future)

For semantic pattern matching:
```toml
[condition]
semantic = { concept = "authentication", field = "tool.args.query", threshold = 0.7 }
```

**Implementation Path**:
1. Pre-compute concept embeddings (auth, security, performance, etc.)
2. Embed incoming text at runtime
3. Compare via cosine similarity
4. Cache embeddings per session

**Recommended Model**: `sentence-transformers/all-MiniLM-L6-v2` (22M params, fast)

### LISP Rule Language (Future)

S-expression alternative for power users:
```lisp
(and (> (ctx-token-usage) 0.8)
     (any (ctx-recent-tools 5)
          (lambda (t) (eq (tool-name t) "vault_search"))))
```

**Implementation**: Use `hy` (Python-hosted Lisp) or custom S-expression parser.

### Skills/Behaviors (Future)

Higher-level plugins with complex logic:
- **Deep Researcher**: Multi-step research with source synthesis
- **Agent Swarm**: Coordinate multiple sub-agents

These require the rule engine to support:
- State machines
- Multi-turn workflows
- Agent communication protocols

---

## 7. Competitor Research: Game Plugin Systems

### 7.1 BakkesMod (Rocket League)

**Architecture**: C++ plugin DLLs with optional Python bindings (not Lua as initially expected).

**Key Patterns**:

1. **Three-Layer Architecture**:
   ```
   Plugin Layer (User C++ DLLs) → SDK Core (GameWrapper, CVarManager) → Wrapper Objects → Game Engine
   ```

2. **Plugin Interface**:
   ```cpp
   class BakkesModPlugin {
       std::shared_ptr<CVarManagerWrapper> cvarManager;
       std::shared_ptr<GameWrapper> gameWrapper;
       virtual void onLoad() {};
       virtual void onUnload() {};
   };
   ```

3. **Hook System** - Function hooking for game events:
   ```cpp
   gameWrapper->HookEvent("TAGame.Car_TA.SetVehicleInput", [this](std::string fname) {
       // Executes when RL calls this function
   });
   ```

4. **CVars Pattern** - User-configurable variables with automatic cleanup:
   ```cpp
   cvarManager->registerCvar("cl_my_setting", "default", "Description", true, true, 0, true, 100);
   ```

**Documentation Excellence**:
- Progressive tutorials (simple → complex)
- Code examples before abstract concepts
- Direct links to SDK header files
- Emphasis on null checks and thread safety
- Real plugin examples on GitHub

**Lessons for Oracle**:
- Type-safe wrapper objects (CarWrapper, BallWrapper) for game state access
- Automatic cleanup on plugin unload
- Configuration variables exposed via API

---

### 7.2 Honorbuddy & WoW Bot Frameworks

**Architecture**: TreeSharp behavior trees with C# plugins.

**Key Patterns**:

1. **Behavior Tree Composites**:
   - `PrioritySelector`: First successful child wins (decision trees)
   - `Sequence`: Execute until failure (task chains)
   - `Decorator`: Modify child behavior (guards, timeouts)

2. **Priority-Based Selection**:
   ```csharp
   new PrioritySelector(
       new Decorator(ret => target.HealthPercent < 25, Cast("Execute")),
       new Decorator(ret => Me.HealthPercent < 50, Cast("Heal Self")),
       new Decorator(ret => true, Cast("Fireball"))  // Default
   )
   ```

3. **Context Detection** (Singular innovation):
   ```
   Context Detection Loop:
     In Instance? → Use Instance rotation
     In Battleground? → Use BG rotation
     Else → Use Solo rotation
   ```

4. **Frame Locking Optimization**:
   - Cache running node to avoid O(n) tree traversal
   - Only re-evaluate on state change

5. **RunStatus Pattern**:
   - `Success`: Task done, continue
   - `Failure`: Try next option
   - `Running`: Resume next tick (coroutine-like)

**Lessons for Oracle**:
- Priority-based rule selection reduces conditional nesting
- Context awareness enables reuse (same rules, different behaviors)
- Frame locking prevents exponential traversal
- Coroutine support for multi-turn operations

---

### 7.3 Lua Plugin Systems (Games & Servers)

#### World of Warcraft Addon API

**Event Registration**:
```lua
local frame = CreateFrame("FRAME", "MyAddonFrame")
frame:RegisterEvent("PLAYER_ENTERING_WORLD")
frame:SetScript("OnEvent", function(self, event, ...)
    events[event](self, ...)  -- Table-dispatch pattern
end)
```

**Persistence**: SavedVariables declared in `.toc` file, auto-saved on logout.

#### Garry's Mod Hook System

**Simple String-Based Registration**:
```lua
hook.Add("Think", "FramerateDisplay", function()
    print("Think called")
end)
```

**Return Value Semantics**: First non-nil return stops propagation.

#### Roblox (Luau)

**Signal/Connection Pattern**:
```lua
local connection = part.Touched:Connect(function(otherPart)
    otherPart.Transparency = 1
end)
connection:Disconnect()  -- Cleanup
```

**Async Suffix Convention**: Methods that yield marked with `Async` suffix.

#### Redis Lua Scripting

**Atomic Execution Model**:
```lua
-- Rate limiting (atomic transaction)
local current = redis.call("get", KEYS[1])
if tonumber(current) >= limit then
    return 0  -- Rate limit exceeded
else
    redis.call("incr", key)
    return 1  -- Allowed
end
```

**Sandboxing**: No globals, no require(), limited stdlib, 5s timeout.

#### OpenResty (nginx + Lua)

**Per-Request Isolation**:
```lua
ngx.ctx.user_id = 123  -- Request-scoped context
local res = ngx.location.capture("/auth", { ctx = ngx.ctx })  -- Share context
```

**Phase-Based Execution**: init, rewrite, access, content, log phases.

---

### 7.4 Comparative Analysis

| System | Registration | Execution | Context | Sandboxing |
|--------|-------------|-----------|---------|------------|
| BakkesMod | DLL + macro | Hooks | GameWrapper | Trust model |
| Honorbuddy | C# classes | Behavior tree | Styx namespace | AppDomain |
| WoW Addons | Frame objects | Events | Frame tables | Secure funcs |
| GMod | String IDs | Hook chain | Globals | Partial |
| Roblox | Signal/Connect | Callbacks | Object-based | VM-level |
| Redis | EVAL | Atomic | KEYS/ARGV | Strict |
| OpenResty | Directive | Non-blocking | ngx.ctx | Strict |

---

### 7.5 Patterns to Adopt

1. **Priority-Based Selection** (Honorbuddy)
   - Rules ordered by priority, first match wins
   - Reduces nested conditionals

2. **Context Detection** (Singular)
   - Detect environment and switch behavior schemas
   - Same rules work in different contexts

3. **String-Based Hook IDs** (GMod)
   - Simpler than object-based registration
   - Easy to add/remove

4. **Connection Objects** (Roblox)
   - Enable automatic cleanup
   - Clear ownership semantics

5. **Request-Scoped Context** (OpenResty)
   - Isolated per-request state
   - Explicit sharing when needed

6. **Atomic Execution** (Redis)
   - Guarantees consistency
   - No partial rule effects

---

## 8. Competitor Research: Documentation Best Practices

### 8.1 Top Documentation Sites Analyzed

| Platform | Strength |
|----------|----------|
| **VS Code Extension API** | Progressive disclosure, UX guidelines |
| **HashiCorp Terraform** | API reference organization, versioning |
| **Grafana** | Consistent endpoint documentation |
| **Drools** | Concept progression (foundational → advanced) |
| **Easy Rules** | Archetype generator, runnable tutorials |
| **Obsidian** | Sample plugin repos, community forum |

### 8.2 Recommended Documentation Structure

**8-Section Progression**:
```
docs/plugin-api/
├── 00-getting-started/     # Entry point (5-10 min)
├── 01-core-concepts/       # Mental models
├── 02-architecture/        # System design
├── 03-guides/              # Detailed how-to
├── 04-api-reference/       # Exhaustive lookup
├── 05-examples/            # Working code
├── 06-testing-debugging/   # Developer workflows
├── 07-migration/           # Version upgrades
└── INDEX.md                # Navigation hub
```

### 8.3 Getting Started Pattern (VS Code)

**Part 1: Your First Plugin** (~10 min)
- Scaffolding command
- Complete runnable example
- Screenshot of result

**Part 2: Plugin Anatomy** (~15 min)
- Dissect generated code
- Highlight key concepts
- Cross-links to deeper sections

**Part 3: What's Next** (~5 min)
- Capability table with links
- Learning path by use case

### 8.4 API Reference Pattern

Each endpoint/hook includes:
1. **Signature** with types
2. **Parameters table** (type, required, description)
3. **Return value** with TypeScript type
4. **Example** with real values
5. **Error cases**
6. **See also** cross-links

### 8.5 Hook Documentation Format

```markdown
## Hook: beforeRuleEval

**Type:** `(context: EvaluationContext) => void | Promise<void>`

**When it fires:** Before rule engine evaluates each rule

**Event payload:**
```typescript
interface BeforeRuleEvalEvent {
  ruleId: string;
  ruleName: string;
  timestamp: number;
  context: Record<string, unknown>;
}
```

**Example:**
```typescript
oracle.hooks.beforeRuleEval((event) => {
  console.log(`Evaluating: ${event.ruleName}`);
});
```

**Error handling:** If hook throws, rule evaluation...
```

### 8.6 Code Examples Tiers

**Tier 1: Hello World** (30 seconds)
```typescript
const plugin = new OraclePlugin({ id: 'hello' });
plugin.registerRule({
  condition: () => true,
  action: () => console.log('Hello!'),
});
```

**Tier 2: Intermediate** (5 min) - Conditional rules with context

**Tier 3: Advanced** (15 min) - State management, hooks, multi-step

---

## 9. Documentation Structure

Created at `docs/plugin-api/`:

```
docs/plugin-api/
├── README.md                    # Overview, getting started
├── architecture/
│   ├── overview.md              # System architecture
│   ├── performance.md           # Performance considerations
│   └── roadmap.md               # Stretch goals, future work
├── rules/
│   ├── format.md                # TOML rule schema
│   ├── conditions.md            # Expression language
│   ├── actions.md               # Available actions
│   └── examples.md              # Common patterns
├── context-api/
│   ├── reference.md             # Full API reference
│   ├── turn.md                  # Turn state
│   ├── history.md               # History access
│   └── state.md                 # Plugin state
├── hooks/
│   ├── lifecycle.md             # Hook point reference
│   └── events.md                # Event types
├── scripting/
│   ├── lua-guide.md             # Lua scripting guide
│   ├── sandbox.md               # Security model
│   └── examples.md              # Lua examples
└── built-ins/
    ├── token-budget.md          # Built-in rule docs
    ├── iteration-budget.md
    ├── large-result.md
    └── repeated-failure.md
```

---

## 8. Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Performance tier 1 | Python + simpleeval | 80% TOML conditions |
| Performance tier 2 | lupa (LuaJIT) | 20% complex scripts |
| Performance tier 3 | Rust + PyO3 | Future: >10k rules/sec |
| Expression language | simpleeval | Safe, supports boolean ops |
| Scripting language | Lua (via lupa) | Fast, sandboxable, game industry proven |
| Hook integration | Extend ANS EventBus | Reuse existing infrastructure |
| Rule format | TOML | Matches ANS subscribers |
| Sandbox approach | Env whitelisting + timeout | Defense in depth |
| State storage | SQLite (existing) | Per-plugin key-value |

---

## Sources

- [lupa GitHub](https://github.com/scoder/lupa)
- [simpleeval GitHub](https://github.com/danthedeckie/simpleeval)
- [PyO3 User Guide](https://pyo3.rs/)
- [Lua Sandboxes Wiki](http://lua-users.org/wiki/SandBoxes)
- [Google CEL](https://cel.dev/)
- [LuaJIT Benchmarks](https://staff.fnwi.uva.nl/h.vandermeer/docs/lua/luajit/luajit_performance.html)
