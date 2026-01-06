# Component Audit: Vlt-Bridge Deep Code Analysis

**Audit Date:** 2026-01-05
**Auditor:** Claude Code (Opus 4.5)
**Scope:** All major backend components that could form the backbone of a unified agent system

---

## 1. VLT CLI - Daemon Architecture

### File Locations and Sizes

| File | Lines | Purpose |
|------|-------|---------|
| `packages/vlt-cli/src/vlt/daemon/server.py` | 981 | ZMQ-based daemon server |
| `packages/vlt-cli/src/vlt/daemon/manager.py` | 388 | Daemon lifecycle management |
| `packages/vlt-cli/src/vlt/daemon/client.py` | 299 | Client for daemon communication |
| `packages/vlt-cli/src/vlt/core/service.py` | ~400 | Core service registry |

### Core Classes and Signatures

```python
# server.py - VltDaemonServer
class VltDaemonServer:
    """ZeroMQ-based daemon for background task execution."""

    def __init__(
        self,
        socket_path: Optional[str] = None,
        max_concurrent_jobs: int = 3,
    ) -> None

    async def start(self) -> None
    async def stop(self) -> None
    async def handle_message(self, message: bytes) -> bytes

    # Job management
    async def submit_job(self, job_type: str, params: Dict[str, Any]) -> str
    async def get_job_status(self, job_id: str) -> JobStatus
    async def cancel_job(self, job_id: str) -> bool
```

```python
# manager.py - DaemonManager
class DaemonManager:
    """Manages daemon lifecycle (start/stop/status)."""

    def start(self, foreground: bool = False) -> bool
    def stop(self) -> bool
    def status(self) -> DaemonStatus
    def is_running(self) -> bool
    def get_socket_path(self) -> Path
```

```python
# client.py - DaemonClient
class DaemonClient:
    """Client for communicating with the daemon over ZMQ."""

    async def connect(self) -> bool
    async def submit_job(self, job_type: str, params: Dict) -> JobResponse
    async def get_status(self, job_id: str) -> JobStatus
    async def cancel(self, job_id: str) -> bool
    async def ping(self) -> bool
```

### What It Does Well (Reusable)

1. **ZeroMQ-based IPC**: Clean async messaging pattern
2. **Job queue system**: Supports job submission, status tracking, cancellation
3. **Lifecycle management**: Proper daemon start/stop with PID file
4. **Concurrent job limits**: Configurable max_concurrent_jobs
5. **Socket-based communication**: Avoids HTTP overhead for local ops

### What's Missing/Broken

1. **No agent process management**: Only handles indexing jobs
2. **No multi-agent coordination**: Single-job-type focus
3. **No event streaming**: Request/response only, no pub/sub
4. **No distributed mode**: Single-machine only

### Code Pattern Example

```python
# From server.py - Job handling pattern
async def handle_message(self, message: bytes) -> bytes:
    """Route incoming messages to handlers."""
    try:
        request = json.loads(message.decode())
        msg_type = request.get("type")

        handlers = {
            "submit": self._handle_submit,
            "status": self._handle_status,
            "cancel": self._handle_cancel,
            "ping": self._handle_ping,
        }

        handler = handlers.get(msg_type, self._handle_unknown)
        result = await handler(request)
        return json.dumps(result).encode()
    except Exception as e:
        return json.dumps({"error": str(e)}).encode()
```

---

## 2. Agent Notification System (ANS)

### File Locations and Sizes

| File | Lines | Purpose |
|------|-------|---------|
| `backend/src/services/ans/bus.py` | ~200 | EventBus pub/sub |
| `backend/src/services/ans/event.py` | ~150 | Event model and types |
| `backend/src/services/ans/subscriber.py` | ~300 | Subscriber loading/matching |
| `backend/src/services/ans/accumulator.py` | ~250 | Event batching/dedup |
| `backend/src/services/ans/toon_formatter.py` | ~200 | Jinja2 TOON templates |
| `backend/src/services/ans/persistence.py` | ~200 | Cross-session storage |
| `backend/src/services/ans/deferred.py` | ~100 | Deferred delivery queue |
| **Subscribers (TOML)** | 9 files | Event configuration |
| **Templates (Jinja2)** | 9 files | Notification formatting |

**Total ANS lines:** ~1,600 Python + configs

### Core Classes and Signatures

```python
# bus.py - EventBus
class EventBus:
    """Singleton pub/sub event bus with wildcard subscription."""

    def __init__(self, max_queue_size: int = 1000) -> None

    def subscribe(
        self,
        event_pattern: str,  # e.g., "tool.*", "budget.token.*"
        handler: Callable[[Event], None],
    ) -> str  # Returns subscription_id

    def unsubscribe(self, subscription_id: str) -> bool
    def emit(self, event: Event) -> None
    def emit_async(self, event: Event) -> None  # Non-blocking

    # Pattern matching
    def _matches_pattern(self, event_type: str, pattern: str) -> bool
```

```python
# event.py - Event Model
@dataclass
class Event:
    type: EventType  # Enum: TOOL_CALL_FAILURE, BUDGET_WARNING, etc.
    source: str
    severity: Severity  # DEBUG, INFO, WARNING, ERROR
    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

class EventType(str, Enum):
    TOOL_CALL_FAILURE = "tool.call.failure"
    TOOL_CALL_TIMEOUT = "tool.call.timeout"
    BUDGET_TOKEN_WARNING = "budget.token.warning"
    BUDGET_ITERATION_WARNING = "budget.iteration.warning"
    BUDGET_TOKEN_EXCEEDED = "budget.token.exceeded"
    BUDGET_ITERATION_EXCEEDED = "budget.iteration.exceeded"
    AGENT_LOOP_DETECTED = "agent.loop.detected"
    CONTEXT_APPROACHING_LIMIT = "context.approaching.limit"
    AGENT_SELF_NOTIFY = "agent.self.notify"
    AGENT_SELF_REMIND = "agent.self.remind"
    SOURCE_STALE = "source.stale"
    QUERY_START = "query.start"
    # ... more types
```

```python
# accumulator.py - NotificationAccumulator
class NotificationAccumulator:
    """Batches and deduplicates events before rendering."""

    def __init__(self) -> None

    def register_subscribers(self, subscribers: List[Subscriber]) -> None
    def accumulate(self, event: Event) -> None
    def flush(self, inject_at: InjectionPoint) -> List[str]
    def flush_all(self) -> List[str]

    # Deduplication
    def _get_dedupe_key(self, event: Event, subscriber: Subscriber) -> str
```

### Subscriber TOML Pattern

```toml
# tool_failure.toml
[subscriber]
id = "tool_failure"
name = "Tool Failure Notifications"
description = "Notifies agent when tool calls fail or timeout"
version = "1.0.0"

[events]
types = ["tool.call.failure", "tool.call.timeout"]
severity_filter = "warning"

[batching]
window_ms = 2000
max_size = 10
dedupe_key = "type:payload.tool_name"
dedupe_window_ms = 5000

[output]
priority = "high"
inject_at = "after_tool"
template = "tool_failure.toon.j2"
core = true  # Cannot be disabled by user
```

### Template Pattern (TOON Jinja2)

```jinja2
{# tool_failure.toon.j2 #}
{% if events|length == 1 %}
{% set e = events[0] %}
tool_fail: {{ e.payload.tool_name }} {{ e.payload.error_type }}{% if e.payload.error_message %} - {{ e.payload.error_message }}{% endif %}

{% else %}
tool_fails[{{ events|length }}]{tool,error,message}:
{% for e in events %}
  {{ e.payload.tool_name }},{{ e.payload.error_type }},{{ e.payload.error_message | default("") }}
{% endfor %}
{% endif %}
```

### What It Does Well (Reusable)

1. **Pub/sub with wildcards**: `tool.*` matches all tool events
2. **Event batching**: Groups similar events within time windows
3. **Deduplication**: Prevents notification spam
4. **Injection points**: `turn_start`, `after_tool`, `immediate`
5. **TOML-based configuration**: Easy to add new subscribers
6. **Jinja2 templates**: Flexible output formatting
7. **Severity filtering**: Only process relevant severity levels
8. **Core vs user-disableable**: Some notifications cannot be turned off

### What's Missing/Broken

1. **No distributed pub/sub**: In-process only
2. **No persistence of event history**: Only current session
3. **No replay capability**: Cannot re-process events
4. **Limited cross-agent communication**: Designed for single agent

### Integration Opportunity

The EventBus could be the backbone for agent communication:
- Extend to cross-process pub/sub via ZMQ (from daemon)
- Add event sourcing for replay
- Add distributed broker support (Redis pub/sub)

---

## 3. Plugin System (015-oracle-plugin-system)

### File Locations and Sizes

| File | Lines | Purpose |
|------|-------|---------|
| `backend/src/services/plugins/engine.py` | ~400 | RuleEngine core |
| `backend/src/services/plugins/rule.py` | ~250 | Rule model and types |
| `backend/src/services/plugins/context.py` | ~300 | RuleContext for evaluation |
| `backend/src/services/plugins/loader.py` | 244 | TOML rule file loading |
| `backend/src/services/plugins/expression.py` | ~200 | Condition evaluation |
| `backend/src/services/plugins/actions.py` | ~250 | Action dispatch |
| `backend/src/services/plugins/lua_sandbox.py` | ~300 | Lua script execution |
| `backend/src/services/plugins/state.py` | 294 | Plugin state management |

**Total Plugin lines:** ~2,238 Python

### Behavior Tree Subsystem

| File | Lines | Purpose |
|------|-------|---------|
| `behavior_tree/node.py` | 311 | Base BehaviorNode, Composite, Decorator, Leaf |
| `behavior_tree/composites.py` | 470 | Selector, Sequence, Parallel |
| `behavior_tree/decorators.py` | 779 | Inverter, Repeater, Guard, Cooldown, etc. |
| `behavior_tree/leaves.py` | 707 | Condition, Action, Wait, Script |
| `behavior_tree/tree.py` | 511 | BehaviorTree orchestration |
| `behavior_tree/builder.py` | 537 | Fluent tree builder API |
| `behavior_tree/types.py` | 287 | RunStatus, TickContext |

**Total Behavior Tree lines:** ~3,602 Python

### Core Classes and Signatures

```python
# engine.py - RuleEngine
class RuleEngine:
    """Executes rules in response to events."""

    def __init__(
        self,
        loader: RuleLoader,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        context_builder: Callable[[Event], RuleContext],
        auto_subscribe: bool = True,
    ) -> None

    def start(self) -> None  # Subscribe to events
    def stop(self) -> None
    def evaluate_rules(self, event: Event) -> List[RuleResult]

    @property
    def rule_count(self) -> int
```

```python
# rule.py - Rule Model
@dataclass
class Rule:
    id: str
    name: str
    description: str
    version: str
    trigger: HookPoint  # on_turn_start, on_tool_call, on_response, etc.
    condition: Optional[str]  # Expression like "turn.number > 5"
    script: Optional[str]  # Lua script path
    action: Optional[RuleAction]
    priority: int = 100
    enabled: bool = True
    core: bool = False
    plugin_id: Optional[str] = None
    source_path: Optional[str] = None

    def validate(self) -> List[str]  # Returns validation errors

class HookPoint(str, Enum):
    ON_TURN_START = "on_turn_start"
    ON_TOOL_CALL = "on_tool_call"
    ON_TOOL_RESULT = "on_tool_result"
    ON_RESPONSE = "on_response"
    ON_ERROR = "on_error"
    ON_EVENT = "on_event"

class ActionType(str, Enum):
    LOG = "log"
    NOTIFY = "notify"
    SET_STATE = "set_state"
    EMIT_EVENT = "emit_event"
```

```python
# context.py - RuleContext
@dataclass
class RuleContext:
    """Snapshot of agent state for rule evaluation."""
    turn: TurnState
    history: HistoryState
    user: UserState
    project: ProjectState
    state: PluginState
    event: EventData

@dataclass
class TurnState:
    number: int
    token_usage: float  # 0.0 to 1.0
    context_usage: float
    iteration_count: int

@dataclass
class HistoryState:
    messages: List[Dict[str, Any]]
    tools: List[ToolCallRecord]
    failures: Dict[str, int]  # tool_name -> failure_count
```

```python
# behavior_tree/node.py - Base Classes
class BehaviorNode(ABC):
    """Abstract base for all BT nodes."""

    def __init__(self, name: Optional[str] = None) -> None

    def tick(self, context: TickContext) -> RunStatus
    @abstractmethod
    def _tick(self, context: TickContext) -> RunStatus
    def reset(self) -> None
    def debug_info(self) -> dict

class Composite(BehaviorNode):
    """Manages multiple children (Selector, Sequence, Parallel)."""

    @property
    def children(self) -> list[BehaviorNode]
    def add_child(self, child: BehaviorNode) -> "Composite"

class Decorator(BehaviorNode):
    """Wraps single child with modified behavior."""

    @property
    def child(self) -> Optional[BehaviorNode]

class Leaf(BehaviorNode):
    """Terminal nodes (Condition, Action)."""
    pass

class RunStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"
```

### Lua Sandbox Pattern

```python
# lua_sandbox.py
class LuaSandbox:
    """Secure Lua execution environment."""

    def __init__(
        self,
        max_instructions: int = 10000,
        allowed_modules: List[str] = None,
    ) -> None

    def execute(
        self,
        script: str,
        context: Dict[str, Any],
        timeout: float = 5.0,
    ) -> Dict[str, Any]

    def register_function(self, name: str, func: Callable) -> None
```

### What It Does Well (Reusable)

1. **Full behavior tree implementation**: Composites, Decorators, Leaves
2. **Fluent builder API**: Easy tree construction
3. **TOML-based rule definitions**: Declarative configuration
4. **Expression evaluation**: Simple DSL for conditions
5. **Lua sandboxing**: Safe script execution
6. **Event-driven rule triggers**: Integrates with EventBus
7. **State management**: Plugin-scoped persistent state
8. **Context snapshots**: Full agent state for rule evaluation

### What's Missing/Broken

1. **Rules directory empty/missing**: `plugins/rules/` not populated
2. **Lua sandbox untested**: May have security gaps
3. **No visual debugger**: Hard to trace tree execution
4. **No hot-reload**: Must restart for rule changes
5. **Database persistence incomplete**: `plugin_state` table not used

### Integration Opportunity

The Behavior Tree could orchestrate multi-agent workflows:
- Root Selector chooses between agent types
- Parallel runs agents concurrently
- Guards check budget/context limits
- Actions delegate to subagents

---

## 4. Research Orchestrator (Behavior Pattern)

### File Locations and Sizes

| File | Lines | Purpose |
|------|-------|---------|
| `backend/src/services/research/behaviors.py` | 976 | Research behavior nodes |
| `backend/src/services/research/orchestrator.py` | 509 | Research workflow coordinator |
| `backend/src/services/research/llm_service.py` | ~300 | LLM API wrapper |
| `backend/src/services/research/vault_persister.py` | ~200 | Report saving |

**Total Research lines:** ~1,985 Python

### Core Classes and Signatures

```python
# behaviors.py - Research Behaviors
class ResearchBehavior(ABC):
    """Base class for research behavior nodes."""

    @abstractmethod
    async def run(self, state: ResearchState) -> ResearchState
    def get_progress_message(self) -> str

class GenerateBriefBehavior(ResearchBehavior):
    """Transform query into ResearchBrief."""

    def __init__(
        self,
        llm_service: ResearchLLMService,
        prompt_loader: Optional[PromptLoader] = None,
    )

    async def run(self, state: ResearchState) -> ResearchState

class ParallelResearchersBehavior(ResearchBehavior):
    """Run multiple researchers in parallel."""

    def __init__(
        self,
        llm_service: ResearchLLMService,
        tavily_service: Optional[TavilySearchService] = None,
        openrouter_search: Optional[OpenRouterSearchService] = None,
        search_provider: SearchProvider = "none",
        max_concurrent: int = 5,
    )

    async def run(self, state: ResearchState) -> ResearchState

class CompressFindingsBehavior(ResearchBehavior):
    """Synthesize findings from all researchers."""

class GenerateReportBehavior(ResearchBehavior):
    """Generate final research report."""

class PersistToVaultBehavior(ResearchBehavior):
    """Save research to vault."""
```

```python
# orchestrator.py - ResearchOrchestrator
class ResearchOrchestrator:
    """Orchestrates the deep research workflow."""

    def __init__(
        self,
        user_id: str,
        vault_path: Optional[str] = None,
        search_provider: SearchProvider = "none",
        tavily_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        config: Optional[ResearchConfig] = None,
    )

    async def run_research(
        self,
        request: ResearchRequest,
    ) -> ResearchState

    async def run_research_streaming(
        self,
        request: ResearchRequest,
    ) -> AsyncGenerator[ResearchProgress, None]
```

### Workflow Pattern

```
1. GenerateBriefBehavior     -> Create research plan
2. PlanSubtopicsBehavior     -> Assign subtopics to researchers
3. ParallelResearchersBehavior -> Run 5 concurrent researchers
4. CompressFindingsBehavior  -> Synthesize findings
5. GenerateReportBehavior    -> Create final report
6. PersistToVaultBehavior    -> Save to vault
```

### What It Does Well (Reusable)

1. **Sequential behavior composition**: Clean workflow stages
2. **Parallel execution**: Semaphore-limited concurrency
3. **Progress streaming**: Async generator for UI updates
4. **State machine pattern**: ResearchStatus transitions
5. **Configurable depth**: Quick/Standard/Thorough modes

### What's Missing/Broken

1. **Not using plugin behavior trees**: Separate implementation
2. **No cancellation support**: Cannot abort mid-research
3. **No partial results**: All or nothing

### Integration Opportunity

Could be rewritten using the Plugin system's behavior trees:
- `Sequence` of research stages
- `Parallel` for concurrent researchers
- `Guard` for API key checks
- `Cooldown` for rate limiting

---

## 5. Oracle Agent - Execution Loop

### File Location and Size

| File | Lines | Purpose |
|------|-------|---------|
| `backend/src/services/oracle_agent.py` | 2,765 | Main agent loop |

### Core Classes and Signatures

```python
# oracle_agent.py - OracleAgent
class OracleAgent:
    """AI project manager agent with tool calling."""

    OPENROUTER_BASE = "https://openrouter.ai/api/v1"
    MAX_TURNS = 30
    DEFAULT_MODEL = "anthropic/claude-sonnet-4"
    DEFAULT_SUBAGENT_MODEL = "deepseek/deepseek-chat"

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        subagent_model: Optional[str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context_service: Optional[OracleContextService] = None,
        tree_service: Optional[ContextTreeService] = None,
    )

    async def query(
        self,
        question: str,
        user_id: str,
        stream: bool = True,
        thinking: bool = False,
        max_tokens: int = 4000,
        project_id: Optional[str] = None,
        context_id: Optional[str] = None,
    ) -> AsyncGenerator[OracleStreamChunk, None]

    def cancel(self) -> None
    def is_cancelled(self) -> bool
```

### Integrated Systems

```python
# ANS Integration (Agent Notification System)
self._event_bus = get_event_bus()
self._accumulator = NotificationAccumulator()
self._toon_formatter = get_toon_formatter()
self._subscriber_loader = SubscriberLoader()

# Plugin System Integration (RuleEngine)
self._rule_engine: Optional[RuleEngine] = None
self._init_rule_engine()

# Context Tracking
self._context_service = context_service or get_context_service()
self._tree_service = tree_service or get_context_tree_service()
```

### Key Methods

```python
# Loop detection
def _detect_loop(self, tool_calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Detect repetitive tool call patterns."""
    # Tracks last 6 patterns, triggers at 3 repetitions

# Budget tracking
def _check_iteration_budget(self, turn: int) -> None:
    """Emit warning at 70% of MAX_TURNS."""

def _check_token_budget(self, tokens_used: int) -> None:
    """Emit warning at 80% of max_tokens."""

def _check_context_limit(self) -> None:
    """Emit warning at 70% of context window."""

# Rule context building
def _build_rule_context(self, event: Event) -> RuleContext:
    """Create snapshot of agent state for rule evaluation."""
```

### What It Does Well (Reusable)

1. **Streaming response**: AsyncGenerator for SSE
2. **Tool call parsing**: XML and function calling formats
3. **Budget tracking**: Iteration, token, context warnings
4. **Loop detection**: Prevents infinite tool loops
5. **ANS integration**: Full notification pipeline
6. **Plugin integration**: RuleEngine hooks
7. **Context trees**: Conversation history management
8. **Cancellation**: Task cancellation support

### What's Missing/Broken

1. **Monolithic design**: 2,765 lines in one file
2. **No agent-to-agent communication**: Single agent focus
3. **Hardcoded model knowledge**: Context sizes in dict
4. **No plugin isolation**: Runs in same process

---

## 6. Tool Executor

### File Location and Size

| File | Lines | Purpose |
|------|-------|---------|
| `backend/src/services/tool_executor.py` | 2,539 | Tool dispatch and execution |

### Core Classes and Signatures

```python
# tool_executor.py - ToolExecutor
class ToolExecutor:
    """Executes tool calls by routing to backend services."""

    DEFAULT_TIMEOUT: float = 30.0

    TOOL_TIMEOUTS: Dict[str, float] = {
        "web_search": 60.0,
        "web_fetch": 60.0,
        "search_code": 30.0,
        "delegate_librarian": 1200.0,  # 20 minutes
        "deep_research": 1800.0,  # 30 minutes
        # ...
    }

    def __init__(
        self,
        vault_service: Optional[VaultService] = None,
        indexer_service: Optional[IndexerService] = None,
        thread_service: Optional[ThreadService] = None,
        oracle_bridge: Optional[OracleBridge] = None,
        db_service: Optional[DatabaseService] = None,
        user_settings_service: Optional[UserSettingsService] = None,
        librarian_service: Optional[LibrarianService] = None,
        github_service: Optional[GitHubService] = None,
        default_timeout: Optional[float] = None,
    )

    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        user_id: str,
        timeout: Optional[float] = None,
    ) -> str  # JSON result

    async def execute_batch(
        self,
        tool_calls: List[Dict[str, Any]],
        user_id: str,
        timeout: Optional[float] = None,
        include_call_ids: bool = False,
    ) -> List[Any]
```

### Tool Registry

```python
self._tools: Dict[str, Any] = {
    # Code tools
    "search_code": self._search_code,
    "find_definition": self._find_definition,
    "find_references": self._find_references,
    "get_repo_map": self._get_repo_map,
    # Vault tools
    "vault_read": self._vault_read,
    "vault_write": self._vault_write,
    "vault_search": self._vault_search,
    "vault_list": self._vault_list,
    "vault_move": self._vault_move,
    "vault_create_index": self._vault_create_index,
    # Thread tools
    "thread_push": self._thread_push,
    "thread_read": self._thread_read,
    "thread_seek": self._thread_seek,
    "thread_list": self._thread_list,
    # Web tools
    "web_search": self._web_search,
    "web_fetch": self._web_fetch,
    # Meta tools
    "delegate_librarian": self._delegate_librarian,
    "notify_self": self._notify_self,
    # GitHub tools
    "github_read": self._github_read,
    "github_search": self._github_search,
    # Research tools
    "deep_research": self._deep_research,
}
```

### ANS Integration

```python
# On timeout
get_event_bus().emit(Event(
    type=EventType.TOOL_CALL_TIMEOUT,
    source="tool_executor",
    severity=Severity.WARNING,
    payload={
        "tool_name": name,
        "error_type": "timeout",
        "error_message": f"Timed out after {actual_timeout}s",
        "timeout_seconds": actual_timeout,
    }
))

# Staleness detection
def _check_file_staleness(self, user_id, path, project_id, current_content):
    """Emit SOURCE_STALE event if file changed since last read."""
```

### What It Does Well (Reusable)

1. **Centralized dispatch**: Single entry point for all tools
2. **Timeout management**: Per-tool and default timeouts
3. **Batch execution**: Parallel tool calls
4. **Error categorization**: Structured error types
5. **ANS integration**: Event emission on failures
6. **Staleness tracking**: Detects file changes

### What's Missing/Broken

1. **No tool versioning**: Cannot evolve tool schemas
2. **No sandboxing**: Tools run in same process
3. **Limited parallelism**: Sequential within batch
4. **No retry logic**: Single-shot execution

---

## 7. MCP Server

### File Location and Size

| File | Lines | Purpose |
|------|-------|---------|
| `backend/src/mcp/server.py` | 735 | FastMCP tool server |

### Core Pattern

```python
from fastmcp import FastMCP

mcp = FastMCP(
    "obsidian-docs-viewer",
    instructions="Multi-tenant vault tools...",
)

@mcp.resource("ui://widget/note.html", mime_type="text/html+skybridge")
def widget_resource() -> str:
    """Return the widget HTML bundle."""

@mcp.tool(name="list_notes", description="...")
def list_notes(folder: Optional[str] = None) -> List[Dict]:
    user_id = _current_user_id()
    return vault_service.list_notes(user_id, folder)

@mcp.tool(name="search_code", description="...")
async def search_code(query: str, limit: int = 10) -> Dict:
    result = await oracle_bridge.search_code(query, limit)
    return result
```

### Tools Exposed

| Tool | Description |
|------|-------------|
| `list_notes` | List vault notes |
| `read_note` | Read note content |
| `write_note` | Create/update note |
| `delete_note` | Delete note |
| `search_notes` | Full-text search |
| `get_backlinks` | Note references |
| `get_tags` | Tag list |
| `ask_oracle` | Query codebase |
| `search_code` | Hybrid code search |
| `find_definition` | Symbol definition |
| `find_references` | Symbol usages |
| `get_repo_map` | Repo structure |

### What It Does Well (Reusable)

1. **FastMCP integration**: Clean decorator API
2. **Widget resource**: iFrame for ChatGPT
3. **Auth handling**: JWT validation
4. **STDIO/HTTP modes**: Local and remote

### What's Missing/Broken

1. **Subset of ToolExecutor**: Not all tools exposed
2. **No streaming**: Request/response only
3. **No agent tools**: Only document tools

---

## 8. Dependency Graph

```
                    ┌──────────────────┐
                    │   Oracle Agent   │
                    │   (2,765 lines)  │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  Tool Executor │  │   ANS EventBus │  │  Plugin Engine │
│  (2,539 lines) │  │   (~1,600)     │  │  (~2,238)      │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                   │
        ▼                   ▼                   │
┌───────────────────────────────────────┐      │
│           Backend Services            │      │
│  - VaultService                       │      │
│  - IndexerService                     │      │
│  - ThreadService                      │      │
│  - GitHubService                      │      │
│  - Research Orchestrator (~1,985)     │◄─────┘
│  - LibrarianService                   │
│  - OracleBridge                       │
└───────────────────┬───────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   MCP Server    │
          │   (735 lines)   │
          └─────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  VLT CLI Daemon │
          │   (1,668 lines) │
          └─────────────────┘
```

---

## 9. Reuse Analysis

### Can Be Reused Directly

| Component | Lines | Readiness | Notes |
|-----------|-------|-----------|-------|
| ANS EventBus | ~200 | **High** | Clean pub/sub, extend to ZMQ |
| ANS Subscribers | 9 TOML | **High** | Declarative configs |
| Behavior Tree (plugin) | ~3,600 | **High** | Full implementation |
| TOML Rule Loader | 244 | **High** | Clean parser |
| Tool Executor dispatch | ~500 | **Medium** | Extract core pattern |
| Research Behaviors | ~976 | **Medium** | Refactor to use plugin BT |

### Needs Rewriting

| Component | Lines | Reason |
|-----------|-------|--------|
| Oracle Agent loop | 2,765 | Monolithic, extract agent interface |
| VLT Daemon | 1,668 | Add multi-agent support |
| Lua Sandbox | ~300 | Security audit needed |

### Integration Opportunities

1. **EventBus + ZMQ Daemon**
   - Extend EventBus to emit over ZMQ
   - Daemon becomes event broker for multiple agents
   - Cross-process pub/sub for agent communication

2. **Behavior Trees for Agent Orchestration**
   ```
   Root: Selector
   ├── Sequence (Oracle Flow)
   │   ├── Guard: HasContext?
   │   ├── Action: LoadContext
   │   └── Action: RunOracleLoop
   └── Sequence (Librarian Flow)
       ├── Guard: NeedsSummarization?
       └── Action: RunLibrarianLoop
   ```

3. **Plugin Rules for Agent Behavior**
   ```toml
   [rule]
   id = "delegate_on_large_results"
   trigger = "on_tool_result"

   [condition]
   expression = "history.tools[-1].name == 'vault_search' and len(history.tools[-1].result.results) > 6"

   [action]
   type = "emit_event"
   event_type = "agent.delegate.librarian"
   payload = { task = "summarize_results" }
   ```

4. **Research Orchestrator on Plugin BT**
   - Replace custom behaviors with BT nodes
   - Use `Parallel` composite for concurrent researchers
   - Use `Cooldown` decorator for rate limiting
   - Use `Guard` for API key checks

---

## 10. Summary

### Total Lines Audited

| Component | Lines |
|-----------|-------|
| VLT CLI Daemon | 1,668 |
| ANS (Python + configs) | ~1,600 |
| Plugin System | ~2,238 |
| Plugin Behavior Tree | ~3,602 |
| Research System | ~1,985 |
| Oracle Agent | 2,765 |
| Tool Executor | 2,539 |
| MCP Server | 735 |
| **Total** | **~17,132** |

### Key Findings

1. **Strong foundations exist**: ANS EventBus, Behavior Trees, TOML configs
2. **Duplication**: Research has its own behavior pattern, not using plugin BT
3. **Monolithic agents**: Oracle Agent is too large, needs interface extraction
4. **Missing glue**: No cross-process communication for multi-agent
5. **Plugin system underused**: Rules directory empty, Lua untested

### Recommended Architecture Direction

1. **Extract Agent Interface** from OracleAgent
2. **Unify on Plugin Behavior Trees** for all agent orchestration
3. **Extend EventBus** with ZMQ for distributed pub/sub
4. **Use Daemon** as agent process manager
5. **Keep TOML configs** for rules, subscribers, agent definitions
6. **Add Redis/SQLite** for distributed state

---

**Confidence: 8/10** - I have read the actual code files and understand the implementation patterns. Some files were not fully read (daemon server internals), and the interaction between components in production may have nuances not captured in static analysis.
