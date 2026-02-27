# Data Model: Behavior Tree Universal Runtime

## Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   TreeRegistry  │──────▶│  BehaviorTree   │──────▶│  BehaviorNode   │
│                 │  1:N  │                 │  1:N  │                 │
└─────────────────┘       └────────┬────────┘       └────────┬────────┘
                                   │                         │
                                   │ 1:1                     │ type
                                   ▼                         ▼
                          ┌─────────────────┐       ┌─────────────────┐
                          │   Blackboard    │       │ NodeType Enum   │
                          │   (tree scope)  │       │ COMPOSITE       │
                          └────────┬────────┘       │ DECORATOR       │
                                   │                │ LEAF            │
                                   │ parent         └─────────────────┘
                                   ▼
                          ┌─────────────────┐
                          │   Blackboard    │
                          │ (global scope)  │
                          └─────────────────┘

┌─────────────────┐       ┌─────────────────┐
│  TickContext    │──────▶│  Blackboard     │
│                 │  ref  │  (current scope)│
└────────┬────────┘       └─────────────────┘
         │
         │ contains
         ▼
┌─────────────────┐
│   Services      │
│ (DI container)  │
└─────────────────┘
```

---

## Core Entities

### 1. Blackboard

Hierarchical key-value state for inter-node communication.

```python
@dataclass
class Blackboard:
    """Hierarchical key-value state."""

    # Identity
    id: str                          # UUID
    scope: BlackboardScope           # GLOBAL | TREE | SUBTREE

    # Data
    data: Dict[str, Any]             # Key-value pairs
    schema: Optional[Dict[str, Any]] # Expected structure (for validation)

    # Hierarchy
    parent: Optional["Blackboard"]   # Parent scope (None for global)

    # Metadata
    version: int                     # Optimistic concurrency
    created_at: datetime
    modified_at: datetime

    # Methods
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...
    def set_global(self, key: str, value: Any) -> None: ...
    def has(self, key: str) -> bool: ...
    def delete(self, key: str) -> bool: ...
    def snapshot(self) -> Dict[str, Any]: ...
    def validate(self) -> List[str]: ...  # Returns validation errors


class BlackboardScope(Enum):
    GLOBAL = "global"      # Persists across sessions
    TREE = "tree"          # Scoped to tree execution
    SUBTREE = "subtree"    # Scoped to subtree
```

**Validation Rules:**
- Keys must be strings
- Values must be JSON-serializable for persistence
- Schema validation optional but recommended
- Version increments on every write

**State Transitions:**
- Created → Active → Archived (on tree completion for TREE/SUBTREE scope)
- Global blackboard never archived, only persisted

---

### 2. BehaviorNode

Base abstraction for all tree nodes. Extends existing implementation.

```python
@dataclass
class BehaviorNode:
    """Base abstraction for all tree nodes."""

    # Identity
    id: str                          # Unique within tree
    name: str                        # Human-readable
    node_type: NodeType              # COMPOSITE | DECORATOR | LEAF

    # State
    status: RunStatus                # SUCCESS | FAILURE | RUNNING | FRESH
    tick_count: int                  # Times ticked
    running_since: Optional[datetime] # When entered RUNNING

    # Timing
    last_tick_duration_ms: float     # Duration of last tick
    total_running_time_ms: float     # Cumulative RUNNING time

    # Hierarchy
    parent: Optional["BehaviorNode"] # Parent node
    children: List["BehaviorNode"]   # Child nodes (composites)

    # Metadata
    metadata: Dict[str, Any]         # Arbitrary data
    source_location: Optional[str]   # Lua file:line

    # Methods (abstract)
    def tick(self, ctx: TickContext) -> RunStatus: ...
    def reset(self) -> None: ...
    def debug_info(self) -> Dict[str, Any]: ...


class NodeType(Enum):
    COMPOSITE = "composite"    # Sequence, Selector, Parallel
    DECORATOR = "decorator"    # Guard, Repeater, Cooldown, etc.
    LEAF = "leaf"              # Action, Condition, LLMCall


class RunStatus(Enum):
    SUCCESS = "success"        # Node completed successfully
    FAILURE = "failure"        # Node failed
    RUNNING = "running"        # Node still executing
    FRESH = "fresh"            # Node not yet ticked
```

**Validation Rules:**
- id unique within tree
- COMPOSITE nodes must have children
- DECORATOR nodes have exactly one child
- LEAF nodes have no children

---

### 3. BehaviorTree

A named, reusable composition of nodes.

```python
@dataclass
class BehaviorTree:
    """Named tree composition."""

    # Identity
    id: str                          # Unique identifier
    name: str                        # Human-readable name
    description: str                 # Purpose of tree

    # Structure
    root: BehaviorNode               # Root node
    node_count: int                  # Total nodes
    max_depth: int                   # Tree depth

    # State
    blackboard: Blackboard           # Tree-scoped blackboard
    status: TreeStatus               # IDLE | RUNNING | COMPLETED | FAILED
    tick_count: int                  # Total ticks executed

    # Source
    source_path: str                 # Path to Lua file
    source_hash: str                 # Hash for change detection
    loaded_at: datetime              # When parsed

    # Configuration
    max_tick_duration_ms: int        # Stuck detection threshold
    tick_budget: int                 # Max ticks per event

    # Methods
    def tick(self, ctx: TickContext) -> RunStatus: ...
    def reset(self) -> None: ...
    def reload(self, policy: ReloadPolicy) -> None: ...
    def debug_info(self) -> Dict[str, Any]: ...


class TreeStatus(Enum):
    IDLE = "idle"              # Not currently executing
    RUNNING = "running"        # Mid-execution
    COMPLETED = "completed"    # Last execution succeeded
    FAILED = "failed"          # Last execution failed


class ReloadPolicy(Enum):
    CANCEL_AND_RESTART = "cancel_and_restart"
    LET_FINISH_THEN_SWAP = "let_finish_then_swap"
    IMMEDIATE = "immediate"
```

---

### 4. TickContext

Context passed to every node on tick.

```python
@dataclass
class TickContext:
    """Execution context for tree ticks."""

    # Event that triggered this tick
    event: Event                     # From ANS EventBus

    # State access
    blackboard: Blackboard           # Current scope
    services: Services               # Dependency injection

    # Tick tracking
    tick_count: int                  # Ticks in current execution
    tick_budget: int                 # Max before yield
    start_time: datetime             # Execution start

    # Debugging
    parent_path: List[str]           # Path of parent node IDs
    trace_enabled: bool              # Whether to log every tick

    # Async coordination
    async_pending: Set[str]          # Pending async operation IDs

    # Methods
    def elapsed_ms(self) -> float: ...
    def budget_remaining(self) -> int: ...
    def budget_exceeded(self) -> bool: ...
    def push_path(self, node_id: str) -> None: ...
    def pop_path(self) -> None: ...
    def add_async(self, op_id: str) -> None: ...
    def complete_async(self, op_id: str) -> None: ...
    def has_pending_async(self) -> bool: ...
```

---

### 5. LLMCallNode

Specialized node for LLM API calls. Extends Leaf.

```python
@dataclass
class LLMCallNode(BehaviorNode):
    """Specialized node for LLM calls."""

    node_type: NodeType = NodeType.LEAF

    # LLM configuration
    model: str                       # Model identifier
    prompt_key: str                  # Blackboard key for prompt
    response_key: str                # Blackboard key for response

    # Streaming
    stream_to: Optional[str]         # Blackboard key for chunks
    on_chunk: Optional[str]          # Python callback path

    # Budget
    budget_tokens: int               # Max tokens
    tokens_used: int                 # Current usage

    # Control
    interruptible: bool              # Can be cancelled
    timeout_seconds: float           # Wall-clock timeout
    retry_on: List[str]              # Error types to retry
    retry_count: int                 # Current retries
    max_retries: int                 # Max retry attempts

    # Internal state
    request_id: Optional[str]        # In-flight request ID
    partial_response: str            # Accumulated chunks

    # Methods
    def tick(self, ctx: TickContext) -> RunStatus: ...
    def interrupt(self) -> None: ...
    def get_token_usage(self) -> Dict[str, int]: ...
```

**Tick Behavior:**
1. FRESH → Initiate request, return RUNNING
2. RUNNING → Check completion:
   - Streaming: Update blackboard, continue RUNNING
   - Complete: Write response, return SUCCESS
   - Error (retryable): Increment retry, return RUNNING
   - Error (fatal): Write error, return FAILURE
   - Budget exceeded: Write error, return FAILURE
   - Timeout: Write error, return FAILURE
   - Interrupted: Cleanup, return FAILURE

---

### 6. TreeRegistry

Manages loaded trees.

```python
@dataclass
class TreeRegistry:
    """Registry of loaded behavior trees."""

    # Storage
    trees: Dict[str, BehaviorTree]   # id -> tree

    # Configuration
    tree_dir: str                    # Directory to watch
    reload_policy: ReloadPolicy      # Default reload behavior

    # File watching
    watcher: Optional[FileWatcher]   # watchdog observer

    # Methods
    def load(self, path: str) -> BehaviorTree: ...
    def get(self, tree_id: str) -> Optional[BehaviorTree]: ...
    def reload(self, tree_id: str) -> None: ...
    def reload_all(self) -> None: ...
    def unload(self, tree_id: str) -> None: ...
    def list_trees(self) -> List[str]: ...
    def start_watching(self) -> None: ...
    def stop_watching(self) -> None: ...
```

---

### 7. Services (DI Container)

Injected dependencies available in TickContext.

```python
@dataclass
class Services:
    """Dependency injection container."""

    # LLM
    llm_client: LLMClient            # OpenRouter/Anthropic client

    # MCP Tools (NEW)
    tool_registry: ToolRegistry      # Registry of all MCP tools
    tool_executor: ToolExecutor      # Tool execution with async support

    # CodeRAG (NEW)
    coderag: CodeRAGService          # Direct CodeRAG search access
    oracle_bridge: OracleBridge      # Multi-source oracle queries

    # Tree Management (NEW)
    tree_registry: TreeRegistry      # Loaded behavior trees

    # Events
    event_bus: EventBus              # ANS EventBus

    # Persistence
    database: DatabaseService        # SQLite access

    # Context
    context_service: ContextService  # Conversation context

    # Configuration
    config: Config                   # Runtime configuration


@dataclass
class ToolRegistry:
    """Registry of MCP tools available as leaf nodes."""

    def get_tool(self, name: str) -> ToolDefinition: ...
    def list_tools(self) -> List[str]: ...
    def get_contract(self, name: str) -> NodeContract: ...


@dataclass
class CodeRAGService:
    """Direct CodeRAG access via vlt-cli/daemon."""

    def search_bm25(
        self,
        query: str,
        limit: int = 20,
        project_id: Optional[str] = None
    ) -> List[CodeChunk]: ...

    def get_project_status(self, project_id: str) -> ProjectStatus: ...

    async def trigger_reindex(
        self,
        project_id: str,
        force: bool = False
    ) -> str: ...  # Returns job_id
```

---

## Database Schema Extensions

### Blackboard Persistence Table

```sql
CREATE TABLE IF NOT EXISTS blackboard_state (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,           -- 'global', 'tree', 'subtree'
    tree_id TEXT,                  -- NULL for global
    data_json TEXT NOT NULL,       -- JSON serialized data
    schema_json TEXT,              -- Optional schema
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (tree_id) REFERENCES behavior_trees(id)
);

CREATE INDEX idx_blackboard_scope ON blackboard_state(scope);
CREATE INDEX idx_blackboard_tree ON blackboard_state(tree_id);
```

### Tree Metadata Table

```sql
CREATE TABLE IF NOT EXISTS behavior_trees (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_tick_at TIMESTAMP,
    total_ticks INTEGER DEFAULT 0,
    status TEXT DEFAULT 'idle'
);
```

### Tick History Table (for debugging)

```sql
CREATE TABLE IF NOT EXISTS tick_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tree_id TEXT NOT NULL,
    tick_number INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    node_path TEXT NOT NULL,        -- JSON array of node IDs
    status TEXT NOT NULL,           -- SUCCESS, FAILURE, RUNNING
    duration_ms REAL NOT NULL,
    blackboard_snapshot TEXT,       -- Optional JSON snapshot
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (tree_id) REFERENCES behavior_trees(id)
);

CREATE INDEX idx_tick_history_tree ON tick_history(tree_id, created_at DESC);
```

---

## Validation Summary

| Entity | Validation Rules |
|--------|------------------|
| Blackboard | Keys are strings, values JSON-serializable |
| BehaviorNode | Unique ID, correct children for type |
| BehaviorTree | Valid root, non-circular, schema matches |
| TickContext | Valid event, non-negative budgets |
| LLMCallNode | Valid model, positive budget |
| TreeRegistry | Unique tree IDs, valid paths |

---

## State Transitions

### RunStatus Transitions

```
     ┌─────────┐
     │  FRESH  │ ◄──── reset()
     └────┬────┘
          │ first tick
          ▼
     ┌─────────┐
┌───▶│ RUNNING │◄────┐
│    └────┬────┘     │
│         │          │ async continues
│    ┌────┴────┐     │
│    ▼         ▼     │
│ ┌─────┐  ┌─────┐   │
│ │SUCCESS│  │FAILURE│ │
│ └───┬───┘  └───┬───┘ │
│     │          │     │
│     └────┬─────┘     │
│          │           │
│     (completion)     │
│          │           │
└──────────┴───────────┘
```

### TreeStatus Transitions

```
     ┌─────────┐
     │  IDLE   │◄──────────────────┐
     └────┬────┘                   │
          │ tick()                 │
          ▼                        │
     ┌─────────┐                   │
     │ RUNNING │──────┬────────────┤
     └────┬────┘      │            │
          │           │            │
     ┌────┴────┐      │            │
     ▼         ▼      │            │
┌─────────┐ ┌─────────┐           │
│COMPLETED│ │ FAILED  │           │
└────┬────┘ └────┬────┘           │
     │           │                 │
     └─────┬─────┘                 │
           │ reset()               │
           └───────────────────────┘
```
