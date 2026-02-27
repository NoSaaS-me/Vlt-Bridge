# VLT-CLI Package Thorough Exploration Report

## 1. DIRECTORY STRUCTURE

```
packages/vlt-cli/
├── src/vlt/
│   ├── main.py                 # 4561 lines: CLI entry point (typer app)
│   ├── config.py               # Settings management (profile-aware)
│   ├── profile.py              # ProfileManager: named profile system
│   ├── db.py                   # SQLAlchemy setup (SQLite)
│   ├── mcp_server.py           # FastMCP server entry point
│   ├── mcp/                    # MCP tools modules
│   │   ├── __init__.py
│   │   ├── thread_tools.py     # vlt_thread_* MCP tools
│   │   ├── meta_tools.py       # vlt_* MCP tools
│   │   ├── code_tools.py       # vlt_code_* MCP tools
│   │   ├── oracle_tools.py     # vlt_oracle_* MCP tools
│   │   └── vault_tools.py      # vlt_note_* MCP tools
│   ├── daemon/                 # Background sync daemon
│   │   ├── __init__.py
│   │   ├── server.py           # FastAPI daemon server (34KB)
│   │   ├── client.py           # DaemonClient for CLI
│   │   └── manager.py          # DaemonManager: process lifecycle
│   ├── core/                   # Business logic
│   │   ├── models.py           # SQLAlchemy models
│   │   ├── service.py          # SqliteVaultService
│   │   ├── oracle.py           # Oracle client
│   │   ├── sync.py             # ThreadSyncClient
│   │   ├── migrations.py       # DB schema
│   │   ├── identity.py         # Project identity (vlt.toml)
│   │   ├── coderag/            # CodeRAG indexing
│   │   │   └── indexer.py      # CodeRAGIndexer
│   │   ├── retrievers/         # RAG retrievers
│   │   └── [other modules]     # oracle_client, query_analyzer, etc.
│   └── tests/
│       └── unit/               # Unit tests
├── pyproject.toml              # Dependencies, entry points
└── [venv, tests root level]
```

## 2. CLI COMMAND REGISTRATION PATTERN (typer)

**Main entry point** (pyproject.toml):
```
[project.scripts]
vlt = "vlt.main:app"
vlt-mcp = "vlt.mcp_server:main"
```

**Command hierarchy in main.py** (lines 77-87):
```python
app = typer.Typer(name="vlt", help=APP_HELP, no_args_is_help=True)

# Sub-apps (command groups)
thread_app = typer.Typer(name="thread", help=...)
config_app = typer.Typer(name="config", help=...)
sync_app = typer.Typer(name="sync", help=...)
daemon_app = typer.Typer(name="daemon", help=...)
profile_app = typer.Typer(name="profile", help=...)

# Register sub-apps with main app
app.add_typer(thread_app, name="thread")
app.add_typer(config_app, name="config")
app.add_typer(sync_app, name="sync")
app.add_typer(daemon_app, name="daemon")
app.add_typer(profile_app, name="profile")
```

**Command decorator pattern**:
```python
@thread_app.command("new")
def new_thread(
    name: str = typer.Argument(..., help="..."),
    initial_thought: str = typer.Argument(..., help="..."),
    project: str = typer.Option(None, "--project", "-p", help="..."),
    author: str = typer.Option(None, "--author", help="...")
):
    """Docstring becomes help text"""
    # Implementation
```

**Full command list**:
- `vlt config set-key`
- `vlt sync status`, `retry`, `all`
- `vlt profile list`, `show`, `add`, `use`, `delete`, `init`
- `vlt daemon start`, `stop`, `status`, `restart`, `logs`
- `vlt thread new`, `push`, `read`, `move`
- `vlt librarian run`
- Plus coderag, oracle, overview commands

## 3. DAEMON ARCHITECTURE

### 3a. Daemon Server (FastAPI, localhost)

**File**: `daemon/server.py` (34KB, 998 lines)

**Startup** (run_server function, line 929):
```python
def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    profile_name: Optional[str] = None,
):
    # Uses uvicorn.run(app, host, port)
```

**Lifespan management** (lines 664-714):
- Initializes DaemonState (settings, sync_client, http_client)
- Creates persistent httpx.AsyncClient with auth header
- Checks backend connection
- Starts 3 background tasks:
  1. `process_sync_queue()` - Retries failed syncs (every 30s)
  2. `process_dirty_threads()` - Lazy summarization with debouncing
  3. `process_coderag_jobs()` - CodeRAG background indexing
- Graceful shutdown: sets _shutdown_event, cancels tasks, closes HTTP client

**HTTP Endpoints**:
1. **GET /health** → HealthResponse
   - status, uptime_seconds, backend_url, backend_connected, queue_size, dirty_threads

2. **POST /sync/enqueue** (EnqueueRequest) → EnqueueResponse
   - Queues sync entry, tries immediate sync if backend connected
   - Marks thread as dirty for lazy summarization

3. **GET /sync/status** → SyncStatusResponse
   - Queue size, pending items, daemon uptime, backend connection

4. **POST /sync/retry** → RetryResponse
   - Retries all queued entries

5. **GET /summarize/pending** → DirtyThreadsStatusResponse
   - Threads pending summarization with timing info

6. **POST /summarize/{thread_id}** (SummarizeRequest) → SummarizeResponse
   - Proxies summarization request to backend

### 3b. Background Task: Lazy Summarization (lines 286-399)

```
SUMMARIZE_DELAY_SECONDS = 30     # Debounce window
SUMMARIZE_CHECK_INTERVAL_SECONDS = 10

async def process_dirty_threads():
    # Track DirtyThreadInfo(thread_id, project_id, last_push_time, retry_count)
    # After 30s of inactivity, request server-side summarization
    # Retries up to 3 times on failure
    # Updates local State table with summary
```

### 3c. Background Task: CodeRAG Indexing (lines 630-657)

- Polls for pending CodeRAGIndexJob records (priority DESC, created_at ASC)
- Runs indexer in thread pool via asyncio.to_thread()
- Progress callback updates job record in real-time
- Handles JobCancelledException, OSError (disk space), permission errors
- Stores embedding API key per job (allows daemon to be started without key)

### 3d. DaemonState (lines 139-166)

```python
class DaemonState:
    start_time: datetime
    settings: Settings
    sync_client: ThreadSyncClient
    http_client: httpx.AsyncClient      # Persistent connection to backend
    backend_connected: bool
    processing_queue: bool
    dirty_threads: Dict[str, DirtyThreadInfo]  # For lazy summarization
    _summarize_lock: asyncio.Lock
    _shutdown_event: asyncio.Event
```

### 3e. Daemon Manager (Process Lifecycle)

**File**: `daemon/manager.py` (324 lines)

**DaemonManager class**:
- `__init__(port, profile_name)`: Sets up PID file, log file, client
- `start(foreground=False)`: Starts daemon background process
  - Checks if already running via health check
  - Cleans up stale PID files
  - Runs in foreground (blocking) or background (subprocess)
  - Writes PID to `~/.vlt/profiles/{profile}/daemon.pid`
  - Logs to `~/.vlt/profiles/{profile}/daemon.log`

- `_run_foreground()`: Calls `run_server()` (blocking)
- `_run_background()`: Uses `subprocess.Popen()` with `start_new_session=True`
  - Command: `python -m vlt.daemon.server --port {port} --profile {profile}`
  - Detaches from parent, runs from home directory
  - Returns after verifying startup

- `stop()`: Sends SIGTERM to PID, waits 3s, then SIGKILL
- Process ops: `_read_pid()`, `_write_pid()`, `_remove_pid()`, `_is_process_running()`

### 3f. Daemon Client (CLI Communication)

**File**: `daemon/client.py` (200+ lines)

**DaemonClient class**:
```python
HEALTH_TIMEOUT = 0.5        # Fast-fail for health checks
OPERATION_TIMEOUT = 5.0     # Normal operations

async def is_running() -> bool
async def get_status() -> DaemonStatus
async def enqueue_sync(thread_id, project_id, name, entry) -> EnqueueResult
async def request_summarize(thread_id) -> SummarizeResult
async def retry_sync() -> RetryResult
```

**Used in commands** (e.g., thread push, line 1399):
```python
if settings.daemon_enabled:
    client = DaemonClient(settings.daemon_url)
    if await client.is_running():
        result = await client.enqueue_sync(...)
        return result.success, not result.queued
    # Falls back to direct sync if daemon not available
```

## 4. SETTINGS/CONFIG SYSTEM

**File**: `config.py` (306 lines)

**Profile-aware configuration flow**:
1. `ProfileManager.get_active_profile()` → determines active profile
2. `ProfileManager.get_env_file()` → `~/.vlt/profiles/{profile}/.env`
3. `Settings._load_env_file()` → reads VLT_* environment variables
4. Precedence: env vars > .env file > defaults

**Settings class**:
```python
class Settings(BaseSettings):
    # Database
    database_url: str = ""                  # Profile-specific SQLite

    # Server sync
    sync_token: Optional[str] = None        # VLT_SYNC_TOKEN
    vault_url: str = "http://localhost:8000"  # VLT_VAULT_URL

    # Oracle
    oracle_timeout: float = 60.0
    oracle_prefer_backend: bool = True
    oracle_enabled: bool = True             # VLT_ORACLE_ENABLED

    # Daemon
    daemon_port: int = 8765                 # Profile-specific
    daemon_enabled: bool = True
    daemon_url: str = ""                    # Computed from daemon_port

    # Profile (informational)
    profile: Optional[str] = None           # VLT_PROFILE

    @property
    def is_server_configured(self) -> bool:
        return bool(self.sync_token)

    @property
    def can_use_backend_oracle(self) -> bool:
        return self.is_server_configured and self.oracle_prefer_backend
```

**Global settings instance**:
```python
_settings: Optional[Settings] = None

def get_settings(profile_name=None, force_reload=False) -> Settings:
    # Lazy singleton pattern

def reload_settings(profile_name=None) -> Settings:
    # Clears cache, reloads everything

settings = get_settings()  # Created at import time
```

## 5. PROFILE SYSTEM

**File**: `profile.py` (400+ lines)

**Directory structure**:
```
~/.vlt/
├── config.toml              # Active profile pointer, global settings
└── profiles/
    ├── default/
    │   ├── vault.db
    │   ├── .env
    │   ├── daemon.pid
    │   └── daemon.log
    └── {profile-name}/
```

**ProfileManager class**:
```python
get_active_profile() -> str
    # 1. Check VLT_PROFILE env var
    # 2. Check .vlt/profile file in cwd
    # 3. Check config.toml active_profile
    # 4. Default to "default"

get_profile_dir(profile_name) -> Path
    # ~/.vlt/profiles/{profile_name}/

get_env_file(profile_name) -> Path
get_database_url(profile_name) -> str
    # sqlite:///{profile_dir}/vault.db

get_daemon_port(profile_name) -> int
    # Hash-based unique port per profile

create_profile(name, description) -> bool
delete_profile(name) -> bool
list_profiles() -> List[dict]
```

**Profile validation**:
```
PROFILE_NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
MAX_PROFILE_NAME_LENGTH = 64
RESERVED_NAMES = {"profiles", "config", "global"}
```

## 6. EXAMPLE COMMANDS: thread new & thread push

### thread new (lines 1270-1340)

```python
@thread_app.command("new")
def new_thread(
    name: str,
    initial_thought: str,
    project: str = typer.Option(None, "--project", "-p"),
    author: str = typer.Option(None, "--author"),
):
    # 1. Resolve author (from arg or state["author"])
    # 2. Resolve project (from arg, vlt.toml, or error)
    # 3. Ensure project exists
    # 4. Create thread via service
    # 5. Sync to backend (full thread + all entries)
    # 6. Print success + sync status
```

### thread push (lines 1341-1444)

```python
@thread_app.command("push")
def push_thought(
    thread_id: str,
    content: str,
    author: str = typer.Option(None, "--author"),
):
    # 1. Resolve author
    # 2. Parse thread_id (handle project/thread format)
    # 3. Add thought to thread via service
    # 4. Print success
    # 5. If server configured:
    #    a. Try daemon first (if enabled)
    #    b. Fallback to direct sync if daemon not running
    # 6. Print sync status
```

**Daemon integration in push** (lines 1396-1412):
```python
if settings.daemon_enabled:
    async def try_daemon():
        client = DaemonClient(settings.daemon_url)
        if await client.is_running():
            result = await client.enqueue_sync(...)
            return result.success, not result.queued
        return False, False

    daemon_ok, synced = asyncio.run(try_daemon())

# Fallback to direct sync if daemon not available
if not via_daemon:
    synced = asyncio.run(sync_thread_entry(...))
```

## 7. DATABASE MODELS

**File**: `core/models.py`

**Key models**:
1. **Project**: id (slug), name, description, created_at, threads (rel)
2. **Thread**: id (slug), project_id, status, created_at, nodes (rel)
3. **Node**: id (uuid), thread_id, sequence_id, content, author, timestamp, embedding, prev_node_id, tags, refs
4. **State**: id (uuid), target_id, target_type ("thread"|"project"), summary, head_node_id, updated_at, meta
5. **Tag**: id (int), name, nodes (m2m)
6. **Reference**: id, source_node_id, target_thread_id, note

**CodeRAG model** (T024-T029):
- **CodeRAGIndexJob**: id, project_id, target_path, status (PENDING|RUNNING|COMPLETED|FAILED|CANCELLED), priority, progress_percent, files_processed, files_total, chunks_created, started_at, completed_at, error_message, embedding_api_key, force

## 8. DATABASE CONNECTION

**File**: `db.py` (106 lines)

```python
# SQLite pragmas for performance
PRAGMA foreign_keys=ON
PRAGMA journal_mode=WAL          # Better concurrency
PRAGMA synchronous=NORMAL        # Faster with WAL

# Engine setup
engine = create_engine(settings.database_url, check_same_thread=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Profile-specific
get_engine_for_profile(profile_name) -> Engine
get_session_for_profile(profile_name) -> sessionmaker
```

## 9. PYPROJECT.TOML DEPENDENCIES

```toml
[project]
requires-python = ">=3.11"

dependencies = [
    "typer>=0.9.0",             # CLI framework
    "pydantic>=2.0.0",          # Validation
    "pydantic-settings>=2.0.0", # Config management
    "sqlalchemy>=2.0.0",        # ORM
    "rich>=13.0.0",             # TUI/formatting
    "numpy>=1.26.0",            # Arrays
    "openai>=1.0.0",            # LLM
    "httpx>=0.25.0",            # Async HTTP
    "alembic>=1.13.0",          # DB migrations
    "fastapi>=0.109.0",         # Web framework (daemon)
    "uvicorn>=0.27.0",          # ASGI server (daemon)
    "tomli-w>=1.0.0",           # TOML writing
    "fastmcp>=2.13.1",          # MCP server
]

[project.scripts]
vlt = "vlt.main:app"
vlt-mcp = "vlt.mcp_server:main"
```

## 10. DAEMON LIFECYCLE COMMANDS

```python
@daemon_app.command("start")
def daemon_start(port=None, foreground=False):
    # Uses DaemonManager.start(foreground)

@daemon_app.command("stop")
def daemon_stop():
    # Uses DaemonManager.stop()

@daemon_app.command("status")
def daemon_status(json_output=False):
    # Shows uptime, backend connection, queue size

@daemon_app.command("restart")
def daemon_restart():
    # stop() then start()

@daemon_app.command("logs")
def daemon_logs(follow=False, lines=100):
    # Tail daemon.log file
```

## 11. NO WEBSOCKET INFRASTRUCTURE (Current)

**Current implementation**:
- Daemon uses synchronous HTTP endpoints (FastAPI)
- Client uses short-lived httpx.AsyncClient connections
- No persistent WebSocket connection between CLI and daemon
- All communication is request-response via HTTP

**Potential for WebSocket**:
- Could add `/ws` endpoint to daemon server
- Could maintain persistent connection for real-time updates
- Would be useful for `vlt daemon logs --follow` (streaming)
- Would be useful for progress updates on long-running jobs (CodeRAG indexing)

## 12. SESSION/PROCESS MANAGEMENT

**Current process management**:
1. **Profile isolation**: Each profile has own PID file, log file, port
2. **PID tracking**: `~/.vlt/profiles/{profile}/daemon.pid`
3. **Health checks**: `DaemonClient.is_running()` via GET /health
4. **Graceful shutdown**: SIGTERM → 3s wait → SIGKILL
5. **Stale PID cleanup**: Check if process still running before restart

**No explicit session management**:
- No session store for daemon
- No user authentication (local daemon = trusted)
- No request/session tracking beyond HTTP request lifecycle
- State is transient (in-memory) except for database

## 13. KEY RESOURCES FOR SESSION-RELAY IMPLEMENTATION

**To implement `session-relay` command + daemon WebSocket endpoint**:

1. **Add WebSocket endpoint** to daemon/server.py
   - Pattern: `GET /ws/session/{session_id}` with upgrade
   - Maintain open connection with reconnect support

2. **Add DaemonClient method** for WebSocket subscription
   - `async def subscribe_session(session_id) -> AsyncIterator[str]`

3. **Add CLI command** in main.py
   - `@daemon_app.command("relay")`
   - Connects via WebSocket, streams messages

4. **Message format**
   - Define event protocol (JSON on wire)
   - Support subscription patterns (e.g., "thread:*", "sync:status")

5. **Session store in daemon**
   - Track active sessions + subscribers
   - Broadcast status/progress updates
   - Handle reconnection with message buffering

**Infrastructure ready**:
- FastAPI app with lifespan management ✓
- Background task pattern ✓
- State tracking (dirty_threads model) ✓
- Config/profile support ✓
- Error handling patterns ✓
