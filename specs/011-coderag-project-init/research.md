# Research: CodeRAG Project Integration

**Feature Branch**: `011-coderag-project-init`
**Date**: 2026-01-01
**Status**: Complete

## Executive Summary

This research consolidates findings from four parallel investigations into the current codebase to inform the implementation of interactive CodeRAG project integration with background indexing and visibility.

---

## 1. CLI Init Workflow Research

### Current Implementation

**Location**: `packages/vlt-cli/src/vlt/main.py` (lines 820-925)

**Current command signature**:
```python
@coderag_app.command("init")
def coderag_init(
    project: str = typer.Option(None, "--project", "-p", ...),
    path: Path = typer.Option(None, "--path", ...),
    force: bool = typer.Option(False, "--force", ...),
):
```

**Current flow**:
1. Check if `--project` flag provided (explicit)
2. If not, call `load_project_identity()` from `vlt.core.identity`
3. If that fails, error: `"No project specified and no vlt.toml found."`
4. Run `CodeRAGIndexer.full_index()` synchronously (blocking)

### Decision: Interactive Prompting Library

| Option | Available | Recommendation |
|--------|-----------|----------------|
| `rich.prompt.Prompt` | Yes (imported) | **Selected** - Natural match with existing rich usage |
| `rich.prompt.Confirm` | Yes (imported) | **Selected** - Consistent with typer.confirm pattern |
| `typer.confirm()` | Yes (used in main.py) | Use for simple yes/no |
| questionary | Not installed | Rejected - adds new dependency |

**Rationale**: Rich is already heavily used for console output, tables, progress bars, and panels. Adding `rich.prompt` maintains consistency without new dependencies.

### Decision: Project Selection UI

```python
# Proposed implementation pattern
from rich.prompt import Prompt, Confirm

projects = service.list_projects()  # Need to add this method
if projects:
    console.print("[bold]Select a project:[/bold]")
    for i, proj in enumerate(projects, 1):
        has_index = "✓" if has_coderag_index(proj.id) else " "
        console.print(f"  {i}. [{has_index}] {proj.name} ({proj.id})")
    console.print(f"  {len(projects)+1}. Create new project")

    choice = Prompt.ask("Enter number", default="1")
```

**Rationale**: Simple numbered list with rich formatting is more terminal-friendly than complex TUI widgets.

---

## 2. Daemon Background Jobs Research

### Current Architecture

**Framework**: FastAPI async HTTP server
**Transport**: localhost:8765 (default)
**Storage**: SQLite at `~/.vlt/vault.db`

**Key files**:
- `daemon/server.py` - 669 lines, FastAPI app with lifespan
- `daemon/manager.py` - Process lifecycle with PID tracking
- `daemon/client.py` - CLI client for daemon communication

### Existing Job Patterns

The daemon has two queue types, neither is a generic job model:

1. **Sync Queue** - Thread entry syncing (JSON file storage)
2. **Dirty Threads** - Auto-summarization debouncing (in-memory)

### Decision: CodeRAG Job Model

**Schema addition to models.py**:
```python
class CodeRAGIndexJob(Base):
    __tablename__ = "coderag_index_jobs"

    id: Mapped[str]  # UUID
    project_id: Mapped[str]  # FK to projects
    status: Mapped[JobStatus]  # pending, running, completed, failed, cancelled
    target_path: Mapped[str]

    # Progress tracking
    progress_percent: Mapped[int]
    files_total: Mapped[int]
    files_processed: Mapped[int]
    chunks_created: Mapped[int]

    # Timestamps
    created_at: Mapped[datetime]
    started_at: Mapped[Optional[datetime]]
    completed_at: Mapped[Optional[datetime]]
    error_message: Mapped[Optional[str]]
```

**Rationale**: SQLite ORM model (like `IndexDeltaQueue`) provides durability across daemon restarts and queryable progress.

### Decision: Background Task Pattern

Add to `daemon/server.py` lifespan:
```python
async def process_coderag_jobs():
    """Background task polling for pending indexing jobs."""
    while not state._shutdown_event.is_set():
        job = get_next_pending_job()
        if job:
            await run_indexing_with_progress(job)
        await asyncio.sleep(10)  # Check every 10 seconds
```

**Rationale**: Follows existing `process_sync_queue()` and `process_dirty_threads()` patterns.

---

## 3. Backend API Research

### Current State

**No dedicated CodeRAG routes exist**. CodeRAG is accessed via:
- `OracleBridge.search_code()` - subprocess to `vlt coderag search`
- `OracleBridge._check_coderag_initialized()` - internal method, not exposed

### Decision: New API Routes

Create `backend/src/api/routes/coderag.py`:

```python
router = APIRouter(prefix="/api/coderag", tags=["coderag"])

@router.get("/status", response_model=CodeRAGStatus)
async def get_coderag_status(project_id: str = Query(...)):
    """Get CodeRAG index status for a project."""

@router.post("/init", response_model=InitJobResponse)
async def init_coderag(request: InitCodeRAGRequest):
    """Queue a CodeRAG indexing job."""

@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of an indexing job."""

@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel an in-progress indexing job."""
```

**Rationale**: Follows established patterns in `index.py` and `rag.py`.

### Decision: Project Deletion Cascade

**Location**: `backend/src/services/project_service.py` line ~365

Add after RAG directory cleanup:
```python
# Delete project CodeRAG data via vlt CLI
try:
    bridge = OracleBridge()
    await bridge._run_vlt_command(["coderag", "delete", "--project", project_id])
except Exception as e:
    logger.warning(f"CodeRAG cleanup failed: {e}")
```

**Alternatives considered**:
- Direct SQLite deletion (rejected: vlt DB is separate from backend DB)
- Ignore cleanup (rejected: orphans data)

---

## 4. Frontend Settings UI Research

### Current Structure

Settings page has 6 Card sections:
1. Profile
2. API Token
3. **Index Health** ← Pattern to follow
4. AI Models
5. Context Settings
6. System Logs

### Decision: CodeRAG Status Panel Pattern

Follow Index Health section pattern (lines 403-459):
```tsx
<Card>
  <CardHeader>
    <CardTitle>Code Index</CardTitle>
    <CardDescription>CodeRAG indexing status for code search</CardDescription>
  </CardHeader>
  <CardContent className="space-y-4">
    <div className="grid grid-cols-2 gap-4">
      <div>
        <div className="text-sm text-muted-foreground">Chunks Indexed</div>
        <div className="text-2xl font-bold">{status.chunk_count}</div>
      </div>
      <div>
        <div className="text-sm text-muted-foreground">Status</div>
        <Badge variant={getBadgeVariant(status.status)}>
          {status.status}
        </Badge>
      </div>
    </div>
    {/* Progress bar for active indexing */}
    {status.status === 'indexing' && (
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Progress</span>
          <span>{status.progress}%</span>
        </div>
        <div className="w-full bg-muted rounded-full h-2">
          <div
            className="bg-primary h-2 rounded-full transition-all"
            style={{ width: `${status.progress}%` }}
          />
        </div>
      </div>
    )}
  </CardContent>
</Card>
```

### Decision: Progress Bar Implementation

**No Progress component in shadcn/ui** is installed. Options:

| Option | Recommendation |
|--------|----------------|
| Install `@radix-ui/react-progress` | Overkill for simple bar |
| Custom div with width% | **Selected** - matches Alert/Badge simplicity |
| Use spinning icon only | Insufficient for long operations |

**Rationale**: A simple styled div with `style={{ width: percentage }}` is lightweight and adequate.

---

## 5. Service Layer Gaps Identified

### Missing in vlt CLI (`SqliteVaultService`):

1. `list_projects() -> List[Project]` - Query all projects
2. `has_coderag_index(project_id) -> bool` - Check if index exists
3. `delete_coderag_index(project_id)` - Clean up index data

### Missing in Backend:

1. `CodeRAGService` - New service to wrap OracleBridge calls
2. Routes for `/api/coderag/*`
3. Project cascade delete for CodeRAG data

### Missing in Frontend:

1. `coderagService.ts` - API client for CodeRAG endpoints
2. `CodeRAGStatus` type definition
3. Settings section for CodeRAG status/control

---

## 6. Consolidated Technical Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Prompting Library** | rich.prompt.Prompt | Already used, no new deps |
| **Project List UI** | Numbered list with status | Terminal-friendly |
| **Job Storage** | SQLAlchemy model | Durable, queryable |
| **Background Task** | asyncio in daemon | Follows existing patterns |
| **API Pattern** | FastAPI router with Depends | Consistent with codebase |
| **Frontend Progress** | Custom div percentage bar | No additional packages |
| **Cascade Delete** | Via OracleBridge subprocess | Separate DB constraint |

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Daemon not running during init | Add `--foreground` fallback to run sync |
| Large repo timeout | Stream progress, allow cancellation |
| Embedding API rate limits | Queue with exponential backoff |
| Frontend polling overhead | WebSocket or SSE for real-time (future) |

---

## Next Steps

1. Generate data-model.md with entity definitions
2. Generate API contracts in `/contracts/`
3. Create quickstart.md implementation guide
