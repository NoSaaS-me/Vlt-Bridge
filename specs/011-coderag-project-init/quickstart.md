# Quickstart: CodeRAG Project Integration

**Feature Branch**: `011-coderag-project-init`
**Date**: 2026-01-01

## Implementation Overview

This feature adds interactive project selection to CodeRAG initialization, background indexing via daemon, and progress visibility in CLI and web UI.

---

## Phase 1: CLI Interactive Init

### 1.1 Add Project Listing to Service

**File**: `packages/vlt-cli/src/vlt/core/service.py`

```python
def list_projects(self) -> List[Project]:
    """List all projects in the database."""
    with Session(engine) as session:
        return list(session.scalars(select(Project).order_by(Project.name)))

def has_coderag_index(self, project_id: str) -> bool:
    """Check if project has an existing CodeRAG index."""
    with Session(engine) as session:
        count = session.scalar(
            select(func.count()).select_from(CodeChunk)
            .where(CodeChunk.project_id == project_id)
        )
        return count > 0
```

### 1.2 Update Init Command with Interactive Flow

**File**: `packages/vlt-cli/src/vlt/main.py`

Replace the current `coderag_init` command:

```python
from rich.prompt import Prompt, Confirm

@coderag_app.command("init")
def coderag_init(
    project: str = typer.Option(None, "--project", "-p"),
    path: Path = typer.Option(None, "--path"),
    force: bool = typer.Option(False, "--force"),
    background: bool = typer.Option(True, "--background/--foreground"),
):
    """Initialize CodeRAG index for a project."""
    console = Console()
    service = SqliteVaultService()

    # Step 1: Determine project (interactive if not provided)
    if not project:
        project = _interactive_project_selection(console, service)
        if not project:
            raise typer.Exit(code=1)

    # Step 2: Check for existing index
    if service.has_coderag_index(project) and not force:
        console.print(f"[yellow]Project '{project}' already has a code index.[/yellow]")
        if not Confirm.ask("Force re-index?"):
            console.print("Cancelled.")
            raise typer.Exit(code=0)
        force = True

    # Step 3: Queue indexing job
    target_path = path or Path.cwd()
    if background:
        job_id = _queue_background_indexing(project, target_path, force)
        console.print(f"[green]Indexing queued.[/green] Job ID: {job_id}")
        console.print(f"Check status: [bold]vlt coderag status --project {project}[/bold]")
    else:
        _run_foreground_indexing(project, target_path, force, console)


def _interactive_project_selection(console: Console, service: SqliteVaultService) -> Optional[str]:
    """Interactive project selection with create option."""
    projects = service.list_projects()

    if not projects:
        if Confirm.ask("No projects found. Create a new one?"):
            name = Prompt.ask("Project name")
            project = service.create_project(name, "")
            return project.id
        return None

    console.print("\n[bold]Available Projects:[/bold]")
    for i, proj in enumerate(projects, 1):
        has_index = "✓" if service.has_coderag_index(proj.id) else " "
        console.print(f"  {i}. [{has_index}] {proj.name} ({proj.id})")
    console.print(f"  {len(projects)+1}. Create new project")

    choice = Prompt.ask(
        "Select project",
        choices=[str(i) for i in range(1, len(projects)+2)],
        default="1"
    )

    idx = int(choice)
    if idx <= len(projects):
        return projects[idx-1].id
    else:
        name = Prompt.ask("New project name")
        project = service.create_project(name, "")
        return project.id
```

---

## Phase 2: Background Job Infrastructure

### 2.1 Add Job Model

**File**: `packages/vlt-cli/src/vlt/core/models.py`

```python
class JobStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CodeRAGIndexJob(Base):
    __tablename__ = "coderag_index_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    status: Mapped[JobStatus] = mapped_column(SQLAEnum(JobStatus), default=JobStatus.PENDING)
    target_path: Mapped[str] = mapped_column(String(512))
    force: Mapped[bool] = mapped_column(Boolean, default=False)

    files_total: Mapped[int] = mapped_column(Integer, default=0)
    files_processed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, default=0)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
```

### 2.2 Add Background Task to Daemon

**File**: `packages/vlt-cli/src/vlt/daemon/server.py`

Add to lifespan context manager:

```python
async def process_coderag_jobs():
    """Background task to process pending CodeRAG indexing jobs."""
    from ..core.coderag.indexer import CodeRAGIndexer

    while not state._shutdown_event.is_set():
        try:
            job = _get_next_pending_job()
            if job:
                await _run_indexing_job(job)
        except Exception as e:
            logger.error(f"Error in coderag job processor: {e}")

        try:
            await asyncio.wait_for(state._shutdown_event.wait(), timeout=10.0)
            break
        except asyncio.TimeoutError:
            continue


async def _run_indexing_job(job: CodeRAGIndexJob):
    """Execute a single indexing job with progress updates."""
    with Session(engine) as session:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        session.commit()

    try:
        indexer = CodeRAGIndexer(Path(job.target_path), job.project_id)

        # Progress callback
        def on_progress(files_done: int, files_total: int, chunks: int):
            with Session(engine) as session:
                j = session.get(CodeRAGIndexJob, job.id)
                j.files_processed = files_done
                j.files_total = files_total
                j.chunks_created = chunks
                j.progress_percent = int((files_done / max(files_total, 1)) * 100)
                session.commit()

        stats = await asyncio.to_thread(
            indexer.index_full,
            force=job.force,
            progress_callback=on_progress
        )

        with Session(engine) as session:
            j = session.get(CodeRAGIndexJob, job.id)
            j.status = JobStatus.COMPLETED
            j.completed_at = datetime.now(timezone.utc)
            session.commit()

    except Exception as e:
        with Session(engine) as session:
            j = session.get(CodeRAGIndexJob, job.id)
            j.status = JobStatus.FAILED
            j.error_message = str(e)
            j.completed_at = datetime.now(timezone.utc)
            session.commit()
```

---

## Phase 3: Backend API Routes

### 3.1 Create CodeRAG Router

**File**: `backend/src/api/routes/coderag.py`

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from ...services.oracle_bridge import OracleBridge
from ...models.coderag import (
    CodeRAGStatusResponse, InitCodeRAGRequest, InitCodeRAGResponse,
    JobStatusResponse
)

router = APIRouter(prefix="/api/coderag", tags=["coderag"])


def get_oracle_bridge() -> OracleBridge:
    return OracleBridge()


@router.get("/status", response_model=CodeRAGStatusResponse)
async def get_coderag_status(
    project_id: str = Query(..., description="Project ID"),
    bridge: OracleBridge = Depends(get_oracle_bridge),
):
    """Get CodeRAG index status for a project."""
    result = await bridge._run_vlt_command([
        "coderag", "status", "--project", project_id, "--json"
    ])
    return CodeRAGStatusResponse(**result)


@router.post("/init", response_model=InitCodeRAGResponse, status_code=201)
async def init_coderag(
    request: InitCodeRAGRequest,
    bridge: OracleBridge = Depends(get_oracle_bridge),
):
    """Queue a CodeRAG indexing job."""
    args = ["coderag", "init", "--project", request.project_id,
            "--path", request.target_path, "--background"]
    if request.force:
        args.append("--force")

    result = await bridge._run_vlt_command(args)
    return InitCodeRAGResponse(**result)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    bridge: OracleBridge = Depends(get_oracle_bridge),
):
    """Get status of an indexing job."""
    result = await bridge._run_vlt_command([
        "coderag", "job", job_id, "--json"
    ])
    if "error" in result:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**result)
```

### 3.2 Register Router

**File**: `backend/src/api/main.py`

```python
from .routes import coderag  # Add import

app.include_router(coderag.router, tags=["coderag"])  # Add registration
```

---

## Phase 4: Frontend Status Panel

### 4.1 Create CodeRAG Service

**File**: `frontend/src/services/coderag.ts`

```typescript
import { apiFetch } from './api';

export interface CodeRAGStatus {
  project_id: string;
  status: 'not_initialized' | 'indexing' | 'ready' | 'failed' | 'stale';
  file_count: number;
  chunk_count: number;
  last_indexed_at: string | null;
  error_message: string | null;
  active_job: {
    job_id: string;
    progress_percent: number;
    files_processed: number;
    files_total: number;
  } | null;
}

export async function getCodeRAGStatus(projectId: string): Promise<CodeRAGStatus> {
  return apiFetch<CodeRAGStatus>(`/api/coderag/status?project_id=${projectId}`);
}

export async function initCodeRAG(
  projectId: string,
  targetPath: string,
  force: boolean = false
): Promise<{ job_id: string; status: string }> {
  return apiFetch('/api/coderag/init', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, target_path: targetPath, force }),
  });
}
```

### 4.2 Add Settings Section

**File**: `frontend/src/pages/Settings.tsx`

Add after Index Health section:

```tsx
{/* Code Index Section */}
{coderagStatus ? (
  <Card>
    <CardHeader>
      <CardTitle>Code Index</CardTitle>
      <CardDescription>
        CodeRAG indexing status for code search
      </CardDescription>
    </CardHeader>
    <CardContent className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm text-muted-foreground">Chunks Indexed</div>
          <div className="text-2xl font-bold">{coderagStatus.chunk_count}</div>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">Status</div>
          <Badge variant={getStatusVariant(coderagStatus.status)}>
            {coderagStatus.status}
          </Badge>
        </div>
      </div>

      {coderagStatus.status === 'indexing' && coderagStatus.active_job && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Progress</span>
            <span>
              {coderagStatus.active_job.files_processed} / {coderagStatus.active_job.files_total} files
            </span>
          </div>
          <div className="w-full bg-muted rounded-full h-2">
            <div
              className="bg-primary h-2 rounded-full transition-all"
              style={{ width: `${coderagStatus.active_job.progress_percent}%` }}
            />
          </div>
        </div>
      )}

      {coderagStatus.last_indexed_at && (
        <div className="text-sm text-muted-foreground">
          Last indexed: {new Date(coderagStatus.last_indexed_at).toLocaleString()}
        </div>
      )}

      <Button
        variant="outline"
        disabled={coderagStatus.status === 'indexing'}
        onClick={handleReindex}
      >
        <RefreshCw className={`h-4 w-4 mr-2 ${coderagStatus.status === 'indexing' ? 'animate-spin' : ''}`} />
        {coderagStatus.status === 'indexing' ? 'Indexing...' : 'Re-index Code'}
      </Button>
    </CardContent>
  </Card>
) : (
  <SettingsSectionSkeleton title="Code Index" description="CodeRAG indexing status" />
)}
```

---

## Phase 5: Cascade Delete

### 5.1 Add CLI Delete Command

**File**: `packages/vlt-cli/src/vlt/main.py`

```python
@coderag_app.command("delete")
def coderag_delete(
    project: str = typer.Option(..., "--project", "-p"),
    confirm: bool = typer.Option(False, "--yes", "-y"),
):
    """Delete CodeRAG index for a project."""
    if not confirm:
        if not typer.confirm(f"Delete all CodeRAG data for project '{project}'?"):
            raise typer.Abort()

    service = SqliteVaultService()
    deleted = service.delete_coderag_index(project)
    console.print(f"[green]Deleted {deleted['chunks']} chunks, {deleted['nodes']} nodes[/green]")
```

### 5.2 Update Project Service

**File**: `backend/src/services/project_service.py`

Add after RAG directory cleanup (~line 365):

```python
# Delete project CodeRAG index
try:
    bridge = OracleBridge()
    result = bridge._run_vlt_command([
        "coderag", "delete", "--project", project_id, "--yes"
    ])
    logger.info(f"Deleted CodeRAG index for project {project_id}: {result}")
except Exception as e:
    logger.warning(f"Failed to delete CodeRAG index for {project_id}: {e}")
```

---

## Testing Checklist

- [ ] `vlt coderag init` prompts for project when none specified
- [ ] Existing projects with indexes show `[✓]` in selection list
- [ ] Force flag is required to re-index existing project
- [ ] Background indexing continues after terminal close
- [ ] `vlt coderag status` shows progress during indexing
- [ ] Web UI Settings shows Code Index status panel
- [ ] Progress bar updates during active indexing
- [ ] Project deletion cleans up CodeRAG data
- [ ] Job cancellation works from CLI and API

---

## End-to-End Validation Scenarios (T067)

This section documents the scenarios that should be manually tested to validate the full feature.

### Scenario 1: Fresh Project Interactive Init

**Steps:**
1. Ensure no projects exist: `vlt project list`
2. Navigate to a codebase directory with Python files
3. Run `vlt coderag init` (no flags)
4. When prompted, select "Create new project"
5. Enter project name and confirm

**Expected:**
- Interactive prompt appears for project creation
- New project is created
- Indexing starts (foreground or background depending on daemon status)
- Progress is displayed
- Status command shows indexed files

### Scenario 2: Background Indexing with Daemon

**Steps:**
1. Start the daemon: `vlt daemon start`
2. Run `vlt coderag init --project <id> --path <codebase>`
3. Immediately check status: `vlt coderag status --project <id>`
4. Wait for progress updates
5. Close terminal
6. Open new terminal and check status again

**Expected:**
- Job is queued (displays job ID)
- Status shows "running" with progress percentage
- After terminal close, indexing continues
- Status shows "completed" when done

### Scenario 3: Daemon Not Running Fallback (T064)

**Steps:**
1. Stop daemon if running: `vlt daemon stop`
2. Run `vlt coderag init --project <id>`
3. Observe warning message
4. Accept prompt to run in foreground

**Expected:**
- Warning: "Daemon is not running"
- Options displayed for starting daemon or foreground mode
- Prompt to run in foreground instead
- Foreground indexing with progress bar runs successfully

### Scenario 4: Overwrite Protection

**Steps:**
1. Run `vlt coderag init --project <id>` on a project with existing index
2. Observe warning and prompt
3. Decline to overwrite (answer "no")
4. Run again with `--force` flag

**Expected:**
- Warning about existing index is displayed
- Confirmation prompt appears
- Declining aborts without data loss
- `--force` flag bypasses the prompt

### Scenario 5: No Indexable Files (T063)

**Steps:**
1. Create empty directory or directory with only non-code files
2. Run `vlt coderag init --project <id> --path <empty-dir> --foreground`

**Expected:**
- Warning: "No indexable files found"
- Supported languages listed
- Recovery suggestions displayed
- No indexing job created

### Scenario 6: Job Cancellation (T062)

**Steps:**
1. Start daemon: `vlt daemon start`
2. Start large indexing job: `vlt coderag init --project <id> --path <large-codebase>`
3. Note the job ID from output
4. Cancel via API: `curl -X POST http://localhost:8000/api/coderag/jobs/<job_id>/cancel`
5. Check status: `vlt coderag status --project <id>`

**Expected:**
- Job starts running
- Cancel request succeeds
- Job status changes to "cancelled"
- Error message shows "Cancelled by user"

### Scenario 7: Status Command JSON Output

**Steps:**
1. Run `vlt coderag status --project <id> --json`
2. Pipe to jq: `vlt coderag status --project <id> --json | jq .`

**Expected:**
- Valid JSON output
- Contains: job_id, status, progress_percent, files_processed, files_total
- Contains: index_stats with chunks_count, symbols_count

### Scenario 8: Progress Visibility During Active Indexing

**Steps:**
1. Start indexing on medium-sized codebase
2. Run `vlt coderag status --project <id>` repeatedly
3. Observe progress percentage changes

**Expected:**
- progress_percent increases over time
- files_processed/files_total updates
- Time elapsed and ETA displayed
- Upon completion, status changes to "completed"

### Scenario 9: Error Recovery - Disk Space (T065)

**Note:** This is a destructive test - simulate only if you have a test environment.

**Steps:**
1. Fill disk to near capacity (simulate low space)
2. Run `vlt coderag init --project <id> --foreground`
3. Observe error when disk fills

**Expected:**
- Error: "Disk space exhausted during indexing"
- Recovery suggestions displayed
- Retry command provided

### Scenario 10: Web UI Status Panel

**Steps:**
1. Start backend: `./start-dev.sh`
2. Navigate to Settings page in browser
3. Observe Code Index section
4. Start indexing via CLI
5. Refresh Settings page

**Expected:**
- Code Index card displays status
- During indexing: progress bar with percentage
- After completion: chunk count and last indexed timestamp
- Re-index button available when not indexing

---

## Validation Commands Quick Reference

```bash
# Check project list
vlt project list

# Interactive init
vlt coderag init

# Specific project init
vlt coderag init --project <id> --path .

# Foreground with force
vlt coderag init --project <id> --foreground --force

# Check status
vlt coderag status --project <id>
vlt coderag status --project <id> --json

# Daemon management
vlt daemon start
vlt daemon stop
vlt daemon status

# Cancel job via API
curl -X POST http://localhost:8000/api/coderag/jobs/<job_id>/cancel
```
