"""
VLT Daemon Server - FastAPI server for background sync operations.

This server runs on localhost and provides:
- POST /sync/enqueue - Queue a sync operation (thread push uses this)
- POST /sync/retry - Retry failed syncs
- GET /sync/status - Get queue status
- POST /summarize/{thread_id} - Request summarization
- GET /summarize/pending - Get status of threads pending auto-summarization
- GET /health - Health check

The daemon maintains a persistent httpx.AsyncClient to the backend server,
eliminating connection overhead for each CLI call.

LAZY AUTO-SUMMARIZATION:
When a thread is synced via /sync/enqueue, it is marked as "dirty" for
summarization. After SUMMARIZE_DELAY_SECONDS (default: 30) of inactivity
on that thread, the daemon automatically requests summarization from the
backend and updates the local State table. This provides debouncing so that
rapid pushes only trigger one summarization after the flurry stops.

CODERAG BACKGROUND INDEXING:
The daemon also processes CodeRAG indexing jobs queued via `vlt coderag init`.
Jobs are stored in the coderag_index_jobs table and processed one at a time
in priority order (DESC priority, ASC created_at). The daemon updates job
progress in real-time via progress callbacks during indexing.
"""

import asyncio
import fcntl
import json
import logging
import os
import pty
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from vlt.config import Settings
from vlt.core.sync import ThreadSyncClient, SyncQueueItem

logger = logging.getLogger(__name__)


# =============================================================================
# Project ID Inference from CWD (cached)
# =============================================================================

_cwd_project_cache: Dict[str, Optional[str]] = {}


def _infer_project_id(cwd: str) -> Optional[str]:
    """Walk up from cwd looking for vlt.toml and extract project.id.

    Results (including None) are cached per resolved path for the lifetime
    of the daemon process to avoid repeated filesystem walks.
    """
    resolved = os.path.realpath(cwd)
    if resolved in _cwd_project_cache:
        return _cwd_project_cache[resolved]

    result = None
    current = Path(resolved)
    while True:
        candidate = current / "vlt.toml"
        if candidate.exists():
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore
                except ImportError:
                    break
            try:
                with open(candidate, "rb") as f:
                    data = tomllib.load(f)
                result = data.get("project", {}).get("id") or data.get("project_id")
            except Exception:
                pass
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    _cwd_project_cache[resolved] = result
    return result


# =============================================================================
# Session Relay — In-Memory Stream Registries
# =============================================================================

# Maps session_id → asyncio.Queue of bytes chunks for WebSocket relay
_session_streams: Dict[str, asyncio.Queue] = {}
# Maps session_id → asyncio.Queue for injecting keystrokes/text into PTY
# Items are raw bytes for inject OR JSON bytes for control messages (resize, etc.)
_session_inject_queues: Dict[str, asyncio.Queue] = {}
# Scrollback buffer: last 64 KB of PTY output per session (for late-joining browsers)
_session_scrollback: Dict[str, bytearray] = {}
_MAX_SCROLLBACK = 64 * 1024

# Scrollback persistence — written to disk so content survives daemon restarts
_scrollback_dir = Path.home() / ".vlt" / "scrollback"
_scrollback_dir.mkdir(parents=True, exist_ok=True)
_scrollback_write_tasks: Dict[str, "asyncio.Task[None]"] = {}


def _scrollback_path(session_id: str) -> Path:
    return _scrollback_dir / f"{session_id}.bin"


async def _persist_scrollback(session_id: str, buf: bytearray) -> None:
    """Write scrollback buffer to disk (called from debounced task)."""
    try:
        await asyncio.to_thread(_scrollback_path(session_id).write_bytes, bytes(buf))
    except Exception as e:
        logger.debug(f"Failed to persist scrollback for {session_id}: {e}")


def _schedule_scrollback_persist(session_id: str, buf: bytearray) -> None:
    """Debounce scrollback disk writes — coalesces rapid chunks into one write."""
    existing = _scrollback_write_tasks.get(session_id)
    if existing and not existing.done():
        existing.cancel()

    async def _delayed() -> None:
        await asyncio.sleep(0.5)
        await _persist_scrollback(session_id, buf)

    _scrollback_write_tasks[session_id] = asyncio.create_task(_delayed())

# Ring buffer for recent hook events (max 500 entries) — served via GET /api/hooks/recent
_hook_events: List[Dict[str, Any]] = []
_MAX_HOOK_EVENTS = 500


# =============================================================================
# Request/Response Models
# =============================================================================

class EnqueueRequest(BaseModel):
    """Request to queue a sync entry."""
    thread_id: str
    project_id: str
    name: str
    entry: Dict[str, Any]  # entry_id, sequence_id, content, author, timestamp


class EnqueueResponse(BaseModel):
    """Response from enqueue operation."""
    queued: bool
    message: str
    queue_size: int


class SyncStatusResponse(BaseModel):
    """Response with sync queue status."""
    pending: int
    items: List[Dict[str, Any]]
    daemon_uptime_seconds: float
    backend_connected: bool


class RetryResponse(BaseModel):
    """Response from retry operation."""
    success: int
    failed: int
    skipped: int


class SummarizeRequest(BaseModel):
    """Request for server-side summarization."""
    current_summary: Optional[str] = None


class SummarizeResponse(BaseModel):
    """Response from summarization request."""
    thread_id: str
    summary: str
    model: Optional[str] = None
    tokens_used: int = 0
    success: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    uptime_seconds: float
    backend_url: str
    backend_connected: bool
    queue_size: int
    dirty_threads: int


class DirtyThreadsStatusResponse(BaseModel):
    """Response with dirty threads status."""
    count: int
    threads: List[Dict[str, Any]]
    summarize_delay_seconds: int


# =============================================================================
# Lazy Summarization Config
# =============================================================================

# Wait this many seconds after the last push before triggering summarization
SUMMARIZE_DELAY_SECONDS = 30

# How often to check for dirty threads that need summarization
SUMMARIZE_CHECK_INTERVAL_SECONDS = 10


# =============================================================================
# Daemon State
# =============================================================================

class DirtyThreadInfo(BaseModel):
    """Tracking info for a thread that needs summarization."""
    thread_id: str
    project_id: str
    last_push_time: float  # time.time() value
    retry_count: int = 0


class DaemonState:
    """Shared state for the daemon server."""

    def __init__(self):
        self.start_time: datetime = datetime.now(timezone.utc)
        self.settings: Settings = Settings()
        self.sync_client: Optional[ThreadSyncClient] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.backend_connected: bool = False
        self.processing_queue: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        # Dirty threads needing summarization: thread_id -> DirtyThreadInfo
        self.dirty_threads: Dict[str, DirtyThreadInfo] = {}
        self._summarize_lock: asyncio.Lock = asyncio.Lock()

    @property
    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    @property
    def vault_url(self) -> str:
        return self.settings.vault_url.rstrip("/")

    @property
    def sync_token(self) -> Optional[str]:
        return self.settings.sync_token


state = DaemonState()


# =============================================================================
# Background Tasks
# =============================================================================

async def check_backend_connection():
    """Check if we can connect to the backend."""
    if not state.http_client:
        state.backend_connected = False
        return

    try:
        response = await state.http_client.get(
            f"{state.vault_url}/health",
            timeout=5.0
        )
        state.backend_connected = response.status_code == 200
    except Exception as e:
        logger.debug(f"Backend health check failed: {e}")
        state.backend_connected = False


async def process_sync_queue():
    """Background task that processes the sync queue periodically."""
    while not state._shutdown_event.is_set():
        try:
            # Only process if not already processing
            if not state.processing_queue and state.sync_client:
                state.processing_queue = True
                try:
                    result = await state.sync_client.retry_queue()
                    if result["success"] > 0:
                        logger.info(f"Processed {result['success']} queued entries")
                finally:
                    state.processing_queue = False

            # Also check backend connection
            await check_backend_connection()

        except Exception as e:
            logger.error(f"Error in queue processor: {e}")

        # Wait before next check
        try:
            await asyncio.wait_for(
                state._shutdown_event.wait(),
                timeout=30.0  # Process queue every 30 seconds
            )
            break  # Shutdown requested
        except asyncio.TimeoutError:
            continue  # Continue processing


async def update_local_state(thread_id: str, summary: str, last_node_id: Optional[str] = None) -> bool:
    """
    Update the local State table with the new summary.

    Args:
        thread_id: Thread identifier
        summary: New summary from server
        last_node_id: ID of the last node processed (optional)

    Returns:
        True if successful, False otherwise
    """
    try:
        from vlt.db import SessionLocal
        from vlt.core.models import State, Node

        db = SessionLocal()
        try:
            # Get current state
            current_state = db.scalars(
                select(State)
                .where(State.target_id == thread_id)
                .where(State.target_type == "thread")
            ).first()

            # If no last_node_id provided, get the latest node for this thread
            if not last_node_id:
                latest_node = db.scalars(
                    select(Node)
                    .where(Node.thread_id == thread_id)
                    .order_by(Node.sequence_id.desc())
                ).first()
                if latest_node:
                    last_node_id = latest_node.id

            if current_state:
                # Update existing state
                current_state.summary = summary
                if last_node_id:
                    current_state.head_node_id = last_node_id
                logger.info(f"Updated State for thread {thread_id}")
            else:
                # Create new state
                new_state = State(
                    id=str(uuid.uuid4()),
                    target_id=thread_id,
                    target_type="thread",
                    summary=summary,
                    head_node_id=last_node_id,
                )
                db.add(new_state)
                logger.info(f"Created new State for thread {thread_id}")

            db.commit()
            return True

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Failed to update local state for thread {thread_id}: {e}")
        return False


async def summarize_dirty_thread(thread_info: DirtyThreadInfo) -> bool:
    """
    Request summarization for a dirty thread and update local state.

    Args:
        thread_info: Info about the dirty thread

    Returns:
        True if summarization succeeded, False otherwise
    """
    if not state.http_client or not state.sync_token:
        logger.warning(f"Cannot summarize thread {thread_info.thread_id}: no HTTP client or token")
        return False

    try:
        # Get current summary from local State (for incremental update)
        current_summary = None
        try:
            from vlt.db import SessionLocal
            from vlt.core.models import State

            db = SessionLocal()
            try:
                current_state = db.scalars(
                    select(State)
                    .where(State.target_id == thread_info.thread_id)
                    .where(State.target_type == "thread")
                ).first()
                if current_state:
                    current_summary = current_state.summary
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"Could not get current summary: {e}")

        # Request summarization from backend
        logger.info(f"Requesting summarization for thread {thread_info.thread_id}")
        response = await state.http_client.post(
            f"{state.vault_url}/api/threads/{thread_info.thread_id}/summarize",
            json={"current_summary": current_summary} if current_summary else {},
            timeout=60.0,  # Summarization can take a while
        )
        response.raise_for_status()
        data = response.json()

        if data.get("success", True) and data.get("summary"):
            # Update local State table
            success = await update_local_state(
                thread_id=thread_info.thread_id,
                summary=data["summary"],
            )
            if success:
                logger.info(f"Successfully summarized thread {thread_info.thread_id}")
            return success
        else:
            error = data.get("error", "Unknown error")
            logger.warning(f"Server returned error for thread {thread_info.thread_id}: {error}")
            return False

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error summarizing thread {thread_info.thread_id}: {e.response.status_code}")
        return False
    except Exception as e:
        logger.error(f"Error summarizing thread {thread_info.thread_id}: {e}")
        return False


async def process_dirty_threads():
    """
    Background task that checks for dirty threads and triggers summarization.

    Implements debouncing: only summarizes a thread after SUMMARIZE_DELAY_SECONDS
    have passed since the last push to that thread.
    """
    while not state._shutdown_event.is_set():
        try:
            async with state._summarize_lock:
                now = time.time()
                threads_to_summarize: List[DirtyThreadInfo] = []

                # Find threads ready for summarization
                for thread_id, info in list(state.dirty_threads.items()):
                    if now - info.last_push_time >= SUMMARIZE_DELAY_SECONDS:
                        threads_to_summarize.append(info)

                # Process each ready thread
                for info in threads_to_summarize:
                    # Remove from dirty list before attempting (prevents duplicate processing)
                    state.dirty_threads.pop(info.thread_id, None)

                    success = await summarize_dirty_thread(info)

                    if not success and info.retry_count < 3:
                        # Re-add to dirty list for retry (with incremented count)
                        state.dirty_threads[info.thread_id] = DirtyThreadInfo(
                            thread_id=info.thread_id,
                            project_id=info.project_id,
                            last_push_time=time.time(),  # Reset timer for retry
                            retry_count=info.retry_count + 1,
                        )
                        logger.info(f"Queued thread {info.thread_id} for retry ({info.retry_count + 1}/3)")
                    elif not success:
                        logger.warning(f"Giving up on summarizing thread {info.thread_id} after 3 retries")

        except Exception as e:
            logger.error(f"Error in dirty thread processor: {e}")

        # Wait before next check
        try:
            await asyncio.wait_for(
                state._shutdown_event.wait(),
                timeout=SUMMARIZE_CHECK_INTERVAL_SECONDS,
            )
            break  # Shutdown requested
        except asyncio.TimeoutError:
            continue  # Continue processing


# =============================================================================
# CodeRAG Background Indexing (T024-T029)
# =============================================================================

# How often to check for pending CodeRAG indexing jobs
CODERAG_JOB_CHECK_INTERVAL_SECONDS = 10


def _get_next_pending_job():
    """Get the next pending CodeRAG indexing job.

    Finds job with status=PENDING, ordered by priority DESC, created_at ASC.
    This ensures high-priority jobs are processed first, and among same
    priority, older jobs are processed first (FIFO within priority level).

    Returns:
        CodeRAGIndexJob or None if no pending jobs
    """
    from sqlalchemy.orm import Session
    from vlt.db import engine
    from vlt.core.models import CodeRAGIndexJob, JobStatus

    with Session(engine) as session:
        job = session.scalar(
            select(CodeRAGIndexJob)
            .where(CodeRAGIndexJob.status == JobStatus.PENDING)
            .order_by(
                CodeRAGIndexJob.priority.desc(),
                CodeRAGIndexJob.created_at.asc()
            )
            .limit(1)
        )
        if job:
            # Detach from session for use outside
            session.expunge(job)
        return job


def _check_job_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled.

    This is called during indexing to support job cancellation (T062).
    The indexer's progress callback checks this flag between file processing.

    Args:
        job_id: Job identifier to check

    Returns:
        True if job status is CANCELLED, False otherwise
    """
    from sqlalchemy.orm import Session
    from vlt.db import engine
    from vlt.core.models import CodeRAGIndexJob, JobStatus

    with Session(engine) as session:
        job = session.get(CodeRAGIndexJob, job_id)
        if job and job.status == JobStatus.CANCELLED:
            return True
        return False


class JobCancelledException(Exception):
    """Raised when a job is cancelled during indexing."""
    pass


async def _run_indexing_job(job):
    """Execute a single CodeRAG indexing job with progress updates.

    This function:
    - Updates job status to RUNNING when processing starts (T027)
    - Runs the indexer in a thread pool via asyncio.to_thread()
    - Updates progress via callback during indexing (T029)
    - Checks for cancellation flag during indexing (T062)
    - Updates job status to COMPLETED, FAILED, or CANCELLED when done (T028)

    Args:
        job: CodeRAGIndexJob instance to process
    """
    from pathlib import Path
    from sqlalchemy.orm import Session
    from vlt.db import engine
    from vlt.core.models import CodeRAGIndexJob, JobStatus
    from vlt.core.coderag.indexer import CodeRAGIndexer

    job_id = job.id
    logger.info(f"Starting CodeRAG indexing job {job_id} for project {job.project_id}")

    # T027: Update job status to RUNNING
    with Session(engine) as session:
        j = session.get(CodeRAGIndexJob, job_id)
        if j:
            j.status = JobStatus.RUNNING
            j.started_at = datetime.now(timezone.utc)
            session.commit()
        else:
            logger.error(f"Job {job_id} not found in database")
            return

    # Temporarily set the embedding API key for this job if stored with the job.
    # This allows the daemon (which may have been started without the key) to
    # generate embeddings using the user's key passed at job-creation time.
    _prev_api_key = os.environ.get("VLT_OPENROUTER_API_KEY")
    if job.embedding_api_key:
        os.environ["VLT_OPENROUTER_API_KEY"] = job.embedding_api_key
        logger.debug(f"Set VLT_OPENROUTER_API_KEY from job {job_id} for embedding generation")

    try:
        # Create indexer
        target_path = Path(job.target_path)
        indexer = CodeRAGIndexer(target_path, job.project_id)

        # T029 + T062: Progress callback with cancellation check
        def on_progress(files_done: int, files_total: int, chunks: int):
            """Progress callback invoked by indexer.

            Also checks for cancellation flag (T062) - if job is cancelled,
            raises JobCancelledException to break out of indexing loop.
            """
            # Check cancellation flag (T062)
            if _check_job_cancelled(job_id):
                logger.info(f"Job {job_id} cancelled, aborting indexing")
                raise JobCancelledException(f"Job {job_id} was cancelled by user")

            # Update progress
            with Session(engine) as session:
                j = session.get(CodeRAGIndexJob, job_id)
                if j:
                    j.files_processed = files_done
                    j.files_total = files_total
                    j.chunks_created = chunks
                    # Calculate progress percent (avoid division by zero)
                    if files_total > 0:
                        j.progress_percent = int((files_done / files_total) * 100)
                    else:
                        j.progress_percent = 0
                    session.commit()

        # Run indexer in thread pool (it's synchronous)
        stats = await asyncio.to_thread(
            indexer.index_full,
            force=job.force,
            progress_callback=on_progress
        )

        # T028: Update job status to COMPLETED
        with Session(engine) as session:
            j = session.get(CodeRAGIndexJob, job_id)
            if j:
                j.status = JobStatus.COMPLETED
                j.completed_at = datetime.now(timezone.utc)
                j.files_processed = stats.files_indexed
                j.files_total = stats.files_discovered
                j.chunks_created = stats.chunks_created
                j.progress_percent = 100
                session.commit()

        logger.info(
            f"CodeRAG indexing job {job_id} completed: "
            f"{stats.files_indexed} files, {stats.chunks_created} chunks, "
            f"{stats.duration_seconds:.2f}s"
        )

    except JobCancelledException:
        # T062: Job was cancelled - update status and exit cleanly
        logger.info(f"CodeRAG indexing job {job_id} was cancelled")
        with Session(engine) as session:
            j = session.get(CodeRAGIndexJob, job_id)
            if j:
                j.status = JobStatus.CANCELLED
                j.completed_at = datetime.now(timezone.utc)
                j.error_message = "Cancelled by user"
                session.commit()

    except OSError as e:
        # T065: Handle disk space and other OS errors with clear messages
        error_msg = str(e)
        if "No space left on device" in error_msg or e.errno == 28:  # ENOSPC
            error_msg = (
                "Disk space exhausted during indexing. "
                "Free up disk space and retry with: vlt coderag init --force"
            )
            logger.error(f"CodeRAG indexing job {job_id} failed: disk space exhausted")
        else:
            logger.error(f"CodeRAG indexing job {job_id} failed with OS error: {e}")

        with Session(engine) as session:
            j = session.get(CodeRAGIndexJob, job_id)
            if j:
                j.status = JobStatus.FAILED
                j.completed_at = datetime.now(timezone.utc)
                j.error_message = error_msg
                session.commit()

    except Exception as e:
        # T028 + T066: Update job status to FAILED with recovery suggestions
        error_msg = str(e)
        logger.error(f"CodeRAG indexing job {job_id} failed: {e}")

        # T066: Add recovery suggestions for common errors
        if "database is locked" in error_msg.lower():
            error_msg = (
                f"{error_msg}. Recovery: Close other vlt processes and retry."
            )
        elif "permission denied" in error_msg.lower():
            error_msg = (
                f"{error_msg}. Recovery: Check file permissions for the target directory."
            )

        with Session(engine) as session:
            j = session.get(CodeRAGIndexJob, job_id)
            if j:
                j.status = JobStatus.FAILED
                j.completed_at = datetime.now(timezone.utc)
                j.error_message = error_msg
                session.commit()

    finally:
        # Restore the previous API key state (remove or revert to prior value)
        if job.embedding_api_key:
            if _prev_api_key is None:
                os.environ.pop("VLT_OPENROUTER_API_KEY", None)
            else:
                os.environ["VLT_OPENROUTER_API_KEY"] = _prev_api_key


async def process_coderag_jobs():
    """Background task to process pending CodeRAG indexing jobs.

    This task runs continuously in the daemon, checking for pending jobs
    at regular intervals and processing them one at a time.
    """
    logger.info("CodeRAG job processor started")

    while not state._shutdown_event.is_set():
        try:
            # Check for pending job
            job = _get_next_pending_job()
            if job:
                await _run_indexing_job(job)
        except Exception as e:
            logger.error(f"Error in CodeRAG job processor: {e}")

        # Wait before next check (or until shutdown)
        try:
            await asyncio.wait_for(
                state._shutdown_event.wait(),
                timeout=CODERAG_JOB_CHECK_INTERVAL_SECONDS
            )
            break  # Shutdown requested
        except asyncio.TimeoutError:
            continue  # Continue processing

    logger.info("CodeRAG job processor stopped")


# =============================================================================
# Session Relay — ~/.claude/history.jsonl Discovery Watcher
# =============================================================================

async def watch_claude_history():
    """
    Watches ~/.claude/history.jsonl for new session entries.

    Discovers Claude Code sessions by project cwd without requiring the relay
    command. Runs in a poll loop every 5 seconds.  Sessions discovered here
    get source="discovery" and only provide status metadata — no PTY stream.
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession

    history_path = Path.home() / ".claude" / "history.jsonl"

    if not history_path.exists():
        logger.debug("~/.claude/history.jsonl not found, history watcher idle")
        return

    seen_sessions: set = set()

    while not state._shutdown_event.is_set():
        try:
            with open(history_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        sid = entry.get("sessionId")
                        if not sid or sid in seen_sessions:
                            continue
                        seen_sessions.add(sid)
                        cwd = (entry.get("cwd")
                               or entry.get("projectPath")
                               or entry.get("project")
                               or "")
                        with Session(engine) as db:
                            existing = db.get(AgentSession, sid)
                            if not existing:
                                db.add(AgentSession(
                                    id=sid,
                                    cwd=cwd,
                                    name=Path(cwd).name if cwd else "unknown",
                                    status="idle",
                                    source="discovery",
                                ))
                                db.commit()
                                logger.debug(f"Discovered session {sid} from history (cwd={cwd})")
                            elif cwd and not existing.cwd:
                                # Backfill cwd for sessions stored before this fix
                                existing.cwd = cwd
                                existing.name = Path(cwd).name or existing.name
                                db.commit()
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug(f"History watcher error: {e}")

        try:
            await asyncio.wait_for(
                state._shutdown_event.wait(),
                timeout=5.0,
            )
            break  # Shutdown requested
        except asyncio.TimeoutError:
            continue  # Keep polling


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage daemon lifecycle."""
    logger.info("VLT Daemon starting...")

    # Initialize state
    state.start_time = datetime.now(timezone.utc)
    state.settings = Settings()
    state.sync_client = ThreadSyncClient()

    # Ensure all DB tables exist (including new agent_sessions)
    try:
        from vlt.core.migrations import init_db
        init_db()
        logger.info("Database schema initialized")
    except Exception as e:
        logger.warning(f"DB init warning (may be ok): {e}")

    # Create persistent HTTP client for backend communication
    state.http_client = httpx.AsyncClient(
        timeout=30.0,
        headers={
            "Authorization": f"Bearer {state.sync_token}" if state.sync_token else "",
            "Content-Type": "application/json",
        }
    )

    # Check initial backend connection
    await check_backend_connection()

    # Start background queue processor
    queue_task = asyncio.create_task(process_sync_queue())

    # Start background dirty thread processor (lazy summarization)
    summarize_task = asyncio.create_task(process_dirty_threads())

    # Start background CodeRAG indexing job processor (T026)
    coderag_task = asyncio.create_task(process_coderag_jobs())

    # Start background ~/.claude/history.jsonl session discovery watcher
    history_task = asyncio.create_task(watch_claude_history())

    logger.info(f"VLT Daemon started (backend: {state.vault_url}, connected: {state.backend_connected})")

    yield

    # Shutdown
    logger.info("VLT Daemon shutting down...")
    state._shutdown_event.set()

    # Wait for background tasks to finish
    for task in [queue_task, summarize_task, coderag_task, history_task]:
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()

    # Close HTTP client
    if state.http_client:
        await state.http_client.aclose()

    logger.info("VLT Daemon stopped")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="VLT Daemon",
    description="Background sync service for vlt-cli",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    queue_status = state.sync_client.get_queue_status() if state.sync_client else {"pending": 0}

    return HealthResponse(
        status="healthy",
        uptime_seconds=state.uptime_seconds,
        backend_url=state.vault_url,
        backend_connected=state.backend_connected,
        queue_size=queue_status.get("pending", 0),
        dirty_threads=len(state.dirty_threads),
    )


@app.post("/sync/enqueue", response_model=EnqueueResponse)
async def enqueue_sync(request: EnqueueRequest, background_tasks: BackgroundTasks):
    """
    Queue a sync entry for background processing.

    This endpoint is designed to be fast - it queues the entry and returns immediately.
    The actual sync happens in the background.

    After a successful sync, the thread is marked as "dirty" for lazy summarization.
    The daemon will automatically trigger summarization after SUMMARIZE_DELAY_SECONDS
    of inactivity on the thread (debouncing).
    """
    if not state.sync_client:
        raise HTTPException(status_code=503, detail="Sync client not initialized")

    synced = False

    # Try immediate sync first if backend is connected
    if state.backend_connected and state.sync_token:
        try:
            await state.sync_client.sync_entries(
                thread_id=request.thread_id,
                project_id=request.project_id,
                name=request.name,
                entries=[request.entry],
            )
            synced = True
        except Exception as e:
            logger.warning(f"Immediate sync failed, queuing: {e}")

    if synced:
        # Mark thread as dirty for lazy summarization
        # If already dirty, this updates the last_push_time (debouncing)
        state.dirty_threads[request.thread_id] = DirtyThreadInfo(
            thread_id=request.thread_id,
            project_id=request.project_id,
            last_push_time=time.time(),
            retry_count=0,  # Reset retry count on new push
        )
        logger.debug(f"Marked thread {request.thread_id} as dirty for summarization")

        queue_status = state.sync_client.get_queue_status()
        return EnqueueResponse(
            queued=False,
            message="Synced immediately",
            queue_size=queue_status.get("pending", 0),
        )

    # Queue for later processing
    state.sync_client.queue_entry(
        thread_id=request.thread_id,
        project_id=request.project_id,
        name=request.name,
        entry=request.entry,
        error="Queued via daemon",
    )

    queue_status = state.sync_client.get_queue_status()
    return EnqueueResponse(
        queued=True,
        message="Queued for background sync",
        queue_size=queue_status.get("pending", 0),
    )


@app.get("/sync/status", response_model=SyncStatusResponse)
async def get_sync_status():
    """Get current sync queue status."""
    if not state.sync_client:
        raise HTTPException(status_code=503, detail="Sync client not initialized")

    queue_status = state.sync_client.get_queue_status()

    return SyncStatusResponse(
        pending=queue_status.get("pending", 0),
        items=queue_status.get("items", []),
        daemon_uptime_seconds=state.uptime_seconds,
        backend_connected=state.backend_connected,
    )


@app.post("/sync/retry", response_model=RetryResponse)
async def retry_sync():
    """Retry all queued sync entries."""
    if not state.sync_client:
        raise HTTPException(status_code=503, detail="Sync client not initialized")

    result = await state.sync_client.retry_queue()

    return RetryResponse(
        success=result.get("success", 0),
        failed=result.get("failed", 0),
        skipped=result.get("skipped", 0),
    )


@app.get("/summarize/pending", response_model=DirtyThreadsStatusResponse)
async def get_dirty_threads_status():
    """
    Get status of threads pending summarization.

    Shows which threads are marked as dirty and waiting for summarization,
    along with timing information.
    """
    now = time.time()
    threads_info = []

    for thread_id, info in state.dirty_threads.items():
        seconds_since_push = now - info.last_push_time
        seconds_until_summarize = max(0, SUMMARIZE_DELAY_SECONDS - seconds_since_push)

        threads_info.append({
            "thread_id": info.thread_id,
            "project_id": info.project_id,
            "seconds_since_push": round(seconds_since_push, 1),
            "seconds_until_summarize": round(seconds_until_summarize, 1),
            "retry_count": info.retry_count,
        })

    return DirtyThreadsStatusResponse(
        count=len(state.dirty_threads),
        threads=threads_info,
        summarize_delay_seconds=SUMMARIZE_DELAY_SECONDS,
    )


@app.post("/summarize/{thread_id}", response_model=SummarizeResponse)
async def request_summarize(thread_id: str, request: SummarizeRequest):
    """
    Request server-side summarization for a thread.

    This proxies the request to the backend server.
    """
    if not state.http_client:
        raise HTTPException(status_code=503, detail="HTTP client not initialized")

    if not state.sync_token:
        return SummarizeResponse(
            thread_id=thread_id,
            summary="",
            success=False,
            error="No sync token configured",
        )

    try:
        response = await state.http_client.post(
            f"{state.vault_url}/api/threads/{thread_id}/summarize",
            json={"current_summary": request.current_summary} if request.current_summary else {},
        )
        response.raise_for_status()
        data = response.json()

        return SummarizeResponse(
            thread_id=thread_id,
            summary=data.get("summary", ""),
            model=data.get("model"),
            tokens_used=data.get("tokens_used", 0),
            success=data.get("success", True),
            error=data.get("error"),
        )
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text[:200] if e.response.text else str(e)
        return SummarizeResponse(
            thread_id=thread_id,
            summary="",
            success=False,
            error=f"HTTP {e.response.status_code}: {error_detail}",
        )
    except Exception as e:
        return SummarizeResponse(
            thread_id=thread_id,
            summary="",
            success=False,
            error=str(e),
        )


# =============================================================================
# Session Relay Endpoints
# =============================================================================

@app.post("/api/sessions/register")
async def register_session(request: Request):
    """
    Register a new Claude Code agent session.

    Called by `vlt relay` when it wraps a claude process in a PTY.
    Creates an in-memory stream queue pair so the WebSocket endpoint
    can bridge terminal output to web UI clients.

    Body: {session_id, cwd, project_id, pid, args}
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession

    payload = await request.json()
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    cwd = payload.get("cwd", "")
    project_id = payload.get("project_id")
    pid = payload.get("pid")
    args = payload.get("args", [])

    # Create in-memory queues for PTY relay
    if session_id not in _session_streams:
        _session_streams[session_id] = asyncio.Queue(maxsize=4096)
    if session_id not in _session_inject_queues:
        _session_inject_queues[session_id] = asyncio.Queue(maxsize=256)

    # Upsert into DB
    with Session(engine) as db:
        existing = db.get(AgentSession, session_id)
        if existing:
            existing.cwd = cwd
            existing.project_id = project_id
            existing.pid = pid
            existing.args = json.dumps(args)
            existing.status = "idle"
            existing.source = "relay"
            existing.last_activity = datetime.utcnow().isoformat()
        else:
            db.add(AgentSession(
                id=session_id,
                cwd=cwd,
                name=Path(cwd).name if cwd else "unknown",
                project_id=project_id,
                pid=pid,
                args=json.dumps(args),
                status="idle",
                source="relay",
            ))
        db.commit()

    logger.info(f"Session registered: {session_id} (pid={pid}, cwd={cwd})")
    return {"ok": True}


@app.post("/api/sessions/deregister")
async def deregister_session(request: Request):
    """
    Deregister a Claude Code agent session.

    Called by `vlt relay` when the claude process exits.
    Sends a None sentinel into the stream queue to close any active WebSockets,
    then removes the queues from memory.

    Body: {session_id}
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession

    payload = await request.json()
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    # Send close sentinel to any active WebSocket listeners
    if session_id in _session_streams:
        await _session_streams[session_id].put(None)
        del _session_streams[session_id]
    _session_inject_queues.pop(session_id, None)
    # Clean up scrollback (in-memory + disk) since session ended cleanly
    _session_scrollback.pop(session_id, None)
    try:
        _scrollback_path(session_id).unlink(missing_ok=True)
    except Exception:
        pass

    # Mark as dead in DB
    with Session(engine) as db:
        existing = db.get(AgentSession, session_id)
        if existing:
            existing.status = "dead"
            existing.last_activity = datetime.utcnow().isoformat()
            db.commit()

    logger.info(f"Session deregistered: {session_id}")
    return {"ok": True}


@app.get("/api/sessions")
async def list_sessions(
    project_id: Optional[str] = None,
    include_all: bool = False,
):
    """
    List Claude Code agent sessions.

    Query params:
        project_id: Filter to sessions belonging to this project.
        include_all: If true, include dead sessions too (for find dialog).

    Sessions with no project_id and a cwd are lazily backfilled via
    _infer_project_id on each list call.
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession

    with Session(engine) as db:
        # Step 1: Backfill project_id for any sessions that lack one.
        # This must happen BEFORE the filtered query so that newly-tagged
        # sessions are included in project-scoped results.
        untagged = db.scalars(
            select(AgentSession)
            .where(AgentSession.project_id.is_(None))
            .where(AgentSession.cwd.isnot(None))
        ).all()
        for s in untagged:
            inferred = _infer_project_id(s.cwd)
            if inferred:
                s.project_id = inferred
        if untagged:
            db.commit()

        # Step 2: Build the filtered query
        query = select(AgentSession)
        if not include_all:
            query = query.where(AgentSession.status != "dead")
        if project_id:
            query = query.where(AgentSession.project_id == project_id)
        query = query.order_by(AgentSession.last_activity.desc())
        sessions = db.scalars(query).all()

    result = []
    for s in sessions:
        result.append({
            "id": s.id,
            "project_id": s.project_id,
            "name": s.name,
            "cwd": s.cwd,
            "status": s.status,
            "model": s.model,
            "ctx_pct": s.ctx_pct,
            "pid": s.pid,
            "bypass_perms": s.bypass_perms,
            "source": s.source,
            "created_at": s.created_at,
            "last_activity": s.last_activity,
        })
    return result


@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, request: Request):
    """
    Update a session's metadata (e.g. assign it to a project).

    Body: {"project_id": "vlt-bridge"}
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession

    body = await request.json()
    with Session(engine) as db:
        session = db.get(AgentSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if "project_id" in body:
            session.project_id = body["project_id"]
        db.commit()
    return {"ok": True}


@app.post("/api/sessions/{session_id}/inject")
async def inject_to_session(session_id: str, request: Request):
    """
    Inject text into a running Claude Code PTY session.

    Puts the text onto the inject queue, which the `vlt relay` process
    polls and writes into the PTY master fd.

    Body: {text: str, press_enter: bool}
    """
    if session_id not in _session_inject_queues:
        raise HTTPException(status_code=404, detail="Session not found or not a relay session")

    payload = await request.json()
    text = payload.get("text", "")
    press_enter = payload.get("press_enter", False)

    if press_enter and not text.endswith("\n"):
        text = text + "\n"

    await _session_inject_queues[session_id].put(text.encode())
    logger.debug(f"Injected {len(text)} bytes into session {session_id}")
    return {"ok": True}


@app.websocket("/ws/relay/{session_id}")
async def relay_ws(websocket: WebSocket, session_id: str):
    """
    PTY relay → daemon WebSocket bridge.

    The `vlt session-relay` process connects here to stream terminal output.

    Relay→Daemon: binary PTY chunks  → put in _session_streams[session_id]
    Daemon→Relay: JSON text messages ← drained from _session_inject_queues[session_id]
      {"type": "inject", "data": "text"}
      {"type": "resize", "cols": N, "rows": N}
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession

    await websocket.accept()

    # Verify session is known (registered via HTTP first, or already in DB)
    with Session(engine) as db:
        existing = db.get(AgentSession, session_id)
    if not existing:
        await websocket.close(1008, "Session not registered")
        return

    # Create queues on-demand — survives daemon restarts
    if session_id not in _session_streams:
        _session_streams[session_id] = asyncio.Queue(maxsize=4096)
    if session_id not in _session_inject_queues:
        _session_inject_queues[session_id] = asyncio.Queue(maxsize=256)

    stream_q = _session_streams[session_id]
    inject_q = _session_inject_queues[session_id]

    async def receive_pty():
        """Receive binary PTY chunks from relay → fill scrollback + stream queue."""
        if session_id not in _session_scrollback:
            _session_scrollback[session_id] = bytearray()
        buf = _session_scrollback[session_id]
        try:
            async for chunk in websocket.iter_bytes():
                # Maintain rolling scrollback buffer
                buf.extend(chunk)
                if len(buf) > _MAX_SCROLLBACK:
                    del buf[:len(buf) - _MAX_SCROLLBACK]
                # Debounced disk persist so content survives daemon restarts
                _schedule_scrollback_persist(session_id, buf)
                try:
                    stream_q.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"Relay WS receive_pty error for {session_id}: {e}")

    async def forward_inject():
        """Forward inject/control messages from inject queue → relay as JSON text.

        Items in inject_q can be:
          - raw bytes (keyboard input) → wrapped as {"type": "inject", "data": "..."}
          - JSON bytes (resize/control) → forwarded as-is
        """
        while True:
            try:
                data = await inject_q.get()
                text = data.decode(errors="replace")
                # Already JSON (resize, etc.) — forward as-is; else wrap as inject
                if text.startswith("{"):
                    await websocket.send_text(text)
                else:
                    await websocket.send_text(json.dumps({"type": "inject", "data": text}))
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.debug(f"Relay WS forward_inject error for {session_id}: {e}")
                break

    logger.info(f"Relay WS connected: {session_id}")
    await asyncio.gather(receive_pty(), forward_inject(), return_exceptions=True)
    logger.info(f"Relay WS disconnected: {session_id}")


@app.websocket("/ws/sessions/{session_id}")
async def session_ws(websocket: WebSocket, session_id: str):
    """
    Browser → daemon WebSocket bridge for live terminal viewing.

    Server→Client: raw bytes (terminal output chunks from PTY via _session_streams)
    Client→Server JSON:
      {"type": "inject", "data": "text to send\\n"}
      {"type": "resize", "cols": N, "rows": N}
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession

    await websocket.accept()

    if session_id not in _session_streams:
        # On daemon restart, queues are wiped — recreate on-demand for DB sessions
        with Session(engine) as db:
            existing = db.get(AgentSession, session_id)
        if not existing:
            await websocket.close(1008, "Session not found")
            return
        _session_streams[session_id] = asyncio.Queue(maxsize=4096)
        if session_id not in _session_inject_queues:
            _session_inject_queues[session_id] = asyncio.Queue(maxsize=256)

    queue = _session_streams[session_id]

    # Send scrollback buffer to late-joining browser clients.
    # If in-memory buffer is empty (e.g. after daemon restart), fall back to disk.
    scrollback = _session_scrollback.get(session_id)
    if not scrollback:
        sb_file = _scrollback_path(session_id)
        if sb_file.exists():
            try:
                disk_data = await asyncio.to_thread(sb_file.read_bytes)
                if disk_data:
                    _session_scrollback[session_id] = bytearray(disk_data)
                    scrollback = disk_data
            except Exception as e:
                logger.debug(f"Failed to load scrollback from disk for {session_id}: {e}")
    if scrollback:
        try:
            await websocket.send_bytes(bytes(scrollback))
        except Exception:
            pass

    async def send_loop():
        """Forward PTY output chunks from the queue to the WebSocket."""
        while True:
            try:
                chunk = await queue.get()
                if chunk is None:  # Sentinel: session ended
                    await websocket.close(1000, "Session ended")
                    # Put sentinel back so other listeners also get it
                    await queue.put(None)
                    break
                await websocket.send_bytes(chunk)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.debug(f"WS send_loop error for {session_id}: {e}")
                break

    async def recv_loop():
        """Receive inject/resize control messages from the browser."""
        try:
            async for msg in websocket.iter_text():
                try:
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    if msg_type == "inject" and session_id in _session_inject_queues:
                        text = data.get("data", "")
                        await _session_inject_queues[session_id].put(text.encode())
                    elif msg_type == "resize":
                        cols = data.get("cols", 80)
                        rows = data.get("rows", 24)
                        logger.debug(f"Resize request for {session_id}: {cols}x{rows}")
                        # Forward resize to relay so PTY dimensions stay in sync
                        if session_id in _session_inject_queues:
                            resize_cmd = json.dumps({"type": "resize", "cols": cols, "rows": rows}).encode()
                            try:
                                _session_inject_queues[session_id].put_nowait(resize_cmd)
                            except asyncio.QueueFull:
                                pass
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"WS recv_loop error for {session_id}: {e}")

    await asyncio.gather(send_loop(), recv_loop(), return_exceptions=True)


# =============================================================================
# Hook Receiver Endpoint
# =============================================================================

# Maps Claude Code hook event names to internal session status strings
_HOOK_STATUS_MAP = {
    "SessionStart": "idle",
    "UserPromptSubmit": "thinking",
    "PreToolUse": "executing",
    "PostToolUse": "idle",
    "Stop": "idle",
    "SessionEnd": "dead",
}


@app.post("/api/hooks")
async def claude_hook(request: Request):
    """
    Receives Claude Code lifecycle hook payloads.

    Claude Code fires hooks at key lifecycle events. This endpoint updates
    the local AgentSession record so the UI can show live status without
    needing the full PTY relay.

    Common fields: session_id, hook_event_name, cwd, transcript_path
    Returns {} for most events. Returns {"additionalContext": "..."} for
    UserPromptSubmit if context injection is desired (future feature).

    Registered in ~/.claude/settings.json hooks config:
      "hooks": {"UserPromptSubmit": [{"matcher": "", "hooks": [{"type": "command",
        "command": "curl -s -X POST http://localhost:8765/api/hooks ..."}]}]}
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession

    payload = await request.json()
    event = payload.get("hook_event_name", "")
    session_id = payload.get("session_id")
    cwd = payload.get("cwd", "")
    transcript_path = payload.get("transcript_path")
    model = payload.get("model")

    if not session_id:
        return {}

    new_status = _HOOK_STATUS_MAP.get(event)
    now = datetime.utcnow().isoformat()

    # Check if this session belongs to a managed session — don't mark as "dead"
    # since managed sessions persist between individual claude -p invocations
    is_managed = any(
        m["session_id"] == session_id for m in _managed_sessions.values()
    )
    if is_managed and new_status == "dead":
        new_status = "idle"

    with Session(engine) as db:
        session = db.get(AgentSession, session_id)
        if session:
            if new_status:
                session.status = new_status
            session.last_activity = now
            if transcript_path:
                session.transcript_path = transcript_path
            if model and not session.model:
                session.model = model
            if is_managed:
                session.source = "managed"
            # Backfill project_id if missing
            if not session.project_id and session.cwd:
                inferred = _infer_project_id(session.cwd)
                if inferred:
                    session.project_id = inferred
            db.commit()
        else:
            # Session discovered for the first time via a hook (no relay started)
            inferred_pid = _infer_project_id(cwd) if cwd else None
            db.add(AgentSession(
                id=session_id,
                cwd=cwd,
                name=Path(cwd).name if cwd else "unknown",
                project_id=inferred_pid,
                status=new_status or "idle",
                source="managed" if is_managed else "hook",
                transcript_path=transcript_path,
                model=model,
            ))
            db.commit()

    logger.debug(f"Hook received: event={event}, session={session_id}, status={new_status}")

    # Append to in-memory ring buffer for the Events panel
    _hook_events.append({
        "id": str(uuid.uuid4()),
        "ts": now,
        "event": event,
        "session_id": session_id,
        "cwd": cwd,
        "status": new_status,
    })
    if len(_hook_events) > _MAX_HOOK_EVENTS:
        del _hook_events[:-_MAX_HOOK_EVENTS]

    # Push new JSONL entries to any live-streaming browser clients
    if transcript_path and session_id in _live_stream_queues:
        asyncio.create_task(_push_new_jsonl_entries(session_id, transcript_path))

    # Notify live stream clients of status change
    _push_status_to_live(session_id, new_status or "idle", event)

    # Context injection: return queued messages on UserPromptSubmit
    if event == "UserPromptSubmit" and session_id in _injection_queues:
        pending = _injection_queues.pop(session_id, [])
        if pending:
            context = "\n\n".join(pending)
            return {"additionalContext": context}

    return {}


@app.get("/api/hooks/recent")
async def list_hook_events(limit: int = 100):
    """Return the most recent Claude Code hook events (in-memory, resets on restart)."""
    return {"events": _hook_events[-limit:]}


# =============================================================================
# Session Management — Dismiss / force-cull stale sessions
# =============================================================================

@app.delete("/api/sessions/{session_id}")
async def dismiss_session(session_id: str):
    """Mark a session as dead so it disappears from active listings."""
    from vlt.db import engine
    from vlt.core.models import AgentSession
    from sqlmodel import Session as DBSession
    with DBSession(engine) as db:
        session = db.get(AgentSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session.status = "dead"
        db.add(session)
        db.commit()
    # Also clean up in-memory relay state and persisted scrollback
    _session_streams.pop(session_id, None)
    _session_inject_queues.pop(session_id, None)
    _session_scrollback.pop(session_id, None)
    try:
        _scrollback_path(session_id).unlink(missing_ok=True)
    except Exception:
        pass
    return {"ok": True}


# =============================================================================
# Transcript API — Read/Write JSONL session transcripts
# =============================================================================

@app.get("/api/sessions/{session_id}/transcript")
async def get_transcript(session_id: str, types: str = "user,assistant,system"):
    """
    Read a Claude Code session transcript (JSONL file).

    Returns parsed entries filtered by type. The JSONL path is derived from
    the session's cwd: ~/.claude/projects/{slug}/{session_id}.jsonl

    Query params:
        types: comma-separated entry types to include (default: user,assistant,system)
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession

    # Look up session to get cwd / transcript_path
    with Session(engine) as db:
        session = db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    jsonl_path = Path(_resolve_transcript_path(session))

    if not jsonl_path.exists():
        raise HTTPException(status_code=404, detail=f"Transcript not found: {jsonl_path}")

    type_filter = set(t.strip() for t in types.split(",") if t.strip())

    entries = []
    total_lines = 0
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_type = obj.get("type", "")
                if type_filter and entry_type not in type_filter:
                    continue
                entries.append({
                    "lineIndex": line_idx,
                    "type": entry_type,
                    "message": obj.get("message"),
                    "timestamp": obj.get("timestamp"),
                    "uuid": obj.get("uuid"),
                    "sessionId": obj.get("sessionId"),
                    "cwd": obj.get("cwd"),
                    "raw": obj,
                })
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to read transcript: {e}")

    return {
        "path": str(jsonl_path),
        "entries": entries,
        "total_lines": total_lines,
    }


@app.post("/api/sessions/{session_id}/transcript")
async def save_transcript(session_id: str, request: Request):
    """
    Write back a modified transcript (full JSONL replacement).

    Body: { entries: [ {...raw json objects...} ] }
    Each entry is written as one JSON line.
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession

    with Session(engine) as db:
        session = db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    jsonl_path = Path(_resolve_transcript_path(session))

    payload = await request.json()
    raw_entries = payload.get("entries", [])

    if not isinstance(raw_entries, list):
        raise HTTPException(status_code=400, detail="entries must be an array")

    try:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in raw_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write transcript: {e}")

    return {"ok": True, "lines_written": len(raw_entries)}


# =============================================================================
# Live Session Streaming + Context Injection
# =============================================================================

# Per-session live streaming: browser WebSocket clients receive JSONL entries
_live_stream_queues: Dict[str, Set[asyncio.Queue]] = {}

# Per-session byte offset for incremental JSONL reads
_live_file_offsets: Dict[str, int] = {}

# Per-session context injection queue: messages from browser → Claude via hooks
_injection_queues: Dict[str, list] = {}


def _find_managed_by_session_id(sid: str) -> Optional[Dict]:
    """Look up a managed session entry by session_id (not CWD)."""
    for managed in _managed_sessions.values():
        if managed.get("session_id") == sid:
            return managed
    return None


def _push_status_to_live(session_id: str, status: str, event: str) -> None:
    """Push a status update message to all live stream clients for a session."""
    clients = _live_stream_queues.get(session_id)
    if not clients:
        return
    msg = {"type": "status", "status": status, "event": event}
    for q in list(clients):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


async def _push_new_jsonl_entries(session_id: str, transcript_path: str) -> None:
    """Read new JSONL entries since last offset and push to live stream clients."""
    clients = _live_stream_queues.get(session_id)
    if not clients:
        return

    path = Path(transcript_path)
    if not path.exists():
        return

    offset = _live_file_offsets.get(session_id, 0)

    try:
        def _read_new():
            with open(path, "r", encoding="utf-8") as f:
                f.seek(offset)
                entries = []
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        entries.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue
                return entries, f.tell()

        new_entries, new_offset = await asyncio.to_thread(_read_new)
        _live_file_offsets[session_id] = new_offset
    except Exception:
        return

    _STREAM_TYPES = {"user", "assistant", "system"}
    for entry in new_entries:
        if entry.get("type") not in _STREAM_TYPES:
            continue
        msg = {"type": "entry", "entry": entry}
        for q in list(clients):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass


def _resolve_transcript_path(session) -> str:
    """Resolve JSONL transcript path from session (prefers stored, falls back to slug)."""
    if session.transcript_path:
        return session.transcript_path
    cwd = session.cwd or ""
    slug = cwd.replace("/", "-")
    return str(Path.home() / ".claude" / "projects" / slug / f"{session.id}.jsonl")


@app.websocket("/ws/session/{session_id}/live")
async def session_live_ws(websocket: WebSocket, session_id: str):
    """
    Live JSONL transcript streaming to browser clients.

    On connect: sends all existing entries as {"type": "initial", "entries": [...]}.
    Then streams new entries as {"type": "entry", "entry": {...}} triggered by hooks
    and a 1-second polling fallback.

    Also sends {"type": "status", "status": "thinking"|"idle"|...} on hook events.
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession

    await websocket.accept()

    with Session(engine) as db:
        session = db.get(AgentSession, session_id)
    if not session:
        await websocket.close(1008, "Session not found")
        return

    transcript_path = _resolve_transcript_path(session)

    # Register client queue
    client_queue: asyncio.Queue = asyncio.Queue(maxsize=2048)
    if session_id not in _live_stream_queues:
        _live_stream_queues[session_id] = set()
    _live_stream_queues[session_id].add(client_queue)

    # Only include conversation-relevant entry types in initial + streaming
    _LIVE_TYPES = {"user", "assistant", "system"}

    try:
        # Send initial batch (last 200 conversation entries, filtered)
        path = Path(transcript_path)
        initial_entries = []
        if path.exists():
            def _read_initial():
                entries = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            obj = json.loads(stripped)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("type") in _LIVE_TYPES:
                            entries.append(obj)
                    return entries[-200:], f.tell()

            initial_entries, offset = await asyncio.to_thread(_read_initial)
            _live_file_offsets[session_id] = offset

        await websocket.send_json({
            "type": "initial",
            "entries": initial_entries,
            "session": {
                "id": session.id,
                "cwd": session.cwd,
                "status": session.status,
                "model": session.model,
                "name": session.name,
            },
        })

        # Poll for new entries between hook events (1-second fallback)
        async def poll_loop():
            while True:
                await asyncio.sleep(1.0)
                await _push_new_jsonl_entries(session_id, transcript_path)

        poll_task = asyncio.create_task(poll_loop())

        # Dual loop: forward queued messages to browser + receive browser input
        async def send_loop():
            while True:
                try:
                    msg = await asyncio.wait_for(client_queue.get(), timeout=30.0)
                    await websocket.send_json(msg)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})

        async def recv_loop():
            """Route browser input based on session type:
            - relay → direct PTY write via inject queue
            - managed (web-spawned) → queue for --resume worker
            - hook (terminal CLI) → additionalContext via hook response
            """
            async for raw in websocket.iter_text():
                try:
                    msg = json.loads(raw)
                    msg_type = msg.get("type")
                    text = msg.get("text", "")
                    if not text or msg_type not in ("inject", "input"):
                        continue

                    # 1. Relay session — direct PTY stdin write
                    if session_id in _session_inject_queues:
                        await _session_inject_queues[session_id].put(text + "\n")
                        await websocket.send_json({
                            "type": "input_sent",
                            "text": text,
                        })

                    # 2. Web-spawned managed session — queue for --resume worker
                    elif (managed := _find_managed_by_session_id(session_id)) is not None:
                        managed["queue"].append({"text": text})
                        if not managed["processing"]:
                            asyncio.create_task(
                                _managed_message_worker(managed["cwd"])
                            )
                        await websocket.send_json({
                            "type": "input_sent",
                            "text": text,
                        })

                    # 3. Hook-discovered session (CLI in terminal) —
                    #    queue as additionalContext, delivered on next
                    #    UserPromptSubmit hook.
                    else:
                        _injection_queues.setdefault(session_id, []).append(text)
                        await websocket.send_json({
                            "type": "injected",
                            "text": text,
                            "queued": len(_injection_queues[session_id]),
                        })
                except (json.JSONDecodeError, Exception):
                    pass

        try:
            await asyncio.gather(send_loop(), recv_loop(), return_exceptions=True)
        finally:
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"Live stream error for {session_id}: {e}")
    finally:
        _live_stream_queues.get(session_id, set()).discard(client_queue)
        if not _live_stream_queues.get(session_id):
            _live_stream_queues.pop(session_id, None)
            _live_file_offsets.pop(session_id, None)


@app.post("/api/sessions/{session_id}/context")
async def inject_context(session_id: str, request: Request):
    """
    Queue a context message to be injected into the agent's next turn.

    When the agent's next UserPromptSubmit hook fires, the daemon returns
    the queued message as additionalContext, which Claude Code injects into
    the conversation. This enables remote interaction without a PTY relay.
    """
    body = await request.json()
    message = body.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="message required")

    if session_id not in _injection_queues:
        _injection_queues[session_id] = []
    _injection_queues[session_id].append(message)

    return {"ok": True, "queued": len(_injection_queues[session_id])}


# =============================================================================
# Daemon-Managed Sessions (Message Queue)
# =============================================================================
#
# Instead of maintaining a long-running interactive PTY (Claude's TUI doesn't
# accept raw PTY input reliably), we use a message-queue approach:
#
#   1. New session:      claude -p "msg" --session-id <uuid> --dangerously-skip-permissions
#   2. Resume existing:  claude -p "msg" --resume <session_id> --dangerously-skip-permissions
#
# The session_id is pre-assigned (uuid4) for new sessions so we know the JSONL
# file path before the process starts. For existing sessions (user clicked in UI),
# we use --resume to append to that conversation. Hooks fire normally.

# Per-cwd managed session state
_managed_sessions: Dict[str, dict] = {}  # cwd → {session_id, processing, queue}


async def _run_claude_message(
    cwd: str,
    message: str,
    session_id: Optional[str] = None,
    is_first: bool = True,
    model: Optional[str] = None,
) -> Optional[str]:
    """
    Run a single claude -p message in a PTY and wait for completion.

    For the first message: uses --session-id to pre-assign the session UUID.
    For subsequent messages: uses --resume to target that specific session.
    Returns the session_id used, or None on failure.
    """
    claude_bin = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    cmd = [claude_bin, "-p", message, "--dangerously-skip-permissions"]
    if session_id:
        if is_first:
            cmd.extend(["--session-id", session_id])
        else:
            cmd.extend(["--resume", session_id])
    if model:
        cmd.extend(["--model", model])

    # Clean env: remove Claude nesting detection vars
    _STRIP_VARS = {"CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"}
    spawn_env = {k: v for k, v in os.environ.items() if k not in _STRIP_VARS}
    spawn_env["TERM"] = "xterm-256color"

    master_fd, slave_fd = pty.openpty()
    winsize = struct.pack("HHHH", 50, 120, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    if not os.path.isdir(cwd):
        os.close(master_fd)
        os.close(slave_fd)
        logger.error(f"CWD does not exist: {cwd}")
        return None

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=cwd,
            env=spawn_env,
            preexec_fn=os.setsid,
        )
    except FileNotFoundError:
        os.close(master_fd)
        os.close(slave_fd)
        logger.error(f"'claude' not found at {claude_bin}")
        return None

    os.close(slave_fd)
    logger.info(f"Managed message PID={proc.pid} cwd={cwd} session={session_id} first={is_first}")

    # Drain PTY output in background thread
    loop = asyncio.get_event_loop()

    async def drain():
        while proc.poll() is None:
            try:
                await loop.run_in_executor(
                    None, lambda: os.read(master_fd, 16384)
                )
            except OSError:
                break
        try:
            os.close(master_fd)
        except OSError:
            pass

    drain_task = asyncio.create_task(drain())

    # Wait for process to complete (timeout 5 minutes)
    try:
        exit_code = await asyncio.wait_for(
            loop.run_in_executor(None, proc.wait),
            timeout=300.0,
        )
        logger.info(f"Managed message PID={proc.pid} exited code={exit_code}")
    except asyncio.TimeoutError:
        proc.kill()
        logger.warning(f"Managed message PID={proc.pid} timed out, killed")

    drain_task.cancel()
    try:
        await drain_task
    except (asyncio.CancelledError, Exception):
        pass

    return session_id


async def _managed_message_worker(cwd: str):
    """
    Process queued messages for a managed session one at a time.
    Each message spawns claude -p, waits for completion, then processes next.
    """
    managed = _managed_sessions.get(cwd)
    if not managed:
        return

    managed["processing"] = True

    # Pre-generate session_id if not already set
    if not managed["session_id"]:
        managed["session_id"] = str(uuid.uuid4())
        managed["is_new"] = True
        logger.info(f"Pre-assigned session_id={managed['session_id']} for cwd={cwd}")

    sid = managed["session_id"]
    is_new_session = managed.get("is_new", True)

    try:
        while managed["queue"]:
            msg = managed["queue"].pop(0)
            text = msg["text"]
            model = msg.get("model")

            _push_status_to_live(sid, "thinking", "ManagedMessage")

            await _run_claude_message(
                cwd=cwd,
                message=text,
                session_id=sid,
                is_first=is_new_session,  # True = --session-id, False = --resume
                model=model,
            )

            # After first message, always resume
            is_new_session = False
            managed["is_new"] = False
            _push_status_to_live(sid, "idle", "ManagedMessageDone")
    finally:
        managed["processing"] = False


@app.post("/api/sessions/spawn")
async def spawn_managed_session(request: Request):
    """
    Send a message to a daemon-managed Claude Code session.

    First message creates a new session. Subsequent messages use --resume
    to target the specific session by ID. Each message runs as a discrete
    process with hooks firing normally.
    """
    body = await request.json()
    cwd = body.get("cwd", os.path.expanduser("~"))
    message = body.get("prompt", body.get("message", "Hello! Ready for instructions."))
    model = body.get("model")

    cwd_path = str(Path(cwd).expanduser().resolve())
    if not Path(cwd_path).is_dir():
        raise HTTPException(status_code=400, detail=f"Invalid cwd: {cwd}")

    # Initialize managed session state for this cwd
    if cwd_path not in _managed_sessions:
        _managed_sessions[cwd_path] = {
            "session_id": str(uuid.uuid4()),  # Pre-assigned, used with --session-id
            "processing": False,
            "queue": [],
            "cwd": cwd_path,
            "is_new": True,  # brand new session → first msg uses --session-id
        }

    managed = _managed_sessions[cwd_path]
    managed["queue"].append({"text": message, "model": model})

    # Start worker if not already running
    if not managed["processing"]:
        asyncio.create_task(_managed_message_worker(cwd_path))

    return {
        "ok": True,
        "cwd": cwd_path,
        "queued": len(managed["queue"]),
        "session_id": managed.get("session_id"),
    }



# =============================================================================
# Entry Point
# =============================================================================

def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    profile_name: Optional[str] = None,
):
    """
    Run the daemon server.

    Args:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to listen on (default: 8765)
        profile_name: Profile to use for settings. If None, uses active profile.
    """
    import uvicorn

    # Set up profile before loading settings
    if profile_name:
        import os
        os.environ["VLT_PROFILE"] = profile_name

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting daemon for profile: {profile_name or 'default'}")

    # Handle signals for graceful shutdown
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        state._shutdown_event.set()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLT Daemon Server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile to use (default: active profile)",
    )

    args = parser.parse_args()
    run_server(host=args.host, port=args.port, profile_name=args.profile)
