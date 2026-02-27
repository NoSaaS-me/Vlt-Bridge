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
import logging
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import select

from vlt.config import Settings
from vlt.core.sync import ThreadSyncClient, SyncQueueItem

logger = logging.getLogger(__name__)


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

    logger.info(f"VLT Daemon started (backend: {state.vault_url}, connected: {state.backend_connected})")

    yield

    # Shutdown
    logger.info("VLT Daemon shutting down...")
    state._shutdown_event.set()

    # Wait for background tasks to finish
    for task in [queue_task, summarize_task, coderag_task]:
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
