"""Thread API endpoints - Sync and query threads from vlt-cli.

This module provides two types of endpoints:

1. **Database-backed endpoints** (T014-T018, T031): These work with the
   ThreadService which stores threads in SQLite for syncing and searching.

2. **CLI-backed endpoints** (T037-T039): These call the vlt CLI directly
   to create threads, push entries, and perform semantic search. These
   enable the web UI to interact with the local vlt vault.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..middleware import AuthContext, get_auth_context
from ...models.thread import (
    Thread,
    SyncRequest,
    SyncResponse,
    SyncStatus,
    ThreadListResponse,
    ThreadSearchResponse,
    SummarizeRequest,
    SummarizeResponse,
    # CLI-based models (T037-T039)
    CreateThreadRequest,
    CreateThreadResponse,
    PushEntryRequest,
    PushEntryResponse,
    SeekResult,
    SeekResponse,
)
from ...services.thread_service import ThreadService, get_thread_service
from ...services.librarian_service import LibrarianService, get_librarian_service

logger = logging.getLogger(__name__)


# ============================================================================
# VLT CLI Helper Functions
# ============================================================================

async def run_vlt_command(
    args: list[str],
    timeout: int = 30,
) -> tuple[bool, str, str]:
    """
    Run a vlt CLI command asynchronously.

    Args:
        args: Command arguments (after 'vlt')
        timeout: Command timeout in seconds

    Returns:
        Tuple of (success, stdout, stderr)
    """
    cmd = ["vlt"] + args

    try:
        logger.info(f"Running vlt command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return False, "", f"Command timed out after {timeout}s"

        stdout_str = stdout.decode("utf-8").strip()
        stderr_str = stderr.decode("utf-8").strip()

        if process.returncode != 0:
            logger.error(f"vlt command failed: {stderr_str}")
            return False, stdout_str, stderr_str

        return True, stdout_str, stderr_str

    except FileNotFoundError:
        return False, "", "vlt CLI not found. Please ensure vlt is installed and in PATH."
    except Exception as e:
        logger.exception(f"Failed to run vlt command: {e}")
        return False, "", str(e)

router = APIRouter(prefix="/api/threads", tags=["threads"])


# Constants for validation
MAX_ENTRIES_PER_SYNC = 100
MAX_ENTRY_SIZE_BYTES = 100_000  # 100KB per entry


# T014: POST /api/threads/sync
@router.post("/sync", response_model=SyncResponse)
async def sync_thread(
    request: SyncRequest,
    auth: AuthContext = Depends(get_auth_context),
    service: ThreadService = Depends(get_thread_service),
):
    """
    Sync thread entries from vlt-cli.

    Creates the thread if it doesn't exist, or updates existing thread.
    Supports incremental sync - only new entries need to be sent.

    **Request Body:**
    - `thread_id`: Unique identifier for the thread
    - `project_id`: Project the thread belongs to
    - `name`: Human-readable thread name
    - `status`: Thread status ("active", "archived", "blocked")
    - `created_at`: Original creation timestamp (optional)
    - `updated_at`: Last update timestamp (optional)
    - `entries`: List of thread entries to sync (max 100 per request)

    **Response:**
    - `thread_id`: The synced thread's ID
    - `synced_count`: Number of new entries synced
    - `last_synced_sequence`: Highest sequence_id synced

    **Limits:**
    - Maximum 100 entries per sync request
    - Maximum 100KB per entry content
    """
    # T041: Request validation - size limits
    if len(request.entries) > MAX_ENTRIES_PER_SYNC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many entries: {len(request.entries)} exceeds limit of {MAX_ENTRIES_PER_SYNC}",
        )

    for i, entry in enumerate(request.entries):
        content_size = len(entry.content.encode("utf-8"))
        if content_size > MAX_ENTRY_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Entry {i} content too large: {content_size} bytes exceeds limit of {MAX_ENTRY_SIZE_BYTES}",
            )

    try:
        logger.info(f"Sync request for thread {request.thread_id} from user {auth.user_id} ({len(request.entries)} entries)")

        # Create or update thread
        service.create_or_update_thread(
            user_id=auth.user_id,
            thread_id=request.thread_id,
            project_id=request.project_id,
            name=request.name,
            status=request.status,
            created_at=request.created_at,
            updated_at=request.updated_at,
        )

        # Add entries
        synced_count, last_seq = service.add_entries(
            user_id=auth.user_id,
            thread_id=request.thread_id,
            entries=request.entries,
        )

        return SyncResponse(
            thread_id=request.thread_id,
            synced_count=synced_count,
            last_synced_sequence=last_seq,
        )
    except Exception as e:
        logger.exception(f"Sync failed for thread {request.thread_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sync failed: {str(e)}",
        )


# T031: GET /api/threads/search (for Phase 4, but add endpoint now)
# NOTE: This route MUST be defined BEFORE the /{thread_id} route to avoid
# FastAPI treating "search" as a thread_id parameter
@router.get("/search", response_model=ThreadSearchResponse)
async def search_threads(
    q: str = Query(..., min_length=1, max_length=256, description="Search query"),
    project_id: Optional[str] = Query(None, description="Filter by project"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    auth: AuthContext = Depends(get_auth_context),
    service: ThreadService = Depends(get_thread_service),
):
    """
    Full-text search across thread entries.

    Uses SQLite FTS5 with BM25 ranking to search thread entry content.
    Returns matching entries with highlighted snippets.

    **Query Parameters:**
    - `q`: Search query (required, 1-256 chars)
    - `project_id`: Filter results to a specific project
    - `limit`: Maximum results to return (1-50, default: 10)

    **Response:**
    - `results`: List of matching entries with snippets and relevance scores
    - `total`: Total number of matching results
    """
    return service.search_threads(
        user_id=auth.user_id,
        query=q,
        project_id=project_id,
        limit=limit,
    )


# ============================================================================
# CLI-based Thread API Endpoints (T037-T039)
# These endpoints call vlt CLI directly for local vault operations
# ============================================================================


# T039: GET /api/threads/seek - Semantic search via vlt CLI
# NOTE: This route MUST be defined BEFORE the /{thread_id} route to avoid
# FastAPI treating "seek" as a thread_id parameter
@router.get("/seek", response_model=SeekResponse)
async def seek_threads(
    q: str = Query(..., min_length=1, max_length=512, description="Semantic search query"),
    project: Optional[str] = Query(None, description="Filter by project slug"),
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Semantic search across threads via vlt CLI.

    Uses vlt's semantic search capabilities to find relevant entries
    based on meaning rather than exact keyword matches. This enables
    natural language queries like "How did I solve the caching problem?"

    **Query Parameters:**
    - `q`: Search query (required, 1-512 chars)
    - `project`: Filter results to a specific project

    **Response:**
    - `query`: The original search query
    - `results`: List of matching entries with content and relevance scores
    - `total`: Total number of matches
    - `project`: Project filter applied, if any

    **Example:**
    ```
    GET /api/threads/seek?q=how%20to%20implement%20caching
    ```

    **Notes:**
    - Requires vlt CLI to be installed and accessible
    - Uses the local vlt vault (not synced threads in backend DB)
    - Results are ranked by semantic similarity
    """
    # Build vlt command: vlt thread seek "query" [--project project]
    args = ["thread", "seek", q]

    if project:
        args.extend(["--project", project])

    success, stdout, stderr = await run_vlt_command(args, timeout=30)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"vlt seek failed: {stderr or 'Unknown error'}",
        )

    # Parse vlt output - it may output formatted text, not JSON
    # We'll parse it as best we can
    results = []

    # Try to parse as JSON first (if vlt supports --json flag)
    try:
        # Attempt JSON parsing
        data = json.loads(stdout)
        if isinstance(data, list):
            for item in data:
                results.append(SeekResult(
                    thread_id=item.get("thread_id", "unknown"),
                    project_id=item.get("project_id"),
                    content=item.get("content", ""),
                    score=float(item.get("score", 0.5)),
                    author=item.get("author"),
                    timestamp=datetime.fromisoformat(item["timestamp"]) if item.get("timestamp") else None,
                ))
    except json.JSONDecodeError:
        # Fall back to text parsing
        # vlt seek typically outputs formatted text with thread/project/content
        if stdout:
            # Simple heuristic: each non-empty block is a result
            lines = stdout.split("\n")
            current_content = []

            for line in lines:
                if line.strip():
                    current_content.append(line)
                elif current_content:
                    # End of a result block
                    results.append(SeekResult(
                        thread_id="parsed",
                        content="\n".join(current_content),
                        score=0.5,  # Default score when not available
                    ))
                    current_content = []

            # Don't forget last block
            if current_content:
                results.append(SeekResult(
                    thread_id="parsed",
                    content="\n".join(current_content),
                    score=0.5,
                ))

    return SeekResponse(
        query=q,
        results=results,
        total=len(results),
        project=project,
    )


# T037: POST /api/threads - Create a new thread via vlt CLI
# NOTE: This route MUST be defined BEFORE the /{thread_id} routes
@router.post("/create", response_model=CreateThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread_via_cli(
    request: CreateThreadRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Create a new thread via vlt CLI.

    Creates a new reasoning chain in the local vlt vault. This is useful
    when the web UI needs to initiate a new thread without going through
    the CLI directly.

    **Request Body:**
    - `name`: Thread slug/name (e.g., 'optimization-strategy')
    - `initial_thought`: Initial thought/content for the thread
    - `project`: Project slug (optional, defaults to auto-detected)
    - `author`: Override the author (optional)

    **Response:**
    - `thread_id`: Created thread identifier
    - `project_id`: Project the thread belongs to
    - `name`: Thread name
    - `success`: Whether creation succeeded
    - `message`: Status message

    **Example:**
    ```json
    POST /api/threads/create
    {
        "name": "api-redesign",
        "initial_thought": "Starting to redesign the API layer for better ergonomics",
        "project": "my-project"
    }
    ```

    **Notes:**
    - Requires vlt CLI to be installed and accessible
    - Thread is created in the local vlt vault
    - Use /api/threads/sync to sync to the backend database
    """
    # Build vlt command: vlt thread new NAME "INITIAL_THOUGHT" [--project P] [--author A]
    args = ["thread", "new", request.name, request.initial_thought]

    if request.project:
        args.extend(["--project", request.project])

    if request.author:
        args.extend(["--author", request.author])

    success, stdout, stderr = await run_vlt_command(args, timeout=30)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create thread: {stderr or 'Unknown error'}",
        )

    # Parse the output to extract thread info
    # vlt thread new typically outputs something like:
    # "Created thread 'name' in project 'project'"
    thread_id = request.name
    project_id = request.project or "unknown"

    # Try to extract project from stdout if available
    if "project" in stdout.lower():
        # Simple extraction - look for project name in quotes
        match = re.search(r"project\s+['\"]?(\S+)['\"]?", stdout, re.IGNORECASE)
        if match:
            project_id = match.group(1).strip("'\"")

    return CreateThreadResponse(
        thread_id=thread_id,
        project_id=project_id,
        name=request.name,
        success=True,
        message=stdout or "Thread created successfully",
    )


# T015: GET /api/threads
@router.get("", response_model=ThreadListResponse)
async def list_threads(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    thread_status: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum threads to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    auth: AuthContext = Depends(get_auth_context),
    service: ThreadService = Depends(get_thread_service),
):
    """
    List user's synced threads with optional filters.

    Returns threads ordered by most recently updated first.

    **Query Parameters:**
    - `project_id`: Filter by project ID
    - `status`: Filter by thread status ("active", "archived", "blocked")
    - `limit`: Maximum threads to return (1-100, default: 50)
    - `offset`: Pagination offset (default: 0)

    **Response:**
    - `threads`: List of thread objects (without entries)
    - `total`: Total count of matching threads
    """
    return service.list_threads(
        user_id=auth.user_id,
        project_id=project_id,
        status=thread_status,
        limit=limit,
        offset=offset,
    )


# T016: GET /api/threads/{thread_id}
@router.get("/{thread_id}", response_model=Thread)
async def get_thread(
    thread_id: str,
    include_entries: bool = Query(True, description="Include thread entries"),
    entries_limit: int = Query(50, ge=1, le=500, description="Maximum entries to return"),
    auth: AuthContext = Depends(get_auth_context),
    service: ThreadService = Depends(get_thread_service),
):
    """
    Get a specific thread with its entries.

    **Path Parameters:**
    - `thread_id`: The thread's unique identifier

    **Query Parameters:**
    - `include_entries`: Whether to include thread entries (default: true)
    - `entries_limit`: Maximum entries to return (1-500, default: 50)

    **Response:**
    - Thread object with optional entries list
    """
    thread = service.get_thread(
        user_id=auth.user_id,
        thread_id=thread_id,
        include_entries=include_entries,
        entries_limit=entries_limit,
    )

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    return thread


# T038: POST /api/threads/{thread_id}/entries - Push entry via vlt CLI
@router.post("/{thread_id}/entries", response_model=PushEntryResponse, status_code=status.HTTP_201_CREATED)
async def push_thread_entry(
    thread_id: str,
    request: PushEntryRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Push a new entry to a thread via vlt CLI.

    Commits a thought to permanent memory in the local vlt vault. This is
    a fire-and-forget operation designed for logging intermediate reasoning
    steps so you can free up context window space.

    **Path Parameters:**
    - `thread_id`: Thread slug or path to push to

    **Request Body:**
    - `content`: The thought/content to log (required)
    - `author`: Override the author (optional)

    **Response:**
    - `thread_id`: Thread the entry was added to
    - `success`: Whether push succeeded
    - `message`: Status message

    **Example:**
    ```json
    POST /api/threads/api-redesign/entries
    {
        "content": "Decided to use FastAPI for its async support and automatic OpenAPI generation"
    }
    ```

    **Notes:**
    - Requires vlt CLI to be installed and accessible
    - Entry is pushed to the local vlt vault
    - Optimized for speed (<50ms target)
    - If vlt daemon is running, sync is routed through it for better performance
    """
    # Build vlt command: vlt thread push THREAD_ID "CONTENT" [--author A]
    args = ["thread", "push", thread_id, request.content]

    if request.author:
        args.extend(["--author", request.author])

    success, stdout, stderr = await run_vlt_command(args, timeout=10)  # Short timeout for push

    if not success:
        # Check for specific error cases
        if "not found" in stderr.lower() or "does not exist" in stderr.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread '{thread_id}' not found. Create it first with POST /api/threads/create",
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to push entry: {stderr or 'Unknown error'}",
        )

    return PushEntryResponse(
        thread_id=thread_id,
        success=True,
        message=stdout or "Entry pushed successfully",
    )


# T017: GET /api/threads/{thread_id}/status
@router.get("/{thread_id}/status", response_model=SyncStatus)
async def get_sync_status(
    thread_id: str,
    auth: AuthContext = Depends(get_auth_context),
    service: ThreadService = Depends(get_thread_service),
):
    """
    Get sync status for a thread.

    Returns information about the last sync operation, including
    the highest synced sequence_id and any sync errors.

    **Path Parameters:**
    - `thread_id`: The thread's unique identifier

    **Response:**
    - `thread_id`: The thread's ID
    - `last_synced_sequence`: Highest sequence_id synced (-1 if never synced)
    - `last_sync_at`: Timestamp of last sync
    - `sync_error`: Error message from last failed sync (null if successful)
    """
    sync_status = service.get_sync_status(
        user_id=auth.user_id,
        thread_id=thread_id,
    )

    if not sync_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    return sync_status


# T018: DELETE /api/threads/{thread_id}
@router.delete("/{thread_id}")
async def delete_thread(
    thread_id: str,
    auth: AuthContext = Depends(get_auth_context),
    service: ThreadService = Depends(get_thread_service),
):
    """
    Delete a synced thread and all its entries.

    This permanently removes the thread, all its entries, and sync status.
    The operation cannot be undone.

    **Path Parameters:**
    - `thread_id`: The thread's unique identifier

    **Response:**
    ```json
    {"status": "ok", "message": "Thread deleted"}
    ```
    """
    deleted = service.delete_thread(
        user_id=auth.user_id,
        thread_id=thread_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    return {"status": "ok", "message": "Thread deleted"}


# Summarization endpoint - server-side Librarian
@router.post("/{thread_id}/summarize", response_model=SummarizeResponse)
async def summarize_thread(
    thread_id: str,
    request: SummarizeRequest = None,
    auth: AuthContext = Depends(get_auth_context),
    thread_service: ThreadService = Depends(get_thread_service),
    librarian: LibrarianService = Depends(get_librarian_service),
):
    """
    Summarize a thread using server-side LLM.

    Generates an AI-powered summary of the thread's entries. The summary
    provides a structured "State Object" that allows agents to quickly
    resume context after a context window reset.

    This endpoint replaces the CLI-side Librarian functionality, centralizing
    LLM operations on the server where API keys and billing are managed.

    **Path Parameters:**
    - `thread_id`: The thread's unique identifier

    **Request Body (optional):**
    - `current_summary`: Existing summary to update (for incremental summarization)
    - `entries_limit`: Maximum entries to include (default: 100, max: 500)

    **Response:**
    - `thread_id`: The thread's ID
    - `summary`: Generated summary text (Markdown formatted)
    - `model`: Model used for summarization
    - `tokens_used`: Approximate tokens consumed
    - `success`: Whether summarization succeeded
    - `error`: Error message if failed

    **Notes:**
    - Uses the user's configured oracle model from settings
    - Falls back to system API keys if user hasn't configured their own
    - Summary format is optimized for agent context restoration
    """
    # Handle case where request body is not provided
    if request is None:
        request = SummarizeRequest()

    # Get the thread with entries
    thread = thread_service.get_thread(
        user_id=auth.user_id,
        thread_id=thread_id,
        include_entries=True,
        entries_limit=request.entries_limit,
    )

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    if not thread.entries:
        return SummarizeResponse(
            thread_id=thread_id,
            summary=request.current_summary or "No entries to summarize.",
            model=None,
            tokens_used=0,
            success=True,
            error=None,
        )

    try:
        # Call the librarian service
        result = await librarian.summarize_thread(
            user_id=auth.user_id,
            entries=thread.entries,
            current_summary=request.current_summary,
        )

        return SummarizeResponse(
            thread_id=thread_id,
            summary=result["summary"],
            model=result["model"],
            tokens_used=result["tokens_used"],
            success=result["success"],
            error=result["error"],
        )

    except Exception as e:
        logger.exception(f"Summarization failed for thread {thread_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}",
        )


__all__ = ["router"]
