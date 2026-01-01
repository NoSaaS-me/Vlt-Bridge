"""CodeRAG API endpoints - Code index management and status tracking.

This module provides endpoints for:
- Getting CodeRAG index status for a project
- Triggering CodeRAG indexing jobs
- Monitoring job progress
- Cancelling indexing jobs

Based on spec: specs/011-coderag-project-init/contracts/coderag-api.yaml
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..middleware import AuthContext, get_auth_context
from ...models.coderag import (
    CodeRAGStatusResponse,
    InitCodeRAGRequest,
    InitCodeRAGResponse,
    JobStatusResponse,
    ErrorResponse,
    JobSummary,
)
from ...services.oracle_bridge import OracleBridge

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/coderag", tags=["coderag"])

# Singleton oracle bridge instance
_oracle_bridge: OracleBridge | None = None


def get_oracle_bridge() -> OracleBridge:
    """Get or create the oracle bridge instance."""
    global _oracle_bridge
    if _oracle_bridge is None:
        _oracle_bridge = OracleBridge()
    return _oracle_bridge


@router.get("/status", response_model=CodeRAGStatusResponse)
async def get_coderag_status(
    project_id: str = Query(..., description="Project ID to check status for"),
    auth: AuthContext = Depends(get_auth_context),
    bridge: OracleBridge = Depends(get_oracle_bridge),
):
    """
    Get CodeRAG index status for a project.

    Returns the current status of the CodeRAG index including:
    - Whether the index is initialized
    - Number of indexed files and chunks
    - Last indexed timestamp
    - Active indexing job status (if any)

    **Query Parameters:**
    - `project_id`: Project identifier (required)

    **Response:**
    - `project_id`: Project identifier
    - `status`: Index status (not_initialized, indexing, ready, failed, stale)
    - `file_count`: Number of indexed files
    - `chunk_count`: Number of code chunks
    - `last_indexed_at`: Last successful index timestamp
    - `error_message`: Error details if status is failed
    - `active_job`: Current indexing job details if any
    """
    try:
        logger.info(f"Getting CodeRAG status for project {project_id} (user: {auth.user_id})")

        result = bridge.get_coderag_status(project_id)

        if result.get("error"):
            # Handle error response from vlt CLI
            error_msg = result.get("message", "Unknown error")
            logger.warning(f"CodeRAG status error for project {project_id}: {error_msg}")

            # Return a not_initialized status if vlt is not available or index doesn't exist
            return CodeRAGStatusResponse(
                project_id=project_id,
                status="not_initialized",
                file_count=0,
                chunk_count=0,
                last_indexed_at=None,
                error_message=error_msg if "not found" not in error_msg.lower() else None,
                active_job=None,
            )

        # Parse active job if present
        active_job = None
        if result.get("active_job"):
            job_data = result["active_job"]
            active_job = JobSummary(
                job_id=job_data.get("job_id", ""),
                progress_percent=job_data.get("progress_percent", 0),
                files_processed=job_data.get("files_processed", 0),
                files_total=job_data.get("files_total", 0),
                started_at=job_data.get("started_at"),
            )

        # Extract stats from index_stats if present, else use top-level (for compatibility)
        index_stats = result.get("index_stats", {})
        file_count = index_stats.get("files_count", result.get("file_count", 0))
        chunk_count = index_stats.get("chunks_count", result.get("chunk_count", 0))

        return CodeRAGStatusResponse(
            project_id=result.get("project_id", project_id),
            status=result.get("status", "not_initialized"),
            file_count=file_count,
            chunk_count=chunk_count,
            last_indexed_at=result.get("last_indexed_at") or result.get("completed_at"),
            error_message=result.get("error_message"),
            active_job=active_job,
        )

    except Exception as e:
        logger.exception(f"Failed to get CodeRAG status for project {project_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get CodeRAG status: {str(e)}",
        )


@router.post("/init", response_model=InitCodeRAGResponse, status_code=201)
async def init_coderag(
    request: InitCodeRAGRequest,
    auth: AuthContext = Depends(get_auth_context),
    bridge: OracleBridge = Depends(get_oracle_bridge),
):
    """
    Initialize or re-index CodeRAG for a project.

    Triggers a new CodeRAG indexing job. By default, indexing runs in the
    background via the daemon process.

    **Request Body:**
    - `project_id`: Project to associate with index (required)
    - `target_path`: Directory path to index (required)
    - `force`: Force re-index even if index exists (default: false)
    - `background`: Run indexing in background (default: true)

    **Response:**
    - `job_id`: Identifier for tracking the job
    - `status`: Whether job is queued or started
    - `message`: Human-readable status message

    **Errors:**
    - 400: Invalid request (path doesn't exist)
    - 409: Index already exists (use force=true to re-index)
    """
    try:
        logger.info(
            f"Initializing CodeRAG for project {request.project_id} "
            f"(path: {request.target_path}, force: {request.force}, user: {auth.user_id})"
        )

        result = bridge.init_coderag(
            project_id=request.project_id,
            target_path=request.target_path,
            force=request.force,
            background=request.background,
        )

        if result.get("error"):
            error_msg = result.get("message", "Unknown error")
            logger.warning(f"CodeRAG init failed for project {request.project_id}: {error_msg}")

            # Determine appropriate error status code
            if "already exists" in error_msg.lower():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=error_msg,
                )
            elif "not found" in error_msg.lower() or "invalid" in error_msg.lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_msg,
                )

        return InitCodeRAGResponse(
            job_id=result.get("job_id", ""),
            status=result.get("status", "queued"),
            message=result.get("message", "Indexing job queued"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to initialize CodeRAG for project {request.project_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize CodeRAG: {str(e)}",
        )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    auth: AuthContext = Depends(get_auth_context),
    bridge: OracleBridge = Depends(get_oracle_bridge),
):
    """
    Get detailed status of an indexing job.

    **Path Parameters:**
    - `job_id`: Job identifier (UUID)

    **Response:**
    - `job_id`: Job identifier
    - `project_id`: Associated project ID
    - `status`: Current job status (pending, running, completed, failed, cancelled)
    - `progress_percent`: Completion percentage (0-100)
    - `files_total`: Total files to process
    - `files_processed`: Files completed
    - `chunks_created`: Code chunks generated
    - `started_at`: Processing start time
    - `completed_at`: Processing end time (if finished)
    - `error_message`: Error details if failed
    - `duration_seconds`: Elapsed time in seconds

    **Errors:**
    - 404: Job not found
    """
    try:
        logger.info(f"Getting job status for {job_id} (user: {auth.user_id})")

        result = bridge.get_job_status(job_id)

        if result.get("error"):
            error_msg = result.get("message", "Unknown error")
            if "not found" in error_msg.lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job not found: {job_id}",
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg,
            )

        return JobStatusResponse(
            job_id=result.get("job_id", job_id),
            project_id=result.get("project_id", ""),
            status=result.get("status", "pending"),
            progress_percent=result.get("progress_percent", 0),
            files_total=result.get("files_total", 0),
            files_processed=result.get("files_processed", 0),
            chunks_created=result.get("chunks_created", 0),
            started_at=result.get("started_at"),
            completed_at=result.get("completed_at"),
            error_message=result.get("error_message"),
            duration_seconds=result.get("duration_seconds"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get job status for {job_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}",
        )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    auth: AuthContext = Depends(get_auth_context),
    bridge: OracleBridge = Depends(get_oracle_bridge),
):
    """
    Cancel an indexing job.

    Cancels an in-progress or pending indexing job. Completed or failed
    jobs cannot be cancelled.

    **Path Parameters:**
    - `job_id`: Job identifier (UUID)

    **Response:**
    - `status`: "cancelled"
    - `message`: Confirmation message

    **Errors:**
    - 400: Job cannot be cancelled (already completed/failed)
    - 404: Job not found
    """
    try:
        logger.info(f"Cancelling job {job_id} (user: {auth.user_id})")

        result = bridge.cancel_job(job_id)

        if result.get("error"):
            error_msg = result.get("message", "Unknown error")
            if "not found" in error_msg.lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job not found: {job_id}",
                )
            elif "cannot be cancelled" in error_msg.lower() or "already" in error_msg.lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg,
            )

        return {
            "status": "cancelled",
            "message": result.get("message", f"Job {job_id} has been cancelled"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to cancel job {job_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}",
        )
