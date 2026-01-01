"""Pydantic models for CodeRAG API endpoints.

Implements data models for:
- CodeRAG index status tracking
- Indexing job management
- Progress monitoring

Based on spec: specs/011-coderag-project-init/contracts/coderag-api.yaml
"""

from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field


# Status enum types
CodeRAGIndexStatus = Literal["not_initialized", "indexing", "ready", "failed", "stale"]
JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
InitJobStatus = Literal["queued", "started"]


class JobSummary(BaseModel):
    """Summary of an active indexing job."""
    job_id: str = Field(..., description="Job identifier (UUID)")
    progress_percent: int = Field(..., ge=0, le=100, description="Completion percentage")
    files_processed: int = Field(..., ge=0, description="Number of files processed")
    files_total: int = Field(..., ge=0, description="Total number of files to process")
    started_at: datetime = Field(..., description="Job start timestamp")


class CodeRAGStatusResponse(BaseModel):
    """Response for GET /api/coderag/status endpoint.

    Returns the current status of a project's CodeRAG index.
    """
    project_id: str = Field(..., description="Project identifier")
    status: CodeRAGIndexStatus = Field(..., description="Current index status")
    file_count: int = Field(default=0, ge=0, description="Number of indexed files")
    chunk_count: int = Field(default=0, ge=0, description="Number of code chunks")
    last_indexed_at: Optional[datetime] = Field(
        None, description="Last successful index timestamp"
    )
    error_message: Optional[str] = Field(
        None, description="Error details if status is failed"
    )
    active_job: Optional[JobSummary] = Field(
        None, description="Currently active indexing job"
    )


class InitCodeRAGRequest(BaseModel):
    """Request payload for POST /api/coderag/init endpoint.

    Triggers CodeRAG indexing for a project.
    """
    project_id: str = Field(..., description="Project to associate with index")
    target_path: str = Field(..., description="Directory path to index")
    force: bool = Field(
        default=False, description="Force re-index even if index exists"
    )
    background: bool = Field(
        default=True, description="Run indexing in background (via daemon)"
    )


class InitCodeRAGResponse(BaseModel):
    """Response for POST /api/coderag/init endpoint.

    Confirms that indexing job was queued or started.
    """
    job_id: str = Field(..., description="Identifier for tracking the job (UUID)")
    status: InitJobStatus = Field(
        ..., description="Whether job is queued or immediately started"
    )
    message: str = Field(..., description="Human-readable status message")


class JobStatusResponse(BaseModel):
    """Response for GET /api/coderag/jobs/{job_id} endpoint.

    Detailed status of an indexing job.
    """
    job_id: str = Field(..., description="Job identifier (UUID)")
    project_id: str = Field(..., description="Associated project ID")
    status: JobStatus = Field(..., description="Current job status")
    progress_percent: int = Field(default=0, ge=0, le=100, description="Completion percentage")
    files_total: int = Field(default=0, ge=0, description="Total files to process")
    files_processed: int = Field(default=0, ge=0, description="Files completed")
    chunks_created: int = Field(default=0, ge=0, description="Code chunks generated")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing end time")
    error_message: Optional[str] = Field(None, description="Error details if failed")
    duration_seconds: Optional[float] = Field(None, description="Elapsed time in seconds")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
