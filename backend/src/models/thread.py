"""Pydantic models for Thread Sync feature."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ThreadEntry(BaseModel):
    """Single entry/node within a thread."""

    entry_id: str = Field(..., description="UUID of the entry")
    sequence_id: int = Field(..., ge=0, description="Order within thread")
    content: str = Field(..., min_length=1, max_length=100000)
    author: str = Field("user", max_length=64)
    timestamp: datetime


class Thread(BaseModel):
    """Thread synced from vlt-cli."""

    thread_id: str = Field(..., min_length=1, max_length=128)
    project_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=256)
    status: Literal["active", "archived", "blocked"] = "active"
    created_at: datetime
    updated_at: datetime
    entries: Optional[List[ThreadEntry]] = None


class SyncRequest(BaseModel):
    """Request to sync thread entries from CLI."""

    thread_id: str = Field(..., min_length=1, max_length=128)
    project_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=256)
    status: Literal["active", "archived", "blocked"] = "active"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    entries: List[ThreadEntry] = Field(..., min_length=1)


class SyncResponse(BaseModel):
    """Response after syncing entries."""

    thread_id: str
    synced_count: int
    last_synced_sequence: int


class SyncStatus(BaseModel):
    """Sync status for a thread."""

    thread_id: str
    last_synced_sequence: int = Field(
        -1, description="Highest synced sequence_id (-1 if never synced)"
    )
    last_sync_at: datetime
    sync_error: Optional[str] = None


class ThreadListResponse(BaseModel):
    """List of threads for a user."""

    threads: List[Thread]
    total: int


class ThreadSearchResult(BaseModel):
    """Search result from thread FTS."""

    thread_id: str
    entry_id: str
    content: str = Field(..., description="Matching content snippet")
    author: str
    timestamp: datetime
    score: float = Field(..., description="Relevance score")


class ThreadSearchResponse(BaseModel):
    """Response for thread search."""

    results: List[ThreadSearchResult]
    total: int


class SummarizeRequest(BaseModel):
    """Request to summarize a thread."""

    current_summary: Optional[str] = Field(
        None,
        description="Existing summary to update (for incremental summarization)"
    )
    entries_limit: int = Field(
        100,
        ge=1,
        le=500,
        description="Maximum number of recent entries to include in summarization"
    )


class SummarizeResponse(BaseModel):
    """Response from thread summarization."""

    thread_id: str
    summary: str = Field(..., description="Generated summary text")
    model: Optional[str] = Field(None, description="Model used for summarization")
    tokens_used: int = Field(0, description="Approximate tokens consumed")
    success: bool = Field(True, description="Whether summarization succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# CLI-based Thread API Models (T037-T039)
# These models are for endpoints that interact with vlt CLI directly
# ============================================================================


class CreateThreadRequest(BaseModel):
    """Request to create a new thread via vlt CLI."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Thread slug/name (e.g., 'optimization-strategy')"
    )
    initial_thought: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Initial thought/content for the thread"
    )
    project: Optional[str] = Field(
        None,
        max_length=128,
        description="Project slug. Defaults to auto-detected from vlt.toml context."
    )
    author: Optional[str] = Field(
        None,
        max_length=64,
        description="Override the author for this thread"
    )


class CreateThreadResponse(BaseModel):
    """Response after creating a thread via vlt CLI."""

    thread_id: str = Field(..., description="Created thread identifier")
    project_id: str = Field(..., description="Project the thread belongs to")
    name: str = Field(..., description="Thread name")
    success: bool = Field(True, description="Whether creation succeeded")
    message: Optional[str] = Field(None, description="Status message")


class PushEntryRequest(BaseModel):
    """Request to push an entry to a thread via vlt CLI."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="The thought/content to log"
    )
    author: Optional[str] = Field(
        None,
        max_length=64,
        description="Override the author for this thought"
    )


class PushEntryResponse(BaseModel):
    """Response after pushing an entry via vlt CLI."""

    thread_id: str = Field(..., description="Thread the entry was added to")
    success: bool = Field(True, description="Whether push succeeded")
    message: Optional[str] = Field(None, description="Status message")


class SeekResult(BaseModel):
    """Single result from semantic search."""

    thread_id: str = Field(..., description="Thread ID containing the match")
    project_id: Optional[str] = Field(None, description="Project the thread belongs to")
    content: str = Field(..., description="Matching content/snippet")
    score: float = Field(..., description="Relevance score (0-1)")
    author: Optional[str] = Field(None, description="Author of the entry")
    timestamp: Optional[datetime] = Field(None, description="When the entry was created")


class SeekResponse(BaseModel):
    """Response from semantic thread search via vlt CLI."""

    query: str = Field(..., description="Original search query")
    results: List[SeekResult] = Field(default_factory=list, description="Matching results")
    total: int = Field(0, description="Total number of matches")
    project: Optional[str] = Field(None, description="Project filter applied, if any")


__all__ = [
    "ThreadEntry",
    "Thread",
    "SyncRequest",
    "SyncResponse",
    "SyncStatus",
    "ThreadListResponse",
    "ThreadSearchResult",
    "ThreadSearchResponse",
    "SummarizeRequest",
    "SummarizeResponse",
    # CLI-based Thread API Models (T037-T039)
    "CreateThreadRequest",
    "CreateThreadResponse",
    "PushEntryRequest",
    "PushEntryResponse",
    "SeekResult",
    "SeekResponse",
]
