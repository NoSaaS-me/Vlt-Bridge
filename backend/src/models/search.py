"""Search request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Full-text search result payload."""

    note_path: str
    title: str
    snippet: str = Field(..., description="Highlighted body excerpt")
    score: float = Field(..., description="Relevance score (weighted by field)")
    updated: datetime


class SearchRequest(BaseModel):
    """Full-text search query parameters."""

    query: str = Field(..., min_length=1, max_length=256)
    limit: int = Field(50, ge=1, le=100)


__all__ = ["SearchResult", "SearchRequest"]
