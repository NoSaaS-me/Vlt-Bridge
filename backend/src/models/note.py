"""Note-related Pydantic models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NoteMetadata(BaseModel):
    """Frontmatter metadata (allows arbitrary keys)."""

    model_config = ConfigDict(extra="allow")

    title: Optional[str] = None
    tags: Optional[list[str]] = None
    project: Optional[str] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class Note(BaseModel):
    """Complete note with content and metadata."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "alice",
                "note_path": "api/design.md",
                "version": 5,
                "title": "API Design",
                "metadata": {
                    "tags": ["backend", "api"],
                    "project": "auth-service",
                },
                "body": "# API Design\\n\\nThis document describes...",
                "created": "2025-01-10T09:00:00Z",
                "updated": "2025-01-15T14:30:00Z",
                "size_bytes": 4096,
            }
        }
    )

    user_id: str = Field(..., description="Owner user ID")
    note_path: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Relative path to vault root (includes .md)",
    )
    version: int = Field(..., ge=1, description="Optimistic concurrency version")
    title: str = Field(..., min_length=1, description="Display title")
    metadata: NoteMetadata = Field(default_factory=NoteMetadata, description="Frontmatter")
    body: str = Field(..., description="Markdown content")
    created: datetime = Field(..., description="Creation timestamp")
    updated: datetime = Field(..., description="Last update timestamp")
    size_bytes: int = Field(..., ge=0, le=1_048_576, description="File size in bytes")

    @field_validator("note_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value.endswith(".md"):
            raise ValueError("Note path must end with .md")
        if ".." in value:
            raise ValueError("Note path must not contain '..'")
        if "\\" in value:
            raise ValueError("Note path must use Unix-style separators (/)")
        if value.startswith("/"):
            raise ValueError("Note path must be relative (no leading /)")
        return value


class NoteCreate(BaseModel):
    """Request payload to create a note."""

    note_path: str = Field(..., min_length=1, max_length=256)
    title: Optional[str] = None
    metadata: Optional[NoteMetadata] = None
    body: str = Field(..., max_length=1_048_576)


class NoteUpdate(BaseModel):
    """Request payload to update a note."""

    title: Optional[str] = None
    metadata: Optional[NoteMetadata] = None
    body: str = Field(..., max_length=1_048_576)
    if_version: Optional[int] = Field(
        None, ge=1, description="Expected version for concurrency check"
    )


class NoteSummary(BaseModel):
    """Lightweight representation used for listings."""

    note_path: str
    title: str
    updated: datetime


__all__ = ["NoteMetadata", "Note", "NoteCreate", "NoteUpdate", "NoteSummary"]
