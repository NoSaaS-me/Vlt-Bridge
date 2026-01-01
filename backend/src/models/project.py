"""Project-related Pydantic models for multi-project architecture."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


def generate_project_slug(name: str) -> str:
    """
    Generate a URL-friendly slug from a project name.

    Args:
        name: Human-readable project name (e.g., "My Project Name")

    Returns:
        Slug suitable for use as project_id (e.g., "my-project-name")
    """
    # Convert to lowercase
    slug = name.lower()
    # Replace spaces and underscores with hyphens
    slug = re.sub(r'[\s_]+', '-', slug)
    # Remove any characters that aren't alphanumeric or hyphens
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    # Collapse multiple hyphens
    slug = re.sub(r'-+', '-', slug)
    # Strip leading/trailing hyphens
    slug = slug.strip('-')
    # Limit length
    return slug[:50] if slug else 'project'


class ProjectBase(BaseModel):
    """Base project fields shared between create/update."""

    name: str = Field(..., min_length=1, max_length=100, description="Human-readable project name")
    description: Optional[str] = Field(None, max_length=500, description="Optional project description")


class ProjectCreate(ProjectBase):
    """Request payload to create a new project."""

    id: Optional[str] = Field(
        None,
        pattern=r'^[a-z0-9-]+$',
        max_length=50,
        description="Optional custom project ID (slug). Auto-generated from name if not provided."
    )


class ProjectUpdate(BaseModel):
    """Request payload to update a project."""

    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Updated project name")
    description: Optional[str] = Field(None, max_length=500, description="Updated project description")


class Project(ProjectBase):
    """Complete project model with all fields."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "vlt-bridge",
                "user_id": "user-123",
                "name": "Vlt Bridge",
                "description": "AI memory and context retrieval system",
                "created_at": "2025-01-10T09:00:00Z",
                "updated_at": "2025-01-15T14:30:00Z",
                "note_count": 42,
                "thread_count": 7,
            }
        }
    )

    id: str = Field(..., description="Unique project identifier (slug)")
    user_id: str = Field(..., description="Owner user ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    note_count: Optional[int] = Field(0, description="Number of notes in project")
    thread_count: Optional[int] = Field(0, description="Number of threads in project")

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not re.match(r'^[a-z0-9-]+$', value):
            raise ValueError("Project ID must contain only lowercase letters, numbers, and hyphens")
        if len(value) > 50:
            raise ValueError("Project ID must be 50 characters or less")
        return value


class ProjectList(BaseModel):
    """Response model for listing projects."""

    projects: list[Project] = Field(default_factory=list, description="List of projects")
    total: int = Field(..., description="Total number of projects")


class ProjectStats(BaseModel):
    """Statistics for a project."""

    project_id: str = Field(..., description="Project identifier")
    note_count: int = Field(0, description="Number of notes")
    thread_count: int = Field(0, description="Number of threads")
    last_note_update: Optional[datetime] = Field(None, description="Last note update timestamp")
    last_thread_update: Optional[datetime] = Field(None, description="Last thread update timestamp")


# Default project ID for migration compatibility
DEFAULT_PROJECT_ID = "default"


__all__ = [
    "Project",
    "ProjectCreate",
    "ProjectUpdate",
    "ProjectBase",
    "ProjectList",
    "ProjectStats",
    "generate_project_slug",
    "DEFAULT_PROJECT_ID",
]
