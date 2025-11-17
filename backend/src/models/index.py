"""Index and metadata models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Wikilink(BaseModel):
    """Bidirectional link between notes."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "alice",
                "source_path": "api/design.md",
                "target_path": "api/endpoints.md",
                "link_text": "Endpoints",
                "is_resolved": True,
            }
        }
    )

    user_id: str
    source_path: str
    target_path: Optional[str] = Field(None, description="Null if unresolved")
    link_text: str
    is_resolved: bool


class Tag(BaseModel):
    """Tag with aggregated count."""

    tag_name: str
    count: int = Field(..., ge=0)


class IndexHealth(BaseModel):
    """Index health metrics per user."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "alice",
                "note_count": 142,
                "last_full_rebuild": "2025-01-01T00:00:00Z",
                "last_incremental_update": "2025-01-15T14:30:00Z",
            }
        }
    )

    user_id: str
    note_count: int = Field(..., ge=0)
    last_full_rebuild: Optional[datetime] = None
    last_incremental_update: Optional[datetime] = None


__all__ = ["Wikilink", "Tag", "IndexHealth"]
