"""User and profile models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GHProfile(BaseModel):
    """GitHub OAuth profile information."""

    username: str = Field(..., description="GitHub username")
    name: Optional[str] = Field(None, description="Display name")
    avatar_url: Optional[str] = Field(None, description="Profile picture URL")


class User(BaseModel):
    """User account with authentication details."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "gh-alice",
                "gh_profile": {
                    "username": "alice",
                    "name": "Alice Smith",
                    "avatar_url": "https://github.com/alice.png",
                },
                "vault_path": "/data/vaults/gh-alice",
                "created": "2025-01-15T10:30:00Z",
            }
        }
    )

    user_id: str = Field(..., min_length=1, max_length=64, description="Internal user ID")
    gh_profile: Optional[GHProfile] = Field(None, description="GitHub OAuth profile data")
    vault_path: str = Field(..., description="Absolute path to the user's vault")
    created: datetime = Field(..., description="Account creation timestamp")


__all__ = ["User", "GHProfile"]
