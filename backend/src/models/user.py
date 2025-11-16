"""User and profile models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HFProfile(BaseModel):
    """Hugging Face OAuth profile information."""

    username: str = Field(..., description="HF username")
    name: Optional[str] = Field(None, description="Display name")
    avatar_url: Optional[str] = Field(None, description="Profile picture URL")


class User(BaseModel):
    """User account with authentication details."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "alice",
                "hf_profile": {
                    "username": "alice",
                    "name": "Alice Smith",
                    "avatar_url": "https://cdn-avatars.huggingface.co/v1/alice",
                },
                "vault_path": "/data/vaults/alice",
                "created": "2025-01-15T10:30:00Z",
            }
        }
    )

    user_id: str = Field(..., min_length=1, max_length=64, description="Internal user ID")
    hf_profile: Optional[HFProfile] = Field(None, description="HF OAuth profile data")
    vault_path: str = Field(..., description="Absolute path to the user's vault")
    created: datetime = Field(..., description="Account creation timestamp")


__all__ = ["User", "HFProfile"]
