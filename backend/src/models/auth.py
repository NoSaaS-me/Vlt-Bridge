"""Authentication models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TokenResponse(BaseModel):
    """JWT issuance response."""

    token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type (always bearer)")
    expires_at: datetime = Field(..., description="Expiration timestamp")


class JWTPayload(BaseModel):
    """JWT claims payload."""

    sub: str = Field(..., description="Subject (user_id)")
    iat: int = Field(..., description="Issued at timestamp")
    exp: int = Field(..., description="Expiration timestamp")


__all__ = ["TokenResponse", "JWTPayload"]
