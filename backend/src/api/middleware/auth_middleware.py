"""Authentication dependency helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Optional

from fastapi import Header, HTTPException, status

from ...models.auth import JWTPayload
from ...services.auth import AuthError, AuthService
from ...services.config import get_config
from datetime import datetime, timezone

auth_service = AuthService()


def _unauthorized(message: str, error: str = "unauthorized") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": error, "message": message},
    )


@dataclass
class AuthContext:
    """Context extracted from a bearer token."""

    user_id: str
    token: str
    payload: JWTPayload


def get_auth_context(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
) -> AuthContext:
    """
    Extract and validate the user_id from a Bearer token.

    Raises HTTPException if the header is missing/invalid.
    """
    if not authorization:
        # Check for No-Auth mode (Hackathon/Demo)
        config = get_config()
        if config.enable_noauth_mcp:
            # Create a dummy payload for demo user
            payload = JWTPayload(
                sub="demo-user",
                iat=int(datetime.now(timezone.utc).timestamp()),
                exp=int(datetime.now(timezone.utc).timestamp()) + 3600
            )
            return AuthContext(user_id="demo-user", token="no-auth", payload=payload)
            
        raise _unauthorized("Authorization header required")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise _unauthorized("Authorization header must be in format: Bearer <token>")

    try:
        payload = auth_service.validate_jwt(token)
    except AuthError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error": exc.error, "message": exc.message, "detail": exc.detail},
        ) from exc

    return AuthContext(user_id=payload.sub, token=token, payload=payload)


def extract_user_id_from_jwt(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
) -> str:
    """Compatibility helper that returns only the user_id."""
    return get_auth_context(authorization).user_id


__all__ = ["AuthContext", "extract_user_id_from_jwt", "get_auth_context"]
