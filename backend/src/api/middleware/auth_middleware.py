"""Authentication dependency helpers."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import Header, HTTPException, status

from ...services.auth import AuthError, AuthService

auth_service = AuthService()


def _unauthorized(message: str, error: str = "unauthorized") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": error, "message": message},
    )


def extract_user_id_from_jwt(
    authorization: Annotated[Optional[str], Header(default=None, alias="Authorization")] = None,
) -> str:
    """
    Extract and validate the user_id from a Bearer token.

    Raises HTTPException if the header is missing/invalid.
    """
    if not authorization:
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

    return payload.sub


__all__ = ["extract_user_id_from_jwt"]
