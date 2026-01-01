"""Authentication dependency helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Callable, Optional

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


def _forbidden(message: str, error: str = "forbidden") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"error": error, "message": message},
    )


class AuthMode(Enum):
    """
    Authentication mode for API routes.

    - OPTIONAL: Authentication is optional; falls back to demo-user if ENABLE_NOAUTH_MCP=true
    - STRICT: Authentication is required; never falls back to demo-user
    - ADMIN: Authentication is required AND user must have admin privileges
    """
    OPTIONAL = "optional"
    STRICT = "strict"
    ADMIN = "admin"


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


def require_auth_context(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
) -> AuthContext:
    """
    Extract and validate the user_id from a Bearer token.

    This dependency NEVER falls back to demo-user, regardless of ENABLE_NOAUTH_MCP.
    Use this for routes that must enforce strict authentication (sensitive data,
    paid APIs, administrative functions).

    Raises HTTPException(401) if the header is missing/invalid.
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

    return AuthContext(user_id=payload.sub, token=token, payload=payload)


def require_admin_context(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
) -> AuthContext:
    """
    Extract and validate the user_id from a Bearer token, then verify admin privileges.

    This dependency enforces strict authentication (no demo-user fallback) and then
    checks if the authenticated user has admin privileges.

    Use this for administrative routes like system logs, user management, etc.

    Raises HTTPException(401) if the header is missing/invalid.
    Raises HTTPException(403) if the user is not an admin.
    """
    # First, enforce strict authentication
    auth_context = require_auth_context(authorization)

    # Then, check if the user is an admin
    config = get_config()
    if auth_context.user_id not in config.admin_user_ids:
        raise _forbidden(
            "Admin privileges required",
            error="insufficient_permissions"
        )

    return auth_context


def extract_user_id_from_jwt(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
) -> str:
    """Compatibility helper that returns only the user_id."""
    return get_auth_context(authorization).user_id


def get_auth_dependency(mode: AuthMode) -> Callable[[Optional[str]], AuthContext]:
    """
    Factory function to get the appropriate authentication dependency based on mode.

    This provides a more explicit and type-safe way to specify authentication requirements
    for routes.

    Args:
        mode: The authentication mode (OPTIONAL, STRICT, or ADMIN)

    Returns:
        The appropriate authentication dependency function

    Raises:
        ValueError: If an unknown auth mode is provided

    Example:
        @router.get("/api/notes")
        async def list_notes(auth: AuthContext = Depends(get_auth_dependency(AuthMode.STRICT))):
            # This route requires strict authentication
            ...

        @router.get("/api/system/logs")
        async def get_logs(auth: AuthContext = Depends(get_auth_dependency(AuthMode.ADMIN))):
            # This route requires admin privileges
            ...

        @router.get("/api/index/health")
        async def health_check(auth: AuthContext = Depends(get_auth_dependency(AuthMode.OPTIONAL))):
            # This route allows optional authentication
            ...
    """
    if mode == AuthMode.OPTIONAL:
        return get_auth_context
    elif mode == AuthMode.STRICT:
        return require_auth_context
    elif mode == AuthMode.ADMIN:
        return require_admin_context
    else:
        raise ValueError(f"Unknown auth mode: {mode}")


__all__ = [
    "AuthContext",
    "AuthMode",
    "extract_user_id_from_jwt",
    "get_auth_context",
    "get_auth_dependency",
    "require_auth_context",
    "require_admin_context",
]
