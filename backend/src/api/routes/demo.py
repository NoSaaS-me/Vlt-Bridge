"""Public endpoints for demo mode access."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter

from ...services.auth import AuthService
from ...services.seed import ensure_welcome_note

DEMO_USER_ID = "demo-user"
DEMO_TOKEN_TTL_HOURS = 12

router = APIRouter()
auth_service = AuthService()


@router.get("/api/demo/token")
async def issue_demo_token():
    """
    Issue a short-lived JWT for the shared demo vault.

    The caller can use this token to explore the application in read-only mode.
    """
    ensure_welcome_note(DEMO_USER_ID)
    token, expires_at = auth_service.issue_token_response(
        DEMO_USER_ID, expires_in=timedelta(hours=DEMO_TOKEN_TTL_HOURS)
    )
    return {
        "token": token,
        "token_type": "bearer",
        "expires_at": expires_at.isoformat(),
        "user_id": DEMO_USER_ID,
    }


__all__ = ["router", "DEMO_USER_ID"]

