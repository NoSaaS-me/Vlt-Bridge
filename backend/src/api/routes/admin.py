"""Admin routes for user management."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..middleware import AuthContext, require_admin_context
from ...services.user_service import UserService

router = APIRouter(prefix="/api/admin", tags=["admin"])


class UserResponse(BaseModel):
    user_id: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: str
    approved_by: Optional[str] = None
    created_at: str
    last_login_at: Optional[str] = None


class UpdateRoleRequest(BaseModel):
    role: str  # 'admin', 'user', 'pending', 'blocked'


@router.get("/users")
async def list_users(
    role: Optional[str] = None,
    auth: AuthContext = Depends(require_admin_context),
):
    """List all users. Admin only."""
    svc = UserService()
    users = svc.list_users(role=role)
    return [UserResponse(**u.__dict__) for u in users]


@router.patch("/users/{user_id}")
async def update_user_role(
    user_id: str,
    body: UpdateRoleRequest,
    auth: AuthContext = Depends(require_admin_context),
):
    """Update a user's role. Admin only."""
    if body.role not in ("admin", "user", "pending", "blocked"):
        raise HTTPException(400, "Invalid role")
    if user_id == auth.user_id and body.role != "admin":
        raise HTTPException(400, "Cannot demote yourself")
    svc = UserService()
    updated = svc.update_role(user_id, body.role, approved_by=auth.user_id)
    if not updated:
        raise HTTPException(404, "User not found")
    return UserResponse(**updated.__dict__)


@router.get("/setup-status")
async def setup_status():
    """Check if initial setup is complete. Public endpoint (no auth required)."""
    from ...services.config import get_config

    config = get_config()
    if not config.require_user_approval:
        return {"setup_complete": True, "approval_required": False}
    svc = UserService()
    return {"setup_complete": svc.is_setup_complete(), "approval_required": True}


__all__ = ["router"]
