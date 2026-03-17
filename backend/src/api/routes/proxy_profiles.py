"""API routes for proxy profile management."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..middleware import AuthContext, require_auth_context
from ...services.proxy_service import ProxyProfileService, get_proxy_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/proxy-profiles", tags=["proxy-profiles"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CreateProxyProfileRequest(BaseModel):
    name: str
    proxy_url: str
    proxy_username: str = ""
    proxy_password: str = ""


class UpdateProxyProfileRequest(BaseModel):
    proxy_url: Optional[str] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("")
async def list_proxy_profiles(
    auth: AuthContext = Depends(require_auth_context),
    svc: ProxyProfileService = Depends(get_proxy_service),
) -> dict:
    """List all proxy profiles for the authenticated user (passwords masked)."""
    profiles = svc.list_profiles(auth.user_id)
    return {"profiles": profiles, "total": len(profiles)}


@router.post("", status_code=201)
async def create_proxy_profile(
    body: CreateProxyProfileRequest,
    auth: AuthContext = Depends(require_auth_context),
    svc: ProxyProfileService = Depends(get_proxy_service),
) -> dict:
    """Create a new proxy profile."""
    try:
        profile = svc.create_profile(
            user_id=auth.user_id,
            name=body.name,
            proxy_url=body.proxy_url,
            proxy_username=body.proxy_username,
            proxy_password=body.proxy_password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return profile


@router.get("/{name}")
async def get_proxy_profile(
    name: str,
    auth: AuthContext = Depends(require_auth_context),
    svc: ProxyProfileService = Depends(get_proxy_service),
) -> dict:
    """Get a single proxy profile (password masked)."""
    profile = svc.get_profile(auth.user_id, name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Proxy profile '{name}' not found")
    return profile


@router.put("/{name}")
async def update_proxy_profile(
    name: str,
    body: UpdateProxyProfileRequest,
    auth: AuthContext = Depends(require_auth_context),
    svc: ProxyProfileService = Depends(get_proxy_service),
) -> dict:
    """Update an existing proxy profile."""
    updated = svc.update_profile(
        user_id=auth.user_id,
        name=name,
        proxy_url=body.proxy_url,
        proxy_username=body.proxy_username,
        proxy_password=body.proxy_password,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Proxy profile '{name}' not found")
    return updated


@router.delete("/{name}", status_code=204)
async def delete_proxy_profile(
    name: str,
    auth: AuthContext = Depends(require_auth_context),
    svc: ProxyProfileService = Depends(get_proxy_service),
):
    """Delete a proxy profile."""
    deleted = svc.delete_profile(auth.user_id, name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Proxy profile '{name}' not found")
