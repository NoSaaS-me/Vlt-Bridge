"""Composio Integration Hub routes.

Manages user connections to 100+ third-party apps via Composio.

Routes:
    GET    /api/composio/status           — check if Composio is configured
    GET    /api/composio/apps             — catalog with connected status
    GET    /api/composio/connected        — user's connected apps
    POST   /api/composio/connect/{app}   — initiate OAuth, return redirect URL
    DELETE /api/composio/{app}           — disconnect an app
    GET    /api/composio/{app}/actions   — list actions for an app
    POST   /api/composio/{app}/invoke    — execute a Composio action
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..middleware import AuthContext, require_auth_context

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/composio", tags=["composio-hub"])


def _get_composio_service():
    """Get the ComposioService from the vlt-connectors registry."""
    from vlt_connectors.registry import get_registry
    svc = get_registry().get_service("composio")
    if svc is None:
        raise HTTPException(503, "Composio service is not available. Check vlt-connectors installation.")
    return svc


class InvokeRequest(BaseModel):
    action: str
    params: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/status")
async def composio_status() -> dict:
    """Check if Composio is configured (API key present). No auth required."""
    try:
        svc = _get_composio_service()
        return {"configured": svc.is_configured()}
    except HTTPException:
        raise
    except Exception as exc:
        return {"configured": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

@router.get("/apps")
async def list_composio_apps(
    auth: AuthContext = Depends(require_auth_context),
) -> dict:
    """List all Composio apps with connected status for the current user."""
    svc = _get_composio_service()

    try:
        catalog = svc.catalog()
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch Composio catalog: {exc}")

    connected_names: set[str] = set()
    try:
        user_connections = svc.connected(entity_id=auth.user_id)
        connected_names = {c["app_name"].lower() for c in user_connections}
    except Exception:
        pass  # Connection fetch failure doesn't break catalog display

    annotated = [
        {**app, "connected": app["name"].lower() in connected_names}
        for app in catalog
    ]
    return {"apps": annotated, "total": len(annotated)}


# ---------------------------------------------------------------------------
# Connected apps
# ---------------------------------------------------------------------------

@router.get("/connected")
async def list_connected_apps(
    auth: AuthContext = Depends(require_auth_context),
) -> dict:
    """List apps the current user has connected via Composio."""
    svc = _get_composio_service()
    try:
        connections = svc.connected(entity_id=auth.user_id)
        return {"connections": connections, "total": len(connections)}
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch connections: {exc}")


# ---------------------------------------------------------------------------
# Connect (initiate OAuth)
# ---------------------------------------------------------------------------

@router.post("/connect/{app_name}")
async def connect_app(
    app_name: str,
    auth: AuthContext = Depends(require_auth_context),
) -> dict:
    """Initiate OAuth connection for a Composio app. Returns the redirect URL."""
    svc = _get_composio_service()
    try:
        redirect_url = svc.initiate_connection(
            app_name=app_name.lower(),
            entity_id=auth.user_id,
        )
        return {"app": app_name, "redirect_url": redirect_url}
    except Exception as exc:
        raise HTTPException(502, f"Failed to initiate connection: {exc}")


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------

@router.delete("/{app_name}")
async def disconnect_app(
    app_name: str,
    auth: AuthContext = Depends(require_auth_context),
) -> dict:
    """Disconnect a Composio app for the current user."""
    svc = _get_composio_service()
    try:
        svc.disconnect(app_name=app_name.lower(), entity_id=auth.user_id)
        return {"app": app_name, "disconnected": True}
    except Exception as exc:
        raise HTTPException(502, f"Failed to disconnect: {exc}")


# ---------------------------------------------------------------------------
# List actions for an app
# ---------------------------------------------------------------------------

@router.get("/{app_name}/actions")
async def list_app_actions(
    app_name: str,
    auth: AuthContext = Depends(require_auth_context),
) -> dict:
    """List available actions for a Composio app."""
    svc = _get_composio_service()
    try:
        actions = svc.get_actions(app_name=app_name.lower())
        return {"app": app_name, "actions": actions, "total": len(actions)}
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch actions: {exc}")


# ---------------------------------------------------------------------------
# Invoke an action
# ---------------------------------------------------------------------------

@router.post("/{app_name}/invoke")
async def invoke_composio_action(
    app_name: str,
    body: InvokeRequest,
    auth: AuthContext = Depends(require_auth_context),
) -> dict:
    """Execute a Composio action on behalf of the current user.

    action should be the Composio action name e.g. 'GMAIL_SEND_EMAIL'.
    Lowercase names are normalized to uppercase automatically.
    """
    svc = _get_composio_service()
    result = svc.execute(
        app_name=app_name.lower(),
        action_name=body.action,
        params=body.params,
        entity_id=auth.user_id,
    )
    if not result.get("success"):
        raise HTTPException(502, result.get("error", "Composio action failed"))
    return {"success": True, "data": result.get("data", {})}
