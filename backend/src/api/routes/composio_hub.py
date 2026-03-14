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
from ...services.connector_service import ConnectorService, get_connector_service

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
# Action permission config
# ---------------------------------------------------------------------------

class ComposioConfigUpdate(BaseModel):
    config: dict[str, str]


@router.put("/{app_name}/config")
async def update_composio_config(
    app_name: str,
    body: ComposioConfigUpdate,
    auth: AuthContext = Depends(require_auth_context),
    connector_svc: ConnectorService = Depends(get_connector_service),
) -> dict:
    """Save action permission config for a composio connector.

    Only accepts __action_* keys. Stored under connector_name='composio:{app}'.
    """
    connector_name = f"composio:{app_name.lower()}"
    # Only allow __action_* keys through this endpoint
    filtered = {k: v for k, v in body.config.items() if k.startswith("__action_")}
    if not filtered:
        raise HTTPException(400, "No valid __action_* keys provided")
    connector_svc.set_config(auth.user_id, connector_name, filtered)
    return {"connector": connector_name, "saved": True}


@router.get("/{app_name}/config")
async def get_composio_config(
    app_name: str,
    auth: AuthContext = Depends(require_auth_context),
    connector_svc: ConnectorService = Depends(get_connector_service),
) -> dict:
    """Get action permission config for a composio connector."""
    connector_name = f"composio:{app_name.lower()}"
    config = connector_svc.get_config(auth.user_id, connector_name)
    # Only return __action_* keys
    action_config = {k: v for k, v in config.items() if k.startswith("__action_")}
    return {"connector": connector_name, "config": action_config}


# ---------------------------------------------------------------------------
# Invoke an action
# ---------------------------------------------------------------------------

@router.post("/{app_name}/invoke")
async def invoke_composio_action(
    app_name: str,
    body: InvokeRequest,
    auth: AuthContext = Depends(require_auth_context),
    connector_svc: ConnectorService = Depends(get_connector_service),
) -> dict:
    """Execute a Composio action on behalf of the current user.

    action should be the Composio action name e.g. 'GMAIL_SEND_EMAIL'.
    Lowercase names are normalized to uppercase automatically.
    """
    # Per-action permission check for composio connectors — stored under "composio:{app_name}"
    composio_connector_name = f"composio:{app_name.lower()}"
    permission = connector_svc.get_action_permission(auth.user_id, composio_connector_name, body.action)
    if permission == "off":
        raise HTTPException(
            status_code=403,
            detail=f"Action '{body.action}' is disabled for '{composio_connector_name}'. Enable it in Connectors settings.",
        )

    svc = _get_composio_service()
    try:
        result = svc.execute(
            app_name=app_name.lower(),
            action_name=body.action,
            params=body.params,
            entity_id=auth.user_id,
        )
    except Exception as exc:
        logger.exception("Composio action failed: %s/%s", app_name, body.action)
        raise HTTPException(502, f"Composio action failed: {exc}")
    if not result.get("success"):
        raise HTTPException(502, result.get("error", "Composio action failed"))
    return {"success": True, "data": result.get("data", {})}
