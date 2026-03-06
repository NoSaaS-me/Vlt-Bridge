"""API routes for the connector system."""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException

from ..middleware import AuthContext, require_auth_context
from ...connectors.registry import get_registry
from ...models.connectors import (
    ConnectorInfo,
    ConnectorListResponse,
    ConnectorConfigUpdate,
    ConnectorInvokeRequest,
    ConnectorInvokeResponse,
)
from ...services.connector_service import ConnectorService, get_connector_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/connectors", tags=["connectors"])


@router.get("", response_model=ConnectorListResponse)
async def list_connectors(
    auth: AuthContext = Depends(require_auth_context),
    svc: ConnectorService = Depends(get_connector_service),
) -> ConnectorListResponse:
    """List all registered connectors with per-user enabled/configured state."""
    registry = get_registry()
    infos = []
    for connector in registry.list_all():
        enabled = svc.is_enabled(auth.user_id, connector.name)
        configured = svc.is_configured(auth.user_id, connector)
        infos.append(ConnectorInfo(
            name=connector.name,
            display_name=connector.display_name,
            description=connector.description,
            credential_fields=connector.credential_fields,
            actions=connector.actions,
            enabled=enabled,
            configured=configured,
        ))
    return ConnectorListResponse(connectors=infos)


@router.get("/{connector_name}/config")
async def get_connector_config(
    connector_name: str,
    auth: AuthContext = Depends(require_auth_context),
    svc: ConnectorService = Depends(get_connector_service),
) -> dict:
    """Get a connector's config for the authenticated user (secrets masked)."""
    registry = get_registry()
    connector = registry.get(connector_name)
    if not connector:
        raise HTTPException(status_code=404, detail=f"Connector '{connector_name}' not found")

    raw = svc.get_config(auth.user_id, connector_name)
    secret_field_names = {f.name for f in connector.credential_fields if f.secret}
    masked = {
        k: ("••••••••" if k in secret_field_names and v else v)
        for k, v in raw.items()
    }
    return {"connector": connector_name, "config": masked}


@router.put("/{connector_name}/config")
async def update_connector_config(
    connector_name: str,
    body: ConnectorConfigUpdate,
    auth: AuthContext = Depends(require_auth_context),
    svc: ConnectorService = Depends(get_connector_service),
) -> dict:
    """Save connector config (credentials + __enabled flag) for the authenticated user."""
    registry = get_registry()
    if not registry.get(connector_name):
        raise HTTPException(status_code=404, detail=f"Connector '{connector_name}' not found")
    # Filter out masked placeholder values so existing encrypted secrets are not overwritten.
    filtered = {k: v for k, v in body.config.items() if v != "••••••••"}
    svc.set_config(auth.user_id, connector_name, filtered)
    return {"connector": connector_name, "saved": True}


@router.get("/{connector_name}/credentials")
async def get_connector_credentials(
    connector_name: str,
    auth: AuthContext = Depends(require_auth_context),
    svc: ConnectorService = Depends(get_connector_service),
) -> dict:
    """
    Get decrypted credentials for a connector (for daemon/service use).
    Returns only non-metadata keys (no __ prefixed keys).
    Requires a valid Bearer token — daemon uses LOCAL_USER_ID-scoped token.
    """
    registry = get_registry()
    connector = registry.get(connector_name)
    if not connector:
        raise HTTPException(status_code=404, detail=f"Connector '{connector_name}' not found")

    credentials = svc.get_credentials(auth.user_id, connector_name)
    return {"connector": connector_name, "credentials": credentials}


@router.post("/{connector_name}/invoke", response_model=ConnectorInvokeResponse)
async def invoke_connector(
    connector_name: str,
    body: ConnectorInvokeRequest,
    auth: AuthContext = Depends(require_auth_context),
    svc: ConnectorService = Depends(get_connector_service),
) -> ConnectorInvokeResponse:
    """Invoke a connector action on behalf of the authenticated user."""
    registry = get_registry()
    connector = registry.get(connector_name)
    if not connector:
        raise HTTPException(status_code=404, detail=f"Connector '{connector_name}' not found")

    if not svc.is_enabled(auth.user_id, connector_name):
        raise HTTPException(status_code=403, detail=f"Connector '{connector_name}' is not enabled for this user")

    credentials = svc.get_credentials(auth.user_id, connector_name)

    try:
        result = await connector.invoke(body.action, body.params, credentials)
        success = result.get("success", True)
        error = result.get("error") if not success else None
        payload = {k: v for k, v in result.items() if k not in ("success", "error")}
        return ConnectorInvokeResponse(success=success, result=payload, error=error)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Connector invoke failed: %s/%s", connector_name, body.action)
        raise HTTPException(status_code=502, detail=str(exc))
