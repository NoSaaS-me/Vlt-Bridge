"""API routes for user settings (oracle toggle, etc.)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..middleware import AuthContext, get_auth_context
from ...models.settings import OracleSettings, OracleSettingsUpdate
from ...services.user_settings import UserSettingsService, get_user_settings_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/oracle", response_model=OracleSettings)
async def get_oracle_settings(
    auth: AuthContext = Depends(get_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
) -> OracleSettings:
    """Get oracle MCP toggle state for the authenticated user."""
    enabled = settings_service.get_oracle_mcp_enabled(auth.user_id)
    return OracleSettings(oracle_mcp_enabled=enabled)


@router.put("/oracle", response_model=OracleSettings)
async def update_oracle_settings(
    body: OracleSettingsUpdate,
    auth: AuthContext = Depends(get_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
) -> OracleSettings:
    """Update oracle MCP toggle state for the authenticated user."""
    settings_service.set_oracle_mcp_enabled(auth.user_id, body.oracle_mcp_enabled)
    return OracleSettings(oracle_mcp_enabled=body.oracle_mcp_enabled)
