"""Pydantic models for the connector system."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class CredentialField(BaseModel):
    """A credential input field for a connector's config dialog."""
    name: str
    label: str
    secret: bool = True
    placeholder: str = ""
    field_type: str = "text"       # "text" | "select"
    options: list[str] = Field(default_factory=list)  # only used when field_type="select"


class ConnectorParam(BaseModel):
    """A single parameter for a connector action."""
    name: str
    description: str
    required: bool = False
    default: Optional[str] = None


class ConnectorAction(BaseModel):
    """A callable action exposed by a connector."""
    name: str
    description: str
    params: list[ConnectorParam] = Field(default_factory=list)


class ConnectorInfo(BaseModel):
    """Public info about a connector, including per-user state."""
    name: str
    display_name: str
    description: str
    credential_fields: list[CredentialField]
    actions: list[ConnectorAction]
    enabled: bool = False
    configured: bool = False


class ConnectorListResponse(BaseModel):
    connectors: list[ConnectorInfo] = Field(default_factory=list)


class ConnectorConfigUpdate(BaseModel):
    """Write one or more config keys for a connector."""
    config: dict[str, str] = Field(
        default_factory=dict,
        description="Map of config_key -> config_value. Set '__enabled' to 'true'/'false'.",
    )


class ConnectorInvokeRequest(BaseModel):
    action: str
    params: dict[str, Any] = Field(default_factory=dict)


class ConnectorInvokeResponse(BaseModel):
    success: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
