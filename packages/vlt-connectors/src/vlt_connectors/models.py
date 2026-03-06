"""Shared connector Pydantic models."""
from __future__ import annotations
from pydantic import BaseModel, Field


class CredentialField(BaseModel):
    name: str
    label: str
    secret: bool = True
    placeholder: str = ""
    field_type: str = "text"          # "text" | "select"
    options: list[str] = Field(default_factory=list)


class ConnectorParam(BaseModel):
    name: str
    description: str
    required: bool = True
    default: str | None = None


class ConnectorAction(BaseModel):
    name: str
    description: str
    params: list[ConnectorParam] = Field(default_factory=list)


class ConnectorInfo(BaseModel):
    name: str
    display_name: str
    description: str
    connector_type: str = "action"    # "action" | "service" | "webhook"
    credential_fields: list[CredentialField] = Field(default_factory=list)
    actions: list[ConnectorAction] = Field(default_factory=list)
    enabled: bool = False
    configured: bool = False
    auth_type: str = "api_key"        # "api_key" | "oauth2"
