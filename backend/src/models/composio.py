"""Pydantic models for Composio Connection Vault (024)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- Auth info (GET /auth-info response) ---

class AuthFieldInfo(BaseModel):
    name: str
    display_name: str
    description: str = ""
    type: str = "string"
    required: bool = False
    expected_from_customer: bool = False


class AuthSchemeInfo(BaseModel):
    auth_mode: str
    integration_fields: list[AuthFieldInfo] = []
    user_fields: list[AuthFieldInfo] = []


class AppAuthInfo(BaseModel):
    has_managed_auth: bool
    primary_auth_mode: str
    auth_schemes: list[AuthSchemeInfo] = []


# --- Connect request/response ---

class ConnectRequest(BaseModel):
    label: str = Field(default="", max_length=100)
    auth_mode: str | None = None
    auth_config: dict[str, str] | None = None
    connected_account_params: dict[str, str] | None = None
    redirect_url: str | None = None


class ConnectResponse(BaseModel):
    app: str
    connection_id: str
    label: str = ""
    redirect_url: str | None = None
    status: str = "initiated"


# --- Connection info (list endpoint) ---

class ConnectionInfo(BaseModel):
    connection_id: str
    app_name: str
    label: str = ""
    auth_mode: str = ""
    status: str = "active"
    created_at: str = ""


# --- Invoke request (updated with connection_id) ---

class InvokeRequest(BaseModel):
    action: str
    params: dict[str, Any] = {}
    connection_id: str | None = None
