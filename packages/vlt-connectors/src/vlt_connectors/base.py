"""Connector base classes."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field
from typing import Any, ClassVar, Optional
from .models import CredentialField, ConnectorAction, ConnectorInfo


@dataclass
class WebhookEventDef:
    """Definition of one type of webhook event this connector can emit."""
    event_type: str
    description: str
    fields: list[str]
    example_fields: dict[str, str] = dataclass_field(default_factory=dict)


@dataclass
class ConnectorEvent:
    """A parsed webhook event from a connector."""
    connector: str
    event_type: str
    timestamp: str
    fields: dict[str, str]
    raw_payload: dict


class BaseConnector(ABC):
    """All connector types inherit from this."""
    name: ClassVar[str]
    display_name: ClassVar[str]
    description: ClassVar[str]
    connector_type: ClassVar[str] = "action"
    credential_fields: ClassVar[list[CredentialField]] = []
    actions: ClassVar[list[ConnectorAction]] = []

    def get_info(self, enabled: bool = False, configured: bool = False) -> ConnectorInfo:
        return ConnectorInfo(
            name=self.name,
            display_name=self.display_name,
            description=self.description,
            connector_type=self.connector_type,
            credential_fields=self.credential_fields,
            actions=self.actions,
            enabled=enabled,
            configured=configured,
            auth_type=getattr(self, "auth_type", "api_key"),
        )


class ActionConnector(BaseConnector):
    """User/agent-triggered action connectors. Exposed via connector_call MCP tool."""
    connector_type: ClassVar[str] = "action"
    webhook_events: ClassVar[list[WebhookEventDef]] = []  # non-empty = also a webhook source

    @abstractmethod
    async def invoke(self, action: str, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        """Invoke a named action with user-provided params and decrypted credentials."""
        ...

    def verify_webhook(self, headers: dict[str, str], body_bytes: bytes, credentials: dict[str, str]) -> bool:
        """Verify webhook signature. Default: no verification (subclass overrides)."""
        return True

    def parse_webhook(self, headers: dict[str, str], body_bytes: bytes, credentials: dict[str, str]) -> Optional[ConnectorEvent]:
        """Parse raw webhook into a ConnectorEvent. Default: not implemented."""
        return None


class ServiceConnector(BaseConnector):
    """System-internal service connectors. Per-user credentials, but called by system components."""
    connector_type: ClassVar[str] = "service"
    available_contexts: ClassVar[list[str]] = ["backend"]
    # subclasses override with ["backend", "daemon", "cli"] as appropriate
