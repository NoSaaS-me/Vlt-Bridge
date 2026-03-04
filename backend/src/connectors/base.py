"""Abstract base class for connectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from ..models.connectors import ConnectorAction, CredentialField


class BaseConnector(ABC):
    """All connectors extend this. Define name, credential_fields, actions, and invoke()."""

    name: str
    display_name: str
    description: str
    credential_fields: list[CredentialField]
    actions: list[ConnectorAction]

    @abstractmethod
    async def invoke(self, action: str, params: dict, credentials: dict) -> dict:
        """Execute the given action with params and credentials. Return a result dict."""
        ...
