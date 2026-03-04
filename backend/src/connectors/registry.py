"""Global connector registry — singleton."""
from __future__ import annotations

from typing import Optional, Type
from .base import BaseConnector

_registry: Optional["ConnectorRegistry"] = None


class ConnectorRegistry:
    def __init__(self):
        self._connectors: dict[str, Type[BaseConnector]] = {}

    def register(self, connector_class: Type[BaseConnector]) -> None:
        self._connectors[connector_class.name] = connector_class

    def get(self, name: str) -> Optional[BaseConnector]:
        cls = self._connectors.get(name)
        return cls() if cls else None

    def list_all(self) -> list[BaseConnector]:
        return [cls() for cls in self._connectors.values()]


def get_registry() -> ConnectorRegistry:
    global _registry
    if _registry is None:
        _registry = ConnectorRegistry()
        _load_builtin_connectors(_registry)
    return _registry


def _load_builtin_connectors(registry: ConnectorRegistry) -> None:
    from .mailgun import MailgunConnector
    registry.register(MailgunConnector)
