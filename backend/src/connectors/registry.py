"""Compatibility shim — use vlt_connectors.registry instead."""
from vlt_connectors.registry import ConnectorRegistry, get_registry

__all__ = ["ConnectorRegistry", "get_registry"]
