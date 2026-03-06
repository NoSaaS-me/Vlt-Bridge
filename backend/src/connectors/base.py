"""Compatibility shim — use vlt_connectors.base instead."""
from vlt_connectors.base import BaseConnector, ActionConnector, ServiceConnector

__all__ = ["BaseConnector", "ActionConnector", "ServiceConnector"]
