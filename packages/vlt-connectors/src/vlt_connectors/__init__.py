"""vlt-connectors: Shared connector system for Vlt-Bridge."""
from .base import BaseConnector, ActionConnector, ServiceConnector
from .models import CredentialField, ConnectorAction, ConnectorParam, ConnectorInfo
from .registry import ConnectorRegistry, get_registry
from .encryption import encrypt_value, decrypt_value, get_fernet, generate_key
from .daemon_client import CredentialCache
from .declarative import DeclarativeConnector, load_connector_from_toml, load_connectors_from_dir

__all__ = [
    "BaseConnector", "ActionConnector", "ServiceConnector",
    "CredentialField", "ConnectorAction", "ConnectorParam", "ConnectorInfo",
    "ConnectorRegistry", "get_registry",
    "encrypt_value", "decrypt_value", "get_fernet", "generate_key",
    "CredentialCache",
    "DeclarativeConnector", "load_connector_from_toml", "load_connectors_from_dir",
]
