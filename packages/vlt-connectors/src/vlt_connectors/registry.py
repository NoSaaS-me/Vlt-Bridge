"""Connector registry supporting action and service connector types."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base import ActionConnector, ServiceConnector

logger = logging.getLogger(__name__)

_registry: "ConnectorRegistry | None" = None


class ConnectorRegistry:
    def __init__(self) -> None:
        self._action: dict[str, "ActionConnector"] = {}
        self._service: dict[str, "ServiceConnector"] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load built-in connectors."""
        try:
            from .connectors.mailgun import MailgunConnector
            self.register_action(MailgunConnector())
        except ImportError as e:
            logger.warning("Could not load MailgunConnector: %s", e)
        try:
            from .service.huggingface import HuggingFaceInferenceConnector
            self.register_service(HuggingFaceInferenceConnector())
        except ImportError:
            pass
        try:
            from .service.ollama import OllamaConnector
            self.register_service(OllamaConnector())
        except ImportError:
            pass
        # Load built-in declarative TOML connectors
        from pathlib import Path
        from .declarative import load_connectors_from_dir
        connectors_dir = Path(__file__).parent / "connectors"
        if connectors_dir.exists():
            for c in load_connectors_from_dir(connectors_dir):
                self.register_action(c)
        try:
            from .connectors.github_oauth import GitHubOAuthConnector
            self.register_action(GitHubOAuthConnector())
        except ImportError as e:
            logger.warning("Could not load GitHubOAuthConnector: %s", e)

    def register_action(self, connector: "ActionConnector") -> None:
        self._action[connector.name] = connector
        logger.debug("Registered action connector: %s", connector.name)

    def register_service(self, connector: "ServiceConnector") -> None:
        self._service[connector.name] = connector
        logger.debug("Registered service connector: %s", connector.name)

    def get_action(self, name: str) -> "ActionConnector | None":
        return self._action.get(name)

    def get_service(self, name: str) -> "ServiceConnector | None":
        return self._service.get(name)

    def get(self, name: str) -> "ActionConnector | None":
        """Compatibility shim: look up by name across action connectors."""
        return self._action.get(name)

    def list_action_connectors(self) -> list["ActionConnector"]:
        return list(self._action.values())

    def list_service_connectors(self) -> list["ServiceConnector"]:
        return list(self._service.values())

    def list_all(self) -> list["ActionConnector"]:
        """Compatibility shim: return all action connectors (matches old backend registry API)."""
        return list(self._action.values())


def get_registry() -> ConnectorRegistry:
    global _registry
    if _registry is None:
        _registry = ConnectorRegistry()
    return _registry
