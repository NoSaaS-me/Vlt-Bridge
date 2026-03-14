"""Composio Integration Hub service connector.

Wraps the Composio SDK to provide a managed 100+ app integration catalog.
Users connect apps via Composio's OAuth flow; this service brokers invocations.

COMPOSIO_API_KEY env variable is the operator's global API key.
Per-user state is keyed by entity_id = user_id in Composio.
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar

from ..base import ServiceConnector
from ..models import CredentialField

logger = logging.getLogger(__name__)


class ComposioService(ServiceConnector):
    """Service connector for Composio Integration Hub.

    Provides access to 100+ pre-built app integrations.
    API key is operator-level (env var), not per-user.
    User connections are keyed by entity_id (= user_id).
    """

    name = "composio"
    display_name = "Composio Integration Hub"
    description = (
        "Managed integration hub providing 100+ pre-built app connectors. "
        "Connect Gmail, Slack, Linear, Notion, GitHub, Stripe, and more via OAuth — "
        "no per-user API key needed."
    )
    connector_type: ClassVar[str] = "service"
    available_contexts: ClassVar[list[str]] = ["backend", "daemon"]

    # No per-user credential fields — API key is operator-level env var
    credential_fields: ClassVar[list[CredentialField]] = []

    def _api_key(self) -> str:
        key = os.environ.get("COMPOSIO_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "COMPOSIO_API_KEY environment variable is not set. "
                "Get your API key from https://app.composio.dev/settings."
            )
        return key

    def _toolset(self):
        """Return a ComposioToolSet instance. Import is lazy to avoid dep errors if composio not installed.

        The instance is cached on self._toolset_instance to avoid rebuilding the SDK client
        on every call.
        """
        if not hasattr(self, "_toolset_instance") or self._toolset_instance is None:
            try:
                from composio import ComposioToolSet
            except ImportError:
                raise RuntimeError(
                    "composio-core is not installed. Run: uv pip install composio-core"
                )
            self._toolset_instance = ComposioToolSet(api_key=self._api_key())
        return self._toolset_instance

    def is_configured(self) -> bool:
        """True if COMPOSIO_API_KEY is set."""
        return bool(os.environ.get("COMPOSIO_API_KEY", "").strip())

    def catalog(self) -> list[dict]:
        """List all available apps in the Composio catalog.

        Returns list of dicts with keys: name, display_name, description, categories.
        """
        toolset = self._toolset()
        try:
            apps = toolset.client.apps.get()
            return [
                {
                    "name": getattr(a, "name", ""),
                    "display_name": getattr(a, "display_name", "") or getattr(a, "name", ""),
                    "description": getattr(a, "description", "") or "",
                    "categories": list(getattr(a, "categories", []) or []),
                }
                for a in apps
                if getattr(a, "name", "") and not getattr(a, "no_auth", False)
            ]
        except Exception as exc:
            logger.exception("Composio catalog() failed")
            raise RuntimeError(f"Failed to fetch Composio app catalog: {exc}") from exc

    def connected(self, entity_id: str) -> list[dict]:
        """List apps connected by a specific user (entity).

        Returns list of dicts with keys: app_name, status, connection_id.
        """
        toolset = self._toolset()
        try:
            entity = toolset.get_entity(id=entity_id)
            connections = entity.get_connections()
            return [
                {
                    "app_name": getattr(c, "appName", "") or getattr(c, "app_name", ""),
                    "status": getattr(c, "status", "unknown"),
                    "connection_id": getattr(c, "id", ""),
                }
                for c in connections
            ]
        except Exception as exc:
            logger.exception("Composio connected(%s) failed", entity_id)
            raise RuntimeError(f"Failed to fetch connected apps for user: {exc}") from exc

    def initiate_connection(self, app_name: str, entity_id: str) -> str:
        """Start OAuth flow for an app. Returns the redirect URL to send the user to."""
        toolset = self._toolset()
        try:
            entity = toolset.get_entity(id=entity_id)
            request = entity.initiate_connection(app_name=app_name)
            redirect_url: str = (
                getattr(request, "redirectUrl", None)
                or getattr(request, "redirect_url", None)
                or ""
            )
            if not redirect_url:
                raise RuntimeError("Composio returned no redirect URL")
            return redirect_url
        except Exception as exc:
            logger.exception("Composio initiate_connection(%s, %s) failed", app_name, entity_id)
            raise RuntimeError(f"Failed to initiate connection: {exc}") from exc

    def disconnect(self, app_name: str, entity_id: str) -> None:
        """Disconnect an app for a user."""
        toolset = self._toolset()
        try:
            entity = toolset.get_entity(id=entity_id)
            connections = entity.get_connections()
            for conn in connections:
                name = getattr(conn, "appName", "") or getattr(conn, "app_name", "")
                if name.lower() == app_name.lower():
                    conn_id = getattr(conn, "id", "")
                    if conn_id:
                        toolset.client.connected_accounts.delete(connection_id=conn_id)
                        return
            # No connection found — silently succeed (already disconnected)
            logger.debug(
                "Composio disconnect: no active connection for app=%s entity=%s (already disconnected)",
                app_name,
                entity_id,
            )
            return
        except Exception as exc:
            logger.exception("Composio disconnect(%s, %s) failed", app_name, entity_id)
            raise RuntimeError(f"Failed to disconnect app: {exc}") from exc

    def get_actions(self, app_name: str) -> list[dict]:
        """List actions available for an app.

        Returns list of dicts with keys: name, display_name, description, parameters.
        """
        toolset = self._toolset()
        try:
            actions = toolset.client.actions.get(apps=[app_name])
            return [
                {
                    "name": getattr(a, "name", ""),
                    "display_name": getattr(a, "display_name", "") or getattr(a, "name", ""),
                    "description": getattr(a, "description", "") or "",
                    "parameters": getattr(a, "parameters", {}) or {},
                }
                for a in actions
                if getattr(a, "name", "")
            ]
        except Exception as exc:
            logger.exception("Composio get_actions(%s) failed", app_name)
            raise RuntimeError(f"Failed to fetch actions for app '{app_name}': {exc}") from exc

    def execute(
        self,
        app_name: str,
        action_name: str,
        params: dict[str, Any],
        entity_id: str,
    ) -> dict:
        """Execute a Composio action on behalf of a user.

        Returns dict with keys: success (bool), data (dict), error (str|None).

        Note: Unlike other methods, this never raises — exceptions are caught and
        returned as {"success": False, "data": {}, "error": str(exc)}.
        """
        toolset = self._toolset()
        try:
            result = toolset.execute_action(
                action=action_name.upper(),  # Composio uses UPPER_SNAKE_CASE action names
                params=params,
                entity_id=entity_id,
            )
            # Composio result has "successfull" (sic) key in some versions
            success = result.get("successfull", result.get("successful", True))
            data = result.get("data", result)
            error = result.get("error") if not success else None
            return {"success": success, "data": data, "error": error}
        except Exception as exc:
            logger.exception("Composio execute(%s/%s) failed", app_name, action_name)
            return {"success": False, "data": {}, "error": str(exc)}
