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

    def app_auth_info(self, app_name: str) -> dict:
        """Query auth requirements for connecting an app.

        Returns {has_managed_auth, primary_auth_mode, auth_schemes} where each scheme
        contains integration_fields (operator-level) and user_fields (user-level).
        """
        toolset = self._toolset()
        app = toolset.client.apps.get(name=app_name.lower())
        has_managed = bool(app.testConnectors)
        schemes = []
        for scheme in (app.auth_schemes or []):
            integration_fields = []
            user_fields = []
            for f in scheme.fields:
                entry = {
                    "name": f.name,
                    "display_name": getattr(f, "display_name", None) or f.name,
                    "description": getattr(f, "description", ""),
                    "type": getattr(f, "type", "string"),
                    "required": f.required,
                    "expected_from_customer": f.expected_from_customer,
                }
                if f.expected_from_customer:
                    user_fields.append(entry)
                else:
                    integration_fields.append(entry)
            schemes.append({
                "auth_mode": scheme.auth_mode,
                "integration_fields": integration_fields,
                "user_fields": user_fields,
            })
        primary = schemes[0]["auth_mode"] if schemes else "OAUTH2"
        return {
            "has_managed_auth": has_managed,
            "primary_auth_mode": primary,
            "auth_schemes": schemes,
        }

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

    def initiate_connection(
        self,
        app_name: str,
        entity_id: str,
        label: str = "",
        auth_mode: str | None = None,
        auth_config: dict[str, str] | None = None,
        connected_account_params: dict[str, str] | None = None,
        redirect_url: str | None = None,
    ) -> dict:
        """Start connection flow for an app.

        Returns dict with {connection_id, redirect_url, status}.
        For OAuth apps, redirect_url points to the auth provider.
        For API_KEY apps, status is 'active' immediately (no redirect needed).
        """
        toolset = self._toolset()
        try:
            app = toolset.client.apps.get(name=app_name.lower())
            has_managed = bool(app.testConnectors)

            # Auto-detect auth mode from app's first scheme if not provided
            if not auth_mode:
                for scheme in (app.auth_schemes or []):
                    auth_mode = scheme.auth_mode
                    break

            entity = toolset.get_entity(id=entity_id)
            use_composio = has_managed and not auth_config

            request = entity.initiate_connection(
                app_name=app_name,
                auth_mode=auth_mode,
                auth_config=auth_config or {},
                use_composio_auth=use_composio,
                force_new_integration=bool(auth_config),
                connected_account_params=connected_account_params or {},
                redirect_url=redirect_url,
                labels=[label] if label else None,
            )

            return {
                "connection_id": getattr(request, "connectedAccountId", ""),
                "redirect_url": getattr(request, "redirectUrl", None) or "",
                "status": getattr(request, "connectionStatus", "initiated"),
            }
        except Exception as exc:
            logger.exception("Composio initiate_connection(%s, %s) failed", app_name, entity_id)
            raise RuntimeError(f"Failed to initiate connection: {exc}") from exc

    def disconnect_by_id(self, connection_id: str) -> None:
        """Disconnect a specific connection by its Composio ID using raw HTTP DELETE."""
        from composio.client.endpoints import v1

        toolset = self._toolset()
        try:
            toolset.client.http.delete(url=str(v1 / "connectedAccounts" / connection_id))
        except Exception as exc:
            logger.exception("Composio disconnect_by_id(%s) failed", connection_id)
            raise RuntimeError(f"Failed to disconnect connection: {exc}") from exc

    def disconnect(self, app_name: str, entity_id: str) -> int:
        """Disconnect all connections for an app for a user. Returns count disconnected."""
        toolset = self._toolset()
        try:
            entity = toolset.get_entity(id=entity_id)
            connections = entity.get_connections()
            count = 0
            for conn in connections:
                name = getattr(conn, "appName", "") or getattr(conn, "app_name", "")
                if name.lower() == app_name.lower():
                    conn_id = getattr(conn, "id", "")
                    if conn_id:
                        self.disconnect_by_id(conn_id)
                        count += 1
            if count == 0:
                logger.debug(
                    "Composio disconnect: no active connection for app=%s entity=%s (already disconnected)",
                    app_name,
                    entity_id,
                )
            return count
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
        connected_account_id: str | None = None,
    ) -> dict:
        """Execute a Composio action on behalf of a user.

        If connected_account_id is provided, routes to that specific connection.
        Otherwise the SDK picks the default connection for the entity.

        Returns dict with keys: success (bool), data (dict), error (str|None).
        """
        toolset = self._toolset()
        try:
            kwargs: dict[str, Any] = {
                "action": action_name.upper(),
                "params": params,
                "entity_id": entity_id,
            }
            if connected_account_id:
                kwargs["connected_account_id"] = connected_account_id
            result = toolset.execute_action(**kwargs)
            # Composio result has "successfull" (sic) key in some versions
            success = result.get("successfull", result.get("successful", True))
            data = result.get("data", result)
            error = result.get("error") if not success else None
            return {"success": success, "data": data, "error": error}
        except Exception as exc:
            logger.exception("Composio execute(%s/%s) failed", app_name, action_name)
            return {"success": False, "data": {}, "error": str(exc)}
