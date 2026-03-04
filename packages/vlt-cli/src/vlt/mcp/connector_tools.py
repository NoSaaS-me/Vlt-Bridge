"""Connector tools for the vlt MCP server.

Exposes the generic connector dispatch interface so agents can list
and invoke connectors enabled for their user session.

Tools registered:
    connector_list   — list connectors available to the current user
    connector_call   — invoke a connector action with JSON params
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def register_connector_tools(mcp) -> None:
    """Register connector tools onto a FastMCP server instance."""

    @mcp.tool()
    def connector_list() -> dict:
        """List connectors enabled for the current user.

        Returns each connector's name, description, available actions,
        and parameter schemas — so you know exactly what to pass to
        connector_call.

        Returns:
            {status, connectors: [{name, display_name, description, actions: [{name, description, params}]}]}
        """
        from vlt.mcp import _ok, _err

        try:
            from vlt.config import get_settings
            import httpx

            settings = get_settings()
            vault_url = getattr(settings, "vault_url", None) or getattr(settings, "sync_url", None)
            sync_token = getattr(settings, "sync_token", None)

            if not vault_url or not sync_token:
                return _err(
                    "NOT_CONFIGURED",
                    "Document-MCP backend not configured. Set VLT_VAULT_URL and VLT_SYNC_TOKEN.",
                )

            resp = httpx.get(
                f"{vault_url}/api/connectors",
                headers={"Authorization": f"Bearer {sync_token}"},
                timeout=10.0,
            )
            if resp.status_code != 200:
                return _err("API_ERROR", f"Backend returned HTTP {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            # Filter to only enabled+configured connectors for the agent
            enabled = [
                {
                    "name": c["name"],
                    "display_name": c["display_name"],
                    "description": c["description"],
                    "actions": c["actions"],
                }
                for c in data.get("connectors", [])
                if c.get("enabled") and c.get("configured")
            ]
            return _ok(connectors=enabled, total=len(enabled))

        except Exception as e:
            logger.exception("connector_list failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def connector_call(connector: str, action: str, params: str = "{}") -> dict:
        """Invoke a connector action.

        Use connector_list first to discover available connectors and their
        parameter schemas.

        Args:
            connector: Connector name, e.g. "mailgun"
            action: Action name, e.g. "send_email"
            params: JSON string of action parameters, e.g. '{"to": "user@example.com", "subject": "Hi", "body": "Hello"}'

        Returns:
            {status, success, result} on success or {status: "error", ...} on failure
        """
        from vlt.mcp import _ok, _err

        try:
            try:
                params_dict = json.loads(params) if params else {}
            except json.JSONDecodeError as e:
                return _err("INVALID_PARAMS", f"params must be valid JSON: {e}")

            from vlt.config import get_settings
            import httpx

            settings = get_settings()
            vault_url = getattr(settings, "vault_url", None) or getattr(settings, "sync_url", None)
            sync_token = getattr(settings, "sync_token", None)

            if not vault_url or not sync_token:
                return _err(
                    "NOT_CONFIGURED",
                    "Document-MCP backend not configured. Set VLT_VAULT_URL and VLT_SYNC_TOKEN.",
                )

            resp = httpx.post(
                f"{vault_url}/api/connectors/{connector}/invoke",
                json={"action": action, "params": params_dict},
                headers={"Authorization": f"Bearer {sync_token}"},
                timeout=30.0,
            )

            if resp.status_code == 404:
                return _err("CONNECTOR_NOT_FOUND", f"Connector '{connector}' not found or not enabled for your account")
            if resp.status_code == 403:
                return _err("CONNECTOR_DISABLED", f"Connector '{connector}' is not enabled. Configure it in the Connectors page.")
            if resp.status_code >= 400:
                return _err("API_ERROR", f"Backend returned HTTP {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            return _ok(success=data.get("success", True), result=data.get("result", {}))

        except Exception as e:
            logger.exception("connector_call failed")
            return _err("INTERNAL_ERROR", str(e))
