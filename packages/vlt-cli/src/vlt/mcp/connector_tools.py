"""Connector tools for the vlt MCP server.

Exposes the generic connector dispatch interface so agents can list
and invoke connectors enabled for their user session.

Tools registered:
    connector_list   — list connectors (native + Composio Hub apps)
    connector_call   — invoke a connector action with JSON params;
                       prefix connector name with 'composio:' for Hub apps
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

        Returns native connectors plus Composio Integration Hub apps (prefixed
        with 'composio:'). Use connector_call with name 'composio:gmail' to
        invoke a Composio action.

        Returns:
            {status, connectors: [{name, display_name, description, actions}], total}
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

            headers = {"Authorization": f"Bearer {sync_token}"}

            # Fetch native connectors
            native_connectors = []
            resp = httpx.get(f"{vault_url}/api/connectors", headers=headers, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                native_connectors = [
                    {
                        "name": c["name"],
                        "display_name": c["display_name"],
                        "description": c["description"],
                        "actions": c["actions"],
                    }
                    for c in data.get("connectors", [])
                    if c.get("enabled") and c.get("configured")
                ]
            elif resp.status_code != 200:
                return _err("API_ERROR", f"Backend returned HTTP {resp.status_code}: {resp.text[:200]}")

            # Fetch Composio connected apps (errors here don't block native connectors)
            composio_connectors = []
            try:
                hub_resp = httpx.get(f"{vault_url}/api/composio/connected", headers=headers, timeout=10.0)
                if hub_resp.status_code == 200:
                    hub_data = hub_resp.json()
                    for conn in hub_data.get("connections", []):
                        app_name = conn.get("app_name", "")
                        if app_name:
                            composio_connectors.append({
                                "name": f"composio:{app_name}",
                                "display_name": f"{app_name.title()} (via Composio)",
                                "description": (
                                    f"Composio-managed {app_name} integration. "
                                    f"Use 'vlt connectors hub actions {app_name}' to see available actions."
                                ),
                                "actions": [],
                            })
            except Exception:
                pass  # Composio hub errors don't affect native connectors

            all_connectors = native_connectors + composio_connectors
            return _ok(connectors=all_connectors, total=len(all_connectors))

        except Exception as e:
            logger.exception("connector_list failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def connector_call(connector: str, action: str, params: str = "{}") -> dict:
        """Invoke a connector action.

        Use connector_list first to discover available connectors and their
        parameter schemas.

        For Composio integrations, prefix the connector name with 'composio:',
        e.g. 'composio:gmail'. Action names for Composio use UPPER_SNAKE_CASE,
        e.g. 'GMAIL_SEND_EMAIL'.

        Args:
            connector: Connector name, e.g. "mailgun" or "composio:gmail"
            action: Action name, e.g. "send_email" or "GMAIL_SEND_EMAIL"
            params: JSON string of action parameters

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

            headers = {"Authorization": f"Bearer {sync_token}"}

            # Route composio: prefixed connectors to the Hub API
            if connector.startswith("composio:"):
                app_name = connector[len("composio:"):]
                resp = httpx.post(
                    f"{vault_url}/api/composio/{app_name}/invoke",
                    json={"action": action, "params": params_dict},
                    headers=headers,
                    timeout=30.0,
                )
                if resp.status_code >= 400:
                    return _err("API_ERROR", f"Composio Hub returned HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                return _ok(success=data.get("success", True), result=data.get("data", {}))

            # Native connector path
            resp = httpx.post(
                f"{vault_url}/api/connectors/{connector}/invoke",
                json={"action": action, "params": params_dict},
                headers=headers,
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
