"""Connector tools for the vlt MCP server.

Exposes the generic connector dispatch interface so agents can list
and invoke connectors enabled for their user session.

Tools registered:
    connector_list    — list connectors (native + Composio Hub apps)
    connector_actions — list available actions + parameter schemas for a connector
    connector_call    — invoke a connector action with JSON params;
                        prefix connector name with 'composio:' for Hub apps

Typical agent workflow:
    1. connector_list           → discover which connectors are available
    2. connector_actions(name)  → get action names + parameter schemas
    3. connector_call(name, action, params)  → execute
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def register_connector_tools(mcp) -> None:
    """Register connector tools onto a FastMCP server instance."""

    def _get_backend(settings):
        """Return (vault_url, headers) or raise ValueError if not configured."""
        vault_url = getattr(settings, "vault_url", None) or getattr(settings, "sync_url", None)
        sync_token = getattr(settings, "sync_token", None)
        if not vault_url or not sync_token:
            raise ValueError(
                "Document-MCP backend not configured. Set VLT_VAULT_URL and VLT_SYNC_TOKEN."
            )
        return vault_url, {"Authorization": f"Bearer {sync_token}"}

    @mcp.tool()
    def connector_list() -> dict:
        """List connectors available for the current user.

        Returns native connectors (fully configured) plus Composio Integration Hub
        apps the user has connected. Each Composio connector includes a summary of
        its available actions so you know what you can call.

        To get the full parameter schema for a specific connector's actions, call
        connector_actions(connector_name).

        Returns:
            {status, connectors: [{name, display_name, description, actions: [{name, description}]}], total}
        """
        import httpx
        from vlt.mcp import _err, _ok
        from vlt.config import get_settings

        try:
            settings = get_settings()
            vault_url, headers = _get_backend(settings)
        except ValueError as e:
            return _err("NOT_CONFIGURED", str(e))
        except Exception as e:
            return _err("INTERNAL_ERROR", str(e))

        try:
            # Native connectors
            native_connectors = []
            resp = httpx.get(f"{vault_url}/api/connectors", headers=headers, timeout=10.0)
            if resp.status_code == 200:
                native_connectors = [
                    {
                        "name": c["name"],
                        "display_name": c["display_name"],
                        "description": c["description"],
                        "actions": c["actions"],
                    }
                    for c in resp.json().get("connectors", [])
                    if c.get("enabled") and c.get("configured")
                ]
            elif resp.status_code >= 400:
                return _err("API_ERROR", f"Backend returned HTTP {resp.status_code}: {resp.text[:200]}")

            # Composio connected apps — fetch actions for each so agents know what's callable
            composio_connectors = []
            hub_resp = httpx.get(f"{vault_url}/api/composio/connected", headers=headers, timeout=10.0)
            if hub_resp.status_code == 200:
                connections = hub_resp.json().get("connections", [])
                for conn in connections:
                    app_name = conn.get("app_name", "").lower()
                    if not app_name:
                        continue
                    # Fetch action summaries for this app (name + description only, no schemas)
                    # This gives agents enough to pick an action; use connector_actions for schemas.
                    action_summaries: list[dict] = []
                    try:
                        act_resp = httpx.get(
                            f"{vault_url}/api/composio/{app_name}/actions",
                            headers=headers,
                            timeout=15.0,
                        )
                        if act_resp.status_code == 200:
                            action_summaries = [
                                {"name": a["name"], "description": a.get("description", "")}
                                for a in act_resp.json().get("actions", [])
                            ]
                    except Exception:
                        pass  # Missing action list is non-fatal

                    composio_connectors.append({
                        "name": f"composio:{app_name}",
                        "display_name": f"{app_name.title()} (via Composio)",
                        "description": (
                            f"Composio-managed {app_name} integration. "
                            f"{len(action_summaries)} actions available. "
                            "Call connector_actions('composio:" + app_name + "') for full parameter schemas."
                        ),
                        "actions": action_summaries,
                    })

            all_connectors = native_connectors + composio_connectors
            return _ok(connectors=all_connectors, total=len(all_connectors))

        except Exception as e:
            logger.exception("connector_list failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def connector_actions(connector: str) -> dict:
        """Get the full list of actions and parameter schemas for a connector.

        Use this after connector_list to discover exactly what parameters an action
        needs before calling connector_call.

        For Composio connectors, action names use UPPER_SNAKE_CASE (e.g. GMAIL_SEND_EMAIL).
        The 'parameters' field contains a JSON Schema describing required/optional fields.

        Args:
            connector: Connector name, e.g. "mailgun" or "composio:gmail"

        Returns:
            {status, connector, actions: [{name, display_name, description, parameters}], total}
        """
        import httpx
        from vlt.mcp import _err, _ok
        from vlt.config import get_settings

        try:
            settings = get_settings()
            vault_url, headers = _get_backend(settings)
        except ValueError as e:
            return _err("NOT_CONFIGURED", str(e))
        except Exception as e:
            return _err("INTERNAL_ERROR", str(e))

        try:
            if connector.startswith("composio:"):
                app_name = connector[len("composio:"):]
                resp = httpx.get(
                    f"{vault_url}/api/composio/{app_name}/actions",
                    headers=headers,
                    timeout=15.0,
                )
                if resp.status_code == 404:
                    return _err("NOT_FOUND", f"No Composio app '{app_name}' found or it is not connected.")
                if resp.status_code >= 400:
                    return _err("API_ERROR", f"Backend returned HTTP {resp.status_code}: {resp.text[:200]}")
                actions = resp.json().get("actions", [])
                return _ok(connector=connector, actions=actions, total=len(actions))

            # Native connector: actions are returned inline from /api/connectors
            resp = httpx.get(f"{vault_url}/api/connectors", headers=headers, timeout=10.0)
            if resp.status_code >= 400:
                return _err("API_ERROR", f"Backend returned HTTP {resp.status_code}: {resp.text[:200]}")
            connectors = resp.json().get("connectors", [])
            match = next((c for c in connectors if c["name"] == connector), None)
            if not match:
                return _err("NOT_FOUND", f"Connector '{connector}' not found or not enabled.")
            return _ok(connector=connector, actions=match.get("actions", []), total=len(match.get("actions", [])))

        except Exception as e:
            logger.exception("connector_actions failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def connector_call(connector: str, action: str, params: str = "{}") -> dict:
        """Invoke a connector action.

        Recommended workflow:
            1. connector_list()              — see which connectors are available
            2. connector_actions(connector)  — get action names + parameter schemas
            3. connector_call(connector, action, params)  — execute

        For Composio integrations, prefix the connector name with 'composio:',
        e.g. 'composio:gmail'. Action names for Composio use UPPER_SNAKE_CASE,
        e.g. 'GMAIL_SEND_EMAIL'. Use connector_actions to get the exact names and
        required parameters before calling.

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
