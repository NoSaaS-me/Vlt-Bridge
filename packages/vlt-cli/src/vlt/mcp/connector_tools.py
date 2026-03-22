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
        apps the user has connected. Each connector includes a summary of its
        available actions so you know what you can call.

        Composio connectors are prefixed with ``composio:`` (e.g. ``composio:gmail``).
        Use this exact prefixed name when calling connector_actions or connector_call.

        Each action has a ``permission`` field:
          - ``"allow"`` — agent can call freely (default)
          - ``"ask"``   — agent must request user approval before executing
          - ``"off"``   — action is disabled and hidden (not returned in this list)

        To get parameter schemas before calling an action, use
        connector_actions(connector_name).

        Returns:
            {status, connectors: [{name, display_name, description,
             actions: [{name, description, permission}]}], total}
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

        def _fetch_config(connector_name: str) -> dict:
            """Fetch raw connector config dict; returns {} on failure."""
            try:
                cfg_resp = httpx.get(
                    f"{vault_url}/api/connectors/{connector_name}/config",
                    headers=headers,
                    timeout=5.0,
                )
                if cfg_resp.status_code == 200:
                    return cfg_resp.json().get("config", {})
            except Exception:
                pass
            return {}

        def _annotate_and_filter_actions(actions: list[dict], cfg: dict) -> list[dict]:
            """Attach 'permission' to each action and strip any set to 'off'."""
            annotated = []
            for act in actions:
                perm = cfg.get(f"__action_{act['name']}", "allow")
                if perm == "off":
                    continue
                annotated.append({**act, "permission": perm})
            return annotated

        try:
            # Native connectors
            native_connectors = []
            resp = httpx.get(f"{vault_url}/api/connectors", headers=headers, timeout=10.0)
            if resp.status_code == 200:
                for c in resp.json().get("connectors", []):
                    if not (c.get("enabled") and c.get("configured")):
                        continue
                    cfg = _fetch_config(c["name"])
                    actions = _annotate_and_filter_actions(c.get("actions", []), cfg)
                    native_connectors.append({
                        "name": c["name"],
                        "display_name": c["display_name"],
                        "description": c["description"],
                        "actions": actions,
                    })
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

                    # Annotate composio actions with per-action permissions
                    composio_cfg = _fetch_config(f"composio:{app_name}")
                    action_summaries = _annotate_and_filter_actions(action_summaries, composio_cfg)

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

        Call this before connector_call to discover exactly what parameters an
        action expects. Works for both connected and unconnected Composio apps
        (useful for previewing capabilities before connecting).

        For Composio connectors, use the ``composio:`` prefix (e.g. ``composio:gmail``).
        Composio action names use UPPER_SNAKE_CASE (e.g. ``GMAIL_SEND_EMAIL``).

        The ``parameters`` field on each action is a JSON Schema object with
        ``type``, ``properties``, and ``required`` keys describing the expected
        input. Use this schema to construct the ``params`` JSON string for
        connector_call.

        Args:
            connector: Connector name, e.g. ``"mailgun"`` or ``"composio:gmail"``.

        Returns:
            {status, connector, actions: [{name, display_name, description,
             parameters: {type, properties, required}}], total}
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
    def connector_call(connector: str, action: str, params: str = "{}", instance_id: str = "default") -> dict:
        """Invoke a connector action.

        Recommended workflow:
            1. connector_list()              — discover available connectors
            2. connector_actions(connector)  — get action names + parameter schemas
            3. connector_call(connector, action, params)  — execute

        For Composio integrations, prefix the connector with ``composio:``
        (e.g. ``composio:gmail``). Action names use UPPER_SNAKE_CASE
        (e.g. ``GMAIL_SEND_EMAIL``). Always call connector_actions first to get
        exact names and required parameters.

        **WARNING**: ``params`` must be a JSON **string**, not a dict/object.
        Example: ``params='{"to": "user@example.com", "subject": "Hello"}'``

        Permission behaviour (set by the user in Connectors settings):
          - ``allow`` → action executes normally
          - ``ask`` → returns ``{success: false, result: {requires_approval: true}}``
            with a message asking you to tell the user to approve it
          - ``off`` → returns an ACTION_DISABLED error

        Args:
            connector:   Connector name, e.g. ``"mailgun"`` or ``"composio:gmail"``.
            action:      Action name, e.g. ``"send_email"`` or ``"GMAIL_SEND_EMAIL"``.
            params:      **JSON string** of action parameters (default ``"{}"``).
            instance_id: Instance of the connector to use (default ``"default"``).
                         Multi-instance support allows separate credential sets for
                         the same connector type (e.g. two Mailgun accounts).

        Returns:
            {status, success, result} on success.
            {status, success: false, result: {requires_approval: true}, message} when permission is "ask".
            {status: "error", error_code, message} on failure.
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

            # Check per-action permission before dispatching to the backend.
            # For composio: connectors, config is stored under "composio:{app_name}".
            # A 404 on the config endpoint means no config exists → default "allow".
            config_connector = connector  # e.g. "mailgun" or "composio:gmail"
            try:
                cfg_resp = httpx.get(
                    f"{vault_url}/api/connectors/{config_connector}/instances/{instance_id}/config",
                    headers=headers,
                    timeout=5.0,
                )
                if cfg_resp.status_code == 200:
                    cfg_data = cfg_resp.json().get("config", {})
                    perm = cfg_data.get(f"__action_{action}", "allow")
                    if perm == "off":
                        return _err(
                            "ACTION_DISABLED",
                            f"Action '{action}' is disabled for '{connector}' (instance '{instance_id}'). Enable it in the Connectors settings.",
                        )
                    if perm == "ask":
                        return _ok(
                            success=False,
                            result={"requires_approval": True},
                            message=(
                                f"Action '{action}' on '{connector}' (instance '{instance_id}') requires user approval. "
                                "This action is in 'ask' mode — the user must grant permission before it executes. "
                                "Ask the user to set it to 'allow' in Connectors settings, then retry."
                            ),
                        )
            except Exception:
                pass  # Config fetch failure is non-fatal — proceed with default "allow"

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

            # Native connector path — pass instance_id through to backend
            resp = httpx.post(
                f"{vault_url}/api/connectors/{connector}/invoke",
                json={"action": action, "params": params_dict, "instance_id": instance_id},
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
