"""Oracle tools for the vlt MCP server.

Exposes oracle status and query capabilities via MCP. The oracle toggle
state is read from the Document-MCP backend when configured; otherwise
falls back to the local VLT_ORACLE_ENABLED setting.

Tools registered:
    vlt_oracle_status  — three-state oracle availability check
    vlt_oracle_query   — AI-powered multi-source codebase query
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def register_oracle_tools(mcp) -> None:
    """Register oracle tools onto a FastMCP server instance."""

    @mcp.tool()
    def vlt_oracle_status() -> dict:
        """Check oracle availability: enabled state, backend config, and model in use.

        Returns a three-state dict so callers know exactly why oracle is or is not available:
        - enabled: oracle MCP toggle is on (from backend or env)
        - configured: an API key and backend URL are set
        - backend_available: the Document-MCP backend responded to a health check

        Returns:
            {status, enabled, configured, backend_available, model, guidance}
        """
        from vlt.mcp import _ok, _err

        try:
            from vlt.config import get_settings
            settings = get_settings()

            # Determine if backend is available and oracle enabled
            backend_available = False
            enabled_from_backend: Optional[bool] = None

            vault_url = getattr(settings, "vault_url", None) or getattr(settings, "sync_url", None)
            sync_token = getattr(settings, "sync_token", None)

            if vault_url and sync_token:
                try:
                    import httpx
                    r = httpx.get(f"{vault_url}/api/settings/oracle",
                                  headers={"Authorization": f"Bearer {sync_token}"},
                                  timeout=2.0)
                    if r.status_code == 200:
                        backend_available = True
                        data = r.json()
                        enabled_from_backend = bool(data.get("oracle_mcp_enabled", True))
                except Exception:
                    pass

            # Fall back to local settings if backend unreachable
            local_enabled = getattr(settings, "oracle_enabled", True)
            enabled = enabled_from_backend if enabled_from_backend is not None else local_enabled

            # Is oracle configured? Needs an API key
            api_key = (
                getattr(settings, "openrouter_api_key", None)
                or getattr(settings, "google_api_key", None)
            )
            configured = bool(api_key) and bool(vault_url)

            # Current model
            model = getattr(settings, "oracle_model", "deepseek/deepseek-chat-v3")

            guidance = None
            if not enabled:
                guidance = "Oracle is disabled. Enable it in Settings > Oracle or set VLT_ORACLE_ENABLED=true."
            elif not configured:
                guidance = "Oracle requires an API key. Run: vlt config set-key <YOUR_OPENROUTER_KEY>"

            return _ok(
                enabled=enabled,
                configured=configured,
                backend_available=backend_available,
                model=model,
                guidance=guidance,
            )

        except Exception as e:
            logger.exception("vlt_oracle_status failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_oracle_query(
        query: str,
        project_id: Optional[str] = None,
        context_id: Optional[str] = None,
    ) -> dict:
        """Ask the oracle a question about your codebase, documentation, or development history.

        The oracle synthesises context from vlt threads (reasoning history), CodeRAG
        (code index), and the Markdown vault. Returns a single consolidated answer
        (synchronous, not streamed). Timeout: 60 seconds.

        Before using: call vlt_oracle_status() to verify the oracle is enabled and
        configured. If status shows guidance text, follow it before querying.

        For multi-turn conversations, pass context_id from a previous response to
        continue with accumulated context.

        Args:
            query: Your question or task description.
            project_id: Project scope (optional — oracle infers from context if omitted).
            context_id: Continue a previous conversation. Pass the context_id value
                        from the previous vlt_oracle_query response.

        Returns:
            {status, answer, context_id, sources_used}
            Pass context_id to the next call for multi-turn conversation.
        """
        from vlt.mcp import _ok, _err

        try:
            # Check oracle status first
            status_result = vlt_oracle_status()
            if status_result.get("status") == "error":
                return status_result

            if not status_result.get("enabled"):
                return _err(
                    "ORACLE_DISABLED",
                    status_result.get(
                        "guidance",
                        "Oracle is disabled. Enable it in Settings > Oracle.",
                    ),
                )

            if not status_result.get("configured"):
                return _err(
                    "ORACLE_NOT_CONFIGURED",
                    status_result.get(
                        "guidance",
                        "Oracle requires API credentials. Run: vlt config set-key <token>",
                    ),
                )

            from vlt.config import get_settings
            settings = get_settings()

            vault_url = getattr(settings, "vault_url", None) or getattr(settings, "sync_url", None)
            sync_token = getattr(settings, "sync_token", None)

            if not vault_url or not sync_token:
                return _err(
                    "ORACLE_NOT_CONFIGURED",
                    "Document-MCP backend not configured. Set VLT_VAULT_URL and VLT_SYNC_TOKEN.",
                )

            # Proxy to backend oracle endpoint (non-streaming for MCP context)
            import httpx

            payload: dict = {"query": query}
            if project_id:
                payload["project_id"] = project_id
            if context_id:
                payload["context_id"] = context_id

            r = httpx.post(
                f"{vault_url}/api/oracle",
                json=payload,
                headers={"Authorization": f"Bearer {sync_token}"},
                timeout=60.0,
            )

            if r.status_code == 200:
                data = r.json()
                return _ok(
                    answer=data.get("response") or data.get("answer") or "",
                    context_id=data.get("context_id"),
                    sources_used=data.get("sources_used", []),
                )
            else:
                return _err(
                    "ORACLE_ERROR",
                    f"Backend returned HTTP {r.status_code}: {r.text[:200]}",
                )

        except Exception as e:
            logger.exception("vlt_oracle_query failed")
            return _err("INTERNAL_ERROR", str(e))
