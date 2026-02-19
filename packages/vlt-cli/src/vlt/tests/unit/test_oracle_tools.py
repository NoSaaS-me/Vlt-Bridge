"""Unit tests for oracle_tools MCP tools.

Mocks httpx and settings to test vlt_oracle_status and vlt_oracle_query in isolation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_and_get(tool_name: str):
    from fastmcp import FastMCP
    from vlt.mcp.oracle_tools import register_oracle_tools

    mcp = FastMCP("test")
    register_oracle_tools(mcp)

    async def _get():
        tool = await mcp.get_tool(tool_name)
        return tool.fn

    return asyncio.run(_get())


def _make_settings(
    oracle_enabled=True,
    vault_url=None,
    sync_token=None,
    oracle_model="deepseek/deepseek-chat-v3",
    openrouter_api_key=None,
):
    s = MagicMock()
    s.oracle_enabled = oracle_enabled
    s.vault_url = vault_url
    s.sync_url = None
    s.sync_token = sync_token
    s.oracle_model = oracle_model
    s.openrouter_api_key = openrouter_api_key
    s.google_api_key = None
    return s


# ---------------------------------------------------------------------------
# vlt_oracle_status
# ---------------------------------------------------------------------------

class TestVltOracleStatus:
    def test_enabled_true_from_local_settings(self):
        fn = _register_and_get("vlt_oracle_status")

        with patch("vlt.config.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(oracle_enabled=True)

            result = fn()

        assert result["status"] == "ok"
        assert result["enabled"] is True

    def test_enabled_false_from_local_settings(self):
        fn = _register_and_get("vlt_oracle_status")

        with patch("vlt.config.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(oracle_enabled=False)

            result = fn()

        assert result["status"] == "ok"
        assert result["enabled"] is False
        assert result["guidance"] is not None

    def test_configured_false_when_no_api_key(self):
        fn = _register_and_get("vlt_oracle_status")

        with patch("vlt.config.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(
                vault_url="http://localhost:8000",
                sync_token="tok",
                openrouter_api_key=None,
            )

            # Mock httpx to return 200 (backend available)
            http_resp = MagicMock()
            http_resp.status_code = 200
            http_resp.json.return_value = {"oracle_mcp_enabled": True}

            with patch("httpx.get", return_value=http_resp):
                result = fn()

        assert result["status"] == "ok"
        assert result["configured"] is False  # no api key
        assert result["backend_available"] is True

    def test_backend_unavailable_falls_back_to_local(self):
        fn = _register_and_get("vlt_oracle_status")

        with patch("vlt.config.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(
                oracle_enabled=True,
                vault_url="http://localhost:8000",
                sync_token="tok",
            )

            with patch("httpx.get", side_effect=Exception("connection refused")):
                result = fn()

        assert result["status"] == "ok"
        assert result["backend_available"] is False
        assert result["enabled"] is True  # falls back to local setting

    def test_model_included_in_response(self):
        fn = _register_and_get("vlt_oracle_status")

        with patch("vlt.config.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(
                oracle_model="anthropic/claude-sonnet-4"
            )

            result = fn()

        assert result["model"] == "anthropic/claude-sonnet-4"


# ---------------------------------------------------------------------------
# vlt_oracle_query
# ---------------------------------------------------------------------------

class TestVltOracleQuery:
    def test_returns_oracle_disabled_when_disabled(self):
        fn = _register_and_get("vlt_oracle_query")

        with patch("vlt.config.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(oracle_enabled=False)

            result = fn(query="what does this do?")

        assert result["status"] == "error"
        assert result["code"] == "ORACLE_DISABLED"

    def test_returns_oracle_not_configured_when_no_key(self):
        fn = _register_and_get("vlt_oracle_query")

        with patch("vlt.config.get_settings") as mock_settings:
            # Oracle enabled but no vault URL → not configured
            mock_settings.return_value = _make_settings(
                oracle_enabled=True,
                vault_url=None,
                sync_token=None,
            )

            result = fn(query="what does this do?")

        assert result["status"] == "error"
        assert result["code"] in ("ORACLE_NOT_CONFIGURED", "ORACLE_DISABLED")

    def test_proxies_to_backend_when_configured(self):
        fn = _register_and_get("vlt_oracle_query")

        backend_resp = MagicMock()
        backend_resp.status_code = 200
        backend_resp.json.return_value = {
            "response": "The codebase implements a REST API.",
            "context_id": "ctx-123",
            "sources_used": ["threads", "coderag"],
        }

        with (
            patch("vlt.config.get_settings") as mock_settings,
            patch("httpx.get") as mock_get,
            patch("httpx.post", return_value=backend_resp),
        ):
            mock_settings.return_value = _make_settings(
                oracle_enabled=True,
                vault_url="http://localhost:8000",
                sync_token="test-token",
                openrouter_api_key="sk-test",
            )
            # Oracle status health check (backend returns oracle enabled)
            status_resp = MagicMock()
            status_resp.status_code = 200
            status_resp.json.return_value = {"oracle_mcp_enabled": True}
            mock_get.return_value = status_resp

            result = fn(query="what does this do?")

        assert result["status"] == "ok"
        assert "REST API" in result["answer"]
        assert result["context_id"] == "ctx-123"
        assert "coderag" in result["sources_used"]
