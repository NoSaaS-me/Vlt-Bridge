"""Unit tests for artifact MCP tools — registration, parameter validation, error paths."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest


def _make_mcp():
    """Create a mock FastMCP server to capture tool registrations."""
    tools = {}

    class MockMCP:
        def tool(self):
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

    return MockMCP(), tools


class TestArtifactToolRegistration:
    """Test that all expected tools get registered."""

    def test_registers_all_tools(self):
        from vlt.mcp.artifact_tools import register_artifact_tools

        mcp, tools = _make_mcp()
        register_artifact_tools(mcp)

        expected = {
            "vlt_artifact_create",
            "vlt_artifact_list",
            "vlt_artifact_update",
            "vlt_artifact_state",
            "vlt_artifact_test",
            "vlt_artifact_screenshot",
        }
        assert set(tools.keys()) == expected

    def test_tool_has_docstrings(self):
        from vlt.mcp.artifact_tools import register_artifact_tools

        mcp, tools = _make_mcp()
        register_artifact_tools(mcp)

        for name, fn in tools.items():
            assert fn.__doc__, f"Tool {name} is missing docstring"


class TestArtifactUpdate:
    """Test the vlt_artifact_update file-writing tool."""

    def test_invalid_json_files(self):
        from vlt.mcp.artifact_tools import register_artifact_tools

        mcp, tools = _make_mcp()
        register_artifact_tools(mcp)

        result = tools["vlt_artifact_update"](artifact_id="test", files="not json")
        assert result["status"] == "error"
        assert "INVALID_FILES" in result.get("error_code", result.get("code", ""))

    def test_empty_files_dict(self):
        """Empty files dict should still fetch artifact and succeed."""
        from vlt.mcp.artifact_tools import register_artifact_tools

        mcp, tools = _make_mcp()
        register_artifact_tools(mcp)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "test",
            "disk_path": "/tmp/nonexistent",
            "state": "draft",
        }

        with patch("httpx.get", return_value=mock_resp):
            result = tools["vlt_artifact_update"](artifact_id="test", files="{}")
            assert result["status"] == "ok"
            assert result["files_written"] == []


class TestArtifactScreenshot:
    """Test the screenshot tool (currently returns not-implemented)."""

    def test_returns_not_implemented(self):
        from vlt.mcp.artifact_tools import register_artifact_tools

        mcp, tools = _make_mcp()
        register_artifact_tools(mcp)

        result = tools["vlt_artifact_screenshot"](artifact_id="test-123")
        assert result["status"] == "error"
        assert "NOT_IMPLEMENTED" in result.get("error_code", result.get("code", ""))


class TestArtifactCreate:
    """Test the create tool with mocked daemon."""

    def test_successful_create(self):
        from vlt.mcp.artifact_tools import register_artifact_tools

        mcp, tools = _make_mcp()
        register_artifact_tools(mcp)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "abc123",
            "name": "Test Artifact",
            "project_id": "default",
            "state": "draft",
            "disk_path": "/data/artifacts/local-dev/abc123",
            "type": "ephemeral",
        }

        with patch("httpx.post", return_value=mock_resp):
            result = tools["vlt_artifact_create"](name="Test Artifact")
            assert result["status"] == "ok"
            assert result["artifact_id"] == "abc123"
            assert result["state"] == "draft"

    def test_daemon_error(self):
        from vlt.mcp.artifact_tools import register_artifact_tools

        mcp, tools = _make_mcp()
        register_artifact_tools(mcp)

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("httpx.post", return_value=mock_resp):
            result = tools["vlt_artifact_create"](name="Test")
            assert result["status"] == "error"


class TestArtifactState:
    """Test the state transition tool."""

    def test_successful_transition(self):
        from vlt.mcp.artifact_tools import register_artifact_tools

        mcp, tools = _make_mcp()
        register_artifact_tools(mcp)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "abc123",
            "state": "building",
            "state_history": [
                {"state": "draft", "at": "2026-03-15T10:00:00Z", "by": "user"},
                {"state": "building", "at": "2026-03-15T10:05:00Z", "by": "agent"},
            ],
            "version": 2,
        }

        with patch("httpx.post", return_value=mock_resp):
            result = tools["vlt_artifact_state"](artifact_id="abc123", target_state="building")
            assert result["status"] == "ok"
            assert result["state"] == "building"
            assert result["previous_state"] == "draft"
