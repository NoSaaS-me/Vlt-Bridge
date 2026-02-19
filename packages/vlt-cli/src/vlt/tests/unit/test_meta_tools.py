"""Unit tests for meta_tools MCP tools (vlt_status, vlt_project_detect).

Mocks SqliteVaultService, DaemonClient, and filesystem operations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_and_get(tool_name: str):
    """Create a server, register meta tools, and return the named tool function."""
    import asyncio
    from fastmcp import FastMCP
    from vlt.mcp.meta_tools import register_meta_tools

    mcp = FastMCP("test")
    register_meta_tools(mcp)

    async def _get():
        tool = await mcp.get_tool(tool_name)
        return tool.fn

    return asyncio.run(_get())


def _make_project(pid="test-proj", name="Test Project"):
    p = MagicMock()
    p.id = pid
    p.name = name
    return p


# ---------------------------------------------------------------------------
# vlt_status
# ---------------------------------------------------------------------------

class TestVltStatus:
    def test_returns_projects_list(self):
        fn = _register_and_get("vlt_status")

        projects = [_make_project("proj-a", "Project A"), _make_project("proj-b", "Project B")]

        with (
            patch("vlt.config.get_settings") as mock_settings,
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("httpx.get") as mock_http,
        ):
            settings = mock_settings.return_value
            settings.daemon_url = "http://127.0.0.1:8765"
            settings.is_server_configured = False
            settings.oracle_enabled = True
            settings.get_db_path.return_value = Path("/tmp/vault.db")

            svc = MockSvc.return_value
            svc.list_projects.return_value = projects

            execute_mock = MagicMock()
            execute_mock.all.return_value = []
            svc.db.execute.return_value = execute_mock

            mock_http.side_effect = Exception("no daemon")

            result = fn()

        assert result["status"] == "ok"
        assert len(result["projects"]) == 2
        assert result["projects"][0]["id"] == "proj-a"
        assert result["daemon_running"] is False

    def test_daemon_running_true_on_healthy_response(self):
        fn = _register_and_get("vlt_status")

        with (
            patch("vlt.config.get_settings") as mock_settings,
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("httpx.get") as mock_http,
        ):
            settings = mock_settings.return_value
            settings.daemon_url = "http://127.0.0.1:8765"
            settings.is_server_configured = True
            settings.oracle_enabled = True
            settings.get_db_path.return_value = Path("/tmp/vault.db")

            svc = MockSvc.return_value
            svc.list_projects.return_value = []

            execute_mock = MagicMock()
            execute_mock.all.return_value = []
            svc.db.execute.return_value = execute_mock

            http_resp = MagicMock()
            http_resp.status_code = 200
            mock_http.return_value = http_resp

            result = fn()

        assert result["status"] == "ok"
        assert result["daemon_running"] is True
        assert result["backend_configured"] is True

    def test_oracle_enabled_reflected_in_response(self):
        fn = _register_and_get("vlt_status")

        with (
            patch("vlt.config.get_settings") as mock_settings,
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("httpx.get") as mock_http,
        ):
            settings = mock_settings.return_value
            settings.daemon_url = "http://127.0.0.1:8765"
            settings.is_server_configured = False
            settings.oracle_enabled = False
            settings.get_db_path.return_value = Path("/tmp/vault.db")

            svc = MockSvc.return_value
            svc.list_projects.return_value = []

            execute_mock = MagicMock()
            execute_mock.all.return_value = []
            svc.db.execute.return_value = execute_mock

            mock_http.side_effect = Exception("no daemon")

            result = fn()

        assert result["oracle_enabled"] is False


# ---------------------------------------------------------------------------
# vlt_project_detect
# ---------------------------------------------------------------------------

class TestVltProjectDetect:
    def test_returns_found_true_when_vlt_toml_present(self, tmp_path):
        fn = _register_and_get("vlt_project_detect")

        # Write a vlt.toml in tmp_path
        toml_content = b'[project]\nid = "my-project"\nname = "My Project"\n'
        (tmp_path / "vlt.toml").write_bytes(toml_content)

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = True

            result = fn(path=str(tmp_path))

        assert result["status"] == "ok"
        assert result["found"] is True
        assert result["project_id"] == "my-project"
        assert result["project_name"] == "My Project"
        assert result["has_coderag_index"] is True

    def test_returns_found_false_when_no_vlt_toml(self, tmp_path):
        fn = _register_and_get("vlt_project_detect")

        result = fn(path=str(tmp_path))

        assert result["status"] == "ok"
        assert result["found"] is False
        assert result["project_id"] is None

    def test_walks_up_directory_tree(self, tmp_path):
        fn = _register_and_get("vlt_project_detect")

        # vlt.toml in parent, call from sub-directory
        toml_content = b'[project]\nid = "root-proj"\nname = "Root Project"\n'
        (tmp_path / "vlt.toml").write_bytes(toml_content)

        subdir = tmp_path / "src" / "deep"
        subdir.mkdir(parents=True)

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = False

            result = fn(path=str(subdir))

        assert result["found"] is True
        assert result["project_id"] == "root-proj"

    def test_defaults_to_cwd(self, tmp_path, monkeypatch):
        fn = _register_and_get("vlt_project_detect")

        toml_content = b'[project]\nid = "cwd-proj"\nname = "CWD Project"\n'
        (tmp_path / "vlt.toml").write_bytes(toml_content)
        monkeypatch.chdir(tmp_path)

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = False

            result = fn()  # no path arg

        assert result["found"] is True
        assert result["project_id"] == "cwd-proj"
