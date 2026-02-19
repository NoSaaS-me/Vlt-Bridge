"""Unit tests for code_tools MCP tools.

Mocks CodeRAGStore, SqliteVaultService, and DaemonClient.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_and_get(tool_name: str):
    from fastmcp import FastMCP
    from vlt.mcp.code_tools import register_code_tools

    mcp = FastMCP("test")
    register_code_tools(mcp)

    async def _get():
        tool = await mcp.get_tool(tool_name)
        return tool.fn

    return asyncio.run(_get())


def _make_job(status="running", progress=50, job_id="job-1"):
    from vlt.core.models import JobStatus
    job = MagicMock()
    job.id = job_id
    job.status = JobStatus.RUNNING if status == "running" else JobStatus.PENDING
    job.progress_percent = progress
    job.files_processed = 10
    job.files_total = 20
    job.chunks_created = 50
    job.error_message = None
    return job


def _make_symbol(name="my_func", file_path="src/foo.py", lineno=10, kind="function"):
    s = MagicMock()
    s.name = name
    s.file_path = file_path
    s.lineno = lineno
    s.kind = kind
    s.scope = None
    s.signature = f"def {name}()"
    s.language = "python"
    return s


# ---------------------------------------------------------------------------
# vlt_code_init
# ---------------------------------------------------------------------------

class TestVltCodeInit:
    def test_returns_already_running_when_active_job_no_force(self, tmp_path):
        fn = _register_and_get("vlt_code_init")

        active_job = _make_job("running")

        with (
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("httpx.get"),
        ):
            svc = MockSvc.return_value
            svc.ensure_project_exists.return_value = MagicMock()
            svc.get_active_job_for_project.return_value = active_job

            result = fn(project_id="test-proj", path=str(tmp_path), force=False)

        assert result["status"] == "ok"
        assert result["job_id"] == "job-1"
        assert "already in progress" in result["message"].lower()

    def test_creates_job_when_daemon_running(self, tmp_path):
        """Test job creation path when daemon is available (no background thread spawned)."""
        fn = _register_and_get("vlt_code_init")

        session_mock = MagicMock()
        session_mock.__enter__ = MagicMock(return_value=session_mock)
        session_mock.__exit__ = MagicMock(return_value=False)

        http_resp = MagicMock()
        http_resp.status_code = 200

        with (
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("vlt.config.get_settings") as mock_settings,
            patch("vlt.core.models.CodeRAGIndexJob"),
            patch("vlt.db.engine"),
            patch("httpx.get", return_value=http_resp),
            # Patch Session at sqlalchemy level; daemon is running so no bg thread
            patch("sqlalchemy.orm.Session", return_value=session_mock),
        ):
            svc = MockSvc.return_value
            svc.ensure_project_exists.return_value = MagicMock()
            svc.get_active_job_for_project.return_value = None

            settings = mock_settings.return_value
            settings.daemon_url = "http://127.0.0.1:8765"

            result = fn(project_id="test-proj", path=str(tmp_path))

        assert result["status"] == "ok"
        assert "job_id" in result
        assert result["daemon_handling"] is True

    def test_returns_error_for_nonexistent_path(self):
        fn = _register_and_get("vlt_code_init")

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.ensure_project_exists.return_value = MagicMock()
            svc.get_active_job_for_project.return_value = None

            result = fn(project_id="test-proj", path="/nonexistent/path/xyz")

        assert result["status"] == "error"
        assert result["code"] == "INVALID_PATH"


# ---------------------------------------------------------------------------
# vlt_code_search
# ---------------------------------------------------------------------------

class TestVltCodeSearch:
    def test_returns_index_not_found_when_not_indexed(self):
        fn = _register_and_get("vlt_code_search")

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = False

            result = fn(query="authentication", project_id="test-proj")

        assert result["status"] == "error"
        assert result["code"] == "INDEX_NOT_FOUND"

    def test_returns_results_when_indexed(self):
        fn = _register_and_get("vlt_code_search")

        mock_chunks = [
            {
                "chunk_id": "c1",
                "file_path": "src/auth.py",
                "qualified_name": "authenticate",
                "lineno": 42,
                "score": 0.9,
                "body": "def authenticate(token): ...",
                "language": "python",
            }
        ]

        with (
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("vlt.core.coderag.bm25.search_bm25") as mock_search,
        ):
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = True
            mock_search.return_value = mock_chunks

            result = fn(query="authenticate", project_id="test-proj", limit=5)

        assert result["status"] == "ok"
        assert len(result["results"]) == 1
        assert result["results"][0]["file_path"] == "src/auth.py"
        assert result["results"][0]["lineno"] == 42

    def test_result_has_required_fields(self):
        fn = _register_and_get("vlt_code_search")

        mock_chunk = {
            "chunk_id": "c1",
            "file_path": "src/foo.py",
            "qualified_name": "foo",
            "lineno": 1,
            "score": 0.5,
            "body": "def foo(): pass",
        }

        with (
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("vlt.core.coderag.bm25.search_bm25") as mock_search,
        ):
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = True
            mock_search.return_value = [mock_chunk]

            result = fn(query="foo", project_id="test-proj")

        assert result["status"] == "ok"
        entry = result["results"][0]
        for field in ("chunk_id", "file_path", "qualified_name", "lineno", "score", "snippet"):
            assert field in entry, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# vlt_code_map
# ---------------------------------------------------------------------------

class TestVltCodeMap:
    def test_returns_index_not_found_when_not_indexed(self):
        fn = _register_and_get("vlt_code_map")

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = False

            result = fn(project_id="test-proj")

        assert result["status"] == "error"
        assert result["code"] == "INDEX_NOT_FOUND"

    def test_returns_map_text_when_available(self):
        fn = _register_and_get("vlt_code_map")

        mock_repo_map = MagicMock()
        mock_repo_map.map_text = "# Repo Map\n- src/foo.py\n"
        mock_repo_map.token_count = 100
        mock_repo_map.files_included = 5
        mock_repo_map.symbols_included = 20
        mock_repo_map.symbols_total = 25

        with (
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("vlt.core.coderag.store.CodeRAGStore") as MockStore,
        ):
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = True

            store_instance = MockStore.return_value
            store_instance.__enter__ = MagicMock(return_value=store_instance)
            store_instance.__exit__ = MagicMock(return_value=False)
            store_instance.get_repo_map.return_value = mock_repo_map

            result = fn(project_id="test-proj")

        assert result["status"] == "ok"
        assert "# Repo Map" in result["map_text"]
        assert result["token_count"] == 100


# ---------------------------------------------------------------------------
# vlt_code_status
# ---------------------------------------------------------------------------

class TestVltCodeStatus:
    def test_returns_indexed_true_after_completion(self):
        fn = _register_and_get("vlt_code_status")

        with (
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("vlt.core.coderag.store.CodeRAGStore") as MockStore,
        ):
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = True
            svc.get_active_job_for_project.return_value = None

            store_instance = MockStore.return_value
            store_instance.__enter__ = MagicMock(return_value=store_instance)
            store_instance.__exit__ = MagicMock(return_value=False)
            store_instance.get_project_stats.return_value = {"chunks": 500, "symbols": 120}

            result = fn(project_id="test-proj")

        assert result["status"] == "ok"
        assert result["indexed"] is True
        assert result["active_job"] is None

    def test_active_job_info_included(self):
        fn = _register_and_get("vlt_code_status")

        active_job = _make_job("running", progress=60)

        with (
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("vlt.core.coderag.store.CodeRAGStore") as MockStore,
        ):
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = False
            svc.get_active_job_for_project.return_value = active_job

            store_instance = MockStore.return_value
            store_instance.__enter__ = MagicMock(return_value=store_instance)
            store_instance.__exit__ = MagicMock(return_value=False)
            store_instance.get_project_stats.return_value = {}

            result = fn(project_id="test-proj")

        assert result["status"] == "ok"
        assert result["active_job"] is not None
        assert result["active_job"]["progress_percent"] == 60


# ---------------------------------------------------------------------------
# vlt_code_lookup
# ---------------------------------------------------------------------------

class TestVltCodeLookup:
    def test_returns_definitions(self):
        fn = _register_and_get("vlt_code_lookup")

        mock_sym = _make_symbol("authenticate", "src/auth.py", 42, "function")

        with (
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("vlt.core.coderag.store.CodeRAGStore") as MockStore,
        ):
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = True
            store_instance = MockStore.return_value
            store_instance.__enter__ = MagicMock(return_value=store_instance)
            store_instance.__exit__ = MagicMock(return_value=False)
            store_instance.get_symbols_by_name.return_value = [mock_sym]

            result = fn(symbol="authenticate", project_id="test-proj")

        assert result["status"] == "ok"
        assert result["found"] is True
        assert len(result["definitions"]) == 1
        assert result["definitions"][0]["file_path"] == "src/auth.py"

    def test_returns_found_false_when_no_match(self):
        fn = _register_and_get("vlt_code_lookup")

        with (
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("vlt.core.coderag.store.CodeRAGStore") as MockStore,
        ):
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = True
            store_instance = MockStore.return_value
            store_instance.__enter__ = MagicMock(return_value=store_instance)
            store_instance.__exit__ = MagicMock(return_value=False)
            store_instance.get_symbols_by_name.return_value = []

            result = fn(symbol="nonexistent_symbol", project_id="test-proj")

        assert result["status"] == "ok"
        assert result["found"] is False

    def test_kind_filter_applied(self):
        fn = _register_and_get("vlt_code_lookup")

        symbols = [
            _make_symbol("Foo", "src/foo.py", 1, "class"),
            _make_symbol("Foo", "src/bar.py", 5, "function"),
        ]

        with (
            patch("vlt.core.service.SqliteVaultService") as MockSvc,
            patch("vlt.core.coderag.store.CodeRAGStore") as MockStore,
        ):
            svc = MockSvc.return_value
            svc.has_coderag_index.return_value = True
            store_instance = MockStore.return_value
            store_instance.__enter__ = MagicMock(return_value=store_instance)
            store_instance.__exit__ = MagicMock(return_value=False)
            store_instance.get_symbols_by_name.return_value = symbols

            result = fn(symbol="Foo", project_id="test-proj", kind="class")

        assert result["status"] == "ok"
        assert len(result["definitions"]) == 1
        assert result["definitions"][0]["kind"] == "class"
