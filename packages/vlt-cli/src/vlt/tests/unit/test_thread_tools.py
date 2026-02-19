"""Unit tests for thread_tools MCP tools.

Mocks SqliteVaultService to test tool logic in isolation.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(thread_id="test-thread", seq=0, content="hello"):
    node = MagicMock()
    node.id = f"node-{seq}"
    node.thread_id = thread_id
    node.sequence_id = seq
    node.content = content
    node.author = "agent"
    node.timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return node


def _make_thread(thread_id="test-thread", project_id="test-proj", status="active"):
    t = MagicMock()
    t.id = thread_id
    t.project_id = project_id
    t.status = status
    t.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return t


def _register_and_get(tool_name: str):
    """Create a server, register thread tools, and return the named tool function."""
    import asyncio
    from fastmcp import FastMCP
    from vlt.mcp.thread_tools import register_thread_tools

    mcp = FastMCP("test")
    register_thread_tools(mcp)

    async def _get():
        tool = await mcp.get_tool(tool_name)
        return tool.fn

    return asyncio.run(_get())


# ---------------------------------------------------------------------------
# vlt_thread_create
# ---------------------------------------------------------------------------

class TestVltThreadCreate:
    def test_creates_thread_and_returns_ids(self):
        fn = _register_and_get("vlt_thread_create")

        mock_thread = _make_thread()
        mock_node = _make_node()

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.ensure_project_exists.return_value = MagicMock()
            svc.create_thread.return_value = mock_thread

            # Mock the DB scalar query returning first_node
            scalars_mock = MagicMock()
            scalars_mock.first.return_value = mock_node
            svc.db.scalars.return_value = scalars_mock

            result = fn(
                project_id="test-proj",
                name="test-thread",
                initial_thought="First thought",
            )

        assert result["status"] == "ok"
        assert result["thread_id"] == "test-thread"
        assert result["project_id"] == "test-proj"
        assert result["node_id"] == "node-0"
        assert result["sequence_id"] == 0

    def test_returns_error_on_vault_error(self):
        fn = _register_and_get("vlt_thread_create")

        from vlt.core.service import VaultError

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.ensure_project_exists.side_effect = VaultError("DB error")

            result = fn(
                project_id="bad-proj",
                name="test-thread",
                initial_thought="ignored",
            )

        assert result["status"] == "error"
        assert result["code"] == "THREAD_CREATE_FAILED"


# ---------------------------------------------------------------------------
# vlt_thread_push
# ---------------------------------------------------------------------------

class TestVltThreadPush:
    def test_push_returns_duration_ms(self):
        fn = _register_and_get("vlt_thread_push")

        mock_thread = _make_thread()
        mock_node = _make_node(seq=1, content="second thought")

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.db.get.return_value = mock_thread
            svc.add_thought.return_value = mock_node

            result = fn(thread_id="test-thread", content="second thought")

        assert result["status"] == "ok"
        assert result["thread_id"] == "test-thread"
        assert result["node_id"] == "node-1"
        assert result["sequence_id"] == 1
        assert isinstance(result["duration_ms"], float)
        assert result["duration_ms"] >= 0

    def test_push_to_nonexistent_thread_returns_error(self):
        fn = _register_and_get("vlt_thread_push")

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.db.get.return_value = None  # thread not found

            result = fn(thread_id="no-such-thread", content="hello")

        assert result["status"] == "error"
        assert result["code"] == "THREAD_NOT_FOUND"


# ---------------------------------------------------------------------------
# vlt_thread_read
# ---------------------------------------------------------------------------

class TestVltThreadRead:
    def test_read_returns_summary_and_nodes(self):
        fn = _register_and_get("vlt_thread_read")

        from vlt.core.interfaces import ThreadStateView, NodeView

        node_view = NodeView(
            id="node-0",
            content="hello",
            author="agent",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            sequence_id=0,
        )
        state = ThreadStateView(
            thread_id="test-thread",
            project_id="test-proj",
            summary="A summary.",
            recent_nodes=[node_view],
            meta={},
        )

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.get_thread_state.return_value = state
            svc.db.scalar.return_value = 1  # node_count

            result = fn(thread_id="test-thread", limit=10)

        assert result["status"] == "ok"
        assert result["thread_id"] == "test-thread"
        assert result["project_id"] == "test-proj"
        assert result["summary"] == "A summary."
        assert result["node_count"] == 1
        assert len(result["recent_nodes"]) == 1
        assert result["recent_nodes"][0]["content"] == "hello"


# ---------------------------------------------------------------------------
# vlt_thread_seek
# ---------------------------------------------------------------------------

class TestVltThreadSeek:
    def test_seek_returns_semantic_results(self):
        fn = _register_and_get("vlt_thread_seek")

        mock_results = [
            {
                "thread_id": "auth-thread",
                "node_id": 3,
                "content": "Use JWT tokens",
                "author": "agent",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "score": 0.9,
            }
        ]

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.seek_threads.return_value = mock_results

            result = fn(query="authentication", project_id="test-proj")

        assert result["status"] == "ok"
        assert result["search_mode"] == "semantic"
        assert result["total"] == 1
        assert result["results"][0]["thread_id"] == "auth-thread"

    def test_seek_falls_back_to_keyword_on_error(self):
        fn = _register_and_get("vlt_thread_seek")

        mock_node = _make_node(content="Use JWT tokens")

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            # Semantic search fails (no API key)
            svc.seek_threads.side_effect = Exception("No API key")

            scalars_mock = MagicMock()
            scalars_mock.all.return_value = [mock_node]
            svc.db.scalars.return_value = scalars_mock

            result = fn(query="JWT", project_id="test-proj")

        assert result["status"] == "ok"
        assert result["search_mode"] == "keyword"
        assert result["total"] >= 0

    def test_seek_includes_search_mode_field(self):
        fn = _register_and_get("vlt_thread_seek")

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.seek_threads.return_value = []

            result = fn(query="anything", project_id="test-proj")

        assert "search_mode" in result


# ---------------------------------------------------------------------------
# vlt_thread_list
# ---------------------------------------------------------------------------

class TestVltThreadList:
    def test_list_scoped_by_project(self):
        fn = _register_and_get("vlt_thread_list")

        threads = [_make_thread(), _make_thread("other-thread")]

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.list_threads.return_value = threads

            # Mock node count query
            execute_mock = MagicMock()
            execute_mock.all.return_value = [
                MagicMock(thread_id="test-thread", cnt=5),
            ]
            svc.db.execute.return_value = execute_mock

            result = fn(project_id="test-proj")

        assert result["status"] == "ok"
        assert result["project_id"] == "test-proj"
        assert len(result["threads"]) == 2

    def test_list_all_when_no_project(self):
        fn = _register_and_get("vlt_thread_list")

        threads = [_make_thread(), _make_thread("other-thread", "other-proj")]

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value

            scalars_mock = MagicMock()
            scalars_mock.all.return_value = threads
            svc.db.scalars.return_value = scalars_mock

            execute_mock = MagicMock()
            execute_mock.all.return_value = []
            svc.db.execute.return_value = execute_mock

            result = fn()

        assert result["status"] == "ok"
        assert result["project_id"] == "all"
        assert len(result["threads"]) == 2

    def test_thread_entry_has_required_fields(self):
        fn = _register_and_get("vlt_thread_list")

        with patch("vlt.core.service.SqliteVaultService") as MockSvc:
            svc = MockSvc.return_value
            svc.list_threads.return_value = [_make_thread()]

            execute_mock = MagicMock()
            execute_mock.all.return_value = []
            svc.db.execute.return_value = execute_mock

            result = fn(project_id="test-proj")

        assert result["status"] == "ok"
        entry = result["threads"][0]
        assert "id" in entry
        assert "project_id" in entry
        assert "status" in entry
        assert "created_at" in entry
        assert "node_count" in entry
