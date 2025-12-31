"""Unit tests for ToolExecutor service (009-oracle-agent T015, T028)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.src.services.tool_executor import ToolExecutor, get_tool_executor


@pytest.fixture
def mock_services():
    """Create mock services for ToolExecutor."""
    vault = MagicMock()
    indexer = MagicMock()
    threads = MagicMock()
    oracle_bridge = MagicMock()
    db = MagicMock()

    return {
        "vault_service": vault,
        "indexer_service": indexer,
        "thread_service": threads,
        "oracle_bridge": oracle_bridge,
        "db_service": db,
    }


@pytest.fixture
def executor(mock_services) -> ToolExecutor:
    """Create a ToolExecutor with mocked dependencies."""
    return ToolExecutor(**mock_services)


class TestToolExecutorInit:
    """Tests for ToolExecutor initialization."""

    def test_init_with_provided_services(self, mock_services) -> None:
        """Executor uses provided service instances."""
        executor = ToolExecutor(**mock_services)

        assert executor.vault is mock_services["vault_service"]
        assert executor.indexer is mock_services["indexer_service"]
        assert executor.threads is mock_services["thread_service"]

    def test_init_creates_default_services_when_none_provided(self) -> None:
        """Executor creates default services when none provided."""
        # This will create real service instances
        executor = ToolExecutor()

        assert executor.vault is not None
        assert executor.indexer is not None
        assert executor.threads is not None


class TestToolExecutorGetToolSchemas:
    """Tests for get_tool_schemas() method."""

    def test_get_oracle_tools(self, executor: ToolExecutor) -> None:
        """get_tool_schemas('oracle') returns Oracle-scoped tools."""
        tools = executor.get_tool_schemas(agent="oracle")

        # Should have multiple tools
        assert len(tools) > 0

        # All should be function type
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]

        # Check for expected Oracle tools
        tool_names = [t["function"]["name"] for t in tools]
        assert "search_code" in tool_names
        assert "vault_read" in tool_names
        assert "thread_push" in tool_names

    def test_get_librarian_tools(self, executor: ToolExecutor) -> None:
        """get_tool_schemas('librarian') returns Librarian-scoped tools."""
        tools = executor.get_tool_schemas(agent="librarian")

        tool_names = [t["function"]["name"] for t in tools]

        # Librarian should have fewer tools
        assert len(tools) < len(executor.get_tool_schemas(agent="oracle"))

        # Librarian-only tools
        assert "vault_move" in tool_names
        assert "vault_create_index" in tool_names

        # Both agents have these (including web tools for Librarian)
        assert "vault_read" in tool_names
        assert "vault_search" in tool_names
        assert "web_search" in tool_names
        assert "web_fetch" in tool_names

        # Oracle-only tools should not be present
        assert "thread_push" not in tool_names
        assert "delegate_librarian" not in tool_names

    def test_schemas_are_cached(self, executor: ToolExecutor) -> None:
        """Tool schemas are cached after first load."""
        tools1 = executor.get_tool_schemas(agent="oracle")
        tools2 = executor.get_tool_schemas(agent="oracle")

        # Same cache, filtered each time
        assert executor._schema_cache is not None


class TestToolExecutorExecute:
    """Tests for execute() method."""

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(
        self, executor: ToolExecutor
    ) -> None:
        """execute() returns error JSON for unknown tool."""
        result = await executor.execute("unknown_tool", {}, "user-123")

        data = json.loads(result)
        assert "error" in data
        assert "Unknown tool" in data["error"]

    @pytest.mark.asyncio
    async def test_execute_vault_read_calls_service(
        self, executor: ToolExecutor, mock_services
    ) -> None:
        """execute('vault_read') calls VaultService.read_note()."""
        mock_services["vault_service"].read_note.return_value = {
            "title": "Test Note",
            "body": "Content here",
            "metadata": {},
        }

        result = await executor.execute(
            "vault_read",
            {"path": "test/note.md"},
            "user-123",
        )

        data = json.loads(result)
        assert data["path"] == "test/note.md"
        assert data["title"] == "Test Note"
        assert data["content"] == "Content here"

        mock_services["vault_service"].read_note.assert_called_once_with(
            "user-123", "test/note.md"
        )

    @pytest.mark.asyncio
    async def test_execute_vault_search_calls_indexer(
        self, executor: ToolExecutor, mock_services
    ) -> None:
        """execute('vault_search') calls IndexerService.search_notes()."""
        mock_services["indexer_service"].search_notes.return_value = [
            {"path": "result1.md", "score": 0.9},
            {"path": "result2.md", "score": 0.8},
        ]

        result = await executor.execute(
            "vault_search",
            {"query": "test query", "limit": 5},
            "user-123",
        )

        data = json.loads(result)
        assert data["query"] == "test query"
        assert data["count"] == 2
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_execute_search_code_calls_oracle_bridge(
        self, executor: ToolExecutor, mock_services
    ) -> None:
        """execute('search_code') calls OracleBridge.search_code()."""
        mock_services["oracle_bridge"].search_code = AsyncMock(
            return_value={"results": [{"file": "test.py"}]}
        )

        result = await executor.execute(
            "search_code",
            {"query": "function definition", "limit": 5},
            "user-123",
        )

        data = json.loads(result)
        assert "results" in data

        mock_services["oracle_bridge"].search_code.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_handles_file_not_found(
        self, executor: ToolExecutor, mock_services
    ) -> None:
        """execute() returns error for FileNotFoundError."""
        mock_services["vault_service"].read_note.side_effect = FileNotFoundError(
            "Note not found"
        )

        result = await executor.execute(
            "vault_read",
            {"path": "missing.md"},
            "user-123",
        )

        data = json.loads(result)
        assert "error" in data
        assert "File not found" in data["error"]

    @pytest.mark.asyncio
    async def test_execute_handles_generic_exception(
        self, executor: ToolExecutor, mock_services
    ) -> None:
        """execute() returns error for generic exceptions."""
        mock_services["vault_service"].read_note.side_effect = Exception(
            "Something went wrong"
        )

        result = await executor.execute(
            "vault_read",
            {"path": "test.md"},
            "user-123",
        )

        data = json.loads(result)
        assert "error" in data
        assert "execution failed" in data["error"]


class TestToolExecutorUnimplementedTools:
    """Tests for placeholder tool implementations."""

    @pytest.mark.asyncio
    async def test_find_definition_returns_not_implemented(
        self, executor: ToolExecutor
    ) -> None:
        """find_definition returns placeholder error."""
        result = await executor.execute(
            "find_definition",
            {"symbol": "TestClass"},
            "user-123",
        )

        data = json.loads(result)
        assert "error" in data
        assert "not yet implemented" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_delegate_librarian_requires_api_key(
        self, executor: ToolExecutor
    ) -> None:
        """delegate_librarian requires OpenRouter API key to be configured."""
        # Without a configured API key, the tool returns an error
        result = await executor.execute(
            "delegate_librarian",
            {"task": "organize docs", "task_type": "summarize"},
            "user-123",
        )

        data = json.loads(result)
        assert "error" in data
        # Now implemented - returns API key error instead of "not implemented"
        assert "api key" in data["error"].lower() or "not configured" in data["error"].lower()


class TestToolExecutorRepoMap:
    """Tests for get_repo_map tool implementation (T031)."""

    @pytest.mark.asyncio
    async def test_get_repo_map_uses_oracle_bridge_when_available(
        self, executor: ToolExecutor, mock_services
    ) -> None:
        """get_repo_map uses OracleBridge when it returns valid map."""
        mock_services["oracle_bridge"].get_repo_map = AsyncMock(
            return_value={
                "map_text": "test map content",
                "token_count": 100,
                "max_tokens": 2000,
                "files_included": 5,
                "symbols_included": 10,
                "symbols_total": 15,
                "scope": None,
            }
        )

        result = await executor.execute(
            "get_repo_map",
            {"max_tokens": 2000},
            "user-123",
        )

        data = json.loads(result)
        assert "error" not in data
        assert data["map_text"] == "test map content"
        assert data["files_included"] == 5
        mock_services["oracle_bridge"].get_repo_map.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_repo_map_falls_back_to_filesystem(
        self, executor: ToolExecutor, mock_services
    ) -> None:
        """get_repo_map falls back to filesystem when OracleBridge fails."""
        mock_services["oracle_bridge"].get_repo_map = AsyncMock(
            side_effect=Exception("vlt coderag not available")
        )

        result = await executor.execute(
            "get_repo_map",
            {"max_tokens": 500},
            "user-123",
        )

        data = json.loads(result)
        assert "error" not in data
        assert "map_text" in data
        assert data.get("source") == "filesystem"
        assert "token_count" in data
        assert "files_included" in data

    @pytest.mark.asyncio
    async def test_get_repo_map_with_scope(
        self, executor: ToolExecutor, mock_services
    ) -> None:
        """get_repo_map passes scope parameter correctly."""
        mock_services["oracle_bridge"].get_repo_map = AsyncMock(
            return_value={
                "map_text": "scoped map",
                "token_count": 50,
                "max_tokens": 1000,
                "files_included": 2,
                "symbols_included": 5,
                "symbols_total": 5,
                "scope": "src/api/",
            }
        )

        result = await executor.execute(
            "get_repo_map",
            {"scope": "src/api/", "max_tokens": 1000},
            "user-123",
        )

        data = json.loads(result)
        assert data["scope"] == "src/api/"
        mock_services["oracle_bridge"].get_repo_map.assert_called_with(
            scope="src/api/",
            max_tokens=1000,
            include_signatures=True,
            include_docstrings=False,
            project=None,
        )

    @pytest.mark.asyncio
    async def test_get_repo_map_filesystem_fallback_respects_max_tokens(
        self, executor: ToolExecutor, mock_services
    ) -> None:
        """Filesystem fallback respects max_tokens parameter."""
        mock_services["oracle_bridge"].get_repo_map = AsyncMock(
            return_value={"error": "coderag not initialized"}
        )

        result = await executor.execute(
            "get_repo_map",
            {"max_tokens": 100},
            "user-123",
        )

        data = json.loads(result)
        assert "error" not in data
        assert data["max_tokens"] == 100
        # Token count should be within the budget
        assert data["token_count"] <= 100 or "truncated" in data["map_text"]

    @pytest.mark.asyncio
    async def test_get_repo_map_filesystem_includes_key_files(
        self, executor: ToolExecutor, mock_services
    ) -> None:
        """Filesystem fallback includes key_files in response."""
        mock_services["oracle_bridge"].get_repo_map = AsyncMock(
            side_effect=Exception("not available")
        )

        result = await executor.execute(
            "get_repo_map",
            {"max_tokens": 2000},
            "user-123",
        )

        data = json.loads(result)
        assert "key_files" in data
        assert isinstance(data["key_files"], list)
        assert "file_types" in data
        assert isinstance(data["file_types"], dict)


class TestToolExecutorSingleton:
    """Tests for singleton pattern."""

    def test_get_tool_executor_returns_singleton(self) -> None:
        """get_tool_executor() returns the same instance."""
        # Clear any existing singleton
        import backend.src.services.tool_executor as te_module

        te_module._tool_executor = None

        executor1 = get_tool_executor()
        executor2 = get_tool_executor()

        assert executor1 is executor2

        # Clean up
        te_module._tool_executor = None


class TestToolExecutorParallel:
    """Tests for parallel tool execution (009-oracle-agent T025).

    These tests verify that the execute_batch() method properly executes
    multiple tool calls concurrently, handles partial failures gracefully,
    respects timeouts, and preserves result ordering.
    """

    @pytest.fixture
    def parallel_executor(self, mock_services) -> ToolExecutor:
        """Create a ToolExecutor configured for parallel execution testing."""
        return ToolExecutor(**mock_services)

    @pytest.mark.asyncio
    async def test_parallel_tools_execute_concurrently(
        self, parallel_executor: ToolExecutor, mock_services
    ) -> None:
        """Verify that multiple tools run in parallel, not sequentially.

        If tools run sequentially, total time would be ~0.3s (3 * 0.1s).
        If tools run in parallel, total time should be ~0.1s + overhead.
        """
        import asyncio
        import time

        execution_times = []
        start_times = []

        async def slow_vault_read(*args, **kwargs):
            start_times.append(time.time())
            await asyncio.sleep(0.1)  # Simulate work
            execution_times.append(time.time())
            return {"title": "Test", "body": "Content", "metadata": {}}

        # Patch vault service to use slow async operation
        mock_services["vault_service"].read_note = slow_vault_read

        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "note1.md"}},
            {"name": "vault_read", "arguments": {"path": "note2.md"}},
            {"name": "vault_read", "arguments": {"path": "note3.md"}},
        ]

        overall_start = time.time()
        results = await parallel_executor.execute_batch(tool_calls, "user-123")
        overall_end = time.time()

        total_time = overall_end - overall_start

        # All three should return results
        assert len(results) == 3

        # Total time should be closer to 0.1s than 0.3s
        # Allow some overhead, but it should definitely be < 0.25s
        assert total_time < 0.25, f"Expected parallel execution, but took {total_time:.3f}s"

        # All three tools should have started within a short window
        if len(start_times) == 3:
            start_spread = max(start_times) - min(start_times)
            assert start_spread < 0.05, f"Tools didn't start concurrently, spread: {start_spread:.3f}s"

    @pytest.mark.asyncio
    async def test_parallel_tools_handle_partial_failure(
        self, parallel_executor: ToolExecutor, mock_services
    ) -> None:
        """One tool failing shouldn't block others from completing.

        Tool 1: succeeds with note content
        Tool 2: raises exception (simulated failure)
        Tool 3: succeeds with search results
        All three should return results (success or error).
        """
        # Tool 1 - vault_read succeeds
        mock_services["vault_service"].read_note.return_value = {
            "title": "Success Note",
            "body": "Content",
            "metadata": {},
        }

        # Tool 2 - search_code raises exception
        mock_services["oracle_bridge"].search_code = AsyncMock(
            side_effect=RuntimeError("Simulated failure in search_code")
        )

        # Tool 3 - vault_search succeeds
        mock_services["indexer_service"].search_notes.return_value = [
            {"path": "result.md", "score": 0.9}
        ]

        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "note.md"}},
            {"name": "search_code", "arguments": {"query": "test"}},
            {"name": "vault_search", "arguments": {"query": "search term"}},
        ]

        results = await parallel_executor.execute_batch(tool_calls, "user-123")

        # Should have 3 results
        assert len(results) == 3

        # Parse results
        result_0 = json.loads(results[0])
        result_1 = json.loads(results[1])
        result_2 = json.loads(results[2])

        # Tool 1 should succeed
        assert "error" not in result_0
        assert result_0["title"] == "Success Note"

        # Tool 2 should have error
        assert "error" in result_1

        # Tool 3 should succeed despite Tool 2 failing
        assert "error" not in result_2
        assert result_2["count"] == 1

    @pytest.mark.asyncio
    async def test_parallel_tools_timeout(
        self, parallel_executor: ToolExecutor, mock_services
    ) -> None:
        """Tools that exceed timeout should return error without blocking others.

        Tool 1: takes 0.5s (exceeds 0.2s timeout)
        Tool 2: completes quickly
        Both should return results, with Tool 1 having a timeout error.
        """
        import asyncio

        async def slow_search(*args, **kwargs):
            await asyncio.sleep(0.5)  # Takes longer than timeout
            return {"results": []}

        mock_services["oracle_bridge"].search_code = slow_search

        mock_services["vault_service"].read_note.return_value = {
            "title": "Quick Note",
            "body": "Fast content",
            "metadata": {},
        }

        tool_calls = [
            {"name": "search_code", "arguments": {"query": "slow query"}},
            {"name": "vault_read", "arguments": {"path": "fast.md"}},
        ]

        # Execute with a short timeout (0.2s)
        results = await parallel_executor.execute_batch(
            tool_calls, "user-123", timeout=0.2
        )

        assert len(results) == 2

        # Parse results
        result_0 = json.loads(results[0])
        result_1 = json.loads(results[1])

        # Tool 1 (slow) should have timeout error
        assert "error" in result_0
        assert "timeout" in result_0["error"].lower() or "timed out" in result_0["error"].lower()

        # Tool 2 (fast) should still succeed
        assert "error" not in result_1
        assert result_1["title"] == "Quick Note"

    @pytest.mark.asyncio
    async def test_parallel_tools_preserve_order(
        self, parallel_executor: ToolExecutor, mock_services
    ) -> None:
        """Results should match the order of the original tool calls.

        Execute tools with varying completion times to verify that
        result ordering matches input ordering, not completion order.
        Uses search_code (async) rather than vault_read (sync) for true parallel testing.
        """
        import asyncio

        completion_order = []

        async def slow_search(*args, **kwargs):
            await asyncio.sleep(0.15)
            completion_order.append("slow")
            return {"results": [{"file": "slow.py", "score": 0.9}]}

        async def medium_search(*args, **kwargs):
            await asyncio.sleep(0.10)
            completion_order.append("medium")
            return {"results": [{"file": "medium.py", "score": 0.8}]}

        async def fast_search(*args, **kwargs):
            await asyncio.sleep(0.05)
            completion_order.append("fast")
            return {"results": [{"file": "fast.py", "score": 0.7}]}

        call_count = 0

        async def variable_delay_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return await slow_search()
            elif call_count == 2:
                return await medium_search()
            else:
                return await fast_search()

        mock_services["oracle_bridge"].search_code = variable_delay_search

        # Order: slow, medium, fast - but fast completes first
        tool_calls = [
            {"name": "search_code", "arguments": {"query": "slow"}},
            {"name": "search_code", "arguments": {"query": "medium"}},
            {"name": "search_code", "arguments": {"query": "fast"}},
        ]

        results = await parallel_executor.execute_batch(tool_calls, "user-123")

        # All should complete
        assert len(results) == 3

        # Results should be in INPUT order (slow, medium, fast)
        # not completion order (fast, medium, slow)
        result_0 = json.loads(results[0])
        result_1 = json.loads(results[1])
        result_2 = json.loads(results[2])

        assert result_0["results"][0]["file"] == "slow.py"
        assert result_1["results"][0]["file"] == "medium.py"
        assert result_2["results"][0]["file"] == "fast.py"

        # Verify completion order was different (fast finished first)
        assert completion_order[0] == "fast"
        assert completion_order[-1] == "slow"

    @pytest.mark.asyncio
    async def test_parallel_tools_all_fail(
        self, parallel_executor: ToolExecutor, mock_services
    ) -> None:
        """When all tools fail, all results should contain errors."""
        # All tools raise exceptions
        mock_services["vault_service"].read_note.side_effect = FileNotFoundError("Not found")
        mock_services["oracle_bridge"].search_code = AsyncMock(
            side_effect=RuntimeError("Search failed")
        )
        mock_services["indexer_service"].search_notes.side_effect = ValueError("Invalid query")

        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "missing.md"}},
            {"name": "search_code", "arguments": {"query": "fail"}},
            {"name": "vault_search", "arguments": {"query": "invalid"}},
        ]

        results = await parallel_executor.execute_batch(tool_calls, "user-123")

        # All should have results (error responses)
        assert len(results) == 3

        # All should contain errors
        for i, result in enumerate(results):
            data = json.loads(result)
            assert "error" in data, f"Result {i} should have error: {data}"

    @pytest.mark.asyncio
    async def test_parallel_tools_empty_batch(
        self, parallel_executor: ToolExecutor
    ) -> None:
        """Empty batch should return empty results list."""
        results = await parallel_executor.execute_batch([], "user-123")
        assert results == []

    @pytest.mark.asyncio
    async def test_parallel_tools_single_tool(
        self, parallel_executor: ToolExecutor, mock_services
    ) -> None:
        """Single tool in batch should work correctly."""
        mock_services["vault_service"].read_note.return_value = {
            "title": "Single Note",
            "body": "Single content",
            "metadata": {},
        }

        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "single.md"}},
        ]

        results = await parallel_executor.execute_batch(tool_calls, "user-123")

        assert len(results) == 1
        data = json.loads(results[0])
        assert data["title"] == "Single Note"

    @pytest.mark.asyncio
    async def test_parallel_tools_mixed_known_unknown(
        self, parallel_executor: ToolExecutor, mock_services
    ) -> None:
        """Batch with unknown tools should return error for unknown, success for known."""
        mock_services["vault_service"].read_note.return_value = {
            "title": "Known Tool Result",
            "body": "Content",
            "metadata": {},
        }

        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "known.md"}},
            {"name": "nonexistent_tool", "arguments": {"foo": "bar"}},
            {"name": "vault_read", "arguments": {"path": "another.md"}},
        ]

        results = await parallel_executor.execute_batch(tool_calls, "user-123")

        assert len(results) == 3

        # Tool 1 (known) - success
        result_0 = json.loads(results[0])
        assert "error" not in result_0
        assert result_0["title"] == "Known Tool Result"

        # Tool 2 (unknown) - error
        result_1 = json.loads(results[1])
        assert "error" in result_1
        assert "Unknown tool" in result_1["error"]

        # Tool 3 (known) - success
        result_2 = json.loads(results[2])
        assert "error" not in result_2

    @pytest.mark.asyncio
    async def test_parallel_tools_default_timeout(
        self, parallel_executor: ToolExecutor, mock_services
    ) -> None:
        """Batch execution should have a sensible default timeout."""
        import asyncio

        call_count = 0

        async def quick_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"results": []}

        mock_services["oracle_bridge"].search_code = quick_tool

        tool_calls = [
            {"name": "search_code", "arguments": {"query": "test"}},
        ]

        # Execute without explicit timeout - should use default
        results = await parallel_executor.execute_batch(tool_calls, "user-123")

        assert len(results) == 1
        assert call_count == 1  # Tool was actually called

    @pytest.mark.asyncio
    async def test_parallel_tools_with_tool_call_ids(
        self, parallel_executor: ToolExecutor, mock_services
    ) -> None:
        """Results should include tool_call_id when provided in input."""
        mock_services["vault_service"].read_note.return_value = {
            "title": "Test",
            "body": "Content",
            "metadata": {},
        }
        mock_services["indexer_service"].search_notes.return_value = []

        tool_calls = [
            {"id": "call_abc123", "name": "vault_read", "arguments": {"path": "a.md"}},
            {"id": "call_def456", "name": "vault_search", "arguments": {"query": "q"}},
        ]

        results = await parallel_executor.execute_batch(
            tool_calls, "user-123", include_call_ids=True
        )

        assert len(results) == 2

        # Results should be tuples of (call_id, result)
        assert results[0][0] == "call_abc123"
        assert results[1][0] == "call_def456"

        # Verify the actual results are still valid
        data_0 = json.loads(results[0][1])
        assert data_0["title"] == "Test"


class TestToolExecutorTimeout:
    """Tests for tool execution timeout handling (009-oracle-agent T028).

    These tests verify that the ToolExecutor properly handles timeout
    configuration and protects against hanging tool calls.
    """

    @pytest.fixture
    def timeout_executor(self, mock_services) -> ToolExecutor:
        """Create a ToolExecutor configured for timeout testing."""
        return ToolExecutor(**mock_services)

    def test_default_timeout_class_attribute(self) -> None:
        """Verify DEFAULT_TIMEOUT class attribute is defined."""
        assert hasattr(ToolExecutor, "DEFAULT_TIMEOUT")
        assert ToolExecutor.DEFAULT_TIMEOUT == 30.0

    def test_tool_timeouts_class_attribute(self) -> None:
        """Verify TOOL_TIMEOUTS class attribute has expected entries."""
        assert hasattr(ToolExecutor, "TOOL_TIMEOUTS")
        timeouts = ToolExecutor.TOOL_TIMEOUTS

        # Check web tools have longer timeouts
        assert timeouts.get("web_search") == 60.0
        assert timeouts.get("web_fetch") == 60.0

        # Check vault tools have shorter timeouts
        assert timeouts.get("vault_read") == 10.0
        assert timeouts.get("vault_write") == 10.0

        # Check delegate has longest timeout (20 minutes for large summarizations and web research)
        assert timeouts.get("delegate_librarian") == 1200.0

    def test_init_with_custom_default_timeout(self, mock_services) -> None:
        """Executor accepts custom default_timeout in constructor."""
        executor = ToolExecutor(**mock_services, default_timeout=45.0)
        assert executor._default_timeout == 45.0

    def test_init_without_custom_timeout_uses_class_default(self, mock_services) -> None:
        """Executor uses DEFAULT_TIMEOUT when no custom timeout provided."""
        executor = ToolExecutor(**mock_services)
        assert executor._default_timeout == ToolExecutor.DEFAULT_TIMEOUT

    def test_get_timeout_returns_tool_specific(self, timeout_executor: ToolExecutor) -> None:
        """get_timeout() returns tool-specific timeout when defined."""
        assert timeout_executor.get_timeout("web_search") == 60.0
        assert timeout_executor.get_timeout("vault_read") == 10.0

    def test_get_timeout_returns_default_for_unknown(self, timeout_executor: ToolExecutor) -> None:
        """get_timeout() returns default for tools not in TOOL_TIMEOUTS."""
        # Tool exists but has no specific timeout
        result = timeout_executor.get_timeout("some_unknown_tool")
        assert result == ToolExecutor.DEFAULT_TIMEOUT

    def test_get_timeout_override_takes_precedence(self, timeout_executor: ToolExecutor) -> None:
        """get_timeout() respects override parameter over tool-specific."""
        # web_search normally has 60s, but override should win
        result = timeout_executor.get_timeout("web_search", override=5.0)
        assert result == 5.0

    def test_get_timeout_none_override_uses_tool_specific(self, timeout_executor: ToolExecutor) -> None:
        """get_timeout() with None override uses tool-specific timeout."""
        result = timeout_executor.get_timeout("web_fetch", override=None)
        assert result == 60.0

    @pytest.mark.asyncio
    async def test_execute_timeout_returns_error(
        self, timeout_executor: ToolExecutor, mock_services
    ) -> None:
        """execute() returns timeout error when tool exceeds timeout."""
        # Use search_code which is already async (oracle_bridge.search_code)
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(0.5)  # Takes longer than timeout
            return {"results": []}

        mock_services["oracle_bridge"].search_code = slow_search

        # Execute with short timeout
        result = await timeout_executor.execute(
            "search_code",
            {"query": "test"},
            "user-123",
            timeout=0.1,
        )

        data = json.loads(result)
        assert "error" in data
        assert "timed out" in data["error"]
        assert data.get("timed_out") is True
        assert data.get("timeout") == 0.1
        assert data.get("tool") == "search_code"

    @pytest.mark.asyncio
    async def test_execute_timeout_error_is_helpful(
        self, timeout_executor: ToolExecutor, mock_services
    ) -> None:
        """Timeout error message helps agent understand what happened."""
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(0.5)
            return {"results": []}

        mock_services["oracle_bridge"].search_code = slow_search

        result = await timeout_executor.execute(
            "search_code",
            {"query": "test"},
            "user-123",
            timeout=0.1,
        )

        data = json.loads(result)

        # Error should mention the tool name
        assert "search_code" in data["error"]

        # Error should mention suggestions for recovery
        assert "reducing the scope" in data["error"] or "smaller" in data["error"]

    @pytest.mark.asyncio
    async def test_execute_success_within_timeout(
        self, timeout_executor: ToolExecutor, mock_services
    ) -> None:
        """execute() returns result when tool completes within timeout."""
        # Use search_code which is already async
        async def fast_search(*args, **kwargs):
            await asyncio.sleep(0.01)  # Fast enough
            return {"results": [{"file": "test.py", "match": "found"}]}

        mock_services["oracle_bridge"].search_code = fast_search

        result = await timeout_executor.execute(
            "search_code",
            {"query": "test"},
            "user-123",
            timeout=1.0,
        )

        data = json.loads(result)
        assert "error" not in data
        assert "results" in data

    @pytest.mark.asyncio
    async def test_execute_uses_tool_specific_timeout(
        self, timeout_executor: ToolExecutor, mock_services
    ) -> None:
        """execute() uses tool-specific timeout when no override provided."""
        # web_search has 60s timeout, so 0.5s should not timeout
        async def medium_search(*args, **kwargs):
            await asyncio.sleep(0.05)
            return {"query": "test", "results": [], "count": 0}

        mock_services["oracle_bridge"].search_code = medium_search

        # Don't provide timeout - should use TOOL_TIMEOUTS[search_code] = 30.0
        result = await timeout_executor.execute(
            "search_code",
            {"query": "test"},
            "user-123",
        )

        data = json.loads(result)
        assert "error" not in data

    @pytest.mark.asyncio
    async def test_execute_override_timeout_takes_precedence(
        self, timeout_executor: ToolExecutor, mock_services
    ) -> None:
        """execute() timeout parameter overrides tool-specific timeout."""
        # Use search_code which is already async
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(0.3)  # Longer than override, shorter than default
            return {"results": []}

        mock_services["oracle_bridge"].search_code = slow_search

        # search_code has 30s default, but we override to 0.1s
        result = await timeout_executor.execute(
            "search_code",
            {"query": "test"},
            "user-123",
            timeout=0.1,
        )

        data = json.loads(result)
        assert "error" in data
        assert data.get("timed_out") is True

    @pytest.mark.asyncio
    async def test_timeout_does_not_affect_other_exceptions(
        self, timeout_executor: ToolExecutor, mock_services
    ) -> None:
        """Other exceptions are still handled correctly with timeout wrapping."""
        mock_services["vault_service"].read_note.side_effect = FileNotFoundError("Not found")

        result = await timeout_executor.execute(
            "vault_read",
            {"path": "missing.md"},
            "user-123",
            timeout=10.0,
        )

        data = json.loads(result)
        assert "error" in data
        assert "File not found" in data["error"]
        assert data.get("timed_out") is not True

    @pytest.mark.asyncio
    async def test_timeout_with_instance_default(self, mock_services) -> None:
        """Executor uses instance default timeout when tool has no specific timeout."""
        executor = ToolExecutor(**mock_services, default_timeout=0.1)

        async def slow_handler(*args, **kwargs):
            await asyncio.sleep(0.3)
            return {"result": "never returned"}

        # Inject a custom tool that has no specific timeout
        async def custom_tool(user_id, **kwargs):
            return await slow_handler()

        executor._tools["custom_tool"] = custom_tool

        result = await executor.execute(
            "custom_tool",
            {},
            "user-123",
        )

        data = json.loads(result)
        assert "error" in data
        assert data.get("timed_out") is True
        assert data.get("timeout") == 0.1  # Instance default
