"""
Unit tests for MCP Tool Leaf Nodes.

Tests cover:
- Tool: Generic MCP tool wrapper
- Oracle: Multi-source oracle query with streaming
- CodeSearch: Code search via CodeRAG
- VaultSearch: Vault note search via BM25
- Parameter interpolation (${bb.key} syntax)
- Error handling (E4003, E6001, E2001)

Part of the BT Universal Runtime (spec 019).
Tasks covered: 4.1.1-4.3.8 from tasks.md
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel

# Import the module under test
from backend.src.bt.nodes.tools import (
    Tool,
    Oracle,
    CodeSearch,
    VaultSearch,
    ToolLeaf,
    interpolate_params,
    ToolNotFoundError,
    ToolTimeoutError,
    MissingToolParameterError,
    ToolResult,
    _interpolate_string,
    _get_bb_value,
    _parse_default,
)
from backend.src.bt.state.base import RunStatus
from backend.src.bt.state.blackboard import TypedBlackboard
from backend.src.bt.core.context import TickContext


# =============================================================================
# Test Models
# =============================================================================


class QueryModel(BaseModel):
    """Test model for query data."""
    text: str
    limit: int = 10


class ContextModel(BaseModel):
    """Test model for nested context data."""
    session_id: str
    user_id: str


class IdentityModel(BaseModel):
    """Test model for identity data."""
    user_id: str


# =============================================================================
# Mock Services
# =============================================================================


class MockToolExecutor:
    """Mock tool executor for testing."""

    def __init__(self, results: Optional[Dict[str, Any]] = None):
        self.results = results or {}
        self.execute_calls: List[tuple] = []
        self.check_calls: List[str] = []
        self.cancel_calls: List[str] = []
        self.pending_async: Dict[str, Any] = {}

    def execute(self, tool_name: str, params: Dict, ctx: Any) -> Any:
        self.execute_calls.append((tool_name, params, ctx))

        if tool_name in self.results:
            result = self.results[tool_name]
            if isinstance(result, str) and result.startswith("async:"):
                self.pending_async[result[6:]] = {"pending": True}
            return result

        raise ToolNotFoundError(tool_name, "test-node", list(self.results.keys()))

    def check_completion(self, request_id: str) -> Optional[Any]:
        self.check_calls.append(request_id)
        if request_id in self.pending_async:
            pending = self.pending_async.pop(request_id)
            return pending.get("result", {"success": True})
        return None

    def cancel(self, request_id: str) -> None:
        self.cancel_calls.append(request_id)
        self.pending_async.pop(request_id, None)


class MockOracleBridge:
    """Mock oracle bridge for testing."""

    def __init__(self):
        self.ask_oracle_calls: List[Dict] = []
        self.search_code_calls: List[Dict] = []
        self.find_definition_calls: List[Dict] = []
        self.find_references_calls: List[Dict] = []
        self.get_repo_map_calls: List[Dict] = []
        self.ask_result: Dict[str, Any] = {}
        self.search_result: Dict[str, Any] = {}

    async def ask_oracle(
        self,
        question: str,
        sources: Optional[List[str]] = None,
        explain: bool = False,
    ) -> Dict[str, Any]:
        self.ask_oracle_calls.append({
            "question": question,
            "sources": sources,
            "explain": explain,
        })
        return self.ask_result or {
            "answer": "Test answer",
            "sources": [],
            "tokens_used": 100,
        }

    async def search_code(
        self,
        query: str,
        limit: int = 10,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.search_code_calls.append({
            "query": query,
            "limit": limit,
            "language": language,
            "file_pattern": file_pattern,
        })
        return self.search_result or {
            "results": [{"file": "test.py", "line": 1}],
        }

    async def find_definition(
        self,
        symbol: str,
        scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.find_definition_calls.append({
            "symbol": symbol,
            "scope": scope,
        })
        return {"definitions": []}

    async def find_references(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        self.find_references_calls.append({
            "symbol": symbol,
            "limit": limit,
        })
        return {"references": []}

    async def get_repo_map(
        self,
        scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.get_repo_map_calls.append({"scope": scope})
        return {"map": "test map"}


class MockIndexer:
    """Mock indexer service for testing."""

    def __init__(self):
        self.search_calls: List[Dict] = []
        self.search_result: List[Dict] = []

    def search_notes(
        self,
        user_id: str,
        query: str,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict]:
        self.search_calls.append({
            "user_id": user_id,
            "query": query,
            "tags": tags,
            "limit": limit,
        })
        return self.search_result or [{"path": "note.md", "title": "Test Note"}]


class MockServices:
    """Mock services container."""

    def __init__(self):
        self.tool_executor = MockToolExecutor()
        self.oracle_bridge = MockOracleBridge()
        self.indexer = MockIndexer()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def blackboard() -> TypedBlackboard:
    """Create a blackboard for testing."""
    bb = TypedBlackboard(scope_name="test")
    bb.register("query", QueryModel)
    bb.register("context", ContextModel)
    bb.register("identity", IdentityModel)
    return bb


@pytest.fixture
def context(blackboard: TypedBlackboard) -> TickContext:
    """Create a tick context for testing."""
    return TickContext(
        blackboard=blackboard,
        services=MockServices(),
    )


# =============================================================================
# Parameter Interpolation Tests
# =============================================================================


class TestInterpolation:
    """Tests for ${bb.key} parameter interpolation."""

    def test_simple_key_interpolation(self, blackboard: TypedBlackboard):
        """Test basic key interpolation."""
        blackboard.set("query", QueryModel(text="hello", limit=5))

        params = {"search": "${bb.query}"}
        result = interpolate_params(params, blackboard)

        assert isinstance(result["search"], QueryModel)
        assert result["search"].text == "hello"

    def test_nested_key_interpolation(self, blackboard: TypedBlackboard):
        """Test nested key access with dot notation."""
        blackboard.set("context", ContextModel(session_id="sess123", user_id="user456"))

        # Test with the internal lookup
        value = _get_bb_value(blackboard, "context.session_id")
        assert value == "sess123"

    def test_default_value_interpolation(self, blackboard: TypedBlackboard):
        """Test default value when key is missing."""
        params = {"limit": "${bb.missing_key | default:20}"}
        result = interpolate_params(params, blackboard)

        assert result["limit"] == 20

    def test_default_string_interpolation(self, blackboard: TypedBlackboard):
        """Test default string value."""
        params = {"name": "${bb.missing | default:unknown}"}
        result = interpolate_params(params, blackboard)

        assert result["name"] == "unknown"

    def test_embedded_interpolation(self, blackboard: TypedBlackboard):
        """Test interpolation embedded in string."""
        blackboard.set("query", QueryModel(text="test", limit=10))

        s = "Search for: ${bb.query.text} with limit ${bb.query.limit}"
        result = _interpolate_string(s, blackboard)

        assert "Search for: test with limit 10" == result

    def test_no_interpolation_needed(self, blackboard: TypedBlackboard):
        """Test params without interpolation patterns."""
        params = {"static": "value", "number": 42}
        result = interpolate_params(params, blackboard)

        assert result == params

    def test_nested_dict_interpolation(self, blackboard: TypedBlackboard):
        """Test interpolation in nested dicts."""
        blackboard.set("query", QueryModel(text="nested", limit=5))

        params = {
            "outer": {
                "inner": "${bb.query}",
            }
        }
        result = interpolate_params(params, blackboard)

        assert isinstance(result["outer"]["inner"], QueryModel)

    def test_list_interpolation(self, blackboard: TypedBlackboard):
        """Test interpolation in lists."""
        blackboard.set("query", QueryModel(text="list", limit=3))

        params = {
            "items": ["${bb.query}", "static"]
        }
        result = interpolate_params(params, blackboard)

        assert isinstance(result["items"][0], QueryModel)
        assert result["items"][1] == "static"


class TestParseDefault:
    """Tests for default value parsing."""

    def test_parse_int(self):
        assert _parse_default("42") == 42

    def test_parse_float(self):
        assert _parse_default("3.14") == 3.14

    def test_parse_true(self):
        assert _parse_default("true") is True
        assert _parse_default("True") is True

    def test_parse_false(self):
        assert _parse_default("false") is False
        assert _parse_default("False") is False

    def test_parse_null(self):
        assert _parse_default("null") is None
        assert _parse_default("none") is None

    def test_parse_quoted_string(self):
        assert _parse_default('"hello"') == "hello"
        assert _parse_default("'world'") == "world"

    def test_parse_plain_string(self):
        assert _parse_default("plain") == "plain"


# =============================================================================
# Tool Node Tests
# =============================================================================


class TestTool:
    """Tests for the generic Tool node."""

    def test_tool_initialization(self):
        """Test Tool node initialization."""
        tool = Tool(
            id="test-tool",
            tool_name="search_notes",
            params={"query": "test"},
            output="results",
        )

        assert tool._id == "test-tool"
        assert tool._tool_name == "search_notes"
        assert tool._params == {"query": "test"}
        assert tool._output_key == "results"

    def test_tool_sync_execution_success(self, context: TickContext):
        """Test successful sync tool execution."""
        # Setup mock
        context.services.tool_executor.results = {
            "search_notes": [{"title": "Note 1"}]
        }

        tool = Tool(
            id="sync-tool",
            tool_name="search_notes",
            params={"query": "test"},
            output="results",
        )

        result = tool.tick(context)

        assert result == RunStatus.SUCCESS
        assert len(context.services.tool_executor.execute_calls) == 1

    def test_tool_async_execution(self, context: TickContext):
        """Test async tool execution."""
        # Setup mock to return async indicator
        context.services.tool_executor.results = {
            "async_tool": "async:request-123"
        }
        context.services.tool_executor.pending_async["request-123"] = {
            "pending": True,
            "result": {"success": True},
        }

        tool = Tool(
            id="async-tool",
            tool_name="async_tool",
            params={},
            output="results",
        )

        # First tick starts async operation
        result = tool.tick(context)
        assert result == RunStatus.RUNNING

        # Second tick completes
        result = tool.tick(context)
        assert result == RunStatus.SUCCESS

    def test_tool_not_found(self, context: TickContext):
        """Test E4003 error when tool not found."""
        tool = Tool(
            id="missing-tool",
            tool_name="nonexistent",
            params={},
            output="results",
        )

        result = tool.tick(context)

        assert result == RunStatus.FAILURE
        # Check error was written
        error = context.blackboard._lookup("_tool_error")
        assert error is not None
        assert "E4003" in error

    def test_tool_param_interpolation(self, context: TickContext):
        """Test parameter interpolation before execution."""
        context.blackboard.set("query", QueryModel(text="interpolated", limit=5))
        context.services.tool_executor.results = {
            "search_notes": []
        }

        tool = Tool(
            id="interp-tool",
            tool_name="search_notes",
            params={"query": "${bb.query}"},
            output="results",
        )

        tool.tick(context)

        # Check that interpolated params were passed
        _, params, _ = context.services.tool_executor.execute_calls[0]
        assert isinstance(params["query"], QueryModel)

    def test_tool_debug_info(self):
        """Test debug info output."""
        tool = Tool(
            id="debug-tool",
            tool_name="test",
            params={"key": "value"},
            output="out",
            timeout_ms=5000,
        )

        info = tool.debug_info()

        assert info["tool_name"] == "test"
        assert info["output_key"] == "out"
        assert info["timeout_ms"] == 5000
        assert info["params_template"] == {"key": "value"}


# =============================================================================
# Oracle Node Tests
# =============================================================================


class TestOracle:
    """Tests for the Oracle node."""

    def test_oracle_initialization(self):
        """Test Oracle node initialization."""
        oracle = Oracle(
            id="test-oracle",
            question="What is this?",
            sources=["code", "vault"],
            explain=True,
            stream_to="chunks",
            output="answer",
        )

        assert oracle._id == "test-oracle"
        assert oracle._question_template == "What is this?"
        assert oracle._sources == ["code", "vault"]
        assert oracle._explain is True
        assert oracle._stream_to == "chunks"
        assert oracle._output_key == "answer"

    def test_oracle_default_values(self):
        """Test Oracle with default values."""
        oracle = Oracle(
            id="default-oracle",
            question="Test?",
        )

        assert oracle._sources == ["code", "vault", "threads"]
        assert oracle._explain is False
        assert oracle._stream_to is None
        assert oracle._output_key == "oracle_answer"

    def test_oracle_question_interpolation(self, context: TickContext):
        """Test question interpolation from blackboard."""
        context.blackboard.set("query", QueryModel(text="my question", limit=1))

        oracle = Oracle(
            id="interp-oracle",
            question="${bb.query.text}",
        )

        # First tick starts the request
        result = oracle.tick(context)

        # Check stored request params
        request_key = f"_oracle_request_{oracle._request_id}"
        request = context.blackboard._lookup(request_key)
        assert request is not None
        assert request["question"] == "my question"

    def test_oracle_empty_question_fails(self, context: TickContext):
        """Test failure on empty question."""
        oracle = Oracle(
            id="empty-oracle",
            question="${bb.missing}",
        )

        result = oracle.tick(context)

        assert result == RunStatus.FAILURE

    def test_oracle_debug_info(self):
        """Test Oracle debug info."""
        oracle = Oracle(
            id="debug-oracle",
            question="test?",
            sources=["code"],
            stream_to="stream",
        )

        info = oracle.debug_info()

        assert info["question_template"] == "test?"
        assert info["sources"] == ["code"]
        assert info["stream_to"] == "stream"


# =============================================================================
# CodeSearch Node Tests
# =============================================================================


class TestCodeSearch:
    """Tests for the CodeSearch node."""

    def test_codesearch_initialization(self):
        """Test CodeSearch node initialization."""
        cs = CodeSearch(
            id="test-cs",
            operation="search",
            query="function",
            limit=20,
            language="python",
            file_pattern="*.py",
            output="results",
        )

        assert cs._id == "test-cs"
        assert cs._operation == "search"
        assert cs._query_template == "function"
        assert cs._limit == 20
        assert cs._language == "python"
        assert cs._file_pattern == "*.py"
        assert cs._output_key == "results"

    def test_codesearch_valid_operations(self):
        """Test that only valid operations are accepted."""
        for op in ["search", "definition", "references", "repo_map"]:
            cs = CodeSearch(id=f"cs-{op}", operation=op, query="test")
            assert cs._operation == op

    def test_codesearch_invalid_operation(self):
        """Test that invalid operation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operation"):
            CodeSearch(id="bad", operation="invalid")

    def test_codesearch_repo_map_no_query(self, context: TickContext):
        """Test repo_map operation doesn't require query."""
        cs = CodeSearch(
            id="repo-map",
            operation="repo_map",
            scope="src/",
        )

        # First tick starts search
        result = cs.tick(context)

        # Should be RUNNING (async operation)
        assert result == RunStatus.RUNNING

    def test_codesearch_search_requires_query(self, context: TickContext):
        """Test search operation requires query."""
        cs = CodeSearch(
            id="no-query",
            operation="search",
            query=None,
        )

        result = cs.tick(context)

        assert result == RunStatus.FAILURE

    def test_codesearch_debug_info(self):
        """Test CodeSearch debug info."""
        cs = CodeSearch(
            id="debug-cs",
            operation="definition",
            query="MyClass",
            language="typescript",
        )

        info = cs.debug_info()

        assert info["operation"] == "definition"
        assert info["query_template"] == "MyClass"
        assert info["language"] == "typescript"


# =============================================================================
# VaultSearch Node Tests
# =============================================================================


class TestVaultSearch:
    """Tests for the VaultSearch node."""

    def test_vaultsearch_initialization(self):
        """Test VaultSearch node initialization."""
        vs = VaultSearch(
            id="test-vs",
            query="project notes",
            tags=["project", "design"],
            limit=5,
            output="docs",
        )

        assert vs._id == "test-vs"
        assert vs._query_template == "project notes"
        assert vs._tags == ["project", "design"]
        assert vs._limit == 5
        assert vs._output_key == "docs"

    def test_vaultsearch_defaults(self):
        """Test VaultSearch default values."""
        vs = VaultSearch(
            id="default-vs",
            query="test",
        )

        assert vs._tags is None
        assert vs._limit == 10
        assert vs._output_key == "notes"

    def test_vaultsearch_execution(self, context: TickContext):
        """Test VaultSearch execution."""
        context.services.indexer.search_result = [
            {"path": "note1.md", "title": "Note 1", "score": 0.9},
            {"path": "note2.md", "title": "Note 2", "score": 0.8},
        ]

        vs = VaultSearch(
            id="exec-vs",
            query="test query",
            tags=["design"],
            limit=10,
        )

        result = vs.tick(context)

        assert result == RunStatus.SUCCESS

        # Check search was called correctly
        call = context.services.indexer.search_calls[0]
        assert call["query"] == "test query"
        assert call["tags"] == ["design"]
        assert call["limit"] == 10

    def test_vaultsearch_query_interpolation(self, context: TickContext):
        """Test VaultSearch query interpolation."""
        context.blackboard.set("query", QueryModel(text="interpolated query", limit=5))

        vs = VaultSearch(
            id="interp-vs",
            query="${bb.query.text}",
        )

        vs.tick(context)

        call = context.services.indexer.search_calls[0]
        assert call["query"] == "interpolated query"

    def test_vaultsearch_empty_query_fails(self, context: TickContext):
        """Test failure on empty query."""
        vs = VaultSearch(
            id="empty-vs",
            query="${bb.missing}",
        )

        result = vs.tick(context)

        assert result == RunStatus.FAILURE

    def test_vaultsearch_gets_user_id(self, context: TickContext):
        """Test VaultSearch extracts user_id from identity."""
        context.blackboard.set("identity", IdentityModel(user_id="test-user"))

        vs = VaultSearch(
            id="user-vs",
            query="test",
        )

        vs.tick(context)

        call = context.services.indexer.search_calls[0]
        assert call["user_id"] == "test-user"

    def test_vaultsearch_debug_info(self):
        """Test VaultSearch debug info."""
        vs = VaultSearch(
            id="debug-vs",
            query="debug query",
            tags=["tag1", "tag2"],
        )

        info = vs.debug_info()

        assert info["query_template"] == "debug query"
        assert info["tags"] == ["tag1", "tag2"]


# =============================================================================
# Error Class Tests
# =============================================================================


class TestErrorClasses:
    """Tests for error classes."""

    def test_tool_not_found_error(self):
        """Test ToolNotFoundError E4003."""
        error = ToolNotFoundError(
            tool_name="missing_tool",
            node_id="test-node",
            available_tools=["tool1", "tool2"],
        )

        assert error.error_code == "E4003"
        assert "missing_tool" in str(error)
        assert "E4003" in str(error)

    def test_tool_timeout_error(self):
        """Test ToolTimeoutError E6001."""
        error = ToolTimeoutError(
            tool_name="slow_tool",
            node_id="test-node",
            timeout_ms=5000,
            operation_id="op-123",
        )

        assert error.error_code == "E6001"
        assert "5000ms" in str(error)
        assert "E6001" in str(error)

    def test_missing_tool_parameter_error(self):
        """Test MissingToolParameterError E2001."""
        error = MissingToolParameterError(
            tool_name="my_tool",
            node_id="test-node",
            param_name="required_param",
            expected_type="str",
        )

        assert error.error_code == "E2001"
        assert "required_param" in str(error)
        assert "E2001" in str(error)


# =============================================================================
# ToolLeaf Base Class Tests
# =============================================================================


class TestToolLeaf:
    """Tests for ToolLeaf base class."""

    def test_timeout_check(self):
        """Test timeout checking logic."""
        tool = Tool(
            id="timeout-test",
            tool_name="test",
            params={},
            output="out",
            timeout_ms=100,  # Very short timeout
        )

        # Start the timer
        tool._tool_start_time = datetime.now(timezone.utc)

        # Should not be timed out immediately
        assert not tool._check_timeout()

        # Set start time to past
        from datetime import timedelta
        tool._tool_start_time = datetime.now(timezone.utc) - timedelta(milliseconds=200)

        # Should be timed out now
        assert tool._check_timeout()

    def test_elapsed_ms(self):
        """Test elapsed time calculation."""
        tool = Tool(
            id="elapsed-test",
            tool_name="test",
            params={},
            output="out",
        )

        # Not started
        assert tool._get_elapsed_ms() == 0.0

        # Start timer
        tool._tool_start_time = datetime.now(timezone.utc)

        # Should have some elapsed time (small but > 0)
        elapsed = tool._get_elapsed_ms()
        assert elapsed >= 0

    def test_reset_clears_state(self):
        """Test reset clears internal state."""
        tool = Tool(
            id="reset-test",
            tool_name="test",
            params={},
            output="out",
        )

        # Set some state
        tool._request_id = "request-123"
        tool._tool_start_time = datetime.now(timezone.utc)

        # Reset
        tool.reset()

        assert tool._request_id is None
        assert tool._tool_start_time is None


# =============================================================================
# ToolResult Tests
# =============================================================================


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(success=True, value={"data": "test"})

        assert result.success
        assert result.value == {"data": "test"}
        assert result.error is None

    def test_failure_result(self):
        """Test failure result."""
        result = ToolResult(success=False, error="Something went wrong")

        assert not result.success
        assert result.value is None
        assert result.error == "Something went wrong"

    def test_duration_tracking(self):
        """Test duration_ms field."""
        result = ToolResult(success=True, duration_ms=123.45)

        assert result.duration_ms == 123.45


# =============================================================================
# Integration Tests
# =============================================================================


class TestToolIntegration:
    """Integration tests for tool nodes."""

    def test_tool_node_in_tree(self, context: TickContext):
        """Test tool node works in a tree context."""
        context.services.tool_executor.results = {
            "search_notes": [{"title": "Found"}]
        }

        tool = Tool(
            id="tree-tool",
            tool_name="search_notes",
            params={"query": "test"},
            output="results",
        )

        # Simulate tree tick
        context.push_path("root")
        result = tool.tick(context)
        context.pop_path()

        assert result == RunStatus.SUCCESS

    def test_multiple_tool_executions(self, context: TickContext):
        """Test multiple sequential tool executions."""
        context.services.tool_executor.results = {
            "tool1": "result1",
            "tool2": "result2",
        }

        tool1 = Tool(id="t1", tool_name="tool1", params={}, output="out1")
        tool2 = Tool(id="t2", tool_name="tool2", params={}, output="out2")

        r1 = tool1.tick(context)
        r2 = tool2.tick(context)

        assert r1 == RunStatus.SUCCESS
        assert r2 == RunStatus.SUCCESS
        assert len(context.services.tool_executor.execute_calls) == 2

    def test_tool_reset_allows_reexecution(self, context: TickContext):
        """Test that reset allows re-execution."""
        context.services.tool_executor.results = {
            "reusable": "result"
        }

        tool = Tool(
            id="reusable",
            tool_name="reusable",
            params={},
            output="out",
        )

        # First execution
        r1 = tool.tick(context)
        assert r1 == RunStatus.SUCCESS

        # Reset and re-execute
        tool.reset()
        r2 = tool.tick(context)
        assert r2 == RunStatus.SUCCESS

        # Should have two calls
        assert len(context.services.tool_executor.execute_calls) == 2
