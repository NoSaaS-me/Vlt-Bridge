"""
Unit tests for TreeLoader.

Tests the TreeLoader class from lua/loader.py:
- Loading from file
- Loading from string
- Error handling (E4001, E4002, E5001, E5003)
- BT.* API injection
- Tree validation

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.6.1-2.6.6 from tasks.md
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from backend.src.bt.lua.loader import (
    TreeLoader,
    TreeLoadError,
    validate_tree_name,
    validate_node_id,
)
from backend.src.bt.lua.api import TreeDefinition, NodeDefinition


# =============================================================================
# Helper to check if lupa is available
# =============================================================================


def lupa_available() -> bool:
    """Check if lupa is installed."""
    try:
        import lupa
        return True
    except ImportError:
        return False


# Mark for skipping if lupa not available
requires_lupa = pytest.mark.skipif(
    not lupa_available(),
    reason="lupa not installed"
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def loader() -> TreeLoader:
    """Create a test TreeLoader."""
    return TreeLoader(sandbox_timeout=2.0)


@pytest.fixture
def temp_lua_file():
    """Create a temporary Lua file for testing."""
    files_created = []

    def _create_file(content: str, suffix: str = ".lua") -> Path:
        f = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            delete=False,
            encoding="utf-8",
        )
        f.write(content)
        f.close()
        path = Path(f.name)
        files_created.append(path)
        return path

    yield _create_file

    # Cleanup
    for path in files_created:
        if path.exists():
            path.unlink()


# =============================================================================
# TreeLoadError Tests
# =============================================================================


class TestTreeLoadError:
    """Tests for TreeLoadError exception."""

    def test_basic_error(self) -> None:
        """Test creating a basic error."""
        error = TreeLoadError(
            error_code="E4001",
            message="File not found",
        )

        assert error.error_code == "E4001"
        assert "E4001" in str(error)
        assert "File not found" in str(error)

    def test_error_with_source_path(self) -> None:
        """Test error with source path context."""
        error = TreeLoadError(
            error_code="E5001",
            message="Unexpected symbol",
            source_path="/trees/oracle.lua",
        )

        assert error.source_path == "/trees/oracle.lua"
        assert "oracle.lua" in str(error)

    def test_error_with_line_number(self) -> None:
        """Test error with line number context."""
        error = TreeLoadError(
            error_code="E5001",
            message="Unexpected symbol",
            source_path="/trees/oracle.lua",
            line_number=42,
        )

        assert error.line_number == 42
        assert "line 42" in str(error)


# =============================================================================
# Validation Function Tests
# =============================================================================


class TestValidationFunctions:
    """Tests for tree/node name validation."""

    def test_valid_tree_names(self) -> None:
        """Test valid tree names per tree-loader.yaml."""
        valid_names = [
            "oracle-agent",
            "research_runner",
            "MyTree123",
            "a",
            "A",
            "test",
            "node-1",
            "node_1",
        ]

        for name in valid_names:
            assert validate_tree_name(name), f"'{name}' should be valid"

    def test_invalid_tree_names(self) -> None:
        """Test invalid tree names per tree-loader.yaml."""
        invalid_names = [
            "../parent",      # Path traversal
            "my.tree",        # Contains dot
            "tree/sub",       # Contains slash
            "123start",       # Starts with number
            "-start",         # Starts with hyphen
            "_start",         # Starts with underscore
            "",               # Empty
            " ",              # Whitespace
            "tree name",      # Contains space
        ]

        for name in invalid_names:
            assert not validate_tree_name(name), f"'{name}' should be invalid"

    def test_validate_node_id(self) -> None:
        """Test node ID validation (same rules as tree names)."""
        assert validate_node_id("action-1")
        assert validate_node_id("MyAction")
        assert not validate_node_id("123action")
        assert not validate_node_id("")


# =============================================================================
# TreeLoader File Loading Tests
# =============================================================================


class TestTreeLoaderFileLoading:
    """Tests for loading trees from files."""

    def test_file_not_found_error(self, loader: TreeLoader) -> None:
        """Test E4001 error when file doesn't exist."""
        path = Path("/nonexistent/tree.lua")

        with pytest.raises(TreeLoadError) as exc_info:
            loader.load(path)

        assert exc_info.value.error_code == "E4001"
        assert "not found" in str(exc_info.value).lower()

    def test_wrong_extension_error(
        self,
        loader: TreeLoader,
        temp_lua_file,
    ) -> None:
        """Test E4001 error when file has wrong extension."""
        path = temp_lua_file("content", suffix=".txt")

        with pytest.raises(TreeLoadError) as exc_info:
            loader.load(path)

        assert exc_info.value.error_code == "E4001"
        assert ".lua" in str(exc_info.value)

    @requires_lupa
    def test_load_valid_file(
        self,
        loader: TreeLoader,
        temp_lua_file,
    ) -> None:
        """Test loading a valid Lua tree file."""
        lua_content = """
return BT.tree("test-tree", {
    description = "A test tree",
    root = BT.sequence({
        BT.action("step1", {fn = "test.step1"}),
        BT.action("step2", {fn = "test.step2"}),
    })
})
"""
        path = temp_lua_file(lua_content)

        tree = loader.load(path)

        assert isinstance(tree, TreeDefinition)
        assert tree.name == "test-tree"
        assert tree.description == "A test tree"
        assert tree.root.type == "sequence"
        assert len(tree.root.children) == 2
        assert tree.source_path == str(path)
        assert tree.source_hash != ""  # Hash should be computed

    @requires_lupa
    def test_source_hash_computed(
        self,
        loader: TreeLoader,
        temp_lua_file,
    ) -> None:
        """Test that source hash is computed on load."""
        lua_content = "return BT.tree('test', {root = BT.sequence({})})"
        path = temp_lua_file(lua_content)

        tree = loader.load(path)

        assert tree.source_hash != ""
        assert len(tree.source_hash) == 64  # SHA256 hex


# =============================================================================
# TreeLoader String Loading Tests
# =============================================================================


class TestTreeLoaderStringLoading:
    """Tests for loading trees from strings."""

    @requires_lupa
    def test_load_simple_tree(self, loader: TreeLoader) -> None:
        """Test loading a simple tree from string."""
        lua_code = """
return BT.tree("simple", {
    root = BT.sequence({})
})
"""
        tree = loader.load_string(lua_code)

        assert isinstance(tree, TreeDefinition)
        assert tree.name == "simple"
        assert tree.root.type == "sequence"

    @requires_lupa
    def test_load_complex_tree(self, loader: TreeLoader) -> None:
        """Test loading a complex tree from string."""
        lua_code = """
return BT.tree("complex", {
    description = "Complex tree",
    blackboard = {
        query = "QueryModel",
        response = "ResponseModel",
    },
    root = BT.selector({
        BT.guard("bb.has_context",
            BT.sequence({
                BT.action("load", {fn = "ctx.load"}),
                BT.action("process", {fn = "ctx.process"}),
            })
        ),
        BT.sequence({
            BT.action("init", {fn = "ctx.init"}),
            BT.action("process", {fn = "ctx.process"}),
        })
    })
})
"""
        tree = loader.load_string(lua_code)

        assert tree.name == "complex"
        assert tree.description == "Complex tree"
        assert tree.blackboard_schema["query"] == "QueryModel"
        assert tree.root.type == "selector"
        assert len(tree.root.children) == 2
        assert tree.root.children[0].type == "guard"
        assert tree.root.children[1].type == "sequence"

    @requires_lupa
    def test_load_with_decorators(self, loader: TreeLoader) -> None:
        """Test loading tree with decorator nodes."""
        lua_code = """
return BT.tree("decorated", {
    root = BT.timeout(30000,
        BT.retry(3,
            BT.action("api_call", {fn = "api.call"}),
            {backoff_ms = 1000}
        )
    )
})
"""
        tree = loader.load_string(lua_code)

        assert tree.root.type == "timeout"
        assert tree.root.config["timeout_ms"] == 30000
        assert tree.root.children[0].type == "retry"
        assert tree.root.children[0].config["max_retries"] == 3

    @requires_lupa
    def test_load_with_parallel(self, loader: TreeLoader) -> None:
        """Test loading tree with parallel node."""
        lua_code = """
return BT.tree("parallel-tree", {
    root = BT.parallel({
        policy = "require_all",
        merge_strategy = "last_wins",
    }, {
        BT.action("search1", {fn = "search.code"}),
        BT.action("search2", {fn = "search.vault"}),
    })
})
"""
        tree = loader.load_string(lua_code)

        assert tree.root.type == "parallel"
        assert tree.root.config["policy"] == "require_all"
        assert len(tree.root.children) == 2


# =============================================================================
# TreeLoader Error Handling Tests
# =============================================================================


class TestTreeLoaderErrors:
    """Tests for TreeLoader error handling."""

    @requires_lupa
    def test_syntax_error(self, loader: TreeLoader) -> None:
        """Test E5001 on Lua syntax error."""
        lua_code = "return {{{{invalid syntax"

        with pytest.raises(TreeLoadError) as exc_info:
            loader.load_string(lua_code)

        assert exc_info.value.error_code == "E5001"

    @requires_lupa
    def test_no_return_error(self, loader: TreeLoader) -> None:
        """Test E4002 when script doesn't return a tree."""
        lua_code = """
local x = 1
local y = 2
-- No return statement
"""
        with pytest.raises(TreeLoadError) as exc_info:
            loader.load_string(lua_code)

        assert exc_info.value.error_code == "E4002"
        assert "did not return" in str(exc_info.value).lower()

    @requires_lupa
    def test_returns_node_instead_of_tree(self, loader: TreeLoader) -> None:
        """Test E4002 when script returns node instead of tree."""
        lua_code = """
return BT.sequence({
    BT.action("test", {fn = "test.fn"})
})
"""
        with pytest.raises(TreeLoadError) as exc_info:
            loader.load_string(lua_code)

        assert exc_info.value.error_code == "E4002"
        assert "NodeDefinition" in str(exc_info.value)
        assert "TreeDefinition" in str(exc_info.value)

    @requires_lupa
    def test_returns_wrong_type(self, loader: TreeLoader) -> None:
        """Test E4002 when script returns wrong type."""
        lua_code = "return 42"

        with pytest.raises(TreeLoadError) as exc_info:
            loader.load_string(lua_code)

        assert exc_info.value.error_code == "E4002"

    @requires_lupa
    def test_tree_requires_root_error(self, loader: TreeLoader) -> None:
        """Test error when BT.tree() called without root."""
        lua_code = """
return BT.tree("test", {
    description = "No root!"
})
"""
        # This should raise a ValueError from BTApiBuilder
        with pytest.raises((TreeLoadError, ValueError)):
            loader.load_string(lua_code)

    @requires_lupa
    def test_tree_requires_name_error(self, loader: TreeLoader) -> None:
        """Test error when BT.tree() called without name."""
        lua_code = """
return BT.tree("", {
    root = BT.sequence({})
})
"""
        with pytest.raises((TreeLoadError, ValueError)):
            loader.load_string(lua_code)


# =============================================================================
# TreeLoader BT.* API Injection Tests
# =============================================================================


@requires_lupa
class TestBTApiInjection:
    """Tests for BT.* API availability in Lua environment."""

    def test_bt_namespace_available(self, loader: TreeLoader) -> None:
        """Test that BT namespace is available in Lua."""
        lua_code = """
return BT.tree("test", {
    root = BT.sequence({})
})
"""
        tree = loader.load_string(lua_code)
        assert tree.name == "test"

    def test_all_composites_available(self, loader: TreeLoader) -> None:
        """Test that all composite functions are available."""
        lua_code = """
return BT.tree("test", {
    root = BT.selector({
        BT.sequence({}),
        BT.parallel({}, {}),
    })
})
"""
        tree = loader.load_string(lua_code)
        assert tree.root.type == "selector"

    def test_all_decorators_available(self, loader: TreeLoader) -> None:
        """Test that all decorator functions are available."""
        lua_code = """
local action = BT.action("test", {fn = "test.fn"})
return BT.tree("test", {
    root = BT.sequence({
        BT.timeout(1000, action),
        BT.retry(3, action),
        BT.guard("true", action),
        BT.cooldown(1000, action),
        BT.inverter(action),
        BT.always_succeed(action),
        BT.always_fail(action),
    })
})
"""
        tree = loader.load_string(lua_code)
        children = tree.root.children
        assert children[0].type == "timeout"
        assert children[1].type == "retry"
        assert children[2].type == "guard"
        assert children[3].type == "cooldown"
        assert children[4].type == "inverter"
        assert children[5].type == "always_succeed"
        assert children[6].type == "always_fail"

    def test_all_leaves_available(self, loader: TreeLoader) -> None:
        """Test that all leaf functions are available."""
        lua_code = """
return BT.tree("test", {
    root = BT.sequence({
        BT.action("a", {fn = "test.fn"}),
        BT.condition("c", {condition = "true"}),
        BT.llm_call({model = "test", prompt_key = "p", response_key = "r"}),
        BT.subtree_ref("other"),
        BT.script("s", {lua = "return {status = 'success'}"}),
    })
})
"""
        tree = loader.load_string(lua_code)
        children = tree.root.children
        assert children[0].type == "action"
        assert children[1].type == "condition"
        assert children[2].type == "llm_call"
        assert children[3].type == "subtree_ref"
        assert children[4].type == "script"

    def test_mcp_integration_nodes_available(self, loader: TreeLoader) -> None:
        """Test that MCP integration nodes are available."""
        lua_code = """
return BT.tree("test", {
    root = BT.sequence({
        BT.tool("search_notes", {query = "test", output = "r1"}),
        BT.oracle({question = "test", output = "r2"}),
        BT.code_search({operation = "search", query = "test", output = "r3"}),
        BT.vault_search({query = "test", output = "r4"}),
    })
})
"""
        tree = loader.load_string(lua_code)
        children = tree.root.children
        assert children[0].type == "tool"
        assert children[1].type == "oracle"
        assert children[2].type == "code_search"
        assert children[3].type == "vault_search"

    def test_for_each_available(self, loader: TreeLoader) -> None:
        """Test that for_each is available."""
        lua_code = """
return BT.tree("test", {
    root = BT.for_each("items", {
        item_key = "item",
        children = {
            BT.action("process", {fn = "test.process"})
        }
    })
})
"""
        tree = loader.load_string(lua_code)
        assert tree.root.type == "for_each"
        assert tree.root.config["collection_key"] == "items"

    def test_contract_available(self, loader: TreeLoader) -> None:
        """Test that BT.contract is available."""
        lua_code = """
local contract = BT.contract({
    inputs = {session_id = "SessionIdModel"},
    outputs = {context = "ContextModel"},
})
-- Contract is used within scripts, just verify it works
return BT.tree("test", {
    root = BT.sequence({})
})
"""
        tree = loader.load_string(lua_code)
        assert tree.name == "test"


# =============================================================================
# TreeLoader Timeout Tests
# =============================================================================


@requires_lupa
class TestTreeLoaderTimeout:
    """Tests for TreeLoader timeout handling."""

    def test_timeout_on_infinite_loop(self) -> None:
        """Test E5003 on infinite loop in tree definition."""
        loader = TreeLoader(sandbox_timeout=0.5)

        lua_code = """
while true do end
return BT.tree("never", {root = BT.sequence({})})
"""
        with pytest.raises(TreeLoadError) as exc_info:
            loader.load_string(lua_code)

        assert exc_info.value.error_code == "E5003"

    def test_fast_load_succeeds(self) -> None:
        """Test that fast loads complete successfully."""
        loader = TreeLoader(sandbox_timeout=5.0)

        lua_code = """
return BT.tree("fast", {
    root = BT.sequence({})
})
"""
        tree = loader.load_string(lua_code)
        assert tree.name == "fast"


# =============================================================================
# Integration Tests
# =============================================================================


@requires_lupa
class TestTreeLoaderIntegration:
    """Integration tests for TreeLoader."""

    def test_oracle_agent_like_tree(self, loader: TreeLoader) -> None:
        """Test loading a tree structure similar to oracle-agent."""
        lua_code = """
return BT.tree("oracle-agent", {
    description = "Oracle agent behavior tree",
    blackboard = {
        query = "QueryModel",
        context = "ContextModel",
        response = "ResponseModel",
    },
    root = BT.sequence({
        -- Load context
        BT.action("load_context", {fn = "oracle.load_context"}),

        -- Main loop
        BT.retry(10,
            BT.sequence({
                -- Process query
                BT.selector({
                    -- Try tools first
                    BT.guard("bb.needs_tools",
                        BT.parallel({policy = "require_all"}, {
                            BT.tool("search_notes", {
                                query = "${bb.tool_query}",
                                output = "search_results",
                            }),
                        })
                    ),
                    -- Otherwise generate response
                    BT.action("generate", {fn = "oracle.generate"}),
                }),

                -- Emit response
                BT.action("emit", {fn = "oracle.emit_response"}),
            }),
            {backoff_ms = 100}
        ),
    })
})
"""
        tree = loader.load_string(lua_code)

        assert tree.name == "oracle-agent"
        assert "query" in tree.blackboard_schema
        assert tree.root.type == "sequence"
        # First child is action
        assert tree.root.children[0].type == "action"
        # Second child is retry
        assert tree.root.children[1].type == "retry"

    def test_research_runner_like_tree(self, loader: TreeLoader) -> None:
        """Test loading a tree structure similar to research-runner."""
        lua_code = """
return BT.tree("research-runner", {
    description = "Research runner subtree",
    root = BT.for_each("research_queries", {
        item_key = "current_query",
        continue_on_failure = true,
        children = {
            BT.timeout(60000,
                BT.sequence({
                    BT.code_search({
                        operation = "search",
                        query = "${bb.current_query}",
                        limit = 10,
                        output = "code_results",
                    }),
                    BT.vault_search({
                        query = "${bb.current_query}",
                        limit = 5,
                        output = "vault_results",
                    }),
                    BT.action("merge_results", {fn = "research.merge"}),
                })
            )
        }
    })
})
"""
        tree = loader.load_string(lua_code)

        assert tree.name == "research-runner"
        assert tree.root.type == "for_each"
        assert tree.root.config["continue_on_failure"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
