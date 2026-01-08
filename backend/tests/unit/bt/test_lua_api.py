"""
Unit tests for BT.* Lua API.

Tests the BTApiBuilder class from lua/api.py:
- NodeDefinition dataclass
- TreeDefinition dataclass
- All BT.* functions (tree, sequence, selector, parallel, etc.)
- MCP integration nodes (tool, oracle, code_search, vault_search)
- Contract declaration

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.5.1-2.5.8 from tasks.md
"""

import pytest
from typing import Dict, Any

from backend.src.bt.lua.api import (
    NodeDefinition,
    TreeDefinition,
    BTApiBuilder,
    compute_source_hash,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def api_builder() -> BTApiBuilder:
    """Create a test API builder."""
    return BTApiBuilder(source_path="<test>")


@pytest.fixture
def bt_api(api_builder: BTApiBuilder) -> Dict[str, Any]:
    """Create the BT.* API dictionary."""
    return api_builder.build_api()


# =============================================================================
# NodeDefinition Tests
# =============================================================================


class TestNodeDefinition:
    """Tests for NodeDefinition dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a NodeDefinition with required fields."""
        node = NodeDefinition(type="action")

        assert node.type == "action"
        assert node.id is not None  # Auto-generated
        assert node.id.startswith("action_")
        assert node.name == node.id
        assert node.config == {}
        assert node.children == []
        assert node.source_line is None

    def test_with_all_fields(self) -> None:
        """Test creating a NodeDefinition with all fields."""
        child = NodeDefinition(type="action", id="child-1")
        node = NodeDefinition(
            type="sequence",
            id="seq-1",
            name="Main Sequence",
            config={"timeout": 5000},
            children=[child],
            source_line=42,
        )

        assert node.type == "sequence"
        assert node.id == "seq-1"
        assert node.name == "Main Sequence"
        assert node.config == {"timeout": 5000}
        assert len(node.children) == 1
        assert node.children[0].id == "child-1"
        assert node.source_line == 42

    def test_to_dict(self) -> None:
        """Test NodeDefinition serialization."""
        child = NodeDefinition(type="action", id="child-1", name="Child Action")
        node = NodeDefinition(
            type="sequence",
            id="seq-1",
            name="Main Sequence",
            children=[child],
        )

        data = node.to_dict()

        assert data["type"] == "sequence"
        assert data["id"] == "seq-1"
        assert data["name"] == "Main Sequence"
        assert len(data["children"]) == 1
        assert data["children"][0]["id"] == "child-1"

    def test_auto_id_generation(self) -> None:
        """Test that IDs are auto-generated when not provided."""
        node1 = NodeDefinition(type="action")
        node2 = NodeDefinition(type="action")

        # IDs should be unique
        assert node1.id != node2.id
        assert node1.id.startswith("action_")
        assert node2.id.startswith("action_")


# =============================================================================
# TreeDefinition Tests
# =============================================================================


class TestTreeDefinition:
    """Tests for TreeDefinition dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a TreeDefinition with required fields."""
        root = NodeDefinition(type="sequence", id="root")
        tree = TreeDefinition(name="test-tree", root=root)

        assert tree.name == "test-tree"
        assert tree.root.id == "root"
        assert tree.description == ""
        assert tree.blackboard_schema == {}
        assert tree.config == {}
        assert tree.source_path == ""
        assert tree.source_hash == ""

    def test_with_all_fields(self) -> None:
        """Test creating a TreeDefinition with all fields."""
        root = NodeDefinition(type="sequence", id="root")
        tree = TreeDefinition(
            name="oracle-agent",
            root=root,
            description="Main Oracle agent tree",
            blackboard_schema={
                "query": "QueryModel",
                "response": "ResponseModel",
            },
            config={"max_retries": 3},
            source_path="/trees/oracle.lua",
            source_hash="abc123",
        )

        assert tree.name == "oracle-agent"
        assert tree.description == "Main Oracle agent tree"
        assert tree.blackboard_schema["query"] == "QueryModel"
        assert tree.config["max_retries"] == 3
        assert tree.source_path == "/trees/oracle.lua"
        assert tree.source_hash == "abc123"

    def test_to_dict(self) -> None:
        """Test TreeDefinition serialization."""
        root = NodeDefinition(type="sequence", id="root")
        tree = TreeDefinition(
            name="test-tree",
            root=root,
            description="A test tree",
        )

        data = tree.to_dict()

        assert data["name"] == "test-tree"
        assert data["description"] == "A test tree"
        assert data["root"]["id"] == "root"


# =============================================================================
# BTApiBuilder Tests - Tree Creation
# =============================================================================


class TestBTApiTree:
    """Tests for BT.tree() function."""

    def test_tree_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a tree with BT.tree()."""
        root = NodeDefinition(type="sequence", id="root")
        tree = bt_api["tree"]("oracle-agent", {
            "root": root,
            "description": "Main agent tree",
        })

        assert isinstance(tree, TreeDefinition)
        assert tree.name == "oracle-agent"
        assert tree.root.id == "root"
        assert tree.description == "Main agent tree"

    def test_tree_with_blackboard_schema(self, bt_api: Dict[str, Any]) -> None:
        """Test tree creation with blackboard schema."""
        root = NodeDefinition(type="sequence", id="root")
        tree = bt_api["tree"]("test", {
            "root": root,
            "blackboard": {
                "query": "QueryModel",
                "response": "ResponseModel",
            },
        })

        assert tree.blackboard_schema["query"] == "QueryModel"
        assert tree.blackboard_schema["response"] == "ResponseModel"

    def test_tree_requires_name(self, bt_api: Dict[str, Any]) -> None:
        """Test that BT.tree() requires a name."""
        root = NodeDefinition(type="sequence", id="root")

        with pytest.raises(ValueError, match="requires a name"):
            bt_api["tree"]("", {"root": root})

    def test_tree_requires_root(self, bt_api: Dict[str, Any]) -> None:
        """Test that BT.tree() requires a root node."""
        with pytest.raises(ValueError, match="requires a 'root' node"):
            bt_api["tree"]("test", {"description": "No root"})

    def test_tree_root_must_be_node(self, bt_api: Dict[str, Any]) -> None:
        """Test that BT.tree() root must be NodeDefinition."""
        with pytest.raises(ValueError, match="must be a node definition"):
            bt_api["tree"]("test", {"root": "not a node"})


# =============================================================================
# BTApiBuilder Tests - Composite Nodes
# =============================================================================


class TestBTApiComposites:
    """Tests for composite node creation functions."""

    def test_sequence_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a sequence with BT.sequence()."""
        children = [
            NodeDefinition(type="action", id="step1"),
            NodeDefinition(type="action", id="step2"),
        ]
        node = bt_api["sequence"](children)

        assert isinstance(node, NodeDefinition)
        assert node.type == "sequence"
        assert len(node.children) == 2
        assert node.children[0].id == "step1"
        assert node.children[1].id == "step2"

    def test_sequence_with_config(self, bt_api: Dict[str, Any]) -> None:
        """Test sequence with optional config."""
        children = [NodeDefinition(type="action", id="step1")]
        node = bt_api["sequence"](children, {"id": "main-seq", "name": "Main Sequence"})

        assert node.id == "main-seq"
        assert node.name == "Main Sequence"

    def test_selector_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a selector with BT.selector()."""
        children = [
            NodeDefinition(type="condition", id="check"),
            NodeDefinition(type="action", id="fallback"),
        ]
        node = bt_api["selector"](children)

        assert node.type == "selector"
        assert len(node.children) == 2

    def test_parallel_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a parallel with BT.parallel()."""
        children = [
            NodeDefinition(type="action", id="search1"),
            NodeDefinition(type="action", id="search2"),
        ]
        node = bt_api["parallel"]({
            "policy": "require_all",
            "required_successes": 2,
        }, children)

        assert node.type == "parallel"
        assert node.config["policy"] == "require_all"
        assert node.config["required_successes"] == 2
        assert len(node.children) == 2

    def test_parallel_children_as_first_arg(self, bt_api: Dict[str, Any]) -> None:
        """Test parallel with children as first arg (convenience)."""
        children = [NodeDefinition(type="action", id="a1")]
        node = bt_api["parallel"](children)

        assert node.type == "parallel"
        assert len(node.children) == 1

    def test_for_each_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a for_each with BT.for_each()."""
        child = NodeDefinition(type="action", id="process")
        node = bt_api["for_each"]("results", {
            "item_key": "current_result",
            "children": [child],
            "continue_on_failure": True,
        })

        assert node.type == "for_each"
        assert node.config["collection_key"] == "results"
        assert node.config["item_key"] == "current_result"
        assert node.config["continue_on_failure"] is True
        assert len(node.children) == 1


# =============================================================================
# BTApiBuilder Tests - Leaf Nodes
# =============================================================================


class TestBTApiLeaves:
    """Tests for leaf node creation functions."""

    def test_action_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating an action with BT.action()."""
        node = bt_api["action"]("load_context", {
            "fn": "oracle.actions.load_context",
        })

        assert isinstance(node, NodeDefinition)
        assert node.type == "action"
        assert node.id == "load_context"
        assert node.config["fn"] == "oracle.actions.load_context"
        assert len(node.children) == 0

    def test_action_with_custom_id(self, bt_api: Dict[str, Any]) -> None:
        """Test action with custom ID."""
        node = bt_api["action"]("load", {"id": "custom-id", "fn": "test.fn"})

        assert node.id == "custom-id"

    def test_condition_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a condition with BT.condition()."""
        node = bt_api["condition"]("has_budget", {
            "condition": "bb.budget > 0",
        })

        assert node.type == "condition"
        assert node.id == "has_budget"
        assert node.config["condition"] == "bb.budget > 0"

    def test_llm_call_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating an LLM call with BT.llm_call()."""
        node = bt_api["llm_call"]({
            "model": "claude-3-opus",
            "prompt_key": "prompt",
            "response_key": "response",
            "stream_to": "partial",
        })

        assert node.type == "llm_call"
        assert node.config["model"] == "claude-3-opus"
        assert node.config["prompt_key"] == "prompt"
        assert node.config["response_key"] == "response"

    def test_subtree_ref_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a subtree ref with BT.subtree_ref()."""
        node = bt_api["subtree_ref"]("research-runner", {"lazy": True})

        assert node.type == "subtree_ref"
        assert node.config["tree_name"] == "research-runner"
        assert node.config["lazy"] is True

    def test_subtree_ref_requires_name(self, bt_api: Dict[str, Any]) -> None:
        """Test that BT.subtree_ref() requires a tree name."""
        with pytest.raises(ValueError, match="requires a tree name"):
            bt_api["subtree_ref"]("")

    def test_script_creation_inline(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a script with inline code."""
        node = bt_api["script"]("compute", {
            "lua": "local x = bb.get('input'); return {status = 'success'}",
        })

        assert node.type == "script"
        assert node.id == "compute"
        assert "lua" in node.config

    def test_script_creation_file(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a script with file reference."""
        node = bt_api["script"]("process", {
            "file": "scripts/process.lua",
        })

        assert node.type == "script"
        assert node.config["file"] == "scripts/process.lua"


# =============================================================================
# BTApiBuilder Tests - Decorator Nodes
# =============================================================================


class TestBTApiDecorators:
    """Tests for decorator node creation functions."""

    def test_timeout_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a timeout with BT.timeout()."""
        child = NodeDefinition(type="action", id="api_call")
        node = bt_api["timeout"](30000, child)

        assert node.type == "timeout"
        assert node.config["timeout_ms"] == 30000
        assert len(node.children) == 1
        assert node.children[0].id == "api_call"

    def test_retry_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a retry with BT.retry()."""
        child = NodeDefinition(type="action", id="flaky_op")
        node = bt_api["retry"](3, child, {"backoff_ms": 1000})

        assert node.type == "retry"
        assert node.config["max_retries"] == 3
        assert node.config["backoff_ms"] == 1000

    def test_guard_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a guard with BT.guard()."""
        child = NodeDefinition(type="action", id="expensive_op")
        node = bt_api["guard"]("bb.budget > 0", child)

        assert node.type == "guard"
        assert node.config["condition"] == "bb.budget > 0"

    def test_cooldown_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a cooldown with BT.cooldown()."""
        child = NodeDefinition(type="action", id="rate_limited")
        node = bt_api["cooldown"](5000, child)

        assert node.type == "cooldown"
        assert node.config["cooldown_ms"] == 5000

    def test_inverter_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating an inverter with BT.inverter()."""
        child = NodeDefinition(type="condition", id="is_empty")
        node = bt_api["inverter"](child)

        assert node.type == "inverter"
        assert len(node.children) == 1

    def test_always_succeed_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating an always_succeed with BT.always_succeed()."""
        child = NodeDefinition(type="action", id="optional")
        node = bt_api["always_succeed"](child)

        assert node.type == "always_succeed"

    def test_always_fail_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating an always_fail with BT.always_fail()."""
        child = NodeDefinition(type="action", id="force_retry")
        node = bt_api["always_fail"](child)

        assert node.type == "always_fail"

    def test_decorator_requires_node_child(self, bt_api: Dict[str, Any]) -> None:
        """Test that decorators require NodeDefinition as child."""
        with pytest.raises(ValueError, match="requires a node definition"):
            bt_api["timeout"](5000, "not a node")

        with pytest.raises(ValueError, match="requires a node definition"):
            bt_api["inverter"]({"type": "action"})


# =============================================================================
# BTApiBuilder Tests - Contract Declaration
# =============================================================================


class TestBTApiContract:
    """Tests for BT.contract() function."""

    def test_contract_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a contract with BT.contract()."""
        contract = bt_api["contract"]({
            "inputs": {"session_id": "SessionIdModel"},
            "optional_inputs": {"limit": "LimitModel"},
            "outputs": {"context": "ContextModel"},
            "description": "Load conversation context",
        })

        assert contract["inputs"]["session_id"] == "SessionIdModel"
        assert contract["optional_inputs"]["limit"] == "LimitModel"
        assert contract["outputs"]["context"] == "ContextModel"
        assert contract["description"] == "Load conversation context"

    def test_contract_defaults(self, bt_api: Dict[str, Any]) -> None:
        """Test contract with default values."""
        contract = bt_api["contract"]({})

        assert contract["inputs"] == {}
        assert contract["optional_inputs"] == {}
        assert contract["outputs"] == {}
        assert contract["description"] == ""


# =============================================================================
# BTApiBuilder Tests - MCP Integration
# =============================================================================


class TestBTApiMCPIntegration:
    """Tests for MCP integration nodes."""

    def test_tool_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a tool node with BT.tool()."""
        node = bt_api["tool"]("search_notes", {
            "query": "${bb.user_query}",
            "limit": 10,
            "output": "search_results",
        })

        assert node.type == "tool"
        assert node.config["tool_name"] == "search_notes"
        assert node.config["params"]["query"] == "${bb.user_query}"
        assert node.config["params"]["limit"] == 10
        assert node.config["output"] == "search_results"

    def test_tool_requires_name(self, bt_api: Dict[str, Any]) -> None:
        """Test that BT.tool() requires a tool name."""
        with pytest.raises(ValueError, match="requires a tool name"):
            bt_api["tool"]("")

    def test_oracle_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating an oracle node with BT.oracle()."""
        node = bt_api["oracle"]({
            "question": "${bb.user_question}",
            "sources": ["code", "vault"],
            "stream_to": "partial_response",
            "output": "oracle_answer",
        })

        assert node.type == "oracle"
        assert node.config["question"] == "${bb.user_question}"
        assert node.config["sources"] == ["code", "vault"]
        assert node.config["output"] == "oracle_answer"

    def test_code_search_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a code_search node with BT.code_search()."""
        node = bt_api["code_search"]({
            "operation": "search",
            "query": "${bb.search_query}",
            "limit": 20,
            "output": "code_results",
        })

        assert node.type == "code_search"
        assert node.config["operation"] == "search"
        assert node.config["query"] == "${bb.search_query}"
        assert node.config["output"] == "code_results"

    def test_vault_search_creation(self, bt_api: Dict[str, Any]) -> None:
        """Test creating a vault_search node with BT.vault_search()."""
        node = bt_api["vault_search"]({
            "query": "${bb.query}",
            "tags": ["project", "design"],
            "limit": 5,
            "output": "notes",
        })

        assert node.type == "vault_search"
        assert node.config["query"] == "${bb.query}"
        assert node.config["tags"] == ["project", "design"]
        assert node.config["output"] == "notes"


# =============================================================================
# Nested Tree Definition Tests
# =============================================================================


class TestNestedTreeDefinition:
    """Tests for creating nested tree structures."""

    def test_nested_sequence_in_selector(self, bt_api: Dict[str, Any]) -> None:
        """Test nesting sequence inside selector."""
        action1 = NodeDefinition(type="action", id="a1")
        action2 = NodeDefinition(type="action", id="a2")
        sequence = bt_api["sequence"]([action1, action2], {"id": "inner-seq"})
        selector = bt_api["selector"]([sequence], {"id": "outer-sel"})

        assert selector.type == "selector"
        assert selector.id == "outer-sel"
        assert len(selector.children) == 1
        assert selector.children[0].type == "sequence"
        assert len(selector.children[0].children) == 2

    def test_decorator_chain(self, bt_api: Dict[str, Any]) -> None:
        """Test chaining decorators."""
        action = NodeDefinition(type="action", id="api_call")
        with_retry = bt_api["retry"](3, action)
        with_timeout = bt_api["timeout"](30000, with_retry)

        assert with_timeout.type == "timeout"
        assert with_timeout.children[0].type == "retry"
        assert with_timeout.children[0].children[0].type == "action"

    def test_complex_tree_structure(self, bt_api: Dict[str, Any]) -> None:
        """Test building a complex tree structure."""
        # Build a tree like:
        # selector
        #   -> guard("has_context")
        #        -> sequence
        #             -> action("load_context")
        #             -> action("process")
        #   -> sequence
        #        -> action("initialize")
        #        -> action("process")

        load_ctx = bt_api["action"]("load_context", {"fn": "ctx.load"})
        process1 = bt_api["action"]("process", {"fn": "ctx.process"})
        guarded_seq = bt_api["sequence"]([load_ctx, process1])
        guarded = bt_api["guard"]("bb.has_context", guarded_seq)

        init = bt_api["action"]("initialize", {"fn": "ctx.init"})
        process2 = bt_api["action"]("process2", {"fn": "ctx.process"})
        fallback_seq = bt_api["sequence"]([init, process2])

        root = bt_api["selector"]([guarded, fallback_seq])

        tree = bt_api["tree"]("complex-tree", {
            "root": root,
            "description": "A complex tree structure",
        })

        assert tree.name == "complex-tree"
        assert tree.root.type == "selector"
        assert len(tree.root.children) == 2
        assert tree.root.children[0].type == "guard"
        assert tree.root.children[1].type == "sequence"


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_source_hash(self) -> None:
        """Test computing source hash."""
        content = "return BT.tree('test', {})"
        hash1 = compute_source_hash(content)

        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex length

        # Same content should give same hash
        hash2 = compute_source_hash(content)
        assert hash1 == hash2

        # Different content should give different hash
        hash3 = compute_source_hash(content + " ")
        assert hash1 != hash3


# =============================================================================
# API Completeness Test
# =============================================================================


class TestAPICompleteness:
    """Tests to verify all expected API functions are present."""

    def test_all_functions_present(self, bt_api: Dict[str, Any]) -> None:
        """Verify all expected BT.* functions are present."""
        expected_functions = [
            # Tree
            "tree",
            # Composites
            "sequence",
            "selector",
            "parallel",
            "for_each",
            # Leaves
            "action",
            "condition",
            "llm_call",
            "subtree_ref",
            "script",
            # Decorators
            "timeout",
            "retry",
            "guard",
            "cooldown",
            "inverter",
            "always_succeed",
            "always_fail",
            # Contract
            "contract",
            # MCP integration
            "tool",
            "oracle",
            "code_search",
            "vault_search",
        ]

        for func_name in expected_functions:
            assert func_name in bt_api, f"Missing BT.{func_name}()"
            assert callable(bt_api[func_name]), f"BT.{func_name} is not callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
