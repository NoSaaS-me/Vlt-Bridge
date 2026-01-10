"""
Unit tests for TreeBuilder.

Tests the TreeBuilder class from lua/builder.py:
- Building BehaviorTree from TreeDefinition
- Node type mapping to node classes
- Error handling (E2005, E4003)
- Blackboard schema registration
- Stub node creation for MCP leaves

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.7.6-2.7.10 from tasks.md
"""

import pytest
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from backend.src.bt.lua.definitions import (
    NodeDefinition,
    TreeDefinition,
)
from backend.src.bt.lua.builder import (
    TreeBuilder,
    TreeBuildError,
    DuplicateNodeIdError,
    NODE_BUILDERS,
)
from backend.src.bt.nodes import (
    Sequence,
    Selector,
    Parallel,
    Timeout,
    Retry,
    Action,
    Condition,
    SubtreeRef,
    Script,
    # Tool nodes
    Tool,
    Oracle,
    CodeSearch,
    VaultSearch,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def builder() -> TreeBuilder:
    """Create a test builder without registry."""
    return TreeBuilder(registry=None)


def make_tree(root: NodeDefinition, name: str = "test-tree") -> TreeDefinition:
    """Helper to create a tree definition."""
    return TreeDefinition(name=name, root=root)


# =============================================================================
# NODE_BUILDERS Tests
# =============================================================================


class TestNodeBuilders:
    """Tests for NODE_BUILDERS mapping."""

    def test_all_composites_mapped(self) -> None:
        """Test all composite types are in NODE_BUILDERS."""
        composite_types = ["sequence", "selector", "parallel", "for_each"]
        for node_type in composite_types:
            assert node_type in NODE_BUILDERS, f"Missing composite: {node_type}"

    def test_all_decorators_mapped(self) -> None:
        """Test all decorator types are in NODE_BUILDERS."""
        decorator_types = [
            "timeout", "retry", "guard", "cooldown",
            "inverter", "always_succeed", "always_fail",
        ]
        for node_type in decorator_types:
            assert node_type in NODE_BUILDERS, f"Missing decorator: {node_type}"

    def test_all_leaves_mapped(self) -> None:
        """Test all leaf types are in NODE_BUILDERS."""
        leaf_types = ["action", "condition", "subtree_ref", "script"]
        for node_type in leaf_types:
            assert node_type in NODE_BUILDERS, f"Missing leaf: {node_type}"


# =============================================================================
# Basic Build Tests
# =============================================================================


class TestBasicBuild:
    """Tests for basic tree building."""

    def test_build_simple_sequence(self, builder: TreeBuilder) -> None:
        """Test building a simple sequence tree."""
        def dummy_action(ctx):
            return ctx

        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(type="action", id="step-1", config={"fn": "test.action"}),
                    NodeDefinition(type="action", id="step-2", config={"fn": "test.action"}),
                ],
            )
        )

        # Patch function resolution
        with patch.object(builder, "_resolve_function", return_value=dummy_action):
            tree = builder.build(tree_def)

        assert tree is not None
        assert tree.id == "test-tree"
        assert tree.name == "test-tree"
        assert isinstance(tree.root, Sequence)
        assert len(tree.root.children) == 2

    def test_build_nested_tree(self, builder: TreeBuilder) -> None:
        """Test building a nested tree structure."""
        def dummy_action(ctx):
            return ctx

        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="selector",
                        id="fallback",
                        children=[
                            NodeDefinition(type="condition", id="check", config={"condition": "true"}),
                            NodeDefinition(type="action", id="backup", config={"fn": "test.action"}),
                        ],
                    ),
                    NodeDefinition(type="action", id="final", config={"fn": "test.action"}),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=dummy_action):
            tree = builder.build(tree_def)

        assert tree is not None
        assert isinstance(tree.root, Sequence)
        assert isinstance(tree.root.children[0], Selector)
        assert isinstance(tree.root.children[1], Action)


# =============================================================================
# Composite Node Build Tests
# =============================================================================


class TestCompositeBuild:
    """Tests for building composite nodes."""

    def test_build_selector(self, builder: TreeBuilder) -> None:
        """Test building a selector node."""
        def dummy_action(ctx):
            return ctx

        tree_def = make_tree(
            NodeDefinition(
                type="selector",
                id="root",
                children=[
                    NodeDefinition(type="action", id="a1", config={"fn": "test.a"}),
                    NodeDefinition(type="action", id="a2", config={"fn": "test.b"}),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=dummy_action):
            tree = builder.build(tree_def)

        assert isinstance(tree.root, Selector)
        assert len(tree.root.children) == 2

    def test_build_parallel(self, builder: TreeBuilder) -> None:
        """Test building a parallel node with policy."""
        def dummy_action(ctx):
            return ctx

        tree_def = make_tree(
            NodeDefinition(
                type="parallel",
                id="root",
                config={
                    "policy": "require_one",
                    "merge_strategy": "last_wins",
                },
                children=[
                    NodeDefinition(type="action", id="a1", config={"fn": "test.a"}),
                    NodeDefinition(type="action", id="a2", config={"fn": "test.b"}),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=dummy_action):
            tree = builder.build(tree_def)

        assert isinstance(tree.root, Parallel)
        assert len(tree.root.children) == 2


# =============================================================================
# Decorator Node Build Tests
# =============================================================================


class TestDecoratorBuild:
    """Tests for building decorator nodes."""

    def test_build_timeout(self, builder: TreeBuilder) -> None:
        """Test building a timeout decorator."""
        def dummy_action(ctx):
            return ctx

        tree_def = make_tree(
            NodeDefinition(
                type="timeout",
                id="timeout",
                config={"timeout_ms": 5000},
                children=[
                    NodeDefinition(type="action", id="action", config={"fn": "test.a"}),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=dummy_action):
            tree = builder.build(tree_def)

        assert isinstance(tree.root, Timeout)
        assert tree.root._timeout_ms == 5000
        assert isinstance(tree.root.child, Action)

    def test_build_retry(self, builder: TreeBuilder) -> None:
        """Test building a retry decorator."""
        def dummy_action(ctx):
            return ctx

        tree_def = make_tree(
            NodeDefinition(
                type="retry",
                id="retry",
                config={"max_retries": 3, "backoff_ms": 1000},
                children=[
                    NodeDefinition(type="action", id="action", config={"fn": "test.a"}),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=dummy_action):
            tree = builder.build(tree_def)

        assert isinstance(tree.root, Retry)
        assert tree.root._max_retries == 3
        assert tree.root._backoff_ms == 1000

    def test_build_decorator_chain(self, builder: TreeBuilder) -> None:
        """Test building chained decorators."""
        def dummy_action(ctx):
            return ctx

        tree_def = make_tree(
            NodeDefinition(
                type="timeout",
                id="timeout",
                config={"timeout_ms": 10000},
                children=[
                    NodeDefinition(
                        type="retry",
                        id="retry",
                        config={"max_retries": 2},
                        children=[
                            NodeDefinition(type="action", id="action", config={"fn": "test.a"}),
                        ],
                    ),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=dummy_action):
            tree = builder.build(tree_def)

        assert isinstance(tree.root, Timeout)
        assert isinstance(tree.root.child, Retry)
        assert isinstance(tree.root.child.child, Action)


# =============================================================================
# Leaf Node Build Tests
# =============================================================================


class TestLeafBuild:
    """Tests for building leaf nodes."""

    def test_build_action(self, builder: TreeBuilder) -> None:
        """Test building an action node."""
        def my_action(ctx):
            return ctx

        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="action",
                        id="my-action",
                        name="My Action",
                        config={"fn": "test.my_action"},
                    ),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=my_action):
            tree = builder.build(tree_def)

        action = tree.root.children[0]
        assert isinstance(action, Action)
        assert action._id == "my-action"

    def test_build_condition_with_fn(self, builder: TreeBuilder) -> None:
        """Test building a condition node with function."""
        def my_condition(ctx):
            return True

        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="condition",
                        id="check",
                        config={"fn": "test.my_condition"},
                    ),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=my_condition):
            tree = builder.build(tree_def)

        condition = tree.root.children[0]
        assert isinstance(condition, Condition)

    def test_build_condition_with_expression(self, builder: TreeBuilder) -> None:
        """Test building a condition node with expression."""
        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="condition",
                        id="check",
                        config={"expression": "bb.budget > 0"},
                    ),
                ],
            )
        )

        tree = builder.build(tree_def)

        condition = tree.root.children[0]
        assert isinstance(condition, Condition)

    def test_build_condition_with_fn_and_args(self, builder: TreeBuilder) -> None:
        """Test building a condition node with function and args (functools.partial binding)."""
        def my_condition(ctx, expected_type: str):
            """Condition function that takes an expected_type argument."""
            return expected_type == "test_value"

        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="condition",
                        id="check",
                        config={
                            "fn": "test.my_condition",
                            "args": {"expected_type": "test_value"},
                        },
                    ),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=my_condition):
            tree = builder.build(tree_def)

        condition = tree.root.children[0]
        assert isinstance(condition, Condition)
        # The condition function should be wrapped with functools.partial
        # We can verify by checking that the partial was applied
        assert hasattr(condition._condition_fn, "func")  # functools.partial objects have a 'func' attribute

    def test_build_action_with_fn_and_args(self, builder: TreeBuilder) -> None:
        """Test building an action node with function and args (functools.partial binding)."""
        def my_action(ctx, count: int):
            """Action function that takes a count argument."""
            return ctx

        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="action",
                        id="my-action",
                        name="My Action",
                        config={
                            "fn": "test.my_action",
                            "args": {"count": 5},
                        },
                    ),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=my_action):
            tree = builder.build(tree_def)

        action = tree.root.children[0]
        assert isinstance(action, Action)
        # The action function should be wrapped with functools.partial
        assert hasattr(action._fn, "func")  # functools.partial objects have a 'func' attribute

    def test_build_subtree_ref(self, builder: TreeBuilder) -> None:
        """Test building a subtree reference node."""
        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="subtree_ref",
                        id="ref",
                        config={"tree": "other-tree", "lazy": True},
                    ),
                ],
            )
        )

        tree = builder.build(tree_def)

        ref = tree.root.children[0]
        assert isinstance(ref, SubtreeRef)
        assert ref._tree_name == "other-tree"
        assert ref._lazy is True

    def test_build_script_inline(self, builder: TreeBuilder) -> None:
        """Test building a script node with inline code."""
        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="script",
                        id="script",
                        config={"code": "return {status='success'}"},
                    ),
                ],
            )
        )

        tree = builder.build(tree_def)

        script = tree.root.children[0]
        assert isinstance(script, Script)


# =============================================================================
# Stub Node Tests
# =============================================================================


class TestStubNodes:
    """Tests for stub node creation for unimplemented MCP leaves."""

    def test_tool_node_built(self, builder: TreeBuilder) -> None:
        """Test that tool nodes are built as Tool instances."""
        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="tool",
                        id="tool",
                        config={
                            "tool_name": "search_notes",
                            "output": "results",
                            "params": {"query": "test"},
                        },
                    ),
                ],
            )
        )

        tree = builder.build(tree_def)

        tool = tree.root.children[0]
        assert isinstance(tool, Tool)
        assert tool._tool_name == "search_notes"
        assert tool._output_key == "results"

    def test_oracle_node_built(self, builder: TreeBuilder) -> None:
        """Test that oracle nodes are built as Oracle instances."""
        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="oracle",
                        id="oracle",
                        config={"question": "test question"},
                    ),
                ],
            )
        )

        tree = builder.build(tree_def)

        oracle = tree.root.children[0]
        assert isinstance(oracle, Oracle)
        assert oracle._question_template == "test question"

    def test_code_search_node_built(self, builder: TreeBuilder) -> None:
        """Test that code_search nodes are built as CodeSearch instances."""
        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="code_search",
                        id="cs",
                        config={
                            "operation": "search",
                            "query": "my query",
                        },
                    ),
                ],
            )
        )

        tree = builder.build(tree_def)

        cs = tree.root.children[0]
        assert isinstance(cs, CodeSearch)
        assert cs._operation == "search"
        assert cs._query_template == "my query"

    def test_vault_search_node_built(self, builder: TreeBuilder) -> None:
        """Test that vault_search nodes are built as VaultSearch instances."""
        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="vault_search",
                        id="vs",
                        config={
                            "query": "vault query",
                            "tags": ["tag1"],
                        },
                    ),
                ],
            )
        )

        tree = builder.build(tree_def)

        vs = tree.root.children[0]
        assert isinstance(vs, VaultSearch)
        assert vs._query_template == "vault query"
        assert vs._tags == ["tag1"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestBuildErrors:
    """Tests for error handling during build."""

    def test_duplicate_node_id_error(self, builder: TreeBuilder) -> None:
        """Test that duplicate node IDs produce DuplicateNodeIdError."""
        def dummy_action(ctx):
            return ctx

        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(type="action", id="dup", config={"fn": "test.a"}),
                    NodeDefinition(type="action", id="dup", config={"fn": "test.b"}),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=dummy_action):
            with pytest.raises(DuplicateNodeIdError) as exc_info:
                builder.build(tree_def)

            assert "dup" in str(exc_info.value)
            assert exc_info.value.code == "E2005"

    def test_unknown_node_type_error(self, builder: TreeBuilder) -> None:
        """Test that unknown node types produce TreeBuildError."""
        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="totally_unknown",
                        id="unknown",
                        children=[],
                    ),
                ],
            )
        )

        with pytest.raises(TreeBuildError) as exc_info:
            builder.build(tree_def)

        assert exc_info.value.code == "E4002"
        assert "totally_unknown" in str(exc_info.value)

    def test_missing_fn_error(self, builder: TreeBuilder) -> None:
        """Test that missing fn path produces TreeBuildError."""
        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="action",
                        id="action",
                        config={},  # No fn!
                    ),
                ],
            )
        )

        with pytest.raises(TreeBuildError) as exc_info:
            builder.build(tree_def)

        assert exc_info.value.code == "E4003"
        assert "fn" in str(exc_info.value).lower()

    def test_fn_resolution_error(self, builder: TreeBuilder) -> None:
        """Test that unresolvable fn produces TreeBuildError."""
        tree_def = make_tree(
            NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(
                        type="action",
                        id="action",
                        config={"fn": "nonexistent.module.function"},
                    ),
                ],
            )
        )

        with pytest.raises(TreeBuildError) as exc_info:
            builder.build(tree_def)

        assert exc_info.value.code == "E4003"

    def test_decorator_wrong_child_count(self, builder: TreeBuilder) -> None:
        """Test that decorator with wrong children produces TreeBuildError."""
        def dummy_action(ctx):
            return ctx

        tree_def = make_tree(
            NodeDefinition(
                type="timeout",
                id="timeout",
                config={"timeout_ms": 5000},
                children=[
                    NodeDefinition(type="action", id="a1", config={"fn": "test.a"}),
                    NodeDefinition(type="action", id="a2", config={"fn": "test.b"}),
                ],
            )
        )

        with patch.object(builder, "_resolve_function", return_value=dummy_action):
            with pytest.raises(TreeBuildError) as exc_info:
                builder.build(tree_def)

            assert exc_info.value.code == "E2006"


# =============================================================================
# Blackboard Schema Tests
# =============================================================================


class TestBlackboardSchema:
    """Tests for blackboard schema registration during build."""

    def test_schema_registered(self, builder: TreeBuilder) -> None:
        """Test that blackboard schema is registered."""
        tree_def = TreeDefinition(
            name="test-tree",
            root=NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(type="action", id="action", config={"fn": "os.getcwd"}),
                ],
            ),
            blackboard_schema={
                "query": "pydantic.BaseModel",
            },
        )

        # This should not raise (schema may not resolve but tree builds)
        tree = builder.build(tree_def)
        assert tree is not None


# =============================================================================
# Tree Metadata Tests
# =============================================================================


class TestTreeMetadata:
    """Tests for tree metadata from definition."""

    def test_tree_id_from_name(self, builder: TreeBuilder) -> None:
        """Test tree ID comes from definition name."""
        tree_def = TreeDefinition(
            name="my-tree",
            root=NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(type="action", id="action", config={"fn": "os.getcwd"}),
                ],
            ),
        )

        tree = builder.build(tree_def)

        assert tree.id == "my-tree"
        assert tree.name == "my-tree"

    def test_tree_description(self, builder: TreeBuilder) -> None:
        """Test tree description is preserved."""
        tree_def = TreeDefinition(
            name="test-tree",
            root=NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(type="action", id="action", config={"fn": "os.getcwd"}),
                ],
            ),
            description="This is a test tree.",
        )

        tree = builder.build(tree_def)

        assert tree.description == "This is a test tree."

    def test_tree_source_path(self, builder: TreeBuilder) -> None:
        """Test tree source path is preserved."""
        tree_def = TreeDefinition(
            name="test-tree",
            root=NodeDefinition(
                type="sequence",
                id="root",
                children=[
                    NodeDefinition(type="action", id="action", config={"fn": "os.getcwd"}),
                ],
            ),
            source_path="/path/to/tree.lua",
        )

        tree = builder.build(tree_def)

        assert tree.source_path == "/path/to/tree.lua"


# =============================================================================
# Exports
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
