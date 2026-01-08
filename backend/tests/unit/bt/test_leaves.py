"""
Unit tests for BT Leaf Nodes.

Tests the leaf node implementations from nodes/leaves.py:
- LeafNode base class
- Action node (function execution)
- Condition node (boolean evaluation)
- SubtreeRef node (subtree references)
- Script node (Lua execution)

Part of the BT Universal Runtime (spec 019).
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict

from pydantic import BaseModel

from backend.src.bt.nodes.leaves import (
    LeafNode,
    Action,
    Condition,
    SubtreeRef,
    Script,
    FunctionNotFoundError,
    TreeNotFoundError,
    CircularReferenceError,
    ActionFunction,
)
from backend.src.bt.state.base import RunStatus, NodeType
from backend.src.bt.state.blackboard import TypedBlackboard
from backend.src.bt.state.contracts import NodeContract
from backend.src.bt.core.context import TickContext


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleModel(BaseModel):
    """Simple test model."""
    value: int


class StringModel(BaseModel):
    """String test model."""
    text: str


@pytest.fixture
def blackboard() -> TypedBlackboard:
    """Create a test blackboard."""
    bb = TypedBlackboard(scope_name="test")
    bb.register("input", SimpleModel)
    bb.register("output", SimpleModel)
    bb.register("text", StringModel)
    return bb


@pytest.fixture
def tick_context(blackboard: TypedBlackboard) -> TickContext:
    """Create a test tick context."""
    return TickContext(blackboard=blackboard)


# =============================================================================
# LeafNode Base Class Tests
# =============================================================================


class TestLeafNode:
    """Tests for LeafNode base class."""

    def test_leaf_node_type(self) -> None:
        """Leaf nodes should have LEAF node type."""
        # Create a concrete LeafNode subclass for testing
        class TestLeaf(LeafNode):
            def _tick(self, ctx: TickContext) -> RunStatus:
                return RunStatus.SUCCESS

        node = TestLeaf(id="test-leaf")
        assert node.node_type == NodeType.LEAF

    def test_leaf_node_no_children(self) -> None:
        """Leaf nodes should always have empty children list."""
        class TestLeaf(LeafNode):
            def _tick(self, ctx: TickContext) -> RunStatus:
                return RunStatus.SUCCESS

        node = TestLeaf(id="test-leaf")
        assert node.children == []

    def test_leaf_node_cannot_add_children(self) -> None:
        """Leaf nodes should raise error when trying to add children."""
        class TestLeaf(LeafNode):
            def _tick(self, ctx: TickContext) -> RunStatus:
                return RunStatus.SUCCESS

        node = TestLeaf(id="test-leaf")
        child = TestLeaf(id="child")

        with pytest.raises(ValueError, match="Cannot add child to leaf node"):
            node._add_child(child)


# =============================================================================
# Action Node Tests
# =============================================================================


class TestAction:
    """Tests for Action node."""

    def test_action_with_callable(self, tick_context: TickContext) -> None:
        """Action should execute callable and return its status."""
        def my_action(ctx: TickContext) -> RunStatus:
            return RunStatus.SUCCESS

        action = Action(id="test-action", fn=my_action)
        result = action.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert action.tick_count == 1

    def test_action_returns_failure(self, tick_context: TickContext) -> None:
        """Action should return FAILURE from function."""
        def failing_action(ctx: TickContext) -> RunStatus:
            return RunStatus.FAILURE

        action = Action(id="failing-action", fn=failing_action)
        result = action.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_action_returns_running(self, tick_context: TickContext) -> None:
        """Action may return RUNNING for async operations."""
        def async_action(ctx: TickContext) -> RunStatus:
            return RunStatus.RUNNING

        action = Action(id="async-action", fn=async_action)
        result = action.tick(tick_context)

        assert result == RunStatus.RUNNING
        assert action.running_since is not None

    def test_action_exception_returns_failure(self, tick_context: TickContext) -> None:
        """Action should return FAILURE if function raises exception."""
        def bad_action(ctx: TickContext) -> RunStatus:
            raise ValueError("Something went wrong")

        action = Action(id="bad-action", fn=bad_action)
        result = action.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_action_invalid_return_type(self, tick_context: TickContext) -> None:
        """Action should return FAILURE if function returns non-RunStatus."""
        def bad_return(ctx: TickContext) -> Any:
            return "not a RunStatus"

        action = Action(id="bad-return", fn=bad_return)
        result = action.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_action_with_blackboard_access(
        self,
        blackboard: TypedBlackboard,
        tick_context: TickContext,
    ) -> None:
        """Action should have access to blackboard."""
        # Set up input
        blackboard.set("input", SimpleModel(value=42))

        def read_write_action(ctx: TickContext) -> RunStatus:
            input_val = ctx.blackboard.get("input", SimpleModel)
            if input_val:
                ctx.blackboard.set("output", SimpleModel(value=input_val.value * 2))
                return RunStatus.SUCCESS
            return RunStatus.FAILURE

        action = Action(id="rw-action", fn=read_write_action)
        result = action.tick(tick_context)

        assert result == RunStatus.SUCCESS
        output = blackboard.get("output", SimpleModel)
        assert output is not None
        assert output.value == 84

    def test_action_invalid_fn_type(self) -> None:
        """Action should reject non-callable, non-string fn."""
        with pytest.raises(TypeError, match="must be a string path or callable"):
            Action(id="bad", fn=123)  # type: ignore

    def test_action_function_path_resolution_failure(self) -> None:
        """Action should raise E4003 for invalid function path."""
        with pytest.raises(FunctionNotFoundError) as exc_info:
            Action(id="bad-path", fn="nonexistent.module.function")

        assert exc_info.value.error_code == "E4003"

    def test_action_debug_info(self) -> None:
        """Action debug_info should include function path."""
        action = Action(id="test", fn=lambda ctx: RunStatus.SUCCESS)
        info = action.debug_info()

        assert info["id"] == "test"
        assert info["node_type"] == "leaf"
        assert "fn_path" in info
        assert "has_function_contract" in info


# =============================================================================
# Condition Node Tests
# =============================================================================


class TestCondition:
    """Tests for Condition node."""

    def test_condition_true_returns_success(self, tick_context: TickContext) -> None:
        """Condition returning True should yield SUCCESS."""
        condition = Condition(
            id="true-cond",
            condition=lambda ctx: True,
        )
        result = condition.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_condition_false_returns_failure(self, tick_context: TickContext) -> None:
        """Condition returning False should yield FAILURE."""
        condition = Condition(
            id="false-cond",
            condition=lambda ctx: False,
        )
        result = condition.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_condition_never_returns_running(self, tick_context: TickContext) -> None:
        """Condition should never return RUNNING."""
        # Even if the lambda somehow returns something truthy
        condition = Condition(
            id="cond",
            condition=lambda ctx: True,
        )
        result = condition.tick(tick_context)

        # Should be SUCCESS, not RUNNING
        assert result in (RunStatus.SUCCESS, RunStatus.FAILURE)
        assert condition.running_since is None

    def test_condition_with_blackboard(
        self,
        blackboard: TypedBlackboard,
        tick_context: TickContext,
    ) -> None:
        """Condition should be able to check blackboard values."""
        blackboard.set("input", SimpleModel(value=50))

        condition = Condition(
            id="check-value",
            condition=lambda ctx: (
                ctx.blackboard.get("input", SimpleModel).value > 40
            ),
        )
        result = condition.tick(tick_context)

        assert result == RunStatus.SUCCESS

    def test_condition_exception_returns_failure(
        self,
        tick_context: TickContext,
    ) -> None:
        """Condition should return FAILURE if expression raises."""
        condition = Condition(
            id="bad-cond",
            condition=lambda ctx: 1 / 0,  # Division by zero
        )
        result = condition.tick(tick_context)

        assert result == RunStatus.FAILURE

    def test_condition_invalid_type(self) -> None:
        """Condition should reject invalid condition type."""
        with pytest.raises(TypeError, match="must be a string or callable"):
            Condition(id="bad", condition=123)  # type: ignore

    def test_condition_debug_info(self) -> None:
        """Condition debug_info should include condition type."""
        cond = Condition(id="test", condition=lambda ctx: True)
        info = cond.debug_info()

        assert info["id"] == "test"
        assert info["condition_type"] == "python"

    def test_condition_lua_expression_debug_info(self) -> None:
        """Condition with Lua expression should show in debug_info."""
        cond = Condition(id="test", condition="bb.value > 0")
        info = cond.debug_info()

        assert info["condition_type"] == "lua"
        assert info["condition_expr"] == "bb.value > 0"


# =============================================================================
# SubtreeRef Node Tests
# =============================================================================


class TestSubtreeRef:
    """Tests for SubtreeRef node."""

    def test_subtree_ref_resolution(self) -> None:
        """SubtreeRef should resolve tree from registry."""
        # Create a mock tree
        mock_root = MagicMock()
        mock_root.status = RunStatus.FRESH

        mock_tree = MagicMock()
        mock_tree.id = "target-tree"
        mock_tree.root = mock_root

        registry = {"target-tree": mock_tree}

        ref = SubtreeRef(id="ref", tree_name="target-tree")
        ref.resolve(registry)

        assert ref._resolved_tree == mock_tree

    def test_subtree_ref_not_found(self) -> None:
        """SubtreeRef should raise E3001 for missing tree."""
        registry: Dict[str, Any] = {}

        ref = SubtreeRef(id="ref", tree_name="nonexistent")

        with pytest.raises(TreeNotFoundError) as exc_info:
            ref.resolve(registry)

        assert exc_info.value.error_code == "E3001"
        assert exc_info.value.tree_name == "nonexistent"

    def test_subtree_ref_circular_detection(self) -> None:
        """SubtreeRef should detect circular references (E3002)."""
        # Create trees that reference each other
        mock_tree_a = MagicMock()
        mock_tree_a.id = "tree-a"

        registry = {"tree-a": mock_tree_a}

        ref = SubtreeRef(id="ref", tree_name="tree-a")

        # Simulate being called from tree-a (circular)
        with pytest.raises(CircularReferenceError) as exc_info:
            ref.resolve(registry, reference_chain=["tree-a"])

        assert exc_info.value.error_code == "E3002"
        assert "tree-a" in exc_info.value.cycle_path

    def test_subtree_ref_tick_executes_subtree(
        self,
        tick_context: TickContext,
    ) -> None:
        """SubtreeRef tick should execute the subtree root."""
        # Create mock tree
        mock_root = MagicMock()
        mock_root.tick.return_value = RunStatus.SUCCESS
        mock_root.children = []

        mock_tree = MagicMock()
        mock_tree.id = "target-tree"
        mock_tree.root = mock_root

        registry = {"target-tree": mock_tree}

        ref = SubtreeRef(id="ref", tree_name="target-tree")
        ref.resolve(registry)

        result = ref.tick(tick_context)

        assert result == RunStatus.SUCCESS
        mock_root.tick.assert_called_once()

    def test_subtree_ref_lazy_resolution(self) -> None:
        """SubtreeRef with lazy=True should not resolve at init."""
        ref = SubtreeRef(id="ref", tree_name="target", lazy=True)

        # Should not be resolved yet
        assert ref._resolved_tree is None

    def test_subtree_ref_creates_child_scope(
        self,
        blackboard: TypedBlackboard,
    ) -> None:
        """SubtreeRef should create child blackboard scope."""
        # Create mock tree
        mock_root = MagicMock()
        mock_root.children = []

        def check_scope(ctx):
            # Should have a child scope
            assert ctx.blackboard._scope_name.startswith("subtree:")
            return RunStatus.SUCCESS

        mock_root.tick = check_scope

        mock_tree = MagicMock()
        mock_tree.id = "target-tree"
        mock_tree.root = mock_root

        registry = {"target-tree": mock_tree}

        ref = SubtreeRef(id="ref", tree_name="target-tree")
        ref.resolve(registry)

        tick_context = TickContext(blackboard=blackboard)
        ref.tick(tick_context)

    def test_subtree_ref_debug_info(self) -> None:
        """SubtreeRef debug_info should include tree reference info."""
        ref = SubtreeRef(id="ref", tree_name="target", lazy=True)
        info = ref.debug_info()

        assert info["tree_name"] == "target"
        assert info["lazy"] is True
        assert info["resolved"] is False


# =============================================================================
# Script Node Tests
# =============================================================================


class TestScript:
    """Tests for Script node (with mocked Lua sandbox)."""

    def test_script_requires_code_or_file(self) -> None:
        """Script should require either code or file."""
        with pytest.raises(ValueError, match="Either 'code' or 'file' must be provided"):
            Script(id="empty")

    def test_script_rejects_both_code_and_file(self) -> None:
        """Script should reject both code and file specified."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            Script(id="both", code="return 1", file="test.lua")

    def test_script_with_inline_code(self) -> None:
        """Script should accept inline code."""
        script = Script(id="inline", code="return {status = 'success'}")

        assert script._code is not None
        assert script._file is None

    def test_script_with_file(self) -> None:
        """Script should accept file path."""
        script = Script(id="file", file="test.lua")

        assert script._file == "test.lua"
        assert script._code is None

    def test_script_timeout_clamped(self) -> None:
        """Script timeout should be clamped to max."""
        script = Script(id="test", code="return 1", timeout_ms=100000)

        assert script._timeout_ms == Script.MAX_TIMEOUT_MS

    def test_script_debug_info(self) -> None:
        """Script debug_info should include source info."""
        script = Script(id="test", code="return 1", timeout_ms=1000)
        info = script.debug_info()

        assert info["source_type"] == "inline"
        assert info["timeout_ms"] == 1000
        assert info["code_length"] == len("return 1")

    def test_script_file_debug_info(self) -> None:
        """Script with file should show file in debug_info."""
        script = Script(id="test", file="scripts/test.lua")
        info = script.debug_info()

        assert info["source_type"] == "file"
        assert info["file"] == "scripts/test.lua"

    @patch("backend.src.bt.lua.sandbox.LuaSandbox")
    def test_script_execution_success(
        self,
        mock_sandbox_class: MagicMock,
        tick_context: TickContext,
    ) -> None:
        """Script should execute Lua and return SUCCESS."""
        from backend.src.bt.lua.sandbox import LuaExecutionResult

        # Mock sandbox behavior
        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = LuaExecutionResult.ok(
            {"status": "success"}
        )
        mock_sandbox_class.return_value = mock_sandbox

        script = Script(id="test", code="return {status = 'success'}")
        result = script.tick(tick_context)

        assert result == RunStatus.SUCCESS

    @patch("backend.src.bt.lua.sandbox.LuaSandbox")
    def test_script_execution_failure(
        self,
        mock_sandbox_class: MagicMock,
        tick_context: TickContext,
    ) -> None:
        """Script should return FAILURE when Lua returns failure status."""
        from backend.src.bt.lua.sandbox import LuaExecutionResult

        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = LuaExecutionResult.ok(
            {"status": "failure", "reason": "Intentional failure"}
        )
        mock_sandbox_class.return_value = mock_sandbox

        script = Script(id="test", code="return {status = 'failure'}")
        result = script.tick(tick_context)

        assert result == RunStatus.FAILURE

    @patch("backend.src.bt.lua.sandbox.LuaSandbox")
    def test_script_execution_running(
        self,
        mock_sandbox_class: MagicMock,
        tick_context: TickContext,
    ) -> None:
        """Script should return RUNNING when Lua returns running status."""
        from backend.src.bt.lua.sandbox import LuaExecutionResult

        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = LuaExecutionResult.ok(
            {"status": "running"}
        )
        mock_sandbox_class.return_value = mock_sandbox

        script = Script(id="test", code="return {status = 'running'}")
        result = script.tick(tick_context)

        assert result == RunStatus.RUNNING

    @patch("backend.src.bt.lua.sandbox.LuaSandbox")
    def test_script_syntax_error(
        self,
        mock_sandbox_class: MagicMock,
        tick_context: TickContext,
    ) -> None:
        """Script should return FAILURE on Lua syntax error (E5001)."""
        from backend.src.bt.lua.sandbox import LuaExecutionResult

        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = LuaExecutionResult.syntax_error(
            message="unexpected symbol near 'return'",
            line_number=1,
        )
        mock_sandbox_class.return_value = mock_sandbox

        script = Script(id="test", code="return {{{")
        result = script.tick(tick_context)

        assert result == RunStatus.FAILURE

    @patch("backend.src.bt.lua.sandbox.LuaSandbox")
    def test_script_runtime_error(
        self,
        mock_sandbox_class: MagicMock,
        tick_context: TickContext,
    ) -> None:
        """Script should return FAILURE on Lua runtime error (E5002)."""
        from backend.src.bt.lua.sandbox import LuaExecutionResult

        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = LuaExecutionResult.runtime_error(
            message="attempt to call a nil value",
            line_number=2,
        )
        mock_sandbox_class.return_value = mock_sandbox

        script = Script(id="test", code="undefined_function()")
        result = script.tick(tick_context)

        assert result == RunStatus.FAILURE

    @patch("backend.src.bt.lua.sandbox.LuaSandbox")
    def test_script_timeout(
        self,
        mock_sandbox_class: MagicMock,
        tick_context: TickContext,
    ) -> None:
        """Script should return FAILURE on timeout (E5003)."""
        from backend.src.bt.lua.sandbox import LuaExecutionResult

        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = LuaExecutionResult.timeout_error(5.0)
        mock_sandbox_class.return_value = mock_sandbox

        script = Script(id="test", code="while true do end")
        result = script.tick(tick_context)

        assert result == RunStatus.FAILURE


# =============================================================================
# Integration Tests
# =============================================================================


class TestLeafNodeIntegration:
    """Integration tests for leaf nodes working together."""

    def test_action_sets_value_for_condition(
        self,
        blackboard: TypedBlackboard,
    ) -> None:
        """Action can set a value that a Condition later checks."""
        def set_value(ctx: TickContext) -> RunStatus:
            ctx.blackboard.set("input", SimpleModel(value=100))
            return RunStatus.SUCCESS

        action = Action(id="setter", fn=set_value)
        condition = Condition(
            id="checker",
            condition=lambda ctx: ctx.blackboard.get("input", SimpleModel).value > 50,
        )

        tick_context = TickContext(blackboard=blackboard)

        # First action sets value
        assert action.tick(tick_context) == RunStatus.SUCCESS

        # Then condition checks it
        assert condition.tick(tick_context) == RunStatus.SUCCESS

    def test_reset_clears_running_state(
        self,
        tick_context: TickContext,
    ) -> None:
        """Reset should clear running state on all leaf types."""
        # Test Action
        action = Action(
            id="action",
            fn=lambda ctx: RunStatus.RUNNING,
        )
        action.tick(tick_context)
        assert action.running_since is not None
        action.reset()
        assert action.running_since is None
        assert action.status == RunStatus.FRESH

        # Test Condition
        condition = Condition(id="cond", condition=lambda ctx: True)
        condition.tick(tick_context)
        condition.reset()
        assert condition.status == RunStatus.FRESH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
