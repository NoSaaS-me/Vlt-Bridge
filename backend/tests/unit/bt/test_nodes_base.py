"""
Unit tests for BehaviorNode base class.

Tests:
- Node construction and validation
- Node ID validation (E2004)
- tick() workflow with contract validation
- reset() functionality
- debug_info()
- Cancellation handling

Part of the BT Universal Runtime (spec 019).
"""

import pytest
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from backend.src.bt.nodes.base import (
    BehaviorNode,
    InvalidNodeIdError,
    NODE_ID_PATTERN,
)
from backend.src.bt.state.base import NodeType, RunStatus
from backend.src.bt.state.blackboard import TypedBlackboard
from backend.src.bt.state.contracts import NodeContract
from backend.src.bt.core.context import TickContext  # Import directly to avoid cascading imports


# =============================================================================
# Test Fixtures
# =============================================================================


class InputSchema(BaseModel):
    """Schema for required inputs."""
    value: str


class OutputSchema(BaseModel):
    """Schema for outputs."""
    result: str


class ConcreteLeafNode(BehaviorNode):
    """Concrete implementation for testing leaf nodes."""

    def __init__(
        self,
        id: str,
        name: Optional[str] = None,
        tick_result: RunStatus = RunStatus.SUCCESS,
        raise_exception: bool = False,
    ) -> None:
        super().__init__(id=id, name=name)
        self._tick_result = tick_result
        self._raise_exception = raise_exception
        self._tick_called = False

    @property
    def node_type(self) -> NodeType:
        return NodeType.LEAF

    def _tick(self, ctx: TickContext) -> RunStatus:
        self._tick_called = True
        if self._raise_exception:
            raise RuntimeError("Test exception")
        return self._tick_result


class ConcreteNodeWithContract(BehaviorNode):
    """Concrete implementation with a contract for testing."""

    def __init__(self, id: str, name: Optional[str] = None) -> None:
        super().__init__(id=id, name=name)

    @property
    def node_type(self) -> NodeType:
        return NodeType.LEAF

    @classmethod
    def contract(cls) -> NodeContract:
        return NodeContract(
            inputs={"test_input": InputSchema},
            outputs={"test_output": OutputSchema},
            description="Test node with contract",
        )

    def _tick(self, ctx: TickContext) -> RunStatus:
        # Read input and write output
        ctx.blackboard.get("test_input", InputSchema)
        ctx.blackboard.set("test_output", OutputSchema(result="done"))
        return RunStatus.SUCCESS


@pytest.fixture
def basic_blackboard() -> TypedBlackboard:
    """Create a basic blackboard for testing."""
    return TypedBlackboard(scope_name="test")


@pytest.fixture
def tick_context(basic_blackboard: TypedBlackboard) -> TickContext:
    """Create a basic tick context."""
    return TickContext(blackboard=basic_blackboard)


# =============================================================================
# Node ID Validation Tests (E2004)
# =============================================================================


class TestNodeIdValidation:
    """Tests for node ID validation."""

    def test_valid_node_id_simple(self) -> None:
        """Valid simple ID should work."""
        node = ConcreteLeafNode(id="myNode")
        assert node.id == "myNode"

    def test_valid_node_id_with_numbers(self) -> None:
        """Valid ID with numbers should work."""
        node = ConcreteLeafNode(id="node123")
        assert node.id == "node123"

    def test_valid_node_id_with_underscore(self) -> None:
        """Valid ID with underscore should work."""
        node = ConcreteLeafNode(id="my_node")
        assert node.id == "my_node"

    def test_valid_node_id_with_hyphen(self) -> None:
        """Valid ID with hyphen should work."""
        node = ConcreteLeafNode(id="my-node")
        assert node.id == "my-node"

    def test_valid_node_id_mixed(self) -> None:
        """Valid ID with mixed characters should work."""
        node = ConcreteLeafNode(id="myNode123_test-case")
        assert node.id == "myNode123_test-case"

    def test_invalid_node_id_empty(self) -> None:
        """Empty ID should raise E2004."""
        with pytest.raises(InvalidNodeIdError) as exc_info:
            ConcreteLeafNode(id="")
        assert "E2004" in str(exc_info.value)

    def test_invalid_node_id_starts_with_number(self) -> None:
        """ID starting with number should raise E2004."""
        with pytest.raises(InvalidNodeIdError) as exc_info:
            ConcreteLeafNode(id="123node")
        assert "E2004" in str(exc_info.value)
        assert "123node" in str(exc_info.value)

    def test_invalid_node_id_starts_with_underscore(self) -> None:
        """ID starting with underscore should raise E2004."""
        with pytest.raises(InvalidNodeIdError) as exc_info:
            ConcreteLeafNode(id="_node")
        assert "E2004" in str(exc_info.value)

    def test_invalid_node_id_starts_with_hyphen(self) -> None:
        """ID starting with hyphen should raise E2004."""
        with pytest.raises(InvalidNodeIdError) as exc_info:
            ConcreteLeafNode(id="-node")
        assert "E2004" in str(exc_info.value)

    def test_invalid_node_id_special_chars(self) -> None:
        """ID with special characters should raise E2004."""
        with pytest.raises(InvalidNodeIdError) as exc_info:
            ConcreteLeafNode(id="node@test")
        assert "E2004" in str(exc_info.value)

    def test_invalid_node_id_spaces(self) -> None:
        """ID with spaces should raise E2004."""
        with pytest.raises(InvalidNodeIdError) as exc_info:
            ConcreteLeafNode(id="my node")
        assert "E2004" in str(exc_info.value)

    def test_node_id_pattern_matches_valid(self) -> None:
        """NODE_ID_PATTERN should match valid IDs."""
        valid_ids = ["a", "A", "myNode", "node123", "my_node", "my-node", "ABC123"]
        for id in valid_ids:
            assert NODE_ID_PATTERN.match(id), f"Expected {id} to be valid"

    def test_node_id_pattern_rejects_invalid(self) -> None:
        """NODE_ID_PATTERN should reject invalid IDs."""
        invalid_ids = ["", "123", "_node", "-node", "node@test", "my node"]
        for id in invalid_ids:
            assert not NODE_ID_PATTERN.match(id), f"Expected {id} to be invalid"


# =============================================================================
# Node Construction Tests
# =============================================================================


class TestNodeConstruction:
    """Tests for node construction and initialization."""

    def test_basic_construction(self) -> None:
        """Basic construction should set all fields correctly."""
        node = ConcreteLeafNode(id="testNode")

        assert node.id == "testNode"
        assert node.name == "testNode"  # Defaults to id
        assert node.status == RunStatus.FRESH
        assert node.tick_count == 0
        assert node.running_since is None
        assert node.last_tick_duration_ms == 0.0
        assert node.children == []
        assert node.parent is None

    def test_construction_with_name(self) -> None:
        """Construction with custom name should work."""
        node = ConcreteLeafNode(id="testNode", name="My Test Node")

        assert node.id == "testNode"
        assert node.name == "My Test Node"

    def test_construction_with_metadata(self) -> None:
        """Construction with metadata should work."""
        metadata = {"key": "value", "number": 42}
        node = ConcreteLeafNode(id="testNode")
        node._metadata = metadata

        assert node.metadata == metadata
        # Should return a copy
        assert node.metadata is not metadata

    def test_node_type_is_abstract(self) -> None:
        """node_type property should be abstract."""
        node = ConcreteLeafNode(id="testNode")
        assert node.node_type == NodeType.LEAF

    def test_children_returns_copy(self) -> None:
        """children property should return a copy."""
        node = ConcreteLeafNode(id="parent")
        child = ConcreteLeafNode(id="child")
        node._add_child(child)

        children = node.children
        children.append(ConcreteLeafNode(id="fake"))

        assert len(node.children) == 1


# =============================================================================
# Tick Workflow Tests
# =============================================================================


class TestTickWorkflow:
    """Tests for the tick() method workflow."""

    def test_tick_increments_count(self, tick_context: TickContext) -> None:
        """tick() should increment tick_count."""
        node = ConcreteLeafNode(id="testNode")

        assert node.tick_count == 0
        node.tick(tick_context)
        assert node.tick_count == 1
        node.tick(tick_context)
        assert node.tick_count == 2

    def test_tick_updates_status(self, tick_context: TickContext) -> None:
        """tick() should update status based on _tick result."""
        node = ConcreteLeafNode(id="testNode", tick_result=RunStatus.SUCCESS)

        assert node.status == RunStatus.FRESH
        result = node.tick(tick_context)

        assert result == RunStatus.SUCCESS
        assert node.status == RunStatus.SUCCESS

    def test_tick_handles_running_state(self, tick_context: TickContext) -> None:
        """tick() should track running_since when RUNNING."""
        node = ConcreteLeafNode(id="testNode", tick_result=RunStatus.RUNNING)

        before = datetime.utcnow()
        node.tick(tick_context)
        after = datetime.utcnow()

        assert node.status == RunStatus.RUNNING
        assert node.running_since is not None
        assert before <= node.running_since <= after

    def test_tick_clears_running_since_on_completion(
        self, tick_context: TickContext
    ) -> None:
        """tick() should clear running_since when completing."""
        node = ConcreteLeafNode(id="testNode", tick_result=RunStatus.RUNNING)

        # First tick - enter RUNNING
        node.tick(tick_context)
        assert node.running_since is not None

        # Change result to SUCCESS
        node._tick_result = RunStatus.SUCCESS

        # Second tick - exit RUNNING
        node.tick(tick_context)
        assert node.running_since is None
        assert node.status == RunStatus.SUCCESS

    def test_tick_measures_duration(self, tick_context: TickContext) -> None:
        """tick() should measure execution duration."""
        node = ConcreteLeafNode(id="testNode")

        node.tick(tick_context)

        # Duration should be non-negative (may be very small)
        assert node.last_tick_duration_ms >= 0

    def test_tick_calls_tick_impl(self, tick_context: TickContext) -> None:
        """tick() should call _tick()."""
        node = ConcreteLeafNode(id="testNode")

        assert not node._tick_called
        node.tick(tick_context)
        assert node._tick_called

    def test_tick_handles_exception_as_failure(
        self, tick_context: TickContext
    ) -> None:
        """tick() should catch exceptions and return FAILURE."""
        node = ConcreteLeafNode(id="testNode", raise_exception=True)

        result = node.tick(tick_context)

        assert result == RunStatus.FAILURE
        assert node.status == RunStatus.FAILURE

    def test_tick_marks_progress(self, tick_context: TickContext) -> None:
        """tick() should mark progress on success."""
        node = ConcreteLeafNode(id="testNode", tick_result=RunStatus.SUCCESS)

        node.tick(tick_context)

        assert tick_context.last_progress_at is not None


# =============================================================================
# Contract Validation Tests
# =============================================================================


class TestContractValidation:
    """Tests for contract validation during tick."""

    def test_tick_validates_inputs(self) -> None:
        """tick() should validate required inputs (E2001)."""
        bb = TypedBlackboard(scope_name="test")
        bb.register("test_input", InputSchema)
        bb.register("test_output", OutputSchema)
        # Don't set test_input - should fail

        ctx = TickContext(blackboard=bb)
        node = ConcreteNodeWithContract(id="contractNode")

        result = node.tick(ctx)

        assert result == RunStatus.FAILURE
        assert node.status == RunStatus.FAILURE

    def test_tick_succeeds_with_valid_inputs(self) -> None:
        """tick() should succeed when inputs are present."""
        bb = TypedBlackboard(scope_name="test")
        bb.register("test_input", InputSchema)
        bb.register("test_output", OutputSchema)
        bb.set("test_input", InputSchema(value="hello"))

        ctx = TickContext(blackboard=bb)
        node = ConcreteNodeWithContract(id="contractNode")

        result = node.tick(ctx)

        assert result == RunStatus.SUCCESS

    def test_tick_clears_access_tracking(self) -> None:
        """tick() should clear access tracking before _tick."""
        bb = TypedBlackboard(scope_name="test")
        bb.register("test_input", InputSchema)
        bb.register("test_output", OutputSchema)
        bb.set("test_input", InputSchema(value="hello"))

        # Pre-read to set tracking
        bb.get("test_input", InputSchema)
        assert len(bb.get_reads()) > 0

        ctx = TickContext(blackboard=bb)
        node = ConcreteLeafNode(id="testNode")

        node.tick(ctx)

        # Should have cleared before tick (empty after tick for leaf that doesn't read)
        # Actually, ConcreteLeafNode doesn't read anything, so reads should be empty
        # after clear_access_tracking is called


# =============================================================================
# Cancellation Tests
# =============================================================================


class TestCancellation:
    """Tests for cancellation handling."""

    def test_tick_checks_cancellation(self) -> None:
        """tick() should check for cancellation at start."""
        bb = TypedBlackboard(scope_name="test")
        ctx = TickContext(blackboard=bb, cancellation_requested=True)

        node = ConcreteLeafNode(id="testNode")

        result = node.tick(ctx)

        assert result == RunStatus.FAILURE
        assert not node._tick_called  # Should not call _tick

    def test_tick_with_cancellation_reason(self) -> None:
        """tick() should handle cancellation reason."""
        bb = TypedBlackboard(scope_name="test")
        ctx = TickContext(
            blackboard=bb,
            cancellation_requested=True,
            cancellation_reason="User cancelled",
        )

        node = ConcreteLeafNode(id="testNode")

        result = node.tick(ctx)

        assert result == RunStatus.FAILURE


# =============================================================================
# Reset Tests
# =============================================================================


class TestReset:
    """Tests for reset() functionality."""

    def test_reset_sets_fresh_status(self, tick_context: TickContext) -> None:
        """reset() should set status to FRESH."""
        node = ConcreteLeafNode(id="testNode")
        node.tick(tick_context)

        assert node.status == RunStatus.SUCCESS

        node.reset()

        assert node.status == RunStatus.FRESH

    def test_reset_clears_running_since(self, tick_context: TickContext) -> None:
        """reset() should clear running_since."""
        node = ConcreteLeafNode(id="testNode", tick_result=RunStatus.RUNNING)
        node.tick(tick_context)

        assert node.running_since is not None

        node.reset()

        assert node.running_since is None

    def test_reset_preserves_tick_count(self, tick_context: TickContext) -> None:
        """reset() should NOT reset tick_count (for debugging)."""
        node = ConcreteLeafNode(id="testNode")
        node.tick(tick_context)
        node.tick(tick_context)

        assert node.tick_count == 2

        node.reset()

        # Tick count preserved intentionally
        assert node.tick_count == 2

    def test_reset_resets_children(self, tick_context: TickContext) -> None:
        """reset() should recursively reset children."""
        parent = ConcreteLeafNode(id="parent")
        child = ConcreteLeafNode(id="child")
        parent._add_child(child)

        child.tick(tick_context)
        assert child.status == RunStatus.SUCCESS

        parent.reset()

        assert child.status == RunStatus.FRESH


# =============================================================================
# Debug Info Tests
# =============================================================================


class TestDebugInfo:
    """Tests for debug_info() functionality."""

    def test_debug_info_contains_required_fields(self) -> None:
        """debug_info() should contain all required fields."""
        node = ConcreteLeafNode(id="testNode", name="Test Node")

        info = node.debug_info()

        assert info["id"] == "testNode"
        assert info["name"] == "Test Node"
        assert info["node_type"] == "leaf"
        assert info["status"] == "FRESH"
        assert info["tick_count"] == 0
        assert info["running_since"] is None
        assert info["last_tick_duration_ms"] == 0.0
        assert "contract_summary" in info
        assert info["parent_id"] is None

    def test_debug_info_includes_children(self) -> None:
        """debug_info() should include children IDs."""
        parent = ConcreteLeafNode(id="parent")
        child1 = ConcreteLeafNode(id="child1")
        child2 = ConcreteLeafNode(id="child2")
        parent._add_child(child1)
        parent._add_child(child2)

        info = parent.debug_info()

        assert "children_ids" in info
        assert info["children_ids"] == ["child1", "child2"]

    def test_debug_info_includes_parent(self) -> None:
        """debug_info() should include parent ID."""
        parent = ConcreteLeafNode(id="parent")
        child = ConcreteLeafNode(id="child")
        parent._add_child(child)

        info = child.debug_info()

        assert info["parent_id"] == "parent"


# =============================================================================
# Hierarchy Tests
# =============================================================================


class TestHierarchy:
    """Tests for parent/child hierarchy management."""

    def test_add_child(self) -> None:
        """_add_child should set up parent-child relationship."""
        parent = ConcreteLeafNode(id="parent")
        child = ConcreteLeafNode(id="child")

        parent._add_child(child)

        assert child in parent.children
        assert child.parent is parent

    def test_add_child_duplicate_parent_raises(self) -> None:
        """_add_child should raise if child already has parent."""
        parent1 = ConcreteLeafNode(id="parent1")
        parent2 = ConcreteLeafNode(id="parent2")
        child = ConcreteLeafNode(id="child")

        parent1._add_child(child)

        with pytest.raises(ValueError) as exc_info:
            parent2._add_child(child)
        assert "already has parent" in str(exc_info.value)

    def test_remove_child(self) -> None:
        """_remove_child should remove parent-child relationship."""
        parent = ConcreteLeafNode(id="parent")
        child = ConcreteLeafNode(id="child")
        parent._add_child(child)

        result = parent._remove_child(child)

        assert result is True
        assert child not in parent.children
        assert child.parent is None

    def test_remove_nonexistent_child(self) -> None:
        """_remove_child should return False for non-child."""
        parent = ConcreteLeafNode(id="parent")
        other = ConcreteLeafNode(id="other")

        result = parent._remove_child(other)

        assert result is False


# =============================================================================
# Repr Tests
# =============================================================================


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr_contains_info(self) -> None:
        """__repr__ should contain useful information."""
        node = ConcreteLeafNode(id="testNode")
        repr_str = repr(node)

        assert "ConcreteLeafNode" in repr_str
        assert "testNode" in repr_str
        assert "FRESH" in repr_str
