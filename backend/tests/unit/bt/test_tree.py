"""
Unit tests for BehaviorTree class.

Tests tasks 1.3.1-1.3.6 from tasks.md:
- BehaviorTree creation
- Tick execution
- Status transitions
- Reset and cancel

Part of the BT Universal Runtime (spec 019).
"""

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from backend.src.bt.state import RunStatus, TypedBlackboard
from backend.src.bt.core import BehaviorTree, TreeStatus, TickContext


# =============================================================================
# Mock Node Implementation
# =============================================================================


class MockNode:
    """Mock behavior node for testing."""

    def __init__(
        self,
        node_id: str = "mock-node",
        status: RunStatus = RunStatus.FRESH,
        tick_result: Optional[RunStatus] = None,
        children: Optional[List["MockNode"]] = None,
    ):
        self._id = node_id
        self._status = status
        self._tick_result = tick_result or RunStatus.SUCCESS
        self.children = children or []
        self.tick_count = 0
        self.reset_count = 0
        self.running_since: Optional[datetime] = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> RunStatus:
        return self._status

    def tick(self, ctx: TickContext) -> RunStatus:
        self.tick_count += 1
        self._status = self._tick_result
        if self._tick_result == RunStatus.RUNNING:
            self.running_since = datetime.now(timezone.utc)
        return self._tick_result

    def reset(self) -> None:
        self.reset_count += 1
        self._status = RunStatus.FRESH
        self.running_since = None


# =============================================================================
# Test BehaviorTree Creation
# =============================================================================


class TestBehaviorTreeCreation:
    """Tests for BehaviorTree instantiation."""

    def test_create_basic_tree(self) -> None:
        """Test creating a basic tree with required fields."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(
            id="test-tree",
            name="Test Tree",
            root=root,
        )

        assert tree.id == "test-tree"
        assert tree.name == "Test Tree"
        assert tree.root is root
        assert tree.status == TreeStatus.IDLE
        assert tree.tick_count == 0

    def test_create_tree_with_all_fields(self) -> None:
        """Test creating a tree with all optional fields."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(
            id="test-tree",
            name="Test Tree",
            root=root,
            description="A test tree",
            source_path="/path/to/tree.lua",
            source_hash="abc123",
            max_tick_duration_ms=30000,
            tick_budget=500,
        )

        assert tree.description == "A test tree"
        assert tree.source_path == "/path/to/tree.lua"
        assert tree.source_hash == "abc123"
        assert tree.max_tick_duration_ms == 30000
        assert tree.tick_budget == 500

    def test_tree_creates_blackboard(self) -> None:
        """Test that tree creates its own blackboard."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(
            id="test-tree",
            name="Test Tree",
            root=root,
        )

        assert tree.blackboard is not None
        assert isinstance(tree.blackboard, TypedBlackboard)

    def test_tree_with_empty_id_raises(self) -> None:
        """Test that empty ID raises ValueError."""
        root = MockNode(node_id="root")
        with pytest.raises(ValueError, match="id cannot be empty"):
            BehaviorTree(id="", name="Test Tree", root=root)

    def test_tree_with_empty_name_raises(self) -> None:
        """Test that empty name raises ValueError."""
        root = MockNode(node_id="root")
        with pytest.raises(ValueError, match="name cannot be empty"):
            BehaviorTree(id="test", name="", root=root)

    def test_tree_with_none_root_raises(self) -> None:
        """Test that None root raises ValueError."""
        with pytest.raises(ValueError, match="root cannot be None"):
            BehaviorTree(id="test", name="Test Tree", root=None)


# =============================================================================
# Test Tick Execution
# =============================================================================


class TestBehaviorTreeTick:
    """Tests for BehaviorTree tick execution."""

    def test_tick_increments_count(self) -> None:
        """Test that each tick increments tick_count."""
        root = MockNode(node_id="root", tick_result=RunStatus.RUNNING)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        tree.tick(ctx)
        assert tree.tick_count == 1

        tree.tick(ctx)
        assert tree.tick_count == 2

    def test_tick_calls_root(self) -> None:
        """Test that tick calls root node's tick method."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        tree.tick(ctx)
        assert root.tick_count == 1

    def test_tick_returns_root_status(self) -> None:
        """Test that tick returns root node's status."""
        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        status = tree.tick(ctx)
        assert status == RunStatus.SUCCESS

    def test_tick_pushes_and_pops_path(self) -> None:
        """Test that tick pushes tree id to path and pops it."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        tree.tick(ctx)
        # Path should be empty after tick completes
        assert len(ctx.parent_path) == 0


# =============================================================================
# Test Status Transitions
# =============================================================================


class TestBehaviorTreeStatusTransitions:
    """Tests for TreeStatus state machine."""

    def test_idle_to_running_on_first_tick(self) -> None:
        """Test IDLE -> RUNNING transition on first tick."""
        root = MockNode(node_id="root", tick_result=RunStatus.RUNNING)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        assert tree.status == TreeStatus.IDLE
        tree.tick(ctx)
        assert tree.status == TreeStatus.RUNNING

    def test_running_to_completed_on_success(self) -> None:
        """Test RUNNING -> COMPLETED on SUCCESS."""
        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        tree.tick(ctx)
        assert tree.status == TreeStatus.COMPLETED

    def test_running_to_failed_on_failure(self) -> None:
        """Test RUNNING -> FAILED on FAILURE."""
        root = MockNode(node_id="root", tick_result=RunStatus.FAILURE)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        tree.tick(ctx)
        assert tree.status == TreeStatus.FAILED

    def test_yielded_on_budget_exceeded(self) -> None:
        """Test YIELDED status when budget exceeded."""
        root = MockNode(node_id="root", tick_result=RunStatus.RUNNING)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard, tick_budget=0)  # Immediate exceed

        # Budget exceeded before tick runs
        tree.tick(ctx)
        assert tree.status == TreeStatus.YIELDED

    def test_status_change_callback(self) -> None:
        """Test status change callback is called."""
        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        callback_calls = []

        def on_change(t: BehaviorTree, old: TreeStatus, new: TreeStatus) -> None:
            callback_calls.append((old, new))

        tree.on_status_change(on_change)
        tree.tick(ctx)

        # Should have IDLE -> RUNNING (on first tick) but we go directly to COMPLETED
        # Actually: IDLE -> COMPLETED since SUCCESS is immediate
        assert len(callback_calls) >= 1

    def test_cancellation_sets_failed(self) -> None:
        """Test cancellation request results in FAILED."""
        root = MockNode(node_id="root", tick_result=RunStatus.RUNNING)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)
        ctx.request_cancellation("test")

        tree.tick(ctx)
        assert tree.status == TreeStatus.FAILED


# =============================================================================
# Test Reset
# =============================================================================


class TestBehaviorTreeReset:
    """Tests for BehaviorTree reset functionality."""

    def test_reset_sets_idle(self) -> None:
        """Test reset sets status to IDLE."""
        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        tree.tick(ctx)
        assert tree.status == TreeStatus.COMPLETED

        tree.reset()
        assert tree.status == TreeStatus.IDLE

    def test_reset_clears_tick_count(self) -> None:
        """Test reset clears tick_count."""
        root = MockNode(node_id="root", tick_result=RunStatus.RUNNING)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        tree.tick(ctx)
        tree.tick(ctx)
        assert tree.tick_count == 2

        tree.reset()
        assert tree.tick_count == 0

    def test_reset_calls_root_reset(self) -> None:
        """Test reset calls root node's reset method."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        tree.tick(ctx)
        tree.reset()
        assert root.reset_count == 1

    def test_reset_clears_blackboard_data(self) -> None:
        """Test reset clears blackboard data."""
        from pydantic import BaseModel

        class TestData(BaseModel):
            value: int

        root = MockNode(node_id="root")
        tree = BehaviorTree(id="test", name="Test", root=root)

        # Add some data
        tree.blackboard.register("test", TestData)
        tree.blackboard.set("test", {"value": 42})
        assert tree.blackboard.has("test")

        tree.reset()
        assert not tree.blackboard.has("test")


# =============================================================================
# Test Cancel
# =============================================================================


class TestBehaviorTreeCancel:
    """Tests for BehaviorTree cancel functionality."""

    def test_cancel_sets_failed(self) -> None:
        """Test cancel sets status to FAILED."""
        root = MockNode(node_id="root", tick_result=RunStatus.RUNNING)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        tree.tick(ctx)
        assert tree.status == TreeStatus.RUNNING

        tree.cancel("test reason")
        assert tree.status == TreeStatus.FAILED

    def test_cancel_with_default_reason(self) -> None:
        """Test cancel works with default reason."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(id="test", name="Test", root=root)

        tree.cancel()  # Default reason
        assert tree.status == TreeStatus.FAILED


# =============================================================================
# Test Node Access
# =============================================================================


class TestBehaviorTreeNodeAccess:
    """Tests for node access methods."""

    def test_get_running_nodes(self) -> None:
        """Test get_running_nodes returns RUNNING nodes."""
        child1 = MockNode(node_id="child1", tick_result=RunStatus.RUNNING)
        child2 = MockNode(node_id="child2", tick_result=RunStatus.SUCCESS)
        child1.running_since = datetime.now(timezone.utc)
        child1._status = RunStatus.RUNNING

        root = MockNode(node_id="root", children=[child1, child2])
        tree = BehaviorTree(id="test", name="Test", root=root)

        running = tree.get_running_nodes()
        assert len(running) == 1
        assert running[0].id == "child1"

    def test_get_node_by_id(self) -> None:
        """Test get_node_by_id finds node."""
        child = MockNode(node_id="child")
        root = MockNode(node_id="root", children=[child])
        tree = BehaviorTree(id="test", name="Test", root=root)

        found = tree.get_node_by_id("child")
        assert found is child

        not_found = tree.get_node_by_id("nonexistent")
        assert not_found is None

    def test_get_node_path(self) -> None:
        """Test get_node_path returns path to node."""
        grandchild = MockNode(node_id="grandchild")
        child = MockNode(node_id="child", children=[grandchild])
        root = MockNode(node_id="root", children=[child])
        tree = BehaviorTree(id="test", name="Test", root=root)

        path = tree.get_node_path("grandchild")
        assert path == ["root", "child", "grandchild"]

    def test_node_count(self) -> None:
        """Test node_count property."""
        grandchild = MockNode(node_id="grandchild")
        child = MockNode(node_id="child", children=[grandchild])
        root = MockNode(node_id="root", children=[child])
        tree = BehaviorTree(id="test", name="Test", root=root)

        assert tree.node_count == 3

    def test_max_depth(self) -> None:
        """Test max_depth property."""
        grandchild = MockNode(node_id="grandchild")
        child = MockNode(node_id="child", children=[grandchild])
        root = MockNode(node_id="root", children=[child])
        tree = BehaviorTree(id="test", name="Test", root=root)

        assert tree.max_depth == 3


# =============================================================================
# Test Debug Info
# =============================================================================


class TestBehaviorTreeDebugInfo:
    """Tests for debug_info method."""

    def test_debug_info_contains_required_fields(self) -> None:
        """Test debug_info returns required fields."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(
            id="test-tree",
            name="Test Tree",
            root=root,
            description="Description",
        )

        info = tree.debug_info()

        assert info["id"] == "test-tree"
        assert info["name"] == "Test Tree"
        assert info["description"] == "Description"
        assert info["status"] == "idle"
        assert info["tick_count"] == 0
        assert info["node_count"] == 1
        assert info["max_depth"] == 1
        assert info["root_id"] == "root"
        assert "loaded_at" in info


# =============================================================================
# Test Reload Pending
# =============================================================================


class TestBehaviorTreeReloadPending:
    """Tests for reload pending functionality."""

    def test_queue_reload(self) -> None:
        """Test queue_reload sets flag."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(id="test", name="Test", root=root)

        assert not tree.reload_pending
        tree.queue_reload()
        assert tree.reload_pending

    def test_clear_reload_pending(self) -> None:
        """Test clear_reload_pending clears flag."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(id="test", name="Test", root=root)

        tree.queue_reload()
        assert tree.reload_pending

        tree.clear_reload_pending()
        assert not tree.reload_pending
