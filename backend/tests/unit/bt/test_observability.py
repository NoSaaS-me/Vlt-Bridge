"""
Unit tests for BT Observability (Phase 7).

Tests for:
- Tree visualization (ASCII, JSON, DOT)
- Tick history tracking
- Breakpoint management
- Debug model serialization

Part of the BT Universal Runtime (spec 019).
"""

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from backend.src.bt.state import RunStatus, NodeType
from backend.src.bt.debug.visualizer import (
    TreeVisualizer,
    ascii_tree,
    json_export,
    dot_export,
)
from backend.src.bt.debug.history import (
    TickEntry,
    NodeStats,
    TickHistoryTracker,
    HistoryRegistry,
    get_history_registry,
)
from backend.src.bt.debug.breakpoints import (
    Breakpoint,
    BreakpointHit,
    BreakpointManager,
    BreakpointRegistry,
    get_breakpoint_registry,
)


# =============================================================================
# Mock Tree and Node Classes
# =============================================================================


class MockNode:
    """Mock behavior node for testing."""

    def __init__(
        self,
        node_id: str = "mock-node",
        name: str = "",
        node_type: NodeType = NodeType.LEAF,
        status: RunStatus = RunStatus.FRESH,
        children: Optional[List["MockNode"]] = None,
        tick_count: int = 0,
        last_tick_duration_ms: float = 0.0,
    ):
        self._id = node_id
        self._name = name or node_id
        self._node_type = node_type
        self._status = status
        self._children = children or []
        self._tick_count = tick_count
        self._last_tick_duration_ms = last_tick_duration_ms
        self.running_since = None
        self.metadata = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def node_type(self) -> NodeType:
        return self._node_type

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def children(self) -> List["MockNode"]:
        return self._children

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def last_tick_duration_ms(self) -> float:
        return self._last_tick_duration_ms


class MockTree:
    """Mock behavior tree for testing."""

    def __init__(
        self,
        tree_id: str = "test-tree",
        name: str = "Test Tree",
        root: Optional[MockNode] = None,
    ):
        self._id = tree_id
        self._name = name
        self._root = root or MockNode("root")
        self._tick_count = 0
        self._description = "A test tree"
        self._source_path = "/path/to/tree.lua"
        self._source_hash = "abc123"
        self._loaded_at = datetime.now(timezone.utc)
        self._last_tick_at = None
        self._max_tick_duration_ms = 30000
        self._tick_budget = 100
        self._reload_pending = False

        # Mock status
        from backend.src.bt.core.tree import TreeStatus
        self._status = TreeStatus.IDLE

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def root(self) -> MockNode:
        return self._root

    @property
    def status(self):
        return self._status

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def description(self) -> str:
        return self._description

    @property
    def source_path(self) -> str:
        return self._source_path

    @property
    def source_hash(self) -> str:
        return self._source_hash

    @property
    def loaded_at(self) -> datetime:
        return self._loaded_at

    @property
    def last_tick_at(self) -> Optional[datetime]:
        return self._last_tick_at

    @property
    def max_tick_duration_ms(self) -> int:
        return self._max_tick_duration_ms

    @property
    def tick_budget(self) -> int:
        return self._tick_budget

    @property
    def reload_pending(self) -> bool:
        return self._reload_pending

    @property
    def node_count(self) -> int:
        return self._count_nodes(self._root)

    @property
    def max_depth(self) -> int:
        return self._calc_depth(self._root)

    def _count_nodes(self, node: MockNode) -> int:
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _calc_depth(self, node: MockNode) -> int:
        if not node.children:
            return 1
        return 1 + max(self._calc_depth(c) for c in node.children)

    def get_running_nodes(self) -> List[MockNode]:
        return []

    def get_node_path(self, node_id: str) -> List[str]:
        return [node_id]


# =============================================================================
# Test Tree Visualizer
# =============================================================================


class TestTreeVisualizer:
    """Tests for TreeVisualizer class."""

    def test_ascii_tree_simple(self) -> None:
        """Test ASCII visualization of simple tree."""
        root = MockNode(
            "root",
            node_type=NodeType.COMPOSITE,
            children=[
                MockNode("child1", status=RunStatus.SUCCESS),
                MockNode("child2", status=RunStatus.FAILURE),
            ],
        )
        tree = MockTree(root=root)

        result = ascii_tree(tree)

        assert "Test Tree" in result
        assert "root" in result
        assert "child1" in result
        assert "child2" in result
        assert "[SUCCESS]" in result
        assert "[FAILURE]" in result

    def test_ascii_tree_with_active_path(self) -> None:
        """Test ASCII visualization with active path highlighting."""
        root = MockNode(
            "root",
            node_type=NodeType.COMPOSITE,
            children=[MockNode("child1")],
        )
        tree = MockTree(root=root)

        result = ascii_tree(tree, active_path=["root", "child1"])

        assert "root" in result
        assert "child1" in result
        # Active nodes should have marker
        assert "*" in result

    def test_ascii_tree_with_timing(self) -> None:
        """Test ASCII visualization with timing info."""
        root = MockNode("root", last_tick_duration_ms=5.5)
        tree = MockTree(root=root)

        result = ascii_tree(tree, show_timing=True)

        assert "5.5ms" in result

    def test_json_export_structure(self) -> None:
        """Test JSON export returns correct structure."""
        root = MockNode(
            "root",
            node_type=NodeType.COMPOSITE,
            children=[MockNode("child1")],
        )
        tree = MockTree(root=root)

        result = json_export(tree)

        assert result["id"] == "test-tree"
        assert result["name"] == "Test Tree"
        assert result["root"]["id"] == "root"
        assert len(result["root"]["children"]) == 1
        assert result["root"]["children"][0]["id"] == "child1"

    def test_json_export_with_active_path(self) -> None:
        """Test JSON export marks active nodes."""
        root = MockNode(
            "root",
            node_type=NodeType.COMPOSITE,
            children=[MockNode("child1")],
        )
        tree = MockTree(root=root)

        result = json_export(tree, active_path=["root"])

        assert result["root"]["is_active"] is True
        assert result["root"]["children"][0]["is_active"] is False

    def test_dot_export_basic(self) -> None:
        """Test DOT export produces valid format."""
        root = MockNode("root", node_type=NodeType.COMPOSITE)
        tree = MockTree(root=root)

        result = dot_export(tree)

        assert 'digraph "Test Tree"' in result
        assert "root" in result
        assert "rankdir=TB" in result

    def test_dot_export_with_children(self) -> None:
        """Test DOT export includes edges for children."""
        root = MockNode(
            "root",
            node_type=NodeType.COMPOSITE,
            children=[MockNode("child1"), MockNode("child2")],
        )
        tree = MockTree(root=root)

        result = dot_export(tree)

        # Check for edge definitions
        assert "root -> child1" in result
        assert "root -> child2" in result

    def test_visualizer_class(self) -> None:
        """Test TreeVisualizer convenience class."""
        visualizer = TreeVisualizer(show_status=True, show_timing=True)

        root = MockNode("root")
        tree = MockTree(root=root)

        ascii_result = visualizer.to_ascii(tree)
        json_result = visualizer.to_json(tree)
        dot_result = visualizer.to_dot(tree)

        assert "root" in ascii_result
        assert json_result["root"]["id"] == "root"
        assert "root" in dot_result


# =============================================================================
# Test Tick History
# =============================================================================


class TestTickHistoryTracker:
    """Tests for TickHistoryTracker class."""

    def test_record_tick(self) -> None:
        """Test recording a tick entry."""
        tracker = TickHistoryTracker("test-tree")

        entry = tracker.record_tick(
            node_id="test-node",
            node_path=["root", "test-node"],
            status=RunStatus.SUCCESS,
            duration_ms=5.0,
        )

        assert entry.tick_number == 1
        assert entry.node_id == "test-node"
        assert entry.status == "SUCCESS"
        assert entry.duration_ms == 5.0
        assert tracker.entry_count == 1

    def test_get_entries_pagination(self) -> None:
        """Test getting entries with pagination."""
        tracker = TickHistoryTracker("test-tree")

        # Record 10 entries
        for i in range(10):
            tracker.record_tick(
                node_id=f"node-{i}",
                node_path=[f"node-{i}"],
                status=RunStatus.SUCCESS,
                duration_ms=1.0,
            )

        # Get with pagination
        entries = tracker.get_entries(limit=5, offset=3)

        assert len(entries) == 5
        # Entries are newest first, so offset=3 skips 3 newest
        assert entries[0].tick_number == 7  # 10 - 3 = 7

    def test_get_entries_filter_by_node(self) -> None:
        """Test filtering entries by node ID."""
        tracker = TickHistoryTracker("test-tree")

        tracker.record_tick("node-a", ["node-a"], RunStatus.SUCCESS, 1.0)
        tracker.record_tick("node-b", ["node-b"], RunStatus.SUCCESS, 1.0)
        tracker.record_tick("node-a", ["node-a"], RunStatus.SUCCESS, 1.0)

        entries = tracker.get_entries(node_id="node-a")

        assert len(entries) == 2
        assert all(e.node_id == "node-a" for e in entries)

    def test_get_entries_filter_by_status(self) -> None:
        """Test filtering entries by status."""
        tracker = TickHistoryTracker("test-tree")

        tracker.record_tick("node-1", ["node-1"], RunStatus.SUCCESS, 1.0)
        tracker.record_tick("node-2", ["node-2"], RunStatus.FAILURE, 1.0)
        tracker.record_tick("node-3", ["node-3"], RunStatus.SUCCESS, 1.0)

        entries = tracker.get_entries(status="FAILURE")

        assert len(entries) == 1
        assert entries[0].status == "FAILURE"

    def test_node_stats_tracking(self) -> None:
        """Test that node statistics are tracked correctly."""
        tracker = TickHistoryTracker("test-tree")

        tracker.record_tick("node-a", ["node-a"], RunStatus.SUCCESS, 5.0)
        tracker.record_tick("node-a", ["node-a"], RunStatus.FAILURE, 3.0)
        tracker.record_tick("node-a", ["node-a"], RunStatus.SUCCESS, 7.0)

        stats = tracker.get_node_stats("node-a")

        assert stats is not None
        assert stats.tick_count == 3
        assert stats.success_count == 2
        assert stats.failure_count == 1
        assert stats.avg_tick_duration_ms == 5.0  # (5+3+7)/3

    def test_max_entries_limit(self) -> None:
        """Test that history respects max entries limit."""
        tracker = TickHistoryTracker("test-tree", max_entries=5)

        for i in range(10):
            tracker.record_tick(f"node-{i}", [f"node-{i}"], RunStatus.SUCCESS, 1.0)

        assert tracker.entry_count == 5
        # Only newest entries kept
        entries = tracker.get_entries()
        assert entries[0].tick_number == 10

    def test_clear_history(self) -> None:
        """Test clearing history."""
        tracker = TickHistoryTracker("test-tree")

        tracker.record_tick("node-1", ["node-1"], RunStatus.SUCCESS, 1.0)
        tracker.clear()

        assert tracker.entry_count == 0
        assert tracker.tick_number == 0


class TestHistoryRegistry:
    """Tests for HistoryRegistry class."""

    def test_get_or_create(self) -> None:
        """Test getting or creating tracker."""
        registry = HistoryRegistry()

        tracker1 = registry.get_or_create("tree-a")
        tracker2 = registry.get_or_create("tree-a")
        tracker3 = registry.get_or_create("tree-b")

        assert tracker1 is tracker2
        assert tracker1 is not tracker3

    def test_list_trees(self) -> None:
        """Test listing trees with history."""
        registry = HistoryRegistry()

        registry.get_or_create("tree-a")
        registry.get_or_create("tree-b")

        trees = registry.list_trees()

        assert "tree-a" in trees
        assert "tree-b" in trees

    def test_remove(self) -> None:
        """Test removing tracker."""
        registry = HistoryRegistry()

        registry.get_or_create("tree-a")
        removed = registry.remove("tree-a")

        assert removed is True
        assert registry.get("tree-a") is None


# =============================================================================
# Test Breakpoints
# =============================================================================


class TestBreakpointManager:
    """Tests for BreakpointManager class."""

    def test_set_breakpoint(self) -> None:
        """Test setting a breakpoint."""
        manager = BreakpointManager("test-tree")

        bp = manager.set_breakpoint("node-a")

        assert bp.node_id == "node-a"
        assert bp.enabled is True
        assert bp.condition is None
        assert len(manager.get_all_breakpoints()) == 1

    def test_set_breakpoint_with_condition(self) -> None:
        """Test setting conditional breakpoint."""
        manager = BreakpointManager("test-tree")

        bp = manager.set_breakpoint("node-a", condition="count > 5")

        assert bp.condition == "count > 5"

    def test_update_breakpoint(self) -> None:
        """Test updating existing breakpoint."""
        manager = BreakpointManager("test-tree")

        manager.set_breakpoint("node-a", enabled=True)
        bp = manager.set_breakpoint("node-a", enabled=False)

        assert bp.enabled is False
        # Should still be only one breakpoint
        assert len(manager.get_all_breakpoints()) == 1

    def test_remove_breakpoint(self) -> None:
        """Test removing a breakpoint."""
        manager = BreakpointManager("test-tree")

        manager.set_breakpoint("node-a")
        removed = manager.remove_breakpoint("node-a")

        assert removed is True
        assert manager.get_breakpoint("node-a") is None

    def test_remove_nonexistent_breakpoint(self) -> None:
        """Test removing nonexistent breakpoint returns False."""
        manager = BreakpointManager("test-tree")

        removed = manager.remove_breakpoint("nonexistent")

        assert removed is False

    def test_enable_disable_breakpoint(self) -> None:
        """Test enabling/disabling breakpoint."""
        manager = BreakpointManager("test-tree")

        manager.set_breakpoint("node-a", enabled=True)

        manager.disable_breakpoint("node-a")
        assert manager.get_breakpoint("node-a").enabled is False

        manager.enable_breakpoint("node-a")
        assert manager.get_breakpoint("node-a").enabled is True

    def test_get_enabled_breakpoints(self) -> None:
        """Test getting only enabled breakpoints."""
        manager = BreakpointManager("test-tree")

        manager.set_breakpoint("node-a", enabled=True)
        manager.set_breakpoint("node-b", enabled=False)
        manager.set_breakpoint("node-c", enabled=True)

        enabled = manager.get_enabled_breakpoints()

        assert len(enabled) == 2
        assert all(bp.enabled for bp in enabled)

    def test_check_breakpoint_triggers(self) -> None:
        """Test that check_breakpoint returns hit when breakpoint set."""
        manager = BreakpointManager("test-tree")

        manager.set_breakpoint("node-a")

        hit = manager.check_breakpoint("node-a")

        assert hit is not None
        assert hit.node_id == "node-a"
        assert manager.is_paused is True

    def test_check_breakpoint_disabled(self) -> None:
        """Test that disabled breakpoint doesn't trigger."""
        manager = BreakpointManager("test-tree")

        manager.set_breakpoint("node-a", enabled=False)

        hit = manager.check_breakpoint("node-a")

        assert hit is None
        assert manager.is_paused is False

    def test_check_breakpoint_condition_true(self) -> None:
        """Test conditional breakpoint when condition is true."""
        manager = BreakpointManager("test-tree")

        manager.set_breakpoint("node-a", condition="value == 5")

        # Create mock blackboard with value=5
        mock_bb = MagicMock()
        mock_bb.snapshot.return_value = {"value": 5}

        hit = manager.check_breakpoint("node-a", mock_bb)

        assert hit is not None

    def test_check_breakpoint_condition_false(self) -> None:
        """Test conditional breakpoint when condition is false."""
        manager = BreakpointManager("test-tree")

        manager.set_breakpoint("node-a", condition="value == 5")

        # Create mock blackboard with value=10
        mock_bb = MagicMock()
        mock_bb.snapshot.return_value = {"value": 10}

        hit = manager.check_breakpoint("node-a", mock_bb)

        assert hit is None

    def test_hit_count_tracking(self) -> None:
        """Test that hit count is tracked."""
        manager = BreakpointManager("test-tree")

        manager.set_breakpoint("node-a")

        manager.check_breakpoint("node-a")
        manager.resume()
        manager.check_breakpoint("node-a")

        bp = manager.get_breakpoint("node-a")
        assert bp.hit_count == 2

    def test_clear_all_breakpoints(self) -> None:
        """Test clearing all breakpoints."""
        manager = BreakpointManager("test-tree")

        manager.set_breakpoint("node-a")
        manager.set_breakpoint("node-b")
        manager.set_breakpoint("node-c")

        count = manager.clear_all()

        assert count == 3
        assert len(manager.get_all_breakpoints()) == 0

    def test_resume(self) -> None:
        """Test resuming after breakpoint hit."""
        manager = BreakpointManager("test-tree")

        manager.set_breakpoint("node-a")
        manager.check_breakpoint("node-a")

        assert manager.is_paused is True

        manager.resume()

        assert manager.is_paused is False


class TestBreakpointRegistry:
    """Tests for BreakpointRegistry class."""

    def test_get_or_create(self) -> None:
        """Test getting or creating manager."""
        registry = BreakpointRegistry()

        manager1 = registry.get_or_create("tree-a")
        manager2 = registry.get_or_create("tree-a")

        assert manager1 is manager2

    def test_remove(self) -> None:
        """Test removing manager."""
        registry = BreakpointRegistry()

        registry.get_or_create("tree-a")
        removed = registry.remove("tree-a")

        assert removed is True
        assert registry.get("tree-a") is None


# =============================================================================
# Test Model Serialization
# =============================================================================


class TestModelSerialization:
    """Tests for debug model serialization."""

    def test_breakpoint_to_dict(self) -> None:
        """Test Breakpoint to_dict method."""
        bp = Breakpoint(
            node_id="test-node",
            tree_id="test-tree",
            enabled=True,
            condition="x > 0",
        )

        result = bp.to_dict()

        assert result["node_id"] == "test-node"
        assert result["tree_id"] == "test-tree"
        assert result["enabled"] is True
        assert result["condition"] == "x > 0"
        assert "created_at" in result

    def test_tick_history_to_dict(self) -> None:
        """Test TickHistoryTracker to_dict method."""
        tracker = TickHistoryTracker("test-tree")

        tracker.record_tick("node-a", ["node-a"], RunStatus.SUCCESS, 5.0)

        result = tracker.to_dict()

        assert result["tree_id"] == "test-tree"
        assert result["entry_count"] == 1
        assert len(result["entries"]) == 1
        assert "node_stats" in result
