"""
Unit tests for TreeWatchdog class.

Tests tasks 1.4.1-1.4.7 from tasks.md:
- Stuck detection based on progress
- Warning vs timeout thresholds
- Progress updates reset stuck timer

From footgun-addendum.md A.1:
- Progress is defined as blackboard write, async completion, or explicit mark_progress()
- Stuck = RUNNING > timeout without progress

Part of the BT Universal Runtime (spec 019).
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from backend.src.bt.state import RunStatus, TypedBlackboard
from backend.src.bt.core import (
    BehaviorTree,
    TreeStatus,
    TickContext,
    TreeWatchdog,
    StuckNodeInfo,
)


# =============================================================================
# Mock Node Implementation
# =============================================================================


class MockNode:
    """Mock behavior node for testing watchdog."""

    def __init__(
        self,
        node_id: str = "mock-node",
        status: RunStatus = RunStatus.FRESH,
        running_since: Optional[datetime] = None,
        children: Optional[List["MockNode"]] = None,
    ):
        self._id = node_id
        self._status = status
        self.running_since = running_since
        self.children = children or []

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> RunStatus:
        return self._status

    def set_running(self, since: Optional[datetime] = None) -> None:
        """Set node to RUNNING state."""
        self._status = RunStatus.RUNNING
        self.running_since = since or datetime.now(timezone.utc)

    def tick(self, ctx: TickContext) -> RunStatus:
        return self._status

    def reset(self) -> None:
        self._status = RunStatus.FRESH
        self.running_since = None


# =============================================================================
# Test StuckNodeInfo
# =============================================================================


class TestStuckNodeInfo:
    """Tests for StuckNodeInfo dataclass."""

    def test_create_stuck_node_info(self) -> None:
        """Test creating StuckNodeInfo."""
        info = StuckNodeInfo(
            node_id="test-node",
            node_path=["root", "child", "test-node"],
            running_duration_ms=45000.0,
            last_progress_at=None,
            is_warning=True,
        )

        assert info.node_id == "test-node"
        assert info.node_path == ["root", "child", "test-node"]
        assert info.running_duration_ms == 45000.0
        assert info.last_progress_at is None
        assert info.is_warning is True

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        now = datetime.now(timezone.utc)
        info = StuckNodeInfo(
            node_id="test-node",
            node_path=["root"],
            running_duration_ms=30000.0,
            last_progress_at=now,
            is_warning=False,
        )

        data = info.to_dict()
        assert data["node_id"] == "test-node"
        assert data["node_path"] == ["root"]
        assert data["running_duration_ms"] == 30000.0
        assert data["last_progress_at"] == now.isoformat()
        assert data["is_warning"] is False


# =============================================================================
# Test TreeWatchdog Creation
# =============================================================================


class TestTreeWatchdogCreation:
    """Tests for TreeWatchdog instantiation."""

    def test_create_with_defaults(self) -> None:
        """Test creating watchdog with default thresholds."""
        watchdog = TreeWatchdog()

        assert watchdog.warning_threshold_ms == 30000
        assert watchdog.timeout_ms == 60000

    def test_create_with_custom_thresholds(self) -> None:
        """Test creating watchdog with custom thresholds."""
        watchdog = TreeWatchdog(
            warning_threshold_ms=10000,
            timeout_ms=20000,
        )

        assert watchdog.warning_threshold_ms == 10000
        assert watchdog.timeout_ms == 20000


# =============================================================================
# Test Stuck Detection
# =============================================================================


class TestTreeWatchdogStuckDetection:
    """Tests for stuck node detection."""

    def test_no_running_nodes_returns_none(self) -> None:
        """Test check_stuck returns None when no nodes running."""
        root = MockNode(node_id="root", status=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)
        watchdog = TreeWatchdog()

        result = watchdog.check_stuck(tree, ctx)
        assert result is None

    def test_running_within_threshold_returns_none(self) -> None:
        """Test running node within warning threshold returns None."""
        # Node just started running (within threshold)
        root = MockNode(node_id="root", status=RunStatus.RUNNING)
        root.running_since = datetime.now(timezone.utc)

        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)
        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)

        # Force check (bypass throttle)
        watchdog._last_check_at = None

        result = watchdog.check_stuck(tree, ctx)
        assert result is None

    def test_running_past_warning_returns_warning(self) -> None:
        """Test running past warning threshold returns warning."""
        # Node running for 35 seconds (past 30s warning)
        root = MockNode(node_id="root", status=RunStatus.RUNNING)
        root.running_since = datetime.now(timezone.utc) - timedelta(seconds=35)

        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)
        # No progress marked (ctx._last_progress_at is None or before running_since)

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        watchdog._last_check_at = None

        result = watchdog.check_stuck(tree, ctx)
        assert result is not None
        assert result.is_warning is True
        assert result.node_id == "root"

    def test_running_past_timeout_returns_timeout(self) -> None:
        """Test running past timeout returns timeout (not warning)."""
        # Node running for 65 seconds (past 60s timeout)
        root = MockNode(node_id="root", status=RunStatus.RUNNING)
        root.running_since = datetime.now(timezone.utc) - timedelta(seconds=65)

        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        watchdog._last_check_at = None

        result = watchdog.check_stuck(tree, ctx)
        assert result is not None
        assert result.is_warning is False  # Timeout, not warning
        assert result.node_id == "root"
        assert result.running_duration_ms >= 65000

    def test_progress_prevents_stuck_detection(self) -> None:
        """Test that progress marks prevent stuck detection."""
        # Node running for 35 seconds
        root = MockNode(node_id="root", status=RunStatus.RUNNING)
        root.running_since = datetime.now(timezone.utc) - timedelta(seconds=35)

        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        # Mark progress recently (after node started running)
        ctx.mark_progress()

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        watchdog._last_check_at = None

        result = watchdog.check_stuck(tree, ctx)
        assert result is None  # Not stuck because progress was made

    def test_checks_child_nodes(self) -> None:
        """Test that check_stuck examines child nodes."""
        # Child node running for 35 seconds
        child = MockNode(node_id="child", status=RunStatus.RUNNING)
        child.running_since = datetime.now(timezone.utc) - timedelta(seconds=35)

        root = MockNode(node_id="root", status=RunStatus.SUCCESS, children=[child])
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        watchdog._last_check_at = None

        result = watchdog.check_stuck(tree, ctx)
        assert result is not None
        assert result.node_id == "child"


# =============================================================================
# Test Progress Tracking
# =============================================================================


class TestTreeWatchdogProgress:
    """Tests for progress-based stuck prevention."""

    def test_async_completion_marks_progress(self) -> None:
        """Test that completing async operation marks progress."""
        root = MockNode(node_id="root", status=RunStatus.RUNNING)
        root.running_since = datetime.now(timezone.utc) - timedelta(seconds=35)

        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        # Add and complete async operation
        ctx.add_async("op-1")
        ctx.complete_async("op-1")  # This marks progress

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        watchdog._last_check_at = None

        result = watchdog.check_stuck(tree, ctx)
        assert result is None  # Not stuck due to progress

    def test_blackboard_write_should_mark_progress(self) -> None:
        """Test concept: blackboard writes mark progress.

        Note: Actual integration requires TypedBlackboard callback.
        This test documents the expected behavior.
        """
        root = MockNode(node_id="root", status=RunStatus.RUNNING)
        root.running_since = datetime.now(timezone.utc) - timedelta(seconds=35)

        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        # Simulate what would happen with proper integration:
        # After bb.set() succeeds, ctx.mark_progress() would be called
        ctx.mark_progress()

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        watchdog._last_check_at = None

        result = watchdog.check_stuck(tree, ctx)
        assert result is None

    def test_explicit_mark_progress(self) -> None:
        """Test explicit mark_progress() prevents stuck."""
        root = MockNode(node_id="root", status=RunStatus.RUNNING)
        root.running_since = datetime.now(timezone.utc) - timedelta(seconds=35)

        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        # Explicitly mark progress
        ctx.mark_progress()

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        watchdog._last_check_at = None

        result = watchdog.check_stuck(tree, ctx)
        assert result is None


# =============================================================================
# Test Warning Deduplication
# =============================================================================


class TestTreeWatchdogWarningDedup:
    """Tests for warning deduplication."""

    def test_warning_only_once_per_node(self) -> None:
        """Test that warnings are only emitted once per node."""
        root = MockNode(node_id="root", status=RunStatus.RUNNING)
        root.running_since = datetime.now(timezone.utc) - timedelta(seconds=35)

        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)

        # First check - should return warning
        watchdog._last_check_at = None
        result1 = watchdog.check_stuck(tree, ctx)
        assert result1 is not None
        assert result1.is_warning is True

        # Second check - should return None (already warned)
        watchdog._last_check_at = None
        result2 = watchdog.check_stuck(tree, ctx)
        assert result2 is None

    def test_warned_nodes_list(self) -> None:
        """Test get_warned_nodes returns warned node IDs."""
        root = MockNode(node_id="root", status=RunStatus.RUNNING)
        root.running_since = datetime.now(timezone.utc) - timedelta(seconds=35)

        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        watchdog._last_check_at = None

        watchdog.check_stuck(tree, ctx)

        warned = watchdog.get_warned_nodes()
        assert "root" in warned

    def test_clear_warning(self) -> None:
        """Test clear_warning removes node from warned list."""
        root = MockNode(node_id="root", status=RunStatus.RUNNING)
        root.running_since = datetime.now(timezone.utc) - timedelta(seconds=35)

        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        watchdog._last_check_at = None

        watchdog.check_stuck(tree, ctx)
        assert "root" in watchdog.get_warned_nodes()

        watchdog.clear_warning("root")
        assert "root" not in watchdog.get_warned_nodes()


# =============================================================================
# Test Event Creation
# =============================================================================


class TestTreeWatchdogEvents:
    """Tests for event creation methods."""

    def test_create_warning_event(self) -> None:
        """Test warning event creation."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(id="test-tree", name="Test", root=root)

        stuck_info = StuckNodeInfo(
            node_id="child",
            node_path=["root", "child"],
            running_duration_ms=35000.0,
            last_progress_at=None,
            is_warning=True,
        )

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        event = watchdog.create_warning_event(tree, stuck_info)

        assert event.type == "tree.watchdog.warning"
        assert event.source == "tree_watchdog"
        assert event.payload["tree_id"] == "test-tree"
        assert event.payload["node_id"] == "child"
        assert event.payload["node_path"] == ["root", "child"]
        assert event.payload["warning_threshold_ms"] == 30000
        assert event.payload["timeout_ms"] == 60000

    def test_create_timeout_event(self) -> None:
        """Test timeout event creation."""
        root = MockNode(node_id="root")
        tree = BehaviorTree(id="test-tree", name="Test", root=root)

        now = datetime.now(timezone.utc)
        stuck_info = StuckNodeInfo(
            node_id="child",
            node_path=["root", "child"],
            running_duration_ms=65000.0,
            last_progress_at=now - timedelta(seconds=70),
            is_warning=False,
        )

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        event = watchdog.create_timeout_event(tree, stuck_info)

        assert event.type == "tree.watchdog.timeout"
        assert event.source == "tree_watchdog"
        assert event.payload["tree_id"] == "test-tree"
        assert event.payload["node_id"] == "child"
        assert event.payload["timeout_ms"] == 60000


# =============================================================================
# Test Reset
# =============================================================================


class TestTreeWatchdogReset:
    """Tests for watchdog reset."""

    def test_reset_clears_warned_nodes(self) -> None:
        """Test reset clears warned nodes."""
        root = MockNode(node_id="root", status=RunStatus.RUNNING)
        root.running_since = datetime.now(timezone.utc) - timedelta(seconds=35)

        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)
        watchdog._last_check_at = None

        watchdog.check_stuck(tree, ctx)
        assert len(watchdog.get_warned_nodes()) > 0

        watchdog.reset()
        assert len(watchdog.get_warned_nodes()) == 0


# =============================================================================
# Test Debug Info
# =============================================================================


class TestTreeWatchdogDebugInfo:
    """Tests for debug_info method."""

    def test_debug_info_contains_fields(self) -> None:
        """Test debug_info returns expected fields."""
        watchdog = TreeWatchdog(warning_threshold_ms=30000, timeout_ms=60000)

        info = watchdog.debug_info()

        assert info["warning_threshold_ms"] == 30000
        assert info["timeout_ms"] == 60000
        assert "warned_nodes" in info
        assert "last_check_at" in info
