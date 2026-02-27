"""
Unit tests for TickScheduler class.

Tests tasks 1.5.1-1.5.6 from tasks.md:
- Event buffering during tick
- Budget tracking
- Tick result generation

From footgun-addendum.md A.2:
- Events buffered during tick
- Buffer overflow drops oldest
- Buffer flushed after tick

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
    TickScheduler,
    TickResult,
)
from backend.src.services.ans.event import Event, Severity
from backend.src.services.ans.bus import EventBus, reset_event_bus


# =============================================================================
# Mock Node Implementation
# =============================================================================


class MockNode:
    """Mock behavior node for testing scheduler."""

    def __init__(
        self,
        node_id: str = "mock-node",
        tick_result: RunStatus = RunStatus.SUCCESS,
        children: Optional[List["MockNode"]] = None,
    ):
        self._id = node_id
        self._status = RunStatus.FRESH
        self._tick_result = tick_result
        self.children = children or []
        self.running_since: Optional[datetime] = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> RunStatus:
        return self._status

    def tick(self, ctx: TickContext) -> RunStatus:
        self._status = self._tick_result
        if self._tick_result == RunStatus.RUNNING:
            self.running_since = datetime.now(timezone.utc)
        return self._tick_result

    def reset(self) -> None:
        self._status = RunStatus.FRESH
        self.running_since = None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clean_event_bus():
    """Reset event bus before and after test."""
    reset_event_bus()
    yield
    reset_event_bus()


# =============================================================================
# Test TickResult
# =============================================================================


class TestTickResult:
    """Tests for TickResult dataclass."""

    def test_create_tick_result(self) -> None:
        """Test creating TickResult."""
        result = TickResult(
            status=RunStatus.SUCCESS,
            tree_status=TreeStatus.COMPLETED,
            tick_count=5,
            duration_ms=100.5,
            nodes_ticked=10,
            budget_exceeded=False,
            stuck_node=None,
            events_emitted=3,
        )

        assert result.status == RunStatus.SUCCESS
        assert result.tree_status == TreeStatus.COMPLETED
        assert result.tick_count == 5
        assert result.duration_ms == 100.5
        assert result.nodes_ticked == 10
        assert result.budget_exceeded is False
        assert result.stuck_node is None
        assert result.events_emitted == 3

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        result = TickResult(
            status=RunStatus.FAILURE,
            tree_status=TreeStatus.FAILED,
            tick_count=3,
            duration_ms=50.0,
        )

        data = result.to_dict()
        assert data["status"] == "FAILURE"
        assert data["tree_status"] == "failed"
        assert data["tick_count"] == 3
        assert data["duration_ms"] == 50.0


# =============================================================================
# Test TickScheduler Creation
# =============================================================================


class TestTickSchedulerCreation:
    """Tests for TickScheduler instantiation."""

    def test_create_basic(self) -> None:
        """Test creating scheduler with defaults."""
        scheduler = TickScheduler()

        assert scheduler.max_buffer_size == 1000
        assert scheduler.emit_tick_events is True
        assert scheduler.event_bus is None
        assert scheduler.watchdog is None

    def test_create_with_event_bus(self) -> None:
        """Test creating scheduler with event bus."""
        bus = EventBus()
        scheduler = TickScheduler(event_bus=bus)

        assert scheduler.event_bus is bus

    def test_create_with_watchdog(self) -> None:
        """Test creating scheduler with watchdog."""
        watchdog = TreeWatchdog()
        scheduler = TickScheduler(watchdog=watchdog)

        assert scheduler.watchdog is watchdog


# =============================================================================
# Test run_tick
# =============================================================================


class TestTickSchedulerRunTick:
    """Tests for run_tick method."""

    def test_run_tick_returns_result(self) -> None:
        """Test run_tick returns TickResult."""
        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        scheduler = TickScheduler()
        result = scheduler.run_tick(tree, ctx)

        assert isinstance(result, TickResult)
        assert result.status == RunStatus.SUCCESS
        assert result.tree_status == TreeStatus.COMPLETED

    def test_run_tick_increments_context_tick(self) -> None:
        """Test run_tick increments context tick count."""
        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        scheduler = TickScheduler()
        scheduler.run_tick(tree, ctx)

        assert ctx.tick_count == 1

    def test_run_tick_measures_duration(self) -> None:
        """Test run_tick measures duration."""
        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        scheduler = TickScheduler()
        result = scheduler.run_tick(tree, ctx)

        assert result.duration_ms >= 0

    def test_run_tick_detects_budget_exceeded(self) -> None:
        """Test run_tick reports budget exceeded."""
        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test", name="Test", root=root)
        # Start with budget already exceeded
        ctx = TickContext(blackboard=tree.blackboard, tick_budget=0)

        scheduler = TickScheduler()
        result = scheduler.run_tick(tree, ctx)

        assert result.budget_exceeded is True


# =============================================================================
# Test Event Buffering
# =============================================================================


class TestTickSchedulerEventBuffering:
    """Tests for event buffering (A.2 from footgun-addendum.md)."""

    def test_events_buffered_during_tick(self, clean_event_bus) -> None:
        """Test events are buffered during tick."""
        bus = EventBus()
        scheduler = TickScheduler(event_bus=bus, emit_tick_events=False)

        # Track dispatched events
        dispatched = []
        bus.subscribe_all(lambda e: dispatched.append(e))

        # Emit event during tick scope
        with scheduler.tick_scope():
            event = Event(
                type="test.event",
                source="test",
                severity=Severity.INFO,
            )
            scheduler.emit(event)

            # Should be buffered, not dispatched
            assert scheduler.buffer_size == 1
            assert len(dispatched) == 0

        # After scope exits, event should be dispatched
        assert scheduler.buffer_size == 0
        assert len(dispatched) == 1
        assert dispatched[0].type == "test.event"

    def test_events_dispatch_immediately_outside_tick(self, clean_event_bus) -> None:
        """Test events dispatch immediately outside tick scope."""
        bus = EventBus()
        scheduler = TickScheduler(event_bus=bus)

        dispatched = []
        bus.subscribe_all(lambda e: dispatched.append(e))

        event = Event(
            type="test.event",
            source="test",
            severity=Severity.INFO,
        )
        scheduler.emit(event)

        # Should dispatch immediately
        assert scheduler.buffer_size == 0
        assert len(dispatched) == 1

    def test_buffer_overflow_drops_oldest(self, clean_event_bus) -> None:
        """Test buffer overflow drops oldest events."""
        scheduler = TickScheduler(max_buffer_size=3)

        with scheduler.tick_scope():
            for i in range(5):
                event = Event(
                    type=f"test.event.{i}",
                    source="test",
                    severity=Severity.INFO,
                )
                scheduler.emit(event)

            # Should only have 3 events (dropped oldest 2)
            assert scheduler.buffer_size == 3

    def test_buffer_fifo_order(self, clean_event_bus) -> None:
        """Test buffer flushes in FIFO order."""
        bus = EventBus()
        scheduler = TickScheduler(event_bus=bus, emit_tick_events=False)

        dispatched = []
        bus.subscribe_all(lambda e: dispatched.append(e))

        with scheduler.tick_scope():
            for i in range(3):
                event = Event(
                    type=f"test.event.{i}",
                    source="test",
                    severity=Severity.INFO,
                )
                scheduler.emit(event)

        # Events should be in order
        assert len(dispatched) == 3
        assert dispatched[0].type == "test.event.0"
        assert dispatched[1].type == "test.event.1"
        assert dispatched[2].type == "test.event.2"


# =============================================================================
# Test tick_scope Context Manager
# =============================================================================


class TestTickSchedulerTickScope:
    """Tests for tick_scope context manager."""

    def test_tick_in_progress_flag(self) -> None:
        """Test tick_in_progress flag is set correctly."""
        scheduler = TickScheduler()

        assert scheduler.is_tick_in_progress is False

        with scheduler.tick_scope():
            assert scheduler.is_tick_in_progress is True

        assert scheduler.is_tick_in_progress is False

    def test_tick_scope_clears_on_exception(self) -> None:
        """Test tick_in_progress cleared even on exception."""
        scheduler = TickScheduler()

        with pytest.raises(ValueError):
            with scheduler.tick_scope():
                assert scheduler.is_tick_in_progress is True
                raise ValueError("test error")

        assert scheduler.is_tick_in_progress is False


# =============================================================================
# Test Watchdog Integration
# =============================================================================


class TestTickSchedulerWatchdogIntegration:
    """Tests for watchdog integration."""

    def test_checks_watchdog_during_tick(self, clean_event_bus) -> None:
        """Test watchdog is checked during run_tick."""
        watchdog = MagicMock(spec=TreeWatchdog)
        watchdog.check_stuck.return_value = None

        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        scheduler = TickScheduler(watchdog=watchdog)
        scheduler.run_tick(tree, ctx)

        watchdog.check_stuck.assert_called_once()

    def test_stuck_node_in_result(self, clean_event_bus) -> None:
        """Test stuck node is included in result."""
        from backend.src.bt.core import StuckNodeInfo

        stuck_info = StuckNodeInfo(
            node_id="stuck-node",
            node_path=["root", "stuck-node"],
            running_duration_ms=65000.0,
            last_progress_at=None,
            is_warning=False,
        )

        watchdog = MagicMock(spec=TreeWatchdog)
        watchdog.check_stuck.return_value = stuck_info
        watchdog.create_timeout_event.return_value = Event(
            type="tree.watchdog.timeout",
            source="test",
            severity=Severity.CRITICAL,
        )

        root = MockNode(node_id="root", tick_result=RunStatus.RUNNING)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        scheduler = TickScheduler(watchdog=watchdog)
        result = scheduler.run_tick(tree, ctx)

        assert result.stuck_node is not None
        assert result.stuck_node.node_id == "stuck-node"


# =============================================================================
# Test Tick Events
# =============================================================================


class TestTickSchedulerTickEvents:
    """Tests for tick start/complete events."""

    def test_emits_tick_start_event(self, clean_event_bus) -> None:
        """Test tick start event is emitted."""
        bus = EventBus()
        dispatched = []
        bus.subscribe_all(lambda e: dispatched.append(e))

        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test-tree", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        scheduler = TickScheduler(event_bus=bus, emit_tick_events=True)
        scheduler.run_tick(tree, ctx)

        start_events = [e for e in dispatched if e.type == "tree.tick.start"]
        assert len(start_events) == 1
        assert start_events[0].payload["tree_id"] == "test-tree"

    def test_emits_tick_complete_event(self, clean_event_bus) -> None:
        """Test tick complete event is emitted."""
        bus = EventBus()
        dispatched = []
        bus.subscribe_all(lambda e: dispatched.append(e))

        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test-tree", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        scheduler = TickScheduler(event_bus=bus, emit_tick_events=True)
        scheduler.run_tick(tree, ctx)

        complete_events = [e for e in dispatched if e.type == "tree.tick.complete"]
        assert len(complete_events) == 1
        assert complete_events[0].payload["tree_id"] == "test-tree"
        assert complete_events[0].payload["status"] == "success"

    def test_can_disable_tick_events(self, clean_event_bus) -> None:
        """Test tick events can be disabled."""
        bus = EventBus()
        dispatched = []
        bus.subscribe_all(lambda e: dispatched.append(e))

        root = MockNode(node_id="root", tick_result=RunStatus.SUCCESS)
        tree = BehaviorTree(id="test", name="Test", root=root)
        ctx = TickContext(blackboard=tree.blackboard)

        scheduler = TickScheduler(event_bus=bus, emit_tick_events=False)
        scheduler.run_tick(tree, ctx)

        tick_events = [e for e in dispatched if e.type.startswith("tree.tick")]
        assert len(tick_events) == 0


# =============================================================================
# Test Buffer Management
# =============================================================================


class TestTickSchedulerBufferManagement:
    """Tests for buffer management methods."""

    def test_clear_buffer(self) -> None:
        """Test clear_buffer clears without dispatching."""
        scheduler = TickScheduler()

        with scheduler.tick_scope():
            for i in range(5):
                event = Event(
                    type=f"test.event.{i}",
                    source="test",
                    severity=Severity.INFO,
                )
                scheduler.emit(event)

            assert scheduler.buffer_size == 5

            count = scheduler.clear_buffer()
            assert count == 5
            assert scheduler.buffer_size == 0


# =============================================================================
# Test Debug Info
# =============================================================================


class TestTickSchedulerDebugInfo:
    """Tests for debug_info method."""

    def test_debug_info_contains_fields(self) -> None:
        """Test debug_info returns expected fields."""
        bus = EventBus()
        watchdog = TreeWatchdog()
        scheduler = TickScheduler(
            event_bus=bus,
            watchdog=watchdog,
            max_buffer_size=500,
        )

        info = scheduler.debug_info()

        assert info["tick_in_progress"] is False
        assert info["buffer_size"] == 0
        assert info["max_buffer_size"] == 500
        assert info["emit_tick_events"] is True
        assert info["has_event_bus"] is True
        assert info["has_watchdog"] is True
