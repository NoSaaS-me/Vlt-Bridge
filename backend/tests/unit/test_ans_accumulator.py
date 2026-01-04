"""Unit tests for the ANS NotificationAccumulator (T018)."""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from backend.src.services.ans.accumulator import (
    Notification,
    NotificationAccumulator,
)
from backend.src.services.ans.event import Event, Severity
from backend.src.services.ans.subscriber import (
    BatchingConfig,
    InjectionPoint,
    Priority,
    Subscriber,
    SubscriberConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def accumulator() -> NotificationAccumulator:
    """Create a fresh NotificationAccumulator instance."""
    return NotificationAccumulator()


@pytest.fixture
def sample_event() -> Event:
    """Create a sample event for testing."""
    return Event(
        type="tool.call.failure",
        source="oracle_agent",
        severity=Severity.ERROR,
        payload={"tool_name": "bash", "error": "timeout"},
    )


@pytest.fixture
def normal_subscriber() -> Subscriber:
    """Create a normal priority subscriber."""
    config = SubscriberConfig(
        id="test_sub",
        name="Test Subscriber",
        description="Test",
        version="1.0.0",
        event_types=["tool.call.failure"],
        template="test.toon.j2",
        priority=Priority.NORMAL,
        inject_at=InjectionPoint.AFTER_TOOL,
        batching=BatchingConfig(
            window_ms=2000,
            max_size=5,
            dedupe_key=None,
            dedupe_window_ms=0,  # No deduplication
        ),
    )
    return Subscriber(config=config)


@pytest.fixture
def critical_subscriber() -> Subscriber:
    """Create a critical priority subscriber."""
    config = SubscriberConfig(
        id="critical_sub",
        name="Critical Subscriber",
        description="Critical",
        version="1.0.0",
        event_types=["tool.call.failure"],
        template="critical.toon.j2",
        priority=Priority.CRITICAL,
        inject_at=InjectionPoint.IMMEDIATE,
        batching=BatchingConfig(
            window_ms=0,
            max_size=1,
        ),
    )
    return Subscriber(config=config)


@pytest.fixture
def deduping_subscriber() -> Subscriber:
    """Create a subscriber with deduplication enabled."""
    config = SubscriberConfig(
        id="dedupe_sub",
        name="Deduping Subscriber",
        description="Deduplication test",
        version="1.0.0",
        event_types=["tool.call.failure"],
        template="test.toon.j2",
        priority=Priority.NORMAL,
        inject_at=InjectionPoint.AFTER_TOOL,
        batching=BatchingConfig(
            window_ms=2000,
            max_size=10,
            dedupe_key="type:payload.tool_name",
            dedupe_window_ms=5000,
        ),
    )
    return Subscriber(config=config)


# =============================================================================
# Notification Dataclass Tests
# =============================================================================


class TestNotificationDataclass:
    """Tests for the Notification dataclass."""

    def test_notification_default_values(self) -> None:
        """Notification has sensible defaults."""
        notification = Notification()

        assert notification.subscriber_id == ""
        assert notification.content == ""
        assert notification.priority == Priority.NORMAL
        assert notification.inject_at == InjectionPoint.AFTER_TOOL
        assert notification.events == []
        assert notification.id is not None
        assert notification.timestamp is not None

    def test_notification_to_dict(self) -> None:
        """Notification.to_dict returns proper dictionary."""
        event = Event(type="test", source="test", severity=Severity.INFO)
        notification = Notification(
            subscriber_id="test_sub",
            content="Test content",
            priority=Priority.HIGH,
            inject_at=InjectionPoint.TURN_START,
            events=[event],
        )

        result = notification.to_dict()

        assert result["subscriber_id"] == "test_sub"
        assert result["content"] == "Test content"
        assert result["priority"] == "high"
        assert result["inject_at"] == "turn_start"
        assert result["event_count"] == 1
        assert "id" in result
        assert "timestamp" in result


# =============================================================================
# Subscriber Registration Tests
# =============================================================================


class TestSubscriberRegistration:
    """Tests for subscriber registration."""

    def test_register_subscriber(
        self, accumulator: NotificationAccumulator, normal_subscriber: Subscriber
    ) -> None:
        """Subscriber can be registered with the accumulator."""
        accumulator.register_subscriber(normal_subscriber)

        assert accumulator.is_subscriber_enabled(normal_subscriber.id)

    def test_register_multiple_subscribers(
        self, accumulator: NotificationAccumulator
    ) -> None:
        """Multiple subscribers can be registered."""
        subscribers = [
            Subscriber(config=SubscriberConfig(
                id=f"sub_{i}",
                name=f"Subscriber {i}",
                description="Test",
                version="1.0.0",
                event_types=["test"],
                template="test.j2",
            ))
            for i in range(3)
        ]

        accumulator.register_subscribers(subscribers)

        for sub in subscribers:
            assert accumulator.is_subscriber_enabled(sub.id)


# =============================================================================
# Disabled Subscriber Tests
# =============================================================================


class TestDisabledSubscribers:
    """Tests for disabled subscriber handling."""

    def test_set_disabled_subscribers_from_set(
        self, accumulator: NotificationAccumulator, normal_subscriber: Subscriber
    ) -> None:
        """Disabled subscribers can be set from a set."""
        accumulator.register_subscriber(normal_subscriber)
        accumulator.set_disabled_subscribers({"test_sub"})

        assert not accumulator.is_subscriber_enabled("test_sub")

    def test_set_disabled_subscribers_from_list(
        self, accumulator: NotificationAccumulator, normal_subscriber: Subscriber
    ) -> None:
        """Disabled subscribers can be set from a list."""
        accumulator.register_subscriber(normal_subscriber)
        accumulator.set_disabled_subscribers(["test_sub"])

        assert not accumulator.is_subscriber_enabled("test_sub")

    def test_disabled_subscriber_skips_accumulation(
        self,
        accumulator: NotificationAccumulator,
        normal_subscriber: Subscriber,
        sample_event: Event,
    ) -> None:
        """Accumulating for a disabled subscriber returns None."""
        accumulator.register_subscriber(normal_subscriber)
        accumulator.set_disabled_subscribers({"test_sub"})

        result = accumulator.accumulate(sample_event, normal_subscriber)

        assert result is None
        assert accumulator.pending_count == 0

    def test_subscriber_with_enabled_false(
        self, accumulator: NotificationAccumulator, sample_event: Event
    ) -> None:
        """Subscriber with enabled=False is skipped."""
        config = SubscriberConfig(
            id="disabled_sub",
            name="Disabled Subscriber",
            description="Test",
            version="1.0.0",
            event_types=["tool.call.failure"],
            template="test.j2",
        )
        subscriber = Subscriber(config=config, enabled=False)
        accumulator.register_subscriber(subscriber)

        result = accumulator.accumulate(sample_event, subscriber)

        assert result is None

    def test_unregistered_subscriber_assumed_enabled(
        self, accumulator: NotificationAccumulator
    ) -> None:
        """Unknown subscriber ID is assumed enabled."""
        assert accumulator.is_subscriber_enabled("unknown_sub")


# =============================================================================
# Event Accumulation Tests
# =============================================================================


class TestEventAccumulation:
    """Tests for event accumulation behavior."""

    def test_accumulate_adds_to_pending(
        self,
        accumulator: NotificationAccumulator,
        normal_subscriber: Subscriber,
        sample_event: Event,
    ) -> None:
        """Accumulating an event adds it to pending."""
        accumulator.register_subscriber(normal_subscriber)

        result = accumulator.accumulate(sample_event, normal_subscriber)

        # Not enough to trigger flush, so no notification returned
        assert result is None
        assert accumulator.pending_count == 1

    def test_accumulate_flushes_at_max_size(
        self,
        accumulator: NotificationAccumulator,
        normal_subscriber: Subscriber,
    ) -> None:
        """Accumulator flushes when batch max_size is reached."""
        accumulator.register_subscriber(normal_subscriber)

        # Send events up to max_size (5)
        for i in range(4):
            event = Event(
                type="tool.call.failure",
                source="test",
                severity=Severity.ERROR,
                payload={"index": i},
            )
            result = accumulator.accumulate(event, normal_subscriber)
            assert result is None  # Not yet at max

        # Fifth event should trigger flush
        fifth_event = Event(
            type="tool.call.failure",
            source="test",
            severity=Severity.ERROR,
            payload={"index": 4},
        )
        result = accumulator.accumulate(fifth_event, normal_subscriber)

        assert result is not None
        assert len(result.events) == 5
        assert accumulator.pending_count == 0

    def test_critical_priority_bypasses_batching(
        self,
        accumulator: NotificationAccumulator,
        critical_subscriber: Subscriber,
        sample_event: Event,
    ) -> None:
        """Critical priority subscriber gets immediate notification."""
        accumulator.register_subscriber(critical_subscriber)

        result = accumulator.accumulate(sample_event, critical_subscriber)

        assert result is not None
        assert result.priority == Priority.CRITICAL
        assert result.inject_at == InjectionPoint.IMMEDIATE
        assert len(result.events) == 1


# =============================================================================
# Deduplication Tests
# =============================================================================


class TestDeduplication:
    """Tests for event deduplication."""

    def test_duplicate_event_within_window_skipped(
        self,
        accumulator: NotificationAccumulator,
        deduping_subscriber: Subscriber,
    ) -> None:
        """Duplicate events within the dedupe window are skipped."""
        accumulator.register_subscriber(deduping_subscriber)

        event1 = Event(
            type="tool.call.failure",
            source="test",
            severity=Severity.ERROR,
            payload={"tool_name": "bash"},
        )
        event2 = Event(
            type="tool.call.failure",
            source="test",
            severity=Severity.ERROR,
            payload={"tool_name": "bash"},  # Same tool_name
        )

        accumulator.accumulate(event1, deduping_subscriber)
        accumulator.accumulate(event2, deduping_subscriber)

        # Only one should be pending due to deduplication
        assert accumulator.pending_count == 1

    def test_different_dedupe_key_not_skipped(
        self,
        accumulator: NotificationAccumulator,
        deduping_subscriber: Subscriber,
    ) -> None:
        """Events with different dedupe keys are not skipped."""
        accumulator.register_subscriber(deduping_subscriber)

        event1 = Event(
            type="tool.call.failure",
            source="test",
            severity=Severity.ERROR,
            payload={"tool_name": "bash"},
        )
        event2 = Event(
            type="tool.call.failure",
            source="test",
            severity=Severity.ERROR,
            payload={"tool_name": "python"},  # Different tool_name
        )

        accumulator.accumulate(event1, deduping_subscriber)
        accumulator.accumulate(event2, deduping_subscriber)

        # Both should be pending
        assert accumulator.pending_count == 2

    def test_no_deduplication_when_window_zero(
        self,
        accumulator: NotificationAccumulator,
        normal_subscriber: Subscriber,
    ) -> None:
        """No deduplication when dedupe_window_ms is 0."""
        accumulator.register_subscriber(normal_subscriber)

        event1 = Event(
            type="tool.call.failure",
            source="test",
            severity=Severity.ERROR,
            payload={"tool_name": "bash"},
        )
        event2 = Event(
            type="tool.call.failure",
            source="test",
            severity=Severity.ERROR,
            payload={"tool_name": "bash"},
        )

        accumulator.accumulate(event1, normal_subscriber)
        accumulator.accumulate(event2, normal_subscriber)

        # Both should be pending
        assert accumulator.pending_count == 2


# =============================================================================
# Flush Tests
# =============================================================================


class TestFlush:
    """Tests for flushing pending events."""

    def test_flush_all_returns_notifications(
        self,
        accumulator: NotificationAccumulator,
        normal_subscriber: Subscriber,
    ) -> None:
        """flush_all returns notifications for all pending events."""
        accumulator.register_subscriber(normal_subscriber)

        for i in range(3):
            event = Event(
                type="tool.call.failure",
                source="test",
                severity=Severity.ERROR,
                payload={"index": i},
            )
            accumulator.accumulate(event, normal_subscriber)

        notifications = accumulator.flush_all()

        assert len(notifications) == 1
        assert len(notifications[0].events) == 3
        assert accumulator.pending_count == 0

    def test_flush_all_with_multiple_subscribers(
        self, accumulator: NotificationAccumulator
    ) -> None:
        """flush_all returns notifications for all subscribers."""
        sub1 = Subscriber(config=SubscriberConfig(
            id="sub1",
            name="Sub 1",
            description="Test",
            version="1.0.0",
            event_types=["tool.call.failure"],
            template="test.j2",
        ))
        sub2 = Subscriber(config=SubscriberConfig(
            id="sub2",
            name="Sub 2",
            description="Test",
            version="1.0.0",
            event_types=["budget.warning"],
            template="test.j2",
        ))

        accumulator.register_subscribers([sub1, sub2])

        event1 = Event(type="tool.call.failure", source="test", severity=Severity.ERROR)
        event2 = Event(type="budget.warning", source="test", severity=Severity.WARNING)

        accumulator.accumulate(event1, sub1)
        accumulator.accumulate(event2, sub2)

        notifications = accumulator.flush_all()

        assert len(notifications) == 2

    def test_flush_all_on_empty_returns_empty(
        self, accumulator: NotificationAccumulator
    ) -> None:
        """flush_all on empty accumulator returns empty list."""
        notifications = accumulator.flush_all()

        assert notifications == []


# =============================================================================
# Drain Tests
# =============================================================================


class TestDrain:
    """Tests for draining notifications by injection point."""

    def test_drain_returns_notifications_for_point(
        self,
        accumulator: NotificationAccumulator,
        normal_subscriber: Subscriber,
        sample_event: Event,
    ) -> None:
        """drain returns notifications for the specified injection point."""
        accumulator.register_subscriber(normal_subscriber)

        # Fill to trigger notification
        for _ in range(5):
            accumulator.accumulate(sample_event, normal_subscriber)

        notifications = accumulator.drain(InjectionPoint.AFTER_TOOL)

        assert len(notifications) == 1
        assert notifications[0].inject_at == InjectionPoint.AFTER_TOOL

    def test_drain_clears_notifications(
        self,
        accumulator: NotificationAccumulator,
        normal_subscriber: Subscriber,
        sample_event: Event,
    ) -> None:
        """drain clears notifications after returning them."""
        accumulator.register_subscriber(normal_subscriber)

        for _ in range(5):
            accumulator.accumulate(sample_event, normal_subscriber)

        first_drain = accumulator.drain(InjectionPoint.AFTER_TOOL)
        second_drain = accumulator.drain(InjectionPoint.AFTER_TOOL)

        assert len(first_drain) == 1
        assert len(second_drain) == 0

    def test_drain_sorts_by_priority(
        self, accumulator: NotificationAccumulator
    ) -> None:
        """drain returns notifications sorted by priority (critical first)."""
        # Create subscribers with different priorities
        low_sub = Subscriber(config=SubscriberConfig(
            id="low_sub",
            name="Low",
            description="Test",
            version="1.0.0",
            event_types=["test"],
            template="test.j2",
            priority=Priority.LOW,
            inject_at=InjectionPoint.TURN_END,
        ))
        high_sub = Subscriber(config=SubscriberConfig(
            id="high_sub",
            name="High",
            description="Test",
            version="1.0.0",
            event_types=["test"],
            template="test.j2",
            priority=Priority.HIGH,
            inject_at=InjectionPoint.TURN_END,
        ))
        normal_sub = Subscriber(config=SubscriberConfig(
            id="normal_sub",
            name="Normal",
            description="Test",
            version="1.0.0",
            event_types=["test"],
            template="test.j2",
            priority=Priority.NORMAL,
            inject_at=InjectionPoint.TURN_END,
        ))

        accumulator.register_subscribers([low_sub, high_sub, normal_sub])

        # Add events in random priority order
        event = Event(type="test", source="test", severity=Severity.INFO)

        # Flush to create notifications
        for _ in range(10):  # Ensure flush
            accumulator.accumulate(event, low_sub)
            accumulator.accumulate(event, high_sub)
            accumulator.accumulate(event, normal_sub)

        accumulator.flush_all()
        notifications = accumulator.drain(InjectionPoint.TURN_END)

        # Should be sorted: high, normal, low
        priorities = [n.priority for n in notifications]
        assert priorities == sorted(
            priorities,
            key=lambda p: {
                Priority.CRITICAL: 0,
                Priority.HIGH: 1,
                Priority.NORMAL: 2,
                Priority.LOW: 3,
            }[p],
        )

    def test_drain_immediate(
        self,
        accumulator: NotificationAccumulator,
        critical_subscriber: Subscriber,
        sample_event: Event,
    ) -> None:
        """drain_immediate returns immediate notifications."""
        accumulator.register_subscriber(critical_subscriber)
        accumulator.accumulate(sample_event, critical_subscriber)

        notifications = accumulator.drain_immediate()

        assert len(notifications) == 1
        assert notifications[0].inject_at == InjectionPoint.IMMEDIATE

    def test_drain_turn_start(self, accumulator: NotificationAccumulator) -> None:
        """drain_turn_start returns turn_start notifications."""
        sub = Subscriber(config=SubscriberConfig(
            id="turn_start_sub",
            name="Turn Start",
            description="Test",
            version="1.0.0",
            event_types=["test"],
            template="test.j2",
            priority=Priority.CRITICAL,  # Bypass batching
            inject_at=InjectionPoint.TURN_START,
        ))
        accumulator.register_subscriber(sub)

        event = Event(type="test", source="test", severity=Severity.INFO)
        accumulator.accumulate(event, sub)

        notifications = accumulator.drain_turn_start()

        assert len(notifications) == 1
        assert notifications[0].inject_at == InjectionPoint.TURN_START

    def test_drain_after_tool(
        self,
        accumulator: NotificationAccumulator,
        normal_subscriber: Subscriber,
        sample_event: Event,
    ) -> None:
        """drain_after_tool returns after_tool notifications."""
        accumulator.register_subscriber(normal_subscriber)

        for _ in range(5):  # Trigger batch
            accumulator.accumulate(sample_event, normal_subscriber)

        notifications = accumulator.drain_after_tool()

        assert len(notifications) == 1
        assert notifications[0].inject_at == InjectionPoint.AFTER_TOOL

    def test_drain_turn_end(self, accumulator: NotificationAccumulator) -> None:
        """drain_turn_end returns turn_end notifications."""
        sub = Subscriber(config=SubscriberConfig(
            id="turn_end_sub",
            name="Turn End",
            description="Test",
            version="1.0.0",
            event_types=["test"],
            template="test.j2",
            priority=Priority.CRITICAL,  # Bypass batching
            inject_at=InjectionPoint.TURN_END,
        ))
        accumulator.register_subscriber(sub)

        event = Event(type="test", source="test", severity=Severity.INFO)
        accumulator.accumulate(event, sub)

        notifications = accumulator.drain_turn_end()

        assert len(notifications) == 1
        assert notifications[0].inject_at == InjectionPoint.TURN_END


# =============================================================================
# Clear Tests
# =============================================================================


class TestClear:
    """Tests for clearing the accumulator."""

    def test_clear_removes_pending_events(
        self,
        accumulator: NotificationAccumulator,
        normal_subscriber: Subscriber,
        sample_event: Event,
    ) -> None:
        """clear removes all pending events."""
        accumulator.register_subscriber(normal_subscriber)
        accumulator.accumulate(sample_event, normal_subscriber)

        accumulator.clear()

        assert accumulator.pending_count == 0

    def test_clear_removes_notifications(
        self,
        accumulator: NotificationAccumulator,
        critical_subscriber: Subscriber,
        sample_event: Event,
    ) -> None:
        """clear removes all pending notifications."""
        accumulator.register_subscriber(critical_subscriber)
        accumulator.accumulate(sample_event, critical_subscriber)

        accumulator.clear()

        assert accumulator.notification_count == 0

    def test_clear_removes_dedupe_tracking(
        self,
        accumulator: NotificationAccumulator,
        deduping_subscriber: Subscriber,
    ) -> None:
        """clear removes deduplication tracking."""
        accumulator.register_subscriber(deduping_subscriber)

        event = Event(
            type="tool.call.failure",
            source="test",
            severity=Severity.ERROR,
            payload={"tool_name": "bash"},
        )

        accumulator.accumulate(event, deduping_subscriber)
        accumulator.clear()

        # After clear, same event should not be deduplicated
        accumulator.accumulate(event, deduping_subscriber)

        assert accumulator.pending_count == 1


# =============================================================================
# Properties Tests
# =============================================================================


class TestProperties:
    """Tests for accumulator properties."""

    def test_pending_count_tracks_events(
        self,
        accumulator: NotificationAccumulator,
        normal_subscriber: Subscriber,
    ) -> None:
        """pending_count tracks total pending events across all subscribers."""
        accumulator.register_subscriber(normal_subscriber)

        assert accumulator.pending_count == 0

        event = Event(type="tool.call.failure", source="test", severity=Severity.ERROR)
        accumulator.accumulate(event, normal_subscriber)

        assert accumulator.pending_count == 1

        accumulator.accumulate(event, normal_subscriber)

        assert accumulator.pending_count == 2

    def test_notification_count_tracks_notifications(
        self,
        accumulator: NotificationAccumulator,
        critical_subscriber: Subscriber,
        sample_event: Event,
    ) -> None:
        """notification_count tracks pending notifications across all injection points."""
        accumulator.register_subscriber(critical_subscriber)

        assert accumulator.notification_count == 0

        accumulator.accumulate(sample_event, critical_subscriber)

        assert accumulator.notification_count == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_accumulate_without_registration(
        self, accumulator: NotificationAccumulator, sample_event: Event
    ) -> None:
        """Accumulating for an unregistered subscriber still works."""
        config = SubscriberConfig(
            id="unregistered",
            name="Unregistered",
            description="Test",
            version="1.0.0",
            event_types=["tool.call.failure"],
            template="test.j2",
            priority=Priority.CRITICAL,
        )
        subscriber = Subscriber(config=config)

        # Should not raise
        result = accumulator.accumulate(sample_event, subscriber)

        assert result is not None

    def test_notification_content_initially_empty(
        self,
        accumulator: NotificationAccumulator,
        critical_subscriber: Subscriber,
        sample_event: Event,
    ) -> None:
        """Notification content is empty (to be filled by formatter)."""
        accumulator.register_subscriber(critical_subscriber)

        result = accumulator.accumulate(sample_event, critical_subscriber)

        assert result.content == ""

    def test_dedupe_key_pattern_parsing(
        self, accumulator: NotificationAccumulator
    ) -> None:
        """Custom dedupe key pattern is correctly parsed."""
        config = SubscriberConfig(
            id="custom_dedupe",
            name="Custom Dedupe",
            description="Test",
            version="1.0.0",
            event_types=["test"],
            template="test.j2",
            batching=BatchingConfig(
                dedupe_key="type:payload.foo:payload.bar",
                dedupe_window_ms=5000,
            ),
        )
        subscriber = Subscriber(config=config)
        accumulator.register_subscriber(subscriber)

        event1 = Event(
            type="test",
            source="test",
            severity=Severity.INFO,
            payload={"foo": "a", "bar": "b"},
        )
        event2 = Event(
            type="test",
            source="test",
            severity=Severity.INFO,
            payload={"foo": "a", "bar": "b"},  # Same
        )
        event3 = Event(
            type="test",
            source="test",
            severity=Severity.INFO,
            payload={"foo": "a", "bar": "c"},  # Different bar
        )

        accumulator.accumulate(event1, subscriber)
        accumulator.accumulate(event2, subscriber)
        accumulator.accumulate(event3, subscriber)

        # event2 should be deduplicated, event3 should not
        assert accumulator.pending_count == 2
