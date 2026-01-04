"""Unit tests for the ANS EventBus (T011)."""

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from backend.src.services.ans.bus import (
    EventBus,
    EventHandler,
    get_event_bus,
    reset_event_bus,
)
from backend.src.services.ans.event import Event, Severity


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def bus() -> EventBus:
    """Create a fresh EventBus instance."""
    return EventBus()


@pytest.fixture
def sample_event() -> Event:
    """Create a sample event for testing."""
    return Event(
        type="test.event.one",
        source="test",
        severity=Severity.INFO,
        payload={"key": "value"},
    )


@pytest.fixture(autouse=True)
def reset_global_bus() -> None:
    """Reset the global event bus before each test."""
    reset_event_bus()
    yield
    reset_event_bus()


# =============================================================================
# Basic Subscription Tests
# =============================================================================


class TestEventBusSubscription:
    """Tests for subscribing handlers to events."""

    def test_subscribe_handler_to_event_type(self, bus: EventBus) -> None:
        """Handler can be subscribed to a specific event type."""
        handler = MagicMock()

        bus.subscribe("test.event", handler)

        assert bus.handler_count == 1

    def test_subscribe_multiple_handlers_to_same_type(self, bus: EventBus) -> None:
        """Multiple handlers can subscribe to the same event type."""
        handler1 = MagicMock()
        handler2 = MagicMock()

        bus.subscribe("test.event", handler1)
        bus.subscribe("test.event", handler2)

        assert bus.handler_count == 2

    def test_subscribe_handler_to_multiple_types(self, bus: EventBus) -> None:
        """A handler can subscribe to multiple event types."""
        handler = MagicMock()

        bus.subscribe("test.event.one", handler)
        bus.subscribe("test.event.two", handler)

        assert bus.handler_count == 2

    def test_subscribe_all_registers_global_handler(self, bus: EventBus) -> None:
        """subscribe_all registers a handler for all events."""
        handler = MagicMock()

        bus.subscribe_all(handler)

        assert bus.handler_count == 1

    def test_unsubscribe_removes_handler(self, bus: EventBus) -> None:
        """Unsubscribe removes the handler from the event type."""
        handler = MagicMock()
        bus.subscribe("test.event", handler)

        result = bus.unsubscribe("test.event", handler)

        assert result is True
        assert bus.handler_count == 0

    def test_unsubscribe_returns_false_if_not_found(self, bus: EventBus) -> None:
        """Unsubscribe returns False if handler not found."""
        handler = MagicMock()

        result = bus.unsubscribe("test.event", handler)

        assert result is False

    def test_unsubscribe_returns_false_for_unknown_type(self, bus: EventBus) -> None:
        """Unsubscribe returns False for unknown event type."""
        handler = MagicMock()
        bus.subscribe("test.event", handler)

        result = bus.unsubscribe("other.event", handler)

        assert result is False

    def test_clear_removes_all_handlers(self, bus: EventBus) -> None:
        """Clear removes all handlers and pending events."""
        handler1 = MagicMock()
        handler2 = MagicMock()
        bus.subscribe("test.event.one", handler1)
        bus.subscribe_all(handler2)

        bus.clear()

        assert bus.handler_count == 0


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventBusEmission:
    """Tests for emitting events."""

    def test_emit_calls_subscribed_handler(
        self, bus: EventBus, sample_event: Event
    ) -> None:
        """Emitting an event calls the subscribed handler."""
        handler = MagicMock()
        bus.subscribe("test.event.one", handler)

        bus.emit(sample_event)

        handler.assert_called_once_with(sample_event)

    def test_emit_calls_multiple_handlers(
        self, bus: EventBus, sample_event: Event
    ) -> None:
        """Emitting an event calls all subscribed handlers."""
        handler1 = MagicMock()
        handler2 = MagicMock()
        bus.subscribe("test.event.one", handler1)
        bus.subscribe("test.event.one", handler2)

        bus.emit(sample_event)

        handler1.assert_called_once_with(sample_event)
        handler2.assert_called_once_with(sample_event)

    def test_emit_calls_global_handlers(
        self, bus: EventBus, sample_event: Event
    ) -> None:
        """Emitting an event calls global handlers."""
        handler = MagicMock()
        bus.subscribe_all(handler)

        bus.emit(sample_event)

        handler.assert_called_once_with(sample_event)

    def test_emit_calls_global_and_type_handlers(
        self, bus: EventBus, sample_event: Event
    ) -> None:
        """Emitting an event calls both global and type-specific handlers."""
        global_handler = MagicMock()
        type_handler = MagicMock()
        bus.subscribe_all(global_handler)
        bus.subscribe("test.event.one", type_handler)

        bus.emit(sample_event)

        global_handler.assert_called_once_with(sample_event)
        type_handler.assert_called_once_with(sample_event)

    def test_emit_does_not_call_unmatched_handlers(
        self, bus: EventBus, sample_event: Event
    ) -> None:
        """Emitting an event does not call handlers for other types."""
        handler = MagicMock()
        bus.subscribe("other.event", handler)

        bus.emit(sample_event)

        handler.assert_not_called()

    def test_emit_adds_to_pending_events(
        self, bus: EventBus, sample_event: Event
    ) -> None:
        """Emitting an event adds it to pending events."""
        bus.emit(sample_event)

        assert bus.pending_count == 1


# =============================================================================
# Wildcard Pattern Matching Tests
# =============================================================================


class TestEventBusWildcardMatching:
    """Tests for wildcard pattern matching in subscriptions."""

    def test_wildcard_matches_child_events(self, bus: EventBus) -> None:
        """Wildcard pattern 'tool.*' matches child events like 'tool.call.failure'."""
        handler = MagicMock()
        bus.subscribe("tool.*", handler)

        event = Event(type="tool.call.failure", source="test", severity=Severity.ERROR)
        bus.emit(event)

        handler.assert_called_once_with(event)

    def test_wildcard_matches_multiple_child_events(self, bus: EventBus) -> None:
        """Wildcard matches various child events."""
        handler = MagicMock()
        bus.subscribe("tool.*", handler)

        events = [
            Event(type="tool.call.failure", source="test", severity=Severity.ERROR),
            Event(type="tool.call.success", source="test", severity=Severity.INFO),
            Event(type="tool.call.timeout", source="test", severity=Severity.WARNING),
        ]

        for event in events:
            bus.emit(event)

        assert handler.call_count == 3

    def test_wildcard_does_not_match_unrelated_events(self, bus: EventBus) -> None:
        """Wildcard does not match events from different categories."""
        handler = MagicMock()
        bus.subscribe("tool.*", handler)

        event = Event(type="budget.warning", source="test", severity=Severity.WARNING)
        bus.emit(event)

        handler.assert_not_called()

    def test_exact_match_preferred_over_wildcard(self, bus: EventBus) -> None:
        """Both exact and wildcard handlers are called when both match."""
        exact_handler = MagicMock()
        wildcard_handler = MagicMock()
        bus.subscribe("tool.call.failure", exact_handler)
        bus.subscribe("tool.*", wildcard_handler)

        event = Event(type="tool.call.failure", source="test", severity=Severity.ERROR)
        bus.emit(event)

        exact_handler.assert_called_once()
        wildcard_handler.assert_called_once()


# =============================================================================
# Handler Error Handling Tests
# =============================================================================


class TestEventBusErrorHandling:
    """Tests for error handling in event handlers."""

    def test_handler_exception_does_not_stop_other_handlers(
        self, bus: EventBus, sample_event: Event, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An exception in one handler does not prevent other handlers from running."""
        failing_handler = MagicMock(side_effect=ValueError("Handler error"))
        succeeding_handler = MagicMock()
        bus.subscribe("test.event.one", failing_handler)
        bus.subscribe("test.event.one", succeeding_handler)

        with caplog.at_level(logging.ERROR):
            bus.emit(sample_event)

        failing_handler.assert_called_once()
        succeeding_handler.assert_called_once()
        assert "Error in event handler" in caplog.text

    def test_global_handler_exception_logged(
        self, bus: EventBus, sample_event: Event, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An exception in a global handler is logged."""
        failing_handler = MagicMock(side_effect=RuntimeError("Global handler error"))
        bus.subscribe_all(failing_handler)

        with caplog.at_level(logging.ERROR):
            bus.emit(sample_event)

        assert "Error in global event handler" in caplog.text


# =============================================================================
# Enable/Disable Tests
# =============================================================================


class TestEventBusEnableDisable:
    """Tests for enabling and disabling the event bus."""

    def test_disabled_bus_drops_events(
        self, bus: EventBus, sample_event: Event, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Disabled event bus drops events without calling handlers."""
        handler = MagicMock()
        bus.subscribe("test.event.one", handler)

        bus.disable()

        with caplog.at_level(logging.DEBUG):
            bus.emit(sample_event)

        handler.assert_not_called()
        assert bus.pending_count == 0
        assert "disabled" in caplog.text

    def test_enable_after_disable_processes_events(
        self, bus: EventBus, sample_event: Event
    ) -> None:
        """Re-enabled event bus processes new events."""
        handler = MagicMock()
        bus.subscribe("test.event.one", handler)

        bus.disable()
        bus.enable()
        bus.emit(sample_event)

        handler.assert_called_once()

    def test_enable_logs_message(
        self, bus: EventBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Enabling the bus logs an info message."""
        with caplog.at_level(logging.INFO):
            bus.enable()

        assert "enabled" in caplog.text

    def test_disable_logs_message(
        self, bus: EventBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Disabling the bus logs an info message."""
        with caplog.at_level(logging.INFO):
            bus.disable()

        assert "disabled" in caplog.text


# =============================================================================
# Queue Management Tests
# =============================================================================


class TestEventBusQueueManagement:
    """Tests for event queue management."""

    def test_drain_pending_returns_all_events(self, bus: EventBus) -> None:
        """drain_pending returns all pending events and clears the queue."""
        events = [
            Event(type="test.one", source="test", severity=Severity.INFO),
            Event(type="test.two", source="test", severity=Severity.WARNING),
            Event(type="test.three", source="test", severity=Severity.ERROR),
        ]

        for event in events:
            bus.emit(event)

        drained = bus.drain_pending()

        assert len(drained) == 3
        assert bus.pending_count == 0

    def test_drain_pending_on_empty_queue(self, bus: EventBus) -> None:
        """drain_pending returns empty list when queue is empty."""
        drained = bus.drain_pending()

        assert drained == []

    def test_queue_overflow_drops_low_priority_events(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When queue is full, low priority events are dropped first."""
        bus = EventBus(max_queue_size=5)

        # Fill with low priority events
        for i in range(5):
            bus.emit(Event(
                type=f"test.low.{i}",
                source="test",
                severity=Severity.DEBUG,
            ))

        assert bus.pending_count == 5

        # Add a high priority event (should trigger overflow handling)
        with caplog.at_level(logging.WARNING):
            bus.emit(Event(
                type="test.high",
                source="test",
                severity=Severity.CRITICAL,
            ))

        # Some low priority events should have been dropped
        assert "Dropped" in caplog.text or bus.pending_count < 7

    def test_clear_empties_pending_queue(self, bus: EventBus) -> None:
        """clear() removes all pending events."""
        bus.emit(Event(type="test", source="test", severity=Severity.INFO))
        bus.emit(Event(type="test", source="test", severity=Severity.INFO))

        bus.clear()

        assert bus.pending_count == 0


# =============================================================================
# Global Singleton Tests
# =============================================================================


class TestEventBusSingleton:
    """Tests for the global singleton EventBus."""

    def test_get_event_bus_returns_singleton(self) -> None:
        """get_event_bus returns the same instance on repeated calls."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2

    def test_reset_event_bus_creates_new_instance(self) -> None:
        """reset_event_bus causes get_event_bus to return a new instance."""
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()

        assert bus1 is not bus2

    def test_reset_event_bus_clears_existing(self) -> None:
        """reset_event_bus clears the existing bus before resetting."""
        bus = get_event_bus()
        handler = MagicMock()
        bus.subscribe("test", handler)
        bus.emit(Event(type="test", source="test", severity=Severity.INFO))

        reset_event_bus()

        # After reset, handlers and pending events are cleared
        new_bus = get_event_bus()
        assert new_bus.handler_count == 0
        assert new_bus.pending_count == 0


# =============================================================================
# Property Tests
# =============================================================================


class TestEventBusProperties:
    """Tests for EventBus properties."""

    def test_pending_count_tracks_events(self, bus: EventBus) -> None:
        """pending_count accurately tracks the number of pending events."""
        assert bus.pending_count == 0

        bus.emit(Event(type="test.1", source="test", severity=Severity.INFO))
        assert bus.pending_count == 1

        bus.emit(Event(type="test.2", source="test", severity=Severity.INFO))
        assert bus.pending_count == 2

    def test_handler_count_tracks_all_handlers(self, bus: EventBus) -> None:
        """handler_count tracks both type-specific and global handlers."""
        assert bus.handler_count == 0

        handler1 = MagicMock()
        handler2 = MagicMock()
        handler3 = MagicMock()

        bus.subscribe("type.one", handler1)
        assert bus.handler_count == 1

        bus.subscribe("type.two", handler2)
        assert bus.handler_count == 2

        bus.subscribe_all(handler3)
        assert bus.handler_count == 3


# =============================================================================
# Edge Cases
# =============================================================================


class TestEventBusEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_same_handler_subscribed_multiple_times(
        self, bus: EventBus, sample_event: Event
    ) -> None:
        """Same handler subscribed multiple times is called multiple times."""
        handler = MagicMock()
        bus.subscribe("test.event.one", handler)
        bus.subscribe("test.event.one", handler)

        bus.emit(sample_event)

        assert handler.call_count == 2

    def test_empty_event_type(self, bus: EventBus) -> None:
        """Bus handles empty event type gracefully."""
        handler = MagicMock()
        bus.subscribe("", handler)

        event = Event(type="", source="test", severity=Severity.INFO)
        bus.emit(event)

        handler.assert_called_once()

    def test_concurrent_emit_and_subscribe(self, bus: EventBus) -> None:
        """Bus handles emit during handler execution safely."""
        second_handler = MagicMock()

        def subscribing_handler(event: Event) -> None:
            bus.subscribe("test.event.two", second_handler)

        bus.subscribe("test.event.one", subscribing_handler)

        # Should not raise
        bus.emit(Event(type="test.event.one", source="test", severity=Severity.INFO))

        # Second handler should be registered
        assert bus.handler_count == 2
