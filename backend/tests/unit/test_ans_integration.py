"""Integration tests for the Agent Notification System (ANS).

T033 [US1] Test the full notification flow:
1. EventBus emits a tool.call.failure event
2. Subscriber processes it
3. Accumulator batches/formats it
4. The notification is yielded as OracleStreamChunk(type="system")
"""

import logging
from pathlib import Path
from textwrap import dedent
from typing import Optional
from unittest.mock import MagicMock

import pytest

from backend.src.services.ans.bus import EventBus, reset_event_bus
from backend.src.services.ans.event import Event, EventType, Severity
from backend.src.services.ans.subscriber import (
    InjectionPoint,
    Priority,
    Subscriber,
    SubscriberConfig,
    SubscriberLoader,
)
from backend.src.services.ans.accumulator import Notification, NotificationAccumulator
from backend.src.services.ans.toon_formatter import ToonFormatter
from backend.src.models.oracle import OracleStreamChunk


# =============================================================================
# Fixtures for test setup
# =============================================================================


@pytest.fixture
def event_bus() -> EventBus:
    """Create a fresh event bus for each test."""
    reset_event_bus()
    return EventBus()


@pytest.fixture
def temp_subscribers_dir(tmp_path: Path) -> Path:
    """Create a temporary subscribers directory."""
    subscribers_dir = tmp_path / "subscribers"
    subscribers_dir.mkdir()
    return subscribers_dir


@pytest.fixture
def temp_templates_dir(tmp_path: Path) -> Path:
    """Create a temporary templates directory."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    return templates_dir


@pytest.fixture
def tool_failure_subscriber(temp_subscribers_dir: Path, temp_templates_dir: Path) -> Subscriber:
    """Create a tool_failure subscriber similar to the actual one."""
    # Create the template
    template_content = dedent("""
    {% if events|length == 1 %}
    {% set e = events[0] %}
    tool_fail: {{ e.payload.tool_name }} {{ e.payload.error_type }}{% if e.payload.error_message %} - {{ e.payload.error_message }}{% endif %}
    {% else %}
    tool_fails[{{ events|length }}]{tool,error,message}:
    {% for e in events %}
      {{ e.payload.tool_name }},{{ e.payload.error_type }},{{ e.payload.error_message | default("") }}
    {% endfor %}
    {% endif %}
    """).strip()
    (temp_templates_dir / "tool_failure.toon.j2").write_text(template_content)

    # Create the subscriber TOML
    toml_content = dedent("""
    [subscriber]
    id = "tool_failure"
    name = "Tool Failure Notifications"
    description = "Notifies agent when tool calls fail or timeout"
    version = "1.0.0"

    [events]
    types = ["tool.call.failure", "tool.call.timeout"]
    severity_filter = "warning"

    [batching]
    window_ms = 2000
    max_size = 10
    dedupe_key = "type:payload.tool_name"
    dedupe_window_ms = 5000

    [output]
    priority = "high"
    inject_at = "after_tool"
    template = "tool_failure.toon.j2"
    core = true
    """).strip()
    (temp_subscribers_dir / "tool_failure.toml").write_text(toml_content)

    # Load the subscriber
    loader = SubscriberLoader(
        subscribers_dir=temp_subscribers_dir,
        templates_dir=temp_templates_dir,
    )
    subscribers = loader.load_all()
    return subscribers["tool_failure"]


@pytest.fixture
def toon_formatter(temp_templates_dir: Path) -> ToonFormatter:
    """Create a ToonFormatter with the test templates directory."""
    return ToonFormatter(templates_dir=temp_templates_dir)


@pytest.fixture
def accumulator() -> NotificationAccumulator:
    """Create a fresh accumulator for each test."""
    return NotificationAccumulator()


@pytest.fixture
def tool_failure_event() -> Event:
    """Create a sample tool failure event."""
    return Event(
        type=EventType.TOOL_CALL_FAILURE,
        source="oracle_agent",
        severity=Severity.ERROR,
        payload={
            "tool_name": "read_file",
            "tool_id": "call_123",
            "error_type": "FileNotFoundError",
            "error_message": "File /path/to/missing.txt not found",
        }
    )


# =============================================================================
# T033: Integration test - tool failure → notification → SSE chunk
# =============================================================================


class TestToolFailureNotificationFlow:
    """
    Test the complete notification flow from tool failure to SSE chunk.

    This validates that:
    1. An event bus emits a tool.call.failure event
    2. The subscriber processes it
    3. The accumulator batches/formats it
    4. The notification is yielded as OracleStreamChunk(type="system")
    """

    def test_event_bus_emits_tool_failure(self, event_bus: EventBus, tool_failure_event: Event) -> None:
        """EventBus correctly emits tool.call.failure events."""
        received_events: list[Event] = []

        def handler(event: Event) -> None:
            received_events.append(event)

        event_bus.subscribe(EventType.TOOL_CALL_FAILURE, handler)
        event_bus.emit(tool_failure_event)

        assert len(received_events) == 1
        assert received_events[0].type == EventType.TOOL_CALL_FAILURE
        assert received_events[0].payload["tool_name"] == "read_file"

    def test_subscriber_matches_tool_failure_event(
        self, tool_failure_subscriber: Subscriber, tool_failure_event: Event
    ) -> None:
        """Tool failure subscriber matches tool.call.failure events."""
        assert tool_failure_subscriber.matches_event(EventType.TOOL_CALL_FAILURE) is True
        assert tool_failure_subscriber.matches_event(EventType.TOOL_CALL_TIMEOUT) is True
        assert tool_failure_subscriber.matches_event(EventType.TOOL_CALL_SUCCESS) is False
        assert tool_failure_subscriber.matches_event("budget.warning") is False

    def test_accumulator_creates_notification_for_high_priority(
        self,
        accumulator: NotificationAccumulator,
        tool_failure_subscriber: Subscriber,
        tool_failure_event: Event,
    ) -> None:
        """Accumulator creates notification for high-priority subscriber."""
        accumulator.register_subscriber(tool_failure_subscriber)

        # High priority should accumulate but not immediately return
        # (only CRITICAL bypasses batching)
        notification = accumulator.accumulate(tool_failure_event, tool_failure_subscriber)

        # High priority means it's added to pending, not immediately returned
        # (CRITICAL would return immediately)
        assert notification is None
        assert accumulator.pending_count == 1

    def test_accumulator_drains_after_tool_notifications(
        self,
        accumulator: NotificationAccumulator,
        tool_failure_subscriber: Subscriber,
        tool_failure_event: Event,
    ) -> None:
        """Accumulator correctly drains after_tool injection point notifications."""
        accumulator.register_subscriber(tool_failure_subscriber)
        accumulator.accumulate(tool_failure_event, tool_failure_subscriber)

        # Manually flush to create notifications
        accumulator.flush_all()

        # Drain after_tool notifications (where tool_failure is injected)
        notifications = accumulator.drain_after_tool()

        assert len(notifications) == 1
        assert notifications[0].subscriber_id == "tool_failure"
        assert notifications[0].priority == Priority.HIGH
        assert notifications[0].inject_at == InjectionPoint.AFTER_TOOL
        assert len(notifications[0].events) == 1

    def test_toon_formatter_formats_notification(
        self,
        temp_templates_dir: Path,
        tool_failure_subscriber: Subscriber,
        tool_failure_event: Event,
    ) -> None:
        """ToonFormatter correctly formats tool failure notification."""
        formatter = ToonFormatter(templates_dir=temp_templates_dir)

        # Create a notification with the event
        notification = Notification(
            subscriber_id="tool_failure",
            priority=Priority.HIGH,
            inject_at=InjectionPoint.AFTER_TOOL,
            events=[tool_failure_event],
        )

        content = formatter.format_notification(
            notification,
            tool_failure_subscriber.config.template
        )

        # Check that the formatted content contains expected elements
        assert "tool_fail:" in content
        assert "read_file" in content
        assert "FileNotFoundError" in content
        assert "File /path/to/missing.txt not found" in content

    def test_notification_to_oracle_stream_chunk(
        self,
        temp_templates_dir: Path,
        accumulator: NotificationAccumulator,
        tool_failure_subscriber: Subscriber,
        tool_failure_event: Event,
    ) -> None:
        """Notification is correctly converted to OracleStreamChunk(type='system')."""
        formatter = ToonFormatter(templates_dir=temp_templates_dir)
        accumulator.register_subscriber(tool_failure_subscriber)

        # Simulate the flow
        accumulator.accumulate(tool_failure_event, tool_failure_subscriber)
        accumulator.flush_all()
        notifications = accumulator.drain_after_tool()

        assert len(notifications) == 1
        notification = notifications[0]

        # Format the notification
        content = formatter.format_notification(
            notification,
            tool_failure_subscriber.config.template
        )
        notification.content = content

        # Create the OracleStreamChunk
        chunk = OracleStreamChunk(
            type="system",
            content=notification.content,
            metadata={
                "subscriber_id": notification.subscriber_id,
                "priority": notification.priority.value,
                "event_count": len(notification.events),
            }
        )

        assert chunk.type == "system"
        assert "tool_fail:" in chunk.content
        assert "read_file" in chunk.content
        assert chunk.metadata["subscriber_id"] == "tool_failure"
        assert chunk.metadata["priority"] == "high"
        assert chunk.metadata["event_count"] == 1

    def test_full_flow_event_to_chunk(
        self,
        event_bus: EventBus,
        temp_templates_dir: Path,
        temp_subscribers_dir: Path,
    ) -> None:
        """
        Full integration test: Event emission → Subscriber processing →
        Accumulator batching → ToonFormatter → OracleStreamChunk.
        """
        # Setup: Create subscriber and load it
        template_content = dedent("""
        {% if events|length == 1 %}
        {% set e = events[0] %}
        tool_fail: {{ e.payload.tool_name }} {{ e.payload.error_type }}{% if e.payload.error_message %} - {{ e.payload.error_message }}{% endif %}
        {% else %}
        tool_fails[{{ events|length }}]{tool,error,message}:
        {% for e in events %}
          {{ e.payload.tool_name }},{{ e.payload.error_type }},{{ e.payload.error_message | default("") }}
        {% endfor %}
        {% endif %}
        """).strip()
        (temp_templates_dir / "tool_failure.toon.j2").write_text(template_content)

        toml_content = dedent("""
        [subscriber]
        id = "tool_failure"
        name = "Tool Failure Notifications"

        [events]
        types = ["tool.call.failure", "tool.call.timeout"]

        [output]
        priority = "high"
        inject_at = "after_tool"
        template = "tool_failure.toon.j2"
        core = true
        """).strip()
        (temp_subscribers_dir / "tool_failure.toml").write_text(toml_content)

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()
        subscriber = subscribers["tool_failure"]

        accumulator = NotificationAccumulator()
        accumulator.register_subscriber(subscriber)
        formatter = ToonFormatter(templates_dir=temp_templates_dir)

        # Track chunks produced
        chunks: list[OracleStreamChunk] = []

        # Step 1: Subscribe to events and accumulate
        def handle_event(event: Event) -> None:
            if subscriber.matches_event(event.type):
                accumulator.accumulate(event, subscriber)

        event_bus.subscribe_all(handle_event)

        # Step 2: Emit a tool failure event
        event = Event(
            type=EventType.TOOL_CALL_FAILURE,
            source="oracle_agent",
            severity=Severity.ERROR,
            payload={
                "tool_name": "search_code",
                "tool_id": "call_456",
                "error_type": "TimeoutError",
                "error_message": "Search timed out after 30s",
            }
        )
        event_bus.emit(event)

        # Step 3: Flush and drain notifications
        accumulator.flush_all()
        notifications = accumulator.drain_after_tool()

        # Step 4: Format and convert to chunks
        for notification in notifications:
            content = formatter.format_notification(
                notification,
                subscriber.config.template
            )
            notification.content = content

            chunk = OracleStreamChunk(
                type="system",
                content=notification.content,
                metadata={
                    "subscriber_id": notification.subscriber_id,
                    "priority": notification.priority.value,
                    "event_count": len(notification.events),
                }
            )
            chunks.append(chunk)

        # Verify the full flow produced expected output
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.type == "system"
        assert "tool_fail:" in chunk.content
        assert "search_code" in chunk.content
        assert "TimeoutError" in chunk.content
        assert "Search timed out after 30s" in chunk.content


class TestMultipleEventBatching:
    """Test that multiple events are properly batched into a single notification."""

    def test_multiple_failures_batched(
        self,
        temp_templates_dir: Path,
        temp_subscribers_dir: Path,
    ) -> None:
        """Multiple tool failures are batched into a single notification."""
        # Setup template and subscriber
        template_content = dedent("""
        {% if events|length == 1 %}
        {% set e = events[0] %}
        tool_fail: {{ e.payload.tool_name }} {{ e.payload.error_type }}
        {% else %}
        tool_fails[{{ events|length }}]{tool,error,message}:
        {% for e in events %}
          {{ e.payload.tool_name }},{{ e.payload.error_type }},{{ e.payload.error_message | default("") }}
        {% endfor %}
        {% endif %}
        """).strip()
        (temp_templates_dir / "tool_failure.toon.j2").write_text(template_content)

        toml_content = dedent("""
        [subscriber]
        id = "tool_failure"
        name = "Tool Failure Notifications"

        [events]
        types = ["tool.call.failure"]

        [batching]
        window_ms = 2000
        max_size = 10

        [output]
        priority = "high"
        inject_at = "after_tool"
        template = "tool_failure.toon.j2"
        """).strip()
        (temp_subscribers_dir / "tool_failure.toml").write_text(toml_content)

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()
        subscriber = subscribers["tool_failure"]

        accumulator = NotificationAccumulator()
        accumulator.register_subscriber(subscriber)
        formatter = ToonFormatter(templates_dir=temp_templates_dir)

        # Accumulate multiple different events with unique dedupe_keys
        # Each event needs a unique dedupe_key to avoid being deduplicated
        events = [
            Event(
                type=EventType.TOOL_CALL_FAILURE,
                source="oracle_agent",
                severity=Severity.ERROR,
                payload={"tool_name": "read_file", "error_type": "FileNotFoundError", "error_message": "Missing"},
                dedupe_key="tool.call.failure:read_file:1",
            ),
            Event(
                type=EventType.TOOL_CALL_FAILURE,
                source="oracle_agent",
                severity=Severity.ERROR,
                payload={"tool_name": "write_file", "error_type": "PermissionError", "error_message": "Read-only"},
                dedupe_key="tool.call.failure:write_file:2",
            ),
            Event(
                type=EventType.TOOL_CALL_FAILURE,
                source="oracle_agent",
                severity=Severity.ERROR,
                payload={"tool_name": "list_dir", "error_type": "NotADirectoryError", "error_message": "Not a dir"},
                dedupe_key="tool.call.failure:list_dir:3",
            ),
        ]

        for event in events:
            accumulator.accumulate(event, subscriber)

        # Flush and drain
        accumulator.flush_all()
        notifications = accumulator.drain_after_tool()

        assert len(notifications) == 1
        notification = notifications[0]
        assert len(notification.events) == 3

        # Format and verify batched output
        content = formatter.format_notification(notification, subscriber.config.template)

        assert "tool_fails[3]" in content
        assert "read_file" in content
        assert "write_file" in content
        assert "list_dir" in content


class TestDisabledSubscriber:
    """Test that disabled subscribers don't generate notifications."""

    def test_disabled_subscriber_no_notification(
        self,
        temp_templates_dir: Path,
        temp_subscribers_dir: Path,
    ) -> None:
        """Disabled subscribers don't generate notifications."""
        # Setup template and subscriber
        (temp_templates_dir / "tool_failure.toon.j2").write_text("tool_fail: {{ event.payload.tool_name }}")

        toml_content = dedent("""
        [subscriber]
        id = "tool_failure"
        name = "Tool Failure"

        [events]
        types = ["tool.call.failure"]

        [output]
        priority = "high"
        inject_at = "after_tool"
        template = "tool_failure.toon.j2"
        """).strip()
        (temp_subscribers_dir / "tool_failure.toml").write_text(toml_content)

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()
        subscriber = subscribers["tool_failure"]

        # Create accumulator with subscriber disabled
        accumulator = NotificationAccumulator(disabled_subscribers={"tool_failure"})
        accumulator.register_subscriber(subscriber)

        # Try to accumulate an event
        event = Event(
            type=EventType.TOOL_CALL_FAILURE,
            source="oracle_agent",
            severity=Severity.ERROR,
            payload={"tool_name": "test", "error_type": "Error"},
        )
        result = accumulator.accumulate(event, subscriber)

        assert result is None
        assert accumulator.pending_count == 0

        # Flush and drain should also be empty
        accumulator.flush_all()
        notifications = accumulator.drain_after_tool()
        assert len(notifications) == 0


class TestCriticalPriorityBypassesBatching:
    """Test that critical priority events bypass batching."""

    def test_critical_priority_immediate_notification(
        self,
        temp_templates_dir: Path,
        temp_subscribers_dir: Path,
    ) -> None:
        """Critical priority events generate immediate notifications."""
        (temp_templates_dir / "critical.toon.j2").write_text(
            "CRITICAL: {{ event.payload.message }}"
        )

        toml_content = dedent("""
        [subscriber]
        id = "critical_sub"
        name = "Critical Subscriber"

        [events]
        types = ["budget.token.exceeded"]

        [output]
        priority = "critical"
        inject_at = "immediate"
        template = "critical.toon.j2"
        core = true
        """).strip()
        (temp_subscribers_dir / "critical.toml").write_text(toml_content)

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()
        subscriber = subscribers["critical_sub"]

        accumulator = NotificationAccumulator()
        accumulator.register_subscriber(subscriber)

        event = Event(
            type=EventType.BUDGET_TOKEN_EXCEEDED,
            source="oracle_agent",
            severity=Severity.ERROR,
            payload={"message": "Token budget exceeded!"},
        )

        # Critical priority should return notification immediately
        notification = accumulator.accumulate(event, subscriber)

        assert notification is not None
        assert notification.priority == Priority.CRITICAL
        assert notification.inject_at == InjectionPoint.IMMEDIATE
        assert len(notification.events) == 1


class TestFallbackFormatting:
    """Test fallback formatting when template is missing or fails."""

    def test_missing_template_uses_fallback(
        self,
        temp_templates_dir: Path,
    ) -> None:
        """Missing template falls back to simple format."""
        formatter = ToonFormatter(templates_dir=temp_templates_dir)

        event = Event(
            type=EventType.TOOL_CALL_FAILURE,
            source="oracle_agent",
            severity=Severity.ERROR,
            payload={"tool_name": "read_file", "error_type": "Error"},
        )

        notification = Notification(
            subscriber_id="test",
            priority=Priority.HIGH,
            inject_at=InjectionPoint.AFTER_TOOL,
            events=[event],
        )

        # Try to format with non-existent template
        content = formatter.format_notification(notification, "nonexistent.j2")

        # Should use fallback format
        assert "tool.call.failure" in content
        assert "tool_name" in content or "read_file" in content


class TestStreamChunkMetadata:
    """Test that OracleStreamChunk contains proper metadata."""

    def test_chunk_has_required_fields(self) -> None:
        """OracleStreamChunk has all required fields for system type."""
        chunk = OracleStreamChunk(
            type="system",
            content="Test notification content",
            metadata={
                "subscriber_id": "tool_failure",
                "priority": "high",
                "event_count": 1,
            }
        )

        assert chunk.type == "system"
        assert chunk.content == "Test notification content"
        assert chunk.metadata is not None
        assert chunk.metadata["subscriber_id"] == "tool_failure"
        assert chunk.metadata["priority"] == "high"
        assert chunk.metadata["event_count"] == 1

    def test_chunk_serialization(self) -> None:
        """OracleStreamChunk can be serialized to dict/JSON."""
        chunk = OracleStreamChunk(
            type="system",
            content="Tool failure notification",
            metadata={"subscriber_id": "tool_failure"},
        )

        # Convert to dict (for JSON serialization)
        chunk_dict = chunk.model_dump()

        assert chunk_dict["type"] == "system"
        assert chunk_dict["content"] == "Tool failure notification"
        assert chunk_dict["metadata"]["subscriber_id"] == "tool_failure"


# =============================================================================
# T080: Verify <100ms event processing goal
# =============================================================================


import time


class TestPerformanceGoals:
    """
    Performance tests for ANS event processing.

    Requirement: Event processing should complete in under 100ms to avoid
    blocking the agent loop.
    """

    def _make_config(
        self,
        id: str,
        name: str,
        event_types: list[str],
        priority: Priority = Priority.NORMAL,
        inject_at: InjectionPoint = InjectionPoint.AFTER_TOOL,
    ) -> SubscriberConfig:
        """Helper to create a SubscriberConfig with required fields."""
        return SubscriberConfig(
            id=id,
            name=name,
            description=f"Test subscriber: {name}",
            version="1.0.0",
            event_types=event_types,
            template="test.toon.j2",
            priority=priority,
            inject_at=inject_at,
        )

    @pytest.fixture
    def performance_subscriber(self) -> Subscriber:
        """Create a subscriber for performance testing."""
        config = self._make_config(
            id="perf_test",
            name="Performance Test",
            event_types=["tool.call.failure", "tool.call.timeout", "budget.*"],
        )
        return Subscriber(config=config)

    @pytest.fixture
    def performance_accumulator(self, performance_subscriber: Subscriber) -> NotificationAccumulator:
        """Create an accumulator with multiple subscribers for realistic load."""
        acc = NotificationAccumulator()

        # Register the primary subscriber
        acc.register_subscriber(performance_subscriber)

        # Add more subscribers to simulate realistic load
        for i in range(10):
            config = self._make_config(
                id=f"extra_sub_{i}",
                name=f"Extra Subscriber {i}",
                event_types=["tool.*", "budget.*"],
            )
            sub = Subscriber(config=config)
            acc.register_subscriber(sub)

        return acc

    def test_single_event_accumulation_under_100ms(
        self,
        performance_accumulator: NotificationAccumulator,
        performance_subscriber: Subscriber,
    ) -> None:
        """Single event accumulation completes in under 100ms.

        This is the core performance requirement: processing a single event
        through the accumulator must be fast enough to not block the agent.
        """
        event = Event(
            type=EventType.TOOL_CALL_FAILURE,
            source="oracle_agent",
            severity=Severity.ERROR,
            payload={
                "tool_name": "bash",
                "error": "Command failed",
                "duration_ms": 500,
                "output": "A" * 1000,  # 1KB of output
            },
        )

        # Measure single event accumulation time
        start_time = time.perf_counter()
        performance_accumulator.accumulate(event, performance_subscriber)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 100, (
            f"Single event accumulation took {elapsed_ms:.2f}ms, "
            f"exceeding the 100ms performance goal"
        )

    def test_batch_of_10_events_under_100ms(
        self,
        performance_accumulator: NotificationAccumulator,
        performance_subscriber: Subscriber,
    ) -> None:
        """Batch of 10 events processes in under 100ms."""
        events = [
            Event(
                type=EventType.TOOL_CALL_FAILURE,
                source="oracle_agent",
                severity=Severity.ERROR,
                payload={
                    "tool_name": f"tool_{i}",
                    "error": f"Error message {i}",
                    "output": "x" * 500,
                },
                dedupe_key=f"unique_key_{i}",  # Unique keys to avoid deduplication
            )
            for i in range(10)
        ]

        start_time = time.perf_counter()
        for event in events:
            performance_accumulator.accumulate(event, performance_subscriber)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 100, (
            f"Batch of 10 events took {elapsed_ms:.2f}ms, "
            f"exceeding the 100ms performance goal"
        )

    def test_drain_operation_under_100ms(
        self,
        performance_accumulator: NotificationAccumulator,
        performance_subscriber: Subscriber,
    ) -> None:
        """Drain operation completes in under 100ms even with many notifications."""
        # Create many pending events
        for i in range(50):
            event = Event(
                type=EventType.TOOL_CALL_FAILURE,
                source="test",
                severity=Severity.WARNING,
                payload={"index": i},
                dedupe_key=f"drain_test_{i}",
            )
            performance_accumulator.accumulate(event, performance_subscriber)

        # Flush to create notifications
        performance_accumulator.flush_all()

        # Measure drain time
        start_time = time.perf_counter()
        notifications = performance_accumulator.drain_after_tool()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 100, (
            f"Drain operation took {elapsed_ms:.2f}ms, "
            f"exceeding the 100ms performance goal"
        )
        assert len(notifications) > 0  # Verify we actually drained something

    def test_event_bus_emit_under_100ms(self) -> None:
        """Event bus emit operation completes in under 100ms with many handlers."""
        reset_event_bus()
        bus = EventBus(max_queue_size=1000)

        handler_call_count = 0

        def handler(event: Event) -> None:
            nonlocal handler_call_count
            handler_call_count += 1

        # Register many handlers to simulate realistic load
        for event_type in ["tool.*", "budget.*", "agent.*"]:
            for _ in range(5):
                bus.subscribe(event_type, handler)

        # Also add global handlers
        for _ in range(3):
            bus.subscribe_all(handler)

        event = Event(
            type=EventType.TOOL_CALL_FAILURE,
            source="test",
            severity=Severity.ERROR,
            payload={"tool_name": "test_tool", "error": "test error"},
        )

        start_time = time.perf_counter()
        bus.emit(event)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 100, (
            f"Event bus emit took {elapsed_ms:.2f}ms, "
            f"exceeding the 100ms performance goal"
        )
        assert handler_call_count > 0  # Verify handlers were called

    def test_formatter_fallback_under_100ms(self, temp_templates_dir: Path) -> None:
        """ToonFormatter fallback operation completes in under 100ms."""
        formatter = ToonFormatter(templates_dir=temp_templates_dir)

        # Create notification with many events
        events = [
            Event(
                type=EventType.TOOL_CALL_FAILURE,
                source="test",
                severity=Severity.ERROR,
                payload={
                    "tool_name": f"tool_{i}",
                    "error": f"Error message that is quite long: {i}" * 10,
                },
            )
            for i in range(20)
        ]

        notification = Notification(
            subscriber_id="test_sub",
            priority=Priority.NORMAL,
            inject_at=InjectionPoint.AFTER_TOOL,
            events=events,
        )

        start_time = time.perf_counter()
        formatted = formatter._fallback_format(notification)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 100, (
            f"Formatter fallback took {elapsed_ms:.2f}ms, "
            f"exceeding the 100ms performance goal"
        )
        assert len(formatted) > 0

    def test_full_pipeline_under_100ms(
        self,
        temp_templates_dir: Path,
        performance_subscriber: Subscriber,
    ) -> None:
        """Full notification pipeline (emit -> accumulate -> drain -> format) under 100ms."""
        reset_event_bus()
        bus = EventBus()
        accumulator = NotificationAccumulator()
        accumulator.register_subscriber(performance_subscriber)
        formatter = ToonFormatter(templates_dir=temp_templates_dir)

        # Wire up the event handler
        def handle_event(event: Event) -> None:
            if performance_subscriber.matches_event(event.type):
                accumulator.accumulate(event, performance_subscriber)

        bus.subscribe("tool.*", handle_event)

        event = Event(
            type=EventType.TOOL_CALL_FAILURE,
            source="oracle_agent",
            severity=Severity.ERROR,
            payload={
                "tool_name": "bash",
                "error": "Command failed with exit code 1",
                "output": "stderr output here" * 20,
            },
        )

        # Measure the full pipeline
        start_time = time.perf_counter()

        # Step 1: Emit event
        bus.emit(event)

        # Step 2: Flush and drain
        accumulator.flush_all()
        notifications = accumulator.drain_after_tool()

        # Step 3: Format
        for notification in notifications:
            formatter._fallback_format(notification)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 100, (
            f"Full pipeline took {elapsed_ms:.2f}ms, "
            f"exceeding the 100ms performance goal"
        )


class TestEventBusOverflowHandling:
    """Tests for event bus queue overflow handling (T077 verification)."""

    def test_queue_overflow_drops_low_priority(self) -> None:
        """When queue overflows, low priority events are dropped first."""
        bus = EventBus(max_queue_size=10)

        # Fill with low priority events
        for i in range(15):
            event = Event(
                type="test.event",
                source="test",
                severity=Severity.DEBUG if i < 10 else Severity.ERROR,
                payload={"index": i},
            )
            bus.emit(event)

        # Queue should have been trimmed
        assert bus.pending_count <= 10

    def test_critical_events_protected_during_overflow(self) -> None:
        """Critical and error events are never dropped during overflow."""
        bus = EventBus(max_queue_size=10)

        # Add critical events first
        critical_events = [
            Event(
                type="critical.event",
                source="test",
                severity=Severity.CRITICAL,
                payload={"id": f"critical_{i}"},
            )
            for i in range(5)
        ]

        for event in critical_events:
            bus.emit(event)

        # Fill rest with low priority
        for i in range(20):
            bus.emit(
                Event(
                    type="low.event",
                    source="test",
                    severity=Severity.DEBUG,
                    payload={"id": f"low_{i}"},
                )
            )

        # Drain and verify critical events are present
        pending = bus.drain_pending()
        critical_in_pending = [e for e in pending if e.severity == Severity.CRITICAL]

        assert len(critical_in_pending) == 5, "All critical events should be preserved"

    def test_error_events_protected_during_overflow(self) -> None:
        """Error severity events are also protected during overflow."""
        bus = EventBus(max_queue_size=10)

        # Add error events
        for i in range(3):
            bus.emit(
                Event(
                    type="error.event",
                    source="test",
                    severity=Severity.ERROR,
                    payload={"id": f"error_{i}"},
                )
            )

        # Fill with many low priority
        for i in range(30):
            bus.emit(
                Event(
                    type="info.event",
                    source="test",
                    severity=Severity.INFO,
                    payload={"id": f"info_{i}"},
                )
            )

        # Drain and verify error events are present
        pending = bus.drain_pending()
        error_in_pending = [e for e in pending if e.severity == Severity.ERROR]

        assert len(error_in_pending) == 3, "All error events should be preserved"
