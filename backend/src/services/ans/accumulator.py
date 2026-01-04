"""Notification accumulation and batching for the Agent Notification System."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from .event import Event, Severity
from .subscriber import InjectionPoint, Priority, Subscriber


logger = logging.getLogger(__name__)


def _timed_operation(operation_name: str):
    """Decorator to time operations and log at debug level."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"ANS timing: {operation_name} completed in {elapsed_ms:.2f}ms"
            )
            return result
        return wrapper
    return decorator


@dataclass
class Notification:
    """A formatted notification ready for injection."""

    id: UUID = field(default_factory=uuid4)
    subscriber_id: str = ""
    content: str = ""
    priority: Priority = Priority.NORMAL
    inject_at: InjectionPoint = InjectionPoint.AFTER_TOOL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    events: list[Event] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "subscriber_id": self.subscriber_id,
            "content": self.content,
            "priority": self.priority.value,
            "inject_at": self.inject_at.value,
            "timestamp": self.timestamp.isoformat(),
            "event_count": len(self.events),
        }


class NotificationAccumulator:
    """Accumulates and batches events into notifications."""

    def __init__(self, disabled_subscribers: Optional[set[str]] = None):
        """Initialize the accumulator.

        Args:
            disabled_subscribers: Set of subscriber IDs that are disabled.
        """
        self.disabled_subscribers = disabled_subscribers or set()

        # Pending events by subscriber ID
        self._pending: dict[str, list[Event]] = defaultdict(list)

        # Pending notifications by injection point
        self._notifications: dict[InjectionPoint, list[Notification]] = {
            point: [] for point in InjectionPoint
        }

        # Deduplication tracking: {dedupe_key: last_seen_timestamp}
        self._seen_dedupe_keys: dict[str, datetime] = {}

        # Subscribers registry
        self._subscribers: dict[str, Subscriber] = {}

    def register_subscriber(self, subscriber: Subscriber) -> None:
        """Register a subscriber with the accumulator."""
        self._subscribers[subscriber.id] = subscriber

    def register_subscribers(self, subscribers: list[Subscriber]) -> None:
        """Register multiple subscribers."""
        for sub in subscribers:
            self.register_subscriber(sub)

    def set_disabled_subscribers(self, disabled_subscribers: set[str] | list[str]) -> None:
        """Set the disabled subscribers.

        Args:
            disabled_subscribers: Set or list of subscriber IDs that are disabled.
        """
        if isinstance(disabled_subscribers, list):
            self.disabled_subscribers = set(disabled_subscribers)
        else:
            self.disabled_subscribers = disabled_subscribers

    def is_subscriber_enabled(self, subscriber_id: str) -> bool:
        """Check if a subscriber is enabled."""
        if subscriber_id in self.disabled_subscribers:
            return False
        subscriber = self._subscribers.get(subscriber_id)
        if subscriber:
            return subscriber.enabled
        return True

    def accumulate(self, event: Event, subscriber: Subscriber) -> Optional[Notification]:
        """Accumulate an event for a subscriber.

        Returns notification immediately if batching is disabled or priority is critical.
        """
        start_time = time.perf_counter()

        if not self.is_subscriber_enabled(subscriber.id):
            logger.debug(f"Subscriber {subscriber.id} is disabled, skipping event")
            return None

        # Check deduplication
        if not self._should_process_event(event, subscriber):
            logger.debug(f"Event {event.type} deduplicated for {subscriber.id}")
            return None

        # Critical priority bypasses batching
        if subscriber.priority == Priority.CRITICAL:
            notification = self._create_notification(subscriber, [event])
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"ANS timing: accumulate (critical) for {subscriber.id} "
                f"completed in {elapsed_ms:.2f}ms"
            )
            return notification

        # Add to pending batch
        self._pending[subscriber.id].append(event)

        # Check if we should flush (batch size reached)
        batching = subscriber.config.batching
        if len(self._pending[subscriber.id]) >= batching.max_size:
            notification = self._flush_subscriber(subscriber)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"ANS timing: accumulate (batch flush) for {subscriber.id} "
                f"completed in {elapsed_ms:.2f}ms"
            )
            return notification

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"ANS timing: accumulate (pending) for {subscriber.id} "
            f"completed in {elapsed_ms:.2f}ms"
        )
        return None

    def _should_process_event(self, event: Event, subscriber: Subscriber) -> bool:
        """Check if event should be processed (not a duplicate)."""
        batching = subscriber.config.batching

        if batching.dedupe_window_ms <= 0:
            return True  # No deduplication

        dedupe_key = self._get_dedupe_key(event, subscriber)
        now = datetime.now(timezone.utc)

        last_seen = self._seen_dedupe_keys.get(dedupe_key)
        if last_seen is not None:
            elapsed_ms = (now - last_seen).total_seconds() * 1000
            if elapsed_ms < batching.dedupe_window_ms:
                return False

        self._seen_dedupe_keys[dedupe_key] = now
        return True

    def _get_dedupe_key(self, event: Event, subscriber: Subscriber) -> str:
        """Generate deduplication key for an event."""
        dedupe_pattern = subscriber.config.batching.dedupe_key

        if not dedupe_pattern:
            return f"{subscriber.id}:{event.dedupe_key}"

        # Parse pattern like "type:payload.tool_name"
        parts = []
        for part in dedupe_pattern.split(":"):
            if part == "type":
                parts.append(event.type)
            elif part.startswith("payload."):
                key = part[8:]  # Remove "payload." prefix
                value = event.payload.get(key, "")
                parts.append(str(value))
            else:
                parts.append(part)

        return f"{subscriber.id}:" + ":".join(parts)

    def _flush_subscriber(self, subscriber: Subscriber) -> Optional[Notification]:
        """Flush pending events for a subscriber into a notification."""
        events = self._pending.pop(subscriber.id, [])

        if not events:
            return None

        return self._create_notification(subscriber, events)

    def _create_notification(
        self, subscriber: Subscriber, events: list[Event]
    ) -> Notification:
        """Create a notification from events."""
        notification = Notification(
            subscriber_id=subscriber.id,
            priority=subscriber.priority,
            inject_at=subscriber.inject_at,
            events=events,
            content="",  # Content will be set by ToonFormatter
        )

        # Queue for the appropriate injection point
        self._notifications[subscriber.inject_at].append(notification)

        logger.debug(
            f"Created notification for {subscriber.id} with {len(events)} events, "
            f"inject_at={subscriber.inject_at.value}"
        )

        return notification

    def flush_all(self) -> list[Notification]:
        """Flush all pending events and return all notifications."""
        notifications = []

        for subscriber_id, events in list(self._pending.items()):
            if events:
                subscriber = self._subscribers.get(subscriber_id)
                if subscriber:
                    notification = self._create_notification(subscriber, events)
                    notifications.append(notification)

        self._pending.clear()
        return notifications

    def drain(self, injection_point: InjectionPoint) -> list[Notification]:
        """Get and clear notifications for an injection point.

        Args:
            injection_point: The injection point to drain.

        Returns:
            List of notifications, sorted by priority (critical first).
        """
        start_time = time.perf_counter()

        notifications = self._notifications[injection_point]
        self._notifications[injection_point] = []

        # Sort by priority (critical > high > normal > low)
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.NORMAL: 2,
            Priority.LOW: 3,
        }

        notifications.sort(key=lambda n: priority_order.get(n.priority, 99))

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if notifications:
            logger.debug(
                f"ANS timing: drain({injection_point.value}) returned "
                f"{len(notifications)} notifications in {elapsed_ms:.2f}ms"
            )

        return notifications

    def drain_immediate(self) -> list[Notification]:
        """Drain immediate (critical) notifications."""
        return self.drain(InjectionPoint.IMMEDIATE)

    def drain_turn_start(self) -> list[Notification]:
        """Drain turn_start notifications."""
        return self.drain(InjectionPoint.TURN_START)

    def drain_after_tool(self) -> list[Notification]:
        """Drain after_tool notifications."""
        return self.drain(InjectionPoint.AFTER_TOOL)

    def drain_turn_end(self) -> list[Notification]:
        """Drain turn_end notifications."""
        return self.drain(InjectionPoint.TURN_END)

    def clear(self) -> None:
        """Clear all pending events and notifications."""
        self._pending.clear()
        for point in InjectionPoint:
            self._notifications[point].clear()
        self._seen_dedupe_keys.clear()

    @property
    def pending_count(self) -> int:
        """Get total pending event count."""
        return sum(len(events) for events in self._pending.values())

    @property
    def notification_count(self) -> int:
        """Get total pending notification count."""
        return sum(len(notifs) for notifs in self._notifications.values())
