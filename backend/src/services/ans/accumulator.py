"""Notification accumulation and batching for the Agent Notification System."""

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional
from uuid import UUID, uuid4

from .event import Event, Severity
from .subscriber import InjectionPoint, Priority, Subscriber
from .deferred import (
    DeferredDeliveryQueue,
    DeferredNotification,
    DeliveryContext,
    DeliveryTrigger,
    ConditionPredicate,
    context_above_threshold,
    tool_completed,
    message_count_above,
    token_usage_above,
    get_deferred_queue,
)


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
        """Create a notification from events.

        Event payloads can override subscriber defaults for:
        - inject_at: "immediate", "turn_start", "after_tool", "turn_end"
        - priority: "low", "normal", "high", "critical"

        This allows tools like notify_self to specify delivery timing.
        """
        # Start with subscriber defaults
        priority = subscriber.priority
        inject_at = subscriber.inject_at

        # Check if any event has payload overrides
        for event in events:
            if event.payload:
                # Priority override from payload
                if "priority" in event.payload:
                    payload_priority = event.payload["priority"]
                    priority_map = {
                        "low": Priority.LOW,
                        "normal": Priority.NORMAL,
                        "high": Priority.HIGH,
                        "critical": Priority.CRITICAL,
                    }
                    if payload_priority in priority_map:
                        priority = priority_map[payload_priority]

                # Injection point override from payload
                if "inject_at" in event.payload:
                    payload_inject_at = event.payload["inject_at"]
                    inject_at_map = {
                        "immediate": InjectionPoint.IMMEDIATE,
                        "turn_start": InjectionPoint.TURN_START,
                        "after_tool": InjectionPoint.AFTER_TOOL,
                        "turn_end": InjectionPoint.TURN_END,
                    }
                    if payload_inject_at in inject_at_map:
                        inject_at = inject_at_map[payload_inject_at]

                # Only use first event's overrides (for batched events)
                break

        notification = Notification(
            subscriber_id=subscriber.id,
            priority=priority,
            inject_at=inject_at,
            events=events,
            content="",  # Content will be set by ToonFormatter
        )

        # Queue for the appropriate injection point
        self._notifications[inject_at].append(notification)

        logger.debug(
            f"Created notification for {subscriber.id} with {len(events)} events, "
            f"inject_at={inject_at.value}, priority={priority.value}"
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

    # =========================================================================
    # Deferred Delivery Methods
    # =========================================================================

    def defer(
        self,
        event: Optional[Event] = None,
        subscriber_id: str = "",
        content: str = "",
        priority: Priority = Priority.NORMAL,
        trigger: str = "next_turn",
        turns: int = 1,
        tool_name: Optional[str] = None,
        condition: Optional[str] = None,
    ) -> DeferredNotification:
        """Queue an event for deferred delivery.

        This adds a notification to the deferred queue for delivery at
        a later point, based on the specified trigger.

        Args:
            event: The original event (optional)
            subscriber_id: ID of the subscriber
            content: Pre-formatted notification content
            priority: Notification priority
            trigger: Delivery trigger type - one of:
                     "next_turn", "after_n_turns", "after_tool", "on_condition"
            turns: Number of turns to wait (for next_turn/after_n_turns)
            tool_name: Tool to wait for (for after_tool)
            condition: Condition expression string (for on_condition)
                      Supported formats:
                      - "context_above_threshold(0.8)"
                      - "tool_completed(search_code)"
                      - "message_count_above(10)"
                      - "token_usage_above(0.7)"

        Returns:
            The created DeferredNotification
        """
        # Parse trigger type
        trigger_enum = DeliveryTrigger.NEXT_TURN
        if trigger == "next_turn":
            trigger_enum = DeliveryTrigger.NEXT_TURN
            turns = 1  # Always 1 for next_turn
        elif trigger == "after_n_turns":
            trigger_enum = DeliveryTrigger.AFTER_N_TURNS
        elif trigger == "after_tool":
            trigger_enum = DeliveryTrigger.AFTER_TOOL
        elif trigger == "on_condition":
            trigger_enum = DeliveryTrigger.ON_CONDITION

        # Parse condition expression if provided
        condition_predicate: Optional[ConditionPredicate] = None
        if condition and trigger_enum == DeliveryTrigger.ON_CONDITION:
            condition_predicate = self._parse_condition(condition)

        # Get the deferred queue and enqueue
        queue = get_deferred_queue()
        return queue.enqueue(
            event=event,
            subscriber_id=subscriber_id,
            content=content,
            priority=priority,
            trigger=trigger_enum,
            turns=turns,
            tool_name=tool_name,
            condition=condition_predicate,
            condition_expr=condition,
        )

    def _parse_condition(self, condition_expr: str) -> Optional[ConditionPredicate]:
        """Parse a condition expression string into a predicate function.

        Supported expression formats:
        - "context_above_threshold(0.8)" - Fire when context usage >= 80%
        - "tool_completed(search_code)" - Fire when search_code tool completes
        - "message_count_above(10)" - Fire when message count >= 10
        - "token_usage_above(0.7)" - Fire when token budget usage >= 70%

        Args:
            condition_expr: String expression to parse

        Returns:
            Predicate function or None if parsing fails
        """
        if not condition_expr:
            return None

        # Parse function-style expressions: func_name(arg)
        match = re.match(r'(\w+)\(([^)]*)\)', condition_expr.strip())
        if not match:
            logger.warning(f"Unable to parse condition expression: {condition_expr}")
            return None

        func_name = match.group(1)
        arg_str = match.group(2).strip()

        try:
            if func_name == "context_above_threshold":
                threshold = float(arg_str)
                return context_above_threshold(threshold)

            elif func_name == "tool_completed":
                # Remove quotes if present
                tool_name = arg_str.strip('"\'')
                return tool_completed(tool_name)

            elif func_name == "message_count_above":
                count = int(arg_str)
                return message_count_above(count)

            elif func_name == "token_usage_above":
                threshold = float(arg_str)
                return token_usage_above(threshold)

            else:
                logger.warning(f"Unknown condition function: {func_name}")
                return None

        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing condition argument: {condition_expr}: {e}")
            return None

    def drain_deferred_turn_start(self) -> List[Notification]:
        """Drain deferred notifications ready at turn start.

        This should be called at the start of each agent turn. It processes
        NEXT_TURN and AFTER_N_TURNS notifications that are now ready.

        Returns:
            List of Notification objects ready for delivery
        """
        queue = get_deferred_queue()
        deferred_list = queue.on_turn_start()

        # Convert DeferredNotification to Notification
        notifications: List[Notification] = []
        for deferred in deferred_list:
            notification = Notification(
                id=deferred.id,
                subscriber_id=deferred.subscriber_id,
                content=deferred.content,
                priority=deferred.priority,
                inject_at=InjectionPoint.TURN_START,
                timestamp=deferred.created_at,
                events=[deferred.event] if deferred.event else [],
            )
            notifications.append(notification)
            logger.debug(
                f"Converted deferred notification to turn_start notification: {deferred.id}"
            )

        return notifications

    def drain_deferred_after_tool(self, tool_name: str, result: Optional[str] = None) -> List[Notification]:
        """Drain deferred notifications ready after a specific tool completes.

        This should be called after each tool execution. It processes
        AFTER_TOOL notifications waiting for this specific tool.

        Args:
            tool_name: Name of the tool that just completed
            result: Result of the tool execution (optional)

        Returns:
            List of Notification objects ready for delivery
        """
        queue = get_deferred_queue()
        deferred_list = queue.on_tool_complete(tool_name, result)

        # Convert DeferredNotification to Notification
        notifications: List[Notification] = []
        for deferred in deferred_list:
            notification = Notification(
                id=deferred.id,
                subscriber_id=deferred.subscriber_id,
                content=deferred.content,
                priority=deferred.priority,
                inject_at=InjectionPoint.AFTER_TOOL,
                timestamp=deferred.created_at,
                events=[deferred.event] if deferred.event else [],
            )
            notifications.append(notification)
            logger.debug(
                f"Converted deferred notification to after_tool notification: {deferred.id}"
            )

        # Also check condition-based notifications
        condition_list = queue.check_conditions()
        for deferred in condition_list:
            notification = Notification(
                id=deferred.id,
                subscriber_id=deferred.subscriber_id,
                content=deferred.content,
                priority=deferred.priority,
                inject_at=InjectionPoint.AFTER_TOOL,
                timestamp=deferred.created_at,
                events=[deferred.event] if deferred.event else [],
            )
            notifications.append(notification)
            logger.debug(
                f"Converted condition-based notification to after_tool: {deferred.id}"
            )

        return notifications

    def drain_deferred_turn_end(self) -> List[Notification]:
        """Drain deferred notifications ready at turn end.

        This is called at the end of each agent turn.

        Returns:
            List of Notification objects ready for delivery
        """
        queue = get_deferred_queue()
        deferred_list = queue.on_turn_end()

        # Convert DeferredNotification to Notification
        notifications: List[Notification] = []
        for deferred in deferred_list:
            notification = Notification(
                id=deferred.id,
                subscriber_id=deferred.subscriber_id,
                content=deferred.content,
                priority=deferred.priority,
                inject_at=InjectionPoint.TURN_END,
                timestamp=deferred.created_at,
                events=[deferred.event] if deferred.event else [],
            )
            notifications.append(notification)
            logger.debug(
                f"Converted deferred notification to turn_end notification: {deferred.id}"
            )

        return notifications

    def update_deferred_context(
        self,
        turn_number: Optional[int] = None,
        total_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        context_tokens: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
        message_count: Optional[int] = None,
    ) -> None:
        """Update the deferred queue's delivery context.

        Call this to update state information used by condition predicates.

        Args:
            turn_number: Current turn number
            total_tokens: Total tokens used
            max_tokens: Maximum token budget
            context_tokens: Current context window usage
            max_context_tokens: Maximum context window size
            message_count: Total messages in conversation
        """
        queue = get_deferred_queue()
        queue.update_context(
            turn_number=turn_number,
            total_tokens=total_tokens,
            max_tokens=max_tokens,
            context_tokens=context_tokens,
            max_context_tokens=max_context_tokens,
            message_count=message_count,
        )

    @property
    def deferred_count(self) -> int:
        """Get the count of pending deferred notifications."""
        queue = get_deferred_queue()
        return queue.pending_count
