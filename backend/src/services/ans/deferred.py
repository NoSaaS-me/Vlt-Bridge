"""Scheduled/deferred delivery system for the Agent Notification System.

This module implements deferred notification delivery, allowing notifications
to be queued for delivery at specific points:
- NEXT_TURN: Guaranteed delivery at start of next turn
- AFTER_N_TURNS: Deliver after N turns complete
- AFTER_TOOL: Deliver after a specific tool completes
- ON_CONDITION: Deliver when a condition predicate is met
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID, uuid4

from .event import Event, Severity
from .subscriber import InjectionPoint, Priority


logger = logging.getLogger(__name__)


class DeliveryTrigger(str, Enum):
    """When a deferred notification should be delivered."""

    NEXT_TURN = "next_turn"           # Deliver at start of next turn
    AFTER_N_TURNS = "after_n_turns"   # Deliver after N turns complete
    AFTER_TOOL = "after_tool"         # Deliver after specific tool completes
    ON_CONDITION = "on_condition"     # Deliver when condition predicate returns True


@dataclass
class DeliveryContext:
    """Current state information used to evaluate delivery conditions.

    This is passed to condition predicates to determine if a deferred
    notification should be delivered.

    Attributes:
        turn_number: Current turn number (0-indexed)
        total_tokens: Total tokens used so far
        max_tokens: Maximum token budget
        context_tokens: Current context window usage
        max_context_tokens: Maximum context window size
        last_tool_name: Name of the most recently completed tool
        last_tool_result: Result of the most recently completed tool
        message_count: Total messages in conversation
        custom_data: Arbitrary data for custom conditions
    """

    turn_number: int = 0
    total_tokens: int = 0
    max_tokens: int = 0
    context_tokens: int = 0
    max_context_tokens: int = 0
    last_tool_name: Optional[str] = None
    last_tool_result: Optional[str] = None
    message_count: int = 0
    custom_data: Dict[str, Any] = field(default_factory=dict)


# Type alias for condition predicates
ConditionPredicate = Callable[[DeliveryContext], bool]


@dataclass
class DeferredNotification:
    """A notification queued for deferred delivery.

    Attributes:
        id: Unique notification identifier
        event: The original event that triggered this notification
        subscriber_id: ID of the subscriber that created this notification
        content: Pre-formatted notification content (optional)
        priority: Notification priority level
        trigger: Type of delivery trigger
        trigger_config: Configuration for the trigger (e.g., turns_remaining, tool_name)
        created_at: When the notification was created
        turns_remaining: For AFTER_N_TURNS, how many turns left
        tool_name: For AFTER_TOOL, which tool to wait for
        condition: For ON_CONDITION, the predicate function
        condition_expr: For ON_CONDITION, string representation of condition
    """

    id: UUID = field(default_factory=uuid4)
    event: Optional[Event] = None
    subscriber_id: str = ""
    content: str = ""
    priority: Priority = Priority.NORMAL
    trigger: DeliveryTrigger = DeliveryTrigger.NEXT_TURN
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Trigger-specific configuration
    turns_remaining: int = 1  # For NEXT_TURN (1) or AFTER_N_TURNS (N)
    tool_name: Optional[str] = None  # For AFTER_TOOL
    condition: Optional[ConditionPredicate] = None  # For ON_CONDITION
    condition_expr: Optional[str] = None  # String representation of condition

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "event": self.event.to_dict() if self.event else None,
            "subscriber_id": self.subscriber_id,
            "content": self.content,
            "priority": self.priority.value,
            "trigger": self.trigger.value,
            "created_at": self.created_at.isoformat(),
            "turns_remaining": self.turns_remaining,
            "tool_name": self.tool_name,
            "condition_expr": self.condition_expr,
        }


# Built-in condition predicates

def context_above_threshold(threshold: float) -> ConditionPredicate:
    """Create a predicate that fires when context usage exceeds threshold.

    Args:
        threshold: Fraction of context (0.0 to 1.0) to trigger at

    Returns:
        Predicate function that checks context usage
    """
    def predicate(ctx: DeliveryContext) -> bool:
        if ctx.max_context_tokens <= 0:
            return False
        usage = ctx.context_tokens / ctx.max_context_tokens
        return usage >= threshold
    return predicate


def tool_completed(tool_name: str) -> ConditionPredicate:
    """Create a predicate that fires when a specific tool completes.

    Args:
        tool_name: Name of the tool to wait for

    Returns:
        Predicate function that checks last completed tool
    """
    def predicate(ctx: DeliveryContext) -> bool:
        return ctx.last_tool_name == tool_name
    return predicate


def message_count_above(count: int) -> ConditionPredicate:
    """Create a predicate that fires when message count exceeds threshold.

    Args:
        count: Minimum number of messages to trigger at

    Returns:
        Predicate function that checks message count
    """
    def predicate(ctx: DeliveryContext) -> bool:
        return ctx.message_count >= count
    return predicate


def token_usage_above(threshold: float) -> ConditionPredicate:
    """Create a predicate that fires when token usage exceeds threshold.

    Args:
        threshold: Fraction of token budget (0.0 to 1.0) to trigger at

    Returns:
        Predicate function that checks token usage
    """
    def predicate(ctx: DeliveryContext) -> bool:
        if ctx.max_tokens <= 0:
            return False
        usage = ctx.total_tokens / ctx.max_tokens
        return usage >= threshold
    return predicate


class DeferredDeliveryQueue:
    """Queue for managing deferred notifications.

    This class manages notifications that are scheduled for delivery at
    specific points in the agent loop. It tracks turn counts, tool completions,
    and evaluates condition predicates to determine when notifications
    should be delivered.
    """

    def __init__(self):
        """Initialize an empty deferred queue."""
        self._queue: List[DeferredNotification] = []
        self._current_context = DeliveryContext()

    def enqueue(
        self,
        event: Optional[Event] = None,
        subscriber_id: str = "",
        content: str = "",
        priority: Priority = Priority.NORMAL,
        trigger: DeliveryTrigger = DeliveryTrigger.NEXT_TURN,
        turns: int = 1,
        tool_name: Optional[str] = None,
        condition: Optional[ConditionPredicate] = None,
        condition_expr: Optional[str] = None,
    ) -> DeferredNotification:
        """Add a notification to the deferred queue.

        Args:
            event: The original event (optional)
            subscriber_id: ID of the subscriber
            content: Pre-formatted notification content
            priority: Notification priority
            trigger: Type of delivery trigger
            turns: Number of turns to wait (for NEXT_TURN/AFTER_N_TURNS)
            tool_name: Tool to wait for (for AFTER_TOOL)
            condition: Condition predicate (for ON_CONDITION)
            condition_expr: String representation of condition

        Returns:
            The created DeferredNotification
        """
        notification = DeferredNotification(
            event=event,
            subscriber_id=subscriber_id,
            content=content,
            priority=priority,
            trigger=trigger,
            turns_remaining=turns,
            tool_name=tool_name,
            condition=condition,
            condition_expr=condition_expr,
        )
        self._queue.append(notification)
        logger.debug(
            f"Enqueued deferred notification: {notification.id} "
            f"(trigger={trigger.value}, turns={turns}, tool={tool_name})"
        )
        return notification

    def update_context(
        self,
        turn_number: Optional[int] = None,
        total_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        context_tokens: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
        last_tool_name: Optional[str] = None,
        last_tool_result: Optional[str] = None,
        message_count: Optional[int] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the delivery context with current state.

        Args:
            turn_number: Current turn number
            total_tokens: Total tokens used
            max_tokens: Maximum token budget
            context_tokens: Current context window usage
            max_context_tokens: Maximum context window size
            last_tool_name: Name of most recently completed tool
            last_tool_result: Result of most recently completed tool
            message_count: Total messages in conversation
            custom_data: Arbitrary data for custom conditions
        """
        if turn_number is not None:
            self._current_context.turn_number = turn_number
        if total_tokens is not None:
            self._current_context.total_tokens = total_tokens
        if max_tokens is not None:
            self._current_context.max_tokens = max_tokens
        if context_tokens is not None:
            self._current_context.context_tokens = context_tokens
        if max_context_tokens is not None:
            self._current_context.max_context_tokens = max_context_tokens
        if last_tool_name is not None:
            self._current_context.last_tool_name = last_tool_name
        if last_tool_result is not None:
            self._current_context.last_tool_result = last_tool_result
        if message_count is not None:
            self._current_context.message_count = message_count
        if custom_data is not None:
            self._current_context.custom_data.update(custom_data)

    def on_turn_start(self) -> List[DeferredNotification]:
        """Process turn start - decrement counters and return ready notifications.

        This should be called at the start of each agent turn. It:
        1. Decrements turns_remaining for NEXT_TURN and AFTER_N_TURNS
        2. Returns notifications that are now ready (turns_remaining <= 0)
        3. Removes delivered notifications from the queue

        Returns:
            List of notifications ready for delivery
        """
        ready: List[DeferredNotification] = []
        remaining: List[DeferredNotification] = []

        for notification in self._queue:
            if notification.trigger in (
                DeliveryTrigger.NEXT_TURN,
                DeliveryTrigger.AFTER_N_TURNS,
            ):
                notification.turns_remaining -= 1
                if notification.turns_remaining <= 0:
                    ready.append(notification)
                    logger.debug(
                        f"Deferred notification ready at turn_start: {notification.id}"
                    )
                else:
                    remaining.append(notification)
            else:
                remaining.append(notification)

        self._queue = remaining

        # Sort by priority (critical first)
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.NORMAL: 2,
            Priority.LOW: 3,
        }
        ready.sort(key=lambda n: priority_order.get(n.priority, 99))

        if ready:
            logger.debug(f"on_turn_start: {len(ready)} notifications ready")

        return ready

    def on_turn_end(self) -> List[DeferredNotification]:
        """Process turn end notifications.

        Currently, turn_end notifications are handled by checking conditions
        at the end of each turn. This is a hook for future extensions.

        Returns:
            List of notifications ready for delivery (empty for now)
        """
        # Currently no turn_end specific triggers
        # This could be extended for end-of-turn summary notifications
        return []

    def on_tool_complete(self, tool_name: str, result: Optional[str] = None) -> List[DeferredNotification]:
        """Process after-tool notifications for a specific tool.

        This should be called after a tool completes execution. It:
        1. Updates the delivery context with tool info
        2. Returns notifications waiting for this specific tool
        3. Removes delivered notifications from the queue

        Args:
            tool_name: Name of the tool that completed
            result: Result of the tool execution (optional)

        Returns:
            List of notifications ready for delivery
        """
        # Update context
        self.update_context(last_tool_name=tool_name, last_tool_result=result)

        ready: List[DeferredNotification] = []
        remaining: List[DeferredNotification] = []

        for notification in self._queue:
            if notification.trigger == DeliveryTrigger.AFTER_TOOL:
                if notification.tool_name == tool_name or notification.tool_name is None:
                    ready.append(notification)
                    logger.debug(
                        f"Deferred notification ready after tool {tool_name}: {notification.id}"
                    )
                else:
                    remaining.append(notification)
            else:
                remaining.append(notification)

        self._queue = remaining

        # Sort by priority
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.NORMAL: 2,
            Priority.LOW: 3,
        }
        ready.sort(key=lambda n: priority_order.get(n.priority, 99))

        if ready:
            logger.debug(f"on_tool_complete({tool_name}): {len(ready)} notifications ready")

        return ready

    def check_conditions(self) -> List[DeferredNotification]:
        """Evaluate condition-based notifications.

        This should be called periodically to check if any condition-based
        notifications should be delivered. It evaluates each ON_CONDITION
        notification's predicate against the current context.

        Returns:
            List of notifications whose conditions are met
        """
        ready: List[DeferredNotification] = []
        remaining: List[DeferredNotification] = []

        for notification in self._queue:
            if notification.trigger == DeliveryTrigger.ON_CONDITION:
                if notification.condition is not None:
                    try:
                        if notification.condition(self._current_context):
                            ready.append(notification)
                            logger.debug(
                                f"Condition met for deferred notification: {notification.id} "
                                f"(expr={notification.condition_expr})"
                            )
                        else:
                            remaining.append(notification)
                    except Exception as e:
                        logger.warning(
                            f"Error evaluating condition for notification {notification.id}: {e}"
                        )
                        remaining.append(notification)
                else:
                    # No condition function, can't evaluate
                    remaining.append(notification)
            else:
                remaining.append(notification)

        self._queue = remaining

        # Sort by priority
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.NORMAL: 2,
            Priority.LOW: 3,
        }
        ready.sort(key=lambda n: priority_order.get(n.priority, 99))

        if ready:
            logger.debug(f"check_conditions: {len(ready)} notifications ready")

        return ready

    def clear(self) -> int:
        """Clear all deferred notifications.

        Returns:
            Number of notifications cleared
        """
        count = len(self._queue)
        self._queue.clear()
        self._current_context = DeliveryContext()
        if count > 0:
            logger.debug(f"Cleared {count} deferred notifications")
        return count

    @property
    def pending_count(self) -> int:
        """Get the number of pending deferred notifications."""
        return len(self._queue)

    @property
    def context(self) -> DeliveryContext:
        """Get the current delivery context."""
        return self._current_context

    def get_pending(self) -> List[DeferredNotification]:
        """Get a copy of all pending notifications.

        Returns:
            List of pending notifications (copy)
        """
        return list(self._queue)


# Module-level singleton management

_deferred_queue: Optional[DeferredDeliveryQueue] = None


def get_deferred_queue() -> DeferredDeliveryQueue:
    """Get the global deferred delivery queue singleton.

    Returns:
        The global DeferredDeliveryQueue instance
    """
    global _deferred_queue
    if _deferred_queue is None:
        _deferred_queue = DeferredDeliveryQueue()
    return _deferred_queue


def reset_deferred_queue() -> DeferredDeliveryQueue:
    """Reset and return a fresh deferred delivery queue.

    This creates a new queue instance and replaces the global singleton.
    Use this at the start of a new query to ensure clean state.

    Returns:
        The new DeferredDeliveryQueue instance
    """
    global _deferred_queue
    _deferred_queue = DeferredDeliveryQueue()
    logger.debug("Reset deferred delivery queue")
    return _deferred_queue


__all__ = [
    "DeliveryTrigger",
    "DeliveryContext",
    "DeferredNotification",
    "DeferredDeliveryQueue",
    "ConditionPredicate",
    # Built-in predicates
    "context_above_threshold",
    "tool_completed",
    "message_count_above",
    "token_usage_above",
    # Module-level functions
    "get_deferred_queue",
    "reset_deferred_queue",
]
