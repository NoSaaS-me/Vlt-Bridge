"""Event bus for the Agent Notification System."""

import logging
from collections import defaultdict
from threading import Lock
from typing import Callable, Optional

from .event import Event, Severity


logger = logging.getLogger(__name__)


# Type alias for event handlers
EventHandler = Callable[[Event], None]


class EventBus:
    """Pub/sub event bus for ANS events."""
    
    def __init__(self, max_queue_size: int = 1000):
        """Initialize the event bus.
        
        Args:
            max_queue_size: Maximum pending events before dropping low-priority ones.
        """
        self.max_queue_size = max_queue_size
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._global_handlers: list[EventHandler] = []
        self._pending_events: list[Event] = []
        self._lock = Lock()
        self._enabled = True
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler to a specific event type.
        
        Args:
            event_type: Event type to subscribe to (supports wildcards like "tool.*").
            handler: Callback function to handle matching events.
        """
        with self._lock:
            self._handlers[event_type].append(handler)
            logger.debug(f"Subscribed handler to {event_type}")
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe a handler to all events.
        
        Args:
            handler: Callback function to handle all events.
        """
        with self._lock:
            self._global_handlers.append(handler)
            logger.debug("Subscribed global handler")
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> bool:
        """Unsubscribe a handler from an event type.
        
        Returns:
            True if handler was removed, False if not found.
        """
        with self._lock:
            if event_type in self._handlers:
                try:
                    self._handlers[event_type].remove(handler)
                    return True
                except ValueError:
                    return False
            return False
    
    def emit(self, event: Event) -> None:
        """Emit an event to all matching subscribers.
        
        Args:
            event: The event to emit.
        """
        if not self._enabled:
            logger.debug(f"Event bus disabled, dropping event: {event.type}")
            return
        
        with self._lock:
            # Check queue size and drop low-priority events if needed
            if len(self._pending_events) >= self.max_queue_size:
                self._drop_low_priority_events()
            
            self._pending_events.append(event)
        
        # Dispatch immediately
        self._dispatch(event)
    
    def _dispatch(self, event: Event) -> None:
        """Dispatch an event to all matching handlers."""
        handlers_called = 0
        
        # Call global handlers first
        for handler in self._global_handlers:
            try:
                handler(event)
                handlers_called += 1
            except Exception as e:
                logger.error(f"Error in global event handler: {e}")
        
        # Call type-specific handlers
        with self._lock:
            handlers = list(self._handlers.items())
        
        for event_type, type_handlers in handlers:
            if self._matches_type(event.type, event_type):
                for handler in type_handlers:
                    try:
                        handler(event)
                        handlers_called += 1
                    except Exception as e:
                        logger.error(f"Error in event handler for {event_type}: {e}")
        
        logger.debug(f"Dispatched event {event.type} to {handlers_called} handlers")
    
    def _matches_type(self, actual_type: str, subscribed_type: str) -> bool:
        """Check if an actual event type matches a subscribed pattern."""
        if actual_type == subscribed_type:
            return True
        # Support wildcard: "tool.*" matches "tool.call.failure"
        if subscribed_type.endswith(".*"):
            prefix = subscribed_type[:-2]
            return actual_type.startswith(prefix + ".")
        return False
    
    def _drop_low_priority_events(self) -> None:
        """Drop the oldest low-priority events when queue is full.

        Strategy:
        - Never drop CRITICAL or ERROR severity events
        - Sort remaining by severity (low first) then timestamp (oldest first)
        - Drop until queue is at 50% capacity
        """
        # Separate critical/error events from droppable events
        protected_events = [
            e for e in self._pending_events
            if e.severity in (Severity.CRITICAL, Severity.ERROR)
        ]
        droppable_events = [
            e for e in self._pending_events
            if e.severity not in (Severity.CRITICAL, Severity.ERROR)
        ]

        if not droppable_events:
            # All events are critical/error - can't drop any
            logger.warning(
                f"Queue overflow with {len(protected_events)} protected events, "
                "no low-priority events to drop"
            )
            return

        # Sort droppable by severity (low priority first) then by timestamp (oldest first)
        droppable_events.sort(
            key=lambda e: (e.severity.value_int, e.timestamp)
        )

        # Calculate how many to drop (target 50% capacity minus protected events)
        target_size = max(0, (self.max_queue_size // 2) - len(protected_events))
        events_to_keep = min(len(droppable_events), target_size)
        events_to_drop = len(droppable_events) - events_to_keep

        if events_to_drop > 0:
            dropped = droppable_events[:events_to_drop]
            kept_droppable = droppable_events[events_to_drop:]

            # Log with severity breakdown
            severity_counts = {}
            for e in dropped:
                severity_counts[e.severity.value] = severity_counts.get(e.severity.value, 0) + 1

            logger.warning(
                f"Queue overflow: dropped {len(dropped)} events "
                f"(by severity: {severity_counts}), "
                f"kept {len(protected_events)} protected + {len(kept_droppable)} droppable"
            )

            # Rebuild queue: protected events first, then remaining droppable
            self._pending_events = protected_events + kept_droppable
    
    def drain_pending(self) -> list[Event]:
        """Get and clear all pending events.
        
        Returns:
            List of pending events.
        """
        with self._lock:
            events = self._pending_events
            self._pending_events = []
            return events
    
    def clear(self) -> None:
        """Clear all handlers and pending events."""
        with self._lock:
            self._handlers.clear()
            self._global_handlers.clear()
            self._pending_events.clear()
    
    def enable(self) -> None:
        """Enable the event bus."""
        self._enabled = True
        logger.info("Event bus enabled")
    
    def disable(self) -> None:
        """Disable the event bus (events will be dropped)."""
        self._enabled = False
        logger.info("Event bus disabled")
    
    @property
    def pending_count(self) -> int:
        """Get the number of pending events."""
        return len(self._pending_events)
    
    @property
    def handler_count(self) -> int:
        """Get total number of handlers."""
        with self._lock:
            type_handlers = sum(len(h) for h in self._handlers.values())
            return type_handlers + len(self._global_handlers)


# Global singleton instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _event_bus
    if _event_bus is not None:
        _event_bus.clear()
    _event_bus = None
