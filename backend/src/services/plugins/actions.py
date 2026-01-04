"""Action dispatcher for rule actions.

This module provides the ActionDispatcher class that executes rule actions
when conditions are met, including notify_self, log, set_state, and emit_event.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from jinja2 import Environment, BaseLoader, TemplateSyntaxError, UndefinedError

from .context import RuleContext
from .rule import ActionType, RuleAction


logger = logging.getLogger(__name__)


class ActionError(Exception):
    """Raised when action execution fails."""

    pass


# Type alias for state setter callback
StateSetter = Callable[[str, Any], None]


class ActionDispatcher:
    """Dispatches and executes rule actions.

    The dispatcher handles all action types defined in RuleAction:
    - notify_self: Inject notification into agent context
    - log: Write to system log
    - set_state: Store plugin-scoped persistent state
    - emit_event: Emit ANS event

    Example:
        from services.ans.bus import EventBus

        bus = EventBus()
        dispatcher = ActionDispatcher(event_bus=bus)

        # Execute an action
        success = dispatcher.dispatch(rule.action, context)
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        state_setter: Optional[StateSetter] = None,
    ) -> None:
        """Initialize the action dispatcher.

        Args:
            event_bus: EventBus instance for emit_event actions.
            state_setter: Callback for set_state actions: (key, value) -> None
        """
        self._event_bus = event_bus
        self._state_setter = state_setter
        self._jinja_env = Environment(loader=BaseLoader(), autoescape=False)

        # Accumulated notifications for notify_self actions
        self._pending_notifications: list[dict[str, Any]] = []

    @property
    def pending_notifications(self) -> list[dict[str, Any]]:
        """Get list of pending notifications.

        Returns:
            List of notification dictionaries with message, category, priority.
        """
        return self._pending_notifications

    def clear_notifications(self) -> list[dict[str, Any]]:
        """Clear and return pending notifications.

        Returns:
            List of cleared notifications.
        """
        notifications = self._pending_notifications
        self._pending_notifications = []
        return notifications

    def dispatch(self, action: RuleAction, context: RuleContext) -> bool:
        """Execute an action with the given context.

        Args:
            action: The action to execute.
            context: The rule context for template rendering.

        Returns:
            True if action executed successfully, False otherwise.
        """
        try:
            if action.type == ActionType.NOTIFY_SELF:
                return self._notify_self(action, context)
            elif action.type == ActionType.LOG:
                return self._log(action, context)
            elif action.type == ActionType.SET_STATE:
                return self._set_state(action, context)
            elif action.type == ActionType.EMIT_EVENT:
                return self._emit_event(action, context)
            else:
                logger.error(f"Unknown action type: {action.type}")
                return False

        except Exception as e:
            logger.error(f"Error executing action {action.type}: {e}")
            return False

    def _render_template(self, template: str, context: RuleContext) -> str:
        """Render a Jinja2 template with the rule context.

        Args:
            template: Jinja2 template string.
            context: Rule context for template variables.

        Returns:
            Rendered string.

        Raises:
            ActionError: If template rendering fails.
        """
        try:
            tpl = self._jinja_env.from_string(template)
            return tpl.render(context=context)
        except TemplateSyntaxError as e:
            raise ActionError(f"Template syntax error: {e}")
        except UndefinedError as e:
            raise ActionError(f"Template undefined variable: {e}")

    def _notify_self(self, action: RuleAction, context: RuleContext) -> bool:
        """Execute a notify_self action.

        Adds a notification to the pending list for injection into agent context.

        Args:
            action: The notify_self action.
            context: Rule context for template rendering.

        Returns:
            True on success.
        """
        if not action.message:
            logger.warning("notify_self action has no message")
            return False

        # Render the message template
        try:
            message = self._render_template(action.message, context)
        except ActionError as e:
            logger.error(f"Failed to render notification message: {e}")
            return False

        # Create notification entry
        notification = {
            "message": message,
            "category": action.category or "info",
            "priority": action.priority.value,
            "deliver_at": action.deliver_at.value,
        }

        self._pending_notifications.append(notification)
        logger.debug(f"Queued notification: {message[:50]}...")
        return True

    def _log(self, action: RuleAction, context: RuleContext) -> bool:
        """Execute a log action.

        Writes to the system log at the specified level.

        Args:
            action: The log action.
            context: Rule context for template rendering.

        Returns:
            True on success.
        """
        message = action.message or "Rule triggered"

        # Render message template if needed
        if "{{" in message:
            try:
                message = self._render_template(message, context)
            except ActionError as e:
                logger.error(f"Failed to render log message: {e}")
                return False

        # Log at appropriate level
        level = action.level.lower()
        if level == "debug":
            logger.debug(f"[Rule] {message}")
        elif level == "info":
            logger.info(f"[Rule] {message}")
        elif level == "warning":
            logger.warning(f"[Rule] {message}")
        elif level == "error":
            logger.error(f"[Rule] {message}")
        else:
            logger.info(f"[Rule] {message}")

        return True

    def _set_state(self, action: RuleAction, context: RuleContext) -> bool:
        """Execute a set_state action.

        Stores a value in plugin-scoped persistent state.

        Args:
            action: The set_state action.
            context: Rule context for template rendering.

        Returns:
            True on success, False if no state setter configured.
        """
        if not action.key:
            logger.warning("set_state action has no key")
            return False

        if self._state_setter is None:
            logger.warning("set_state called but no state_setter configured")
            return False

        # Render value template if needed
        value = action.value
        if isinstance(value, str) and "{{" in value:
            try:
                value = self._render_template(value, context)
            except ActionError as e:
                logger.error(f"Failed to render state value: {e}")
                return False

        # Set the state
        try:
            self._state_setter(action.key, value)
            logger.debug(f"Set state: {action.key} = {value}")
            return True
        except Exception as e:
            logger.error(f"Failed to set state {action.key}: {e}")
            return False

    def _emit_event(self, action: RuleAction, context: RuleContext) -> bool:
        """Execute an emit_event action.

        Emits an ANS event to the event bus.

        Args:
            action: The emit_event action.
            context: Rule context for template rendering.

        Returns:
            True on success, False if no event bus configured.
        """
        if not action.event_type:
            logger.warning("emit_event action has no event_type")
            return False

        if self._event_bus is None:
            logger.warning("emit_event called but no event_bus configured")
            return False

        # Build payload, rendering any templates
        payload = {}
        if action.payload:
            for key, value in action.payload.items():
                if isinstance(value, str) and "{{" in value:
                    try:
                        payload[key] = self._render_template(value, context)
                    except ActionError as e:
                        logger.warning(f"Failed to render payload value for {key}: {e}")
                        payload[key] = value
                else:
                    payload[key] = value

        # Create and emit event
        try:
            # Import here to avoid circular dependency
            from ..ans.event import Event, Severity

            event = Event(
                type=action.event_type,
                source="plugin_rule",
                severity=Severity.INFO,
                payload=payload,
            )
            self._event_bus.emit(event)
            logger.debug(f"Emitted event: {action.event_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to emit event {action.event_type}: {e}")
            return False


__all__ = [
    "ActionDispatcher",
    "ActionError",
    "StateSetter",
]
