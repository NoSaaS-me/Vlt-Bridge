"""Unit tests for ActionDispatcher (T028-T030).

This module tests:
- notify_self action (T028)
- log and set_state actions (T029)
- emit_event action (T030)
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from backend.src.services.plugins.actions import (
    ActionDispatcher,
    ActionError,
)
from backend.src.services.plugins.context import (
    HistoryState,
    PluginState,
    ProjectState,
    RuleContext,
    TurnState,
    UserState,
)
from backend.src.services.plugins.rule import (
    ActionType,
    InjectionPoint,
    Priority,
    RuleAction,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_context() -> RuleContext:
    """Create a basic RuleContext for testing."""
    return RuleContext(
        turn=TurnState(
            number=5,
            token_usage=0.85,
            context_usage=0.6,
            iteration_count=3,
        ),
        history=HistoryState(),
        user=UserState(id="user-123"),
        project=ProjectState(id="project-456"),
        state=PluginState(_store={"counter": 5}),
    )


@pytest.fixture
def mock_event_bus() -> MagicMock:
    """Create a mock event bus."""
    return MagicMock()


@pytest.fixture
def mock_state_setter() -> MagicMock:
    """Create a mock state setter."""
    return MagicMock()


@pytest.fixture
def dispatcher(
    mock_event_bus: MagicMock,
    mock_state_setter: MagicMock,
) -> ActionDispatcher:
    """Create an ActionDispatcher with mocks."""
    return ActionDispatcher(
        event_bus=mock_event_bus,
        state_setter=mock_state_setter,
    )


# =============================================================================
# T028: notify_self Action Tests
# =============================================================================


class TestNotifySelfAction:
    """Tests for notify_self action (T028)."""

    def test_notify_self_adds_to_pending(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """notify_self adds notification to pending list."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message="Test notification",
            category="info",
            priority=Priority.NORMAL,
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is True
        assert len(dispatcher.pending_notifications) == 1
        assert dispatcher.pending_notifications[0]["message"] == "Test notification"

    def test_notify_self_renders_template(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """notify_self renders Jinja2 templates in message."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message="Token usage at {{ (context.turn.token_usage * 100) | int }}%",
            category="warning",
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is True
        assert len(dispatcher.pending_notifications) == 1
        assert "85%" in dispatcher.pending_notifications[0]["message"]

    def test_notify_self_includes_category(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """notify_self includes category in notification."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message="Test",
            category="warning",
        )

        dispatcher.dispatch(action, basic_context)

        assert dispatcher.pending_notifications[0]["category"] == "warning"

    def test_notify_self_includes_priority(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """notify_self includes priority in notification."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message="Test",
            priority=Priority.HIGH,
        )

        dispatcher.dispatch(action, basic_context)

        assert dispatcher.pending_notifications[0]["priority"] == "high"

    def test_notify_self_includes_deliver_at(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """notify_self includes deliver_at in notification."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message="Test",
            deliver_at=InjectionPoint.IMMEDIATE,
        )

        dispatcher.dispatch(action, basic_context)

        assert dispatcher.pending_notifications[0]["deliver_at"] == "immediate"

    def test_notify_self_without_message_fails(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """notify_self without message returns False."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message=None,
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is False
        assert len(dispatcher.pending_notifications) == 0

    def test_notify_self_with_invalid_template(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """notify_self with invalid template returns False."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message="Invalid {{ missing_close",
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is False

    def test_clear_notifications(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """clear_notifications returns and clears pending notifications."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message="Test 1",
        )
        dispatcher.dispatch(action, basic_context)

        action2 = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message="Test 2",
        )
        dispatcher.dispatch(action2, basic_context)

        cleared = dispatcher.clear_notifications()

        assert len(cleared) == 2
        assert len(dispatcher.pending_notifications) == 0


# =============================================================================
# T029: log and set_state Action Tests
# =============================================================================


class TestLogAction:
    """Tests for log action (T029)."""

    def test_log_at_info_level(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """log action writes to info level."""
        action = RuleAction(
            type=ActionType.LOG,
            message="Test log message",
            level="info",
        )

        with caplog.at_level("INFO"):
            result = dispatcher.dispatch(action, basic_context)

        assert result is True
        assert "Test log message" in caplog.text

    def test_log_at_debug_level(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """log action writes to debug level."""
        action = RuleAction(
            type=ActionType.LOG,
            message="Debug message",
            level="debug",
        )

        with caplog.at_level("DEBUG"):
            result = dispatcher.dispatch(action, basic_context)

        assert result is True
        assert "Debug message" in caplog.text

    def test_log_at_warning_level(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """log action writes to warning level."""
        action = RuleAction(
            type=ActionType.LOG,
            message="Warning message",
            level="warning",
        )

        with caplog.at_level("WARNING"):
            result = dispatcher.dispatch(action, basic_context)

        assert result is True
        assert "Warning message" in caplog.text

    def test_log_at_error_level(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """log action writes to error level."""
        action = RuleAction(
            type=ActionType.LOG,
            message="Error message",
            level="error",
        )

        with caplog.at_level("ERROR"):
            result = dispatcher.dispatch(action, basic_context)

        assert result is True
        assert "Error message" in caplog.text

    def test_log_renders_template(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """log action renders Jinja2 templates."""
        action = RuleAction(
            type=ActionType.LOG,
            message="Turn {{ context.turn.number }} started",
            level="info",
        )

        with caplog.at_level("INFO"):
            result = dispatcher.dispatch(action, basic_context)

        assert result is True
        assert "Turn 5 started" in caplog.text

    def test_log_without_message_uses_default(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """log action without message uses default message."""
        action = RuleAction(
            type=ActionType.LOG,
            message=None,
            level="info",
        )

        with caplog.at_level("INFO"):
            result = dispatcher.dispatch(action, basic_context)

        assert result is True
        assert "Rule triggered" in caplog.text


class TestSetStateAction:
    """Tests for set_state action (T029)."""

    def test_set_state_calls_setter(
        self,
        dispatcher: ActionDispatcher,
        mock_state_setter: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """set_state calls the state setter callback."""
        action = RuleAction(
            type=ActionType.SET_STATE,
            key="my_key",
            value="my_value",
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is True
        mock_state_setter.assert_called_once_with("my_key", "my_value")

    def test_set_state_renders_value_template(
        self,
        dispatcher: ActionDispatcher,
        mock_state_setter: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """set_state renders Jinja2 templates in value."""
        action = RuleAction(
            type=ActionType.SET_STATE,
            key="last_turn",
            value="{{ context.turn.number }}",
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is True
        mock_state_setter.assert_called_once_with("last_turn", "5")

    def test_set_state_without_key_fails(
        self,
        dispatcher: ActionDispatcher,
        mock_state_setter: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """set_state without key returns False."""
        action = RuleAction(
            type=ActionType.SET_STATE,
            key=None,
            value="test",
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is False
        mock_state_setter.assert_not_called()

    def test_set_state_without_setter_fails(
        self,
        mock_event_bus: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """set_state without state_setter returns False."""
        dispatcher = ActionDispatcher(event_bus=mock_event_bus, state_setter=None)

        action = RuleAction(
            type=ActionType.SET_STATE,
            key="test_key",
            value="test_value",
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is False

    def test_set_state_handles_setter_exception(
        self,
        dispatcher: ActionDispatcher,
        mock_state_setter: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """set_state handles exceptions from state setter."""
        mock_state_setter.side_effect = RuntimeError("Database error")

        action = RuleAction(
            type=ActionType.SET_STATE,
            key="test_key",
            value="test_value",
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is False


# =============================================================================
# T030: emit_event Action Tests
# =============================================================================


class TestEmitEventAction:
    """Tests for emit_event action (T030)."""

    def test_emit_event_calls_bus(
        self,
        dispatcher: ActionDispatcher,
        mock_event_bus: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """emit_event calls event bus emit method."""
        action = RuleAction(
            type=ActionType.EMIT_EVENT,
            event_type="custom.event.fired",
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is True
        mock_event_bus.emit.assert_called_once()

    def test_emit_event_includes_event_type(
        self,
        dispatcher: ActionDispatcher,
        mock_event_bus: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """emit_event includes correct event type."""
        action = RuleAction(
            type=ActionType.EMIT_EVENT,
            event_type="custom.budget.warning",
        )

        dispatcher.dispatch(action, basic_context)

        call_args = mock_event_bus.emit.call_args
        event = call_args[0][0]
        assert event.type == "custom.budget.warning"

    def test_emit_event_includes_payload(
        self,
        dispatcher: ActionDispatcher,
        mock_event_bus: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """emit_event includes payload in event."""
        action = RuleAction(
            type=ActionType.EMIT_EVENT,
            event_type="custom.event",
            payload={"key": "value", "number": 42},
        )

        dispatcher.dispatch(action, basic_context)

        call_args = mock_event_bus.emit.call_args
        event = call_args[0][0]
        assert event.payload["key"] == "value"
        assert event.payload["number"] == 42

    def test_emit_event_renders_payload_templates(
        self,
        dispatcher: ActionDispatcher,
        mock_event_bus: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """emit_event renders Jinja2 templates in payload."""
        action = RuleAction(
            type=ActionType.EMIT_EVENT,
            event_type="custom.event",
            payload={
                "turn": "{{ context.turn.number }}",
                "usage": "{{ context.turn.token_usage }}",
            },
        )

        dispatcher.dispatch(action, basic_context)

        call_args = mock_event_bus.emit.call_args
        event = call_args[0][0]
        assert event.payload["turn"] == "5"
        assert event.payload["usage"] == "0.85"

    def test_emit_event_without_event_type_fails(
        self,
        dispatcher: ActionDispatcher,
        mock_event_bus: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """emit_event without event_type returns False."""
        action = RuleAction(
            type=ActionType.EMIT_EVENT,
            event_type=None,
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is False
        mock_event_bus.emit.assert_not_called()

    def test_emit_event_without_bus_fails(
        self,
        mock_state_setter: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """emit_event without event_bus returns False."""
        dispatcher = ActionDispatcher(event_bus=None, state_setter=mock_state_setter)

        action = RuleAction(
            type=ActionType.EMIT_EVENT,
            event_type="custom.event",
        )

        result = dispatcher.dispatch(action, basic_context)

        assert result is False

    def test_emit_event_source_is_plugin_rule(
        self,
        dispatcher: ActionDispatcher,
        mock_event_bus: MagicMock,
        basic_context: RuleContext,
    ) -> None:
        """emit_event sets source to 'plugin_rule'."""
        action = RuleAction(
            type=ActionType.EMIT_EVENT,
            event_type="custom.event",
        )

        dispatcher.dispatch(action, basic_context)

        call_args = mock_event_bus.emit.call_args
        event = call_args[0][0]
        assert event.source == "plugin_rule"


# =============================================================================
# General Dispatcher Tests
# =============================================================================


class TestDispatcherGeneral:
    """General tests for ActionDispatcher."""

    def test_unknown_action_type_returns_false(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """Unknown action type returns False."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,  # Will be modified
            message="Test",
        )
        # Manually set to invalid type
        action.type = "invalid_type"  # type: ignore

        result = dispatcher.dispatch(action, basic_context)

        assert result is False

    def test_dispatcher_handles_exceptions(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """Dispatcher handles exceptions gracefully."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message="{{ undefined_var.nested }}",  # Will fail
        )

        # Should not raise, should return False
        result = dispatcher.dispatch(action, basic_context)

        assert result is False

    def test_multiple_notifications_accumulate(
        self,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """Multiple notify_self actions accumulate notifications."""
        actions = [
            RuleAction(type=ActionType.NOTIFY_SELF, message="First"),
            RuleAction(type=ActionType.NOTIFY_SELF, message="Second"),
            RuleAction(type=ActionType.NOTIFY_SELF, message="Third"),
        ]

        for action in actions:
            dispatcher.dispatch(action, basic_context)

        assert len(dispatcher.pending_notifications) == 3
        assert dispatcher.pending_notifications[0]["message"] == "First"
        assert dispatcher.pending_notifications[1]["message"] == "Second"
        assert dispatcher.pending_notifications[2]["message"] == "Third"
