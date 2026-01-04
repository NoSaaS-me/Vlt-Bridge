"""API routes for notification subscriber management."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..middleware import AuthContext, get_auth_context
from ...models.notifications import (
    NotificationSettings,
    SubscriberInfo,
    SubscriberListResponse,
    SubscriberToggleRequest,
    SubscriberToggleResponse,
)
from ...services.user_settings import UserSettingsService, get_user_settings_service
from ...services.ans.subscriber import SubscriberLoader
from ...services.ans.bus import get_event_bus
from ...services.ans.event import Event, Severity

DEMO_USER_ID = "demo-user"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["notifications"])

# Singleton subscriber loader
_subscriber_loader: SubscriberLoader | None = None


def get_subscriber_loader() -> SubscriberLoader:
    """Get or create the subscriber loader singleton."""
    global _subscriber_loader
    if _subscriber_loader is None:
        _subscriber_loader = SubscriberLoader()
        _subscriber_loader.load_all()
    return _subscriber_loader


@router.get("/subscribers", response_model=SubscriberListResponse)
async def list_subscribers(
    auth: AuthContext = Depends(get_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
    loader: SubscriberLoader = Depends(get_subscriber_loader),
):
    """
    Get all available notification subscribers.

    Returns a list of all subscribers with their enabled status for the current user.
    Core subscribers are always enabled and cannot be disabled.
    """
    try:
        # Get user's disabled subscribers
        disabled_subscribers = settings_service.get_disabled_subscribers(auth.user_id)
        disabled_set = set(disabled_subscribers)

        # Get all subscribers from loader
        all_subscribers = loader.get_all_subscribers()

        # Build response with enabled status
        subscriber_infos: List[SubscriberInfo] = []
        for sub in all_subscribers:
            # Core subscribers are always enabled
            is_enabled = sub.core or (sub.id not in disabled_set)

            subscriber_infos.append(
                SubscriberInfo(
                    id=sub.id,
                    name=sub.name,
                    description=sub.config.description,
                    version=sub.config.version,
                    events=sub.event_types,
                    is_core=sub.core,
                    enabled=is_enabled,
                )
            )

        return SubscriberListResponse(subscribers=subscriber_infos)

    except Exception as e:
        logger.error(f"Failed to list subscribers for user {auth.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list subscribers: {str(e)}"
        )


@router.post("/subscribers/{subscriber_id}/toggle", response_model=SubscriberToggleResponse)
async def toggle_subscriber(
    subscriber_id: str,
    request: SubscriberToggleRequest,
    auth: AuthContext = Depends(get_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
    loader: SubscriberLoader = Depends(get_subscriber_loader),
):
    """
    Toggle a subscriber's enabled state for the current user.

    Core subscribers cannot be disabled and will return an error.
    """
    try:
        # Check if subscriber exists
        subscriber = loader.get_subscriber(subscriber_id)
        if subscriber is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Subscriber '{subscriber_id}' not found"
            )

        # Check if it's a core subscriber
        if subscriber.core and not request.enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot disable core subscriber '{subscriber_id}'. Core subscribers are required for system operation."
            )

        # Get current disabled subscribers
        disabled_subscribers = settings_service.get_disabled_subscribers(auth.user_id)
        disabled_set = set(disabled_subscribers)

        # Update disabled set based on request
        if request.enabled:
            # Enable: remove from disabled set
            disabled_set.discard(subscriber_id)
            message = f"Subscriber '{subscriber.name}' enabled"
        else:
            # Disable: add to disabled set
            disabled_set.add(subscriber_id)
            message = f"Subscriber '{subscriber.name}' disabled"

        # Save updated disabled list
        settings_service.set_disabled_subscribers(auth.user_id, list(disabled_set))

        return SubscriberToggleResponse(
            id=subscriber_id,
            enabled=request.enabled,
            message=message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle subscriber {subscriber_id} for user {auth.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to toggle subscriber: {str(e)}"
        )


class TestNotificationResponse(BaseModel):
    """Response from testing a notification subscriber."""

    subscriber_id: str = Field(..., description="The subscriber ID that was tested")
    event_type: str = Field(..., description="The event type that was emitted")
    inject_at: str = Field(..., description="When the notification will appear (turn_start, after_tool, immediate)")
    message: str = Field(..., description="Status message")


# Map subscriber IDs to their primary test event types
# NOTE: Test events are emitted to the EventBus. For notifications to appear:
# - turn_start subscribers: Will show at the start of the next oracle turn
# - after_tool subscribers: Will show after the next tool execution
# - immediate subscribers: Will show when critical events (like max turns) occur
#
# For testing purposes, payloads should match the structure emitted by oracle_agent.py
SUBSCRIBER_TEST_EVENTS = {
    "tool_failure": {
        "type": "tool.call.failure",
        "severity": Severity.ERROR,  # Match oracle_agent severity
        "payload": {
            "tool_name": "test_tool",
            "error_message": "This is a test notification for tool failure",
            "error_type": "TestError",
            "duration_ms": 150,
        },
        # inject_at: after_tool - only shows after next tool execution
    },
    "budget_warning": {
        "type": "budget.token.warning",
        "severity": Severity.WARNING,
        "payload": {
            "budget_type": "token",  # Required for deduplication
            "current": 8000,
            "max": 10000,
            "percent": 80,
            "message": "Used 8000 of 10000 tokens",
        },
        # inject_at: turn_start - shows at next turn
    },
    "budget_exceeded": {
        "type": "budget.token.exceeded",
        "severity": Severity.ERROR,  # Match oracle_agent severity
        "payload": {
            "budget_type": "token",
            "current": 10500,
            "max": 10000,
            "percent": 105,
            "message": "Exceeded token budget: 10500 of 10000 tokens",
        },
        # inject_at: immediate - only shows on critical events (max turns)
    },
    "loop_detected": {
        "type": "agent.loop.detected",
        "severity": Severity.WARNING,
        "payload": {
            "iteration_count": 5,
            "pattern": "repeated tool calls",
            "tool_sequence": ["search_code", "search_code", "search_code"],
        },
        # inject_at: immediate - only shows on critical events (max turns)
    },
    "self_notify": {
        "type": "agent.self.notify",
        "severity": Severity.INFO,
        "payload": {
            "message": "This is a test self-notification from the agent",
            "priority": "normal",
            "inject_at": "turn_start",
        },
        # inject_at: turn_start - shows at next turn
    },
    "context_limit": {
        "type": "context.approaching_limit",
        "severity": Severity.WARNING,
        "payload": {
            "current_tokens": 95000,
            "max_tokens": 100000,
            "percent": 95,
            "message": "Context window at 95% capacity",
        },
        # inject_at: turn_start - shows at next turn
    },
    "session_resumed": {
        "type": "session.resumed",
        "severity": Severity.INFO,
        "payload": {
            "tree_id": "test-tree-123",
            "previous_session_id": "test-session-123",
            "context_nodes_restored": 5,
            "last_question": "What was the previous question?",
        },
        # inject_at: turn_start - shows at next turn
    },
    "source_stale": {
        "type": "source.stale",
        "severity": Severity.WARNING,
        "payload": {
            "source_type": "coderag",
            "path": "/path/to/stale/file.py",
            "last_updated": "2024-01-01T00:00:00Z",
            "staleness_hours": 48,
        },
        # inject_at: after_tool - only shows after next tool execution
    },
    "task_checkpoint": {
        "type": "task.checkpoint",
        "severity": Severity.INFO,
        "payload": {
            "task_id": "test-task-001",
            "checkpoint_name": "Test checkpoint",
            "progress_percent": 50,
            "items_completed": 5,
            "items_total": 10,
        },
        # inject_at: turn_start - shows at next turn
    },
}


@router.post("/subscribers/{subscriber_id}/test", response_model=TestNotificationResponse)
async def test_subscriber(
    subscriber_id: str,
    auth: AuthContext = Depends(get_auth_context),
    loader: SubscriberLoader = Depends(get_subscriber_loader),
):
    """
    Emit a test event for a subscriber.

    This endpoint is only available for demo users.
    It emits a test event that the specified subscriber will catch,
    useful for verifying that the notification system is working.
    """
    # Only allow demo users to trigger test notifications
    if auth.user_id != DEMO_USER_ID:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Test notifications are only available for demo users"
        )

    # Check if subscriber exists
    subscriber = loader.get_subscriber(subscriber_id)
    if subscriber is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Subscriber '{subscriber_id}' not found"
        )

    # Get test event config for this subscriber
    test_config = SUBSCRIBER_TEST_EVENTS.get(subscriber_id)
    if test_config is None:
        # Fallback: use first event type from subscriber
        if subscriber.event_types:
            event_type = subscriber.event_types[0]
            test_config = {
                "type": event_type,
                "severity": Severity.INFO,
                "payload": {"test": True, "message": f"Test event for {subscriber.name}"},
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No test event configured for subscriber '{subscriber_id}'"
            )

    try:
        # Emit the test event
        bus = get_event_bus()
        event = Event(
            type=test_config["type"],
            source="notification_test_api",
            severity=test_config["severity"],
            payload=test_config["payload"],
        )
        bus.emit(event)

        # Get inject_at from subscriber config
        inject_at = subscriber.inject_at.value

        logger.info(f"Emitted test event for subscriber {subscriber_id}: {test_config['type']} (inject_at={inject_at})")

        # Build message with timing info
        timing_messages = {
            "turn_start": "Will appear at the start of the next oracle turn",
            "after_tool": "Will appear after the next tool execution",
            "immediate": "Will appear on critical events (max turns reached)",
            "turn_end": "Will appear at the end of the current turn",
        }
        timing_hint = timing_messages.get(inject_at, "")

        return TestNotificationResponse(
            subscriber_id=subscriber_id,
            event_type=test_config["type"],
            inject_at=inject_at,
            message=f"Test event emitted for {subscriber.name}. {timing_hint}",
        )

    except Exception as e:
        logger.error(f"Failed to emit test event for {subscriber_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to emit test event: {str(e)}"
        )
