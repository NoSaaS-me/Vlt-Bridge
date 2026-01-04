"""API routes for notification subscriber management."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

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
