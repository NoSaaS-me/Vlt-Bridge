"""Pydantic models for notification settings and subscriber info."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List


class NotificationSettings(BaseModel):
    """User's notification settings."""

    disabled_subscribers: List[str] = Field(
        default_factory=list,
        description="List of subscriber IDs that are disabled for this user"
    )


class SubscriberInfo(BaseModel):
    """Information about a notification subscriber."""

    id: str = Field(..., description="Unique subscriber identifier")
    name: str = Field(..., description="Human-readable subscriber name")
    description: str = Field(..., description="Description of what this subscriber does")
    version: str = Field(..., description="Subscriber version")
    events: List[str] = Field(
        default_factory=list,
        description="List of event types this subscriber listens to"
    )
    is_core: bool = Field(
        default=False,
        description="Whether this is a core subscriber that cannot be disabled"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this subscriber is enabled for the current user"
    )


class SubscriberListResponse(BaseModel):
    """Response containing list of subscribers."""

    subscribers: List[SubscriberInfo] = Field(
        default_factory=list,
        description="List of all available subscribers"
    )


class SubscriberToggleRequest(BaseModel):
    """Request to toggle a subscriber's enabled state."""

    enabled: bool = Field(..., description="Whether to enable or disable the subscriber")


class SubscriberToggleResponse(BaseModel):
    """Response after toggling a subscriber."""

    id: str = Field(..., description="Subscriber ID that was toggled")
    enabled: bool = Field(..., description="New enabled state")
    message: str = Field(..., description="Status message")
