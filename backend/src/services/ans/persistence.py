"""Cross-session notification persistence for the Agent Notification System.

This module handles storing and retrieving notifications that need to survive
session restarts. When an agent resumes a session, pending cross-session
notifications can be loaded and re-injected into context.

Feature: 014-ans-enhancements, Feature 3: Cross-Session Persistence
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from ..database import DatabaseService, DEFAULT_DB_PATH

logger = logging.getLogger(__name__)


class NotificationStatus(Enum):
    """Status of a cross-session notification."""

    PENDING = "pending"  # Not yet delivered
    DELIVERED = "delivered"  # Delivered to agent but not acknowledged
    ACKNOWLEDGED = "acknowledged"  # Agent confirmed receipt
    EXPIRED = "expired"  # Past expiration time
    CANCELLED = "cancelled"  # Manually cancelled


@dataclass
class CrossSessionNotification:
    """A notification that persists across session restarts.

    Attributes:
        id: Unique notification identifier.
        user_id: User this notification belongs to.
        project_id: Project context for scoping.
        tree_id: Optional context tree ID for association.
        event_type: Original event type that triggered this notification.
        source: Component that generated the notification.
        severity: Notification severity level.
        payload: Original event payload data.
        formatted_content: Pre-formatted TOON content for display.
        priority: Notification priority (low, normal, high, critical).
        inject_at: When to inject (turn_start, after_tool, immediate).
        created_at: When the notification was created.
        expires_at: When the notification expires (None = never).
        delivered_at: When the notification was delivered.
        acknowledged_at: When the notification was acknowledged.
        status: Current notification status.
        category: Optional category for grouping (discovery, warning, etc.).
        dedupe_key: Key for deduplication.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    project_id: str = ""
    tree_id: Optional[str] = None
    event_type: str = ""
    source: str = ""
    severity: str = "info"
    payload: dict[str, Any] = field(default_factory=dict)
    formatted_content: Optional[str] = None
    priority: str = "normal"
    inject_at: str = "turn_start"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    status: NotificationStatus = NotificationStatus.PENDING
    category: Optional[str] = None
    dedupe_key: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if the notification has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "tree_id": self.tree_id,
            "event_type": self.event_type,
            "source": self.source,
            "severity": self.severity,
            "payload": self.payload,
            "formatted_content": self.formatted_content,
            "priority": self.priority,
            "inject_at": self.inject_at,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "status": self.status.value,
            "category": self.category,
            "dedupe_key": self.dedupe_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossSessionNotification":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            user_id=data.get("user_id", ""),
            project_id=data.get("project_id", ""),
            tree_id=data.get("tree_id"),
            event_type=data.get("event_type", ""),
            source=data.get("source", ""),
            severity=data.get("severity", "info"),
            payload=data.get("payload", {}),
            formatted_content=data.get("formatted_content"),
            priority=data.get("priority", "normal"),
            inject_at=data.get("inject_at", "turn_start"),
            created_at=_parse_datetime(data.get("created_at")),
            expires_at=_parse_datetime(data.get("expires_at")),
            delivered_at=_parse_datetime(data.get("delivered_at")),
            acknowledged_at=_parse_datetime(data.get("acknowledged_at")),
            status=NotificationStatus(data.get("status", "pending")),
            category=data.get("category"),
            dedupe_key=data.get("dedupe_key"),
        )


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO datetime string, returning None if empty."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class CrossSessionPersistenceService:
    """Service for persisting cross-session notifications.

    This service manages the lifecycle of notifications that need to survive
    session restarts. Notifications are stored in SQLite and can be:
    - Stored when created
    - Retrieved when a session resumes
    - Marked as delivered/acknowledged
    - Cancelled if no longer needed
    - Cleaned up when expired
    """

    # Default expiration time for notifications (24 hours)
    DEFAULT_EXPIRY_HOURS = 24

    def __init__(self, db_service: Optional[DatabaseService] = None):
        """Initialize the persistence service.

        Args:
            db_service: Database service to use. If None, uses default.
        """
        self._db_service = db_service or DatabaseService()

    def store(
        self,
        notification: CrossSessionNotification,
        expiry_hours: Optional[int] = None,
    ) -> CrossSessionNotification:
        """Store a notification for cross-session persistence.

        Args:
            notification: The notification to store.
            expiry_hours: Hours until expiration. None uses default, 0 = never expires.

        Returns:
            The stored notification with updated fields.
        """
        conn = self._db_service.connect()
        try:
            # Set expiration if not already set
            if notification.expires_at is None and expiry_hours != 0:
                hours = expiry_hours if expiry_hours is not None else self.DEFAULT_EXPIRY_HOURS
                notification.expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)

            # Ensure created_at is set
            if notification.created_at is None:
                notification.created_at = datetime.now(timezone.utc)

            conn.execute(
                """
                INSERT INTO cross_session_notifications (
                    id, user_id, project_id, tree_id,
                    event_type, source, severity, payload_json,
                    formatted_content, priority, inject_at,
                    created_at, expires_at, delivered_at, acknowledged_at,
                    status, category, dedupe_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    notification.id,
                    notification.user_id,
                    notification.project_id,
                    notification.tree_id,
                    notification.event_type,
                    notification.source,
                    notification.severity,
                    json.dumps(notification.payload),
                    notification.formatted_content,
                    notification.priority,
                    notification.inject_at,
                    notification.created_at.isoformat() if notification.created_at else None,
                    notification.expires_at.isoformat() if notification.expires_at else None,
                    notification.delivered_at.isoformat() if notification.delivered_at else None,
                    notification.acknowledged_at.isoformat() if notification.acknowledged_at else None,
                    notification.status.value,
                    notification.category,
                    notification.dedupe_key,
                ),
            )
            conn.commit()

            logger.debug(
                f"Stored cross-session notification {notification.id} "
                f"for user {notification.user_id} project {notification.project_id}"
            )

            return notification

        except Exception as e:
            logger.error(f"Failed to store cross-session notification: {e}")
            raise
        finally:
            conn.close()

    def get_pending(
        self,
        user_id: str,
        project_id: str,
        tree_id: Optional[str] = None,
        include_delivered: bool = False,
    ) -> list[CrossSessionNotification]:
        """Get pending notifications for a user/project.

        Args:
            user_id: User ID to filter by.
            project_id: Project ID to filter by.
            tree_id: Optional tree ID to filter by.
            include_delivered: Whether to include already-delivered notifications.

        Returns:
            List of pending notifications, sorted by priority and created_at.
        """
        conn = self._db_service.connect()
        try:
            # Build query
            query = """
                SELECT * FROM cross_session_notifications
                WHERE user_id = ? AND project_id = ?
                AND (expires_at IS NULL OR expires_at > ?)
            """
            params: list[Any] = [
                user_id,
                project_id,
                datetime.now(timezone.utc).isoformat(),
            ]

            # Filter by status
            if include_delivered:
                query += " AND status IN ('pending', 'delivered')"
            else:
                query += " AND status = 'pending'"

            # Filter by tree_id if provided
            if tree_id is not None:
                query += " AND (tree_id = ? OR tree_id IS NULL)"
                params.append(tree_id)

            # Order by priority (critical first) then created_at
            query += """
                ORDER BY
                    CASE priority
                        WHEN 'critical' THEN 0
                        WHEN 'high' THEN 1
                        WHEN 'normal' THEN 2
                        WHEN 'low' THEN 3
                        ELSE 4
                    END,
                    created_at ASC
            """

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            notifications = []
            for row in rows:
                notification = self._row_to_notification(row)
                notifications.append(notification)

            logger.debug(
                f"Found {len(notifications)} pending cross-session notifications "
                f"for user {user_id} project {project_id}"
            )

            return notifications

        except Exception as e:
            logger.error(f"Failed to get pending notifications: {e}")
            raise
        finally:
            conn.close()

    def mark_delivered(
        self,
        notification_id: str,
    ) -> bool:
        """Mark a notification as delivered.

        Args:
            notification_id: ID of the notification to mark.

        Returns:
            True if updated, False if not found.
        """
        conn = self._db_service.connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """
                UPDATE cross_session_notifications
                SET status = ?, delivered_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (NotificationStatus.DELIVERED.value, now, notification_id),
            )
            conn.commit()

            updated = cursor.rowcount > 0
            if updated:
                logger.debug(f"Marked notification {notification_id} as delivered")
            return updated

        except Exception as e:
            logger.error(f"Failed to mark notification as delivered: {e}")
            raise
        finally:
            conn.close()

    def mark_acknowledged(
        self,
        notification_id: str,
    ) -> bool:
        """Mark a notification as acknowledged.

        Args:
            notification_id: ID of the notification to mark.

        Returns:
            True if updated, False if not found.
        """
        conn = self._db_service.connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """
                UPDATE cross_session_notifications
                SET status = ?, acknowledged_at = ?
                WHERE id = ? AND status IN ('pending', 'delivered')
                """,
                (NotificationStatus.ACKNOWLEDGED.value, now, notification_id),
            )
            conn.commit()

            updated = cursor.rowcount > 0
            if updated:
                logger.debug(f"Marked notification {notification_id} as acknowledged")
            return updated

        except Exception as e:
            logger.error(f"Failed to mark notification as acknowledged: {e}")
            raise
        finally:
            conn.close()

    def cancel(
        self,
        notification_id: str,
    ) -> bool:
        """Cancel a pending notification.

        Args:
            notification_id: ID of the notification to cancel.

        Returns:
            True if cancelled, False if not found or already processed.
        """
        conn = self._db_service.connect()
        try:
            cursor = conn.execute(
                """
                UPDATE cross_session_notifications
                SET status = ?
                WHERE id = ? AND status IN ('pending', 'delivered')
                """,
                (NotificationStatus.CANCELLED.value, notification_id),
            )
            conn.commit()

            cancelled = cursor.rowcount > 0
            if cancelled:
                logger.debug(f"Cancelled notification {notification_id}")
            return cancelled

        except Exception as e:
            logger.error(f"Failed to cancel notification: {e}")
            raise
        finally:
            conn.close()

    def cancel_by_dedupe_key(
        self,
        user_id: str,
        project_id: str,
        dedupe_key: str,
    ) -> int:
        """Cancel all pending notifications with a specific dedupe key.

        Args:
            user_id: User ID to filter by.
            project_id: Project ID to filter by.
            dedupe_key: Dedupe key to match.

        Returns:
            Number of notifications cancelled.
        """
        conn = self._db_service.connect()
        try:
            cursor = conn.execute(
                """
                UPDATE cross_session_notifications
                SET status = ?
                WHERE user_id = ? AND project_id = ?
                AND dedupe_key = ? AND status IN ('pending', 'delivered')
                """,
                (NotificationStatus.CANCELLED.value, user_id, project_id, dedupe_key),
            )
            conn.commit()

            count = cursor.rowcount
            if count > 0:
                logger.debug(
                    f"Cancelled {count} notifications with dedupe_key {dedupe_key}"
                )
            return count

        except Exception as e:
            logger.error(f"Failed to cancel notifications by dedupe key: {e}")
            raise
        finally:
            conn.close()

    def cleanup_expired(
        self,
        user_id: Optional[str] = None,
    ) -> int:
        """Remove expired notifications.

        Args:
            user_id: Optional user ID to limit cleanup scope.

        Returns:
            Number of notifications cleaned up.
        """
        conn = self._db_service.connect()
        try:
            now = datetime.now(timezone.utc).isoformat()

            if user_id:
                cursor = conn.execute(
                    """
                    DELETE FROM cross_session_notifications
                    WHERE user_id = ? AND expires_at IS NOT NULL AND expires_at < ?
                    """,
                    (user_id, now),
                )
            else:
                cursor = conn.execute(
                    """
                    DELETE FROM cross_session_notifications
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                    """,
                    (now,),
                )

            conn.commit()

            count = cursor.rowcount
            if count > 0:
                logger.info(f"Cleaned up {count} expired cross-session notifications")
            return count

        except Exception as e:
            logger.error(f"Failed to cleanup expired notifications: {e}")
            raise
        finally:
            conn.close()

    def get_by_id(
        self,
        notification_id: str,
    ) -> Optional[CrossSessionNotification]:
        """Get a notification by ID.

        Args:
            notification_id: ID of the notification to retrieve.

        Returns:
            The notification if found, None otherwise.
        """
        conn = self._db_service.connect()
        try:
            cursor = conn.execute(
                "SELECT * FROM cross_session_notifications WHERE id = ?",
                (notification_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_notification(row)

        except Exception as e:
            logger.error(f"Failed to get notification by ID: {e}")
            raise
        finally:
            conn.close()

    def _row_to_notification(self, row) -> CrossSessionNotification:
        """Convert a database row to a CrossSessionNotification."""
        return CrossSessionNotification(
            id=row["id"],
            user_id=row["user_id"],
            project_id=row["project_id"],
            tree_id=row["tree_id"],
            event_type=row["event_type"],
            source=row["source"],
            severity=row["severity"],
            payload=json.loads(row["payload_json"]) if row["payload_json"] else {},
            formatted_content=row["formatted_content"],
            priority=row["priority"],
            inject_at=row["inject_at"],
            created_at=_parse_datetime(row["created_at"]) or datetime.now(timezone.utc),
            expires_at=_parse_datetime(row["expires_at"]),
            delivered_at=_parse_datetime(row["delivered_at"]),
            acknowledged_at=_parse_datetime(row["acknowledged_at"]),
            status=NotificationStatus(row["status"]),
            category=row["category"],
            dedupe_key=row["dedupe_key"],
        )


# Singleton instance
_persistence_service: Optional[CrossSessionPersistenceService] = None


def get_persistence_service() -> CrossSessionPersistenceService:
    """Get the singleton persistence service instance.

    Returns:
        The CrossSessionPersistenceService singleton.
    """
    global _persistence_service
    if _persistence_service is None:
        _persistence_service = CrossSessionPersistenceService()
    return _persistence_service


__all__ = [
    "CrossSessionNotification",
    "CrossSessionPersistenceService",
    "NotificationStatus",
    "get_persistence_service",
]
