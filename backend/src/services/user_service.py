"""User management service for multi-tenancy auth."""

import logging
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from .database import DatabaseService

logger = logging.getLogger(__name__)


@dataclass
class UserRecord:
    user_id: str
    display_name: Optional[str]
    avatar_url: Optional[str]
    role: str  # 'admin', 'user', 'pending', 'blocked'
    approved_by: Optional[str]
    created_at: str
    last_login_at: Optional[str]


def _row_to_record(row: sqlite3.Row) -> UserRecord:
    """Convert a sqlite3.Row to a UserRecord."""
    return UserRecord(
        user_id=row["user_id"],
        display_name=row["display_name"],
        avatar_url=row["avatar_url"],
        role=row["role"],
        approved_by=row["approved_by"],
        created_at=row["created_at"],
        last_login_at=row["last_login_at"],
    )


class UserService:
    def __init__(self, db: Optional[DatabaseService] = None):
        self._db = db or DatabaseService()

    def _conn(self) -> sqlite3.Connection:
        return self._db.connect()

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        """Get user by ID. Returns None if not found."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            return _row_to_record(row) if row else None
        finally:
            conn.close()

    def list_users(self, role: Optional[str] = None) -> list[UserRecord]:
        """List all users, optionally filtered by role."""
        conn = self._conn()
        try:
            if role:
                rows = conn.execute(
                    "SELECT * FROM users WHERE role = ? ORDER BY created_at DESC",
                    (role,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM users ORDER BY created_at DESC"
                ).fetchall()
            return [_row_to_record(r) for r in rows]
        finally:
            conn.close()

    def has_any_admin(self) -> bool:
        """Check if any admin user exists (for first-launch detection)."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM users WHERE role = 'admin' LIMIT 1"
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def create_user(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        avatar_url: Optional[str] = None,
        role: str = "pending",
    ) -> UserRecord:
        """Create a new user with the given role."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO users (user_id, display_name, avatar_url, role, created_at, last_login_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, display_name, avatar_url, role, now, now),
            )
            conn.commit()
            logger.info("Created user %s with role=%s", user_id, role)
            return UserRecord(
                user_id=user_id,
                display_name=display_name,
                avatar_url=avatar_url,
                role=role,
                approved_by=None,
                created_at=now,
                last_login_at=now,
            )
        finally:
            conn.close()

    def ensure_user(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> UserRecord:
        """Get or create user. Updates last_login_at if exists. Does NOT change role."""
        existing = self.get_user(user_id)
        if existing:
            self.update_last_login(user_id)
            return existing

        # New user -- determine role
        if not self.has_any_admin():
            role = "admin"  # First user is admin
        else:
            from .config import get_config

            config = get_config()
            role = "pending" if config.require_user_approval else "user"

        return self.create_user(
            user_id, display_name=display_name, avatar_url=avatar_url, role=role
        )

    def update_role(
        self, user_id: str, new_role: str, approved_by: Optional[str] = None
    ) -> Optional[UserRecord]:
        """Update user role. Returns updated record or None if not found."""
        conn = self._conn()
        try:
            cur = conn.execute(
                """UPDATE users SET role = ?, approved_by = ? WHERE user_id = ?""",
                (new_role, approved_by, user_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
            logger.info(
                "Updated user %s role to %s (approved_by=%s)",
                user_id,
                new_role,
                approved_by,
            )
            return self.get_user(user_id)
        finally:
            conn.close()

    def update_last_login(self, user_id: str) -> None:
        """Touch last_login_at timestamp."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE users SET last_login_at = ? WHERE user_id = ?",
                (now, user_id),
            )
            conn.commit()
        finally:
            conn.close()

    def is_setup_complete(self) -> bool:
        """True if at least one admin exists (setup wizard done)."""
        return self.has_any_admin()


__all__ = ["UserService", "UserRecord"]
