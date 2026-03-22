"""Service for managing per-user proxy profiles in SQLite, with encrypted credentials."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from cryptography.fernet import Fernet
from fastapi import Depends

from .database import DatabaseService, get_db_service

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProxyProfileService:
    def __init__(self, db_service: Optional[DatabaseService] = None):
        self.db = db_service or DatabaseService()
        self._fernet: Optional[Fernet] = None  # lazy init

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _get_fernet(self) -> Fernet:
        if self._fernet is None:
            from vlt_connectors.encryption import get_fernet
            from .config import get_config
            cfg = get_config()
            enc_key = cfg.connector_encryption_key or ""
            jwt_key = cfg.jwt_secret_key or ""
            self._fernet = get_fernet(
                encryption_key=enc_key if enc_key else None,
                jwt_secret=jwt_key if jwt_key else None,
            )
        return self._fernet

    def _encrypt(self, value: str) -> str:
        from vlt_connectors.encryption import encrypt_value
        return encrypt_value(value, self._get_fernet())

    def _decrypt(self, value: str) -> str:
        from vlt_connectors.encryption import decrypt_value
        try:
            return decrypt_value(value, self._get_fernet())
        except Exception:
            logger.warning("Failed to decrypt proxy credential; returning raw value")
            return value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_dict(self, row) -> dict:
        """Convert a sqlite3.Row to a public dict (password masked)."""
        return {
            "name": row["name"],
            "proxy_url": row["proxy_url"],
            "proxy_username": row["proxy_username"] or "",
            "proxy_password": "••••••••" if row["proxy_password"] else "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_dict_with_creds(self, row) -> dict:
        """Convert row including decrypted password (for internal/daemon use)."""
        password = ""
        if row["proxy_password"]:
            password = self._decrypt(row["proxy_password"])
        return {
            "name": row["name"],
            "proxy_url": row["proxy_url"],
            "proxy_username": row["proxy_username"] or "",
            "proxy_password": password,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_profiles(self, user_id: str) -> list[dict]:
        """Return all proxy profiles for a user (passwords masked)."""
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                "SELECT name, proxy_url, proxy_username, proxy_password, created_at, updated_at"
                " FROM proxy_profiles WHERE user_id=? ORDER BY name",
                (user_id,),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_profile(self, user_id: str, name: str, decrypt: bool = False) -> dict | None:
        """Return a single proxy profile or None if not found."""
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                "SELECT name, proxy_url, proxy_username, proxy_password, created_at, updated_at"
                " FROM proxy_profiles WHERE user_id=? AND name=?",
                (user_id, name),
            )
            row = cursor.fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._row_to_dict_with_creds(row) if decrypt else self._row_to_dict(row)

    def create_profile(
        self,
        user_id: str,
        name: str,
        proxy_url: str,
        proxy_username: str = "",
        proxy_password: str = "",
    ) -> dict:
        """Create a new proxy profile. Raises ValueError if the name already exists."""
        now = _now_iso()
        encrypted_pw = self._encrypt(proxy_password) if proxy_password else None
        conn = self.db.connect()
        try:
            conn.execute(
                """
                INSERT INTO proxy_profiles
                    (user_id, name, proxy_url, proxy_username, proxy_password, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, name, proxy_url, proxy_username or None, encrypted_pw, now, now),
            )
            conn.commit()
        except Exception as exc:
            if "UNIQUE constraint failed" in str(exc):
                raise ValueError(f"Proxy profile '{name}' already exists")
            raise
        finally:
            conn.close()
        profile = self.get_profile(user_id, name)
        assert profile is not None
        return profile

    def update_profile(
        self,
        user_id: str,
        name: str,
        proxy_url: Optional[str] = None,
        proxy_username: Optional[str] = None,
        proxy_password: Optional[str] = None,
    ) -> dict | None:
        """Update fields on an existing proxy profile. Returns updated profile or None if not found."""
        existing = self.get_profile(user_id, name)
        if existing is None:
            return None

        conn = self.db.connect()
        try:
            updates: list[str] = ["updated_at=?"]
            values: list = [_now_iso()]

            if proxy_url is not None:
                updates.append("proxy_url=?")
                values.append(proxy_url)
            if proxy_username is not None:
                updates.append("proxy_username=?")
                values.append(proxy_username or None)
            if proxy_password is not None:
                updates.append("proxy_password=?")
                values.append(self._encrypt(proxy_password) if proxy_password else None)

            values.extend([user_id, name])
            conn.execute(
                f"UPDATE proxy_profiles SET {', '.join(updates)} WHERE user_id=? AND name=?",
                values,
            )
            conn.commit()
        finally:
            conn.close()

        return self.get_profile(user_id, name)

    def delete_profile(self, user_id: str, name: str) -> bool:
        """Delete a proxy profile. Returns True if it existed."""
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                "DELETE FROM proxy_profiles WHERE user_id=? AND name=?",
                (user_id, name),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


def get_proxy_service(db: DatabaseService = Depends(get_db_service)) -> ProxyProfileService:
    return ProxyProfileService(db_service=db)
