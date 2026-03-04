"""Service for per-user connector configuration in SQLite."""
from __future__ import annotations

from typing import Optional
from fastapi import Depends

from .database import DatabaseService, get_db_service
from ..connectors.base import BaseConnector


class ConnectorService:
    def __init__(self, db_service: Optional[DatabaseService] = None):
        self.db = db_service or DatabaseService()

    def get_config(self, user_id: str, connector_name: str) -> dict[str, str]:
        """Return all config key/value pairs for a connector, including __enabled."""
        conn = self.db.connect()
        cursor = conn.execute(
            "SELECT config_key, config_value FROM connector_configs WHERE user_id=? AND connector_name=?",
            (user_id, connector_name),
        )
        return {row["config_key"]: row["config_value"] or "" for row in cursor.fetchall()}

    def set_config(self, user_id: str, connector_name: str, updates: dict[str, str]) -> None:
        """Upsert config keys for a connector."""
        conn = self.db.connect()
        for key, value in updates.items():
            conn.execute(
                """
                INSERT INTO connector_configs (user_id, connector_name, config_key, config_value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (user_id, connector_name, config_key)
                DO UPDATE SET config_value=excluded.config_value, updated_at=datetime('now')
                """,
                (user_id, connector_name, key, value),
            )
        conn.commit()

    def is_enabled(self, user_id: str, connector_name: str) -> bool:
        config = self.get_config(user_id, connector_name)
        return config.get("__enabled", "false").lower() == "true"

    def get_credentials(self, user_id: str, connector_name: str) -> dict[str, str]:
        """Return config minus any keys starting with __ (internal control keys)."""
        return {k: v for k, v in self.get_config(user_id, connector_name).items() if not k.startswith("__")}

    def is_configured(self, user_id: str, connector: BaseConnector) -> bool:
        """True if all secret credential fields have non-empty values."""
        creds = self.get_credentials(user_id, connector.name)
        secret_fields = [f.name for f in connector.credential_fields if f.secret]
        return all(creds.get(f, "").strip() for f in secret_fields)


def get_connector_service(db: DatabaseService = Depends(get_db_service)) -> ConnectorService:
    return ConnectorService(db_service=db)
