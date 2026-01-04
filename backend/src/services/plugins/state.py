"""Plugin state persistence service (T047).

This module provides SQLite-based persistence for plugin state,
allowing rules to store and retrieve state across sessions.

State is scoped by:
- user_id: Each user has isolated state
- project_id: Each project has isolated state
- plugin_id: Each plugin has isolated state
- key: Individual state keys within a plugin
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from ..database import DatabaseService

logger = logging.getLogger(__name__)


class PluginStateService:
    """Service for persisting plugin state in SQLite.

    Provides CRUD operations for plugin state with user/project/plugin scoping.
    Values are JSON-serialized for storage, supporting any JSON-compatible type.

    Thread Safety:
        All operations are thread-safe. SQLite handles concurrency at the
        connection level, and we use fresh connections per operation.

    Example:
        >>> service = PluginStateService(database_service)
        >>> service.set("user1", "proj1", "my-plugin", "counter", 42)
        >>> service.get("user1", "proj1", "my-plugin", "counter")
        42
        >>> service.get("user1", "proj1", "my-plugin", "missing", default=0)
        0
    """

    def __init__(self, database_service: DatabaseService):
        """Initialize the plugin state service.

        Args:
            database_service: DatabaseService instance for SQLite access.
        """
        self._db = database_service
        self._lock = threading.Lock()

    def get(
        self,
        user_id: str,
        project_id: str,
        plugin_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a value from plugin state.

        Args:
            user_id: User identifier.
            project_id: Project identifier.
            plugin_id: Plugin identifier.
            key: State key to retrieve.
            default: Value to return if key not found.

        Returns:
            The stored value, or default if not found.
        """
        conn = self._db.connect()
        try:
            cursor = conn.execute(
                """
                SELECT value_json
                FROM plugin_state
                WHERE user_id = ? AND project_id = ? AND plugin_id = ? AND key = ?
                """,
                (user_id, project_id, plugin_id, key),
            )
            row = cursor.fetchone()

            if row is None:
                return default

            try:
                return json.loads(row["value_json"])
            except json.JSONDecodeError:
                logger.warning(
                    f"Invalid JSON in plugin state: {user_id}/{project_id}/{plugin_id}/{key}"
                )
                return default

        except Exception as e:
            logger.error(f"Failed to get plugin state: {e}")
            return default
        finally:
            conn.close()

    def set(
        self,
        user_id: str,
        project_id: str,
        plugin_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Set a value in plugin state.

        Args:
            user_id: User identifier.
            project_id: Project identifier.
            plugin_id: Plugin identifier.
            key: State key to set.
            value: Value to store (must be JSON-serializable).

        Raises:
            ValueError: If value is not JSON-serializable.
        """
        try:
            value_json = json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Value is not JSON-serializable: {e}") from e

        now = datetime.now(timezone.utc).isoformat()

        conn = self._db.connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO plugin_state (user_id, project_id, plugin_id, key, value_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, project_id, plugin_id, key) DO UPDATE SET
                        value_json = excluded.value_json,
                        updated_at = excluded.updated_at
                    """,
                    (user_id, project_id, plugin_id, key, value_json, now, now),
                )
        except Exception as e:
            logger.error(f"Failed to set plugin state: {e}")
            raise
        finally:
            conn.close()

    def get_all(
        self,
        user_id: str,
        project_id: str,
        plugin_id: str,
    ) -> dict[str, Any]:
        """Get all state for a plugin.

        Args:
            user_id: User identifier.
            project_id: Project identifier.
            plugin_id: Plugin identifier.

        Returns:
            Dictionary of all key-value pairs for the plugin.
        """
        conn = self._db.connect()
        try:
            cursor = conn.execute(
                """
                SELECT key, value_json
                FROM plugin_state
                WHERE user_id = ? AND project_id = ? AND plugin_id = ?
                """,
                (user_id, project_id, plugin_id),
            )

            result = {}
            for row in cursor.fetchall():
                try:
                    result[row["key"]] = json.loads(row["value_json"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON in plugin state: {user_id}/{project_id}/{plugin_id}/{row['key']}"
                    )
                    continue

            return result

        except Exception as e:
            logger.error(f"Failed to get all plugin state: {e}")
            return {}
        finally:
            conn.close()

    def clear(
        self,
        user_id: str,
        project_id: str,
        plugin_id: str,
    ) -> None:
        """Clear all state for a plugin.

        Args:
            user_id: User identifier.
            project_id: Project identifier.
            plugin_id: Plugin identifier.
        """
        conn = self._db.connect()
        try:
            with conn:
                conn.execute(
                    """
                    DELETE FROM plugin_state
                    WHERE user_id = ? AND project_id = ? AND plugin_id = ?
                    """,
                    (user_id, project_id, plugin_id),
                )
        except Exception as e:
            logger.error(f"Failed to clear plugin state: {e}")
            raise
        finally:
            conn.close()

    def delete(
        self,
        user_id: str,
        project_id: str,
        plugin_id: str,
        key: str,
    ) -> bool:
        """Delete a specific key from plugin state.

        Args:
            user_id: User identifier.
            project_id: Project identifier.
            plugin_id: Plugin identifier.
            key: State key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        conn = self._db.connect()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    DELETE FROM plugin_state
                    WHERE user_id = ? AND project_id = ? AND plugin_id = ? AND key = ?
                    """,
                    (user_id, project_id, plugin_id, key),
                )
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete plugin state key: {e}")
            return False
        finally:
            conn.close()


# Singleton instance
_plugin_state_service: Optional[PluginStateService] = None


def get_plugin_state_service(
    database_service: Optional[DatabaseService] = None,
) -> PluginStateService:
    """Get the singleton PluginStateService instance.

    Args:
        database_service: Optional DatabaseService to use. If not provided,
                         a default DatabaseService will be created.

    Returns:
        The singleton PluginStateService instance.
    """
    global _plugin_state_service

    if _plugin_state_service is None:
        db = database_service or DatabaseService()
        _plugin_state_service = PluginStateService(db)

    return _plugin_state_service


def reset_plugin_state_service() -> None:
    """Reset the singleton instance (for testing)."""
    global _plugin_state_service
    _plugin_state_service = None


__all__ = [
    "PluginStateService",
    "get_plugin_state_service",
    "reset_plugin_state_service",
]
