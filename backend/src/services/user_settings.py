"""Service for managing user settings in the database."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from ..models.settings import ModelSettings, ModelProvider
from .database import DatabaseService

logger = logging.getLogger(__name__)


class UserSettingsService:
    """Service for reading and writing user settings."""

    def __init__(self, db_service: Optional[DatabaseService] = None):
        """Initialize with optional database service."""
        self.db = db_service or DatabaseService()

    def get_settings(self, user_id: str) -> ModelSettings:
        """
        Get user's model settings.

        Args:
            user_id: User identifier

        Returns:
            ModelSettings object (returns defaults if not found)
            Note: openrouter_api_key is NOT returned for security, only openrouter_api_key_set flag
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                """
                SELECT oracle_model, oracle_provider, subagent_model,
                       subagent_provider, thinking_enabled, openrouter_api_key
                FROM user_settings
                WHERE user_id = ?
                """,
                (user_id,)
            )
            row = cursor.fetchone()

            if row:
                # Check if API key is set (but don't return the actual key)
                api_key = row["openrouter_api_key"]
                has_api_key = api_key is not None and len(api_key) > 0

                return ModelSettings(
                    oracle_model=row["oracle_model"],
                    oracle_provider=ModelProvider(row["oracle_provider"]),
                    subagent_model=row["subagent_model"],
                    subagent_provider=ModelProvider(row["subagent_provider"]),
                    thinking_enabled=bool(row["thinking_enabled"]),
                    openrouter_api_key=None,  # Never return the actual key
                    openrouter_api_key_set=has_api_key
                )
            else:
                # Return defaults for new users
                return ModelSettings()

        except Exception as e:
            logger.error(f"Failed to get settings for user {user_id}: {e}")
            # Return defaults on error
            return ModelSettings()
        finally:
            conn.close()

    def get_openrouter_api_key(self, user_id: str) -> Optional[str]:
        """
        Get user's OpenRouter API key (for internal use only).

        Args:
            user_id: User identifier

        Returns:
            The API key or None if not set
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                "SELECT openrouter_api_key FROM user_settings WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row and row["openrouter_api_key"]:
                return row["openrouter_api_key"]
            return None
        except Exception as e:
            logger.error(f"Failed to get API key for user {user_id}: {e}")
            return None
        finally:
            conn.close()

    def update_settings(
        self,
        user_id: str,
        oracle_model: Optional[str] = None,
        oracle_provider: Optional[ModelProvider] = None,
        subagent_model: Optional[str] = None,
        subagent_provider: Optional[ModelProvider] = None,
        thinking_enabled: Optional[bool] = None,
        openrouter_api_key: Optional[str] = None
    ) -> ModelSettings:
        """
        Update user's model settings.

        Args:
            user_id: User identifier
            oracle_model: Oracle model ID (optional)
            oracle_provider: Oracle provider (optional)
            subagent_model: Subagent model ID (optional)
            subagent_provider: Subagent provider (optional)
            thinking_enabled: Enable thinking mode (optional)
            openrouter_api_key: OpenRouter API key (optional, empty string to clear)

        Returns:
            Updated ModelSettings object
        """
        conn = self.db.connect()
        try:
            # Get current settings
            current = self.get_settings(user_id)

            # Get current API key (not returned by get_settings for security)
            current_api_key = self.get_openrouter_api_key(user_id)

            # Determine new API key value
            # None = don't change, "" = clear, otherwise = set new key
            if openrouter_api_key is None:
                new_api_key = current_api_key
            elif openrouter_api_key == "":
                new_api_key = None  # Clear the key
            else:
                new_api_key = openrouter_api_key

            # Apply updates (only non-None values)
            updated = ModelSettings(
                oracle_model=oracle_model if oracle_model is not None else current.oracle_model,
                oracle_provider=oracle_provider if oracle_provider is not None else current.oracle_provider,
                subagent_model=subagent_model if subagent_model is not None else current.subagent_model,
                subagent_provider=subagent_provider if subagent_provider is not None else current.subagent_provider,
                thinking_enabled=thinking_enabled if thinking_enabled is not None else current.thinking_enabled,
                openrouter_api_key=None,  # Never return the key
                openrouter_api_key_set=new_api_key is not None and len(new_api_key) > 0
            )

            now = datetime.now(timezone.utc).isoformat()

            # Upsert settings
            with conn:
                conn.execute(
                    """
                    INSERT INTO user_settings (
                        user_id, oracle_model, oracle_provider,
                        subagent_model, subagent_provider, thinking_enabled,
                        openrouter_api_key, created, updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        oracle_model = excluded.oracle_model,
                        oracle_provider = excluded.oracle_provider,
                        subagent_model = excluded.subagent_model,
                        subagent_provider = excluded.subagent_provider,
                        thinking_enabled = excluded.thinking_enabled,
                        openrouter_api_key = excluded.openrouter_api_key,
                        updated = excluded.updated
                    """,
                    (
                        user_id,
                        updated.oracle_model,
                        updated.oracle_provider.value,
                        updated.subagent_model,
                        updated.subagent_provider.value,
                        int(updated.thinking_enabled),
                        new_api_key,
                        now,
                        now
                    )
                )

            logger.info(f"Updated settings for user {user_id}")
            return updated

        except Exception as e:
            logger.error(f"Failed to update settings for user {user_id}: {e}")
            raise
        finally:
            conn.close()


def get_user_settings_service() -> UserSettingsService:
    """Get instance of UserSettingsService."""
    return UserSettingsService()
