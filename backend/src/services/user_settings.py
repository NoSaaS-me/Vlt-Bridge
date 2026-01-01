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
                       subagent_provider, thinking_enabled, chat_center_mode,
                       librarian_timeout, max_context_nodes, openrouter_api_key
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

                # Handle librarian_timeout - may be None for legacy rows
                librarian_timeout = row["librarian_timeout"] if row["librarian_timeout"] is not None else 1200

                # Handle max_context_nodes - may be None for legacy rows
                max_context_nodes = row["max_context_nodes"] if row["max_context_nodes"] is not None else 30

                return ModelSettings(
                    oracle_model=row["oracle_model"],
                    oracle_provider=ModelProvider(row["oracle_provider"]),
                    subagent_model=row["subagent_model"],
                    subagent_provider=ModelProvider(row["subagent_provider"]),
                    thinking_enabled=bool(row["thinking_enabled"]),
                    chat_center_mode=bool(row["chat_center_mode"]) if row["chat_center_mode"] is not None else False,
                    librarian_timeout=librarian_timeout,
                    max_context_nodes=max_context_nodes,
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

    def get_oracle_model(self, user_id: str) -> str:
        """
        Get user's preferred Oracle model.

        This is used by OracleAgent for the primary model.
        Returns the configured model or a default if not set.

        Args:
            user_id: User identifier

        Returns:
            Model identifier string (e.g., "deepseek/deepseek-chat")
        """
        settings = self.get_settings(user_id)
        return settings.oracle_model or "anthropic/claude-sonnet-4"

    def get_subagent_model(self, user_id: str) -> str:
        """
        Get user's preferred subagent model.

        This is used by OracleAgent when delegating to Librarian.
        Returns the configured model or a default if not set.

        Args:
            user_id: User identifier

        Returns:
            Model identifier string (e.g., "deepseek/deepseek-chat")
        """
        settings = self.get_settings(user_id)
        return settings.subagent_model or "deepseek/deepseek-chat"

    def get_subagent_provider(self, user_id: str) -> "ModelProvider":
        """
        Get user's preferred subagent provider.

        Args:
            user_id: User identifier

        Returns:
            ModelProvider enum value
        """
        settings = self.get_settings(user_id)
        return settings.subagent_provider

    def get_librarian_timeout(self, user_id: str) -> int:
        """
        Get user's configured librarian timeout in seconds.

        This is used by ToolExecutor when running delegate_librarian operations.
        Returns the configured timeout or the default (1200 seconds = 20 minutes).

        Args:
            user_id: User identifier

        Returns:
            Timeout in seconds (60-3600, default 1200)
        """
        settings = self.get_settings(user_id)
        return settings.librarian_timeout

    def set_librarian_timeout(self, user_id: str, timeout_seconds: int) -> None:
        """
        Set user's librarian timeout.

        Args:
            user_id: User identifier
            timeout_seconds: Timeout in seconds (60-3600)
        """
        # Clamp to valid range
        timeout_seconds = max(60, min(timeout_seconds, 3600))
        self.update_settings(user_id=user_id, librarian_timeout=timeout_seconds)

    def get_max_context_nodes(self, user_id: str) -> int:
        """
        Get user's configured max context nodes per tree.

        This is used by ContextTreeService when creating new trees or pruning.
        Returns the configured limit or the default (30 nodes).

        Args:
            user_id: User identifier

        Returns:
            Max nodes per tree (5-100, default 30)
        """
        settings = self.get_settings(user_id)
        return settings.max_context_nodes

    def set_max_context_nodes(self, user_id: str, max_nodes: int) -> None:
        """
        Set user's max context nodes per tree.

        Args:
            user_id: User identifier
            max_nodes: Max nodes per tree (5-100)
        """
        # Clamp to valid range
        max_nodes = max(5, min(max_nodes, 100))
        self.update_settings(user_id=user_id, max_context_nodes=max_nodes)

    def set_subagent_model(self, user_id: str, model: str, provider: Optional["ModelProvider"] = None) -> None:
        """
        Set user's preferred subagent model.

        Args:
            user_id: User identifier
            model: Model identifier (e.g., "deepseek/deepseek-chat")
            provider: Optional provider override (defaults to OPENROUTER)
        """
        self.update_settings(
            user_id=user_id,
            subagent_model=model,
            subagent_provider=provider,
        )

    def update_settings(
        self,
        user_id: str,
        oracle_model: Optional[str] = None,
        oracle_provider: Optional[ModelProvider] = None,
        subagent_model: Optional[str] = None,
        subagent_provider: Optional[ModelProvider] = None,
        thinking_enabled: Optional[bool] = None,
        chat_center_mode: Optional[bool] = None,
        librarian_timeout: Optional[int] = None,
        max_context_nodes: Optional[int] = None,
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
            chat_center_mode: Show AI chat in center view (optional)
            librarian_timeout: Timeout in seconds for Librarian operations (optional, 60-3600)
            max_context_nodes: Max nodes per context tree (optional, 5-100)
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

            # Clamp librarian_timeout to valid range if provided
            new_librarian_timeout = current.librarian_timeout
            if librarian_timeout is not None:
                new_librarian_timeout = max(60, min(librarian_timeout, 3600))

            # Clamp max_context_nodes to valid range if provided
            new_max_context_nodes = current.max_context_nodes
            if max_context_nodes is not None:
                new_max_context_nodes = max(5, min(max_context_nodes, 100))

            # Apply updates (only non-None values)
            updated = ModelSettings(
                oracle_model=oracle_model if oracle_model is not None else current.oracle_model,
                oracle_provider=oracle_provider if oracle_provider is not None else current.oracle_provider,
                subagent_model=subagent_model if subagent_model is not None else current.subagent_model,
                subagent_provider=subagent_provider if subagent_provider is not None else current.subagent_provider,
                thinking_enabled=thinking_enabled if thinking_enabled is not None else current.thinking_enabled,
                chat_center_mode=chat_center_mode if chat_center_mode is not None else current.chat_center_mode,
                librarian_timeout=new_librarian_timeout,
                max_context_nodes=new_max_context_nodes,
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
                        chat_center_mode, librarian_timeout, max_context_nodes,
                        openrouter_api_key, created, updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        oracle_model = excluded.oracle_model,
                        oracle_provider = excluded.oracle_provider,
                        subagent_model = excluded.subagent_model,
                        subagent_provider = excluded.subagent_provider,
                        thinking_enabled = excluded.thinking_enabled,
                        chat_center_mode = excluded.chat_center_mode,
                        librarian_timeout = excluded.librarian_timeout,
                        max_context_nodes = excluded.max_context_nodes,
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
                        int(updated.chat_center_mode),
                        updated.librarian_timeout,
                        updated.max_context_nodes,
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
