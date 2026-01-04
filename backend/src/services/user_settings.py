"""Service for managing user settings in the database."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from ..models.settings import ModelSettings, ModelProvider, ReasoningEffort
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
                       subagent_provider, thinking_enabled, reasoning_effort,
                       chat_center_mode, librarian_timeout, max_context_nodes,
                       openrouter_api_key, max_iterations, soft_warning_percent,
                       token_budget, token_warning_percent, timeout_seconds,
                       max_tool_calls_per_turn, max_parallel_tools
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

                # Handle AgentConfig fields - may be None for legacy rows
                max_iterations = row["max_iterations"] if row["max_iterations"] is not None else 15
                soft_warning_percent = row["soft_warning_percent"] if row["soft_warning_percent"] is not None else 70
                token_budget = row["token_budget"] if row["token_budget"] is not None else 50000
                token_warning_percent = row["token_warning_percent"] if row["token_warning_percent"] is not None else 80
                timeout_seconds = row["timeout_seconds"] if row["timeout_seconds"] is not None else 120
                max_tool_calls_per_turn = row["max_tool_calls_per_turn"] if row["max_tool_calls_per_turn"] is not None else 100
                max_parallel_tools = row["max_parallel_tools"] if row["max_parallel_tools"] is not None else 3

                # Handle reasoning_effort - may be None for legacy rows
                reasoning_effort_str = row["reasoning_effort"] if row["reasoning_effort"] is not None else "medium"
                reasoning_effort = ReasoningEffort(reasoning_effort_str)

                return ModelSettings(
                    oracle_model=row["oracle_model"],
                    oracle_provider=ModelProvider(row["oracle_provider"]),
                    subagent_model=row["subagent_model"],
                    subagent_provider=ModelProvider(row["subagent_provider"]),
                    thinking_enabled=bool(row["thinking_enabled"]),
                    reasoning_effort=reasoning_effort,
                    chat_center_mode=bool(row["chat_center_mode"]) if row["chat_center_mode"] is not None else False,
                    librarian_timeout=librarian_timeout,
                    max_context_nodes=max_context_nodes,
                    openrouter_api_key=None,  # Never return the actual key
                    openrouter_api_key_set=has_api_key,
                    max_iterations=max_iterations,
                    soft_warning_percent=soft_warning_percent,
                    token_budget=token_budget,
                    token_warning_percent=token_warning_percent,
                    timeout_seconds=timeout_seconds,
                    max_tool_calls_per_turn=max_tool_calls_per_turn,
                    max_parallel_tools=max_parallel_tools,
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

    def get_reasoning_effort(self, user_id: str) -> str:
        """
        Get user's configured reasoning effort level.

        This is used when making LLM calls to set the reasoning_effort parameter
        for models that support extended thinking.

        Args:
            user_id: The user's ID

        Returns:
            Reasoning effort level: "low", "medium", or "high"
        """
        settings = self.get_settings(user_id)
        return settings.reasoning_effort.value

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
        reasoning_effort: Optional[ReasoningEffort] = None,
        chat_center_mode: Optional[bool] = None,
        librarian_timeout: Optional[int] = None,
        max_context_nodes: Optional[int] = None,
        openrouter_api_key: Optional[str] = None,
        max_iterations: Optional[int] = None,
        soft_warning_percent: Optional[int] = None,
        token_budget: Optional[int] = None,
        token_warning_percent: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_tool_calls_per_turn: Optional[int] = None,
        max_parallel_tools: Optional[int] = None,
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
            reasoning_effort: Reasoning effort level (optional, low/medium/high)
            chat_center_mode: Show AI chat in center view (optional)
            librarian_timeout: Timeout in seconds for Librarian operations (optional, 60-3600)
            max_context_nodes: Max nodes per context tree (optional, 5-100)
            openrouter_api_key: OpenRouter API key (optional, empty string to clear)
            max_iterations: Maximum agent turns per query (optional, 1-50)
            soft_warning_percent: Percentage of max iterations to trigger warning (optional, 50-90)
            token_budget: Maximum tokens per session (optional, 1000-200000)
            token_warning_percent: Percentage of token budget to trigger warning (optional, 50-95)
            timeout_seconds: Overall query timeout in seconds (optional, 10-600)
            max_tool_calls_per_turn: Maximum tool calls per agent turn (optional, 1-200)
            max_parallel_tools: Maximum concurrent tool executions (optional, 1-10)

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

            # Clamp AgentConfig fields to valid ranges if provided
            new_max_iterations = current.max_iterations
            if max_iterations is not None:
                new_max_iterations = max(1, min(max_iterations, 50))

            new_soft_warning_percent = current.soft_warning_percent
            if soft_warning_percent is not None:
                new_soft_warning_percent = max(50, min(soft_warning_percent, 90))

            new_token_budget = current.token_budget
            if token_budget is not None:
                new_token_budget = max(1000, min(token_budget, 200000))

            new_token_warning_percent = current.token_warning_percent
            if token_warning_percent is not None:
                new_token_warning_percent = max(50, min(token_warning_percent, 95))

            new_timeout_seconds = current.timeout_seconds
            if timeout_seconds is not None:
                new_timeout_seconds = max(10, min(timeout_seconds, 600))

            new_max_tool_calls_per_turn = current.max_tool_calls_per_turn
            if max_tool_calls_per_turn is not None:
                new_max_tool_calls_per_turn = max(1, min(max_tool_calls_per_turn, 200))

            new_max_parallel_tools = current.max_parallel_tools
            if max_parallel_tools is not None:
                new_max_parallel_tools = max(1, min(max_parallel_tools, 10))

            # Apply updates (only non-None values)
            updated = ModelSettings(
                oracle_model=oracle_model if oracle_model is not None else current.oracle_model,
                oracle_provider=oracle_provider if oracle_provider is not None else current.oracle_provider,
                subagent_model=subagent_model if subagent_model is not None else current.subagent_model,
                subagent_provider=subagent_provider if subagent_provider is not None else current.subagent_provider,
                thinking_enabled=thinking_enabled if thinking_enabled is not None else current.thinking_enabled,
                reasoning_effort=reasoning_effort if reasoning_effort is not None else current.reasoning_effort,
                chat_center_mode=chat_center_mode if chat_center_mode is not None else current.chat_center_mode,
                librarian_timeout=new_librarian_timeout,
                max_context_nodes=new_max_context_nodes,
                openrouter_api_key=None,  # Never return the key
                openrouter_api_key_set=new_api_key is not None and len(new_api_key) > 0,
                max_iterations=new_max_iterations,
                soft_warning_percent=new_soft_warning_percent,
                token_budget=new_token_budget,
                token_warning_percent=new_token_warning_percent,
                timeout_seconds=new_timeout_seconds,
                max_tool_calls_per_turn=new_max_tool_calls_per_turn,
                max_parallel_tools=new_max_parallel_tools,
            )

            now = datetime.now(timezone.utc).isoformat()

            # Upsert settings
            with conn:
                conn.execute(
                    """
                    INSERT INTO user_settings (
                        user_id, oracle_model, oracle_provider,
                        subagent_model, subagent_provider, thinking_enabled,
                        reasoning_effort, chat_center_mode, librarian_timeout,
                        max_context_nodes, openrouter_api_key, max_iterations,
                        soft_warning_percent, token_budget, token_warning_percent,
                        timeout_seconds, max_tool_calls_per_turn, max_parallel_tools,
                        created, updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        oracle_model = excluded.oracle_model,
                        oracle_provider = excluded.oracle_provider,
                        subagent_model = excluded.subagent_model,
                        subagent_provider = excluded.subagent_provider,
                        thinking_enabled = excluded.thinking_enabled,
                        reasoning_effort = excluded.reasoning_effort,
                        chat_center_mode = excluded.chat_center_mode,
                        librarian_timeout = excluded.librarian_timeout,
                        max_context_nodes = excluded.max_context_nodes,
                        openrouter_api_key = excluded.openrouter_api_key,
                        max_iterations = excluded.max_iterations,
                        soft_warning_percent = excluded.soft_warning_percent,
                        token_budget = excluded.token_budget,
                        token_warning_percent = excluded.token_warning_percent,
                        timeout_seconds = excluded.timeout_seconds,
                        max_tool_calls_per_turn = excluded.max_tool_calls_per_turn,
                        max_parallel_tools = excluded.max_parallel_tools,
                        updated = excluded.updated
                    """,
                    (
                        user_id,
                        updated.oracle_model,
                        updated.oracle_provider.value,
                        updated.subagent_model,
                        updated.subagent_provider.value,
                        int(updated.thinking_enabled),
                        updated.reasoning_effort.value,
                        int(updated.chat_center_mode),
                        updated.librarian_timeout,
                        updated.max_context_nodes,
                        new_api_key,
                        updated.max_iterations,
                        updated.soft_warning_percent,
                        updated.token_budget,
                        updated.token_warning_percent,
                        updated.timeout_seconds,
                        updated.max_tool_calls_per_turn,
                        updated.max_parallel_tools,
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

    def get_agent_config(self, user_id: str) -> "AgentConfig":
        """
        Get user's agent configuration settings.

        Args:
            user_id: User identifier

        Returns:
            AgentConfig object with turn control settings
        """
        from ..models.settings import AgentConfig

        settings = self.get_settings(user_id)
        config = AgentConfig(
            max_iterations=settings.max_iterations,
            soft_warning_percent=settings.soft_warning_percent,
            token_budget=settings.token_budget,
            token_warning_percent=settings.token_warning_percent,
            timeout_seconds=settings.timeout_seconds,
            max_tool_calls_per_turn=settings.max_tool_calls_per_turn,
            max_parallel_tools=settings.max_parallel_tools,
        )
        logger.debug(
            f"[AGENT_CONFIG] Loaded for {user_id}: "
            f"max_tool_calls_per_turn={config.max_tool_calls_per_turn}, "
            f"max_iterations={config.max_iterations}, "
            f"max_parallel_tools={config.max_parallel_tools}"
        )
        return config

    def update_agent_config(
        self,
        user_id: str,
        max_iterations: Optional[int] = None,
        soft_warning_percent: Optional[int] = None,
        token_budget: Optional[int] = None,
        token_warning_percent: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_tool_calls_per_turn: Optional[int] = None,
        max_parallel_tools: Optional[int] = None,
    ) -> "AgentConfig":
        """
        Update user's agent configuration.

        Args:
            user_id: User identifier
            max_iterations: Maximum agent turns per query (1-50)
            soft_warning_percent: Percentage of max iterations to trigger warning (50-90)
            token_budget: Maximum tokens per session (1000-200000)
            token_warning_percent: Percentage of token budget to trigger warning (50-95)
            timeout_seconds: Overall query timeout in seconds (10-600)
            max_tool_calls_per_turn: Maximum tool calls per agent turn (1-200)
            max_parallel_tools: Maximum concurrent tool executions (1-10)

        Returns:
            Updated AgentConfig object
        """
        from ..models.settings import AgentConfig

        # Get current settings
        current = self.get_agent_config(user_id)

        # Apply updates with validation (clamping to bounds)
        def clamp(value: Optional[int], min_val: int, max_val: int, default: int) -> int:
            if value is None:
                return default
            return max(min_val, min(value, max_val))

        new_config = AgentConfig(
            max_iterations=clamp(max_iterations, 1, 50, current.max_iterations),
            soft_warning_percent=clamp(soft_warning_percent, 50, 90, current.soft_warning_percent),
            token_budget=clamp(token_budget, 1000, 200000, current.token_budget),
            token_warning_percent=clamp(token_warning_percent, 50, 95, current.token_warning_percent),
            timeout_seconds=clamp(timeout_seconds, 10, 600, current.timeout_seconds),
            max_tool_calls_per_turn=clamp(max_tool_calls_per_turn, 1, 200, current.max_tool_calls_per_turn),
            max_parallel_tools=clamp(max_parallel_tools, 1, 10, current.max_parallel_tools),
        )

        # Call existing update_settings to persist
        self.update_settings(
            user_id=user_id,
            max_iterations=new_config.max_iterations,
            soft_warning_percent=new_config.soft_warning_percent,
            token_budget=new_config.token_budget,
            token_warning_percent=new_config.token_warning_percent,
            timeout_seconds=new_config.timeout_seconds,
            max_tool_calls_per_turn=new_config.max_tool_calls_per_turn,
            max_parallel_tools=new_config.max_parallel_tools,
        )

        return new_config


def get_user_settings_service() -> UserSettingsService:
    """Get instance of UserSettingsService."""
    return UserSettingsService()
