"""
vlt-cli Configuration

This module manages CLI configuration via environment variables and ~/.vlt/.env file.

Configuration is loaded from:
1. Environment variables (prefixed with VLT_)
2. ~/.vlt/.env file

Key settings:
- VLT_SYNC_TOKEN: Authentication token for backend server sync
- VLT_VAULT_URL: Backend server URL (default: http://localhost:8000)
- VLT_DATABASE_URL: Local SQLite database path

DEPRECATED (will be removed):
- VLT_OPENROUTER_API_KEY: No longer used - LLM calls are handled server-side
"""

import os
import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """vlt-cli configuration settings."""

    app_name: str = "Vault CLI"

    # Database
    database_url: str = f"sqlite:///{Path.home()}/.vlt/vault.db"

    # Server sync configuration (primary)
    sync_token: str | None = None
    vault_url: str = "http://localhost:8000"

    # Daemon configuration
    daemon_port: int = 8765
    daemon_enabled: bool = True  # Whether to use daemon for sync operations
    daemon_url: str = "http://127.0.0.1:8765"

    # DEPRECATED: OpenRouter settings (kept for backward compatibility)
    # These are no longer used - LLM operations are handled server-side
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "x-ai/grok-4.1-fast"
    openrouter_embedding_model: str = "qwen/qwen3-embedding-8b"

    model_config = SettingsConfigDict(
        env_prefix="VLT_",
        env_file=Path.home() / ".vlt" / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Warn if deprecated OpenRouter key is still set
        if self.openrouter_api_key:
            logger.warning(
                "VLT_OPENROUTER_API_KEY is deprecated. "
                "LLM operations are now handled server-side. "
                "Use 'vlt config set-key <sync-token>' to configure server authentication."
            )

    def get_db_path(self) -> Path:
        """Get the SQLite database file path."""
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", ""))
        return Path.home() / ".vlt" / "vault.db"

    @property
    def is_server_configured(self) -> bool:
        """Check if server sync is properly configured."""
        return bool(self.sync_token)


settings = Settings()

# Ensure the directory exists
db_path = settings.get_db_path()
if not db_path.parent.exists():
    db_path.parent.mkdir(parents=True, exist_ok=True)
