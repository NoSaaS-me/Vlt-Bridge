"""
VLT-CLI Configuration

This module manages CLI configuration via environment variables and profile-specific .env files.

Configuration is loaded from (in order of precedence):
1. Environment variables (prefixed with VLT_)
2. Profile-specific .env file (~/.vlt/profiles/{profile}/.env)

Key settings:
- VLT_SYNC_TOKEN: Authentication token for backend server sync
- VLT_VAULT_URL: Backend server URL (default: http://localhost:8000)
- VLT_DATABASE_URL: Local SQLite database path (auto-configured per profile)
- VLT_PROFILE: Override active profile (env var only)

Profile System:
Each profile has isolated storage in ~/.vlt/profiles/{profile}/:
- vault.db: SQLite database
- .env: Profile-specific environment variables
- daemon.pid: Daemon process ID file
- daemon.log: Daemon log file

Oracle Thin Client Mode:
When VLT_SYNC_TOKEN is set and backend is available, the CLI uses thin client mode:
- Oracle queries are sent to the backend API instead of running locally
- Context is shared with the web UI (same conversation tree)
- LLM API keys are managed server-side

Local Mode (fallback):
When backend is unavailable or --local flag is used:
- Uses local OracleOrchestrator with local indexes
- Requires VLT_OPENROUTER_API_KEY for LLM synthesis

DEPRECATED (will be removed):
- VLT_OPENROUTER_API_KEY: No longer used - LLM calls are handled server-side
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _get_profile_env_file() -> Path:
    """
    Get the .env file path for the current profile.

    This is called during Settings initialization to determine which .env file to load.
    We import ProfileManager here to avoid circular imports.
    """
    # Import here to avoid circular dependency
    from vlt.profile import get_profile_manager, ensure_profile_migrated

    # Ensure legacy data is migrated before loading settings
    ensure_profile_migrated()

    manager = get_profile_manager()
    return manager.get_env_file()


def _get_profile_database_url() -> str:
    """
    Get the database URL for the current profile.

    Returns the SQLite URL pointing to the profile's vault.db.
    """
    # Import here to avoid circular dependency
    from vlt.profile import get_profile_manager, ensure_profile_migrated

    ensure_profile_migrated()
    manager = get_profile_manager()
    return manager.get_database_url()


def _get_profile_daemon_port() -> int:
    """
    Get the daemon port for the current profile.

    Each profile can have its own daemon on a unique port.
    """
    # Import here to avoid circular dependency
    from vlt.profile import get_profile_manager

    manager = get_profile_manager()
    return manager.get_daemon_port()


class Settings(BaseSettings):
    """
    VLT-CLI configuration settings.

    Settings are loaded from environment variables and profile-specific .env file.
    The active profile is determined by VLT_PROFILE env var, project .vlt/profile file,
    or ~/.vlt/config.toml (see ProfileManager for details).
    """

    app_name: str = "Vault CLI"

    # Database - will be overridden with profile-specific path
    database_url: str = ""

    # Server sync configuration (primary)
    sync_token: Optional[str] = None
    vault_url: str = "http://localhost:8000"

    # Oracle configuration
    oracle_timeout: float = 60.0  # Request timeout for Oracle queries
    oracle_prefer_backend: bool = True  # Prefer backend when available

    # Daemon configuration - will be overridden with profile-specific values
    daemon_port: int = 8765
    daemon_enabled: bool = True  # Whether to use daemon for sync operations
    daemon_url: str = ""  # Will be computed from daemon_port

    # Profile name (can only be set via VLT_PROFILE env var, not .env file)
    # This is informational - actual profile is determined by ProfileManager
    profile: Optional[str] = None

    # DEPRECATED: OpenRouter settings (kept for backward compatibility)
    # These are no longer used - LLM operations are handled server-side
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "x-ai/grok-4.1-fast"
    openrouter_embedding_model: str = "qwen/qwen3-embedding-8b"

    model_config = SettingsConfigDict(
        env_prefix="VLT_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, profile_name: Optional[str] = None, **kwargs):
        """
        Initialize settings with profile-aware configuration.

        Args:
            profile_name: Override profile name (for --profile flag).
                          If not provided, profile is determined by ProfileManager.
            **kwargs: Additional settings overrides
        """
        # Handle profile override via environment variable or explicit parameter
        if profile_name:
            os.environ["VLT_PROFILE"] = profile_name

        # Determine the env file path based on active profile
        try:
            env_file = _get_profile_env_file()
        except Exception as e:
            logger.warning(f"Could not determine profile env file: {e}")
            env_file = Path.home() / ".vlt" / ".env"  # Fallback

        # Update model config with the correct env file
        # Note: We can't modify model_config at runtime, so we read it ourselves
        if env_file.exists():
            self._load_env_file(env_file)

        super().__init__(**kwargs)

        # Set profile-specific database URL if not explicitly provided
        if not self.database_url or self.database_url == "":
            try:
                self.database_url = _get_profile_database_url()
            except Exception as e:
                logger.warning(f"Could not get profile database URL: {e}")
                self.database_url = f"sqlite:///{Path.home()}/.vlt/vault.db"

        # Set profile-specific daemon port if not explicitly provided
        if self.daemon_port == 8765:  # Default value
            try:
                self.daemon_port = _get_profile_daemon_port()
            except Exception:
                pass  # Keep default

        # Compute daemon URL from port
        if not self.daemon_url:
            self.daemon_url = f"http://127.0.0.1:{self.daemon_port}"

        # Warn if deprecated OpenRouter key is still set
        if self.openrouter_api_key:
            logger.warning(
                "VLT_OPENROUTER_API_KEY is deprecated. "
                "LLM operations are now handled server-side. "
                "Use 'vlt config set-key <sync-token>' to configure server authentication."
            )

    def _load_env_file(self, env_file: Path) -> None:
        """
        Load environment variables from a .env file.

        Only sets variables that are not already set in the environment.

        Args:
            env_file: Path to the .env file
        """
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
        except Exception as e:
            logger.debug(f"Error loading env file {env_file}: {e}")

    def get_db_path(self) -> Path:
        """Get the SQLite database file path."""
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", ""))
        return Path.home() / ".vlt" / "vault.db"

    @property
    def is_server_configured(self) -> bool:
        """Check if server sync is properly configured."""
        return bool(self.sync_token)

    @property
    def backend_url(self) -> str:
        """Get the backend URL (alias for vault_url)."""
        return self.vault_url

    @property
    def can_use_backend_oracle(self) -> bool:
        """Check if backend Oracle can be used (token + prefer backend)."""
        return self.is_server_configured and self.oracle_prefer_backend


# Global settings instance - will be created lazily
_settings: Optional[Settings] = None


def get_settings(profile_name: Optional[str] = None, force_reload: bool = False) -> Settings:
    """
    Get the global Settings instance.

    Args:
        profile_name: Override the active profile (for --profile flag)
        force_reload: Force reload settings (useful after profile switch)

    Returns:
        Settings instance
    """
    global _settings

    if _settings is None or force_reload or profile_name:
        _settings = Settings(profile_name=profile_name)

    return _settings


def reload_settings(profile_name: Optional[str] = None) -> Settings:
    """
    Reload settings, optionally with a different profile.

    This clears the profile cache and reloads everything.

    Args:
        profile_name: Optional profile to switch to

    Returns:
        New Settings instance
    """
    # Clear profile manager cache
    from vlt.profile import get_profile_manager
    get_profile_manager().clear_cache()

    return get_settings(profile_name=profile_name, force_reload=True)


# Legacy compatibility - create settings on import
# This maintains backward compatibility with code that imports `settings` directly
try:
    settings = get_settings()
except Exception as e:
    # Fallback for import-time errors
    logger.warning(f"Failed to load settings: {e}. Using defaults.")
    settings = Settings()

# Ensure the profile directory exists
try:
    db_path = settings.get_db_path()
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.warning(f"Failed to create database directory: {e}")
