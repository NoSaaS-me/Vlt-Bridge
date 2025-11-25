"""Application configuration helpers."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VAULT_BASE = PROJECT_ROOT / "data" / "vaults"


class AppConfig(BaseModel):
    """Runtime configuration loaded from environment variables."""

    model_config = ConfigDict(frozen=True)

    jwt_secret_key: Optional[str] = Field(
        default=None,
        description="HMAC secret for JWT signing (required for JWT/HTTP auth)",
    )
    enable_local_mode: bool = Field(
        default=True,
        description="Allow local-dev token bypass when running locally",
    )
    local_dev_token: Optional[str] = Field(
        default="local-dev-token",
        description="Static token accepted in local mode for development",
    )
    vault_base_path: Path = Field(..., description="Base directory for per-user vaults")
    hf_oauth_client_id: Optional[str] = Field(
        None, description="Hugging Face OAuth client ID (optional)"
    )
    hf_oauth_client_secret: Optional[str] = Field(
        None, description="Hugging Face OAuth client secret (optional)"
    )
    hf_space_url: str = Field(
        default="http://localhost:5173",
        description="Base URL of the HF Space or local dev server"
    )

    @field_validator("vault_base_path", mode="before")
    @classmethod
    def _normalize_vault_path(cls, value: str | Path | None) -> Path:
        if value is None or value == "":
            raise ValueError("VAULT_BASE_PATH is required")
        if isinstance(value, Path):
            path = value
        else:
            path = Path(value)
        return path.expanduser().resolve()

    @field_validator("jwt_secret_key", mode="before")
    @classmethod
    def _ensure_secret(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(
                "JWT_SECRET_KEY cannot be empty; unset the variable to disable JWT auth in local mode"
            )
        if len(cleaned) < 16:
            raise ValueError("JWT_SECRET_KEY must be at least 16 characters")
        return cleaned


def _read_env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache application configuration."""
    jwt_secret = _read_env("JWT_SECRET_KEY")
    vault_base = _read_env("VAULT_BASE_PATH", str(DEFAULT_VAULT_BASE))
    hf_client_id = _read_env("HF_OAUTH_CLIENT_ID")
    hf_client_secret = _read_env("HF_OAUTH_CLIENT_SECRET")
    hf_space_url = _read_env("HF_SPACE_URL", "http://localhost:5173")
    enable_local_mode = _read_env("ENABLE_LOCAL_MODE", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    local_dev_token = _read_env("LOCAL_DEV_TOKEN", "local-dev-token")

    config = AppConfig(
        jwt_secret_key=jwt_secret,
        enable_local_mode=enable_local_mode,
        local_dev_token=local_dev_token,
        vault_base_path=vault_base,
        hf_oauth_client_id=hf_client_id,
        hf_oauth_client_secret=hf_client_secret,
        hf_space_url=hf_space_url,
    )
    # Ensure vault base directory exists for downstream services.
    config.vault_base_path.mkdir(parents=True, exist_ok=True)
    return config


def reload_config() -> AppConfig:
    """Clear cached config (useful for tests) and reload."""
    get_config.cache_clear()
    return get_config()


__all__ = ["AppConfig", "get_config", "reload_config", "PROJECT_ROOT", "DEFAULT_VAULT_BASE"]
