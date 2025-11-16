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

    jwt_secret_key: str = Field(..., min_length=16, description="HMAC secret for JWT signing")
    vault_base_path: Path = Field(..., description="Base directory for per-user vaults")
    hf_oauth_client_id: Optional[str] = Field(
        None, description="Hugging Face OAuth client ID (optional)"
    )
    hf_oauth_client_secret: Optional[str] = Field(
        None, description="Hugging Face OAuth client secret (optional)"
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

    @field_validator("jwt_secret_key")
    @classmethod
    def _ensure_secret(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("JWT_SECRET_KEY is required")
        return value.strip()


def _read_env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache application configuration."""
    jwt_secret = _read_env("JWT_SECRET_KEY")
    vault_base = _read_env("VAULT_BASE_PATH", str(DEFAULT_VAULT_BASE))
    hf_client_id = _read_env("HF_OAUTH_CLIENT_ID")
    hf_client_secret = _read_env("HF_OAUTH_CLIENT_SECRET")

    config = AppConfig(
        jwt_secret_key=jwt_secret,
        vault_base_path=vault_base,
        hf_oauth_client_id=hf_client_id,
        hf_oauth_client_secret=hf_client_secret,
    )
    # Ensure vault base directory exists for downstream services.
    config.vault_base_path.mkdir(parents=True, exist_ok=True)
    return config


def reload_config() -> AppConfig:
    """Clear cached config (useful for tests) and reload."""
    get_config.cache_clear()
    return get_config()


__all__ = ["AppConfig", "get_config", "reload_config", "PROJECT_ROOT", "DEFAULT_VAULT_BASE"]
