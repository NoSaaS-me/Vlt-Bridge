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
        description="Static token accepted in local mode (maps to 'demo-user')",
    )
    chatgpt_service_token: Optional[str] = Field(
        default=None,
        description="Static token for ChatGPT Apps SDK auth",
    )
    chatgpt_cors_origin: str = Field(
        default="https://chatgpt.com",
        description="Allowed CORS origin for ChatGPT",
    )
    enable_noauth_mcp: bool = Field(
        default=False,
        description="DANGEROUS: Allow unauthenticated MCP access as demo-user (for hackathon)",
    )
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key for RAG features"
    )
    llamaindex_persist_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "llamaindex",
        description="Directory for persisting vector index"
    )
    vault_base_path: Path = Field(..., description="Base directory for per-user vaults")
    csp_policy: Optional[str] = Field(
        default=None,
        description=(
            "Content-Security-Policy header value. If None, uses default policy "
            "suitable for Document-MCP application with React frontend"
        )
    )
    enable_hsts: bool = Field(
        default=False,
        description=(
            "Enable Strict-Transport-Security header (HSTS). "
            "Should only be enabled in production HTTPS deployments"
        )
    )
    frame_options: str = Field(
        default="DENY",
        description=(
            "X-Frame-Options header value. Options: DENY, SAMEORIGIN, or "
            "ALLOW-FROM uri. DENY prevents all framing (most secure)"
        )
    )
    admin_user_ids: set[str] = Field(
        default_factory=set,
        description="Set of user IDs with admin privileges (from ADMIN_USER_IDS env var)"
    )
    base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for the application (used for widget asset URLs, OAuth redirects in production)"
    )

    # BT Oracle Configuration (020-bt-oracle-agent, T052)
    oracle_use_bt: str = Field(
        default="false",
        description=(
            "Oracle execution mode for BT-controlled Oracle (ORACLE_USE_BT). Options:\n"
            "  'false'  - Use legacy OracleAgent only (default, safe)\n"
            "  'true'   - Use BT-controlled Oracle exclusively\n"
            "  'shadow' - Run both in parallel, compare outputs, yield legacy\n"
            "Start with 'shadow' to validate BT behavior before switching to 'true'."
        )
    )
    oracle_prompt_budget: int = Field(
        default=8000,
        ge=1000,
        le=100000,
        description=(
            "Token budget for composed Oracle system prompt (ORACLE_PROMPT_BUDGET).\n"
            "This excludes tool schemas. The prompt composer will truncate or\n"
            "omit segments to stay within this budget. Default: 8000 tokens."
        )
    )

    @field_validator("oracle_use_bt", mode="before")
    @classmethod
    def _validate_oracle_mode(cls, value: str) -> str:
        """Validate ORACLE_USE_BT is one of the allowed values."""
        if value is None:
            return "false"
        v = str(value).lower().strip()
        allowed = {"false", "true", "shadow"}
        if v not in allowed:
            raise ValueError(
                f"ORACLE_USE_BT must be one of {allowed}, got: {value!r}"
            )
        return v

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
    enable_local_mode = _read_env("ENABLE_LOCAL_MODE", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    local_dev_token = _read_env("LOCAL_DEV_TOKEN", "local-dev-token")
    chatgpt_service_token = _read_env("CHATGPT_SERVICE_TOKEN")
    chatgpt_cors_origin = _read_env("CHATGPT_CORS_ORIGIN", "https://chatgpt.com")
    enable_noauth_mcp = _read_env("ENABLE_NOAUTH_MCP", "false").lower() in {"true", "1", "yes"}
    google_api_key = _read_env("GOOGLE_API_KEY")
    llamaindex_persist_dir = _read_env("LLAMAINDEX_PERSIST_DIR", str(PROJECT_ROOT / "data" / "llamaindex"))
    csp_policy = _read_env("CSP_POLICY")
    enable_hsts = _read_env("ENABLE_HSTS", "false").lower() in {"true", "1", "yes"}
    frame_options = _read_env("FRAME_OPTIONS", "DENY")

    # Parse admin user IDs from comma-separated list
    admin_user_ids_str = _read_env("ADMIN_USER_IDS", "")
    admin_user_ids = {uid.strip() for uid in admin_user_ids_str.split(",") if uid.strip()} if admin_user_ids_str else set()

    # Base URL for application (for widget URLs, etc.)
    base_url = _read_env("BASE_URL", "http://localhost:8000")

    # BT Oracle configuration (020-bt-oracle-agent, T052)
    oracle_use_bt = _read_env("ORACLE_USE_BT", "false")
    oracle_prompt_budget_str = _read_env("ORACLE_PROMPT_BUDGET", "8000")
    try:
        oracle_prompt_budget = int(oracle_prompt_budget_str)
    except ValueError:
        oracle_prompt_budget = 8000

    config = AppConfig(
        jwt_secret_key=jwt_secret,
        enable_local_mode=enable_local_mode,
        local_dev_token=local_dev_token,
        chatgpt_service_token=chatgpt_service_token,
        chatgpt_cors_origin=chatgpt_cors_origin,
        enable_noauth_mcp=enable_noauth_mcp,
        google_api_key=google_api_key,
        llamaindex_persist_dir=llamaindex_persist_dir,
        vault_base_path=vault_base,
        csp_policy=csp_policy,
        enable_hsts=enable_hsts,
        frame_options=frame_options,
        admin_user_ids=admin_user_ids,
        base_url=base_url,
        # BT Oracle (020-bt-oracle-agent)
        oracle_use_bt=oracle_use_bt,
        oracle_prompt_budget=oracle_prompt_budget,
    )
    # Ensure vault base directory and index persist directory exist for downstream services.
    config.vault_base_path.mkdir(parents=True, exist_ok=True)
    config.llamaindex_persist_dir.mkdir(parents=True, exist_ok=True)
    return config


def reload_config() -> AppConfig:
    """Clear cached config (useful for tests) and reload."""
    get_config.cache_clear()
    return get_config()


# =============================================================================
# Oracle Agent Configuration (T031 - US3: Budget and Loop Enforcement)
# =============================================================================


class OracleConfig(BaseModel):
    """Oracle agent configuration loaded from environment.

    Provides configurable budget limits and warning thresholds for the
    BT-controlled Oracle agent. Per spec FR-007: Configurable max turn limits.

    Environment Variables:
        ORACLE_MAX_TURNS: Maximum agent turns per query (default: 30)
        ORACLE_ITERATION_WARNING_THRESHOLD: Warn at this % of max turns (default: 0.70)
        ORACLE_TOKEN_WARNING_THRESHOLD: Warn at this % of token budget (default: 0.80)
        ORACLE_CONTEXT_WARNING_THRESHOLD: Warn at this % of context window (default: 0.70)
        ORACLE_LOOP_THRESHOLD: Consecutive same-reason count for stuck detection (default: 3)
    """

    model_config = ConfigDict(frozen=True)

    max_turns: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Maximum agent turns per query (ORACLE_MAX_TURNS)"
    )
    iteration_warning_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Warn at this percentage of max turns (0.70 = 70%)"
    )
    token_warning_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Warn at this percentage of token budget"
    )
    context_warning_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Warn at this percentage of context window"
    )
    loop_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive same-reason signals before stuck detection"
    )


@lru_cache(maxsize=1)
def get_oracle_config() -> OracleConfig:
    """Load and cache Oracle agent configuration from environment.

    Returns:
        OracleConfig instance with values from environment or defaults.
    """
    # Parse max_turns
    max_turns_str = _read_env("ORACLE_MAX_TURNS", "30")
    try:
        max_turns = int(max_turns_str)
    except ValueError:
        max_turns = 30

    # Parse iteration warning threshold
    iteration_warning_str = _read_env("ORACLE_ITERATION_WARNING_THRESHOLD", "0.70")
    try:
        iteration_warning = float(iteration_warning_str)
    except ValueError:
        iteration_warning = 0.70

    # Parse token warning threshold
    token_warning_str = _read_env("ORACLE_TOKEN_WARNING_THRESHOLD", "0.80")
    try:
        token_warning = float(token_warning_str)
    except ValueError:
        token_warning = 0.80

    # Parse context warning threshold
    context_warning_str = _read_env("ORACLE_CONTEXT_WARNING_THRESHOLD", "0.70")
    try:
        context_warning = float(context_warning_str)
    except ValueError:
        context_warning = 0.70

    # Parse loop threshold
    loop_threshold_str = _read_env("ORACLE_LOOP_THRESHOLD", "3")
    try:
        loop_threshold = int(loop_threshold_str)
    except ValueError:
        loop_threshold = 3

    return OracleConfig(
        max_turns=max_turns,
        iteration_warning_threshold=iteration_warning,
        token_warning_threshold=token_warning,
        context_warning_threshold=context_warning,
        loop_threshold=loop_threshold,
    )


def reload_oracle_config() -> OracleConfig:
    """Clear cached Oracle config and reload from environment.

    Useful for tests that need to change configuration.

    Returns:
        Fresh OracleConfig instance.
    """
    get_oracle_config.cache_clear()
    return get_oracle_config()


__all__ = [
    "AppConfig", "get_config", "reload_config",
    "OracleConfig", "get_oracle_config", "reload_oracle_config",
    "PROJECT_ROOT", "DEFAULT_VAULT_BASE"
]