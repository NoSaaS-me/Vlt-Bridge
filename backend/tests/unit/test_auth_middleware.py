"""Tests for authentication middleware dependencies."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.src.api.middleware.auth_middleware import (
    AuthContext,
    AuthMode,
    get_auth_context,
    get_auth_dependency,
    require_admin_context,
    require_auth_context,
)
from backend.src.services import config as config_module
from backend.src.services.auth import AuthService


@pytest.fixture(autouse=True)
def restore_config_cache():
    """Restore config cache after each test."""
    config_module.reload_config()
    yield
    config_module.reload_config()


@pytest.fixture
def auth_service(monkeypatch, tmp_path: Path):
    """Create auth service with a test secret."""
    secret = "test-secret-key-at-least-16-chars"
    monkeypatch.setenv("JWT_SECRET_KEY", secret)
    monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
    cfg = config_module.reload_config()
    return AuthService(config=cfg)


@pytest.fixture
def valid_token(auth_service: AuthService):
    """Generate a valid JWT token for testing."""
    return auth_service.create_jwt("test-user-123")


@pytest.fixture
def expired_token(auth_service: AuthService):
    """Generate an expired JWT token for testing."""
    # Create token that expired 1 hour ago
    return auth_service.create_jwt(
        "test-user-123",
        expires_in=timedelta(hours=-1)
    )


@pytest.fixture
def admin_token(auth_service: AuthService, monkeypatch, tmp_path: Path):
    """Generate a valid JWT token for an admin user."""
    # Set up admin user
    monkeypatch.setenv("ADMIN_USER_IDS", "admin-user-123,another-admin")
    monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
    config_module.reload_config()
    return auth_service.create_jwt("admin-user-123")


class TestGetAuthContext:
    """Tests for get_auth_context (OPTIONAL authentication mode)."""

    def test_valid_jwt_returns_auth_context(self, valid_token: str):
        """Valid JWT should return AuthContext with correct user_id."""
        auth_header = f"Bearer {valid_token}"

        context = get_auth_context(authorization=auth_header)

        assert isinstance(context, AuthContext)
        assert context.user_id == "test-user-123"
        assert context.token == valid_token
        assert context.payload.sub == "test-user-123"

    def test_expired_jwt_raises_401(self, expired_token: str):
        """Expired JWT should raise 401 Unauthorized."""
        auth_header = f"Bearer {expired_token}"

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(authorization=auth_header)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "token_expired"
        assert "expired" in exc_info.value.detail["message"].lower()

    def test_missing_header_with_noauth_disabled_raises_401(self, monkeypatch, tmp_path: Path):
        """Missing header with ENABLE_NOAUTH_MCP=false should raise 401."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(authorization=None)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "unauthorized"
        assert "required" in exc_info.value.detail["message"].lower()

    def test_missing_header_with_noauth_enabled_returns_demo_user(self, monkeypatch, tmp_path: Path):
        """Missing header with ENABLE_NOAUTH_MCP=true should return demo-user."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "true")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        context = get_auth_context(authorization=None)

        assert isinstance(context, AuthContext)
        assert context.user_id == "demo-user"
        assert context.token == "no-auth"
        assert context.payload.sub == "demo-user"

    def test_invalid_header_format_no_bearer_raises_401(self, valid_token: str):
        """Invalid header format (missing 'Bearer') should raise 401."""
        auth_header = valid_token  # Missing "Bearer " prefix

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(authorization=auth_header)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "unauthorized"
        assert "bearer" in exc_info.value.detail["message"].lower()

    def test_invalid_header_format_no_token_raises_401(self):
        """Invalid header format (Bearer without token) should raise 401."""
        auth_header = "Bearer "  # No token after "Bearer "

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(authorization=auth_header)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "unauthorized"
        assert "bearer" in exc_info.value.detail["message"].lower()

    def test_malformed_token_raises_401(self):
        """Malformed JWT token should raise 401."""
        auth_header = "Bearer not-a-valid-jwt-token"

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(authorization=auth_header)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "invalid_token"

    def test_invalid_signature_raises_401(self, monkeypatch, tmp_path: Path):
        """JWT with invalid signature should raise 401."""
        # Create token with one secret
        secret1 = "first-secret-key-at-least-16-chars"
        monkeypatch.setenv("JWT_SECRET_KEY", secret1)
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        cfg1 = config_module.reload_config()
        service1 = AuthService(config=cfg1)
        token = service1.create_jwt("test-user")

        # Try to validate with different secret
        secret2 = "second-secret-key-at-least-16-chars"
        monkeypatch.setenv("JWT_SECRET_KEY", secret2)
        config_module.reload_config()

        auth_header = f"Bearer {token}"

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(authorization=auth_header)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "invalid_token"


class TestRequireAuthContext:
    """Tests for require_auth_context (STRICT authentication mode)."""

    def test_valid_jwt_returns_auth_context(self, valid_token: str):
        """Valid JWT should return AuthContext with correct user_id."""
        auth_header = f"Bearer {valid_token}"

        context = require_auth_context(authorization=auth_header)

        assert isinstance(context, AuthContext)
        assert context.user_id == "test-user-123"
        assert context.token == valid_token
        assert context.payload.sub == "test-user-123"

    def test_expired_jwt_raises_401(self, expired_token: str):
        """Expired JWT should raise 401 Unauthorized."""
        auth_header = f"Bearer {expired_token}"

        with pytest.raises(HTTPException) as exc_info:
            require_auth_context(authorization=auth_header)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "token_expired"

    def test_missing_header_raises_401_regardless_of_noauth(self, monkeypatch, tmp_path: Path):
        """Missing header should raise 401 even with ENABLE_NOAUTH_MCP=true."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "true")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        with pytest.raises(HTTPException) as exc_info:
            require_auth_context(authorization=None)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "unauthorized"
        assert "required" in exc_info.value.detail["message"].lower()

    def test_missing_header_with_noauth_disabled_raises_401(self, monkeypatch, tmp_path: Path):
        """Missing header with ENABLE_NOAUTH_MCP=false should raise 401."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        with pytest.raises(HTTPException) as exc_info:
            require_auth_context(authorization=None)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "unauthorized"

    def test_invalid_header_format_raises_401(self, valid_token: str):
        """Invalid header format should raise 401."""
        auth_header = valid_token  # Missing "Bearer " prefix

        with pytest.raises(HTTPException) as exc_info:
            require_auth_context(authorization=auth_header)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "unauthorized"
        assert "bearer" in exc_info.value.detail["message"].lower()

    def test_malformed_token_raises_401(self):
        """Malformed JWT token should raise 401."""
        auth_header = "Bearer invalid-token"

        with pytest.raises(HTTPException) as exc_info:
            require_auth_context(authorization=auth_header)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "invalid_token"


class TestRequireAdminContext:
    """Tests for require_admin_context (ADMIN authentication mode)."""

    def test_valid_jwt_admin_user_returns_auth_context(self, admin_token: str):
        """Valid JWT for admin user should return AuthContext."""
        auth_header = f"Bearer {admin_token}"

        context = require_admin_context(authorization=auth_header)

        assert isinstance(context, AuthContext)
        assert context.user_id == "admin-user-123"
        assert context.token == admin_token

    def test_valid_jwt_non_admin_user_raises_403(self, valid_token: str, monkeypatch, tmp_path: Path):
        """Valid JWT for non-admin user should raise 403 Forbidden."""
        # Set up admin list that doesn't include test-user-123
        monkeypatch.setenv("ADMIN_USER_IDS", "admin-user-123,another-admin")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        auth_header = f"Bearer {valid_token}"

        with pytest.raises(HTTPException) as exc_info:
            require_admin_context(authorization=auth_header)

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "insufficient_permissions"
        assert "admin" in exc_info.value.detail["message"].lower()

    def test_missing_header_raises_401(self, monkeypatch, tmp_path: Path):
        """Missing authorization header should raise 401."""
        monkeypatch.setenv("ADMIN_USER_IDS", "admin-user-123")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        with pytest.raises(HTTPException) as exc_info:
            require_admin_context(authorization=None)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "unauthorized"

    def test_invalid_token_raises_401(self, monkeypatch, tmp_path: Path):
        """Invalid token should raise 401 before checking admin status."""
        monkeypatch.setenv("ADMIN_USER_IDS", "admin-user-123")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        auth_header = "Bearer invalid-token"

        with pytest.raises(HTTPException) as exc_info:
            require_admin_context(authorization=auth_header)

        # Should fail at authentication, not authorization
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "invalid_token"

    def test_expired_token_raises_401(self, expired_token: str, monkeypatch, tmp_path: Path):
        """Expired token should raise 401."""
        monkeypatch.setenv("ADMIN_USER_IDS", "test-user-123")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        auth_header = f"Bearer {expired_token}"

        with pytest.raises(HTTPException) as exc_info:
            require_admin_context(authorization=auth_header)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "token_expired"

    def test_empty_admin_list_rejects_all_users(self, valid_token: str, monkeypatch, tmp_path: Path):
        """Empty admin list should reject all users with 403."""
        monkeypatch.setenv("ADMIN_USER_IDS", "")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        auth_header = f"Bearer {valid_token}"

        with pytest.raises(HTTPException) as exc_info:
            require_admin_context(authorization=auth_header)

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "insufficient_permissions"


class TestGetAuthDependency:
    """Tests for get_auth_dependency factory function."""

    def test_optional_mode_returns_get_auth_context(self):
        """OPTIONAL mode should return get_auth_context."""
        dependency = get_auth_dependency(AuthMode.OPTIONAL)
        assert dependency == get_auth_context

    def test_strict_mode_returns_require_auth_context(self):
        """STRICT mode should return require_auth_context."""
        dependency = get_auth_dependency(AuthMode.STRICT)
        assert dependency == require_auth_context

    def test_admin_mode_returns_require_admin_context(self):
        """ADMIN mode should return require_admin_context."""
        dependency = get_auth_dependency(AuthMode.ADMIN)
        assert dependency == require_admin_context

    def test_invalid_mode_raises_value_error(self):
        """Invalid auth mode should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_auth_dependency("invalid_mode")  # type: ignore

        assert "unknown auth mode" in str(exc_info.value).lower()


class TestAuthContextDataclass:
    """Tests for AuthContext dataclass."""

    def test_auth_context_creation(self, valid_token: str, auth_service: AuthService):
        """AuthContext should be created with all required fields."""
        payload = auth_service.validate_jwt(valid_token)

        context = AuthContext(
            user_id="test-user-123",
            token=valid_token,
            payload=payload
        )

        assert context.user_id == "test-user-123"
        assert context.token == valid_token
        assert context.payload.sub == "test-user-123"
        assert isinstance(context.payload.iat, int)
        assert isinstance(context.payload.exp, int)
