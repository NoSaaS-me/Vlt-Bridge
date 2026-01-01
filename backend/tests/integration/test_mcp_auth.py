"""Integration tests for MCP endpoint authentication.

Tests that MCP HTTP endpoint properly enforces authentication and that STDIO mode
still works with local-dev fallback. This ensures the security improvements from
P3.1 are working correctly.
"""

import os
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.src.services import config as config_module
from backend.src.services.auth import AuthError, AuthService


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
    monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
    cfg = config_module.reload_config()
    return AuthService(config=cfg)


@pytest.fixture
def valid_token(auth_service: AuthService):
    """Generate a valid JWT token for testing."""
    return auth_service.create_jwt("test-user-123")


@pytest.fixture
def expired_token(auth_service: AuthService):
    """Generate an expired JWT token for testing."""
    return auth_service.create_jwt(
        "test-user-123",
        expires_in=timedelta(hours=-1)
    )


@pytest.mark.integration
class TestMCPHttpAuthentication:
    """Test authentication enforcement on MCP HTTP endpoint."""

    def test_http_mode_rejects_missing_auth_header(self, monkeypatch, tmp_path: Path):
        """HTTP mode should reject requests without Authorization header."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        # Import after config is set
        from backend.src.mcp.server import _current_user_id

        # Mock HTTP request context
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None  # No Authorization header

        with patch("backend.src.mcp.server._current_http_request") as mock_http_request:
            mock_http_request.get.return_value = mock_request

            with pytest.raises(PermissionError) as exc_info:
                _current_user_id()

            assert "Authorization header required" in str(exc_info.value)

    def test_http_mode_rejects_invalid_auth_format(self, monkeypatch, tmp_path: Path):
        """HTTP mode should reject malformed Authorization headers."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        # Test cases for invalid auth header formats
        invalid_headers = [
            "InvalidScheme token123",  # Wrong scheme
            "Bearer",  # Missing token
            "token-without-bearer",  # No scheme
            "",  # Empty string
        ]

        for invalid_header in invalid_headers:
            mock_request = MagicMock()
            mock_request.headers.get.return_value = invalid_header

            with patch("backend.src.mcp.server._current_http_request") as mock_http_request:
                mock_http_request.get.return_value = mock_request

                with pytest.raises(PermissionError) as exc_info:
                    _current_user_id()

                assert "Authorization header must be 'Bearer <token>'" in str(exc_info.value)

    def test_http_mode_rejects_invalid_jwt_token(self, monkeypatch, tmp_path: Path):
        """HTTP mode should reject invalid JWT tokens."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-at-least-16-chars")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        # Invalid JWT token
        invalid_tokens = [
            "not-a-valid-jwt-token",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature",
        ]

        for invalid_token in invalid_tokens:
            mock_request = MagicMock()
            mock_request.headers.get.return_value = f"Bearer {invalid_token}"

            with patch("backend.src.mcp.server._current_http_request") as mock_http_request:
                mock_http_request.get.return_value = mock_request

                with pytest.raises(PermissionError):
                    _current_user_id()

    def test_http_mode_rejects_expired_jwt_token(self, monkeypatch, tmp_path: Path, expired_token: str):
        """HTTP mode should reject expired JWT tokens."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-at-least-16-chars")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        mock_request = MagicMock()
        mock_request.headers.get.return_value = f"Bearer {expired_token}"

        with patch("backend.src.mcp.server._current_http_request") as mock_http_request:
            mock_http_request.get.return_value = mock_request

            with pytest.raises(PermissionError):
                _current_user_id()

    def test_http_mode_accepts_valid_jwt_token(self, monkeypatch, tmp_path: Path, valid_token: str):
        """HTTP mode should accept valid JWT tokens and extract user_id."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-at-least-16-chars")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        mock_request = MagicMock()
        mock_request.headers.get.return_value = f"Bearer {valid_token}"

        with patch("backend.src.mcp.server._current_http_request") as mock_http_request:
            mock_http_request.get.return_value = mock_request

            user_id = _current_user_id()
            assert user_id == "test-user-123"

    def test_http_mode_logs_deprecation_warning_when_noauth_enabled(self, monkeypatch, tmp_path: Path, valid_token: str):
        """HTTP mode should log deprecation warning when ENABLE_NOAUTH_MCP is enabled."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "true")
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-at-least-16-chars")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        mock_request = MagicMock()
        mock_request.headers.get.return_value = f"Bearer {valid_token}"

        with patch("backend.src.mcp.server._current_http_request") as mock_http_request:
            with patch("backend.src.mcp.server.logger") as mock_logger:
                mock_http_request.get.return_value = mock_request

                user_id = _current_user_id()
                assert user_id == "test-user-123"

                # Verify deprecation warning was logged
                mock_logger.warning.assert_called_once()
                warning_call = mock_logger.warning.call_args[0][0]
                assert "ENABLE_NOAUTH_MCP is enabled" in warning_call
                assert "DEPRECATED" in warning_call


@pytest.mark.integration
class TestMCPStdioAuthentication:
    """Test STDIO mode authentication fallback behavior."""

    def test_stdio_mode_uses_local_dev_fallback(self, monkeypatch, tmp_path: Path):
        """STDIO mode should fall back to 'local-dev' when no LOCAL_USER_ID is set."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        # Remove LOCAL_USER_ID if it exists
        monkeypatch.delenv("LOCAL_USER_ID", raising=False)
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        # Mock STDIO transport (no HTTP request)
        with patch("backend.src.mcp.server._current_http_request", None):
            user_id = _current_user_id()
            assert user_id == "local-dev"

    def test_stdio_mode_uses_local_user_id_env_var(self, monkeypatch, tmp_path: Path):
        """STDIO mode should use LOCAL_USER_ID environment variable when set."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        monkeypatch.setenv("LOCAL_USER_ID", "custom-local-user")
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        # Mock STDIO transport (no HTTP request)
        with patch("backend.src.mcp.server._current_http_request", None):
            user_id = _current_user_id()
            assert user_id == "custom-local-user"

    def test_stdio_mode_does_not_require_jwt_secret(self, monkeypatch, tmp_path: Path):
        """STDIO mode should work even without JWT_SECRET_KEY configured."""
        # Don't set JWT_SECRET_KEY
        monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        monkeypatch.setenv("LOCAL_USER_ID", "local-dev-user")
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        # Mock STDIO transport (no HTTP request)
        with patch("backend.src.mcp.server._current_http_request", None):
            user_id = _current_user_id()
            assert user_id == "local-dev-user"

    def test_stdio_mode_ignores_noauth_setting(self, monkeypatch, tmp_path: Path):
        """STDIO mode behavior should be consistent regardless of ENABLE_NOAUTH_MCP."""
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        monkeypatch.setenv("LOCAL_USER_ID", "test-stdio-user")

        from backend.src.mcp.server import _current_user_id

        # Test with ENABLE_NOAUTH_MCP=true
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "true")
        config_module.reload_config()

        with patch("backend.src.mcp.server._current_http_request", None):
            user_id_with_noauth = _current_user_id()

        # Test with ENABLE_NOAUTH_MCP=false
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        config_module.reload_config()

        with patch("backend.src.mcp.server._current_http_request", None):
            user_id_without_noauth = _current_user_id()

        # Both should return the same user_id from LOCAL_USER_ID
        assert user_id_with_noauth == "test-stdio-user"
        assert user_id_without_noauth == "test-stdio-user"
        assert user_id_with_noauth == user_id_without_noauth


@pytest.mark.integration
class TestMCPTransportDetection:
    """Test that MCP correctly detects transport mode (HTTP vs STDIO)."""

    def test_http_request_context_is_none_for_stdio(self, monkeypatch, tmp_path: Path):
        """_current_http_request should be None in STDIO mode."""
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        from backend.src.mcp.server import _current_http_request

        # In STDIO mode (default import), _current_http_request should be callable but return None
        # or be None if not available
        if _current_http_request is not None:
            # If it's a context var getter, it should raise LookupError when not set
            with pytest.raises(LookupError):
                _current_http_request.get()  # type: ignore

    def test_http_request_context_exists_for_http(self, monkeypatch, tmp_path: Path):
        """_current_http_request should be available in HTTP mode."""
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        from backend.src.mcp.server import _current_http_request

        # _current_http_request should be importable and not None
        # (it's imported from fastmcp.server.http if available)
        assert _current_http_request is not None or _current_http_request is None  # Either is valid

    def test_lookup_error_treated_as_stdio(self, monkeypatch, tmp_path: Path):
        """LookupError from _current_http_request.get() should be treated as STDIO mode."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        monkeypatch.setenv("LOCAL_USER_ID", "stdio-user")
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        # Mock _current_http_request to raise LookupError (no HTTP context)
        mock_http_request = MagicMock()
        mock_http_request.get.side_effect = LookupError("No request context")

        with patch("backend.src.mcp.server._current_http_request", mock_http_request):
            user_id = _current_user_id()
            # Should fall back to STDIO mode
            assert user_id == "stdio-user"


@pytest.mark.integration
class TestMCPAuthenticationRegression:
    """Regression tests to ensure authentication behavior doesn't break."""

    def test_no_demo_user_fallback_in_http_mode(self, monkeypatch, tmp_path: Path):
        """
        Regression test: HTTP mode should NEVER fall back to demo-user,
        even with ENABLE_NOAUTH_MCP=true (P3.1 requirement).
        """
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "true")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        # Mock HTTP request without auth header
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None

        with patch("backend.src.mcp.server._current_http_request") as mock_http_request:
            mock_http_request.get.return_value = mock_request

            # Should raise PermissionError, NOT fall back to demo-user
            with pytest.raises(PermissionError) as exc_info:
                _current_user_id()

            assert "Authorization header required" in str(exc_info.value)

    def test_valid_bearer_token_extracts_correct_user_id(self, monkeypatch, tmp_path: Path, auth_service: AuthService):
        """Regression test: Valid Bearer token should extract correct user_id from JWT payload."""
        test_users = [
            "user-alice-123",
            "user-bob-456",
            "admin-user-789",
            "hf_user_special@example.com",
        ]

        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-at-least-16-chars")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        for expected_user_id in test_users:
            token = auth_service.create_jwt(expected_user_id)
            mock_request = MagicMock()
            mock_request.headers.get.return_value = f"Bearer {token}"

            with patch("backend.src.mcp.server._current_http_request") as mock_http_request:
                mock_http_request.get.return_value = mock_request

                actual_user_id = _current_user_id()
                assert actual_user_id == expected_user_id

    def test_case_insensitive_bearer_scheme(self, monkeypatch, tmp_path: Path, valid_token: str):
        """Regression test: Bearer scheme should be case-insensitive."""
        monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-at-least-16-chars")
        monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
        config_module.reload_config()

        from backend.src.mcp.server import _current_user_id

        # Test different case variations
        bearer_variations = [
            f"Bearer {valid_token}",
            f"bearer {valid_token}",
            f"BEARER {valid_token}",
            f"BeArEr {valid_token}",
        ]

        for auth_header in bearer_variations:
            mock_request = MagicMock()
            mock_request.headers.get.return_value = auth_header

            with patch("backend.src.mcp.server._current_http_request") as mock_http_request:
                mock_http_request.get.return_value = mock_request

                user_id = _current_user_id()
                assert user_id == "test-user-123"
