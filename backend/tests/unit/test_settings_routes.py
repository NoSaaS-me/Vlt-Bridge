"""Tests for GET/PUT /api/settings/oracle endpoint."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from backend.src.api.main import app
from backend.src.api.middleware import AuthContext, get_auth_context
from backend.src.services.user_settings import UserSettingsService, get_user_settings_service

client = TestClient(app)


@pytest.fixture
def mock_auth():
    """Mock authentication context."""
    mock = Mock(spec=AuthContext)
    mock.user_id = "test-user"
    return mock


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Clean up dependency overrides after each test."""
    yield
    app.dependency_overrides = {}


class TestGetOracleSettings:
    def test_returns_oracle_mcp_enabled_true_by_default(self, mock_auth):
        mock_svc = Mock(spec=UserSettingsService)
        mock_svc.get_oracle_mcp_enabled.return_value = True

        app.dependency_overrides[get_auth_context] = lambda: mock_auth
        app.dependency_overrides[get_user_settings_service] = lambda: mock_svc

        response = client.get("/api/settings/oracle")

        assert response.status_code == 200
        assert response.json()["oracle_mcp_enabled"] is True
        mock_svc.get_oracle_mcp_enabled.assert_called_once_with("test-user")

    def test_returns_oracle_mcp_enabled_false_when_disabled(self, mock_auth):
        mock_svc = Mock(spec=UserSettingsService)
        mock_svc.get_oracle_mcp_enabled.return_value = False

        app.dependency_overrides[get_auth_context] = lambda: mock_auth
        app.dependency_overrides[get_user_settings_service] = lambda: mock_svc

        response = client.get("/api/settings/oracle")

        assert response.status_code == 200
        assert response.json()["oracle_mcp_enabled"] is False

    def test_returns_401_without_auth(self):
        response = client.get("/api/settings/oracle")
        assert response.status_code == 401


class TestPutOracleSettings:
    def test_sets_oracle_mcp_enabled_false(self, mock_auth):
        mock_svc = Mock(spec=UserSettingsService)
        mock_svc.set_oracle_mcp_enabled.return_value = None

        app.dependency_overrides[get_auth_context] = lambda: mock_auth
        app.dependency_overrides[get_user_settings_service] = lambda: mock_svc

        response = client.put(
            "/api/settings/oracle",
            json={"oracle_mcp_enabled": False},
        )

        assert response.status_code == 200
        assert response.json()["oracle_mcp_enabled"] is False
        mock_svc.set_oracle_mcp_enabled.assert_called_once_with("test-user", False)

    def test_sets_oracle_mcp_enabled_true(self, mock_auth):
        mock_svc = Mock(spec=UserSettingsService)
        mock_svc.set_oracle_mcp_enabled.return_value = None

        app.dependency_overrides[get_auth_context] = lambda: mock_auth
        app.dependency_overrides[get_user_settings_service] = lambda: mock_svc

        response = client.put(
            "/api/settings/oracle",
            json={"oracle_mcp_enabled": True},
        )

        assert response.status_code == 200
        assert response.json()["oracle_mcp_enabled"] is True

    def test_returns_401_without_auth(self):
        response = client.put(
            "/api/settings/oracle",
            json={"oracle_mcp_enabled": False},
        )
        assert response.status_code == 401
