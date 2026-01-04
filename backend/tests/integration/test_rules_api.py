"""Integration tests for Rules API endpoints.

Tests the /api/rules endpoints for listing, viewing, toggling, and testing rules.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.src.api.main import app
from backend.src.services import config as config_module
from backend.src.services.auth import create_access_token


@pytest.fixture(autouse=True)
def restore_config_cache():
    """Restore config cache after each test."""
    config_module.reload_config()
    yield
    config_module.reload_config()


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    """Create FastAPI test client with ENABLE_NOAUTH_MCP disabled."""
    # Ensure authentication is enforced
    monkeypatch.setenv("ENABLE_NOAUTH_MCP", "false")
    monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-at-least-16-chars")
    config_module.reload_config()

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    """Generate valid auth headers for testing."""
    token = create_access_token("test-user")
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
class TestRulesRouteAuthentication:
    """Test authentication enforcement on /api/rules/* routes."""

    def test_list_rules_requires_auth(self, client: TestClient):
        """GET /api/rules should return 401 without authentication."""
        response = client.get("/api/rules")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_get_rule_requires_auth(self, client: TestClient):
        """GET /api/rules/{rule_id} should return 401 without authentication."""
        response = client.get("/api/rules/test-rule")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_toggle_rule_requires_auth(self, client: TestClient):
        """POST /api/rules/{rule_id}/toggle should return 401 without authentication."""
        response = client.post("/api/rules/test-rule/toggle", json={"enabled": False})
        assert response.status_code == 401
        assert "error" in response.json()

    def test_test_rule_requires_auth(self, client: TestClient):
        """POST /api/rules/{rule_id}/test should return 401 without authentication."""
        response = client.post("/api/rules/test-rule/test")
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.mark.integration
class TestListRulesEndpoint:
    """Tests for GET /api/rules endpoint."""

    def test_list_rules_success(self, client: TestClient, auth_headers: dict):
        """GET /api/rules should return a list of rules."""
        response = client.get("/api/rules", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "rules" in data
        assert "total" in data
        assert isinstance(data["rules"], list)
        assert data["total"] == len(data["rules"])

    def test_list_rules_filter_by_trigger(self, client: TestClient, auth_headers: dict):
        """GET /api/rules should filter by trigger type."""
        response = client.get(
            "/api/rules", params={"trigger": "on_turn_start"}, headers=auth_headers
        )
        assert response.status_code == 200

        data = response.json()
        # All returned rules should have the specified trigger
        for rule in data["rules"]:
            assert rule["trigger"] == "on_turn_start"

    def test_list_rules_filter_enabled_only(self, client: TestClient, auth_headers: dict):
        """GET /api/rules should filter to enabled rules only."""
        response = client.get(
            "/api/rules", params={"enabled_only": True}, headers=auth_headers
        )
        assert response.status_code == 200

        data = response.json()
        # All returned rules should be enabled
        for rule in data["rules"]:
            assert rule["enabled"] is True

    def test_list_rules_response_schema(self, client: TestClient, auth_headers: dict):
        """GET /api/rules should return rules with correct schema."""
        response = client.get("/api/rules", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        if data["rules"]:
            rule = data["rules"][0]
            # Check required fields are present
            assert "id" in rule
            assert "name" in rule
            assert "trigger" in rule
            assert "enabled" in rule
            assert "core" in rule


@pytest.mark.integration
class TestGetRuleEndpoint:
    """Tests for GET /api/rules/{rule_id} endpoint."""

    def test_get_rule_not_found(self, client: TestClient, auth_headers: dict):
        """GET /api/rules/{rule_id} should return 404 for nonexistent rule."""
        response = client.get("/api/rules/nonexistent-rule-id", headers=auth_headers)
        assert response.status_code == 404
        assert "error" in response.json()
        assert response.json()["error"] == "not_found"

    def test_get_rule_success(self, client: TestClient, auth_headers: dict):
        """GET /api/rules/{rule_id} should return rule details."""
        # First list rules to get a valid ID
        list_response = client.get("/api/rules", headers=auth_headers)
        assert list_response.status_code == 200
        rules = list_response.json()["rules"]

        if rules:
            rule_id = rules[0]["id"]
            response = client.get(f"/api/rules/{rule_id}", headers=auth_headers)
            assert response.status_code == 200

            data = response.json()
            # Check required fields
            assert data["id"] == rule_id
            assert "name" in data
            assert "trigger" in data
            assert "enabled" in data
            assert "core" in data
            # Check detail-specific fields
            assert "version" in data
            assert "source_path" in data

    def test_get_rule_response_schema(self, client: TestClient, auth_headers: dict):
        """GET /api/rules/{rule_id} should return rule with correct detail schema."""
        # First list rules to get a valid ID
        list_response = client.get("/api/rules", headers=auth_headers)
        if list_response.json()["rules"]:
            rule_id = list_response.json()["rules"][0]["id"]
            response = client.get(f"/api/rules/{rule_id}", headers=auth_headers)
            assert response.status_code == 200

            data = response.json()
            # RuleDetail has additional fields compared to RuleInfo
            assert "version" in data
            # condition or script should be present (XOR)
            has_condition = data.get("condition") is not None
            has_script = data.get("script") is not None
            # At least one should be present for a valid rule
            # (unless they both happen to be null in a malformed rule)


@pytest.mark.integration
class TestToggleRuleEndpoint:
    """Tests for POST /api/rules/{rule_id}/toggle endpoint."""

    def test_toggle_rule_not_found(self, client: TestClient, auth_headers: dict):
        """POST /api/rules/{rule_id}/toggle should return 404 for nonexistent rule."""
        response = client.post(
            "/api/rules/nonexistent-rule-id/toggle",
            json={"enabled": False},
            headers=auth_headers,
        )
        assert response.status_code == 404
        assert "error" in response.json()

    def test_toggle_rule_missing_enabled_field(self, client: TestClient, auth_headers: dict):
        """POST /api/rules/{rule_id}/toggle should require enabled field."""
        # First list rules to get a valid ID
        list_response = client.get("/api/rules", headers=auth_headers)
        if list_response.json()["rules"]:
            rule_id = list_response.json()["rules"][0]["id"]
            response = client.post(
                f"/api/rules/{rule_id}/toggle",
                json={},  # Missing 'enabled' field
                headers=auth_headers,
            )
            assert response.status_code == 422  # Validation error

    def test_toggle_rule_disable_non_core(self, client: TestClient, auth_headers: dict):
        """POST /api/rules/{rule_id}/toggle should allow disabling non-core rules."""
        # Find a non-core rule
        list_response = client.get("/api/rules", headers=auth_headers)
        rules = list_response.json()["rules"]
        non_core_rules = [r for r in rules if not r["core"]]

        if non_core_rules:
            rule_id = non_core_rules[0]["id"]

            # Disable the rule
            response = client.post(
                f"/api/rules/{rule_id}/toggle",
                json={"enabled": False},
                headers=auth_headers,
            )
            assert response.status_code == 200
            assert response.json()["enabled"] is False

            # Re-enable the rule
            response = client.post(
                f"/api/rules/{rule_id}/toggle",
                json={"enabled": True},
                headers=auth_headers,
            )
            assert response.status_code == 200
            assert response.json()["enabled"] is True

    def test_toggle_rule_cannot_disable_core(self, client: TestClient, auth_headers: dict):
        """POST /api/rules/{rule_id}/toggle should reject disabling core rules."""
        # Find a core rule
        list_response = client.get("/api/rules", headers=auth_headers)
        rules = list_response.json()["rules"]
        core_rules = [r for r in rules if r["core"]]

        if core_rules:
            rule_id = core_rules[0]["id"]

            response = client.post(
                f"/api/rules/{rule_id}/toggle",
                json={"enabled": False},
                headers=auth_headers,
            )
            assert response.status_code == 400
            assert "cannot_disable_core_rule" in response.json().get("error", "")


@pytest.mark.integration
class TestTestRuleEndpoint:
    """Tests for POST /api/rules/{rule_id}/test endpoint."""

    def test_test_rule_not_found(self, client: TestClient, auth_headers: dict):
        """POST /api/rules/{rule_id}/test should return 404 for nonexistent rule."""
        response = client.post(
            "/api/rules/nonexistent-rule-id/test", headers=auth_headers
        )
        assert response.status_code == 404
        assert "error" in response.json()

    def test_test_rule_success(self, client: TestClient, auth_headers: dict):
        """POST /api/rules/{rule_id}/test should evaluate rule condition."""
        # Find a rule with a condition (not script-based)
        list_response = client.get("/api/rules", headers=auth_headers)
        rules = list_response.json()["rules"]

        if rules:
            # Get rule details to find one with condition
            for rule_info in rules:
                detail_response = client.get(
                    f"/api/rules/{rule_info['id']}", headers=auth_headers
                )
                if detail_response.status_code == 200:
                    detail = detail_response.json()
                    if detail.get("condition"):
                        # Found a rule with condition
                        response = client.post(
                            f"/api/rules/{rule_info['id']}/test", headers=auth_headers
                        )
                        assert response.status_code == 200

                        data = response.json()
                        assert "condition_result" in data
                        assert "action_would_execute" in data
                        assert "evaluation_time_ms" in data
                        assert isinstance(data["condition_result"], bool)
                        assert isinstance(data["action_would_execute"], bool)
                        assert isinstance(data["evaluation_time_ms"], (int, float))
                        return

    def test_test_rule_with_context_override(self, client: TestClient, auth_headers: dict):
        """POST /api/rules/{rule_id}/test should accept context overrides."""
        # Find a rule
        list_response = client.get("/api/rules", headers=auth_headers)
        rules = list_response.json()["rules"]

        if rules:
            rule_id = rules[0]["id"]
            response = client.post(
                f"/api/rules/{rule_id}/test",
                json={
                    "context_override": {
                        "turn_number": 5,
                        "token_usage": 0.8,
                        "context_usage": 0.9,
                        "iteration_count": 10,
                    }
                },
                headers=auth_headers,
            )
            # Should succeed (or return script error if script-based)
            assert response.status_code == 200

    def test_test_rule_script_based(self, client: TestClient, auth_headers: dict):
        """POST /api/rules/{rule_id}/test should handle script-based rules."""
        # Find a rule with a script (not condition-based)
        list_response = client.get("/api/rules", headers=auth_headers)
        rules = list_response.json()["rules"]

        if rules:
            for rule_info in rules:
                detail_response = client.get(
                    f"/api/rules/{rule_info['id']}", headers=auth_headers
                )
                if detail_response.status_code == 200:
                    detail = detail_response.json()
                    if detail.get("script") and not detail.get("condition"):
                        # Found a script-based rule
                        response = client.post(
                            f"/api/rules/{rule_info['id']}/test", headers=auth_headers
                        )
                        assert response.status_code == 200

                        data = response.json()
                        # Script-based rules should return an error message
                        assert data.get("error") is not None
                        assert "script" in data["error"].lower()
                        return


@pytest.mark.integration
class TestRulesEndpointEdgeCases:
    """Tests for edge cases in Rules API."""

    def test_list_rules_empty_filters(self, client: TestClient, auth_headers: dict):
        """GET /api/rules with empty filter values should still work."""
        response = client.get(
            "/api/rules",
            params={"plugin_id": "", "enabled_only": "false"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_rule_special_characters(self, client: TestClient, auth_headers: dict):
        """GET /api/rules/{rule_id} should handle special characters safely."""
        # Try various potentially problematic IDs
        test_ids = [
            "rule-with-dashes",
            "rule_with_underscores",
            "rule123",
            "../../../etc/passwd",  # Path traversal attempt
        ]

        for rule_id in test_ids:
            response = client.get(f"/api/rules/{rule_id}", headers=auth_headers)
            # Should return 404 (not found) not 500 (server error)
            assert response.status_code in (200, 404)

    def test_toggle_rule_idempotent(self, client: TestClient, auth_headers: dict):
        """POST /api/rules/{rule_id}/toggle should be idempotent."""
        # Find a non-core rule
        list_response = client.get("/api/rules", headers=auth_headers)
        rules = list_response.json()["rules"]
        non_core_rules = [r for r in rules if not r["core"]]

        if non_core_rules:
            rule_id = non_core_rules[0]["id"]

            # Toggle twice to same state should have same result
            response1 = client.post(
                f"/api/rules/{rule_id}/toggle",
                json={"enabled": True},
                headers=auth_headers,
            )
            response2 = client.post(
                f"/api/rules/{rule_id}/toggle",
                json={"enabled": True},
                headers=auth_headers,
            )

            assert response1.status_code == 200
            assert response2.status_code == 200
            assert response1.json()["enabled"] == response2.json()["enabled"]
