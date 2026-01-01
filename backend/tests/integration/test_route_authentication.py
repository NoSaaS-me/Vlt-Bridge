"""Integration tests for route authentication enforcement.

Tests that all protected routes return 401 Unauthorized when accessed without
valid authentication headers, ensuring the authentication middleware is properly
enforced across the API.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.src.api.main import app
from backend.src.services import config as config_module


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


@pytest.mark.integration
class TestNotesRouteAuthentication:
    """Test authentication enforcement on /api/notes/* routes."""

    def test_list_notes_requires_auth(self, client: TestClient):
        """GET /api/notes should return 401 without authentication."""
        response = client.get("/api/notes")
        assert response.status_code == 401
        assert "error" in response.json()
        assert response.json()["error"] == "unauthorized"

    def test_create_note_requires_auth(self, client: TestClient):
        """POST /api/notes should return 401 without authentication."""
        payload = {
            "note_path": "test.md",
            "content": "Test content"
        }
        response = client.post("/api/notes", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_get_note_requires_auth(self, client: TestClient):
        """GET /api/notes/{path} should return 401 without authentication."""
        response = client.get("/api/notes/test.md")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_update_note_requires_auth(self, client: TestClient):
        """PUT /api/notes/{path} should return 401 without authentication."""
        payload = {
            "content": "Updated content"
        }
        response = client.put("/api/notes/test.md", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_delete_note_requires_auth(self, client: TestClient):
        """DELETE /api/notes/{path} should return 401 without authentication."""
        response = client.delete("/api/notes/test.md")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_move_note_requires_auth(self, client: TestClient):
        """POST /api/notes/move should return 401 without authentication."""
        payload = {
            "from_path": "old.md",
            "to_path": "new.md"
        }
        response = client.post("/api/notes/move", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.mark.integration
class TestIndexRouteAuthentication:
    """Test authentication enforcement on /api/index/* routes."""

    def test_rebuild_index_requires_auth(self, client: TestClient):
        """POST /api/index/rebuild should return 401 without authentication."""
        response = client.post("/api/index/rebuild")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_index_health_allows_unauthenticated(self, client: TestClient):
        """GET /api/index/health should allow access without authentication."""
        # This route uses OPTIONAL auth, so it should NOT return 401
        # It might return 500 or other errors, but NOT 401
        response = client.get("/api/index/health")
        assert response.status_code != 401


@pytest.mark.integration
class TestSearchRouteAuthentication:
    """Test authentication enforcement on /api/search/* routes."""

    def test_search_requires_auth(self, client: TestClient):
        """POST /api/search should return 401 without authentication."""
        payload = {"query": "test"}
        response = client.post("/api/search", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_backlinks_requires_auth(self, client: TestClient):
        """GET /api/backlinks/{path} should return 401 without authentication."""
        response = client.get("/api/backlinks/test.md")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_tags_requires_auth(self, client: TestClient):
        """GET /api/tags should return 401 without authentication."""
        response = client.get("/api/tags")
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.mark.integration
class TestGraphRouteAuthentication:
    """Test authentication enforcement on /api/graph route."""

    def test_graph_requires_auth(self, client: TestClient):
        """GET /api/graph should return 401 without authentication."""
        response = client.get("/api/graph")
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.mark.integration
class TestOracleRouteAuthentication:
    """Test authentication enforcement on /api/oracle/* routes."""

    def test_query_oracle_requires_auth(self, client: TestClient):
        """POST /api/oracle should return 401 without authentication."""
        payload = {"query": "test question"}
        response = client.post("/api/oracle", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_query_oracle_stream_requires_auth(self, client: TestClient):
        """POST /api/oracle/stream should return 401 without authentication."""
        payload = {"query": "test question"}
        response = client.post("/api/oracle/stream", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_cancel_oracle_session_requires_auth(self, client: TestClient):
        """POST /api/oracle/cancel should return 401 without authentication."""
        response = client.post("/api/oracle/cancel")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_get_conversation_history_requires_auth(self, client: TestClient):
        """GET /api/oracle/history should return 401 without authentication."""
        response = client.get("/api/oracle/history")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_clear_conversation_history_requires_auth(self, client: TestClient):
        """DELETE /api/oracle/history should return 401 without authentication."""
        response = client.delete("/api/oracle/history")
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.mark.integration
class TestOracleContextRouteAuthentication:
    """Test authentication enforcement on /api/oracle/context/* routes."""

    def test_list_trees_requires_auth(self, client: TestClient):
        """GET /api/oracle/context/trees should return 401 without authentication."""
        response = client.get("/api/oracle/context/trees")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_get_tree_requires_auth(self, client: TestClient):
        """GET /api/oracle/context/trees/{root_id} should return 401 without authentication."""
        response = client.get("/api/oracle/context/trees/test-tree-id")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_create_tree_requires_auth(self, client: TestClient):
        """POST /api/oracle/context/trees should return 401 without authentication."""
        payload = {"initial_message": "test"}
        response = client.post("/api/oracle/context/trees", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_delete_tree_requires_auth(self, client: TestClient):
        """DELETE /api/oracle/context/trees/{root_id} should return 401 without authentication."""
        response = client.delete("/api/oracle/context/trees/test-tree-id")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_activate_tree_requires_auth(self, client: TestClient):
        """POST /api/oracle/context/trees/{root_id}/activate should return 401 without authentication."""
        response = client.post("/api/oracle/context/trees/test-tree-id/activate")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_prune_tree_requires_auth(self, client: TestClient):
        """POST /api/oracle/context/trees/{root_id}/prune should return 401 without authentication."""
        payload = {"from_node_id": "node-123"}
        response = client.post("/api/oracle/context/trees/test-tree-id/prune", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_checkout_node_requires_auth(self, client: TestClient):
        """POST /api/oracle/context/nodes/{node_id}/checkout should return 401 without authentication."""
        response = client.post("/api/oracle/context/nodes/node-123/checkout")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_update_node_label_requires_auth(self, client: TestClient):
        """PUT /api/oracle/context/nodes/{node_id}/label should return 401 without authentication."""
        payload = {"label": "new label"}
        response = client.put("/api/oracle/context/nodes/node-123/label", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_checkpoint_node_requires_auth(self, client: TestClient):
        """PUT /api/oracle/context/nodes/{node_id}/checkpoint should return 401 without authentication."""
        payload = {"is_checkpoint": True}
        response = client.put("/api/oracle/context/nodes/node-123/checkpoint", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_get_settings_requires_auth(self, client: TestClient):
        """GET /api/oracle/context/settings should return 401 without authentication."""
        response = client.get("/api/oracle/context/settings")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_update_settings_requires_auth(self, client: TestClient):
        """PUT /api/oracle/context/settings should return 401 without authentication."""
        payload = {"max_context_tokens": 8000}
        response = client.put("/api/oracle/context/settings", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.mark.integration
class TestThreadsRouteAuthentication:
    """Test authentication enforcement on /api/threads/* routes."""

    def test_sync_threads_requires_auth(self, client: TestClient):
        """POST /api/threads/sync should return 401 without authentication."""
        payload = {"entries": []}
        response = client.post("/api/threads/sync", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_search_threads_requires_auth(self, client: TestClient):
        """GET /api/threads/search should return 401 without authentication."""
        response = client.get("/api/threads/search?q=test")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_seek_threads_requires_auth(self, client: TestClient):
        """GET /api/threads/seek should return 401 without authentication."""
        response = client.get("/api/threads/seek?date=2024-01-01")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_create_thread_requires_auth(self, client: TestClient):
        """POST /api/threads/create should return 401 without authentication."""
        payload = {
            "thread_id": "test-thread",
            "title": "Test Thread",
            "entries": []
        }
        response = client.post("/api/threads/create", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_list_threads_requires_auth(self, client: TestClient):
        """GET /api/threads should return 401 without authentication."""
        response = client.get("/api/threads")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_get_thread_requires_auth(self, client: TestClient):
        """GET /api/threads/{thread_id} should return 401 without authentication."""
        response = client.get("/api/threads/test-thread-id")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_add_thread_entry_requires_auth(self, client: TestClient):
        """POST /api/threads/{thread_id}/entries should return 401 without authentication."""
        payload = {"content": "test entry"}
        response = client.post("/api/threads/test-thread-id/entries", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()

    def test_get_thread_status_requires_auth(self, client: TestClient):
        """GET /api/threads/{thread_id}/status should return 401 without authentication."""
        response = client.get("/api/threads/test-thread-id/status")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_delete_thread_requires_auth(self, client: TestClient):
        """DELETE /api/threads/{thread_id} should return 401 without authentication."""
        response = client.delete("/api/threads/test-thread-id")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_summarize_thread_requires_auth(self, client: TestClient):
        """POST /api/threads/{thread_id}/summarize should return 401 without authentication."""
        response = client.post("/api/threads/test-thread-id/summarize")
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.mark.integration
class TestRagRouteAuthentication:
    """Test authentication enforcement on /api/rag/* routes."""

    def test_rag_status_requires_auth(self, client: TestClient):
        """GET /api/rag/status should return 401 without authentication."""
        response = client.get("/api/rag/status")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_rag_chat_requires_auth(self, client: TestClient):
        """POST /api/rag/chat should return 401 without authentication."""
        payload = {"message": "test question"}
        response = client.post("/api/rag/chat", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.mark.integration
class TestTtsRouteAuthentication:
    """Test authentication enforcement on /api/tts route."""

    def test_tts_requires_auth(self, client: TestClient):
        """POST /api/tts should return 401 without authentication."""
        payload = {"text": "test text"}
        response = client.post("/api/tts", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.mark.integration
class TestSystemRouteAuthentication:
    """Test authentication enforcement on /api/system/* routes."""

    def test_system_logs_requires_admin_auth(self, client: TestClient):
        """GET /api/system/logs should return 401 without authentication."""
        response = client.get("/api/system/logs")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_system_debug_widget_requires_admin_auth(self, client: TestClient):
        """GET /api/system/debug/widget should return 401 without authentication."""
        response = client.get("/api/system/debug/widget")
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.mark.integration
class TestAuthRouteAuthentication:
    """Test authentication enforcement on /api/auth/* routes."""

    def test_tokens_requires_auth(self, client: TestClient):
        """POST /api/tokens should return 401 without authentication."""
        response = client.post("/api/tokens")
        assert response.status_code == 401
        assert "error" in response.json()

    def test_me_requires_auth(self, client: TestClient):
        """GET /api/me should return 401 without authentication."""
        response = client.get("/api/me")
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.mark.integration
class TestPublicRoutes:
    """Test that intentionally public routes remain accessible."""

    def test_demo_token_is_public(self, client: TestClient):
        """POST /api/demo/token should NOT require authentication."""
        response = client.post("/api/demo/token")
        # Should not return 401 (may return other errors, but not auth error)
        assert response.status_code != 401

    def test_health_endpoint_is_public(self, client: TestClient):
        """GET /health should NOT require authentication."""
        response = client.get("/health")
        # Should not return 401 (may return 404 if not defined, but not 401)
        assert response.status_code != 401

    def test_auth_login_is_public(self, client: TestClient):
        """GET /auth/login should NOT require authentication."""
        # This endpoint handles OAuth flow and must be public
        response = client.get("/auth/login")
        # Should not return 401 (may redirect or return other status, but not 401)
        assert response.status_code != 401

    def test_auth_callback_is_public(self, client: TestClient):
        """GET /auth/callback should NOT require authentication."""
        # This endpoint handles OAuth callback and must be public
        response = client.get("/auth/callback")
        # Should not return 401 (may return error for missing code, but not 401)
        assert response.status_code != 401
