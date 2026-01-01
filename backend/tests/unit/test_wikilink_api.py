"""Tests for wikilink resolution and note preview API endpoints."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from backend.src.api.main import app
from backend.src.api.middleware import AuthContext, get_auth_context

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


class TestWikilinkResolveEndpoint:
    """Tests for GET /api/wikilinks/resolve endpoint."""

    @patch("backend.src.api.routes.search.IndexerService")
    def test_resolve_wikilink_success(self, mock_indexer_cls, mock_auth):
        """Test successful wikilink resolution."""
        # Setup mock
        mock_indexer = mock_indexer_cls.return_value
        mock_indexer.resolve_single_wikilink.return_value = {
            "link_text": "API Design",
            "target_path": "docs/api-design.md",
            "is_resolved": True,
        }

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request
        response = client.get("/api/wikilinks/resolve?link=API%20Design")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["link_text"] == "API Design"
        assert data["target_path"] == "docs/api-design.md"
        assert data["is_resolved"] is True

        # Verify service was called correctly
        mock_indexer.resolve_single_wikilink.assert_called_once_with(
            "test-user", "API Design", ""
        )

    @patch("backend.src.api.routes.search.IndexerService")
    def test_resolve_wikilink_with_context(self, mock_indexer_cls, mock_auth):
        """Test wikilink resolution with context path for same-folder preference."""
        # Setup mock
        mock_indexer = mock_indexer_cls.return_value
        mock_indexer.resolve_single_wikilink.return_value = {
            "link_text": "Related Note",
            "target_path": "docs/related-note.md",
            "is_resolved": True,
        }

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request with context
        response = client.get(
            "/api/wikilinks/resolve?link=Related%20Note&context=docs/current.md"
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["is_resolved"] is True

        # Verify service was called with context
        mock_indexer.resolve_single_wikilink.assert_called_once_with(
            "test-user", "Related Note", "docs/current.md"
        )

    @patch("backend.src.api.routes.search.IndexerService")
    def test_resolve_wikilink_not_found(self, mock_indexer_cls, mock_auth):
        """Test wikilink resolution when link doesn't exist (broken link)."""
        # Setup mock to return unresolved link
        mock_indexer = mock_indexer_cls.return_value
        mock_indexer.resolve_single_wikilink.return_value = {
            "link_text": "Nonexistent Page",
            "target_path": None,
            "is_resolved": False,
        }

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request
        response = client.get("/api/wikilinks/resolve?link=Nonexistent%20Page")

        # Verify response - should return 200 with is_resolved=False
        assert response.status_code == 200
        data = response.json()
        assert data["link_text"] == "Nonexistent Page"
        assert data["target_path"] is None
        assert data["is_resolved"] is False

    def test_resolve_wikilink_missing_link_param(self, mock_auth):
        """Test error when link parameter is missing."""
        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request without link parameter
        response = client.get("/api/wikilinks/resolve")

        # Should return 422 (validation error)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    @patch("backend.src.api.routes.search.IndexerService")
    def test_resolve_wikilink_url_decoding(self, mock_indexer_cls, mock_auth):
        """Test that URL-encoded link text is properly decoded."""
        # Setup mock
        mock_indexer = mock_indexer_cls.return_value
        mock_indexer.resolve_single_wikilink.return_value = {
            "link_text": "Special Chars !@#",
            "target_path": "special.md",
            "is_resolved": True,
        }

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request with URL-encoded special characters
        response = client.get("/api/wikilinks/resolve?link=Special%20Chars%20%21%40%23")

        # Verify response
        assert response.status_code == 200

        # Verify service received decoded text
        mock_indexer.resolve_single_wikilink.assert_called_once()
        call_args = mock_indexer.resolve_single_wikilink.call_args[0]
        assert call_args[1] == "Special Chars !@#"

    @patch("backend.src.api.routes.search.IndexerService")
    def test_resolve_wikilink_service_error(self, mock_indexer_cls, mock_auth):
        """Test error handling when service raises exception."""
        # Setup mock to raise exception
        mock_indexer = mock_indexer_cls.return_value
        mock_indexer.resolve_single_wikilink.side_effect = Exception("Database connection failed")

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request
        response = client.get("/api/wikilinks/resolve?link=Any%20Link")

        # Should return 500
        assert response.status_code == 500
        data = response.json()
        assert "Failed to resolve wikilink" in data["detail"]


class TestNotePreviewEndpoint:
    """Tests for GET /api/notes/{path}/preview endpoint."""

    @patch("backend.src.api.routes.notes.DatabaseService")
    @patch("backend.src.api.routes.notes.VaultService")
    def test_get_note_preview_success(self, mock_vault_cls, mock_db_cls, mock_auth):
        """Test successful note preview retrieval."""
        # Setup vault mock
        mock_vault = mock_vault_cls.return_value
        mock_vault.read_note.return_value = {
            "title": "API Design",
            "body": "# API Design\n\nThis document describes **our API** design patterns.\n\nWe use REST.",
            "metadata": {
                "tags": ["backend", "api"],
                "updated": "2025-01-15T14:30:00Z",
            },
        }

        # Setup database mock (not used when tags in metadata)
        mock_db = mock_db_cls.return_value
        mock_conn = Mock()
        mock_db.connect.return_value = mock_conn

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request
        response = client.get("/api/notes/docs/api-design.md/preview")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "API Design"
        assert "API Design" in data["snippet"]
        assert "**our API**" not in data["snippet"]  # Markdown should be stripped
        assert "our API" in data["snippet"]  # Text should remain
        assert data["tags"] == ["backend", "api"]
        assert "updated" in data

        # Verify vault service was called
        mock_vault.read_note.assert_called_once_with("test-user", "docs/api-design.md")

    @patch("backend.src.api.routes.notes.DatabaseService")
    @patch("backend.src.api.routes.notes.VaultService")
    def test_get_note_preview_strips_markdown(self, mock_vault_cls, mock_db_cls, mock_auth):
        """Test that markdown formatting is stripped from preview snippet."""
        # Setup vault mock with various markdown elements
        mock_vault = mock_vault_cls.return_value
        mock_vault.read_note.return_value = {
            "title": "Markdown Test",
            "body": """# Header
## Subheader

**Bold text** and *italic text* and __underlined__.

`inline code` and:

```python
code block
```

[Link text](http://example.com)
[[Wikilink]]

![Image](image.png)

> Blockquote

- List item
* Another item
1. Numbered

---

Regular text.""",
            "metadata": {
                "updated": "2025-01-15T14:30:00Z",
            },
        }

        # Setup database mock for tags query
        mock_db = mock_db_cls.return_value
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor
        mock_db.connect.return_value = mock_conn

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request
        response = client.get("/api/notes/test.md/preview")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        snippet = data["snippet"]

        # Verify markdown elements are stripped
        assert "#" not in snippet  # No headers
        assert "**" not in snippet  # No bold markers
        assert "*" not in snippet or "Another" in snippet  # No italic markers (or list markers)
        assert "`" not in snippet  # No code markers
        assert "```" not in snippet  # No code block markers
        assert "[[" not in snippet  # No wikilink markers
        assert "![]" not in snippet  # No image markers
        assert "---" not in snippet  # No horizontal rules

        # Verify text content remains
        assert "Bold text" in snippet or "italic text" in snippet

    @patch("backend.src.api.routes.notes.DatabaseService")
    @patch("backend.src.api.routes.notes.VaultService")
    def test_get_note_preview_tags_from_database(self, mock_vault_cls, mock_db_cls, mock_auth):
        """Test that tags are fetched from database when not in metadata."""
        # Setup vault mock without tags in metadata
        mock_vault = mock_vault_cls.return_value
        mock_vault.read_note.return_value = {
            "title": "Test Note",
            "body": "Some content",
            "metadata": {
                "updated": "2025-01-15T14:30:00Z",
            },
        }

        # Setup database mock to return tags
        mock_db = mock_db_cls.return_value
        mock_conn = Mock()
        mock_cursor = Mock()

        # Mock rows with dict-like access
        class MockRow:
            def __init__(self, tag):
                self.tag = tag
                self._data = {"tag": tag}

            def keys(self):
                return self._data.keys()

            def __getitem__(self, key):
                return self._data[key]

        mock_cursor.fetchall.return_value = [
            MockRow("database-tag"),
            MockRow("indexed-tag"),
        ]
        mock_conn.execute.return_value = mock_cursor
        mock_db.connect.return_value = mock_conn

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request
        response = client.get("/api/notes/test.md/preview")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data["tags"]) == 2
        assert "database-tag" in data["tags"]
        assert "indexed-tag" in data["tags"]

        # Verify database was queried for tags
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert "SELECT tag FROM note_tags" in call_args[0]

    @patch("backend.src.api.routes.notes.VaultService")
    def test_get_note_preview_snippet_length(self, mock_vault_cls, mock_auth):
        """Test that snippet is truncated to 200 characters."""
        # Setup vault mock with long content
        long_text = "A" * 500  # 500 characters
        mock_vault = mock_vault_cls.return_value
        mock_vault.read_note.return_value = {
            "title": "Long Note",
            "body": long_text,
            "metadata": {
                "updated": "2025-01-15T14:30:00Z",
            },
        }

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request
        response = client.get("/api/notes/long.md/preview")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data["snippet"]) == 200

    @patch("backend.src.api.routes.notes.VaultService")
    def test_get_note_preview_not_found(self, mock_vault_cls, mock_auth):
        """Test 404 error when note doesn't exist."""
        # Setup vault mock to raise FileNotFoundError
        mock_vault = mock_vault_cls.return_value
        mock_vault.read_note.side_effect = FileNotFoundError("Note not found")

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request
        response = client.get("/api/notes/nonexistent.md/preview")

        # Verify 404 response
        assert response.status_code == 404
        data = response.json()
        assert "Note not found" in data["detail"]

    @patch("backend.src.api.routes.notes.VaultService")
    def test_get_note_preview_url_decoding(self, mock_vault_cls, mock_auth):
        """Test that URL-encoded paths are properly decoded."""
        # Setup vault mock
        mock_vault = mock_vault_cls.return_value
        mock_vault.read_note.return_value = {
            "title": "Spaced Path",
            "body": "Content",
            "metadata": {"updated": "2025-01-15T14:30:00Z"},
        }

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request with URL-encoded path
        response = client.get("/api/notes/folder%20with%20spaces/note.md/preview")

        # Verify response
        assert response.status_code == 200

        # Verify vault service received decoded path
        mock_vault.read_note.assert_called_once_with("test-user", "folder with spaces/note.md")

    @patch("backend.src.api.routes.notes.VaultService")
    def test_get_note_preview_service_error(self, mock_vault_cls, mock_auth):
        """Test error handling when service raises exception."""
        # Setup vault mock to raise exception
        mock_vault = mock_vault_cls.return_value
        mock_vault.read_note.side_effect = Exception("I/O error")

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request
        response = client.get("/api/notes/test.md/preview")

        # Should return 500
        assert response.status_code == 500
        data = response.json()
        assert "Failed to get note preview" in data["detail"]

    @patch("backend.src.api.routes.notes.DatabaseService")
    @patch("backend.src.api.routes.notes.VaultService")
    def test_get_note_preview_empty_body(self, mock_vault_cls, mock_db_cls, mock_auth):
        """Test preview generation for note with empty body."""
        # Setup vault mock with empty body
        mock_vault = mock_vault_cls.return_value
        mock_vault.read_note.return_value = {
            "title": "Empty Note",
            "body": "",
            "metadata": {
                "tags": ["empty"],
                "updated": "2025-01-15T14:30:00Z",
            },
        }

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request
        response = client.get("/api/notes/empty.md/preview")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Empty Note"
        assert data["snippet"] == ""
        assert data["tags"] == ["empty"]

    @patch("backend.src.api.routes.notes.DatabaseService")
    @patch("backend.src.api.routes.notes.VaultService")
    def test_get_note_preview_timestamp_parsing(self, mock_vault_cls, mock_db_cls, mock_auth):
        """Test that various timestamp formats are handled correctly."""
        # Setup vault mock with ISO timestamp
        mock_vault = mock_vault_cls.return_value
        mock_vault.read_note.return_value = {
            "title": "Test Note",
            "body": "Content",
            "metadata": {
                "tags": [],
                "updated": "2025-01-15T14:30:00+00:00",  # ISO format with timezone
            },
        }

        app.dependency_overrides[get_auth_context] = lambda: mock_auth

        # Make request
        response = client.get("/api/notes/test.md/preview")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "updated" in data
        # Verify it's a valid ISO timestamp string in response
        datetime.fromisoformat(data["updated"].replace("Z", "+00:00"))
