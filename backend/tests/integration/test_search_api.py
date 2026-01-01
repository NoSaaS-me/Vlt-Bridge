#!/usr/bin/env python3
"""Integration tests for /api/search endpoint with tag filtering."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import pytest
from httpx import AsyncClient, ASGITransport

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from backend.src.api.middleware import AuthContext, get_auth_context
from backend.src.models.auth import JWTPayload
from backend.src.services.database import DatabaseService
from backend.src.services.indexer import IndexerService


TEST_USER_ID = "test-user-search"


def _create_mock_auth_context() -> AuthContext:
    """Create a mock auth context for testing."""
    payload = JWTPayload(sub=TEST_USER_ID, iat=0, exp=9999999999)
    return AuthContext(user_id=TEST_USER_ID, token="test-token", payload=payload)


# Global variables to store test database and indexer
_test_db_service: DatabaseService | None = None
_test_indexer: IndexerService | None = None


def _get_test_indexer() -> IndexerService:
    """Get the test indexer instance."""
    if _test_indexer is None:
        raise RuntimeError("Test indexer not initialized")
    return _test_indexer


@pytest.fixture()
def db_service(tmp_path: Path) -> Generator[DatabaseService, None, None]:
    """Create a temporary database for testing."""
    global _test_db_service
    db_path = tmp_path / "test_search.db"
    db = DatabaseService(db_path)
    db.initialize()
    _test_db_service = db
    yield db
    _test_db_service = None


@pytest.fixture()
def indexer(db_service: DatabaseService) -> Generator[IndexerService, None, None]:
    """Create an indexer service with the test database."""
    global _test_indexer
    _test_indexer = IndexerService(db_service=db_service)
    yield _test_indexer
    _test_indexer = None


@pytest.fixture()
def seeded_indexer(indexer: IndexerService) -> IndexerService:
    """Seed the indexer with test notes for search tests."""
    # Note with python tag
    indexer.index_note(
        TEST_USER_ID,
        {
            "path": "docs/python-guide.md",
            "metadata": {
                "title": "Python Programming Guide",
                "tags": ["python", "programming", "tutorial"],
            },
            "body": "A comprehensive guide to Python programming language.",
        },
    )

    # Note with javascript tag
    indexer.index_note(
        TEST_USER_ID,
        {
            "path": "docs/javascript-guide.md",
            "metadata": {
                "title": "JavaScript Programming Guide",
                "tags": ["javascript", "programming", "tutorial"],
            },
            "body": "A comprehensive guide to JavaScript programming language.",
        },
    )

    # Note with python and web tags
    indexer.index_note(
        TEST_USER_ID,
        {
            "path": "docs/python-web.md",
            "metadata": {
                "title": "Python Web Development",
                "tags": ["python", "web", "backend"],
            },
            "body": "Building web applications with Python frameworks.",
        },
    )

    # Note without tags
    indexer.index_note(
        TEST_USER_ID,
        {
            "path": "docs/general-guide.md",
            "metadata": {"title": "General Programming Guide"},
            "body": "General programming concepts and best practices.",
        },
    )

    return indexer


@pytest.fixture()
def test_client(seeded_indexer: IndexerService) -> Generator[AsyncClient, None, None]:
    """Create a test client with dependency overrides."""
    # Import app after setting up the path
    from backend.src.api.main import app

    # Store original IndexerService __init__
    original_init = IndexerService.__init__

    # Create a patched __init__ that uses our test db
    def patched_init(self, db_service=None):
        if _test_db_service is not None:
            original_init(self, db_service=_test_db_service)
        else:
            original_init(self, db_service=db_service)

    # Apply patches
    IndexerService.__init__ = patched_init
    app.dependency_overrides[get_auth_context] = _create_mock_auth_context

    try:
        transport = ASGITransport(app=app)
        client = AsyncClient(transport=transport, base_url="http://test")
        yield client
    finally:
        # Cleanup
        IndexerService.__init__ = original_init
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_with_single_tag_filter(test_client: AsyncClient) -> None:
    """GET /api/search?q=...&tags=... returns filtered results."""
    response = await test_client.get(
        "/api/search", params={"q": "guide", "tags": "python"}
    )

    assert response.status_code == 200
    results = response.json()

    # Should only return notes with "python" tag
    assert len(results) == 2  # python-guide.md and python-web.md
    paths = {r["note_path"] for r in results}
    assert "docs/python-guide.md" in paths
    assert "docs/python-web.md" in paths
    assert "docs/javascript-guide.md" not in paths


@pytest.mark.asyncio
async def test_search_with_multiple_tags_repeated_param(test_client: AsyncClient) -> None:
    """Multiple tags parameter (repeated) works correctly with AND logic."""
    # Use repeated tags parameters for AND logic
    response = await test_client.get(
        "/api/search",
        params=[("q", "python"), ("tags", "python"), ("tags", "web")],
    )

    assert response.status_code == 200
    results = response.json()

    # Only python-web.md has both "python" AND "web" tags
    assert len(results) == 1
    assert results[0]["note_path"] == "docs/python-web.md"


@pytest.mark.asyncio
async def test_search_with_empty_tags_returns_all_matches(test_client: AsyncClient) -> None:
    """Empty tags parameter is handled gracefully, returns all FTS matches."""
    # Search without tags should return all matching results
    response = await test_client.get("/api/search", params={"q": "guide"})

    assert response.status_code == 200
    results = response.json()

    # Should return all notes matching "guide"
    assert len(results) >= 3
    paths = {r["note_path"] for r in results}
    assert "docs/python-guide.md" in paths
    assert "docs/javascript-guide.md" in paths
    assert "docs/general-guide.md" in paths


@pytest.mark.asyncio
async def test_search_with_nonexistent_tag_returns_empty(test_client: AsyncClient) -> None:
    """Search with non-existent tag returns empty results."""
    response = await test_client.get(
        "/api/search",
        params={"q": "guide", "tags": "nonexistent-tag"},
    )

    assert response.status_code == 200
    results = response.json()

    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_with_empty_string_tag_ignored(test_client: AsyncClient) -> None:
    """Empty string tags are ignored gracefully."""
    # Empty string tag should be filtered out
    response = await test_client.get(
        "/api/search",
        params=[("q", "guide"), ("tags", ""), ("tags", "python")],
    )

    assert response.status_code == 200
    results = response.json()

    # Should behave as if only "python" tag was passed (2 python notes match "guide")
    assert len(results) == 2
    paths = {r["note_path"] for r in results}
    assert "docs/python-guide.md" in paths
    assert "docs/python-web.md" in paths


@pytest.mark.asyncio
async def test_search_with_case_insensitive_tags(test_client: AsyncClient) -> None:
    """Tag filtering is case-insensitive."""
    # Use uppercase tag
    response = await test_client.get(
        "/api/search", params={"q": "guide", "tags": "PYTHON"}
    )

    assert response.status_code == 200
    results = response.json()

    # Should match notes with "python" tag (case-insensitive)
    assert len(results) >= 1
    paths = {r["note_path"] for r in results}
    assert "docs/python-guide.md" in paths


@pytest.mark.asyncio
async def test_search_response_contains_expected_fields(test_client: AsyncClient) -> None:
    """Search results contain all expected fields."""
    response = await test_client.get(
        "/api/search", params={"q": "python", "tags": "python"}
    )

    assert response.status_code == 200
    results = response.json()

    assert len(results) >= 1
    result = results[0]

    # Verify response schema
    assert "note_path" in result
    assert "title" in result
    assert "snippet" in result
    assert "score" in result
    assert "updated" in result
