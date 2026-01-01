from pathlib import Path

import pytest

from backend.src.services.database import DatabaseService
from backend.src.services.indexer import IndexerService


@pytest.fixture()
def indexer(tmp_path: Path) -> IndexerService:
    db_path = tmp_path / "index.db"
    db_service = DatabaseService(db_path)
    db_service.initialize()
    return IndexerService(db_service=db_service)


def _note(path: str, title: str, body: str, tags: list[str] | None = None) -> dict:
    metadata = {"title": title}
    if tags is not None:
        metadata["tags"] = tags
    return {
        "path": path,
        "metadata": metadata,
        "body": body,
    }


def test_search_notes_handles_apostrophes(indexer: IndexerService) -> None:
    indexer.index_note(
        "local-dev",
        _note(
            "notes/obrien.md",
            "O'Brien Authentication",
            "Details about O'Brien's authentication flow.",
        ),
    )

    results = indexer.search_notes("local-dev", "O'Brien")

    assert results
    assert results[0]["path"] == "notes/obrien.md"


def test_search_notes_preserves_prefix_queries(indexer: IndexerService) -> None:
    indexer.index_note(
        "local-dev",
        _note(
            "notes/auth.md",
            "Authorization Overview",
            "Prefix search should match auth prefix tokens.",
        ),
    )

    results = indexer.search_notes("local-dev", "auth*")

    assert results
    assert results[0]["path"] == "notes/auth.md"


def test_search_notes_handles_symbol_tokens(indexer: IndexerService) -> None:
    indexer.index_note(
        "local-dev",
        _note(
            "notes/api-docs.md",
            "API & Documentation Guide",
            "Overview covering API & documentation best practices.",
        ),
    )

    results = indexer.search_notes("local-dev", "API & documentation")

    assert results
    assert results[0]["path"] == "notes/api-docs.md"


# --- Tag filtering tests ---


def test_search_notes_filters_by_single_tag(indexer: IndexerService) -> None:
    """Search with a single tag filter returns only notes with that tag."""
    indexer.index_note(
        "local-dev",
        _note(
            "notes/python-guide.md",
            "Python Guide",
            "A comprehensive guide to Python programming.",
            tags=["python", "programming"],
        ),
    )
    indexer.index_note(
        "local-dev",
        _note(
            "notes/javascript-guide.md",
            "JavaScript Guide",
            "A comprehensive guide to JavaScript programming.",
            tags=["javascript", "programming"],
        ),
    )

    # Search for "guide" filtered by "python" tag
    results = indexer.search_notes("local-dev", "guide", tags=["python"])

    assert len(results) == 1
    assert results[0]["path"] == "notes/python-guide.md"


def test_search_notes_filters_by_multiple_tags_with_and_logic(indexer: IndexerService) -> None:
    """Search with multiple tags uses AND logic - notes must have ALL tags."""
    indexer.index_note(
        "local-dev",
        _note(
            "notes/python-web.md",
            "Python Web Development",
            "Building web applications with Python.",
            tags=["python", "web", "backend"],
        ),
    )
    indexer.index_note(
        "local-dev",
        _note(
            "notes/python-data.md",
            "Python Data Science",
            "Data science with Python.",
            tags=["python", "data"],
        ),
    )
    indexer.index_note(
        "local-dev",
        _note(
            "notes/js-web.md",
            "JavaScript Web Development",
            "Building web applications with JavaScript.",
            tags=["javascript", "web", "frontend"],
        ),
    )

    # Search filtered by both "python" AND "web" tags
    results = indexer.search_notes("local-dev", "development", tags=["python", "web"])

    assert len(results) == 1
    assert results[0]["path"] == "notes/python-web.md"


def test_search_notes_returns_empty_when_no_matching_tags(indexer: IndexerService) -> None:
    """Search with non-matching tags returns empty results."""
    indexer.index_note(
        "local-dev",
        _note(
            "notes/python-guide.md",
            "Python Guide",
            "A comprehensive guide to Python programming.",
            tags=["python", "programming"],
        ),
    )

    # Search for "guide" filtered by non-existent tag
    results = indexer.search_notes("local-dev", "guide", tags=["nonexistent-tag"])

    assert len(results) == 0


def test_search_notes_with_empty_tags_array_returns_all_matches(indexer: IndexerService) -> None:
    """Search with empty tags array behaves same as no tag filter."""
    indexer.index_note(
        "local-dev",
        _note(
            "notes/python-guide.md",
            "Python Guide",
            "A comprehensive guide to Python programming.",
            tags=["python"],
        ),
    )
    indexer.index_note(
        "local-dev",
        _note(
            "notes/javascript-guide.md",
            "JavaScript Guide",
            "A comprehensive guide to JavaScript programming.",
            tags=["javascript"],
        ),
    )
    indexer.index_note(
        "local-dev",
        _note(
            "notes/untagged-guide.md",
            "Untagged Guide",
            "A guide with no tags.",
        ),
    )

    # Search with empty tags array
    results = indexer.search_notes("local-dev", "guide", tags=[])

    # Should return all matching notes regardless of tags
    assert len(results) == 3
    paths = {r["path"] for r in results}
    assert paths == {
        "notes/python-guide.md",
        "notes/javascript-guide.md",
        "notes/untagged-guide.md",
    }


def test_search_notes_tag_filter_normalizes_tags(indexer: IndexerService) -> None:
    """Tag filter normalizes tags (case-insensitive matching)."""
    indexer.index_note(
        "local-dev",
        _note(
            "notes/python-guide.md",
            "Python Guide",
            "A comprehensive guide to Python programming.",
            tags=["Python", "Programming"],  # Mixed case in source
        ),
    )

    # Search with lowercase tag filter
    results = indexer.search_notes("local-dev", "guide", tags=["python"])
    assert len(results) == 1
    assert results[0]["path"] == "notes/python-guide.md"

    # Search with uppercase tag filter
    results = indexer.search_notes("local-dev", "guide", tags=["PYTHON"])
    assert len(results) == 1
    assert results[0]["path"] == "notes/python-guide.md"

