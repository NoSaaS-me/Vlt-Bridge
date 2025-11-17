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


def _note(path: str, title: str, body: str) -> dict:
    return {
        "path": path,
        "metadata": {"title": title},
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

