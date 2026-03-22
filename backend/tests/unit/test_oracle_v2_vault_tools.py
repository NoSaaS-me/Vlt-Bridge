"""Unit tests for vault_list and vault_backlinks tools in oracle_v2/tools/vault_tools.py.

Tests:
    1. vault_list() without directory filter returns all notes sorted by date desc
    2. vault_list() with directory argument passes folder to VaultService
    3. vault_list() when no notes returns descriptive empty message
    4. vault_list() clamps limit > 200 to 200
    5. vault_backlinks() formats backlinks correctly when found
    6. vault_backlinks() returns no-backlinks message when list is empty
    7. vault_backlinks() returns error string when IndexerService raises
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import guard — skip suite if oracle_v2 deps are unavailable
# ---------------------------------------------------------------------------

try:
    from backend.src.services.oracle_v2.tools.vault_tools import make_vault_tools
    _IMPORTS_OK = True
except ImportError as _err:
    _IMPORTS_OK = False
    _IMPORT_ERR = _err

pytestmark = pytest.mark.skipif(
    not _IMPORTS_OK, reason="oracle_v2 vault_tools deps unavailable"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _make_note(path: str, title: str, last_modified: datetime) -> dict:
    return {"path": path, "title": title, "last_modified": last_modified}


def _make_backlink(path: str, title: str) -> dict:
    return {"path": path, "title": title}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_services():
    """Patch VaultService and IndexerService at the vault_tools module level."""
    with (
        patch("backend.src.services.oracle_v2.tools.vault_tools.VaultService") as MockVault,
        patch("backend.src.services.oracle_v2.tools.vault_tools.IndexerService") as MockIndexer,
    ):
        vault_instance = MagicMock()
        indexer_instance = MagicMock()
        MockVault.return_value = vault_instance
        MockIndexer.return_value = indexer_instance
        yield vault_instance, indexer_instance


# ---------------------------------------------------------------------------
# T1: vault_list — all notes, sorted by last_modified descending
# ---------------------------------------------------------------------------

def test_vault_list_all(mock_services):
    vault_svc, _ = mock_services
    vault_svc.list_notes.return_value = [
        _make_note("Alpha.md", "Alpha", _dt(2026, 1, 10)),
        _make_note("Beta.md", "Beta", _dt(2026, 3, 5)),
        _make_note("Gamma.md", "Gamma", _dt(2025, 12, 1)),
    ]

    tools = make_vault_tools("user1", "proj1")
    vault_list = tools[3]

    result = vault_list()

    # Should call with no folder
    vault_svc.list_notes.assert_called_once_with("user1", folder=None, project_id="proj1")

    # Header checks
    assert "Vault notes in: (all)" in result
    assert "showing 3 of 3" in result

    # Most recent (Beta, 2026-03-05) should appear before Alpha (2026-01-10)
    beta_pos = result.index("Beta.md")
    alpha_pos = result.index("Alpha.md")
    gamma_pos = result.index("Gamma.md")
    assert beta_pos < alpha_pos < gamma_pos

    # Date formatting
    assert "2026-03-05" in result
    assert "2026-01-10" in result
    assert "2025-12-01" in result


# ---------------------------------------------------------------------------
# T2: vault_list — scoped to a directory
# ---------------------------------------------------------------------------

def test_vault_list_with_directory(mock_services):
    vault_svc, _ = mock_services
    vault_svc.list_notes.return_value = [
        _make_note("Research/Note1.md", "Note One", _dt(2026, 2, 20)),
    ]

    tools = make_vault_tools("user1", "proj1")
    vault_list = tools[3]

    result = vault_list(directory="Research")

    vault_svc.list_notes.assert_called_once_with("user1", folder="Research", project_id="proj1")
    assert "Vault notes in: Research" in result
    assert "Research/Note1.md" in result
    assert "Note One" in result


# ---------------------------------------------------------------------------
# T3: vault_list — no notes returns descriptive message
# ---------------------------------------------------------------------------

def test_vault_list_empty(mock_services):
    vault_svc, _ = mock_services
    vault_svc.list_notes.return_value = []

    tools = make_vault_tools("user1", "proj1")
    vault_list = tools[3]

    result = vault_list(directory="NonExistent")

    assert "No notes found" in result
    assert "NonExistent" in result
    assert "vault_list()" in result


# ---------------------------------------------------------------------------
# T4: vault_list — limit > 200 is clamped to 200
# ---------------------------------------------------------------------------

def test_vault_list_limit_clamped(mock_services):
    vault_svc, _ = mock_services
    # Return 210 notes (all same date for simplicity)
    vault_svc.list_notes.return_value = [
        _make_note(f"Note{i}.md", f"Title {i}", _dt(2026, 1, 1))
        for i in range(210)
    ]

    tools = make_vault_tools("user1", "proj1")
    vault_list = tools[3]

    result = vault_list(limit=500)

    # Should show only 200 despite requesting 500
    assert "showing 200 of 210" in result


# ---------------------------------------------------------------------------
# T5: vault_backlinks — found, formatted correctly
# ---------------------------------------------------------------------------

def test_vault_backlinks_found(mock_services):
    _, indexer_svc = mock_services
    indexer_svc.get_backlinks.return_value = [
        _make_backlink("Projects/Plan.md", "Project Plan"),
        _make_backlink("Notes/Summary.md", "Summary"),
    ]

    tools = make_vault_tools("user1", "proj1")
    vault_backlinks = tools[4]

    result = vault_backlinks("Architecture/Overview.md")

    indexer_svc.get_backlinks.assert_called_once_with(
        "user1", target_path="Architecture/Overview.md", project_id="proj1"
    )

    assert "Backlinks for: Architecture/Overview.md" in result
    assert "2 notes link to this" in result
    assert "Projects/Plan.md" in result
    assert "Project Plan" in result
    assert "Notes/Summary.md" in result
    assert "Summary" in result


# ---------------------------------------------------------------------------
# T6: vault_backlinks — no backlinks returns simple message
# ---------------------------------------------------------------------------

def test_vault_backlinks_none(mock_services):
    _, indexer_svc = mock_services
    indexer_svc.get_backlinks.return_value = []

    tools = make_vault_tools("user1", "proj1")
    vault_backlinks = tools[4]

    result = vault_backlinks("Orphan/Note.md")

    assert "No notes link to 'Orphan/Note.md'" in result


# ---------------------------------------------------------------------------
# T7: vault_backlinks — service raises, tool returns error string
# ---------------------------------------------------------------------------

def test_vault_backlinks_error(mock_services):
    _, indexer_svc = mock_services
    indexer_svc.get_backlinks.side_effect = RuntimeError("DB connection lost")

    tools = make_vault_tools("user1", "proj1")
    vault_backlinks = tools[4]

    result = vault_backlinks("Any/Note.md")

    assert "Error" in result
    assert "DB connection lost" in result
