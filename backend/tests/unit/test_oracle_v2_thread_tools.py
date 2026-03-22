"""Unit tests for thread_tools (thread_create / thread_push) — T039.

Tests:
    1. test_thread_create_basic — create succeeds, returns confirmation string
    2. test_thread_create_with_goal — create + goal entry pushed as oracle:init
    3. test_thread_create_invalid_name — special chars rejected
    4. test_thread_create_name_too_long — 81+ chars rejected
    5. test_thread_push_basic — push succeeds, returns confirmation
    6. test_thread_push_thread_not_found — returns error string mentioning thread_create
    7. test_thread_push_content_too_long — over 10KB rejected
    8. test_thread_push_empty_content — empty string rejected
    9. test_thread_push_author_sanitization — invalid author chars rejected
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

try:
    from backend.src.services.oracle_v2.tools.thread_tools import make_thread_tools
    from backend.src.models.thread import Thread, ThreadEntry, ThreadListResponse
    _IMPORTS_OK = True
except ImportError as _err:
    _IMPORTS_OK = False
    _IMPORT_ERR = _err

pytestmark = pytest.mark.skipif(not _IMPORTS_OK, reason="oracle_v2 thread_tools unavailable")

_USER_ID = "test-user"
_PROJECT_ID = "test-project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_thread(thread_id: str, entries=None) -> Thread:
    """Build a minimal Thread model for mock returns."""
    return Thread(
        thread_id=thread_id,
        project_id=_PROJECT_ID,
        name=thread_id,
        status="active",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        entries=entries,
    )


def _make_entry(seq: int) -> ThreadEntry:
    return ThreadEntry(
        entry_id="00000000-0000-0000-0000-000000000001",
        sequence_id=seq,
        content="existing entry",
        author="user",
        timestamp=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_svc():
    """Mock ThreadService with sensible defaults."""
    svc = MagicMock()
    svc.create_or_update_thread.return_value = _make_thread("my-thread")
    svc.add_entries.return_value = (1, 0)
    svc.get_thread.return_value = None
    return svc


@pytest.fixture
def tools(mock_svc):
    """Build tool list with patched get_thread_service.

    The import is deferred inside make_thread_tools, so we patch at the
    source module (backend.src.services.thread_service) — the same object
    that the lazy import resolves to.
    """
    with patch(
        "backend.src.services.thread_service.get_thread_service",
        return_value=mock_svc,
    ):
        tool_list = make_thread_tools(user_id=_USER_ID, project_id=_PROJECT_ID)
    # Closures already captured mock_svc via the patched call; no patch needed after factory.
    return tool_list, mock_svc


def _get(tools_fixture, name: str):
    tool_list, svc = tools_fixture
    for fn in tool_list:
        if fn.__name__ == name:
            return fn, svc
    raise KeyError(f"Tool {name!r} not found")


# ---------------------------------------------------------------------------
# T1: thread_create basic
# ---------------------------------------------------------------------------

def test_thread_create_basic(tools):
    thread_create, svc = _get(tools, "thread_create")
    svc.create_or_update_thread.return_value = _make_thread("my-thread")

    result = thread_create("my-thread")

    assert "Created thread: my-thread" in result
    assert f"Project: {_PROJECT_ID}" in result
    assert "(no goal specified)" in result
    svc.create_or_update_thread.assert_called_once_with(
        user_id=_USER_ID,
        thread_id="my-thread",
        project_id=_PROJECT_ID,
        name="my-thread",
        status="active",
    )
    # No goal → add_entries not called
    svc.add_entries.assert_not_called()


# ---------------------------------------------------------------------------
# T2: thread_create with goal
# ---------------------------------------------------------------------------

def test_thread_create_with_goal(tools):
    thread_create, svc = _get(tools, "thread_create")
    svc.create_or_update_thread.return_value = _make_thread("research-track")

    result = thread_create("research-track", goal="Investigate auth flow")

    assert "Created thread: research-track" in result
    assert "Investigate auth flow" in result

    svc.add_entries.assert_called_once()
    call_kwargs = svc.add_entries.call_args
    entries = call_kwargs.kwargs.get("entries") or call_kwargs[1].get("entries") or call_kwargs[0][2]
    assert len(entries) == 1
    assert entries[0].author == "oracle:init"
    assert entries[0].content == "Investigate auth flow"
    assert entries[0].sequence_id == 0


# ---------------------------------------------------------------------------
# T3: thread_create invalid name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_name", [
    "has space",
    "has!bang",
    "dot.dot",
    "slash/slash",
    "at@sign",
])
def test_thread_create_invalid_name(tools, bad_name):
    thread_create, svc = _get(tools, "thread_create")

    result = thread_create(bad_name)

    assert result.startswith("Error:")
    assert "invalid characters" in result
    svc.create_or_update_thread.assert_not_called()


# ---------------------------------------------------------------------------
# T4: thread_create name too long
# ---------------------------------------------------------------------------

def test_thread_create_name_too_long(tools):
    thread_create, svc = _get(tools, "thread_create")
    long_name = "a" * 81

    result = thread_create(long_name)

    assert result.startswith("Error:")
    assert "too long" in result
    svc.create_or_update_thread.assert_not_called()


# ---------------------------------------------------------------------------
# T5: thread_push basic
# ---------------------------------------------------------------------------

def test_thread_push_basic(tools):
    thread_push, svc = _get(tools, "thread_push")
    # Thread exists with one entry at seq 3
    svc.get_thread.return_value = _make_thread("my-thread", entries=[_make_entry(3)])

    result = thread_push("my-thread", "Important finding here")

    assert "Pushed to thread: my-thread" in result
    assert "Entry #: 4" in result
    assert "Author: oracle" in result
    assert "Important finding here" in result

    svc.add_entries.assert_called_once()
    call_kwargs = svc.add_entries.call_args
    entries = call_kwargs.kwargs.get("entries") or call_kwargs[1].get("entries") or call_kwargs[0][2]
    assert entries[0].sequence_id == 4
    assert entries[0].author == "oracle"


# ---------------------------------------------------------------------------
# T6: thread_push thread not found
# ---------------------------------------------------------------------------

def test_thread_push_thread_not_found(tools):
    thread_push, svc = _get(tools, "thread_push")
    svc.get_thread.return_value = None

    result = thread_push("ghost-thread", "some content")

    assert result.startswith("Error:")
    assert "not found" in result
    assert "thread_create" in result
    svc.add_entries.assert_not_called()


# ---------------------------------------------------------------------------
# T7: thread_push content too large
# ---------------------------------------------------------------------------

def test_thread_push_content_too_long(tools):
    thread_push, svc = _get(tools, "thread_push")

    big_content = "x" * 10241  # just over 10 KB

    result = thread_push("my-thread", big_content)

    assert result.startswith("Error:")
    assert "too large" in result
    svc.add_entries.assert_not_called()


# ---------------------------------------------------------------------------
# T8: thread_push empty content
# ---------------------------------------------------------------------------

def test_thread_push_empty_content(tools):
    thread_push, svc = _get(tools, "thread_push")

    result = thread_push("my-thread", "")

    assert result.startswith("Error:")
    assert "empty" in result
    svc.add_entries.assert_not_called()


# ---------------------------------------------------------------------------
# T9: thread_push author sanitization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_author", [
    "has space",
    "has!bang",
    "dot.dot",
    "slash/slash",
    "at@sign",
])
def test_thread_push_author_sanitization(tools, bad_author):
    thread_push, svc = _get(tools, "thread_push")

    result = thread_push("my-thread", "valid content", author=bad_author)

    assert result.startswith("Error:")
    assert "invalid characters" in result
    svc.add_entries.assert_not_called()
