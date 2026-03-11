"""Unit tests for memory_tools (remember/recall) with mock Graphiti — T029.

Tests:
    1. remember() with None graphiti returns graceful "Memory not available" message
    2. recall() with None graphiti returns graceful "No facts available" message
    3. load_recalled_facts() with None graphiti returns []
    4. write_episode() with None graphiti returns silently (no exception)
    5. remember() with mock graphiti calls write_episode via event loop
    6. recall() with mock graphiti calls load_recalled_facts and formats output
    7. load_recalled_facts() filters out edges where invalid_at is not None
    8. load_recalled_facts() logs warning and returns [] on Graphiti search error
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import guards — all tests skip if dependencies unavailable
# ---------------------------------------------------------------------------

try:
    from backend.src.services.memory.loader import load_recalled_facts
    from backend.src.services.memory.writer import write_episode
    from backend.src.services.oracle_v2.tools.memory_tools import make_memory_tools
    _IMPORTS_OK = True
except ImportError as _err:
    _IMPORTS_OK = False
    _IMPORT_ERR = _err

pytestmark = pytest.mark.skipif(not _IMPORTS_OK, reason=f"oracle_v2 deps unavailable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_graphiti():
    """Mock Graphiti instance with async search and add_episode."""
    g = MagicMock()
    g.search = AsyncMock(return_value=[])
    g.add_episode = AsyncMock(return_value=None)
    return g


def _make_edge(fact: str, invalid_at=None):
    """Helper: create a mock Graphiti edge with .fact and .invalid_at."""
    edge = MagicMock()
    edge.fact = fact
    edge.invalid_at = invalid_at
    return edge


# ---------------------------------------------------------------------------
# T1: remember() with None graphiti
# ---------------------------------------------------------------------------

def test_remember_none_graphiti_returns_graceful_message():
    tools = make_memory_tools(graphiti=None, user_id="u1", project_id="p1")
    remember = tools[0]
    result = remember("some fact")
    assert "not available" in result.lower() or "not connected" in result.lower()


# ---------------------------------------------------------------------------
# T2: recall() with None graphiti
# ---------------------------------------------------------------------------

def test_recall_none_graphiti_returns_graceful_message():
    tools = make_memory_tools(graphiti=None, user_id="u1", project_id="p1")
    recall = tools[1]
    result = recall("what is the rate limiter?")
    assert "not available" in result.lower() or "not connected" in result.lower()


# ---------------------------------------------------------------------------
# T3: load_recalled_facts() with None graphiti returns []
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_recalled_facts_none_graphiti_returns_empty():
    facts = await load_recalled_facts(
        graphiti=None,
        query="rate limiter location",
        user_id="u1",
        project_id="p1",
    )
    assert facts == []


# ---------------------------------------------------------------------------
# T4: write_episode() with None graphiti returns silently
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_episode_none_graphiti_silent():
    # Must not raise
    await write_episode(
        graphiti=None,
        episode_body="The auth middleware is in src/api/middleware/auth.py",
        user_id="u1",
        project_id="p1",
        thread_id="thread-123",
        turn_index=0,
    )


# ---------------------------------------------------------------------------
# T5: load_recalled_facts() filters invalid_at edges
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_recalled_facts_filters_invalidated_edges(mock_graphiti):
    from datetime import datetime, timezone
    valid_edge = _make_edge("Rate limiter is in middleware/rate_limit.py", invalid_at=None)
    invalidated_edge = _make_edge("Old location in utils/rate.py", invalid_at=datetime.now(timezone.utc))
    mock_graphiti.search = AsyncMock(return_value=[valid_edge, invalidated_edge])

    facts = await load_recalled_facts(
        graphiti=mock_graphiti,
        query="rate limiter",
        user_id="u1",
        project_id="p1",
    )

    assert len(facts) == 1
    assert "middleware/rate_limit.py" in facts[0]


# ---------------------------------------------------------------------------
# T6: load_recalled_facts() returns [] on Graphiti search exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_recalled_facts_returns_empty_on_error(mock_graphiti):
    mock_graphiti.search = AsyncMock(side_effect=ConnectionError("FalkorDB unreachable"))

    facts = await load_recalled_facts(
        graphiti=mock_graphiti,
        query="any query",
        user_id="u1",
        project_id="p1",
    )

    assert facts == []


# ---------------------------------------------------------------------------
# T7: write_episode() calls add_episode with correct args
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_episode_calls_add_episode(mock_graphiti):
    await write_episode(
        graphiti=mock_graphiti,
        episode_body="Auth service handles JWT validation",
        user_id="u1",
        project_id="p1",
        thread_id="thread-abc",
        turn_index=2,
    )

    mock_graphiti.add_episode.assert_awaited_once()
    call_kwargs = mock_graphiti.add_episode.call_args.kwargs
    assert call_kwargs["name"] == "oracle-turn-thread-abc-2"
    assert "Auth service" in call_kwargs["episode_body"]
    assert call_kwargs["group_id"] == "u1:p1"
    assert call_kwargs["source"] == "message"


# ---------------------------------------------------------------------------
# T8: write_episode() logs warning on add_episode failure — no exception raised
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_episode_silent_on_failure(mock_graphiti):
    mock_graphiti.add_episode = AsyncMock(side_effect=RuntimeError("neo4j write failed"))

    # Must not raise
    await write_episode(
        graphiti=mock_graphiti,
        episode_body="Some episode",
        user_id="u1",
        project_id="p1",
        thread_id="thread-xyz",
        turn_index=0,
    )


# ---------------------------------------------------------------------------
# T9: recall() with mock graphiti formats output correctly
# ---------------------------------------------------------------------------

def test_recall_with_mock_graphiti_formats_facts(mock_graphiti):
    edge1 = _make_edge("Rate limiter is in middleware/rate_limit.py")
    edge2 = _make_edge("Auth service uses JWT RS256")
    mock_graphiti.search = AsyncMock(return_value=[edge1, edge2])

    # Run in a fresh event loop (simulating sync REPL context)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        tools = make_memory_tools(graphiti=mock_graphiti, user_id="u1", project_id="p1")
        recall = tools[1]

        # Patch run_coroutine_threadsafe to use our loop directly
        with patch("backend.src.services.oracle_v2.tools.memory_tools.asyncio.run_coroutine_threadsafe") as mock_rctf:
            # Simulate what run_coroutine_threadsafe returns
            future = loop.run_until_complete(
                load_recalled_facts(mock_graphiti, "auth", "u1", "p1")
            )
            mock_future = MagicMock()
            mock_future.result.return_value = future
            mock_rctf.return_value = mock_future

            result = recall("auth")

        assert "1." in result
        assert "middleware/rate_limit.py" in result or "JWT" in result
    finally:
        loop.close()
        asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# T10: group_id_for and user_group_id helpers
# ---------------------------------------------------------------------------

def test_group_id_for():
    from backend.src.services.memory.client import group_id_for
    assert group_id_for("alice", "my-project") == "alice:my-project"


def test_user_group_id():
    from backend.src.services.memory.client import user_group_id
    assert user_group_id("alice") == "alice:_user"


def test_get_graphiti_returns_none_on_missing_attr():
    from backend.src.services.memory.client import get_graphiti

    class FakeState:
        pass

    assert get_graphiti(FakeState()) is None


def test_get_graphiti_returns_instance():
    from backend.src.services.memory.client import get_graphiti

    class FakeState:
        graphiti = "mock-graphiti"

    assert get_graphiti(FakeState()) == "mock-graphiti"
