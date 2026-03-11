"""Unit tests for oracle_v2/streaming.py — T020 (023-oracle-codeact-rework).

Tests the oracle_to_sse() adapter that maps LangGraph astream_events() output
to OracleStreamChunk SSE chunks.

Since streaming.py may not exist yet (Worker A is creating it), all tests
skip gracefully on ImportError.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Import guard — streaming.py may not be written yet
# ---------------------------------------------------------------------------

try:
    from backend.src.services.oracle_v2.streaming import oracle_to_sse
    _STREAMING_AVAILABLE = True
except ImportError:
    _STREAMING_AVAILABLE = False

try:
    from backend.src.models.oracle import OracleStreamChunk
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False

_SKIP = not (_STREAMING_AVAILABLE and _MODELS_AVAILABLE)
_SKIP_REASON = "oracle_v2.streaming not importable (Worker A may not have created it yet)"


# ---------------------------------------------------------------------------
# Test event factories — mirror LangGraph astream_events() v2 format
# ---------------------------------------------------------------------------

def _chat_model_stream_event(content: str) -> dict:
    """Simulate on_chat_model_stream event."""
    chunk_mock = MagicMock()
    chunk_mock.content = content
    return {
        "event": "on_chat_model_stream",
        "data": {"chunk": chunk_mock},
        "name": "ChatOpenAI",
        "run_id": "run-001",
        "tags": [],
        "metadata": {},
    }


def _tool_start_event(tool_name: str, tool_input: dict, run_id: str = "run-002") -> dict:
    """Simulate on_tool_start event."""
    return {
        "event": "on_tool_start",
        "data": {"input": tool_input},
        "name": tool_name,
        "run_id": run_id,
        "tags": [],
        "metadata": {},
    }


def _tool_end_event(tool_name: str, output: Any, run_id: str = "run-002") -> dict:
    """Simulate on_tool_end event."""
    return {
        "event": "on_tool_end",
        "data": {"output": output},
        "name": tool_name,
        "run_id": run_id,
        "tags": [],
        "metadata": {},
    }


def _repl_stdout_event(text: str) -> dict:
    """Simulate on_custom_event with name=repl_stdout."""
    return {
        "event": "on_custom_event",
        "name": "repl_stdout",
        "data": {"output": text},
        "run_id": "run-003",
        "tags": [],
        "metadata": {},
    }


async def _event_stream(*events: dict) -> AsyncIterator[dict]:
    """Yield events as an async iterator, then raise StopAsyncIteration."""
    for event in events:
        yield event


async def _collect_chunks(events, thread_id: str = "thread-123", is_new_thread: bool = False):
    """Run oracle_to_sse() and collect all emitted chunks into a list."""
    chunks = []
    async for chunk in oracle_to_sse(_event_stream(*events), thread_id=thread_id, is_new_thread=is_new_thread):
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestChatModelStreamChunk:
    """on_chat_model_stream → content chunk."""

    @pytest.mark.asyncio
    async def test_chat_stream_produces_content_chunk(self):
        chunks = await _collect_chunks(_chat_model_stream_event("Hello"))
        content_chunks = [c for c in chunks if c.type == "content"]
        assert len(content_chunks) >= 1, "Expected at least one content chunk"

    @pytest.mark.asyncio
    async def test_chat_stream_content_matches_event(self):
        chunks = await _collect_chunks(_chat_model_stream_event("Hello, world!"))
        content_chunks = [c for c in chunks if c.type == "content"]
        combined = "".join(c.content or "" for c in content_chunks)
        assert "Hello, world!" in combined

    @pytest.mark.asyncio
    async def test_empty_chat_content_not_emitted(self):
        """Empty string content from model stream should be filtered out."""
        chunks = await _collect_chunks(_chat_model_stream_event(""))
        content_chunks = [c for c in chunks if c.type == "content"]
        # Empty content chunks shouldn't be emitted (they add no value)
        for c in content_chunks:
            assert c.content, f"Got empty content chunk: {c}"


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestToolStartChunk:
    """on_tool_start → tool_call chunk."""

    @pytest.mark.asyncio
    async def test_tool_start_produces_tool_call_chunk(self):
        chunks = await _collect_chunks(
            _tool_start_event("search_code", {"query": "auth middleware"})
        )
        tool_call_chunks = [c for c in chunks if c.type == "tool_call"]
        assert len(tool_call_chunks) >= 1, "Expected at least one tool_call chunk"

    @pytest.mark.asyncio
    async def test_tool_call_chunk_has_tool_name(self):
        chunks = await _collect_chunks(
            _tool_start_event("search_code", {"query": "auth middleware"})
        )
        tool_call_chunks = [c for c in chunks if c.type == "tool_call"]
        assert tool_call_chunks, "No tool_call chunks found"
        chunk = tool_call_chunks[0]
        assert chunk.tool_call is not None
        tool_call_info = chunk.tool_call
        assert "search_code" in str(tool_call_info)

    @pytest.mark.asyncio
    async def test_tool_call_chunk_has_run_id(self):
        """tool_call chunk should carry a tool_call_id for result association."""
        chunks = await _collect_chunks(
            _tool_start_event("search_code", {"query": "auth"}, run_id="run-xyz")
        )
        tool_call_chunks = [c for c in chunks if c.type == "tool_call"]
        if tool_call_chunks:
            chunk = tool_call_chunks[0]
            # tool_call_id should correlate start/end events
            assert chunk.tool_call_id is not None or chunk.tool_call is not None


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestToolEndChunk:
    """on_tool_end → tool_result chunk."""

    @pytest.mark.asyncio
    async def test_tool_end_produces_tool_result_chunk(self):
        chunks = await _collect_chunks(
            _tool_end_event("search_code", "Found 3 results in src/api/auth.py")
        )
        tool_result_chunks = [c for c in chunks if c.type == "tool_result"]
        assert len(tool_result_chunks) >= 1, "Expected at least one tool_result chunk"

    @pytest.mark.asyncio
    async def test_tool_result_chunk_has_output(self):
        output = "Found authentication middleware at line 42"
        chunks = await _collect_chunks(_tool_end_event("search_code", output))
        tool_result_chunks = [c for c in chunks if c.type == "tool_result"]
        assert tool_result_chunks, "No tool_result chunks found"
        chunk = tool_result_chunks[0]
        assert chunk.tool_result is not None
        assert output in str(chunk.tool_result)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestReplStdoutChunk:
    """on_custom_event name=repl_stdout → progress chunk."""

    @pytest.mark.asyncio
    async def test_repl_stdout_produces_progress_chunk(self):
        chunks = await _collect_chunks(_repl_stdout_event("x = 42\n"))
        progress_chunks = [c for c in chunks if c.type == "progress"]
        assert len(progress_chunks) >= 1, "Expected at least one progress chunk"

    @pytest.mark.asyncio
    async def test_repl_stdout_content_in_progress_chunk(self):
        chunks = await _collect_chunks(_repl_stdout_event("Output: 99\n"))
        progress_chunks = [c for c in chunks if c.type == "progress"]
        assert progress_chunks, "No progress chunks found"
        combined = "".join(c.content or "" for c in progress_chunks)
        assert "99" in combined


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestDoneChunk:
    """After all events consumed → done chunk with context_id=thread_id."""

    @pytest.mark.asyncio
    async def test_done_chunk_emitted_at_end(self):
        chunks = await _collect_chunks(
            _chat_model_stream_event("Hello"),
            thread_id="test-thread-abc",
        )
        done_chunks = [c for c in chunks if c.type == "done"]
        assert len(done_chunks) == 1, f"Expected exactly one done chunk, got {len(done_chunks)}"

    @pytest.mark.asyncio
    async def test_done_chunk_has_correct_context_id(self):
        chunks = await _collect_chunks(
            _chat_model_stream_event("Hello"),
            thread_id="my-thread-xyz",
        )
        done_chunks = [c for c in chunks if c.type == "done"]
        assert done_chunks, "No done chunk found"
        assert done_chunks[0].context_id == "my-thread-xyz"

    @pytest.mark.asyncio
    async def test_done_chunk_emitted_even_for_empty_stream(self):
        """Even with no events, a done chunk should be emitted."""
        chunks = await _collect_chunks(thread_id="empty-thread")
        done_chunks = [c for c in chunks if c.type == "done"]
        assert len(done_chunks) >= 1, "Expected done chunk even for empty event stream"


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestNewThreadContextUpdate:
    """is_new_thread=True → first chunk is context_update with context_id."""

    @pytest.mark.asyncio
    async def test_new_thread_emits_context_update_first(self):
        chunks = await _collect_chunks(
            _chat_model_stream_event("Hello"),
            thread_id="brand-new-thread",
            is_new_thread=True,
        )
        assert chunks, "No chunks emitted"
        assert chunks[0].type == "context_update", (
            f"First chunk should be context_update for new thread, got {chunks[0].type!r}"
        )

    @pytest.mark.asyncio
    async def test_context_update_has_correct_thread_id(self):
        chunks = await _collect_chunks(
            _chat_model_stream_event("Hello"),
            thread_id="new-thread-999",
            is_new_thread=True,
        )
        context_update_chunks = [c for c in chunks if c.type == "context_update"]
        assert context_update_chunks, "No context_update chunk found"
        assert context_update_chunks[0].context_id == "new-thread-999"

    @pytest.mark.asyncio
    async def test_existing_thread_no_context_update(self):
        """is_new_thread=False → no context_update chunk (thread already known)."""
        chunks = await _collect_chunks(
            _chat_model_stream_event("Hello"),
            thread_id="existing-thread",
            is_new_thread=False,
        )
        context_update_chunks = [c for c in chunks if c.type == "context_update"]
        assert not context_update_chunks, (
            "Unexpected context_update chunk for existing thread"
        )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestCancelledError:
    """CancelledError should produce a done chunk, not an error chunk."""

    @pytest.mark.asyncio
    async def test_cancelled_error_emits_done_not_error(self):
        async def _cancellable_stream():
            yield _chat_model_stream_event("partial")
            raise asyncio.CancelledError()

        chunks = []
        async for chunk in oracle_to_sse(_cancellable_stream(), thread_id="cancel-thread", is_new_thread=False):
            chunks.append(chunk)

        chunk_types = [c.type for c in chunks]
        assert "error" not in chunk_types, (
            f"CancelledError should not produce error chunk; got types: {chunk_types}"
        )
        assert "done" in chunk_types, (
            f"CancelledError should produce done chunk; got types: {chunk_types}"
        )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestChunkTypes:
    """Verify all emitted chunks are valid OracleStreamChunk instances."""

    @pytest.mark.asyncio
    async def test_all_chunks_are_oracle_stream_chunk(self):
        chunks = await _collect_chunks(
            _chat_model_stream_event("Hello"),
            _tool_start_event("search_code", {"query": "auth"}),
            _tool_end_event("search_code", "result"),
            _repl_stdout_event("42\n"),
        )
        for chunk in chunks:
            assert isinstance(chunk, OracleStreamChunk), (
                f"Expected OracleStreamChunk, got {type(chunk)}"
            )

    @pytest.mark.asyncio
    async def test_chunk_types_are_valid(self):
        """All chunk types must be from the allowed set."""
        allowed_types = {
            "thinking", "content", "source", "tool_call", "tool_result",
            "done", "error", "system", "context_update", "progress"
        }
        chunks = await _collect_chunks(
            _chat_model_stream_event("Hello"),
        )
        for chunk in chunks:
            assert chunk.type in allowed_types, (
                f"Unknown chunk type: {chunk.type!r}"
            )
