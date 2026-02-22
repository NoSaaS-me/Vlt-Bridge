"""Unit tests for RLM Oracle — T018 (022-rlm-oracle).

Tests RLMSession, RLMPromptBuilder, _extract_code, SubOracleCallable, and
RLMOracleWrapper.  LLM and ProjectContext are mocked; REPLExecutor runs real
restricted Python.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.src.services.rlm_oracle import (
    RLMSession,
    RLMPromptBuilder,
    SubOracleCallable,
    RLMOracleWrapper,
    RecursionDepthExceeded,
    _extract_code,
    ROOT_MAX_ITERATIONS,
    SUB_MAX_ITERATIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project_context_mock(summary: str = "5 files, 10KB. Languages: python:5") -> MagicMock:
    """Return a mock ProjectContext whose get_manifest().summary() returns *summary*."""
    manifest_mock = MagicMock()
    manifest_mock.summary.return_value = summary
    ctx_mock = MagicMock()
    ctx_mock.get_manifest.return_value = manifest_mock
    return ctx_mock


def _make_llm_mock(content: str) -> AsyncMock:
    """Return an AsyncMock OpenRouterClient whose complete() always returns *content*."""
    llm_mock = AsyncMock()
    llm_mock.complete = AsyncMock(return_value={"content": content, "usage": {}})
    return llm_mock


async def _collect_chunks(wrapper: RLMOracleWrapper, query: str = "test query"):
    """Drive process_query() to completion and return all chunks."""
    chunks = []
    async for chunk in wrapper.process_query(query):
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# TestRLMSession
# ---------------------------------------------------------------------------

class TestRLMSession:

    def test_create_root_has_correct_fields(self):
        session = RLMSession.create_root(user_id="u1", query="What is X?")
        # session_id must be a valid UUID4
        parsed = uuid.UUID(session.session_id, version=4)
        assert str(parsed) == session.session_id
        assert session.recursion_depth == 0
        assert session.max_iterations == ROOT_MAX_ITERATIONS
        assert session.status == "running"
        assert session.iteration_count == 0
        assert session.user_id == "u1"
        assert session.query == "What is X?"
        assert session.llm_history == []
        assert session.final_value is None

    def test_create_sub_inherits_parent_fields(self):
        parent = RLMSession.create_root(user_id="u2", query="parent query", project_id="proj1")
        sub = RLMSession.create_sub(parent, prompt="sub prompt")
        assert sub.recursion_depth == parent.recursion_depth + 1
        assert sub.max_iterations == SUB_MAX_ITERATIONS
        assert sub.user_id == parent.user_id
        assert sub.project_id == parent.project_id
        assert sub.context_id == parent.context_id
        assert sub.query == "sub prompt"
        # Sub session has its own session_id
        assert sub.session_id != parent.session_id

    def test_budget_remaining_decreases(self):
        session = RLMSession.create_root(user_id="u3", query="q")
        assert session.budget_remaining == ROOT_MAX_ITERATIONS
        session.iteration_count = 5
        assert session.budget_remaining == ROOT_MAX_ITERATIONS - 5
        session.iteration_count = ROOT_MAX_ITERATIONS
        assert session.budget_remaining == 0

    def test_is_budget_exhausted_true_at_limit(self):
        session = RLMSession.create_root(user_id="u4", query="q")
        session.iteration_count = session.max_iterations
        assert session.is_budget_exhausted() is True

    def test_is_budget_exhausted_false_below_limit(self):
        session = RLMSession.create_root(user_id="u5", query="q")
        session.iteration_count = session.max_iterations - 1
        assert session.is_budget_exhausted() is False


# ---------------------------------------------------------------------------
# TestRLMPromptBuilder
# ---------------------------------------------------------------------------

class TestRLMPromptBuilder:

    def setup_method(self):
        self.builder = RLMPromptBuilder()

    def test_build_system_prompt_contains_project_summary(self):
        summary = "42 files, 850KB. Languages: python:30"
        prompt = self.builder.build_system_prompt(summary)
        assert summary in prompt

    def test_build_iteration_message_metadata_only(self):
        # 500-char stdout should appear only as a truncated preview (<= 200 chars)
        long_stdout = "x" * 500
        msg = self.builder.build_iteration_message(
            stdout_full=long_stdout,
            stdout_total_chars=500,
            error=None,
            iteration_number=1,
        )
        # The full content must NOT appear verbatim
        assert long_stdout not in msg
        # The preview must be present but truncated
        preview = long_stdout[:200]
        assert preview in msg or repr(preview) in msg
        # "[truncated]" must appear
        assert "truncated" in msg.lower()

    def test_build_iteration_message_with_error(self):
        msg = self.builder.build_iteration_message(
            stdout_full="",
            stdout_total_chars=0,
            error="NameError: x is not defined",
            iteration_number=2,
        )
        assert "NameError: x is not defined" in msg

    def test_build_initial_user_message_contains_query(self):
        query = "Where is the authentication logic?"
        msg = self.builder.build_initial_user_message(query, "10 files, 20KB. Languages: python:10")
        assert query in msg


# ---------------------------------------------------------------------------
# TestExtractCode
# ---------------------------------------------------------------------------

class TestExtractCode:

    def test_extract_python_fenced_block(self):
        response = "Some prose.\n```python\nresult = 42\n```\nEnd."
        code = _extract_code(response)
        assert code == "result = 42"

    def test_extract_bare_fenced_block(self):
        response = "See below:\n```\nFinal = 'done'\n```"
        code = _extract_code(response)
        assert code == "Final = 'done'"

    def test_extract_no_block_returns_empty(self):
        response = "The answer is forty-two. No code block here."
        code = _extract_code(response)
        assert code == ""


# ---------------------------------------------------------------------------
# TestRLMOracleWrapper — mock LLM + ProjectContext
# ---------------------------------------------------------------------------

PATCH_LLM = "backend.src.services.openrouter_client.OpenRouterClient"
PATCH_CTX = "backend.src.services.project_context.build_project_context"


class TestRLMOracleWrapper:

    def _make_wrapper(self, **kwargs) -> RLMOracleWrapper:
        return RLMOracleWrapper(
            user_id="test-user",
            api_key="test-key",
            project_id="test-project",
            model="test/model",
            max_tokens=512,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_process_query_yields_oracle_stream_chunks(self):
        ctx_mock = _make_project_context_mock()
        llm_mock = _make_llm_mock("```python\nFinal = 'done'\n```")

        with patch(PATCH_LLM, return_value=llm_mock), \
             patch(PATCH_CTX, return_value=ctx_mock):
            wrapper = self._make_wrapper()
            chunks = await _collect_chunks(wrapper, "What is X?")

        assert len(chunks) > 0
        types = [c.type for c in chunks]
        assert "content" in types

    @pytest.mark.asyncio
    async def test_loop_terminates_when_final_set(self):
        ctx_mock = _make_project_context_mock()
        llm_mock = _make_llm_mock("```python\nFinal = 'answer'\n```")

        with patch(PATCH_LLM, return_value=llm_mock), \
             patch(PATCH_CTX, return_value=ctx_mock):
            wrapper = self._make_wrapper()
            chunks = await _collect_chunks(wrapper, "Tell me about Y")

        types = [c.type for c in chunks]
        # Must yield a done chunk
        assert "done" in types
        # Content chunk must carry the final value
        content_chunks = [c for c in chunks if c.type == "content"]
        assert any("answer" in (c.content or "") for c in content_chunks)
        # Done chunk must not carry incomplete flag
        done_chunk = next(c for c in chunks if c.type == "done")
        assert not (done_chunk.metadata or {}).get("incomplete")

    @pytest.mark.asyncio
    async def test_loop_terminates_at_budget_with_no_final(self):
        ctx_mock = _make_project_context_mock()
        # LLM always returns code that prints but never sets Final
        llm_mock = _make_llm_mock("```python\nprint('working')\n```")

        with patch(PATCH_LLM, return_value=llm_mock), \
             patch(PATCH_CTX, return_value=ctx_mock), \
             patch("backend.src.services.rlm_oracle.ROOT_MAX_ITERATIONS", 2):
            wrapper = self._make_wrapper()
            chunks = await _collect_chunks(wrapper, "Loop me")

        types = [c.type for c in chunks]
        assert "done" in types
        done_chunk = next(c for c in chunks if c.type == "done")
        assert (done_chunk.metadata or {}).get("incomplete") is True

    @pytest.mark.asyncio
    async def test_cancel_stops_iteration(self):
        ctx_mock = _make_project_context_mock()
        llm_mock = _make_llm_mock("```python\nFinal = 'done'\n```")

        with patch(PATCH_LLM, return_value=llm_mock), \
             patch(PATCH_CTX, return_value=ctx_mock):
            wrapper = self._make_wrapper()
            wrapper.cancel()
            chunks = await _collect_chunks(wrapper, "Cancelled query")

        assert len(chunks) >= 1
        assert chunks[0].type == "error"

    @pytest.mark.asyncio
    async def test_yields_progress_chunk_for_stdout(self):
        ctx_mock = _make_project_context_mock()
        # First call: print and don't set Final; second call: set Final
        call_count = 0

        async def _complete_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"content": "```python\nprint('hello')\n```", "usage": {}}
            return {"content": "```python\nFinal = 'done'\n```", "usage": {}}

        llm_mock = AsyncMock()
        llm_mock.complete = AsyncMock(side_effect=_complete_side_effect)

        with patch(PATCH_LLM, return_value=llm_mock), \
             patch(PATCH_CTX, return_value=ctx_mock):
            wrapper = self._make_wrapper()
            chunks = await _collect_chunks(wrapper, "Show me progress")

        progress_chunks = [c for c in chunks if c.type == "progress"]
        assert len(progress_chunks) >= 1
        combined_content = "".join(c.content or "" for c in progress_chunks)
        assert "hello" in combined_content


# ---------------------------------------------------------------------------
# TestSubOracleCallable
# ---------------------------------------------------------------------------

class TestSubOracleCallable:

    def _make_sub_oracle(self, recursion_depth: int = 0, call_count: int = 0) -> SubOracleCallable:
        parent = RLMSession.create_root(user_id="u", query="q")
        parent.recursion_depth = recursion_depth
        parent.sub_oracle_call_count = call_count
        return SubOracleCallable(
            parent_session=parent,
            api_key="key",
            model="model",
            project_id="proj",
        )

    def test_recursion_depth_exceeded_raises(self):
        """Parent at depth >= 2 should raise RecursionDepthExceeded immediately."""
        sub_oracle = self._make_sub_oracle(recursion_depth=2, call_count=0)
        with pytest.raises(RecursionDepthExceeded):
            sub_oracle("some prompt")

    def test_call_count_exceeded_raises(self):
        """More than MAX_SUB_ORACLE_CALLS should raise RecursionDepthExceeded."""
        sub_oracle = self._make_sub_oracle(recursion_depth=0, call_count=3)
        with pytest.raises(RecursionDepthExceeded):
            sub_oracle("some prompt")


# ---------------------------------------------------------------------------
# Phase 4: US2 — Focused Query Efficiency (T021)
# ---------------------------------------------------------------------------

class TestFocusedQueryEfficiency:
    """T021 — focused query should complete in ≤3 iterations."""

    def _make_wrapper(self, **kwargs) -> RLMOracleWrapper:
        return RLMOracleWrapper(
            user_id="test-user",
            api_key="test-key",
            project_id="test-project",
            model="test/model",
            max_tokens=512,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_focused_query_completes_in_few_iterations(self):
        """Mock LLM returns Final on 2nd iteration; done chunk must have iteration_count <= 3."""
        ctx_mock = _make_project_context_mock()
        call_count = 0

        async def _complete_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First iteration: print something (simulate a search result), no Final yet
                return {"content": "```python\nresults = project.search('foo')\nprint(results)\n```", "usage": {}}
            # Second iteration: set Final
            return {"content": "```python\nFinal = 'The function is in services/foo.py'\n```", "usage": {}}

        llm_mock = AsyncMock()
        llm_mock.complete = AsyncMock(side_effect=_complete_side_effect)

        with patch(PATCH_LLM, return_value=llm_mock), \
             patch(PATCH_CTX, return_value=ctx_mock):
            wrapper = self._make_wrapper()
            chunks = await _collect_chunks(wrapper, "Where is the foo function defined?")

        done_chunks = [c for c in chunks if c.type == "done"]
        assert len(done_chunks) == 1

        done_chunk = done_chunks[0]
        iteration_count = (done_chunk.metadata or {}).get("iteration_count")
        assert iteration_count is not None, "done chunk must carry iteration_count in metadata"
        assert iteration_count <= 3, f"Focused query took {iteration_count} iterations, expected ≤3"

    @pytest.mark.asyncio
    async def test_done_chunk_always_carries_iteration_count(self):
        """iteration_count must be present in done chunk metadata for all termination paths."""
        ctx_mock = _make_project_context_mock()
        # Direct Final on first iteration
        llm_mock = _make_llm_mock("```python\nFinal = 'quick answer'\n```")

        with patch(PATCH_LLM, return_value=llm_mock), \
             patch(PATCH_CTX, return_value=ctx_mock):
            wrapper = self._make_wrapper()
            chunks = await _collect_chunks(wrapper, "Quick question")

        done_chunk = next((c for c in chunks if c.type == "done"), None)
        assert done_chunk is not None
        assert "iteration_count" in (done_chunk.metadata or {}), \
            "iteration_count must always be in done chunk metadata"
        assert (done_chunk.metadata or {})["iteration_count"] >= 1
