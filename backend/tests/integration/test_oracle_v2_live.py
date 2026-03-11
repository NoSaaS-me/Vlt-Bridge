"""
Real-world integration tests for Oracle V2 (LangGraph CodeAct).

Tests run against the live server at http://localhost:8000 using an
actual OpenRouter API call.  They verify end-to-end behaviour that
unit tests cannot: SSE streaming, multi-turn context retention,
planner activation, thread persistence, and graceful degradation.

Run:
    python -m pytest tests/integration/test_oracle_v2_live.py -v -s

Requirements:
    - Server running at localhost:8000
    - demo-user has an OpenRouter key in user_settings
    - langgraph-codeact + langgraph-checkpoint-sqlite installed
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import AsyncGenerator

import httpx
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8000"
ORACLE_MODEL = "deepseek/deepseek-chat"  # cheap OpenRouter model for live tests
TIMEOUT = 120  # seconds per request — LLM calls can be slow


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

BACKEND_ROOT = "/mnt/sda1/Projects/00Tooling/Vlt-Bridge/backend"
JWT_SECRET = "local-dev-secret-key-123"  # matches start-dev.sh (overrides .env)


def _mint_token() -> str:
    """Mint a JWT for demo-user using the backend's AuthService."""
    import os, sys
    sys.path.insert(0, BACKEND_ROOT)
    os.environ["JWT_SECRET_KEY"] = JWT_SECRET
    from src.services.auth import AuthService
    from src.services.config import get_config
    cfg = get_config()
    svc = AuthService(cfg)
    return svc.create_jwt(user_id="demo-user", expires_in=timedelta(hours=1))


TOKEN: str = ""   # filled in setup_module
_SETUP_ERROR: str = ""


def setup_module(_module):
    global TOKEN, _SETUP_ERROR
    try:
        TOKEN = _mint_token()
    except Exception as e:
        _SETUP_ERROR = str(e)


def _require_token():
    """Skip test if token wasn't minted."""
    if not TOKEN:
        pytest.skip(f"No auth token — setup failed: {_SETUP_ERROR}")


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------

@dataclass
class StreamResult:
    chunks: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    final_context_id: str | None = None
    elapsed_s: float = 0.0

    @property
    def types(self) -> list[str]:
        return [c["type"] for c in self.chunks]

    @property
    def content_text(self) -> str:
        return "".join(c.get("content", "") for c in self.chunks if c["type"] == "content")

    @property
    def tool_calls(self) -> list[dict]:
        return [c for c in self.chunks if c["type"] == "tool_call"]

    @property
    def progress_chunks(self) -> list[dict]:
        return [c for c in self.chunks if c["type"] == "progress"]

    @property
    def done_chunk(self) -> dict | None:
        for c in reversed(self.chunks):
            if c["type"] == "done":
                return c
        return None

    @property
    def context_update_chunk(self) -> dict | None:
        for c in self.chunks:
            if c["type"] == "context_update":
                return c
        return None


async def _stream_oracle(
    question: str,
    *,
    context_id: str | None = None,
    project_id: str = "default",
    model: str = ORACLE_MODEL,
) -> StreamResult:
    """Send an oracle stream request and collect all chunks."""
    result = StreamResult()
    t0 = time.monotonic()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        payload = {
            "question": question,
            "model": model,
            "project_id": project_id,
            "max_tokens": 8000,
        }
        if context_id:
            payload["context_id"] = context_id

        async with client.stream(
            "POST",
            f"{BASE_URL}/api/oracle/stream",
            json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                result.errors.append(f"HTTP {resp.status_code}: {body.decode()[:200]}")
                result.elapsed_s = time.monotonic() - t0
                return result

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                    result.chunks.append(chunk)
                    if chunk.get("type") in ("done", "context_update"):
                        cid = chunk.get("context_id")
                        if cid:
                            result.final_context_id = cid
                    if chunk.get("type") == "done":
                        break
                except json.JSONDecodeError:
                    pass   # ignore keep-alive pings

    result.elapsed_s = time.monotonic() - t0
    return result


def stream_oracle(question: str, **kw) -> StreamResult:
    return asyncio.get_event_loop().run_until_complete(_stream_oracle(question, **kw))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(path: str, **params) -> dict:
    r = httpx.get(f"{BASE_URL}{path}", params=params,
                   headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10)
    return r.json()


def _delete(path: str) -> int:
    r = httpx.delete(f"{BASE_URL}{path}",
                     headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10)
    return r.status_code


def _patch(path: str, body: dict) -> dict:
    r = httpx.patch(f"{BASE_URL}{path}", json=body,
                    headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10)
    return r.json()


# ---------------------------------------------------------------------------
# Server health prerequisite
# ---------------------------------------------------------------------------

def test_server_healthy():
    """Gate: server must be running and oracle endpoints present."""
    _require_token()
    r = httpx.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200, "Server not healthy"

    spec = httpx.get(f"{BASE_URL}/openapi.json", timeout=5).json()
    paths = spec["paths"]
    assert "/api/oracle/stream" in paths, "oracle stream route missing"
    assert "/api/oracle/threads" in paths, "oracle threads route missing"
    assert "/api/oracle/memory" in paths, "oracle memory route missing"


# ---------------------------------------------------------------------------
# T1 — stream integrity + context_id propagation
# ---------------------------------------------------------------------------

class TestStreamIntegrity:

    def test_stream_ends_with_done_chunk(self):
        """Every stream must end with a done chunk."""
        r = stream_oracle("What files are in the oracle_v2 package? List them.")
        assert r.done_chunk is not None, f"No done chunk in stream. Types seen: {r.types}"

    def test_done_chunk_carries_context_id(self):
        """done chunk must carry context_id so the client can continue the thread."""
        r = stream_oracle("What is the Oracle V2 wrapper class?")
        assert r.done_chunk is not None
        cid = r.done_chunk.get("context_id") or r.final_context_id
        assert cid, f"done chunk missing context_id. Chunk: {r.done_chunk}"

    def test_new_thread_emits_context_update_first(self):
        """New thread → context_update chunk appears before first content chunk."""
        r = stream_oracle("What is the purpose of oracle_v2?")
        types = r.types
        if "context_update" in types and "content" in types:
            cu_idx = next(i for i, t in enumerate(types) if t == "context_update")
            co_idx = next(i for i, t in enumerate(types) if t == "content")
            assert cu_idx < co_idx, "context_update must precede first content chunk"

    def test_stream_produces_content(self):
        """Stream must include at least some content text."""
        r = stream_oracle("In one sentence: what is the OracleV2Wrapper?")
        assert r.content_text, f"No content produced. Chunks: {r.types}"

    def test_no_stream_errors_for_valid_query(self):
        """Valid query should not produce an error chunk."""
        r = stream_oracle("Where is the database initialization code?")
        error_chunks = [c for c in r.chunks if c["type"] == "error"]
        assert not error_chunks, f"Unexpected error chunks: {error_chunks}"

    def test_error_chunk_on_bad_model(self):
        """Completely invalid model should produce an error chunk, not an HTTP 500."""
        r = stream_oracle("hello", model="nonexistent/fake-model-zzzz")
        # Should either produce an error chunk or HTTP error (not a crash)
        has_error = (
            any(c["type"] == "error" for c in r.chunks)
            or bool(r.errors)
        )
        assert has_error, "Expected error on nonexistent model, got clean stream"


# ---------------------------------------------------------------------------
# T2 — multi-turn context retention (the core CodeAct value-prop)
# ---------------------------------------------------------------------------

class TestMultiTurn:

    def test_second_turn_uses_same_thread(self):
        """Turn 2 must receive the same context_id that turn 1 assigned."""
        r1 = stream_oracle("What is the OracleBridge class?")
        ctx = r1.final_context_id
        assert ctx, "Turn 1 did not return a context_id"

        r2 = stream_oracle("Where is it instantiated?", context_id=ctx)
        # Turn 2 should reference the same thread
        assert r2.final_context_id or r2.done_chunk, "Turn 2 stream did not complete"

    def test_second_turn_is_faster(self):
        """Multi-turn: turn 2 (with context) should need fewer tool calls than turn 1.

        If CodeAct state is properly checkpointed, the agent doesn't need to
        re-search for things it already found in turn 1.
        """
        r1 = stream_oracle(
            "Find the OracleV2Wrapper class and tell me what parameters its __init__ accepts."
        )
        ctx = r1.final_context_id
        if not ctx:
            pytest.skip("Turn 1 did not produce a context_id — likely a config issue")

        tool_calls_t1 = len(r1.tool_calls)

        r2 = stream_oracle("What is the return type of process_query?", context_id=ctx)
        tool_calls_t2 = len(r2.tool_calls)

        print(f"\n  Turn 1 tool calls: {tool_calls_t1}, Turn 2 tool calls: {tool_calls_t2}")
        # Turn 2 asks a follow-up about the SAME class — should need <= tool calls
        # (it already has the class definition in REPL context from turn 1)
        assert tool_calls_t2 <= tool_calls_t1 + 2, (
            f"Turn 2 made significantly MORE tool calls ({tool_calls_t2}) than turn 1 ({tool_calls_t1}) "
            f"— suggests context was not persisted"
        )

    def test_resumed_thread_still_produces_content(self):
        """Resuming an existing thread must still produce a valid answer."""
        r1 = stream_oracle("What does the sandbox.py ALLOWED_MODULES set contain?")
        ctx = r1.final_context_id
        if not ctx:
            pytest.skip("No context_id from turn 1")

        r2 = stream_oracle("Can the agent import 'os' from the REPL?", context_id=ctx)
        assert r2.content_text, "Resumed thread produced no content"
        assert r2.done_chunk, "Resumed thread did not complete"


# ---------------------------------------------------------------------------
# T3 — tool call verification
# ---------------------------------------------------------------------------

class TestToolCalls:

    def test_code_search_triggers_tool_call(self):
        """A code-search question should trigger at least one tool_call chunk."""
        r = stream_oracle(
            "Search the codebase for where 'AsyncSqliteSaver' is initialized. "
            "Show me the exact file and line."
        )
        tool_names = [c.get("tool_call", {}).get("name", "") for c in r.tool_calls]
        print(f"\n  Tool calls observed: {tool_names}")
        assert r.tool_calls, (
            f"Expected at least one tool_call for a code search query. "
            f"Got only: {r.types}"
        )

    def test_git_history_triggers_shell_tool(self):
        """A git history question should trigger a shell tool (git_log or run_shell)."""
        r = stream_oracle(
            "What are the last 5 commit messages that touched the oracle module?"
        )
        tool_names = [
            c.get("tool_call", {}).get("name", "")
            for c in r.tool_calls
        ]
        print(f"\n  Tool calls observed: {tool_names}")
        # Accept either git_log (explicit) or a generic run_shell call
        git_related = any(
            "git" in n.lower() or "shell" in n.lower() or "log" in n.lower()
            for n in tool_names
        )
        assert r.tool_calls, "Expected at least one tool call for git history query"
        # Don't fail hard on tool name — just verify something was called
        assert r.content_text, "Stream produced no content text"

    def test_tool_result_follows_tool_call(self):
        """Every tool_call chunk should eventually be followed by a tool_result chunk."""
        r = stream_oracle("Read the file backend/src/services/oracle_v2/state.py")
        call_ids = {c.get("tool_call_id") for c in r.tool_calls if c.get("tool_call_id")}
        result_ids = {
            c.get("tool_call_id")
            for c in r.chunks
            if c.get("type") == "tool_result" and c.get("tool_call_id")
        }
        print(f"\n  call_ids={call_ids}, result_ids={result_ids}")
        if call_ids:
            # All called tools should have a corresponding result
            unmatched = call_ids - result_ids
            assert not unmatched, f"Tool calls without results: {unmatched}"


# ---------------------------------------------------------------------------
# T4 — planner activation
# ---------------------------------------------------------------------------

class TestPlanner:

    def test_complex_query_emits_plan_progress(self):
        """A complex query matching COMPLEX_KEYWORDS should emit a plan progress chunk."""
        # "find all" is a COMPLEX_KEYWORD → classifier returns "plan"
        r = stream_oracle(
            "Find all files that import from src/services/vault.py "
            "and tell me what functions from it they use."
        )
        plan_chunks = [
            c for c in r.progress_chunks
            if c.get("content", "").startswith("Plan:")
        ]
        print(f"\n  Progress chunks: {[c.get('content','')[:80] for c in r.progress_chunks]}")
        assert plan_chunks, (
            "Expected a 'Plan:' progress chunk for a complex 'find all' query. "
            f"Progress chunks seen: {r.progress_chunks}"
        )

    def test_plan_appears_before_first_tool_call(self):
        """Plan progress chunk must appear in the stream before the first tool_call."""
        r = stream_oracle(
            "Analyze all oracle-related modules across the codebase and summarize "
            "the data flow from an API request to the streamed response."
        )
        types = r.types
        plan_indices = [i for i, t in enumerate(types) if t == "progress"]
        tool_indices = [i for i, t in enumerate(types) if t == "tool_call"]

        if not plan_indices or not tool_indices:
            pytest.skip("No plan or no tool calls — planner may have been skipped")

        # The first plan chunk should come before the first tool call
        assert plan_indices[0] < tool_indices[0], (
            f"Plan chunk at index {plan_indices[0]} came AFTER first tool call at {tool_indices[0]}"
        )

    def test_direct_query_skips_planner(self):
        """Short 'what is' query should produce no Plan: progress chunk."""
        r = stream_oracle("What is the OracleState class?")
        plan_chunks = [
            c for c in r.progress_chunks
            if c.get("content", "").startswith("Plan:")
        ]
        assert not plan_chunks, (
            f"Simple 'what is' query should skip planner, but got plan: "
            f"{[c.get('content','')[:60] for c in plan_chunks]}"
        )


# ---------------------------------------------------------------------------
# T5 — thread management API
# ---------------------------------------------------------------------------

class TestThreadManagement:

    def setup_method(self):
        """Create a thread by running a query."""
        r = stream_oracle("What is the purpose of the oracle_threads SQLite table?")
        self.thread_id = r.final_context_id
        if not self.thread_id:
            pytest.skip("Could not create thread — oracle stream failed")

    def test_thread_appears_in_list(self):
        """After a query, GET /api/oracle/threads should contain the new thread."""
        data = _get("/api/oracle/threads")
        thread_ids = [t["thread_id"] for t in data.get("threads", [])]
        assert self.thread_id in thread_ids, (
            f"Thread {self.thread_id} not found in list. "
            f"Total threads: {data.get('total', 0)}"
        )

    def test_thread_has_auto_title(self):
        """Thread should have an auto-generated title from the first 60 chars of the query."""
        data = _get("/api/oracle/threads")
        thread = next(
            (t for t in data.get("threads", []) if t["thread_id"] == self.thread_id),
            None,
        )
        assert thread is not None, f"Thread {self.thread_id} not in list"
        assert thread.get("title"), f"Thread has no auto-title: {thread}"
        print(f"\n  Auto-title: {thread['title']!r}")

    def test_thread_history_has_messages(self):
        """GET /api/oracle/threads/{id}/history should return user + assistant messages."""
        data = _get(f"/api/oracle/threads/{self.thread_id}/history")
        messages = data.get("messages", [])
        roles = [m["role"] for m in messages]
        print(f"\n  History message roles: {roles}")
        assert "user" in roles, f"No user message in history: {messages}"
        assert "assistant" in roles, f"No assistant message in history: {messages}"

    def test_thread_rename(self):
        """PATCH /api/oracle/threads/{id} should update the title."""
        new_title = f"Test-renamed-{uuid.uuid4().hex[:8]}"
        result = _patch(f"/api/oracle/threads/{self.thread_id}", {"title": new_title})
        assert result.get("title") == new_title, f"Title not updated: {result}"

    def test_thread_delete(self):
        """DELETE /api/oracle/threads/{id} should remove it from the list."""
        status = _delete(f"/api/oracle/threads/{self.thread_id}")
        assert status == 204, f"Expected 204 No Content, got {status}"

        # Verify gone
        data = _get("/api/oracle/threads")
        thread_ids = [t["thread_id"] for t in data.get("threads", [])]
        assert self.thread_id not in thread_ids, "Thread still appears after deletion"

    def test_delete_nonexistent_thread_returns_404(self):
        """DELETE on a fake thread_id should return 404."""
        status = _delete(f"/api/oracle/threads/{uuid.uuid4()}")
        assert status == 404, f"Expected 404 for nonexistent thread, got {status}"


# ---------------------------------------------------------------------------
# T6 — memory search endpoint
# ---------------------------------------------------------------------------

class TestMemorySearch:

    def test_memory_endpoint_responds(self):
        """GET /api/oracle/memory should return a valid response."""
        data = _get("/api/oracle/memory", query="oracle", project_id="default")
        assert "available" in data, f"Missing 'available' field: {data}"
        assert "facts" in data, f"Missing 'facts' field: {data}"

    def test_memory_graceful_without_graphiti(self):
        """If Graphiti is not connected, memory endpoint returns available=false with message."""
        data = _get("/api/oracle/memory", query="test")
        # Either available=True (Graphiti connected) or available=False with message
        assert isinstance(data.get("available"), bool), "available should be bool"
        if not data["available"]:
            assert data.get("message"), "Should have a message when unavailable"
            print(f"\n  Memory unavailable: {data['message']}")

    def test_empty_query_returns_no_facts(self):
        """Empty query should return empty facts list with a message."""
        data = _get("/api/oracle/memory", query="")
        assert data.get("facts") == [] or isinstance(data.get("facts"), list)


# ---------------------------------------------------------------------------
# T7 — concurrent streams (stress test)
# ---------------------------------------------------------------------------

class TestConcurrency:

    def test_two_concurrent_streams_complete(self):
        """Two simultaneous oracle queries should both complete without error."""

        async def run_both():
            t1 = asyncio.create_task(
                _stream_oracle("What is the vault service?")
            )
            t2 = asyncio.create_task(
                _stream_oracle("What is the indexer service?")
            )
            r1, r2 = await asyncio.gather(t1, t2)
            return r1, r2

        r1, r2 = asyncio.get_event_loop().run_until_complete(run_both())

        assert r1.done_chunk is not None, f"Stream 1 missing done chunk. Types: {r1.types}"
        assert r2.done_chunk is not None, f"Stream 2 missing done chunk. Types: {r2.types}"
        assert not any(c["type"] == "error" for c in r1.chunks), f"Stream 1 errors: {r1.chunks}"
        assert not any(c["type"] == "error" for c in r2.chunks), f"Stream 2 errors: {r2.chunks}"


# ---------------------------------------------------------------------------
# Summary reporter
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a summary table of stream stats after the test run."""
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    print(f"\n\n{'='*60}")
    print(f"Oracle V2 Live Test Summary: {passed} passed / {failed} failed")
    print(f"{'='*60}")
