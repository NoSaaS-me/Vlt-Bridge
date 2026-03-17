"""Unit tests for sub_oracle subagent patterns (023 Phase 0).

Tests:
- T001: Result truncation at 2000 chars
- T002: Toolkit exclusion (child namespace has no sub_oracle)
- T003: Wall-clock timeout (60s)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.src.services.rlm_oracle import (
    RLMSession,
    SubOracleCallable,
    RecursionDepthExceeded,
)
from backend.src.services.repl_executor import REPLNamespace


# ---------------------------------------------------------------------------
# T001: Result truncation
# ---------------------------------------------------------------------------

class TestResultTruncation:

    def test_short_result_unchanged(self):
        result = "Short answer."
        truncated = SubOracleCallable._truncate_result(result, 2000)
        assert truncated == result

    def test_exact_limit_unchanged(self):
        result = "x" * 2000
        truncated = SubOracleCallable._truncate_result(result, 2000)
        assert truncated == result

    def test_long_result_truncated_with_notice(self):
        result = "x" * 5000
        truncated = SubOracleCallable._truncate_result(result, 2000)
        assert len(truncated) < len(result)
        assert truncated.startswith("x" * 2000)
        assert "[truncated — 5000 chars total]" in truncated

    def test_truncation_preserves_prefix(self):
        result = "A" * 1000 + "B" * 3000
        truncated = SubOracleCallable._truncate_result(result, 2000)
        # First 2000 chars: 1000 A's + 1000 B's
        assert truncated.startswith("A" * 1000 + "B" * 1000)

    def test_class_constant_is_2000(self):
        assert SubOracleCallable.RESULT_MAX_CHARS == 2000


# ---------------------------------------------------------------------------
# T002: Toolkit exclusion
# ---------------------------------------------------------------------------

class TestToolkitExclusion:

    def test_child_namespace_has_no_sub_oracle_when_none(self):
        """When sub_oracle_fn is None, 'sub_oracle' must not appear in globals."""
        ns = REPLNamespace()
        mock_project = MagicMock()
        ns.inject(mock_project, sub_oracle_fn=None)

        import io
        glb = ns.build_restricted_globals(io.StringIO())

        assert "sub_oracle" not in glb, \
            "Child namespace must not contain sub_oracle (toolkit exclusion)"
        assert "project" in glb, \
            "project must still be present in child namespace"
        assert "Final" in glb, \
            "Final must still be present in child namespace"

    def test_root_namespace_has_sub_oracle_when_provided(self):
        """When sub_oracle_fn is a callable, it must appear in globals."""
        ns = REPLNamespace()
        mock_project = MagicMock()
        mock_sub_oracle = MagicMock()
        ns.inject(mock_project, sub_oracle_fn=mock_sub_oracle)

        import io
        glb = ns.build_restricted_globals(io.StringIO())

        assert "sub_oracle" in glb
        assert glb["sub_oracle"] is mock_sub_oracle

    def test_child_loop_injects_none_sub_oracle(self):
        """Verify _run_rlm_child_loop passes sub_oracle_fn=None to namespace.

        We check this by inspecting the source code pattern — the child loop
        must set sub_oracle_fn = None before injecting into namespace.
        """
        import inspect
        from backend.src.services.rlm_oracle import _run_rlm_child_loop
        source = inspect.getsource(_run_rlm_child_loop)
        assert "sub_oracle_fn = None" in source, \
            "Child loop must set sub_oracle_fn = None (toolkit exclusion)"

    def test_no_depth_guard_in_sub_oracle_call(self):
        """Depth-based recursion GUARD should be removed from __call__.

        recursion_depth may still appear in debug logs, but it must NOT
        be used in an if/raise guard. Recursion is prevented by toolkit
        exclusion instead.
        """
        import inspect
        source = inspect.getsource(SubOracleCallable.__call__)
        # The old guard was: if self._parent.recursion_depth >= 2: raise
        assert "recursion_depth >= " not in source, \
            "SubOracleCallable should not guard on recursion_depth (use toolkit exclusion)"

    def test_call_count_guard_still_works(self):
        """MAX_SUB_ORACLE_CALLS limit must still be enforced."""
        parent = RLMSession.create_root(user_id="u", query="q")
        parent.sub_oracle_call_count = 3
        sub = SubOracleCallable(
            parent_session=parent,
            api_key="key",
            model="model",
            project_id="proj",
        )
        with pytest.raises(RecursionDepthExceeded, match="exhausted"):
            sub("test prompt")


# ---------------------------------------------------------------------------
# T003: Wall-clock timeout
# ---------------------------------------------------------------------------

class TestWallClockTimeout:

    def test_timeout_constant_is_60s(self):
        assert SubOracleCallable.CHILD_TIMEOUT_S == 60.0

    def test_timeout_returns_partial_result_or_message(self):
        """When child loop times out, should return partial_result or timeout message."""
        parent = RLMSession.create_root(user_id="u", query="q")
        parent.sub_oracle_call_count = 0
        sub = SubOracleCallable(
            parent_session=parent,
            api_key="key",
            model="model",
            project_id="proj",
        )

        # Mock _run_rlm_child_loop to hang forever
        async def _hang(**kwargs):
            await asyncio.sleep(999)
            return "never reached"

        with patch("backend.src.services.rlm_oracle._run_rlm_child_loop", _hang), \
             patch.object(SubOracleCallable, "CHILD_TIMEOUT_S", 0.1):
            result = sub("test prompt")

        assert "timed out" in result.lower()

    def test_timeout_returns_partial_result_when_available(self):
        """When child has partial_result, timeout should return it."""
        parent = RLMSession.create_root(user_id="u", query="q")
        parent.sub_oracle_call_count = 0
        sub = SubOracleCallable(
            parent_session=parent,
            api_key="key",
            model="model",
            project_id="proj",
        )

        async def _hang_with_partial(**kwargs):
            session = kwargs.get("session")
            if session:
                session.partial_result = "partial findings so far"
            await asyncio.sleep(999)
            return "never reached"

        with patch("backend.src.services.rlm_oracle._run_rlm_child_loop", _hang_with_partial), \
             patch.object(SubOracleCallable, "CHILD_TIMEOUT_S", 0.1):
            result = sub("test prompt")

        assert result == "partial findings so far"

    def test_successful_child_not_truncated_under_limit(self):
        """Normal child result under 2000 chars should pass through unchanged."""
        parent = RLMSession.create_root(user_id="u", query="q")
        parent.sub_oracle_call_count = 0
        sub = SubOracleCallable(
            parent_session=parent,
            api_key="key",
            model="model",
            project_id="proj",
        )

        short_result = "Found the auth middleware in src/api/middleware/auth.py"

        async def _quick(**kwargs):
            return short_result

        with patch("backend.src.services.rlm_oracle._run_rlm_child_loop", _quick):
            result = sub("test prompt")

        assert result == short_result

    def test_successful_long_child_result_truncated(self):
        """Normal child result over 2000 chars should be truncated."""
        parent = RLMSession.create_root(user_id="u", query="q")
        parent.sub_oracle_call_count = 0
        sub = SubOracleCallable(
            parent_session=parent,
            api_key="key",
            model="model",
            project_id="proj",
        )

        long_result = "A" * 5000

        async def _verbose(**kwargs):
            return long_result

        with patch("backend.src.services.rlm_oracle._run_rlm_child_loop", _verbose):
            result = sub("test prompt")

        assert len(result) < 5000
        assert "[truncated — 5000 chars total]" in result
