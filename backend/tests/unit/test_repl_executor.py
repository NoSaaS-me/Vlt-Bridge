"""Unit tests for REPLExecutor — T011 (022-rlm-oracle).

Tests sandbox security, Final detection, stdout capture, approved modules,
and the timeout mechanism using a short timeout to keep tests fast.
"""
from __future__ import annotations

import asyncio
import pytest

from backend.src.services.repl_executor import (
    ExecutionResult,
    REPLExecutor,
    REPLNamespace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run(code: str, timeout_s: float = 5.0) -> ExecutionResult:
    """Execute *code* and return the single ExecutionResult."""
    ns = REPLNamespace()
    ex = REPLExecutor(namespace=ns, timeout_s=timeout_s)
    result = None
    async for r in ex.execute(code):
        result = r
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------

class TestBasicExec:
    @pytest.mark.asyncio
    async def test_simple_expression_succeeds(self):
        result = await _run("x = 1 + 1")
        assert result.success is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_print_captured_in_stdout_full(self):
        result = await _run("print('hello world')")
        assert result.success is True
        assert "hello world" in result.stdout_full

    @pytest.mark.asyncio
    async def test_stdout_preview_max_200_chars(self):
        # Generate output longer than 200 chars
        result = await _run("print('x' * 500)")
        assert result.stdout_total_chars > 200
        assert len(result.stdout_preview) <= 200

    @pytest.mark.asyncio
    async def test_stdout_total_chars_matches_full_length(self):
        result = await _run("print('abc')")
        assert result.stdout_total_chars == len(result.stdout_full)

    @pytest.mark.asyncio
    async def test_duration_ms_is_positive(self):
        result = await _run("pass")
        assert result.duration_ms > 0


# ---------------------------------------------------------------------------
# Final sentinel
# ---------------------------------------------------------------------------

class TestFinalSentinel:
    @pytest.mark.asyncio
    async def test_final_string_detected(self):
        result = await _run("Final = 'done'")
        assert result.has_final is True
        assert result.final_value == "done"

    @pytest.mark.asyncio
    async def test_final_non_string_coerced_to_str(self):
        result = await _run("Final = 42")
        assert result.has_final is True
        assert result.final_value == "42"

    @pytest.mark.asyncio
    async def test_final_dict_coerced_to_str(self):
        result = await _run("Final = {'key': 'val'}")
        assert result.has_final is True
        assert isinstance(result.final_value, str)

    @pytest.mark.asyncio
    async def test_no_final_when_not_set(self):
        result = await _run("x = 1")
        assert result.has_final is False
        assert result.final_value is None


# ---------------------------------------------------------------------------
# Sandbox security
# ---------------------------------------------------------------------------

class TestSandboxSecurity:
    @pytest.mark.asyncio
    async def test_import_os_blocked(self):
        """RestrictedPython removes __import__ — importing any module is blocked."""
        result = await _run("import os")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_open_blocked(self):
        """open() is removed from safe_builtins."""
        result = await _run("open('/etc/passwd')")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_eval_blocked(self):
        result = await _run("eval('1+1')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_exec_blocked(self):
        result = await _run("exec('x=1')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_subclasses_traversal_blocked(self):
        """safer_getattr should block __subclasses__ and similar dunder attrs."""
        result = await _run("object.__subclasses__()")
        # safer_getattr blocks __subclasses__ access
        assert result.success is False

    @pytest.mark.asyncio
    async def test_globals_builtin_blocked(self):
        result = await _run("globals()")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_locals_builtin_blocked(self):
        result = await _run("locals()")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_dunder_class_attr_blocked(self):
        """Accessing __class__ via attribute should be blocked."""
        result = await _run("x = (1).__class__")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_getattr_available(self):
        """getattr() should work for safe attributes."""
        result = await _run("x = getattr('hello', 'upper')\nFinal = x()")
        assert result.success is True
        assert result.final_value == "HELLO"

    @pytest.mark.asyncio
    async def test_hasattr_available(self):
        """hasattr() should work for safe attributes."""
        result = await _run("Final = str(hasattr('hello', 'upper'))")
        assert result.success is True
        assert result.final_value == "True"

    @pytest.mark.asyncio
    async def test_getattr_blocks_dunder(self):
        """getattr() must still block dunder access."""
        result = await _run("x = getattr(1, '__class__')")
        assert result.success is False


# ---------------------------------------------------------------------------
# Approved stdlib modules
# ---------------------------------------------------------------------------

class TestApprovedModules:
    """Approved modules are pre-injected as globals — no 'import' needed."""

    @pytest.mark.asyncio
    async def test_re_module_available(self):
        # re is injected as a global name, not imported
        result = await _run(
            "m = re.match(r'(\\w+)', 'hello')\n"
            "print(m.group(1))"
        )
        assert result.success is True
        assert "hello" in result.stdout_full

    @pytest.mark.asyncio
    async def test_json_module_available(self):
        result = await _run("print(json.dumps({'key': 'val'}))")
        assert result.success is True
        assert "key" in result.stdout_full

    @pytest.mark.asyncio
    async def test_math_module_available(self):
        result = await _run("print(math.sqrt(9))")
        assert result.success is True
        assert "3.0" in result.stdout_full

    @pytest.mark.asyncio
    async def test_datetime_module_available(self):
        result = await _run(
            "d = datetime.date(2024, 1, 1)\n"
            "print(d.year)"
        )
        assert result.success is True
        assert "2024" in result.stdout_full

    @pytest.mark.asyncio
    async def test_collections_module_available(self):
        result = await _run(
            "c = collections.Counter([1, 2, 2, 3])\n"
            "print(c[2])"
        )
        assert result.success is True
        assert "2" in result.stdout_full

    @pytest.mark.asyncio
    async def test_itertools_module_available(self):
        result = await _run(
            "pairs = list(itertools.chain([1, 2], [3]))\n"
            "print(pairs)"
        )
        assert result.success is True
        assert "1" in result.stdout_full


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    @pytest.mark.asyncio
    async def test_infinite_loop_times_out(self):
        """An infinite loop should be caught by the per-step timeout."""
        # Use a very short timeout (0.5s) so the test doesn't hang
        result = await _run("while True: pass", timeout_s=0.5)
        assert result.success is False
        assert "timed out" in (result.error or "").lower() or "TIMEOUT" in result.stdout_full

    @pytest.mark.asyncio
    async def test_fast_code_completes_within_timeout(self):
        result = await _run("Final = sum(range(1000))", timeout_s=5.0)
        assert result.success is True
        assert result.has_final is True


# ---------------------------------------------------------------------------
# Variables persist across iterations (REPLNamespace)
# ---------------------------------------------------------------------------

class TestNamespacePersistence:
    @pytest.mark.asyncio
    async def test_variable_from_first_exec_visible_in_second(self):
        ns = REPLNamespace()
        ex = REPLExecutor(namespace=ns, timeout_s=5.0)

        # First execution sets x
        async for _ in ex.execute("x = 42"):
            pass

        # Second execution can read x
        result2 = None
        async for r in ex.execute("print(x)"):
            result2 = r

        assert result2 is not None
        assert "42" in result2.stdout_full

    @pytest.mark.asyncio
    async def test_internal_names_not_persisted(self):
        """_INTERNAL_NAMES like 'project' shouldn't leak into _variables."""
        ns = REPLNamespace()
        ex = REPLExecutor(namespace=ns, timeout_s=5.0)
        async for _ in ex.execute("x = 1"):
            pass
        # project should not be in _variables
        assert "project" not in ns._variables
        assert "__builtins__" not in ns._variables


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------

class TestSyntaxErrors:
    @pytest.mark.asyncio
    async def test_syntax_error_returns_failure(self):
        result = await _run("def broken(:\n    pass")
        assert result.success is False
        assert result.error is not None
        assert "SyntaxError" in result.error

    @pytest.mark.asyncio
    async def test_runtime_error_captured_in_error_field(self):
        result = await _run("raise ValueError('oops')")
        assert result.success is False
        assert "ValueError" in (result.error or "")
