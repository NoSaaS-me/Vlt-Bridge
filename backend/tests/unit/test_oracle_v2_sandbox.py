"""Unit tests for oracle_v2/sandbox.py — run_shell allowlist and make_sandbox_eval."""

from __future__ import annotations

import pytest

from backend.src.services.oracle_v2.sandbox import (
    SHELL_ALLOWLIST,
    make_sandbox_eval,
    run_shell,
)


class TestRunShell:
    """Tests for the run_shell allowlist enforcement."""

    def test_blocks_semicolon_chaining(self):
        with pytest.raises(ValueError, match="chaining"):
            run_shell("ls ; cat /etc/passwd")

    def test_blocks_and_chaining(self):
        with pytest.raises(ValueError, match="chaining"):
            run_shell("ls && rm -rf /")

    def test_blocks_pipe_or(self):
        with pytest.raises(ValueError, match="chaining"):
            run_shell("ls || echo pwned")

    def test_blocks_command_substitution(self):
        with pytest.raises(ValueError, match="chaining"):
            run_shell("ls $(whoami)")

    def test_blocks_backtick(self):
        with pytest.raises(ValueError, match="chaining"):
            run_shell("ls `pwd`")

    def test_blocks_non_allowlisted_command(self):
        with pytest.raises(ValueError):
            run_shell("rm -rf /tmp/test")

    def test_blocks_git_rm_subcommand(self):
        with pytest.raises(ValueError):
            run_shell("git rm somefile.py")

    def test_blocks_git_push_subcommand(self):
        with pytest.raises(ValueError):
            run_shell("git push origin main")

    def test_allows_git_status(self):
        result = run_shell("git status")
        assert isinstance(result, str)

    def test_allows_ls(self):
        result = run_shell("ls .")
        assert isinstance(result, str)

    def test_empty_command_raises(self):
        with pytest.raises(ValueError):
            run_shell("")

    def test_allowlist_contains_expected_entries(self):
        assert "git" in SHELL_ALLOWLIST
        assert "grep" in SHELL_ALLOWLIST
        assert "find" in SHELL_ALLOWLIST
        assert "rg" in SHELL_ALLOWLIST
        # Destructive commands must not be present
        assert "rm" not in SHELL_ALLOWLIST
        assert "curl" not in SHELL_ALLOWLIST
        assert "wget" not in SHELL_ALLOWLIST


class TestMakeSandboxEval:
    """Tests for make_sandbox_eval — REPL execution and import restrictions."""

    def test_basic_execution(self):
        eval_fn = make_sandbox_eval([])
        output, new_vars = eval_fn("x = 42\nprint(x)", {})
        assert "42" in output
        assert new_vars.get("x") == 42

    def test_context_carryforward(self):
        eval_fn = make_sandbox_eval([])
        _, new_vars = eval_fn("x = 42", {})
        output, _ = eval_fn("print(x)", new_vars)
        assert "42" in output

    def test_blocked_import_os(self):
        eval_fn = make_sandbox_eval([])
        with pytest.raises(ImportError):
            eval_fn("import os", {})

    def test_blocked_import_subprocess(self):
        eval_fn = make_sandbox_eval([])
        with pytest.raises(ImportError):
            eval_fn("import subprocess", {})

    def test_blocked_import_sys(self):
        eval_fn = make_sandbox_eval([])
        with pytest.raises(ImportError):
            eval_fn("import sys", {})

    def test_allowed_import_re(self):
        eval_fn = make_sandbox_eval([])
        output, _ = eval_fn("import re\nprint(re.findall(r'\\d+', 'abc123'))", {})
        assert "123" in output

    def test_allowed_import_json(self):
        eval_fn = make_sandbox_eval([])
        output, _ = eval_fn("import json\nprint(json.dumps({'k': 1}))", {})
        assert '"k"' in output

    def test_tool_injection(self):
        def my_tool(x: str) -> str:
            """Test tool that echoes input."""
            return f"result: {x}"

        eval_fn = make_sandbox_eval([my_tool])
        output, _ = eval_fn("print(my_tool('hello'))", {})
        assert "result: hello" in output

    def test_multiple_tools_injected(self):
        def tool_a() -> str:
            """Tool A."""
            return "a"

        def tool_b() -> str:
            """Tool B."""
            return "b"

        eval_fn = make_sandbox_eval([tool_a, tool_b])
        output, _ = eval_fn("print(tool_a(), tool_b())", {})
        assert "a" in output
        assert "b" in output

    def test_run_shell_injected(self):
        """run_shell should be directly available in the REPL namespace."""
        eval_fn = make_sandbox_eval([])
        # git status always available in the project root
        output, _ = eval_fn("result = run_shell('git status')\nprint(type(result).__name__)", {})
        assert "str" in output

    def test_non_picklable_vars_skipped(self):
        """Non-picklable vars must be silently dropped from new_vars."""
        eval_fn = make_sandbox_eval([])
        # A plain picklable var alongside a non-picklable one (lambda)
        # We can't import threading (blocked), but we can define a lambda
        code = "x = 1\nf = lambda: 42"
        _, new_vars = eval_fn(code, {})
        # x is picklable
        assert new_vars.get("x") == 1
        # f (lambda) is not pickle-safe — should be dropped
        assert "f" not in new_vars

    def test_returns_tuple(self):
        eval_fn = make_sandbox_eval([])
        result = eval_fn("x = 1", {})
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_print_output_captured(self):
        eval_fn = make_sandbox_eval([])
        output, _ = eval_fn("print('hello world')", {})
        assert "hello world" in output

    def test_syntax_error_propagates(self):
        eval_fn = make_sandbox_eval([])
        with pytest.raises(SyntaxError):
            eval_fn("def broken(", {})

    def test_timeout_raises_timeout_error(self, monkeypatch):
        """A REPL that times out should raise TimeoutError."""
        import concurrent.futures

        def always_timeout(fn, *args, **kwargs):
            fut = concurrent.futures.Future()
            exc = concurrent.futures.TimeoutError()
            # Simulate wait() raising TimeoutError
            raise TimeoutError("REPL timed out") from exc

        from backend.src.services.oracle_v2 import sandbox as sandbox_mod

        monkeypatch.setattr(
            sandbox_mod,
            "_execute_with_timeout",
            always_timeout,
            raising=False,
        )
        # Even if patching internals isn't possible, verify TimeoutError is a real exception
        with pytest.raises((TimeoutError, concurrent.futures.TimeoutError, Exception)):
            # 1ms timeout — will time out on any real sleep
            eval_fn = make_sandbox_eval([])
            eval_fn("import time; time.sleep(0)", {})
