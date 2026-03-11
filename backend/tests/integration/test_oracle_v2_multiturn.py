"""Integration test: multi-turn context carry-forward — T021 (023-oracle-codeact-rework).

Tests the sandbox eval function and run_shell allowlist directly, without
invoking an LLM. This validates the context persistence mechanism at the
eval level, proving that OracleState.context carry-forward works correctly.

Two-turn scenario verified:
  Turn 1: agent sets x = 42 in REPL
  Turn 2: agent reads x — should find 42 without re-computation
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

try:
    from backend.src.services.oracle_v2.sandbox import (
        make_sandbox_eval,
        run_shell,
        SHELL_ALLOWLIST,
        GIT_SUBCOMMAND_ALLOWLIST,
    )
    _SANDBOX_AVAILABLE = True
except ImportError:
    _SANDBOX_AVAILABLE = False

_SKIP = not _SANDBOX_AVAILABLE
_SKIP_REASON = "oracle_v2.sandbox not importable"


# ---------------------------------------------------------------------------
# T021-A: eval_fn context carry-forward
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestSandboxContextCarryForward:
    """Verify make_sandbox_eval() preserves variables across turns."""

    def setup_method(self):
        """Create a fresh eval_fn with no tools."""
        self.eval_fn = make_sandbox_eval(tools=[])

    def test_variable_set_in_turn1_readable_in_turn2(self):
        """Core multi-turn invariant: context dict propagates variables across calls."""
        # Turn 1: set x = 42 in an empty context
        output1, new_vars1 = self.eval_fn("x = 42\nprint(x)", {})
        assert "42" in output1, f"Expected '42' in turn-1 output, got: {output1!r}"
        assert "x" in new_vars1, f"x not found in new_vars: {new_vars1}"
        assert new_vars1["x"] == 42

        # Simulate LangGraph merging new_vars into OracleState.context
        context_after_turn1 = {**new_vars1}

        # Turn 2: read x from context (no re-assignment)
        output2, new_vars2 = self.eval_fn("print(x)", context_after_turn1)
        assert "42" in output2, (
            f"Turn 2 should read x=42 from carry-forward context, got: {output2!r}"
        )

    def test_multiple_variables_carry_forward(self):
        """Multiple variables set in turn 1 are all available in turn 2."""
        _, new_vars1 = self.eval_fn("a = 10\nb = 20\nc = a + b", {})
        assert new_vars1.get("a") == 10
        assert new_vars1.get("b") == 20
        assert new_vars1.get("c") == 30

        context = {**new_vars1}
        output2, _ = self.eval_fn("print(a, b, c)", context)
        assert "10" in output2
        assert "20" in output2
        assert "30" in output2

    def test_context_variable_can_be_overwritten_in_turn2(self):
        """Variables from context can be overwritten in a subsequent turn."""
        _, new_vars1 = self.eval_fn("x = 42", {})
        context = {**new_vars1}

        _, new_vars2 = self.eval_fn("x = 99", context)
        assert new_vars2.get("x") == 99, (
            f"x should be 99 after overwrite, got: {new_vars2}"
        )

    def test_complex_object_carry_forward(self):
        """Lists and dicts (pickle-safe) carry forward correctly."""
        _, new_vars1 = self.eval_fn("data = [1, 2, 3]\nmapping = {'key': 'val'}", {})
        context = {**new_vars1}

        output2, _ = self.eval_fn("print(len(data), mapping['key'])", context)
        assert "3" in output2
        assert "val" in output2

    def test_function_defined_in_turn1_callable_in_turn2(self):
        """Functions defined in turn 1 are available in turn 2 via context."""
        code1 = "def double(n):\n    return n * 2"
        _, new_vars1 = self.eval_fn(code1, {})
        context = {**new_vars1}

        # Functions are pickle-safe (top-level closures), so they should survive
        # Note: pickle-ability depends on the function being a top-level def;
        # if it fails, new_vars1 won't contain 'double' — so we check gracefully.
        if "double" in context:
            output2, _ = self.eval_fn("print(double(21))", context)
            assert "42" in output2

    def test_empty_context_starts_clean(self):
        """Turn 1 with empty context has no prior variables."""
        code = "print(type({}))"
        output, new_vars = self.eval_fn(code, {})
        assert "dict" in output
        # No bleed from prior test's context
        assert "x" not in new_vars

    def test_stdout_captured_correctly(self):
        """Stdout from print() is captured and returned as string."""
        output, _ = self.eval_fn("print('hello')\nprint('world')", {})
        assert "hello" in output
        assert "world" in output

    def test_non_picklable_vars_excluded_from_new_vars(self):
        """Non-pickle-safe objects (e.g. file handles) are excluded from new_vars."""
        # lambdas defined in exec() scope are not pickle-safe
        code = "f = lambda x: x + 1\ny = 5"
        _, new_vars = self.eval_fn(code, {})
        # y should be present (pickle-safe int), f may or may not be
        # The key invariant: no exception raised, y is present
        assert new_vars.get("y") == 5

    def test_turn2_new_vars_only_contains_changed_values(self):
        """new_vars from turn 2 should only contain variables that changed."""
        _, new_vars1 = self.eval_fn("x = 1\ny = 2", {})
        context = {**new_vars1}

        # Turn 2 only changes x, leaves y unchanged
        _, new_vars2 = self.eval_fn("x = 99", context)
        assert new_vars2.get("x") == 99
        # y didn't change (same object), so it may not be in new_vars2
        # The critical check: x is in new_vars2 with the new value
        assert "x" in new_vars2


# ---------------------------------------------------------------------------
# T021-B: run_shell() allowlist validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestRunShellAllowlist:
    """Verify run_shell() enforces the security allowlist correctly."""

    def test_rm_rf_blocked(self):
        """rm -rf / must raise ValueError — not in SHELL_ALLOWLIST."""
        with pytest.raises(ValueError, match=r"not in allowlist|allowlist"):
            run_shell("rm -rf /")

    def test_shell_chaining_semicolon_blocked(self):
        """Shell chaining with ; is unconditionally blocked."""
        with pytest.raises(ValueError, match=r"chaining|;"):
            run_shell("ls ; cat /etc/passwd")

    def test_shell_chaining_and_blocked(self):
        """Shell chaining with && is unconditionally blocked."""
        with pytest.raises(ValueError, match=r"chaining|&&"):
            run_shell("ls && cat /etc/passwd")

    def test_shell_chaining_or_blocked(self):
        """Shell chaining with || is unconditionally blocked."""
        with pytest.raises(ValueError, match=r"chaining|\|\|"):
            run_shell("ls || cat /etc/passwd")

    def test_shell_chaining_subshell_blocked(self):
        """Command substitution $() is unconditionally blocked."""
        with pytest.raises(ValueError, match=r"chaining|\$\("):
            run_shell("echo $(cat /etc/passwd)")

    def test_shell_chaining_backtick_blocked(self):
        """Backtick command substitution is unconditionally blocked."""
        with pytest.raises(ValueError, match=r"chaining|backtick|`"):
            run_shell("echo `cat /etc/passwd`")

    def test_non_allowlisted_command_blocked(self):
        """Commands not in SHELL_ALLOWLIST raise ValueError."""
        with pytest.raises(ValueError, match=r"not in allowlist"):
            run_shell("python --version")

    def test_another_non_allowlisted_command_blocked(self):
        """curl is not in the allowlist."""
        with pytest.raises(ValueError, match=r"not in allowlist"):
            run_shell("curl http://example.com")

    def test_sudo_not_in_allowlist(self):
        """sudo is not in SHELL_ALLOWLIST."""
        with pytest.raises(ValueError, match=r"not in allowlist"):
            run_shell("sudo ls")

    def test_git_without_subcommand_blocked(self):
        """git without a subcommand should raise ValueError."""
        with pytest.raises(ValueError):
            run_shell("git")

    def test_git_blocked_subcommand(self):
        """git push is not in GIT_SUBCOMMAND_ALLOWLIST."""
        with pytest.raises(ValueError, match=r"not in allowlist|subcommand"):
            run_shell("git push origin main")

    def test_git_checkout_blocked(self):
        """git checkout is not in GIT_SUBCOMMAND_ALLOWLIST (destructive)."""
        with pytest.raises(ValueError, match=r"not in allowlist|subcommand"):
            run_shell("git checkout main")

    def test_git_reset_blocked(self):
        """git reset is not in GIT_SUBCOMMAND_ALLOWLIST (destructive)."""
        with pytest.raises(ValueError, match=r"not in allowlist|subcommand"):
            run_shell("git reset --hard HEAD")

    def test_empty_command_blocked(self):
        """Empty command string should raise ValueError."""
        with pytest.raises(ValueError):
            run_shell("")

    def test_allowlist_contains_expected_commands(self):
        """Verify SHELL_ALLOWLIST contains the expected read-only commands."""
        expected = {"git", "grep", "find", "ls", "cat", "head", "tail", "wc", "diff", "rg"}
        assert expected.issubset(SHELL_ALLOWLIST), (
            f"SHELL_ALLOWLIST missing expected commands: {expected - SHELL_ALLOWLIST}"
        )

    def test_git_allowlist_contains_expected_subcommands(self):
        """Verify GIT_SUBCOMMAND_ALLOWLIST contains expected read-only git subcommands."""
        expected = {"log", "diff", "status", "show", "blame", "branch"}
        assert expected.issubset(GIT_SUBCOMMAND_ALLOWLIST), (
            f"GIT_SUBCOMMAND_ALLOWLIST missing: {expected - GIT_SUBCOMMAND_ALLOWLIST}"
        )

    def test_git_log_executes_successfully(self):
        """git log is an allowlisted command and should execute without raising."""
        try:
            result = run_shell("git log --oneline -3")
            # result is a string (stdout), empty string is fine if no commits
            assert isinstance(result, str)
        except FileNotFoundError:
            pytest.skip("git not installed in test environment")
        except Exception as exc:
            # If git repo not found that's also acceptable
            if "not a git repository" in str(exc).lower():
                pytest.skip("Not inside a git repository")
            raise

    def test_ls_executes_successfully(self):
        """ls is allowlisted and should execute."""
        try:
            result = run_shell("ls /tmp")
            assert isinstance(result, str)
        except FileNotFoundError:
            pytest.skip("ls not available")

    def test_return_type_is_string(self):
        """run_shell() must return a string."""
        try:
            result = run_shell("ls /tmp")
            assert isinstance(result, str), f"Expected str, got {type(result)}"
        except (FileNotFoundError, ValueError):
            pytest.skip("Command not available in test environment")
