"""Sandbox execution environment for the Oracle CodeAct agent.

Provides:
    make_sandbox_eval(tools) -> eval_fn   — BYOS eval function for create_codeact()
    run_shell(cmd)                         — allowlisted shell callable injected into _locals

Security model:
    - Custom __import__ restricts Python imports to ALLOWED_MODULES
    - run_shell() is a Python callable in _locals, NOT accessible via exec string
    - Shell chaining (;, &&, ||, $(), backticks) is blocked
    - Shell commands restricted to SHELL_ALLOWLIST
    - git subcommands restricted to GIT_SUBCOMMAND_ALLOWLIST
    - Thread-based 30s timeout kills runaway REPL code
    - new_vars extraction skips non-JSON-serializable values (functions, classes, modules)
      to prevent LangGraph checkpoint serde and astream_events JSON streaming failures

Known constraint (Issue #6447):
    Use sync eval_fn — async get_stream_writer() silently drops custom events.
    Async tool calls invoked via asyncio.run_coroutine_threadsafe() when needed.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import io
import json
import logging
import re
import shlex
import subprocess
import sys
import traceback
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module allowlist
# ---------------------------------------------------------------------------

ALLOWED_MODULES: frozenset[str] = frozenset({
    "re",
    "json",
    "math",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "string",
    "textwrap",
    "pathlib",
})

# ---------------------------------------------------------------------------
# Shell allowlist
# ---------------------------------------------------------------------------

SHELL_ALLOWLIST: frozenset[str] = frozenset({
    "git", "grep", "find", "ls", "cat", "head", "tail", "wc", "diff", "rg",
})

GIT_SUBCOMMAND_ALLOWLIST: frozenset[str] = frozenset({
    "log", "diff", "status", "show", "blame", "branch", "tag", "stash", "shortlog",
})

# Regex patterns for shell chaining — blocked unconditionally
_SHELL_CHAIN_RE = re.compile(r";|&&|\|\||`|\$\(")


def run_shell(cmd: str, timeout: int = 30) -> str:
    """Run an allowlisted shell command and return stdout as a string.

    Only read-only operations are permitted. Shell chaining is blocked.
    Command must start with an allowlisted binary.

    Args:
        cmd: Shell command string (e.g. "git log --oneline -10 src/api/").
        timeout: Seconds before the subprocess is killed (default 30).

    Returns:
        stdout as a string (stderr included on non-zero exit).

    Raises:
        ValueError: If the command is blocked by the allowlist or chain check.
        subprocess.TimeoutExpired: If the command exceeds timeout.
    """
    cmd = cmd.strip()

    # Reject shell chaining
    if _SHELL_CHAIN_RE.search(cmd):
        raise ValueError(
            f"Shell chaining is not allowed (;, &&, ||, $(), backticks): {cmd!r}"
        )

    try:
        parts = shlex.split(cmd)
    except ValueError as exc:
        raise ValueError(f"Cannot parse shell command {cmd!r}: {exc}") from exc

    if not parts:
        raise ValueError("Empty shell command")

    binary = parts[0].lower()
    if binary not in SHELL_ALLOWLIST:
        raise ValueError(
            f"Shell command {binary!r} not in allowlist. "
            f"Allowed: {sorted(SHELL_ALLOWLIST)}"
        )

    # Extra check for git subcommands
    if binary == "git":
        if len(parts) < 2:
            raise ValueError("git requires a subcommand")
        subcommand = parts[1].lower()
        # Strip any leading -- flags before subcommand check
        subcommand = subcommand.lstrip("-")
        if subcommand not in GIT_SUBCOMMAND_ALLOWLIST:
            raise ValueError(
                f"git subcommand {subcommand!r} not in allowlist. "
                f"Allowed: {sorted(GIT_SUBCOMMAND_ALLOWLIST)}"
            )

    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += f"\n[stderr]: {result.stderr.strip()}"
        return output
    except subprocess.TimeoutExpired:
        raise
    except FileNotFoundError:
        raise ValueError(f"Command not found: {binary!r}")


# ---------------------------------------------------------------------------
# Restricted __import__
# ---------------------------------------------------------------------------

def _make_restricted_import(original_import: Callable) -> Callable:
    """Return a replacement __import__ that only allows ALLOWED_MODULES."""

    def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
        base = name.split(".")[0]
        if base in ALLOWED_MODULES:
            return original_import(name, *args, **kwargs)
        raise ImportError(
            f"Import of {name!r} is not allowed in the Oracle sandbox. "
            f"Allowed modules: {sorted(ALLOWED_MODULES)}"
        )

    return _restricted_import


# ---------------------------------------------------------------------------
# Pickle-safe variable extraction
# ---------------------------------------------------------------------------

def _extract_new_vars(local_ns: dict[str, Any], prev_context: dict[str, Any]) -> dict[str, Any]:
    """Extract variables added or changed in local_ns relative to prev_context.

    Skips non-JSON-serializable values (functions, classes, modules, etc.) to
    avoid corrupting LangGraph state in both the checkpoint serde path and the
    astream_events JSON streaming path.
    """
    new_vars: dict[str, Any] = {}
    for key, value in local_ns.items():
        if key.startswith("_"):
            continue
        if key in prev_context and prev_context[key] is value:
            continue
        # Skip types that are never JSON-serializable and would break
        # LangGraph's astream_events event stream serialization.
        if callable(value) or isinstance(value, type) or inspect.ismodule(value):
            logger.debug("Skipping non-JSON-safe REPL var %r (type=%s)", key, type(value).__name__)
            continue
        try:
            json.dumps(value)
            new_vars[key] = value
        except (TypeError, ValueError):
            logger.debug("Skipping non-JSON-safe REPL var %r (type=%s)", key, type(value).__name__)
    return new_vars


# ---------------------------------------------------------------------------
# Eval function factory
# ---------------------------------------------------------------------------

REPL_TIMEOUT_SECONDS = 300  # 5 minutes — soft limit, not a crash


def make_sandbox_eval(tools: list[Callable]) -> Callable[[str, dict], tuple[str, dict]]:
    """Build the sync eval_fn expected by create_codeact().

    The returned function is a sync callable: (code: str, locals: dict) -> (output: str, new_vars: dict)

    Tool callables are injected directly into _locals so the REPL can call
    them as plain Python functions (e.g. search_code("auth middleware")).

    Note: async tool methods must be wrapped to use asyncio.run_coroutine_threadsafe()
    before being passed in as tools — eval_fn runs in a sync thread.

    Args:
        tools: List of callable tools with __name__ and __doc__ attributes.

    Returns:
        eval_fn compatible with create_codeact(eval_fn=...).
    """
    # Mutable timeout override — agent can call set_timeout(seconds) to
    # increase the limit for the next iteration (one-shot, resets after use).
    _timeout_override: list[float] = [0]

    def set_timeout(seconds: int) -> str:
        """Set the REPL timeout for the next code execution.

        Use this before running long operations (e.g. delegate_task, large
        file processing). The override applies to the NEXT iteration only,
        then resets to the default (5 minutes).

        Args:
            seconds: Timeout in seconds (max 600 = 10 minutes).

        Returns:
            Confirmation message.
        """
        clamped = max(10, min(seconds, 600))
        _timeout_override[0] = clamped
        return f"Timeout set to {clamped}s for next execution."

    # Build tool namespace: {tool.__name__: tool, ...}
    tool_namespace: dict[str, Callable] = {t.__name__: t for t in tools}
    tool_namespace["set_timeout"] = set_timeout

    # Pre-import allowed modules into a shared base namespace
    base_globals: dict[str, Any] = {
        "__builtins__": {
            # Safe builtins subset
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "frozenset": frozenset,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "bytes": bytes,
            "bytearray": bytearray,
            "memoryview": memoryview,
            "type": type,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "repr": repr,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "any": any,
            "all": all,
            "next": next,
            "iter": iter,
            "vars": vars,
            "dir": dir,
            "help": help,
            "format": format,
            "chr": chr,
            "ord": ord,
            "hex": hex,
            "oct": oct,
            "bin": bin,
            "hash": hash,
            "id": id,
            "None": None,
            "True": True,
            "False": False,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "NotImplementedError": NotImplementedError,
            "StopIteration": StopIteration,
            "ImportError": ImportError,
            "__import__": _make_restricted_import(__import__),
        },
    }

    def eval_fn(code: str, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Execute code in the sandbox and return (stdout, new_vars).

        Args:
            code: Python code string from the LLM.
            context: Current REPL context (OracleState.context) — persisted variables.

        Returns:
            (stdout_output, new_variables_to_merge_into_context)
        """
        stdout_capture = io.StringIO()
        local_ns: dict[str, Any] = {
            **context,         # existing REPL state
            **tool_namespace,  # tool callables
            "run_shell": run_shell,  # shell access (allowlisted)
        }
        exec_globals = dict(base_globals)

        exec_error: str | None = None

        def _run() -> None:
            nonlocal exec_error
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            try:
                exec(compile(code, "<oracle_repl>", "exec"), exec_globals, local_ns)  # noqa: S102
            except Exception:
                # Behave like a real REPL: print the traceback to stdout
                # so the LLM can see the error and adjust, rather than
                # crashing the entire LangGraph stream.
                exec_error = traceback.format_exc()
                stdout_capture.write(exec_error)
            finally:
                sys.stdout = old_stdout

        # Allow the agent to override timeout per-iteration via set_timeout()
        effective_timeout = _timeout_override[0] if _timeout_override[0] > 0 else REPL_TIMEOUT_SECONDS
        # Reset override after use (one-shot)
        _timeout_override[0] = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            try:
                future.result(timeout=effective_timeout)
            except concurrent.futures.TimeoutError:
                # Soft error — return as stdout so the agent can see it and
                # adjust, instead of crashing the entire LangGraph stream.
                timeout_msg = (
                    f"[TIMEOUT] Code execution exceeded {effective_timeout:.0f}s. "
                    f"Use set_timeout(seconds) before your next code block to increase "
                    f"the limit, or break the work into smaller steps."
                )
                partial_output = stdout_capture.getvalue()
                if partial_output:
                    partial_output += "\n"
                return partial_output + timeout_msg, {}

        output = stdout_capture.getvalue()
        # Only extract new vars if execution succeeded — don't persist
        # partial state from a failed exec.
        new_vars = _extract_new_vars(local_ns, context) if exec_error is None else {}
        return output, new_vars

    return eval_fn
