"""RLM REPL Executor — Restricted Python execution engine for RLM Oracle.

Implements FR-015 (restricted namespace), FR-016 (per-step timeout),
FR-017 (iteration budget enforcement).

Components:
    QueuedStringIO — io.StringIO subclass that pushes chunks to asyncio.Queue (thread-safe).
    ExecutionResult — Dataclass capturing the outcome of one REPL step.
    REPLNamespace   — RestrictedPython execution environment for one oracle session.
    REPLExecutor    — Async executor that bridges sync exec() to async SSE streaming.

Part of 022-rlm-oracle (RLM Oracle replacement).
"""
from __future__ import annotations

import asyncio
import collections
import datetime
import io
import itertools
import json
import logging
import math
import re
import string
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom RestrictedPython policy that allows safe dunder attributes
# ---------------------------------------------------------------------------

# Dunder attributes that LLMs commonly use for introspection but are safe.
_ALLOWED_DUNDER_ATTRS = frozenset({
    "__name__", "__qualname__", "__module__", "__doc__",
    "__len__", "__str__", "__repr__", "__bool__",
    "__iter__", "__next__", "__contains__",
    "__hash__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__",
    "__abs__", "__neg__", "__pos__", "__invert__",
    "__int__", "__float__", "__complex__",
    "__enter__", "__exit__",
    "__dict__",  # read-only introspection of dataclasses, etc.
})


def _make_policy():
    """Create a custom RestrictedPython policy at import time."""
    import ast
    from RestrictedPython import RestrictingNodeTransformer
    from RestrictedPython.transformer import INSPECT_ATTRIBUTES, copy_locations

    class _RLMNodeTransformer(RestrictingNodeTransformer):
        """Allow access to safe dunder attributes like __name__, __doc__.

        Fully overrides visit_Attribute to relax the blanket underscore ban
        for a curated set of safe dunder names while keeping all other
        security transformations intact.
        """

        def visit_Attribute(self, node):
            # Block non-allowed underscore attributes
            if node.attr.startswith('_') and node.attr != '_':
                if node.attr not in _ALLOWED_DUNDER_ATTRS:
                    self.error(
                        node,
                        f'"{node.attr}" is an invalid attribute name because it '
                        f'starts with "_".',
                    )

            if node.attr.endswith('__roles__'):
                self.error(
                    node,
                    f'"{node.attr}" is an invalid attribute name because it ends '
                    f'with "__roles__".',
                )

            if node.attr in INSPECT_ATTRIBUTES:
                self.error(
                    node,
                    f'"{node.attr}" is a restricted name,'
                    ' that is forbidden to access in RestrictedPython.',
                )

            # Transform reads: obj.attr → _getattr_(obj, "attr")
            if isinstance(node.ctx, ast.Load):
                node = self.node_contents_visit(node)
                new_node = ast.Call(
                    func=ast.Name('_getattr_', ast.Load()),
                    args=[node.value, ast.Constant(node.attr)],
                    keywords=[])
                copy_locations(new_node, node)
                return new_node

            # Transform writes/deletes: obj.attr = val → _write_(obj).attr = val
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                node = self.node_contents_visit(node)
                new_value = ast.Call(
                    func=ast.Name('_write_', ast.Load()),
                    args=[node.value],
                    keywords=[])
                copy_locations(new_value, node.value)
                node.value = new_value
                return node

            else:
                raise NotImplementedError(f"Unknown ctx type: {type(node.ctx)}")

    return _RLMNodeTransformer


try:
    _RLMPolicy = _make_policy()
except ImportError:
    _RLMPolicy = None  # RestrictedPython not installed; will fail at execute()

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

# Marks that the Final variable was not set by the LLM in this iteration.
_FINAL_NOT_SET = object()

# EOF marker pushed to the queue once the thread finishes executing.
_QUEUE_EOF = object()

# Internal variable names that should never be copied into _variables.
_INTERNAL_NAMES = frozenset({
    "project",
    "sub_oracle",
    "Final",
    "__builtins__",
    "_getattr_",
    "_getiter_",
    "_iter_unpack_sequence_",
    "_print_",
    "_print",
    "_inplacevar_",
    "_getitem_",
    "_write_",
    "_apply_",
})


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Captures the outcome of one REPL execution step."""

    success: bool
    stdout_full: str           # All stdout captured during this step.
    stdout_preview: str        # First 200 chars of stdout (for LLM history).
    stdout_total_chars: int    # Total length of stdout.
    error: Optional[str]       # Exception message/traceback if failed.
    has_final: bool            # Whether Final variable was set this iteration.
    final_value: Any           # The value of Final if has_final is True.
    duration_ms: float


# ---------------------------------------------------------------------------
# QueuedStringIO
# ---------------------------------------------------------------------------

class QueuedStringIO(io.StringIO):
    """StringIO that streams written chunks to an asyncio.Queue (thread-safe).

    This runs inside a thread-pool thread, so it uses
    ``asyncio.run_coroutine_threadsafe`` to push chunks back to the event loop.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        chunk_size: int = 256,
    ) -> None:
        super().__init__()
        self._queue = queue
        self._loop = loop
        self._chunk_size = chunk_size
        self._pending_buffer: str = ""

    def write(self, s: str) -> int:
        """Accumulate text; flush to queue when the buffer reaches chunk_size."""
        self._pending_buffer += s
        if len(self._pending_buffer) >= self._chunk_size:
            chunk = self._pending_buffer
            self._pending_buffer = ""
            try:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(chunk), self._loop
                ).result(timeout=0.1)
            except Exception:
                # Non-fatal: the main loop may have timed out.
                logger.debug("QueuedStringIO: failed to push chunk", exc_info=True)
        return len(s)

    def flush_remaining(self) -> None:
        """Push any remaining buffered text to the queue."""
        if self._pending_buffer:
            chunk = self._pending_buffer
            self._pending_buffer = ""
            try:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(chunk), self._loop
                ).result(timeout=0.1)
            except Exception:
                logger.debug("QueuedStringIO: failed to flush remaining", exc_info=True)


# ---------------------------------------------------------------------------
# REPLNamespace
# ---------------------------------------------------------------------------

class REPLNamespace:
    """Restricted execution namespace for one RLM Oracle session.

    Variables accumulated across iterations are stored in ``_variables`` so
    each successive ``exec()`` call can see the results of prior steps.
    """

    ALLOWED_MODULES: dict[str, Any] = {
        "re": re,
        "json": json,
        "math": math,
        "datetime": datetime,
        "collections": collections,
        "itertools": itertools,
        "string": string,
    }

    def __init__(self) -> None:
        self._variables: dict[str, Any] = {}
        self._sub_oracle_fn: Any = None
        self._project_context: Any = None

    def inject(self, project_context: Any, sub_oracle_fn: Any) -> None:
        """Inject project context and sub_oracle callable from RLMOracleWrapper."""
        self._project_context = project_context
        self._sub_oracle_fn = sub_oracle_fn

    def build_restricted_globals(
        self,
        stdout_capture: "QueuedStringIO | io.StringIO",
    ) -> dict:
        """Build the RestrictedPython globals dict for exec().

        The ``_print_`` key must be the *class* (factory), not an instance.
        RestrictedPython compiles ``print(x)`` into:
            ``_print = _print_(_getattr_=_getattr_)``
            ``_print._call_print(x)``

        We provide a subclass of PrintCollector whose ``write()`` method
        tees output to ``stdout_capture``.

        RestrictedPython rewrites several Python constructs into guarded forms:
            - ``obj[key]``        -> ``_getitem_(obj, key)``
            - ``obj[key] = val``  -> ``_write_(obj)[key] = val``
            - ``obj.attr = val``  -> ``_write_(obj).attr = val``
            - ``x += y``         -> ``x = _inplacevar_(op, x, y)``
            - ``a, b = seq``     -> ``_unpack_sequence_(seq, 2)``
            - ``for x in it``    -> ``_getiter_(it)``
        """
        from RestrictedPython import safe_builtins
        from RestrictedPython.Guards import (
            guarded_iter_unpack_sequence,
            guarded_unpack_sequence,
            safer_getattr,
        )
        from RestrictedPython.PrintCollector import PrintCollector

        def _dict_aware_write_guard(ob: Any) -> Any:
            """Write guard that allows dict/list subclasses (Counter, defaultdict, etc.).

            RestrictedPython's full_write_guard uses ``type(ob) is dict`` — an exact
            type check — which rejects Counter, defaultdict, OrderedDict, etc.
            Using isinstance() is equally safe because dict subclasses have the
            same item-assignment semantics.
            """
            if isinstance(ob, (dict, list)):
                return ob
            if hasattr(ob, "_guarded_writes"):
                return ob
            raise TypeError("object does not support item or slice assignment")

        # Build a per-call PrintCollector subclass that streams to our capture.
        _capture = stdout_capture  # closed over

        class _StreamingPrintCollector(PrintCollector):
            """PrintCollector that also tees output to ``_capture``."""

            def write(self, text: str) -> None:  # type: ignore[override]
                super().write(text)
                _capture.write(text)

        # Strip dangerous builtins.
        safe_builtins_dict: dict = dict(safe_builtins)
        for blocked in ("__import__", "open", "eval", "exec", "compile", "globals", "locals"):
            safe_builtins_dict.pop(blocked, None)

        # Dunder names that are safe to read (introspection, not jail-break)
        _SAFE_DUNDERS = frozenset({
            "__name__", "__qualname__", "__module__", "__doc__",
            "__len__", "__str__", "__repr__", "__bool__",
            "__iter__", "__next__", "__contains__",
            "__hash__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
            "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__",
            "__abs__", "__neg__", "__pos__", "__invert__",
            "__int__", "__float__", "__complex__",
            "__enter__", "__exit__",
        })

        # Dangerous dunders that enable sandbox escape
        _BLOCKED_DUNDERS = frozenset({
            "__subclasses__", "__bases__", "__mro__", "__class__",
            "__init_subclass__", "__set_name__",
            "__globals__", "__code__", "__func__",
            "__self__", "__builtins__",
            "__import__", "__loader__", "__spec__",
            "__reduce__", "__reduce_ex__",
            "__getattribute__", "__setattr__", "__delattr__",
        })

        def _guarded_getattr(obj: Any, name: str) -> Any:
            """Attribute access guard: allows safe dunders, blocks dangerous ones."""
            if name.startswith("_"):
                if name in _SAFE_DUNDERS:
                    return getattr(obj, name)
                if name in _BLOCKED_DUNDERS:
                    raise AttributeError(
                        f"Access to '{name}' is restricted in the sandbox."
                    )
                # Other underscore names: use safer_getattr's default policy
                return safer_getattr(obj, name)
            return getattr(obj, name)

        # Provide safe getattr/hasattr that go through our guard
        def _safe_getattr_builtin(obj: Any, name: str, *default: Any) -> Any:
            try:
                return _guarded_getattr(obj, name)
            except AttributeError:
                if default:
                    return default[0]
                raise

        def _safe_hasattr_builtin(obj: Any, name: str) -> bool:
            try:
                _guarded_getattr(obj, name)
                return True
            except (AttributeError, TypeError):
                return False

        # RestrictedPython 8.x safe_builtins is very minimal — re-add the
        # common safe builtins that were omitted.  Deliberately excluded:
        # object/vars/dir (enable class hierarchy traversal / namespace inspection).
        _extra_safe = {
            "getattr": _safe_getattr_builtin,
            "hasattr": _safe_hasattr_builtin,
            "type": type,
            "list": list,
            "dict": dict,
            "set": set,
            "frozenset": frozenset,
            "enumerate": enumerate,
            "filter": filter,
            "map": map,
            "min": min,
            "max": max,
            "sum": sum,
            "all": all,
            "any": any,
            "next": next,
            "iter": iter,
            "reversed": reversed,
            "format": format,
            # super() omitted: not useful in top-level REPL scripts and can aid
            # class hierarchy traversal attacks (__mro__ / __subclasses__ via super()).
        }
        safe_builtins_dict.update(_extra_safe)

        def _safe_getitem(obj: Any, key: Any) -> Any:
            """Guard for subscript reads: obj[key]."""
            return obj[key]

        def _safe_inplacevar(op: str, x: Any, y: Any) -> Any:
            """Guard for augmented assignment operators (+=, -=, *=, etc.)."""
            _OPS = {
                "+=": lambda a, b: a + b,
                "-=": lambda a, b: a - b,
                "*=": lambda a, b: a * b,
                "/=": lambda a, b: a / b,
                "//=": lambda a, b: a // b,
                "%=": lambda a, b: a % b,
                "**=": lambda a, b: a ** b,
                "&=": lambda a, b: a & b,
                "|=": lambda a, b: a | b,
                "^=": lambda a, b: a ^ b,
                ">>=": lambda a, b: a >> b,
                "<<=": lambda a, b: a << b,
            }
            fn = _OPS.get(op)
            if fn is None:
                raise NotImplementedError(f"Unsupported in-place operator: {op!r}")
            return fn(x, y)

        # Restricted __import__ that only allows approved modules.
        # LLMs naturally write "import re" etc. — this makes it work
        # without opening the full import system.
        _allowed_import_names = set(self.ALLOWED_MODULES.keys())
        _allowed_import_map = dict(self.ALLOWED_MODULES)

        def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in _allowed_import_names:
                return _allowed_import_map[name]
            raise ImportError(
                f"Module '{name}' is not available. "
                f"Allowed: {', '.join(sorted(_allowed_import_names))}"
            )

        safe_builtins_dict["__import__"] = _restricted_import

        # Compose globals.
        glb: dict = {
            "__builtins__": safe_builtins_dict,
            # Attribute access guard (allows safe dunders, blocks dangerous ones).
            "_getattr_": _guarded_getattr,
            # Subscript read guard: obj[key].
            "_getitem_": _safe_getitem,
            # Write guard: obj[key]=val and obj.attr=val.
            # Uses isinstance() so dict subclasses (Counter, defaultdict, etc.) work.
            "_write_": _dict_aware_write_guard,
            # Iteration guard: for x in iterable.
            "_getiter_": iter,
            # Sequence unpacking guards: a, b = seq.
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_unpack_sequence_": guarded_unpack_sequence,
            # In-place operator guard: x += y.
            "_inplacevar_": _safe_inplacevar,
            # Print routing — must be the class (factory), not an instance.
            "_print_": _StreamingPrintCollector,
            # Allowed stdlib modules.
            **self.ALLOWED_MODULES,
            # Accumulated session variables from prior iterations.
            **self._variables,
            # REPL-injected objects.
            "project": self._project_context,
            # sub_oracle is only injected for root sessions. Children get
            # None (toolkit exclusion) — sub_oracle won't appear in their
            # namespace, causing NameError if the LLM tries to call it.
            **({"sub_oracle": self._sub_oracle_fn} if self._sub_oracle_fn is not None else {}),
            # Final sentinel — LLM sets this to signal completion.
            "Final": _FINAL_NOT_SET,
        }
        return glb

    def has_final(self, namespace: dict) -> bool:
        """Return True if the LLM set the ``Final`` variable this iteration."""
        return namespace.get("Final", _FINAL_NOT_SET) is not _FINAL_NOT_SET

    def get_final(self, namespace: dict) -> Any:
        """Return the Final value, coerced to a human-readable string.

        Preference order:
        1. str  → returned as-is
        2. dict/list → JSON pretty-printed (readable; avoids Python repr single-quotes)
        3. anything else → str() fallback
        """
        value = namespace.get("Final", _FINAL_NOT_SET)
        if value is _FINAL_NOT_SET:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, indent=2, default=str)
            except Exception:
                pass
        return str(value)

    def update_variables(self, namespace: dict) -> None:
        """Extract user-defined variables from namespace into ``_variables``.

        Skips internal / dunder names and the pre-injected reserved names so
        that only the LLM's own output variables persist across iterations.
        """
        for key, value in namespace.items():
            if key.startswith("_"):
                continue
            if key in _INTERNAL_NAMES:
                continue
            self._variables[key] = value


# ---------------------------------------------------------------------------
# REPLExecutor
# ---------------------------------------------------------------------------

class REPLExecutor:
    """Executes LLM-generated Python in a REPLNamespace with streaming stdout.

    The user-supplied code is compiled by RestrictedPython and then run inside
    a daemon thread (via ``loop.run_in_executor``).  A ``QueuedStringIO``
    bridges the thread's synchronous writes back to an ``asyncio.Queue``,
    which this async generator drains to build ``ExecutionResult``.
    """

    # Shared EOF sentinel across all instances.
    SENTINEL: object = _QUEUE_EOF

    def __init__(
        self,
        namespace: REPLNamespace,
        timeout_s: float = 30.0,
    ) -> None:
        self._namespace = namespace
        self._timeout_s = timeout_s

    async def execute(self, code: str) -> AsyncGenerator[ExecutionResult, None]:  # type: ignore[override]
        """Execute *code* in the restricted namespace and yield one ExecutionResult.

        Although declared as an ``AsyncGenerator`` for API consistency, this
        implementation yields exactly **one** item — the final result after all
        stdout has been collected.  Callers that want live streaming should
        relay ``result.stdout_full`` over SSE after receiving the yield.

        Raises nothing — all exceptions are captured into the returned
        ``ExecutionResult``.
        """
        from RestrictedPython import compile_restricted

        start = time.monotonic()

        # ------------------------------------------------------------------ #
        # 1. Compile with RestrictedPython (fast path — no thread needed).
        # ------------------------------------------------------------------ #
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                policy = _RLMPolicy  # Our custom policy allowing safe dunders
                if policy is not None:
                    byte_code = compile_restricted(
                        code, "<rlm-repl>", "exec", policy=policy,
                    )
                else:
                    byte_code = compile_restricted(code, "<rlm-repl>", "exec")
        except SyntaxError as exc:
            yield ExecutionResult(
                success=False,
                stdout_full="",
                stdout_preview="",
                stdout_total_chars=0,
                error=f"SyntaxError: {exc}",
                has_final=False,
                final_value=None,
                duration_ms=(time.monotonic() - start) * 1000,
            )
            return

        # ------------------------------------------------------------------ #
        # 2. Set up the async queue + streaming stdout.
        # ------------------------------------------------------------------ #
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        queued_io = QueuedStringIO(queue, loop, chunk_size=256)

        # Build the execution globals once (includes injected variables).
        exec_globals = self._namespace.build_restricted_globals(queued_io)

        # Shared error slot: [exception_str | None]
        error_slot: list[Optional[str]] = [None]

        # ------------------------------------------------------------------ #
        # 3. Thread target: runs exec() then signals EOF.
        # ------------------------------------------------------------------ #
        def _sync_exec() -> None:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = queued_io
            sys.stderr = queued_io
            try:
                exec(byte_code, exec_globals)  # noqa: S102
            except Exception as exc:
                tb = traceback.format_exc()
                error_msg = f"{type(exc).__name__}: {exc}\n{tb}"
                error_slot[0] = error_msg
                # Also push the error text to stdout so the LLM sees it.
                try:
                    queued_io.write(error_msg)
                except Exception:
                    pass
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                queued_io.flush_remaining()
                # Signal EOF — always, even on exception.
                asyncio.run_coroutine_threadsafe(
                    queue.put(_QUEUE_EOF), loop
                ).result(timeout=1.0)

        # ------------------------------------------------------------------ #
        # 4. Launch daemon thread and drain the queue.
        # ------------------------------------------------------------------ #
        stdout_chunks: list[str] = []
        timed_out = False

        # Use a daemon thread so that an infinite-loop timeout does NOT block
        # process exit.  We can't use run_in_executor (non-daemon threads).
        thread_done = asyncio.Event()

        def _sync_exec_with_signal() -> None:
            try:
                _sync_exec()
            finally:
                loop.call_soon_threadsafe(thread_done.set)

        t = threading.Thread(target=_sync_exec_with_signal, daemon=True)
        t.start()

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        queue.get(), timeout=self._timeout_s
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    error_slot[0] = (
                        f"Execution timed out after {self._timeout_s:.1f}s"
                    )
                    break

                if chunk is _QUEUE_EOF:
                    break

                stdout_chunks.append(chunk)  # type: ignore[arg-type]
        finally:
            # Give the thread a moment to finish naturally; we don't forcibly
            # kill it (Python threads can't be reliably killed), but we don't
            # block the event loop waiting for it either.
            if not timed_out:
                try:
                    await asyncio.wait_for(thread_done.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass

        # ------------------------------------------------------------------ #
        # 5. Persist session variables and extract Final.
        # ------------------------------------------------------------------ #
        self._namespace.update_variables(exec_globals)
        _has_final = self._namespace.has_final(exec_globals)
        _final_value = self._namespace.get_final(exec_globals) if _has_final else None

        # ------------------------------------------------------------------ #
        # 6. Assemble result.
        # ------------------------------------------------------------------ #
        stdout_full = "".join(stdout_chunks)
        if timed_out and error_slot[0]:
            # Append timeout notice to whatever was captured before the timeout.
            stdout_full = stdout_full + f"\n[TIMEOUT] {error_slot[0]}"

        duration_ms = (time.monotonic() - start) * 1000

        yield ExecutionResult(
            success=error_slot[0] is None and not timed_out,
            stdout_full=stdout_full,
            stdout_preview=stdout_full[:200],
            stdout_total_chars=len(stdout_full),
            error=error_slot[0],
            has_final=_has_final,
            final_value=_final_value,
            duration_ms=duration_ms,
        )
