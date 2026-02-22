"""REPL Executor — Restricted Python execution engine for RLM Oracle.

Implements FR-015 (restricted namespace), FR-016 (per-step timeout),
FR-017 (iteration budget enforcement).

Components:
    REPLNamespace  — RestrictedPython execution environment for one oracle session.
    REPLExecutor   — Async executor that bridges sync exec() to async SSE streaming.
    QueuedStringIO — io.StringIO subclass that pushes chunks to asyncio.Queue.

Part of 022-rlm-oracle (RLM Oracle replacement).
"""
from __future__ import annotations
