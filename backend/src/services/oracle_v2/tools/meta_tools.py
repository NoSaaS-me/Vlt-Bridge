"""Meta tools for the Oracle CodeAct agent — self-awareness, plan management, and delegation.

Subagent patterns (from validated architecture research):
- delegate_task: spawns a child CodeAct graph with toolkit exclusion (no delegate_task
  in child), context isolation (clean prompt + task only), and result truncation (2000 tokens).
- Full results saved to /tmp/delegate_results/ for recovery when truncated.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_VALID_STATUSES = frozenset({"done", "in_progress", "blocked", "skipped"})

_STATUS_SYMBOLS: dict[str, str] = {
    "pending": " ",
    "in_progress": "→",
    "done": "✓",
    "blocked": "✗",
    "skipped": "–",
}


DELEGATE_RESULT_MAX_TOKENS = 2000
DELEGATE_RESULT_CHARS_PER_TOKEN = 4  # conservative estimate
DELEGATE_RESULT_MAX_CHARS = DELEGATE_RESULT_MAX_TOKENS * DELEGATE_RESULT_CHARS_PER_TOKEN  # ~8000
DELEGATE_CHILD_RECURSION_LIMIT = 25
DELEGATE_RESULTS_DIR = "/tmp/delegate_results"


def make_meta_tools(
    all_tools: list[Callable],
    plan_ref: list | None = None,
    *,
    delegate_config: Optional[dict[str, Any]] = None,
) -> list[Callable]:
    """Build meta tool callables bound to the full tool list.

    Args:
        all_tools: Every tool callable available in this REPL session.
        plan_ref: Optional mutable list of plan step dicts. Each dict has keys:
                  ``description`` (str), ``status`` (str), ``notes`` (str).
                  When None, update_plan returns a "no plan active" message.
        delegate_config: Optional config for delegate_task. Keys:
            - checkpointer: LangGraph checkpointer for child graph
            - model: Pre-built LangChain chat model
            When None, delegate_task returns a "not configured" message.

    Returns:
        [list_tools, update_plan, delegate_task]
    """
    _plan_ref = plan_ref

    def list_tools() -> str:
        """List all available tools with their descriptions.

        Returns:
            Formatted list of tool names and first line of their docstrings.
        """
        if not all_tools:
            return "No tools available."

        lines = ["Available tools:"]
        for tool in all_tools:
            name = getattr(tool, "__name__", repr(tool))
            doc = getattr(tool, "__doc__", None) or ""
            # First non-empty line of the docstring
            first_line = next(
                (ln.strip() for ln in doc.splitlines() if ln.strip()),
                "(no description)",
            )
            # Build a simple signature hint from annotations
            import inspect
            try:
                sig = inspect.signature(tool)
                params = []
                for pname, param in sig.parameters.items():
                    if param.default is inspect.Parameter.empty:
                        params.append(pname)
                    else:
                        params.append(f"{pname}={param.default!r}")
                sig_str = f"({', '.join(params)})"
            except (ValueError, TypeError):
                sig_str = "(...)"

            lines.append(f"  {name}{sig_str}: {first_line}")

        return "\n".join(lines)

    def update_plan(step_index: int, status: str, notes: str = "") -> str:
        """Update the status of a plan step.

        Args:
            step_index: Zero-based index of the plan step.
            status: New status: 'done', 'in_progress', 'blocked', 'skipped'.
            notes: Optional context about what was done or found.

        Returns:
            Updated plan view showing all steps with current status.
        """
        if _plan_ref is None or len(_plan_ref) == 0:
            return "No plan is active. Plans are created automatically for complex queries."

        if status not in _VALID_STATUSES:
            valid = ", ".join(sorted(_VALID_STATUSES))
            return f"[update_plan error] Invalid status {status!r}. Valid values: {valid}"

        if not (0 <= step_index < len(_plan_ref)):
            return (
                f"[update_plan error] step_index {step_index} out of range "
                f"(plan has {len(_plan_ref)} steps, indices 0–{len(_plan_ref) - 1})"
            )

        # Mutate in place so caller's reference reflects the update
        _plan_ref[step_index]["status"] = status
        if notes:
            _plan_ref[step_index]["notes"] = notes

        # Render plan view
        lines = [f"Plan updated — step {step_index}: {status}\n", "Current plan:"]
        for idx, step in enumerate(_plan_ref):
            s = step.get("status", "pending")
            symbol = _STATUS_SYMBOLS.get(s, " ")
            desc = step.get("description", "")
            step_notes = step.get("notes", "")

            # Align: symbol (2 chars) + status (12 chars padded) + description
            status_label = s.ljust(12)
            lines.append(f"  [{idx}] {symbol} {status_label} {desc}")
            if step_notes:
                lines.append(f"               Notes: {step_notes}")

        return "\n".join(lines)

    def delegate_task(task_description: str, context: str = "") -> str:
        """Delegate a complex multi-step task to an isolated subagent.

        The subagent has access to all your tools EXCEPT delegate_task (no
        recursion). It receives a clean prompt with only the task description
        and optional context — it does NOT inherit your conversation history.

        Use when:
          - The task requires 3+ tool calls
          - It involves research across multiple sources
          - It benefits from focused execution without your accumulated context

        Do NOT use for:
          - Simple single-tool lookups (use the tool directly)
          - Tasks requiring your conversation context (include it in ``context``)

        Args:
            task_description: Clear, self-contained description of what to accomplish.
            context: Optional additional context the subagent needs (file contents,
                     search results, etc.). Keep this concise.

        Returns:
            The subagent's final response (capped at ~2000 tokens). If truncated,
            the full result is saved to /tmp/delegate_results/<task_id>.md and the
            path is included in the truncation notice.
        """
        if delegate_config is None:
            return "[delegate_task] Not configured — checkpointer or model not available."

        import uuid
        child_thread_id = str(uuid.uuid4())

        try:
            import asyncio as _asyncio
            from ..graph import build_oracle_graph, _PickleSerde

            model = delegate_config.get("model")
            if model is None:
                return "[delegate_task] Missing model in config."

            # Toolkit exclusion: child gets ALL tools except delegate_task
            child_tools = [t for t in all_tools if getattr(t, "__name__", "") != "delegate_task"]

            # Child uses a fresh MemorySaver — NOT the parent's AsyncSqliteSaver.
            # The parent's checkpointer has async locks bound to the FastAPI event
            # loop, but delegate_task runs in a sync REPL thread via asyncio.run()
            # which creates a new loop. Using MemorySaver avoids the cross-loop
            # lock conflict. Child state is ephemeral anyway — no need to persist.
            from langgraph.checkpoint.memory import MemorySaver
            child_checkpointer = MemorySaver(serde=_PickleSerde())

            # Build child graph with separate recursion limit
            child_graph = build_oracle_graph(
                tools=child_tools,
                checkpointer=child_checkpointer,
                model=model,
            )

            # Context isolation: child gets clean prompt with task only
            child_prompt = task_description
            if context:
                child_prompt = f"{task_description}\n\nContext:\n{context}"

            config = {
                "configurable": {"thread_id": child_thread_id},
                "recursion_limit": DELEGATE_CHILD_RECURSION_LIMIT,
            }

            # Run child graph synchronously (we're in a sync REPL thread).
            # asyncio.run() creates a fresh event loop — safe because the
            # child's MemorySaver has no pre-bound locks.
            result_messages = _asyncio.run(
                child_graph.ainvoke(
                    {"messages": [("user", child_prompt)]},
                    config=config,
                )
            )

            # Extract the last assistant message as the result
            messages = result_messages.get("messages", [])
            if messages:
                last_msg = messages[-1]
                raw_result = getattr(last_msg, "content", str(last_msg))
            else:
                raw_result = "(subagent produced no output)"

        except Exception as exc:
            logger.error("delegate_task failed: %s", exc, exc_info=True)
            raw_result = f"[delegate_task error] {type(exc).__name__}: {exc}"

        # Always save full result to file for recovery
        os.makedirs(DELEGATE_RESULTS_DIR, exist_ok=True)
        result_path = os.path.join(DELEGATE_RESULTS_DIR, f"{child_thread_id}.md")
        try:
            with open(result_path, "w") as f:
                f.write(f"# Delegate Task Result\n\n")
                f.write(f"**Task:** {task_description}\n\n")
                f.write(f"---\n\n{raw_result}\n")
        except OSError as write_err:
            logger.warning("Failed to save delegate result to %s: %s", result_path, write_err)
            result_path = None

        # Result truncation (~2000 tokens ≈ 8000 chars)
        if len(raw_result) > DELEGATE_RESULT_MAX_CHARS:
            truncation_notice = f"\n...\n[truncated — {len(raw_result)} chars total (~{len(raw_result) // DELEGATE_RESULT_CHARS_PER_TOKEN} tokens)]"
            if result_path:
                truncation_notice += f"\nFull result saved to: {result_path}"
                truncation_notice += f"\nUse run_shell('cat {result_path}') to read the complete output."
            raw_result = raw_result[:DELEGATE_RESULT_MAX_CHARS] + truncation_notice

        return raw_result

    return [list_tools, update_plan, delegate_task]
