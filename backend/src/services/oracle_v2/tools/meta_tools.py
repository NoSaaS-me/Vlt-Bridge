"""Meta tools for the Oracle CodeAct agent — self-awareness and plan management."""

from __future__ import annotations

from typing import Callable

_VALID_STATUSES = frozenset({"done", "in_progress", "blocked", "skipped"})

_STATUS_SYMBOLS: dict[str, str] = {
    "pending": " ",
    "in_progress": "→",
    "done": "✓",
    "blocked": "✗",
    "skipped": "–",
}


def make_meta_tools(all_tools: list[Callable], plan_ref: list | None = None) -> list[Callable]:
    """Build meta tool callables bound to the full tool list.

    Args:
        all_tools: Every tool callable available in this REPL session.
        plan_ref: Optional mutable list of plan step dicts. Each dict has keys:
                  ``description`` (str), ``status`` (str), ``notes`` (str).
                  When None, update_plan returns a "no plan active" message.

    Returns:
        [list_tools, update_plan]
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

    return [list_tools, update_plan]
