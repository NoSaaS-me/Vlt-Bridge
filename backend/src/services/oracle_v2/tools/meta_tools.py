"""Meta tools for the Oracle CodeAct agent — self-awareness and plan management."""

from __future__ import annotations

from typing import Callable


def make_meta_tools(all_tools: list[Callable]) -> list[Callable]:
    """Build meta tool callables bound to the full tool list.

    Args:
        all_tools: Every tool callable available in this REPL session.

    Returns:
        [list_tools]

    Note:
        update_plan and delegate_librarian will be added once graph.py
        wiring is complete (T034, handled by orchestrator).
    """

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

    return [list_tools]
