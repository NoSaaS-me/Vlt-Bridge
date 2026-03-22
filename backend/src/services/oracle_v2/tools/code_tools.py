"""Oracle V2 code tools — wrapped callables for the CodeAct REPL.

Each function is injected into the REPL namespace so the LLM can call them
as plain Python (e.g. ``results = search_code("auth middleware")``).

Functions are bound to project_id via closure. All return plain strings so
the LLM can process them in REPL code.

Available tools (returned by make_code_tools):
  search_code      — hybrid semantic+BM25 search over indexed code
  read_file        — read file contents with line-number bounds
  get_repo_map     — structural overview / repository map
  symbol_lookup    — precise symbol definition lookup by name
  find_callers     — find functions that call a given function
  find_callees     — find functions called by a given function
  class_hierarchy  — get class inheritance hierarchy (parents + children)
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _find_vlt_command() -> str:
    """Locate the vlt binary on PATH."""
    import shutil
    return shutil.which("vlt") or "vlt"


def _run_vlt_coderag(
    args: list[str],
    timeout: int = 60,
    env_override: Optional[dict] = None,
) -> dict:
    """Run a vlt coderag sub-command and return parsed JSON result.

    Falls back to returning an error dict on failure.
    """
    import os
    cmd = [_find_vlt_command()] + args
    env = {**os.environ}
    if env_override:
        env.update(env_override)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        stdout = result.stdout.strip()
        if not stdout:
            return {
                "error": True,
                "message": f"vlt returned empty output (rc={result.returncode}). stderr: {result.stderr[:300]}",
            }
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            # Return raw text wrapped in a dict
            return {"map_text": stdout, "raw": True}
    except subprocess.TimeoutExpired:
        return {"error": True, "message": f"vlt command timed out after {timeout}s"}
    except FileNotFoundError:
        return {"error": True, "message": "vlt command not found. Is vlt-cli installed?"}
    except Exception as exc:
        return {"error": True, "message": f"vlt command failed: {exc}"}


def _format_search_results(raw: dict | list, query: str) -> str:
    """Format code search results into a human-readable string for the LLM."""
    if isinstance(raw, dict) and raw.get("error"):
        return f"[code search error] {raw.get('message', 'Unknown error')}"

    # Results may be a list directly, or wrapped in {"results": [...]}
    results: list = []
    if isinstance(raw, list):
        results = raw
    elif isinstance(raw, dict):
        results = raw.get("results", [])

    if not results:
        return f"[no results found for query: {query!r}]"

    lines = [f"Code search results for {query!r} ({len(results)} matches):\n"]
    for i, item in enumerate(results, start=1):
        file_path = item.get("file_path") or item.get("path") or "unknown"
        start_line = item.get("start_line") or item.get("line") or ""
        snippet = item.get("content") or item.get("code") or item.get("snippet") or ""
        score = item.get("score")
        score_str = f" (score={score:.3f})" if score else ""
        header = f"[{i}] {file_path}"
        if start_line:
            header += f":{start_line}"
        header += score_str
        lines.append(header)
        if snippet:
            # Limit snippet to keep context tight
            snippet_preview = snippet[:500].rstrip()
            lines.append(snippet_preview)
        lines.append("")

    return "\n".join(lines)


_VALID_SYMBOL_KINDS = frozenset({"function", "class", "method", "variable"})


def make_code_tools(
    project_id: str,
    openrouter_api_key: Optional[str] = None,
) -> list[Callable]:
    """Factory: returns [search_code, read_file, get_repo_map, symbol_lookup, find_callers, find_callees, class_hierarchy] bound to project_id.

    Args:
        project_id: The coderag project ID to scope all tool calls.
        openrouter_api_key: Optional API key passed to vlt for vector search.

    Returns:
        List of plain callables ready to inject into the CodeAct REPL namespace.
    """
    _project_id = project_id
    _api_key = openrouter_api_key

    def search_code(query: str, limit: int = 10) -> str:
        """Search the codebase for relevant code using semantic + BM25 hybrid search.

        Args:
            query: Natural language search query or code pattern.
            limit: Maximum number of results to return (default 10).

        Returns:
            Formatted string with file paths, line numbers, and code snippets.
        """
        args = ["coderag", "search", query, "--limit", str(limit), "--project", _project_id]
        env = {}
        if _api_key:
            env["VLT_OPENROUTER_API_KEY"] = _api_key

        raw = _run_vlt_coderag(args, timeout=60, env_override=env if env else None)
        return _format_search_results(raw, query)

    def read_file(path: str, start_line: int = 1, end_line: int = 100) -> str:
        """Read a file from the codebase.

        Args:
            path: File path relative to project root.
            start_line: First line to read (1-indexed, default 1).
            end_line: Last line to read (default 100).

        Returns:
            File contents as string with line numbers.
        """
        # Resolve project root from vlt coderag status
        try:
            status_raw = _run_vlt_coderag(
                ["coderag", "status", "--project", _project_id, "--json"],
                timeout=10,
            )
            project_root_str = status_raw.get("project_path") or status_raw.get("path")
        except Exception:
            project_root_str = None

        # Attempt to read the file directly
        # If project_root resolves, use it; otherwise try relative to cwd
        import os
        candidate_paths: list[Path] = []
        if project_root_str:
            candidate_paths.append(Path(project_root_str) / path)
        candidate_paths.append(Path(path))
        candidate_paths.append(Path(os.getcwd()) / path)

        file_path: Optional[Path] = None
        for candidate in candidate_paths:
            if candidate.exists() and candidate.is_file():
                file_path = candidate
                break

        if file_path is None:
            return f"[read_file error] File not found: {path!r} (tried {[str(p) for p in candidate_paths]})"

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"[read_file error] Cannot read {path!r}: {exc}"

        all_lines = content.splitlines()
        total = len(all_lines)

        # Clamp indices (1-indexed → 0-indexed)
        lo = max(0, start_line - 1)
        hi = min(total, end_line)
        selected = all_lines[lo:hi]

        # Format with line numbers
        result_lines = [f"{file_path} (lines {lo + 1}-{lo + len(selected)} of {total}):"]
        for idx, line in enumerate(selected, start=lo + 1):
            result_lines.append(f"{idx:6d}  {line}")

        if hi < total:
            result_lines.append(f"... ({total - hi} more lines, use end_line to see more)")

        return "\n".join(result_lines)

    def get_repo_map(scope: str = "") -> str:
        """Get a structural overview of the codebase (repository map).

        Args:
            scope: Optional directory path to scope the map (e.g. 'src/api/').

        Returns:
            Formatted repository map with file tree and key symbols.
        """
        args = ["coderag", "map", "--project", _project_id]

        raw = _run_vlt_coderag(args, timeout=90)

        if isinstance(raw, dict) and raw.get("error"):
            return f"[get_repo_map error] {raw.get('message', 'Unknown error')}"

        # map_text key is set by _run_vlt_coderag for raw text output
        if isinstance(raw, dict) and "map_text" in raw:
            map_text = raw["map_text"]
            if scope:
                # Filter lines that mention the scope prefix
                filtered = [
                    line for line in map_text.splitlines()
                    if scope in line or not line.strip()
                ]
                if filtered:
                    return "\n".join(filtered)
            return map_text

        # Structured response
        if isinstance(raw, dict):
            return json.dumps(raw, indent=2)

        return str(raw)

    def symbol_lookup(name: str, kind: str = "") -> str:
        """Look up a symbol definition by exact name (function, class, method, variable).

        More precise than search_code — filters for exact name matches in symbol definitions.
        Use search_code for fuzzy/conceptual queries, symbol_lookup for known names.

        Args:
            name: Symbol name to look up (e.g. 'OracleV2Wrapper', 'make_sandbox_eval').
            kind: Optional filter: 'function', 'class', 'method', 'variable'. Empty matches all.

        Returns:
            Symbol definitions with file path, line number, kind, and signature.
        """
        if kind and kind not in _VALID_SYMBOL_KINDS:
            valid = ", ".join(sorted(_VALID_SYMBOL_KINDS))
            return f"[symbol_lookup error] Invalid kind {kind!r}. Valid values: {valid}"

        args = ["coderag", "search", name, "--limit", "10", "--project", _project_id, "--json"]
        env = {}
        if _api_key:
            env["VLT_OPENROUTER_API_KEY"] = _api_key

        raw = _run_vlt_coderag(args, timeout=60, env_override=env if env else None)

        if isinstance(raw, dict) and raw.get("error"):
            return f"[symbol_lookup error] {raw.get('message', 'Unknown error')}"

        # Normalise results to list
        results: list = []
        if isinstance(raw, list):
            results = raw
        elif isinstance(raw, dict):
            results = raw.get("results", [])

        if not results:
            return (
                f"No symbol {name!r} found. "
                f"Try search_code({name!r}) for broader matching."
            )

        kind_header = f" (kind: {kind})" if kind else ""
        lines = [f"Symbol lookup for {name!r}{kind_header}:\n"]
        displayed = 0
        for i, item in enumerate(results, start=1):
            item_kind = item.get("kind") or item.get("type") or ""
            # Apply kind filter when specified
            if kind and item_kind and item_kind.lower() != kind.lower():
                continue

            file_path = item.get("file_path") or item.get("path") or "unknown"
            start_line = item.get("start_line") or item.get("line") or ""
            snippet = item.get("content") or item.get("code") or item.get("snippet") or ""

            loc = f"{file_path}"
            if start_line:
                loc += f":{start_line}"
            lines.append(f"[{displayed + 1}] {loc}")

            if item_kind:
                lines.append(f"    Kind: {item_kind}")

            # Show first meaningful line of snippet as the signature
            if snippet:
                sig_line = next(
                    (ln.strip() for ln in snippet.splitlines() if ln.strip()),
                    "",
                )
                if sig_line:
                    lines.append(f"    {sig_line}")

            lines.append("")
            displayed += 1

        if displayed == 0:
            return (
                f"No symbol {name!r} found with kind={kind!r}. "
                f"Try symbol_lookup({name!r}) without a kind filter."
            )

        return "\n".join(lines)

    def find_callers(function_name: str, transitive: bool = False) -> str:
        """Find what functions call a given function.

        Args:
            function_name: Name of the target function.
            transitive: If True, find all callers recursively (full call chain). Default False.

        Returns:
            Formatted list of caller functions with file paths and line numbers.
        """
        args = ["coderag", "callers", function_name, "--project", _project_id, "--json"]
        if transitive:
            args.append("--transitive")
        raw = _run_vlt_coderag(args, timeout=60)

        if isinstance(raw, dict) and raw.get("error"):
            return f"[find_callers error] {raw.get('message', 'Unknown error')}"

        results = raw.get("results", []) if isinstance(raw, dict) else []
        if not results:
            return f"No callers found for {function_name!r}."

        lines = [f"Callers of {function_name!r} ({len(results)} found):\n"]
        for i, item in enumerate(results, 1):
            name = item.get("caller_name") or item.get("name") or "unknown"
            fpath = item.get("caller_file_path") or item.get("file_path") or item.get("path") or ""
            lineno = item.get("caller_line_number") or item.get("line_number") or ""
            loc = f"{fpath}:{lineno}" if lineno else fpath
            lines.append(f"  [{i}] {name}  ({loc})")
        return "\n".join(lines)

    def find_callees(function_name: str, transitive: bool = False) -> str:
        """Find what a function calls (its dependencies).

        Args:
            function_name: Name of the source function.
            transitive: If True, find all callees recursively (full dependency chain). Default False.

        Returns:
            Formatted list of called functions with file paths and line numbers.
        """
        args = ["coderag", "callees", function_name, "--project", _project_id, "--json"]
        if transitive:
            args.append("--transitive")
        raw = _run_vlt_coderag(args, timeout=60)

        if isinstance(raw, dict) and raw.get("error"):
            return f"[find_callees error] {raw.get('message', 'Unknown error')}"

        results = raw.get("results", []) if isinstance(raw, dict) else []
        if not results:
            return f"No callees found for {function_name!r}."

        lines = [f"Functions called by {function_name!r} ({len(results)} found):\n"]
        for i, item in enumerate(results, 1):
            name = item.get("callee_name") or item.get("name") or "unknown"
            fpath = item.get("callee_file_path") or item.get("file_path") or item.get("path") or ""
            lineno = item.get("callee_line_number") or item.get("line_number") or ""
            loc = f"{fpath}:{lineno}" if lineno else fpath
            lines.append(f"  [{i}] {name}  ({loc})")
        return "\n".join(lines)

    def class_hierarchy(class_name: str) -> str:
        """Get class inheritance hierarchy — parents and children.

        Args:
            class_name: Name of the class to inspect.

        Returns:
            Formatted hierarchy showing parents (superclasses) and children (subclasses).
        """
        args = ["coderag", "hierarchy", class_name, "--project", _project_id, "--json"]
        raw = _run_vlt_coderag(args, timeout=60)

        if isinstance(raw, dict) and raw.get("error"):
            return f"[class_hierarchy error] {raw.get('message', 'Unknown error')}"

        parents = raw.get("parents", []) if isinstance(raw, dict) else []
        children = raw.get("children", []) if isinstance(raw, dict) else []

        lines = [f"Class hierarchy for {class_name!r}:\n"]

        if parents:
            lines.append("  Parents (superclasses):")
            for p in parents:
                name = p.get("name") if isinstance(p, dict) else str(p)
                lines.append(f"    - {name}")
        else:
            lines.append("  Parents: (none — base class)")

        lines.append(f"\n  -> {class_name}\n")

        if children:
            lines.append("  Children (subclasses):")
            for c in children:
                name = c.get("name") if isinstance(c, dict) else str(c)
                lines.append(f"    - {name}")
        else:
            lines.append("  Children: (none)")

        return "\n".join(lines)

    return [search_code, read_file, get_repo_map, symbol_lookup, find_callers, find_callees, class_hierarchy]
