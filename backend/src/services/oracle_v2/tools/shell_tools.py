"""Shell tools for the Oracle CodeAct agent — allowlisted read-only shell commands."""

from __future__ import annotations

import shutil
from typing import Callable

from ..sandbox import run_shell


def git_log(path: str = "", n: int = 10, oneline: bool = True) -> str:
    """Show git commit history for a file or the whole repo.

    Args:
        path: File path to scope the log (empty = whole repo).
        n: Number of commits to show (default 10).
        oneline: Show compact one-line format (default True).

    Returns:
        Git log output as string.
    """
    parts = ["git", "log", f"-{n}"]
    if oneline:
        parts.append("--oneline")
    if path:
        parts.extend(["--", path])
    return run_shell(" ".join(parts))


def git_diff(ref1: str = "HEAD~1", ref2: str = "HEAD", path: str = "") -> str:
    """Show diff between two git refs.

    Args:
        ref1: First ref (default HEAD~1).
        ref2: Second ref (default HEAD).
        path: Optional file path to scope the diff.

    Returns:
        Git diff output as string.
    """
    parts = ["git", "diff", ref1, ref2]
    if path:
        parts.extend(["--", path])
    return run_shell(" ".join(parts))


def git_blame(path: str, start_line: int = 1, end_line: int = 50) -> str:
    """Show git blame for a file range (who last modified each line).

    Args:
        path: File path to blame.
        start_line: First line (1-indexed).
        end_line: Last line.

    Returns:
        Git blame output as string.
    """
    return run_shell(f"git blame -L {start_line},{end_line} -- {path}")


def git_status() -> str:
    """Show current git working tree status.

    Returns:
        Git status output.
    """
    return run_shell("git status")


def find_files(pattern: str, directory: str = ".") -> str:
    """Find files matching a pattern.

    Args:
        pattern: Filename pattern (e.g. '*.py', 'oracle*.py').
        directory: Directory to search (default current dir).

    Returns:
        Newline-separated list of matching file paths.
    """
    return run_shell(f"find {directory} -name {pattern}")


def grep_files(pattern: str, path: str = ".", flags: str = "") -> str:
    """Search file contents for a pattern using rg or grep.

    Args:
        pattern: Regex pattern to search for.
        path: File or directory to search.
        flags: Optional flags (e.g. '-i' for case-insensitive, '-l' for filenames only).

    Returns:
        Matching lines with file:line format.
    """
    # Prefer ripgrep if available (faster and in allowlist)
    if shutil.which("rg"):
        flag_str = f" {flags}" if flags else ""
        return run_shell(f"rg{flag_str} {pattern} {path}")
    flag_str = f" {flags}" if flags else ""
    return run_shell(f"grep -r{flag_str} {pattern} {path}")


def make_shell_tools() -> list[Callable]:
    """Return shell tool callables for injection into the CodeAct REPL.

    Returns:
        [git_log, git_diff, git_blame, git_status, find_files, grep_files]
    """
    return [git_log, git_diff, git_blame, git_status, find_files, grep_files]
