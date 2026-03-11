"""Thread tools for the Oracle CodeAct agent — access vlt development threads."""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def make_thread_tools(user_id: str) -> list[Callable]:
    """Build thread tool callables bound to user_id.

    Args:
        user_id: User ID used to scope all thread operations.

    Returns:
        [read_thread, search_threads, list_threads]
    """
    from backend.src.services.thread_service import get_thread_service

    svc = get_thread_service()

    def read_thread(thread_name: str) -> str:
        """Read a vlt development thread by name.

        Args:
            thread_name: Thread name or ID.

        Returns:
            Thread entries as formatted text.
        """
        try:
            thread = svc.get_thread(user_id, thread_name, include_entries=True)
            if thread is None:
                return f"Thread {thread_name!r} not found."

            lines = [f"Thread: {thread.thread_id} ({thread.name})\n"]
            if not thread.entries:
                lines.append("(no entries)")
            else:
                for entry in thread.entries:
                    ts = entry.timestamp.strftime("%Y-%m-%d %H:%M") if entry.timestamp else "unknown"
                    lines.append(f"[{ts}] {entry.author}: {entry.content}")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("read_thread failed for %r: %s", thread_name, exc)
            return f"Error reading thread {thread_name!r}: {exc}"

    def search_threads(query: str) -> str:
        """Search vlt threads for relevant entries.

        Args:
            query: Search query.

        Returns:
            Matching thread entries with context.
        """
        try:
            response = svc.search_threads(user_id=user_id, query=query, limit=10)
            if not response.results:
                return f"No thread entries found for query: {query!r}"

            lines = [f"Search results for {query!r}:\n"]
            for result in response.results:
                ts = result.timestamp.strftime("%Y-%m-%d %H:%M") if result.timestamp else "unknown"
                lines.append(
                    f"Thread {result.thread_id} [{ts}] {result.author}:\n  {result.content}\n"
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("search_threads failed for %r: %s", query, exc)
            return f"Error searching threads: {exc}"

    def list_threads(limit: int = 20) -> str:
        """List available vlt development threads.

        Args:
            limit: Max threads to list (default 20).

        Returns:
            Formatted list of thread names and summaries.
        """
        try:
            response = svc.list_threads(user_id=user_id, limit=limit)
            if not response.threads:
                return "No threads found."

            lines = [f"Threads ({response.total} total):\n"]
            for t in response.threads:
                updated = t.updated_at.strftime("%Y-%m-%d") if t.updated_at else "unknown"
                lines.append(f"  {t.thread_id}  [{updated}]  {t.name}  (project: {t.project_id})")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("list_threads failed: %s", exc)
            return f"Error listing threads: {exc}"

    return [read_thread, search_threads, list_threads]
