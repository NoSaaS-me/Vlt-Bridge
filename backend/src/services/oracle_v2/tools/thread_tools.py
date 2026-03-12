"""Thread tools for the Oracle CodeAct agent — access vlt development threads."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Callable

logger = logging.getLogger(__name__)

_THREAD_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_AUTHOR_RE = re.compile(r"^[a-zA-Z0-9_:-]+$")
_MAX_CONTENT_BYTES = 10240  # 10 KB


def make_thread_tools(user_id: str, project_id: str) -> list[Callable]:
    """Build thread tool callables bound to user_id and project_id.

    Args:
        user_id: User ID used to scope all thread operations.
        project_id: Project ID used when creating new threads.

    Returns:
        [read_thread, search_threads, list_threads, thread_create, thread_push]
    """
    from ...thread_service import get_thread_service
    from ....models.thread import ThreadEntry

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

    def thread_create(name: str, goal: str = "") -> str:
        """Create a new vlt development thread to track work or log findings.

        Args:
            name: Thread slug used as both the ID and display name.
                  Only [a-zA-Z0-9_-] allowed, max 80 characters.
            goal: Optional initial goal entry stored as the first entry (authored as
                  'oracle:init'). Pass an empty string to skip.

        Returns:
            Confirmation string with thread details, or an error string.
        """
        try:
            # Validate name
            if not name:
                return "Error: thread name must not be empty."
            if len(name) > 80:
                return f"Error: thread name too long ({len(name)} chars, max 80)."
            if not _THREAD_NAME_RE.match(name):
                return (
                    f"Error: thread name {name!r} contains invalid characters. "
                    "Only [a-zA-Z0-9_-] are allowed."
                )

            svc.create_or_update_thread(
                user_id=user_id,
                thread_id=name,
                project_id=project_id,
                name=name,
                status="active",
            )

            effective_goal = goal.strip() if goal else ""
            if effective_goal:
                svc.add_entries(
                    user_id=user_id,
                    thread_id=name,
                    entries=[ThreadEntry(
                        entry_id=str(uuid.uuid4()),
                        sequence_id=0,
                        content=effective_goal,
                        author="oracle:init",
                        timestamp=datetime.utcnow(),
                    )],
                )

            return (
                f"Created thread: {name}\n"
                f"  Project: {project_id}\n"
                f"  Goal: {effective_goal or '(no goal specified)'}"
            )
        except Exception as exc:
            logger.warning("thread_create failed for %r: %s", name, exc)
            return f"Error creating thread {name!r}: {exc}"

    def thread_push(thread_name: str, content: str, author: str = "oracle") -> str:
        """Push a new entry to an existing vlt thread.

        Args:
            thread_name: The thread ID / slug to push to.
            content: Entry content (max 10 KB).
            author: Entry author label. Only [a-zA-Z0-9_:-] allowed, max 40 chars.
                    Defaults to 'oracle'.

        Returns:
            Confirmation string, or an error string if the operation fails.
        """
        try:
            # Validate content
            if not content:
                return "Error: content must not be empty."
            if len(content.encode("utf-8")) > _MAX_CONTENT_BYTES:
                return (
                    f"Error: content too large ({len(content.encode('utf-8'))} bytes, "
                    f"max {_MAX_CONTENT_BYTES})."
                )

            # Validate / sanitize author
            effective_author = (author or "oracle").strip()
            if not effective_author:
                effective_author = "oracle"
            if len(effective_author) > 40:
                return f"Error: author too long ({len(effective_author)} chars, max 40)."
            if not _AUTHOR_RE.match(effective_author):
                return (
                    f"Error: author {effective_author!r} contains invalid characters. "
                    "Only [a-zA-Z0-9_:-] are allowed."
                )

            # Resolve next sequence_id
            thread = svc.get_thread(
                user_id, thread_name, include_entries=True, entries_limit=1
            )
            if thread is None:
                return (
                    f"Error: thread {thread_name!r} not found. "
                    "Use thread_create() to create it first."
                )

            if thread.entries:
                next_seq = thread.entries[0].sequence_id + 1
            else:
                next_seq = 0

            svc.add_entries(
                user_id=user_id,
                thread_id=thread_name,
                entries=[ThreadEntry(
                    entry_id=str(uuid.uuid4()),
                    sequence_id=next_seq,
                    content=content,
                    author=effective_author,
                    timestamp=datetime.utcnow(),
                )],
            )

            preview = content[:80]
            ellipsis = "..." if len(content) > 80 else ""
            return (
                f"Pushed to thread: {thread_name}\n"
                f"  Entry #: {next_seq}\n"
                f"  Author: {effective_author}\n"
                f'  Content: "{preview}{ellipsis}"'
            )
        except Exception as exc:
            logger.warning("thread_push failed for %r: %s", thread_name, exc)
            return f"Error pushing to thread {thread_name!r}: {exc}"

    return [read_thread, search_threads, list_threads, thread_create, thread_push]
