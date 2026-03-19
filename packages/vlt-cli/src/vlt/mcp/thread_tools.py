"""Thread memory tools for the vlt MCP server.

Exposes vlt thread operations via MCP so AI agents can create, update, read,
and search threads without CLI subprocess overhead.

Tools registered:
    vlt_thread_create  — create a new thread (auto-creates project)
    vlt_thread_push    — append a node to an existing thread
    vlt_thread_read    — read thread state (summary + recent nodes)
    vlt_thread_seek    — semantic or keyword search across nodes
    vlt_thread_list    — list threads, optionally filtered by project
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def register_thread_tools(mcp) -> None:
    """Register all thread tools onto a FastMCP server instance."""

    @mcp.tool()
    def vlt_thread_create(
        project_id: str,
        name: str,
        initial_thought: str,
        author: str = "agent",
    ) -> dict:
        """Create a new vlt thread in a project.

        The thread ID is derived from ``name`` (lowercased, spaces → hyphens).
        Auto-creates the project if it does not exist — always safe to call.

        TIP: Don't know the project_id? Call vlt_project_detect() first —
        it reads vlt.toml from your working directory and returns the slug.

        Args:
            project_id: Project identifier slug (e.g. "my-project"). Created if new.
                        Use vlt_project_detect() to find this from cwd.
            name: Thread name. Becomes the thread_id slug (e.g. "Auth Design" → "auth-design").
            initial_thought: First entry written into the new thread.
            author: Author label for the initial node (default: "agent").

        Returns:
            {status, thread_id, project_id, node_id, sequence_id}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService, VaultError
        from vlt.core.models import Node
        from sqlalchemy import select

        try:
            svc = SqliteVaultService()
            svc.ensure_project_exists(project_id)
            thread = svc.create_thread(project_id, name, initial_thought, author=author)

            # Fetch the initial node (sequence_id == 0) for the return payload
            first_node = svc.db.scalars(
                select(Node)
                .where(Node.thread_id == thread.id)
                .order_by(Node.sequence_id)
                .limit(1)
            ).first()

            return _ok(
                thread_id=thread.id,
                project_id=thread.project_id,
                node_id=first_node.id if first_node else None,
                sequence_id=first_node.sequence_id if first_node else 0,
            )
        except VaultError as e:
            return _err("THREAD_CREATE_FAILED", str(e))
        except Exception as e:
            logger.exception("vlt_thread_create failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_thread_push(
        thread_id: str,
        content: str,
        author: str = "agent",
    ) -> dict:
        """Append a thought to an existing vlt thread.

        Args:
            thread_id: Thread identifier slug (e.g. "auth-design").
            content: The thought or message to append.
            author: Author label (default: "agent").

        Returns:
            {status, thread_id, node_id, sequence_id, duration_ms}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService, VaultError
        from vlt.core.models import Thread

        try:
            svc = SqliteVaultService()

            thread = svc.db.get(Thread, thread_id)
            if not thread:
                return _err("THREAD_NOT_FOUND", f"Thread '{thread_id}' not found.")

            t0 = time.monotonic()
            node = svc.add_thought(thread_id, content, author=author)
            duration_ms = round((time.monotonic() - t0) * 1000, 1)

            return _ok(
                thread_id=thread_id,
                node_id=node.id,
                sequence_id=node.sequence_id,
                duration_ms=duration_ms,
            )
        except VaultError as e:
            return _err("THREAD_NOT_FOUND", str(e))
        except Exception as e:
            logger.exception("vlt_thread_push failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_thread_read(
        thread_id: str,
        limit: int = 10,
    ) -> dict:
        """Read the current state of a vlt thread.

        Returns a summary (LLM-generated if available, otherwise guidance text)
        plus the N most recent nodes.

        Args:
            thread_id: Thread identifier slug.
            limit: Number of recent nodes to include (0 = all). Default: 10.

        Returns:
            {status, thread_id, project_id, summary, recent_nodes, node_count}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService, VaultError
        from vlt.core.models import Node
        from sqlalchemy import select, func

        try:
            svc = SqliteVaultService()
            state = svc.get_thread_state(thread_id, limit=limit)

            node_count = svc.db.scalar(
                select(func.count())
                .select_from(Node)
                .where(Node.thread_id == state.thread_id)
            ) or 0

            # Replace CLI-only librarian guidance with MCP-appropriate message
            summary = state.summary
            if summary and "vlt librarian run" in summary:
                summary = (
                    "No summary generated yet (requires librarian background process). "
                    "Read recent_nodes directly for full thread content."
                )

            return _ok(
                thread_id=state.thread_id,
                project_id=state.project_id,
                summary=summary,
                recent_nodes=[
                    {
                        "id": n.id,
                        "sequence_id": n.sequence_id,
                        "author": n.author,
                        "timestamp": n.timestamp.isoformat(),
                        "content": n.content,
                    }
                    for n in state.recent_nodes
                ],
                node_count=node_count,
            )
        except VaultError as e:
            return _err("THREAD_NOT_FOUND", str(e))
        except Exception as e:
            logger.exception("vlt_thread_read failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_thread_seek(
        query: str,
        project_id: Optional[str] = None,
        limit: int = 10,
    ) -> dict:
        """Search across thread nodes using semantic or keyword search.

        Tries vector similarity search first. Falls back automatically to
        SQLite LIKE keyword search when: (a) no embedding API key is configured,
        (b) semantic search returns empty results (e.g. freshly pushed nodes
        not yet embedded — usually takes <30s for background embedding).

        The ``search_mode`` field in the response tells you which strategy
        was used, so you know if results may be incomplete.

        TIP: Pass project_id to scope results to the current project.
        Use vlt_project_detect() to get it from cwd.

        Args:
            query: Natural language or keyword search query.
            project_id: Scope search to a specific project (recommended).
                        Use vlt_project_detect() to find this from cwd.
            limit: Maximum number of results. Default: 10.

        Returns:
            {status, results: [{thread_id, node_id, content, author, timestamp, score}],
             search_mode: "semantic"|"keyword", total}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService
        from vlt.core.models import Node, Thread
        from sqlalchemy import select

        svc = SqliteVaultService()
        search_mode = "semantic"
        results = []

        try:
            if project_id:
                raw = svc.seek_threads(project_id, query, limit=limit)
                results = [
                    {
                        "thread_id": r["thread_id"],
                        "node_id": r["node_id"],
                        "content": r["content"],
                        "author": r.get("author"),
                        "timestamp": r.get("timestamp"),
                        "score": r.get("score"),
                    }
                    for r in raw
                ]
            else:
                raw_results = svc.search(query, project_id=None)
                results = [
                    {
                        "thread_id": r.thread_id,
                        "node_id": r.node_id,
                        "content": r.content,
                        "score": r.score,
                    }
                    for r in raw_results[:limit]
                ]
            # Semantic returned empty — embeddings may not be computed yet.
            # Fall through to keyword search so freshly-pushed nodes are findable.
            if not results:
                raise ValueError("semantic search returned no results")
        except Exception:
            # Fallback: keyword LIKE search (no embedding required)
            search_mode = "keyword"
            like_q = f"%{query}%"
            stmt = (
                select(Node)
                .where(Node.content.like(like_q))
                .limit(limit)
            )
            if project_id:
                stmt = stmt.join(Thread, Node.thread_id == Thread.id).where(
                    Thread.project_id == project_id
                )
            nodes = svc.db.scalars(stmt).all()
            results = [
                {
                    "thread_id": n.thread_id,
                    "node_id": n.id,
                    "content": n.content,
                    "author": n.author,
                    "timestamp": n.timestamp.isoformat(),
                    "score": None,
                }
                for n in nodes
            ]

        return _ok(
            results=results,
            search_mode=search_mode,
            total=len(results),
        )

    @mcp.tool()
    def vlt_thread_list(
        project_id: Optional[str] = None,
    ) -> dict:
        """List vlt threads, optionally filtered by project.

        WARNING: Omitting project_id returns threads from ALL projects, which
        can be very large. Always pass project_id to scope results.
        Use vlt_project_detect() to get the current project_id from cwd.

        Args:
            project_id: Filter to this project slug. STRONGLY RECOMMENDED.
                        Use vlt_project_detect() to find this from cwd.
                        Lists threads across ALL projects if omitted (can be 100+).

        Returns:
            {status, project_id, threads: [{id, project_id, status, created_at, node_count}]}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService, VaultError
        from vlt.core.models import Thread, Node
        from sqlalchemy import select, func

        try:
            svc = SqliteVaultService()

            if project_id:
                threads = svc.list_threads(project_id)
            else:
                threads = list(svc.db.scalars(select(Thread)).all())

            # Node counts per thread in one query
            count_rows = svc.db.execute(
                select(Node.thread_id, func.count(Node.id).label("cnt"))
                .group_by(Node.thread_id)
            ).all()
            node_counts = {row.thread_id: row.cnt for row in count_rows}

            return _ok(
                project_id=project_id or "all",
                threads=[
                    {
                        "id": t.id,
                        "project_id": t.project_id,
                        "status": t.status,
                        "created_at": t.created_at.isoformat(),
                        "node_count": node_counts.get(t.id, 0),
                    }
                    for t in threads
                ],
            )
        except VaultError as e:
            return _err("PROJECT_NOT_FOUND", str(e))
        except Exception as e:
            logger.exception("vlt_thread_list failed")
            return _err("INTERNAL_ERROR", str(e))
