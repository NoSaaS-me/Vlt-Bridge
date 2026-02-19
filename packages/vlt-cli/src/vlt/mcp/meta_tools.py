"""Meta tools for the vlt MCP server.

Provides health/status and project auto-detection capabilities.

Tools registered:
    vlt_status          — health check: projects, threads, daemon, backend, oracle
    vlt_project_detect  — detect project context from directory tree
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def register_meta_tools(mcp) -> None:
    """Register meta tools onto a FastMCP server instance."""

    @mcp.tool()
    def vlt_status() -> dict:
        """Return vlt system health and a summary of all projects and threads.

        Checks: database connectivity, project/thread counts, daemon status,
        backend configuration, and oracle toggle state.

        Returns:
            {status, projects, daemon_running, backend_configured, oracle_enabled, db_path}
        """
        from vlt.mcp import _ok, _err
        from vlt.config import get_settings
        from vlt.core.service import SqliteVaultService, VaultError
        from vlt.core.models import Project, Thread, Node
        from sqlalchemy import select, func
        import httpx

        try:
            settings = get_settings()
            svc = SqliteVaultService()

            # Projects with thread and node counts (two queries, no N+1)
            projects = svc.list_projects()

            thread_counts_rows = svc.db.execute(
                select(Thread.project_id, func.count(Thread.id).label("cnt"))
                .group_by(Thread.project_id)
            ).all()
            thread_counts = {r.project_id: r.cnt for r in thread_counts_rows}

            node_counts_rows = svc.db.execute(
                select(Thread.project_id, func.count(Node.id).label("cnt"))
                .join(Node, Thread.id == Node.thread_id)
                .group_by(Thread.project_id)
            ).all()
            node_counts = {r.project_id: r.cnt for r in node_counts_rows}

            project_summaries = [
                {
                    "id": p.id,
                    "name": p.name,
                    "thread_count": thread_counts.get(p.id, 0),
                    "node_count": node_counts.get(p.id, 0),
                }
                for p in projects
            ]

            # Daemon check (sync HTTP, short timeout)
            daemon_running = False
            try:
                r = httpx.get(f"{settings.daemon_url}/health", timeout=1.0)
                daemon_running = r.status_code == 200
            except Exception:
                pass

            return _ok(
                projects=project_summaries,
                daemon_running=daemon_running,
                backend_configured=settings.is_server_configured,
                oracle_enabled=settings.oracle_enabled,
                db_path=str(settings.get_db_path()),
            )
        except VaultError as e:
            return _err("DB_ERROR", str(e))
        except Exception as e:
            logger.exception("vlt_status failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_project_detect(
        path: Optional[str] = None,
    ) -> dict:
        """Detect vlt project context from a directory path.

        Walks up the directory tree from ``path`` (defaults to cwd) looking
        for a ``vlt.toml`` file. If found, reads the project_id and checks
        whether a CodeRAG index exists for that project.

        Args:
            path: Starting directory (defaults to current working directory).

        Returns:
            {status, found, project_id, project_name, project_path, has_coderag_index}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService, VaultError

        try:
            start = Path(path or os.getcwd()).resolve()

            # Walk up looking for vlt.toml
            config_path: Optional[Path] = None
            current = start
            while True:
                candidate = current / "vlt.toml"
                if candidate.exists():
                    config_path = candidate
                    break
                parent = current.parent
                if parent == current:
                    break
                current = parent

            if config_path is None:
                return _ok(
                    found=False,
                    project_id=None,
                    project_name=None,
                    project_path=None,
                    has_coderag_index=False,
                )

            # Parse vlt.toml
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            project_section = config.get("project", {})
            project_id = project_section.get("id") or project_section.get("name")
            project_name = project_section.get("name") or project_id

            if not project_id:
                return _ok(
                    found=True,
                    project_id=None,
                    project_name=None,
                    project_path=str(config_path.parent),
                    has_coderag_index=False,
                )

            # Check for CodeRAG index
            has_index = False
            try:
                svc = SqliteVaultService()
                has_index = svc.has_coderag_index(project_id)
            except Exception:
                pass

            return _ok(
                found=True,
                project_id=project_id,
                project_name=project_name,
                project_path=str(config_path.parent),
                has_coderag_index=has_index,
            )
        except Exception as e:
            logger.exception("vlt_project_detect failed")
            return _err("INTERNAL_ERROR", str(e))
