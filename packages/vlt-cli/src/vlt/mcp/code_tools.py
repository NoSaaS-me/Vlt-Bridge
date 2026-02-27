"""Code intelligence tools for the vlt MCP server.

Exposes CodeRAG functionality via MCP so AI agents can index, search,
and navigate codebases without CLI subprocess overhead.

Tools registered:
    vlt_code_init    — start code indexing for a project (daemon or background thread)
    vlt_code_search  — hybrid BM25 + vector search across indexed code
    vlt_code_map     — retrieve the cached repository structure map
    vlt_code_status  — check index status and job progress
    vlt_code_lookup  — find symbol definitions by name
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


def register_code_tools(mcp) -> None:
    """Register all code intelligence tools onto a FastMCP server instance."""

    @mcp.tool()
    def vlt_code_init(
        project_id: str,
        path: str,
        force: bool = False,
    ) -> dict:
        """Start code indexing for a project.

        Queues an indexing job. If the vlt daemon is running it will process
        the job in the background. Otherwise indexing runs in a detached thread
        so this call returns immediately.

        Call ``vlt_code_status`` to poll progress.

        Args:
            project_id: Project identifier slug.
            path: Absolute path to the codebase root directory to index.
            force: Re-index even if an index already exists. Default: False.

        Returns:
            {status, project_id, job_id, job_status, daemon_handling}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService, VaultError
        from vlt.core.models import CodeRAGIndexJob, JobStatus
        from vlt.db import engine
        from sqlalchemy.orm import Session
        import httpx

        try:
            svc = SqliteVaultService()
            svc.ensure_project_exists(project_id)

            # Check for an active job unless force=True
            if not force:
                active_job = svc.get_active_job_for_project(project_id)
                if active_job:
                    return _ok(
                        project_id=project_id,
                        job_id=active_job.id,
                        job_status=active_job.status.value,
                        progress_percent=active_job.progress_percent,
                        daemon_handling=True,
                        message="Indexing already in progress. Use force=True to restart.",
                    )

            target_path = Path(path).resolve()
            if not target_path.exists():
                return _err("INVALID_PATH", f"Path does not exist: {path}")

            # Create a pending job record
            job_id = str(uuid4())
            with Session(engine) as session:
                job = CodeRAGIndexJob(
                    id=job_id,
                    project_id=project_id,
                    status=JobStatus.PENDING,
                    target_path=str(target_path),
                    force=force,
                    priority=0,
                    files_total=0,
                    files_processed=0,
                    chunks_created=0,
                    progress_percent=0,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(job)
                session.commit()

            # Check if daemon is running (sync)
            from vlt.config import get_settings
            settings = get_settings()
            daemon_running = False
            try:
                r = httpx.get(f"{settings.daemon_url}/health", timeout=1.0)
                daemon_running = r.status_code == 200
            except Exception:
                pass

            if not daemon_running:
                # Run indexing in a detached background thread
                def _run():
                    try:
                        from vlt.core.coderag.indexer import CodeRAGIndexer
                        from vlt.core.models import JobStatus as JS
                        from vlt.db import engine as _eng
                        from sqlalchemy.orm import Session as _Session

                        indexer = CodeRAGIndexer(target_path, project_id)

                        # Mark job as running
                        with _Session(_eng) as s:
                            j = s.get(CodeRAGIndexJob, job_id)
                            if j:
                                j.status = JS.RUNNING
                                j.started_at = datetime.now(timezone.utc)
                                s.commit()

                        indexer.index_full(force=force)

                        # Mark completed
                        with _Session(_eng) as s:
                            j = s.get(CodeRAGIndexJob, job_id)
                            if j:
                                j.status = JS.COMPLETED
                                j.completed_at = datetime.now(timezone.utc)
                                j.progress_percent = 100
                                s.commit()
                    except Exception as exc:
                        logger.exception("Background indexing failed for %s", project_id)
                        try:
                            from vlt.core.models import JobStatus as JS
                            from vlt.db import engine as _eng
                            from sqlalchemy.orm import Session as _Session
                            with _Session(_eng) as s:
                                j = s.get(CodeRAGIndexJob, job_id)
                                if j:
                                    j.status = JS.FAILED
                                    j.error_message = str(exc)
                                    s.commit()
                        except Exception:
                            pass

                t = threading.Thread(target=_run, daemon=True)
                t.start()

            return _ok(
                project_id=project_id,
                job_id=job_id,
                job_status="queued",
                daemon_handling=daemon_running,
                message=(
                    "Job queued — daemon will process it."
                    if daemon_running
                    else "Indexing started in background thread (daemon not running)."
                ),
            )
        except VaultError as e:
            return _err("PROJECT_ERROR", str(e))
        except Exception as e:
            logger.exception("vlt_code_init failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_code_search(
        query: str,
        project_id: str,
        limit: int = 10,
        language: Optional[str] = None,
    ) -> dict:
        """Search indexed code using hybrid BM25 + keyword retrieval.

        Returns code chunks ranked by relevance, with file path and line numbers.

        Args:
            query: Natural language or keyword search query.
            project_id: Project to search in.
            limit: Maximum results to return. Default: 10.
            language: Filter results to this programming language (optional).

        Returns:
            {status, results: [{chunk_id, file_path, qualified_name, lineno, score, snippet}]}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService, VaultError
        from vlt.core.coderag.bm25 import search_bm25

        try:
            svc = SqliteVaultService()
            if not svc.has_coderag_index(project_id):
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No code index for project '{project_id}'. Call vlt_code_init first.",
                )

            raw = search_bm25(query, limit=limit * 2 if language else limit, project_id=project_id)

            if language:
                # Filter by language if the result has a language field
                raw = [r for r in raw if r.get("language", "").lower() == language.lower()]
                raw = raw[:limit]

            results = [
                {
                    "chunk_id": r.get("chunk_id"),
                    "file_path": r.get("file_path"),
                    "qualified_name": r.get("qualified_name"),
                    "lineno": r.get("lineno"),
                    "score": r.get("score"),
                    "snippet": (r.get("body") or "")[:300],
                }
                for r in raw
            ]

            return _ok(results=results, total=len(results))
        except VaultError as e:
            return _err("PROJECT_ERROR", str(e))
        except Exception as e:
            logger.exception("vlt_code_search failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_code_map(
        project_id: str,
        scope: Optional[str] = None,
    ) -> dict:
        """Retrieve the cached repository structure map for a project.

        Returns a compact textual overview of the codebase: files, classes,
        functions, and their relationships. Suitable for providing to an LLM
        as initial context.

        Args:
            project_id: Project identifier slug.
            scope: Limit map to files under this path prefix (optional).

        Returns:
            {status, map_text, token_count, files_included, symbols_included, symbols_total}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.coderag.store import CodeRAGStore
        from vlt.core.service import SqliteVaultService

        try:
            svc = SqliteVaultService()
            if not svc.has_coderag_index(project_id):
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No code index for project '{project_id}'. Call vlt_code_init first.",
                )

            with CodeRAGStore() as store:
                repo_map = store.get_repo_map(project_id, scope=scope)

            if not repo_map:
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No repo map cached for '{project_id}'. Re-run vlt_code_init to generate one.",
                )

            return _ok(
                map_text=repo_map.map_text,
                token_count=repo_map.token_count,
                files_included=repo_map.files_included,
                symbols_included=repo_map.symbols_included,
                symbols_total=repo_map.symbols_total,
            )
        except Exception as e:
            logger.exception("vlt_code_map failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_code_status(
        project_id: str,
    ) -> dict:
        """Check code index status and active job progress for a project.

        Args:
            project_id: Project identifier slug.

        Returns:
            {status, indexed, active_job: {job_id, status, progress_percent,
            files_processed, files_total, chunks_created, error_message} | null, stats}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService, VaultError
        from vlt.core.coderag.store import CodeRAGStore

        try:
            svc = SqliteVaultService()
            indexed = svc.has_coderag_index(project_id)
            active_job = svc.get_active_job_for_project(project_id)

            job_info = None
            if active_job:
                job_info = {
                    "job_id": active_job.id,
                    "status": active_job.status.value,
                    "progress_percent": active_job.progress_percent,
                    "files_processed": active_job.files_processed,
                    "files_total": active_job.files_total,
                    "chunks_created": active_job.chunks_created,
                    "error_message": active_job.error_message,
                }

            # Project stats (chunk/symbol/graph counts)
            stats = {}
            try:
                with CodeRAGStore() as store:
                    stats = store.get_project_stats(project_id)
            except Exception:
                pass

            return _ok(
                project_id=project_id,
                indexed=indexed,
                active_job=job_info,
                stats=stats,
            )
        except VaultError as e:
            return _err("PROJECT_ERROR", str(e))
        except Exception as e:
            logger.exception("vlt_code_status failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_code_lookup(
        symbol: str,
        project_id: str,
        kind: Optional[str] = None,
    ) -> dict:
        """Look up symbol definitions by name in the code index.

        Finds where a function, class, method, or variable is defined.

        Args:
            symbol: Symbol name to look up (e.g. "authenticate", "UserService").
            project_id: Project to search in.
            kind: Filter by symbol kind: "function", "class", "method", "variable", etc.

        Returns:
            {status, found, definitions: [{name, file_path, lineno, kind, scope, signature, language}]}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.coderag.store import CodeRAGStore
        from vlt.core.service import SqliteVaultService

        try:
            svc = SqliteVaultService()
            if not svc.has_coderag_index(project_id):
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No code index for project '{project_id}'. Call vlt_code_init first.",
                )

            with CodeRAGStore() as store:
                symbols = store.get_symbols_by_name(symbol, project_id)

            if kind:
                symbols = [s for s in symbols if s.kind.lower() == kind.lower()]

            definitions = [
                {
                    "name": s.name,
                    "file_path": s.file_path,
                    "lineno": s.lineno,
                    "kind": s.kind,
                    "scope": s.scope,
                    "signature": s.signature,
                    "language": s.language,
                }
                for s in symbols
            ]

            return _ok(
                found=len(definitions) > 0,
                definitions=definitions,
            )
        except Exception as e:
            logger.exception("vlt_code_lookup failed")
            return _err("INTERNAL_ERROR", str(e))
