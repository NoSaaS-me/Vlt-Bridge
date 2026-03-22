"""Code intelligence tools for the vlt MCP server.

Exposes CodeRAG + CGC (CodeGraphContext) functionality via MCP so AI agents
can index, search, and navigate codebases without CLI subprocess overhead.

Tools registered:
    vlt_code_init      — start code indexing for a project (CodeRAG + CGC)
    vlt_code_search    — hybrid BM25 + vector search across indexed code (unchanged)
    vlt_code_map       — retrieve the repository structure map via CGC KùzuDB
    vlt_code_status    — check index status, job progress, and CGC graph stats
    vlt_code_lookup    — find symbol definitions by name via CGC
    vlt_code_callers   — find functions that call a target function
    vlt_code_callees   — find functions that a target function calls
    vlt_code_hierarchy — get class inheritance hierarchy (parents + children)
    vlt_code_dead_code — detect potentially unused functions
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

    # ------------------------------------------------------------------
    # Shared helper — resolve absolute project path from last completed job
    # ------------------------------------------------------------------

    def _get_project_target_path(project_id: str) -> Optional[str]:
        """Resolve absolute project path from the most recent completed index job."""
        from sqlalchemy.orm import Session
        from sqlalchemy import select
        from vlt.db import engine
        from vlt.core.models import CodeRAGIndexJob, JobStatus

        with Session(engine) as session:
            job = session.scalars(
                select(CodeRAGIndexJob)
                .where(CodeRAGIndexJob.project_id == project_id)
                .where(CodeRAGIndexJob.status == JobStatus.COMPLETED)
                .order_by(CodeRAGIndexJob.completed_at.desc())
                .limit(1)
            ).first()
            return job.target_path if job else None

    # ------------------------------------------------------------------
    # vlt_code_init
    # ------------------------------------------------------------------

    @mcp.tool()
    def vlt_code_init(
        project_id: str,
        path: str,
        force: bool = False,
    ) -> dict:
        """Start code indexing for a project. Returns immediately — indexing runs async.

        Runs two indexers in sequence:
          1. CodeRAG BM25/vector indexer (for vlt_code_search)
          2. CGC KùzuDB graph indexer (for vlt_code_lookup, vlt_code_map,
             vlt_code_callers, vlt_code_callees, vlt_code_hierarchy, vlt_code_dead_code)

        If daemon is running: CodeRAG job is queued for background processing and
        CGC indexing runs after it completes.
        If daemon is not running: both run in a detached thread.

        After calling this, poll vlt_code_status(project_id) until indexed=true
        before using search/map/lookup tools.

        If force=False and indexing is already in progress, returns the existing
        job info instead of creating a new one. Use force=True to cancel any
        in-flight job and restart from scratch.

        Args:
            project_id: Project identifier slug (e.g. "my-project").
            path: Absolute path to the codebase root directory to index.
            force: Re-index even if an index already exists. Default: False.

        Returns:
            {status, project_id, job_id, job_status: "queued"|"running"|"completed"|"failed",
             daemon_handling: bool, message}
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
                # Run both CodeRAG and CGC indexing in a detached background thread
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

                        # CGC indexing — failure does NOT fail the overall job
                        try:
                            from vlt.core.coderag.code_graph import get_code_graph_service
                            graph_svc = get_code_graph_service()
                            graph_svc.index_project_sync(target_path)
                            logger.info("CGC indexing complete for project %s", project_id)
                        except Exception as cgc_exc:
                            logger.warning(
                                "CGC indexing failed for project %s (CodeRAG index still valid): %s",
                                project_id,
                                cgc_exc,
                            )

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

    # ------------------------------------------------------------------
    # vlt_code_search  (unchanged — BM25 works fine)
    # ------------------------------------------------------------------

    @mcp.tool()
    def vlt_code_search(
        query: str,
        project_id: str,
        limit: int = 10,
        language: Optional[str] = None,
    ) -> dict:
        """Search indexed code using hybrid BM25 + keyword retrieval.

        PREREQUISITE: Project must have a completed code index. If you get
        INDEX_NOT_FOUND, call vlt_code_init first, then poll vlt_code_status
        until indexed=true.

        Returns code chunks ranked by relevance with file path and line numbers.
        Snippets are truncated to 300 chars.

        Args:
            query: Natural language or keyword search query (e.g. "authentication middleware").
            project_id: Project to search in.
            limit: Maximum results to return. Default: 10.
            language: Filter to a specific language, e.g. "python", "typescript" (optional).
                      May return fewer than limit results due to post-filtering.

        Returns:
            {status, results: [{chunk_id, file_path, qualified_name, lineno, score, snippet}], total}
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

    # ------------------------------------------------------------------
    # vlt_code_map  (rewired — CGC KùzuDB)
    # ------------------------------------------------------------------

    @mcp.tool()
    def vlt_code_map(
        project_id: str,
        scope: Optional[str] = None,
    ) -> dict:
        """Retrieve the repository structure map for a project via CGC KùzuDB.

        Returns a compact textual overview of the codebase: files, classes,
        functions, and their relationships (Aider-style tree). Good for
        initial codebase orientation.

        Requires a completed code index (call vlt_code_init first if needed).
        The map is generated live from the KùzuDB graph — always up-to-date
        after indexing completes.

        Args:
            project_id: Project identifier slug.
            scope: Limit map to files under this path prefix (optional, e.g. "src/api/").

        Returns:
            {status, map_text: str, source: "cgc-kuzudb"}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService

        try:
            svc = SqliteVaultService()
            if not svc.has_coderag_index(project_id):
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No code index for project '{project_id}'. Call vlt_code_init first.",
                )

            target_path = _get_project_target_path(project_id)
            if not target_path:
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No completed index job found for project '{project_id}'.",
                )

            from vlt.core.coderag.code_graph import get_code_graph_service
            graph_svc = get_code_graph_service()
            map_text = graph_svc.get_repo_map(repo_path=target_path, scope=scope or "")

            return _ok(map_text=map_text, source="cgc-kuzudb")
        except Exception as e:
            logger.exception("vlt_code_map failed")
            return _err("INTERNAL_ERROR", str(e))

    # ------------------------------------------------------------------
    # vlt_code_status  (enhanced — adds CGC graph stats)
    # ------------------------------------------------------------------

    @mcp.tool()
    def vlt_code_status(
        project_id: str,
    ) -> dict:
        """Check code index status and active job progress for a project.

        Returns CodeRAG stats (chunks, symbols) merged with CGC graph stats
        (functions, classes, files, call edges) when available.

        Args:
            project_id: Project identifier slug.

        Returns:
            {status, indexed, active_job: {job_id, status, progress_percent,
            files_processed, files_total, chunks_created, error_message} | null,
            stats: {cgc_functions, cgc_classes, cgc_files, cgc_call_edges, ...}}
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

            # CodeRAG chunk/symbol/graph counts
            stats: dict = {}
            try:
                with CodeRAGStore() as store:
                    stats = store.get_project_stats(project_id)
            except Exception:
                pass

            # CGC graph stats — failure is non-fatal
            try:
                from vlt.core.coderag.code_graph import get_code_graph_service
                graph_svc = get_code_graph_service()
                cgc_raw = graph_svc.get_stats()
                stats["cgc_functions"] = cgc_raw.get("functions", 0)
                stats["cgc_classes"] = cgc_raw.get("classes", 0)
                stats["cgc_files"] = cgc_raw.get("files", 0)
                stats["cgc_call_edges"] = cgc_raw.get("call_edges", 0)
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

    # ------------------------------------------------------------------
    # vlt_code_lookup  (rewired — CGC KùzuDB)
    # ------------------------------------------------------------------

    @mcp.tool()
    def vlt_code_lookup(
        symbol: str,
        project_id: str,
        kind: Optional[str] = None,
    ) -> dict:
        """Look up symbol definitions by name in the code graph (via CGC KùzuDB).

        Finds where a function, class, method, or variable is defined.
        Queries the live KùzuDB graph — more accurate and up-to-date than
        the previous FTS symbol table approach.

        Requires a completed code index (call vlt_code_init first if needed).

        Args:
            symbol: Symbol name to look up (e.g. "authenticate", "UserService").
            project_id: Project to search in.
            kind: Filter by symbol kind (case-insensitive). Common values:
                  "function", "class", "method", "variable".
                  Omit to search all symbol types.

        Returns:
            {status, found: bool, definitions: [{name, file_path, lineno, kind,
             scope, signature, language}]}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService

        try:
            svc = SqliteVaultService()
            if not svc.has_coderag_index(project_id):
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No code index for project '{project_id}'. Call vlt_code_init first.",
                )

            target_path = _get_project_target_path(project_id)
            if not target_path:
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No completed index job found for project '{project_id}'.",
                )

            from vlt.core.coderag.code_graph import get_code_graph_service
            graph_svc = get_code_graph_service()
            raw = graph_svc.lookup_symbol(
                name=symbol,
                kind=kind or "",
                repo_path=target_path,
            )

            definitions = [
                {
                    "name": r.get("name", symbol),
                    "file_path": r.get("path", ""),
                    "lineno": r.get("line_number"),
                    "kind": r.get("label") or r.get("type") or r.get("kind", ""),
                    "scope": "",
                    "signature": "",
                    "language": r.get("lang", ""),
                }
                for r in raw
            ]

            return _ok(found=len(definitions) > 0, definitions=definitions)
        except Exception as e:
            logger.exception("vlt_code_lookup failed")
            return _err("INTERNAL_ERROR", str(e))

    # ------------------------------------------------------------------
    # vlt_code_callers  (new — CGC call graph)
    # ------------------------------------------------------------------

    @mcp.tool()
    def vlt_code_callers(
        function_name: str,
        project_id: str,
        transitive: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """Find all functions that call the specified function (callers / call sites).

        Queries the CGC KùzuDB call graph for direct callers, or the full
        transitive set of callers when transitive=True.

        Requires a completed code index (call vlt_code_init first if needed).

        Args:
            function_name: Name of the function to find callers of.
            project_id: Project to search in.
            transitive: If True, return all transitive callers (full call chain).
                        Default: False (direct callers only).
            path: Optionally scope the query to a specific file path.

        Returns:
            {status, results: [{caller_name, caller_file_path, caller_line_number}]}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService

        try:
            svc = SqliteVaultService()
            if not svc.has_coderag_index(project_id):
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No code index for project '{project_id}'. Call vlt_code_init first.",
                )

            target_path = _get_project_target_path(project_id)
            if not target_path:
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No completed index job found for project '{project_id}'.",
                )

            from vlt.core.coderag.code_graph import get_code_graph_service
            graph_svc = get_code_graph_service()
            raw = graph_svc.find_callers(
                function_name=function_name,
                transitive=transitive,
                path=path or "",
                repo_path=target_path,
            )

            results = [
                {
                    "caller_name": r.get("name") or r.get("caller_name", ""),
                    "caller_file_path": r.get("path") or r.get("file_path", ""),
                    "caller_line_number": r.get("line_number") or r.get("lineno"),
                }
                for r in raw
            ]

            return _ok(results=results)
        except Exception as e:
            logger.exception("vlt_code_callers failed")
            return _err("INTERNAL_ERROR", str(e))

    # ------------------------------------------------------------------
    # vlt_code_callees  (new — CGC call graph)
    # ------------------------------------------------------------------

    @mcp.tool()
    def vlt_code_callees(
        function_name: str,
        project_id: str,
        transitive: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """Find all functions that the specified function calls (callees / dependencies).

        Queries the CGC KùzuDB call graph for direct callees, or the full
        transitive set when transitive=True. Useful for understanding a
        function's dependency footprint.

        Requires a completed code index (call vlt_code_init first if needed).

        Args:
            function_name: Name of the function to inspect.
            project_id: Project to search in.
            transitive: If True, return all transitive callees (full dependency tree).
                        Default: False (direct callees only).
            path: Optionally scope the query to a specific file path.

        Returns:
            {status, results: [{callee_name, callee_file_path, callee_line_number}]}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService

        try:
            svc = SqliteVaultService()
            if not svc.has_coderag_index(project_id):
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No code index for project '{project_id}'. Call vlt_code_init first.",
                )

            target_path = _get_project_target_path(project_id)
            if not target_path:
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No completed index job found for project '{project_id}'.",
                )

            from vlt.core.coderag.code_graph import get_code_graph_service
            graph_svc = get_code_graph_service()
            raw = graph_svc.find_callees(
                function_name=function_name,
                transitive=transitive,
                path=path or "",
                repo_path=target_path,
            )

            results = [
                {
                    "callee_name": r.get("name") or r.get("callee_name", ""),
                    "callee_file_path": r.get("path") or r.get("file_path", ""),
                    "callee_line_number": r.get("line_number") or r.get("lineno"),
                }
                for r in raw
            ]

            return _ok(results=results)
        except Exception as e:
            logger.exception("vlt_code_callees failed")
            return _err("INTERNAL_ERROR", str(e))

    # ------------------------------------------------------------------
    # vlt_code_hierarchy  (new — CGC class graph)
    # ------------------------------------------------------------------

    @mcp.tool()
    def vlt_code_hierarchy(
        class_name: str,
        project_id: str,
    ) -> dict:
        """Get the class inheritance hierarchy for a class (parents and children).

        Queries the CGC KùzuDB graph for the class's superclasses (parents)
        and subclasses (children). Useful for understanding polymorphic
        relationships and refactoring impact.

        Requires a completed code index (call vlt_code_init first if needed).

        Args:
            class_name: Name of the class to inspect.
            project_id: Project to search in.

        Returns:
            {status, class_name: str, parents: [...], children: [...]}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService

        try:
            svc = SqliteVaultService()
            if not svc.has_coderag_index(project_id):
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No code index for project '{project_id}'. Call vlt_code_init first.",
                )

            target_path = _get_project_target_path(project_id)
            if not target_path:
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No completed index job found for project '{project_id}'.",
                )

            from vlt.core.coderag.code_graph import get_code_graph_service
            graph_svc = get_code_graph_service()
            raw = graph_svc.class_hierarchy(class_name=class_name, repo_path=target_path)

            # Normalise — CGC may return various shapes
            parents = raw.get("parents") or raw.get("superclasses") or []
            children = raw.get("children") or raw.get("subclasses") or []

            return _ok(
                class_name=class_name,
                parents=parents,
                children=children,
            )
        except Exception as e:
            logger.exception("vlt_code_hierarchy failed")
            return _err("INTERNAL_ERROR", str(e))

    # ------------------------------------------------------------------
    # vlt_code_dead_code  (new — CGC call graph)
    # ------------------------------------------------------------------

    @mcp.tool()
    def vlt_code_dead_code(
        project_id: str,
    ) -> dict:
        """Detect potentially unused (dead) functions in the project.

        Queries the CGC KùzuDB graph for functions with no callers.
        Useful for codebase cleanup and identifying orphaned code.

        Note: Functions decorated with framework decorators (e.g. @app.route,
        @pytest.mark) are typically entry points and may still be needed even
        without call-graph callers.

        Requires a completed code index (call vlt_code_init first if needed).

        Args:
            project_id: Project to search in.

        Returns:
            {status, results: [{function_name, path, line_number}]}
        """
        from vlt.mcp import _ok, _err
        from vlt.core.service import SqliteVaultService

        try:
            svc = SqliteVaultService()
            if not svc.has_coderag_index(project_id):
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No code index for project '{project_id}'. Call vlt_code_init first.",
                )

            target_path = _get_project_target_path(project_id)
            if not target_path:
                return _err(
                    "INDEX_NOT_FOUND",
                    f"No completed index job found for project '{project_id}'.",
                )

            from vlt.core.coderag.code_graph import get_code_graph_service
            graph_svc = get_code_graph_service()
            raw = graph_svc.find_dead_code(repo_path=target_path)

            # CGC returns dict with "results" key, or possibly a direct list
            items = raw.get("results") or raw.get("dead_functions") or []
            if isinstance(items, dict):
                items = list(items.values())

            results = [
                {
                    "function_name": item.get("name") or item.get("function_name", ""),
                    "path": item.get("path") or item.get("file_path", ""),
                    "line_number": item.get("line_number") or item.get("lineno"),
                }
                for item in items
                if isinstance(item, dict)
            ]

            return _ok(results=results)
        except Exception as e:
            logger.exception("vlt_code_dead_code failed")
            return _err("INTERNAL_ERROR", str(e))
