"""CGC (CodeGraphContext) integration — structural code intelligence via graph DB.

Wraps CGC's GraphBuilder and CodeFinder for use within vlt's CodeRAG subsystem.
CGC handles symbol extraction, call graphs, class hierarchies, and repo maps
via tree-sitter + KùzuDB. Our existing BM25 search handles semantic queries.

Usage:
    from vlt.core.coderag.code_graph import get_code_graph_service

    svc = get_code_graph_service()
    await svc.index_project(Path("/my/project"))
    results = svc.lookup_symbol("MyClass")
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_instance: Optional["CodeGraphService"] = None


def get_code_graph_service(db_path: Optional[Path] = None) -> "CodeGraphService":
    """Get or create the singleton CodeGraphService.

    Args:
        db_path: KùzuDB storage directory. Defaults to ~/.vlt/cgc-graph.
                 Only used on first call (singleton).
    """
    global _instance
    if _instance is None:
        _instance = CodeGraphService(db_path=db_path)
    return _instance


def reset_code_graph_service() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class CodeGraphService:
    """Thin wrapper around CGC for structural code intelligence.

    Provides: symbol lookup, call graph queries, class hierarchy,
    dead code detection, and repo map generation from the KùzuDB graph.
    """

    def __init__(self, db_path: Optional[Path] = None):
        effective_path = db_path or Path.home() / ".vlt" / "cgc-graph"
        # KùzuDB creates the DB directory itself — only ensure the parent exists
        effective_path.parent.mkdir(parents=True, exist_ok=True)

        os.environ["DATABASE_TYPE"] = "kuzudb"
        os.environ["KUZUDB_PATH"] = str(effective_path)
        # Store full source in graph nodes for find_code
        os.environ.setdefault("INDEX_SOURCE", "true")

        from codegraphcontext.core import get_database_manager
        from codegraphcontext.tools.graph_builder import GraphBuilder
        from codegraphcontext.tools.code_finder import CodeFinder
        from codegraphcontext.core.jobs import JobManager

        self._db = get_database_manager()
        self._job_manager = JobManager()
        # Get or create event loop — needed for GraphBuilder even in sync contexts
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        self._builder = GraphBuilder(self._db, self._job_manager, loop)
        self._finder = CodeFinder(self._db)
        self._db_path = effective_path

        logger.info("CodeGraphService initialized (KùzuDB at %s)", effective_path)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_project(self, project_path: Path, is_dependency: bool = False) -> str:
        """Index a project into the KùzuDB graph. Returns job_id."""
        job_id = self._job_manager.create_job(str(project_path))
        await self._builder.build_graph_from_path_async(
            path=project_path,
            is_dependency=is_dependency,
            job_id=job_id,
        )
        logger.info("CGC indexing complete for %s (job %s)", project_path, job_id)
        return job_id

    def index_project_sync(self, project_path: Path, is_dependency: bool = False) -> str:
        """Synchronous wrapper for index_project."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.index_project(project_path, is_dependency))
        finally:
            loop.close()

    def get_job_status(self, job_id: str) -> dict:
        """Poll CGC indexing job progress."""
        job = self._job_manager.get_job(job_id)
        if not job:
            return {"status": "not_found"}
        return {
            "status": job.status.value,
            "progress": job.progress_percentage,
            "files_processed": job.processed_files,
            "files_total": job.total_files,
        }

    def delete_project(self, project_path: str) -> None:
        """Remove a project from the graph."""
        self._builder.delete_repository_from_graph(project_path)

    def is_indexed(self, project_path: str) -> bool:
        """Check if a project has been indexed in the graph."""
        try:
            repos = self._finder.list_indexed_repositories()
            if isinstance(repos, list):
                return any(
                    r.get("path", "") == project_path or r.get("name", "") == project_path
                    for r in repos
                )
            if isinstance(repos, dict):
                repo_list = repos.get("repositories", [])
                return any(
                    r.get("path", "") == project_path or r.get("name", "") == project_path
                    for r in repo_list
                )
        except Exception as exc:
            logger.debug("is_indexed check failed: %s", exc)
        return False

    # ------------------------------------------------------------------
    # Symbol lookup
    # ------------------------------------------------------------------

    def lookup_symbol(
        self,
        name: str,
        kind: str = "",
        repo_path: str = "",
    ) -> list[dict]:
        """Find symbol definitions by name.

        Args:
            name: Symbol name (e.g. "authenticate", "UserService").
            kind: Optional filter: "function", "class", "variable".
            repo_path: Optional repo path to scope the query.

        Returns:
            List of dicts with name, path, line_number, source, docstring, etc.
        """
        kwargs = {}
        if repo_path:
            kwargs["repo_path"] = repo_path

        if kind == "function" or kind == "method":
            return self._finder.find_by_function_name(name, fuzzy_search=False, **kwargs)
        elif kind == "class":
            return self._finder.find_by_class_name(name, fuzzy_search=False, **kwargs)
        elif kind == "variable":
            return self._finder.find_by_variable_name(name, **kwargs)
        else:
            results = []
            results.extend(self._finder.find_by_function_name(name, fuzzy_search=False, **kwargs))
            results.extend(self._finder.find_by_class_name(name, fuzzy_search=False, **kwargs))
            results.extend(self._finder.find_by_variable_name(name, **kwargs))
            return results

    # ------------------------------------------------------------------
    # Relationship queries
    # ------------------------------------------------------------------

    def find_callers(
        self,
        function_name: str,
        transitive: bool = False,
        path: str = "",
        repo_path: str = "",
    ) -> list[dict]:
        """Find functions that call the target function."""
        qtype = "find_all_callers" if transitive else "find_callers"
        result = self._finder.analyze_code_relationships(
            query_type=qtype,
            target=function_name,
            context=path or None,
            repo_path=repo_path or None,
        )
        return result if isinstance(result, list) else result.get("results", [])

    def find_callees(
        self,
        function_name: str,
        transitive: bool = False,
        path: str = "",
        repo_path: str = "",
    ) -> list[dict]:
        """Find functions that the target function calls."""
        qtype = "find_all_callees" if transitive else "find_callees"
        result = self._finder.analyze_code_relationships(
            query_type=qtype,
            target=function_name,
            context=path or None,
            repo_path=repo_path or None,
        )
        return result if isinstance(result, list) else result.get("results", [])

    def class_hierarchy(self, class_name: str, repo_path: str = "") -> dict:
        """Get parents and children of a class."""
        result = self._finder.analyze_code_relationships(
            query_type="class_hierarchy",
            target=class_name,
            repo_path=repo_path or None,
        )
        return result if isinstance(result, dict) else {"class_name": class_name, "results": result}

    def find_dead_code(
        self,
        repo_path: str = "",
        exclude_decorated: list[str] | None = None,
    ) -> dict:
        """Find functions with no callers (potentially unused)."""
        result = self._finder.find_dead_code(
            exclude_decorated_with=exclude_decorated or [],
            repo_path=repo_path or None,
        )
        return result if isinstance(result, dict) else {"results": result}

    # ------------------------------------------------------------------
    # Repo map
    # ------------------------------------------------------------------

    def get_repo_map(self, repo_path: str, scope: str = "") -> str:
        """Generate repo map from the KùzuDB graph.

        Queries Repository→File→Function/Class hierarchy and formats
        as an Aider-style tree string with symbols and line numbers.

        Args:
            repo_path: Absolute path of the indexed repository.
            scope: Optional path prefix filter (e.g. "src/api/").

        Returns:
            Formatted tree string, or "(empty graph)" if not indexed.
        """
        driver = self._db.get_driver()
        with driver.session() as s:
            scope_filter = ""
            if scope:
                scope_filter = f'AND f.relative_path STARTS WITH "{scope}"'

            result = s.run(f"""
                MATCH (r:Repository)-[:CONTAINS*]->(f:File)
                WHERE r.path = $repo {scope_filter}
                OPTIONAL MATCH (f)-[:CONTAINS]->(sym)
                WHERE label(sym) IN ['Function', 'Class', 'Variable',
                                      'Struct', 'Enum', 'Interface', 'Trait']
                RETURN f.relative_path AS file_path,
                       label(sym) AS sym_type,
                       sym.name AS sym_name,
                       sym.line_number AS sym_line
                ORDER BY f.relative_path, sym.line_number
            """, repo=repo_path)

            tree: dict[str, list[str]] = {}
            for row in result:
                fp = row.get("file_path") or row.get("f.relative_path")
                if not fp:
                    continue
                if fp not in tree:
                    tree[fp] = []
                sym_name = row.get("sym_name") or row.get("sym.name")
                sym_type = row.get("sym_type") or row.get("label(sym)")
                sym_line = row.get("sym_line") or row.get("sym.line_number")
                if sym_name and sym_type:
                    tree[fp].append(f"  {sym_type.lower()} {sym_name} (L{sym_line})")

            if not tree:
                return "(empty graph — run vlt_code_init with force=True to index)"

            lines = []
            for fp in sorted(tree.keys()):
                lines.append(fp)
                lines.extend(sorted(tree[fp]))
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, repo_path: str = "") -> dict:
        """Get graph statistics."""
        driver = self._db.get_driver()
        counts: dict[str, int] = {}
        with driver.session() as s:
            for label in ("Function", "Class", "File", "Variable"):
                try:
                    result = s.run(f"MATCH (n:{label}) RETURN count(n) AS c")
                    for row in result:
                        counts[label.lower() + "s"] = row.get("c", 0)
                except Exception:
                    counts[label.lower() + "s"] = 0
            try:
                result = s.run("MATCH ()-[r:CALLS]->() RETURN count(r) AS c")
                for row in result:
                    counts["call_edges"] = row.get("c", 0)
            except Exception:
                counts["call_edges"] = 0
        return counts
