# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2026-01-01 14:25]
FTS5 functions like bm25() and snippet() cannot be used with GROUP BY clauses - the FTS5 context is lost. Use a subquery to filter results first, then join with FTS5 for scoring.

_Context: When adding tag filtering to search_notes(), the original implementation used GROUP BY/HAVING to find notes with all required tags, but this broke bm25() and snippet(). The fix is to use a subquery: first find note_paths matching all tags via GROUP BY in the subquery, then use that result in a WHERE IN clause for the main FTS5 query._

## [2026-01-01 14:31]
Tests cannot run in worktree due to pydantic_core module conflict between Vlt-Bridge venv and Auto-Claude venv. The PYTHONPATH appears to pick up the wrong pydantic from /Auto-Claude/apps/backend/.venv instead of the project venv.

_Context: Running pytest in worktree .worktrees/005-add-tag-filtering-to-search-endpoint with /mnt/Samsung2tb/Projects/00Tooling/Vlt-Bridge/backend/.venv/bin/python_
