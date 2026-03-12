"""Vault tools for the Oracle CodeAct agent — wraps existing vault/indexer services."""

from __future__ import annotations

from typing import Callable

from ...indexer import IndexerService
from ...vault import VaultService


def make_vault_tools(user_id: str, project_id: str) -> list[Callable]:
    """Factory: returns [vault_search, vault_read, vault_write, vault_list, vault_backlinks] bound to user/project.

    Args:
        user_id: The authenticated user's ID (captured in closure).
        project_id: The active project ID (captured in closure).

    Returns:
        List of five callables: [vault_search, vault_read, vault_write, vault_list, vault_backlinks]
    """
    # Instantiate services once — shared across all three tools
    _vault_svc = VaultService()
    _indexer_svc = IndexerService()

    def vault_search(query: str, limit: int = 10) -> str:
        """Search vault notes using full-text search (BM25 ranking).

        Args:
            query: Search query string.
            limit: Maximum results (default 10).

        Returns:
            Formatted list of matching notes with titles, paths, and snippets.
        """
        try:
            results = _indexer_svc.search_notes(
                user_id, query, limit=limit, project_id=project_id
            )
            if not results:
                return f"No results found for query: {query!r}"

            lines = [f"Found {len(results)} result(s) for {query!r}:\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "Untitled")
                path = r.get("path", "")
                snippet = r.get("snippet", "").strip()
                score = r.get("score", 0.0)
                lines.append(f"{i}. [{title}] ({path}) — score={score:.3f}")
                if snippet:
                    lines.append(f"   {snippet}")
            return "\n".join(lines)
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error searching vault: {exc}"

    def vault_read(path: str) -> str:
        """Read a vault note by its path.

        Args:
            path: Note path relative to vault root (e.g. 'Architecture/Overview.md').

        Returns:
            Full markdown content of the note.
        """
        try:
            note = _vault_svc.read_note(user_id, path, project_id=project_id)
            body = note.get("body", "")
            title = note.get("title", "")
            header = f"# {title}\n\n" if title else ""
            return f"{header}{body}"
        except FileNotFoundError as exc:
            return f"Error: {exc}"
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error reading vault note: {exc}"

    def vault_write(path: str, content: str) -> str:
        """Write or update a vault note.

        Args:
            path: Note path relative to vault root (e.g. 'Research/Finding.md').
            content: Full markdown content to write.

        Returns:
            Confirmation string with the path that was written.
        """
        try:
            note = _vault_svc.write_note(
                user_id,
                path,
                body=content,
                project_id=project_id,
            )
            written_path = note.get("path", path)
            version = note.get("metadata", {}).get("version")
            version_str = f" (version {version})" if version else ""
            return f"Written: {written_path}{version_str}"
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error writing vault note: {exc}"

    def vault_list(directory: str = "", limit: int = 50) -> str:
        """List vault notes, optionally scoped to a directory prefix.

        Args:
            directory: Optional folder prefix to scope the listing (e.g. 'Research').
                       Leave empty to list all notes.
            limit: Maximum number of notes to return (default 50, capped at 200).

        Returns:
            Formatted table of notes sorted by last_modified descending,
            or a message if no notes are found.
        """
        try:
            limit = min(limit, 200)
            folder = directory.strip() or None
            notes = _vault_svc.list_notes(
                user_id, folder=folder, project_id=project_id
            )
            if not notes:
                label = directory if directory else "(all)"
                return f"No notes found in '{label}'. Use vault_list() without arguments to see all notes."

            # Re-sort by last_modified descending (service returns alphabetical)
            notes_sorted = sorted(
                notes,
                key=lambda n: n.get("last_modified") or "",
                reverse=True,
            )
            shown = notes_sorted[:limit]
            total = len(notes_sorted)

            label = directory if directory else "(all)"
            lines = [
                f"Vault notes in: {label}",
                f"(showing {len(shown)} of {total} notes)",
                "",
            ]
            for note in shown:
                path = note.get("path", "")
                title = note.get("title", "")
                raw_date = note.get("last_modified")
                if raw_date:
                    date_str = str(raw_date)[:10]  # YYYY-MM-DD from ISO string or datetime
                else:
                    date_str = "unknown"
                lines.append(f"  {path}  {date_str}  {title}")

            return "\n".join(lines)
        except Exception as exc:
            return f"Error listing vault notes: {exc}"

    def vault_backlinks(path: str) -> str:
        """Get notes that link TO the given note via wikilinks.

        Args:
            path: Note path relative to vault root (e.g. 'Architecture/Overview.md').

        Returns:
            Formatted list of source notes that contain a wikilink to this note,
            or a message if none exist.
        """
        try:
            backlinks = _indexer_svc.get_backlinks(
                user_id, target_path=path, project_id=project_id
            )
            if not backlinks:
                return f"No notes link to '{path}'."

            lines = [
                f"Backlinks for: {path}",
                f"({len(backlinks)} notes link to this)",
                "",
            ]
            for bl in backlinks:
                source_path = bl.get("path", "")
                source_title = bl.get("title", "")
                lines.append(f"  {source_path}  {source_title}")

            return "\n".join(lines)
        except Exception as exc:
            return f"Error retrieving backlinks: {exc}"

    return [vault_search, vault_read, vault_write, vault_list, vault_backlinks]


__all__ = ["make_vault_tools"]
