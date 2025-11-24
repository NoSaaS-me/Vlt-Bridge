"""SQLite-backed indexing utilities for notes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, List, Sequence

from .database import DatabaseService
from .vault import VaultNote

logger = logging.getLogger(__name__)

WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z]+(?:\*)?")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_slug(text: str | None) -> str:
    """Normalize text into a slug suitable for wikilink matching."""
    if not text:
        return ""
    slug = text.lower()
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def normalize_tag(tag: str | None) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.strip().lower()


def _prepare_match_query(query: str) -> str:
    """
    Sanitize user-supplied query text for FTS5 MATCH usage.

    - Extracts tokens comprised of alphanumeric characters (per spec: split on non-alphanum).
    - Preserves a single trailing '*' to allow prefix searches.
    - Wraps each token in double quotes to neutralize MATCH operators.
    """
    sanitized_terms: List[str] = []

    for match in TOKEN_PATTERN.finditer(query or ""):
        token = match.group()
        has_prefix_star = token.endswith("*")
        core = token[:-1] if has_prefix_star else token
        if not core:
            continue
        sanitized_terms.append(f'"{core}"{"*" if has_prefix_star else ""}')

    if not sanitized_terms:
        raise ValueError("Search query must contain alphanumeric characters")

    return " ".join(sanitized_terms)


class IndexerService:
    """Manage SQLite-backed metadata, tags, search index, and link graph."""

    def __init__(self, db_service: DatabaseService | None = None) -> None:
        self.db_service = db_service or DatabaseService()

    def index_note(self, user_id: str, note: VaultNote) -> int:
        """Insert or update index rows for a note."""
        start_time = time.time()
        
        note_path = note["path"]
        metadata = dict(note.get("metadata") or {})
        title = note.get("title") or metadata.get("title") or Path(note_path).stem
        body = note.get("body", "") or ""
        size_bytes = int(note.get("size_bytes") or len(body.encode("utf-8")))
        created = str(metadata.get("created") or _utcnow_iso())
        updated = str(metadata.get("updated") or _utcnow_iso())

        normalized_title_slug = normalize_slug(title)
        normalized_path_slug = normalize_slug(Path(note_path).stem)
        if not normalized_title_slug:
            normalized_title_slug = normalized_path_slug

        tags = self._prepare_tags(metadata.get("tags"))
        wikilinks = self.extract_wikilinks(body)

        conn = self.db_service.connect()
        try:
            with conn:
                version = self.increment_version(conn, user_id, note_path)
                self._delete_current_entries(conn, user_id, note_path)

                conn.execute(
                    """
                    INSERT INTO note_metadata (
                        user_id, note_path, version, title,
                        created, updated, size_bytes,
                        normalized_title_slug, normalized_path_slug
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        note_path,
                        version,
                        title,
                        created,
                        updated,
                        size_bytes,
                        normalized_title_slug,
                        normalized_path_slug,
                    ),
                )

                conn.execute(
                    """
                    INSERT INTO note_fts (user_id, note_path, title, body)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, note_path, title, body),
                )

                if tags:
                    conn.executemany(
                        """
                        INSERT INTO note_tags (user_id, note_path, tag)
                        VALUES (?, ?, ?)
                        """,
                        [(user_id, note_path, tag) for tag in tags],
                    )

                if wikilinks:
                    resolved = self.resolve_wikilinks(conn, user_id, note_path, wikilinks)
                    conn.executemany(
                        """
                        INSERT INTO note_links (user_id, source_path, target_path, link_text, is_resolved)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                user_id,
                                note_path,
                                entry["target_path"],
                                entry["link_text"],
                                1 if entry["is_resolved"] else 0,
                            )
                            for entry in resolved
                        ],
                    )

                self.update_index_health(conn, user_id)

            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "Note indexed successfully",
                extra={
                    "user_id": user_id,
                    "note_path": note_path,
                    "version": version,
                    "tags_count": len(tags),
                    "wikilinks_count": len(wikilinks),
                    "duration_ms": f"{duration_ms:.2f}"
                }
            )

            return version
        finally:
            conn.close()

    def delete_note_index(self, user_id: str, note_path: str) -> None:
        """Remove all index data for a note and update backlinks."""
        conn = self.db_service.connect()
        try:
            with conn:
                self._delete_current_entries(conn, user_id, note_path)
                conn.execute(
                    """
                    UPDATE note_links
                    SET target_path = NULL, is_resolved = 0
                    WHERE user_id = ? AND target_path = ?
                    """,
                    (user_id, note_path),
                )
                self.update_index_health(conn, user_id)
        finally:
            conn.close()

    def extract_wikilinks(self, body: str) -> List[str]:
        """Extract wikilink text from Markdown body."""
        links = []
        for match in WIKILINK_PATTERN.finditer(body or ""):
            link_text = match.group(1).strip()
            if link_text:
                links.append(link_text)
        # Preserve order but drop duplicates
        seen: Dict[str, None] = {}
        for link in links:
            if link not in seen:
                seen[link] = None
        return list(seen.keys())

    def resolve_wikilinks(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        note_path: str,
        link_texts: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """Resolve wikilinks to target note paths using slug comparison."""
        if not link_texts:
            return []

        results: List[Dict[str, Any]] = []
        note_folder = Path(note_path).parent

        for text in link_texts:
            slug = normalize_slug(text)
            if not slug:
                results.append({"link_text": text, "target_path": None, "is_resolved": False})
                continue

            rows = conn.execute(
                """
                SELECT note_path
                FROM note_metadata
                WHERE user_id = ?
                  AND (normalized_title_slug = ? OR normalized_path_slug = ?)
                """,
                (user_id, slug, slug),
            ).fetchall()

            if not rows:
                results.append({"link_text": text, "target_path": None, "is_resolved": False})
                continue

            candidates = [row["note_path"] if isinstance(row, sqlite3.Row) else row[0] for row in rows]
            target = sorted(
                candidates,
                key=lambda candidate: (Path(candidate).parent != note_folder, candidate),
            )[0]

            results.append({"link_text": text, "target_path": target, "is_resolved": True})

        return results

    def increment_version(
        self, conn: sqlite3.Connection, user_id: str, note_path: str
    ) -> int:
        """Return the next version number for a note."""
        row = conn.execute(
            "SELECT version FROM note_metadata WHERE user_id = ? AND note_path = ?",
            (user_id, note_path),
        ).fetchone()
        if row is None:
            return 1
        current_version = row["version"] if isinstance(row, sqlite3.Row) else row[0]
        return int(current_version) + 1

    def update_index_health(self, conn: sqlite3.Connection, user_id: str) -> None:
        """Update per-user index health stats."""
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM note_metadata WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        note_count = int(row["count"] if isinstance(row, sqlite3.Row) else row[0])
        now_iso = _utcnow_iso()
        conn.execute(
            """
            INSERT INTO index_health (user_id, note_count, last_incremental_update)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                note_count = excluded.note_count,
                last_incremental_update = excluded.last_incremental_update
            """,
            (user_id, note_count, now_iso),
        )

    def search_notes(self, user_id: str, query: str, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Execute a full-text search with recency bonus scoring."""
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        sanitized_query = _prepare_match_query(query)

        conn = self.db_service.connect()
        try:
            rows = conn.execute(
                """
                SELECT
                    m.note_path,
                    m.title,
                    m.updated,
                    snippet(note_fts, 3, '<mark>', '</mark>', '...', 32) AS snippet,
                    bm25(note_fts, 3.0, 1.0) AS score
                FROM note_fts
                JOIN note_metadata m USING (user_id, note_path)
                WHERE note_fts.user_id = ? AND note_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (user_id, sanitized_query, limit),
            ).fetchall()
        finally:
            conn.close()

        now = datetime.now(timezone.utc)
        seven_days = timedelta(days=7)
        thirty_days = timedelta(days=30)

        results: List[Dict[str, Any]] = []
        for row in rows:
            updated_raw = row["updated"] if isinstance(row, sqlite3.Row) else row[2]
            snippet = row["snippet"] if isinstance(row, sqlite3.Row) else row[3]
            base_score = float(row["score"] if isinstance(row, sqlite3.Row) else row[4])
            try:
                updated_dt = datetime.fromisoformat(str(updated_raw))
            except ValueError:
                updated_dt = now
            delta = now - updated_dt
            if delta <= seven_days:
                bonus = 1.0
            elif delta <= thirty_days:
                bonus = 0.5
            else:
                bonus = 0.0

            results.append(
                {
                    "path": row["note_path"] if isinstance(row, sqlite3.Row) else row[0],
                    "title": row["title"] if isinstance(row, sqlite3.Row) else row[1],
                    "snippet": snippet or "",
                    "score": base_score + bonus,
                    "updated": updated_raw,
                }
            )

        return sorted(results, key=lambda item: item["score"], reverse=True)

    def get_backlinks(self, user_id: str, target_path: str) -> List[Dict[str, Any]]:
        """Return backlinks for a note."""
        conn = self.db_service.connect()
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT l.source_path, m.title
                FROM note_links l
                JOIN note_metadata m
                  ON l.user_id = m.user_id AND l.source_path = m.note_path
                WHERE l.user_id = ? AND l.target_path = ?
                ORDER BY m.updated DESC
                """,
                (user_id, target_path),
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "path": row["source_path"] if isinstance(row, sqlite3.Row) else row[0],
                "title": row["title"] if isinstance(row, sqlite3.Row) else row[1],
            }
            for row in rows
        ]

    def get_tags(self, user_id: str) -> List[Dict[str, Any]]:
        """Return tag counts for a user."""
        conn = self.db_service.connect()
        try:
            rows = conn.execute(
                """
                SELECT tag, COUNT(DISTINCT note_path) AS count
                FROM note_tags
                WHERE user_id = ?
                GROUP BY tag
                ORDER BY count DESC, tag ASC
                """,
                (user_id,),
            ).fetchall()
        finally:
            conn.close()

        return [
            {"tag": row["tag"] if isinstance(row, sqlite3.Row) else row[0], "count": int(row["count"] if isinstance(row, sqlite3.Row) else row[1])}
            for row in rows
        ]

    def _delete_current_entries(self, conn: sqlite3.Connection, user_id: str, note_path: str) -> None:
        """Delete existing index rows for a note."""
        conn.execute(
            "DELETE FROM note_metadata WHERE user_id = ? AND note_path = ?",
            (user_id, note_path),
        )
        conn.execute(
            "DELETE FROM note_fts WHERE user_id = ? AND note_path = ?",
            (user_id, note_path),
        )
        conn.execute(
            "DELETE FROM note_tags WHERE user_id = ? AND note_path = ?",
            (user_id, note_path),
        )
        conn.execute(
            "DELETE FROM note_links WHERE user_id = ? AND source_path = ?",
            (user_id, note_path),
        )

    def _prepare_tags(self, tags: Any) -> List[str]:
        if not isinstance(tags, list):
            return []
        normalized: List[str] = []
        for tag in tags:
            cleaned = normalize_tag(tag)
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized


__all__ = ["IndexerService", "normalize_slug", "normalize_tag"]
