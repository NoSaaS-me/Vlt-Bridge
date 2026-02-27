"""Thread Service - CRUD operations for synced threads."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple

from ..services.database import DatabaseService
from ..models.thread import (
    Thread,
    ThreadEntry,
    SyncStatus,
    ThreadListResponse,
    ThreadSearchResult,
    ThreadSearchResponse,
)

logger = logging.getLogger(__name__)


class ThreadService:
    """Service for thread CRUD operations."""

    def __init__(self, db_service: DatabaseService | None = None):
        """Initialize with database service."""
        self._db = db_service or DatabaseService()

    # T006: create_or_update_thread
    def create_or_update_thread(
        self,
        user_id: str,
        thread_id: str,
        project_id: str,
        name: str,
        status: str = "active",
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> Thread:
        """Create a new thread or update an existing one."""
        conn = self._db.connect()
        now = datetime.utcnow().isoformat()
        created = created_at.isoformat() if created_at else now
        updated = updated_at.isoformat() if updated_at else now

        try:
            # Check if thread exists
            cursor = conn.execute(
                "SELECT 1 FROM threads WHERE user_id = ? AND thread_id = ?",
                (user_id, thread_id)
            )
            exists = cursor.fetchone() is not None

            if exists:
                # Update existing thread
                conn.execute(
                    """
                    UPDATE threads
                    SET project_id = ?, name = ?, status = ?, updated_at = ?
                    WHERE user_id = ? AND thread_id = ?
                    """,
                    (project_id, name, status, updated, user_id, thread_id)
                )
            else:
                # Insert new thread
                conn.execute(
                    """
                    INSERT INTO threads (user_id, thread_id, project_id, name, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, thread_id, project_id, name, status, created, updated)
                )
                # Initialize sync status
                conn.execute(
                    """
                    INSERT INTO thread_sync_status (user_id, thread_id, last_synced_sequence, last_sync_at)
                    VALUES (?, ?, -1, ?)
                    """,
                    (user_id, thread_id, now)
                )

            conn.commit()
            logger.info(f"{'Updated' if exists else 'Created'} thread {thread_id} for user {user_id}")

            return Thread(
                thread_id=thread_id,
                project_id=project_id,
                name=name,
                status=status,
                created_at=datetime.fromisoformat(created),
                updated_at=datetime.fromisoformat(updated),
            )
        finally:
            conn.close()

    # T007: add_entries
    def add_entries(
        self,
        user_id: str,
        thread_id: str,
        entries: List[ThreadEntry],
    ) -> Tuple[int, int]:
        """
        Add entries to a thread. Returns (synced_count, last_synced_sequence).
        Handles duplicates gracefully (INSERT OR IGNORE).
        """
        conn = self._db.connect()
        synced_count = 0
        last_seq = -1

        try:
            for entry in entries:
                # Insert entry (ignore if already exists)
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO thread_entries
                    (user_id, entry_id, thread_id, sequence_id, content, author, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        entry.entry_id,
                        thread_id,
                        entry.sequence_id,
                        entry.content,
                        entry.author,
                        entry.timestamp.isoformat(),
                    )
                )
                if cursor.rowcount > 0:
                    synced_count += 1
                    # Update FTS index
                    conn.execute(
                        """
                        INSERT INTO thread_entries_fts (rowid, content)
                        SELECT rowid, content FROM thread_entries
                        WHERE user_id = ? AND entry_id = ?
                        """,
                        (user_id, entry.entry_id)
                    )

                if entry.sequence_id > last_seq:
                    last_seq = entry.sequence_id

            # Update sync status
            now = datetime.utcnow().isoformat()
            conn.execute(
                """
                UPDATE thread_sync_status
                SET last_synced_sequence = MAX(last_synced_sequence, ?), last_sync_at = ?, sync_error = NULL
                WHERE user_id = ? AND thread_id = ?
                """,
                (last_seq, now, user_id, thread_id)
            )

            # Update thread's updated_at
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE user_id = ? AND thread_id = ?",
                (now, user_id, thread_id)
            )

            conn.commit()
            logger.info(f"Synced {synced_count} entries for thread {thread_id}")

            return synced_count, last_seq
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add entries: {e}")
            raise
        finally:
            conn.close()

    # T008: get_thread
    def get_thread(
        self,
        user_id: str,
        thread_id: str,
        include_entries: bool = True,
        entries_limit: int = 50,
    ) -> Optional[Thread]:
        """Get a thread by ID, optionally with entries."""
        conn = self._db.connect()

        try:
            cursor = conn.execute(
                """
                SELECT thread_id, project_id, name, status, created_at, updated_at
                FROM threads WHERE user_id = ? AND thread_id = ?
                """,
                (user_id, thread_id)
            )
            row = cursor.fetchone()

            if not row:
                return None

            entries = None
            if include_entries:
                entries_cursor = conn.execute(
                    """
                    SELECT entry_id, sequence_id, content, author, timestamp
                    FROM thread_entries
                    WHERE user_id = ? AND thread_id = ?
                    ORDER BY sequence_id DESC
                    LIMIT ?
                    """,
                    (user_id, thread_id, entries_limit)
                )
                entries = [
                    ThreadEntry(
                        entry_id=e["entry_id"],
                        sequence_id=e["sequence_id"],
                        content=e["content"],
                        author=e["author"],
                        timestamp=datetime.fromisoformat(e["timestamp"]),
                    )
                    for e in entries_cursor.fetchall()
                ]
                # Reverse to get chronological order
                entries = list(reversed(entries))

            return Thread(
                thread_id=row["thread_id"],
                project_id=row["project_id"],
                name=row["name"],
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                entries=entries,
            )
        finally:
            conn.close()

    # T009: list_threads
    def list_threads(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ThreadListResponse:
        """List threads for a user with optional filters."""
        conn = self._db.connect()

        try:
            # Build query with optional filters
            query = "SELECT thread_id, project_id, name, status, created_at, updated_at FROM threads WHERE user_id = ?"
            count_query = "SELECT COUNT(*) FROM threads WHERE user_id = ?"
            params = [user_id]

            if project_id:
                query += " AND project_id = ?"
                count_query += " AND project_id = ?"
                params.append(project_id)

            if status:
                query += " AND status = ?"
                count_query += " AND status = ?"
                params.append(status)

            # Get total count
            cursor = conn.execute(count_query, params)
            total = cursor.fetchone()[0]

            # Get threads with pagination
            query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            cursor = conn.execute(query, params + [limit, offset])

            threads = [
                Thread(
                    thread_id=row["thread_id"],
                    project_id=row["project_id"],
                    name=row["name"],
                    status=row["status"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
                for row in cursor.fetchall()
            ]

            return ThreadListResponse(threads=threads, total=total)
        finally:
            conn.close()

    # T010: get_sync_status
    def get_sync_status(self, user_id: str, thread_id: str) -> Optional[SyncStatus]:
        """Get sync status for a thread."""
        conn = self._db.connect()

        try:
            cursor = conn.execute(
                """
                SELECT thread_id, last_synced_sequence, last_sync_at, sync_error
                FROM thread_sync_status WHERE user_id = ? AND thread_id = ?
                """,
                (user_id, thread_id)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return SyncStatus(
                thread_id=row["thread_id"],
                last_synced_sequence=row["last_synced_sequence"],
                last_sync_at=datetime.fromisoformat(row["last_sync_at"]),
                sync_error=row["sync_error"],
            )
        finally:
            conn.close()

    @staticmethod
    def _prepare_fts_query(query: str) -> str:
        """Sanitize user-supplied query for FTS5 MATCH usage.

        Extracts alphanumeric tokens and wraps each in double quotes to
        neutralize FTS5 operator injection (AND, OR, NOT, NEAR, etc.).
        A single trailing '*' is preserved to allow prefix searches.
        Raises ValueError if no valid tokens are found.
        """
        token_pattern = re.compile(r"[0-9A-Za-z]+(?:\*)?")
        sanitized_terms: List[str] = []
        for match in token_pattern.finditer(query or ""):
            token = match.group()
            has_prefix_star = token.endswith("*")
            core = token[:-1] if has_prefix_star else token
            if not core:
                continue
            sanitized_terms.append(f'"{core}"{"*" if has_prefix_star else ""}')
        if not sanitized_terms:
            raise ValueError("Search query must contain alphanumeric characters")
        return " ".join(sanitized_terms)

    # T011: search_threads
    def search_threads(
        self,
        user_id: str,
        query: str,
        project_id: Optional[str] = None,
        limit: int = 10,
    ) -> ThreadSearchResponse:
        """Full-text search across thread entries."""
        conn = self._db.connect()

        try:
            # Sanitize query to prevent FTS5 operator injection
            sanitized_query = self._prepare_fts_query(query)

            # Use FTS5 for search with BM25 ranking
            sql = """
                SELECT
                    te.thread_id,
                    te.entry_id,
                    snippet(thread_entries_fts, 0, '<mark>', '</mark>', '...', 64) as content,
                    te.author,
                    te.timestamp,
                    bm25(thread_entries_fts) as score
                FROM thread_entries_fts fts
                JOIN thread_entries te ON fts.rowid = te.rowid
                JOIN threads t ON te.user_id = t.user_id AND te.thread_id = t.thread_id
                WHERE thread_entries_fts MATCH ? AND te.user_id = ?
            """
            params = [sanitized_query, user_id]

            if project_id:
                sql += " AND t.project_id = ?"
                params.append(project_id)

            sql += " ORDER BY score LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)

            results = [
                ThreadSearchResult(
                    thread_id=row["thread_id"],
                    entry_id=row["entry_id"],
                    content=row["content"],
                    author=row["author"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    score=abs(row["score"]),  # BM25 returns negative scores
                )
                for row in cursor.fetchall()
            ]

            return ThreadSearchResponse(results=results, total=len(results))
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ThreadSearchResponse(results=[], total=0)
        finally:
            conn.close()

    def delete_thread(self, user_id: str, thread_id: str) -> bool:
        """Delete a thread and all its entries."""
        conn = self._db.connect()

        try:
            # Delete from FTS first (get rowids)
            conn.execute(
                """
                DELETE FROM thread_entries_fts
                WHERE rowid IN (
                    SELECT rowid FROM thread_entries
                    WHERE user_id = ? AND thread_id = ?
                )
                """,
                (user_id, thread_id)
            )

            # CASCADE will handle entries and sync_status
            cursor = conn.execute(
                "DELETE FROM threads WHERE user_id = ? AND thread_id = ?",
                (user_id, thread_id)
            )

            conn.commit()
            deleted = cursor.rowcount > 0

            if deleted:
                logger.info(f"Deleted thread {thread_id} for user {user_id}")

            return deleted
        finally:
            conn.close()


# Singleton instance for dependency injection
_thread_service: ThreadService | None = None


def get_thread_service() -> ThreadService:
    """Get or create the thread service singleton."""
    global _thread_service
    if _thread_service is None:
        _thread_service = ThreadService()
    return _thread_service
