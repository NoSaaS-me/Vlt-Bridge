"""SQLite database helpers for document indexing schema."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "index.db"

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS note_metadata (
        user_id TEXT NOT NULL,
        note_path TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        title TEXT NOT NULL,
        created TEXT NOT NULL,
        updated TEXT NOT NULL,
        size_bytes INTEGER NOT NULL DEFAULT 0,
        normalized_title_slug TEXT,
        normalized_path_slug TEXT,
        PRIMARY KEY (user_id, note_path)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_metadata_user ON note_metadata(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_metadata_updated ON note_metadata(user_id, updated DESC)",
    "CREATE INDEX IF NOT EXISTS idx_metadata_title_slug ON note_metadata(user_id, normalized_title_slug)",
    "CREATE INDEX IF NOT EXISTS idx_metadata_path_slug ON note_metadata(user_id, normalized_path_slug)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
        user_id UNINDEXED,
        note_path UNINDEXED,
        title,
        body,
        content='',
        tokenize='porter unicode61',
        prefix='2 3'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS note_tags (
        user_id TEXT NOT NULL,
        note_path TEXT NOT NULL,
        tag TEXT NOT NULL,
        PRIMARY KEY (user_id, note_path, tag)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tags_user_tag ON note_tags(user_id, tag)",
    "CREATE INDEX IF NOT EXISTS idx_tags_user_path ON note_tags(user_id, note_path)",
    """
    CREATE TABLE IF NOT EXISTS note_links (
        user_id TEXT NOT NULL,
        source_path TEXT NOT NULL,
        target_path TEXT,
        link_text TEXT NOT NULL,
        is_resolved INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (user_id, source_path, link_text)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_links_user_source ON note_links(user_id, source_path)",
    "CREATE INDEX IF NOT EXISTS idx_links_user_target ON note_links(user_id, target_path)",
    "CREATE INDEX IF NOT EXISTS idx_links_unresolved ON note_links(user_id, is_resolved)",
    """
    CREATE TABLE IF NOT EXISTS index_health (
        user_id TEXT PRIMARY KEY,
        note_count INTEGER NOT NULL DEFAULT 0,
        last_full_rebuild TEXT,
        last_incremental_update TEXT
    )
    """,
)


class DatabaseService:
    """Manage SQLite connections and schema initialization."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH

    def _ensure_directory(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """Return a sqlite3 connection with the proper data directory created."""
        self._ensure_directory()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self, statements: Iterable[str] | None = None) -> Path:
        """Create all schema artifacts required for indexing."""
        conn = self.connect()
        try:
            with conn:  # Transactional apply of DDL
                for statement in statements or DDL_STATEMENTS:
                    conn.execute(statement)
        finally:
            conn.close()
        return self.db_path


def init_database(db_path: str | Path | None = None) -> Path:
    """Convenience wrapper matching the quickstart instructions."""
    return DatabaseService(db_path).initialize()


__all__ = ["DatabaseService", "init_database", "DEFAULT_DB_PATH"]
