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
    """
    CREATE TABLE IF NOT EXISTS user_settings (
        user_id TEXT PRIMARY KEY,
        oracle_model TEXT NOT NULL DEFAULT 'gemini-2.0-flash-exp',
        oracle_provider TEXT NOT NULL DEFAULT 'google',
        subagent_model TEXT NOT NULL DEFAULT 'gemini-2.0-flash-exp',
        subagent_provider TEXT NOT NULL DEFAULT 'google',
        thinking_enabled INTEGER NOT NULL DEFAULT 0,
        chat_center_mode INTEGER NOT NULL DEFAULT 0,
        librarian_timeout INTEGER NOT NULL DEFAULT 1200,
        max_context_nodes INTEGER NOT NULL DEFAULT 30,
        openrouter_api_key TEXT,
        created TEXT NOT NULL,
        updated TEXT NOT NULL
    )
    """,
    # Thread Sync tables (T001)
    """
    CREATE TABLE IF NOT EXISTS threads (
        user_id TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'archived', 'blocked')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (user_id, thread_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_threads_user_project ON threads(user_id, project_id)",
    "CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(user_id, status)",
    """
    CREATE TABLE IF NOT EXISTS thread_entries (
        user_id TEXT NOT NULL,
        entry_id TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        sequence_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        author TEXT NOT NULL DEFAULT 'user',
        timestamp TEXT NOT NULL,
        PRIMARY KEY (user_id, entry_id),
        FOREIGN KEY (user_id, thread_id) REFERENCES threads(user_id, thread_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entries_thread_seq ON thread_entries(user_id, thread_id, sequence_id)",
    "CREATE INDEX IF NOT EXISTS idx_entries_timestamp ON thread_entries(user_id, timestamp)",
    """
    CREATE TABLE IF NOT EXISTS thread_sync_status (
        user_id TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        last_synced_sequence INTEGER NOT NULL DEFAULT -1,
        last_sync_at TEXT NOT NULL,
        sync_error TEXT,
        PRIMARY KEY (user_id, thread_id),
        FOREIGN KEY (user_id, thread_id) REFERENCES threads(user_id, thread_id) ON DELETE CASCADE
    )
    """,
    # Thread entries FTS5 (T002)
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS thread_entries_fts USING fts5(
        content,
        content='thread_entries',
        content_rowid=rowid,
        tokenize='porter unicode61'
    )
    """,
    # Oracle context persistence (009-oracle-agent T010) - Legacy flat context (deprecated)
    """
    CREATE TABLE IF NOT EXISTS oracle_contexts (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        session_start TEXT NOT NULL,
        last_activity TEXT,
        last_model TEXT,
        token_budget INTEGER DEFAULT 16000,
        tokens_used INTEGER DEFAULT 0,
        compressed_summary TEXT,
        recent_exchanges_json TEXT DEFAULT '[]',
        key_decisions_json TEXT DEFAULT '[]',
        mentioned_symbols TEXT,
        mentioned_files TEXT,
        status TEXT DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'closed')),
        compression_count INTEGER DEFAULT 0,
        UNIQUE(user_id, project_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_oracle_contexts_user_project ON oracle_contexts(user_id, project_id)",
    "CREATE INDEX IF NOT EXISTS idx_oracle_contexts_last_activity ON oracle_contexts(last_activity)",
    # Context tree tables (009-oracle-agent - branching conversation history)
    # Individual conversation nodes in the tree
    """
    CREATE TABLE IF NOT EXISTS context_nodes (
        id TEXT PRIMARY KEY,
        root_id TEXT NOT NULL,
        parent_id TEXT,
        user_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        created_at TEXT NOT NULL,

        -- Content
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        tool_calls_json TEXT DEFAULT '[]',
        tokens_used INTEGER DEFAULT 0,
        model_used TEXT,

        -- Metadata
        label TEXT,
        is_checkpoint INTEGER DEFAULT 0,
        is_root INTEGER DEFAULT 0,

        FOREIGN KEY (parent_id) REFERENCES context_nodes(id) ON DELETE SET NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_context_nodes_user_project ON context_nodes(user_id, project_id)",
    "CREATE INDEX IF NOT EXISTS idx_context_nodes_root ON context_nodes(root_id)",
    "CREATE INDEX IF NOT EXISTS idx_context_nodes_parent ON context_nodes(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_context_nodes_checkpoint ON context_nodes(user_id, project_id, is_checkpoint)",
    # Tree metadata (one per root)
    """
    CREATE TABLE IF NOT EXISTS context_trees (
        root_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        current_node_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_activity TEXT NOT NULL,
        node_count INTEGER DEFAULT 1,
        max_nodes INTEGER DEFAULT 30,
        label TEXT,

        FOREIGN KEY (root_id) REFERENCES context_nodes(id) ON DELETE CASCADE,
        FOREIGN KEY (current_node_id) REFERENCES context_nodes(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_context_trees_user_project ON context_trees(user_id, project_id)",
    "CREATE INDEX IF NOT EXISTS idx_context_trees_last_activity ON context_trees(last_activity)",
)

# Migration statements for existing databases
MIGRATION_STATEMENTS: tuple[str, ...] = (
    # Add openrouter_api_key column if it doesn't exist
    "ALTER TABLE user_settings ADD COLUMN openrouter_api_key TEXT",
    # Add librarian_timeout column if it doesn't exist (default 1200 = 20 minutes)
    "ALTER TABLE user_settings ADD COLUMN librarian_timeout INTEGER NOT NULL DEFAULT 1200",
    # Add max_context_nodes column if it doesn't exist (default 30 nodes per tree)
    "ALTER TABLE user_settings ADD COLUMN max_context_nodes INTEGER NOT NULL DEFAULT 30",
    # Add chat_center_mode column if it doesn't exist (default 0 = flyout panel)
    "ALTER TABLE user_settings ADD COLUMN chat_center_mode INTEGER NOT NULL DEFAULT 0",
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
            # Run migrations for existing databases (ignore errors for already-applied migrations)
            for migration in MIGRATION_STATEMENTS:
                try:
                    conn.execute(migration)
                    conn.commit()
                except sqlite3.OperationalError:
                    pass  # Column/table already exists
        finally:
            conn.close()
        return self.db_path


def init_database(db_path: str | Path | None = None) -> Path:
    """Convenience wrapper matching the quickstart instructions."""
    return DatabaseService(db_path).initialize()


__all__ = ["DatabaseService", "init_database", "DEFAULT_DB_PATH"]
