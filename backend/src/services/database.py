"""SQLite database helpers for document indexing schema."""

from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
from typing import Iterable

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "index.db"

DDL_STATEMENTS: tuple[str, ...] = (
    # Projects table (new for multi-project support)
    """
    CREATE TABLE IF NOT EXISTS projects (
        user_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        settings_json TEXT,
        PRIMARY KEY (user_id, project_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id)",
    # Note metadata with project_id
    """
    CREATE TABLE IF NOT EXISTS note_metadata (
        user_id TEXT NOT NULL,
        project_id TEXT NOT NULL DEFAULT 'default',
        note_path TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        title TEXT NOT NULL,
        created TEXT NOT NULL,
        updated TEXT NOT NULL,
        size_bytes INTEGER NOT NULL DEFAULT 0,
        normalized_title_slug TEXT,
        normalized_path_slug TEXT,
        PRIMARY KEY (user_id, project_id, note_path)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_metadata_user ON note_metadata(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_metadata_user_project ON note_metadata(user_id, project_id)",
    "CREATE INDEX IF NOT EXISTS idx_metadata_updated ON note_metadata(user_id, project_id, updated DESC)",
    "CREATE INDEX IF NOT EXISTS idx_metadata_title_slug ON note_metadata(user_id, project_id, normalized_title_slug)",
    "CREATE INDEX IF NOT EXISTS idx_metadata_path_slug ON note_metadata(user_id, project_id, normalized_path_slug)",
    # FTS with project_id
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
        user_id UNINDEXED,
        project_id UNINDEXED,
        note_path UNINDEXED,
        title,
        body,
        tokenize='porter unicode61',
        prefix='2 3'
    )
    """,
    # Note tags with project_id
    """
    CREATE TABLE IF NOT EXISTS note_tags (
        user_id TEXT NOT NULL,
        project_id TEXT NOT NULL DEFAULT 'default',
        note_path TEXT NOT NULL,
        tag TEXT NOT NULL,
        PRIMARY KEY (user_id, project_id, note_path, tag)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tags_user_project ON note_tags(user_id, project_id)",
    "CREATE INDEX IF NOT EXISTS idx_tags_user_tag ON note_tags(user_id, project_id, tag)",
    "CREATE INDEX IF NOT EXISTS idx_tags_user_path ON note_tags(user_id, project_id, note_path)",
    # Note links with project_id
    """
    CREATE TABLE IF NOT EXISTS note_links (
        user_id TEXT NOT NULL,
        project_id TEXT NOT NULL DEFAULT 'default',
        source_path TEXT NOT NULL,
        target_path TEXT,
        link_text TEXT NOT NULL,
        is_resolved INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (user_id, project_id, source_path, link_text)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_links_user_project ON note_links(user_id, project_id)",
    "CREATE INDEX IF NOT EXISTS idx_links_user_source ON note_links(user_id, project_id, source_path)",
    "CREATE INDEX IF NOT EXISTS idx_links_user_target ON note_links(user_id, project_id, target_path)",
    "CREATE INDEX IF NOT EXISTS idx_links_unresolved ON note_links(user_id, project_id, is_resolved)",
    # Index health with project_id
    """
    CREATE TABLE IF NOT EXISTS index_health (
        user_id TEXT NOT NULL,
        project_id TEXT NOT NULL DEFAULT 'default',
        note_count INTEGER NOT NULL DEFAULT 0,
        last_full_rebuild TEXT,
        last_incremental_update TEXT,
        PRIMARY KEY (user_id, project_id)
    )
    """,
    # User settings with default_project_id
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
        default_project_id TEXT DEFAULT 'default',
        disabled_subscribers_json TEXT DEFAULT '[]',
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
        system_messages_json TEXT DEFAULT '[]',

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
    # Add default_project_id column to user_settings (010-multi-project)
    "ALTER TABLE user_settings ADD COLUMN default_project_id TEXT DEFAULT 'default'",
    # Add disabled_subscribers_json column to user_settings (013-agent-notification-system T006)
    "ALTER TABLE user_settings ADD COLUMN disabled_subscribers_json TEXT DEFAULT '[]'",
    # Add system_messages_json column to context_nodes (013-agent-notification-system T007)
    "ALTER TABLE context_nodes ADD COLUMN system_messages_json TEXT DEFAULT '[]'",
)


# Multi-project migration (010-multi-project)
# These are complex migrations that need table recreation
MULTI_PROJECT_MIGRATION_VERSION = "010"


def _check_has_project_id_column(conn: sqlite3.Connection, table: str) -> bool:
    """Check if a table has the project_id column."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return "project_id" in columns


def _check_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Check if a table exists."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    )
    return cursor.fetchone() is not None


def run_multi_project_migration(conn: sqlite3.Connection) -> bool:
    """
    Run the multi-project migration to add project_id to vault tables.

    This migration:
    1. Creates the projects table
    2. Adds project_id column to note_metadata, note_tags, note_links, index_health
    3. Recreates note_fts with project_id
    4. Creates default project for existing users
    5. Migrates existing data to project_id='default'

    Returns True if migration was performed, False if already migrated.
    """
    from datetime import datetime, timezone

    # Check if already migrated by looking for project_id in note_metadata
    if _check_has_project_id_column(conn, "note_metadata"):
        logger.debug("Multi-project migration already applied (note_metadata has project_id)")
        return False

    logger.info("Starting multi-project migration (010)...")

    try:
        # 1. Create projects table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                settings_json TEXT,
                PRIMARY KEY (user_id, project_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id)")

        # 2. Get all existing users from note_metadata
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cursor = conn.execute("SELECT DISTINCT user_id FROM note_metadata")
        existing_users = [row[0] for row in cursor.fetchall()]

        # 3. Create default project for each existing user
        for user_id in existing_users:
            conn.execute("""
                INSERT OR IGNORE INTO projects (user_id, project_id, name, description, created_at, updated_at)
                VALUES (?, 'default', 'Default Project', 'Migrated from single-vault', ?, ?)
            """, (user_id, now_iso, now_iso))

        # 4. Migrate note_metadata - recreate with project_id
        conn.execute("""
            CREATE TABLE IF NOT EXISTS note_metadata_new (
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL DEFAULT 'default',
                note_path TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                title TEXT NOT NULL,
                created TEXT NOT NULL,
                updated TEXT NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                normalized_title_slug TEXT,
                normalized_path_slug TEXT,
                PRIMARY KEY (user_id, project_id, note_path)
            )
        """)

        conn.execute("""
            INSERT INTO note_metadata_new
            SELECT user_id, 'default', note_path, version, title,
                   created, updated, size_bytes, normalized_title_slug, normalized_path_slug
            FROM note_metadata
        """)

        conn.execute("DROP TABLE note_metadata")
        conn.execute("ALTER TABLE note_metadata_new RENAME TO note_metadata")

        # Recreate indexes for note_metadata
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_user ON note_metadata(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_user_project ON note_metadata(user_id, project_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_updated ON note_metadata(user_id, project_id, updated DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_title_slug ON note_metadata(user_id, project_id, normalized_title_slug)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_path_slug ON note_metadata(user_id, project_id, normalized_path_slug)")

        # 5. Migrate note_tags - recreate with project_id
        conn.execute("""
            CREATE TABLE IF NOT EXISTS note_tags_new (
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL DEFAULT 'default',
                note_path TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (user_id, project_id, note_path, tag)
            )
        """)

        conn.execute("""
            INSERT INTO note_tags_new
            SELECT user_id, 'default', note_path, tag FROM note_tags
        """)

        conn.execute("DROP TABLE note_tags")
        conn.execute("ALTER TABLE note_tags_new RENAME TO note_tags")

        # Recreate indexes for note_tags
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_user_project ON note_tags(user_id, project_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_user_tag ON note_tags(user_id, project_id, tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_user_path ON note_tags(user_id, project_id, note_path)")

        # 6. Migrate note_links - recreate with project_id
        conn.execute("""
            CREATE TABLE IF NOT EXISTS note_links_new (
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL DEFAULT 'default',
                source_path TEXT NOT NULL,
                target_path TEXT,
                link_text TEXT NOT NULL,
                is_resolved INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, project_id, source_path, link_text)
            )
        """)

        conn.execute("""
            INSERT INTO note_links_new
            SELECT user_id, 'default', source_path, target_path, link_text, is_resolved
            FROM note_links
        """)

        conn.execute("DROP TABLE note_links")
        conn.execute("ALTER TABLE note_links_new RENAME TO note_links")

        # Recreate indexes for note_links
        conn.execute("CREATE INDEX IF NOT EXISTS idx_links_user_project ON note_links(user_id, project_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_links_user_source ON note_links(user_id, project_id, source_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_links_user_target ON note_links(user_id, project_id, target_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_links_unresolved ON note_links(user_id, project_id, is_resolved)")

        # 7. Migrate index_health - recreate with project_id
        conn.execute("""
            CREATE TABLE IF NOT EXISTS index_health_new (
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL DEFAULT 'default',
                note_count INTEGER NOT NULL DEFAULT 0,
                last_full_rebuild TEXT,
                last_incremental_update TEXT,
                PRIMARY KEY (user_id, project_id)
            )
        """)

        conn.execute("""
            INSERT INTO index_health_new
            SELECT user_id, 'default', note_count, last_full_rebuild, last_incremental_update
            FROM index_health
        """)

        conn.execute("DROP TABLE index_health")
        conn.execute("ALTER TABLE index_health_new RENAME TO index_health")

        # 8. Rebuild note_fts with project_id
        conn.execute("DROP TABLE IF EXISTS note_fts")
        conn.execute("""
            CREATE VIRTUAL TABLE note_fts USING fts5(
                user_id UNINDEXED,
                project_id UNINDEXED,
                note_path UNINDEXED,
                title,
                body,
                tokenize='porter unicode61',
                prefix='2 3'
            )
        """)

        # Re-populate FTS from note_metadata (body needs separate process via indexer)
        conn.execute("""
            INSERT INTO note_fts(user_id, project_id, note_path, title, body)
            SELECT user_id, project_id, note_path, title, ''
            FROM note_metadata
        """)

        logger.info("Multi-project migration (010) completed successfully")
        return True

    except Exception as e:
        logger.error(f"Multi-project migration failed: {e}")
        raise


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
            # Check if this is an existing database needing migration
            needs_migration = _check_table_exists(conn, "note_metadata") and not _check_has_project_id_column(conn, "note_metadata")

            if needs_migration:
                # Run multi-project migration for existing databases
                with conn:
                    run_multi_project_migration(conn)
                conn.commit()
            else:
                # Fresh install - apply DDL statements
                with conn:  # Transactional apply of DDL
                    for statement in statements or DDL_STATEMENTS:
                        conn.execute(statement)

            # Run simple column migrations for existing databases (ignore errors for already-applied migrations)
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


__all__ = ["DatabaseService", "init_database", "DEFAULT_DB_PATH", "run_multi_project_migration"]
