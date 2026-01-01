from vlt.db import engine, Base
from vlt.core import models # Import models to register them with Base
from sqlalchemy import text

def init_db():
    """Initializes the database schema."""
    Base.metadata.create_all(bind=engine)

    # Apply additional migrations not covered by SQLAlchemy ORM
    apply_oracle_migrations()


def apply_oracle_migrations():
    """
    Apply Oracle feature migrations:
    - T013: Create FTS5 virtual table for BM25 search
    - T014: Add additional indexes for Oracle tables

    All operations are idempotent (safe to run multiple times).
    """
    with engine.connect() as conn:
        # T014 - Create FTS5 virtual table for BM25 full-text search
        # This is a standalone FTS5 table that will be manually synchronized with code_chunks
        # We use standalone instead of contentless because code_chunks uses VARCHAR primary key
        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS code_chunk_fts USING fts5(
                chunk_id UNINDEXED,
                name,
                qualified_name,
                signature,
                docstring,
                body,
                tokenize='porter unicode61'
            )
        """))

        # Create indexes for code_chunks (if not exists)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_code_chunk_project_id
            ON code_chunks(project_id)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_code_chunk_file_path
            ON code_chunks(project_id, file_path)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_code_chunk_qualified_name
            ON code_chunks(qualified_name)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_code_chunk_name
            ON code_chunks(name)
        """))

        # Create indexes for code_nodes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_code_node_project_id
            ON code_nodes(project_id)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_code_node_file_path
            ON code_nodes(file_path)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_code_node_name
            ON code_nodes(name)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_code_node_centrality
            ON code_nodes(project_id, centrality_score DESC)
        """))

        # Create indexes for code_edges
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_code_edge_source
            ON code_edges(source_id)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_code_edge_target
            ON code_edges(target_id)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_code_edge_type
            ON code_edges(project_id, edge_type)
        """))

        # Create indexes for symbol_definitions
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_symbol_def_name
            ON symbol_definitions(project_id, name)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_symbol_def_file
            ON symbol_definitions(file_path)
        """))

        # Create indexes for repo_maps
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_repo_map_project
            ON repo_maps(project_id, scope)
        """))

        # Create indexes for oracle_sessions
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_oracle_session_project
            ON oracle_sessions(project_id, created_at DESC)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_oracle_session_thread
            ON oracle_sessions(thread_id)
        """))

        # Create indexes for oracle_conversations
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_oracle_conv_project_user
            ON oracle_conversations(project_id, user_id, status)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_oracle_conv_activity
            ON oracle_conversations(last_activity DESC)
        """))

        # Partial index for active conversations with expiry
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_oracle_conv_expires
            ON oracle_conversations(expires_at)
            WHERE status = 'active'
        """))

        # Create indexes for index_delta_queue
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_delta_queue_project_status
            ON index_delta_queue(project_id, status)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_delta_queue_priority
            ON index_delta_queue(project_id, priority DESC, queued_at ASC)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_delta_queue_file
            ON index_delta_queue(project_id, file_path)
        """))

        # Create unique index for thread_summary_cache
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS ix_thread_summary_thread
            ON thread_summary_cache(thread_id)
        """))

        # ============================================================
        # CodeRAG Index Jobs - T005
        # ============================================================

        # Index for finding jobs by project and status
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_coderag_job_project_status
            ON coderag_index_jobs(project_id, status)
        """))

        # Index for finding pending jobs by priority (for daemon job picker)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_coderag_job_pending_priority
            ON coderag_index_jobs(status, priority DESC, created_at ASC)
        """))

        # Index for finding active job for a project
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_coderag_job_active
            ON coderag_index_jobs(project_id)
            WHERE status IN ('pending', 'running')
        """))

        conn.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialized with Oracle migrations.")
