from vlt.db import engine, Base
from vlt.core import models # Import models to register them with Base
from sqlalchemy import text

def init_db():
    """Initializes the database schema."""
    Base.metadata.create_all(bind=engine)

    # Apply additional migrations not covered by SQLAlchemy ORM
    apply_oracle_migrations()
    apply_cronban_migrations()
    apply_pipeline_migrations()


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

        # Add embedding_api_key column if it doesn't exist (idempotent)
        try:
            conn.execute(text(
                "ALTER TABLE coderag_index_jobs ADD COLUMN embedding_api_key TEXT"
            ))
        except Exception:
            pass  # Column already exists

        # ============================================================
        # Agent Sessions - Session Relay
        # ============================================================

        # Index for listing active sessions quickly
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_agent_session_status
            ON agent_sessions(status)
        """))

        # Index for looking up by project
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_agent_session_project
            ON agent_sessions(project_id)
        """))

        # Index for ordering by activity
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_agent_session_activity
            ON agent_sessions(last_activity DESC)
        """))

        # Add transcript_path column if it doesn't exist (idempotent)
        try:
            conn.execute(text(
                "ALTER TABLE agent_sessions ADD COLUMN transcript_path TEXT"
            ))
        except Exception:
            pass  # Column already exists

        conn.commit()


def apply_cronban_migrations():
    """
    Apply Cronban feature migrations (idempotent).
    Tables are created by SQLAlchemy ORM; this adds indexes only.
    """
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_cronban_entries_project_type
            ON cronban_entries(project_id, entry_type, status)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_cronban_entries_next_fire
            ON cronban_entries(next_fire_at, status)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_cronban_entries_gate
            ON cronban_entries(entry_type, status, gate_last_checked_at)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_cronban_skills_project
            ON cronban_skills(project_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_cronban_columns_project
            ON cronban_kanban_columns(project_id, col_order)
        """))

        # Graduation columns (idempotent)
        try:
            conn.execute(text(
                "ALTER TABLE cronban_kanban_columns ADD COLUMN auto_graduate INTEGER NOT NULL DEFAULT 1"
            ))
        except Exception:
            pass  # Already exists
        try:
            conn.execute(text(
                "ALTER TABLE cronban_kanban_columns ADD COLUMN graduation_column_id TEXT"
            ))
        except Exception:
            pass  # Already exists

        # Helper evaluator columns (idempotent)
        try:
            conn.execute(text(
                "ALTER TABLE agent_sessions ADD COLUMN is_cronban_helper INTEGER NOT NULL DEFAULT 0"
            ))
        except Exception:
            pass  # Already exists
        try:
            conn.execute(text(
                "ALTER TABLE cronban_entries ADD COLUMN eval_model TEXT NOT NULL DEFAULT 'haiku'"
            ))
        except Exception:
            pass  # Already exists
        try:
            conn.execute(text(
                "ALTER TABLE cronban_entries ADD COLUMN gate_eval_pending INTEGER NOT NULL DEFAULT 0"
            ))
        except Exception:
            pass  # Already exists

        # Gates library table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cronban_gates (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                name TEXT NOT NULL,
                description TEXT,
                prompt_markdown TEXT NOT NULL,
                tags_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_cronban_gates_project
            ON cronban_gates(project_id)
        """))

        # gate_id column on entries (idempotent)
        try:
            conn.execute(text(
                "ALTER TABLE cronban_entries ADD COLUMN gate_id TEXT"
            ))
        except Exception:
            pass  # Already exists
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_cronban_fire_logs_entry
            ON cronban_fire_logs(entry_id, fired_at)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_cronban_gate_logs_entry
            ON cronban_gate_logs(entry_id, checked_at)
        """))
        conn.commit()


def apply_pipeline_migrations():
    """
    Create Pipeline system tables and migrate data from legacy Cronban tables.

    New tables: pipelines, pipeline_stages, pipeline_cards, cron_triggers, webhook_listeners.
    Legacy tables (cronban_entries, cronban_kanban_columns) are left intact — their data
    is migrated to the new schema on first run (idempotent guard: skips if new tables have data).
    """
    import uuid
    import json
    from datetime import datetime

    def now() -> str:
        return datetime.utcnow().isoformat()

    with engine.connect() as conn:
        # ── Create new tables ─────────────────────────────────────────────
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipelines (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipeline_stages (
                id TEXT PRIMARY KEY,
                pipeline_id TEXT NOT NULL,
                name TEXT NOT NULL,
                stage_order INTEGER NOT NULL DEFAULT 0,
                skill_id TEXT,
                prompt_text TEXT,
                gate_id TEXT,
                eval_model TEXT NOT NULL DEFAULT 'haiku',
                auto_advance INTEGER NOT NULL DEFAULT 1,
                is_terminal INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipeline_cards (
                id TEXT PRIMARY KEY,
                pipeline_id TEXT NOT NULL,
                project_id TEXT,
                title TEXT NOT NULL,
                color TEXT,
                current_stage_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                target_session_id TEXT,
                target_cwd TEXT,
                gate_eval_pending INTEGER NOT NULL DEFAULT 0,
                gate_last_result TEXT,
                gate_last_checked_at TEXT,
                gate_consecutive_not_met INTEGER NOT NULL DEFAULT 0,
                fire_count INTEGER NOT NULL DEFAULT 0,
                last_fired_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cron_triggers (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                color TEXT,
                pipeline_id TEXT,
                skill_id TEXT,
                prompt_text TEXT,
                cron_expression TEXT,
                rrule_str TEXT,
                next_fire_at TEXT,
                timezone TEXT NOT NULL DEFAULT 'UTC',
                target_session_id TEXT,
                target_cwd TEXT,
                create_new_session INTEGER NOT NULL DEFAULT 0,
                fire_count INTEGER NOT NULL DEFAULT 0,
                last_fired_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS webhook_listeners (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                pipeline_id TEXT,
                skill_id TEXT,
                prompt_text TEXT,
                webhook_secret TEXT,
                target_session_id TEXT,
                target_cwd TEXT,
                create_new_session INTEGER NOT NULL DEFAULT 0,
                fire_count INTEGER NOT NULL DEFAULT 0,
                last_fired_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))

        # ── Indexes ───────────────────────────────────────────────────────
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_pipelines_project ON pipelines(project_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_pipeline_stages_pipeline ON pipeline_stages(pipeline_id, stage_order)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_pipeline_cards_pipeline ON pipeline_cards(pipeline_id, current_stage_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_pipeline_cards_project ON pipeline_cards(project_id, status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_cron_triggers_project ON cron_triggers(project_id, status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_cron_triggers_next_fire ON cron_triggers(next_fire_at, status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_webhook_listeners_project ON webhook_listeners(project_id)"))

        # ── Add source_type to fire logs (idempotent) ─────────────────────
        try:
            conn.execute(text("ALTER TABLE cronban_fire_logs ADD COLUMN source_type TEXT"))
        except Exception:
            pass

        # One-shot triggers (idempotent)
        try:
            conn.execute(text(
                "ALTER TABLE cron_triggers ADD COLUMN fire_once INTEGER NOT NULL DEFAULT 0"
            ))
        except Exception:
            pass  # Already exists

        # Gate settings: base_url for z.ai / self-hosted providers (idempotent)
        try:
            conn.execute(text("ALTER TABLE cronban_settings ADD COLUMN gate_base_url TEXT"))
        except Exception:
            pass  # Already exists

        # gate_id on cron_triggers (idempotent)
        try:
            conn.execute(text("ALTER TABLE cron_triggers ADD COLUMN gate_id TEXT"))
        except Exception:
            pass  # Already exists

        # skill_id / prompt_text / gate_id on pipeline_cards (idempotent)
        try:
            conn.execute(text("ALTER TABLE pipeline_cards ADD COLUMN skill_id TEXT"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE pipeline_cards ADD COLUMN prompt_text TEXT"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE pipeline_cards ADD COLUMN gate_id TEXT"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE pipeline_cards ADD COLUMN use_helper_session INTEGER NOT NULL DEFAULT 0"))
        except Exception:
            pass

        # WebhookListener connector columns (idempotent)
        for col in ["connector_name TEXT", "pattern_filter_json TEXT", "backend_user_id TEXT"]:
            try:
                conn.execute(text(f"ALTER TABLE webhook_listeners ADD COLUMN {col}"))
            except Exception:
                pass  # Already exists

        # Webhook event log table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS webhook_event_logs (
                id TEXT PRIMARY KEY,
                listener_id TEXT NOT NULL,
                connector_name TEXT,
                event_type TEXT,
                received_at TEXT NOT NULL,
                matched INTEGER NOT NULL DEFAULT 0,
                match_details TEXT,
                fields_json TEXT,
                fired INTEGER NOT NULL DEFAULT 0,
                fire_log_id TEXT
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_webhook_event_logs_listener "
            "ON webhook_event_logs(listener_id, received_at DESC)"
        ))

        conn.commit()

        # ── Data migration (only if new tables are empty) ─────────────────
        # Guard: skip if we've already migrated
        n_pipes = conn.execute(text("SELECT COUNT(*) FROM pipelines")).scalar()
        if n_pipes and n_pipes > 0:
            return  # Already migrated

        # Check if there's legacy data to migrate
        try:
            n_cols = conn.execute(text("SELECT COUNT(*) FROM cronban_kanban_columns")).scalar() or 0
            n_entries = conn.execute(text("SELECT COUNT(*) FROM cronban_entries")).scalar() or 0
        except Exception:
            return  # Legacy tables don't exist, nothing to migrate

        if n_cols == 0 and n_entries == 0:
            return  # Nothing to migrate

        # ── Migrate columns → "Default Pipeline" + stages ─────────────────
        # Group columns by project_id
        cols = conn.execute(text(
            "SELECT id, project_id, name, col_order, is_terminal, auto_graduate, "
            "graduation_column_id, created_at FROM cronban_kanban_columns ORDER BY project_id, col_order"
        )).fetchall()

        # Build project → pipeline_id map
        project_pipeline: dict = {}
        col_to_stage: dict = {}  # old column_id → new stage_id

        for col in cols:
            col_id, proj_id, col_name, col_order, is_terminal, auto_advance, grad_col_id, col_created = col
            proj_key = proj_id or "__global__"

            if proj_key not in project_pipeline:
                pipe_id = str(uuid.uuid4())
                conn.execute(text(
                    "INSERT INTO pipelines(id, project_id, name, description, created_at, updated_at) "
                    "VALUES(:id, :proj, :name, :desc, :ca, :ua)"
                ), {"id": pipe_id, "proj": proj_id, "name": "Default Pipeline",
                    "desc": "Migrated from legacy Kanban columns", "ca": col_created or now(), "ua": now()})
                project_pipeline[proj_key] = pipe_id

            pipeline_id = project_pipeline[proj_key]
            stage_id = str(uuid.uuid4())
            col_to_stage[col_id] = stage_id

            conn.execute(text(
                "INSERT INTO pipeline_stages(id, pipeline_id, name, stage_order, "
                "gate_id, eval_model, auto_advance, is_terminal, created_at, updated_at) "
                "VALUES(:id, :pip, :name, :ord, NULL, 'haiku', :adv, :term, :ca, :ua)"
            ), {"id": stage_id, "pip": pipeline_id, "name": col_name, "ord": col_order,
                "adv": 1 if auto_advance else 0, "term": 1 if is_terminal else 0,
                "ca": col_created or now(), "ua": now()})

        # ── Migrate entries ────────────────────────────────────────────────
        entries = conn.execute(text(
            "SELECT id, project_id, title, entry_type, status, color, skill_id, prompt_text, "
            "gate_id, eval_text, eval_model, gate_eval_pending, target_session_id, target_cwd, "
            "create_new_session, cron_expression, rrule_str, next_fire_at, timezone, "
            "kanban_column_id, gate_check_interval_minutes, gate_last_checked_at, "
            "gate_last_result, gate_consecutive_not_met, webhook_secret, fire_count, "
            "last_fired_at, created_at, updated_at FROM cronban_entries"
        )).fetchall()

        for e in entries:
            (eid, proj_id, title, etype, estatus, color, skill_id, prompt_text,
             gate_id, eval_text, eval_model, gate_eval_pending, target_sid, target_cwd,
             create_new, cron_expr, rrule, next_fire, tz, col_id, gate_interval,
             gate_last_checked, gate_last_result, gate_consec, webhook_secret,
             fire_count, last_fired, created_at, updated_at) = e

            ts = created_at or now()
            ua = updated_at or now()

            if etype == "cron":
                conn.execute(text(
                    "INSERT INTO cron_triggers(id, project_id, title, status, color, "
                    "skill_id, prompt_text, cron_expression, rrule_str, next_fire_at, timezone, "
                    "target_session_id, target_cwd, create_new_session, fire_count, last_fired_at, "
                    "created_at, updated_at) VALUES("
                    ":id,:proj,:title,:status,:color,:skill,:prompt,:cron,:rrule,:next,:tz,"
                    ":tsid,:tcwd,:create,:fires,:last,:ca,:ua)"
                ), {"id": eid, "proj": proj_id, "title": title, "status": estatus,
                    "color": color, "skill": skill_id, "prompt": prompt_text,
                    "cron": cron_expr, "rrule": rrule, "next": next_fire, "tz": tz or "UTC",
                    "tsid": target_sid, "tcwd": target_cwd, "create": 1 if create_new else 0,
                    "fires": fire_count or 0, "last": last_fired, "ca": ts, "ua": ua})

            elif etype == "gate":
                # Find the pipeline + stage for this card's column
                stage_id = col_to_stage.get(col_id) if col_id else None
                if not stage_id:
                    # No matching column — place in first available pipeline
                    proj_key = proj_id or "__global__"
                    pipeline_id = project_pipeline.get(proj_key)
                    if pipeline_id:
                        first_stage = conn.execute(text(
                            "SELECT id FROM pipeline_stages WHERE pipeline_id=:pid ORDER BY stage_order LIMIT 1"
                        ), {"pid": pipeline_id}).scalar()
                        stage_id = first_stage
                    if not stage_id:
                        continue  # Can't place this card

                # Find pipeline from stage
                pipeline_id = conn.execute(text(
                    "SELECT pipeline_id FROM pipeline_stages WHERE id=:sid"
                ), {"sid": stage_id}).scalar()
                if not pipeline_id:
                    continue

                # If this entry had a gate_id on the stage, update the stage
                if gate_id:
                    conn.execute(text(
                        "UPDATE pipeline_stages SET gate_id=:gid, eval_model=:em WHERE id=:sid "
                        "AND gate_id IS NULL"
                    ), {"gid": gate_id, "em": eval_model or "haiku", "sid": stage_id})
                elif eval_text:
                    # Legacy inline eval — store as gate prompt in a new gate entry
                    # Just skip for now; eval_text is not migrated to gate library
                    pass

                conn.execute(text(
                    "INSERT INTO pipeline_cards(id, pipeline_id, project_id, title, color, "
                    "current_stage_id, status, target_session_id, target_cwd, gate_eval_pending, "
                    "gate_last_result, gate_last_checked_at, gate_consecutive_not_met, "
                    "fire_count, last_fired_at, created_at, updated_at) VALUES("
                    ":id,:pip,:proj,:title,:color,:stage,:status,:tsid,:tcwd,:gep,"
                    ":glr,:glc,:gcnm,:fires,:last,:ca,:ua)"
                ), {"id": eid, "pip": pipeline_id, "proj": proj_id, "title": title,
                    "color": color, "stage": stage_id, "status": estatus,
                    "tsid": target_sid, "tcwd": target_cwd,
                    "gep": 1 if gate_eval_pending else 0,
                    "glr": gate_last_result, "glc": gate_last_checked,
                    "gcnm": gate_consec or 0, "fires": fire_count or 0,
                    "last": last_fired, "ca": ts, "ua": ua})

            elif etype == "webhook":
                conn.execute(text(
                    "INSERT INTO webhook_listeners(id, project_id, name, status, "
                    "skill_id, prompt_text, webhook_secret, target_session_id, target_cwd, "
                    "create_new_session, fire_count, last_fired_at, created_at, updated_at) "
                    "VALUES(:id,:proj,:name,:status,:skill,:prompt,:secret,:tsid,:tcwd,"
                    ":create,:fires,:last,:ca,:ua)"
                ), {"id": eid, "proj": proj_id, "name": title, "status": estatus,
                    "skill": skill_id, "prompt": prompt_text, "secret": webhook_secret,
                    "tsid": target_sid, "tcwd": target_cwd,
                    "create": 1 if create_new else 0,
                    "fires": fire_count or 0, "last": last_fired, "ca": ts, "ua": ua})

        conn.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialized with Oracle migrations.")
