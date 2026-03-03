from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import String, ForeignKey, Integer, Text, LargeBinary, DateTime, JSON, Table, Column, Float, Boolean, Enum as SQLAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from vlt.db import Base
import enum

# Association table for Node-Tag
node_tags = Table(
    "node_tags",
    Base.metadata,
    Column("node_id", String, ForeignKey("nodes.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)

class Tag(Base):
    __tablename__ = "tags"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    
    nodes: Mapped[List["Node"]] = relationship(secondary=node_tags, back_populates="tags")

class Reference(Base):
    __tablename__ = "references"
    
    id: Mapped[str] = mapped_column(String, primary_key=True) # UUID
    source_node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id"))
    target_thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id"))
    note: Mapped[str] = mapped_column(String) # Relationship type e.g. "relates_to"
    
    source_node: Mapped["Node"] = relationship(back_populates="outbound_refs")
    target_thread: Mapped["Thread"] = relationship()

class Project(Base):
    __tablename__ = "projects"
    
    id: Mapped[str] = mapped_column(String, primary_key=True) # Slug
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    threads: Mapped[List["Thread"]] = relationship(back_populates="project")

class Thread(Base):
    __tablename__ = "threads"
    
    id: Mapped[str] = mapped_column(String, primary_key=True) # Slug
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    status: Mapped[str] = mapped_column(String, default="active") # Active, Archived, Blocked
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    project: Mapped["Project"] = relationship(back_populates="threads")
    nodes: Mapped[List["Node"]] = relationship(back_populates="thread")

class Node(Base):
    __tablename__ = "nodes"
    
    id: Mapped[str] = mapped_column(String, primary_key=True) # UUID
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id"))
    sequence_id: Mapped[int] = mapped_column(Integer) # Ordered within thread
    content: Mapped[str] = mapped_column(Text)
    author: Mapped[str] = mapped_column(String, default="user")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    prev_node_id: Mapped[Optional[str]] = mapped_column(ForeignKey("nodes.id"), nullable=True)

    thread: Mapped["Thread"] = relationship(back_populates="nodes")
    prev_node: Mapped[Optional["Node"]] = relationship(remote_side=[id])
    
    tags: Mapped[List["Tag"]] = relationship(secondary=node_tags, back_populates="nodes")
    outbound_refs: Mapped[List["Reference"]] = relationship(back_populates="source_node")

class State(Base):
    __tablename__ = "states"

    id: Mapped[str] = mapped_column(String, primary_key=True) # UUID
    target_id: Mapped[str] = mapped_column(String) # Thread.id or Project.id
    target_type: Mapped[str] = mapped_column(String) # "thread" or "project"
    summary: Mapped[str] = mapped_column(Text)
    head_node_id: Mapped[Optional[str]] = mapped_column(ForeignKey("nodes.id"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    meta: Mapped[dict] = mapped_column(JSON, default={})

    head_node: Mapped[Optional["Node"]] = relationship()


# ============================================================================
# Oracle Feature - Enums
# ============================================================================

class ChunkType(enum.Enum):
    """Type of code chunk."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"


class NodeType(enum.Enum):
    """Type of code graph node."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"


class EdgeType(enum.Enum):
    """Type of code graph edge."""
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    USES = "uses"
    DECORATES = "decorates"


class ConversationStatus(enum.Enum):
    """Status of oracle conversation."""
    ACTIVE = "active"
    COMPRESSED = "compressed"
    CLOSED = "closed"


class ChangeType(enum.Enum):
    """Type of file change."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


class QueueStatus(enum.Enum):
    """Status of queued index update."""
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class JobStatus(enum.Enum):
    """Status of CodeRAG indexing job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# Oracle Feature - Models
# ============================================================================

class CodeChunk(Base):
    """Context-enriched semantic unit from source code."""
    __tablename__ = "code_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    file_path: Mapped[str] = mapped_column(String(512))
    file_hash: Mapped[str] = mapped_column(String(32))  # MD5 hash
    chunk_type: Mapped[ChunkType] = mapped_column(SQLAEnum(ChunkType))
    name: Mapped[str] = mapped_column(String(256))
    qualified_name: Mapped[str] = mapped_column(String(512))
    language: Mapped[str] = mapped_column(String)
    lineno: Mapped[int] = mapped_column(Integer)
    end_lineno: Mapped[int] = mapped_column(Integer)
    imports: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    class_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    signature: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    decorators: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    body: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    embedding_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    project: Mapped["Project"] = relationship()


class CodeNode(Base):
    """Node in code dependency graph."""
    __tablename__ = "code_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # Qualified name
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    file_path: Mapped[str] = mapped_column(String(512))
    node_type: Mapped[NodeType] = mapped_column(SQLAEnum(NodeType))
    name: Mapped[str] = mapped_column(String(256))
    signature: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    lineno: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    centrality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    project: Mapped["Project"] = relationship()
    outgoing_edges: Mapped[List["CodeEdge"]] = relationship(foreign_keys="CodeEdge.source_id", back_populates="source_node")
    incoming_edges: Mapped[List["CodeEdge"]] = relationship(foreign_keys="CodeEdge.target_id", back_populates="target_node")


class CodeEdge(Base):
    """Directed edge in code dependency graph."""
    __tablename__ = "code_edges"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    source_id: Mapped[str] = mapped_column(ForeignKey("code_nodes.id"))
    target_id: Mapped[str] = mapped_column(ForeignKey("code_nodes.id"))
    edge_type: Mapped[EdgeType] = mapped_column(SQLAEnum(EdgeType))
    lineno: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    count: Mapped[int] = mapped_column(Integer, default=1)

    project: Mapped["Project"] = relationship()
    source_node: Mapped["CodeNode"] = relationship(foreign_keys=[source_id], back_populates="outgoing_edges")
    target_node: Mapped["CodeNode"] = relationship(foreign_keys=[target_id], back_populates="incoming_edges")


class SymbolDefinition(Base):
    """ctags symbol definition."""
    __tablename__ = "symbol_definitions"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    name: Mapped[str] = mapped_column(String(256))
    file_path: Mapped[str] = mapped_column(String(512))
    lineno: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String)
    scope: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    signature: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    language: Mapped[str] = mapped_column(String)

    project: Mapped["Project"] = relationship()


class RepoMap(Base):
    """Cached repository structure map."""
    __tablename__ = "repo_maps"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    scope: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    map_text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    max_tokens: Mapped[int] = mapped_column(Integer)
    files_included: Mapped[int] = mapped_column(Integer)
    symbols_included: Mapped[int] = mapped_column(Integer)
    symbols_total: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    project: Mapped["Project"] = relationship()


class OracleSession(Base):
    """Logged oracle conversation."""
    __tablename__ = "oracle_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id"))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    query_type: Mapped[str] = mapped_column(String)
    sources_json: Mapped[str] = mapped_column(Text)
    retrieval_traces_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_used: Mapped[str] = mapped_column(String)
    tokens_used: Mapped[int] = mapped_column(Integer)
    cost_cents: Mapped[float] = mapped_column(Float)
    duration_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    project: Mapped["Project"] = relationship()
    thread: Mapped["Thread"] = relationship()


class OracleConversation(Base):
    """Shared context across MCP tools."""
    __tablename__ = "oracle_conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    user_id: Mapped[str] = mapped_column(String)
    session_start: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_activity: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    token_budget: Mapped[int] = mapped_column(Integer, default=16000)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    compressed_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recent_exchanges_json: Mapped[str] = mapped_column(Text, default='[]')
    mentioned_symbols: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mentioned_files: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[ConversationStatus] = mapped_column(SQLAEnum(ConversationStatus), default=ConversationStatus.ACTIVE)
    compression_count: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    project: Mapped["Project"] = relationship()


class IndexDeltaQueue(Base):
    """Pending file changes for indexing."""
    __tablename__ = "index_delta_queue"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    file_path: Mapped[str] = mapped_column(String(512))
    change_type: Mapped[ChangeType] = mapped_column(SQLAEnum(ChangeType))
    old_hash: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    new_hash: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    lines_changed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    queued_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    priority: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[QueueStatus] = mapped_column(SQLAEnum(QueueStatus), default=QueueStatus.PENDING)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    project: Mapped["Project"] = relationship()


class ThreadSummaryCache(Base):
    """Cached thread summaries for lazy LLM."""
    __tablename__ = "thread_summary_cache"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id"), unique=True)
    summary: Mapped[str] = mapped_column(Text)
    last_node_id: Mapped[str] = mapped_column(String)
    node_count: Mapped[int] = mapped_column(Integer)
    model_used: Mapped[str] = mapped_column(String)
    tokens_used: Mapped[int] = mapped_column(Integer)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    thread: Mapped["Thread"] = relationship()


# ============================================================================
# CodeRAG Project Integration - Index Job Model
# ============================================================================

class CodeRAGIndexJob(Base):
    """Background CodeRAG indexing job with progress tracking.

    Tracks the state and progress of background code indexing jobs
    for CodeRAG project integration. Each project can have multiple
    jobs over time, but only one running job at a time.
    """
    __tablename__ = "coderag_index_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    status: Mapped[JobStatus] = mapped_column(SQLAEnum(JobStatus), default=JobStatus.PENDING)
    target_path: Mapped[str] = mapped_column(String(512))
    force: Mapped[bool] = mapped_column(Boolean, default=False)
    priority: Mapped[int] = mapped_column(Integer, default=0)

    # Progress tracking
    files_total: Mapped[int] = mapped_column(Integer, default=0)
    files_processed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, default=0)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # API key for embedding generation (stored with job so daemon can use it)
    embedding_api_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Relationship
    project: Mapped["Project"] = relationship()


# ============================================================================
# Session Relay - Agent Session Model
# ============================================================================

class AgentSession(Base):
    """Live Claude Code agent session tracked by the daemon.

    Sessions are discovered via three sources:
    - "relay": The `vlt relay` command wraps claude in a PTY and registers
      the session directly (full terminal stream relay).
    - "hook": Claude Code lifecycle hooks POST to /api/hooks as events fire
      (status updates only, no PTY stream).
    - "discovery": The daemon's background ~/.claude/history.jsonl watcher
      finds sessions that were never explicitly registered.
    """
    __tablename__ = "agent_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # Claude session_id UUID
    project_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # derived from cwd basename
    cwd: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="idle")  # idle/thinking/executing/dead
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ctx_pct: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    args: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    bypass_perms: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String, default="relay")  # relay/discovery/hook
    transcript_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # path from hook payload
    is_cronban_helper: Mapped[bool] = mapped_column(Boolean, default=False)  # project-level helper evaluator session
    created_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    last_activity: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


# ============================================================================
# Cronban — Cron + Kanban + Calendar job scheduling
# ============================================================================

class CronbanEntryType(enum.Enum):
    CRON    = "cron"     # DEPRECATED — use CronTrigger
    GATE    = "gate"     # DEPRECATED — use PipelineCard
    WEBHOOK = "webhook"  # DEPRECATED — use WebhookListener


class CronbanEntryStatus(enum.Enum):
    ACTIVE    = "active"
    PAUSED    = "paused"
    COMPLETED = "completed"
    FAILED    = "failed"


class CronbanSkill(Base):
    """Reusable prompt template library — the 'skill' a Claude session is given."""
    __tablename__ = "cronban_skills"

    id: Mapped[str] = mapped_column(String, primary_key=True)           # UUID
    project_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # None = global
    name: Mapped[str] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_markdown: Mapped[str] = mapped_column(Text)                  # What Claude sees
    tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string[]
    created_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class CronbanGate(Base):
    """
    Reusable gate definition — the eval criterion a helper Claude runs after the
    working agent's turn ends. Gates are like skills but for the verifier side.

    The gate's prompt_markdown is what the helper Claude receives, alongside:
      - The original task that was given to the working agent
      - The working agent's most recent output

    The helper evaluates and responds with GATE_RESULT: PASS or GATE_RESULT: FAIL.
    """
    __tablename__ = "cronban_gates"

    id: Mapped[str] = mapped_column(String, primary_key=True)           # UUID
    project_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # None = global
    name: Mapped[str] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_markdown: Mapped[str] = mapped_column(Text)                  # Helper sees this
    tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string[]
    created_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class KanbanColumn(Base):
    """Kanban board column definition, per-project."""
    __tablename__ = "cronban_kanban_columns"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    col_order: Mapped[int] = mapped_column(Integer, default=0)
    color: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # hex
    is_terminal: Mapped[bool] = mapped_column(Boolean, default=False)   # "Done"-type
    # Graduation: when a gate check passes, auto-move the card to graduation_column_id
    # (or the next column by col_order if graduation_column_id is NULL)
    auto_graduate: Mapped[bool] = mapped_column(Boolean, default=True)
    graduation_column_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class CronbanEntry(Base):
    """
    A cronban entry — one of: calendar event (CRON), kanban card (GATE), or webhook trigger.

    Three-part structure:
      1. Schedule (target_time / cron / rrule)
      2. Skill    — prompt Claude sees (prompt_text or skill_id)
      3. Eval     — hidden criterion Claude NEVER sees; used by verifier only
    """
    __tablename__ = "cronban_entries"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    title: Mapped[str] = mapped_column(String)
    entry_type: Mapped[str] = mapped_column(String, default="cron")   # CronbanEntryType value
    status: Mapped[str] = mapped_column(String, default="active")     # CronbanEntryStatus value
    color: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # ── SKILL (what Claude is told to do) ──────────────────────────────────
    skill_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)   # FK cronban_skills.id
    prompt_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Inline fallback

    # ── GATE (helper Claude runs this after working agent's turn ends) ────
    gate_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # FK cronban_gates.id
    # Legacy inline eval_text — kept for backwards compat; gate_id takes priority
    eval_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    eval_model: Mapped[str] = mapped_column(String, default="haiku")  # haiku|sonnet|opus — helper model
    gate_eval_pending: Mapped[bool] = mapped_column(Boolean, default=False)  # True while helper is evaluating

    # ── WHERE TO FIRE ──────────────────────────────────────────────────────
    target_session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    target_cwd: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    create_new_session: Mapped[bool] = mapped_column(Boolean, default=False)

    # ── SCHEDULE (CRON type) ───────────────────────────────────────────────
    cron_expression: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # "0 9 * * 1-5"
    rrule_str: Mapped[Optional[str]] = mapped_column(Text, nullable=True)          # RFC 5545 RRULE
    next_fire_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)     # ISO UTC
    timezone: Mapped[str] = mapped_column(String, default="UTC")

    # ── GATE / KANBAN (GATE type) ──────────────────────────────────────────
    kanban_column_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    kanban_order: Mapped[int] = mapped_column(Integer, default=0)
    gate_check_interval_minutes: Mapped[int] = mapped_column(Integer, default=30)
    gate_last_checked_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    gate_last_result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON {met,reasoning}
    gate_consecutive_not_met: Mapped[int] = mapped_column(Integer, default=0)
    gate_question_injected_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # ── WEBHOOK type ───────────────────────────────────────────────────────
    webhook_secret: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # ── STATS ──────────────────────────────────────────────────────────────
    fire_count: Mapped[int] = mapped_column(Integer, default=0)
    last_fired_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class CronbanFireLog(Base):
    """Audit log of every time a cronban entry fired."""
    __tablename__ = "cronban_fire_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    entry_id: Mapped[str] = mapped_column(String)
    fired_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    trigger_type: Mapped[str] = mapped_column(String)    # "cron"|"gate"|"webhook"|"manual"
    target_session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    prompt_sent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result: Mapped[str] = mapped_column(String, default="success")  # "success"|"failed"|"session_dead"
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class CronbanGateLog(Base):
    """Audit log of each gate evaluation run (verifier model call)."""
    __tablename__ = "cronban_gate_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    entry_id: Mapped[str] = mapped_column(String)
    checked_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    gate_met: Mapped[bool] = mapped_column(Boolean)
    reasoning: Mapped[str] = mapped_column(Text)
    model_used: Mapped[str] = mapped_column(String)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)


class CronbanSettings(Base):
    """Per-profile gate verifier model settings."""
    __tablename__ = "cronban_settings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default="default")
    gate_provider: Mapped[str] = mapped_column(String, default="claude_code")  # "claude_code"|"zai"|"openrouter"|"gemini"
    gate_model: Mapped[str] = mapped_column(String, default="sonnet")
    gate_api_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # encrypted at rest
    gate_base_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # for z.ai / self-hosted
    updated_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


# ============================================================================
# Pipeline system — replaces KanbanColumn + CronbanEntry (gate/cron/webhook)
# ============================================================================

class Pipeline(Base):
    """
    A workflow definition: an ordered sequence of stages a PipelineCard
    progresses through as work completes and gates are satisfied.

    Cron triggers and webhook listeners create PipelineCard instances;
    the Kanban board displays those cards grouped by their current stage.
    """
    __tablename__ = "pipelines"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class PipelineStage(Base):
    """
    One stage within a Pipeline.

    When a PipelineCard enters this stage:
      - If skill_id / prompt_text is set, that prompt is fired at the target session.
      - Helper Claude evaluates the agent output against gate_id after each turn.
      - If gate passes and auto_advance=True, the card moves to the next stage.
      - is_terminal stages mark completion.
    """
    __tablename__ = "pipeline_stages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    pipeline_id: Mapped[str] = mapped_column(String)                         # FK → pipelines.id
    name: Mapped[str] = mapped_column(String)
    stage_order: Mapped[int] = mapped_column(Integer, default=0)
    # What Claude does when a card enters this stage
    skill_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)   # FK → cronban_skills.id
    prompt_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Inline fallback
    # Gate: evaluated by helper Claude after each agent turn ends
    gate_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)    # FK → cronban_gates.id
    eval_model: Mapped[str] = mapped_column(String, default="haiku")         # haiku|sonnet|opus
    auto_advance: Mapped[bool] = mapped_column(Boolean, default=True)        # Auto-move on gate pass
    is_terminal: Mapped[bool] = mapped_column(Boolean, default=False)        # Final stage
    created_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class PipelineCard(Base):
    """
    An instance of work moving through a Pipeline, one stage at a time.
    Created by a CronTrigger, WebhookListener, or manually.

    The card tracks which stage it's currently in, the gate evaluation state
    for that stage, and the target session where the agent runs.
    """
    __tablename__ = "pipeline_cards"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    pipeline_id: Mapped[str] = mapped_column(String)                         # FK → pipelines.id
    project_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    title: Mapped[str] = mapped_column(String)
    color: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    current_stage_id: Mapped[str] = mapped_column(String)                    # FK → pipeline_stages.id
    status: Mapped[str] = mapped_column(String, default="active")            # active/completed/paused/failed
    # Agent target
    target_session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    target_cwd: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Gate evaluation state (for current stage)
    gate_eval_pending: Mapped[bool] = mapped_column(Boolean, default=False)
    gate_last_result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)    # JSON {met, reasoning}
    gate_last_checked_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    gate_consecutive_not_met: Mapped[int] = mapped_column(Integer, default=0)
    # Stats
    fire_count: Mapped[int] = mapped_column(Integer, default=0)
    last_fired_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class CronTrigger(Base):
    """
    Time-based trigger that fires on a cron schedule.

    On fire, either:
      - Creates a new PipelineCard in pipeline_id (placed at the first stage), OR
      - Fires skill_id / prompt_text directly at a session (standalone, no pipeline).
    """
    __tablename__ = "cron_triggers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    title: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="active")            # active/paused
    color: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # What to create/fire
    pipeline_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)   # FK → pipelines.id
    skill_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)      # FK → cronban_skills.id
    prompt_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)     # Inline fallback
    # Schedule
    cron_expression: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    rrule_str: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    next_fire_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timezone: Mapped[str] = mapped_column(String, default="UTC")
    # Target session (used for standalone skill fires; pipeline cards inherit from card)
    target_session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    target_cwd: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    create_new_session: Mapped[bool] = mapped_column(Boolean, default=False)
    fire_once: Mapped[bool] = mapped_column(Boolean, default=False)   # one-shot: completed after first fire
    # Stats
    fire_count: Mapped[int] = mapped_column(Integer, default=0)
    last_fired_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class WebhookListener(Base):
    """
    External HTTP trigger — fires when a POST arrives at /api/cronban/webhooks/{id}/fire.
    Authenticated via HMAC-SHA256 if webhook_secret is set.

    The caller may append an extra message to the skill prompt via the POST body.
    On fire, either:
      - Creates a new PipelineCard in pipeline_id, OR
      - Fires skill_id / prompt_text directly at a session.
    """
    __tablename__ = "webhook_listeners"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="active")            # active/paused
    # What to create/fire
    pipeline_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)   # FK → pipelines.id
    skill_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)      # FK → cronban_skills.id
    prompt_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)     # Inline fallback
    # Auth
    webhook_secret: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Target session (for standalone skill fires)
    target_session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    target_cwd: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    create_new_session: Mapped[bool] = mapped_column(Boolean, default=False)
    # Stats
    fire_count: Mapped[int] = mapped_column(Integer, default=0)
    last_fired_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())