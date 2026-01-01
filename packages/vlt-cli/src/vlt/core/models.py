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

    # Relationship
    project: Mapped["Project"] = relationship()