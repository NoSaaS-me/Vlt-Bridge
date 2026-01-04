"""Pydantic models for Oracle context persistence (009-oracle-agent).

This module provides two context models:
1. Tree-based context (ContextNode, ContextTree) - branching conversation history
2. Legacy flat context (OracleContext) - deprecated but kept for migration
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContextStatus(str, Enum):
    """Status of an Oracle context session."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class ExchangeRole(str, Enum):
    """Role of a participant in an Oracle exchange."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class ToolCallStatus(str, Enum):
    """Status of a tool call execution."""

    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"


class ToolCategory(str, Enum):
    """Category of Oracle tools."""

    VAULT = "vault"
    THREAD = "thread"
    CODE = "code"
    WEB = "web"
    META = "meta"


class ToolCall(BaseModel):
    """A single tool invocation within an exchange."""

    id: str = Field(..., description="OpenRouter tool_call_id")
    name: str = Field(..., description="Tool name (e.g., 'search_code')")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    result: Optional[str] = Field(None, description="Tool execution result")
    status: ToolCallStatus = Field(
        default=ToolCallStatus.PENDING, description="Execution status"
    )
    duration_ms: Optional[int] = Field(None, description="Execution time in milliseconds")


# =============================================================================
# Tree-Based Context Models
# =============================================================================


class ContextNode(BaseModel):
    """A single node in the context tree representing one conversation exchange.

    Each node contains a user question and the Oracle's answer, along with
    metadata for tree navigation, pruning, and organization.
    """

    id: str = Field(..., description="UUID primary key")
    root_id: str = Field(..., description="ID of the root node for this tree")
    parent_id: Optional[str] = Field(
        None, description="Parent node ID (None for root nodes)"
    )
    user_id: str = Field(..., description="User identifier")
    project_id: str = Field(..., description="Project scope")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this node was created"
    )

    # Content
    question: str = Field(..., description="User's question")
    answer: str = Field(..., description="Oracle's answer")
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Tools invoked for this exchange"
    )
    tokens_used: int = Field(0, description="Total tokens consumed for this exchange")
    model_used: Optional[str] = Field(None, description="Model used for this exchange")

    # System messages (notifications from ANS)
    system_messages: List[str] = Field(
        default_factory=list, description="System messages/notifications for this exchange"
    )

    # Tree metadata
    label: Optional[str] = Field(
        None, description="User-assigned label for easy reference"
    )
    is_checkpoint: bool = Field(
        False, description="Protected from pruning when True"
    )
    is_root: bool = Field(
        False, description="True if this is a root node"
    )
    child_count: int = Field(
        0, description="Number of child nodes (for pruning decisions)"
    )


class ContextTree(BaseModel):
    """Metadata for a conversation tree.

    Each tree has one root node and tracks the current HEAD position
    (where new messages will be added). Users can have multiple trees
    per project, allowing for parallel conversation threads.
    """

    root_id: str = Field(..., description="ID of the root node for this tree")
    user_id: str = Field(..., description="User identifier")
    project_id: str = Field(..., description="Project scope")
    current_node_id: str = Field(
        ..., description="HEAD - active node where next message will be added"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this tree was created"
    )
    last_activity: datetime = Field(
        default_factory=datetime.utcnow, description="Last interaction timestamp"
    )
    node_count: int = Field(1, description="Current number of nodes in this tree")
    max_nodes: int = Field(
        30, description="Maximum nodes before pruning (from user settings)"
    )
    label: Optional[str] = Field(None, description="User-assigned tree name")


class ContextTreeResponse(BaseModel):
    """API response for context tree retrieval.

    Contains all trees for a user/project combination along with
    the full node structure of the active tree.
    """

    trees: List[ContextTree] = Field(
        default_factory=list, description="All trees for user/project"
    )
    active_tree: Optional[ContextTree] = Field(
        None, description="Currently active tree"
    )
    nodes: Dict[str, ContextNode] = Field(
        default_factory=dict, description="All nodes in active tree, keyed by ID"
    )
    path_to_head: List[str] = Field(
        default_factory=list,
        description="Node IDs from root to current HEAD (for context building)"
    )


class ContextNodeCreateRequest(BaseModel):
    """Request to create a new context node."""

    parent_id: Optional[str] = Field(
        None, description="Parent node ID (None to create new root)"
    )
    question: str = Field(..., description="User's question")
    answer: str = Field(..., description="Oracle's answer")
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Tools invoked"
    )
    tokens_used: int = Field(0, description="Tokens consumed")
    model_used: Optional[str] = Field(None, description="Model used")
    label: Optional[str] = Field(None, description="Optional label")
    is_checkpoint: bool = Field(False, description="Mark as checkpoint")


class ContextNodeUpdateRequest(BaseModel):
    """Request to update an existing context node."""

    label: Optional[str] = Field(None, description="New label (None to keep current)")
    is_checkpoint: Optional[bool] = Field(
        None, description="New checkpoint status (None to keep current)"
    )


class ContextTreeCreateRequest(BaseModel):
    """Request to create a new context tree (with initial root node)."""

    question: str = Field(..., description="Initial question for root node")
    answer: str = Field(..., description="Initial answer for root node")
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Tools invoked"
    )
    tokens_used: int = Field(0, description="Tokens consumed")
    model_used: Optional[str] = Field(None, description="Model used")
    label: Optional[str] = Field(None, description="Tree label")


class ContextTreeUpdateRequest(BaseModel):
    """Request to update tree metadata."""

    label: Optional[str] = Field(None, description="New tree label")
    max_nodes: Optional[int] = Field(
        None, ge=5, le=100, description="New max nodes limit"
    )
    current_node_id: Optional[str] = Field(
        None, description="Move HEAD to different node"
    )


# =============================================================================
# Legacy Flat Context Models (Deprecated)
# =============================================================================


class OracleExchange(BaseModel):
    """A single turn in the Oracle conversation (question + tool calls + answer)."""

    id: str = Field(..., description="UUID for this exchange")
    role: ExchangeRole = Field(..., description="Role of the participant")
    content: str = Field(..., description="Message content")
    tool_calls: Optional[List[ToolCall]] = Field(
        None, description="Tools invoked in this exchange"
    )
    tool_call_id: Optional[str] = Field(
        None, description="For tool results, the ID of the originating call"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When this exchange occurred"
    )
    token_count: int = Field(0, description="Tokens in this exchange")
    key_insight: Optional[str] = Field(
        None, description="Summary for compression (important exchanges only)"
    )
    mentioned_symbols: List[str] = Field(
        default_factory=list, description="Code symbols referenced in this turn"
    )
    mentioned_files: List[str] = Field(
        default_factory=list, description="Files referenced in this turn"
    )


class OracleContext(BaseModel):
    """Persistent conversation state for a user+project pair.

    Survives sessions and model changes. Contains compressed summary,
    recent exchanges, key decisions, and token usage tracking.
    """

    id: str = Field(..., description="UUID primary key")
    user_id: str = Field(..., description="User identifier")
    project_id: str = Field(..., description="Project scope")
    session_start: datetime = Field(
        default_factory=datetime.utcnow, description="When context was created"
    )
    last_activity: Optional[datetime] = Field(
        None, description="Last interaction timestamp"
    )
    last_model: Optional[str] = Field(None, description="Model used in last turn")
    token_budget: int = Field(16000, description="Max tokens allowed")
    tokens_used: int = Field(0, description="Current token count")
    compressed_summary: Optional[str] = Field(
        None, description="Older exchanges compressed into summary"
    )
    recent_exchanges: List[OracleExchange] = Field(
        default_factory=list, description="Last N exchanges (preserved verbatim)"
    )
    key_decisions: List[str] = Field(
        default_factory=list, description="Important decisions (never compressed)"
    )
    mentioned_symbols: Optional[str] = Field(
        None, description="Comma-separated symbols mentioned across session"
    )
    mentioned_files: Optional[str] = Field(
        None, description="Comma-separated files mentioned across session"
    )
    status: ContextStatus = Field(
        default=ContextStatus.ACTIVE, description="Context session status"
    )
    compression_count: int = Field(0, description="Times context has been compressed")


class OracleContextResponse(BaseModel):
    """API response for context retrieval."""

    id: str
    project_id: str
    session_start: datetime
    last_activity: Optional[datetime]
    token_budget: int
    tokens_used: int
    tokens_remaining: int
    compressed_summary: Optional[str]
    recent_exchanges: List[OracleExchange]
    key_decisions: List[str]
    status: ContextStatus


class Tool(BaseModel):
    """Definition of a tool the Oracle can invoke."""

    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="What the tool does")
    parameters: Dict[str, Any] = Field(..., description="JSON Schema for params")
    agent_scope: List[str] = Field(
        default=["oracle"], description="Which agents can use this tool"
    )
    category: ToolCategory = Field(..., description="Tool category for organization")


class Subagent(BaseModel):
    """Definition of a specialized subagent."""

    name: str = Field(..., description="Subagent identifier")
    system_prompt_path: str = Field(..., description="Path to prompt file")
    allowed_tools: List[str] = Field(..., description="Tool names it can use")
    max_turns: int = Field(10, description="Max agent loop iterations")
    model_override: Optional[str] = Field(
        None, description="Different model for this subagent"
    )


__all__ = [
    # Enums
    "ContextStatus",
    "ExchangeRole",
    "ToolCallStatus",
    "ToolCategory",
    # Shared models
    "ToolCall",
    # Tree-based context models (new)
    "ContextNode",
    "ContextTree",
    "ContextTreeResponse",
    "ContextNodeCreateRequest",
    "ContextNodeUpdateRequest",
    "ContextTreeCreateRequest",
    "ContextTreeUpdateRequest",
    # Legacy flat context models (deprecated)
    "OracleExchange",
    "OracleContext",
    "OracleContextResponse",
    # Tool and agent definitions
    "Tool",
    "Subagent",
]
