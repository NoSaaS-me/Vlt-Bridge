"""
Pydantic models for BT Debug API (FR-8: Observability).

Part of the BT Universal Runtime (spec 019).
Implements data models for debug endpoints from tasks.md 7.1.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class TreeStatusResponse(str, Enum):
    """Tree execution status for API responses."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    YIELDED = "yielded"


class NodeTypeResponse(str, Enum):
    """Node type for API responses."""

    COMPOSITE = "composite"
    DECORATOR = "decorator"
    LEAF = "leaf"


class RunStatusResponse(str, Enum):
    """Node run status for API responses."""

    FRESH = "fresh"
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


# =============================================================================
# Tree List Response
# =============================================================================


class TreeSummary(BaseModel):
    """Summary of a registered tree for list endpoint."""

    id: str = Field(..., description="Unique tree identifier")
    name: str = Field(..., description="Human-readable tree name")
    status: TreeStatusResponse = Field(..., description="Current execution status")
    tick_count: int = Field(..., description="Total ticks executed")
    node_count: int = Field(..., description="Number of nodes in tree")
    source_path: str = Field(..., description="Path to Lua source file")
    loaded_at: datetime = Field(..., description="When tree was loaded")
    last_tick_at: Optional[datetime] = Field(None, description="When last tick occurred")


class TreeListResponse(BaseModel):
    """Response for GET /api/bt/debug/trees."""

    trees: List[TreeSummary] = Field(default_factory=list, description="All registered trees")
    total: int = Field(..., description="Total number of trees")


# =============================================================================
# Tree State Response
# =============================================================================


class NodeInfo(BaseModel):
    """Information about a single node."""

    id: str = Field(..., description="Node identifier")
    name: str = Field(..., description="Human-readable name")
    node_type: NodeTypeResponse = Field(..., description="Node type")
    status: RunStatusResponse = Field(..., description="Current run status")
    tick_count: int = Field(..., description="Times ticked")
    running_since: Optional[datetime] = Field(None, description="When entered RUNNING")
    last_tick_duration_ms: float = Field(..., description="Duration of last tick")
    is_active: bool = Field(False, description="True if on active execution path")
    children: List["NodeInfo"] = Field(default_factory=list, description="Child nodes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Node metadata")


# Enable forward references for recursive model
NodeInfo.model_rebuild()


class TreeStateResponse(BaseModel):
    """Response for GET /api/bt/debug/tree/{id}."""

    id: str = Field(..., description="Tree identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field("", description="Tree description")
    status: TreeStatusResponse = Field(..., description="Current status")
    tick_count: int = Field(..., description="Total ticks executed")
    node_count: int = Field(..., description="Number of nodes")
    max_depth: int = Field(..., description="Tree depth")
    max_tick_duration_ms: int = Field(..., description="Watchdog timeout")
    tick_budget: int = Field(..., description="Max ticks per event")
    reload_pending: bool = Field(False, description="Reload queued")
    source_path: str = Field("", description="Lua source path")
    source_hash: str = Field("", description="Source file hash")
    loaded_at: datetime = Field(..., description="When loaded")
    last_tick_at: Optional[datetime] = Field(None, description="Last tick time")
    execution_start: Optional[datetime] = Field(None, description="Current execution start")
    blackboard_size_bytes: int = Field(..., description="Blackboard memory usage")
    active_path: List[str] = Field(default_factory=list, description="Currently active node path")
    root: NodeInfo = Field(..., description="Root node with full tree structure")


# =============================================================================
# Blackboard Response
# =============================================================================


class BlackboardKeyInfo(BaseModel):
    """Information about a blackboard key."""

    key: str = Field(..., description="Key name")
    schema_type: str = Field(..., description="Registered schema type name")
    has_value: bool = Field(..., description="Whether key has a value")
    value_preview: Optional[str] = Field(None, description="Value preview (truncated)")
    size_bytes: int = Field(0, description="Approximate size in bytes")


class BlackboardResponse(BaseModel):
    """Response for GET /api/bt/debug/tree/{id}/blackboard."""

    tree_id: str = Field(..., description="Tree identifier")
    scope_name: str = Field(..., description="Blackboard scope name")
    parent_scope: Optional[str] = Field(None, description="Parent scope name")
    size_bytes: int = Field(..., description="Total size in bytes")
    max_size_bytes: int = Field(..., description="Size limit")
    key_count: int = Field(..., description="Number of keys")
    keys: List[BlackboardKeyInfo] = Field(default_factory=list, description="Key details")
    snapshot: Dict[str, Any] = Field(default_factory=dict, description="Full data snapshot")
    reads_this_tick: List[str] = Field(default_factory=list, description="Keys read this tick")
    writes_this_tick: List[str] = Field(default_factory=list, description="Keys written this tick")


# =============================================================================
# Tick History Response
# =============================================================================


class TickHistoryEntry(BaseModel):
    """Single tick history entry."""

    tick_number: int = Field(..., description="Tick number in current execution")
    timestamp: datetime = Field(..., description="When tick occurred")
    node_id: str = Field(..., description="Node that was ticked")
    node_path: List[str] = Field(default_factory=list, description="Path to node")
    status: RunStatusResponse = Field(..., description="Result of tick")
    duration_ms: float = Field(..., description="Tick duration in milliseconds")
    event_type: Optional[str] = Field(None, description="Event that triggered tick")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class TickHistoryResponse(BaseModel):
    """Response for GET /api/bt/debug/tree/{id}/history."""

    tree_id: str = Field(..., description="Tree identifier")
    total_entries: int = Field(..., description="Total history entries available")
    returned_entries: int = Field(..., description="Number returned in this response")
    offset: int = Field(0, description="Offset for pagination")
    limit: int = Field(100, description="Limit for pagination")
    entries: List[TickHistoryEntry] = Field(
        default_factory=list, description="History entries (newest first)"
    )


# =============================================================================
# Breakpoint Models
# =============================================================================


class BreakpointInfo(BaseModel):
    """Information about a breakpoint."""

    node_id: str = Field(..., description="Node where breakpoint is set")
    enabled: bool = Field(True, description="Whether breakpoint is active")
    condition: Optional[str] = Field(None, description="Optional condition expression")
    hit_count: int = Field(0, description="Times breakpoint was hit")
    created_at: datetime = Field(..., description="When breakpoint was created")


class SetBreakpointRequest(BaseModel):
    """Request to set a breakpoint."""

    node_id: str = Field(..., description="Node ID to set breakpoint on")
    enabled: bool = Field(True, description="Whether breakpoint is enabled")
    condition: Optional[str] = Field(None, description="Optional condition expression")


class BreakpointResponse(BaseModel):
    """Response for breakpoint operations."""

    tree_id: str = Field(..., description="Tree identifier")
    breakpoints: List[BreakpointInfo] = Field(
        default_factory=list, description="All breakpoints"
    )
    message: str = Field("", description="Operation result message")


class DeleteBreakpointRequest(BaseModel):
    """Request to delete a breakpoint."""

    node_id: str = Field(..., description="Node ID of breakpoint to delete")


# =============================================================================
# Visualization Models
# =============================================================================


class TreeVisualization(BaseModel):
    """Tree visualization in multiple formats."""

    tree_id: str = Field(..., description="Tree identifier")
    ascii_tree: str = Field("", description="ASCII art representation")
    json_structure: Dict[str, Any] = Field(
        default_factory=dict, description="JSON tree structure"
    )
    dot_graph: str = Field("", description="Graphviz DOT format")


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "TreeStatusResponse",
    "NodeTypeResponse",
    "RunStatusResponse",
    # List
    "TreeSummary",
    "TreeListResponse",
    # State
    "NodeInfo",
    "TreeStateResponse",
    # Blackboard
    "BlackboardKeyInfo",
    "BlackboardResponse",
    # History
    "TickHistoryEntry",
    "TickHistoryResponse",
    # Breakpoints
    "BreakpointInfo",
    "SetBreakpointRequest",
    "BreakpointResponse",
    "DeleteBreakpointRequest",
    # Visualization
    "TreeVisualization",
]
