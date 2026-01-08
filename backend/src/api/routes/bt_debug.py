"""
BT Debug API Routes - Observability endpoints for behavior trees.

Part of the BT Universal Runtime (spec 019).
Implements FR-8 from spec.md and Phase 7 from tasks.md:
- /api/bt/debug/trees - List all registered trees
- /api/bt/debug/tree/{id} - Get tree state with visualization
- /api/bt/debug/tree/{id}/blackboard - Get blackboard state
- /api/bt/debug/tree/{id}/history - Get tick history
- /api/bt/debug/tree/{id}/breakpoint - Manage breakpoints

Security: Debug endpoints are only available when DEBUG_MODE=true
or when authenticated as admin.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from ..middleware import AuthContext, require_auth_context
from ...models.bt_debug import (
    TreeListResponse,
    TreeSummary,
    TreeStateResponse,
    NodeInfo,
    BlackboardResponse,
    BlackboardKeyInfo,
    TickHistoryResponse,
    TickHistoryEntry,
    BreakpointResponse,
    BreakpointInfo,
    SetBreakpointRequest,
    DeleteBreakpointRequest,
    TreeVisualization,
    TreeStatusResponse,
    NodeTypeResponse,
    RunStatusResponse,
)
from ...bt.debug import (
    TreeVisualizer,
    ascii_tree,
    json_export,
    dot_export,
    TickHistoryTracker,
    BreakpointManager,
)
from ...bt.debug.history import get_history_registry, HistoryRegistry
from ...bt.debug.breakpoints import get_breakpoint_registry, BreakpointRegistry
from ...services.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bt/debug", tags=["bt-debug"])

# =============================================================================
# Configuration and Dependencies
# =============================================================================


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled."""
    config = get_config()
    # Debug is enabled in local mode or when explicitly set
    return config.enable_local_mode or getattr(config, "enable_bt_debug", False)


def require_debug_mode():
    """Dependency that requires debug mode to be enabled."""
    if not is_debug_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug endpoints are disabled. Set DEBUG_MODE=true to enable.",
        )


def get_tree_registry():
    """Get the global tree registry.

    This is a lazy import to avoid circular dependencies.
    The registry may not exist if no trees have been loaded.
    """
    try:
        from ...bt.lua.registry import TreeRegistry
        from pathlib import Path

        # Try to get existing registry or create one
        # In production, this would be a singleton managed by the app
        tree_dir = Path(__file__).parents[3] / "trees"
        if not tree_dir.exists():
            tree_dir.mkdir(parents=True, exist_ok=True)

        return TreeRegistry(tree_dir)
    except Exception as e:
        logger.warning(f"Failed to get tree registry: {e}")
        return None


# =============================================================================
# 7.1.1 - List All Trees
# =============================================================================


@router.get("/trees", response_model=TreeListResponse)
async def list_trees(
    auth: AuthContext = Depends(require_auth_context),
    _debug: None = Depends(require_debug_mode),
) -> TreeListResponse:
    """
    List all registered behavior trees.

    Returns summary information for all trees in the registry including:
    - Tree ID and name
    - Current execution status
    - Tick count and node count
    - Source file path
    - Load and last tick timestamps

    **Requires:** Debug mode enabled
    """
    registry = get_tree_registry()
    trees: List[TreeSummary] = []

    if registry:
        for tree_id in registry.list_trees():
            tree = registry.get(tree_id)
            if tree:
                source_path = registry._source_paths.get(tree_id, "")

                trees.append(TreeSummary(
                    id=tree.id,
                    name=tree.name,
                    status=TreeStatusResponse(tree.status.value),
                    tick_count=tree.tick_count,
                    node_count=tree.node_count,
                    source_path=str(source_path) if source_path else "",
                    loaded_at=tree.loaded_at or datetime.now(timezone.utc),
                    last_tick_at=tree.last_tick_at,
                ))

    return TreeListResponse(
        trees=trees,
        total=len(trees),
    )


# =============================================================================
# 7.1.2 - Get Tree State
# =============================================================================


@router.get("/tree/{tree_id}", response_model=TreeStateResponse)
async def get_tree_state(
    tree_id: str,
    auth: AuthContext = Depends(require_auth_context),
    _debug: None = Depends(require_debug_mode),
) -> TreeStateResponse:
    """
    Get detailed state of a specific behavior tree.

    Returns full tree structure with:
    - Tree metadata and status
    - Root node with recursive children
    - Active execution path highlighting
    - Blackboard size and stats

    **Requires:** Debug mode enabled

    **Path Parameters:**
    - tree_id: Tree identifier
    """
    registry = get_tree_registry()

    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tree registry not available",
        )

    tree = registry.get(tree_id)
    if not tree:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tree '{tree_id}' not found",
        )

    # Get active path from running nodes
    active_path = []
    for node in tree.get_running_nodes():
        path = tree.get_node_path(node.id)
        if path:
            active_path = path
            break

    # Build node tree recursively
    root_info = _build_node_info(tree.root, set(active_path))

    # Get source info
    source_path = registry._source_paths.get(tree_id, "")

    return TreeStateResponse(
        id=tree.id,
        name=tree.name,
        description=tree.description or "",
        status=TreeStatusResponse(tree.status.value),
        tick_count=tree.tick_count,
        node_count=tree.node_count,
        max_depth=tree.max_depth,
        max_tick_duration_ms=tree.max_tick_duration_ms,
        tick_budget=tree.tick_budget,
        reload_pending=tree.reload_pending,
        source_path=str(source_path) if source_path else "",
        source_hash=tree.source_hash or "",
        loaded_at=tree.loaded_at or datetime.now(timezone.utc),
        last_tick_at=tree.last_tick_at,
        execution_start=None,  # Would need to track this in tree
        blackboard_size_bytes=tree.blackboard.get_size_bytes() if tree.blackboard else 0,
        active_path=active_path,
        root=root_info,
    )


def _build_node_info(node, active_set: set) -> NodeInfo:
    """Build NodeInfo recursively from a node."""
    # Get node type
    node_type_str = "leaf"
    if hasattr(node, "node_type"):
        node_type_str = node.node_type.value

    # Get status
    status_str = "fresh"
    if hasattr(node, "status"):
        status_str = node.status.name.lower()

    # Build children recursively
    children = []
    if hasattr(node, "children"):
        for child in node.children:
            children.append(_build_node_info(child, active_set))
    elif hasattr(node, "child") and node.child:
        children.append(_build_node_info(node.child, active_set))

    return NodeInfo(
        id=node.id,
        name=getattr(node, "name", node.id),
        node_type=NodeTypeResponse(node_type_str),
        status=RunStatusResponse(status_str),
        tick_count=getattr(node, "tick_count", 0),
        running_since=getattr(node, "running_since", None),
        last_tick_duration_ms=getattr(node, "last_tick_duration_ms", 0.0),
        is_active=node.id in active_set,
        children=children,
        metadata=dict(getattr(node, "metadata", {})),
    )


# =============================================================================
# 7.1.3 - Get Blackboard State
# =============================================================================


@router.get("/tree/{tree_id}/blackboard", response_model=BlackboardResponse)
async def get_blackboard_state(
    tree_id: str,
    auth: AuthContext = Depends(require_auth_context),
    _debug: None = Depends(require_debug_mode),
) -> BlackboardResponse:
    """
    Get blackboard state for a behavior tree.

    Returns:
    - All registered keys with types
    - Current values (with size limits for large values)
    - Read/write tracking for current tick
    - Memory usage stats

    **Requires:** Debug mode enabled

    **Path Parameters:**
    - tree_id: Tree identifier
    """
    registry = get_tree_registry()

    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tree registry not available",
        )

    tree = registry.get(tree_id)
    if not tree:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tree '{tree_id}' not found",
        )

    bb = tree.blackboard
    if not bb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tree '{tree_id}' has no blackboard",
        )

    # Get key information
    keys: List[BlackboardKeyInfo] = []
    snapshot = bb.snapshot()

    for key, schema in bb._schemas.items():
        has_value = key in snapshot
        value_preview = None
        size_bytes = 0

        if has_value:
            value = snapshot[key]
            # Truncate preview for large values
            value_str = str(value)
            if len(value_str) > 200:
                value_preview = value_str[:200] + "..."
            else:
                value_preview = value_str
            size_bytes = bb._get_key_size(key)

        keys.append(BlackboardKeyInfo(
            key=key,
            schema_type=schema.__name__ if schema else "unknown",
            has_value=has_value,
            value_preview=value_preview,
            size_bytes=size_bytes,
        ))

    return BlackboardResponse(
        tree_id=tree_id,
        scope_name=bb._scope_name,
        parent_scope=bb._parent._scope_name if bb._parent else None,
        size_bytes=bb.get_size_bytes(),
        max_size_bytes=bb._max_size_bytes,
        key_count=len(bb._data),
        keys=keys,
        snapshot=snapshot,
        reads_this_tick=list(bb.get_reads()),
        writes_this_tick=list(bb.get_writes()),
    )


# =============================================================================
# 7.1.4 - Get Tick History
# =============================================================================


@router.get("/tree/{tree_id}/history", response_model=TickHistoryResponse)
async def get_tick_history(
    tree_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    node_id: Optional[str] = Query(default=None),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    auth: AuthContext = Depends(require_auth_context),
    _debug: None = Depends(require_debug_mode),
) -> TickHistoryResponse:
    """
    Get tick history for a behavior tree.

    Returns chronological list of tick events with:
    - Tick number and timestamp
    - Node that was ticked
    - Result status
    - Duration

    Supports pagination and filtering.

    **Requires:** Debug mode enabled

    **Path Parameters:**
    - tree_id: Tree identifier

    **Query Parameters:**
    - limit: Maximum entries to return (default: 100)
    - offset: Entries to skip for pagination
    - node_id: Filter by specific node
    - status: Filter by status (success, failure, running)
    """
    history_registry = get_history_registry()
    tracker = history_registry.get(tree_id)

    if not tracker:
        # No history yet - return empty
        return TickHistoryResponse(
            tree_id=tree_id,
            total_entries=0,
            returned_entries=0,
            offset=offset,
            limit=limit,
            entries=[],
        )

    # Get entries with filters
    entries = tracker.get_entries(
        limit=limit,
        offset=offset,
        node_id=node_id,
        status=status_filter,
    )

    # Convert to response format
    response_entries = [
        TickHistoryEntry(
            tick_number=e.tick_number,
            timestamp=e.timestamp,
            node_id=e.node_id,
            node_path=e.node_path,
            status=RunStatusResponse(e.status.lower()),
            duration_ms=e.duration_ms,
            event_type=e.event_type,
            details=e.details,
        )
        for e in entries
    ]

    return TickHistoryResponse(
        tree_id=tree_id,
        total_entries=tracker.entry_count,
        returned_entries=len(response_entries),
        offset=offset,
        limit=limit,
        entries=response_entries,
    )


# =============================================================================
# 7.1.5 - Breakpoint Management
# =============================================================================


@router.get("/tree/{tree_id}/breakpoint", response_model=BreakpointResponse)
async def list_breakpoints(
    tree_id: str,
    auth: AuthContext = Depends(require_auth_context),
    _debug: None = Depends(require_debug_mode),
) -> BreakpointResponse:
    """
    List all breakpoints for a behavior tree.

    Returns all set breakpoints with:
    - Node ID where set
    - Enabled/disabled state
    - Condition expression (if any)
    - Hit count

    **Requires:** Debug mode enabled

    **Path Parameters:**
    - tree_id: Tree identifier
    """
    bp_registry = get_breakpoint_registry()
    manager = bp_registry.get_or_create(tree_id)

    breakpoints = [
        BreakpointInfo(
            node_id=bp.node_id,
            enabled=bp.enabled,
            condition=bp.condition,
            hit_count=bp.hit_count,
            created_at=bp.created_at,
        )
        for bp in manager.get_all_breakpoints()
    ]

    return BreakpointResponse(
        tree_id=tree_id,
        breakpoints=breakpoints,
        message=f"{len(breakpoints)} breakpoint(s) set",
    )


@router.post("/tree/{tree_id}/breakpoint", response_model=BreakpointResponse)
async def set_breakpoint(
    tree_id: str,
    request: SetBreakpointRequest,
    auth: AuthContext = Depends(require_auth_context),
    _debug: None = Depends(require_debug_mode),
) -> BreakpointResponse:
    """
    Set or update a breakpoint on a node.

    **Requires:** Debug mode enabled

    **Path Parameters:**
    - tree_id: Tree identifier

    **Request Body:**
    - node_id: Node to set breakpoint on
    - enabled: Whether breakpoint is active (default: true)
    - condition: Optional condition expression
    """
    # Verify tree exists
    registry = get_tree_registry()
    if registry:
        tree = registry.get(tree_id)
        if tree:
            # Verify node exists
            node = tree.get_node_by_id(request.node_id)
            if not node:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Node '{request.node_id}' not found in tree '{tree_id}'",
                )

    bp_registry = get_breakpoint_registry()
    manager = bp_registry.get_or_create(tree_id)

    bp = manager.set_breakpoint(
        node_id=request.node_id,
        enabled=request.enabled,
        condition=request.condition,
    )

    logger.info(f"Set breakpoint on node '{request.node_id}' in tree '{tree_id}'")

    # Return all breakpoints
    breakpoints = [
        BreakpointInfo(
            node_id=b.node_id,
            enabled=b.enabled,
            condition=b.condition,
            hit_count=b.hit_count,
            created_at=b.created_at,
        )
        for b in manager.get_all_breakpoints()
    ]

    return BreakpointResponse(
        tree_id=tree_id,
        breakpoints=breakpoints,
        message=f"Breakpoint set on node '{request.node_id}'",
    )


@router.delete("/tree/{tree_id}/breakpoint", response_model=BreakpointResponse)
async def delete_breakpoint(
    tree_id: str,
    request: DeleteBreakpointRequest,
    auth: AuthContext = Depends(require_auth_context),
    _debug: None = Depends(require_debug_mode),
) -> BreakpointResponse:
    """
    Delete a breakpoint from a node.

    **Requires:** Debug mode enabled

    **Path Parameters:**
    - tree_id: Tree identifier

    **Request Body:**
    - node_id: Node to remove breakpoint from
    """
    bp_registry = get_breakpoint_registry()
    manager = bp_registry.get_or_create(tree_id)

    removed = manager.remove_breakpoint(request.node_id)

    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No breakpoint found on node '{request.node_id}'",
        )

    logger.info(f"Removed breakpoint from node '{request.node_id}' in tree '{tree_id}'")

    # Return remaining breakpoints
    breakpoints = [
        BreakpointInfo(
            node_id=b.node_id,
            enabled=b.enabled,
            condition=b.condition,
            hit_count=b.hit_count,
            created_at=b.created_at,
        )
        for b in manager.get_all_breakpoints()
    ]

    return BreakpointResponse(
        tree_id=tree_id,
        breakpoints=breakpoints,
        message=f"Breakpoint removed from node '{request.node_id}'",
    )


@router.delete("/tree/{tree_id}/breakpoint/all", response_model=BreakpointResponse)
async def clear_all_breakpoints(
    tree_id: str,
    auth: AuthContext = Depends(require_auth_context),
    _debug: None = Depends(require_debug_mode),
) -> BreakpointResponse:
    """
    Clear all breakpoints for a tree.

    **Requires:** Debug mode enabled

    **Path Parameters:**
    - tree_id: Tree identifier
    """
    bp_registry = get_breakpoint_registry()
    manager = bp_registry.get_or_create(tree_id)

    count = manager.clear_all()

    logger.info(f"Cleared {count} breakpoint(s) from tree '{tree_id}'")

    return BreakpointResponse(
        tree_id=tree_id,
        breakpoints=[],
        message=f"Cleared {count} breakpoint(s)",
    )


# =============================================================================
# 7.2 - Tree Visualization Endpoints
# =============================================================================


@router.get("/tree/{tree_id}/visualization", response_model=TreeVisualization)
async def get_tree_visualization(
    tree_id: str,
    format: str = Query(default="all", description="Format: json, ascii, dot, or all"),
    show_status: bool = Query(default=True),
    show_timing: bool = Query(default=False),
    auth: AuthContext = Depends(require_auth_context),
    _debug: None = Depends(require_debug_mode),
) -> TreeVisualization:
    """
    Get tree visualization in various formats.

    **Requires:** Debug mode enabled

    **Path Parameters:**
    - tree_id: Tree identifier

    **Query Parameters:**
    - format: Output format (json, ascii, dot, or all)
    - show_status: Include node status in visualization
    - show_timing: Include timing information

    **Response:**
    - ascii_tree: ASCII art representation
    - json_structure: JSON tree structure
    - dot_graph: Graphviz DOT format
    """
    registry = get_tree_registry()

    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tree registry not available",
        )

    tree = registry.get(tree_id)
    if not tree:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tree '{tree_id}' not found",
        )

    # Get active path
    active_path = []
    for node in tree.get_running_nodes():
        path = tree.get_node_path(node.id)
        if path:
            active_path = path
            break

    # Create visualizer
    visualizer = TreeVisualizer(
        show_status=show_status,
        show_timing=show_timing,
    )

    # Generate requested formats
    ascii_result = ""
    json_result = {}
    dot_result = ""

    if format in ("all", "ascii"):
        ascii_result = visualizer.to_ascii(tree, active_path)

    if format in ("all", "json"):
        json_result = visualizer.to_json(tree, active_path)

    if format in ("all", "dot"):
        dot_result = visualizer.to_dot(tree, active_path)

    return TreeVisualization(
        tree_id=tree_id,
        ascii_tree=ascii_result,
        json_structure=json_result,
        dot_graph=dot_result,
    )


# =============================================================================
# 7.x - Validation Endpoint
# =============================================================================


class ValidateRequest(BaseModel):
    """Request body for tree validation."""
    content: str
    strict: bool = False


class ValidateResponse(BaseModel):
    """Response from tree validation."""
    valid: bool
    node_count: int = 0
    tree_name: str = ""
    errors: List[str] = []
    warnings: List[str] = []


@router.post("/validate", response_model=ValidateResponse)
async def validate_tree(
    request: ValidateRequest,
    auth: AuthContext = Depends(require_auth_context),
    _debug: None = Depends(require_debug_mode),
) -> ValidateResponse:
    """
    Validate a Lua behavior tree definition.

    Checks the tree definition for:
    - Lua syntax errors
    - Invalid node references
    - Missing required properties
    - Circular subtree references

    **Requires:** Debug mode enabled

    **Request Body:**
    - content: Lua tree definition string
    - strict: Enable strict validation mode (default: false)
    """
    try:
        from ...bt.lua.loader import TreeLoader
        from ...bt.lua.validator import TreeValidator

        def count_nodes(node) -> int:
            """Count total nodes in tree recursively."""
            if node is None:
                return 0
            return 1 + sum(count_nodes(child) for child in (node.children or []))

        # Load tree from string
        loader = TreeLoader()
        tree_def = loader.load_string(request.content, "inline")

        # Count nodes
        node_count = count_nodes(tree_def.root) if tree_def and tree_def.root else 0

        # Validate tree
        # In strict mode, resolve function references; otherwise skip resolution
        validator = TreeValidator(resolve_functions=request.strict)
        validation_errors = validator.validate(tree_def)

        if validation_errors:
            return ValidateResponse(
                valid=False,
                node_count=node_count,
                tree_name=tree_def.name if tree_def else "",
                errors=[f"{e.code}: {e.message}" for e in validation_errors],
                warnings=[],
            )

        return ValidateResponse(
            valid=True,
            node_count=node_count,
            tree_name=tree_def.name if tree_def else "",
            errors=[],
            warnings=[],
        )

    except Exception as e:
        return ValidateResponse(
            valid=False,
            node_count=0,
            tree_name="",
            errors=[str(e)],
            warnings=[],
        )


# =============================================================================
# Exports
# =============================================================================


__all__ = ["router"]
