"""
Tree Visualizer - Export behavior trees in multiple formats.

Part of the BT Universal Runtime (spec 019).
Implements task 7.2 from tasks.md:
- JSON export of tree structure
- ASCII tree visualization
- Graphviz DOT export
- Active node highlighting

From spec.md FR-8:
- Output formats: JSON for programmatic access, tree visualization (ASCII or graphviz)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.tree import BehaviorTree, BehaviorNodeProtocol
    from ..state import RunStatus


# =============================================================================
# ASCII Tree Visualization
# =============================================================================


def ascii_tree(
    tree: "BehaviorTree",
    active_path: Optional[List[str]] = None,
    show_status: bool = True,
    show_timing: bool = False,
) -> str:
    """
    Generate ASCII art representation of a behavior tree.

    From footgun-addendum.md E.1:
    ```
    oracle-agent (Sequence)
    |- load-context (Action: oracle.load_context)
    |   |- inputs: session_id
    |   |- outputs: context, turn_number
    |- check-budget (Guard: ctx.budget > 0)
    |   |- process-query (Sequence)
    |       |- search-code (CodeSearch)
    ...
    ```

    Args:
        tree: The behavior tree to visualize.
        active_path: List of node IDs on active path (for highlighting).
        show_status: Whether to show node status.
        show_timing: Whether to show timing information.

    Returns:
        ASCII string representation of the tree.
    """
    active_set = set(active_path) if active_path else set()
    lines: List[str] = []

    # Tree header
    status_str = f" [{tree.status.value}]" if show_status else ""
    lines.append(f"{tree.name} (Tree){status_str}")

    # Render root node and children
    _render_node_ascii(
        node=tree.root,
        lines=lines,
        prefix="",
        is_last=True,
        active_set=active_set,
        show_status=show_status,
        show_timing=show_timing,
    )

    return "\n".join(lines)


def _render_node_ascii(
    node: "BehaviorNodeProtocol",
    lines: List[str],
    prefix: str,
    is_last: bool,
    active_set: Set[str],
    show_status: bool,
    show_timing: bool,
) -> None:
    """Recursively render a node and its children as ASCII."""
    # Determine connector
    connector = "\\-- " if is_last else "|-- "

    # Build node line
    is_active = node.id in active_set
    active_marker = " *" if is_active else ""

    # Get node type name
    node_type = type(node).__name__

    # Status suffix
    status_suffix = ""
    if show_status:
        status_suffix = f" [{node.status.name}]"

    # Timing suffix
    timing_suffix = ""
    if show_timing and hasattr(node, "last_tick_duration_ms"):
        timing_suffix = f" ({node.last_tick_duration_ms:.1f}ms)"

    lines.append(f"{prefix}{connector}{node.id} ({node_type}){status_suffix}{timing_suffix}{active_marker}")

    # Prepare prefix for children
    child_prefix = prefix + ("    " if is_last else "|   ")

    # Get children
    children = _get_children(node)

    # Render children
    for i, child in enumerate(children):
        child_is_last = i == len(children) - 1
        _render_node_ascii(
            node=child,
            lines=lines,
            prefix=child_prefix,
            is_last=child_is_last,
            active_set=active_set,
            show_status=show_status,
            show_timing=show_timing,
        )


def _get_children(node: "BehaviorNodeProtocol") -> List["BehaviorNodeProtocol"]:
    """Get children of a node, handling both composite and decorator nodes."""
    children = []

    # Composite nodes have children attribute
    if hasattr(node, "children"):
        children.extend(node.children)

    # Decorator nodes have child attribute
    if hasattr(node, "child") and node.child is not None:
        children.append(node.child)

    return children


# =============================================================================
# JSON Export
# =============================================================================


def json_export(
    tree: "BehaviorTree",
    active_path: Optional[List[str]] = None,
    include_timing: bool = True,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Export behavior tree as JSON structure.

    Suitable for programmatic access and frontend visualization.

    Args:
        tree: The behavior tree to export.
        active_path: List of node IDs on active path.
        include_timing: Include timing information.
        include_metadata: Include node metadata.

    Returns:
        Dictionary representing the tree structure.
    """
    active_set = set(active_path) if active_path else set()

    return {
        "id": tree.id,
        "name": tree.name,
        "description": tree.description,
        "status": tree.status.value,
        "tick_count": tree.tick_count,
        "node_count": tree.node_count,
        "max_depth": tree.max_depth,
        "source_path": tree.source_path,
        "loaded_at": tree.loaded_at.isoformat() if tree.loaded_at else None,
        "last_tick_at": tree.last_tick_at.isoformat() if tree.last_tick_at else None,
        "active_path": list(active_set),
        "root": _node_to_dict(
            node=tree.root,
            active_set=active_set,
            include_timing=include_timing,
            include_metadata=include_metadata,
        ),
    }


def _node_to_dict(
    node: "BehaviorNodeProtocol",
    active_set: Set[str],
    include_timing: bool,
    include_metadata: bool,
) -> Dict[str, Any]:
    """Convert a node and its children to dictionary."""
    result: Dict[str, Any] = {
        "id": node.id,
        "name": getattr(node, "name", node.id),
        "type": type(node).__name__,
        "node_type": node.node_type.value if hasattr(node, "node_type") else "unknown",
        "status": node.status.name,
        "tick_count": getattr(node, "tick_count", 0),
        "is_active": node.id in active_set,
    }

    # Add timing info
    if include_timing:
        result["last_tick_duration_ms"] = getattr(node, "last_tick_duration_ms", 0.0)
        running_since = getattr(node, "running_since", None)
        result["running_since"] = running_since.isoformat() if running_since else None

    # Add metadata
    if include_metadata and hasattr(node, "metadata"):
        result["metadata"] = dict(node.metadata)

    # Add children
    children = _get_children(node)
    if children:
        result["children"] = [
            _node_to_dict(child, active_set, include_timing, include_metadata)
            for child in children
        ]
    else:
        result["children"] = []

    return result


# =============================================================================
# Graphviz DOT Export
# =============================================================================


def dot_export(
    tree: "BehaviorTree",
    active_path: Optional[List[str]] = None,
    show_status: bool = True,
    rankdir: str = "TB",
) -> str:
    """
    Export behavior tree as Graphviz DOT format.

    Useful for rendering with graphviz tools or web-based viewers.

    Args:
        tree: The behavior tree to export.
        active_path: List of node IDs on active path (highlighted in green).
        show_status: Whether to show status in node labels.
        rankdir: Graph direction (TB=top-bottom, LR=left-right).

    Returns:
        DOT format string.
    """
    active_set = set(active_path) if active_path else set()
    lines: List[str] = []

    # Graph header
    lines.append(f'digraph "{tree.name}" {{')
    lines.append(f'    rankdir={rankdir};')
    lines.append('    node [shape=box, fontname="monospace"];')
    lines.append('    edge [fontname="monospace"];')
    lines.append("")

    # Add nodes
    _add_dot_nodes(
        node=tree.root,
        lines=lines,
        active_set=active_set,
        show_status=show_status,
    )

    lines.append("")

    # Add edges
    _add_dot_edges(
        node=tree.root,
        lines=lines,
        active_set=active_set,
    )

    lines.append("}")

    return "\n".join(lines)


def _add_dot_nodes(
    node: "BehaviorNodeProtocol",
    lines: List[str],
    active_set: Set[str],
    show_status: bool,
) -> None:
    """Add DOT node definitions recursively."""
    # Node styling based on status and active state
    is_active = node.id in active_set
    status = node.status.name if hasattr(node, "status") else "UNKNOWN"

    # Determine colors
    if is_active:
        fillcolor = "#90EE90"  # Light green for active
        style = "filled,bold"
    elif status == "RUNNING":
        fillcolor = "#87CEEB"  # Light blue for running
        style = "filled"
    elif status == "SUCCESS":
        fillcolor = "#98FB98"  # Pale green for success
        style = "filled"
    elif status == "FAILURE":
        fillcolor = "#FFB6C1"  # Light pink for failure
        style = "filled"
    else:
        fillcolor = "#FFFFFF"  # White for fresh/unknown
        style = "filled"

    # Build label
    node_type = type(node).__name__
    label = f"{node.id}\\n({node_type})"
    if show_status:
        label += f"\\n[{status}]"

    # Escape node ID for DOT
    node_id = _escape_dot_id(node.id)

    lines.append(
        f'    {node_id} [label="{label}", fillcolor="{fillcolor}", style="{style}"];'
    )

    # Recurse for children
    for child in _get_children(node):
        _add_dot_nodes(child, lines, active_set, show_status)


def _add_dot_edges(
    node: "BehaviorNodeProtocol",
    lines: List[str],
    active_set: Set[str],
) -> None:
    """Add DOT edge definitions recursively."""
    parent_id = _escape_dot_id(node.id)
    children = _get_children(node)

    for i, child in enumerate(children):
        child_id = _escape_dot_id(child.id)

        # Highlight active edges
        is_active_edge = node.id in active_set and child.id in active_set
        edge_style = 'color="green", penwidth=2' if is_active_edge else ""

        # Add edge label for ordering in sequences/selectors
        if len(children) > 1:
            lines.append(f'    {parent_id} -> {child_id} [label="{i+1}" {edge_style}];')
        else:
            lines.append(f"    {parent_id} -> {child_id} [{edge_style}];")

        # Recurse
        _add_dot_edges(child, lines, active_set)


def _escape_dot_id(node_id: str) -> str:
    """Escape node ID for DOT format (handle hyphens etc.)."""
    # Replace characters that need escaping
    escaped = node_id.replace("-", "_").replace(" ", "_")
    # If it starts with a number, prefix with n_
    if escaped and escaped[0].isdigit():
        escaped = "n_" + escaped
    return escaped


# =============================================================================
# TreeVisualizer Class
# =============================================================================


class TreeVisualizer:
    """
    Convenience class for tree visualization.

    Wraps the visualization functions with caching and configuration.
    """

    def __init__(
        self,
        show_status: bool = True,
        show_timing: bool = False,
        include_metadata: bool = True,
    ) -> None:
        """
        Initialize visualizer with default settings.

        Args:
            show_status: Default for showing status in visualizations.
            show_timing: Default for showing timing info.
            include_metadata: Default for including metadata in JSON.
        """
        self.show_status = show_status
        self.show_timing = show_timing
        self.include_metadata = include_metadata

    def to_ascii(
        self,
        tree: "BehaviorTree",
        active_path: Optional[List[str]] = None,
    ) -> str:
        """Generate ASCII visualization."""
        return ascii_tree(
            tree=tree,
            active_path=active_path,
            show_status=self.show_status,
            show_timing=self.show_timing,
        )

    def to_json(
        self,
        tree: "BehaviorTree",
        active_path: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate JSON export."""
        return json_export(
            tree=tree,
            active_path=active_path,
            include_timing=self.show_timing,
            include_metadata=self.include_metadata,
        )

    def to_dot(
        self,
        tree: "BehaviorTree",
        active_path: Optional[List[str]] = None,
        rankdir: str = "TB",
    ) -> str:
        """Generate Graphviz DOT export."""
        return dot_export(
            tree=tree,
            active_path=active_path,
            show_status=self.show_status,
            rankdir=rankdir,
        )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "TreeVisualizer",
    "ascii_tree",
    "json_export",
    "dot_export",
]
