"""
BT Debug Module - Observability and debugging tools for behavior trees.

Part of the BT Universal Runtime (spec 019).
Implements FR-8: Observability and Debugging from spec.md.
"""

from .visualizer import TreeVisualizer, ascii_tree, json_export, dot_export
from .history import TickHistoryTracker
from .breakpoints import BreakpointManager

__all__ = [
    "TreeVisualizer",
    "ascii_tree",
    "json_export",
    "dot_export",
    "TickHistoryTracker",
    "BreakpointManager",
]
