"""
BehaviorTree - Named tree composition with tick execution.

Part of the BT Universal Runtime (spec 019).
Implements the BehaviorTree interface from tree-loader.yaml and data-model.md.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING

from ..state import RunStatus, TypedBlackboard

if TYPE_CHECKING:
    from .context import TickContext


logger = logging.getLogger(__name__)


class TreeStatus(str, Enum):
    """
    Current status of a behavior tree.

    From tree-loader.yaml TreeStatus enum:
    - IDLE: Tree is not executing
    - RUNNING: Tree is mid-execution
    - COMPLETED: Last execution completed with SUCCESS
    - FAILED: Last execution completed with FAILURE
    - YIELDED: Tree yielded due to tick budget
    """

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    YIELDED = "yielded"

    def is_terminal(self) -> bool:
        """Check if status indicates execution finished."""
        return self in (TreeStatus.COMPLETED, TreeStatus.FAILED)

    def is_active(self) -> bool:
        """Check if tree is actively executing or yielded."""
        return self in (TreeStatus.RUNNING, TreeStatus.YIELDED)


class BehaviorNodeProtocol(Protocol):
    """Protocol for behavior tree nodes."""

    @property
    def id(self) -> str:
        """Node identifier."""
        ...

    @property
    def status(self) -> RunStatus:
        """Current run status."""
        ...

    @property
    def running_since(self) -> Optional[datetime]:
        """When node entered RUNNING state."""
        ...

    def tick(self, ctx: "TickContext") -> RunStatus:
        """Execute one tick of this node."""
        ...

    def reset(self) -> None:
        """Reset node to initial state."""
        ...


@dataclass
class BehaviorTree:
    """
    Named tree composition with tick execution.

    From tree-loader.yaml BehaviorTree interface:
    - Manages a root node and tree-scoped blackboard
    - Tracks tick count and execution status
    - Supports reset and cancel operations
    - Provides debug info for observability

    Invariants:
    - id is unique and non-empty
    - root node is not None
    - blackboard is tree-scoped
    - tick_count >= 0
    - status transitions follow state machine (data-model.md)

    Example:
        >>> root = MySequenceNode(id="root", children=[...])
        >>> tree = BehaviorTree(id="oracle-agent", name="Oracle Agent", root=root)
        >>> ctx = TickContext(blackboard=tree.blackboard)
        >>> status = tree.tick(ctx)
        >>> print(tree.status)
    """

    # Identity
    id: str
    name: str
    root: BehaviorNodeProtocol
    description: str = ""

    # Source tracking
    source_path: str = ""
    source_hash: str = ""

    # Configuration (from tree-loader.yaml)
    max_tick_duration_ms: int = 60000  # 60s watchdog timeout
    tick_budget: int = 1000  # Max ticks per event before yield

    # Internal state - using default_factory for mutable defaults
    _blackboard: Optional[TypedBlackboard] = field(default=None, repr=False)
    _status: TreeStatus = field(default=TreeStatus.IDLE, repr=False)
    _tick_count: int = field(default=0, repr=False)
    _reload_pending: bool = field(default=False, repr=False)
    _loaded_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        repr=False,
    )
    _last_tick_at: Optional[datetime] = field(default=None, repr=False)
    _execution_start: Optional[datetime] = field(default=None, repr=False)

    # Callbacks
    _on_status_change: Optional[Callable[["BehaviorTree", TreeStatus, TreeStatus], None]] = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize tree with validation."""
        if not self.id:
            raise ValueError("Tree id cannot be empty")
        if not self.name:
            raise ValueError("Tree name cannot be empty")
        if self.root is None:
            raise ValueError("Tree root cannot be None")

        # Create tree-scoped blackboard if not provided
        if self._blackboard is None:
            self._blackboard = TypedBlackboard(scope_name=f"tree:{self.id}")

    # =========================================================================
    # Properties (readonly as per tree-loader.yaml)
    # =========================================================================

    @property
    def blackboard(self) -> TypedBlackboard:
        """Tree-scoped blackboard for data storage."""
        return self._blackboard

    @property
    def status(self) -> TreeStatus:
        """Current tree execution status."""
        return self._status

    @property
    def tick_count(self) -> int:
        """Total ticks executed in current/last execution."""
        return self._tick_count

    @property
    def reload_pending(self) -> bool:
        """True if a reload is queued."""
        return self._reload_pending

    @property
    def loaded_at(self) -> datetime:
        """When tree was loaded."""
        return self._loaded_at

    @property
    def last_tick_at(self) -> Optional[datetime]:
        """When last tick was executed."""
        return self._last_tick_at

    @property
    def node_count(self) -> int:
        """Total nodes in tree (computed)."""
        return self._count_nodes(self.root)

    @property
    def max_depth(self) -> int:
        """Tree depth (computed)."""
        return self._compute_depth(self.root)

    # =========================================================================
    # Tick Execution
    # =========================================================================

    def tick(self, ctx: "TickContext") -> RunStatus:
        """
        Execute one tick cycle.

        From tree-loader.yaml tick method:
        - Increments tick_count
        - Updates status based on result
        - Handles budget exceeded -> YIELDED

        Args:
            ctx: Tick context with blackboard and services.

        Returns:
            RunStatus from root node tick.

        Raises:
            RuntimeError: If tree is in invalid state for tick.
        """
        # Check for cancellation
        if ctx.cancellation_requested:
            self._set_status(TreeStatus.FAILED)
            return RunStatus.FAILURE

        # Transition to RUNNING on first tick
        if self._status == TreeStatus.IDLE:
            self._execution_start = datetime.now(timezone.utc)
            self._set_status(TreeStatus.RUNNING)

        # Check if resuming from YIELDED
        if self._status == TreeStatus.YIELDED:
            self._set_status(TreeStatus.RUNNING)

        # Check budget
        if ctx.budget_exceeded():
            self._set_status(TreeStatus.YIELDED)
            return RunStatus.RUNNING

        # Execute tick
        self._tick_count += 1
        self._last_tick_at = datetime.now(timezone.utc)
        ctx.increment_tick()

        # Trace if enabled
        ctx.trace(self.id, "tree.tick.start", tick=self._tick_count)

        try:
            # Push tree as root of path
            ctx.push_path(self.id)

            # Tick root node
            result = self.root.tick(ctx)

            # Pop tree from path
            ctx.pop_path()

        except Exception as e:
            ctx.pop_path()
            logger.error(f"Tree {self.id} tick failed: {e}")
            ctx.trace(self.id, "tree.tick.error", error=str(e))
            self._set_status(TreeStatus.FAILED)
            return RunStatus.FAILURE

        # Trace completion
        ctx.trace(
            self.id,
            "tree.tick.complete",
            tick=self._tick_count,
            result=result.name,
        )

        # Update status based on result
        if result == RunStatus.SUCCESS:
            self._set_status(TreeStatus.COMPLETED)
        elif result == RunStatus.FAILURE:
            self._set_status(TreeStatus.FAILED)
        elif result == RunStatus.RUNNING:
            self._set_status(TreeStatus.RUNNING)

        return result

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def reset(self) -> None:
        """
        Reset tree to initial state.

        From tree-loader.yaml reset method:
        - Sets status = IDLE
        - Resets all nodes
        - Clears blackboard (except global)
        """
        logger.debug(f"Resetting tree {self.id}")

        # Reset status
        self._set_status(TreeStatus.IDLE)
        self._tick_count = 0
        self._execution_start = None

        # Reset root node (which should cascade to children)
        if hasattr(self.root, "reset"):
            self.root.reset()

        # Clear blackboard (but preserve schemas)
        # Note: We clear data but keep schema registrations
        self._blackboard._data.clear()
        self._blackboard._size_bytes = 0
        self._blackboard.clear_access_tracking()

    def cancel(self, reason: str = "user_request") -> None:
        """
        Cancel current execution.

        From tree-loader.yaml cancel method:
        - Sets ctx.cancellation_requested = True (via callback)
        - In-flight async operations should be cancelled by nodes
        - Emits E3006 event

        Args:
            reason: Reason for cancellation.
        """
        logger.info(f"Cancelling tree {self.id}: {reason}")

        # Transition to FAILED (cancellation is a form of failure)
        self._set_status(TreeStatus.FAILED)

        # Note: The actual cancellation of async operations
        # happens in the TickScheduler or node implementations
        # which check the status or cancellation flags

    def queue_reload(self) -> None:
        """Mark tree for reload after current execution completes."""
        self._reload_pending = True

    def clear_reload_pending(self) -> None:
        """Clear the reload pending flag."""
        self._reload_pending = False

    # =========================================================================
    # Node Access
    # =========================================================================

    def get_running_nodes(self) -> List[BehaviorNodeProtocol]:
        """
        Get all nodes currently in RUNNING status.

        Used by TreeWatchdog for stuck detection.

        Returns:
            List of nodes with status == RUNNING.
        """
        running = []
        self._collect_running_nodes(self.root, running)
        return running

    def _collect_running_nodes(
        self,
        node: BehaviorNodeProtocol,
        result: List[BehaviorNodeProtocol],
    ) -> None:
        """Recursively collect running nodes."""
        if node.status == RunStatus.RUNNING:
            result.append(node)

        # Check children if composite
        if hasattr(node, "children"):
            for child in node.children:
                self._collect_running_nodes(child, result)

        # Check child if decorator
        if hasattr(node, "child") and node.child is not None:
            self._collect_running_nodes(node.child, result)

    def get_node_by_id(self, node_id: str) -> Optional[BehaviorNodeProtocol]:
        """
        Find a node by ID.

        Args:
            node_id: The node ID to find.

        Returns:
            The node if found, None otherwise.
        """
        return self._find_node(self.root, node_id)

    def _find_node(
        self,
        node: BehaviorNodeProtocol,
        node_id: str,
    ) -> Optional[BehaviorNodeProtocol]:
        """Recursively find node by ID."""
        if node.id == node_id:
            return node

        # Check children if composite
        if hasattr(node, "children"):
            for child in node.children:
                found = self._find_node(child, node_id)
                if found:
                    return found

        # Check child if decorator
        if hasattr(node, "child") and node.child is not None:
            found = self._find_node(node.child, node_id)
            if found:
                return found

        return None

    def get_node_path(self, node_id: str) -> List[str]:
        """
        Get path from root to a node.

        Args:
            node_id: The node ID to find path to.

        Returns:
            List of node IDs from root to target, empty if not found.
        """
        path: List[str] = []
        self._find_path(self.root, node_id, path)
        return path

    def _find_path(
        self,
        node: BehaviorNodeProtocol,
        target_id: str,
        path: List[str],
    ) -> bool:
        """Recursively find path to node."""
        path.append(node.id)

        if node.id == target_id:
            return True

        # Check children if composite
        if hasattr(node, "children"):
            for child in node.children:
                if self._find_path(child, target_id, path):
                    return True

        # Check child if decorator
        if hasattr(node, "child") and node.child is not None:
            if self._find_path(node.child, target_id, path):
                return True

        path.pop()
        return False

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_status_change(
        self,
        callback: Callable[["BehaviorTree", TreeStatus, TreeStatus], None],
    ) -> None:
        """
        Register callback for status changes.

        Args:
            callback: Function called with (tree, old_status, new_status).
        """
        self._on_status_change = callback

    def _set_status(self, new_status: TreeStatus) -> None:
        """Set status and fire callback if registered."""
        if new_status != self._status:
            old_status = self._status
            self._status = new_status
            logger.debug(f"Tree {self.id} status: {old_status.value} -> {new_status.value}")
            if self._on_status_change:
                try:
                    self._on_status_change(self, old_status, new_status)
                except Exception as e:
                    logger.error(f"Status change callback failed: {e}")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _count_nodes(self, node: BehaviorNodeProtocol) -> int:
        """Recursively count nodes in tree."""
        count = 1

        if hasattr(node, "children"):
            for child in node.children:
                count += self._count_nodes(child)

        if hasattr(node, "child") and node.child is not None:
            count += self._count_nodes(node.child)

        return count

    def _compute_depth(self, node: BehaviorNodeProtocol, current: int = 1) -> int:
        """Recursively compute max depth."""
        max_child_depth = current

        if hasattr(node, "children"):
            for child in node.children:
                child_depth = self._compute_depth(child, current + 1)
                max_child_depth = max(max_child_depth, child_depth)

        if hasattr(node, "child") and node.child is not None:
            child_depth = self._compute_depth(node.child, current + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    # =========================================================================
    # Debugging
    # =========================================================================

    def debug_info(self) -> Dict[str, Any]:
        """
        Return debug information about this tree.

        From tree-loader.yaml debug_info method.

        Returns:
            Dictionary with tree state for debugging.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self._status.value,
            "tick_count": self._tick_count,
            "node_count": self.node_count,
            "max_depth": self.max_depth,
            "max_tick_duration_ms": self.max_tick_duration_ms,
            "tick_budget": self.tick_budget,
            "reload_pending": self._reload_pending,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "loaded_at": self._loaded_at.isoformat(),
            "last_tick_at": (
                self._last_tick_at.isoformat()
                if self._last_tick_at else None
            ),
            "execution_start": (
                self._execution_start.isoformat()
                if self._execution_start else None
            ),
            "blackboard_size_bytes": self._blackboard.get_size_bytes(),
            "root_id": self.root.id,
            "root_status": self.root.status.name,
        }


__all__ = ["BehaviorTree", "TreeStatus", "BehaviorNodeProtocol"]
