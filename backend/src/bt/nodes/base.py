"""
BehaviorNode Base Class - Base abstraction for all behavior tree nodes.

Implements the BehaviorNode interface from contracts/nodes.yaml.
All tree nodes MUST inherit from this class.

Tasks covered: 1.1.1-1.1.8 from tasks.md

Error codes:
- E2001: Missing required input (causes FAILURE)
- E2002: Undeclared read (warning only)
- E2003: Undeclared write (warning only)
- E2004: Invalid node ID format

Part of the BT Universal Runtime (spec 019).
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..state.base import NodeType, RunStatus
from ..state.contracts import ContractedNode, NodeContract

if TYPE_CHECKING:
    from ..core.context import TickContext

logger = logging.getLogger(__name__)

# Node ID validation pattern from nodes.yaml
# Must start with letter, followed by letters, numbers, underscores, or hyphens
NODE_ID_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


class InvalidNodeIdError(ValueError):
    """Exception raised when a node ID doesn't match the required pattern.

    Error code: E2004

    Node IDs must:
    - Start with a letter (a-z, A-Z)
    - Contain only letters, numbers, underscores, and hyphens
    - Not be empty
    """

    def __init__(self, node_id: str, message: Optional[str] = None) -> None:
        self.node_id = node_id
        self.error_code = "E2004"

        if message is None:
            message = (
                f"[E2004] Invalid node ID '{node_id}': "
                f"must match pattern ^[a-zA-Z][a-zA-Z0-9_-]*$ "
                f"(start with letter, followed by letters/numbers/underscores/hyphens)"
            )

        super().__init__(message)


class BehaviorNode(ABC, ContractedNode):
    """Base class for all behavior tree nodes.

    All nodes MUST inherit from this class. Provides:
    - State tracking (status, tick count, timing)
    - Contract validation (inputs/outputs)
    - Hierarchical structure (parent/children)
    - Debug information

    Invariants (from nodes.yaml):
    - _id is unique within tree (checked at tree build time)
    - _status reflects result of most recent tick
    - _tick_count >= 0
    - If _status == RUNNING, _running_since is not None
    - If _status != RUNNING, _running_since is None

    Usage:
        class MyAction(BehaviorNode):
            @property
            def node_type(self) -> NodeType:
                return NodeType.LEAF

            def _tick(self, ctx: TickContext) -> RunStatus:
                # Implement your logic here
                return RunStatus.SUCCESS

    The tick() method handles:
    - Cancellation checking
    - Contract input validation
    - Timing measurement
    - State updates
    - Contract access validation (warnings)

    Override _tick(), NOT tick().
    """

    def __init__(
        self,
        id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a behavior node.

        Args:
            id: Unique identifier within the tree. Must match pattern
                ^[a-zA-Z][a-zA-Z0-9_-]*$
            name: Human-readable name. Defaults to id if not provided.
            metadata: Arbitrary metadata for debugging/tooling.

        Raises:
            InvalidNodeIdError: If id doesn't match the required pattern (E2004).

        Postconditions:
            - _status = RunStatus.FRESH
            - _tick_count = 0
            - _running_since = None
            - _last_tick_duration_ms = 0.0
            - _name = name or id
        """
        # Validate node ID (E2004)
        if not id or not isinstance(id, str):
            raise InvalidNodeIdError(id, "[E2004] Node ID must be a non-empty string")

        if not NODE_ID_PATTERN.match(id):
            raise InvalidNodeIdError(id)

        # Identity
        self._id = id
        self._name = name if name else id
        self._metadata = metadata or {}

        # State (postconditions from nodes.yaml)
        self._status = RunStatus.FRESH
        self._tick_count = 0
        self._running_since: Optional[datetime] = None
        self._last_tick_duration_ms = 0.0

        # Hierarchy
        self._parent: Optional[BehaviorNode] = None
        self._children: List[BehaviorNode] = []

    # =========================================================================
    # Properties (readonly per nodes.yaml)
    # =========================================================================

    @property
    def id(self) -> str:
        """Unique identifier within tree."""
        return self._id

    @property
    def name(self) -> str:
        """Human-readable name (defaults to id)."""
        return self._name

    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        """COMPOSITE, DECORATOR, or LEAF.

        Subclasses must implement this to declare their type.
        """
        pass

    @property
    def status(self) -> RunStatus:
        """Result of most recent tick."""
        return self._status

    @property
    def tick_count(self) -> int:
        """Total times this node has been ticked."""
        return self._tick_count

    @property
    def running_since(self) -> Optional[datetime]:
        """When node entered RUNNING state (None if not RUNNING)."""
        return self._running_since

    @property
    def last_tick_duration_ms(self) -> float:
        """Duration of most recent tick in milliseconds."""
        return self._last_tick_duration_ms

    @property
    def children(self) -> List["BehaviorNode"]:
        """Child nodes (empty for LEAF)."""
        return list(self._children)  # Return copy to prevent mutation

    @property
    def parent(self) -> Optional["BehaviorNode"]:
        """Parent node (None for root)."""
        return self._parent

    @property
    def metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata for debugging/tooling."""
        return dict(self._metadata)  # Return copy

    # =========================================================================
    # Hierarchy Management
    # =========================================================================

    def _add_child(self, child: "BehaviorNode") -> None:
        """Add a child node. Internal method for tree construction.

        Args:
            child: Node to add as a child.

        Raises:
            ValueError: If child already has a parent.
        """
        if child._parent is not None:
            raise ValueError(
                f"Node '{child.id}' already has parent '{child._parent.id}'"
            )

        child._parent = self
        self._children.append(child)

    def _remove_child(self, child: "BehaviorNode") -> bool:
        """Remove a child node. Internal method.

        Args:
            child: Node to remove.

        Returns:
            True if child was found and removed, False otherwise.
        """
        if child in self._children:
            self._children.remove(child)
            child._parent = None
            return True
        return False

    # =========================================================================
    # Main Execution Methods
    # =========================================================================

    def tick(self, ctx: "TickContext") -> RunStatus:
        """Execute node for one tick.

        DO NOT override this method - override _tick() instead.

        This method handles:
        1. Cancellation checking
        2. Contract input validation (E2001)
        3. Timing measurement
        4. Calling _tick() for actual work
        5. State updates (tick_count, status, running_since)
        6. Contract access validation (E2002, E2003 warnings)

        Args:
            ctx: The execution context with blackboard, budget, etc.

        Returns:
            RunStatus from the node execution.

        Postconditions (from nodes.yaml):
        - _tick_count incremented by 1
        - _status set to return value
        - If RUNNING and was not RUNNING: _running_since = now
        - If not RUNNING: _running_since = None
        - _last_tick_duration_ms updated
        - Contract violations logged as warnings
        """
        # Step 1: Check cancellation
        if ctx.cancellation_requested:
            self._handle_cancellation(ctx.cancellation_reason)
            self._status = RunStatus.FAILURE
            return RunStatus.FAILURE

        # Step 2: Clear access tracking for contract validation
        ctx.blackboard.clear_access_tracking()

        # Step 3: Validate contract inputs (E2001)
        missing = self._validate_contract_inputs(ctx.blackboard)
        if missing:
            self._log_missing_inputs(missing)
            self._status = RunStatus.FAILURE
            return RunStatus.FAILURE

        # Step 4: Execute the actual node logic with timing
        start_time = time.perf_counter()

        try:
            status = self._tick(ctx)
        except Exception as e:
            logger.error(
                f"Node '{self._id}' raised exception during _tick: {e}",
                exc_info=True,
            )
            status = RunStatus.FAILURE

        end_time = time.perf_counter()

        # Step 5: Update state
        self._tick_count += 1
        self._last_tick_duration_ms = (end_time - start_time) * 1000
        self._status = status

        # Update running_since based on new status
        if status == RunStatus.RUNNING:
            if self._running_since is None:
                self._running_since = datetime.utcnow()
        else:
            self._running_since = None

        # Step 6: Validate contract access (warnings only)
        violations = self._validate_contract_access(ctx.blackboard)
        for violation in violations:
            logger.warning(violation)

        # Mark progress on successful tick (footgun A.1)
        if status != RunStatus.FAILURE:
            ctx.mark_progress()

        return status

    @abstractmethod
    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Subclass implementation of the tick logic.

        Override THIS method, not tick().
        All actual node logic goes here.

        Args:
            ctx: The execution context.

        Returns:
            RunStatus indicating the result:
            - SUCCESS: Node completed successfully
            - FAILURE: Node failed
            - RUNNING: Node is mid-execution, continue next tick

        Note:
            Contract validation happens in tick() before this is called.
            You can assume all required inputs are present.
        """
        pass

    def reset(self) -> None:
        """Reset node to initial state.

        Resets:
        - _status = RunStatus.FRESH
        - _running_since = None

        Does NOT reset:
        - _tick_count (intentional for debugging)

        Subclasses should override to reset their specific state,
        and call super().reset().
        """
        self._status = RunStatus.FRESH
        self._running_since = None

        # Reset children recursively
        for child in self._children:
            child.reset()

    # =========================================================================
    # Contract Methods (inherited from ContractedNode, can be overridden)
    # =========================================================================

    @classmethod
    def contract(cls) -> NodeContract:
        """Declare state requirements for this node.

        Override in subclasses to declare required inputs, optional inputs,
        and outputs. The default implementation returns an empty contract.

        Returns:
            NodeContract declaring this node's state requirements.

        Example:
            @classmethod
            def contract(cls) -> NodeContract:
                return NodeContract(
                    inputs={"session_id": SessionId},
                    outputs={"context": ConversationContext},
                    description="Load user context"
                )
        """
        return NodeContract()

    # =========================================================================
    # Debug and Logging Methods
    # =========================================================================

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information dictionary.

        Contains:
        - id, name, node_type
        - status, tick_count
        - running_since, last_tick_duration_ms
        - contract_summary
        - children_ids (for composites/decorators)
        - parent_id
        - metadata

        Returns:
            Dictionary with debug information.
        """
        contract = self.__class__.contract()

        info: Dict[str, Any] = {
            "id": self._id,
            "name": self._name,
            "node_type": self.node_type.value,
            "status": self._status.name,
            "tick_count": self._tick_count,
            "running_since": (
                self._running_since.isoformat() if self._running_since else None
            ),
            "last_tick_duration_ms": self._last_tick_duration_ms,
            "contract_summary": contract.summary(),
            "parent_id": self._parent._id if self._parent else None,
            "metadata": self._metadata,
        }

        # Add children info for composites/decorators
        if self._children:
            info["children_ids"] = [child._id for child in self._children]

        return info

    def _handle_cancellation(self, reason: Optional[str] = None) -> None:
        """Handle cancellation request.

        Called when ctx.cancellation_requested is True.
        Subclasses can override to perform cleanup.

        Args:
            reason: Optional reason for cancellation.
        """
        reason_str = reason or "unspecified"
        logger.info(f"Node '{self._id}' cancelled: {reason_str}")

        # Reset running state
        self._running_since = None

    def _log_missing_inputs(self, missing: List[str]) -> None:
        """Log E2001 errors for missing inputs.

        Args:
            missing: List of missing input key names.
        """
        contract = self.__class__.contract()

        for key in missing:
            expected_type = "unknown"
            if key in contract.inputs:
                expected_type = contract.inputs[key].__name__

            logger.error(
                f"[E2001] Node '{self._id}' missing required input: "
                f"'{key}' (type: {expected_type})"
            )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"id='{self._id}', "
            f"status={self._status.name}, "
            f"ticks={self._tick_count})"
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BehaviorNode",
    "InvalidNodeIdError",
    "NODE_ID_PATTERN",
]
