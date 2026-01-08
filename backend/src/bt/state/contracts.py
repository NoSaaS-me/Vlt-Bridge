"""
NodeContract System - Contract-based validation for behavior tree nodes.

This module implements the contract system from nodes.yaml (Phase 0.4):
- NodeContract: Declares node state requirements and outputs
- ContractViolationError: Exception for contract violations
- ContractedNode: Mixin class for nodes with contracts
- action_contract: Decorator for declaring contracts on action functions

Error codes:
- E2001: Missing required input (error severity, causes FAILURE)
- E2002: Undeclared read (warning severity, logged but execution continues)
- E2003: Undeclared write (warning severity, logged but execution continues)

Part of the BT Universal Runtime (spec 019).

Reference:
- contracts/nodes.yaml - NodeContract specification
- contracts/errors.yaml - Error codes E2001-E2003
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel

from .base import (
    BTError,
    ErrorContext,
    ErrorResult,
    RecoveryAction,
    RecoveryInfo,
    Severity,
)
from .blackboard import TypedBlackboard


# =============================================================================
# Contract Violation Types
# =============================================================================


class ViolationType(str, Enum):
    """Types of contract violations.

    Used by ContractViolationError to classify the violation.
    """

    MISSING_INPUT = "missing_input"  # E2001 - Required input not present
    UNDECLARED_READ = "undeclared_read"  # E2002 - Read key not in contract
    UNDECLARED_WRITE = "undeclared_write"  # E2003 - Wrote key not in contract


# =============================================================================
# ContractViolationError Exception
# =============================================================================


class ContractViolationError(Exception):
    """Exception raised when a node contract is violated.

    This exception carries structured information about the violation
    for error handling and logging.

    Attributes:
        violation_type: Type of violation (missing_input, undeclared_read, undeclared_write)
        key: The blackboard key involved in the violation
        node_id: ID of the node that violated the contract
        contract: The NodeContract that was violated
        error_code: The error code (E2001, E2002, E2003)
        message: Human-readable error message

    Example:
        >>> raise ContractViolationError(
        ...     violation_type=ViolationType.MISSING_INPUT,
        ...     key="session_id",
        ...     node_id="load-context",
        ...     contract=my_contract,
        ... )
    """

    def __init__(
        self,
        violation_type: ViolationType,
        key: str,
        node_id: Optional[str] = None,
        contract: Optional["NodeContract"] = None,
        message: Optional[str] = None,
    ) -> None:
        self.violation_type = violation_type
        self.key = key
        self.node_id = node_id
        self.contract = contract

        # Map violation type to error code
        error_code_map = {
            ViolationType.MISSING_INPUT: "E2001",
            ViolationType.UNDECLARED_READ: "E2002",
            ViolationType.UNDECLARED_WRITE: "E2003",
        }
        self.error_code = error_code_map[violation_type]

        # Generate message if not provided
        if message is None:
            message = self._generate_message()
        self.message = message

        super().__init__(self.message)

    def _generate_message(self) -> str:
        """Generate a human-readable error message."""
        node_name = self.node_id or "unknown"

        if self.violation_type == ViolationType.MISSING_INPUT:
            expected_type = "unknown"
            if self.contract and self.key in self.contract.inputs:
                expected_type = self.contract.inputs[self.key].__name__
            return (
                f"Node '{node_name}' missing required input: "
                f"'{self.key}' (type: {expected_type})"
            )

        elif self.violation_type == ViolationType.UNDECLARED_READ:
            declared = []
            if self.contract:
                declared = list(self.contract.all_input_keys)
            return (
                f"Node '{node_name}' read undeclared key '{self.key}'. "
                f"Add to contract inputs or optional_inputs. "
                f"Declared inputs: {declared}"
            )

        else:  # UNDECLARED_WRITE
            declared = []
            if self.contract:
                declared = list(self.contract.outputs.keys())
            return (
                f"Node '{node_name}' wrote undeclared key '{self.key}'. "
                f"Add to contract outputs. "
                f"Declared outputs: {declared}"
            )

    def to_bt_error(self) -> BTError:
        """Convert to BTError for structured error handling."""
        # Determine severity based on violation type
        severity_map = {
            ViolationType.MISSING_INPUT: Severity.ERROR,
            ViolationType.UNDECLARED_READ: Severity.WARNING,
            ViolationType.UNDECLARED_WRITE: Severity.WARNING,
        }
        severity = severity_map[self.violation_type]

        # Determine recovery action
        recovery_action_map = {
            ViolationType.MISSING_INPUT: RecoveryAction.ABORT,
            ViolationType.UNDECLARED_READ: RecoveryAction.SKIP,
            ViolationType.UNDECLARED_WRITE: RecoveryAction.SKIP,
        }
        recovery_action = recovery_action_map[self.violation_type]

        # Build extra context
        extra: Dict[str, Any] = {"key": self.key}
        if self.contract:
            if self.violation_type == ViolationType.MISSING_INPUT:
                extra["expected_type"] = (
                    self.contract.inputs[self.key].__name__
                    if self.key in self.contract.inputs
                    else "unknown"
                )
                extra["contract_description"] = self.contract.description
            elif self.violation_type == ViolationType.UNDECLARED_READ:
                extra["declared_inputs"] = list(self.contract.all_input_keys)
            else:
                extra["declared_outputs"] = list(self.contract.outputs.keys())

        return BTError(
            code=self.error_code,
            category="node",
            severity=severity,
            message=self.message,
            context=ErrorContext(
                node_id=self.node_id,
                extra=extra,
            ),
            recovery=RecoveryInfo(
                action=recovery_action,
                manual_steps=self._get_manual_steps(),
            ),
            emit_event=True,
        )

    def _get_manual_steps(self) -> str:
        """Get manual resolution steps."""
        if self.violation_type == ViolationType.MISSING_INPUT:
            return (
                f"Ensure upstream node writes '{self.key}' to blackboard "
                f"before this node runs"
            )
        elif self.violation_type == ViolationType.UNDECLARED_READ:
            return "Update NodeContract to declare this input"
        else:
            return "Update NodeContract to declare this output"


# =============================================================================
# NodeContract Dataclass
# =============================================================================


@dataclass
class NodeContract:
    """Declares a node's state requirements and outputs.

    Used for static validation and runtime enforcement.
    Inspired by function type signatures / C++ header files.

    From contracts/nodes.yaml:

    Attributes:
        inputs: Required inputs. Node FAILS (E2001) if missing.
        optional_inputs: Optional inputs. Node works without them.
        outputs: Keys this node may write.
        description: Human-readable description for documentation.

    Invariants:
        - inputs, optional_inputs, and outputs have disjoint keys
        - All schema types are BaseModel subclasses

    Example:
        >>> contract = NodeContract(
        ...     inputs={"session_id": SessionId},
        ...     optional_inputs={"history_limit": HistoryLimit},
        ...     outputs={"context": ConversationContext, "turn_number": TurnNumber},
        ...     description="Load conversation context from database"
        ... )
    """

    inputs: Dict[str, Type[BaseModel]] = field(default_factory=dict)
    optional_inputs: Dict[str, Type[BaseModel]] = field(default_factory=dict)
    outputs: Dict[str, Type[BaseModel]] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        """Validate contract invariants."""
        # Check for disjoint keys
        input_keys = set(self.inputs.keys())
        optional_keys = set(self.optional_inputs.keys())
        output_keys = set(self.outputs.keys())

        input_optional_overlap = input_keys & optional_keys
        if input_optional_overlap:
            raise ValueError(
                f"Keys cannot be both required and optional: {input_optional_overlap}"
            )

        input_output_overlap = input_keys & output_keys
        if input_output_overlap:
            raise ValueError(
                f"Keys cannot be both input and output: {input_output_overlap}"
            )

        optional_output_overlap = optional_keys & output_keys
        if optional_output_overlap:
            raise ValueError(
                f"Keys cannot be both optional_input and output: {optional_output_overlap}"
            )

        # Validate all schemas are BaseModel subclasses
        for key, schema in self.inputs.items():
            if not isinstance(schema, type) or not issubclass(schema, BaseModel):
                raise TypeError(
                    f"Input schema for '{key}' must be a BaseModel subclass, "
                    f"got {type(schema)}"
                )

        for key, schema in self.optional_inputs.items():
            if not isinstance(schema, type) or not issubclass(schema, BaseModel):
                raise TypeError(
                    f"Optional input schema for '{key}' must be a BaseModel subclass, "
                    f"got {type(schema)}"
                )

        for key, schema in self.outputs.items():
            if not isinstance(schema, type) or not issubclass(schema, BaseModel):
                raise TypeError(
                    f"Output schema for '{key}' must be a BaseModel subclass, "
                    f"got {type(schema)}"
                )

    @property
    def all_input_keys(self) -> Set[str]:
        """Union of required and optional input keys.

        Returns:
            Set containing all keys from inputs and optional_inputs.

        Example:
            >>> contract = NodeContract(
            ...     inputs={"a": SomeModel},
            ...     optional_inputs={"b": OtherModel},
            ... )
            >>> contract.all_input_keys
            {'a', 'b'}
        """
        return set(self.inputs.keys()) | set(self.optional_inputs.keys())

    def validate_inputs(
        self,
        blackboard: TypedBlackboard,
        node_id: Optional[str] = None,
    ) -> List[str]:
        """Check all required inputs exist in blackboard.

        Validates that every key in `inputs` has a value present
        in the blackboard (checking the full scope chain).

        Args:
            blackboard: The TypedBlackboard to check.
            node_id: Optional node ID for error context.

        Returns:
            Empty list on success.
            List of missing key names on failure.

        Error:
            E2001 - Required input missing (for each missing key)

        Example:
            >>> missing = contract.validate_inputs(bb, node_id="my-node")
            >>> if missing:
            ...     for key in missing:
            ...         log_error(f"Missing: {key}")
            ...     return RunStatus.FAILURE
        """
        missing_keys: List[str] = []

        for key in self.inputs.keys():
            if not blackboard.has(key):
                missing_keys.append(key)

        return missing_keys

    def validate_access(
        self,
        reads: Set[str],
        writes: Set[str],
        node_id: Optional[str] = None,
    ) -> List[str]:
        """Check actual access matches declared contract.

        Compares the actual reads and writes performed by a node
        against the declared contract to detect violations.

        Args:
            reads: Set of keys that were read during the tick.
            writes: Set of keys that were written during the tick.
            node_id: Optional node ID for error context.

        Returns:
            Empty list if all access was declared.
            List of violation descriptions on contract mismatch.

        Violations:
            - E2002 (warning): Read undeclared key
            - E2003 (warning): Wrote undeclared key

        Note:
            These are warnings, not errors. Execution continues normally.
            The violations are returned for logging purposes.

        Example:
            >>> violations = contract.validate_access(
            ...     reads=bb.get_reads(),
            ...     writes=bb.get_writes(),
            ...     node_id="my-node",
            ... )
            >>> for v in violations:
            ...     log_warning(v)
        """
        violations: List[str] = []
        node_name = node_id or "unknown"

        # Check for undeclared reads
        declared_inputs = self.all_input_keys
        for key in reads:
            # Skip internal keys (underscore prefix) - they are system-managed
            if key.startswith("_"):
                continue
            if key not in declared_inputs:
                violations.append(
                    f"[E2002] Node '{node_name}' read undeclared key '{key}'. "
                    f"Add to contract inputs or optional_inputs."
                )

        # Check for undeclared writes
        declared_outputs = set(self.outputs.keys())
        for key in writes:
            # Skip internal keys (underscore prefix) - they are system-managed
            if key.startswith("_"):
                continue
            if key not in declared_outputs:
                violations.append(
                    f"[E2003] Node '{node_name}' wrote undeclared key '{key}'. "
                    f"Add to contract outputs."
                )

        return violations

    def get_missing_inputs(
        self,
        blackboard: TypedBlackboard,
    ) -> List[ContractViolationError]:
        """Get ContractViolationError for each missing required input.

        Convenience method that returns full error objects instead of just keys.

        Args:
            blackboard: The TypedBlackboard to check.

        Returns:
            List of ContractViolationError for each missing input.
        """
        missing = self.validate_inputs(blackboard)
        return [
            ContractViolationError(
                violation_type=ViolationType.MISSING_INPUT,
                key=key,
                contract=self,
            )
            for key in missing
        ]

    def summary(self) -> str:
        """Return a summary of the contract for debugging.

        Returns:
            Multi-line string describing the contract.
        """
        lines = []
        if self.description:
            lines.append(f"Description: {self.description}")

        if self.inputs:
            input_strs = [f"{k}: {v.__name__}" for k, v in self.inputs.items()]
            lines.append(f"Required inputs: {', '.join(input_strs)}")

        if self.optional_inputs:
            opt_strs = [f"{k}: {v.__name__}" for k, v in self.optional_inputs.items()]
            lines.append(f"Optional inputs: {', '.join(opt_strs)}")

        if self.outputs:
            out_strs = [f"{k}: {v.__name__}" for k, v in self.outputs.items()]
            lines.append(f"Outputs: {', '.join(out_strs)}")

        if not lines:
            return "Empty contract (no requirements)"

        return "\n".join(lines)


# =============================================================================
# ContractedNode Mixin
# =============================================================================


class ContractedNode:
    """Mixin class for nodes that have contracts.

    Provides the contract() class method and helper methods for
    contract validation. Nodes can inherit from this to gain
    contract functionality.

    Usage:
        class MyAction(LeafNode, ContractedNode):
            @classmethod
            def contract(cls) -> NodeContract:
                return NodeContract(
                    inputs={"query": QueryModel},
                    outputs={"result": ResultModel},
                )

            def _tick(self, ctx: TickContext) -> RunStatus:
                # Contract validation happens in parent tick() method
                query = ctx.blackboard.get("query", QueryModel)
                # ... do work ...
                ctx.blackboard.set("result", result)
                return RunStatus.SUCCESS

    The actual validation is performed by the BehaviorNode.tick() method,
    which calls these helpers before and after _tick().
    """

    # Cache for contract instances (class-level)
    _contract_cache: Optional[NodeContract] = None

    @classmethod
    def contract(cls) -> NodeContract:
        """Declare state requirements for this node.

        Override in subclasses to declare required inputs, optional inputs,
        and outputs. The default implementation returns an empty contract
        (no requirements).

        Returns:
            NodeContract declaring this node's state requirements.

        Example:
            @classmethod
            def contract(cls) -> NodeContract:
                return NodeContract(
                    inputs={"session_id": SessionId},
                    optional_inputs={"limit": LimitConfig},
                    outputs={"context": ConversationContext},
                    description="Load user context"
                )
        """
        # Default: empty contract
        if cls._contract_cache is None:
            cls._contract_cache = NodeContract()
        return cls._contract_cache

    def _validate_contract_inputs(
        self,
        blackboard: TypedBlackboard,
    ) -> Optional[List[str]]:
        """Validate that all required inputs are present.

        Called by tick() before _tick() execution.

        Args:
            blackboard: The blackboard to check.

        Returns:
            None if all inputs present.
            List of missing keys if validation fails.
        """
        node_contract = self.__class__.contract()
        node_id = getattr(self, "_id", None) or getattr(self, "id", None)
        missing = node_contract.validate_inputs(blackboard, node_id)

        if missing:
            return missing
        return None

    def _validate_contract_access(
        self,
        blackboard: TypedBlackboard,
    ) -> List[str]:
        """Validate that all reads/writes were declared in contract.

        Called by tick() after _tick() execution.

        Args:
            blackboard: The blackboard with access tracking.

        Returns:
            List of violation description strings (may be empty).
        """
        node_contract = self.__class__.contract()
        node_id = getattr(self, "_id", None) or getattr(self, "id", None)

        return node_contract.validate_access(
            reads=blackboard.get_reads(),
            writes=blackboard.get_writes(),
            node_id=node_id,
        )

    def get_contract_summary(self) -> str:
        """Get a summary of this node's contract for debugging.

        Returns:
            Human-readable contract summary.
        """
        return self.__class__.contract().summary()


# =============================================================================
# action_contract Decorator
# =============================================================================


F = TypeVar("F", bound=Callable[..., Any])


def action_contract(
    inputs: Optional[Dict[str, Type[BaseModel]]] = None,
    optional_inputs: Optional[Dict[str, Type[BaseModel]]] = None,
    outputs: Optional[Dict[str, Type[BaseModel]]] = None,
    description: str = "",
) -> Callable[[F], F]:
    """Decorator for declaring contracts on action functions.

    Use this decorator to attach a NodeContract to a function that
    will be used as an action node's implementation.

    The contract is stored as a `__contract__` attribute on the function.

    Args:
        inputs: Required input keys and their schema types.
        optional_inputs: Optional input keys and their schema types.
        outputs: Output keys this action may write.
        description: Human-readable description of the action.

    Returns:
        Decorator function.

    Example:
        @action_contract(
            inputs={"session_id": SessionId},
            optional_inputs={"limit": LimitConfig},
            outputs={"context": ConversationContext},
            description="Load user conversation context"
        )
        def load_context(ctx: TickContext) -> RunStatus:
            session_id = ctx.blackboard.get("session_id", SessionId)
            # ... implementation ...
            ctx.blackboard.set("context", context)
            return RunStatus.SUCCESS

    The contract can then be accessed via:
        >>> load_context.__contract__
        NodeContract(inputs={'session_id': SessionId}, ...)
    """
    # Create the contract
    contract = NodeContract(
        inputs=inputs or {},
        optional_inputs=optional_inputs or {},
        outputs=outputs or {},
        description=description,
    )

    def decorator(func: F) -> F:
        # Attach contract to function
        func.__contract__ = contract  # type: ignore[attr-defined]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # Also attach to wrapper
        wrapper.__contract__ = contract  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def get_action_contract(func: Callable[..., Any]) -> Optional[NodeContract]:
    """Get the NodeContract attached to a function by @action_contract.

    Args:
        func: A function that may have been decorated with @action_contract.

    Returns:
        The NodeContract if the function was decorated, None otherwise.

    Example:
        >>> contract = get_action_contract(load_context)
        >>> if contract:
        ...     print(contract.description)
    """
    return getattr(func, "__contract__", None)


# =============================================================================
# Error Factory Functions (for E2001-E2003)
# =============================================================================


def make_missing_input_error(
    node_id: str,
    node_name: str,
    key: str,
    expected_type: str,
    contract_description: str = "",
) -> BTError:
    """Create E2001: Contract violation - missing required input.

    From errors.yaml E2001:
    - Category: node
    - Severity: error
    - Recovery: abort

    Args:
        node_id: The node's unique ID.
        node_name: Human-readable node name.
        key: The missing input key.
        expected_type: The expected type name for the input.
        contract_description: Optional contract description.

    Returns:
        BTError with code E2001.
    """
    return BTError(
        code="E2001",
        category="node",
        severity=Severity.ERROR,
        message=(
            f"Node '{node_name}' missing required input: "
            f"'{key}' (type: {expected_type})"
        ),
        context=ErrorContext(
            node_id=node_id,
            extra={
                "key": key,
                "expected_type": expected_type,
                "contract_description": contract_description,
                "node_name": node_name,
            },
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.ABORT,
            manual_steps=(
                f"Ensure upstream node writes '{key}' to blackboard "
                f"before this node runs"
            ),
        ),
        emit_event=True,
    )


def make_undeclared_read_error(
    node_id: str,
    node_name: str,
    key: str,
    declared_inputs: List[str],
) -> BTError:
    """Create E2002: Contract violation - undeclared read (warning).

    From errors.yaml E2002:
    - Category: node
    - Severity: warning (execution continues)
    - Recovery: skip

    Args:
        node_id: The node's unique ID.
        node_name: Human-readable node name.
        key: The key that was read but not declared.
        declared_inputs: List of declared input keys.

    Returns:
        BTError with code E2002.
    """
    return BTError(
        code="E2002",
        category="node",
        severity=Severity.WARNING,
        message=(
            f"Node '{node_name}' read undeclared key '{key}'. "
            f"Add to contract inputs or optional_inputs."
        ),
        context=ErrorContext(
            node_id=node_id,
            extra={
                "key": key,
                "declared_inputs": declared_inputs,
                "node_name": node_name,
            },
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.SKIP,
            manual_steps="Update NodeContract to declare this input",
        ),
        emit_event=True,
    )


def make_undeclared_write_error(
    node_id: str,
    node_name: str,
    key: str,
    declared_outputs: List[str],
) -> BTError:
    """Create E2003: Contract violation - undeclared write (warning).

    From errors.yaml E2003:
    - Category: node
    - Severity: warning (execution continues)
    - Recovery: skip

    Args:
        node_id: The node's unique ID.
        node_name: Human-readable node name.
        key: The key that was written but not declared.
        declared_outputs: List of declared output keys.

    Returns:
        BTError with code E2003.
    """
    return BTError(
        code="E2003",
        category="node",
        severity=Severity.WARNING,
        message=(
            f"Node '{node_name}' wrote undeclared key '{key}'. "
            f"Add to contract outputs."
        ),
        context=ErrorContext(
            node_id=node_id,
            extra={
                "key": key,
                "declared_outputs": declared_outputs,
                "node_name": node_name,
            },
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.SKIP,
            manual_steps="Update NodeContract to declare this output",
        ),
        emit_event=True,
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Core types
    "NodeContract",
    "ViolationType",
    "ContractViolationError",
    "ContractedNode",
    # Decorator
    "action_contract",
    "get_action_contract",
    # Error factories
    "make_missing_input_error",
    "make_undeclared_read_error",
    "make_undeclared_write_error",
]
