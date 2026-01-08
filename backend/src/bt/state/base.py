"""
BT State - Core Enums and Base Types

This module provides the foundational types for the Behavior Tree runtime:
- RunStatus: Execution status enum (FRESH, RUNNING, SUCCESS, FAILURE)
- NodeType: Classification of node behavior (COMPOSITE, DECORATOR, LEAF)
- BlackboardScope: Hierarchical scope levels (GLOBAL, TREE, SUBTREE)
- ErrorResult: Standard result wrapper for operations that can fail
- BTError: Structured error information matching errors.yaml schema

Reference:
- contracts/nodes.yaml - RunStatus enum specification
- contracts/errors.yaml - Error code patterns and base_error schema
- contracts/blackboard.yaml - BlackboardScope enumeration
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, TypeVar, Generic

T = TypeVar("T")


class RunStatus(IntEnum):
    """Result of a node tick.

    Values follow contracts/nodes.yaml specification:
    - FRESH (0): Node has never been ticked
    - RUNNING (1): Node is mid-execution, will continue next tick
    - SUCCESS (2): Node completed successfully
    - FAILURE (3): Node failed

    IntEnum allows numeric comparisons and ordering.
    """

    FRESH = 0
    RUNNING = 1
    SUCCESS = 2
    FAILURE = 3

    @classmethod
    def from_bool(cls, value: bool) -> "RunStatus":
        """Convert boolean to SUCCESS (True) or FAILURE (False).

        Useful for condition nodes that evaluate boolean expressions.

        Args:
            value: Boolean to convert

        Returns:
            SUCCESS if True, FAILURE if False

        Example:
            >>> RunStatus.from_bool(True)
            <RunStatus.SUCCESS: 2>
            >>> RunStatus.from_bool(False)
            <RunStatus.FAILURE: 3>
        """
        return cls.SUCCESS if value else cls.FAILURE

    def is_complete(self) -> bool:
        """Check if status indicates completion (SUCCESS or FAILURE).

        A complete status means the node has finished its work and won't
        need additional ticks for the current execution.

        Returns:
            True if SUCCESS or FAILURE, False otherwise

        Example:
            >>> RunStatus.SUCCESS.is_complete()
            True
            >>> RunStatus.RUNNING.is_complete()
            False
        """
        return self in (RunStatus.SUCCESS, RunStatus.FAILURE)

    def is_running(self) -> bool:
        """Check if status indicates ongoing execution.

        A RUNNING status means the node needs more ticks to complete.

        Returns:
            True if RUNNING, False otherwise

        Example:
            >>> RunStatus.RUNNING.is_running()
            True
            >>> RunStatus.FRESH.is_running()
            False
        """
        return self == RunStatus.RUNNING


class NodeType(str, Enum):
    """Classification of node behavior.

    From contracts/nodes.yaml:
    - COMPOSITE: Has 1+ children, orchestrates their execution (Sequence, Selector, Parallel)
    - DECORATOR: Has exactly 1 child, modifies its behavior (Timeout, Retry, Guard)
    - LEAF: Has 0 children, performs actual work (Action, Condition, LLMCall)

    Using str Enum for JSON serialization compatibility.
    """

    COMPOSITE = "composite"
    DECORATOR = "decorator"
    LEAF = "leaf"


class BlackboardScope(str, Enum):
    """Hierarchical blackboard scope levels.

    From contracts/blackboard.yaml:
    - GLOBAL: Persists across sessions, shared by all trees
    - TREE: Scoped to current tree execution, cleared on tree completion
    - SUBTREE: Scoped to a subtree, useful for isolated workflows

    Scope hierarchy: SUBTREE -> TREE -> GLOBAL
    Child scopes can read parent scope values but writes are local.
    """

    GLOBAL = "global"
    TREE = "tree"
    SUBTREE = "subtree"


class ErrorSeverity(str, Enum):
    """Error severity levels from contracts/errors.yaml.

    - WARNING: Operation can continue with degraded behavior
    - ERROR: Operation failed, may retry or recover
    - CRITICAL: System-level failure, cannot continue
    """

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Alias for backward compatibility
Severity = ErrorSeverity


class ErrorCategory(str, Enum):
    """Error categories with code ranges from contracts/errors.yaml.

    - BLACKBOARD (E1xxx): Errors related to blackboard state management
    - NODE (E2xxx): Errors in node execution and contracts
    - TREE (E3xxx): Tree-level execution errors
    - LOADER (E4xxx): Tree definition loading and parsing
    - SCRIPT (E5xxx): Lua script execution errors
    - ASYNC (E6xxx): Async operations and timeouts
    - SECURITY (E7xxx): Security violations and sandbox escapes
    - MERGE (E8xxx): Parallel merge conflicts
    """

    BLACKBOARD = "blackboard"
    NODE = "node"
    TREE = "tree"
    LOADER = "loader"
    SCRIPT = "script"
    ASYNC = "async"
    SECURITY = "security"
    MERGE = "merge"


class RecoveryAction(str, Enum):
    """Recovery action types from contracts/errors.yaml.

    - RETRY: Retry the operation (possibly with backoff)
    - ABORT: Stop execution, propagate failure
    - SKIP: Continue execution, ignore the error
    - ESCALATE: Emit event for higher-level handling
    - MANUAL: Requires manual intervention
    """

    RETRY = "retry"
    ABORT = "abort"
    SKIP = "skip"
    ESCALATE = "escalate"
    MANUAL = "manual"


@dataclass
class ErrorContext:
    """Structured context for debugging errors."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    node_id: Optional[str] = None
    node_path: Optional[List[str]] = None
    tree_id: Optional[str] = None
    tick_count: Optional[int] = None
    source_location: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryInfo:
    """How to handle or recover from an error."""

    action: RecoveryAction
    max_retries: int = 0
    backoff_ms: int = 0
    escalate_to: Optional[str] = None
    manual_steps: Optional[str] = None


@dataclass
class BTError:
    """Standard error type for BT operations.

    Matches contracts/errors.yaml base_error schema.

    Error codes follow pattern E[1-8][0-9]{3}:
    - E1xxx: Blackboard errors
    - E2xxx: Node errors
    - E3xxx: Tree errors
    - E4xxx: Loader errors
    - E5xxx: Script errors
    - E6xxx: Async errors
    - E7xxx: Security errors
    - E8xxx: Merge errors

    Example:
        >>> error = BTError(
        ...     code="E1001",
        ...     category="blackboard",
        ...     severity=Severity.ERROR,
        ...     message="Key 'context' is not registered",
        ...     context=ErrorContext(node_id="load-context"),
        ...     recovery=RecoveryInfo(action=RecoveryAction.ABORT),
        ... )
        >>> str(error)
        "[E1001] Key 'context' is not registered"
    """

    code: str
    category: str
    severity: Severity
    message: str
    context: ErrorContext
    recovery: RecoveryInfo
    emit_event: bool = True

    def __post_init__(self) -> None:
        """Validate error code format."""
        import re

        if not re.match(r"^E[1-8][0-9]{3}$", self.code):
            raise ValueError(
                f"Invalid error code '{self.code}': "
                f"must match pattern E[1-8]xxx (e.g., E1001)"
            )

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        context_dict: Dict[str, Any] = {
            "timestamp": self.context.timestamp.isoformat(),
        }
        if self.context.node_id is not None:
            context_dict["node_id"] = self.context.node_id
        if self.context.node_path is not None:
            context_dict["node_path"] = self.context.node_path
        if self.context.tree_id is not None:
            context_dict["tree_id"] = self.context.tree_id
        if self.context.tick_count is not None:
            context_dict["tick_count"] = self.context.tick_count
        if self.context.source_location is not None:
            context_dict["source_location"] = self.context.source_location
        context_dict.update(self.context.extra)

        recovery_dict: Dict[str, Any] = {
            "action": self.recovery.action.value,
        }
        if self.recovery.max_retries > 0:
            recovery_dict["max_retries"] = self.recovery.max_retries
        if self.recovery.backoff_ms > 0:
            recovery_dict["backoff_ms"] = self.recovery.backoff_ms
        if self.recovery.escalate_to is not None:
            recovery_dict["escalate_to"] = self.recovery.escalate_to
        if self.recovery.manual_steps is not None:
            recovery_dict["manual_steps"] = self.recovery.manual_steps

        return {
            "code": self.code,
            "category": self.category,
            "severity": self.severity.value,
            "message": self.message,
            "context": context_dict,
            "recovery": recovery_dict,
            "emit_event": self.emit_event,
        }


@dataclass
class ErrorResult(Generic[T]):
    """
    Standard result wrapper for operations that can fail.

    Usage:
        result = bb.set("key", value)
        if result.is_error:
            handle_error(result.error)
        else:
            use_value(result.value)
    """

    success: bool
    value: Optional[T] = None
    error: Optional[BTError] = None

    @property
    def is_error(self) -> bool:
        """Check if this result represents an error."""
        return not self.success

    @property
    def is_ok(self) -> bool:
        """Check if this result represents success."""
        return self.success

    @classmethod
    def ok(cls, value: T = None) -> "ErrorResult[T]":
        """Create a successful result."""
        return cls(success=True, value=value, error=None)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        category: str = "blackboard",
        severity: Severity = Severity.ERROR,
        context: Optional[ErrorContext] = None,
        recovery: Optional[RecoveryInfo] = None,
        emit_event: bool = True,
    ) -> "ErrorResult[T]":
        """Create a failure result with error details."""
        if context is None:
            context = ErrorContext()
        if recovery is None:
            recovery = RecoveryInfo(action=RecoveryAction.ABORT)

        error = BTError(
            code=code,
            category=category,
            severity=severity,
            message=message,
            context=context,
            recovery=recovery,
            emit_event=emit_event,
        )
        return cls(success=False, value=None, error=error)

    def unwrap(self) -> T:
        """Get the value or raise if error."""
        if self.is_error:
            raise ValueError(f"Cannot unwrap error result: {self.error}")
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get the value or return default if error."""
        if self.is_error:
            return default
        return self.value


# Common error factory functions for blackboard errors
def make_unregistered_key_error(
    key: str,
    available_keys: List[str],
    node_id: Optional[str] = None,
) -> BTError:
    """Create E1001: Key not registered error."""
    return BTError(
        code="E1001",
        category="blackboard",
        severity=Severity.ERROR,
        message=f"Blackboard key '{key}' is not registered. Register with bb.register('{key}', SchemaType) before use.",
        context=ErrorContext(
            node_id=node_id,
            extra={"key": key, "available_keys": available_keys},
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.ABORT,
            manual_steps="Add bb.register() call in node initialization or tree setup",
        ),
        emit_event=True,
    )


def make_schema_validation_error(
    key: str,
    expected_schema: str,
    actual_type: str,
    validation_error: str,
    value_preview: str,
) -> BTError:
    """Create E1002: Schema validation failed error."""
    return BTError(
        code="E1002",
        category="blackboard",
        severity=Severity.ERROR,
        message=f"Blackboard validation failed for key '{key}': {validation_error}",
        context=ErrorContext(
            extra={
                "key": key,
                "expected_schema": expected_schema,
                "actual_type": actual_type,
                "validation_error": validation_error,
                "value_preview": value_preview[:100] if value_preview else "",
            },
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.ABORT,
            manual_steps="Ensure value matches schema. Check Lua type coercion rules.",
        ),
        emit_event=True,
    )


def make_reserved_key_error(key: str, node_id: Optional[str] = None) -> BTError:
    """Create E1003: Reserved key access error."""
    return BTError(
        code="E1003",
        category="blackboard",
        severity=Severity.ERROR,
        message=f"Cannot write to reserved key '{key}'. Keys starting with '_' are system-reserved.",
        context=ErrorContext(
            node_id=node_id,
            extra={"key": key},
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.ABORT,
            manual_steps="Use a key without underscore prefix for user data",
        ),
        emit_event=True,
    )


def make_size_limit_error(
    current_size_bytes: int,
    limit_bytes: int,
    key: str,
    value_size_bytes: int,
) -> BTError:
    """Create E1004: Size limit exceeded error."""
    current_mb = current_size_bytes / (1024 * 1024)
    limit_mb = limit_bytes / (1024 * 1024)
    return BTError(
        code="E1004",
        category="blackboard",
        severity=Severity.ERROR,
        message=f"Blackboard size limit exceeded: {current_mb:.2f}MB > {limit_mb:.2f}MB limit",
        context=ErrorContext(
            extra={
                "current_size_mb": current_mb,
                "limit_mb": limit_mb,
                "key_being_written": key,
                "value_size_bytes": value_size_bytes,
            },
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.ABORT,
            manual_steps="Reduce data stored in blackboard. Consider persisting large data externally.",
        ),
        emit_event=True,
    )


def make_scope_chain_error(scope_name: str, parent_chain: List[str]) -> BTError:
    """Create E1005: Scope chain corruption error."""
    return BTError(
        code="E1005",
        category="blackboard",
        severity=Severity.CRITICAL,
        message=f"Blackboard scope chain corrupted: circular reference detected from '{scope_name}'",
        context=ErrorContext(
            extra={"scope_name": scope_name, "parent_chain": parent_chain},
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.ABORT,
            escalate_to="tree.critical.error",
            manual_steps="This is a bug in the runtime. Report with full context.",
        ),
        emit_event=True,
    )
