"""
BT State Management

This package provides the state type hierarchy for the Behavior Tree runtime:

Core Enums (base.py):
- RunStatus: Execution status (FRESH, RUNNING, SUCCESS, FAILURE)
- NodeType: Node classification (COMPOSITE, DECORATOR, LEAF)
- BlackboardScope: Scope levels (GLOBAL, TREE, SUBTREE)

Error Types (base.py):
- BTError: Structured error information
- ErrorResult: Result wrapper for fallible operations

State Types (types.py):
- BaseState: Abstract base with timestamp
- IdentityState: User/project/session identification
- ConversationState: Messages and context
- BudgetState: Token/iteration/time budgets
- ToolState: Tool call tracking
- ExecutionState: BT tick and path tracking

Composite States (composite.py):
- OracleState: Complete state for Oracle agent
- ResearchState: Complete state for Research workflow
- StateSlice factories for extracting focused views

Part of the BT Universal Runtime (spec 019).

Reference:
- specs/019-bt-universal-runtime/state-architecture.md
- specs/019-bt-universal-runtime/contracts/
"""

# Core enums and error types from base.py
from .base import (
    BlackboardScope,
    BTError,
    ErrorCategory,
    ErrorContext,
    ErrorResult,
    ErrorSeverity,
    NodeType,
    RecoveryAction,
    RecoveryInfo,
    RunStatus,
    Severity,
    make_reserved_key_error,
    make_schema_validation_error,
    make_size_limit_error,
    make_scope_chain_error,
    make_unregistered_key_error,
)

# TypedBlackboard implementation
from .blackboard import (
    MAX_KEY_LENGTH,
    MAX_SIZE_BYTES,
    RESERVED_PREFIX,
    TypedBlackboard,
    lua_to_python,
    python_to_lua,
)

# State types (tasks 0.1.2-0.1.7)
from .types import (
    BaseState,
    IdentityState,
    MessageRole,
    MessageState,
    ConversationState,
    BudgetState,
    ToolCallStatus,
    ToolCallState,
    ToolState,
    ExecutionStatus,
    ExecutionState,
)

# Composite state types (tasks 0.2.1-0.2.3)
from .composite import (
    OracleState,
    OracleStateSlice,
    ResearchPhase,
    ResearcherState,
    ResearchState,
    ResearchStateSlice,
    create_state_slice,
)

# Node contracts (tasks 0.4.1-0.4.6)
from .contracts import (
    NodeContract,
    ViolationType,
    ContractViolationError,
    ContractedNode,
    action_contract,
    get_action_contract,
    make_missing_input_error,
    make_undeclared_read_error,
    make_undeclared_write_error,
)

# State bridges (tasks 0.5.1-0.5.5)
from .bridges import (
    RuleContextBridge,
    LuaStateBridge,
)

# Parallel merge strategies (tasks 0.6.1-0.6.9)
from .merge import (
    MergeStrategy,
    MergeConflict,
    MergeResult,
    ParallelMerger,
    apply_merge_result_to_parent,
    make_merge_conflict_error,
    make_merge_type_mismatch_error,
)

__all__ = [
    # Core enums
    "RunStatus",
    "NodeType",
    "BlackboardScope",
    # Error types
    "BTError",
    "ErrorCategory",
    "ErrorContext",
    "ErrorResult",
    "ErrorSeverity",
    "Severity",
    "RecoveryAction",
    "RecoveryInfo",
    # Error factory functions
    "make_reserved_key_error",
    "make_schema_validation_error",
    "make_size_limit_error",
    "make_scope_chain_error",
    "make_unregistered_key_error",
    # Blackboard
    "TypedBlackboard",
    "MAX_SIZE_BYTES",
    "MAX_KEY_LENGTH",
    "RESERVED_PREFIX",
    # Type coercion
    "lua_to_python",
    "python_to_lua",
    # Base state types (0.1.2-0.1.7)
    "BaseState",
    "IdentityState",
    "MessageRole",
    "MessageState",
    "ConversationState",
    "BudgetState",
    "ToolCallStatus",
    "ToolCallState",
    "ToolState",
    "ExecutionStatus",
    "ExecutionState",
    # Composite state types (0.2.1-0.2.3)
    "OracleState",
    "OracleStateSlice",
    "ResearchPhase",
    "ResearcherState",
    "ResearchState",
    "ResearchStateSlice",
    "create_state_slice",
    # Node contracts (0.4.1-0.4.6)
    "NodeContract",
    "ViolationType",
    "ContractViolationError",
    "ContractedNode",
    "action_contract",
    "get_action_contract",
    "make_missing_input_error",
    "make_undeclared_read_error",
    "make_undeclared_write_error",
    # State bridges (0.5.1-0.5.5)
    "RuleContextBridge",
    "LuaStateBridge",
    # Parallel merge strategies (0.6.1-0.6.9)
    "MergeStrategy",
    "MergeConflict",
    "MergeResult",
    "ParallelMerger",
    "apply_merge_result_to_parent",
    "make_merge_conflict_error",
    "make_merge_type_mismatch_error",
]
