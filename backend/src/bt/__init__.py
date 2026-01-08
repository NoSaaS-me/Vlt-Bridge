"""
Behavior Tree Universal Runtime

A universal composition pattern for agent systems.
Provides Lua DSL for tree definition, hierarchical blackboard state,
LLM-aware nodes with streaming support, and observability infrastructure.

Part of the BT Universal Runtime (spec 019).

Subpackages:
- bt.state: State types and TypedBlackboard
- bt.nodes: Node implementations (future)
- bt.lua: Lua DSL integration (future)
"""

# Core enums and error types
from .state import (
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
    TypedBlackboard,
    lua_to_python,
    python_to_lua,
)

# State types (tasks 0.1.2-0.1.7)
from .state import (
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
from .state import (
    OracleState,
    OracleStateSlice,
    ResearchPhase,
    ResearcherState,
    ResearchState,
    ResearchStateSlice,
    create_state_slice,
)

__all__ = [
    # Core enums
    "RunStatus",
    "NodeType",
    "BlackboardScope",
    # Blackboard
    "TypedBlackboard",
    "lua_to_python",
    "python_to_lua",
    # Error types
    "BTError",
    "ErrorCategory",
    "ErrorContext",
    "ErrorResult",
    "ErrorSeverity",
    "Severity",
    "RecoveryAction",
    "RecoveryInfo",
    # Base state types
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
    # Composite state types
    "OracleState",
    "OracleStateSlice",
    "ResearchPhase",
    "ResearcherState",
    "ResearchState",
    "ResearchStateSlice",
    "create_state_slice",
]
