"""
BT Node Types

Contains all behavior tree node implementations:
- Base: BehaviorNode (abstract base for all nodes)
- Composites: Sequence, Selector, Parallel, ForEach
- Decorators: Timeout, Retry, Guard, Cooldown, etc.
- Leaves: Action, Condition, LLMCall, Script, SubtreeRef, Tool

Part of the BT Universal Runtime (spec 019).
"""

from .base import BehaviorNode, InvalidNodeIdError, NODE_ID_PATTERN

# Composite nodes (tasks 2.1.1-2.1.5)
from .composites import (
    CompositeNode,
    Sequence,
    Selector,
    Parallel,
    ParallelPolicy,
    ForEach,
)

# Decorator nodes (tasks 2.2.1-2.2.6)
from .decorators import (
    DecoratorNode,
    Timeout,
    Retry,
    Guard,
    GuardCondition,
    Cooldown,
    Inverter,
    AlwaysSucceed,
    AlwaysFail,
)

# Leaf nodes (tasks 2.3.1-2.3.4, 2.4.4-2.4.6)
from .leaves import (
    LeafNode,
    Action,
    Condition,
    SubtreeRef,
    Script,
    FunctionNotFoundError,
    TreeNotFoundError,
    CircularReferenceError,
    ActionFunction,
    ConditionFunction,
)

# LLM nodes (tasks 3.1.1-3.5.6)
from .llm import (
    LLMCallNode,
    PromptContent,
    LLMResponse,
    StreamChunk,
    LLMError,
    LLMErrorType,
    LLMClientProtocol,
    make_timeout_error,
    make_cancelled_error,
    make_llm_api_error,
)

# Tool nodes (tasks 4.1.1-4.3.8)
from .tools import (
    ToolLeaf,
    Tool,
    Oracle,
    CodeSearch,
    VaultSearch,
    interpolate_params,
    ToolNotFoundError,
    ToolTimeoutError,
    MissingToolParameterError,
    ToolResult,
)

__all__ = [
    # Base class
    "BehaviorNode",
    "InvalidNodeIdError",
    "NODE_ID_PATTERN",
    # Composite nodes
    "CompositeNode",
    "Sequence",
    "Selector",
    "Parallel",
    "ParallelPolicy",
    "ForEach",
    # Decorator nodes
    "DecoratorNode",
    "Timeout",
    "Retry",
    "Guard",
    "GuardCondition",
    "Cooldown",
    "Inverter",
    "AlwaysSucceed",
    "AlwaysFail",
    # Leaf nodes
    "LeafNode",
    "Action",
    "Condition",
    "SubtreeRef",
    "Script",
    "FunctionNotFoundError",
    "TreeNotFoundError",
    "CircularReferenceError",
    "ActionFunction",
    "ConditionFunction",
    # LLM nodes
    "LLMCallNode",
    "PromptContent",
    "LLMResponse",
    "StreamChunk",
    "LLMError",
    "LLMErrorType",
    "LLMClientProtocol",
    "make_timeout_error",
    "make_cancelled_error",
    "make_llm_api_error",
    # Tool nodes (MCP integration)
    "ToolLeaf",
    "Tool",
    "Oracle",
    "CodeSearch",
    "VaultSearch",
    "interpolate_params",
    "ToolNotFoundError",
    "ToolTimeoutError",
    "MissingToolParameterError",
    "ToolResult",
]
