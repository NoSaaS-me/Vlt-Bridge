# State Architecture: Unified State Management for BT Runtime

## Problem Statement

The current system has fragmented state management:

| Surface | State Location | Type Safety | Interop |
|---------|---------------|-------------|---------|
| BT Nodes | `Blackboard` (Dict[str, Any]) | None | Poor |
| Plugin Rules | `RuleContext` (typed dataclasses) | Strong | None with BT |
| Oracle Agent | `AgentState`, `OracleContext` | Partial | None with BT |
| Research | `ResearchState` dataclass | Strong | None with BT |
| Tools | Function parameters | Strong | None with BT |
| Lua Scripts | `context` table (mirrors RuleContext) | None | Read-only |

**Core Issues**:
1. No shared type system across surfaces
2. Blackboard has no schema enforcement
3. RuleContext and Blackboard are completely disconnected
4. Each surface reinvents state management
5. No contracts between state producers and consumers

---

## Design Goals

1. **Type Safety**: Runtime validation with clear error messages
2. **Interoperability**: All surfaces share common state types
3. **Extensibility**: Add new state without breaking existing code
4. **Contracts**: Nodes declare inputs/outputs explicitly
5. **Inheritance**: Base state types extended by specific concerns
6. **Performance**: Lazy validation, efficient access patterns

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           State Type Hierarchy                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BaseState (Protocol)                                                        │
│      │                                                                       │
│      ├── IdentityState          user_id, project_id, session_id             │
│      │                                                                       │
│      ├── ConversationState      messages, context_tokens                     │
│      │       │                                                               │
│      │       └── OracleState    + model, streaming_buffer                    │
│      │                                                                       │
│      ├── BudgetState            token_budget, tokens_used, iteration_*       │
│      │                                                                       │
│      ├── ToolState              pending_tools, completed_tools               │
│      │                                                                       │
│      └── ExecutionState         status, current_node, tick_count             │
│                                                                              │
│  CompositeState = IdentityState + ConversationState + BudgetState + ...     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Typed Blackboard                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TypedBlackboard                                                             │
│      │                                                                       │
│      ├── _schemas: Dict[str, Type[BaseModel]]   # Key → Pydantic model      │
│      ├── _data: Dict[str, BaseModel]            # Key → Validated instance  │
│      ├── _parent: Optional[TypedBlackboard]     # Scope hierarchy           │
│      │                                                                       │
│      ├── register_schema(key, model_type)       # Declare expected type     │
│      ├── get[T](key) -> T                       # Type-safe retrieval       │
│      ├── set(key, value)                        # Validates against schema  │
│      └── create_scope() -> TypedBlackboard      # Child scope               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Node Contracts                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  @dataclass                                                                  │
│  class NodeContract:                                                         │
│      inputs: Dict[str, Type[BaseModel]]    # Required state to read         │
│      outputs: Dict[str, Type[BaseModel]]   # State this node produces       │
│      optional: Dict[str, Type[BaseModel]]  # Optional inputs                │
│                                                                              │
│  class ActionNode(Leaf):                                                     │
│      @classmethod                                                            │
│      def contract(cls) -> NodeContract:                                      │
│          return NodeContract(                                                │
│              inputs={"identity": IdentityState},                             │
│              outputs={"tool_result": ToolResultState},                       │
│          )                                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core State Types

### Base Protocol

```python
# backend/src/bt/state/base.py
from typing import Protocol, TypeVar, Generic, Any, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar("T", bound="BaseState")


class BaseState(BaseModel):
    """Base class for all state types.

    Provides:
    - Pydantic validation
    - JSON serialization
    - Immutable by default (frozen=True for critical state)
    - Merge support for state composition
    """

    class Config:
        extra = "forbid"  # Catch typos in state keys

    def merge(self, other: "BaseState") -> "BaseState":
        """Merge another state into this one (other takes precedence)."""
        return self.__class__(**{**self.model_dump(), **other.model_dump()})
```

### Identity State (Shared Everywhere)

```python
# backend/src/bt/state/identity.py
from pydantic import Field
from .base import BaseState


class IdentityState(BaseState):
    """Identity state shared across ALL surfaces.

    This is the root state that everything inherits from.
    Available to: Oracle, Research, Tools, Lua, LLM nodes, Plugins.
    """

    user_id: str = Field(..., description="User identifier")
    project_id: str = Field(default="", description="Project identifier")
    session_id: str = Field(default="", description="Session identifier")
    tree_id: Optional[str] = Field(default=None, description="Current tree ID")

    # Timestamps for debugging
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### Conversation State

```python
# backend/src/bt/state/conversation.py
from typing import List, Optional
from pydantic import Field
from .base import BaseState
from .identity import IdentityState


class MessageState(BaseState):
    """Single message in conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    name: Optional[str] = None  # Tool name if role="tool"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationState(IdentityState):
    """Conversation state - extends Identity.

    Available to: Oracle, Research, LLM nodes.
    """

    messages: List[MessageState] = Field(default_factory=list)
    context_tokens: int = Field(default=0, ge=0)
    max_context_tokens: int = Field(default=128000, gt=0)

    @property
    def context_usage(self) -> float:
        """Context usage as ratio 0.0-1.0."""
        if self.max_context_tokens == 0:
            return 0.0
        return min(1.0, self.context_tokens / self.max_context_tokens)
```

### Budget State

```python
# backend/src/bt/state/budget.py
from pydantic import Field
from .base import BaseState


class BudgetState(BaseState):
    """Budget tracking state.

    Available to: Oracle, Research, LLM nodes, Watchdog.
    """

    # Token budget
    token_budget: int = Field(default=100000, gt=0)
    tokens_used: int = Field(default=0, ge=0)

    # Iteration budget
    iteration_budget: int = Field(default=100, gt=0)
    iterations_used: int = Field(default=0, ge=0)

    # Time budget
    timeout_seconds: float = Field(default=300.0, gt=0)
    elapsed_seconds: float = Field(default=0.0, ge=0)

    @property
    def token_usage(self) -> float:
        """Token usage as ratio 0.0-1.0."""
        return min(1.0, self.tokens_used / self.token_budget)

    @property
    def iteration_usage(self) -> float:
        """Iteration usage as ratio 0.0-1.0."""
        return min(1.0, self.iterations_used / self.iteration_budget)

    @property
    def any_budget_exceeded(self) -> bool:
        """Check if any budget is exceeded."""
        return (
            self.tokens_used >= self.token_budget or
            self.iterations_used >= self.iteration_budget or
            self.elapsed_seconds >= self.timeout_seconds
        )
```

### Tool State

```python
# backend/src/bt/state/tool.py
from typing import List, Dict, Any, Optional
from pydantic import Field
from datetime import datetime
from .base import BaseState


class ToolCallState(BaseState):
    """State for a single tool call."""

    tool_id: str  # Unique ID for this call
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # pending, running, success, failure
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None


class ToolState(BaseState):
    """Tool execution state.

    Available to: Oracle, Tool nodes, Research.
    """

    pending_tools: List[ToolCallState] = Field(default_factory=list)
    running_tools: List[ToolCallState] = Field(default_factory=list)
    completed_tools: List[ToolCallState] = Field(default_factory=list)

    # Failure tracking for retry logic
    failure_counts: Dict[str, int] = Field(default_factory=dict)

    @property
    def has_pending(self) -> bool:
        return len(self.pending_tools) > 0

    @property
    def has_running(self) -> bool:
        return len(self.running_tools) > 0

    def get_failure_count(self, tool_name: str) -> int:
        return self.failure_counts.get(tool_name, 0)
```

### Execution State (BT-Specific)

```python
# backend/src/bt/state/execution.py
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import Field
from .base import BaseState


class ExecutionStatus(str, Enum):
    """Tree execution status."""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"


class NodeExecutionState(BaseState):
    """State for a single node's execution."""

    node_id: str
    node_type: str
    status: str = "fresh"  # fresh, running, success, failure
    tick_count: int = 0
    last_tick_ms: float = 0.0
    total_running_ms: float = 0.0
    error: Optional[str] = None


class ExecutionState(BaseState):
    """BT execution state.

    Available to: BT Runtime, Watchdog, Debug API.
    """

    tree_id: str
    tree_name: str
    status: ExecutionStatus = ExecutionStatus.IDLE

    # Tick tracking
    tick_count: int = Field(default=0, ge=0)
    tick_budget: int = Field(default=1000, gt=0)

    # Current execution path
    active_path: List[str] = Field(default_factory=list)  # Node IDs
    running_nodes: Dict[str, NodeExecutionState] = Field(default_factory=dict)

    # Stuck detection
    stuck_node_id: Optional[str] = None
    stuck_duration_ms: float = 0.0
```

---

## Composite State Types

### Oracle Composite State

```python
# backend/src/bt/state/oracle.py
from typing import Optional, List
from pydantic import Field
from .identity import IdentityState
from .conversation import ConversationState, MessageState
from .budget import BudgetState
from .tool import ToolState


class OracleState(ConversationState, BudgetState, ToolState):
    """Complete state for Oracle agent execution.

    Inherits: Identity → Conversation → Budget → Tool

    This is the "God object" for Oracle - contains everything needed.
    Nodes should request ONLY the slices they need via contracts.
    """

    # Oracle-specific
    model: str = Field(default="claude-sonnet-4")
    provider: str = Field(default="anthropic")
    streaming_enabled: bool = Field(default=True)

    # Current turn
    turn_number: int = Field(default=0, ge=0)
    current_query: Optional[str] = None

    # Streaming state
    streaming_buffer: str = Field(default="")
    streaming_chunks: List[str] = Field(default_factory=list)

    # Response
    final_response: Optional[str] = None

    class Config:
        extra = "allow"  # Allow plugin extensions


class OracleStateSlice:
    """Factory for creating state slices from OracleState.

    Nodes should use slices, not the full OracleState.
    """

    @staticmethod
    def identity(state: OracleState) -> IdentityState:
        return IdentityState(
            user_id=state.user_id,
            project_id=state.project_id,
            session_id=state.session_id,
            tree_id=state.tree_id,
        )

    @staticmethod
    def budget(state: OracleState) -> BudgetState:
        return BudgetState(
            token_budget=state.token_budget,
            tokens_used=state.tokens_used,
            iteration_budget=state.iteration_budget,
            iterations_used=state.iterations_used,
            timeout_seconds=state.timeout_seconds,
            elapsed_seconds=state.elapsed_seconds,
        )

    @staticmethod
    def tool(state: OracleState) -> ToolState:
        return ToolState(
            pending_tools=state.pending_tools,
            running_tools=state.running_tools,
            completed_tools=state.completed_tools,
            failure_counts=state.failure_counts,
        )
```

### Research Composite State

```python
# backend/src/bt/state/research.py
from typing import List, Optional, Dict, Any
from pydantic import Field
from enum import Enum
from .identity import IdentityState
from .budget import BudgetState


class ResearchPhase(str, Enum):
    PLANNING = "planning"
    RESEARCHING = "researching"
    COMPRESSING = "compressing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearcherState(BaseState):
    """State for a single researcher."""

    subtopic: str
    queries: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    completed: bool = False
    error: Optional[str] = None


class ResearchState(IdentityState, BudgetState):
    """Complete state for Research subtree.

    Inherits: Identity → Budget
    """

    # Input
    query: str
    depth: str = "standard"  # quick, standard, thorough

    # Phase tracking
    phase: ResearchPhase = ResearchPhase.PLANNING
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)

    # Artifacts
    brief: Optional[Dict[str, Any]] = None
    researchers: List[ResearcherState] = Field(default_factory=list)
    compressed_findings: List[str] = Field(default_factory=list)
    report: Optional[Dict[str, Any]] = None

    # Output
    vault_path: Optional[str] = None
```

---

## Typed Blackboard Implementation

```python
# backend/src/bt/state/blackboard.py
from typing import (
    Dict, Type, TypeVar, Optional, Any, Generic,
    get_type_hints, overload, Union
)
from pydantic import BaseModel, ValidationError
from .base import BaseState

T = TypeVar("T", bound=BaseModel)


class BlackboardKeyError(Exception):
    """Raised when accessing an unregistered key."""
    pass


class BlackboardValidationError(Exception):
    """Raised when value doesn't match schema."""
    pass


class TypedBlackboard:
    """Type-safe blackboard with schema enforcement.

    Unlike Dict[str, Any], this blackboard:
    1. Requires schema registration before use
    2. Validates values against schemas on write
    3. Returns typed values on read
    4. Supports scope hierarchy (parent chain)
    5. Tracks which keys were read/written (for contracts)

    Example:
        bb = TypedBlackboard()
        bb.register("identity", IdentityState)
        bb.register("budget", BudgetState)

        # Type-safe write (validates)
        bb.set("identity", IdentityState(user_id="u1", project_id="p1"))

        # Type-safe read
        identity = bb.get("identity", IdentityState)  # Returns IdentityState

        # Validation error on type mismatch
        bb.set("identity", {"wrong": "type"})  # Raises BlackboardValidationError
    """

    def __init__(
        self,
        parent: Optional["TypedBlackboard"] = None,
        scope_name: str = "root"
    ):
        self._parent = parent
        self._scope_name = scope_name
        self._schemas: Dict[str, Type[BaseModel]] = {}
        self._data: Dict[str, BaseModel] = {}
        self._reads: set[str] = set()  # Keys read this tick
        self._writes: set[str] = set()  # Keys written this tick

        # Inherit parent schemas
        if parent:
            self._schemas.update(parent._schemas)

    def register(self, key: str, schema: Type[T]) -> None:
        """Register a schema for a key.

        Args:
            key: Blackboard key
            schema: Pydantic model class for validation
        """
        self._schemas[key] = schema

    def register_many(self, schemas: Dict[str, Type[BaseModel]]) -> None:
        """Register multiple schemas at once."""
        self._schemas.update(schemas)

    @overload
    def get(self, key: str, schema: Type[T]) -> T: ...

    @overload
    def get(self, key: str, schema: Type[T], default: T) -> T: ...

    def get(
        self,
        key: str,
        schema: Type[T],
        default: Optional[T] = None
    ) -> Optional[T]:
        """Get a typed value from the blackboard.

        Args:
            key: Blackboard key
            schema: Expected type (for type narrowing)
            default: Default value if not found

        Returns:
            Typed value or default

        Raises:
            BlackboardKeyError: If key not registered and no default
        """
        self._reads.add(key)

        # Check local data first
        if key in self._data:
            value = self._data[key]
            if isinstance(value, schema):
                return value
            # Type mismatch - try to convert
            return schema.model_validate(value.model_dump())

        # Check parent scope
        if self._parent:
            return self._parent.get(key, schema, default)

        # Not found
        if default is not None:
            return default

        if key not in self._schemas:
            raise BlackboardKeyError(
                f"Key '{key}' not registered. "
                f"Available keys: {list(self._schemas.keys())}"
            )

        return None

    def set(self, key: str, value: Union[BaseModel, Dict[str, Any]]) -> None:
        """Set a value in the blackboard.

        Args:
            key: Blackboard key
            value: Value to set (must match registered schema)

        Raises:
            BlackboardKeyError: If key not registered
            BlackboardValidationError: If value doesn't match schema
        """
        if key not in self._schemas:
            raise BlackboardKeyError(
                f"Key '{key}' not registered. Call register() first."
            )

        schema = self._schemas[key]

        # Validate and convert
        try:
            if isinstance(value, dict):
                validated = schema.model_validate(value)
            elif isinstance(value, schema):
                validated = value
            else:
                validated = schema.model_validate(value.model_dump())
        except ValidationError as e:
            raise BlackboardValidationError(
                f"Value for '{key}' doesn't match schema {schema.__name__}: {e}"
            )

        self._data[key] = validated
        self._writes.add(key)

    def has(self, key: str) -> bool:
        """Check if key has a value (in this scope or parent)."""
        if key in self._data:
            return True
        if self._parent:
            return self._parent.has(key)
        return False

    def create_child_scope(self, scope_name: str) -> "TypedBlackboard":
        """Create a child scope that inherits from this blackboard.

        Child scope:
        - Inherits all schemas from parent
        - Can read parent values
        - Writes are local to child
        - Cleared when child scope ends
        """
        return TypedBlackboard(parent=self, scope_name=scope_name)

    def get_reads(self) -> set[str]:
        """Get keys that were read this tick (for contract validation)."""
        return self._reads.copy()

    def get_writes(self) -> set[str]:
        """Get keys that were written this tick (for contract validation)."""
        return self._writes.copy()

    def clear_access_tracking(self) -> None:
        """Clear read/write tracking for new tick."""
        self._reads.clear()
        self._writes.clear()

    def snapshot(self) -> Dict[str, Any]:
        """Create a JSON-serializable snapshot of all data."""
        result = {}
        if self._parent:
            result.update(self._parent.snapshot())
        for key, value in self._data.items():
            result[key] = value.model_dump()
        return result
```

---

## Node Contracts

```python
# backend/src/bt/state/contracts.py
from typing import Dict, Type, Set, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel
from .base import BaseState


@dataclass
class NodeContract:
    """Contract declaring a node's state requirements.

    Enables:
    1. Static validation at tree load time
    2. Runtime validation of actual access
    3. Documentation of data flow
    4. Dependency analysis for optimization
    """

    # Required inputs - node will fail if missing
    inputs: Dict[str, Type[BaseModel]] = field(default_factory=dict)

    # Optional inputs - node works without them
    optional_inputs: Dict[str, Type[BaseModel]] = field(default_factory=dict)

    # Outputs - state this node may produce
    outputs: Dict[str, Type[BaseModel]] = field(default_factory=dict)

    # Description for documentation
    description: str = ""

    def validate_inputs(self, blackboard: "TypedBlackboard") -> list[str]:
        """Validate all required inputs are present.

        Returns:
            List of missing input keys (empty if valid)
        """
        missing = []
        for key, schema in self.inputs.items():
            if not blackboard.has(key):
                missing.append(key)
        return missing

    def validate_access(
        self,
        reads: Set[str],
        writes: Set[str]
    ) -> list[str]:
        """Validate actual access matches contract.

        Returns:
            List of contract violations
        """
        violations = []

        # Check for undeclared reads
        declared_reads = set(self.inputs.keys()) | set(self.optional_inputs.keys())
        undeclared_reads = reads - declared_reads
        for key in undeclared_reads:
            violations.append(f"Read undeclared input: {key}")

        # Check for undeclared writes
        declared_writes = set(self.outputs.keys())
        undeclared_writes = writes - declared_writes
        for key in undeclared_writes:
            violations.append(f"Wrote undeclared output: {key}")

        return violations


class ContractedNode:
    """Mixin for nodes that declare contracts."""

    @classmethod
    def contract(cls) -> NodeContract:
        """Override to declare this node's state contract."""
        return NodeContract()

    def validate_contract(self, blackboard: "TypedBlackboard") -> None:
        """Validate contract before tick. Called by runtime."""
        contract = self.contract()
        missing = contract.validate_inputs(blackboard)
        if missing:
            raise ContractViolationError(
                f"{self.__class__.__name__} missing required inputs: {missing}"
            )


class ContractViolationError(Exception):
    """Raised when a node violates its contract."""
    pass
```

---

## Example: LLM Node with Contract

```python
# backend/src/bt/nodes/llm.py
from ..state.contracts import NodeContract, ContractedNode
from ..state.conversation import ConversationState
from ..state.budget import BudgetState
from ..state.blackboard import TypedBlackboard


class LLMStreamState(BaseState):
    """State produced by LLM node during streaming."""

    model: str
    streaming: bool = True
    buffer: str = ""
    chunks: list[str] = []
    tokens_used: int = 0
    complete: bool = False
    error: Optional[str] = None


class LLMCallNode(Leaf, ContractedNode):
    """LLM call node with explicit state contract."""

    @classmethod
    def contract(cls) -> NodeContract:
        return NodeContract(
            inputs={
                "conversation": ConversationState,  # Need conversation history
                "budget": BudgetState,              # Need budget tracking
            },
            optional_inputs={
                "system_prompt": str,  # Optional system prompt override
            },
            outputs={
                "llm_stream": LLMStreamState,       # Streaming state
                "budget": BudgetState,              # Updated budget
            },
            description="Makes LLM API call with streaming support"
        )

    def tick(self, ctx: TickContext) -> RunStatus:
        bb = ctx.blackboard

        # Contract guarantees these exist
        conversation = bb.get("conversation", ConversationState)
        budget = bb.get("budget", BudgetState)

        # Optional - may be None
        system_prompt = bb.get("system_prompt", str, default=None)

        # ... LLM call logic ...

        # Update outputs (contract declares we write these)
        bb.set("llm_stream", LLMStreamState(
            model=self.model,
            buffer=partial_response,
            tokens_used=tokens,
        ))

        bb.set("budget", BudgetState(
            **budget.model_dump(),
            tokens_used=budget.tokens_used + tokens,
        ))

        return RunStatus.RUNNING
```

---

## Example: Tool Node with Contract

```python
# backend/src/bt/nodes/tool.py
from ..state.contracts import NodeContract, ContractedNode
from ..state.tool import ToolState, ToolCallState
from ..state.identity import IdentityState


class ToolResultState(BaseState):
    """Result from tool execution."""

    tool_id: str
    tool_name: str
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None


class ToolExecuteNode(Leaf, ContractedNode):
    """Execute a tool with explicit contract."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    @classmethod
    def contract(cls) -> NodeContract:
        return NodeContract(
            inputs={
                "identity": IdentityState,  # Need user/project context
                "tool": ToolState,          # Need pending tool calls
            },
            outputs={
                "tool": ToolState,          # Updated tool state
                "tool_result": ToolResultState,  # Latest result
            },
            description="Executes a pending tool call"
        )

    def tick(self, ctx: TickContext) -> RunStatus:
        bb = ctx.blackboard

        identity = bb.get("identity", IdentityState)
        tool_state = bb.get("tool", ToolState)

        # Find pending tool
        pending = next(
            (t for t in tool_state.pending_tools if t.name == self.tool_name),
            None
        )

        if not pending:
            return RunStatus.FAILURE

        # Execute tool...
        result = await self._execute(pending, identity)

        # Update tool state
        bb.set("tool_result", result)

        return RunStatus.SUCCESS if result.success else RunStatus.FAILURE
```

---

## Integration with Existing Systems

### RuleContext Bridge

```python
# backend/src/bt/state/bridges.py
from .oracle import OracleState
from .conversation import ConversationState, MessageState
from .budget import BudgetState
from .tool import ToolState, ToolCallState
from ..plugins.context import RuleContext, TurnState, HistoryState


class RuleContextBridge:
    """Bridge between RuleContext and typed state.

    Enables existing plugin rules to work with new state system.
    """

    @staticmethod
    def from_rule_context(ctx: RuleContext) -> OracleState:
        """Convert RuleContext to OracleState."""
        return OracleState(
            user_id=ctx.user.id,
            project_id=ctx.project.id,

            # Budget from turn state
            tokens_used=int(ctx.turn.token_usage * 100000),
            iterations_used=ctx.turn.iteration_count,

            # Messages from history
            messages=[
                MessageState(role=m["role"], content=m["content"])
                for m in ctx.history.messages
            ],

            # Tools from history
            completed_tools=[
                ToolCallState(
                    tool_id=f"tool_{i}",
                    name=t.name,
                    arguments=t.arguments,
                    status="success" if t.success else "failure",
                    result=t.result,
                )
                for i, t in enumerate(ctx.history.tools)
            ],
            failure_counts=ctx.history.failures,

            turn_number=ctx.turn.number,
        )

    @staticmethod
    def to_rule_context(state: OracleState) -> RuleContext:
        """Convert OracleState to RuleContext for plugins."""
        # ... inverse conversion
```

### Lua Bridge

```python
# backend/src/bt/state/lua_bridge.py
from typing import Any, Dict
from .blackboard import TypedBlackboard


class LuaStateBridge:
    """Expose typed blackboard to Lua scripts.

    Converts between Pydantic models and Lua tables.
    """

    def __init__(self, blackboard: TypedBlackboard):
        self._bb = blackboard

    def to_lua_table(self) -> Dict[str, Any]:
        """Convert blackboard to Lua-compatible dict.

        Pydantic models become nested dicts.
        """
        return self._bb.snapshot()

    def from_lua_result(self, result: Dict[str, Any]) -> None:
        """Apply Lua script results back to blackboard.

        Only updates keys that Lua modified (tracks changes).
        """
        if "blackboard" in result:
            for key, value in result["blackboard"].items():
                if key in self._bb._schemas:
                    self._bb.set(key, value)
```

---

## Migration Path

### Phase 1: Add Types Alongside Existing

```python
# Keep existing Blackboard working
class Blackboard:
    # ... existing implementation ...

    def to_typed(self) -> TypedBlackboard:
        """Convert to typed blackboard (migration helper)."""
        tb = TypedBlackboard()
        # Register common schemas
        tb.register("identity", IdentityState)
        tb.register("budget", BudgetState)
        # Copy data with validation
        for key, value in self._data.items():
            if key in tb._schemas:
                tb.set(key, value)
        return tb
```

### Phase 2: Dual-Write

```python
# Nodes write to both during migration
def tick(self, ctx: TickContext) -> RunStatus:
    # Old style (keep working)
    ctx.blackboard.set("result", result_dict)

    # New style (add typed)
    if hasattr(ctx, 'typed_blackboard'):
        ctx.typed_blackboard.set("result", ResultState(**result_dict))
```

### Phase 3: Deprecate Untyped

```python
# Mark old methods deprecated
class Blackboard:
    @deprecated("Use TypedBlackboard.get() instead")
    def get(self, key: str, default: Any = None) -> Any:
        ...
```

---

---

## Consistency Model

### Single-Threaded Guarantee

The BT runtime uses **asyncio** (cooperative multitasking), not multiprocessing. This means:
- Only one tick executes at a time
- No true parallelism, no race conditions between ticks
- Quorum is NOT required for consistency

### Parallel Node Scope Isolation

**Problem**: Current Parallel ticks all children with the same blackboard. Last writer wins.

```python
# BAD: All children share same scope
for child in self._children:
    status = child.tick(context)  # Same context.blackboard for all!
```

**Solution**: Each parallel child gets an isolated child scope. Results merge at completion.

```python
# GOOD: Each child gets isolated scope
class ParallelNode(Composite):
    def tick(self, ctx: TickContext) -> RunStatus:
        child_results: List[TypedBlackboard] = []

        for child in self._children:
            # Create isolated child scope
            child_scope = ctx.blackboard.create_child_scope(
                scope_name=f"parallel_{child.id}"
            )
            child_ctx = ctx.with_blackboard(child_scope)

            # Child writes to its own scope
            status = child.tick(child_ctx)
            child_results.append(child_scope)

        # Merge results based on strategy
        self._merge_results(ctx.blackboard, child_results)
        return self._evaluate_policy()
```

### Merge Strategies

When parallel completes, child scope results must merge into parent. Strategies:

```python
class MergeStrategy(Enum):
    LAST_WINS = "last_wins"           # Default: last child's value wins
    FIRST_WINS = "first_wins"         # First child's value wins
    COLLECT = "collect"               # Collect into list
    MERGE_DICT = "merge_dict"         # Deep merge dictionaries
    FAIL_ON_CONFLICT = "fail"         # Raise error if conflict


@dataclass
class MergeConfig:
    """Per-key merge configuration."""
    key: str
    strategy: MergeStrategy


class ParallelMerger:
    """Merge parallel child scopes into parent."""

    def __init__(self, default_strategy: MergeStrategy = MergeStrategy.COLLECT):
        self.default_strategy = default_strategy
        self.key_strategies: Dict[str, MergeStrategy] = {}

    def configure(self, key: str, strategy: MergeStrategy) -> None:
        """Set merge strategy for a specific key."""
        self.key_strategies[key] = strategy

    def merge(
        self,
        parent: TypedBlackboard,
        children: List[TypedBlackboard]
    ) -> List[str]:
        """Merge child scopes into parent.

        Returns:
            List of conflict warnings (if any)
        """
        conflicts = []

        # Collect all written keys across children
        all_writes: Dict[str, List[Tuple[int, Any]]] = {}
        for i, child in enumerate(children):
            for key in child.get_writes():
                if key not in all_writes:
                    all_writes[key] = []
                all_writes[key].append((i, child.get(key, BaseState)))

        # Apply merge strategy per key
        for key, values in all_writes.items():
            strategy = self.key_strategies.get(key, self.default_strategy)

            if len(values) == 1:
                # No conflict - single writer
                parent.set(key, values[0][1])

            elif strategy == MergeStrategy.LAST_WINS:
                parent.set(key, values[-1][1])

            elif strategy == MergeStrategy.FIRST_WINS:
                parent.set(key, values[0][1])

            elif strategy == MergeStrategy.COLLECT:
                # Collect all values into a list
                collected = [v[1] for v in values]
                # If schema is a list type, extend; otherwise wrap
                if hasattr(parent._schemas.get(key), '__origin__'):
                    parent.set(key, collected)
                else:
                    conflicts.append(f"{key}: collected {len(values)} values")
                    parent.set(key, values[-1][1])  # Fallback

            elif strategy == MergeStrategy.MERGE_DICT:
                # Deep merge dictionaries
                merged = {}
                for _, v in values:
                    if hasattr(v, 'model_dump'):
                        merged.update(v.model_dump())
                    elif isinstance(v, dict):
                        merged.update(v)
                parent.set(key, merged)

            elif strategy == MergeStrategy.FAIL_ON_CONFLICT:
                conflicts.append(
                    f"{key}: conflict between {len(values)} writers"
                )
                # Don't write - leave parent unchanged

        return conflicts
```

### Lua DSL Configuration

```lua
-- Parallel with merge strategy
BT.parallel({
    policy = "require-all",
    merge_strategy = "collect",              -- Default for all keys
    merge = {                                -- Specific key strategies
        sources = "collect",
        summary = "last-wins",
        errors = "collect"
    }
}, {
    BT.subtree_ref("researcher-1"),
    BT.subtree_ref("researcher-2"),
    BT.subtree_ref("researcher-3")
})
```

### Conflict Detection

```python
class ConflictEvent(BaseState):
    """Emitted when parallel merge has conflicts."""
    parallel_node_id: str
    conflicts: List[str]
    resolution: str  # How it was resolved


# Emit ANS event on conflict
if conflicts:
    ctx.event_bus.emit(Event(
        type="tree.parallel.conflict",
        source="parallel_merger",
        payload=ConflictEvent(
            parallel_node_id=self.id,
            conflicts=conflicts,
            resolution=str(self.default_strategy)
        ).model_dump()
    ))
```

### Summary: No Quorum Needed

| Scenario | Consistency Mechanism | Quorum? |
|----------|----------------------|---------|
| Sequential ticks | Single-threaded, no conflicts | No |
| Parallel children | Scope isolation + merge | No |
| LLM streaming | Single writer per stream | No |
| Tool execution | Async but isolated | No |
| Hot reload | Sequential swap | No |
| Global persistence | SQLite + version check | No |

**We are resilient without quorum** because:
1. Asyncio provides sequential execution guarantees
2. Scope isolation prevents parallel children from stomping each other
3. Merge strategies handle result combination deterministically
4. Conflicts are detected and reported (not silently lost)

---

## Benefits

| Aspect | Before (Dict[str, Any]) | After (TypedBlackboard) |
|--------|------------------------|------------------------|
| Type Safety | None - runtime errors | Full - validation on write |
| Contracts | Implicit - hope nodes agree | Explicit - declared inputs/outputs |
| Interop | Poor - each surface different | Strong - shared state types |
| Extension | Risky - may break others | Safe - composition via inheritance |
| Debugging | Hard - unknown structure | Easy - schema + snapshot |
| Lua | Read-only, untyped | Read-write, typed via bridge |
| Documentation | Manual, outdated | Auto-generated from types |

---

## Files to Create

```
backend/src/bt/state/
├── __init__.py
├── base.py              # BaseState protocol
├── identity.py          # IdentityState
├── conversation.py      # ConversationState, MessageState
├── budget.py            # BudgetState
├── tool.py              # ToolState, ToolCallState
├── execution.py         # ExecutionState (BT-specific)
├── oracle.py            # OracleState composite
├── research.py          # ResearchState composite
├── blackboard.py        # TypedBlackboard implementation
├── contracts.py         # NodeContract, ContractedNode
└── bridges/
    ├── __init__.py
    ├── rule_context.py  # RuleContext ↔ typed state
    └── lua.py           # Lua table ↔ typed state
```

---

## References

- BehaviorTree.CPP Ports: https://www.behaviortree.dev/docs/tutorial-basics/tutorial_02_basic_ports/
- Unreal Engine Blackboard: https://dev.epicgames.com/documentation/en-us/unreal-engine/behavior-tree-in-unreal-engine---overview
- Pydantic Discriminated Unions: https://docs.pydantic.dev/latest/concepts/unions/
- Redux Toolkit Entity Adapters: https://redux-toolkit.js.org/api/createEntityAdapter
