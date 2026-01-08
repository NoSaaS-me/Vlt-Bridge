"""
BT State - Pydantic State Models

This module provides the state type hierarchy for the Behavior Tree runtime:
- BaseState: Abstract base for all state types with timestamp
- IdentityState: User/project/session identification
- ConversationState: Messages and context tracking
- BudgetState: Token/iteration/time budget management
- ToolState: Tool call tracking (pending/running/completed)
- ExecutionState: BT-specific tick and path tracking

All state types use Pydantic v2 BaseModel for:
- Type validation at runtime
- JSON serialization/deserialization
- Schema generation for documentation

Reference:
- state-architecture.md - State type hierarchy design
- contracts/blackboard.yaml - Type coercion rules for Lua interop
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class BaseState(BaseModel):
    """Abstract base class for all state types.

    Provides:
    - Pydantic validation
    - JSON serialization
    - Timestamp tracking
    - Extra fields forbidden by default (catch typos)

    All state types should inherit from this class.

    Example:
        >>> class MyState(BaseState):
        ...     name: str
        ...     count: int = 0
        >>> state = MyState(name="test")
        >>> state.timestamp  # auto-populated
        datetime.datetime(...)
    """

    model_config = ConfigDict(
        extra="forbid",  # Catch typos in state keys
        validate_assignment=True,  # Validate on attribute assignment
        frozen=False,  # Allow mutation (override in specific states if needed)
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this state was created or last updated",
    )

    def merge(self, other: "BaseState") -> "BaseState":
        """Merge another state into this one (other takes precedence).

        Creates a new state instance with merged values.
        Fields from 'other' override fields from 'self'.

        Args:
            other: State to merge in (takes precedence)

        Returns:
            New state instance with merged values

        Example:
            >>> state1 = IdentityState(user_id="u1", project_id="p1")
            >>> state2 = IdentityState(user_id="u1", project_id="p2")
            >>> merged = state1.merge(state2)
            >>> merged.project_id
            'p2'
        """
        self_data = self.model_dump()
        other_data = other.model_dump()
        merged = {**self_data, **other_data}
        return self.__class__(**merged)

    def update_timestamp(self) -> "BaseState":
        """Create a copy with updated timestamp.

        Returns:
            New state instance with current timestamp
        """
        data = self.model_dump()
        data["timestamp"] = datetime.utcnow()
        return self.__class__(**data)


class IdentityState(BaseState):
    """Identity state shared across ALL surfaces.

    This is the root state that everything inherits from.
    Available to: Oracle, Research, Tools, Lua, LLM nodes, Plugins.

    Attributes:
        user_id: User identifier (required)
        project_id: Project identifier (optional, defaults to empty)
        session_id: Session identifier (optional, defaults to empty)
        tree_id: Current tree ID (optional, set by runtime)
        created_at: When this identity was established

    Example:
        >>> identity = IdentityState(
        ...     user_id="user-123",
        ...     project_id="vlt-bridge",
        ...     session_id="sess-456"
        ... )
    """

    user_id: str = Field(..., description="User identifier")
    project_id: str = Field(default="", description="Project identifier")
    session_id: str = Field(default="", description="Session identifier")
    tree_id: Optional[str] = Field(default=None, description="Current tree ID")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this identity was established",
    )


class MessageRole(str, Enum):
    """Role of a message in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageState(BaseState):
    """Single message in conversation.

    Attributes:
        role: Who sent this message (user, assistant, system, tool)
        content: Message content text
        name: Tool name if role is "tool"
        tool_call_id: ID linking to tool call if this is tool result
    """

    role: str = Field(..., description="Message role: user, assistant, system, tool")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(default=None, description="Tool name if role='tool'")
    tool_call_id: Optional[str] = Field(
        default=None, description="Tool call ID for tool results"
    )


class ConversationState(BaseState):
    """Conversation state - tracks messages and context.

    Available to: Oracle, Research, LLM nodes.

    Attributes:
        messages: List of messages in conversation
        context_tokens: Current token count in context
        max_context_tokens: Maximum allowed context tokens
        turn_number: Current conversation turn

    Properties:
        context_usage: Ratio of context_tokens to max (0.0-1.0)

    Example:
        >>> conv = ConversationState(
        ...     messages=[MessageState(role="user", content="Hello")],
        ...     context_tokens=10
        ... )
        >>> conv.context_usage
        7.8125e-05  # 10 / 128000
    """

    messages: List[MessageState] = Field(
        default_factory=list, description="Messages in conversation"
    )
    context_tokens: int = Field(
        default=0, ge=0, description="Current token count in context"
    )
    max_context_tokens: int = Field(
        default=128000, gt=0, description="Maximum allowed context tokens"
    )
    turn_number: int = Field(default=0, ge=0, description="Current conversation turn")

    @property
    def context_usage(self) -> float:
        """Context usage as ratio 0.0-1.0.

        Returns:
            Fraction of context window used
        """
        if self.max_context_tokens == 0:
            return 0.0
        return min(1.0, self.context_tokens / self.max_context_tokens)

    def add_message(self, role: str, content: str, **kwargs: Any) -> "ConversationState":
        """Create new state with message added.

        Args:
            role: Message role
            content: Message content
            **kwargs: Additional MessageState fields

        Returns:
            New ConversationState with message appended
        """
        new_message = MessageState(role=role, content=content, **kwargs)
        new_messages = self.messages + [new_message]
        return self.model_copy(
            update={
                "messages": new_messages,
                "timestamp": datetime.utcnow(),
            }
        )


class BudgetState(BaseState):
    """Budget tracking state for resource limits.

    Available to: Oracle, Research, LLM nodes, Watchdog.

    Tracks three budget dimensions:
    - Token budget: Maximum API tokens allowed
    - Iteration budget: Maximum loop iterations
    - Time budget: Maximum execution time

    Attributes:
        token_budget: Maximum tokens allowed
        tokens_used: Tokens consumed so far
        iteration_budget: Maximum iterations allowed
        iterations_used: Iterations consumed so far
        timeout_ms: Maximum execution time in milliseconds
        elapsed_ms: Time elapsed so far in milliseconds

    Properties:
        token_usage: Ratio of tokens used (0.0-1.0)
        iteration_usage: Ratio of iterations used (0.0-1.0)
        time_usage: Ratio of time used (0.0-1.0)
        any_budget_exceeded: True if any budget is exceeded

    Example:
        >>> budget = BudgetState(
        ...     token_budget=100000,
        ...     tokens_used=50000,
        ...     iteration_budget=100,
        ...     iterations_used=10
        ... )
        >>> budget.token_usage
        0.5
        >>> budget.any_budget_exceeded
        False
    """

    # Token budget
    token_budget: int = Field(default=100000, gt=0, description="Maximum tokens allowed")
    tokens_used: int = Field(default=0, ge=0, description="Tokens consumed so far")

    # Iteration budget
    iteration_budget: int = Field(
        default=100, gt=0, description="Maximum iterations allowed"
    )
    iterations_used: int = Field(
        default=0, ge=0, description="Iterations consumed so far"
    )

    # Time budget (in milliseconds for BT consistency)
    timeout_ms: int = Field(
        default=300000, gt=0, description="Maximum execution time in milliseconds"
    )
    elapsed_ms: float = Field(
        default=0.0, ge=0.0, description="Time elapsed so far in milliseconds"
    )

    @property
    def token_usage(self) -> float:
        """Token usage as ratio 0.0-1.0."""
        return min(1.0, self.tokens_used / self.token_budget)

    @property
    def iteration_usage(self) -> float:
        """Iteration usage as ratio 0.0-1.0."""
        return min(1.0, self.iterations_used / self.iteration_budget)

    @property
    def time_usage(self) -> float:
        """Time usage as ratio 0.0-1.0."""
        return min(1.0, self.elapsed_ms / self.timeout_ms)

    @property
    def any_budget_exceeded(self) -> bool:
        """Check if any budget is exceeded.

        Returns:
            True if tokens, iterations, or time budget is exceeded
        """
        return (
            self.tokens_used >= self.token_budget
            or self.iterations_used >= self.iteration_budget
            or self.elapsed_ms >= self.timeout_ms
        )

    @property
    def tokens_remaining(self) -> int:
        """Tokens remaining before budget exceeded."""
        return max(0, self.token_budget - self.tokens_used)

    @property
    def iterations_remaining(self) -> int:
        """Iterations remaining before budget exceeded."""
        return max(0, self.iteration_budget - self.iterations_used)

    @property
    def time_remaining_ms(self) -> float:
        """Time remaining before timeout in milliseconds."""
        return max(0.0, self.timeout_ms - self.elapsed_ms)

    def consume_tokens(self, tokens: int) -> "BudgetState":
        """Create new state with tokens consumed.

        Args:
            tokens: Number of tokens to consume

        Returns:
            New BudgetState with updated tokens_used
        """
        return self.model_copy(
            update={
                "tokens_used": self.tokens_used + tokens,
                "timestamp": datetime.utcnow(),
            }
        )

    def increment_iteration(self) -> "BudgetState":
        """Create new state with iteration incremented.

        Returns:
            New BudgetState with iterations_used += 1
        """
        return self.model_copy(
            update={
                "iterations_used": self.iterations_used + 1,
                "timestamp": datetime.utcnow(),
            }
        )


class ToolCallStatus(str, Enum):
    """Status of a tool call."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolCallState(BaseState):
    """State for a single tool call.

    Tracks the lifecycle of a tool invocation from pending to completion.

    Attributes:
        tool_id: Unique ID for this tool call
        name: Tool name
        arguments: Tool input arguments
        status: Current status (pending, running, success, failure, etc.)
        result: Tool result if completed successfully
        error: Error message if failed
        started_at: When execution started
        completed_at: When execution completed
        duration_ms: Execution duration in milliseconds

    Example:
        >>> tool = ToolCallState(
        ...     tool_id="call-123",
        ...     name="search_code",
        ...     arguments={"query": "authentication"}
        ... )
        >>> tool.status
        <ToolCallStatus.PENDING: 'pending'>
    """

    tool_id: str = Field(..., description="Unique ID for this tool call")
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Tool input arguments"
    )
    status: ToolCallStatus = Field(
        default=ToolCallStatus.PENDING, description="Current status"
    )
    result: Optional[str] = Field(default=None, description="Tool result if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    started_at: Optional[datetime] = Field(
        default=None, description="When execution started"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="When execution completed"
    )
    duration_ms: Optional[float] = Field(
        default=None, description="Execution duration in milliseconds"
    )

    def start(self) -> "ToolCallState":
        """Create new state marking tool as running.

        Returns:
            New ToolCallState with RUNNING status and started_at set
        """
        return self.model_copy(
            update={
                "status": ToolCallStatus.RUNNING,
                "started_at": datetime.utcnow(),
                "timestamp": datetime.utcnow(),
            }
        )

    def complete(self, result: str) -> "ToolCallState":
        """Create new state marking tool as successful.

        Args:
            result: Tool execution result

        Returns:
            New ToolCallState with SUCCESS status
        """
        now = datetime.utcnow()
        duration = None
        if self.started_at:
            duration = (now - self.started_at).total_seconds() * 1000
        return self.model_copy(
            update={
                "status": ToolCallStatus.SUCCESS,
                "result": result,
                "completed_at": now,
                "duration_ms": duration,
                "timestamp": now,
            }
        )

    def fail(self, error: str) -> "ToolCallState":
        """Create new state marking tool as failed.

        Args:
            error: Error message

        Returns:
            New ToolCallState with FAILURE status
        """
        now = datetime.utcnow()
        duration = None
        if self.started_at:
            duration = (now - self.started_at).total_seconds() * 1000
        return self.model_copy(
            update={
                "status": ToolCallStatus.FAILURE,
                "error": error,
                "completed_at": now,
                "duration_ms": duration,
                "timestamp": now,
            }
        )


class ToolState(BaseState):
    """Tool execution state - tracks all tool calls.

    Available to: Oracle, Tool nodes, Research.

    Groups tool calls by their current status for easy access.

    Attributes:
        pending_tools: Tools waiting to be executed
        running_tools: Tools currently executing
        completed_tools: Tools that have finished (success or failure)
        failure_counts: Count of failures per tool name (for retry logic)

    Properties:
        has_pending: True if any tools are pending
        has_running: True if any tools are running
        total_tools: Total number of tool calls tracked

    Example:
        >>> tool_state = ToolState(
        ...     pending_tools=[
        ...         ToolCallState(tool_id="1", name="search")
        ...     ]
        ... )
        >>> tool_state.has_pending
        True
    """

    pending_tools: List[ToolCallState] = Field(
        default_factory=list, description="Tools waiting to be executed"
    )
    running_tools: List[ToolCallState] = Field(
        default_factory=list, description="Tools currently executing"
    )
    completed_tools: List[ToolCallState] = Field(
        default_factory=list, description="Tools that have finished"
    )
    failure_counts: Dict[str, int] = Field(
        default_factory=dict, description="Failure count per tool name"
    )

    @property
    def has_pending(self) -> bool:
        """Check if any tools are pending execution."""
        return len(self.pending_tools) > 0

    @property
    def has_running(self) -> bool:
        """Check if any tools are currently running."""
        return len(self.running_tools) > 0

    @property
    def total_tools(self) -> int:
        """Total number of tool calls tracked."""
        return (
            len(self.pending_tools)
            + len(self.running_tools)
            + len(self.completed_tools)
        )

    def get_failure_count(self, tool_name: str) -> int:
        """Get failure count for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of times this tool has failed
        """
        return self.failure_counts.get(tool_name, 0)

    def add_pending(self, tool: ToolCallState) -> "ToolState":
        """Create new state with tool added to pending.

        Args:
            tool: Tool call to add

        Returns:
            New ToolState with tool in pending_tools
        """
        return self.model_copy(
            update={
                "pending_tools": self.pending_tools + [tool],
                "timestamp": datetime.utcnow(),
            }
        )

    def get_tool_by_id(self, tool_id: str) -> Optional[ToolCallState]:
        """Find a tool call by its ID.

        Args:
            tool_id: Tool call ID to find

        Returns:
            ToolCallState if found, None otherwise
        """
        for tool in self.pending_tools + self.running_tools + self.completed_tools:
            if tool.tool_id == tool_id:
                return tool
        return None


class ExecutionStatus(str, Enum):
    """Tree execution status."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"


class ExecutionState(BaseState):
    """BT execution state - tracks tick and path information.

    Available to: BT Runtime, Watchdog, Debug API.

    This is the BT-specific state that tracks execution progress.

    Attributes:
        tree_id: ID of the tree being executed
        tree_name: Human-readable tree name
        status: Current execution status
        tick_count: Number of ticks executed
        tick_budget: Maximum ticks before yielding
        node_path: Current path of executing nodes (IDs)
        start_time: When execution started
        last_tick_at: When last tick occurred
        running_node_id: Currently executing node ID

    Example:
        >>> exec_state = ExecutionState(
        ...     tree_id="oracle-agent",
        ...     tree_name="Oracle Agent",
        ...     tick_budget=1000
        ... )
        >>> exec_state.status
        <ExecutionStatus.IDLE: 'idle'>
    """

    tree_id: str = Field(default="", description="ID of the tree being executed")
    tree_name: str = Field(default="", description="Human-readable tree name")
    status: ExecutionStatus = Field(
        default=ExecutionStatus.IDLE, description="Current execution status"
    )

    # Tick tracking
    tick_count: int = Field(default=0, ge=0, description="Number of ticks executed")
    tick_budget: int = Field(
        default=1000, gt=0, description="Maximum ticks before yielding"
    )

    # Path tracking
    node_path: List[str] = Field(
        default_factory=list, description="Current path of executing node IDs"
    )
    running_node_id: Optional[str] = Field(
        default=None, description="Currently executing node ID"
    )

    # Timing
    start_time: Optional[datetime] = Field(
        default=None, description="When execution started"
    )
    last_tick_at: Optional[datetime] = Field(
        default=None, description="When last tick occurred"
    )

    @property
    def ticks_remaining(self) -> int:
        """Ticks remaining before budget exceeded."""
        return max(0, self.tick_budget - self.tick_count)

    @property
    def tick_budget_exceeded(self) -> bool:
        """Check if tick budget is exceeded."""
        return self.tick_count >= self.tick_budget

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time since execution started in milliseconds.

        Returns:
            0.0 if not started, otherwise milliseconds since start
        """
        if self.start_time is None:
            return 0.0
        delta = datetime.utcnow() - self.start_time
        return delta.total_seconds() * 1000

    def start_execution(self, tree_id: str, tree_name: str) -> "ExecutionState":
        """Create new state marking execution as started.

        Args:
            tree_id: Tree ID
            tree_name: Tree name

        Returns:
            New ExecutionState with RUNNING status
        """
        now = datetime.utcnow()
        return self.model_copy(
            update={
                "tree_id": tree_id,
                "tree_name": tree_name,
                "status": ExecutionStatus.RUNNING,
                "start_time": now,
                "tick_count": 0,
                "node_path": [],
                "timestamp": now,
            }
        )

    def record_tick(self, node_path: List[str]) -> "ExecutionState":
        """Create new state recording a tick.

        Args:
            node_path: Path of nodes executed in this tick

        Returns:
            New ExecutionState with tick_count incremented
        """
        now = datetime.utcnow()
        return self.model_copy(
            update={
                "tick_count": self.tick_count + 1,
                "node_path": node_path,
                "last_tick_at": now,
                "timestamp": now,
            }
        )

    def complete(self, success: bool) -> "ExecutionState":
        """Create new state marking execution as complete.

        Args:
            success: Whether execution succeeded

        Returns:
            New ExecutionState with SUCCESS or FAILURE status
        """
        status = ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILURE
        return self.model_copy(
            update={
                "status": status,
                "running_node_id": None,
                "timestamp": datetime.utcnow(),
            }
        )
