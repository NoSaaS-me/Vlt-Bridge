"""
BT State - Composite State Types

This module provides composite state types that combine multiple base states:
- OracleState: Complete state for Oracle agent (Identity + Conversation + Budget + Tool)
- ResearchState: Complete state for Research subtree (Identity + Budget + Research-specific)
- StateSliceFactory: Helper for creating state slices from composite states

Composite states are "God objects" that contain everything needed for a specific
agent/workflow. However, nodes should request ONLY the slices they need via contracts.

Reference:
- state-architecture.md - Composite state design
- contracts/nodes.yaml - Node contracts and input/output declarations
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import Field, ConfigDict

from .types import (
    BaseState,
    IdentityState,
    ConversationState,
    MessageState,
    BudgetState,
    ToolState,
    ToolCallState,
    ExecutionState,
)

T = TypeVar("T", bound=BaseState)


class OracleState(BaseState):
    """Complete state for Oracle agent execution.

    Combines: Identity + Conversation + Budget + Tool + Oracle-specific fields.

    This is the "God object" for Oracle - contains everything needed.
    Nodes should request ONLY the slices they need via contracts using
    OracleStateSlice factory methods.

    Attributes:
        # Identity fields
        user_id: User identifier
        project_id: Project identifier
        session_id: Session identifier
        tree_id: Current tree ID

        # Conversation fields
        messages: Messages in conversation
        context_tokens: Current token count
        max_context_tokens: Maximum context tokens
        turn_number: Current turn number

        # Budget fields
        token_budget: Maximum tokens allowed
        tokens_used: Tokens consumed
        iteration_budget: Maximum iterations
        iterations_used: Iterations consumed
        timeout_ms: Maximum execution time
        elapsed_ms: Time elapsed

        # Tool fields
        pending_tools: Tools waiting to execute
        running_tools: Tools currently running
        completed_tools: Tools that have finished
        failure_counts: Failure count per tool

        # Oracle-specific fields
        model: LLM model identifier
        provider: LLM provider
        streaming_enabled: Whether to stream responses
        current_query: Current user query
        streaming_buffer: Accumulated streaming content
        streaming_chunks: Individual streaming chunks
        final_response: Final agent response

    Example:
        >>> state = OracleState(
        ...     user_id="user-123",
        ...     model="claude-sonnet-4",
        ...     current_query="How does auth work?"
        ... )
        >>> identity = OracleStateSlice.identity(state)
        >>> budget = OracleStateSlice.budget(state)
    """

    model_config = ConfigDict(
        extra="allow",  # Allow plugin extensions
        validate_assignment=True,
    )

    # Identity fields
    user_id: str = Field(..., description="User identifier")
    project_id: str = Field(default="", description="Project identifier")
    session_id: str = Field(default="", description="Session identifier")
    tree_id: Optional[str] = Field(default=None, description="Current tree ID")

    # Conversation fields
    messages: List[MessageState] = Field(
        default_factory=list, description="Messages in conversation"
    )
    context_tokens: int = Field(default=0, ge=0, description="Current token count")
    max_context_tokens: int = Field(
        default=128000, gt=0, description="Maximum context tokens"
    )
    turn_number: int = Field(default=0, ge=0, description="Current turn number")

    # Budget fields
    token_budget: int = Field(default=100000, gt=0, description="Maximum tokens")
    tokens_used: int = Field(default=0, ge=0, description="Tokens consumed")
    iteration_budget: int = Field(default=100, gt=0, description="Maximum iterations")
    iterations_used: int = Field(default=0, ge=0, description="Iterations consumed")
    timeout_ms: int = Field(default=300000, gt=0, description="Max time in ms")
    elapsed_ms: float = Field(default=0.0, ge=0.0, description="Time elapsed in ms")

    # Tool fields
    pending_tools: List[ToolCallState] = Field(
        default_factory=list, description="Tools waiting to execute"
    )
    running_tools: List[ToolCallState] = Field(
        default_factory=list, description="Tools currently running"
    )
    completed_tools: List[ToolCallState] = Field(
        default_factory=list, description="Tools that have finished"
    )
    failure_counts: Dict[str, int] = Field(
        default_factory=dict, description="Failure count per tool"
    )

    # Oracle-specific fields
    model: str = Field(default="claude-sonnet-4", description="LLM model identifier")
    provider: str = Field(default="anthropic", description="LLM provider")
    streaming_enabled: bool = Field(default=True, description="Enable streaming")
    current_query: Optional[str] = Field(default=None, description="Current user query")
    streaming_buffer: str = Field(default="", description="Accumulated stream content")
    streaming_chunks: List[str] = Field(
        default_factory=list, description="Individual chunks"
    )
    final_response: Optional[str] = Field(default=None, description="Final response")

    # Computed properties
    @property
    def context_usage(self) -> float:
        """Context usage as ratio 0.0-1.0."""
        if self.max_context_tokens == 0:
            return 0.0
        return min(1.0, self.context_tokens / self.max_context_tokens)

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
            self.tokens_used >= self.token_budget
            or self.iterations_used >= self.iteration_budget
            or self.elapsed_ms >= self.timeout_ms
        )

    @property
    def has_pending_tools(self) -> bool:
        """Check if any tools are pending."""
        return len(self.pending_tools) > 0

    @property
    def has_running_tools(self) -> bool:
        """Check if any tools are running."""
        return len(self.running_tools) > 0


class OracleStateSlice:
    """Factory for creating state slices from OracleState.

    Nodes should use slices, not the full OracleState.
    This promotes loose coupling and makes dependencies explicit.

    Example:
        >>> oracle = OracleState(user_id="u1", tokens_used=5000)
        >>> identity = OracleStateSlice.identity(oracle)
        >>> identity.user_id
        'u1'
        >>> budget = OracleStateSlice.budget(oracle)
        >>> budget.tokens_used
        5000
    """

    @staticmethod
    def identity(state: OracleState) -> IdentityState:
        """Extract IdentityState from OracleState.

        Args:
            state: Full OracleState

        Returns:
            IdentityState containing only identity fields
        """
        return IdentityState(
            user_id=state.user_id,
            project_id=state.project_id,
            session_id=state.session_id,
            tree_id=state.tree_id,
            timestamp=state.timestamp,
        )

    @staticmethod
    def conversation(state: OracleState) -> ConversationState:
        """Extract ConversationState from OracleState.

        Args:
            state: Full OracleState

        Returns:
            ConversationState containing conversation fields
        """
        return ConversationState(
            messages=state.messages,
            context_tokens=state.context_tokens,
            max_context_tokens=state.max_context_tokens,
            turn_number=state.turn_number,
            timestamp=state.timestamp,
        )

    @staticmethod
    def budget(state: OracleState) -> BudgetState:
        """Extract BudgetState from OracleState.

        Args:
            state: Full OracleState

        Returns:
            BudgetState containing budget tracking fields
        """
        return BudgetState(
            token_budget=state.token_budget,
            tokens_used=state.tokens_used,
            iteration_budget=state.iteration_budget,
            iterations_used=state.iterations_used,
            timeout_ms=state.timeout_ms,
            elapsed_ms=state.elapsed_ms,
            timestamp=state.timestamp,
        )

    @staticmethod
    def tools(state: OracleState) -> ToolState:
        """Extract ToolState from OracleState.

        Args:
            state: Full OracleState

        Returns:
            ToolState containing tool tracking fields
        """
        return ToolState(
            pending_tools=state.pending_tools,
            running_tools=state.running_tools,
            completed_tools=state.completed_tools,
            failure_counts=state.failure_counts,
            timestamp=state.timestamp,
        )

    @staticmethod
    def merge_slice(state: OracleState, slice_state: BaseState) -> OracleState:
        """Merge a slice back into OracleState.

        Used to update OracleState after a node modifies a slice.

        Args:
            state: Original OracleState
            slice_state: Modified slice to merge back

        Returns:
            New OracleState with slice values merged

        Example:
            >>> oracle = OracleState(user_id="u1", tokens_used=0)
            >>> budget = OracleStateSlice.budget(oracle)
            >>> budget = budget.consume_tokens(1000)
            >>> oracle = OracleStateSlice.merge_slice(oracle, budget)
            >>> oracle.tokens_used
            1000
        """
        slice_dict = slice_state.model_dump()
        # Remove timestamp to use current time
        slice_dict.pop("timestamp", None)

        state_dict = state.model_dump()
        state_dict.update(slice_dict)
        state_dict["timestamp"] = datetime.utcnow()

        return OracleState(**state_dict)


class ResearchPhase(str, Enum):
    """Phases of a research workflow."""

    PLANNING = "planning"
    RESEARCHING = "researching"
    COMPRESSING = "compressing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearcherState(BaseState):
    """State for a single researcher in the research workflow.

    Tracks the progress of one researcher investigating a subtopic.

    Attributes:
        researcher_id: Unique ID for this researcher
        subtopic: The subtopic being researched
        queries: Search queries executed
        sources: Sources found and analyzed
        findings: Key findings from this researcher
        completed: Whether this researcher has finished
        error: Error message if researcher failed
    """

    researcher_id: str = Field(..., description="Unique researcher ID")
    subtopic: str = Field(..., description="Subtopic being researched")
    queries: List[str] = Field(
        default_factory=list, description="Search queries executed"
    )
    sources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Sources found"
    )
    findings: List[str] = Field(default_factory=list, description="Key findings")
    completed: bool = Field(default=False, description="Whether finished")
    error: Optional[str] = Field(default=None, description="Error if failed")


class ResearchState(BaseState):
    """Complete state for Research subtree.

    Combines: Identity + Budget + Research-specific fields.

    Used by the research workflow to track multi-researcher investigations.

    Attributes:
        # Identity fields
        user_id: User identifier
        project_id: Project identifier
        session_id: Session identifier

        # Budget fields
        token_budget: Maximum tokens
        tokens_used: Tokens consumed
        iteration_budget: Maximum iterations
        iterations_used: Iterations consumed
        timeout_ms: Maximum time
        elapsed_ms: Time elapsed

        # Research-specific fields
        query: Original research query
        depth: Research depth (quick, standard, thorough)
        phase: Current research phase
        progress_pct: Progress percentage (0-100)
        brief: Research brief/plan
        researchers: Individual researcher states
        compressed_findings: Findings after compression
        report: Final research report
        vault_path: Path if persisted to vault

    Example:
        >>> research = ResearchState(
        ...     user_id="u1",
        ...     query="How does authentication work in Vlt-Bridge?",
        ...     depth="standard"
        ... )
        >>> research.phase
        <ResearchPhase.PLANNING: 'planning'>
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Identity fields
    user_id: str = Field(..., description="User identifier")
    project_id: str = Field(default="", description="Project identifier")
    session_id: str = Field(default="", description="Session identifier")

    # Budget fields
    token_budget: int = Field(default=100000, gt=0, description="Maximum tokens")
    tokens_used: int = Field(default=0, ge=0, description="Tokens consumed")
    iteration_budget: int = Field(default=100, gt=0, description="Maximum iterations")
    iterations_used: int = Field(default=0, ge=0, description="Iterations consumed")
    timeout_ms: int = Field(default=600000, gt=0, description="Max time (10 min)")
    elapsed_ms: float = Field(default=0.0, ge=0.0, description="Time elapsed")

    # Research-specific fields
    query: str = Field(..., description="Original research query")
    depth: str = Field(
        default="standard",
        description="Research depth: quick, standard, thorough",
    )
    phase: ResearchPhase = Field(
        default=ResearchPhase.PLANNING, description="Current phase"
    )
    progress_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )

    # Artifacts
    brief: Optional[Dict[str, Any]] = Field(default=None, description="Research brief")
    researchers: List[ResearcherState] = Field(
        default_factory=list, description="Individual researchers"
    )
    compressed_findings: List[str] = Field(
        default_factory=list, description="Compressed findings"
    )
    report: Optional[Dict[str, Any]] = Field(default=None, description="Final report")

    # Output
    vault_path: Optional[str] = Field(
        default=None, description="Vault path if persisted"
    )

    # Computed properties
    @property
    def token_usage(self) -> float:
        """Token usage as ratio 0.0-1.0."""
        return min(1.0, self.tokens_used / self.token_budget)

    @property
    def any_budget_exceeded(self) -> bool:
        """Check if any budget is exceeded."""
        return (
            self.tokens_used >= self.token_budget
            or self.iterations_used >= self.iteration_budget
            or self.elapsed_ms >= self.timeout_ms
        )

    @property
    def completed_researchers(self) -> int:
        """Count of researchers that have completed."""
        return sum(1 for r in self.researchers if r.completed)

    @property
    def total_researchers(self) -> int:
        """Total number of researchers."""
        return len(self.researchers)

    def advance_phase(self, new_phase: ResearchPhase) -> "ResearchState":
        """Create new state with advanced phase.

        Args:
            new_phase: Phase to advance to

        Returns:
            New ResearchState with updated phase
        """
        return self.model_copy(
            update={
                "phase": new_phase,
                "timestamp": datetime.utcnow(),
            }
        )

    def update_progress(self, pct: float) -> "ResearchState":
        """Create new state with updated progress.

        Args:
            pct: Progress percentage (0-100)

        Returns:
            New ResearchState with updated progress
        """
        return self.model_copy(
            update={
                "progress_pct": min(100.0, max(0.0, pct)),
                "timestamp": datetime.utcnow(),
            }
        )


class ResearchStateSlice:
    """Factory for creating state slices from ResearchState.

    Similar to OracleStateSlice, provides focused state views.

    Example:
        >>> research = ResearchState(user_id="u1", query="test")
        >>> identity = ResearchStateSlice.identity(research)
        >>> identity.user_id
        'u1'
    """

    @staticmethod
    def identity(state: ResearchState) -> IdentityState:
        """Extract IdentityState from ResearchState.

        Args:
            state: Full ResearchState

        Returns:
            IdentityState with identity fields
        """
        return IdentityState(
            user_id=state.user_id,
            project_id=state.project_id,
            session_id=state.session_id,
            timestamp=state.timestamp,
        )

    @staticmethod
    def budget(state: ResearchState) -> BudgetState:
        """Extract BudgetState from ResearchState.

        Args:
            state: Full ResearchState

        Returns:
            BudgetState with budget fields
        """
        return BudgetState(
            token_budget=state.token_budget,
            tokens_used=state.tokens_used,
            iteration_budget=state.iteration_budget,
            iterations_used=state.iterations_used,
            timeout_ms=state.timeout_ms,
            elapsed_ms=state.elapsed_ms,
            timestamp=state.timestamp,
        )


def create_state_slice(
    source: BaseState,
    target_type: Type[T],
    field_mapping: Optional[Dict[str, str]] = None,
) -> T:
    """Generic factory for creating state slices.

    Creates a slice of one state type from another by copying
    common fields or using custom field mapping.

    Args:
        source: Source state to extract from
        target_type: Target state type to create
        field_mapping: Optional custom field mapping (target -> source)

    Returns:
        New instance of target_type with fields from source

    Example:
        >>> oracle = OracleState(user_id="u1", tokens_used=5000)
        >>> budget = create_state_slice(oracle, BudgetState)
        >>> budget.tokens_used
        5000
    """
    source_dict = source.model_dump()
    target_fields = target_type.model_fields.keys()

    if field_mapping:
        # Use custom mapping
        result_dict = {}
        for target_field, source_field in field_mapping.items():
            if source_field in source_dict:
                result_dict[target_field] = source_dict[source_field]
    else:
        # Copy fields with matching names
        result_dict = {
            field: source_dict[field]
            for field in target_fields
            if field in source_dict
        }

    # Update timestamp
    result_dict["timestamp"] = datetime.utcnow()

    return target_type(**result_dict)
