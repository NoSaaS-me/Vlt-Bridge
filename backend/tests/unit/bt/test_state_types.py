"""
Unit tests for BT State Types

Tests for:
- base.py: RunStatus, NodeType, BlackboardScope, ErrorResult, BTError
- types.py: BaseState, IdentityState, ConversationState, BudgetState, ToolState, ExecutionState
- composite.py: OracleState, ResearchState, state slice factories

Reference: specs/019-bt-universal-runtime/tasks.md tasks 0.1.1-0.1.8, 0.2.1-0.2.4
"""

import pytest
from datetime import datetime, timedelta
from typing import Any, Dict

# Import base types
from backend.src.bt.state.base import (
    RunStatus,
    NodeType,
    BlackboardScope,
    ErrorSeverity,
    ErrorCategory,
    RecoveryAction,
    ErrorContext,
    RecoveryInfo,
    BTError,
    ErrorResult,
    Severity,  # Alias test
    make_unregistered_key_error,
    make_schema_validation_error,
    make_reserved_key_error,
    make_size_limit_error,
    make_scope_chain_error,
)

# Import state types
from backend.src.bt.state.types import (
    BaseState,
    IdentityState,
    MessageState,
    MessageRole,
    ConversationState,
    BudgetState,
    ToolCallStatus,
    ToolCallState,
    ToolState,
    ExecutionStatus,
    ExecutionState,
)

# Import composite types
from backend.src.bt.state.composite import (
    OracleState,
    OracleStateSlice,
    ResearchPhase,
    ResearcherState,
    ResearchState,
    ResearchStateSlice,
    create_state_slice,
)


# =============================================================================
# RunStatus Tests (0.1.1)
# =============================================================================


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_values_are_integers(self):
        """RunStatus values should be integers per contracts/nodes.yaml."""
        assert RunStatus.FRESH == 0
        assert RunStatus.RUNNING == 1
        assert RunStatus.SUCCESS == 2
        assert RunStatus.FAILURE == 3

    def test_from_bool_true(self):
        """from_bool(True) should return SUCCESS."""
        assert RunStatus.from_bool(True) == RunStatus.SUCCESS

    def test_from_bool_false(self):
        """from_bool(False) should return FAILURE."""
        assert RunStatus.from_bool(False) == RunStatus.FAILURE

    def test_is_complete_success(self):
        """SUCCESS should be complete."""
        assert RunStatus.SUCCESS.is_complete() is True

    def test_is_complete_failure(self):
        """FAILURE should be complete."""
        assert RunStatus.FAILURE.is_complete() is True

    def test_is_complete_running(self):
        """RUNNING should not be complete."""
        assert RunStatus.RUNNING.is_complete() is False

    def test_is_complete_fresh(self):
        """FRESH should not be complete."""
        assert RunStatus.FRESH.is_complete() is False

    def test_is_running_true(self):
        """RUNNING should return True for is_running."""
        assert RunStatus.RUNNING.is_running() is True

    def test_is_running_false_for_others(self):
        """Non-RUNNING statuses should return False for is_running."""
        assert RunStatus.FRESH.is_running() is False
        assert RunStatus.SUCCESS.is_running() is False
        assert RunStatus.FAILURE.is_running() is False

    def test_comparison(self):
        """RunStatus values should be comparable as integers."""
        assert RunStatus.FRESH < RunStatus.RUNNING
        assert RunStatus.RUNNING < RunStatus.SUCCESS
        assert RunStatus.SUCCESS < RunStatus.FAILURE


# =============================================================================
# NodeType Tests
# =============================================================================


class TestNodeType:
    """Tests for NodeType enum."""

    def test_values(self):
        """NodeType should have expected values."""
        assert NodeType.COMPOSITE == "composite"
        assert NodeType.DECORATOR == "decorator"
        assert NodeType.LEAF == "leaf"

    def test_is_string_enum(self):
        """NodeType should be a string enum for JSON serialization."""
        assert isinstance(NodeType.COMPOSITE.value, str)


# =============================================================================
# BlackboardScope Tests
# =============================================================================


class TestBlackboardScope:
    """Tests for BlackboardScope enum."""

    def test_values(self):
        """BlackboardScope should have expected values."""
        assert BlackboardScope.GLOBAL == "global"
        assert BlackboardScope.TREE == "tree"
        assert BlackboardScope.SUBTREE == "subtree"


# =============================================================================
# BTError Tests
# =============================================================================


class TestBTError:
    """Tests for BTError dataclass."""

    def test_valid_error_code(self):
        """Valid error codes should be accepted."""
        error = BTError(
            code="E1001",
            category="blackboard",
            severity=Severity.ERROR,
            message="Test error",
            context=ErrorContext(),
            recovery=RecoveryInfo(action=RecoveryAction.ABORT),
        )
        assert error.code == "E1001"

    def test_invalid_error_code_raises(self):
        """Invalid error codes should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid error code"):
            BTError(
                code="INVALID",
                category="blackboard",
                severity=Severity.ERROR,
                message="Test",
                context=ErrorContext(),
                recovery=RecoveryInfo(action=RecoveryAction.ABORT),
            )

    def test_error_code_range_validation(self):
        """Error codes must be E1xxx through E8xxx."""
        # E0xxx should be invalid
        with pytest.raises(ValueError):
            BTError(
                code="E0001",
                category="blackboard",
                severity=Severity.ERROR,
                message="Test",
                context=ErrorContext(),
                recovery=RecoveryInfo(action=RecoveryAction.ABORT),
            )

        # E9xxx should be invalid
        with pytest.raises(ValueError):
            BTError(
                code="E9001",
                category="blackboard",
                severity=Severity.ERROR,
                message="Test",
                context=ErrorContext(),
                recovery=RecoveryInfo(action=RecoveryAction.ABORT),
            )

    def test_str_representation(self):
        """String representation should include code and message."""
        error = BTError(
            code="E1001",
            category="blackboard",
            severity=Severity.ERROR,
            message="Key not registered",
            context=ErrorContext(),
            recovery=RecoveryInfo(action=RecoveryAction.ABORT),
        )
        assert str(error) == "[E1001] Key not registered"

    def test_to_dict(self):
        """to_dict should produce JSON-serializable output."""
        error = BTError(
            code="E1001",
            category="blackboard",
            severity=Severity.ERROR,
            message="Test error",
            context=ErrorContext(node_id="test-node"),
            recovery=RecoveryInfo(action=RecoveryAction.RETRY, max_retries=3),
        )
        d = error.to_dict()
        assert d["code"] == "E1001"
        assert d["category"] == "blackboard"
        assert d["severity"] == "error"
        assert d["context"]["node_id"] == "test-node"
        assert d["recovery"]["action"] == "retry"
        assert d["recovery"]["max_retries"] == 3


# =============================================================================
# ErrorResult Tests
# =============================================================================


class TestErrorResult:
    """Tests for ErrorResult wrapper."""

    def test_ok_result(self):
        """ok() should create successful result."""
        result = ErrorResult.ok({"data": "test"})
        assert result.success is True
        assert result.is_ok is True
        assert result.is_error is False
        assert result.value == {"data": "test"}
        assert result.error is None

    def test_failure_result(self):
        """failure() should create failed result."""
        result = ErrorResult.failure(
            code="E1001",
            message="Key not registered",
        )
        assert result.success is False
        assert result.is_ok is False
        assert result.is_error is True
        assert result.value is None
        assert result.error is not None
        assert result.error.code == "E1001"

    def test_unwrap_success(self):
        """unwrap() on success should return value."""
        result = ErrorResult.ok(42)
        assert result.unwrap() == 42

    def test_unwrap_failure_raises(self):
        """unwrap() on failure should raise."""
        result = ErrorResult.failure(code="E1001", message="Test")
        with pytest.raises(ValueError, match="Cannot unwrap error"):
            result.unwrap()

    def test_unwrap_or_success(self):
        """unwrap_or() on success should return value."""
        result = ErrorResult.ok(42)
        assert result.unwrap_or(0) == 42

    def test_unwrap_or_failure(self):
        """unwrap_or() on failure should return default."""
        result = ErrorResult.failure(code="E1001", message="Test")
        assert result.unwrap_or(0) == 0


# =============================================================================
# Error Factory Tests
# =============================================================================


class TestErrorFactories:
    """Tests for error factory functions."""

    def test_make_unregistered_key_error(self):
        """make_unregistered_key_error should create E1001."""
        error = make_unregistered_key_error(
            key="missing",
            available_keys=["exists"],
            node_id="test-node",
        )
        assert error.code == "E1001"
        assert "missing" in error.message
        assert error.context.extra["key"] == "missing"

    def test_make_schema_validation_error(self):
        """make_schema_validation_error should create E1002."""
        error = make_schema_validation_error(
            key="data",
            expected_schema="IdentityState",
            actual_type="dict",
            validation_error="missing required field",
            value_preview="{'incomplete': 'data'}",
        )
        assert error.code == "E1002"
        assert "data" in error.message

    def test_make_reserved_key_error(self):
        """make_reserved_key_error should create E1003."""
        error = make_reserved_key_error(key="_internal")
        assert error.code == "E1003"
        assert "_internal" in error.message

    def test_make_size_limit_error(self):
        """make_size_limit_error should create E1004."""
        error = make_size_limit_error(
            current_size_bytes=150 * 1024 * 1024,
            limit_bytes=100 * 1024 * 1024,
            key="large_data",
            value_size_bytes=50 * 1024 * 1024,
        )
        assert error.code == "E1004"
        assert "limit" in error.message.lower()

    def test_make_scope_chain_error(self):
        """make_scope_chain_error should create E1005."""
        error = make_scope_chain_error(
            scope_name="child",
            parent_chain=["root", "parent", "child"],
        )
        assert error.code == "E1005"
        assert error.severity == Severity.CRITICAL


# =============================================================================
# BaseState Tests (0.1.2)
# =============================================================================


class TestBaseState:
    """Tests for BaseState abstract base."""

    def test_timestamp_default(self):
        """BaseState should have auto-populated timestamp."""

        class TestState(BaseState):
            value: int

        before = datetime.utcnow()
        state = TestState(value=42)
        after = datetime.utcnow()

        assert before <= state.timestamp <= after

    def test_extra_fields_forbidden(self):
        """Extra fields should be rejected by default."""

        class TestState(BaseState):
            value: int

        with pytest.raises(Exception):  # Pydantic validation error
            TestState(value=42, extra_field="not allowed")

    def test_merge(self):
        """merge() should combine states with other taking precedence."""

        class TestState(BaseState):
            a: int
            b: int = 0

        state1 = TestState(a=1, b=2)
        state2 = TestState(a=1, b=3)
        merged = state1.merge(state2)

        assert merged.a == 1
        assert merged.b == 3

    def test_update_timestamp(self):
        """update_timestamp() should create copy with new timestamp."""

        class TestState(BaseState):
            value: int

        state = TestState(value=42)
        old_timestamp = state.timestamp

        # Small delay to ensure timestamp changes
        import time

        time.sleep(0.01)

        updated = state.update_timestamp()
        assert updated.timestamp > old_timestamp
        assert updated.value == 42


# =============================================================================
# IdentityState Tests (0.1.3)
# =============================================================================


class TestIdentityState:
    """Tests for IdentityState."""

    def test_required_user_id(self):
        """user_id should be required."""
        with pytest.raises(Exception):
            IdentityState()

    def test_optional_fields_default(self):
        """Optional fields should have defaults."""
        state = IdentityState(user_id="u1")
        assert state.project_id == ""
        assert state.session_id == ""
        assert state.tree_id is None

    def test_all_fields(self):
        """Should accept all fields."""
        state = IdentityState(
            user_id="u1",
            project_id="p1",
            session_id="s1",
            tree_id="t1",
        )
        assert state.user_id == "u1"
        assert state.project_id == "p1"
        assert state.session_id == "s1"
        assert state.tree_id == "t1"

    def test_json_serialization(self):
        """Should be JSON serializable."""
        state = IdentityState(user_id="u1")
        json_dict = state.model_dump()
        assert "user_id" in json_dict
        assert json_dict["user_id"] == "u1"


# =============================================================================
# ConversationState Tests (0.1.4)
# =============================================================================


class TestConversationState:
    """Tests for ConversationState."""

    def test_empty_messages_default(self):
        """Messages should default to empty list."""
        state = ConversationState()
        assert state.messages == []

    def test_context_usage_zero(self):
        """context_usage should be 0 when no tokens used."""
        state = ConversationState(context_tokens=0)
        assert state.context_usage == 0.0

    def test_context_usage_partial(self):
        """context_usage should calculate ratio correctly."""
        state = ConversationState(context_tokens=64000, max_context_tokens=128000)
        assert state.context_usage == 0.5

    def test_context_usage_capped(self):
        """context_usage should be capped at 1.0."""
        state = ConversationState(context_tokens=200000, max_context_tokens=128000)
        assert state.context_usage == 1.0

    def test_add_message(self):
        """add_message should return new state with message appended."""
        state = ConversationState()
        new_state = state.add_message(role="user", content="Hello")

        assert len(state.messages) == 0  # Original unchanged
        assert len(new_state.messages) == 1
        assert new_state.messages[0].role == "user"
        assert new_state.messages[0].content == "Hello"


# =============================================================================
# BudgetState Tests (0.1.5)
# =============================================================================


class TestBudgetState:
    """Tests for BudgetState."""

    def test_defaults(self):
        """Should have sensible defaults."""
        state = BudgetState()
        assert state.token_budget > 0
        assert state.tokens_used == 0
        assert state.iteration_budget > 0
        assert state.iterations_used == 0

    def test_token_usage(self):
        """token_usage should calculate ratio."""
        state = BudgetState(token_budget=100000, tokens_used=50000)
        assert state.token_usage == 0.5

    def test_iteration_usage(self):
        """iteration_usage should calculate ratio."""
        state = BudgetState(iteration_budget=100, iterations_used=25)
        assert state.iteration_usage == 0.25

    def test_any_budget_exceeded_false(self):
        """any_budget_exceeded should be False when under budget."""
        state = BudgetState(
            token_budget=100000,
            tokens_used=50000,
            iteration_budget=100,
            iterations_used=50,
        )
        assert state.any_budget_exceeded is False

    def test_any_budget_exceeded_tokens(self):
        """any_budget_exceeded should be True when tokens exceeded."""
        state = BudgetState(token_budget=100000, tokens_used=100000)
        assert state.any_budget_exceeded is True

    def test_any_budget_exceeded_iterations(self):
        """any_budget_exceeded should be True when iterations exceeded."""
        state = BudgetState(iteration_budget=100, iterations_used=100)
        assert state.any_budget_exceeded is True

    def test_consume_tokens(self):
        """consume_tokens should return new state with updated tokens."""
        state = BudgetState(tokens_used=0)
        new_state = state.consume_tokens(1000)

        assert state.tokens_used == 0  # Original unchanged
        assert new_state.tokens_used == 1000

    def test_increment_iteration(self):
        """increment_iteration should return new state with incremented count."""
        state = BudgetState(iterations_used=5)
        new_state = state.increment_iteration()

        assert state.iterations_used == 5  # Original unchanged
        assert new_state.iterations_used == 6

    def test_tokens_remaining(self):
        """tokens_remaining should calculate correctly."""
        state = BudgetState(token_budget=100000, tokens_used=30000)
        assert state.tokens_remaining == 70000


# =============================================================================
# ToolState Tests (0.1.6)
# =============================================================================


class TestToolCallState:
    """Tests for ToolCallState."""

    def test_creation(self):
        """Should create with required fields."""
        tool = ToolCallState(tool_id="t1", name="search")
        assert tool.tool_id == "t1"
        assert tool.name == "search"
        assert tool.status == ToolCallStatus.PENDING

    def test_start(self):
        """start() should update status to RUNNING."""
        tool = ToolCallState(tool_id="t1", name="search")
        started = tool.start()

        assert tool.status == ToolCallStatus.PENDING  # Original unchanged
        assert started.status == ToolCallStatus.RUNNING
        assert started.started_at is not None

    def test_complete(self):
        """complete() should update status to SUCCESS."""
        tool = ToolCallState(tool_id="t1", name="search").start()
        completed = tool.complete("result data")

        assert completed.status == ToolCallStatus.SUCCESS
        assert completed.result == "result data"
        assert completed.completed_at is not None
        assert completed.duration_ms is not None

    def test_fail(self):
        """fail() should update status to FAILURE."""
        tool = ToolCallState(tool_id="t1", name="search").start()
        failed = tool.fail("error message")

        assert failed.status == ToolCallStatus.FAILURE
        assert failed.error == "error message"


class TestToolState:
    """Tests for ToolState."""

    def test_empty_default(self):
        """Should default to empty lists."""
        state = ToolState()
        assert state.pending_tools == []
        assert state.running_tools == []
        assert state.completed_tools == []

    def test_has_pending(self):
        """has_pending should detect pending tools."""
        tool = ToolCallState(tool_id="t1", name="search")
        state = ToolState(pending_tools=[tool])
        assert state.has_pending is True

    def test_has_running(self):
        """has_running should detect running tools."""
        tool = ToolCallState(tool_id="t1", name="search").start()
        state = ToolState(running_tools=[tool])
        assert state.has_running is True

    def test_add_pending(self):
        """add_pending should add tool to pending list."""
        state = ToolState()
        tool = ToolCallState(tool_id="t1", name="search")
        new_state = state.add_pending(tool)

        assert len(state.pending_tools) == 0  # Original unchanged
        assert len(new_state.pending_tools) == 1

    def test_get_tool_by_id(self):
        """get_tool_by_id should find tool in any list."""
        tool = ToolCallState(tool_id="t1", name="search")
        state = ToolState(pending_tools=[tool])

        found = state.get_tool_by_id("t1")
        assert found is not None
        assert found.tool_id == "t1"

        assert state.get_tool_by_id("nonexistent") is None


# =============================================================================
# ExecutionState Tests (0.1.7)
# =============================================================================


class TestExecutionState:
    """Tests for ExecutionState."""

    def test_defaults(self):
        """Should have sensible defaults."""
        state = ExecutionState()
        assert state.status == ExecutionStatus.IDLE
        assert state.tick_count == 0
        assert state.tick_budget > 0

    def test_ticks_remaining(self):
        """ticks_remaining should calculate correctly."""
        state = ExecutionState(tick_budget=1000, tick_count=100)
        assert state.ticks_remaining == 900

    def test_tick_budget_exceeded(self):
        """tick_budget_exceeded should detect when exceeded."""
        state = ExecutionState(tick_budget=100, tick_count=100)
        assert state.tick_budget_exceeded is True

    def test_start_execution(self):
        """start_execution should initialize execution state."""
        state = ExecutionState()
        started = state.start_execution("tree-1", "My Tree")

        assert started.status == ExecutionStatus.RUNNING
        assert started.tree_id == "tree-1"
        assert started.tree_name == "My Tree"
        assert started.start_time is not None
        assert started.tick_count == 0

    def test_record_tick(self):
        """record_tick should increment tick count."""
        state = ExecutionState(tick_count=5)
        ticked = state.record_tick(["node1", "node2"])

        assert ticked.tick_count == 6
        assert ticked.node_path == ["node1", "node2"]

    def test_complete_success(self):
        """complete(True) should set SUCCESS status."""
        state = ExecutionState(status=ExecutionStatus.RUNNING)
        completed = state.complete(success=True)

        assert completed.status == ExecutionStatus.SUCCESS

    def test_complete_failure(self):
        """complete(False) should set FAILURE status."""
        state = ExecutionState(status=ExecutionStatus.RUNNING)
        completed = state.complete(success=False)

        assert completed.status == ExecutionStatus.FAILURE


# =============================================================================
# OracleState Tests (0.2.1)
# =============================================================================


class TestOracleState:
    """Tests for OracleState composite."""

    def test_required_user_id(self):
        """user_id should be required."""
        with pytest.raises(Exception):
            OracleState()

    def test_has_all_fields(self):
        """Should include fields from all composed states."""
        state = OracleState(user_id="u1")

        # Identity fields
        assert hasattr(state, "user_id")
        assert hasattr(state, "project_id")
        assert hasattr(state, "session_id")

        # Conversation fields
        assert hasattr(state, "messages")
        assert hasattr(state, "context_tokens")

        # Budget fields
        assert hasattr(state, "token_budget")
        assert hasattr(state, "tokens_used")

        # Tool fields
        assert hasattr(state, "pending_tools")
        assert hasattr(state, "running_tools")

        # Oracle-specific fields
        assert hasattr(state, "model")
        assert hasattr(state, "streaming_enabled")

    def test_computed_properties(self):
        """Computed properties should work."""
        state = OracleState(
            user_id="u1",
            token_budget=100000,
            tokens_used=50000,
        )
        assert state.token_usage == 0.5

    def test_extra_fields_allowed(self):
        """OracleState should allow extra fields for plugins."""
        state = OracleState(
            user_id="u1",
            plugin_data={"custom": "value"},  # Extra field
        )
        assert state.plugin_data == {"custom": "value"}


# =============================================================================
# OracleStateSlice Tests (0.2.3)
# =============================================================================


class TestOracleStateSlice:
    """Tests for OracleStateSlice factory."""

    def test_identity_slice(self):
        """identity() should extract IdentityState."""
        oracle = OracleState(
            user_id="u1",
            project_id="p1",
            session_id="s1",
        )
        identity = OracleStateSlice.identity(oracle)

        assert isinstance(identity, IdentityState)
        assert identity.user_id == "u1"
        assert identity.project_id == "p1"

    def test_conversation_slice(self):
        """conversation() should extract ConversationState."""
        msg = MessageState(role="user", content="Hello")
        oracle = OracleState(
            user_id="u1",
            messages=[msg],
            context_tokens=100,
        )
        conv = OracleStateSlice.conversation(oracle)

        assert isinstance(conv, ConversationState)
        assert len(conv.messages) == 1
        assert conv.context_tokens == 100

    def test_budget_slice(self):
        """budget() should extract BudgetState."""
        oracle = OracleState(
            user_id="u1",
            token_budget=100000,
            tokens_used=5000,
        )
        budget = OracleStateSlice.budget(oracle)

        assert isinstance(budget, BudgetState)
        assert budget.token_budget == 100000
        assert budget.tokens_used == 5000

    def test_tools_slice(self):
        """tools() should extract ToolState."""
        tool = ToolCallState(tool_id="t1", name="search")
        oracle = OracleState(
            user_id="u1",
            pending_tools=[tool],
        )
        tools = OracleStateSlice.tools(oracle)

        assert isinstance(tools, ToolState)
        assert len(tools.pending_tools) == 1

    def test_merge_slice(self):
        """merge_slice() should update OracleState from slice."""
        oracle = OracleState(user_id="u1", tokens_used=0)
        budget = OracleStateSlice.budget(oracle)
        budget = budget.consume_tokens(5000)

        merged = OracleStateSlice.merge_slice(oracle, budget)
        assert merged.tokens_used == 5000


# =============================================================================
# ResearchState Tests (0.2.2)
# =============================================================================


class TestResearchState:
    """Tests for ResearchState composite."""

    def test_required_fields(self):
        """user_id and query should be required."""
        with pytest.raises(Exception):
            ResearchState(user_id="u1")  # Missing query

    def test_defaults(self):
        """Should have sensible defaults."""
        state = ResearchState(user_id="u1", query="test query")
        assert state.depth == "standard"
        assert state.phase == ResearchPhase.PLANNING
        assert state.progress_pct == 0.0

    def test_advance_phase(self):
        """advance_phase should update phase."""
        state = ResearchState(user_id="u1", query="test")
        advanced = state.advance_phase(ResearchPhase.RESEARCHING)

        assert state.phase == ResearchPhase.PLANNING  # Original unchanged
        assert advanced.phase == ResearchPhase.RESEARCHING

    def test_update_progress(self):
        """update_progress should update percentage."""
        state = ResearchState(user_id="u1", query="test")
        updated = state.update_progress(50.0)

        assert updated.progress_pct == 50.0

    def test_progress_capped(self):
        """Progress should be capped at 0-100."""
        state = ResearchState(user_id="u1", query="test")

        # Cap at 100
        updated = state.update_progress(150.0)
        assert updated.progress_pct == 100.0

        # Cap at 0
        updated = state.update_progress(-10.0)
        assert updated.progress_pct == 0.0


class TestResearcherState:
    """Tests for ResearcherState."""

    def test_creation(self):
        """Should create with required fields."""
        researcher = ResearcherState(
            researcher_id="r1",
            subtopic="Authentication",
        )
        assert researcher.researcher_id == "r1"
        assert researcher.subtopic == "Authentication"
        assert researcher.completed is False


# =============================================================================
# Generic State Slice Factory Tests (0.2.3)
# =============================================================================


class TestCreateStateSlice:
    """Tests for create_state_slice generic factory."""

    def test_basic_slice(self):
        """Should copy matching fields."""
        oracle = OracleState(
            user_id="u1",
            project_id="p1",
            token_budget=50000,
            tokens_used=10000,
        )
        budget = create_state_slice(oracle, BudgetState)

        assert isinstance(budget, BudgetState)
        assert budget.token_budget == 50000
        assert budget.tokens_used == 10000

    def test_identity_slice(self):
        """Should work for IdentityState."""
        research = ResearchState(
            user_id="u1",
            project_id="p1",
            query="test",
        )
        identity = create_state_slice(research, IdentityState)

        assert isinstance(identity, IdentityState)
        assert identity.user_id == "u1"
        assert identity.project_id == "p1"


# =============================================================================
# JSON Serialization Tests (0.1.8)
# =============================================================================


class TestJSONSerialization:
    """Tests for JSON serialization of all state types."""

    def test_identity_state_serialization(self):
        """IdentityState should serialize to JSON."""
        state = IdentityState(user_id="u1", project_id="p1")
        json_dict = state.model_dump()
        restored = IdentityState(**json_dict)
        assert restored.user_id == "u1"

    def test_conversation_state_serialization(self):
        """ConversationState should serialize to JSON."""
        msg = MessageState(role="user", content="Hello")
        state = ConversationState(messages=[msg])
        json_dict = state.model_dump()
        restored = ConversationState(**json_dict)
        assert len(restored.messages) == 1

    def test_budget_state_serialization(self):
        """BudgetState should serialize to JSON."""
        state = BudgetState(token_budget=100000, tokens_used=5000)
        json_dict = state.model_dump()
        restored = BudgetState(**json_dict)
        assert restored.tokens_used == 5000

    def test_tool_state_serialization(self):
        """ToolState should serialize to JSON."""
        tool = ToolCallState(tool_id="t1", name="search")
        state = ToolState(pending_tools=[tool])
        json_dict = state.model_dump()
        restored = ToolState(**json_dict)
        assert len(restored.pending_tools) == 1

    def test_execution_state_serialization(self):
        """ExecutionState should serialize to JSON."""
        state = ExecutionState(tree_id="t1", tick_count=10)
        json_dict = state.model_dump()
        restored = ExecutionState(**json_dict)
        assert restored.tick_count == 10

    def test_oracle_state_serialization(self):
        """OracleState should serialize to JSON."""
        state = OracleState(
            user_id="u1",
            model="claude-sonnet-4",
            tokens_used=5000,
        )
        json_dict = state.model_dump()
        restored = OracleState(**json_dict)
        assert restored.tokens_used == 5000

    def test_research_state_serialization(self):
        """ResearchState should serialize to JSON."""
        state = ResearchState(
            user_id="u1",
            query="test query",
            phase=ResearchPhase.RESEARCHING,
        )
        json_dict = state.model_dump()
        restored = ResearchState(**json_dict)
        assert restored.phase == ResearchPhase.RESEARCHING
