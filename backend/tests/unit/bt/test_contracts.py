"""
Unit tests for NodeContract system.

Tests cover:
- NodeContract creation and validation
- validate_inputs() with missing/present inputs
- validate_access() with undeclared reads/writes
- ContractedNode mixin
- action_contract decorator
- all_input_keys property
- Error factory functions (E2001, E2002, E2003)

Part of the BT Universal Runtime (spec 019).
Implements test requirements from tasks 0.4.1-0.4.6.
"""

import pytest
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from backend.src.bt.state import (
    TypedBlackboard,
    RunStatus,
)
from backend.src.bt.state.contracts import (
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


# =============================================================================
# Test Schemas
# =============================================================================


class SessionId(BaseModel):
    """Test schema for session identification."""
    value: str


class UserId(BaseModel):
    """Test schema for user identification."""
    value: str


class HistoryLimit(BaseModel):
    """Test schema for history limit configuration."""
    max_items: int = 100


class ConversationContext(BaseModel):
    """Test schema for conversation context output."""
    session_id: str
    user_id: str
    turn_number: float
    messages: List[Dict[str, str]]


class TurnNumber(BaseModel):
    """Test schema for turn number output."""
    value: float


class SearchQuery(BaseModel):
    """Test schema for search query input."""
    query: str
    limit: int = 10


class SearchResults(BaseModel):
    """Test schema for search results output."""
    items: List[str]
    total_count: int


# =============================================================================
# NodeContract Creation Tests
# =============================================================================


class TestNodeContractCreation:
    """Tests for NodeContract dataclass initialization."""

    def test_create_empty_contract(self):
        """Can create an empty contract with defaults."""
        contract = NodeContract()

        assert contract.inputs == {}
        assert contract.optional_inputs == {}
        assert contract.outputs == {}
        assert contract.description == ""

    def test_create_contract_with_inputs(self):
        """Can create contract with required inputs."""
        contract = NodeContract(
            inputs={"session_id": SessionId, "user_id": UserId},
        )

        assert "session_id" in contract.inputs
        assert "user_id" in contract.inputs
        assert contract.inputs["session_id"] == SessionId

    def test_create_contract_with_optional_inputs(self):
        """Can create contract with optional inputs."""
        contract = NodeContract(
            optional_inputs={"limit": HistoryLimit},
        )

        assert "limit" in contract.optional_inputs
        assert contract.optional_inputs["limit"] == HistoryLimit

    def test_create_contract_with_outputs(self):
        """Can create contract with outputs."""
        contract = NodeContract(
            outputs={"context": ConversationContext, "turn": TurnNumber},
        )

        assert "context" in contract.outputs
        assert "turn" in contract.outputs

    def test_create_full_contract(self):
        """Can create contract with all fields."""
        contract = NodeContract(
            inputs={"session_id": SessionId},
            optional_inputs={"limit": HistoryLimit},
            outputs={"context": ConversationContext},
            description="Load conversation context",
        )

        assert "session_id" in contract.inputs
        assert "limit" in contract.optional_inputs
        assert "context" in contract.outputs
        assert contract.description == "Load conversation context"

    def test_overlapping_input_optional_raises(self):
        """Keys cannot be both required and optional input."""
        with pytest.raises(ValueError, match="both required and optional"):
            NodeContract(
                inputs={"key": SessionId},
                optional_inputs={"key": SessionId},
            )

    def test_overlapping_input_output_raises(self):
        """Keys cannot be both input and output."""
        with pytest.raises(ValueError, match="both input and output"):
            NodeContract(
                inputs={"key": SessionId},
                outputs={"key": ConversationContext},
            )

    def test_overlapping_optional_output_raises(self):
        """Keys cannot be both optional_input and output."""
        with pytest.raises(ValueError, match="both optional_input and output"):
            NodeContract(
                optional_inputs={"key": SessionId},
                outputs={"key": ConversationContext},
            )

    def test_non_basemodel_input_raises(self):
        """Input schema must be BaseModel subclass."""
        with pytest.raises(TypeError, match="BaseModel subclass"):
            NodeContract(inputs={"key": dict})

    def test_non_basemodel_optional_raises(self):
        """Optional input schema must be BaseModel subclass."""
        with pytest.raises(TypeError, match="BaseModel subclass"):
            NodeContract(optional_inputs={"key": str})

    def test_non_basemodel_output_raises(self):
        """Output schema must be BaseModel subclass."""
        with pytest.raises(TypeError, match="BaseModel subclass"):
            NodeContract(outputs={"key": list})


# =============================================================================
# all_input_keys Property Tests
# =============================================================================


class TestAllInputKeys:
    """Tests for all_input_keys property."""

    def test_empty_contract_returns_empty_set(self):
        """Empty contract returns empty set."""
        contract = NodeContract()
        assert contract.all_input_keys == set()

    def test_only_inputs_returns_input_keys(self):
        """Contract with only inputs returns input keys."""
        contract = NodeContract(
            inputs={"a": SessionId, "b": UserId},
        )
        assert contract.all_input_keys == {"a", "b"}

    def test_only_optional_returns_optional_keys(self):
        """Contract with only optional returns optional keys."""
        contract = NodeContract(
            optional_inputs={"c": HistoryLimit},
        )
        assert contract.all_input_keys == {"c"}

    def test_both_returns_union(self):
        """Contract with both returns union of keys."""
        contract = NodeContract(
            inputs={"a": SessionId, "b": UserId},
            optional_inputs={"c": HistoryLimit},
        )
        assert contract.all_input_keys == {"a", "b", "c"}

    def test_outputs_not_included(self):
        """Output keys are not included in all_input_keys."""
        contract = NodeContract(
            inputs={"a": SessionId},
            outputs={"out": ConversationContext},
        )
        assert "out" not in contract.all_input_keys
        assert contract.all_input_keys == {"a"}


# =============================================================================
# validate_inputs Tests
# =============================================================================


class TestValidateInputs:
    """Tests for validate_inputs() method."""

    def test_empty_contract_always_succeeds(self):
        """Empty contract (no required inputs) always succeeds."""
        contract = NodeContract()
        bb = TypedBlackboard()

        missing = contract.validate_inputs(bb)
        assert missing == []

    def test_all_inputs_present_succeeds(self):
        """All required inputs present returns empty list."""
        contract = NodeContract(
            inputs={"session_id": SessionId},
        )

        bb = TypedBlackboard()
        bb.register("session_id", SessionId)
        bb.set("session_id", {"value": "sess-123"})

        missing = contract.validate_inputs(bb)
        assert missing == []

    def test_missing_input_returns_key(self):
        """Missing required input returns the key name."""
        contract = NodeContract(
            inputs={"session_id": SessionId},
        )

        bb = TypedBlackboard()
        bb.register("session_id", SessionId)
        # Note: not setting the value

        missing = contract.validate_inputs(bb)
        assert "session_id" in missing

    def test_multiple_missing_inputs(self):
        """Multiple missing inputs returns all keys."""
        contract = NodeContract(
            inputs={"session_id": SessionId, "user_id": UserId},
        )

        bb = TypedBlackboard()
        bb.register("session_id", SessionId)
        bb.register("user_id", UserId)

        missing = contract.validate_inputs(bb)
        assert "session_id" in missing
        assert "user_id" in missing

    def test_optional_inputs_not_required(self):
        """Optional inputs do not cause validation failure."""
        contract = NodeContract(
            inputs={"session_id": SessionId},
            optional_inputs={"limit": HistoryLimit},
        )

        bb = TypedBlackboard()
        bb.register("session_id", SessionId)
        bb.register("limit", HistoryLimit)
        bb.set("session_id", {"value": "sess-123"})
        # Note: not setting limit (optional)

        missing = contract.validate_inputs(bb)
        assert missing == []

    def test_checks_parent_scope(self):
        """validate_inputs checks parent scope chain."""
        contract = NodeContract(
            inputs={"session_id": SessionId},
        )

        # Set value in parent scope
        parent = TypedBlackboard(scope_name="parent")
        parent.register("session_id", SessionId)
        parent.set("session_id", {"value": "sess-123"})

        # Child scope sees parent value
        child = parent.create_child_scope("child")
        missing = contract.validate_inputs(child)
        assert missing == []


# =============================================================================
# validate_access Tests
# =============================================================================


class TestValidateAccess:
    """Tests for validate_access() method."""

    def test_empty_contract_no_access_succeeds(self):
        """Empty contract with no access returns no violations."""
        contract = NodeContract()

        violations = contract.validate_access(
            reads=set(),
            writes=set(),
        )
        assert violations == []

    def test_declared_read_succeeds(self):
        """Reading a declared input is not a violation."""
        contract = NodeContract(
            inputs={"session_id": SessionId},
        )

        violations = contract.validate_access(
            reads={"session_id"},
            writes=set(),
        )
        assert violations == []

    def test_declared_optional_read_succeeds(self):
        """Reading a declared optional input is not a violation."""
        contract = NodeContract(
            optional_inputs={"limit": HistoryLimit},
        )

        violations = contract.validate_access(
            reads={"limit"},
            writes=set(),
        )
        assert violations == []

    def test_declared_write_succeeds(self):
        """Writing a declared output is not a violation."""
        contract = NodeContract(
            outputs={"context": ConversationContext},
        )

        violations = contract.validate_access(
            reads=set(),
            writes={"context"},
        )
        assert violations == []

    def test_undeclared_read_returns_e2002(self):
        """Reading undeclared key returns E2002 violation."""
        contract = NodeContract(
            inputs={"session_id": SessionId},
        )

        violations = contract.validate_access(
            reads={"session_id", "undeclared_key"},
            writes=set(),
        )

        assert len(violations) == 1
        assert "E2002" in violations[0]
        assert "undeclared_key" in violations[0]

    def test_undeclared_write_returns_e2003(self):
        """Writing undeclared key returns E2003 violation."""
        contract = NodeContract(
            outputs={"context": ConversationContext},
        )

        violations = contract.validate_access(
            reads=set(),
            writes={"context", "undeclared_output"},
        )

        assert len(violations) == 1
        assert "E2003" in violations[0]
        assert "undeclared_output" in violations[0]

    def test_multiple_violations(self):
        """Multiple violations are all reported."""
        contract = NodeContract(
            inputs={"session_id": SessionId},
            outputs={"context": ConversationContext},
        )

        violations = contract.validate_access(
            reads={"session_id", "undeclared_read"},
            writes={"context", "undeclared_write"},
        )

        assert len(violations) == 2
        # One E2002, one E2003
        codes = [v.split("]")[0] for v in violations]
        assert "[E2002" in codes
        assert "[E2003" in codes

    def test_internal_keys_ignored(self):
        """Keys starting with underscore are ignored."""
        contract = NodeContract()

        # Internal keys should not cause violations
        violations = contract.validate_access(
            reads={"_internal_read"},
            writes={"_internal_write"},
        )

        assert violations == []

    def test_includes_node_id_in_message(self):
        """Violation message includes node_id when provided."""
        contract = NodeContract()

        violations = contract.validate_access(
            reads={"undeclared"},
            writes=set(),
            node_id="my-test-node",
        )

        assert "my-test-node" in violations[0]


# =============================================================================
# ContractViolationError Tests
# =============================================================================


class TestContractViolationError:
    """Tests for ContractViolationError exception."""

    def test_missing_input_error(self):
        """Can create missing input violation error."""
        contract = NodeContract(inputs={"session_id": SessionId})

        error = ContractViolationError(
            violation_type=ViolationType.MISSING_INPUT,
            key="session_id",
            node_id="load-context",
            contract=contract,
        )

        assert error.violation_type == ViolationType.MISSING_INPUT
        assert error.key == "session_id"
        assert error.node_id == "load-context"
        assert error.error_code == "E2001"
        assert "missing required input" in error.message

    def test_undeclared_read_error(self):
        """Can create undeclared read violation error."""
        error = ContractViolationError(
            violation_type=ViolationType.UNDECLARED_READ,
            key="unknown_key",
            node_id="process-node",
        )

        assert error.violation_type == ViolationType.UNDECLARED_READ
        assert error.error_code == "E2002"
        assert "undeclared key" in error.message

    def test_undeclared_write_error(self):
        """Can create undeclared write violation error."""
        error = ContractViolationError(
            violation_type=ViolationType.UNDECLARED_WRITE,
            key="extra_output",
            node_id="save-node",
        )

        assert error.violation_type == ViolationType.UNDECLARED_WRITE
        assert error.error_code == "E2003"
        assert "wrote undeclared key" in error.message

    def test_custom_message(self):
        """Can provide custom message."""
        error = ContractViolationError(
            violation_type=ViolationType.MISSING_INPUT,
            key="key",
            message="Custom error message",
        )

        assert error.message == "Custom error message"

    def test_to_bt_error_missing_input(self):
        """to_bt_error() creates correct BTError for missing input."""
        contract = NodeContract(
            inputs={"session_id": SessionId},
            description="Load context",
        )

        error = ContractViolationError(
            violation_type=ViolationType.MISSING_INPUT,
            key="session_id",
            node_id="load-context",
            contract=contract,
        )

        bt_error = error.to_bt_error()

        assert bt_error.code == "E2001"
        assert bt_error.category == "node"
        assert bt_error.severity.value == "error"
        assert bt_error.recovery.action.value == "abort"
        assert "session_id" in bt_error.context.extra["key"]

    def test_to_bt_error_undeclared_read(self):
        """to_bt_error() creates correct BTError for undeclared read."""
        contract = NodeContract(
            inputs={"a": SessionId},
        )

        error = ContractViolationError(
            violation_type=ViolationType.UNDECLARED_READ,
            key="undeclared",
            node_id="node",
            contract=contract,
        )

        bt_error = error.to_bt_error()

        assert bt_error.code == "E2002"
        assert bt_error.severity.value == "warning"
        assert bt_error.recovery.action.value == "skip"

    def test_to_bt_error_undeclared_write(self):
        """to_bt_error() creates correct BTError for undeclared write."""
        error = ContractViolationError(
            violation_type=ViolationType.UNDECLARED_WRITE,
            key="undeclared",
            node_id="node",
        )

        bt_error = error.to_bt_error()

        assert bt_error.code == "E2003"
        assert bt_error.severity.value == "warning"
        assert bt_error.recovery.action.value == "skip"

    def test_raises_as_exception(self):
        """Can raise and catch as exception."""
        with pytest.raises(ContractViolationError) as exc_info:
            raise ContractViolationError(
                violation_type=ViolationType.MISSING_INPUT,
                key="test_key",
            )

        assert exc_info.value.key == "test_key"


# =============================================================================
# ContractedNode Mixin Tests
# =============================================================================


class TestContractedNodeMixin:
    """Tests for ContractedNode mixin class."""

    def test_default_contract_is_empty(self):
        """Default contract() returns empty contract."""

        class MyNode(ContractedNode):
            pass

        contract = MyNode.contract()

        assert contract.inputs == {}
        assert contract.optional_inputs == {}
        assert contract.outputs == {}

    def test_override_contract(self):
        """Can override contract() in subclass."""

        class MyAction(ContractedNode):
            @classmethod
            def contract(cls) -> NodeContract:
                return NodeContract(
                    inputs={"query": SearchQuery},
                    outputs={"results": SearchResults},
                    description="Search for items",
                )

        contract = MyAction.contract()

        assert "query" in contract.inputs
        assert "results" in contract.outputs
        assert contract.description == "Search for items"

    def test_validate_contract_inputs_success(self):
        """_validate_contract_inputs returns None when all present."""

        class MyAction(ContractedNode):
            _id = "my-action"

            @classmethod
            def contract(cls) -> NodeContract:
                return NodeContract(
                    inputs={"session_id": SessionId},
                )

        node = MyAction()

        bb = TypedBlackboard()
        bb.register("session_id", SessionId)
        bb.set("session_id", {"value": "sess-123"})

        missing = node._validate_contract_inputs(bb)
        assert missing is None

    def test_validate_contract_inputs_failure(self):
        """_validate_contract_inputs returns missing keys when absent."""

        class MyAction(ContractedNode):
            _id = "my-action"

            @classmethod
            def contract(cls) -> NodeContract:
                return NodeContract(
                    inputs={"session_id": SessionId},
                )

        node = MyAction()

        bb = TypedBlackboard()
        bb.register("session_id", SessionId)
        # Not setting value

        missing = node._validate_contract_inputs(bb)
        assert missing == ["session_id"]

    def test_validate_contract_access_success(self):
        """_validate_contract_access returns empty list when valid."""

        class MyAction(ContractedNode):
            _id = "my-action"

            @classmethod
            def contract(cls) -> NodeContract:
                return NodeContract(
                    inputs={"query": SearchQuery},
                    outputs={"results": SearchResults},
                )

        node = MyAction()

        bb = TypedBlackboard()
        bb.register("query", SearchQuery)
        bb.register("results", SearchResults)
        bb.set("query", {"query": "test"})
        bb.clear_access_tracking()

        # Simulate access
        bb.get("query", SearchQuery)
        bb.set("results", {"items": [], "total_count": 0})

        violations = node._validate_contract_access(bb)
        assert violations == []

    def test_validate_contract_access_violations(self):
        """_validate_contract_access returns violations when invalid."""

        class MyAction(ContractedNode):
            _id = "my-action"

            @classmethod
            def contract(cls) -> NodeContract:
                return NodeContract(
                    inputs={"query": SearchQuery},
                    outputs={"results": SearchResults},
                )

        node = MyAction()

        bb = TypedBlackboard()
        bb.register("query", SearchQuery)
        bb.register("results", SearchResults)
        bb.register("extra_input", SessionId)
        bb.register("extra_output", TurnNumber)
        bb.set("query", {"query": "test"})
        bb.set("extra_input", {"value": "extra"})
        bb.clear_access_tracking()

        # Simulate access with undeclared keys
        bb.get("query", SearchQuery)
        bb.get("extra_input", SessionId)  # undeclared read
        bb.set("results", {"items": [], "total_count": 0})
        bb.set("extra_output", {"value": 1.0})  # undeclared write

        violations = node._validate_contract_access(bb)

        assert len(violations) == 2

    def test_get_contract_summary(self):
        """get_contract_summary() returns human-readable string."""

        class MyAction(ContractedNode):
            @classmethod
            def contract(cls) -> NodeContract:
                return NodeContract(
                    inputs={"query": SearchQuery},
                    outputs={"results": SearchResults},
                    description="Search for items",
                )

        node = MyAction()
        summary = node.get_contract_summary()

        assert "Search for items" in summary
        assert "query" in summary
        assert "results" in summary


# =============================================================================
# action_contract Decorator Tests
# =============================================================================


class TestActionContractDecorator:
    """Tests for @action_contract decorator."""

    def test_attaches_contract_to_function(self):
        """Decorator attaches __contract__ attribute."""

        @action_contract(
            inputs={"query": SearchQuery},
            outputs={"results": SearchResults},
        )
        def search(ctx):
            pass

        assert hasattr(search, "__contract__")
        assert isinstance(search.__contract__, NodeContract)

    def test_contract_has_correct_values(self):
        """Attached contract has correct values."""

        @action_contract(
            inputs={"session_id": SessionId},
            optional_inputs={"limit": HistoryLimit},
            outputs={"context": ConversationContext},
            description="Load context",
        )
        def load_context(ctx):
            pass

        contract = load_context.__contract__

        assert "session_id" in contract.inputs
        assert "limit" in contract.optional_inputs
        assert "context" in contract.outputs
        assert contract.description == "Load context"

    def test_decorated_function_still_callable(self):
        """Decorated function can still be called normally."""

        @action_contract(
            inputs={"x": SessionId},
        )
        def add_one(value):
            return value + 1

        result = add_one(5)
        assert result == 6

    def test_get_action_contract_retrieves_contract(self):
        """get_action_contract() retrieves attached contract."""

        @action_contract(
            inputs={"query": SearchQuery},
            description="Test function",
        )
        def my_action(ctx):
            pass

        contract = get_action_contract(my_action)

        assert contract is not None
        assert "query" in contract.inputs
        assert contract.description == "Test function"

    def test_get_action_contract_returns_none_for_undecorated(self):
        """get_action_contract() returns None for undecorated function."""

        def plain_function(ctx):
            pass

        contract = get_action_contract(plain_function)
        assert contract is None

    def test_empty_contract_decorator(self):
        """Decorator with no arguments creates empty contract."""

        @action_contract()
        def minimal_action(ctx):
            pass

        contract = get_action_contract(minimal_action)

        assert contract is not None
        assert contract.inputs == {}
        assert contract.optional_inputs == {}
        assert contract.outputs == {}

    def test_preserves_function_metadata(self):
        """Decorator preserves function name and docstring."""

        @action_contract(inputs={"x": SessionId})
        def documented_function(ctx):
            """This is the docstring."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."


# =============================================================================
# Error Factory Function Tests
# =============================================================================


class TestErrorFactoryFunctions:
    """Tests for E2001, E2002, E2003 error factory functions."""

    def test_make_missing_input_error(self):
        """make_missing_input_error creates correct E2001."""
        error = make_missing_input_error(
            node_id="load-context",
            node_name="LoadContext",
            key="session_id",
            expected_type="SessionId",
            contract_description="Load user context",
        )

        assert error.code == "E2001"
        assert error.category == "node"
        assert error.severity.value == "error"
        assert "LoadContext" in error.message
        assert "session_id" in error.message
        assert error.recovery.action.value == "abort"
        assert error.context.node_id == "load-context"

    def test_make_undeclared_read_error(self):
        """make_undeclared_read_error creates correct E2002."""
        error = make_undeclared_read_error(
            node_id="process-node",
            node_name="ProcessNode",
            key="undeclared_key",
            declared_inputs=["input_a", "input_b"],
        )

        assert error.code == "E2002"
        assert error.category == "node"
        assert error.severity.value == "warning"
        assert "ProcessNode" in error.message
        assert "undeclared_key" in error.message
        assert error.recovery.action.value == "skip"

    def test_make_undeclared_write_error(self):
        """make_undeclared_write_error creates correct E2003."""
        error = make_undeclared_write_error(
            node_id="save-node",
            node_name="SaveNode",
            key="extra_output",
            declared_outputs=["output_a"],
        )

        assert error.code == "E2003"
        assert error.category == "node"
        assert error.severity.value == "warning"
        assert "SaveNode" in error.message
        assert "extra_output" in error.message
        assert error.recovery.action.value == "skip"

    def test_errors_have_emit_event_true(self):
        """All factory-created errors have emit_event=True."""
        e2001 = make_missing_input_error("id", "name", "key", "type")
        e2002 = make_undeclared_read_error("id", "name", "key", [])
        e2003 = make_undeclared_write_error("id", "name", "key", [])

        assert e2001.emit_event is True
        assert e2002.emit_event is True
        assert e2003.emit_event is True


# =============================================================================
# NodeContract.summary() Tests
# =============================================================================


class TestNodeContractSummary:
    """Tests for NodeContract.summary() method."""

    def test_empty_contract_summary(self):
        """Empty contract returns appropriate message."""
        contract = NodeContract()
        summary = contract.summary()

        assert "Empty contract" in summary or "no requirements" in summary.lower()

    def test_full_contract_summary(self):
        """Full contract includes all sections."""
        contract = NodeContract(
            inputs={"session_id": SessionId},
            optional_inputs={"limit": HistoryLimit},
            outputs={"context": ConversationContext},
            description="Load conversation context",
        )

        summary = contract.summary()

        assert "Load conversation context" in summary
        assert "session_id" in summary
        assert "limit" in summary
        assert "context" in summary
        assert "Required" in summary or "input" in summary.lower()
        assert "Optional" in summary or "optional" in summary.lower()
        assert "Output" in summary or "output" in summary.lower()


# =============================================================================
# get_missing_inputs Tests
# =============================================================================


class TestGetMissingInputs:
    """Tests for NodeContract.get_missing_inputs() method."""

    def test_no_missing_returns_empty_list(self):
        """No missing inputs returns empty list."""
        contract = NodeContract(
            inputs={"session_id": SessionId},
        )

        bb = TypedBlackboard()
        bb.register("session_id", SessionId)
        bb.set("session_id", {"value": "sess-123"})

        errors = contract.get_missing_inputs(bb)
        assert errors == []

    def test_missing_returns_error_objects(self):
        """Missing inputs returns ContractViolationError objects."""
        contract = NodeContract(
            inputs={"session_id": SessionId, "user_id": UserId},
        )

        bb = TypedBlackboard()
        bb.register("session_id", SessionId)
        bb.register("user_id", UserId)
        # Not setting any values

        errors = contract.get_missing_inputs(bb)

        assert len(errors) == 2
        assert all(isinstance(e, ContractViolationError) for e in errors)
        assert all(e.violation_type == ViolationType.MISSING_INPUT for e in errors)

        keys = {e.key for e in errors}
        assert "session_id" in keys
        assert "user_id" in keys


# =============================================================================
# Integration Tests
# =============================================================================


class TestContractIntegration:
    """Integration tests combining multiple contract features."""

    def test_full_workflow_success(self):
        """Full workflow: declare contract, validate inputs, track access."""

        class SearchAction(ContractedNode):
            _id = "search-action"

            @classmethod
            def contract(cls) -> NodeContract:
                return NodeContract(
                    inputs={"query": SearchQuery},
                    optional_inputs={"limit": HistoryLimit},
                    outputs={"results": SearchResults},
                    description="Search for items",
                )

        node = SearchAction()

        # Setup blackboard
        bb = TypedBlackboard()
        bb.register("query", SearchQuery)
        bb.register("limit", HistoryLimit)
        bb.register("results", SearchResults)

        # Set required input
        bb.set("query", {"query": "test search"})
        bb.clear_access_tracking()

        # Validate inputs
        missing = node._validate_contract_inputs(bb)
        assert missing is None

        # Simulate node execution
        bb.get("query", SearchQuery)
        bb.set("results", {"items": ["item1", "item2"], "total_count": 2})

        # Validate access
        violations = node._validate_contract_access(bb)
        assert violations == []

    def test_full_workflow_with_violations(self):
        """Workflow detects both missing inputs and access violations."""

        class BadAction(ContractedNode):
            _id = "bad-action"

            @classmethod
            def contract(cls) -> NodeContract:
                return NodeContract(
                    inputs={"required": SessionId},
                    outputs={"expected": TurnNumber},
                )

        node = BadAction()

        # Setup blackboard - missing required input
        bb = TypedBlackboard()
        bb.register("required", SessionId)
        bb.register("expected", TurnNumber)
        bb.register("extra", HistoryLimit)
        bb.set("extra", {"max_items": 50})

        # Check missing inputs
        missing = node._validate_contract_inputs(bb)
        assert missing == ["required"]

        # If we ignored that and ran anyway...
        bb.set("required", {"value": "now-present"})
        bb.clear_access_tracking()

        # Simulate bad access
        bb.get("required", SessionId)
        bb.get("extra", HistoryLimit)  # undeclared read
        bb.set("expected", {"value": 1.0})
        bb.set("extra", {"max_items": 100})  # undeclared write

        violations = node._validate_contract_access(bb)
        assert len(violations) == 2

    def test_decorated_action_with_contracted_node(self):
        """Decorated function can work with ContractedNode pattern."""

        # Decorated standalone function
        @action_contract(
            inputs={"query": SearchQuery},
            outputs={"results": SearchResults},
            description="Search implementation",
        )
        def search_impl(query: SearchQuery) -> SearchResults:
            return SearchResults(
                items=[f"Result for: {query.query}"],
                total_count=1,
            )

        # Node that uses the decorated function
        class SearchNode(ContractedNode):
            _id = "search-node"

            @classmethod
            def contract(cls) -> NodeContract:
                # Can reuse the function's contract
                fn_contract = get_action_contract(search_impl)
                return fn_contract or NodeContract()

        node = SearchNode()
        contract = node.__class__.contract()

        assert "query" in contract.inputs
        assert "results" in contract.outputs
        assert contract.description == "Search implementation"
