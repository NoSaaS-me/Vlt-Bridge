"""
Tests for BT State Bridges (Phase 0.5).

Tests RuleContextBridge and LuaStateBridge for:
- Converting between RuleContext and OracleState
- Converting between TypedBlackboard and Lua-compatible data
- Type coercion correctness

Reference:
- tasks.md sections 0.5.1-0.5.5
- contracts/blackboard.yaml type coercion rules
"""

from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from backend.src.bt.state.bridges import RuleContextBridge, LuaStateBridge
from backend.src.bt.state.blackboard import TypedBlackboard
from backend.src.bt.state.composite import OracleState
from backend.src.bt.state.types import (
    MessageState,
    ToolCallState,
    ToolCallStatus,
)

# Import RuleContext types for testing
from backend.src.services.plugins.context import (
    RuleContext,
    TurnState,
    HistoryState,
    UserState,
    ProjectState,
    PluginState,
    ToolCallRecord,
)


# =============================================================================
# Test Pydantic Models for Blackboard
# =============================================================================


class CountModel(BaseModel):
    """Simple model for testing."""
    value: int


class ConfigModel(BaseModel):
    """Model with multiple fields."""
    name: str
    count: int = 0
    enabled: bool = True


class NestedModel(BaseModel):
    """Model with nested structure."""
    config: ConfigModel
    tags: list[str] = []


# =============================================================================
# RuleContextBridge Tests (0.5.1-0.5.2)
# =============================================================================


class TestRuleContextBridgeFromRuleContext:
    """Test RuleContextBridge.from_rule_context() (0.5.1)."""

    def test_minimal_context(self):
        """Convert minimal RuleContext to OracleState."""
        rc = RuleContext.create_minimal(
            user_id="user-123",
            project_id="project-456",
            turn_number=5,
        )

        oracle_state = RuleContextBridge.from_rule_context(rc)

        assert oracle_state.user_id == "user-123"
        assert oracle_state.project_id == "project-456"
        assert oracle_state.turn_number == 5

    def test_turn_state_mapping(self):
        """Token and context usage should be converted to actual values."""
        turn = TurnState(
            number=10,
            token_usage=0.5,  # 50% usage
            context_usage=0.25,  # 25% usage
            iteration_count=3,
        )
        rc = RuleContext(
            turn=turn,
            history=HistoryState(),
            user=UserState(id="user"),
            project=ProjectState(id="project"),
            state=PluginState(),
        )

        oracle_state = RuleContextBridge.from_rule_context(rc)

        # Token usage: 0.5 * 100000 (default budget) = 50000
        assert oracle_state.tokens_used == 50000
        # Context usage: 0.25 * 128000 (default max) = 32000
        assert oracle_state.context_tokens == 32000
        assert oracle_state.iterations_used == 3
        assert oracle_state.turn_number == 10

    def test_messages_conversion(self):
        """Messages from history should be converted to MessageState list."""
        history = HistoryState(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        rc = RuleContext(
            turn=TurnState(number=1, token_usage=0.0, context_usage=0.0, iteration_count=0),
            history=history,
            user=UserState(id="user"),
            project=ProjectState(id="project"),
            state=PluginState(),
        )

        oracle_state = RuleContextBridge.from_rule_context(rc)

        assert len(oracle_state.messages) == 2
        assert oracle_state.messages[0].role == "user"
        assert oracle_state.messages[0].content == "Hello"
        assert oracle_state.messages[1].role == "assistant"
        assert oracle_state.messages[1].content == "Hi there!"

    def test_tool_calls_conversion(self):
        """Tool calls from history should be converted to ToolCallState."""
        now = datetime.now(timezone.utc)
        history = HistoryState(
            messages=[],
            tools=[
                ToolCallRecord(
                    name="search_code",
                    arguments={"query": "test"},
                    result="Found 5 results",
                    success=True,
                    timestamp=now,
                ),
                ToolCallRecord(
                    name="read_file",
                    arguments={"path": "/test.py"},
                    result="File not found",
                    success=False,
                    timestamp=now,
                ),
            ],
            failures={"read_file": 1},
        )
        rc = RuleContext(
            turn=TurnState(number=1, token_usage=0.0, context_usage=0.0, iteration_count=0),
            history=history,
            user=UserState(id="user"),
            project=ProjectState(id="project"),
            state=PluginState(),
        )

        oracle_state = RuleContextBridge.from_rule_context(rc)

        assert len(oracle_state.completed_tools) == 2
        # First tool - success
        assert oracle_state.completed_tools[0].name == "search_code"
        assert oracle_state.completed_tools[0].status == ToolCallStatus.SUCCESS
        assert oracle_state.completed_tools[0].result == "Found 5 results"
        # Second tool - failure
        assert oracle_state.completed_tools[1].name == "read_file"
        assert oracle_state.completed_tools[1].status == ToolCallStatus.FAILURE
        # Failure counts
        assert oracle_state.failure_counts == {"read_file": 1}

    def test_empty_context(self):
        """Handle empty history gracefully."""
        rc = RuleContext.create_minimal("user", "project")

        oracle_state = RuleContextBridge.from_rule_context(rc)

        assert oracle_state.messages == []
        assert oracle_state.completed_tools == []
        assert oracle_state.failure_counts == {}


class TestRuleContextBridgeToRuleContext:
    """Test RuleContextBridge.to_rule_context() (0.5.2)."""

    def test_basic_conversion(self):
        """Convert OracleState to RuleContext-compatible dict."""
        oracle_state = OracleState(
            user_id="user-123",
            project_id="project-456",
            turn_number=5,
            tokens_used=50000,
            token_budget=100000,
            context_tokens=32000,
            max_context_tokens=128000,
            iterations_used=3,
        )

        rc_dict = RuleContextBridge.to_rule_context(oracle_state)

        assert rc_dict["user"]["id"] == "user-123"
        assert rc_dict["project"]["id"] == "project-456"
        assert rc_dict["turn"]["number"] == 5
        assert rc_dict["turn"]["token_usage"] == 0.5  # 50000/100000
        assert rc_dict["turn"]["context_usage"] == 0.25  # 32000/128000
        assert rc_dict["turn"]["iteration_count"] == 3

    def test_messages_conversion(self):
        """Messages should be converted to list of dicts."""
        oracle_state = OracleState(
            user_id="user",
            messages=[
                MessageState(role="user", content="Hello"),
                MessageState(role="assistant", content="Hi!"),
            ],
        )

        rc_dict = RuleContextBridge.to_rule_context(oracle_state)

        assert len(rc_dict["history"]["messages"]) == 2
        assert rc_dict["history"]["messages"][0] == {"role": "user", "content": "Hello"}

    def test_completed_tools_conversion(self):
        """Completed tools should be converted to tool records."""
        now = datetime.now(timezone.utc)
        oracle_state = OracleState(
            user_id="user",
            completed_tools=[
                ToolCallState(
                    tool_id="tool-1",
                    name="search",
                    arguments={"query": "test"},
                    status=ToolCallStatus.SUCCESS,
                    result="Found 3 results",
                    completed_at=now,
                ),
            ],
            failure_counts={"other_tool": 2},
        )

        rc_dict = RuleContextBridge.to_rule_context(oracle_state)

        tools = rc_dict["history"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["success"] is True
        assert tools[0]["result"] == "Found 3 results"
        assert rc_dict["history"]["failures"] == {"other_tool": 2}

    def test_roundtrip_conversion(self):
        """Convert to RuleContext and back should preserve key data."""
        original = OracleState(
            user_id="user-123",
            project_id="project-456",
            turn_number=5,
            tokens_used=25000,
            iterations_used=10,
        )

        # To RuleContext dict
        rc_dict = RuleContextBridge.to_rule_context(original)

        # Recreate RuleContext
        rc = RuleContext(
            turn=TurnState(
                number=rc_dict["turn"]["number"],
                token_usage=rc_dict["turn"]["token_usage"],
                context_usage=rc_dict["turn"]["context_usage"],
                iteration_count=rc_dict["turn"]["iteration_count"],
            ),
            history=HistoryState(
                messages=rc_dict["history"]["messages"],
                failures=rc_dict["history"]["failures"],
            ),
            user=UserState(id=rc_dict["user"]["id"]),
            project=ProjectState(id=rc_dict["project"]["id"]),
            state=PluginState(),
        )

        # Back to OracleState
        restored = RuleContextBridge.from_rule_context(rc)

        assert restored.user_id == original.user_id
        assert restored.project_id == original.project_id
        assert restored.turn_number == original.turn_number


# =============================================================================
# LuaStateBridge Tests (0.5.3-0.5.4)
# =============================================================================


class TestLuaStateBridgeToLuaTable:
    """Test LuaStateBridge.to_lua_table() (0.5.3)."""

    def test_simple_model(self):
        """Convert blackboard with simple model to Lua dict."""
        bb = TypedBlackboard(scope_name="test")
        bb.register("count", CountModel)
        bb.set("count", CountModel(value=42))

        lua_data = LuaStateBridge.to_lua_table(bb)

        # Numbers become floats in Lua
        assert lua_data["count"]["value"] == 42.0
        assert isinstance(lua_data["count"]["value"], float)

    def test_complex_model(self):
        """Convert model with multiple fields."""
        bb = TypedBlackboard(scope_name="test")
        bb.register("config", ConfigModel)
        bb.set("config", ConfigModel(name="test", count=10, enabled=False))

        lua_data = LuaStateBridge.to_lua_table(bb)

        assert lua_data["config"]["name"] == "test"
        assert lua_data["config"]["count"] == 10.0  # float
        assert lua_data["config"]["enabled"] is False

    def test_nested_model(self):
        """Convert nested model structure."""
        bb = TypedBlackboard(scope_name="test")
        bb.register("nested", NestedModel)
        bb.set("nested", NestedModel(
            config=ConfigModel(name="inner", count=5),
            tags=["tag1", "tag2"],
        ))

        lua_data = LuaStateBridge.to_lua_table(bb)

        assert lua_data["nested"]["config"]["name"] == "inner"
        assert lua_data["nested"]["config"]["count"] == 5.0
        assert lua_data["nested"]["tags"] == ["tag1", "tag2"]

    def test_includes_parent_scope(self):
        """Snapshot should include parent scope data."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("parent_key", CountModel)
        parent.set("parent_key", CountModel(value=100))

        child = parent.create_child_scope("child")
        child.register("child_key", CountModel)
        child.set("child_key", CountModel(value=200))

        lua_data = LuaStateBridge.to_lua_table(child)

        # Should include both parent and child data
        assert "parent_key" in lua_data
        assert "child_key" in lua_data
        assert lua_data["parent_key"]["value"] == 100.0
        assert lua_data["child_key"]["value"] == 200.0


class TestLuaStateBridgeFromLuaResult:
    """Test LuaStateBridge.from_lua_result() (0.5.4)."""

    def test_update_from_lua_dict(self):
        """Update blackboard from Lua return dict."""
        bb = TypedBlackboard(scope_name="test")
        bb.register("count", CountModel)
        bb.set("count", CountModel(value=0))

        # Simulate Lua return with float (Lua numbers are always float)
        lua_result = {"count": {"value": 42.0}}

        LuaStateBridge.from_lua_result(lua_result, bb, ["count"])

        # Value should be updated
        count = bb.get("count", CountModel)
        assert count.value == 42  # Pydantic coerces to int

    def test_ignores_undeclared_keys(self):
        """Only output_keys should be written."""
        bb = TypedBlackboard(scope_name="test")
        bb.register("count", CountModel)
        bb.set("count", CountModel(value=0))

        lua_result = {
            "count": {"value": 100.0},
            "other": {"value": 999.0},  # Not in output_keys
        }

        LuaStateBridge.from_lua_result(lua_result, bb, ["count"])

        # count updated, other ignored
        assert bb.get("count", CountModel).value == 100
        assert not bb.has("other")

    def test_ignores_unregistered_keys(self):
        """Unregistered keys should be ignored with warning."""
        bb = TypedBlackboard(scope_name="test")
        # Don't register "count"

        lua_result = {"count": {"value": 42.0}}

        # Should not raise, just log warning
        LuaStateBridge.from_lua_result(lua_result, bb, ["count"])

        # Key should not exist
        assert not bb.has("count")

    def test_handles_none_result(self):
        """None result should be no-op."""
        bb = TypedBlackboard(scope_name="test")
        bb.register("count", CountModel)
        bb.set("count", CountModel(value=42))

        LuaStateBridge.from_lua_result(None, bb, ["count"])

        # Value unchanged
        assert bb.get("count", CountModel).value == 42

    def test_handles_non_dict_result(self):
        """Non-dict result should be no-op."""
        bb = TypedBlackboard(scope_name="test")
        bb.register("count", CountModel)
        bb.set("count", CountModel(value=42))

        LuaStateBridge.from_lua_result("just a string", bb, ["count"])

        # Value unchanged
        assert bb.get("count", CountModel).value == 42


class TestLuaStateBridgeStatusExtraction:
    """Test status/error extraction helpers."""

    def test_extract_success_status(self):
        """Extract 'success' status from Lua result."""
        result = {"status": "success", "data": "some_data"}

        status = LuaStateBridge.extract_status_from_lua(result)

        assert status == "success"

    def test_extract_failure_status(self):
        """Extract 'failure' status from Lua result."""
        result = {"status": "FAILURE"}  # Should be normalized

        status = LuaStateBridge.extract_status_from_lua(result)

        assert status == "failure"

    def test_extract_running_status(self):
        """Extract 'running' status from Lua result."""
        result = {"status": "Running"}

        status = LuaStateBridge.extract_status_from_lua(result)

        assert status == "running"

    def test_no_status_returns_none(self):
        """Missing status field returns None."""
        result = {"data": "some_data"}

        status = LuaStateBridge.extract_status_from_lua(result)

        assert status is None

    def test_extract_error_reason(self):
        """Extract error reason from Lua result."""
        result = {"status": "failure", "reason": "Something went wrong"}

        error = LuaStateBridge.extract_error_from_lua(result)

        assert error == "Something went wrong"

    def test_extract_error_from_error_field(self):
        """Extract error from 'error' field."""
        result = {"status": "failure", "error": "Error message"}

        error = LuaStateBridge.extract_error_from_lua(result)

        assert error == "Error message"

    def test_no_error_returns_none(self):
        """Missing error fields return None."""
        result = {"status": "failure"}

        error = LuaStateBridge.extract_error_from_lua(result)

        assert error is None


# =============================================================================
# Integration Tests (0.5.5)
# =============================================================================


class TestBridgesIntegration:
    """Integration tests for bridges working together."""

    def test_rule_context_to_blackboard_workflow(self):
        """Workflow: RuleContext -> OracleState -> TypedBlackboard -> Lua."""
        # 1. Create RuleContext (from plugin system)
        rc = RuleContext.create_minimal("user-123", "project-456", turn_number=5)

        # 2. Convert to OracleState
        oracle_state = RuleContextBridge.from_rule_context(rc)

        # 3. Set up blackboard with oracle state
        bb = TypedBlackboard(scope_name="oracle")
        bb.register("oracle", OracleState)
        bb.set("oracle", oracle_state)

        # 4. Convert to Lua-compatible data
        lua_data = LuaStateBridge.to_lua_table(bb)

        # Verify chain
        assert lua_data["oracle"]["user_id"] == "user-123"
        assert lua_data["oracle"]["turn_number"] == 5.0  # Lua float

    def test_lua_result_to_rule_context_workflow(self):
        """Workflow: Lua result -> TypedBlackboard -> OracleState -> RuleContext."""
        # 1. Set up blackboard
        bb = TypedBlackboard(scope_name="oracle")
        bb.register("oracle", OracleState)

        # Initial state
        initial = OracleState(user_id="user-123", tokens_used=1000)
        bb.set("oracle", initial)

        # 2. Simulate Lua script modifying state
        lua_result = {
            "oracle": {
                "user_id": "user-123",
                "project_id": "",
                "session_id": "",
                "tree_id": None,
                "messages": [],
                "context_tokens": 0.0,
                "max_context_tokens": 128000.0,
                "turn_number": 10.0,
                "token_budget": 100000.0,
                "tokens_used": 5000.0,  # Modified!
                "iteration_budget": 100.0,
                "iterations_used": 0.0,
                "timeout_ms": 300000.0,
                "elapsed_ms": 0.0,
                "pending_tools": [],
                "running_tools": [],
                "completed_tools": [],
                "failure_counts": {},
                "model": "claude-sonnet-4",
                "provider": "anthropic",
                "streaming_enabled": True,
                "current_query": None,
                "streaming_buffer": "",
                "streaming_chunks": [],
                "final_response": None,
            }
        }

        LuaStateBridge.from_lua_result(lua_result, bb, ["oracle"])

        # 3. Get updated OracleState
        oracle_state = bb.get("oracle", OracleState)

        # 4. Convert to RuleContext for rule evaluation
        rc_dict = RuleContextBridge.to_rule_context(oracle_state)

        # Verify chain
        assert rc_dict["turn"]["number"] == 10
        assert rc_dict["turn"]["token_usage"] == 0.05  # 5000/100000
