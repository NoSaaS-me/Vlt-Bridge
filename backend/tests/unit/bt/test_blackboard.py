"""
Unit tests for TypedBlackboard.

Tests cover:
- Hierarchical scope lookup
- Schema registration and validation
- All error conditions (E1001-E1004)
- Access tracking
- Size limits
- Type coercion edge cases

Part of the BT Universal Runtime (spec 019).
"""

import pytest
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from backend.src.bt.state import (
    TypedBlackboard,
    ErrorResult,
    MAX_SIZE_BYTES,
    MAX_KEY_LENGTH,
    RESERVED_PREFIX,
    lua_to_python,
    python_to_lua,
)


# =============================================================================
# Test Schemas
# =============================================================================


class SimpleValue(BaseModel):
    """Simple test schema."""

    value: str


class ConversationContext(BaseModel):
    """Complex test schema matching spec example."""

    session_id: str
    user_id: str
    turn_number: float  # Uses float for Lua compatibility
    history: List[Dict[str, str]]
    metadata: Optional[Dict[str, Any]] = None


class ToolResult(BaseModel):
    """Test schema for tool results."""

    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    duration_ms: float


class NumberHolder(BaseModel):
    """Schema with numeric fields for coercion tests."""

    int_value: int  # Will coerce from float
    float_value: float


# =============================================================================
# Constructor Tests
# =============================================================================


class TestBlackboardConstructor:
    """Tests for TypedBlackboard.__init__"""

    def test_creates_root_scope(self):
        """Can create a root scope blackboard."""
        bb = TypedBlackboard(scope_name="root")
        assert bb._scope_name == "root"
        assert bb._parent is None
        assert bb._data == {}
        assert bb._schemas == {}
        assert bb._reads == set()
        assert bb._writes == set()
        assert bb._size_bytes == 0

    def test_creates_child_scope(self):
        """Can create a child scope with parent."""
        parent = TypedBlackboard(scope_name="parent")
        child = TypedBlackboard(parent=parent, scope_name="child")

        assert child._scope_name == "child"
        assert child._parent is parent

    def test_empty_scope_name_raises(self):
        """Empty scope_name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TypedBlackboard(scope_name="")

    def test_none_scope_name_raises(self):
        """None scope_name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TypedBlackboard(scope_name=None)

    def test_invalid_parent_type_raises(self):
        """Non-TypedBlackboard parent raises TypeError."""
        with pytest.raises(TypeError, match="must be TypedBlackboard or None"):
            TypedBlackboard(parent="not a blackboard", scope_name="child")

    def test_inherits_schemas_from_parent(self):
        """Child inherits schemas from parent."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("context", ConversationContext)

        child = TypedBlackboard(parent=parent, scope_name="child")
        assert "context" in child._schemas
        assert child._schemas["context"] == ConversationContext

    def test_circular_reference_raises(self):
        """Circular parent reference raises ValueError."""
        bb1 = TypedBlackboard(scope_name="scope1")
        bb2 = TypedBlackboard(parent=bb1, scope_name="scope2")

        # Manually create circular reference for testing
        bb1._parent = bb2  # This creates scope2 -> scope1 -> scope2

        with pytest.raises(ValueError, match="Circular reference"):
            TypedBlackboard(parent=bb1, scope_name="scope3")


# =============================================================================
# Schema Registration Tests
# =============================================================================


class TestSchemaRegistration:
    """Tests for register() and register_many()"""

    def test_register_single_schema(self):
        """Can register a single schema."""
        bb = TypedBlackboard()
        bb.register("context", ConversationContext)

        assert "context" in bb._schemas
        assert bb._schemas["context"] == ConversationContext

    def test_register_is_idempotent(self):
        """Re-registering same key with same schema succeeds."""
        bb = TypedBlackboard()
        bb.register("context", ConversationContext)
        bb.register("context", ConversationContext)  # No error

        assert bb._schemas["context"] == ConversationContext

    def test_register_empty_key_raises(self):
        """Empty key raises ValueError."""
        bb = TypedBlackboard()
        with pytest.raises(ValueError, match="non-empty string"):
            bb.register("", SimpleValue)

    def test_register_long_key_raises(self):
        """Key exceeding MAX_KEY_LENGTH raises ValueError."""
        bb = TypedBlackboard()
        long_key = "x" * (MAX_KEY_LENGTH + 1)

        with pytest.raises(ValueError, match="too long"):
            bb.register(long_key, SimpleValue)

    def test_register_reserved_key_raises(self):
        """Key starting with underscore raises ValueError."""
        bb = TypedBlackboard()
        with pytest.raises(ValueError, match="system-reserved"):
            bb.register("_internal", SimpleValue)

    def test_register_non_basemodel_raises(self):
        """Non-BaseModel schema raises TypeError."""
        bb = TypedBlackboard()

        with pytest.raises(TypeError, match="subclass of pydantic.BaseModel"):
            bb.register("data", dict)

        with pytest.raises(TypeError, match="subclass of pydantic.BaseModel"):
            bb.register("data", str)

    def test_register_many_succeeds(self):
        """Can register multiple schemas atomically."""
        bb = TypedBlackboard()
        bb.register_many({
            "context": ConversationContext,
            "result": ToolResult,
        })

        assert "context" in bb._schemas
        assert "result" in bb._schemas

    def test_register_many_atomic_failure(self):
        """If any schema fails validation, none are registered."""
        bb = TypedBlackboard()

        with pytest.raises(TypeError):
            bb.register_many({
                "context": ConversationContext,  # Valid
                "invalid": dict,  # Invalid - not BaseModel
            })

        # Neither should be registered
        assert "context" not in bb._schemas
        assert "invalid" not in bb._schemas


# =============================================================================
# Data Access Tests
# =============================================================================


class TestDataAccess:
    """Tests for get(), set(), has(), delete()"""

    def test_set_and_get_value(self):
        """Can set and get a validated value."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)

        result = bb.set("simple", {"value": "test"})
        assert result.is_ok

        value = bb.get("simple", SimpleValue)
        assert value is not None
        assert value.value == "test"

    def test_set_with_basemodel_instance(self):
        """Can set using a BaseModel instance."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)

        instance = SimpleValue(value="test")
        result = bb.set("simple", instance)
        assert result.is_ok

        value = bb.get("simple", SimpleValue)
        assert value.value == "test"

    def test_get_returns_default_when_not_found(self):
        """get() returns default when key not found."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)

        default = SimpleValue(value="default")
        value = bb.get("simple", SimpleValue, default=default)
        assert value == default

    def test_get_returns_none_when_not_found_no_default(self):
        """get() returns None when key not found and no default."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)

        value = bb.get("simple", SimpleValue)
        assert value is None

    def test_get_tracks_reads(self):
        """get() adds key to _reads."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)

        bb.get("simple", SimpleValue)
        assert "simple" in bb._reads

    def test_has_returns_true_when_exists(self):
        """has() returns True when key exists."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)
        bb.set("simple", {"value": "test"})

        assert bb.has("simple") is True

    def test_has_returns_false_when_not_exists(self):
        """has() returns False when key doesn't exist."""
        bb = TypedBlackboard()
        assert bb.has("nonexistent") is False

    def test_has_does_not_track_reads(self):
        """has() does NOT add key to _reads."""
        bb = TypedBlackboard()
        bb.has("simple")
        assert "simple" not in bb._reads

    def test_delete_removes_value(self):
        """delete() removes value from data."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)
        bb.set("simple", {"value": "test"})

        deleted = bb.delete("simple")
        assert deleted is True
        assert bb.has("simple") is False

    def test_delete_returns_false_when_not_exists(self):
        """delete() returns False when key doesn't exist."""
        bb = TypedBlackboard()
        deleted = bb.delete("nonexistent")
        assert deleted is False

    def test_delete_does_not_remove_schema(self):
        """delete() leaves schema registered."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)
        bb.set("simple", {"value": "test"})
        bb.delete("simple")

        # Schema still registered
        assert "simple" in bb._schemas

        # Can set again
        result = bb.set("simple", {"value": "new"})
        assert result.is_ok


# =============================================================================
# Error Condition Tests (E1001-E1004)
# =============================================================================


class TestErrorConditions:
    """Tests for blackboard error conditions."""

    def test_unregistered_key_raises_e1001(self):
        """Setting unregistered key returns E1001 error."""
        bb = TypedBlackboard()

        result = bb.set("unknown", {"value": "test"})

        assert result.is_error
        assert result.error.code == "E1001"
        assert "not registered" in result.error.message

    def test_schema_validation_error_e1002(self):
        """Invalid value returns E1002 error."""
        bb = TypedBlackboard()
        bb.register("context", ConversationContext)

        # Missing required fields
        result = bb.set("context", {"session_id": "test"})

        assert result.is_error
        assert result.error.code == "E1002"
        assert "validation failed" in result.error.message

    def test_get_schema_mismatch_raises_e1002(self):
        """Getting with wrong schema raises RuntimeError (E1002)."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)
        bb.set("simple", {"value": "test"})

        with pytest.raises(RuntimeError, match="E1002"):
            bb.get("simple", ConversationContext)  # Wrong schema

    def test_reserved_key_returns_e1003(self):
        """Setting reserved key returns E1003 error."""
        bb = TypedBlackboard()

        result = bb.set("_internal", {"value": "test"})

        assert result.is_error
        assert result.error.code == "E1003"
        assert "system-reserved" in result.error.message

    def test_size_limit_returns_e1004(self):
        """Exceeding size limit returns E1004 error."""
        bb = TypedBlackboard()
        bb._max_size_bytes = 100  # Set a small limit for testing
        bb.register("simple", SimpleValue)

        # This value should exceed 100 bytes when serialized
        large_value = {"value": "x" * 200}
        result = bb.set("simple", large_value)

        assert result.is_error
        assert result.error.code == "E1004"
        assert "size limit exceeded" in result.error.message


# =============================================================================
# Hierarchical Scope Tests
# =============================================================================


class TestHierarchicalScope:
    """Tests for scope chain lookup."""

    def test_child_can_read_parent_value(self):
        """Child scope can read value from parent."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("context", ConversationContext)
        parent.set("context", {
            "session_id": "sess",
            "user_id": "user",
            "turn_number": 1.0,
            "history": [],
        })

        child = parent.create_child_scope("child")
        value = child.get("context", ConversationContext)

        assert value is not None
        assert value.session_id == "sess"

    def test_child_shadows_parent_value(self):
        """Child can shadow parent value without modifying parent."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("simple", SimpleValue)
        parent.set("simple", {"value": "parent_value"})

        child = parent.create_child_scope("child")
        child.set("simple", {"value": "child_value"})

        # Child sees child value
        child_value = child.get("simple", SimpleValue)
        assert child_value.value == "child_value"

        # Parent still has original
        parent_value = parent.get("simple", SimpleValue)
        assert parent_value.value == "parent_value"

    def test_has_checks_parent_chain(self):
        """has() checks parent chain."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("simple", SimpleValue)
        parent.set("simple", {"value": "test"})

        child = parent.create_child_scope("child")
        assert child.has("simple") is True

    def test_delete_only_affects_current_scope(self):
        """delete() only removes from current scope."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("simple", SimpleValue)
        parent.set("simple", {"value": "parent"})

        child = parent.create_child_scope("child")
        child.set("simple", {"value": "child"})

        # Delete from child
        child.delete("simple")

        # Parent value still visible from child
        value = child.get("simple", SimpleValue)
        assert value.value == "parent"

    def test_set_global_writes_to_root(self):
        """set_global() writes to root scope."""
        root = TypedBlackboard(scope_name="root")
        root.register("simple", SimpleValue)

        level1 = root.create_child_scope("level1")
        level2 = level1.create_child_scope("level2")

        # Write from deepest scope to root
        result = level2.set_global("simple", {"value": "from_level2"})
        assert result.is_ok

        # Value visible at root
        value = root.get("simple", SimpleValue)
        assert value.value == "from_level2"

    def test_deep_scope_chain(self):
        """Works with deep scope chains."""
        root = TypedBlackboard(scope_name="root")
        root.register("simple", SimpleValue)
        root.set("simple", {"value": "root_value"})

        # Create deep chain
        current = root
        for i in range(10):
            current = current.create_child_scope(f"level{i}")

        # Deepest level can read root value
        value = current.get("simple", SimpleValue)
        assert value.value == "root_value"


# =============================================================================
# Access Tracking Tests
# =============================================================================


class TestAccessTracking:
    """Tests for read/write tracking."""

    def test_get_tracks_reads(self):
        """get() adds key to reads."""
        bb = TypedBlackboard()
        bb.register("a", SimpleValue)
        bb.register("b", SimpleValue)

        bb.get("a", SimpleValue)
        bb.get("b", SimpleValue)

        reads = bb.get_reads()
        assert "a" in reads
        assert "b" in reads

    def test_set_tracks_writes(self):
        """set() adds key to writes."""
        bb = TypedBlackboard()
        bb.register("a", SimpleValue)
        bb.register("b", SimpleValue)

        bb.set("a", {"value": "test"})
        bb.set("b", {"value": "test"})

        writes = bb.get_writes()
        assert "a" in writes
        assert "b" in writes

    def test_clear_access_tracking(self):
        """clear_access_tracking() resets both sets."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)
        bb.set("simple", {"value": "test"})
        bb.get("simple", SimpleValue)

        assert len(bb.get_reads()) > 0
        assert len(bb.get_writes()) > 0

        bb.clear_access_tracking()

        assert len(bb.get_reads()) == 0
        assert len(bb.get_writes()) == 0

    def test_has_does_not_track_read(self):
        """has() does not track as a read."""
        bb = TypedBlackboard()
        bb.has("simple")

        assert "simple" not in bb.get_reads()

    def test_set_global_tracks_write_locally(self):
        """set_global() tracks write in calling scope."""
        root = TypedBlackboard(scope_name="root")
        root.register("simple", SimpleValue)

        child = root.create_child_scope("child")
        child.set_global("simple", {"value": "test"})

        # Write tracked in child
        assert "simple" in child.get_writes()


# =============================================================================
# Debugging Tests
# =============================================================================


class TestDebugging:
    """Tests for debugging methods."""

    def test_snapshot_returns_merged_data(self):
        """snapshot() returns merged data from all scopes."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("parent_key", SimpleValue)
        parent.set("parent_key", {"value": "parent"})

        child = parent.create_child_scope("child")
        child.register("child_key", SimpleValue)
        child.set("child_key", {"value": "child"})

        snapshot = child.snapshot()

        assert "parent_key" in snapshot
        assert "child_key" in snapshot
        assert snapshot["parent_key"]["value"] == "parent"
        assert snapshot["child_key"]["value"] == "child"

    def test_snapshot_child_wins_on_conflict(self):
        """snapshot() uses child value when key exists in both."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("key", SimpleValue)
        parent.set("key", {"value": "parent"})

        child = parent.create_child_scope("child")
        child.set("key", {"value": "child"})

        snapshot = child.snapshot()
        assert snapshot["key"]["value"] == "child"

    def test_get_size_bytes(self):
        """get_size_bytes() returns current size."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)

        initial_size = bb.get_size_bytes()
        assert initial_size == 0

        bb.set("simple", {"value": "test"})
        new_size = bb.get_size_bytes()
        assert new_size > 0

    def test_debug_info_returns_expected_fields(self):
        """debug_info() returns expected debug information."""
        bb = TypedBlackboard(scope_name="test_scope")
        bb.register("key1", SimpleValue)
        bb.set("key1", {"value": "test"})
        bb.get("key1", SimpleValue)

        info = bb.debug_info()

        assert info["scope_name"] == "test_scope"
        assert info["parent_scope"] is None
        assert info["size_bytes"] > 0
        assert info["key_count"] == 1
        assert "key1" in info["registered_schemas"]
        assert "key1" in info["reads_this_tick"]
        assert "key1" in info["writes_this_tick"]


# =============================================================================
# Size Limit Tests
# =============================================================================


class TestSizeLimits:
    """Tests for size tracking and limits."""

    def test_size_increases_on_set(self):
        """Size increases when setting values."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)

        bb.set("simple", {"value": "a"})
        size1 = bb.get_size_bytes()

        bb.set("simple", {"value": "aaaaaaaaaaaa"})  # Larger value
        size2 = bb.get_size_bytes()

        assert size2 > size1

    def test_size_decreases_on_delete(self):
        """Size decreases when deleting values."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)

        bb.set("simple", {"value": "test"})
        size_before = bb.get_size_bytes()

        bb.delete("simple")
        size_after = bb.get_size_bytes()

        assert size_after < size_before

    def test_size_update_replaces_correctly(self):
        """Size updates correctly when replacing values."""
        bb = TypedBlackboard()
        bb.register("simple", SimpleValue)

        # Set initial value
        bb.set("simple", {"value": "short"})
        size1 = bb.get_size_bytes()

        # Replace with same-ish sized value
        bb.set("simple", {"value": "short"})
        size2 = bb.get_size_bytes()

        # Size should be approximately the same
        assert abs(size1 - size2) < 10


# =============================================================================
# Internal Key Tests
# =============================================================================


class TestInternalKeys:
    """Tests for set_internal() system keys."""

    def test_set_internal_allows_underscore_keys(self):
        """set_internal() can set underscore-prefixed keys."""
        bb = TypedBlackboard()
        bb.set_internal("_failure_trace", {"errors": []})

        assert "_failure_trace" in bb._data
        assert bb._data["_failure_trace"] == {"errors": []}

    def test_set_internal_rejects_non_underscore_keys(self):
        """set_internal() rejects keys not starting with underscore."""
        bb = TypedBlackboard()

        with pytest.raises(ValueError, match="must start with"):
            bb.set_internal("not_internal", {"value": "test"})

    def test_set_internal_tracks_writes(self):
        """set_internal() tracks the write."""
        bb = TypedBlackboard()
        bb.set_internal("_internal", "value")

        assert "_internal" in bb.get_writes()


# =============================================================================
# Type Coercion Tests
# =============================================================================


class TestLuaToPython:
    """Tests for lua_to_python() type coercion."""

    def test_nil_to_none(self):
        """Lua nil converts to Python None."""
        assert lua_to_python(None) is None

    def test_boolean_conversion(self):
        """Lua boolean converts to Python bool."""
        assert lua_to_python(True) is True
        assert lua_to_python(False) is False

    def test_number_always_becomes_float(self):
        """Lua numbers ALWAYS become Python float."""
        result = lua_to_python(42)
        assert isinstance(result, float)
        assert result == 42.0

        result = lua_to_python(3.14)
        assert isinstance(result, float)
        assert result == 3.14

    def test_string_passthrough(self):
        """Lua string converts directly."""
        assert lua_to_python("hello") == "hello"

    def test_dict_conversion(self):
        """Python dicts pass through with type coercion."""
        result = lua_to_python({"name": "test", "count": 5})
        assert result == {"name": "test", "count": 5.0}

    def test_list_conversion(self):
        """Python lists pass through with type coercion."""
        result = lua_to_python([1, 2, 3])
        assert result == [1.0, 2.0, 3.0]

    def test_nested_coercion(self):
        """Nested structures are recursively coerced."""
        result = lua_to_python({"items": [1, 2], "meta": {"count": 3}})
        assert result == {"items": [1.0, 2.0], "meta": {"count": 3.0}}


class TestPythonToLua:
    """Tests for python_to_lua() type coercion."""

    def test_none_to_none(self):
        """Python None stays as None (Lua nil)."""
        assert python_to_lua(None) is None

    def test_bool_passthrough(self):
        """Python bool passes through."""
        assert python_to_lua(True) is True
        assert python_to_lua(False) is False

    def test_int_becomes_float(self):
        """Python int becomes float (Lua number)."""
        result = python_to_lua(42)
        assert isinstance(result, float)
        assert result == 42.0

    def test_float_passthrough(self):
        """Python float passes through."""
        assert python_to_lua(3.14) == 3.14

    def test_string_passthrough(self):
        """Python string passes through."""
        assert python_to_lua("hello") == "hello"

    def test_dict_conversion(self):
        """Python dict converts with string keys."""
        result = python_to_lua({"name": "test", "count": 5})
        assert result == {"name": "test", "count": 5.0}

    def test_list_conversion(self):
        """Python list converts with type coercion."""
        result = python_to_lua([1, 2, 3])
        assert result == [1.0, 2.0, 3.0]

    def test_basemodel_conversion(self):
        """Pydantic BaseModel converts via model_dump()."""
        model = SimpleValue(value="test")
        result = python_to_lua(model)
        assert result == {"value": "test"}

    def test_nested_basemodel(self):
        """Nested BaseModel fields convert correctly."""
        model = ConversationContext(
            session_id="sess",
            user_id="user",
            turn_number=1.0,
            history=[{"role": "user", "content": "hello"}],
        )
        result = python_to_lua(model)

        assert result["session_id"] == "sess"
        assert result["turn_number"] == 1.0
        assert len(result["history"]) == 1


class TestTypeCoercionEdgeCases:
    """Edge case tests for type coercion."""

    def test_empty_dict_stays_dict(self):
        """Empty dict stays as dict."""
        result = lua_to_python({})
        assert result == {}

    def test_empty_list_stays_list(self):
        """Empty list stays as list."""
        result = lua_to_python([])
        assert result == []

    def test_deep_nesting_limit(self):
        """Very deep nesting raises error."""
        # Create deeply nested structure
        value = {"level": 0}
        current = value
        for i in range(25):  # Exceed MAX_LUA_TABLE_DEPTH of 20
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        with pytest.raises(ValueError, match="too deep"):
            lua_to_python(value)

    def test_tuple_converts_to_list(self):
        """Python tuple converts to list."""
        result = python_to_lua((1, 2, 3))
        assert result == [1.0, 2.0, 3.0]


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_parallel_child_isolation(self):
        """Parallel children have isolated scopes."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("shared", SimpleValue)
        parent.register("result", SimpleValue)
        parent.set("shared", {"value": "original"})

        # Create two parallel children
        child1 = parent.create_child_scope("child1")
        child2 = parent.create_child_scope("child2")

        # Each writes to same key
        child1.set("result", {"value": "result1"})
        child2.set("result", {"value": "result2"})

        # Each sees only their own write
        assert child1.get("result", SimpleValue).value == "result1"
        assert child2.get("result", SimpleValue).value == "result2"

        # Parent sees neither (they didn't write to parent)
        assert parent.get("result", SimpleValue) is None

    def test_complex_workflow(self):
        """Complex workflow with multiple scopes and operations."""
        # Root scope with global context
        root = TypedBlackboard(scope_name="global")
        root.register("context", ConversationContext)
        root.register("result", ToolResult)

        root.set("context", {
            "session_id": "sess123",
            "user_id": "user456",
            "turn_number": 1.0,
            "history": [],
        })

        # Tree scope
        tree_bb = root.create_child_scope("tree")

        # Subtree scope
        subtree_bb = tree_bb.create_child_scope("subtree")

        # Subtree can read global context
        ctx = subtree_bb.get("context", ConversationContext)
        assert ctx.session_id == "sess123"

        # Subtree writes result
        subtree_bb.set("result", {
            "tool_name": "search",
            "success": True,
            "result": ["item1", "item2"],
            "duration_ms": 123.0,
        })

        # Result visible in subtree
        assert subtree_bb.has("result")

        # Not visible in parent (no merge yet)
        assert not tree_bb.has("result")

        # Can use set_global to write to root
        subtree_bb.set_global("result", {
            "tool_name": "search",
            "success": True,
            "result": ["item1", "item2"],
            "duration_ms": 123.0,
        })

        # Now visible at root
        result = root.get("result", ToolResult)
        assert result.tool_name == "search"

    def test_access_tracking_workflow(self):
        """Access tracking through a tick-like workflow."""
        bb = TypedBlackboard()
        bb.register("input", SimpleValue)
        bb.register("output", SimpleValue)

        # Simulate pre-tick setup
        bb.set("input", {"value": "user_query"})
        bb.clear_access_tracking()

        # Simulate tick
        input_val = bb.get("input", SimpleValue)
        bb.set("output", {"value": f"processed: {input_val.value}"})

        # Check tracking
        reads = bb.get_reads()
        writes = bb.get_writes()

        assert "input" in reads
        assert "output" in writes
        assert "input" not in writes  # Only read, not written during tick
