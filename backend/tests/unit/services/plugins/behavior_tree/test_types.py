"""Unit tests for behavior tree types."""

import pytest

from backend.src.services.plugins.behavior_tree.types import (
    RunStatus,
    TickContext,
    Blackboard,
)
from backend.src.services.plugins.context import RuleContext


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_success_is_truthy(self):
        """SUCCESS should be truthy."""
        assert bool(RunStatus.SUCCESS) is True

    def test_failure_is_falsy(self):
        """FAILURE should be falsy."""
        assert bool(RunStatus.FAILURE) is False

    def test_running_is_falsy(self):
        """RUNNING should be falsy."""
        assert bool(RunStatus.RUNNING) is False

    def test_from_bool_true(self):
        """from_bool(True) should return SUCCESS."""
        assert RunStatus.from_bool(True) == RunStatus.SUCCESS

    def test_from_bool_false(self):
        """from_bool(False) should return FAILURE."""
        assert RunStatus.from_bool(False) == RunStatus.FAILURE


class TestTickContext:
    """Tests for TickContext dataclass."""

    @pytest.fixture
    def rule_context(self):
        """Create a minimal RuleContext for testing."""
        return RuleContext.create_minimal("user1", "project1")

    def test_create_with_rule_context(self, rule_context):
        """TickContext should store rule_context."""
        ctx = TickContext(rule_context=rule_context)
        assert ctx.rule_context is rule_context

    def test_default_values(self, rule_context):
        """TickContext should have sensible defaults."""
        ctx = TickContext(rule_context=rule_context)
        assert ctx.frame_id == 0
        assert ctx.cache == {}
        assert ctx.blackboard is None
        assert ctx.delta_time_ms == 0.0

    def test_cache_get_set(self, rule_context):
        """Cache get/set should work correctly."""
        ctx = TickContext(rule_context=rule_context)

        # Default value
        assert ctx.get_cached("missing", "default") == "default"

        # Set and get
        ctx.set_cached("key1", "value1")
        assert ctx.get_cached("key1") == "value1"

        # Get without default
        assert ctx.get_cached("missing") is None

    def test_cache_clear(self, rule_context):
        """clear_cache should remove all cached values."""
        ctx = TickContext(rule_context=rule_context)
        ctx.set_cached("key1", "value1")
        ctx.set_cached("key2", "value2")

        ctx.clear_cache()

        assert ctx.get_cached("key1") is None
        assert ctx.get_cached("key2") is None


class TestBlackboard:
    """Tests for Blackboard class."""

    def test_create_empty(self):
        """Blackboard should start empty."""
        bb = Blackboard()
        assert len(bb) == 0
        assert bb.keys() == []

    def test_get_set(self):
        """get/set should work correctly."""
        bb = Blackboard()

        # Default value
        assert bb.get("missing", "default") == "default"
        assert bb.get("missing") is None

        # Set and get
        bb.set("key1", "value1")
        assert bb.get("key1") == "value1"

        # Overwrite
        bb.set("key1", "value2")
        assert bb.get("key1") == "value2"

    def test_has(self):
        """has should check key existence."""
        bb = Blackboard()

        assert bb.has("key1") is False

        bb.set("key1", "value1")
        assert bb.has("key1") is True

    def test_delete(self):
        """delete should remove key and return success."""
        bb = Blackboard()
        bb.set("key1", "value1")

        assert bb.delete("key1") is True
        assert bb.has("key1") is False

        # Delete non-existent
        assert bb.delete("missing") is False

    def test_clear(self):
        """clear should remove all keys."""
        bb = Blackboard()
        bb.set("key1", "value1")
        bb.set("key2", "value2")

        bb.clear()

        assert len(bb) == 0
        assert bb.has("key1") is False
        assert bb.has("key2") is False

    def test_keys(self):
        """keys should return all key names."""
        bb = Blackboard()
        bb.set("key1", "value1")
        bb.set("key2", "value2")

        keys = bb.keys()
        assert sorted(keys) == ["key1", "key2"]

    def test_items(self):
        """items should return all key-value pairs."""
        bb = Blackboard()
        bb.set("key1", "value1")
        bb.set("key2", "value2")

        items = bb.items()
        assert sorted(items) == [("key1", "value1"), ("key2", "value2")]

    def test_copy(self):
        """copy should create independent copy."""
        bb1 = Blackboard()
        bb1.set("key1", "value1")

        bb2 = bb1.copy()
        bb2.set("key1", "modified")
        bb2.set("key2", "value2")

        # Original unchanged
        assert bb1.get("key1") == "value1"
        assert bb1.has("key2") is False

        # Copy modified
        assert bb2.get("key1") == "modified"
        assert bb2.get("key2") == "value2"

    def test_namespace(self):
        """Namespace should isolate keys."""
        bb = Blackboard(namespace="plugin1")

        bb.set("key1", "value1")

        # Key is namespaced internally
        assert bb.get("key1") == "value1"
        assert bb.has("key1") is True

        # Keys returned without namespace
        assert bb.keys() == ["key1"]

    def test_namespace_isolation(self):
        """Different namespaces should not collide."""
        # Shared underlying storage
        bb1 = Blackboard(namespace="plugin1")
        bb2 = Blackboard(namespace="plugin2")

        # Manually share storage for testing
        bb2._data = bb1._data

        bb1.set("key1", "value1")
        bb2.set("key1", "value2")

        # Each namespace sees its own value
        assert bb1.get("key1") == "value1"
        assert bb2.get("key1") == "value2"

    def test_len(self):
        """len should return number of keys."""
        bb = Blackboard()
        assert len(bb) == 0

        bb.set("key1", "value1")
        assert len(bb) == 1

        bb.set("key2", "value2")
        assert len(bb) == 2

    def test_repr(self):
        """repr should include useful info."""
        bb = Blackboard()
        bb.set("key1", "value1")

        repr_str = repr(bb)
        assert "Blackboard" in repr_str
        assert "key1" in repr_str


class TestTickContextWithBlackboard:
    """Tests for TickContext with Blackboard integration."""

    def test_tick_context_with_blackboard(self):
        """TickContext should work with Blackboard."""
        rule_context = RuleContext.create_minimal("user1", "project1")
        blackboard = Blackboard()
        blackboard.set("shared_key", "shared_value")

        ctx = TickContext(
            rule_context=rule_context,
            blackboard=blackboard,
        )

        assert ctx.blackboard is not None
        assert ctx.blackboard.get("shared_key") == "shared_value"
