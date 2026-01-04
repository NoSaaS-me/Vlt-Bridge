"""Unit tests for DecisionTree protocol and registry."""

import pytest

from backend.src.services.decision_tree.protocol import DecisionTree
from backend.src.services.decision_tree.registry import (
    decision_tree,
    get_decision_tree,
    list_decision_trees,
    _decision_trees,
)
from backend.src.models.settings import AgentConfig
from backend.src.models.agent_state import AgentState


class MockDecisionTree:
    """A mock implementation of DecisionTree for testing."""

    def __init__(self, config: AgentConfig):
        self._config = config

    def should_continue(self, state: AgentState) -> tuple[bool, str]:
        return True, ""

    def on_turn_start(self, state: AgentState) -> AgentState:
        return state

    def on_tool_result(self, state: AgentState, result: dict) -> AgentState:
        return state

    def get_config(self) -> AgentConfig:
        return self._config


class TestDecisionTreeProtocol:
    """Test DecisionTree protocol conformance."""

    def test_mock_implements_protocol(self):
        """Verify MockDecisionTree satisfies the Protocol."""
        config = AgentConfig()
        tree = MockDecisionTree(config)
        assert isinstance(tree, DecisionTree)

    def test_protocol_methods_callable(self):
        """Verify all protocol methods are callable."""
        config = AgentConfig()
        tree = MockDecisionTree(config)
        state = AgentState(user_id="test", project_id="proj", config=config)

        # should_continue
        should_continue, reason = tree.should_continue(state)
        assert isinstance(should_continue, bool)
        assert isinstance(reason, str)

        # on_turn_start
        new_state = tree.on_turn_start(state)
        assert isinstance(new_state, AgentState)

        # on_tool_result
        new_state = tree.on_tool_result(state, {"result": "test"})
        assert isinstance(new_state, AgentState)

        # get_config
        returned_config = tree.get_config()
        assert isinstance(returned_config, AgentConfig)


class TestDecisionTreeRegistry:
    """Test decorator-based registry."""

    def setup_method(self):
        """Clear registry before each test."""
        _decision_trees.clear()

    def test_decorator_registers_tree(self):
        """Test that @decision_tree decorator registers implementation."""

        @decision_tree("test_tree")
        class TestTree:
            def should_continue(self, state):
                return True, ""
            def on_turn_start(self, state):
                return state
            def on_tool_result(self, state, result):
                return state
            def get_config(self):
                return AgentConfig()

        assert "test_tree" in list_decision_trees()
        assert get_decision_tree("test_tree") is TestTree

    def test_duplicate_registration_raises(self):
        """Test that registering same name twice raises ValueError."""

        @decision_tree("duplicate")
        class Tree1:
            pass

        with pytest.raises(ValueError, match="already registered"):
            @decision_tree("duplicate")
            class Tree2:
                pass

    def test_get_nonexistent_returns_none(self):
        """Test that getting unregistered tree returns None."""
        assert get_decision_tree("nonexistent") is None

    def test_list_decision_trees(self):
        """Test listing all registered trees."""

        @decision_tree("tree_a")
        class TreeA:
            pass

        @decision_tree("tree_b")
        class TreeB:
            pass

        names = list_decision_trees()
        assert "tree_a" in names
        assert "tree_b" in names
