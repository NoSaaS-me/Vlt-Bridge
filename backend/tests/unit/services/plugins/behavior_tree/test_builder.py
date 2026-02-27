"""Unit tests for TreeBuilder and DeclarativeTreeBuilder."""

import pytest

from backend.src.services.plugins.behavior_tree.types import RunStatus, Blackboard
from backend.src.services.plugins.behavior_tree.builder import (
    TreeBuilder,
    TreeBuilderConfig,
    DeclarativeTreeBuilder,
)
from backend.src.services.plugins.behavior_tree.composites import PrioritySelector
from backend.src.services.plugins.behavior_tree.decorators import Guard
from backend.src.services.plugins.behavior_tree.leaves import ActionNode, SuccessNode
from backend.src.services.plugins.rule import (
    Rule,
    RuleAction,
    ActionType,
    HookPoint,
    Priority,
)
from backend.src.services.plugins.context import RuleContext


@pytest.fixture
def rule_context():
    """Create a RuleContext for testing."""
    return RuleContext.create_minimal("user1", "project1")


@pytest.fixture
def sample_rules():
    """Create sample rules for testing."""
    return [
        Rule(
            id="high-priority",
            name="High Priority Rule",
            description="Test high priority",
            trigger=HookPoint.ON_TURN_START,
            condition="context.turn.token_usage > 0.9",
            action=RuleAction(type=ActionType.LOG, message="High usage"),
            priority=100,
            enabled=True,
        ),
        Rule(
            id="medium-priority",
            name="Medium Priority Rule",
            description="Test medium priority",
            trigger=HookPoint.ON_TURN_START,
            condition="context.turn.token_usage > 0.5",
            action=RuleAction(type=ActionType.LOG, message="Medium usage"),
            priority=50,
            enabled=True,
        ),
        Rule(
            id="low-priority",
            name="Low Priority Rule",
            description="Test low priority",
            trigger=HookPoint.ON_TURN_START,
            condition="True",
            action=RuleAction(type=ActionType.LOG, message="Default"),
            priority=10,
            enabled=True,
        ),
    ]


class TestTreeBuilder:
    """Tests for TreeBuilder class."""

    def test_create_with_config(self):
        """TreeBuilder should accept config."""
        config = TreeBuilderConfig()
        builder = TreeBuilder(config)

        assert builder._config is config

    def test_create_without_config(self):
        """TreeBuilder should work without config."""
        builder = TreeBuilder()
        assert builder._config is not None

    def test_build_empty_rules(self, rule_context):
        """build_from_rules with empty list should return success node."""
        builder = TreeBuilder()

        tree = builder.build_from_rules([], name="EmptyTree")

        assert tree.name == "EmptyTree"
        result = tree.tick(rule_context)
        assert result.status == RunStatus.SUCCESS

    def test_build_single_rule(self, sample_rules, rule_context):
        """build_from_rules with single rule should not use selector."""
        builder = TreeBuilder()
        rules = [sample_rules[0]]

        tree = builder.build_from_rules(rules, name="SingleRule")

        assert tree.name == "SingleRule"
        # Root should be Guard (not selector)
        assert isinstance(tree.root, Guard)

    def test_build_multiple_rules_creates_selector(self, sample_rules):
        """build_from_rules with multiple rules should use PrioritySelector."""
        builder = TreeBuilder()

        tree = builder.build_from_rules(sample_rules, name="MultiRule")

        assert isinstance(tree.root, PrioritySelector)
        # Should have 3 children
        assert len(tree.root.children) == 3

    def test_rules_sorted_by_priority(self, sample_rules):
        """Rules should be sorted by priority (highest first)."""
        builder = TreeBuilder()

        tree = builder.build_from_rules(sample_rules)

        # First child should be high priority
        selector = tree.root
        first_child = selector.children[0]
        assert "high-priority" in first_child.name

    def test_filter_by_hook(self, sample_rules):
        """build_from_rules should filter by hook point."""
        # Add rule with different hook
        sample_rules.append(
            Rule(
                id="different-hook",
                name="Different Hook",
                description="Test different hook",
                trigger=HookPoint.ON_TOOL_COMPLETE,
                condition="True",
                action=RuleAction(type=ActionType.LOG, message="Tool complete"),
                priority=100,
            )
        )

        builder = TreeBuilder()

        tree = builder.build_from_rules(
            sample_rules,
            hook=HookPoint.ON_TURN_START,
        )

        # Should only have turn start rules
        selector = tree.root
        assert len(selector.children) == 3

    def test_filter_disabled_rules(self, sample_rules):
        """build_from_rules should skip disabled rules."""
        sample_rules[0].enabled = False

        builder = TreeBuilder()

        tree = builder.build_from_rules(sample_rules)

        # Should only have 2 children
        selector = tree.root
        assert len(selector.children) == 2

    def test_build_from_single_rule(self, sample_rules, rule_context):
        """build_from_rule should create tree for single rule."""
        builder = TreeBuilder()
        rule = sample_rules[0]

        tree = builder.build_from_rule(rule)

        assert tree.name == f"Rule:{rule.id}"
        assert tree.root is not None

    def test_build_hook_trees(self, sample_rules):
        """build_hook_trees should create tree per hook."""
        # Add rules with different hooks
        sample_rules.append(
            Rule(
                id="tool-rule",
                name="Tool Rule",
                description="Test tool hook",
                trigger=HookPoint.ON_TOOL_COMPLETE,
                condition="True",
                action=RuleAction(type=ActionType.LOG, message="Tool"),
                priority=100,
            )
        )

        builder = TreeBuilder()

        trees = builder.build_hook_trees(sample_rules)

        assert HookPoint.ON_TURN_START in trees
        assert HookPoint.ON_TOOL_COMPLETE in trees
        assert len(trees) == 2


class TestDeclarativeTreeBuilder:
    """Tests for DeclarativeTreeBuilder fluent API."""

    def test_create_simple_tree(self, rule_context):
        """Should create simple tree with fluent API."""
        tree = (
            DeclarativeTreeBuilder("SimpleTree")
            .success()
            .build()
        )

        assert tree.name == "SimpleTree"
        result = tree.tick(rule_context)
        assert result.status == RunStatus.SUCCESS

    def test_selector_with_guards(self, rule_context):
        """Should create selector with guards."""
        tree = (
            DeclarativeTreeBuilder("GuardTree")
            .selector()
                .guard("context.turn.token_usage > 0.9")
                    .success()
                .end()
                .guard("True")
                    .success()
                .end()
            .end()
            .build()
        )

        # Should have selector as root
        assert isinstance(tree.root, PrioritySelector)
        assert len(tree.root.children) == 2

    def test_sequence(self, rule_context):
        """Should create sequence."""
        from backend.src.services.plugins.behavior_tree.composites import Sequence

        tree = (
            DeclarativeTreeBuilder("SeqTree")
            .sequence()
                .success()
                .success()
            .end()
            .build()
        )

        assert isinstance(tree.root, Sequence)
        result = tree.tick(rule_context)
        assert result.status == RunStatus.SUCCESS

    def test_nested_structures(self, rule_context):
        """Should handle nested selectors and sequences."""
        tree = (
            DeclarativeTreeBuilder("Nested")
            .selector()
                .sequence()
                    .condition("True")
                    .success()
                .end()
            .end()
            .build()
        )

        assert isinstance(tree.root, PrioritySelector)
        result = tree.tick(rule_context)
        assert result.status == RunStatus.SUCCESS

    def test_cooldown_decorator(self):
        """Should add cooldown decorator."""
        tree = (
            DeclarativeTreeBuilder("CooldownTree")
            .cooldown(ms=1000)
                .success()
            .end()
            .build()
        )

        from backend.src.services.plugins.behavior_tree.decorators import Cooldown
        assert isinstance(tree.root, Cooldown)

    def test_uses_provided_blackboard(self):
        """Should use blackboard from tree."""
        tree = (
            DeclarativeTreeBuilder("BBTree")
            .success()
            .build()
        )

        assert tree.blackboard is not None


class TestTreeBuilderWithConfig:
    """Tests for TreeBuilder with full configuration."""

    def test_cooldown_config(self, sample_rules, rule_context):
        """Config cooldown should wrap rules."""
        config = TreeBuilderConfig(default_cooldown_ms=1000)
        builder = TreeBuilder(config)

        tree = builder.build_from_rules([sample_rules[0]])

        from backend.src.services.plugins.behavior_tree.decorators import Cooldown
        assert isinstance(tree.root, Cooldown)


class TestScriptRuleBuilding:
    """Tests for building trees from script-based rules."""

    def test_script_rule_creates_script_node(self):
        """Script rules should create ScriptNode."""
        rule = Rule(
            id="script-rule",
            name="Script Rule",
            description="Test script",
            trigger=HookPoint.ON_TURN_START,
            script="test.lua",
            action=RuleAction(type=ActionType.LOG, message="Script ran"),
            priority=100,
        )

        builder = TreeBuilder()

        tree = builder.build_from_rule(rule)

        from backend.src.services.plugins.behavior_tree.leaves import ScriptNode
        assert isinstance(tree.root, ScriptNode)


class TestRulePriorityOrdering:
    """Tests for rule priority ordering in trees."""

    def test_highest_priority_first(self, rule_context):
        """Highest priority rules should be first in selector."""
        rules = [
            Rule(
                id="low",
                name="Low",
                description="Low priority",
                trigger=HookPoint.ON_TURN_START,
                condition="True",
                action=RuleAction(type=ActionType.LOG, message="Low"),
                priority=10,
            ),
            Rule(
                id="high",
                name="High",
                description="High priority",
                trigger=HookPoint.ON_TURN_START,
                condition="True",
                action=RuleAction(type=ActionType.LOG, message="High"),
                priority=100,
            ),
            Rule(
                id="medium",
                name="Medium",
                description="Medium priority",
                trigger=HookPoint.ON_TURN_START,
                condition="True",
                action=RuleAction(type=ActionType.LOG, message="Medium"),
                priority=50,
            ),
        ]

        builder = TreeBuilder()
        tree = builder.build_from_rules(rules)

        # Check order by name
        selector = tree.root
        assert "high" in selector.children[0].name
        assert "medium" in selector.children[1].name
        assert "low" in selector.children[2].name
