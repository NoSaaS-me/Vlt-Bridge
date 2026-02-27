"""TreeBuilder for constructing behavior trees from TOML rules.

This module provides TreeBuilder for converting TOML rule definitions
into behavior tree structures. Key mappings:

- Rule with condition -> Guard(ConditionNode, ActionNode)
- Rule with script -> ScriptNode
- Multiple rules -> PrioritySelector (sorted by priority)
- Rule sequences -> Sequence of Guard+Action pairs

TreeBuilder respects the existing rule format from rule.py and
integrates with ExpressionEvaluator and ActionDispatcher.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from .types import Blackboard
from .node import BehaviorNode
from .composites import PrioritySelector, Sequence
from .decorators import Guard, Cooldown
from .leaves import ConditionNode, ActionNode, ScriptNode, SuccessNode
from .tree import BehaviorTree

if TYPE_CHECKING:
    from ..rule import Rule, RuleAction, HookPoint
    from ..expression import ExpressionEvaluator
    from ..actions import ActionDispatcher
    from ..lua_sandbox import LuaSandbox


logger = logging.getLogger(__name__)


@dataclass
class TreeBuilderConfig:
    """Configuration for TreeBuilder.

    Attributes:
        evaluator: ExpressionEvaluator for condition nodes.
        dispatcher: ActionDispatcher for action nodes.
        sandbox: LuaSandbox for script nodes.
        default_cooldown_ms: Default cooldown for rules (0 = no cooldown).
    """

    evaluator: Optional["ExpressionEvaluator"] = None
    dispatcher: Optional["ActionDispatcher"] = None
    sandbox: Optional["LuaSandbox"] = None
    default_cooldown_ms: float = 0.0


class TreeBuilder:
    """Builds behavior trees from TOML rule definitions.

    The builder converts the rule format used by RuleLoader into
    behavior tree structures suitable for tick-based evaluation.

    Mapping Strategy:

    1. **Single Rule** -> Guard wrapping ActionNode
       ```
       Guard(condition=rule.condition)
         └── ActionNode(action=rule.action)
       ```

    2. **Script Rule** -> ScriptNode directly
       ```
       ScriptNode(script=rule.script)
       ```

    3. **Multiple Rules (same hook)** -> PrioritySelector
       ```
       PrioritySelector
         ├── Guard(P0 condition) -> ActionNode(P0 action)
         ├── Guard(P1 condition) -> ActionNode(P1 action)
         └── Guard(P2 condition) -> ActionNode(P2 action)
       ```

    4. **Rule with cooldown** -> Cooldown decorator
       ```
       Cooldown(ms=5000)
         └── Guard(condition) -> ActionNode(action)
       ```

    Example:
        >>> from behavior_tree.builder import TreeBuilder, TreeBuilderConfig
        >>> from plugins.loader import RuleLoader
        >>>
        >>> loader = RuleLoader(Path("rules/"))
        >>> rules = loader.load_all()
        >>>
        >>> config = TreeBuilderConfig(
        ...     evaluator=ExpressionEvaluator(),
        ...     dispatcher=ActionDispatcher(event_bus),
        ... )
        >>> builder = TreeBuilder(config)
        >>>
        >>> tree = builder.build_from_rules(
        ...     rules=rules,
        ...     hook=HookPoint.ON_TURN_START,
        ...     name="TurnStartRules",
        ... )
    """

    def __init__(self, config: Optional[TreeBuilderConfig] = None) -> None:
        """Initialize the tree builder.

        Args:
            config: Builder configuration. Uses defaults if not provided.
        """
        self._config = config or TreeBuilderConfig()

    def build_from_rules(
        self,
        rules: list["Rule"],
        hook: Optional["HookPoint"] = None,
        name: str = "RuleTree",
    ) -> BehaviorTree:
        """Build a behavior tree from a list of rules.

        Creates a PrioritySelector containing all rules, sorted by
        priority (highest first). Each rule becomes a Guard+Action
        or ScriptNode subtree.

        Args:
            rules: List of rules to include.
            hook: Optional hook point to filter rules by.
            name: Name for the resulting tree.

        Returns:
            BehaviorTree ready for tick() evaluation.
        """
        # Filter by hook if specified
        if hook is not None:
            rules = [r for r in rules if r.trigger == hook]

        # Filter enabled rules
        rules = [r for r in rules if r.enabled]

        # Sort by priority (highest first)
        rules = sorted(rules, key=lambda r: r.priority, reverse=True)

        logger.debug(f"TreeBuilder: Building tree from {len(rules)} rules")

        if not rules:
            # Empty tree - just a success node
            return BehaviorTree(
                root=SuccessNode(name="NoRules"),
                name=name,
            )

        if len(rules) == 1:
            # Single rule - no selector needed
            root = self._build_rule_node(rules[0])
            return BehaviorTree(root=root, name=name)

        # Multiple rules - wrap in PrioritySelector
        children = [self._build_rule_node(rule) for rule in rules]
        root = PrioritySelector(children=children, name=f"{name}Selector")

        return BehaviorTree(root=root, name=name)

    def build_from_rule(self, rule: "Rule", name: Optional[str] = None) -> BehaviorTree:
        """Build a behavior tree from a single rule.

        Args:
            rule: The rule to build from.
            name: Optional tree name. Defaults to rule ID.

        Returns:
            BehaviorTree for the rule.
        """
        tree_name = name or f"Rule:{rule.id}"
        root = self._build_rule_node(rule)
        return BehaviorTree(root=root, name=tree_name)

    def _build_rule_node(self, rule: "Rule") -> BehaviorNode:
        """Build a behavior node for a single rule.

        Args:
            rule: The rule to build.

        Returns:
            BehaviorNode representing the rule.
        """
        if rule.script:
            # Script-based rule
            node = self._build_script_node(rule)
        else:
            # Condition-based rule
            node = self._build_condition_action_node(rule)

        # Wrap with cooldown if configured
        cooldown_ms = self._config.default_cooldown_ms
        if cooldown_ms > 0:
            node = Cooldown(
                child=node,
                cooldown_ms=cooldown_ms,
                name=f"{rule.id}:Cooldown",
            )

        return node

    def _build_condition_action_node(self, rule: "Rule") -> BehaviorNode:
        """Build a Guard+ActionNode for a condition-based rule.

        Structure:
        ```
        Guard(condition)
          └── ActionNode(action)
        ```

        Args:
            rule: The rule with a condition.

        Returns:
            Guard node wrapping ActionNode.
        """
        # Create action node
        action_node = self._create_action_node(rule)

        # Wrap with guard if condition exists
        if rule.condition:
            guard = Guard(
                child=action_node,
                expression=rule.condition,
                name=f"{rule.id}:Guard",
            )
            return guard

        # No condition - just return action
        return action_node

    def _build_script_node(self, rule: "Rule") -> BehaviorNode:
        """Build a ScriptNode for a script-based rule.

        Args:
            rule: The rule with a script path.

        Returns:
            ScriptNode configured for the rule.
        """
        # Resolve script path relative to rule source
        script_path = self._resolve_script_path(rule)

        node = ScriptNode(
            name=f"{rule.id}:Script",
            script_path=script_path,
            sandbox=self._config.sandbox,
        )

        return node

    def _create_action_node(self, rule: "Rule") -> ActionNode:
        """Create an ActionNode for a rule's action.

        Args:
            rule: The rule with an action.

        Returns:
            ActionNode configured for the action.
        """
        node = ActionNode(
            name=f"{rule.id}:Action",
            rule_action=rule.action,
            dispatcher=self._config.dispatcher,
        )

        return node

    def _resolve_script_path(self, rule: "Rule") -> Optional[str]:
        """Resolve script path relative to rule source.

        Args:
            rule: The rule with a script reference.

        Returns:
            Absolute path to script or None.
        """
        if not rule.script:
            return None

        if not rule.source_path:
            # No source path - try as-is
            return rule.script

        try:
            source_dir = Path(rule.source_path).parent
            script_path = (source_dir / rule.script).resolve()
            return str(script_path)
        except Exception as e:
            logger.warning(f"Failed to resolve script path: {e}")
            return rule.script

    def build_hook_trees(
        self,
        rules: list["Rule"],
    ) -> dict["HookPoint", BehaviorTree]:
        """Build separate trees for each hook point.

        Organizes rules by their trigger hook point and builds
        a tree for each.

        Args:
            rules: List of all rules.

        Returns:
            Dictionary mapping HookPoint to BehaviorTree.
        """
        from ..rule import HookPoint

        trees = {}

        for hook in HookPoint:
            hook_rules = [r for r in rules if r.trigger == hook]
            if hook_rules:
                tree = self.build_from_rules(
                    rules=hook_rules,
                    hook=hook,
                    name=f"Hook:{hook.value}",
                )
                trees[hook] = tree
                logger.debug(
                    f"TreeBuilder: Built tree for {hook.value} "
                    f"with {len(hook_rules)} rules"
                )

        return trees


class DeclarativeTreeBuilder:
    """Fluent builder for constructing trees programmatically.

    Provides a more readable API for building trees in code:

    Example:
        >>> tree = (DeclarativeTreeBuilder("MyTree")
        ...     .selector()
        ...         .guard("context.turn.token_usage > 0.8")
        ...             .action(notify_action)
        ...         .end()
        ...         .guard("context.turn.token_usage > 0.5")
        ...             .action(log_action)
        ...         .end()
        ...     .end()
        ...     .build())
    """

    def __init__(self, name: str = "Tree") -> None:
        """Initialize the declarative builder.

        Args:
            name: Name for the resulting tree.
        """
        self._name = name
        self._stack: list[BehaviorNode] = []
        self._root: Optional[BehaviorNode] = None
        self._blackboard = Blackboard()

    def selector(self, name: str = "Selector") -> "DeclarativeTreeBuilder":
        """Start a PrioritySelector.

        Args:
            name: Name for the selector.

        Returns:
            Self for method chaining.
        """
        node = PrioritySelector(name=name)
        self._add_node(node)
        self._stack.append(node)
        return self

    def sequence(self, name: str = "Sequence") -> "DeclarativeTreeBuilder":
        """Start a Sequence.

        Args:
            name: Name for the sequence.

        Returns:
            Self for method chaining.
        """
        from .composites import Sequence
        node = Sequence(name=name)
        self._add_node(node)
        self._stack.append(node)
        return self

    def guard(
        self,
        expression: str,
        name: str = "Guard",
    ) -> "DeclarativeTreeBuilder":
        """Add a Guard decorator.

        Args:
            expression: Condition expression.
            name: Name for the guard.

        Returns:
            Self for method chaining.
        """
        node = Guard(expression=expression, name=name)
        self._add_node(node)
        self._stack.append(node)
        return self

    def condition(
        self,
        expression: str,
        name: str = "Condition",
    ) -> "DeclarativeTreeBuilder":
        """Add a ConditionNode.

        Args:
            expression: Condition expression.
            name: Name for the node.

        Returns:
            Self for method chaining.
        """
        node = ConditionNode(expression=expression, name=name)
        self._add_node(node)
        return self

    def action(
        self,
        rule_action: Optional["RuleAction"] = None,
        dispatcher: Optional["ActionDispatcher"] = None,
        name: str = "Action",
    ) -> "DeclarativeTreeBuilder":
        """Add an ActionNode.

        Args:
            rule_action: The action to execute.
            dispatcher: Action dispatcher.
            name: Name for the node.

        Returns:
            Self for method chaining.
        """
        node = ActionNode(
            rule_action=rule_action,
            dispatcher=dispatcher,
            name=name,
        )
        self._add_node(node)
        return self

    def success(self, name: str = "Success") -> "DeclarativeTreeBuilder":
        """Add a SuccessNode.

        Args:
            name: Name for the node.

        Returns:
            Self for method chaining.
        """
        node = SuccessNode(name=name)
        self._add_node(node)
        return self

    def cooldown(
        self,
        ms: float = 0.0,
        ticks: int = 0,
        name: str = "Cooldown",
    ) -> "DeclarativeTreeBuilder":
        """Add a Cooldown decorator.

        Args:
            ms: Cooldown duration in milliseconds.
            ticks: Cooldown duration in ticks.
            name: Name for the decorator.

        Returns:
            Self for method chaining.
        """
        node = Cooldown(cooldown_ms=ms, cooldown_ticks=ticks, name=name)
        self._add_node(node)
        self._stack.append(node)
        return self

    def end(self) -> "DeclarativeTreeBuilder":
        """End current composite/decorator scope.

        Returns:
            Self for method chaining.
        """
        if self._stack:
            self._stack.pop()
        return self

    def build(self) -> BehaviorTree:
        """Build and return the behavior tree.

        Returns:
            The constructed BehaviorTree.
        """
        return BehaviorTree(
            root=self._root,
            name=self._name,
            blackboard=self._blackboard,
        )

    def _add_node(self, node: BehaviorNode) -> None:
        """Add a node to the current scope.

        Args:
            node: Node to add.
        """
        if not self._stack:
            # No parent - this is the root
            if self._root is None:
                self._root = node
            return

        parent = self._stack[-1]

        # Add to parent based on type
        from .node import Composite, Decorator

        if isinstance(parent, Composite):
            parent.add_child(node)
        elif isinstance(parent, Decorator) and parent.child is None:
            parent.child = node


__all__ = [
    "TreeBuilder",
    "TreeBuilderConfig",
    "DeclarativeTreeBuilder",
]
