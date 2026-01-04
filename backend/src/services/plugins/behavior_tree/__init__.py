"""Behavior tree system for the Oracle Plugin System.

This package provides a behavior tree implementation based on
Honorbuddy patterns (see research.md Section 7.2):

- **RunStatus**: Tri-state result (Success, Failure, Running)
- **Frame locking**: O(1) resume from cached running node
- **Priority-based selection**: First match wins pattern
- **Composites**: PrioritySelector, Sequence, Parallel
- **Decorators**: Guard, Cooldown, Inverter, etc.
- **Leaves**: ConditionNode, ActionNode, ScriptNode, etc.

Key Classes:
- BehaviorTree: Main entry point with tick() and frame locking
- TreeBuilder: Constructs trees from TOML rule definitions
- PrioritySelector: Rule evaluation with short-circuit
- Guard: Condition-gated execution
- ConditionNode: Expression evaluation leaf
- ActionNode: Action execution leaf

Example:
    >>> from behavior_tree import BehaviorTree, PrioritySelector, Guard, ActionNode
    >>> from behavior_tree.builder import TreeBuilder, TreeBuilderConfig
    >>>
    >>> # Build from rules
    >>> config = TreeBuilderConfig(evaluator=eval, dispatcher=dispatch)
    >>> builder = TreeBuilder(config)
    >>> tree = builder.build_from_rules(rules, hook=HookPoint.ON_TURN_START)
    >>>
    >>> # Or construct programmatically
    >>> tree = BehaviorTree(
    ...     root=PrioritySelector([
    ...         Guard(
    ...             expression="context.turn.token_usage > 0.8",
    ...             child=ActionNode(rule_action=notify_action),
    ...         ),
    ...     ]),
    ...     name="TokenWarning",
    ... )
    >>>
    >>> # Evaluate
    >>> result = tree.tick(rule_context)
    >>> print(f"Status: {result.status}, Time: {result.elapsed_ms:.3f}ms")

Performance Targets (from behavior-tree-tasks.md):
- Single node tick: <100ns (Cython, future)
- Tree traversal (100 nodes): <10us with frame locking
- Condition evaluation: <1us with cached expressions
- Full hook point cycle: <100us

Current implementation is pure Python for correctness.
Cython optimization path is documented in behavior-tree-tasks.md.
"""

from .types import (
    RunStatus,
    TickContext,
    Blackboard,
)

from .node import (
    BehaviorNode,
    Composite,
    Decorator,
    Leaf,
)

from .composites import (
    PrioritySelector,
    Sequence,
    Parallel,
    ParallelPolicy,
    MemorySelector,
    MemorySequence,
)

from .decorators import (
    Inverter,
    Succeeder,
    Failer,
    UntilFail,
    UntilSuccess,
    Cooldown,
    Guard,
    Retry,
    Timeout,
    Repeat,
)

from .leaves import (
    SuccessNode,
    FailureNode,
    RunningNode,
    ConditionNode,
    ActionNode,
    WaitNode,
    ScriptNode,
    BlackboardCondition,
    BlackboardSet,
    LogNode,
)

from .tree import (
    BehaviorTree,
    BehaviorTreeManager,
    TickResult,
)

from .builder import (
    TreeBuilder,
    TreeBuilderConfig,
    DeclarativeTreeBuilder,
)


__all__ = [
    # Types
    "RunStatus",
    "TickContext",
    "Blackboard",
    # Base classes
    "BehaviorNode",
    "Composite",
    "Decorator",
    "Leaf",
    # Composites
    "PrioritySelector",
    "Sequence",
    "Parallel",
    "ParallelPolicy",
    "MemorySelector",
    "MemorySequence",
    # Decorators
    "Inverter",
    "Succeeder",
    "Failer",
    "UntilFail",
    "UntilSuccess",
    "Cooldown",
    "Guard",
    "Retry",
    "Timeout",
    "Repeat",
    # Leaves
    "SuccessNode",
    "FailureNode",
    "RunningNode",
    "ConditionNode",
    "ActionNode",
    "WaitNode",
    "ScriptNode",
    "BlackboardCondition",
    "BlackboardSet",
    "LogNode",
    # Tree
    "BehaviorTree",
    "BehaviorTreeManager",
    "TickResult",
    # Builder
    "TreeBuilder",
    "TreeBuilderConfig",
    "DeclarativeTreeBuilder",
]
