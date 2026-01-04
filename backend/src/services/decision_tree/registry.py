"""Registry for DecisionTree implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type, Dict, Optional

if TYPE_CHECKING:
    from src.services.decision_tree.protocol import DecisionTree

# Registry of decision tree implementations
_decision_trees: Dict[str, Type["DecisionTree"]] = {}


def decision_tree(name: str):
    """
    Decorator to register a DecisionTree implementation.

    Usage:
        @decision_tree("default")
        class DefaultDecisionTree:
            ...

        @decision_tree("deep_researcher")
        class DeepResearcherTree:
            ...

    Args:
        name: Unique name for this decision tree

    Returns:
        Decorator function
    """
    def decorator(cls: Type["DecisionTree"]) -> Type["DecisionTree"]:
        if name in _decision_trees:
            raise ValueError(f"Decision tree '{name}' is already registered")
        _decision_trees[name] = cls
        return cls
    return decorator


def get_decision_tree(name: str) -> Optional[Type["DecisionTree"]]:
    """
    Get a registered decision tree by name.

    Args:
        name: Name of the decision tree

    Returns:
        DecisionTree class or None if not found
    """
    return _decision_trees.get(name)


def list_decision_trees() -> list[str]:
    """Return list of all registered decision tree names."""
    return list(_decision_trees.keys())
