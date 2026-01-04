"""
DecisionTree module for Oracle agent turn control.

This module provides:
- DecisionTree Protocol for pluggable control flow
- DefaultDecisionTree implementation with standard termination logic
- Decorator-based registry for skill-specific decision trees

Usage:
    from src.services.decision_tree import (
        DecisionTree,
        DefaultDecisionTree,
        get_decision_tree,
        decision_tree,
    )
"""

from .protocol import DecisionTree
from .default import DefaultDecisionTree
from .registry import decision_tree, get_decision_tree, list_decision_trees

__all__ = [
    "DecisionTree",
    "DefaultDecisionTree",
    "decision_tree",
    "get_decision_tree",
    "list_decision_trees",
]
