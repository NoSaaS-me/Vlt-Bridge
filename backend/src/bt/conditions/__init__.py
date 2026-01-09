"""BT Condition Functions for Oracle Agent.

Conditions are pure predicate functions that check blackboard state
and return True/False to guide BT control flow.

Part of feature 020-bt-oracle-agent.

Usage in Lua tree:
    Condition("needs_code_context")  -- Calls needs_code_context(ctx)
"""

from __future__ import annotations

from .signals import (
    check_signal,
    has_signal,
    signal_type_is,
    signal_confidence_above,
    consecutive_same_reason_gte,
    turns_without_signal_gte,
    signal_is_terminal,
    signal_is_continuation,
)

from .context_needs import (
    has_query_classification,
    needs_code_context,
    needs_vault_context,
    needs_web_context,
    is_conversational,
    any_context_needed,
    query_type_is,
    classification_confidence_above,
)

__all__ = [
    # Signal conditions (T020)
    "check_signal",
    "has_signal",
    "signal_type_is",
    "signal_confidence_above",
    "consecutive_same_reason_gte",
    "turns_without_signal_gte",
    "signal_is_terminal",
    "signal_is_continuation",
    # Context needs conditions (T015)
    "has_query_classification",
    "needs_code_context",
    "needs_vault_context",
    "needs_web_context",
    "is_conversational",
    "any_context_needed",
    "query_type_is",
    "classification_confidence_above",
]
