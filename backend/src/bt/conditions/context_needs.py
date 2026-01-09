"""Context Needs Conditions - Check what context sources are needed.

Conditions that check the query classification to determine which
context sources (code, vault, web) should be searched.

Part of feature 020-bt-oracle-agent.
Tasks covered: T015 from tasks-expanded-us1.md

Acceptance Criteria Mapping:
- AS1.1: Weather query -> needs_web_context returns SUCCESS
- AS1.2: Code query -> needs_code_context returns SUCCESS
- AS1.3: Vault query -> needs_vault_context returns SUCCESS
- AS1.4: Conversational -> is_conversational returns SUCCESS
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..state.base import RunStatus

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _get_classification(ctx: "TickContext") -> Optional[Dict[str, Any]]:
    """Get query classification from blackboard.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        Classification dict if exists and is dict, None otherwise.
    """
    bb = ctx.blackboard
    if bb is None:
        return None

    classification = bb._lookup("query_classification")
    return classification if isinstance(classification, dict) else None


def _get_bool_field(ctx: "TickContext", field: str) -> Optional[bool]:
    """Get a boolean field directly from blackboard.

    Checks the individual flag (e.g., needs_code) directly in blackboard,
    which is set by analyze_query action for easier BT condition access.

    Args:
        ctx: The tick context with blackboard access.
        field: The field name (needs_code, needs_vault, needs_web).

    Returns:
        Boolean value if found, None otherwise.
    """
    bb = ctx.blackboard
    if bb is None:
        return None

    value = bb._lookup(field)
    return bool(value) if value is not None else None


# =============================================================================
# Condition Functions
# =============================================================================


def has_query_classification(ctx: "TickContext") -> RunStatus:
    """Check if query has been classified.

    Used to gate downstream conditions that depend on classification.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if query_classification exists in blackboard,
        RunStatus.FAILURE otherwise.

    Example:
        BT.condition("has-classification", {
            fn = "backend.src.bt.conditions.context_needs.has_query_classification"
        })
    """
    classification = _get_classification(ctx)

    if classification is not None:
        logger.debug("has_query_classification: Classification exists")
        return RunStatus.SUCCESS

    logger.debug("has_query_classification: No classification in blackboard")
    return RunStatus.FAILURE


def needs_code_context(ctx: "TickContext") -> RunStatus:
    """Check if query needs code search.

    Reads bb.query_classification.needs_code or bb.needs_code directly.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if needs_code is True,
        RunStatus.FAILURE if False or not classified.

    Example:
        BT.condition("needs-code-context", {
            fn = "backend.src.bt.conditions.context_needs.needs_code_context"
        })
    """
    # First try direct flag (faster)
    direct_value = _get_bool_field(ctx, "needs_code")
    if direct_value is not None:
        if direct_value:
            logger.debug("needs_code_context: True (from direct flag)")
            return RunStatus.SUCCESS
        else:
            logger.debug("needs_code_context: False (from direct flag)")
            return RunStatus.FAILURE

    # Fall back to classification dict
    classification = _get_classification(ctx)
    if classification is None:
        logger.debug("needs_code_context: No classification available")
        return RunStatus.FAILURE

    needs_code = classification.get("needs_code", False)
    logger.debug(f"needs_code_context: {needs_code} (from classification)")
    return RunStatus.SUCCESS if needs_code else RunStatus.FAILURE


def needs_vault_context(ctx: "TickContext") -> RunStatus:
    """Check if query needs vault/documentation search.

    Reads bb.query_classification.needs_vault or bb.needs_vault directly.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if needs_vault is True,
        RunStatus.FAILURE if False or not classified.

    Example:
        BT.condition("needs-vault-context", {
            fn = "backend.src.bt.conditions.context_needs.needs_vault_context"
        })
    """
    # First try direct flag (faster)
    direct_value = _get_bool_field(ctx, "needs_vault")
    if direct_value is not None:
        if direct_value:
            logger.debug("needs_vault_context: True (from direct flag)")
            return RunStatus.SUCCESS
        else:
            logger.debug("needs_vault_context: False (from direct flag)")
            return RunStatus.FAILURE

    # Fall back to classification dict
    classification = _get_classification(ctx)
    if classification is None:
        logger.debug("needs_vault_context: No classification available")
        return RunStatus.FAILURE

    needs_vault = classification.get("needs_vault", False)
    logger.debug(f"needs_vault_context: {needs_vault} (from classification)")
    return RunStatus.SUCCESS if needs_vault else RunStatus.FAILURE


def needs_web_context(ctx: "TickContext") -> RunStatus:
    """Check if query needs web search.

    Reads bb.query_classification.needs_web or bb.needs_web directly.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if needs_web is True,
        RunStatus.FAILURE if False or not classified.

    Example:
        BT.condition("needs-web-context", {
            fn = "backend.src.bt.conditions.context_needs.needs_web_context"
        })
    """
    # First try direct flag (faster)
    direct_value = _get_bool_field(ctx, "needs_web")
    if direct_value is not None:
        if direct_value:
            logger.debug("needs_web_context: True (from direct flag)")
            return RunStatus.SUCCESS
        else:
            logger.debug("needs_web_context: False (from direct flag)")
            return RunStatus.FAILURE

    # Fall back to classification dict
    classification = _get_classification(ctx)
    if classification is None:
        logger.debug("needs_web_context: No classification available")
        return RunStatus.FAILURE

    needs_web = classification.get("needs_web", False)
    logger.debug(f"needs_web_context: {needs_web} (from classification)")
    return RunStatus.SUCCESS if needs_web else RunStatus.FAILURE


def is_conversational(ctx: "TickContext") -> RunStatus:
    """Check if query is purely conversational (no tools needed).

    Conversational queries are follow-ups, acknowledgments, and other
    responses that don't require any context gathering.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if query_type is "conversational",
        RunStatus.FAILURE otherwise.

    Example:
        BT.condition("is-conversational", {
            fn = "backend.src.bt.conditions.context_needs.is_conversational"
        })
    """
    classification = _get_classification(ctx)
    if classification is None:
        logger.debug("is_conversational: No classification available")
        return RunStatus.FAILURE

    query_type = classification.get("query_type", "")

    # Handle both string and enum values
    query_type_str = str(query_type).lower()

    if query_type_str == "conversational":
        logger.debug("is_conversational: True")
        return RunStatus.SUCCESS

    logger.debug(f"is_conversational: False (type={query_type_str})")
    return RunStatus.FAILURE


def any_context_needed(ctx: "TickContext") -> RunStatus:
    """Check if any context source is needed.

    Returns SUCCESS if at least one of needs_code, needs_vault, needs_web
    is True. Used to gate context gathering phase entirely.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if any context source is needed,
        RunStatus.FAILURE if none needed (conversational).

    Example:
        BT.condition("any-context-needed", {
            fn = "backend.src.bt.conditions.context_needs.any_context_needed"
        })
    """
    classification = _get_classification(ctx)
    if classification is None:
        logger.debug("any_context_needed: No classification, assuming no")
        return RunStatus.FAILURE

    needs_code = classification.get("needs_code", False)
    needs_vault = classification.get("needs_vault", False)
    needs_web = classification.get("needs_web", False)

    if needs_code or needs_vault or needs_web:
        logger.debug(
            f"any_context_needed: True (code={needs_code}, vault={needs_vault}, web={needs_web})"
        )
        return RunStatus.SUCCESS

    logger.debug("any_context_needed: False (no context needed)")
    return RunStatus.FAILURE


def query_type_is(ctx: "TickContext", expected_type: str) -> RunStatus:
    """Check if query type matches expected type exactly.

    Args:
        ctx: The tick context with blackboard access.
        expected_type: Expected query type (code, documentation, research, conversational, action).

    Returns:
        RunStatus.SUCCESS if query_type matches,
        RunStatus.FAILURE otherwise.

    Example:
        BT.condition("is-code-query", {
            fn = "backend.src.bt.conditions.context_needs.query_type_is",
            args = { expected_type = "code" }
        })
    """
    classification = _get_classification(ctx)
    if classification is None:
        logger.debug(f"query_type_is: No classification for '{expected_type}'")
        return RunStatus.FAILURE

    query_type = classification.get("query_type", "")
    query_type_str = str(query_type).lower()
    expected_type_str = str(expected_type).lower()

    if query_type_str == expected_type_str:
        logger.debug(f"query_type_is: Match '{expected_type}'")
        return RunStatus.SUCCESS

    logger.debug(f"query_type_is: '{query_type_str}' != '{expected_type_str}'")
    return RunStatus.FAILURE


def classification_confidence_above(ctx: "TickContext", threshold: float = 0.5) -> RunStatus:
    """Check if classification confidence >= threshold.

    Used to filter low-confidence classifications that might need
    BERT fallback confirmation.

    Args:
        ctx: The tick context with blackboard access.
        threshold: Minimum confidence value (0.0-1.0). Default 0.5.

    Returns:
        RunStatus.SUCCESS if confidence >= threshold,
        RunStatus.FAILURE otherwise.

    Example:
        BT.condition("high-confidence-classification", {
            fn = "backend.src.bt.conditions.context_needs.classification_confidence_above",
            args = { threshold = 0.7 }
        })
    """
    classification = _get_classification(ctx)
    if classification is None:
        logger.debug("classification_confidence_above: No classification")
        return RunStatus.FAILURE

    confidence = classification.get("confidence", 0.0)

    try:
        confidence_float = float(confidence)
    except (ValueError, TypeError):
        logger.warning(f"classification_confidence_above: Invalid confidence '{confidence}'")
        return RunStatus.FAILURE

    if confidence_float >= threshold:
        logger.debug(f"classification_confidence_above: {confidence_float:.2f} >= {threshold}")
        return RunStatus.SUCCESS

    logger.debug(f"classification_confidence_above: {confidence_float:.2f} < {threshold}")
    return RunStatus.FAILURE


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "has_query_classification",
    "needs_code_context",
    "needs_vault_context",
    "needs_web_context",
    "is_conversational",
    "any_context_needed",
    "query_type_is",
    "classification_confidence_above",
]
