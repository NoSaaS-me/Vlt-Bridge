"""Fallback Actions for Oracle Agent BT.

Actions for triggering and applying fallback classification when the agent
fails to emit clear signals.

Part of feature 020-bt-oracle-agent.
Tasks covered: T044 from tasks-expanded-us5.md

Acceptance Criteria Mapping:
- US5-AC-3: Given explicit `stuck` signal, BERT attempts to identify alternative strategy
- US5-AC-4: System functions with heuristic defaults when BERT unavailable
- FR-022: System MUST function with heuristic defaults when BERT unavailable
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..state.base import RunStatus
from .oracle import bb_get, bb_set, _add_pending_chunk

if TYPE_CHECKING:
    from ..core.context import TickContext

logger = logging.getLogger(__name__)


# =============================================================================
# Main Actions
# =============================================================================


def trigger_fallback(ctx: "TickContext") -> RunStatus:
    """Trigger fallback classification and store result in blackboard.

    Algorithm:
    1. Get blackboard state: query, accumulated_content, turns_without_signal, tool_results
    2. Get last signal confidence if available
    3. Call heuristic_classify() from fallback_classifier
    4. Store FallbackClassification in bb.fallback_classification
    5. Log the fallback trigger for audit
    6. Mark progress and return SUCCESS

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS after classification is stored.
        RunStatus.FAILURE if blackboard is unavailable.

    Example:
        In Lua tree:
        BT.action("trigger-fallback", {
            fn = "backend.src.bt.actions.fallback_actions.trigger_fallback"
        })
    """
    bb = ctx.blackboard
    if bb is None:
        logger.warning("trigger_fallback: No blackboard available")
        return RunStatus.FAILURE

    # ==========================================================================
    # Gather state for classification
    # ==========================================================================

    # Get query text (handle different formats)
    query = bb_get(bb, "query") or ""
    if hasattr(query, "question"):
        query = query.question
    elif isinstance(query, dict):
        query = query.get("question", str(query))
    else:
        query = str(query)

    accumulated = bb_get(bb, "accumulated_content") or ""
    turns_without = bb_get(bb, "turns_without_signal") or 0
    tool_results = bb_get(bb, "tool_results") or []

    # Get last signal confidence if available
    last_signal = bb_get(bb, "last_signal")
    last_confidence = None
    last_signal_type = None
    if last_signal:
        if isinstance(last_signal, dict):
            last_confidence = last_signal.get("confidence")
            last_signal_type = last_signal.get("type")
        else:
            last_confidence = getattr(last_signal, "confidence", None)
            last_signal_type = getattr(last_signal, "type", None)

    # ==========================================================================
    # Run classification
    # ==========================================================================

    try:
        # Import classifier (try multiple paths for flexibility)
        try:
            from src.services.fallback_classifier import (
                heuristic_classify,
                log_fallback_trigger,
            )
        except ImportError:
            from backend.src.services.fallback_classifier import (
                heuristic_classify,
                log_fallback_trigger,
            )

        classification = heuristic_classify(
            query=query,
            accumulated_content=accumulated,
            turns_without_signal=turns_without,
            tool_results=tool_results,
            last_signal_confidence=last_confidence,
        )

        # Store classification in blackboard
        bb_set(bb, "fallback_classification", classification.to_dict())

        # Log the trigger for audit (T046)
        log_fallback_trigger(
            reason=classification.reason,
            turns_without_signal=turns_without,
            last_signal_type=last_signal_type,
            last_signal_confidence=last_confidence,
            classification_action=classification.action.value,
            classification_confidence=classification.confidence,
        )

        logger.info(
            f"Fallback triggered: action={classification.action.value}, "
            f"confidence={classification.confidence:.2f}, "
            f"reason={classification.reason}"
        )

    except ImportError as e:
        logger.warning(f"trigger_fallback: fallback_classifier not available: {e}")
        # Store default classification
        bb_set(bb, "fallback_classification", {
            "action": "continue",
            "confidence": 0.5,
            "hint": None,
            "reason": "Fallback classifier unavailable",
        })
    except Exception as e:
        logger.error(f"trigger_fallback: Error during classification: {e}")
        bb_set(bb, "fallback_classification", {
            "action": "continue",
            "confidence": 0.5,
            "hint": None,
            "reason": f"Classification error: {e}",
        })

    ctx.mark_progress()
    return RunStatus.SUCCESS


def apply_heuristic_classification(ctx: "TickContext") -> RunStatus:
    """Apply the fallback classification action.

    Based on FallbackClassification.action:
    - CONTINUE: Do nothing, just log
    - FORCE_RESPONSE: Inject "respond now" system message
    - RETRY_WITH_HINT: Inject hint message
    - ESCALATE: Emit system message to user via ANS/pending chunks

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS after applying the action.

    Example:
        In Lua tree:
        BT.action("apply-fallback", {
            fn = "backend.src.bt.actions.fallback_actions.apply_heuristic_classification"
        })
    """
    bb = ctx.blackboard
    if bb is None:
        logger.warning("apply_heuristic_classification: No blackboard available")
        return RunStatus.FAILURE

    classification = bb_get(bb, "fallback_classification")
    if not classification:
        logger.debug("apply_heuristic_classification: No classification to apply")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    action = classification.get("action", "continue")
    hint = classification.get("hint")
    reason = classification.get("reason", "")
    messages = bb_get(bb, "messages") or []

    # Import FallbackAction enum for comparison
    try:
        try:
            from src.services.fallback_classifier import FallbackAction
        except ImportError:
            from backend.src.services.fallback_classifier import FallbackAction
        action_continue = FallbackAction.CONTINUE.value
        action_force = FallbackAction.FORCE_RESPONSE.value
        action_retry = FallbackAction.RETRY_WITH_HINT.value
        action_escalate = FallbackAction.ESCALATE.value
    except ImportError:
        # Fallback to string constants
        action_continue = "continue"
        action_force = "force_response"
        action_retry = "retry_with_hint"
        action_escalate = "escalate"

    # ==========================================================================
    # Apply action based on classification
    # ==========================================================================

    if action == action_continue:
        logger.debug("apply_heuristic_classification: Action is CONTINUE - no changes")

    elif action == action_force:
        # Inject system message to force response
        force_message = {
            "role": "system",
            "content": (
                "[System] You have gathered sufficient information. "
                "Please provide your final response now."
            ),
        }
        messages.append(force_message)
        bb_set(bb, "messages", messages)
        logger.info("apply_heuristic_classification: Injected FORCE_RESPONSE message")

    elif action == action_retry:
        # Inject hint message to guide retry
        if hint:
            hint_message = {
                "role": "system",
                "content": f"[System Guidance] {hint}",
            }
            messages.append(hint_message)
            bb_set(bb, "messages", messages)
            logger.info(
                f"apply_heuristic_classification: Injected RETRY_WITH_HINT - {hint[:50]}..."
            )
        else:
            logger.warning(
                "apply_heuristic_classification: RETRY_WITH_HINT but no hint provided"
            )

    elif action == action_escalate:
        # Emit system message to user via pending chunks
        escalate_content = (
            "The agent is having difficulty completing this request. "
            "Consider rephrasing your question or breaking it down into smaller parts."
        )
        _add_pending_chunk(bb, {
            "type": "system",
            "content": escalate_content,
            "severity": "warning",
        })

        # Also add to messages so agent sees the escalation
        escalate_message = {
            "role": "system",
            "content": (
                "[System] Unable to make progress on this request. "
                "Acknowledge the limitation and provide what information you can."
            ),
        }
        messages.append(escalate_message)
        bb_set(bb, "messages", messages)
        logger.warning(
            f"apply_heuristic_classification: ESCALATE action - {reason}"
        )

    else:
        logger.warning(
            f"apply_heuristic_classification: Unknown action '{action}'"
        )

    # Clear classification after applying
    bb_set(bb, "fallback_classification", None)

    ctx.mark_progress()
    return RunStatus.SUCCESS


def inject_fallback_hint(ctx: "TickContext") -> RunStatus:
    """Inject fallback hint into conversation messages.

    Adds a system message with the classification hint to guide
    the agent toward a response. Used when a more targeted hint
    injection is needed.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS after injection.

    Example:
        In Lua tree:
        BT.action("inject-fallback-hint", {
            fn = "backend.src.bt.actions.fallback_actions.inject_fallback_hint"
        })
    """
    bb = ctx.blackboard
    if bb is None:
        logger.warning("inject_fallback_hint: No blackboard available")
        return RunStatus.FAILURE

    classification = bb_get(bb, "fallback_classification")
    if not classification:
        logger.debug("inject_fallback_hint: No classification available")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    hint = classification.get("hint")
    if not hint:
        logger.debug("inject_fallback_hint: No hint in classification")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Inject hint as system message
    messages = bb_get(bb, "messages") or []
    messages.append({
        "role": "system",
        "content": f"[Fallback Hint] {hint}",
    })
    bb_set(bb, "messages", messages)

    logger.info(f"inject_fallback_hint: Injected hint - {hint[:50]}...")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def reset_fallback_state(ctx: "TickContext") -> RunStatus:
    """Reset fallback-related state in blackboard.

    Called at the start of each query to ensure clean state.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS after reset.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    bb_set(bb, "fallback_classification", None)
    logger.debug("reset_fallback_state: Cleared fallback classification")

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "trigger_fallback",
    "apply_heuristic_classification",
    "inject_fallback_hint",
    "reset_fallback_state",
]
