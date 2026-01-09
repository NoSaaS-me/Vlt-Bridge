"""Fallback Condition Functions for Oracle Agent BT.

Conditions for determining when BERT fallback classification should trigger.
These conditions check signal state and determine if the agent needs assistance.

Part of feature 020-bt-oracle-agent.
Tasks covered: T043 from tasks-expanded-us5.md

Acceptance Criteria Mapping:
- FR-019: BERT fallback activates when no signal for 3+ turns
- FR-020: BERT fallback activates when signal confidence < 0.3
- FR-021: BERT fallback activates on explicit `stuck` signal
- US5-AC-1: Given agent response with no signal, when 3 turns pass, BERT fallback activates
- US5-AC-2: Given signal with confidence < 0.3, BERT fallback is consulted
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..state.base import RunStatus
from ..actions.oracle import bb_get

if TYPE_CHECKING:
    from ..core.context import TickContext

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Per US5-AC-1: 3+ turns without signal triggers fallback
TURNS_WITHOUT_SIGNAL_THRESHOLD = 3

# Per US5-AC-2: Confidence < 0.3 triggers fallback
LOW_CONFIDENCE_THRESHOLD = 0.3


# =============================================================================
# Composite Condition
# =============================================================================


def needs_fallback(ctx: "TickContext") -> bool:
    """Check if fallback classification should be triggered.

    Fallback triggers when ANY of the following conditions are met:
    1. No signal emitted for 3+ consecutive turns (FR-019)
    2. Last signal has confidence < 0.3 (FR-020)
    3. Last signal is explicitly type "stuck" (FR-021)

    This is used by the BT to decide whether to invoke the fallback classifier
    to provide alternative guidance to the agent.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        True if fallback should be activated, False otherwise.

    Example:
        >>> # In Lua tree:
        >>> # BT.condition("needs-fallback", {
        >>> #     fn = "backend.src.bt.conditions.fallback.needs_fallback"
        >>> # })
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("needs_fallback: No blackboard available")
        return False

    # Condition 1: No signal for 3+ turns (FR-019, US5-AC-1)
    turns_without_signal = bb_get(bb, "turns_without_signal") or 0
    if turns_without_signal >= TURNS_WITHOUT_SIGNAL_THRESHOLD:
        logger.info(
            f"needs_fallback: Triggered - {turns_without_signal} turns without signal "
            f"(threshold: {TURNS_WITHOUT_SIGNAL_THRESHOLD})"
        )
        return True

    # Get last signal for remaining checks
    last_signal = bb_get(bb, "last_signal")
    if last_signal:
        # Condition 2: Low confidence signal (FR-020, US5-AC-2)
        confidence = _get_signal_confidence(last_signal)
        if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
            logger.info(
                f"needs_fallback: Triggered - signal confidence {confidence:.2f} "
                f"< threshold {LOW_CONFIDENCE_THRESHOLD}"
            )
            return True

        # Condition 3: Explicit stuck signal (FR-021)
        signal_type = _get_signal_type(last_signal)
        if signal_type and signal_type.lower() == "stuck":
            logger.info("needs_fallback: Triggered - explicit 'stuck' signal detected")
            return True

    logger.debug("needs_fallback: No fallback trigger conditions met")
    return False


# =============================================================================
# Individual Conditions
# =============================================================================


def no_signal_for_n_turns(ctx: "TickContext", n: int = 3) -> bool:
    """Check if no signal has been emitted for n consecutive turns.

    Used to detect when the agent isn't communicating its state,
    which may indicate confusion or inability to progress.

    Args:
        ctx: The tick context with blackboard access.
        n: Minimum turns without signal (default: 3 per US5-AC-1).

    Returns:
        True if turns_without_signal >= n, False otherwise.

    Example:
        >>> no_signal_for_n_turns(ctx)  # Default threshold
        >>> no_signal_for_n_turns(ctx, n=5)  # Custom threshold
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("no_signal_for_n_turns: No blackboard available")
        return False

    turns_without_signal = bb_get(bb, "turns_without_signal") or 0

    result = turns_without_signal >= n
    if result:
        logger.debug(f"no_signal_for_n_turns: True ({turns_without_signal} >= {n})")
    else:
        logger.debug(f"no_signal_for_n_turns: False ({turns_without_signal} < {n})")

    return result


def signal_confidence_below(ctx: "TickContext", threshold: float = 0.3) -> bool:
    """Check if last signal confidence is below threshold.

    Low confidence signals indicate the agent is uncertain about its state,
    which may warrant fallback assistance.

    Args:
        ctx: The tick context with blackboard access.
        threshold: Confidence threshold (default: 0.3 per US5-AC-2).

    Returns:
        True if last_signal.confidence < threshold, False otherwise.
        Returns False if no signal exists.

    Example:
        >>> signal_confidence_below(ctx)  # Default threshold
        >>> signal_confidence_below(ctx, threshold=0.5)  # Custom threshold
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("signal_confidence_below: No blackboard available")
        return False

    last_signal = bb_get(bb, "last_signal")
    if not last_signal:
        logger.debug("signal_confidence_below: No last signal")
        return False

    confidence = _get_signal_confidence(last_signal)
    if confidence is None:
        logger.debug("signal_confidence_below: Signal has no confidence field")
        return False

    result = confidence < threshold
    if result:
        logger.debug(
            f"signal_confidence_below: True ({confidence:.2f} < {threshold})"
        )
    else:
        logger.debug(
            f"signal_confidence_below: False ({confidence:.2f} >= {threshold})"
        )

    return result


def is_stuck_signal(ctx: "TickContext") -> bool:
    """Check if last signal was type 'stuck'.

    A stuck signal indicates the agent has acknowledged it cannot proceed,
    which should trigger fallback assistance.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        True if last_signal.type == "stuck", False otherwise.

    Example:
        >>> if is_stuck_signal(ctx):
        ...     # Trigger fallback classification
        ...     pass
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("is_stuck_signal: No blackboard available")
        return False

    last_signal = bb_get(bb, "last_signal")
    if not last_signal:
        logger.debug("is_stuck_signal: No last signal")
        return False

    signal_type = _get_signal_type(last_signal)
    if signal_type is None:
        logger.debug("is_stuck_signal: Signal has no type field")
        return False

    # Normalize and compare
    normalized_type = signal_type.lower().replace("signaltype.", "")
    result = normalized_type == "stuck"

    if result:
        logger.debug("is_stuck_signal: True - stuck signal detected")
    else:
        logger.debug(f"is_stuck_signal: False - signal type is '{normalized_type}'")

    return result


# =============================================================================
# Wrapper for BT RunStatus (for condition nodes)
# =============================================================================


def needs_fallback_condition(ctx: "TickContext") -> RunStatus:
    """BT-compatible wrapper for needs_fallback that returns RunStatus.

    This allows the condition to be used directly in the BT where
    conditions return RunStatus instead of bool.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if fallback should trigger,
        RunStatus.FAILURE otherwise.
    """
    return RunStatus.SUCCESS if needs_fallback(ctx) else RunStatus.FAILURE


# =============================================================================
# Helper Functions
# =============================================================================


def _get_signal_type(signal: object) -> str | None:
    """Extract type from signal (dict or object)."""
    if isinstance(signal, dict):
        return signal.get("type")
    return getattr(signal, "type", None)


def _get_signal_confidence(signal: object) -> float | None:
    """Extract confidence from signal (dict or object)."""
    if isinstance(signal, dict):
        conf = signal.get("confidence")
    else:
        conf = getattr(signal, "confidence", None)

    if conf is None:
        return None

    try:
        return float(conf)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Constants
    "TURNS_WITHOUT_SIGNAL_THRESHOLD",
    "LOW_CONFIDENCE_THRESHOLD",
    # Composite condition
    "needs_fallback",
    "needs_fallback_condition",
    # Individual conditions
    "no_signal_for_n_turns",
    "signal_confidence_below",
    "is_stuck_signal",
]
