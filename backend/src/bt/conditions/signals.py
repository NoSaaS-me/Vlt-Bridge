"""Signal Condition Functions for Oracle Agent BT.

Conditions check blackboard state for signal-related values
and return SUCCESS/FAILURE to guide BT control flow.

Part of feature 020-bt-oracle-agent.
Tasks covered: T020 from tasks-expanded-us2.md

Acceptance Criteria Mapping:
- AC-4a: BT parses signal -> check_signal() returns SUCCESS after parsing
- AC-4b: BT acts on signal -> Conditions drive BT routing via selector nodes
- US3-AC-3: 3x same reason = stuck -> consecutive_same_reason_gte(ctx, 3)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..state.base import RunStatus
from ..actions.oracle import bb_get

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard

logger = logging.getLogger(__name__)


def check_signal(ctx: "TickContext") -> RunStatus:
    """Check if a signal was parsed in the current turn.

    Returns SUCCESS if _signal_parsed_this_turn flag is True,
    indicating a signal was extracted from the LLM response.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if signal was parsed this turn,
        RunStatus.FAILURE otherwise.
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("check_signal: No blackboard available")
        return RunStatus.FAILURE

    signal_parsed = bb_get(bb, "_signal_parsed_this_turn", False)

    if signal_parsed:
        logger.debug("check_signal: Signal was parsed this turn")
        return RunStatus.SUCCESS

    logger.debug("check_signal: No signal parsed this turn")
    return RunStatus.FAILURE


def has_signal(ctx: "TickContext", signal_type: Optional[str] = None) -> RunStatus:
    """Check if any/specific signal exists in blackboard.last_signal.

    If signal_type is None, checks if any signal exists.
    If signal_type is provided, checks if the signal matches that type.

    Args:
        ctx: The tick context with blackboard access.
        signal_type: Optional signal type to match (e.g., "need_turn").

    Returns:
        RunStatus.SUCCESS if signal exists (and matches type if specified),
        RunStatus.FAILURE otherwise.

    Example:
        >>> has_signal(ctx)  # Check if any signal exists
        >>> has_signal(ctx, signal_type="need_turn")  # Check for specific type
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("has_signal: No blackboard available")
        return RunStatus.FAILURE

    last_signal = bb_get(bb, "last_signal")
    if last_signal is None:
        logger.debug("has_signal: No signal in blackboard")
        return RunStatus.FAILURE

    # If no specific type required, signal exists = success
    if signal_type is None:
        logger.debug(f"has_signal: Signal exists (any type)")
        return RunStatus.SUCCESS

    # Check signal type - handle both dict and object access
    sig_type = _get_signal_type(last_signal)

    if sig_type is None:
        logger.debug("has_signal: Signal has no type field")
        return RunStatus.FAILURE

    # Normalize for comparison (lowercase, handle enum values)
    sig_type_str = str(sig_type).lower().replace("signaltype.", "")
    expected_type_str = str(signal_type).lower()

    if sig_type_str == expected_type_str:
        logger.debug(f"has_signal: Signal type matches '{signal_type}'")
        return RunStatus.SUCCESS

    logger.debug(f"has_signal: Signal type '{sig_type_str}' != '{expected_type_str}'")
    return RunStatus.FAILURE


def signal_type_is(ctx: "TickContext", expected_type: str) -> RunStatus:
    """Check if last_signal.type matches expected_type exactly.

    This is a stricter version of has_signal for explicit type checking.

    Args:
        ctx: The tick context with blackboard access.
        expected_type: The expected signal type (e.g., "need_turn", "stuck").

    Returns:
        RunStatus.SUCCESS if signal type matches exactly,
        RunStatus.FAILURE otherwise.

    Example:
        >>> signal_type_is(ctx, "need_turn")
        >>> signal_type_is(ctx, "context_sufficient")
        >>> signal_type_is(ctx, "stuck")
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("signal_type_is: No blackboard available")
        return RunStatus.FAILURE

    last_signal = bb_get(bb, "last_signal")
    if last_signal is None:
        logger.debug(f"signal_type_is: No signal to check against '{expected_type}'")
        return RunStatus.FAILURE

    sig_type = _get_signal_type(last_signal)
    if sig_type is None:
        logger.debug("signal_type_is: Signal has no type field")
        return RunStatus.FAILURE

    # Normalize for comparison
    sig_type_str = str(sig_type).lower().replace("signaltype.", "")
    expected_type_str = str(expected_type).lower()

    if sig_type_str == expected_type_str:
        logger.debug(f"signal_type_is: Match '{expected_type}'")
        return RunStatus.SUCCESS

    logger.debug(f"signal_type_is: '{sig_type_str}' != '{expected_type_str}'")
    return RunStatus.FAILURE


def signal_confidence_above(ctx: "TickContext", threshold: float = 0.5) -> RunStatus:
    """Check if last_signal.confidence >= threshold.

    Used to filter out low-confidence signals before acting on them.

    Args:
        ctx: The tick context with blackboard access.
        threshold: Minimum confidence value (0.0-1.0). Default 0.5.

    Returns:
        RunStatus.SUCCESS if confidence >= threshold,
        RunStatus.FAILURE otherwise (including when no signal).

    Example:
        >>> signal_confidence_above(ctx, 0.7)  # High confidence only
        >>> signal_confidence_above(ctx, 0.3)  # Accept moderate confidence
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("signal_confidence_above: No blackboard available")
        return RunStatus.FAILURE

    last_signal = bb_get(bb, "last_signal")
    if last_signal is None:
        logger.debug("signal_confidence_above: No signal in blackboard")
        return RunStatus.FAILURE

    # Get confidence value - handle both dict and object access
    confidence = _get_signal_confidence(last_signal)
    if confidence is None:
        logger.debug("signal_confidence_above: Signal has no confidence field")
        return RunStatus.FAILURE

    try:
        confidence_float = float(confidence)
    except (ValueError, TypeError):
        logger.warning(f"signal_confidence_above: Invalid confidence value '{confidence}'")
        return RunStatus.FAILURE

    if confidence_float >= threshold:
        logger.debug(f"signal_confidence_above: {confidence_float:.2f} >= {threshold}")
        return RunStatus.SUCCESS

    logger.debug(f"signal_confidence_above: {confidence_float:.2f} < {threshold}")
    return RunStatus.FAILURE


def consecutive_same_reason_gte(ctx: "TickContext", count: int = 3) -> RunStatus:
    """Check if consecutive_same_reason >= count (loop detection).

    Used to detect when the agent is stuck in a loop, emitting the same
    need_turn reason multiple times consecutively.

    Per US3-AC-3: 3+ same reason signals indicates stuck state.

    Args:
        ctx: The tick context with blackboard access.
        count: Minimum consecutive same-reason count. Default 3.

    Returns:
        RunStatus.SUCCESS if consecutive_same_reason >= count,
        RunStatus.FAILURE otherwise.

    Example:
        >>> consecutive_same_reason_gte(ctx, 3)  # Default threshold
        >>> consecutive_same_reason_gte(ctx, 5)  # More tolerance
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("consecutive_same_reason_gte: No blackboard available")
        return RunStatus.FAILURE

    consecutive = bb_get(bb, "consecutive_same_reason", 0)

    try:
        consecutive_int = int(consecutive)
    except (ValueError, TypeError):
        logger.warning(f"consecutive_same_reason_gte: Invalid value '{consecutive}'")
        return RunStatus.FAILURE

    if consecutive_int >= count:
        logger.warning(f"consecutive_same_reason_gte: Loop detected ({consecutive_int} >= {count})")
        return RunStatus.SUCCESS

    logger.debug(f"consecutive_same_reason_gte: {consecutive_int} < {count}")
    return RunStatus.FAILURE


def turns_without_signal_gte(ctx: "TickContext", count: int = 3) -> RunStatus:
    """Check if turns_without_signal >= count (fallback trigger).

    Used to trigger BERT fallback when the agent isn't emitting signals.
    Per US5-AC-1: 3+ turns without signal triggers fallback.

    Args:
        ctx: The tick context with blackboard access.
        count: Minimum turns without signal. Default 3.

    Returns:
        RunStatus.SUCCESS if turns_without_signal >= count,
        RunStatus.FAILURE otherwise.
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("turns_without_signal_gte: No blackboard available")
        return RunStatus.FAILURE

    turns = bb_get(bb, "turns_without_signal", 0)

    try:
        turns_int = int(turns)
    except (ValueError, TypeError):
        logger.warning(f"turns_without_signal_gte: Invalid value '{turns}'")
        return RunStatus.FAILURE

    if turns_int >= count:
        logger.warning(f"turns_without_signal_gte: Fallback trigger ({turns_int} >= {count})")
        return RunStatus.SUCCESS

    logger.debug(f"turns_without_signal_gte: {turns_int} < {count}")
    return RunStatus.FAILURE


def signal_is_terminal(ctx: "TickContext") -> RunStatus:
    """Check if the current signal indicates task completion.

    Terminal signals: context_sufficient, stuck, partial_answer

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if signal is terminal,
        RunStatus.FAILURE otherwise.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    last_signal = bb_get(bb, "last_signal")
    if last_signal is None:
        return RunStatus.FAILURE

    sig_type = _get_signal_type(last_signal)
    if sig_type is None:
        return RunStatus.FAILURE

    sig_type_str = str(sig_type).lower().replace("signaltype.", "")
    terminal_types = {"context_sufficient", "stuck", "partial_answer"}

    if sig_type_str in terminal_types:
        logger.debug(f"signal_is_terminal: '{sig_type_str}' is terminal")
        return RunStatus.SUCCESS

    logger.debug(f"signal_is_terminal: '{sig_type_str}' is not terminal")
    return RunStatus.FAILURE


def signal_is_continuation(ctx: "TickContext") -> RunStatus:
    """Check if the current signal requests continuation.

    Continuation signals: need_turn, need_capability, delegation_recommended

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if signal requests continuation,
        RunStatus.FAILURE otherwise.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    last_signal = bb_get(bb, "last_signal")
    if last_signal is None:
        return RunStatus.FAILURE

    sig_type = _get_signal_type(last_signal)
    if sig_type is None:
        return RunStatus.FAILURE

    sig_type_str = str(sig_type).lower().replace("signaltype.", "")
    continuation_types = {"need_turn", "need_capability", "delegation_recommended"}

    if sig_type_str in continuation_types:
        logger.debug(f"signal_is_continuation: '{sig_type_str}' requests continuation")
        return RunStatus.SUCCESS

    logger.debug(f"signal_is_continuation: '{sig_type_str}' does not request continuation")
    return RunStatus.FAILURE


# =============================================================================
# Helper Functions
# =============================================================================


def _get_signal_type(signal: Any) -> Optional[str]:
    """Extract type from signal (dict or object).

    Args:
        signal: Signal data (dict or Signal object).

    Returns:
        Signal type string or None.
    """
    if isinstance(signal, dict):
        return signal.get("type")
    else:
        return getattr(signal, "type", None)


def _get_signal_confidence(signal: Any) -> Optional[float]:
    """Extract confidence from signal (dict or object).

    Args:
        signal: Signal data (dict or Signal object).

    Returns:
        Confidence value or None.
    """
    if isinstance(signal, dict):
        return signal.get("confidence")
    else:
        return getattr(signal, "confidence", None)


def _get_signal_field(signal: Any, field_name: str) -> Any:
    """Extract a specific field from signal's fields dict.

    Args:
        signal: Signal data (dict or Signal object).
        field_name: Name of the field to extract.

    Returns:
        Field value or None.
    """
    if isinstance(signal, dict):
        fields = signal.get("fields", {})
        return fields.get(field_name) if isinstance(fields, dict) else None
    else:
        fields = getattr(signal, "fields", {})
        return fields.get(field_name) if isinstance(fields, dict) else None


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "check_signal",
    "has_signal",
    "signal_type_is",
    "signal_confidence_above",
    "consecutive_same_reason_gte",
    "turns_without_signal_gte",
    "signal_is_terminal",
    "signal_is_continuation",
]
