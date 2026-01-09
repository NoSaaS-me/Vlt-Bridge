"""Loop Detection Conditions for BT-Controlled Oracle Agent.

Conditions for detecting when the agent is stuck in a loop, either:
1. Signal-based: Same need_turn reason emitted 3+ times consecutively
2. Tool-pattern-based: Same tool call sequence repeated (existing detection)

Part of feature 020-bt-oracle-agent.
Tasks covered: T028 from tasks-expanded-us3.md

Acceptance Criteria Mapping:
- US3-AC3: 3 consecutive same reason triggers stuck -> is_stuck_loop returns SUCCESS
- FR-008: Detect loop patterns -> Checks both signal reasons and tool patterns

Environment Variables (via OracleConfig):
- ORACLE_LOOP_THRESHOLD: Consecutive count threshold (default: 3)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..state.base import RunStatus

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


# Default threshold for considering agent stuck (configurable via OracleConfig)
CONSECUTIVE_SAME_REASON_THRESHOLD = 3


# =============================================================================
# Blackboard Helpers
# =============================================================================


def _bb_get(bb: "TypedBlackboard", key: str, default: Any = None) -> Any:
    """Get value from blackboard without schema validation."""
    value = bb._lookup(key)
    return value if value is not None else default


def _get_signal_reason(signal: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract reason from signal fields.

    Args:
        signal: Signal dict with fields key containing reason.

    Returns:
        Reason string or None if not found.
    """
    if signal is None:
        return None

    # Handle both dict and object access
    if isinstance(signal, dict):
        fields = signal.get("fields", {})
        if isinstance(fields, dict):
            return fields.get("reason")
    else:
        fields = getattr(signal, "fields", {})
        if isinstance(fields, dict):
            return fields.get("reason")

    return None


def _get_signal_type(signal: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract type from signal.

    Args:
        signal: Signal dict with type key.

    Returns:
        Type string (normalized lowercase) or None.
    """
    if signal is None:
        return None

    if isinstance(signal, dict):
        sig_type = signal.get("type")
    else:
        sig_type = getattr(signal, "type", None)

    if sig_type is None:
        return None

    # Normalize: handle enum values like "SignalType.NEED_TURN"
    return str(sig_type).lower().replace("signaltype.", "")


def _get_loop_threshold() -> int:
    """Get the configured loop detection threshold.

    Returns:
        Loop threshold from config or default.
    """
    try:
        from ...services.config import get_oracle_config
        config = get_oracle_config()
        return config.loop_threshold
    except ImportError:
        return CONSECUTIVE_SAME_REASON_THRESHOLD


# =============================================================================
# Loop Detection Conditions
# =============================================================================


def is_stuck_loop(ctx: "TickContext") -> RunStatus:
    """Check if agent is stuck in a loop.

    Returns SUCCESS if any loop indicator is triggered:
    1. consecutive_same_reason >= threshold (signal-based loop)
    2. loop_detected == True (tool-pattern loop from existing detection)

    Per US3-AC3: 3 consecutive same need_turn reason signals indicates stuck state.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if stuck loop detected,
        RunStatus.FAILURE if no loop indicators.

    Example:
        >>> is_stuck_loop(ctx)  # Check if should force completion
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("is_stuck_loop: No blackboard available")
        return RunStatus.FAILURE

    threshold = _get_loop_threshold()

    # Check signal-based loop (consecutive same reason)
    consecutive = _bb_get(bb, "consecutive_same_reason", 0)

    try:
        consecutive = int(consecutive)
    except (ValueError, TypeError):
        consecutive = 0

    if consecutive >= threshold:
        logger.warning(
            f"Stuck loop detected: same signal reason {consecutive} times "
            f"(threshold: {threshold})"
        )
        return RunStatus.SUCCESS

    # Check existing tool-pattern loop detection
    loop_detected = _bb_get(bb, "loop_detected", False)
    if loop_detected:
        logger.warning("Stuck loop detected: tool pattern repetition")
        return RunStatus.SUCCESS

    logger.debug(
        f"is_stuck_loop: No loop (consecutive={consecutive}/{threshold}, "
        f"tool_loop={loop_detected})"
    )
    return RunStatus.FAILURE


def has_repeated_signal(ctx: "TickContext") -> RunStatus:
    """Check if last signal reason matches previous signal's reason.

    Used to determine whether consecutive_same_reason should be incremented.
    Only compares need_turn signals since other signal types don't have
    a "reason" field in the same sense.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if last two signals are need_turn with same reason,
        RunStatus.FAILURE otherwise.

    Example:
        >>> has_repeated_signal(ctx)  # Check if should increment loop counter
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("has_repeated_signal: No blackboard available")
        return RunStatus.FAILURE

    signals_emitted = _bb_get(bb, "signals_emitted", [])

    if not isinstance(signals_emitted, list):
        signals_emitted = []

    if len(signals_emitted) < 2:
        logger.debug("has_repeated_signal: Fewer than 2 signals emitted")
        return RunStatus.FAILURE

    last_signal = signals_emitted[-1]
    prev_signal = signals_emitted[-2]

    # Only compare need_turn signals
    last_type = _get_signal_type(last_signal)
    prev_type = _get_signal_type(prev_signal)

    if last_type != "need_turn":
        logger.debug(f"has_repeated_signal: Last signal is '{last_type}', not need_turn")
        return RunStatus.FAILURE

    if prev_type != "need_turn":
        logger.debug(f"has_repeated_signal: Previous signal is '{prev_type}', not need_turn")
        return RunStatus.FAILURE

    # Compare reasons
    last_reason = _get_signal_reason(last_signal)
    prev_reason = _get_signal_reason(prev_signal)

    if last_reason and prev_reason and last_reason == prev_reason:
        logger.debug(f"has_repeated_signal: Same reason '{last_reason}'")
        return RunStatus.SUCCESS

    logger.debug(
        f"has_repeated_signal: Different reasons "
        f"('{last_reason}' vs '{prev_reason}')"
    )
    return RunStatus.FAILURE


def has_repeated_tool_pattern(ctx: "TickContext", window: int = 6) -> RunStatus:
    """Check for tool call loop within a sliding window.

    Examines recent_tool_patterns to detect if the same sequence of
    tool calls is being repeated. This complements signal-based detection.

    Args:
        ctx: The tick context with blackboard access.
        window: Size of the sliding window to check (default: 6).

    Returns:
        RunStatus.SUCCESS if repeated tool pattern detected,
        RunStatus.FAILURE otherwise.

    Example:
        >>> has_repeated_tool_pattern(ctx, window=6)
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("has_repeated_tool_pattern: No blackboard available")
        return RunStatus.FAILURE

    patterns = _bb_get(bb, "recent_tool_patterns", [])

    if not isinstance(patterns, list):
        patterns = []

    if len(patterns) < window:
        logger.debug(
            f"has_repeated_tool_pattern: Not enough patterns "
            f"({len(patterns)} < {window})"
        )
        return RunStatus.FAILURE

    # Get the most recent patterns within window
    recent = patterns[-window:]

    # Check if there's a repeated pattern in the window
    # Simple check: if first half equals second half
    half = window // 2
    first_half = recent[:half]
    second_half = recent[half:half * 2]

    if first_half == second_half and len(first_half) > 0:
        logger.warning(
            f"has_repeated_tool_pattern: Repeated pattern detected: {first_half}"
        )
        return RunStatus.SUCCESS

    # Alternative: check if same tool called 3+ times consecutively
    if len(recent) >= 3:
        last_three = recent[-3:]
        if len(set(last_three)) == 1:
            logger.warning(
                f"has_repeated_tool_pattern: Same tool 3x: {last_three[0]}"
            )
            return RunStatus.SUCCESS

    logger.debug("has_repeated_tool_pattern: No repeated pattern")
    return RunStatus.FAILURE


def consecutive_reason_count(ctx: "TickContext") -> int:
    """Get the current consecutive same-reason count.

    Helper function (not a condition) that returns the actual count
    for use in messages or logging.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        Integer count of consecutive same-reason signals, or 0.
    """
    bb = ctx.blackboard
    if bb is None:
        return 0

    count = _bb_get(bb, "consecutive_same_reason", 0)

    try:
        return int(count)
    except (ValueError, TypeError):
        return 0


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "is_stuck_loop",
    "has_repeated_signal",
    "has_repeated_tool_pattern",
    "consecutive_reason_count",
    "CONSECUTIVE_SAME_REASON_THRESHOLD",
]
