"""Budget Conditions for BT-Controlled Oracle Agent.

Conditions for checking turn budget status to enable budget enforcement
in the behavior tree control layer.

Part of feature 020-bt-oracle-agent.
Tasks covered: T027 from tasks-expanded-us3.md

Acceptance Criteria Mapping:
- US3-AC1: Agent at 29/30 turns gets one more -> is_at_budget_limit returns SUCCESS at turn 29
- US3-AC2: Agent at 30/30 turns forced completion -> is_over_budget returns SUCCESS at turn 30
- FR-007: Configurable max turn limits -> Uses get_oracle_config().max_turns

Environment Variables (via OracleConfig):
- ORACLE_MAX_TURNS: Maximum turns (default: 30)
- ORACLE_ITERATION_WARNING_THRESHOLD: Warning percentage (default: 0.70)
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from ..state.base import RunStatus

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard

logger = logging.getLogger(__name__)


# =============================================================================
# Blackboard Helpers (matches oracle.py pattern)
# =============================================================================


def _bb_get(bb: "TypedBlackboard", key: str, default: Any = None) -> Any:
    """Get value from blackboard without schema validation.

    Uses _lookup which traverses scope chain.
    """
    value = bb._lookup(key)
    return value if value is not None else default


# =============================================================================
# Budget Conditions
# =============================================================================


def turns_remaining(ctx: "TickContext") -> RunStatus:
    """Check if turns remain in budget.

    Calculates remaining turns as max_turns - current_turn.
    Returns SUCCESS if remaining > 0, FAILURE if budget exhausted.

    This condition is useful for deciding whether to continue the agent loop.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if turn < max_turns (turns remaining),
        RunStatus.FAILURE if turn >= max_turns (no turns left).

    Example:
        >>> turns_remaining(ctx)  # Check if can continue looping
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("turns_remaining: No blackboard available")
        return RunStatus.FAILURE

    # Import config here to avoid circular imports
    try:
        from ...services.config import get_oracle_config
        config = get_oracle_config()
        max_turns = config.max_turns
    except ImportError:
        # Fallback for standalone testing
        max_turns = 30

    turn = _bb_get(bb, "turn", 0)

    # Ensure turn is integer
    try:
        turn = int(turn)
    except (ValueError, TypeError):
        turn = 0

    remaining = max_turns - turn

    logger.debug(f"Budget check: {remaining} turns remaining ({turn}/{max_turns})")

    if remaining > 0:
        return RunStatus.SUCCESS

    logger.warning(f"Budget exhausted: {turn}/{max_turns} turns used")
    return RunStatus.FAILURE


def is_at_budget_limit(ctx: "TickContext") -> RunStatus:
    """Check if at the last allowed turn (turn == max_turns - 1).

    Used to identify when the agent should complete on this turn
    or risk budget exceeded on the next turn.

    Per US3-AC1: Agent at 29/30 turns gets one final turn.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if turn == max_turns - 1 (last turn),
        RunStatus.FAILURE otherwise.

    Example:
        >>> is_at_budget_limit(ctx)  # Check if on final turn
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("is_at_budget_limit: No blackboard available")
        return RunStatus.FAILURE

    # Import config here to avoid circular imports
    try:
        from ...services.config import get_oracle_config
        config = get_oracle_config()
        max_turns = config.max_turns
    except ImportError:
        max_turns = 30

    turn = _bb_get(bb, "turn", 0)

    # Ensure turn is integer
    try:
        turn = int(turn)
    except (ValueError, TypeError):
        turn = 0

    at_limit = turn == max_turns - 1

    if at_limit:
        logger.warning(f"At budget limit: turn {turn} of {max_turns} (last turn)")

    return RunStatus.SUCCESS if at_limit else RunStatus.FAILURE


def is_over_budget(ctx: "TickContext") -> RunStatus:
    """Check if over the turn budget (turn >= max_turns).

    Used to detect when the agent has exceeded its turn allocation
    and must be forced to complete.

    Per US3-AC2: Agent at 30/30 turns forced to complete with partial answer.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if turn >= max_turns (over budget),
        RunStatus.FAILURE if turn < max_turns (within budget).

    Example:
        >>> is_over_budget(ctx)  # Check if must force completion
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("is_over_budget: No blackboard available")
        return RunStatus.FAILURE

    # Import config here to avoid circular imports
    try:
        from ...services.config import get_oracle_config
        config = get_oracle_config()
        max_turns = config.max_turns
    except ImportError:
        max_turns = 30

    turn = _bb_get(bb, "turn", 0)

    # Ensure turn is integer
    try:
        turn = int(turn)
    except (ValueError, TypeError):
        turn = 0

    over = turn >= max_turns

    if over:
        logger.error(f"Over budget: turn {turn} >= max {max_turns}")

    return RunStatus.SUCCESS if over else RunStatus.FAILURE


def budget_warning_needed(ctx: "TickContext") -> RunStatus:
    """Check if budget warning should be emitted (at threshold percentage).

    Returns SUCCESS if current turn is at or above the warning threshold
    (default 70% of max_turns) AND warning hasn't been emitted yet.

    This prevents duplicate warnings during the same session.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if at warning threshold and not yet warned,
        RunStatus.FAILURE otherwise.

    Example:
        >>> budget_warning_needed(ctx)  # Check if should emit warning
    """
    bb = ctx.blackboard
    if bb is None:
        logger.debug("budget_warning_needed: No blackboard available")
        return RunStatus.FAILURE

    # Check if warning already emitted
    warning_emitted = _bb_get(bb, "iteration_warning_emitted", False)
    if warning_emitted:
        logger.debug("budget_warning_needed: Warning already emitted")
        return RunStatus.FAILURE

    # Import config here to avoid circular imports
    try:
        from ...services.config import get_oracle_config
        config = get_oracle_config()
        max_turns = config.max_turns
        warning_threshold = config.iteration_warning_threshold
    except ImportError:
        max_turns = 30
        warning_threshold = 0.70

    turn = _bb_get(bb, "turn", 0)

    # Ensure turn is integer
    try:
        turn = int(turn)
    except (ValueError, TypeError):
        turn = 0

    # Calculate threshold turn
    threshold_turn = int(max_turns * warning_threshold)

    if turn >= threshold_turn:
        percentage = (turn / max_turns * 100) if max_turns > 0 else 0
        logger.info(f"Budget warning needed: turn {turn}/{max_turns} ({percentage:.0f}%)")
        return RunStatus.SUCCESS

    return RunStatus.FAILURE


def get_turns_remaining_count(ctx: "TickContext") -> int:
    """Get the number of turns remaining in budget.

    Helper function (not a condition) that returns the actual count
    for use in messages or logging.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        Integer count of remaining turns, or 0 if blackboard unavailable.
    """
    bb = ctx.blackboard
    if bb is None:
        return 0

    try:
        from ...services.config import get_oracle_config
        config = get_oracle_config()
        max_turns = config.max_turns
    except ImportError:
        max_turns = 30

    turn = _bb_get(bb, "turn", 0)

    try:
        turn = int(turn)
    except (ValueError, TypeError):
        turn = 0

    return max(0, max_turns - turn)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "turns_remaining",
    "is_at_budget_limit",
    "is_over_budget",
    "budget_warning_needed",
    "get_turns_remaining_count",
]
