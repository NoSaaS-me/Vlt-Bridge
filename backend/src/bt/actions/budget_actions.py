"""Budget Enforcement Actions for BT-Controlled Oracle Agent.

Actions for budget enforcement including force completion, budget warnings,
and budget exceeded events via the Agent Notification System (ANS).

Part of feature 020-bt-oracle-agent.
Tasks covered: T029 from tasks-expanded-us3.md

Acceptance Criteria Mapping:
- US3-AC2: Force completion with partial answer -> force_completion()
- US3-AC4: BERT fallback on stuck -> force_completion() callable from fallback
- FR-007: Enforce maximum turn limits -> force_completion() enforces limit
- FR-009: Log all signals -> Logger calls for budget events

ANS Event Types Emitted:
- budget.iteration.warning: At 70% of max turns
- budget.iteration.exceeded: When budget is exhausted
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
    """Get value from blackboard without schema validation."""
    value = bb._lookup(key)
    return value if value is not None else default


def _bb_set(bb: "TypedBlackboard", key: str, value: Any) -> None:
    """Set value in blackboard without schema validation."""
    bb._data[key] = value
    bb._writes.add(key)


def _get_oracle_config():
    """Get OracleConfig, handling import gracefully."""
    try:
        from ...services.config import get_oracle_config
        return get_oracle_config()
    except ImportError:
        # Return mock config for standalone testing
        class MockConfig:
            max_turns = 30
            iteration_warning_threshold = 0.70
            token_warning_threshold = 0.80
            context_warning_threshold = 0.70
            loop_threshold = 3
        return MockConfig()


# =============================================================================
# Budget Actions
# =============================================================================


def force_completion(ctx: "TickContext") -> RunStatus:
    """Force agent to complete with current accumulated content.

    Called when budget is exceeded or stuck loop detected.
    Appends a truncation notice to the accumulated content and sets
    the force_complete flag to signal the tree to exit the loop.

    Per US3-AC2: Agent at 30/30 turns forced to complete with partial answer.

    Side Effects:
        - Appends truncation notice to accumulated_content
        - Sets force_complete = True on blackboard
        - Emits budget.iteration.exceeded ANS event

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS always (force completion is authoritative).
    """
    import traceback
    logger.warning(f"force_completion: CALLED! Stack trace:\n{''.join(traceback.format_stack()[-5:])}")

    bb = ctx.blackboard
    if bb is None:
        logger.error("force_completion: No blackboard available")
        return RunStatus.FAILURE

    config = _get_oracle_config()
    turn = _bb_get(bb, "turn", 0)
    max_turns = config.max_turns

    # Ensure turn is integer
    try:
        turn = int(turn)
    except (ValueError, TypeError):
        turn = 0

    # Append truncation notice to accumulated content
    accumulated = _bb_get(bb, "accumulated_content", "")

    if accumulated and isinstance(accumulated, str) and accumulated.strip():
        accumulated += "\n\n[Response truncated due to budget limit]"
    else:
        accumulated = "[Unable to complete: budget limit reached]"

    _bb_set(bb, "accumulated_content", accumulated)

    # Set force complete flag to exit loop
    _bb_set(bb, "force_complete", True)

    # Emit budget exceeded event to ANS
    try:
        from ...services.ans.bus import get_event_bus
        from ...services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="budget.iteration.exceeded",
            source="oracle_bt",
            severity=Severity.ERROR,
            payload={
                "turn": turn,
                "max_turns": max_turns,
                "percentage": (turn / max_turns * 100) if max_turns > 0 else 100,
                "forced": True,
                "reason": "force_completion called"
            }
        ))
        logger.info("force_completion: Emitted budget.iteration.exceeded event")
    except ImportError as e:
        logger.debug(f"force_completion: ANS not available: {e}")
    except Exception as e:
        logger.warning(f"force_completion: Failed to emit budget exceeded event: {e}")

    logger.warning(f"Forced completion at turn {turn}/{max_turns}")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def emit_budget_warning(ctx: "TickContext") -> RunStatus:
    """Emit ANS budget warning event at threshold (70% default).

    Called when the agent has used 70% of its turn budget.
    Only emits once per session (checks iteration_warning_emitted flag).

    This provides visibility to the user/system that the agent is
    approaching its budget limit.

    Side Effects:
        - Sets iteration_warning_emitted = True on blackboard
        - Emits budget.iteration.warning ANS event (if not already emitted)

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS always (warning emission is best-effort).
    """
    bb = ctx.blackboard
    if bb is None:
        logger.warning("emit_budget_warning: No blackboard available")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Check if already emitted
    if _bb_get(bb, "iteration_warning_emitted", False):
        logger.debug("emit_budget_warning: Warning already emitted")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    config = _get_oracle_config()
    turn = _bb_get(bb, "turn", 0)
    max_turns = config.max_turns
    warning_threshold = config.iteration_warning_threshold

    # Ensure turn is integer
    try:
        turn = int(turn)
    except (ValueError, TypeError):
        turn = 0

    # Check if at warning threshold
    threshold_turn = int(max_turns * warning_threshold)

    if turn < threshold_turn:
        logger.debug(
            f"emit_budget_warning: Not at threshold yet "
            f"(turn {turn} < threshold {threshold_turn})"
        )
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Mark as emitted to prevent duplicates
    _bb_set(bb, "iteration_warning_emitted", True)

    # Emit warning event to ANS
    remaining = max_turns - turn
    percentage = (turn / max_turns * 100) if max_turns > 0 else 0

    try:
        from ...services.ans.bus import get_event_bus
        from ...services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="budget.iteration.warning",
            source="oracle_bt",
            severity=Severity.WARNING,
            payload={
                "turn": turn,
                "max_turns": max_turns,
                "percentage": percentage,
                "remaining": remaining,
                "threshold": warning_threshold
            }
        ))
        logger.info(
            f"emit_budget_warning: Emitted budget.iteration.warning "
            f"(turn {turn}/{max_turns}, {percentage:.0f}%)"
        )
    except ImportError as e:
        logger.debug(f"emit_budget_warning: ANS not available: {e}")
    except Exception as e:
        logger.warning(f"emit_budget_warning: Failed to emit warning: {e}")

    logger.info(f"Budget warning: {turn}/{max_turns} turns used ({percentage:.0f}%)")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def emit_budget_exceeded(ctx: "TickContext") -> RunStatus:
    """Emit ANS budget exceeded event.

    Called when the agent has exhausted its turn budget.
    Only emits once per session (checks iteration_exceeded_emitted flag).

    This is separate from force_completion to allow the tree to
    emit the event without necessarily forcing completion.

    Side Effects:
        - Sets iteration_exceeded_emitted = True on blackboard
        - Emits budget.iteration.exceeded ANS event (if not already emitted)

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS always (event emission is best-effort).
    """
    bb = ctx.blackboard
    if bb is None:
        logger.warning("emit_budget_exceeded: No blackboard available")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Check if already emitted
    if _bb_get(bb, "iteration_exceeded_emitted", False):
        logger.debug("emit_budget_exceeded: Event already emitted")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    config = _get_oracle_config()
    turn = _bb_get(bb, "turn", 0)
    max_turns = config.max_turns

    # Ensure turn is integer
    try:
        turn = int(turn)
    except (ValueError, TypeError):
        turn = 0

    # Mark as emitted to prevent duplicates
    _bb_set(bb, "iteration_exceeded_emitted", True)

    # Emit exceeded event to ANS
    try:
        from ...services.ans.bus import get_event_bus
        from ...services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="budget.iteration.exceeded",
            source="oracle_bt",
            severity=Severity.ERROR,
            payload={
                "turn": turn,
                "max_turns": max_turns,
                "percentage": 100.0,
                "reason": "budget_exceeded"
            }
        ))
        logger.info("emit_budget_exceeded: Emitted budget.iteration.exceeded event")
    except ImportError as e:
        logger.debug(f"emit_budget_exceeded: ANS not available: {e}")
    except Exception as e:
        logger.warning(f"emit_budget_exceeded: Failed to emit exceeded event: {e}")

    logger.error(f"Budget exceeded: turn {turn}/{max_turns}")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def emit_loop_warning(ctx: "TickContext") -> RunStatus:
    """Emit ANS loop detection warning event.

    Called when the agent is detected to be in a stuck loop.
    Sets loop_warning_emitted flag and emits agent.loop.detected event.

    Side Effects:
        - Sets loop_warning_emitted = True on blackboard
        - Emits agent.loop.detected ANS event

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS always (event emission is best-effort).
    """
    bb = ctx.blackboard
    if bb is None:
        logger.warning("emit_loop_warning: No blackboard available")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Check if already emitted
    if _bb_get(bb, "loop_warning_emitted", False):
        logger.debug("emit_loop_warning: Warning already emitted")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Mark as emitted
    _bb_set(bb, "loop_warning_emitted", True)

    # Get loop details
    consecutive = _bb_get(bb, "consecutive_same_reason", 0)
    last_signal = _bb_get(bb, "last_signal")

    reason = None
    if last_signal and isinstance(last_signal, dict):
        fields = last_signal.get("fields", {})
        if isinstance(fields, dict):
            reason = fields.get("reason")

    # Emit loop detected event
    try:
        from ...services.ans.bus import get_event_bus
        from ...services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="agent.loop.detected",
            source="oracle_bt",
            severity=Severity.WARNING,
            payload={
                "consecutive_same_reason": consecutive,
                "repeated_reason": reason,
                "action": "force_completion_pending"
            }
        ))
        logger.info(
            f"emit_loop_warning: Emitted agent.loop.detected event "
            f"(consecutive={consecutive}, reason='{reason}')"
        )
    except ImportError as e:
        logger.debug(f"emit_loop_warning: ANS not available: {e}")
    except Exception as e:
        logger.warning(f"emit_loop_warning: Failed to emit loop warning: {e}")

    logger.warning(
        f"Loop detected: {consecutive} consecutive same-reason signals "
        f"(reason: '{reason}')"
    )
    ctx.mark_progress()
    return RunStatus.SUCCESS


def increment_turn(ctx: "TickContext") -> RunStatus:
    """Increment the turn counter.

    Called at the start of each agent turn to track iteration count.
    This is the official turn increment action for the BT.

    Side Effects:
        - Increments turn counter by 1

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if incremented successfully,
        RunStatus.FAILURE if no blackboard.
    """
    bb = ctx.blackboard
    if bb is None:
        logger.error("increment_turn: No blackboard available")
        return RunStatus.FAILURE

    turn = _bb_get(bb, "turn", 0)

    try:
        turn = int(turn)
    except (ValueError, TypeError):
        turn = 0

    turn += 1
    _bb_set(bb, "turn", turn)

    logger.debug(f"increment_turn: Turn is now {turn}")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def check_budget_and_warn(ctx: "TickContext") -> RunStatus:
    """Combined action: check budget and emit warning if needed.

    Convenience action that checks if at warning threshold and emits
    the budget warning if so. Always succeeds to not block the tree.

    This is useful for trees that want a single action for budget checking.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS always.
    """
    bb = ctx.blackboard
    if bb is None:
        ctx.mark_progress()
        return RunStatus.SUCCESS

    config = _get_oracle_config()
    turn = _bb_get(bb, "turn", 0)
    max_turns = config.max_turns
    warning_threshold = config.iteration_warning_threshold

    try:
        turn = int(turn)
    except (ValueError, TypeError):
        turn = 0

    threshold_turn = int(max_turns * warning_threshold)

    # Emit warning if at threshold and not yet warned
    if turn >= threshold_turn and not _bb_get(bb, "iteration_warning_emitted", False):
        emit_budget_warning(ctx)

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "force_completion",
    "emit_budget_warning",
    "emit_budget_exceeded",
    "emit_loop_warning",
    "increment_turn",
    "check_budget_and_warn",
]
