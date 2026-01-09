"""
Oracle-specific BT conditions.

Provides reliable Python-based conditions for checking tool_calls
and other Oracle agent state that may have issues with Lua evaluation.
"""

import logging
from typing import TYPE_CHECKING

from src.bt.state.base import RunStatus

if TYPE_CHECKING:
    from src.bt.state.runtime import TickContext

logger = logging.getLogger(__name__)


def _bb_get(bb, key, default=None):
    """Safely get value from blackboard (SimpleBlackboard or TypedBlackboard)."""
    if bb is None:
        return default
    if hasattr(bb, "get"):
        return bb.get(key, default)
    if hasattr(bb, "_data"):
        return bb._data.get(key, default)
    return default


def no_tool_calls(ctx: "TickContext") -> RunStatus:
    """Check if there are NO tool calls pending.

    Returns:
        SUCCESS if tool_calls is empty or None (no tool calls)
        FAILURE if tool_calls has items (has tool calls)
    """
    bb = ctx.blackboard
    tool_calls = _bb_get(bb, "tool_calls", [])

    if not isinstance(tool_calls, (list, tuple)):
        tool_calls = []

    has_calls = len(tool_calls) > 0

    # Return SUCCESS if NO tool calls, FAILURE if has tool calls
    return RunStatus.FAILURE if has_calls else RunStatus.SUCCESS


def has_tool_calls(ctx: "TickContext") -> RunStatus:
    """Check if there ARE tool calls pending.

    Returns:
        SUCCESS if tool_calls has items (has tool calls)
        FAILURE if tool_calls is empty or None (no tool calls)
    """
    bb = ctx.blackboard
    tool_calls = _bb_get(bb, "tool_calls", [])

    if not isinstance(tool_calls, (list, tuple)):
        logger.warning(f"tool_calls is not a list: {type(tool_calls)}")
        tool_calls = []

    has_calls = len(tool_calls) > 0

    logger.info(f"has_tool_calls condition: tool_calls={len(tool_calls)}, returning {'SUCCESS' if has_calls else 'FAILURE'}")

    # Return SUCCESS if HAS tool calls, FAILURE if no tool calls
    return RunStatus.SUCCESS if has_calls else RunStatus.FAILURE


__all__ = ["no_tool_calls", "has_tool_calls"]
