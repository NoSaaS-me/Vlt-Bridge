"""Signal Actions for Oracle Agent BT.

Actions for parsing, processing, and logging XML signals from LLM responses.

Part of feature 020-bt-oracle-agent.
Tasks covered: T021 from tasks-expanded-us2.md

Acceptance Criteria Mapping:
- AC-1: Emits need_turn -> Parsed by parse_response_signal()
- AC-2: Emits context_sufficient -> Same parsing, different type
- AC-3: Emits stuck -> Same parsing, fields include attempted, blocker
- FR-005: Strip signal XML -> strip_signal_from_response()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..state.base import RunStatus
from .oracle import bb_get, bb_set

if TYPE_CHECKING:
    from ..core.context import TickContext

logger = logging.getLogger(__name__)


def parse_response_signal(ctx: "TickContext") -> RunStatus:
    """Parse XML signal from accumulated_content, store in blackboard.

    Algorithm:
    1. Get accumulated_content from blackboard
    2. Call signal_parser.parse_signal()
    3. Store result in bb.last_signal
    4. Update turns_without_signal counter:
       - If signal found: reset to 0
       - If no signal: increment by 1
    5. Set _signal_parsed_this_turn flag
    6. Return SUCCESS

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS always (signal parsing is best-effort).
    """
    bb = ctx.blackboard
    if bb is None:
        logger.warning("parse_response_signal: No blackboard available")
        return RunStatus.SUCCESS  # Non-blocking

    # Get accumulated content to parse
    accumulated = bb_get(bb, "accumulated_content", "")
    if not accumulated:
        logger.debug("parse_response_signal: No accumulated content to parse")
        bb_set(bb, "last_signal", None)
        _increment_turns_without_signal(bb)
        bb_set(bb, "_signal_parsed_this_turn", True)
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Parse signal using the signal_parser service
    try:
        # Try multiple import paths for flexibility
        try:
            from src.services.signal_parser import parse_signal
        except ImportError:
            from backend.src.services.signal_parser import parse_signal

        signal = parse_signal(accumulated)

        if signal is not None:
            # Convert Signal model to dict for blackboard storage
            signal_dict = {
                "type": signal.type.value if hasattr(signal.type, "value") else str(signal.type),
                "confidence": signal.confidence,
                "fields": signal.fields,
                "raw_xml": signal.raw_xml,
                "timestamp": signal.timestamp.isoformat() if hasattr(signal.timestamp, "isoformat") else str(signal.timestamp),
            }
            bb_set(bb, "last_signal", signal_dict)
            bb_set(bb, "turns_without_signal", 0)
            logger.info(f"parse_response_signal: Parsed signal type='{signal_dict['type']}' confidence={signal.confidence:.2f}")
        else:
            bb_set(bb, "last_signal", None)
            _increment_turns_without_signal(bb)
            logger.debug("parse_response_signal: No signal found in response")

    except ImportError as e:
        logger.warning(f"parse_response_signal: signal_parser not available: {e}")
        bb_set(bb, "last_signal", None)
        _increment_turns_without_signal(bb)
    except Exception as e:
        logger.error(f"parse_response_signal: Error parsing signal: {e}")
        bb_set(bb, "last_signal", None)
        _increment_turns_without_signal(bb)

    # Mark as parsed this turn
    bb_set(bb, "_signal_parsed_this_turn", True)
    ctx.mark_progress()
    return RunStatus.SUCCESS


def strip_signal_from_response(ctx: "TickContext") -> RunStatus:
    """Remove signal XML from accumulated_content for user display.

    Algorithm:
    1. Get accumulated_content from blackboard
    2. Use signal_parser.strip_signal() to remove signal block
    3. Update accumulated_content in blackboard
    4. Return SUCCESS

    This ensures the XML signal is not displayed to the user
    while preserving the rest of the response.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS always (stripping is best-effort).
    """
    bb = ctx.blackboard
    if bb is None:
        logger.warning("strip_signal_from_response: No blackboard available")
        return RunStatus.SUCCESS

    accumulated = bb_get(bb, "accumulated_content", "")
    if not accumulated:
        logger.debug("strip_signal_from_response: No content to strip")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    try:
        # Try multiple import paths for flexibility
        try:
            from src.services.signal_parser import strip_signal
        except ImportError:
            from backend.src.services.signal_parser import strip_signal

        cleaned = strip_signal(accumulated)

        if cleaned != accumulated:
            bb_set(bb, "accumulated_content", cleaned)
            logger.debug(f"strip_signal_from_response: Stripped signal, length {len(accumulated)} -> {len(cleaned)}")
        else:
            logger.debug("strip_signal_from_response: No signal to strip")

    except ImportError as e:
        logger.warning(f"strip_signal_from_response: signal_parser not available: {e}")
    except Exception as e:
        logger.error(f"strip_signal_from_response: Error stripping signal: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def log_signal(ctx: "TickContext") -> RunStatus:
    """Log signal to ANS event bus and append to signals_emitted list.

    Algorithm:
    1. Get last_signal from blackboard
    2. If signal exists:
       a. Append to signals_emitted list
       b. Emit signal event to ANS event bus
    3. Return SUCCESS

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS always (logging is best-effort).
    """
    bb = ctx.blackboard
    if bb is None:
        logger.warning("log_signal: No blackboard available")
        return RunStatus.SUCCESS

    last_signal = bb_get(bb, "last_signal")
    if last_signal is None:
        logger.debug("log_signal: No signal to log")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Append to signals_emitted list
    signals_emitted = bb_get(bb, "signals_emitted", [])
    if not isinstance(signals_emitted, list):
        signals_emitted = []

    # Add timestamp if not present
    if isinstance(last_signal, dict) and "timestamp" not in last_signal:
        last_signal = {**last_signal, "timestamp": datetime.now(timezone.utc).isoformat()}

    signals_emitted.append(last_signal)
    bb_set(bb, "signals_emitted", signals_emitted)

    # Emit to ANS event bus
    try:
        # Try multiple import paths for flexibility
        try:
            from src.services.ans.bus import get_event_bus
            from src.services.ans.event import Event, Severity
        except ImportError:
            from backend.src.services.ans.bus import get_event_bus
            from backend.src.services.ans.event import Event, Severity

        signal_type = last_signal.get("type", "unknown") if isinstance(last_signal, dict) else "unknown"
        confidence = last_signal.get("confidence", 0.5) if isinstance(last_signal, dict) else 0.5
        fields = last_signal.get("fields", {}) if isinstance(last_signal, dict) else {}

        bus = get_event_bus()
        bus.emit(Event(
            type=f"agent.signal.{signal_type}",
            source="oracle_bt",
            severity=Severity.INFO,
            payload={
                "signal_type": signal_type,
                "confidence": confidence,
                "fields": fields,
            }
        ))
        logger.debug(f"log_signal: Emitted event agent.signal.{signal_type}")

    except ImportError as e:
        logger.debug(f"log_signal: ANS not available: {e}")
    except Exception as e:
        logger.warning(f"log_signal: Failed to emit signal event: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def update_signal_state(ctx: "TickContext") -> RunStatus:
    """Update consecutive_same_reason counter based on signal pattern.

    Algorithm:
    1. Get last_signal and prev_signal from blackboard
    2. If both are need_turn signals with same reason:
       - Increment consecutive_same_reason
    3. Else:
       - Reset consecutive_same_reason to 1 (if signal) or 0 (if no signal)
    4. Store current signal as prev_signal for next comparison
    5. Return SUCCESS

    This enables loop detection per US3-AC-3.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS always.
    """
    bb = ctx.blackboard
    if bb is None:
        logger.warning("update_signal_state: No blackboard available")
        return RunStatus.SUCCESS

    last_signal = bb_get(bb, "last_signal")
    prev_signal = bb_get(bb, "_prev_signal")

    if last_signal is None:
        # No signal this turn - reset consecutive counter
        bb_set(bb, "consecutive_same_reason", 0)
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Extract type and reason from signals
    last_type = _get_signal_field(last_signal, "type")
    last_reason = _get_nested_field(last_signal, "fields", "reason")

    prev_type = _get_signal_field(prev_signal, "type") if prev_signal else None
    prev_reason = _get_nested_field(prev_signal, "fields", "reason") if prev_signal else None

    # Normalize types for comparison
    last_type_str = str(last_type).lower().replace("signaltype.", "") if last_type else ""
    prev_type_str = str(prev_type).lower().replace("signaltype.", "") if prev_type else ""

    # Check if both are need_turn with same reason
    if last_type_str == "need_turn" and prev_type_str == "need_turn":
        if last_reason and prev_reason and last_reason == prev_reason:
            # Same reason - increment counter
            consecutive = bb_get(bb, "consecutive_same_reason", 0)
            consecutive = (consecutive or 0) + 1
            bb_set(bb, "consecutive_same_reason", consecutive)
            logger.debug(f"update_signal_state: Consecutive same reason: {consecutive}")
        else:
            # Different reason - reset to 1
            bb_set(bb, "consecutive_same_reason", 1)
            logger.debug("update_signal_state: New need_turn reason, reset to 1")
    elif last_type_str == "need_turn":
        # First need_turn signal
        bb_set(bb, "consecutive_same_reason", 1)
        logger.debug("update_signal_state: First need_turn signal")
    else:
        # Non-need_turn signal - reset
        bb_set(bb, "consecutive_same_reason", 0)
        logger.debug(f"update_signal_state: Non-need_turn signal '{last_type_str}', reset to 0")

    # Store current signal as prev for next turn
    bb_set(bb, "_prev_signal", last_signal)

    ctx.mark_progress()
    return RunStatus.SUCCESS


def reset_signal_state(ctx: "TickContext") -> RunStatus:
    """Reset all signal-related state for a new turn.

    Called at the start of each turn to clear the _signal_parsed_this_turn flag.

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS always.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.SUCCESS

    bb_set(bb, "_signal_parsed_this_turn", False)
    ctx.mark_progress()
    return RunStatus.SUCCESS


def check_fallback_needed(ctx: "TickContext") -> RunStatus:
    """Check if BERT fallback should be triggered.

    Fallback conditions (per US5):
    - turns_without_signal >= 3
    - last_signal.confidence < 0.3
    - explicit stuck signal

    Args:
        ctx: The tick context with blackboard access.

    Returns:
        RunStatus.SUCCESS if fallback should be triggered,
        RunStatus.FAILURE otherwise.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    # Check turns without signal
    turns_without = bb_get(bb, "turns_without_signal", 0)
    if turns_without >= 3:
        logger.warning(f"check_fallback_needed: {turns_without} turns without signal")
        return RunStatus.SUCCESS

    # Check low confidence
    last_signal = bb_get(bb, "last_signal")
    if last_signal:
        confidence = _get_signal_field(last_signal, "confidence")
        if confidence is not None and confidence < 0.3:
            logger.warning(f"check_fallback_needed: Low confidence signal ({confidence})")
            return RunStatus.SUCCESS

        # Check explicit stuck
        sig_type = _get_signal_field(last_signal, "type")
        if str(sig_type).lower().replace("signaltype.", "") == "stuck":
            logger.warning("check_fallback_needed: Explicit stuck signal")
            return RunStatus.SUCCESS

    return RunStatus.FAILURE


# =============================================================================
# Helper Functions
# =============================================================================


def _increment_turns_without_signal(bb: Any) -> None:
    """Increment the turns_without_signal counter."""
    turns = bb_get(bb, "turns_without_signal", 0)
    turns = (turns or 0) + 1
    bb_set(bb, "turns_without_signal", turns)


def _get_signal_field(signal: Any, field: str) -> Any:
    """Get a field from signal (dict or object)."""
    if signal is None:
        return None
    if isinstance(signal, dict):
        return signal.get(field)
    return getattr(signal, field, None)


def _get_nested_field(obj: Any, *keys: str) -> Any:
    """Get a nested field from dict/object."""
    if obj is None:
        return None

    current = obj
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return None

    return current


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "parse_response_signal",
    "strip_signal_from_response",
    "log_signal",
    "update_signal_state",
    "reset_signal_state",
    "check_fallback_needed",
]
