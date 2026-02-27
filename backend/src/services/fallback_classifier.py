"""Fallback Classifier for Oracle Agent.

Provides heuristic-based fallback classification when the agent fails to emit
clear signals. This is a BERT placeholder using simple heuristics.

Part of feature 020-bt-oracle-agent.
Tasks covered: T042, T046 from tasks-expanded-us5.md

Acceptance Criteria Mapping:
- FR-019: BERT fallback activates when no signal for 3+ turns
- FR-020: BERT fallback activates when signal confidence < 0.3
- FR-021: BERT fallback activates on explicit `stuck` signal
- FR-022: System functions with heuristic defaults when BERT unavailable
- SC-006: System functions correctly with BERT disabled (heuristic-only mode)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("fallback")


# =============================================================================
# Data Models
# =============================================================================


class FallbackAction(Enum):
    """Actions the fallback classifier can recommend."""

    CONTINUE = "continue"  # Let agent continue normally
    FORCE_RESPONSE = "force_response"  # Force agent to respond now
    RETRY_WITH_HINT = "retry_with_hint"  # Retry with guidance
    ESCALATE = "escalate"  # Surface to user, acknowledge limitation


@dataclass
class FallbackClassification:
    """Result of fallback classification.

    Attributes:
        action: Recommended fallback action.
        confidence: Classification confidence (0.0-1.0).
        hint: Optional guidance for retry.
        reason: Why this classification was made.
    """

    action: FallbackAction
    confidence: float
    hint: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for blackboard storage."""
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "hint": self.hint,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FallbackClassification":
        """Create from dictionary."""
        return cls(
            action=FallbackAction(data.get("action", "continue")),
            confidence=data.get("confidence", 0.5),
            hint=data.get("hint"),
            reason=data.get("reason", ""),
        )


# =============================================================================
# Classifier Function
# =============================================================================


def heuristic_classify(
    query: str,
    accumulated_content: str,
    turns_without_signal: int,
    tool_results: List[Dict[str, Any]],
    last_signal_confidence: Optional[float] = None,
) -> FallbackClassification:
    """Classify fallback action using heuristics (BERT placeholder).

    Analyzes the current agent state and recommends an appropriate fallback action:
    - CONTINUE: No intervention needed
    - FORCE_RESPONSE: Agent has enough context, should respond now
    - RETRY_WITH_HINT: Agent is stuck, provide guidance for retry
    - ESCALATE: Cannot proceed, surface limitation to user

    Algorithm:
    1. Check accumulated content length - substantial response suggests FORCE_RESPONSE
    2. Analyze tool results - high failure rate suggests ESCALATE
    3. Check query complexity - simple queries without tools suggest FORCE_RESPONSE
    4. Detect stuck patterns - no progress after multiple turns suggests RETRY_WITH_HINT
    5. Calculate confidence based on signal clarity

    Args:
        query: Original user question.
        accumulated_content: Response content accumulated so far.
        turns_without_signal: Count of turns without signal emission.
        tool_results: List of tool execution results with {name, success, result/error}.
        last_signal_confidence: Last signal confidence (if any).

    Returns:
        FallbackClassification with action, confidence, optional hint, and reason.

    Example:
        >>> result = heuristic_classify(
        ...     query="How do I fix this bug?",
        ...     accumulated_content="",
        ...     turns_without_signal=4,
        ...     tool_results=[{"name": "search_code", "success": False}],
        ... )
        >>> result.action
        FallbackAction.RETRY_WITH_HINT
    """
    # Analyze current state
    has_content = len(accumulated_content) > 500
    content_length = len(accumulated_content)
    query_words = len(query.split()) if query else 0

    # Analyze tool results
    tool_count = len(tool_results)
    tool_failures = sum(1 for r in tool_results if not r.get("success", True))
    tool_successes = tool_count - tool_failures
    failure_rate = tool_failures / tool_count if tool_count > 0 else 0.0

    # Get failed tool names for hints
    failed_tools = [
        r.get("name", "unknown") for r in tool_results if not r.get("success", True)
    ]

    # ==========================================================================
    # Decision Logic
    # ==========================================================================

    # Priority 1: High tool failure rate -> ESCALATE
    # If most tools are failing, the agent cannot make progress
    if tool_count >= 3 and failure_rate > 0.7:
        return FallbackClassification(
            action=FallbackAction.ESCALATE,
            confidence=0.7,
            hint=None,
            reason=f"High tool failure rate: {tool_failures}/{tool_count} tools failed",
        )

    # Priority 2: Substantial content accumulated -> FORCE_RESPONSE
    # If we have enough content and the agent isn't signaling, force completion
    if has_content and turns_without_signal >= 2:
        return FallbackClassification(
            action=FallbackAction.FORCE_RESPONSE,
            confidence=0.8,
            hint=None,
            reason=f"Sufficient content accumulated ({content_length} chars) without signal",
        )

    # Priority 3: Simple query with no tool usage -> FORCE_RESPONSE
    # Simple conversational queries don't need tools
    if query_words < 20 and tool_count == 0:
        return FallbackClassification(
            action=FallbackAction.FORCE_RESPONSE,
            confidence=0.75,
            hint="This appears to be a simple query. Please provide a direct answer.",
            reason="Simple query without tool usage",
        )

    # Priority 4: No progress after multiple turns -> RETRY_WITH_HINT
    # Agent is stuck - provide guidance
    if turns_without_signal >= 3 and not has_content:
        hint = "Previous attempts did not yield results. "
        if failed_tools:
            hint += f"Tools {failed_tools} failed. Try alternative approaches or acknowledge the limitation."
        else:
            hint += "Consider whether you have enough context to provide a partial answer."

        return FallbackClassification(
            action=FallbackAction.RETRY_WITH_HINT,
            confidence=0.6,
            hint=hint,
            reason=f"No progress after {turns_without_signal} turns without signal",
        )

    # Priority 5: Some content but many turns without signal -> FORCE_RESPONSE
    # Agent has something but isn't confident enough to finish
    if content_length > 100 and turns_without_signal >= 3:
        return FallbackClassification(
            action=FallbackAction.FORCE_RESPONSE,
            confidence=0.65,
            hint="Please provide your best answer based on available information.",
            reason=f"Partial content ({content_length} chars) after {turns_without_signal} turns",
        )

    # Priority 6: Low confidence signal with some content -> hint toward completion
    if last_signal_confidence is not None and last_signal_confidence < 0.3:
        if content_length > 0:
            return FallbackClassification(
                action=FallbackAction.FORCE_RESPONSE,
                confidence=0.6,
                hint="Your last signal had low confidence. Please provide your response.",
                reason=f"Low signal confidence ({last_signal_confidence:.2f})",
            )
        else:
            return FallbackClassification(
                action=FallbackAction.RETRY_WITH_HINT,
                confidence=0.55,
                hint="Your last signal had low confidence. Try a different approach or acknowledge limitations.",
                reason=f"Low signal confidence ({last_signal_confidence:.2f}) with no content",
            )

    # Default: Continue normally
    return FallbackClassification(
        action=FallbackAction.CONTINUE,
        confidence=0.5,
        hint=None,
        reason="No clear fallback trigger detected",
    )


# =============================================================================
# Logging Function (T046)
# =============================================================================


def log_fallback_trigger(
    reason: str,
    turns_without_signal: int,
    last_signal_type: Optional[str],
    last_signal_confidence: Optional[float],
    classification_action: str,
    classification_confidence: float,
) -> None:
    """Log fallback activation for debugging and audit.

    Logs to both Python logger and ANS event bus for observability.

    Args:
        reason: Why fallback was triggered.
        turns_without_signal: Number of turns without signal emission.
        last_signal_type: Type of the last signal (if any).
        last_signal_confidence: Confidence of the last signal (if any).
        classification_action: The action chosen by the classifier.
        classification_confidence: Confidence of the classification.

    Per FR-009: BT logs all signals for debugging and audit.
    Per SC-009: All signals are logged and auditable.
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "turns_without_signal": turns_without_signal,
        "last_signal_type": last_signal_type,
        "last_signal_confidence": last_signal_confidence,
        "classification_action": classification_action,
        "classification_confidence": classification_confidence,
    }

    # Python logging
    if classification_action == FallbackAction.ESCALATE.value:
        logger.warning(f"Fallback ESCALATE: {log_entry}")
    else:
        logger.info(f"Fallback triggered: {log_entry}")

    # Emit ANS event for system observability
    try:
        # Use relative import within backend package
        try:
            from src.services.ans.bus import get_event_bus
            from src.services.ans.event import Event, Severity
        except ImportError:
            from backend.src.services.ans.bus import get_event_bus
            from backend.src.services.ans.event import Event, Severity

        severity = (
            Severity.WARNING
            if classification_action == FallbackAction.ESCALATE.value
            else Severity.INFO
        )

        bus = get_event_bus()
        bus.emit(
            Event(
                type="fallback.triggered",
                source="fallback_classifier",
                severity=severity,
                payload=log_entry,
            )
        )
        logger.debug("Emitted fallback.triggered event to ANS")

    except ImportError as e:
        logger.debug(f"ANS not available for fallback logging: {e}")
    except Exception as e:
        logger.warning(f"Failed to emit fallback event: {e}")


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "FallbackAction",
    "FallbackClassification",
    "heuristic_classify",
    "log_fallback_trigger",
]
