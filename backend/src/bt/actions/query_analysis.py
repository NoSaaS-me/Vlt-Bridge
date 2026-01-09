"""Query Analysis Action - Classifies user query for context selection.

Analyzes the user query to determine which context sources (code, vault, web)
should be searched. This enables intelligent context selection per User Story 1.

Part of feature 020-bt-oracle-agent.
Tasks covered: T014 from tasks-expanded-us1.md
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..state.base import RunStatus

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard

logger = logging.getLogger(__name__)


# =============================================================================
# Blackboard Helpers (duplicated for standalone module)
# =============================================================================


def bb_get(bb: "TypedBlackboard", key: str, default: Any = None) -> Any:
    """Get value from blackboard without schema validation.

    Uses _lookup which traverses scope chain.
    """
    value = bb._lookup(key)
    return value if value is not None else default


def bb_set(bb: "TypedBlackboard", key: str, value: Any) -> None:
    """Set value in blackboard without schema validation.

    For internal oracle state that doesn't need Pydantic validation.
    """
    bb._data[key] = value
    bb._writes.add(key)


# =============================================================================
# Query Analysis Action
# =============================================================================


def analyze_query(ctx: "TickContext") -> RunStatus:
    """Analyze user query and classify context needs.

    Reads: bb.query (str | OracleQuery | dict)
    Writes:
        bb.query_classification (dict) - Full classification result
        bb.needs_code (bool) - Should search code index
        bb.needs_vault (bool) - Should search vault/docs
        bb.needs_web (bool) - Should search web

    Returns:
        RunStatus.SUCCESS if classification succeeded
        RunStatus.FAILURE if query is missing or blackboard unavailable

    Algorithm:
        1. Extract question text from various query formats
        2. Call classify_query() from query_classifier service
        3. Store results in blackboard for downstream conditions
        4. Log classification for debugging
    """
    bb = ctx.blackboard
    if bb is None:
        logger.error("analyze_query: No blackboard available")
        return RunStatus.FAILURE

    query = bb_get(bb, "query")
    if not query:
        logger.warning("analyze_query: No query in blackboard")
        return RunStatus.FAILURE

    # Extract question text from various formats
    question = _extract_question(query)
    if not question:
        logger.warning("analyze_query: Could not extract question from query")
        return RunStatus.FAILURE

    # Classify using query_classifier service
    try:
        from src.services.query_classifier import classify_query

        classification = classify_query(question)
    except ImportError as e:
        logger.warning(f"query_classifier not available, using fallback: {e}")
        classification = _fallback_classify(question)
    except Exception as e:
        logger.error(f"Query classification failed: {e}")
        classification = _fallback_classify(question)

    # Store full classification as dict for Lua compatibility
    bb_set(bb, "query_classification", {
        "query_type": classification.query_type.value if hasattr(classification.query_type, "value") else classification.query_type,
        "needs_code": classification.needs_code,
        "needs_vault": classification.needs_vault,
        "needs_web": classification.needs_web,
        "confidence": classification.confidence,
        "keywords_matched": classification.keywords_matched or [],
    })

    # Also store individual flags for easy condition checking
    bb_set(bb, "needs_code", classification.needs_code)
    bb_set(bb, "needs_vault", classification.needs_vault)
    bb_set(bb, "needs_web", classification.needs_web)

    logger.debug(
        f"Query classified: type={classification.query_type}, "
        f"needs_code={classification.needs_code}, "
        f"needs_vault={classification.needs_vault}, "
        f"needs_web={classification.needs_web}, "
        f"confidence={classification.confidence:.2f}"
    )

    ctx.mark_progress()
    return RunStatus.SUCCESS


def _extract_question(query: Any) -> str:
    """Extract question text from various query formats.

    Supports:
    - str: Direct question text
    - OracleQuery: Object with .question attribute
    - dict: Dictionary with "question" key
    - Other: str() fallback

    Args:
        query: The query value from blackboard

    Returns:
        Extracted question text, or empty string if extraction failed
    """
    if isinstance(query, str):
        return query.strip()

    if hasattr(query, "question"):
        # OracleQuery object
        return str(query.question).strip()

    if isinstance(query, dict):
        return str(query.get("question", "")).strip()

    # Fallback to string conversion
    return str(query).strip()


# =============================================================================
# Fallback Classification (when query_classifier unavailable)
# =============================================================================


def _fallback_classify(question: str):
    """Fallback heuristic classification when service unavailable.

    Simple keyword matching for basic routing. Less accurate than
    the full query_classifier service.

    Args:
        question: The question text to classify

    Returns:
        A classification-like object with required attributes
    """
    from dataclasses import dataclass, field
    from typing import List, Optional

    @dataclass
    class FallbackClassification:
        """Minimal classification object for fallback."""
        query_type: str = "conversational"
        needs_code: bool = False
        needs_vault: bool = False
        needs_web: bool = False
        confidence: float = 0.5
        keywords_matched: Optional[List[str]] = field(default_factory=list)

    q = question.lower()

    # Web/research keywords (weather, news, latest, etc.)
    web_keywords = ["weather", "latest", "news", "search", "find online", "current", "today"]
    if any(kw in q for kw in web_keywords):
        return FallbackClassification(
            query_type="research",
            needs_web=True,
            confidence=0.8,
            keywords_matched=[kw for kw in web_keywords if kw in q]
        )

    # Code keywords (function, class, implement, how does, etc.)
    code_keywords = ["function", "class", "implement", "code", "where is", "how does", "how do", "method", "variable", "bug", "error", "fix"]
    if any(kw in q for kw in code_keywords):
        return FallbackClassification(
            query_type="code",
            needs_code=True,
            confidence=0.8,
            keywords_matched=[kw for kw in code_keywords if kw in q]
        )

    # Documentation/vault keywords (decision, architecture, what did we, etc.)
    doc_keywords = ["decision", "architecture", "design", "what did we", "why did we", "documentation", "spec"]
    if any(kw in q for kw in doc_keywords):
        return FallbackClassification(
            query_type="documentation",
            needs_vault=True,
            confidence=0.8,
            keywords_matched=[kw for kw in doc_keywords if kw in q]
        )

    # Conversational keywords (thanks, ok, yes, no, etc.)
    conv_keywords = ["thanks", "thank you", "ok", "okay", "yes", "no", "got it", "cool", "great", "perfect"]
    if any(kw in q for kw in conv_keywords):
        return FallbackClassification(
            query_type="conversational",
            needs_code=False,
            needs_vault=False,
            needs_web=False,
            confidence=0.9,
            keywords_matched=[kw for kw in conv_keywords if kw in q]
        )

    # Default: conversational (safest fallback - no tools)
    return FallbackClassification()


__all__ = ["analyze_query"]
