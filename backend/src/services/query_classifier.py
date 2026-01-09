"""Query classifier for Oracle agent.

Classifies user queries to determine which context sources to search.
Uses keyword-based heuristics with fallback to conversational.

Expected accuracy: ~80% on common queries (per research.md)

Part of feature 020-bt-oracle-agent.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from ..models.query_classification import QueryClassification, QueryType

logger = logging.getLogger(__name__)


# =============================================================================
# Keyword Dictionaries
# =============================================================================

# Keywords that indicate CODE queries (implementation details)
CODE_KEYWORDS: Set[str] = {
    # Direct code references
    "function",
    "method",
    "class",
    "variable",
    "module",
    "import",
    "package",
    # Implementation questions
    "implement",
    "implementation",
    "code",
    "coding",
    "syntax",
    "error",
    "bug",
    "fix",
    "debug",
    # Location queries
    "where is",
    "find the",
    "locate",
    "which file",
    "what file",
    # How-it-works queries
    "how does",
    "how do",
    "how is",
    "what does",
    # Code-specific terms
    "line",
    "lines",
    "return",
    "parameter",
    "argument",
    "type",
    "interface",
    "api",
    "endpoint",
    "route",
    "handler",
    "controller",
    "model",
    "service",
    "repository",
    "database",
    "query",
    "schema",
}

# Keywords that indicate DOCUMENTATION queries (decisions, architecture)
DOCUMENTATION_KEYWORDS: Set[str] = {
    # Design documents
    "decision",
    "architecture",
    "design",
    "spec",
    "specification",
    "document",
    "documentation",
    "readme",
    # History queries
    "why did we",
    "why was",
    "what did we",
    "when did we",
    "history of",
    # Planning
    "plan",
    "planning",
    "roadmap",
    "milestone",
    "sprint",
    # Process
    "process",
    "workflow",
    "convention",
    "standard",
    "guideline",
    # Team/project
    "meeting",
    "discussion",
    "agreed",
    "consensus",
}

# Keywords that indicate RESEARCH queries (external info)
RESEARCH_KEYWORDS: Set[str] = {
    # Comparison
    "best practice",
    "best practices",
    "compare",
    "comparison",
    "vs",
    "versus",
    "alternative",
    "alternatives",
    # External info
    "latest",
    "new",
    "recent",
    "update",
    "news",
    "trend",
    "trending",
    # Recommendations
    "recommend",
    "recommendation",
    "should we",
    "should i",
    "better",
    "worse",
    "pros and cons",
    # Learning
    "learn",
    "tutorial",
    "guide",
    "how to",
    "example",
    "examples",
    # External references
    "library",
    "framework",
    "tool",
    "package",
    "npm",
    "pip",
    "crate",
}

# Keywords that indicate ACTION queries (write operations)
ACTION_KEYWORDS: Set[str] = {
    # Create operations
    "create",
    "make",
    "generate",
    "new",
    "add",
    # Update operations
    "update",
    "modify",
    "change",
    "edit",
    "rename",
    "move",
    # Save operations
    "save",
    "write",
    "store",
    "persist",
    # Version control
    "commit",
    "push",
    "merge",
    "branch",
    # File operations
    "delete",
    "remove",
    "file",
    "folder",
    "directory",
}

# Keywords that indicate CONVERSATIONAL queries (follow-ups)
CONVERSATIONAL_KEYWORDS: Set[str] = {
    # Acknowledgment
    "thanks",
    "thank you",
    "great",
    "perfect",
    "awesome",
    "cool",
    "ok",
    "okay",
    "got it",
    "understood",
    # Affirmation
    "yes",
    "yeah",
    "yep",
    "sure",
    "right",
    "correct",
    # Negation
    "no",
    "nope",
    "not",
    "nevermind",
    "never mind",
    # Clarification
    "what do you mean",
    "can you explain",
    "more details",
    "elaborate",
    # Short responses
    "hmm",
    "huh",
    "interesting",
}


# =============================================================================
# Classifier Function
# =============================================================================


def classify_query(
    query: str,
    *,
    conversation_context: Optional[str] = None,
) -> QueryClassification:
    """Classify a user query to determine context needs.

    Algorithm:
    1. Normalize query text (lowercase, strip whitespace)
    2. Check for exact phrase matches (multi-word keywords)
    3. Check for single word matches
    4. Score each query type by keyword matches
    5. Return highest-scoring type (conversational if no matches)

    Args:
        query: User's question or command
        conversation_context: Optional previous context for follow-up detection

    Returns:
        QueryClassification with type and context needs

    Example:
        >>> classify_query("Where is the authentication middleware?")
        QueryClassification(query_type=QueryType.CODE, needs_code=True, ...)

        >>> classify_query("Thanks!")
        QueryClassification(query_type=QueryType.CONVERSATIONAL, ...)
    """
    if not query or not isinstance(query, str):
        return QueryClassification.from_type(
            QueryType.CONVERSATIONAL,
            confidence=0.5,
            keywords_matched=[],
        )

    # Normalize query
    normalized = query.lower().strip()

    # Check for very short queries (likely conversational)
    if len(normalized) < 5:
        return QueryClassification.from_type(
            QueryType.CONVERSATIONAL,
            confidence=0.8,
            keywords_matched=["short_query"],
        )

    # Score each query type
    scores: Dict[QueryType, Tuple[float, List[str]]] = {
        QueryType.CODE: _score_keywords(normalized, CODE_KEYWORDS),
        QueryType.DOCUMENTATION: _score_keywords(normalized, DOCUMENTATION_KEYWORDS),
        QueryType.RESEARCH: _score_keywords(normalized, RESEARCH_KEYWORDS),
        QueryType.ACTION: _score_keywords(normalized, ACTION_KEYWORDS),
        QueryType.CONVERSATIONAL: _score_keywords(normalized, CONVERSATIONAL_KEYWORDS),
    }

    # Find highest scoring type
    best_type = QueryType.CONVERSATIONAL
    best_score = 0.0
    best_keywords: List[str] = []

    for qtype, (score, keywords) in scores.items():
        if score > best_score:
            best_score = score
            best_type = qtype
            best_keywords = keywords

    # Calculate confidence based on score and keyword count
    confidence = _calculate_confidence(best_score, len(best_keywords))

    # Check for question patterns to boost confidence
    if _is_question(normalized):
        confidence = min(1.0, confidence + 0.1)

    # If no keywords matched, check for follow-up patterns
    if best_score == 0 and conversation_context:
        if _is_followup(normalized, conversation_context):
            return QueryClassification.from_type(
                QueryType.CONVERSATIONAL,
                confidence=0.7,
                keywords_matched=["followup_detected"],
            )

    logger.debug(
        f"Classified query: type={best_type.value}, "
        f"confidence={confidence:.2f}, keywords={best_keywords}"
    )

    return QueryClassification.from_type(
        best_type,
        confidence=confidence,
        keywords_matched=best_keywords,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _score_keywords(text: str, keywords: Set[str]) -> Tuple[float, List[str]]:
    """Score text against a set of keywords.

    Returns (score, matched_keywords) where score is weighted by:
    - Multi-word matches count more (phrase matching)
    - More matches = higher score
    - Earlier matches in text slightly preferred

    Args:
        text: Normalized text to score
        keywords: Set of keywords to match

    Returns:
        Tuple of (score, list of matched keywords)
    """
    matched: List[str] = []
    score = 0.0

    for keyword in keywords:
        if keyword in text:
            matched.append(keyword)
            # Multi-word keywords count more
            word_count = len(keyword.split())
            score += word_count * 1.0

    return score, matched


def _calculate_confidence(score: float, keyword_count: int) -> float:
    """Calculate confidence from score and keyword count.

    Confidence ranges:
    - 0.9+: Multiple strong matches
    - 0.7-0.9: Good match
    - 0.5-0.7: Weak match
    - <0.5: Very weak/default

    Args:
        score: Keyword match score
        keyword_count: Number of keywords matched

    Returns:
        Confidence value (0.0-1.0)
    """
    if score == 0:
        return 0.4  # Default for no matches

    # Base confidence from score
    base = min(0.9, 0.5 + score * 0.1)

    # Bonus for multiple keywords
    if keyword_count >= 3:
        base = min(1.0, base + 0.1)
    elif keyword_count >= 2:
        base = min(1.0, base + 0.05)

    return base


def _is_question(text: str) -> bool:
    """Check if text is phrased as a question.

    Args:
        text: Normalized text

    Returns:
        True if text appears to be a question
    """
    question_starters = [
        "what",
        "where",
        "when",
        "why",
        "how",
        "which",
        "who",
        "can",
        "could",
        "would",
        "should",
        "is",
        "are",
        "does",
        "do",
    ]

    # Check for question mark
    if text.endswith("?"):
        return True

    # Check for question starters
    first_word = text.split()[0] if text.split() else ""
    return first_word in question_starters


def _is_followup(text: str, context: str) -> bool:
    """Check if text is a follow-up to previous context.

    Simple heuristics:
    - Very short (< 20 chars)
    - Contains pronouns referring back (it, that, this, they)
    - Contains continuation words (also, more, another)

    Args:
        text: Current query (normalized)
        context: Previous conversation context

    Returns:
        True if text appears to be a follow-up
    """
    if len(text) < 20:
        return True

    followup_words = {"it", "that", "this", "they", "them", "also", "more", "another"}
    words = set(text.split())

    return bool(words & followup_words)


__all__ = [
    "classify_query",
    "CODE_KEYWORDS",
    "DOCUMENTATION_KEYWORDS",
    "RESEARCH_KEYWORDS",
    "ACTION_KEYWORDS",
    "CONVERSATIONAL_KEYWORDS",
]
