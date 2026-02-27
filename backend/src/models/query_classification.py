"""Pydantic models for query classification.

Query classification determines which context sources to search
based on the nature of the user's question.

Part of feature 020-bt-oracle-agent.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class QueryType(str, Enum):
    """Classification of user query intent.

    Each type maps to different context sources and prompt segments.
    """

    CODE = "code"
    """Questions about implementation, function locations, how things work."""

    DOCUMENTATION = "documentation"
    """Questions about decisions, architecture, specs, design docs."""

    RESEARCH = "research"
    """External information needs, best practices, comparisons."""

    CONVERSATIONAL = "conversational"
    """Follow-ups, acknowledgments, clarifications, thanks."""

    ACTION = "action"
    """Write operations: create, update, save, push."""


# Mapping of query type to context needs
QUERY_TYPE_CONTEXT_NEEDS: Dict[QueryType, Dict[str, bool]] = {
    QueryType.CODE: {"needs_code": True, "needs_vault": False, "needs_web": False},
    QueryType.DOCUMENTATION: {"needs_code": False, "needs_vault": True, "needs_web": False},
    QueryType.RESEARCH: {"needs_code": False, "needs_vault": False, "needs_web": True},
    QueryType.CONVERSATIONAL: {"needs_code": False, "needs_vault": False, "needs_web": False},
    QueryType.ACTION: {"needs_code": False, "needs_vault": True, "needs_web": False},
}


class QueryClassification(BaseModel):
    """Result of analyzing user query to determine context needs.

    This classification drives:
    1. Which context sources to search (code, vault, web)
    2. Which prompt segments to include in system prompt
    3. Budget allocation across retrieval sources

    Not persisted - computed fresh for each query.
    """

    query_type: QueryType = Field(
        ...,
        description="Primary classification of the query",
    )
    needs_code: bool = Field(
        ...,
        description="Should search code index (CodeRAG)",
    )
    needs_vault: bool = Field(
        ...,
        description="Should search vault/documentation (FTS5)",
    )
    needs_web: bool = Field(
        ...,
        description="Should search web (Tavily/OpenRouter)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence score",
    )
    keywords_matched: Optional[List[str]] = Field(
        None,
        description="Keywords that triggered this classification",
    )

    @model_validator(mode="after")
    def validate_context_needs(self) -> "QueryClassification":
        """Validate that context needs are reasonable for query type.

        At least one context source should be true unless conversational.
        """
        if self.query_type != QueryType.CONVERSATIONAL:
            if not (self.needs_code or self.needs_vault or self.needs_web):
                # This is a warning case, not an error
                # Conversational queries legitimately need no context
                pass
        return self

    @classmethod
    def from_type(
        cls,
        query_type: QueryType,
        confidence: float = 1.0,
        keywords_matched: Optional[List[str]] = None,
    ) -> "QueryClassification":
        """Create classification from query type using default context needs.

        This factory method applies the standard mapping from query type
        to context needs.

        Args:
            query_type: The classified query type
            confidence: Classification confidence (0.0-1.0)
            keywords_matched: Optional list of matched keywords

        Returns:
            QueryClassification with appropriate context needs set
        """
        needs = QUERY_TYPE_CONTEXT_NEEDS[query_type]
        return cls(
            query_type=query_type,
            needs_code=needs["needs_code"],
            needs_vault=needs["needs_vault"],
            needs_web=needs["needs_web"],
            confidence=confidence,
            keywords_matched=keywords_matched,
        )

    def any_context_needed(self) -> bool:
        """Check if any context source is needed."""
        return self.needs_code or self.needs_vault or self.needs_web

    def get_prompt_segment_id(self) -> str:
        """Get the prompt segment ID for this query type.

        Returns the segment ID to include in the composed prompt.
        """
        segment_map = {
            QueryType.CODE: "code",
            QueryType.DOCUMENTATION: "docs",
            QueryType.RESEARCH: "research",
            QueryType.CONVERSATIONAL: "conversation",
            QueryType.ACTION: "docs",  # Actions use docs segment
        }
        return segment_map[self.query_type]


__all__ = [
    "QueryType",
    "QueryClassification",
    "QUERY_TYPE_CONTEXT_NEEDS",
]
