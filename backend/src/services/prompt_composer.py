"""Prompt composer for Oracle agent.

Composes system prompts from segments based on query classification.
Implements dynamic prompt assembly with priority ordering and token budgeting.

Part of feature 020-bt-oracle-agent.

Module-level functions:
- compose_prompt_with_budget(): Token-budget-aware composition (T038)
- load_segment(): Load a single segment by ID (T036)

Key Requirements:
- FR-011: System prompt MUST be composed from segments
- FR-012: Signal emission instructions MUST always be included
- FR-013: Query-type-specific segments MUST be loaded based on classification
- FR-014: Prompt segments MUST be stored as separate files
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models.query_classification import QueryClassification, QueryType
from .prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class PromptBudgetExceededError(Exception):
    """Raised when required prompt segments exceed token budget.

    This error indicates that the core required segments (base, signals, tools)
    cannot fit within the specified token budget, which is a configuration error.
    """

    pass


# =============================================================================
# Constants
# =============================================================================

# Default token budget for composed prompts
DEFAULT_TOKEN_BUDGET = 8000

# Alias for compatibility with tasks-expanded-us4.md
MAX_PROMPT_TOKENS = DEFAULT_TOKEN_BUDGET

# Approximate tokens per character (rough estimate)
CHARS_PER_TOKEN = 4

# Prompts directory for Oracle BT prompts (Phase 1 placed them in src/prompts/)
ORACLE_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Required segments that must always be included (FR-012)
# These segments are mandatory for proper Oracle operation
REQUIRED_SEGMENTS: Set[str] = {"base", "signals", "tools"}


# =============================================================================
# Segment Registry
# =============================================================================


@dataclass
class PromptSegment:
    """Definition of a reusable prompt segment.

    Segments are loaded from files and composed into system prompts
    based on query type and priority ordering.

    Attributes:
        id: Unique segment identifier
        file_path: Path relative to prompts/oracle/
        priority: Load order (lower = first, 0-99)
        conditions: Query types that include this segment (empty = always)
        required: Whether segment must be present (error if missing)
        token_estimate: Approximate token count for budgeting
    """

    id: str
    file_path: str
    priority: int
    conditions: Set[QueryType] = field(default_factory=set)
    required: bool = True
    token_estimate: int = 500

    def should_include(self, query_type: QueryType) -> bool:
        """Check if segment should be included for query type.

        Args:
            query_type: The classified query type

        Returns:
            True if segment should be included
        """
        # Empty conditions = always include
        if not self.conditions:
            return True
        return query_type in self.conditions


# Segment registry - defines all available prompt segments
# Ordered by priority (lower = loaded first)
SEGMENT_REGISTRY: Dict[str, PromptSegment] = {
    "base": PromptSegment(
        id="base",
        file_path="oracle/base.md",
        priority=0,
        conditions=set(),  # Always included
        required=True,
        token_estimate=400,
    ),
    "signals": PromptSegment(
        id="signals",
        file_path="oracle/signals.md",
        priority=1,
        conditions=set(),  # ALWAYS included (per spec FR-012)
        required=True,
        token_estimate=800,
    ),
    "tools": PromptSegment(
        id="tools",
        file_path="oracle/tools-reference.md",
        priority=2,
        conditions=set(),  # Always included
        required=True,
        token_estimate=600,
    ),
    "code": PromptSegment(
        id="code",
        file_path="oracle/code-analysis.md",
        priority=10,
        conditions={QueryType.CODE},
        required=False,
        token_estimate=300,
    ),
    "docs": PromptSegment(
        id="docs",
        file_path="oracle/documentation.md",
        priority=10,
        conditions={QueryType.DOCUMENTATION, QueryType.ACTION},
        required=False,
        token_estimate=250,
    ),
    "research": PromptSegment(
        id="research",
        file_path="oracle/research.md",
        priority=10,
        conditions={QueryType.RESEARCH},
        required=False,
        token_estimate=250,
    ),
    "conversation": PromptSegment(
        id="conversation",
        file_path="oracle/conversation.md",
        priority=10,
        conditions={QueryType.CONVERSATIONAL},
        required=False,
        token_estimate=150,
    ),
}


# =============================================================================
# Composer Class
# =============================================================================


@dataclass
class ComposedPrompt:
    """Result of prompt composition.

    Attributes:
        content: The full composed prompt text
        segments_included: List of segment IDs that were included
        token_estimate: Estimated token count
        warnings: Any warnings during composition
    """

    content: str
    segments_included: List[str]
    token_estimate: int
    warnings: List[str] = field(default_factory=list)


class PromptComposer:
    """Composes system prompts from segments based on query classification.

    The composer:
    1. Selects segments based on query type
    2. Orders segments by priority
    3. Loads segment content from files
    4. Joins segments with separators
    5. Tracks token budget

    Usage:
        >>> composer = PromptComposer()
        >>> classification = QueryClassification.from_type(QueryType.CODE)
        >>> result = composer.compose(classification)
        >>> print(result.content)
    """

    def __init__(
        self,
        loader: Optional[PromptLoader] = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        """Initialize the prompt composer.

        Args:
            loader: PromptLoader instance (creates default if None)
            token_budget: Maximum tokens for composed prompt
        """
        # Use Oracle prompts directory if no loader provided
        self._loader = loader or PromptLoader(prompts_dir=ORACLE_PROMPTS_DIR)
        self._token_budget = token_budget
        self._cache: Dict[str, str] = {}  # Segment content cache

    def compose(
        self,
        classification: QueryClassification,
        *,
        context: Optional[Dict[str, str]] = None,
        extra_segments: Optional[List[str]] = None,
    ) -> ComposedPrompt:
        """Compose a system prompt based on query classification.

        Algorithm:
        1. Select segments that match query type
        2. Sort by priority (ascending)
        3. Load each segment's content
        4. Join with separators
        5. Track token usage

        Args:
            classification: Query classification result
            context: Optional Jinja2 context for template rendering
            extra_segments: Additional segment IDs to include

        Returns:
            ComposedPrompt with content and metadata
        """
        context = context or {}
        extra_segments = extra_segments or []
        warnings: List[str] = []

        # Step 1: Select segments
        selected = self._select_segments(classification.query_type, extra_segments)

        # Step 2: Sort by priority
        sorted_segments = sorted(selected, key=lambda s: s.priority)

        # Step 3: Load content
        parts: List[str] = []
        included: List[str] = []
        total_tokens = 0

        for segment in sorted_segments:
            # Check token budget
            if total_tokens + segment.token_estimate > self._token_budget:
                if segment.required:
                    warnings.append(
                        f"Required segment '{segment.id}' exceeds token budget"
                    )
                else:
                    warnings.append(
                        f"Skipped segment '{segment.id}' due to token budget"
                    )
                    continue

            # Load content
            try:
                content = self._load_segment(segment, context)
                parts.append(content)
                included.append(segment.id)
                total_tokens += self._estimate_tokens(content)
            except Exception as e:
                if segment.required:
                    raise ValueError(
                        f"Failed to load required segment '{segment.id}': {e}"
                    ) from e
                else:
                    warnings.append(f"Failed to load segment '{segment.id}': {e}")

        # Step 4: Join with separators
        composed = "\n\n---\n\n".join(parts)

        return ComposedPrompt(
            content=composed,
            segments_included=included,
            token_estimate=total_tokens,
            warnings=warnings,
        )

    def compose_for_type(
        self,
        query_type: QueryType,
        *,
        context: Optional[Dict[str, str]] = None,
    ) -> ComposedPrompt:
        """Convenience method to compose prompt from QueryType directly.

        Args:
            query_type: The query type
            context: Optional Jinja2 context

        Returns:
            ComposedPrompt with content and metadata
        """
        classification = QueryClassification.from_type(query_type, confidence=1.0)
        return self.compose(classification, context=context)

    def _select_segments(
        self,
        query_type: QueryType,
        extra_ids: List[str],
    ) -> List[PromptSegment]:
        """Select segments to include based on query type.

        Args:
            query_type: The classified query type
            extra_ids: Additional segment IDs to include

        Returns:
            List of segments to include
        """
        selected: List[PromptSegment] = []
        seen_ids: Set[str] = set()

        # Add matching segments from registry
        for segment in SEGMENT_REGISTRY.values():
            if segment.should_include(query_type):
                selected.append(segment)
                seen_ids.add(segment.id)

        # Add extra segments
        for seg_id in extra_ids:
            if seg_id not in seen_ids and seg_id in SEGMENT_REGISTRY:
                selected.append(SEGMENT_REGISTRY[seg_id])
                seen_ids.add(seg_id)

        return selected

    def _load_segment(
        self,
        segment: PromptSegment,
        context: Dict[str, str],
    ) -> str:
        """Load segment content from file.

        Uses caching to avoid repeated file reads.

        Args:
            segment: Segment to load
            context: Jinja2 context for rendering

        Returns:
            Rendered segment content
        """
        # Check cache (only for context-free segments)
        cache_key = segment.id if not context else None

        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        # Load from file
        content = self._loader.load(segment.file_path, context)

        # Cache if no context
        if cache_key:
            self._cache[cache_key] = content

        return content

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses simple character-based estimation.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // CHARS_PER_TOKEN

    def get_segment_info(self) -> List[Dict[str, object]]:
        """Get information about all registered segments.

        Returns:
            List of segment info dictionaries
        """
        return [
            {
                "id": seg.id,
                "file_path": seg.file_path,
                "priority": seg.priority,
                "conditions": [qt.value for qt in seg.conditions] or ["always"],
                "required": seg.required,
                "token_estimate": seg.token_estimate,
            }
            for seg in sorted(SEGMENT_REGISTRY.values(), key=lambda s: s.priority)
        ]


# =============================================================================
# Module Functions
# =============================================================================


def compose_prompt(
    classification: QueryClassification,
    *,
    context: Optional[Dict[str, str]] = None,
) -> str:
    """Compose a prompt for the given classification.

    Convenience function that creates a PromptComposer and composes.

    Args:
        classification: Query classification result
        context: Optional Jinja2 context

    Returns:
        Composed prompt string

    Example:
        >>> from src.models.query_classification import QueryClassification, QueryType
        >>> classification = QueryClassification.from_type(QueryType.CODE)
        >>> prompt = compose_prompt(classification)
        >>> print(prompt[:100])
    """
    composer = PromptComposer()
    result = composer.compose(classification, context=context)
    return result.content


def load_segment(
    segment_id: str,
    prompts_dir: Optional[Path] = None,
) -> PromptSegment:
    """Load a single prompt segment from filesystem.

    This is a module-level function for loading segments by ID.
    Used for testing and direct segment access.

    Args:
        segment_id: Segment identifier (e.g., "base", "signals", "code")
        prompts_dir: Path to prompts/ directory (defaults to ORACLE_PROMPTS_DIR)

    Returns:
        PromptSegment with content loaded and metadata from registry

    Raises:
        ValueError: If segment_id not in SEGMENT_REGISTRY
        FileNotFoundError: If segment file doesn't exist

    Example:
        >>> segment = load_segment("base")
        >>> print(segment.id, segment.priority)
        base 0
    """
    if segment_id not in SEGMENT_REGISTRY:
        raise ValueError(f"Unknown segment: {segment_id}")

    prompts_dir = prompts_dir or ORACLE_PROMPTS_DIR
    registry_entry = SEGMENT_REGISTRY[segment_id]
    file_path = prompts_dir / registry_entry.file_path

    if not file_path.exists():
        raise FileNotFoundError(f"Segment file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    token_estimate = len(content) // CHARS_PER_TOKEN

    # Return a new PromptSegment with loaded content
    return PromptSegment(
        id=segment_id,
        file_path=registry_entry.file_path,
        priority=registry_entry.priority,
        conditions=registry_entry.conditions,
        required=registry_entry.required,
        token_estimate=token_estimate,
    )


def _validate_required_segments(segments: List[PromptSegment]) -> None:
    """Validate that required segments are present.

    This validation ensures FR-012 compliance: signal emission instructions
    MUST always be included in system prompt.

    Args:
        segments: List of segments to be included in prompt

    Raises:
        ValueError: If required segments are missing (especially signals)
    """
    included_ids = {s.id for s in segments}
    missing = REQUIRED_SEGMENTS - included_ids

    if missing:
        raise ValueError(
            f"Required segments missing from prompt composition: {missing}. "
            f"Signals must ALWAYS be included per FR-012."
        )


def _matches_conditions(conditions: Set[QueryType], query_type: QueryType) -> bool:
    """Check if segment conditions match query type.

    Args:
        conditions: Set of query types that include this segment
        query_type: The current query type

    Returns:
        True if segment should be included
    """
    # Empty conditions = always include
    if not conditions:
        return True
    return query_type in conditions


def _get_filtered_sorted_segments(
    query_type: QueryType,
    prompts_dir: Path,
) -> List[Tuple[PromptSegment, str]]:
    """Get segments filtered by query type and sorted by priority.

    Args:
        query_type: The classified query type
        prompts_dir: Path to prompts directory

    Returns:
        List of (PromptSegment, content) tuples sorted by priority
    """
    segments: List[Tuple[PromptSegment, str]] = []

    for segment_id, segment in SEGMENT_REGISTRY.items():
        if _matches_conditions(segment.conditions, query_type):
            file_path = prompts_dir / segment.file_path
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                token_estimate = len(content) // CHARS_PER_TOKEN
                loaded_segment = PromptSegment(
                    id=segment_id,
                    file_path=segment.file_path,
                    priority=segment.priority,
                    conditions=segment.conditions,
                    required=segment.required,
                    token_estimate=token_estimate,
                )
                segments.append((loaded_segment, content))

    # Sort by priority (ascending: 0, 1, 2, 10, ...)
    segments.sort(key=lambda x: x[0].priority)
    return segments


def compose_prompt_with_budget(
    query_type: QueryType,
    context: Optional[Dict[str, Any]] = None,
    max_tokens: int = MAX_PROMPT_TOKENS,
    prompts_dir: Optional[Path] = None,
) -> Tuple[str, int]:
    """Compose prompt with token budget enforcement.

    This function implements T038 requirements:
    - Track total tokens during composition
    - Skip optional segments if over budget
    - Raise error if required segments exceed budget

    Args:
        query_type: Classification of user query
        context: Variables for Jinja2 rendering (project_name, max_turns, etc.)
        max_tokens: Maximum tokens allowed (default 8000)
        prompts_dir: Override prompts directory (for testing)

    Returns:
        Tuple of (composed_prompt, actual_token_count)

    Raises:
        PromptBudgetExceededError: If required segments exceed budget
        ValueError: If required segments (including signals) are missing

    Example:
        >>> prompt, tokens = compose_prompt_with_budget(QueryType.CODE, {"project_name": "MyApp"})
        >>> assert tokens <= 8000
    """
    import jinja2

    context = context or {}
    prompts_dir = prompts_dir or ORACLE_PROMPTS_DIR

    # Get filtered and sorted segments with content
    segments_with_content = _get_filtered_sorted_segments(query_type, prompts_dir)

    included: List[PromptSegment] = []
    included_content: List[str] = []
    token_count = 0

    for segment, content in segments_with_content:
        projected = token_count + segment.token_estimate

        if projected > max_tokens:
            if segment.required:
                raise PromptBudgetExceededError(
                    f"Required segment '{segment.id}' would exceed budget "
                    f"({projected} > {max_tokens})"
                )
            else:
                logger.warning(
                    f"Skipping optional segment '{segment.id}' due to budget",
                    extra={"token_count": token_count, "max_tokens": max_tokens},
                )
                continue

        included.append(segment)
        included_content.append(content)
        token_count = projected

    # Validate required segments are present (FR-012)
    _validate_required_segments(included)

    # Compose content with separator
    composed = "\n\n---\n\n".join(included_content)

    # Render Jinja2 variables
    template = jinja2.Template(composed)
    rendered = template.render(**context)

    # Re-estimate after rendering (context expansion may change size)
    final_tokens = len(rendered) // CHARS_PER_TOKEN

    return rendered, final_tokens


__all__ = [
    # Classes
    "PromptSegment",
    "ComposedPrompt",
    "PromptComposer",
    # Exceptions
    "PromptBudgetExceededError",
    # Functions
    "compose_prompt",
    "compose_prompt_with_budget",
    "load_segment",
    # Constants
    "SEGMENT_REGISTRY",
    "REQUIRED_SEGMENTS",
    "DEFAULT_TOKEN_BUDGET",
    "MAX_PROMPT_TOKENS",
    "ORACLE_PROMPTS_DIR",
]
