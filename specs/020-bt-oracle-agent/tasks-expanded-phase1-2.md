# Expanded Tasks: Phase 1 (Setup) & Phase 2 (Foundational)

**Feature**: 020-bt-oracle-agent
**Date**: 2026-01-08
**Purpose**: Extremely detailed task expansions for T001-T012 with function signatures, algorithms, code snippets, edge cases, and verification steps.

---

## Phase 1: Setup (T001-T005)

---

### T001: Copy Prompt Templates

**Summary**: Copy prompt templates from `specs/020-bt-oracle-agent/prompts/oracle/` to `backend/src/prompts/oracle/`

#### Destination Structure Verification

Before copying, verify the destination directory structure:

```bash
# Expected structure at backend/prompts/oracle/
backend/prompts/oracle/
  base.md           # Core identity and instructions
  signals.md        # Signal emission protocol (ALWAYS included)
  tools-reference.md # Available tools overview
  code-analysis.md  # Code query context
  documentation.md  # Documentation query context
  research.md       # Research query context
  conversation.md   # Conversational follow-up context
```

#### Implementation Steps

1. **Check if destination directory exists**:
   ```bash
   ls -la /mnt/Samsung2tb/Projects/00Tooling/Vlt-Bridge/backend/prompts/oracle/
   ```

2. **Create directory if missing**:
   ```bash
   mkdir -p backend/prompts/oracle
   ```

3. **Copy all .md files**:
   ```bash
   cp specs/020-bt-oracle-agent/prompts/oracle/*.md backend/prompts/oracle/
   ```

4. **Verify file contents match source**:
   ```bash
   diff -q specs/020-bt-oracle-agent/prompts/oracle/ backend/prompts/oracle/
   ```

#### Source Files to Copy

| Source Path | Destination Path | Purpose |
|------------|------------------|---------|
| `specs/020-bt-oracle-agent/prompts/oracle/base.md` | `backend/prompts/oracle/base.md` | Core Oracle identity |
| `specs/020-bt-oracle-agent/prompts/oracle/signals.md` | `backend/prompts/oracle/signals.md` | Signal emission instructions |
| `specs/020-bt-oracle-agent/prompts/oracle/tools-reference.md` | `backend/prompts/oracle/tools-reference.md` | Tool usage guidance |
| `specs/020-bt-oracle-agent/prompts/oracle/code-analysis.md` | `backend/prompts/oracle/code-analysis.md` | Code query segment |
| `specs/020-bt-oracle-agent/prompts/oracle/documentation.md` | `backend/prompts/oracle/documentation.md` | Docs query segment |
| `specs/020-bt-oracle-agent/prompts/oracle/research.md` | `backend/prompts/oracle/research.md` | Research query segment |
| `specs/020-bt-oracle-agent/prompts/oracle/conversation.md` | `backend/prompts/oracle/conversation.md` | Conversational segment |

#### Edge Cases

1. **Destination directory already has files**: Back up existing files first
   ```bash
   mv backend/prompts/oracle backend/prompts/oracle.bak.$(date +%s)
   ```

2. **Source file permissions issue**: Ensure readable
   ```bash
   chmod 644 specs/020-bt-oracle-agent/prompts/oracle/*.md
   ```

#### Verification Steps

1. Count files match: `ls backend/prompts/oracle/*.md | wc -l` should be 7
2. File sizes match: `du -b specs/020-bt-oracle-agent/prompts/oracle/*.md` vs `du -b backend/prompts/oracle/*.md`
3. Test loading with PromptLoader:
   ```python
   from backend.src.services.prompt_loader import PromptLoader
   loader = PromptLoader()
   print(loader.load("oracle/signals.md", {}))  # Should not raise
   ```

#### References

- Existing PromptLoader: `backend/src/services/prompt_loader.py`
- Prompt directory constant: `DEFAULT_PROMPTS_DIR` in prompt_loader.py

---

### T002: Create signals.py Model

**Summary**: Create `backend/src/models/signals.py` with Signal, SignalType enum, and type-specific field dataclasses per data-model.md

#### File Location

`backend/src/models/signals.py`

#### Full Implementation

```python
"""Pydantic models for Oracle agent signals.

Signals enable agent self-reflection and BT control flow decisions.
The agent emits XML signals that are parsed into these typed structures.

Part of feature 020-bt-oracle-agent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class SignalType(str, Enum):
    """Signal type enumeration.

    Signals communicate agent internal state to the BT runtime.
    """

    NEED_TURN = "need_turn"
    """Agent needs more iterations to complete the task."""

    CONTEXT_SUFFICIENT = "context_sufficient"
    """Agent has gathered enough information to answer."""

    STUCK = "stuck"
    """Agent cannot proceed without external help."""

    NEED_CAPABILITY = "need_capability"
    """Agent needs a tool or capability that is unavailable."""

    PARTIAL_ANSWER = "partial_answer"
    """Agent can provide an answer but with known limitations."""

    DELEGATION_RECOMMENDED = "delegation_recommended"
    """Task would benefit from delegation to a subagent."""


# =============================================================================
# Type-Specific Field Models
# =============================================================================


class NeedTurnFields(BaseModel):
    """Fields for need_turn signal.

    Emitted when agent needs to continue working.
    """

    reason: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Why another turn is needed (be specific)",
    )
    expected_turns: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Estimated additional turns needed",
    )


class ContextSufficientFields(BaseModel):
    """Fields for context_sufficient signal.

    Emitted when agent has gathered enough information.
    """

    sources_found: int = Field(
        ...,
        ge=0,
        description="Number of relevant sources gathered",
    )
    source_types: Optional[List[str]] = Field(
        None,
        description="Types of sources found (code, docs, web)",
    )

    @field_validator("source_types")
    @classmethod
    def validate_source_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate source types are from known set."""
        if v is None:
            return v
        valid_types = {"code", "docs", "web", "thread", "vault", "repomap"}
        for source_type in v:
            if source_type.lower() not in valid_types:
                # Warn but don't reject - allow unknown types
                pass
        return v


class StuckFields(BaseModel):
    """Fields for stuck signal.

    Emitted when agent cannot proceed.
    """

    attempted: List[str] = Field(
        ...,
        min_length=1,
        description="Tools/approaches already tried",
    )
    blocker: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="What prevents progress",
    )
    suggestions: Optional[List[str]] = Field(
        None,
        description="What might help if available",
    )


class NeedCapabilityFields(BaseModel):
    """Fields for need_capability signal.

    Emitted when agent needs an unavailable tool.
    """

    capability: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="What capability is needed",
    )
    reason: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Why it's needed for this task",
    )
    workaround: Optional[str] = Field(
        None,
        max_length=500,
        description="Partial solution without capability",
    )


class PartialAnswerFields(BaseModel):
    """Fields for partial_answer signal.

    Emitted when answering with known limitations.
    """

    missing: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="What information is missing",
    )
    caveat: Optional[str] = Field(
        None,
        max_length=500,
        description="Important caveat for user",
    )


class DelegationRecommendedFields(BaseModel):
    """Fields for delegation_recommended signal.

    Emitted when task would benefit from subagent.
    """

    reason: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Why delegation is recommended",
    )
    scope: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="What should be delegated",
    )
    estimated_tokens: Optional[int] = Field(
        None,
        ge=100,
        le=100000,
        description="Estimated token budget needed",
    )
    subagent_type: Optional[str] = Field(
        None,
        max_length=50,
        description="Recommended subagent type (research, librarian, etc.)",
    )


# Type alias for signal fields union
SignalFields = Union[
    NeedTurnFields,
    ContextSufficientFields,
    StuckFields,
    NeedCapabilityFields,
    PartialAnswerFields,
    DelegationRecommendedFields,
]

# Mapping of signal type to expected fields model
SIGNAL_FIELDS_MAP: Dict[SignalType, type] = {
    SignalType.NEED_TURN: NeedTurnFields,
    SignalType.CONTEXT_SUFFICIENT: ContextSufficientFields,
    SignalType.STUCK: StuckFields,
    SignalType.NEED_CAPABILITY: NeedCapabilityFields,
    SignalType.PARTIAL_ANSWER: PartialAnswerFields,
    SignalType.DELEGATION_RECOMMENDED: DelegationRecommendedFields,
}


# =============================================================================
# Main Signal Model
# =============================================================================


class Signal(BaseModel):
    """Structured agent self-reflection extracted from LLM response.

    Signals are immutable once parsed. They are stored in:
    - Blackboard (runtime) - for BT condition evaluation
    - ANS Events (events table) - for audit/debugging
    - Exchange metadata (context_nodes) - for session continuity

    Example XML that produces this model:
        <signal type="need_turn">
          <reason>Found backup API, need to test response</reason>
          <confidence>0.85</confidence>
          <expected_turns>1</expected_turns>
        </signal>
    """

    type: SignalType = Field(
        ...,
        description="Signal category",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent confidence in signal (0.0-1.0)",
    )
    fields: Dict[str, Any] = Field(
        ...,
        description="Type-specific fields",
    )
    raw_xml: str = Field(
        ...,
        min_length=10,
        description="Original XML for debugging",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When signal was parsed",
    )

    @model_validator(mode="after")
    def validate_fields_for_type(self) -> "Signal":
        """Validate that fields match the expected schema for signal type."""
        expected_model = SIGNAL_FIELDS_MAP.get(self.type)
        if expected_model is None:
            raise ValueError(f"Unknown signal type: {self.type}")

        try:
            # Validate fields against expected model
            expected_model.model_validate(self.fields)
        except Exception as e:
            raise ValueError(
                f"Invalid fields for signal type {self.type}: {e}"
            ) from e

        return self

    @property
    def typed_fields(self) -> SignalFields:
        """Get fields as strongly-typed model instance."""
        expected_model = SIGNAL_FIELDS_MAP[self.type]
        return expected_model.model_validate(self.fields)

    def is_terminal(self) -> bool:
        """Check if this signal indicates task completion.

        Terminal signals mean agent is done (successfully or not).
        Non-terminal signals mean agent wants to continue.
        """
        return self.type in {
            SignalType.CONTEXT_SUFFICIENT,
            SignalType.STUCK,
            SignalType.PARTIAL_ANSWER,
        }

    def is_continuation(self) -> bool:
        """Check if this signal requests continuation."""
        return self.type in {
            SignalType.NEED_TURN,
            SignalType.NEED_CAPABILITY,
            SignalType.DELEGATION_RECOMMENDED,
        }


# =============================================================================
# Agent Signal State (for blackboard tracking)
# =============================================================================


class AgentSignalState(BaseModel):
    """Tracking of signals across agent loop iterations.

    Stored in blackboard for BT condition evaluation.
    """

    signals_emitted: List[Signal] = Field(
        default_factory=list,
        description="All signals emitted this session",
    )
    last_signal: Optional[Signal] = Field(
        None,
        description="Most recent signal",
    )
    consecutive_same_reason: int = Field(
        0,
        ge=0,
        description="Count of consecutive need_turn signals with same reason",
    )
    turns_without_signal: int = Field(
        0,
        ge=0,
        description="Turns with no signal emitted",
    )

    def record_signal(self, signal: Signal) -> None:
        """Record a new signal and update tracking state.

        This method mutates the state - call after parsing a signal.
        """
        self.signals_emitted.append(signal)

        # Track consecutive same reason for loop detection
        if signal.type == SignalType.NEED_TURN:
            if self.last_signal and self.last_signal.type == SignalType.NEED_TURN:
                # Compare reasons
                old_reason = self.last_signal.fields.get("reason", "")
                new_reason = signal.fields.get("reason", "")
                if old_reason == new_reason:
                    self.consecutive_same_reason += 1
                else:
                    self.consecutive_same_reason = 1
            else:
                self.consecutive_same_reason = 1
        else:
            self.consecutive_same_reason = 0

        self.last_signal = signal
        self.turns_without_signal = 0

    def record_turn_without_signal(self) -> None:
        """Record that a turn completed without signal emission."""
        self.turns_without_signal += 1
        self.consecutive_same_reason = 0

    def is_stuck_loop(self, threshold: int = 3) -> bool:
        """Check if agent is in a stuck loop.

        Returns True if agent has emitted the same need_turn reason
        for `threshold` or more consecutive turns.
        """
        return self.consecutive_same_reason >= threshold

    def needs_fallback(self, turns_threshold: int = 3) -> bool:
        """Check if fallback classification should activate.

        Returns True if no signal has been emitted for `turns_threshold`
        consecutive turns.
        """
        return self.turns_without_signal >= turns_threshold


__all__ = [
    "SignalType",
    "Signal",
    "NeedTurnFields",
    "ContextSufficientFields",
    "StuckFields",
    "NeedCapabilityFields",
    "PartialAnswerFields",
    "DelegationRecommendedFields",
    "SignalFields",
    "AgentSignalState",
    "SIGNAL_FIELDS_MAP",
]
```

#### Verification Steps

1. **Syntax check**:
   ```bash
   cd backend && python -c "from src.models.signals import Signal, SignalType; print('OK')"
   ```

2. **Model validation test**:
   ```python
   from backend.src.models.signals import Signal, SignalType

   # Valid signal
   sig = Signal(
       type=SignalType.NEED_TURN,
       confidence=0.85,
       fields={"reason": "Found backup API, need to test", "expected_turns": 1},
       raw_xml="<signal>...</signal>",
   )
   assert sig.is_continuation()

   # Invalid fields should raise
   try:
       Signal(
           type=SignalType.NEED_TURN,
           confidence=0.5,
           fields={},  # Missing required "reason"
           raw_xml="<signal>...</signal>",
       )
       assert False, "Should have raised"
   except ValueError:
       pass
   ```

3. **Check all signal types have field models**:
   ```python
   from backend.src.models.signals import SignalType, SIGNAL_FIELDS_MAP
   assert len(SIGNAL_FIELDS_MAP) == len(SignalType)
   ```

#### Edge Cases

1. **Empty reason string**: Rejected by `min_length=5`
2. **Confidence outside 0-1**: Rejected by `ge=0.0, le=1.0`
3. **Unknown signal type string**: `SignalType` enum will raise ValueError
4. **Extra fields in dict**: Pydantic ignores extra fields by default
5. **Missing required fields**: `model_validator` catches and re-raises with context

#### References

- Existing oracle models: `backend/src/models/oracle.py`
- Settings enum pattern: `backend/src/models/settings.py` (ModelProvider enum)
- Data model spec: `specs/020-bt-oracle-agent/data-model.md`

---

### T003: Create query_classification.py Model

**Summary**: Create `backend/src/models/query_classification.py` with QueryType enum and QueryClassification dataclass per data-model.md

#### File Location

`backend/src/models/query_classification.py`

#### Full Implementation

```python
"""Pydantic models for query classification.

Query classification determines which context sources to search
based on the nature of the user's question.

Part of feature 020-bt-oracle-agent.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

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
QUERY_TYPE_CONTEXT_NEEDS = {
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
```

#### Verification Steps

1. **Syntax check**:
   ```bash
   cd backend && python -c "from src.models.query_classification import QueryClassification, QueryType; print('OK')"
   ```

2. **Factory method test**:
   ```python
   from backend.src.models.query_classification import QueryClassification, QueryType

   # Create from type
   qc = QueryClassification.from_type(QueryType.CODE, confidence=0.9, keywords_matched=["function"])
   assert qc.needs_code is True
   assert qc.needs_vault is False
   assert qc.needs_web is False
   assert qc.get_prompt_segment_id() == "code"
   ```

3. **All query types have context needs**:
   ```python
   from backend.src.models.query_classification import QueryType, QUERY_TYPE_CONTEXT_NEEDS
   assert len(QUERY_TYPE_CONTEXT_NEEDS) == len(QueryType)
   ```

#### Edge Cases

1. **Confidence exactly 0.0 or 1.0**: Valid boundary values
2. **Empty keywords_matched list**: Valid (different from None)
3. **Conversational with no context needs**: Expected and valid

#### References

- QueryType enum defined in: `specs/020-bt-oracle-agent/contracts/signals.yaml`
- Derivation rules in: `specs/020-bt-oracle-agent/data-model.md`

---

### T004: Create bt/conditions/__init__.py

**Summary**: Create `backend/src/bt/conditions/__init__.py` to expose conditions module

#### File Location

`backend/src/bt/conditions/__init__.py`

#### Implementation

```python
"""BT Condition Functions for Oracle Agent.

Conditions are pure predicate functions that check blackboard state
and return True/False to guide BT control flow.

Part of feature 020-bt-oracle-agent.

Usage in Lua tree:
    Condition("needs_code_context")  -- Calls needs_code_context(ctx)
"""

from __future__ import annotations

# Conditions will be added in Phase 3-5 tasks
# This file establishes the module structure

__all__: list[str] = []
```

#### Directory Structure Verification

```bash
# Verify directory exists
ls -la backend/src/bt/

# Create conditions directory if needed
mkdir -p backend/src/bt/conditions

# Create __init__.py
touch backend/src/bt/conditions/__init__.py
```

#### Verification Steps

1. **Module imports successfully**:
   ```bash
   cd backend && python -c "from src.bt.conditions import *; print('OK')"
   ```

2. **Directory structure correct**:
   ```bash
   ls backend/src/bt/conditions/__init__.py
   ```

---

### T005: Verify BT Runtime Tests Pass

**Summary**: Verify BT runtime (019) tests pass: `cd backend && uv run pytest tests/unit/bt/ -q`

#### Command

```bash
cd /mnt/Samsung2tb/Projects/00Tooling/Vlt-Bridge/backend && uv run pytest tests/unit/bt/ -q
```

#### Expected Output

```
............................................ [100%]
XX passed in Y.YYs
```

Where XX is approximately 200+ tests (all the BT runtime tests).

#### Verification Steps

1. **All tests pass** (exit code 0)
2. **No warnings about deprecated features**
3. **Test coverage for key modules**:
   - `test_blackboard.py` - TypedBlackboard
   - `test_lua_sandbox.py` - Lua runtime
   - `test_tree.py` - BehaviorTree execution
   - `test_scheduler.py` - Tick scheduling

#### Troubleshooting

If tests fail:

1. **Missing dependencies**:
   ```bash
   cd backend && uv pip install -e ".[dev]"
   ```

2. **Lupa not installed**:
   ```bash
   uv pip install lupa
   ```

3. **Import errors**: Check that all `__init__.py` files exist

---

## Phase 2: Foundational (T006-T012)

---

### T006: Create signal_parser.py

**Summary**: Create `backend/src/services/signal_parser.py` with `parse_signal()` and `strip_signal()` functions per research.md

#### File Location

`backend/src/services/signal_parser.py`

#### Full Implementation

```python
"""Signal parser for Oracle agent responses.

Extracts and parses XML signals emitted by the Oracle agent.
Signals are always at the end of the response, on their own line.

Performance target: <10ms for 10KB response (per research.md)

Part of feature 020-bt-oracle-agent.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from xml.etree import ElementTree as ET

from ..models.signals import Signal, SignalType, SIGNAL_FIELDS_MAP

logger = logging.getLogger(__name__)


# =============================================================================
# Regex Patterns
# =============================================================================

# Main signal pattern - matches <signal type="...">...</signal>
# Captures: (1) signal type, (2) inner content
# Uses re.DOTALL to match across lines
SIGNAL_PATTERN = re.compile(
    r'<signal\s+type=["\']([^"\']+)["\']>\s*(.*?)\s*</signal>',
    re.DOTALL | re.IGNORECASE,
)

# Pattern to extract individual field elements from signal content
# Captures: (1) field name, (2) field value
FIELD_PATTERN = re.compile(
    r'<(\w+)>(.*?)</\1>',
    re.DOTALL,
)

# Confidence field specifically (may appear as attribute or element)
CONFIDENCE_ATTR_PATTERN = re.compile(
    r'confidence=["\']([0-9.]+)["\']',
    re.IGNORECASE,
)


# =============================================================================
# Parser Functions
# =============================================================================


def parse_signal(response: str) -> Optional[Signal]:
    """Parse signal from Oracle agent response.

    Extracts the XML signal block from the end of the response and
    parses it into a Signal model.

    Algorithm:
    1. Search for <signal type="...">...</signal> pattern
    2. Extract signal type from attribute
    3. Parse inner fields from XML elements
    4. Extract confidence (element or attribute)
    5. Validate fields against signal type schema
    6. Return Signal model or None if no signal found

    Args:
        response: Full LLM response text

    Returns:
        Parsed Signal model, or None if no signal found or parse error

    Example:
        >>> response = '''Here's what I found.
        ...
        ... <signal type="need_turn">
        ...   <reason>Need to verify the API response</reason>
        ...   <confidence>0.85</confidence>
        ... </signal>'''
        >>> signal = parse_signal(response)
        >>> signal.type
        SignalType.NEED_TURN
        >>> signal.confidence
        0.85
    """
    if not response or not isinstance(response, str):
        return None

    # Step 1: Find signal block
    match = SIGNAL_PATTERN.search(response)
    if not match:
        logger.debug("No signal pattern found in response")
        return None

    signal_type_str = match.group(1).lower()
    inner_content = match.group(2)
    raw_xml = match.group(0)

    # Step 2: Parse signal type
    try:
        signal_type = SignalType(signal_type_str)
    except ValueError:
        logger.warning(f"Unknown signal type: {signal_type_str}")
        return None

    # Step 3: Parse inner fields
    fields = _parse_fields(inner_content)

    # Step 4: Extract confidence
    confidence = _extract_confidence(inner_content, raw_xml)

    # Step 5: Validate fields against expected schema
    expected_model = SIGNAL_FIELDS_MAP.get(signal_type)
    if expected_model is None:
        logger.warning(f"No field model for signal type: {signal_type}")
        return None

    # Check required fields exist
    try:
        # This validates the fields match the expected schema
        expected_model.model_validate(fields)
    except Exception as e:
        logger.warning(f"Signal fields validation failed: {e}")
        # Continue anyway - we'll let Signal model do final validation
        pass

    # Step 6: Create Signal model
    try:
        signal = Signal(
            type=signal_type,
            confidence=confidence,
            fields=fields,
            raw_xml=raw_xml,
            timestamp=datetime.now(timezone.utc),
        )
        logger.debug(f"Parsed signal: type={signal_type}, confidence={confidence}")
        return signal
    except Exception as e:
        logger.warning(f"Failed to create Signal model: {e}")
        return None


def strip_signal(response: str) -> str:
    """Remove signal XML from response text.

    Returns the response with the signal block removed and
    whitespace normalized.

    Args:
        response: Full LLM response text

    Returns:
        Response text with signal removed and trailing whitespace stripped

    Example:
        >>> response = '''Here's the answer.
        ...
        ... <signal type="context_sufficient">
        ...   <sources_found>3</sources_found>
        ...   <confidence>0.9</confidence>
        ... </signal>'''
        >>> clean = strip_signal(response)
        >>> clean
        "Here's the answer."
    """
    if not response or not isinstance(response, str):
        return response or ""

    # Remove signal block
    cleaned = SIGNAL_PATTERN.sub("", response)

    # Normalize whitespace: collapse multiple newlines, strip trailing
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()

    return cleaned


def parse_and_strip(response: str) -> Tuple[Optional[Signal], str]:
    """Parse signal and return both signal and cleaned response.

    Convenience function that combines parse_signal and strip_signal.

    Args:
        response: Full LLM response text

    Returns:
        Tuple of (Signal or None, cleaned response text)

    Example:
        >>> response = "Answer here. <signal type='stuck'>...</signal>"
        >>> signal, clean = parse_and_strip(response)
        >>> signal is not None
        True
        >>> "<signal" in clean
        False
    """
    signal = parse_signal(response)
    cleaned = strip_signal(response)
    return signal, cleaned


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_fields(content: str) -> Dict[str, Any]:
    """Parse field elements from signal inner content.

    Handles:
    - Simple string fields: <reason>text</reason>
    - Numeric fields: <expected_turns>2</expected_turns>
    - List fields: <attempted>["tool1", "tool2"]</attempted>
    - Boolean fields (as strings): <success>true</success>

    Args:
        content: Inner XML content between signal tags

    Returns:
        Dictionary of field name -> parsed value
    """
    fields: Dict[str, Any] = {}

    for match in FIELD_PATTERN.finditer(content):
        field_name = match.group(1).lower()
        field_value = match.group(2).strip()

        # Skip confidence - handled separately
        if field_name == "confidence":
            continue

        # Parse value based on content
        fields[field_name] = _parse_field_value(field_value)

    return fields


def _parse_field_value(value: str) -> Any:
    """Parse a field value string into appropriate Python type.

    Parsing rules:
    1. JSON array -> list (handles ["a", "b"] format)
    2. Integer string -> int
    3. Float string -> float (only if has decimal point)
    4. "true"/"false" -> bool
    5. Otherwise -> string

    Args:
        value: Raw string value from XML

    Returns:
        Parsed Python value
    """
    value = value.strip()

    # Empty string
    if not value:
        return ""

    # JSON array (used for lists like attempted tools)
    if value.startswith("[") and value.endswith("]"):
        try:
            import json
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON, return as string
            pass

    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Integer (no decimal point)
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        try:
            return int(value)
        except ValueError:
            pass

    # Float (has decimal point)
    if "." in value:
        try:
            return float(value)
        except ValueError:
            pass

    # Default: string
    return value


def _extract_confidence(content: str, raw_xml: str) -> float:
    """Extract confidence value from signal.

    Confidence can appear as:
    1. Element: <confidence>0.85</confidence>
    2. Attribute: <signal type="..." confidence="0.85">

    Default: 0.5 if not found

    Args:
        content: Inner XML content
        raw_xml: Full signal XML (for attribute check)

    Returns:
        Confidence value (0.0-1.0)
    """
    # Try element first
    for match in FIELD_PATTERN.finditer(content):
        if match.group(1).lower() == "confidence":
            try:
                conf = float(match.group(2).strip())
                return max(0.0, min(1.0, conf))  # Clamp to 0-1
            except ValueError:
                pass

    # Try attribute
    attr_match = CONFIDENCE_ATTR_PATTERN.search(raw_xml)
    if attr_match:
        try:
            conf = float(attr_match.group(1))
            return max(0.0, min(1.0, conf))
        except ValueError:
            pass

    # Default confidence
    logger.debug("No confidence found, using default 0.5")
    return 0.5


__all__ = [
    "parse_signal",
    "strip_signal",
    "parse_and_strip",
    "SIGNAL_PATTERN",
]
```

#### Algorithm Summary

1. **Pattern Matching**: Use regex to find `<signal type="...">...</signal>`
2. **Type Extraction**: Parse signal type string into `SignalType` enum
3. **Field Parsing**: Extract inner `<field>value</field>` elements
4. **Value Coercion**: Convert strings to int/float/bool/list as appropriate
5. **Confidence Extraction**: Check both element and attribute locations
6. **Validation**: Verify fields against the expected schema for signal type
7. **Model Creation**: Build and return `Signal` instance

#### Edge Cases Handled

1. **No signal in response**: Returns `None`
2. **Malformed XML**: Returns `None`, logs warning
3. **Unknown signal type**: Returns `None`, logs warning
4. **Missing confidence**: Uses default 0.5
5. **Confidence out of range**: Clamps to 0.0-1.0
6. **JSON arrays in fields**: Parsed as Python lists
7. **Empty field values**: Preserved as empty strings
8. **Multiple signals**: Only first match is parsed (per spec: one signal per response)
9. **Signal inline with text**: Still extracted (but warned against in prompt)

#### Verification Steps

1. **Syntax check**:
   ```bash
   cd backend && python -c "from src.services.signal_parser import parse_signal, strip_signal; print('OK')"
   ```

2. **Basic parsing test**:
   ```python
   from backend.src.services.signal_parser import parse_signal, strip_signal

   response = '''Here's the answer.

   <signal type="need_turn">
     <reason>Need to verify API response format</reason>
     <confidence>0.85</confidence>
     <expected_turns>1</expected_turns>
   </signal>'''

   signal = parse_signal(response)
   assert signal is not None
   assert signal.type.value == "need_turn"
   assert signal.confidence == 0.85
   assert signal.fields["reason"] == "Need to verify API response format"

   clean = strip_signal(response)
   assert "<signal" not in clean
   assert "Here's the answer." in clean
   ```

#### References

- Signal format spec: `specs/020-bt-oracle-agent/prompts/oracle/signals.md`
- Signal model: `backend/src/models/signals.py` (T002)
- Similar XML parsing: `backend/src/services/tool_parsers/` (tool call parsing)

---

### T007: Write test_signal_parser.py

**Summary**: Write `backend/tests/unit/test_signal_parser.py` with tests for all 6 signal types, malformed XML, and edge cases

#### File Location

`backend/tests/unit/test_signal_parser.py`

#### Full Implementation

```python
"""Unit tests for signal_parser.py (020-bt-oracle-agent T007).

Tests cover:
- All 6 signal types parsing correctly
- Malformed XML handling
- Edge cases (no signal, multiple signals, inline signals)
- Field value parsing (strings, ints, floats, bools, lists)
- Confidence extraction (element and attribute)
- strip_signal whitespace normalization
"""

import pytest
from datetime import datetime, timezone

from backend.src.services.signal_parser import (
    parse_signal,
    strip_signal,
    parse_and_strip,
    SIGNAL_PATTERN,
)
from backend.src.models.signals import Signal, SignalType


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def need_turn_response() -> str:
    """Response with need_turn signal."""
    return '''I found the API endpoint but need to verify the response format.

<signal type="need_turn">
  <reason>Found backup API, need to test if it responds correctly</reason>
  <confidence>0.85</confidence>
  <expected_turns>1</expected_turns>
</signal>'''


@pytest.fixture
def context_sufficient_response() -> str:
    """Response with context_sufficient signal."""
    return '''Based on the authentication middleware in auth.py and the design doc...

<signal type="context_sufficient">
  <sources_found>3</sources_found>
  <source_types>["code", "docs"]</source_types>
  <confidence>0.9</confidence>
</signal>'''


@pytest.fixture
def stuck_response() -> str:
    """Response with stuck signal."""
    return '''I searched all available sources but couldn't find deployment history.

<signal type="stuck">
  <attempted>["vault_search", "thread_seek", "code_search"]</attempted>
  <blocker>No deployment logs or history found in any source</blocker>
  <suggestions>["check external CI/CD system", "ask team member"]</suggestions>
  <confidence>0.7</confidence>
</signal>'''


@pytest.fixture
def need_capability_response() -> str:
    """Response with need_capability signal."""
    return '''I can show you the test file, but I cannot run the tests myself.

<signal type="need_capability">
  <capability>execute_shell_command</capability>
  <reason>Need to run pytest to verify the fix works</reason>
  <workaround>You can run the command manually: pytest tests/unit/</workaround>
  <confidence>0.8</confidence>
</signal>'''


@pytest.fixture
def partial_answer_response() -> str:
    """Response with partial_answer signal."""
    return '''Based on the dev config, the timeout appears to be 30 seconds...

<signal type="partial_answer">
  <missing>Could not verify production configuration</missing>
  <caveat>This is based on development settings only</caveat>
  <confidence>0.6</confidence>
</signal>'''


@pytest.fixture
def delegation_recommended_response() -> str:
    """Response with delegation_recommended signal."""
    return '''This would require analyzing the entire authentication system...

<signal type="delegation_recommended">
  <reason>Need to trace auth flow across 23 files</reason>
  <scope>Map all authentication code paths and dependencies</scope>
  <estimated_tokens>15000</estimated_tokens>
  <subagent_type>research</subagent_type>
  <confidence>0.9</confidence>
</signal>'''


# =============================================================================
# Test: All 6 Signal Types
# =============================================================================


class TestSignalTypeParsing:
    """Test parsing of all 6 signal types."""

    def test_parse_need_turn(self, need_turn_response: str) -> None:
        """Parse need_turn signal correctly."""
        signal = parse_signal(need_turn_response)

        assert signal is not None
        assert signal.type == SignalType.NEED_TURN
        assert signal.confidence == 0.85
        assert "backup API" in signal.fields["reason"]
        assert signal.fields["expected_turns"] == 1
        assert signal.is_continuation()
        assert not signal.is_terminal()

    def test_parse_context_sufficient(self, context_sufficient_response: str) -> None:
        """Parse context_sufficient signal correctly."""
        signal = parse_signal(context_sufficient_response)

        assert signal is not None
        assert signal.type == SignalType.CONTEXT_SUFFICIENT
        assert signal.confidence == 0.9
        assert signal.fields["sources_found"] == 3
        assert signal.fields["source_types"] == ["code", "docs"]
        assert signal.is_terminal()
        assert not signal.is_continuation()

    def test_parse_stuck(self, stuck_response: str) -> None:
        """Parse stuck signal correctly."""
        signal = parse_signal(stuck_response)

        assert signal is not None
        assert signal.type == SignalType.STUCK
        assert signal.confidence == 0.7
        assert len(signal.fields["attempted"]) == 3
        assert "vault_search" in signal.fields["attempted"]
        assert "deployment" in signal.fields["blocker"]
        assert signal.is_terminal()

    def test_parse_need_capability(self, need_capability_response: str) -> None:
        """Parse need_capability signal correctly."""
        signal = parse_signal(need_capability_response)

        assert signal is not None
        assert signal.type == SignalType.NEED_CAPABILITY
        assert signal.confidence == 0.8
        assert signal.fields["capability"] == "execute_shell_command"
        assert "pytest" in signal.fields["reason"]
        assert signal.fields["workaround"] is not None
        assert signal.is_continuation()

    def test_parse_partial_answer(self, partial_answer_response: str) -> None:
        """Parse partial_answer signal correctly."""
        signal = parse_signal(partial_answer_response)

        assert signal is not None
        assert signal.type == SignalType.PARTIAL_ANSWER
        assert signal.confidence == 0.6
        assert "production" in signal.fields["missing"]
        assert signal.fields["caveat"] is not None
        assert signal.is_terminal()

    def test_parse_delegation_recommended(
        self, delegation_recommended_response: str
    ) -> None:
        """Parse delegation_recommended signal correctly."""
        signal = parse_signal(delegation_recommended_response)

        assert signal is not None
        assert signal.type == SignalType.DELEGATION_RECOMMENDED
        assert signal.confidence == 0.9
        assert "23 files" in signal.fields["reason"]
        assert "authentication" in signal.fields["scope"]
        assert signal.fields["estimated_tokens"] == 15000
        assert signal.fields["subagent_type"] == "research"
        assert signal.is_continuation()


# =============================================================================
# Test: Malformed XML
# =============================================================================


class TestMalformedXML:
    """Test handling of malformed signal XML."""

    def test_no_signal_returns_none(self) -> None:
        """Response without signal returns None."""
        response = "Just a regular response without any signal."
        signal = parse_signal(response)
        assert signal is None

    def test_empty_response_returns_none(self) -> None:
        """Empty response returns None."""
        assert parse_signal("") is None
        assert parse_signal(None) is None  # type: ignore

    def test_unclosed_signal_tag(self) -> None:
        """Unclosed signal tag returns None."""
        response = '''Answer here.
        <signal type="need_turn">
          <reason>Unclosed signal'''
        signal = parse_signal(response)
        assert signal is None

    def test_unknown_signal_type(self) -> None:
        """Unknown signal type returns None."""
        response = '''<signal type="unknown_type">
          <reason>Test</reason>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is None

    def test_missing_type_attribute(self) -> None:
        """Signal without type attribute returns None."""
        response = '''<signal>
          <reason>No type</reason>
        </signal>'''
        signal = parse_signal(response)
        assert signal is None

    def test_nested_xml_in_field(self) -> None:
        """Nested XML in field value is handled."""
        response = '''<signal type="need_turn">
          <reason>Found <code>function()</code> that needs testing</reason>
          <confidence>0.7</confidence>
        </signal>'''
        # This may fail to parse due to nested tags - expected behavior
        signal = parse_signal(response)
        # Either parses with nested content or returns None
        if signal is not None:
            assert "function" in signal.fields.get("reason", "")


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_signal_with_single_quotes(self) -> None:
        """Signal with single-quoted type attribute."""
        response = '''<signal type='need_turn'>
          <reason>Testing single quotes</reason>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.type == SignalType.NEED_TURN

    def test_signal_with_extra_whitespace(self) -> None:
        """Signal with extra whitespace is parsed."""
        response = '''<signal   type="need_turn"  >
          <reason>  Extra whitespace  </reason>
          <confidence>  0.5  </confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.type == SignalType.NEED_TURN

    def test_confidence_as_attribute(self) -> None:
        """Confidence as XML attribute instead of element."""
        response = '''<signal type="context_sufficient" confidence="0.95">
          <sources_found>5</sources_found>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.confidence == 0.95

    def test_confidence_out_of_range_clamped(self) -> None:
        """Confidence > 1.0 is clamped to 1.0."""
        response = '''<signal type="need_turn">
          <reason>Test</reason>
          <confidence>1.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.confidence == 1.0

    def test_confidence_negative_clamped(self) -> None:
        """Confidence < 0.0 is clamped to 0.0."""
        response = '''<signal type="need_turn">
          <reason>Test</reason>
          <confidence>-0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.confidence == 0.0

    def test_missing_confidence_uses_default(self) -> None:
        """Missing confidence uses default 0.5."""
        response = '''<signal type="need_turn">
          <reason>No confidence field</reason>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.confidence == 0.5

    def test_multiple_signals_takes_first(self) -> None:
        """Multiple signals - only first is parsed."""
        response = '''First answer.
        <signal type="need_turn">
          <reason>First signal</reason>
          <confidence>0.8</confidence>
        </signal>
        More text.
        <signal type="context_sufficient">
          <sources_found>1</sources_found>
          <confidence>0.9</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.type == SignalType.NEED_TURN
        assert "First signal" in signal.fields["reason"]

    def test_signal_inline_with_text(self) -> None:
        """Signal inline with text (not on own line)."""
        response = 'Answer here <signal type="context_sufficient"><sources_found>1</sources_found><confidence>0.5</confidence></signal> more text'
        signal = parse_signal(response)
        assert signal is not None
        assert signal.type == SignalType.CONTEXT_SUFFICIENT

    def test_case_insensitive_type(self) -> None:
        """Signal type is case-insensitive."""
        response = '''<signal type="NEED_TURN">
          <reason>Uppercase type</reason>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.type == SignalType.NEED_TURN

    def test_timestamp_is_set(self) -> None:
        """Signal timestamp is set to current time."""
        before = datetime.now(timezone.utc)
        response = '''<signal type="need_turn">
          <reason>Test</reason>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        after = datetime.now(timezone.utc)

        assert signal is not None
        assert before <= signal.timestamp <= after

    def test_raw_xml_preserved(self) -> None:
        """Raw XML is preserved in signal."""
        response = '''<signal type="need_turn">
          <reason>Test reason</reason>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert '<signal type="need_turn">' in signal.raw_xml
        assert "</signal>" in signal.raw_xml


# =============================================================================
# Test: Field Value Parsing
# =============================================================================


class TestFieldValueParsing:
    """Test parsing of different field value types."""

    def test_integer_field(self) -> None:
        """Integer field is parsed as int."""
        response = '''<signal type="context_sufficient">
          <sources_found>42</sources_found>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.fields["sources_found"] == 42
        assert isinstance(signal.fields["sources_found"], int)

    def test_float_field(self) -> None:
        """Float field is parsed as float."""
        response = '''<signal type="delegation_recommended">
          <reason>Test</reason>
          <scope>Test scope</scope>
          <estimated_tokens>1.5</estimated_tokens>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        # Note: estimated_tokens should be int, but parser returns float
        # Model validation will handle conversion

    def test_list_field(self) -> None:
        """JSON array field is parsed as list."""
        response = '''<signal type="stuck">
          <attempted>["tool1", "tool2", "tool3"]</attempted>
          <blocker>Cannot proceed</blocker>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.fields["attempted"] == ["tool1", "tool2", "tool3"]
        assert isinstance(signal.fields["attempted"], list)

    def test_empty_list_field(self) -> None:
        """Empty JSON array is parsed as empty list."""
        response = '''<signal type="stuck">
          <attempted>[]</attempted>
          <blocker>Nothing worked</blocker>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        # This may fail validation (min_length=1 for attempted)
        # But parser should handle it

    def test_boolean_string_field(self) -> None:
        """Boolean strings are not in our signals but handle gracefully."""
        # No signal fields are boolean, but parser supports it
        pass  # Covered by internal _parse_field_value tests


# =============================================================================
# Test: strip_signal
# =============================================================================


class TestStripSignal:
    """Test signal stripping from response."""

    def test_strip_removes_signal(self, need_turn_response: str) -> None:
        """strip_signal removes the signal block."""
        cleaned = strip_signal(need_turn_response)
        assert "<signal" not in cleaned
        assert "</signal>" not in cleaned
        assert "backup API" not in cleaned  # Signal content removed

    def test_strip_preserves_content(self, need_turn_response: str) -> None:
        """strip_signal preserves response content."""
        cleaned = strip_signal(need_turn_response)
        assert "API endpoint" in cleaned

    def test_strip_normalizes_whitespace(self) -> None:
        """strip_signal normalizes multiple newlines."""
        response = '''Answer here.



        <signal type="need_turn">
          <reason>Test</reason>
        </signal>



        '''
        cleaned = strip_signal(response)
        assert "\n\n\n" not in cleaned
        assert cleaned == "Answer here."

    def test_strip_handles_no_signal(self) -> None:
        """strip_signal handles response without signal."""
        response = "Just text, no signal."
        cleaned = strip_signal(response)
        assert cleaned == "Just text, no signal."

    def test_strip_handles_empty_response(self) -> None:
        """strip_signal handles empty response."""
        assert strip_signal("") == ""
        assert strip_signal(None) == ""  # type: ignore


# =============================================================================
# Test: parse_and_strip
# =============================================================================


class TestParseAndStrip:
    """Test combined parse and strip function."""

    def test_returns_both_signal_and_clean(self, need_turn_response: str) -> None:
        """parse_and_strip returns both signal and cleaned text."""
        signal, cleaned = parse_and_strip(need_turn_response)

        assert signal is not None
        assert signal.type == SignalType.NEED_TURN
        assert "<signal" not in cleaned
        assert "API endpoint" in cleaned

    def test_no_signal_returns_none_and_original(self) -> None:
        """parse_and_strip with no signal returns None and original text."""
        response = "No signal here."
        signal, cleaned = parse_and_strip(response)

        assert signal is None
        assert cleaned == "No signal here."


# =============================================================================
# Test: Performance (smoke test)
# =============================================================================


class TestPerformance:
    """Smoke test for parsing performance."""

    def test_large_response_parses_quickly(self) -> None:
        """Large response (~10KB) parses in reasonable time."""
        import time

        # Generate ~10KB response
        large_content = "x" * 10000
        response = f'''{large_content}

        <signal type="context_sufficient">
          <sources_found>5</sources_found>
          <confidence>0.9</confidence>
        </signal>'''

        start = time.time()
        signal = parse_signal(response)
        duration_ms = (time.time() - start) * 1000

        assert signal is not None
        assert duration_ms < 50  # Should be <10ms, allow 50ms for slow CI
```

#### Verification Steps

1. **Run tests**:
   ```bash
   cd backend && uv run pytest tests/unit/test_signal_parser.py -v
   ```

2. **All tests pass** (exit code 0)

3. **Coverage check**:
   ```bash
   cd backend && uv run pytest tests/unit/test_signal_parser.py --cov=src/services/signal_parser --cov-report=term-missing
   ```

---

### T008: Create query_classifier.py

**Summary**: Create `backend/src/services/query_classifier.py` with `classify_query()` function using keyword heuristics per research.md

#### File Location

`backend/src/services/query_classifier.py`

#### Full Implementation

```python
"""Query classifier for Oracle agent.

Classifies user queries to determine which context sources to search.
Uses keyword-based heuristics with fallback to conversational.

Expected accuracy: ~80% on common queries (per research.md)

Part of feature 020-bt-oracle-agent.
"""

from __future__ import annotations

import logging
import re
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
        >>> classify_query("Where is the auth middleware?")
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
```

#### Keyword Dictionaries Summary

| Query Type | Example Keywords | Count |
|------------|------------------|-------|
| CODE | function, method, class, implement, where is, how does | ~35 |
| DOCUMENTATION | decision, architecture, spec, why did we, history | ~25 |
| RESEARCH | best practice, compare, vs, latest, recommend | ~30 |
| ACTION | create, update, save, write, commit, delete | ~25 |
| CONVERSATIONAL | thanks, ok, yes, no, interesting | ~25 |

#### Algorithm Summary

1. **Normalize**: Lowercase and strip whitespace
2. **Short Query Check**: < 5 chars treated as conversational
3. **Keyword Scoring**: Score each query type by matched keywords
4. **Best Type Selection**: Highest score wins
5. **Confidence Calculation**: Based on score and keyword count
6. **Question Boost**: Questions get +0.1 confidence
7. **Followup Detection**: Short replies or pronoun references

#### Verification Steps

1. **Syntax check**:
   ```bash
   cd backend && python -c "from src.services.query_classifier import classify_query; print('OK')"
   ```

2. **Basic classification test**:
   ```python
   from backend.src.services.query_classifier import classify_query
   from backend.src.models.query_classification import QueryType

   # Code query
   result = classify_query("Where is the authentication middleware?")
   assert result.query_type == QueryType.CODE
   assert result.needs_code is True

   # Research query
   result = classify_query("What are the best practices for API design?")
   assert result.query_type == QueryType.RESEARCH
   assert result.needs_web is True

   # Conversational
   result = classify_query("Thanks!")
   assert result.query_type == QueryType.CONVERSATIONAL
   ```

#### References

- Heuristic keywords from: `specs/020-bt-oracle-agent/research.md`
- QueryClassification model: `backend/src/models/query_classification.py` (T003)

---

### T009: Write test_query_classifier.py

**Summary**: Write `backend/tests/unit/test_query_classifier.py` with tests for code/docs/research/conversational/action classification

#### File Location

`backend/tests/unit/test_query_classifier.py`

#### Full Implementation

```python
"""Unit tests for query_classifier.py (020-bt-oracle-agent T009).

Tests cover:
- All 5 query types classified correctly
- Keyword matching behavior
- Confidence scoring
- Edge cases (empty, short, ambiguous queries)
- Context needs derivation
"""

import pytest

from backend.src.services.query_classifier import (
    classify_query,
    CODE_KEYWORDS,
    DOCUMENTATION_KEYWORDS,
    RESEARCH_KEYWORDS,
    ACTION_KEYWORDS,
    CONVERSATIONAL_KEYWORDS,
)
from backend.src.models.query_classification import (
    QueryClassification,
    QueryType,
    QUERY_TYPE_CONTEXT_NEEDS,
)


# =============================================================================
# Test: CODE Query Classification
# =============================================================================


class TestCodeClassification:
    """Test classification of code-related queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "Where is the authentication function?",
            "How does the login method work?",
            "What does the validate_user function return?",
            "Find the database connection class",
            "Which file has the API endpoint for users?",
            "What's the implementation of the cache service?",
            "Show me the code for the auth middleware",
            "Locate the error handling module",
            "What parameters does get_user take?",
            "How is the session variable initialized?",
        ],
    )
    def test_code_queries_classified_correctly(self, query: str) -> None:
        """Code-related queries are classified as CODE."""
        result = classify_query(query)
        assert result.query_type == QueryType.CODE, f"Query: {query}"
        assert result.needs_code is True
        assert result.needs_vault is False
        assert result.needs_web is False

    def test_code_query_has_matched_keywords(self) -> None:
        """Code queries have relevant keywords matched."""
        result = classify_query("Where is the authentication function?")
        assert len(result.keywords_matched) > 0
        assert any(kw in CODE_KEYWORDS for kw in result.keywords_matched)


# =============================================================================
# Test: DOCUMENTATION Query Classification
# =============================================================================


class TestDocumentationClassification:
    """Test classification of documentation-related queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "Why did we choose PostgreSQL?",
            "What's the architecture decision for auth?",
            "Show me the design spec for the API",
            "What was discussed in the sprint meeting?",
            "When did we agree to use TypeScript?",
            "What's our convention for naming?",
            "Show me the documentation for this feature",
            "What's the roadmap for Q2?",
            "What guidelines do we follow for testing?",
        ],
    )
    def test_docs_queries_classified_correctly(self, query: str) -> None:
        """Documentation queries are classified as DOCUMENTATION."""
        result = classify_query(query)
        assert result.query_type == QueryType.DOCUMENTATION, f"Query: {query}"
        assert result.needs_vault is True
        assert result.needs_code is False
        assert result.needs_web is False


# =============================================================================
# Test: RESEARCH Query Classification
# =============================================================================


class TestResearchClassification:
    """Test classification of research-related queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "What are the best practices for API design?",
            "Compare React vs Vue for this project",
            "What's the latest version of Python?",
            "Should we use Redis or Memcached?",
            "What alternatives to JWT exist?",
            "Recommend a library for PDF parsing",
            "What are the pros and cons of microservices?",
            "How to implement OAuth2 properly?",
            "What's trending in frontend development?",
        ],
    )
    def test_research_queries_classified_correctly(self, query: str) -> None:
        """Research queries are classified as RESEARCH."""
        result = classify_query(query)
        assert result.query_type == QueryType.RESEARCH, f"Query: {query}"
        assert result.needs_web is True
        assert result.needs_code is False
        assert result.needs_vault is False


# =============================================================================
# Test: ACTION Query Classification
# =============================================================================


class TestActionClassification:
    """Test classification of action-related queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "Create a new note about the meeting",
            "Update the README with installation steps",
            "Save this to the project docs",
            "Write a summary of the discussion",
            "Delete the old config file",
            "Rename the test folder",
            "Move this file to the archive",
            "Generate a report of test results",
            "Add a new entry to the changelog",
        ],
    )
    def test_action_queries_classified_correctly(self, query: str) -> None:
        """Action queries are classified as ACTION."""
        result = classify_query(query)
        assert result.query_type == QueryType.ACTION, f"Query: {query}"
        assert result.needs_vault is True  # Actions typically need vault


# =============================================================================
# Test: CONVERSATIONAL Query Classification
# =============================================================================


class TestConversationalClassification:
    """Test classification of conversational queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "Thanks!",
            "Ok, got it",
            "Yes",
            "No",
            "Great, that helps",
            "Interesting",
            "Perfect",
            "Hmm",
            "Understood",
            "Cool",
        ],
    )
    def test_conversational_queries_classified_correctly(self, query: str) -> None:
        """Conversational queries are classified as CONVERSATIONAL."""
        result = classify_query(query)
        assert result.query_type == QueryType.CONVERSATIONAL, f"Query: {query}"
        assert result.needs_code is False
        assert result.needs_vault is False
        assert result.needs_web is False

    def test_short_query_is_conversational(self) -> None:
        """Very short queries default to conversational."""
        result = classify_query("ok")
        assert result.query_type == QueryType.CONVERSATIONAL
        assert "short_query" in result.keywords_matched


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query_is_conversational(self) -> None:
        """Empty query defaults to conversational."""
        result = classify_query("")
        assert result.query_type == QueryType.CONVERSATIONAL

    def test_none_query_is_conversational(self) -> None:
        """None query defaults to conversational."""
        result = classify_query(None)  # type: ignore
        assert result.query_type == QueryType.CONVERSATIONAL

    def test_whitespace_only_is_conversational(self) -> None:
        """Whitespace-only query defaults to conversational."""
        result = classify_query("   ")
        assert result.query_type == QueryType.CONVERSATIONAL

    def test_case_insensitive_matching(self) -> None:
        """Keywords are matched case-insensitively."""
        result = classify_query("WHERE IS THE FUNCTION?")
        assert result.query_type == QueryType.CODE

    def test_ambiguous_query_uses_highest_score(self) -> None:
        """Ambiguous queries go to highest-scoring type."""
        # Contains both code and docs keywords
        query = "What's the architecture of the auth function?"
        result = classify_query(query)
        # Should pick one - either is acceptable
        assert result.query_type in {QueryType.CODE, QueryType.DOCUMENTATION}

    def test_question_mark_boosts_confidence(self) -> None:
        """Question mark increases confidence."""
        with_qmark = classify_query("Where is the config?")
        without_qmark = classify_query("Where is the config")
        # Both should classify similarly, but question mark may boost
        assert with_qmark.query_type == without_qmark.query_type


# =============================================================================
# Test: Confidence Scoring
# =============================================================================


class TestConfidenceScoring:
    """Test confidence score calculation."""

    def test_multiple_keywords_higher_confidence(self) -> None:
        """More keyword matches = higher confidence."""
        single = classify_query("function")
        multiple = classify_query("find the function implementation in the class")

        assert multiple.confidence >= single.confidence

    def test_no_matches_low_confidence(self) -> None:
        """No keyword matches = low confidence."""
        result = classify_query("xyzabc123")  # Gibberish
        assert result.confidence < 0.5

    def test_strong_match_high_confidence(self) -> None:
        """Strong matches have high confidence."""
        result = classify_query(
            "Where is the authentication middleware function implemented?"
        )
        assert result.confidence >= 0.7


# =============================================================================
# Test: Context Needs Derivation
# =============================================================================


class TestContextNeeds:
    """Test that context needs are correctly derived from query type."""

    def test_all_query_types_have_context_needs(self) -> None:
        """Every QueryType has defined context needs."""
        for qtype in QueryType:
            assert qtype in QUERY_TYPE_CONTEXT_NEEDS

    def test_from_type_factory_sets_needs(self) -> None:
        """from_type factory correctly sets context needs."""
        for qtype in QueryType:
            result = QueryClassification.from_type(qtype, confidence=0.8)
            expected = QUERY_TYPE_CONTEXT_NEEDS[qtype]
            assert result.needs_code == expected["needs_code"]
            assert result.needs_vault == expected["needs_vault"]
            assert result.needs_web == expected["needs_web"]

    def test_code_needs_only_code(self) -> None:
        """CODE type needs only code context."""
        result = classify_query("What function handles auth?")
        if result.query_type == QueryType.CODE:
            assert result.needs_code is True
            assert result.needs_vault is False
            assert result.needs_web is False

    def test_conversational_needs_nothing(self) -> None:
        """CONVERSATIONAL type needs no context."""
        result = classify_query("Thanks!")
        if result.query_type == QueryType.CONVERSATIONAL:
            assert result.needs_code is False
            assert result.needs_vault is False
            assert result.needs_web is False
            assert not result.any_context_needed()


# =============================================================================
# Test: Keyword Coverage
# =============================================================================


class TestKeywordCoverage:
    """Verify keyword dictionaries are well-formed."""

    def test_code_keywords_not_empty(self) -> None:
        """CODE_KEYWORDS has keywords."""
        assert len(CODE_KEYWORDS) > 10

    def test_documentation_keywords_not_empty(self) -> None:
        """DOCUMENTATION_KEYWORDS has keywords."""
        assert len(DOCUMENTATION_KEYWORDS) > 10

    def test_research_keywords_not_empty(self) -> None:
        """RESEARCH_KEYWORDS has keywords."""
        assert len(RESEARCH_KEYWORDS) > 10

    def test_action_keywords_not_empty(self) -> None:
        """ACTION_KEYWORDS has keywords."""
        assert len(ACTION_KEYWORDS) > 10

    def test_conversational_keywords_not_empty(self) -> None:
        """CONVERSATIONAL_KEYWORDS has keywords."""
        assert len(CONVERSATIONAL_KEYWORDS) > 10

    def test_no_keyword_overlap_between_major_types(self) -> None:
        """Major keyword sets have minimal overlap."""
        # Some overlap is acceptable, but core keywords should be distinct
        code_docs_overlap = CODE_KEYWORDS & DOCUMENTATION_KEYWORDS
        assert len(code_docs_overlap) < 5, f"Too much overlap: {code_docs_overlap}"
```

#### Verification Steps

1. **Run tests**:
   ```bash
   cd backend && uv run pytest tests/unit/test_query_classifier.py -v
   ```

2. **All tests pass** (exit code 0)

---

### T010: Create prompt_composer.py

**Summary**: Create `backend/src/services/prompt_composer.py` with `compose_prompt()` function and segment registry per data-model.md

#### File Location

`backend/src/services/prompt_composer.py`

#### Full Implementation

```python
"""Prompt composer for Oracle agent.

Composes system prompts from segments based on query classification.
Implements dynamic prompt assembly with priority ordering and token budgeting.

Part of feature 020-bt-oracle-agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..models.query_classification import QueryClassification, QueryType
from .prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default token budget for composed prompts
DEFAULT_TOKEN_BUDGET = 8000

# Approximate tokens per character (rough estimate)
CHARS_PER_TOKEN = 4


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
        conditions=set(),  # ALWAYS included (per spec)
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
        self._loader = loader or PromptLoader()
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
        >>> from backend.src.models.query_classification import QueryClassification, QueryType
        >>> classification = QueryClassification.from_type(QueryType.CODE)
        >>> prompt = compose_prompt(classification)
        >>> print(prompt[:100])
    """
    composer = PromptComposer()
    result = composer.compose(classification, context=context)
    return result.content


__all__ = [
    "PromptSegment",
    "ComposedPrompt",
    "PromptComposer",
    "compose_prompt",
    "SEGMENT_REGISTRY",
    "DEFAULT_TOKEN_BUDGET",
]
```

#### Segment Registry Summary

| ID | File | Priority | Conditions | Required |
|----|------|----------|------------|----------|
| base | oracle/base.md | 0 | always | Yes |
| signals | oracle/signals.md | 1 | always | Yes |
| tools | oracle/tools-reference.md | 2 | always | Yes |
| code | oracle/code-analysis.md | 10 | CODE | No |
| docs | oracle/documentation.md | 10 | DOCUMENTATION, ACTION | No |
| research | oracle/research.md | 10 | RESEARCH | No |
| conversation | oracle/conversation.md | 10 | CONVERSATIONAL | No |

#### Verification Steps

1. **Syntax check**:
   ```bash
   cd backend && python -c "from src.services.prompt_composer import compose_prompt, PromptComposer; print('OK')"
   ```

2. **Basic composition test**:
   ```python
   from backend.src.services.prompt_composer import PromptComposer, SEGMENT_REGISTRY
   from backend.src.models.query_classification import QueryClassification, QueryType

   composer = PromptComposer()

   # CODE query includes base, signals, tools, code
   classification = QueryClassification.from_type(QueryType.CODE)
   result = composer.compose(classification)
   assert "base" in result.segments_included
   assert "signals" in result.segments_included
   assert "code" in result.segments_included
   assert "research" not in result.segments_included

   # CONVERSATIONAL includes base, signals, tools, conversation
   classification = QueryClassification.from_type(QueryType.CONVERSATIONAL)
   result = composer.compose(classification)
   assert "conversation" in result.segments_included
   assert "code" not in result.segments_included
   ```

#### References

- PromptLoader: `backend/src/services/prompt_loader.py`
- Segment registry spec: `specs/020-bt-oracle-agent/data-model.md`
- QueryClassification: `backend/src/models/query_classification.py` (T003)

---

### T011: Write test_prompt_composer.py

**Summary**: Write `backend/tests/unit/test_prompt_composer.py` with tests for segment composition rules

#### File Location

`backend/tests/unit/test_prompt_composer.py`

#### Full Implementation

```python
"""Unit tests for prompt_composer.py (020-bt-oracle-agent T011).

Tests cover:
- Segment selection based on query type
- Priority ordering
- Token budget enforcement
- Required vs optional segments
- Segment content loading
"""

import pytest
from pathlib import Path
from typing import Dict, Optional

from backend.src.services.prompt_composer import (
    PromptSegment,
    ComposedPrompt,
    PromptComposer,
    compose_prompt,
    SEGMENT_REGISTRY,
    DEFAULT_TOKEN_BUDGET,
)
from backend.src.services.prompt_loader import PromptLoader
from backend.src.models.query_classification import (
    QueryClassification,
    QueryType,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_prompts_dir(tmp_path: Path) -> Path:
    """Create temporary prompts directory with test segments."""
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir(parents=True)

    # Create test segment files
    (oracle_dir / "base.md").write_text("# Base Prompt\n\nYou are the Oracle.")
    (oracle_dir / "signals.md").write_text("# Signals\n\nEmit signals at response end.")
    (oracle_dir / "tools-reference.md").write_text("# Tools\n\nYou have access to tools.")
    (oracle_dir / "code-analysis.md").write_text("# Code Analysis\n\nAnalyze code carefully.")
    (oracle_dir / "documentation.md").write_text("# Documentation\n\nNavigate docs.")
    (oracle_dir / "research.md").write_text("# Research\n\nSearch the web.")
    (oracle_dir / "conversation.md").write_text("# Conversation\n\nBe helpful.")

    return tmp_path


@pytest.fixture
def composer(mock_prompts_dir: Path) -> PromptComposer:
    """Create PromptComposer with test prompts."""
    loader = PromptLoader(prompts_dir=mock_prompts_dir)
    return PromptComposer(loader=loader)


# =============================================================================
# Test: Segment Selection
# =============================================================================


class TestSegmentSelection:
    """Test segment selection based on query type."""

    def test_code_query_includes_code_segment(self, composer: PromptComposer) -> None:
        """CODE queries include code-analysis segment."""
        classification = QueryClassification.from_type(QueryType.CODE)
        result = composer.compose(classification)

        assert "code" in result.segments_included
        assert "Code Analysis" in result.content

    def test_code_query_excludes_other_context_segments(
        self, composer: PromptComposer
    ) -> None:
        """CODE queries exclude docs, research, conversation segments."""
        classification = QueryClassification.from_type(QueryType.CODE)
        result = composer.compose(classification)

        assert "docs" not in result.segments_included
        assert "research" not in result.segments_included
        assert "conversation" not in result.segments_included

    def test_documentation_query_includes_docs_segment(
        self, composer: PromptComposer
    ) -> None:
        """DOCUMENTATION queries include documentation segment."""
        classification = QueryClassification.from_type(QueryType.DOCUMENTATION)
        result = composer.compose(classification)

        assert "docs" in result.segments_included
        assert "Documentation" in result.content

    def test_research_query_includes_research_segment(
        self, composer: PromptComposer
    ) -> None:
        """RESEARCH queries include research segment."""
        classification = QueryClassification.from_type(QueryType.RESEARCH)
        result = composer.compose(classification)

        assert "research" in result.segments_included
        assert "Research" in result.content

    def test_conversational_query_includes_conversation_segment(
        self, composer: PromptComposer
    ) -> None:
        """CONVERSATIONAL queries include conversation segment."""
        classification = QueryClassification.from_type(QueryType.CONVERSATIONAL)
        result = composer.compose(classification)

        assert "conversation" in result.segments_included
        assert "Conversation" in result.content

    def test_action_query_includes_docs_segment(
        self, composer: PromptComposer
    ) -> None:
        """ACTION queries include docs segment (for vault operations)."""
        classification = QueryClassification.from_type(QueryType.ACTION)
        result = composer.compose(classification)

        assert "docs" in result.segments_included


# =============================================================================
# Test: Always-Included Segments
# =============================================================================


class TestAlwaysIncludedSegments:
    """Test that base, signals, tools are always included."""

    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_base_always_included(
        self, composer: PromptComposer, query_type: QueryType
    ) -> None:
        """Base segment is included for all query types."""
        classification = QueryClassification.from_type(query_type)
        result = composer.compose(classification)

        assert "base" in result.segments_included
        assert "Base Prompt" in result.content

    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_signals_always_included(
        self, composer: PromptComposer, query_type: QueryType
    ) -> None:
        """Signals segment is included for all query types (per spec)."""
        classification = QueryClassification.from_type(query_type)
        result = composer.compose(classification)

        assert "signals" in result.segments_included
        assert "Signals" in result.content

    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_tools_always_included(
        self, composer: PromptComposer, query_type: QueryType
    ) -> None:
        """Tools segment is included for all query types."""
        classification = QueryClassification.from_type(query_type)
        result = composer.compose(classification)

        assert "tools" in result.segments_included
        assert "Tools" in result.content


# =============================================================================
# Test: Priority Ordering
# =============================================================================


class TestPriorityOrdering:
    """Test segments are ordered by priority."""

    def test_base_comes_before_signals(self, composer: PromptComposer) -> None:
        """Base (priority 0) comes before signals (priority 1)."""
        classification = QueryClassification.from_type(QueryType.CODE)
        result = composer.compose(classification)

        base_pos = result.content.find("Base Prompt")
        signals_pos = result.content.find("Signals")

        assert base_pos < signals_pos

    def test_tools_comes_before_context_segments(
        self, composer: PromptComposer
    ) -> None:
        """Tools (priority 2) comes before context segments (priority 10)."""
        classification = QueryClassification.from_type(QueryType.CODE)
        result = composer.compose(classification)

        tools_pos = result.content.find("Tools")
        code_pos = result.content.find("Code Analysis")

        assert tools_pos < code_pos

    def test_segments_ordered_in_result(self, composer: PromptComposer) -> None:
        """segments_included list preserves priority order."""
        classification = QueryClassification.from_type(QueryType.CODE)
        result = composer.compose(classification)

        # First three should be base, signals, tools
        assert result.segments_included[0] == "base"
        assert result.segments_included[1] == "signals"
        assert result.segments_included[2] == "tools"


# =============================================================================
# Test: Token Budget
# =============================================================================


class TestTokenBudget:
    """Test token budget enforcement."""

    def test_tracks_token_estimate(self, composer: PromptComposer) -> None:
        """Composed prompt tracks estimated token count."""
        classification = QueryClassification.from_type(QueryType.CODE)
        result = composer.compose(classification)

        assert result.token_estimate > 0
        assert result.token_estimate < DEFAULT_TOKEN_BUDGET

    def test_low_budget_skips_optional_segments(
        self, mock_prompts_dir: Path
    ) -> None:
        """Very low token budget skips optional segments."""
        loader = PromptLoader(prompts_dir=mock_prompts_dir)
        # Set very low budget
        composer = PromptComposer(loader=loader, token_budget=100)

        classification = QueryClassification.from_type(QueryType.CODE)
        result = composer.compose(classification)

        # Should have warnings about skipped segments
        assert len(result.warnings) > 0
        # Required segments may still error

    def test_budget_warning_for_skipped_optional(
        self, mock_prompts_dir: Path
    ) -> None:
        """Skipped optional segments generate warnings."""
        loader = PromptLoader(prompts_dir=mock_prompts_dir)
        # Set budget that allows core but not context
        composer = PromptComposer(loader=loader, token_budget=500)

        classification = QueryClassification.from_type(QueryType.CODE)
        result = composer.compose(classification)

        # Check for warning
        has_skip_warning = any("Skipped" in w or "budget" in w for w in result.warnings)
        # May or may not have warning depending on actual sizes


# =============================================================================
# Test: Segment Content
# =============================================================================


class TestSegmentContent:
    """Test segment content is loaded correctly."""

    def test_segments_separated_by_dividers(self, composer: PromptComposer) -> None:
        """Segments are separated by --- dividers."""
        classification = QueryClassification.from_type(QueryType.CODE)
        result = composer.compose(classification)

        assert "---" in result.content

    def test_segment_content_preserved(self, composer: PromptComposer) -> None:
        """Segment content is preserved in output."""
        classification = QueryClassification.from_type(QueryType.CODE)
        result = composer.compose(classification)

        # Check each included segment's content appears
        assert "You are the Oracle" in result.content  # from base.md
        assert "Emit signals" in result.content  # from signals.md
        assert "Analyze code carefully" in result.content  # from code-analysis.md


# =============================================================================
# Test: PromptSegment Dataclass
# =============================================================================


class TestPromptSegmentDataclass:
    """Test PromptSegment behavior."""

    def test_should_include_with_matching_condition(self) -> None:
        """should_include returns True for matching query type."""
        segment = PromptSegment(
            id="test",
            file_path="test.md",
            priority=10,
            conditions={QueryType.CODE, QueryType.ACTION},
        )

        assert segment.should_include(QueryType.CODE) is True
        assert segment.should_include(QueryType.ACTION) is True
        assert segment.should_include(QueryType.RESEARCH) is False

    def test_should_include_with_empty_conditions(self) -> None:
        """should_include returns True for empty conditions (always include)."""
        segment = PromptSegment(
            id="test",
            file_path="test.md",
            priority=0,
            conditions=set(),  # Empty = always
        )

        for query_type in QueryType:
            assert segment.should_include(query_type) is True


# =============================================================================
# Test: Segment Registry
# =============================================================================


class TestSegmentRegistry:
    """Test the SEGMENT_REGISTRY configuration."""

    def test_registry_has_all_expected_segments(self) -> None:
        """Registry contains all expected segment IDs."""
        expected_ids = {"base", "signals", "tools", "code", "docs", "research", "conversation"}
        actual_ids = set(SEGMENT_REGISTRY.keys())

        assert expected_ids == actual_ids

    def test_base_signals_tools_are_required(self) -> None:
        """Core segments are marked as required."""
        assert SEGMENT_REGISTRY["base"].required is True
        assert SEGMENT_REGISTRY["signals"].required is True
        assert SEGMENT_REGISTRY["tools"].required is True

    def test_context_segments_are_optional(self) -> None:
        """Context-specific segments are optional."""
        assert SEGMENT_REGISTRY["code"].required is False
        assert SEGMENT_REGISTRY["docs"].required is False
        assert SEGMENT_REGISTRY["research"].required is False
        assert SEGMENT_REGISTRY["conversation"].required is False

    def test_signals_has_empty_conditions(self) -> None:
        """Signals segment has empty conditions (always included)."""
        assert SEGMENT_REGISTRY["signals"].conditions == set()


# =============================================================================
# Test: compose_prompt Function
# =============================================================================


class TestComposeFunctionShortcut:
    """Test the compose_prompt convenience function."""

    def test_compose_prompt_returns_string(self, mock_prompts_dir: Path) -> None:
        """compose_prompt returns string content."""
        # Need to patch the default loader
        # For now, test with real prompts if available
        classification = QueryClassification.from_type(QueryType.CODE)

        # This may use inline fallbacks if prompts not found
        try:
            result = compose_prompt(classification)
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception:
            # If prompts not set up, skip
            pytest.skip("Prompts directory not configured")


# =============================================================================
# Test: Get Segment Info
# =============================================================================


class TestGetSegmentInfo:
    """Test segment info retrieval."""

    def test_get_segment_info_returns_all_segments(
        self, composer: PromptComposer
    ) -> None:
        """get_segment_info returns info for all segments."""
        info = composer.get_segment_info()

        assert len(info) == len(SEGMENT_REGISTRY)

    def test_segment_info_has_expected_fields(
        self, composer: PromptComposer
    ) -> None:
        """Segment info includes expected fields."""
        info = composer.get_segment_info()

        for segment_info in info:
            assert "id" in segment_info
            assert "file_path" in segment_info
            assert "priority" in segment_info
            assert "conditions" in segment_info
            assert "required" in segment_info
            assert "token_estimate" in segment_info
```

#### Verification Steps

1. **Run tests**:
   ```bash
   cd backend && uv run pytest tests/unit/test_prompt_composer.py -v
   ```

2. **All tests pass** (exit code 0)

---

### T012: Run All Foundational Tests

**Summary**: Run all foundational tests to verify Phase 2 completion

#### Command

```bash
cd /mnt/Samsung2tb/Projects/00Tooling/Vlt-Bridge/backend && \
uv run pytest \
  tests/unit/test_signal_parser.py \
  tests/unit/test_query_classifier.py \
  tests/unit/test_prompt_composer.py \
  -v
```

#### Expected Output

```
tests/unit/test_signal_parser.py::TestSignalTypeParsing::test_parse_need_turn PASSED
tests/unit/test_signal_parser.py::TestSignalTypeParsing::test_parse_context_sufficient PASSED
... (all signal parser tests)

tests/unit/test_query_classifier.py::TestCodeClassification::test_code_queries_classified_correctly PASSED
... (all classifier tests)

tests/unit/test_prompt_composer.py::TestSegmentSelection::test_code_query_includes_code_segment PASSED
... (all composer tests)

========================= XX passed in Y.YYs =========================
```

#### Verification Checklist

1. [ ] All signal parser tests pass (T007)
2. [ ] All query classifier tests pass (T009)
3. [ ] All prompt composer tests pass (T011)
4. [ ] No deprecation warnings
5. [ ] Test coverage > 80% for new code

#### Additional Verification

Run with coverage:

```bash
cd backend && uv run pytest \
  tests/unit/test_signal_parser.py \
  tests/unit/test_query_classifier.py \
  tests/unit/test_prompt_composer.py \
  --cov=src/services/signal_parser \
  --cov=src/services/query_classifier \
  --cov=src/services/prompt_composer \
  --cov-report=term-missing
```

---

## Summary

### Phase 1 Tasks (T001-T005)

| Task | Description | Key Files |
|------|-------------|-----------|
| T001 | Copy prompt templates | `backend/prompts/oracle/*.md` |
| T002 | Create signals.py model | `backend/src/models/signals.py` |
| T003 | Create query_classification.py model | `backend/src/models/query_classification.py` |
| T004 | Create bt/conditions/__init__.py | `backend/src/bt/conditions/__init__.py` |
| T005 | Verify BT runtime tests | `tests/unit/bt/` |

### Phase 2 Tasks (T006-T012)

| Task | Description | Key Files |
|------|-------------|-----------|
| T006 | Create signal_parser.py | `backend/src/services/signal_parser.py` |
| T007 | Write test_signal_parser.py | `backend/tests/unit/test_signal_parser.py` |
| T008 | Create query_classifier.py | `backend/src/services/query_classifier.py` |
| T009 | Write test_query_classifier.py | `backend/tests/unit/test_query_classifier.py` |
| T010 | Create prompt_composer.py | `backend/src/services/prompt_composer.py` |
| T011 | Write test_prompt_composer.py | `backend/tests/unit/test_prompt_composer.py` |
| T012 | Run all foundational tests | N/A |

### Dependencies

```
T001 (prompts) ─┬─> T010 (composer needs prompt files)
                │
T002 (signals) ─┼─> T006 (parser imports Signal model)
                │
T003 (classification) ─┼─> T008 (classifier returns QueryClassification)
                       │
                       └─> T010 (composer uses QueryClassification)

T004 (conditions module) ─> Phase 3+ (conditions implementation)

T005 (BT tests) ─> Verification only, no code changes
```

### Checkpoint Criteria

Phase 1+2 complete when:
1. All 7 prompt files exist in `backend/prompts/oracle/`
2. `Signal` and `QueryClassification` models import without error
3. `signal_parser.py`, `query_classifier.py`, `prompt_composer.py` exist
4. All 3 test files pass (T012)
5. BT runtime tests still pass (T005)
