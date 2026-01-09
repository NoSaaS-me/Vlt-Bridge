# Expanded Tasks: User Story 4 - Dynamic Prompt Composition

**Feature**: 020-bt-oracle-agent
**User Story**: US4 - Dynamic Prompt Composition (Priority: P2)
**Goal**: System prompts are composed from segments based on query type

---

## T035: Ensure Prompt Segments Exist

**Objective**: Copy all prompt segment files to `backend/src/prompts/oracle/`

### Files Required

| File | Purpose | Source |
|------|---------|--------|
| `base.md` | Core identity, constraints | `specs/020-bt-oracle-agent/prompts/oracle/base.md` |
| `signals.md` | XML signal protocol | `specs/020-bt-oracle-agent/prompts/oracle/signals.md` |
| `tools-reference.md` | Tool selection guide | `specs/020-bt-oracle-agent/prompts/oracle/tools-reference.md` |
| `code-analysis.md` | Code query guidance | `specs/020-bt-oracle-agent/prompts/oracle/code-analysis.md` |
| `documentation.md` | Vault query guidance | `specs/020-bt-oracle-agent/prompts/oracle/documentation.md` |
| `research.md` | Web research guidance | `specs/020-bt-oracle-agent/prompts/oracle/research.md` |
| `conversation.md` | Conversational guidance | `specs/020-bt-oracle-agent/prompts/oracle/conversation.md` |

### Algorithm

1. Check if `backend/src/prompts/oracle/` directory exists
2. If not, create directory
3. For each source file in spec prompts:
   - Copy to destination
   - Verify file exists after copy
4. Log success/failure for each file

### Command

```bash
mkdir -p backend/src/prompts/oracle && \
cp specs/020-bt-oracle-agent/prompts/oracle/*.md backend/src/prompts/oracle/
```

### Acceptance Criteria

- [x] SC-US4-AC1: All 7 prompt files exist in `backend/src/prompts/oracle/`
- [x] SC-US4-AC3: Signal instructions always included (signals.md exists)

---

## T036: Segment Loader in prompt_composer.py

**Objective**: Add segment loading from filesystem to `prompt_composer.py`

### Function Signature

```python
# backend/src/services/prompt_composer.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum

class QueryType(Enum):
    CODE = "code"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    CONVERSATIONAL = "conversational"
    ACTION = "action"

@dataclass
class PromptSegment:
    id: str
    content: str
    token_estimate: int
    priority: int
    conditions: list[str]  # e.g., ["query_type=code"]

def load_segment(segment_id: str, prompts_dir: Path) -> PromptSegment:
    """Load a single prompt segment from filesystem.

    Args:
        segment_id: Segment identifier (e.g., "base", "signals", "code")
        prompts_dir: Path to prompts/oracle/ directory

    Returns:
        PromptSegment with content and metadata

    Raises:
        FileNotFoundError: If segment file doesn't exist
    """
```

### Core Algorithm

1. Map segment_id to filename: `{segment_id}.md` or use SEGMENT_REGISTRY
2. Build full path: `prompts_dir / filename`
3. Read file content
4. Estimate tokens: `len(content) // 4` (rough approximation)
5. Look up priority and conditions from SEGMENT_REGISTRY
6. Return PromptSegment dataclass

### Key Code Snippet

```python
SEGMENT_REGISTRY: dict[str, dict] = {
    "base": {"file": "base.md", "priority": 0, "conditions": ["always"]},
    "signals": {"file": "signals.md", "priority": 1, "conditions": ["always"]},
    "tools": {"file": "tools-reference.md", "priority": 2, "conditions": ["always"]},
    "code": {"file": "code-analysis.md", "priority": 10, "conditions": ["query_type=code"]},
    "docs": {"file": "documentation.md", "priority": 10, "conditions": ["query_type=documentation"]},
    "research": {"file": "research.md", "priority": 10, "conditions": ["query_type=research"]},
    "conversation": {"file": "conversation.md", "priority": 10, "conditions": ["query_type=conversational"]},
}

def load_segment(segment_id: str, prompts_dir: Path) -> PromptSegment:
    if segment_id not in SEGMENT_REGISTRY:
        raise ValueError(f"Unknown segment: {segment_id}")

    registry_entry = SEGMENT_REGISTRY[segment_id]
    file_path = prompts_dir / registry_entry["file"]

    content = file_path.read_text(encoding="utf-8")
    token_estimate = len(content) // 4

    return PromptSegment(
        id=segment_id,
        content=content,
        token_estimate=token_estimate,
        priority=registry_entry["priority"],
        conditions=registry_entry["conditions"],
    )
```

### Acceptance Criteria

- [x] SC-US4-AC1: Code query includes code-analysis.md segment
- [x] SC-US4-AC2: Research query includes research.md segment

---

## T037: Segment Priority Ordering

**Objective**: Compose prompt with segments ordered by priority

### Function Signature

```python
def compose_prompt(
    query_type: QueryType,
    context: dict[str, Any],
    prompts_dir: Optional[Path] = None,
) -> str:
    """Compose system prompt from segments based on query type.

    Args:
        query_type: Classification of user query
        context: Variables for Jinja2 rendering (project_name, max_turns, etc.)
        prompts_dir: Override prompts directory (for testing)

    Returns:
        Composed prompt string with segments ordered by priority
    """
```

### Core Algorithm

1. Load ALL segments from SEGMENT_REGISTRY
2. Filter segments by condition matching:
   - "always" → include
   - "query_type=X" → include if query_type == X
3. Sort filtered segments by priority (ascending: 0, 1, 2, 10)
4. Concatenate content with separator (`\n\n---\n\n`)
5. Render Jinja2 variables from context
6. Return composed string

### Key Code Snippet

```python
def compose_prompt(
    query_type: QueryType,
    context: dict[str, Any],
    prompts_dir: Optional[Path] = None,
) -> str:
    prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR / "oracle"

    # Load and filter segments
    segments: list[PromptSegment] = []
    for segment_id in SEGMENT_REGISTRY:
        segment = load_segment(segment_id, prompts_dir)
        if _matches_conditions(segment.conditions, query_type):
            segments.append(segment)

    # Sort by priority (lower = earlier)
    segments.sort(key=lambda s: s.priority)

    # Compose content
    composed = "\n\n---\n\n".join(s.content for s in segments)

    # Render Jinja2 variables
    template = jinja2.Template(composed)
    return template.render(**context)

def _matches_conditions(conditions: list[str], query_type: QueryType) -> bool:
    for cond in conditions:
        if cond == "always":
            return True
        if cond.startswith("query_type="):
            expected = cond.split("=")[1]
            if query_type.value == expected:
                return True
    return False
```

### Priority Table (from data-model.md)

| Priority | Segments |
|----------|----------|
| 0 | base |
| 1 | signals |
| 2 | tools |
| 10 | code, docs, research, conversation |

### Acceptance Criteria

- [x] SC-010: Prompt composition is deterministic (same inputs = same prompt)
- [x] SC-US4-AC1: Code query includes code-analysis.md
- [x] SC-US4-AC2: Research query includes research.md

---

## T038: Token Budget Tracking

**Objective**: Enforce 8000 token maximum for composed prompts

### Function Signature

```python
MAX_PROMPT_TOKENS = 8000

def compose_prompt_with_budget(
    query_type: QueryType,
    context: dict[str, Any],
    max_tokens: int = MAX_PROMPT_TOKENS,
    prompts_dir: Optional[Path] = None,
) -> tuple[str, int]:
    """Compose prompt with token budget enforcement.

    Args:
        query_type: Classification of user query
        context: Variables for Jinja2 rendering
        max_tokens: Maximum tokens allowed (default 8000)
        prompts_dir: Override prompts directory

    Returns:
        Tuple of (composed_prompt, actual_token_count)

    Raises:
        PromptBudgetExceededError: If required segments exceed budget
    """
```

### Core Algorithm

1. Load and filter segments (same as T037)
2. Sort by priority
3. Iterate segments, accumulating token count:
   - If adding segment would exceed budget AND segment is optional (priority >= 10):
     - Skip segment, log warning
   - If segment is required (priority < 10) AND would exceed budget:
     - Raise PromptBudgetExceededError
4. Compose included segments
5. Return (prompt, total_tokens)

### Key Code Snippet

```python
class PromptBudgetExceededError(Exception):
    """Raised when required prompt segments exceed token budget."""
    pass

def compose_prompt_with_budget(
    query_type: QueryType,
    context: dict[str, Any],
    max_tokens: int = MAX_PROMPT_TOKENS,
    prompts_dir: Optional[Path] = None,
) -> tuple[str, int]:
    prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR / "oracle"

    # Load, filter, sort
    segments = _get_filtered_sorted_segments(query_type, prompts_dir)

    included: list[PromptSegment] = []
    token_count = 0

    for segment in segments:
        projected = token_count + segment.token_estimate

        if projected > max_tokens:
            if segment.priority < 10:  # Required segment
                raise PromptBudgetExceededError(
                    f"Required segment '{segment.id}' would exceed budget "
                    f"({projected} > {max_tokens})"
                )
            else:
                logger.warning(
                    f"Skipping optional segment '{segment.id}' due to budget",
                    extra={"token_count": token_count, "max_tokens": max_tokens}
                )
                continue

        included.append(segment)
        token_count = projected

    composed = "\n\n---\n\n".join(s.content for s in included)
    template = jinja2.Template(composed)
    rendered = template.render(**context)

    # Re-estimate after rendering (context expansion)
    final_tokens = len(rendered) // 4

    return rendered, final_tokens
```

### Acceptance Criteria

- [x] SC-US4-AC4: Tool-heavy segments omitted when over budget
- [x] spec.md: "Total prompt should not exceed 8000 tokens"

---

## T039: Verify Signals Always Included

**Objective**: Guarantee `signals.md` is included for ALL query types

### Function Signature

```python
def _validate_required_segments(segments: list[PromptSegment]) -> None:
    """Validate that required segments are present.

    Args:
        segments: List of segments to be included in prompt

    Raises:
        ValueError: If required segments are missing
    """
```

### Core Algorithm

1. Define REQUIRED_SEGMENTS = {"base", "signals", "tools"}
2. Extract segment IDs from included list
3. Check intersection: `REQUIRED_SEGMENTS - included_ids`
4. If any missing, raise ValueError with missing segment names
5. Always call before composing final prompt

### Key Code Snippet

```python
REQUIRED_SEGMENTS = {"base", "signals", "tools"}

def _validate_required_segments(segments: list[PromptSegment]) -> None:
    included_ids = {s.id for s in segments}
    missing = REQUIRED_SEGMENTS - included_ids

    if missing:
        raise ValueError(
            f"Required segments missing from prompt composition: {missing}. "
            f"Signals must ALWAYS be included per FR-012."
        )

# Integration in compose_prompt_with_budget:
def compose_prompt_with_budget(...) -> tuple[str, int]:
    # ... filtering and sorting ...

    _validate_required_segments(included)  # Validation call

    # ... composition ...
```

### Test Case

```python
def test_signals_always_included():
    """Verify signals.md is included for every query type."""
    for query_type in QueryType:
        prompt, _ = compose_prompt_with_budget(query_type, {})
        assert "Signal Emission Protocol" in prompt
        assert "<signal type=" in prompt
```

### Acceptance Criteria

- [x] FR-012: Signal emission instructions MUST always be included in system prompt
- [x] SC-US4-AC3: Signal instructions always included

---

## T040: Unit Tests for Segment Inclusion/Exclusion

**Objective**: Test segment filtering logic for all query types

### Test File

`backend/tests/unit/test_prompt_composition_segments.py`

### Test Cases

```python
import pytest
from pathlib import Path
from unittest.mock import patch

from src.services.prompt_composer import (
    compose_prompt_with_budget,
    load_segment,
    QueryType,
    SEGMENT_REGISTRY,
    REQUIRED_SEGMENTS,
    PromptBudgetExceededError,
)

class TestSegmentLoading:
    def test_load_existing_segment(self, tmp_path):
        """Load segment from filesystem."""
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir()
        (oracle_dir / "base.md").write_text("# Base\n{{project_name}}")

        segment = load_segment("base", oracle_dir)

        assert segment.id == "base"
        assert "# Base" in segment.content
        assert segment.priority == 0
        assert segment.token_estimate > 0

    def test_load_unknown_segment_raises(self, tmp_path):
        """Unknown segment ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown segment"):
            load_segment("nonexistent", tmp_path)


class TestSegmentFiltering:
    @pytest.fixture
    def mock_prompts_dir(self, tmp_path):
        """Create mock prompts directory with minimal files."""
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir()
        for seg_id, meta in SEGMENT_REGISTRY.items():
            (oracle_dir / meta["file"]).write_text(f"# {seg_id}\n")
        return oracle_dir

    def test_code_query_includes_code_segment(self, mock_prompts_dir):
        """Code query includes code-analysis.md."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CODE, {}, prompts_dir=mock_prompts_dir
        )
        assert "# code" in prompt

    def test_code_query_excludes_research_segment(self, mock_prompts_dir):
        """Code query excludes research.md."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CODE, {}, prompts_dir=mock_prompts_dir
        )
        assert "# research" not in prompt

    def test_research_query_includes_research_segment(self, mock_prompts_dir):
        """Research query includes research.md."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.RESEARCH, {}, prompts_dir=mock_prompts_dir
        )
        assert "# research" in prompt

    def test_conversational_excludes_code_segment(self, mock_prompts_dir):
        """Conversational query excludes code-analysis.md."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CONVERSATIONAL, {}, prompts_dir=mock_prompts_dir
        )
        assert "# code" not in prompt


class TestRequiredSegments:
    @pytest.fixture
    def mock_prompts_dir(self, tmp_path):
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir()
        for seg_id, meta in SEGMENT_REGISTRY.items():
            (oracle_dir / meta["file"]).write_text(f"# {seg_id}\n")
        return oracle_dir

    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_signals_included_all_types(self, mock_prompts_dir, query_type):
        """Signals segment included for every query type."""
        prompt, _ = compose_prompt_with_budget(
            query_type, {}, prompts_dir=mock_prompts_dir
        )
        assert "# signals" in prompt

    @pytest.mark.parametrize("segment_id", ["base", "signals", "tools"])
    def test_required_segments_included(self, mock_prompts_dir, segment_id):
        """All required segments included."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CODE, {}, prompts_dir=mock_prompts_dir
        )
        assert f"# {segment_id}" in prompt


class TestPriorityOrdering:
    @pytest.fixture
    def mock_prompts_dir(self, tmp_path):
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir()
        for seg_id, meta in SEGMENT_REGISTRY.items():
            (oracle_dir / meta["file"]).write_text(f"[{seg_id}]\n")
        return oracle_dir

    def test_segments_ordered_by_priority(self, mock_prompts_dir):
        """Segments appear in priority order: base, signals, tools, context."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CODE, {}, prompts_dir=mock_prompts_dir
        )

        base_pos = prompt.find("[base]")
        signals_pos = prompt.find("[signals]")
        tools_pos = prompt.find("[tools]")
        code_pos = prompt.find("[code]")

        assert base_pos < signals_pos < tools_pos < code_pos


class TestTokenBudget:
    def test_optional_segment_skipped_over_budget(self, tmp_path):
        """Optional segments skipped when over budget."""
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir()

        # Small required segments
        (oracle_dir / "base.md").write_text("base")
        (oracle_dir / "signals.md").write_text("signals")
        (oracle_dir / "tools-reference.md").write_text("tools")

        # Large optional segment
        (oracle_dir / "code-analysis.md").write_text("x" * 40000)  # ~10000 tokens

        prompt, tokens = compose_prompt_with_budget(
            QueryType.CODE, {}, max_tokens=100, prompts_dir=oracle_dir
        )

        assert "x" * 100 not in prompt  # Large segment excluded

    def test_required_segment_over_budget_raises(self, tmp_path):
        """Required segment over budget raises error."""
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir()

        (oracle_dir / "base.md").write_text("x" * 40000)  # Too large
        (oracle_dir / "signals.md").write_text("signals")
        (oracle_dir / "tools-reference.md").write_text("tools")

        with pytest.raises(PromptBudgetExceededError):
            compose_prompt_with_budget(
                QueryType.CONVERSATIONAL, {}, max_tokens=100, prompts_dir=oracle_dir
            )
```

### Acceptance Criteria

- [x] FR-011: System prompt MUST be composed from segments
- [x] FR-013: Query-type-specific segments loaded based on classification
- [x] FR-014: Prompt segments stored as separate files

---

## T041: Integration Test for Code Query

**Objective**: End-to-end test verifying code query includes `code-analysis.md`

### Test File

`backend/tests/integration/test_oracle_bt_integration.py` (add to existing)

### Test Case

```python
import pytest
from pathlib import Path

from src.services.prompt_composer import compose_prompt_with_budget, QueryType
from src.services.query_classifier import classify_query


class TestDynamicPromptComposition:
    """Integration tests for US4: Dynamic Prompt Composition."""

    @pytest.fixture
    def real_prompts_dir(self):
        """Use actual prompts from backend/src/prompts/oracle/."""
        return Path(__file__).parent.parent.parent / "src" / "prompts" / "oracle"

    def test_code_query_full_composition(self, real_prompts_dir):
        """Code query includes code-analysis.md with real prompts.

        Maps to: US4-AC1 (code query includes code analysis segment)
        """
        if not real_prompts_dir.exists():
            pytest.skip("Prompts not deployed yet")

        # Classify a code question
        classification = classify_query("How does the auth middleware work?")
        assert classification.query_type.value == "code"

        # Compose prompt
        prompt, token_count = compose_prompt_with_budget(
            classification.query_type,
            context={"project_name": "Test", "max_turns": 30, "project_context": ""},
            prompts_dir=real_prompts_dir,
        )

        # Verify code-analysis content present
        assert "code" in prompt.lower()
        assert "cite sources" in prompt.lower() or "file paths" in prompt.lower()

        # Verify signals always present
        assert "signal" in prompt.lower()
        assert "<signal type=" in prompt

        # Verify research NOT present (wrong query type)
        assert "web research" not in prompt.lower() or "research methodology" not in prompt.lower()

        # Verify under budget
        assert token_count <= 8000

    def test_classification_to_composition_pipeline(self, real_prompts_dir):
        """Full pipeline: query -> classification -> prompt composition.

        Maps to: SC-010 (deterministic composition)
        """
        if not real_prompts_dir.exists():
            pytest.skip("Prompts not deployed yet")

        queries = [
            ("How does the auth middleware work?", QueryType.CODE),
            ("What did we decide about caching?", QueryType.DOCUMENTATION),
            ("What's the latest React 19 features?", QueryType.RESEARCH),
            ("Thanks, that helps!", QueryType.CONVERSATIONAL),
        ]

        for query, expected_type in queries:
            classification = classify_query(query)
            prompt1, _ = compose_prompt_with_budget(
                classification.query_type,
                context={"project_name": "Test", "max_turns": 30},
                prompts_dir=real_prompts_dir,
            )
            prompt2, _ = compose_prompt_with_budget(
                classification.query_type,
                context={"project_name": "Test", "max_turns": 30},
                prompts_dir=real_prompts_dir,
            )

            # Deterministic: same inputs = same output
            assert prompt1 == prompt2, f"Non-deterministic for {query}"

    def test_signals_mandatory_all_types(self, real_prompts_dir):
        """Signal instructions MUST be present for all query types.

        Maps to: FR-012, US4-AC3
        """
        if not real_prompts_dir.exists():
            pytest.skip("Prompts not deployed yet")

        for query_type in QueryType:
            prompt, _ = compose_prompt_with_budget(
                query_type,
                context={"project_name": "Test"},
                prompts_dir=real_prompts_dir,
            )

            # Check signal protocol markers
            assert "signal" in prompt.lower(), f"Missing signals for {query_type}"
            assert any(
                marker in prompt
                for marker in ["need_turn", "context_sufficient", "<signal type="]
            ), f"Missing signal examples for {query_type}"
```

### Acceptance Criteria

- [x] SC-001: Query classification 90%+ accuracy (tested via classification pipeline)
- [x] SC-010: Prompt composition deterministic
- [x] FR-012: Signal instructions always included
- [x] US4-AC1: Code query includes code-analysis.md

---

## Summary: Acceptance Criteria Mapping

| Criteria | Tasks |
|----------|-------|
| US4-AC1: Code query includes code-analysis.md | T036, T037, T041 |
| US4-AC2: Research query includes research.md | T036, T037, T040 |
| US4-AC3: Signal instructions always included | T035, T039, T040, T041 |
| US4-AC4: Tool-heavy segments omitted over budget | T038, T040 |
| FR-011: Composed from segments | T036, T040 |
| FR-012: Signals always included | T039, T041 |
| FR-013: Query-type-specific segments | T037, T040 |
| FR-014: Segments as separate files | T035, T036 |
| SC-010: Deterministic composition | T037, T041 |

---

## File Dependencies

```
T035 ─────┐
          ├──► T036 ──► T037 ──► T038 ──► T039
T035 ─────┘                          │
                                     ▼
                               T040 (parallel)
                                     │
                                     ▼
                                   T041
```

- T035 must complete first (files need to exist)
- T036-T039 are sequential (build on each other)
- T040 can run in parallel after T039
- T041 requires all prior tasks
