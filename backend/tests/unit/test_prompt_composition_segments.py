"""Unit tests for prompt segment composition (020-bt-oracle-agent T040).

Tests the segment filtering logic for all query types per US4 requirements.

Acceptance Criteria Coverage:
- US4-AC1: Code query includes code-analysis.md segment
- US4-AC2: Research query includes research.md segment
- US4-AC3: Signal instructions always included
- US4-AC4: Tool-heavy segments omitted when over budget

Functional Requirements Coverage:
- FR-011: System prompt MUST be composed from segments
- FR-012: Signal emission instructions MUST always be included
- FR-013: Query-type-specific segments MUST be loaded based on classification
- FR-014: Prompt segments MUST be stored as separate files
- SC-010: Prompt composition is deterministic (same inputs = same prompt)
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from backend.src.services.prompt_composer import (
    compose_prompt_with_budget,
    load_segment,
    PromptSegment,
    SEGMENT_REGISTRY,
    REQUIRED_SEGMENTS,
    PromptBudgetExceededError,
    MAX_PROMPT_TOKENS,
)
from backend.src.models.query_classification import QueryType


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_prompts_dir(tmp_path: Path) -> Path:
    """Create temporary prompts directory with test segments."""
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir(parents=True)

    # Create segment files matching SEGMENT_REGISTRY
    for seg_id, meta in SEGMENT_REGISTRY.items():
        file_path = tmp_path / meta.file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"# {seg_id}\nContent for {seg_id} segment.\n")

    return tmp_path


@pytest.fixture
def real_prompts_dir() -> Path:
    """Return the real prompts directory for integration-style tests."""
    return Path(__file__).parent.parent.parent / "src" / "prompts"


# =============================================================================
# Test: Segment Loading (T036)
# =============================================================================


class TestSegmentLoading:
    """Test segment loading from filesystem."""

    def test_load_existing_segment(self, mock_prompts_dir: Path) -> None:
        """Load segment from filesystem returns PromptSegment with content."""
        segment = load_segment("base", mock_prompts_dir)

        assert segment.id == "base"
        assert segment.priority == 0
        assert segment.token_estimate > 0
        assert segment.required is True

    def test_load_unknown_segment_raises(self, tmp_path: Path) -> None:
        """Unknown segment ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown segment"):
            load_segment("nonexistent", tmp_path)

    def test_load_segment_file_not_found(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError."""
        # Create empty oracle dir without files
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            load_segment("base", tmp_path)

    @pytest.mark.parametrize("segment_id", list(SEGMENT_REGISTRY.keys()))
    def test_load_all_registered_segments(
        self, mock_prompts_dir: Path, segment_id: str
    ) -> None:
        """All registered segments can be loaded."""
        segment = load_segment(segment_id, mock_prompts_dir)
        assert segment.id == segment_id


# =============================================================================
# Test: Segment Filtering by Query Type (T037, FR-013)
# =============================================================================


class TestSegmentFiltering:
    """Test segment selection based on query type."""

    def test_code_query_includes_code_segment(self, mock_prompts_dir: Path) -> None:
        """CODE query includes code-analysis.md (US4-AC1)."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CODE, {}, prompts_dir=mock_prompts_dir
        )
        assert "# code" in prompt

    def test_code_query_excludes_research_segment(self, mock_prompts_dir: Path) -> None:
        """CODE query excludes research.md."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CODE, {}, prompts_dir=mock_prompts_dir
        )
        assert "# research" not in prompt

    def test_code_query_excludes_conversation_segment(
        self, mock_prompts_dir: Path
    ) -> None:
        """CODE query excludes conversation.md."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CODE, {}, prompts_dir=mock_prompts_dir
        )
        assert "# conversation" not in prompt

    def test_research_query_includes_research_segment(
        self, mock_prompts_dir: Path
    ) -> None:
        """RESEARCH query includes research.md (US4-AC2)."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.RESEARCH, {}, prompts_dir=mock_prompts_dir
        )
        assert "# research" in prompt

    def test_research_query_excludes_code_segment(
        self, mock_prompts_dir: Path
    ) -> None:
        """RESEARCH query excludes code-analysis.md."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.RESEARCH, {}, prompts_dir=mock_prompts_dir
        )
        assert "# code" not in prompt

    def test_documentation_query_includes_docs_segment(
        self, mock_prompts_dir: Path
    ) -> None:
        """DOCUMENTATION query includes documentation.md."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.DOCUMENTATION, {}, prompts_dir=mock_prompts_dir
        )
        assert "# docs" in prompt

    def test_conversational_query_includes_conversation_segment(
        self, mock_prompts_dir: Path
    ) -> None:
        """CONVERSATIONAL query includes conversation.md."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CONVERSATIONAL, {}, prompts_dir=mock_prompts_dir
        )
        assert "# conversation" in prompt

    def test_conversational_query_excludes_code_segment(
        self, mock_prompts_dir: Path
    ) -> None:
        """CONVERSATIONAL query excludes code-analysis.md."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CONVERSATIONAL, {}, prompts_dir=mock_prompts_dir
        )
        assert "# code" not in prompt

    def test_action_query_includes_docs_segment(self, mock_prompts_dir: Path) -> None:
        """ACTION query includes documentation.md."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.ACTION, {}, prompts_dir=mock_prompts_dir
        )
        assert "# docs" in prompt


# =============================================================================
# Test: Required Segments Always Included (T039, FR-012)
# =============================================================================


class TestRequiredSegments:
    """Test that required segments are always included."""

    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_signals_included_all_types(
        self, mock_prompts_dir: Path, query_type: QueryType
    ) -> None:
        """Signals segment included for every query type (US4-AC3, FR-012)."""
        prompt, _ = compose_prompt_with_budget(
            query_type, {}, prompts_dir=mock_prompts_dir
        )
        assert "# signals" in prompt

    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_base_included_all_types(
        self, mock_prompts_dir: Path, query_type: QueryType
    ) -> None:
        """Base segment included for every query type."""
        prompt, _ = compose_prompt_with_budget(
            query_type, {}, prompts_dir=mock_prompts_dir
        )
        assert "# base" in prompt

    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_tools_included_all_types(
        self, mock_prompts_dir: Path, query_type: QueryType
    ) -> None:
        """Tools segment included for every query type."""
        prompt, _ = compose_prompt_with_budget(
            query_type, {}, prompts_dir=mock_prompts_dir
        )
        assert "# tools" in prompt

    @pytest.mark.parametrize("segment_id", ["base", "signals", "tools"])
    def test_required_segments_included(
        self, mock_prompts_dir: Path, segment_id: str
    ) -> None:
        """All required segments included for any query type."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CODE, {}, prompts_dir=mock_prompts_dir
        )
        assert f"# {segment_id}" in prompt

    def test_required_segments_constant_matches_registry(self) -> None:
        """REQUIRED_SEGMENTS constant matches registry required flags."""
        required_from_registry = {
            seg_id for seg_id, seg in SEGMENT_REGISTRY.items() if seg.required
        }
        assert REQUIRED_SEGMENTS == required_from_registry


# =============================================================================
# Test: Priority Ordering (T037)
# =============================================================================


class TestPriorityOrdering:
    """Test segments are ordered by priority in composed prompt."""

    @pytest.fixture
    def priority_prompts_dir(self, tmp_path: Path) -> Path:
        """Create prompts with markers for position checking."""
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir(parents=True)

        for seg_id, meta in SEGMENT_REGISTRY.items():
            file_path = tmp_path / meta.file_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f"[{seg_id}]\n")

        return tmp_path

    def test_segments_ordered_by_priority(self, priority_prompts_dir: Path) -> None:
        """Segments appear in priority order: base, signals, tools, context."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CODE, {}, prompts_dir=priority_prompts_dir
        )

        base_pos = prompt.find("[base]")
        signals_pos = prompt.find("[signals]")
        tools_pos = prompt.find("[tools]")
        code_pos = prompt.find("[code]")

        assert base_pos < signals_pos < tools_pos < code_pos

    def test_priority_0_before_priority_1(self, priority_prompts_dir: Path) -> None:
        """Base (priority 0) comes before signals (priority 1)."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.CODE, {}, prompts_dir=priority_prompts_dir
        )

        base_pos = prompt.find("[base]")
        signals_pos = prompt.find("[signals]")

        assert base_pos != -1
        assert signals_pos != -1
        assert base_pos < signals_pos

    def test_priority_2_before_priority_10(self, priority_prompts_dir: Path) -> None:
        """Tools (priority 2) comes before context segments (priority 10)."""
        prompt, _ = compose_prompt_with_budget(
            QueryType.RESEARCH, {}, prompts_dir=priority_prompts_dir
        )

        tools_pos = prompt.find("[tools]")
        research_pos = prompt.find("[research]")

        assert tools_pos != -1
        assert research_pos != -1
        assert tools_pos < research_pos


# =============================================================================
# Test: Token Budget Enforcement (T038)
# =============================================================================


class TestTokenBudget:
    """Test token budget enforcement in prompt composition."""

    def test_optional_segment_skipped_over_budget(self, tmp_path: Path) -> None:
        """Optional segments skipped when over budget (US4-AC4)."""
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir(parents=True)

        # Small required segments
        (oracle_dir / "base.md").write_text("base")
        (oracle_dir / "signals.md").write_text("signals")
        (oracle_dir / "tools-reference.md").write_text("tools")

        # Large optional segment (~10000 tokens)
        (oracle_dir / "code-analysis.md").write_text("x" * 40000)

        prompt, tokens = compose_prompt_with_budget(
            QueryType.CODE, {}, max_tokens=100, prompts_dir=tmp_path
        )

        # Large segment should be excluded
        assert "x" * 100 not in prompt
        # Required segments should be included
        assert "base" in prompt
        assert "signals" in prompt
        assert "tools" in prompt

    def test_required_segment_over_budget_raises(self, tmp_path: Path) -> None:
        """Required segment over budget raises PromptBudgetExceededError."""
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir(parents=True)

        # Oversized base segment (~10000 tokens)
        (oracle_dir / "base.md").write_text("x" * 40000)
        (oracle_dir / "signals.md").write_text("signals")
        (oracle_dir / "tools-reference.md").write_text("tools")

        with pytest.raises(PromptBudgetExceededError):
            compose_prompt_with_budget(
                QueryType.CONVERSATIONAL, {}, max_tokens=100, prompts_dir=tmp_path
            )

    def test_returns_token_count(self, mock_prompts_dir: Path) -> None:
        """compose_prompt_with_budget returns token count."""
        prompt, tokens = compose_prompt_with_budget(
            QueryType.CODE, {}, prompts_dir=mock_prompts_dir
        )

        assert tokens > 0
        assert tokens <= MAX_PROMPT_TOKENS
        # Token count should be roughly proportional to content length
        assert tokens == len(prompt) // 4

    def test_default_budget_is_8000(self) -> None:
        """Default token budget is 8000."""
        assert MAX_PROMPT_TOKENS == 8000


# =============================================================================
# Test: Deterministic Composition (SC-010)
# =============================================================================


class TestDeterministicComposition:
    """Test that prompt composition is deterministic."""

    def test_same_inputs_same_output(self, mock_prompts_dir: Path) -> None:
        """Same inputs produce identical outputs (SC-010)."""
        prompt1, tokens1 = compose_prompt_with_budget(
            QueryType.CODE, {"project_name": "Test"}, prompts_dir=mock_prompts_dir
        )
        prompt2, tokens2 = compose_prompt_with_budget(
            QueryType.CODE, {"project_name": "Test"}, prompts_dir=mock_prompts_dir
        )

        assert prompt1 == prompt2
        assert tokens1 == tokens2

    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_deterministic_for_all_query_types(
        self, mock_prompts_dir: Path, query_type: QueryType
    ) -> None:
        """Composition is deterministic for all query types."""
        prompt1, _ = compose_prompt_with_budget(
            query_type, {}, prompts_dir=mock_prompts_dir
        )
        prompt2, _ = compose_prompt_with_budget(
            query_type, {}, prompts_dir=mock_prompts_dir
        )

        assert prompt1 == prompt2


# =============================================================================
# Test: Jinja2 Context Rendering
# =============================================================================


class TestJinja2Rendering:
    """Test Jinja2 template variable rendering."""

    def test_renders_project_name(self, tmp_path: Path) -> None:
        """project_name variable is rendered in templates."""
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir(parents=True)

        (oracle_dir / "base.md").write_text("Project: {{project_name}}")
        (oracle_dir / "signals.md").write_text("signals")
        (oracle_dir / "tools-reference.md").write_text("tools")

        prompt, _ = compose_prompt_with_budget(
            QueryType.CONVERSATIONAL,
            {"project_name": "TestProject"},
            prompts_dir=tmp_path,
        )

        assert "TestProject" in prompt
        assert "{{project_name}}" not in prompt

    def test_renders_max_turns(self, tmp_path: Path) -> None:
        """max_turns variable is rendered in templates."""
        oracle_dir = tmp_path / "oracle"
        oracle_dir.mkdir(parents=True)

        (oracle_dir / "base.md").write_text("Max turns: {{max_turns}}")
        (oracle_dir / "signals.md").write_text("signals")
        (oracle_dir / "tools-reference.md").write_text("tools")

        prompt, _ = compose_prompt_with_budget(
            QueryType.CONVERSATIONAL,
            {"max_turns": 30},
            prompts_dir=tmp_path,
        )

        assert "30" in prompt


# =============================================================================
# Test: Segment Registry Configuration
# =============================================================================


class TestSegmentRegistryConfiguration:
    """Test SEGMENT_REGISTRY is correctly configured."""

    def test_registry_has_all_expected_segments(self) -> None:
        """Registry contains all expected segment IDs."""
        expected_ids = {"base", "signals", "tools", "code", "docs", "research", "conversation"}
        actual_ids = set(SEGMENT_REGISTRY.keys())

        assert expected_ids == actual_ids

    def test_base_signals_tools_have_empty_conditions(self) -> None:
        """Core segments have empty conditions (always included)."""
        assert SEGMENT_REGISTRY["base"].conditions == set()
        assert SEGMENT_REGISTRY["signals"].conditions == set()
        assert SEGMENT_REGISTRY["tools"].conditions == set()

    def test_context_segments_have_specific_conditions(self) -> None:
        """Context segments are conditioned on query type."""
        assert QueryType.CODE in SEGMENT_REGISTRY["code"].conditions
        assert QueryType.RESEARCH in SEGMENT_REGISTRY["research"].conditions
        assert QueryType.DOCUMENTATION in SEGMENT_REGISTRY["docs"].conditions
        assert QueryType.CONVERSATIONAL in SEGMENT_REGISTRY["conversation"].conditions

    def test_priorities_are_correct(self) -> None:
        """Segment priorities match spec."""
        assert SEGMENT_REGISTRY["base"].priority == 0
        assert SEGMENT_REGISTRY["signals"].priority == 1
        assert SEGMENT_REGISTRY["tools"].priority == 2
        assert SEGMENT_REGISTRY["code"].priority == 10
        assert SEGMENT_REGISTRY["docs"].priority == 10
        assert SEGMENT_REGISTRY["research"].priority == 10
        assert SEGMENT_REGISTRY["conversation"].priority == 10


# =============================================================================
# Test: Real Prompts (Integration-Style)
# =============================================================================


class TestRealPrompts:
    """Tests using actual prompt files from backend/src/prompts/oracle/."""

    def test_signals_always_included_real_prompts(self, real_prompts_dir: Path) -> None:
        """Signal instructions present in real prompts for all query types."""
        if not (real_prompts_dir / "oracle" / "signals.md").exists():
            pytest.skip("Real prompts not deployed yet")

        for query_type in QueryType:
            prompt, _ = compose_prompt_with_budget(
                query_type, {}, prompts_dir=real_prompts_dir
            )

            # Check signal protocol markers from actual signals.md
            assert "signal" in prompt.lower(), f"Missing signals for {query_type}"
            assert any(
                marker in prompt
                for marker in ["need_turn", "context_sufficient", "<signal type="]
            ), f"Missing signal examples for {query_type}"

    def test_code_query_real_prompts(self, real_prompts_dir: Path) -> None:
        """Code query includes code-analysis content from real prompts."""
        if not (real_prompts_dir / "oracle" / "code-analysis.md").exists():
            pytest.skip("Real prompts not deployed yet")

        prompt, token_count = compose_prompt_with_budget(
            QueryType.CODE,
            {"project_name": "Test", "max_turns": 30, "project_context": ""},
            prompts_dir=real_prompts_dir,
        )

        # Check code-analysis markers
        assert "code" in prompt.lower()
        # Verify signals are present
        assert "signal" in prompt.lower()
        # Verify under budget
        assert token_count <= MAX_PROMPT_TOKENS

    def test_research_query_real_prompts(self, real_prompts_dir: Path) -> None:
        """Research query includes research content from real prompts."""
        if not (real_prompts_dir / "oracle" / "research.md").exists():
            pytest.skip("Real prompts not deployed yet")

        prompt, _ = compose_prompt_with_budget(
            QueryType.RESEARCH,
            {"project_name": "Test"},
            prompts_dir=real_prompts_dir,
        )

        # Check research markers
        assert "web" in prompt.lower() or "research" in prompt.lower()
