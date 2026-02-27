"""Integration tests for Oracle BT context selection (US1).

Part of feature 020-bt-oracle-agent (US1 - Intelligent Context Selection).
Tasks covered: T019 from tasks-expanded-us1.md

These tests verify the full flow from query submission through
classification to tool selection, ensuring that:
- Weather queries route to web_search only
- Code queries use code search, not web
- Conversational queries skip tool calls entirely

Acceptance Scenarios:
- AS1.1: Weather query calls web_search without code/vault search
- AS1.2: Code query searches code first, not web
- AS1.3: Documentation query searches vault, not code/web
- AS1.4: Conversational query responds without tool calls
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from backend.src.bt.state.base import RunStatus
from backend.src.bt.state.blackboard import TypedBlackboard
from backend.src.bt.core.context import TickContext
from backend.src.bt.actions.query_analysis import analyze_query
from backend.src.bt.actions.oracle import reset_state, build_system_prompt


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_prompt_loader():
    """Mock prompt loader that returns simple content."""
    with patch("src.services.prompt_loader.PromptLoader") as mock:
        loader_instance = MagicMock()
        loader_instance.load.return_value = "You are Oracle, an AI assistant."
        mock.return_value = loader_instance
        yield mock


@pytest.fixture
def oracle_blackboard():
    """Create a blackboard configured for Oracle BT testing."""
    bb = TypedBlackboard(scope_name="oracle-test")
    # Initialize required fields
    bb._data["user_id"] = "test-user"
    bb._data["project_id"] = "test-project"
    bb._data["model"] = "test-model"
    bb._data["max_tokens"] = 4096
    bb._data["messages"] = []
    bb._data["tools"] = []
    bb._data["tool_calls"] = []
    return bb


@pytest.fixture
def oracle_context(oracle_blackboard):
    """Create a TickContext for Oracle testing."""
    return TickContext(blackboard=oracle_blackboard)


# =============================================================================
# Test: Weather Query Routes to Web Search Only
# =============================================================================


class TestWeatherQueryRouting:
    """Test that weather queries route to web_search only (AS1.1)."""

    def test_weather_query_classification(self, oracle_context):
        """
        GIVEN a weather query
        WHEN analyzed
        THEN classification sets needs_web=True, needs_code=False, needs_vault=False
        """
        oracle_context.blackboard._data["query"] = "What's the weather in Paris?"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        bb = oracle_context.blackboard
        classification = bb._data["query_classification"]

        # Weather = research = needs_web only
        assert classification["query_type"] == "research"
        assert classification["needs_web"] is True
        assert classification["needs_code"] is False
        assert classification["needs_vault"] is False

    def test_news_query_classification(self, oracle_context):
        """
        GIVEN a news query
        WHEN analyzed
        THEN classification sets needs_web=True
        """
        oracle_context.blackboard._data["query"] = "What's the latest news about Python?"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        bb = oracle_context.blackboard
        assert bb._data["needs_web"] is True
        assert bb._data["needs_code"] is False


# =============================================================================
# Test: Code Query Does Not Use Web Search
# =============================================================================


class TestCodeQueryRouting:
    """Test that code queries search code, not web (AS1.2)."""

    def test_code_query_classification(self, oracle_context):
        """
        GIVEN a code query
        WHEN analyzed
        THEN classification sets needs_code=True, needs_web=False
        """
        oracle_context.blackboard._data["query"] = "How does the auth middleware work?"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        bb = oracle_context.blackboard
        classification = bb._data["query_classification"]

        # Code query = needs_code only
        assert classification["query_type"] == "code"
        assert classification["needs_code"] is True
        assert classification["needs_web"] is False
        assert classification["needs_vault"] is False

    def test_function_location_query(self, oracle_context):
        """
        GIVEN a query about function location
        WHEN analyzed
        THEN classification sets needs_code=True
        """
        oracle_context.blackboard._data["query"] = "Where is the validate_token function?"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        bb = oracle_context.blackboard
        assert bb._data["needs_code"] is True
        assert bb._data["needs_web"] is False

    def test_implementation_query(self, oracle_context):
        """
        GIVEN a query about implementation
        WHEN analyzed
        THEN classification sets needs_code=True
        """
        oracle_context.blackboard._data["query"] = "How is the caching implemented?"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        bb = oracle_context.blackboard
        assert bb._data["needs_code"] is True


# =============================================================================
# Test: Documentation Query Routes to Vault
# =============================================================================


class TestDocumentationQueryRouting:
    """Test that documentation queries search vault (AS1.3)."""

    def test_decision_query_classification(self, oracle_context):
        """
        GIVEN a query about past decisions
        WHEN analyzed
        THEN classification sets needs_vault=True
        """
        oracle_context.blackboard._data["query"] = "What did we decide about caching?"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        bb = oracle_context.blackboard
        classification = bb._data["query_classification"]

        # Documentation = needs_vault only
        assert classification["query_type"] == "documentation"
        assert classification["needs_vault"] is True
        assert classification["needs_code"] is False
        assert classification["needs_web"] is False

    def test_architecture_query_classification(self, oracle_context):
        """
        GIVEN a query about architecture
        WHEN analyzed
        THEN classification sets needs_vault=True
        """
        oracle_context.blackboard._data["query"] = "Why did we choose this architecture?"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        bb = oracle_context.blackboard
        assert bb._data["needs_vault"] is True
        assert bb._data["needs_code"] is False


# =============================================================================
# Test: Conversational Query Skips Tools
# =============================================================================


class TestConversationalQueryRouting:
    """Test that conversational queries skip tool calls (AS1.4)."""

    def test_thanks_query_classification(self, oracle_context):
        """
        GIVEN a thanks message
        WHEN analyzed
        THEN classification sets all needs_* to False
        """
        oracle_context.blackboard._data["query"] = "Thanks, that helps!"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        bb = oracle_context.blackboard
        classification = bb._data["query_classification"]

        # Conversational = no context needed
        assert classification["query_type"] == "conversational"
        assert classification["needs_code"] is False
        assert classification["needs_vault"] is False
        assert classification["needs_web"] is False

    def test_ok_query_classification(self, oracle_context):
        """
        GIVEN an 'ok' message
        WHEN analyzed
        THEN classification sets query_type to conversational
        """
        oracle_context.blackboard._data["query"] = "Ok, got it!"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        bb = oracle_context.blackboard
        classification = bb._data["query_classification"]
        assert classification["query_type"] == "conversational"

    def test_affirmation_query_classification(self, oracle_context):
        """
        GIVEN an affirmation message
        WHEN analyzed
        THEN classification sets query_type to conversational
        """
        oracle_context.blackboard._data["query"] = "Yes, that's perfect!"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        bb = oracle_context.blackboard
        classification = bb._data["query_classification"]
        assert classification["query_type"] == "conversational"


# =============================================================================
# Test: Full Flow - Reset -> Analyze -> Build Prompt
# =============================================================================


class TestFullContextSelectionFlow:
    """Integration tests for the full context selection flow."""

    def test_weather_query_full_flow(self, oracle_context):
        """
        GIVEN a weather query
        WHEN processed through reset -> analyze
        THEN classification is available for downstream use
        """
        oracle_context.blackboard._data["query"] = "What's the weather in Paris?"

        # Step 1: Reset state
        reset_result = reset_state(oracle_context)
        assert reset_result == RunStatus.SUCCESS

        # Step 2: Analyze query
        analyze_result = analyze_query(oracle_context)
        assert analyze_result == RunStatus.SUCCESS

        # Verify state after full flow
        bb = oracle_context.blackboard
        assert bb._data["query_classification"] is not None
        assert bb._data["needs_web"] is True
        assert bb._data["needs_code"] is False

    def test_code_query_full_flow(self, oracle_context):
        """
        GIVEN a code query
        WHEN processed through reset -> analyze
        THEN classification is available for downstream use
        """
        oracle_context.blackboard._data["query"] = "How does the auth middleware work?"

        # Step 1: Reset state
        reset_result = reset_state(oracle_context)
        assert reset_result == RunStatus.SUCCESS

        # Step 2: Analyze query
        analyze_result = analyze_query(oracle_context)
        assert analyze_result == RunStatus.SUCCESS

        # Verify state after full flow
        bb = oracle_context.blackboard
        assert bb._data["query_classification"] is not None
        assert bb._data["needs_code"] is True
        assert bb._data["needs_web"] is False

    def test_conversational_query_full_flow(self, oracle_context):
        """
        GIVEN a conversational query
        WHEN processed through reset -> analyze
        THEN no context sources are needed
        """
        oracle_context.blackboard._data["query"] = "Thanks!"

        # Step 1: Reset state
        reset_result = reset_state(oracle_context)
        assert reset_result == RunStatus.SUCCESS

        # Step 2: Analyze query
        analyze_result = analyze_query(oracle_context)
        assert analyze_result == RunStatus.SUCCESS

        # Verify state after full flow
        bb = oracle_context.blackboard
        assert bb._data["query_classification"]["query_type"] == "conversational"
        assert bb._data["needs_code"] is False
        assert bb._data["needs_vault"] is False
        assert bb._data["needs_web"] is False


# =============================================================================
# Test: Classification Confidence
# =============================================================================


class TestClassificationConfidence:
    """Test that classification includes confidence scores."""

    def test_weather_query_has_confidence(self, oracle_context):
        """Classification should include confidence score."""
        oracle_context.blackboard._data["query"] = "What's the weather?"

        analyze_query(oracle_context)

        bb = oracle_context.blackboard
        classification = bb._data["query_classification"]
        assert "confidence" in classification
        assert 0.0 <= classification["confidence"] <= 1.0

    def test_code_query_has_keywords_matched(self, oracle_context):
        """Classification should include matched keywords."""
        oracle_context.blackboard._data["query"] = "Where is the function?"

        analyze_query(oracle_context)

        bb = oracle_context.blackboard
        classification = bb._data["query_classification"]
        assert "keywords_matched" in classification
        assert isinstance(classification["keywords_matched"], list)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for query classification."""

    def test_empty_query_fails(self, oracle_context):
        """Empty query should fail analysis."""
        oracle_context.blackboard._data["query"] = ""

        result = analyze_query(oracle_context)

        assert result == RunStatus.FAILURE

    def test_whitespace_query_fails(self, oracle_context):
        """Whitespace-only query should fail analysis."""
        oracle_context.blackboard._data["query"] = "   "

        result = analyze_query(oracle_context)

        assert result == RunStatus.FAILURE

    def test_very_short_query_is_conversational(self, oracle_context):
        """Very short queries default to conversational."""
        oracle_context.blackboard._data["query"] = "Hi"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        bb = oracle_context.blackboard
        # Short queries are treated as conversational
        assert bb._data["query_classification"]["query_type"] == "conversational"

    def test_mixed_context_query(self, oracle_context):
        """Query with multiple keywords takes highest scoring type."""
        # This query has both code and web keywords
        oracle_context.blackboard._data["query"] = "What's the latest function implementation?"

        result = analyze_query(oracle_context)

        assert result == RunStatus.SUCCESS
        # The classifier should pick the dominant type
        bb = oracle_context.blackboard
        classification = bb._data["query_classification"]
        # Should pick one type (implementation details may vary)
        assert classification["query_type"] in ["code", "research"]


# =============================================================================
# US2 - Agent Self-Reflection via Signals
# Tasks covered: T026 from tasks-expanded-us2.md
#
# Acceptance Criteria Mapping:
# - US2-AC-1: need_turn with reason -> Assert signal type and reason field
# - US2-AC-4: BT parses signal -> Assert last_signal in blackboard
# - FR-005: Strip signal -> Assert <signal not in accumulated_content
# - FR-009: Log signal -> Assert signals_emitted list updated
# =============================================================================


@pytest.fixture
def signal_blackboard():
    """Create a blackboard configured for signal testing."""
    bb = TypedBlackboard(scope_name="signal-test")
    # Initialize signal-related fields
    bb._data["accumulated_content"] = ""
    bb._data["last_signal"] = None
    bb._data["signals_emitted"] = []
    bb._data["consecutive_same_reason"] = 0
    bb._data["turns_without_signal"] = 0
    bb._data["_signal_parsed_this_turn"] = False
    bb._data["_prev_signal"] = None
    bb._data["tool_results"] = []
    return bb


@pytest.fixture
def signal_context(signal_blackboard):
    """Create a TickContext for signal testing."""
    return TickContext(blackboard=signal_blackboard)


# =============================================================================
# Test: Signal Parsing Integration
# =============================================================================


class TestSignalParsingIntegration:
    """Test signal parsing from LLM responses (US2-AC-1, US2-AC-4)."""

    def test_parse_need_turn_signal(self, signal_context):
        """
        GIVEN an LLM response with need_turn signal
        WHEN parsed
        THEN signal is stored in blackboard with correct type and fields
        """
        signal_context.blackboard._data["accumulated_content"] = """
I encountered an error. Let me try a different approach.

<signal type="need_turn">
  <reason>search_code failed, trying vault search instead</reason>
  <confidence>0.8</confidence>
  <expected_turns>1</expected_turns>
</signal>"""

        from backend.src.bt.actions.signal_actions import parse_response_signal

        result = parse_response_signal(signal_context)

        assert result == RunStatus.SUCCESS
        signal = signal_context.blackboard._data["last_signal"]
        assert signal is not None
        assert signal["type"] == "need_turn"
        assert signal["confidence"] == 0.8
        assert "search_code failed" in signal["fields"]["reason"]

    def test_parse_context_sufficient_signal(self, signal_context):
        """
        GIVEN an LLM response with context_sufficient signal
        WHEN parsed
        THEN signal is stored with correct fields
        """
        signal_context.blackboard._data["accumulated_content"] = """
I found the answer.

<signal type="context_sufficient">
  <sources_found>3</sources_found>
  <confidence>0.92</confidence>
</signal>"""

        from backend.src.bt.actions.signal_actions import parse_response_signal

        parse_response_signal(signal_context)

        signal = signal_context.blackboard._data["last_signal"]
        assert signal is not None
        assert signal["type"] == "context_sufficient"
        assert signal["confidence"] == 0.92

    def test_parse_stuck_signal(self, signal_context):
        """
        GIVEN an LLM response with stuck signal
        WHEN parsed
        THEN signal includes attempted tools and blocker
        """
        signal_context.blackboard._data["accumulated_content"] = """
I cannot find the information.

<signal type="stuck">
  <attempted>["search_code", "search_vault"]</attempted>
  <blocker>No relevant results found</blocker>
  <confidence>0.85</confidence>
</signal>"""

        from backend.src.bt.actions.signal_actions import parse_response_signal

        parse_response_signal(signal_context)

        signal = signal_context.blackboard._data["last_signal"]
        assert signal is not None
        assert signal["type"] == "stuck"
        assert "search_code" in signal["fields"]["attempted"]


# =============================================================================
# Test: Signal Stripping (FR-005)
# =============================================================================


class TestSignalStripping:
    """Test signal XML stripping from responses (FR-005)."""

    def test_strip_signal_from_content(self, signal_context):
        """
        GIVEN an LLM response with signal
        WHEN stripped
        THEN signal XML is removed but content preserved
        """
        signal_context.blackboard._data["accumulated_content"] = """
Here is my answer.

<signal type="context_sufficient">
  <confidence>0.9</confidence>
</signal>"""

        from backend.src.bt.actions.signal_actions import strip_signal_from_response

        strip_signal_from_response(signal_context)

        cleaned = signal_context.blackboard._data["accumulated_content"]
        assert "<signal" not in cleaned
        assert "</signal>" not in cleaned
        assert "Here is my answer" in cleaned

    def test_strip_preserves_code_blocks(self, signal_context):
        """
        GIVEN an LLM response with code and signal
        WHEN stripped
        THEN code is preserved, signal removed
        """
        signal_context.blackboard._data["accumulated_content"] = """
Here's the code:

```python
def hello():
    return "world"
```

<signal type="context_sufficient">
  <confidence>0.9</confidence>
</signal>"""

        from backend.src.bt.actions.signal_actions import strip_signal_from_response

        strip_signal_from_response(signal_context)

        cleaned = signal_context.blackboard._data["accumulated_content"]
        assert "def hello():" in cleaned
        assert "<signal" not in cleaned


# =============================================================================
# Test: Signal Logging (FR-009)
# =============================================================================


class TestSignalLogging:
    """Test signal logging to signals_emitted list (FR-009)."""

    def test_signal_logged_to_list(self, signal_context):
        """
        GIVEN a parsed signal
        WHEN logged
        THEN signal is added to signals_emitted list
        """
        signal_context.blackboard._data["accumulated_content"] = """
<signal type="need_turn">
  <reason>Testing the signal logging functionality</reason>
  <confidence>0.8</confidence>
</signal>"""

        from backend.src.bt.actions.signal_actions import (
            parse_response_signal,
            log_signal,
        )

        parse_response_signal(signal_context)
        log_signal(signal_context)

        emitted = signal_context.blackboard._data["signals_emitted"]
        assert len(emitted) == 1
        assert emitted[0]["type"] == "need_turn"

    def test_multiple_signals_accumulated(self, signal_context):
        """
        GIVEN multiple signals over turns
        WHEN logged
        THEN all signals are in signals_emitted
        """
        from backend.src.bt.actions.signal_actions import (
            parse_response_signal,
            log_signal,
        )

        # First signal
        signal_context.blackboard._data["accumulated_content"] = """
<signal type="need_turn">
  <reason>First reason for needing more turns</reason>
  <confidence>0.8</confidence>
</signal>"""
        parse_response_signal(signal_context)
        log_signal(signal_context)

        # Second signal
        signal_context.blackboard._data["_signal_parsed_this_turn"] = False
        signal_context.blackboard._data["accumulated_content"] = """
<signal type="context_sufficient">
  <sources_found>3</sources_found>
  <confidence>0.9</confidence>
</signal>"""
        parse_response_signal(signal_context)
        log_signal(signal_context)

        emitted = signal_context.blackboard._data["signals_emitted"]
        assert len(emitted) == 2


# =============================================================================
# Test: Loop Detection via Signals (US3-AC-3)
# =============================================================================


class TestSignalLoopDetection:
    """Test loop detection via consecutive same-reason signals (US3-AC-3)."""

    def test_consecutive_same_reason_tracking(self, signal_context):
        """
        GIVEN 3 need_turn signals with same reason
        WHEN processed
        THEN consecutive_same_reason reaches 3
        """
        from backend.src.bt.actions.signal_actions import (
            parse_response_signal,
            update_signal_state,
        )

        for i in range(3):
            signal_context.blackboard._data["_signal_parsed_this_turn"] = False
            signal_context.blackboard._data["accumulated_content"] = """
<signal type="need_turn">
  <reason>API unavailable</reason>
  <confidence>0.7</confidence>
</signal>"""
            parse_response_signal(signal_context)
            update_signal_state(signal_context)

        assert signal_context.blackboard._data["consecutive_same_reason"] == 3

    def test_different_reason_resets_counter(self, signal_context):
        """
        GIVEN need_turn signals with different reasons
        WHEN processed
        THEN consecutive_same_reason resets to 1
        """
        from backend.src.bt.actions.signal_actions import (
            parse_response_signal,
            update_signal_state,
        )

        # First signal
        signal_context.blackboard._data["accumulated_content"] = """
<signal type="need_turn">
  <reason>First reason</reason>
  <confidence>0.8</confidence>
</signal>"""
        parse_response_signal(signal_context)
        update_signal_state(signal_context)

        # Different reason
        signal_context.blackboard._data["_signal_parsed_this_turn"] = False
        signal_context.blackboard._data["accumulated_content"] = """
<signal type="need_turn">
  <reason>Different reason</reason>
  <confidence>0.8</confidence>
</signal>"""
        parse_response_signal(signal_context)
        update_signal_state(signal_context)

        assert signal_context.blackboard._data["consecutive_same_reason"] == 1


# =============================================================================
# Test: Full Signal Flow
# =============================================================================


class TestFullSignalFlow:
    """Test complete signal processing sequence."""

    def test_tool_failure_triggers_need_turn_flow(self, signal_context):
        """
        GIVEN a tool failure
        WHEN agent responds with need_turn signal
        THEN full signal flow processes correctly
        """
        # Set up tool failure
        signal_context.blackboard._data["tool_results"] = [
            {"call_id": "call_1", "name": "search_code", "success": False, "error": "Index not ready"}
        ]
        signal_context.blackboard._data["accumulated_content"] = """
The search failed. Let me try another approach.

<signal type="need_turn">
  <reason>search_code failed, trying vault search</reason>
  <confidence>0.85</confidence>
</signal>"""

        from backend.src.bt.actions.signal_actions import (
            parse_response_signal,
            update_signal_state,
            log_signal,
            strip_signal_from_response,
        )

        # Run full sequence
        parse_response_signal(signal_context)
        update_signal_state(signal_context)
        log_signal(signal_context)
        strip_signal_from_response(signal_context)

        # Verify
        bb = signal_context.blackboard
        assert bb._data["last_signal"]["type"] == "need_turn"
        assert bb._data["consecutive_same_reason"] == 1
        assert len(bb._data["signals_emitted"]) == 1
        assert "<signal" not in bb._data["accumulated_content"]

    def test_signal_conditions_after_parsing(self, signal_context):
        """
        GIVEN a parsed stuck signal
        WHEN conditions are checked
        THEN signal_type_is returns SUCCESS for stuck
        """
        signal_context.blackboard._data["accumulated_content"] = """
<signal type="stuck">
  <attempted>["search_code"]</attempted>
  <blocker>No results</blocker>
  <confidence>0.8</confidence>
</signal>"""

        from backend.src.bt.actions.signal_actions import parse_response_signal
        from backend.src.bt.conditions.signals import signal_type_is

        parse_response_signal(signal_context)

        assert signal_type_is(signal_context, "stuck") == RunStatus.SUCCESS
        assert signal_type_is(signal_context, "need_turn") == RunStatus.FAILURE


# =============================================================================
# US4 - Dynamic Prompt Composition
# Tasks covered: T041 from tasks-expanded-us4.md
#
# Acceptance Criteria Mapping:
# - US4-AC1: Code query includes code-analysis.md segment
# - US4-AC2: Research query includes research.md segment
# - US4-AC3: Signal instructions always included
# - SC-010: Prompt composition is deterministic
# - FR-012: Signal emission instructions MUST always be included
# =============================================================================


class TestDynamicPromptComposition:
    """Integration tests for US4: Dynamic Prompt Composition."""

    @pytest.fixture
    def real_prompts_dir(self):
        """Use actual prompts from backend/src/prompts/oracle/."""
        from pathlib import Path

        return Path(__file__).parent.parent.parent / "src" / "prompts"

    def test_code_query_full_composition(self, real_prompts_dir):
        """Code query includes code-analysis.md with real prompts.

        Maps to: US4-AC1 (code query includes code analysis segment)
        """
        from pathlib import Path
        from backend.src.services.prompt_composer import compose_prompt_with_budget
        from backend.src.services.query_classifier import classify_query
        from backend.src.models.query_classification import QueryType

        if not real_prompts_dir.exists():
            pytest.skip("Prompts not deployed yet")

        # Classify a code question
        classification = classify_query("How does the auth middleware work?")
        assert classification.query_type == QueryType.CODE

        # Compose prompt
        prompt, token_count = compose_prompt_with_budget(
            classification.query_type,
            context={"project_name": "Test", "max_turns": 30, "project_context": ""},
            prompts_dir=real_prompts_dir,
        )

        # Verify code-analysis content present (citation format from code-analysis.md)
        assert "code" in prompt.lower()
        # Should mention file paths or line numbers (code citation pattern)
        assert "file" in prompt.lower() or "line" in prompt.lower()

        # Verify signals always present (FR-012)
        assert "signal" in prompt.lower()
        assert "<signal type=" in prompt

        # Verify research NOT present (wrong query type)
        # Research mode has specific markers like "web search first"
        assert "Web search first" not in prompt

        # Verify under budget
        assert token_count <= 8000

    def test_research_query_full_composition(self, real_prompts_dir):
        """Research query includes research.md with real prompts.

        Maps to: US4-AC2 (research query includes research segment)
        """
        from pathlib import Path
        from backend.src.services.prompt_composer import compose_prompt_with_budget
        from backend.src.services.query_classifier import classify_query
        from backend.src.models.query_classification import QueryType

        if not real_prompts_dir.exists():
            pytest.skip("Prompts not deployed yet")

        # Classify a research question
        classification = classify_query("What's the latest React 19 features?")
        assert classification.query_type == QueryType.RESEARCH

        # Compose prompt
        prompt, token_count = compose_prompt_with_budget(
            classification.query_type,
            context={"project_name": "Test", "max_turns": 30, "project_context": ""},
            prompts_dir=real_prompts_dir,
        )

        # Verify research content present
        assert "web" in prompt.lower() or "research" in prompt.lower()

        # Verify signals always present
        assert "signal" in prompt.lower()

        # Verify code analysis NOT present
        assert "Citation Format" not in prompt or "Code Analysis Mode" not in prompt

    def test_classification_to_composition_pipeline(self, real_prompts_dir):
        """Full pipeline: query -> classification -> prompt composition.

        Maps to: SC-010 (deterministic composition)
        """
        from pathlib import Path
        from backend.src.services.prompt_composer import compose_prompt_with_budget
        from backend.src.services.query_classifier import classify_query
        from backend.src.models.query_classification import QueryType

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
        from pathlib import Path
        from backend.src.services.prompt_composer import compose_prompt_with_budget
        from backend.src.models.query_classification import QueryType

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

    def test_query_classification_integration_with_bt(self, oracle_context):
        """Classification from BT action should work with prompt composition.

        This tests the integration between US1 query classification
        and US4 prompt composition.
        """
        from backend.src.services.prompt_composer import compose_prompt_with_budget

        # Set up query
        oracle_context.blackboard._data["query"] = "How does authentication work?"

        # Run BT analysis
        result = analyze_query(oracle_context)
        assert result == RunStatus.SUCCESS

        # Get classification from blackboard
        classification = oracle_context.blackboard._data["query_classification"]
        assert classification["query_type"] == "code"

        # Convert to QueryType and compose
        from backend.src.models.query_classification import QueryType

        query_type = QueryType(classification["query_type"])

        # Compose prompt (using real prompts if available)
        try:
            prompt, tokens = compose_prompt_with_budget(
                query_type,
                context={"project_name": "Test", "max_turns": 30},
            )
            assert len(prompt) > 0
            assert tokens > 0
        except FileNotFoundError:
            # Prompts not deployed yet, acceptable in CI
            pass

    def test_build_system_prompt_uses_classification(self, oracle_context):
        """build_system_prompt BT action uses query classification.

        Verifies the full flow: query -> classify -> build prompt
        """
        # Set up query and run analysis
        oracle_context.blackboard._data["query"] = "Where is the validate function?"

        reset_result = reset_state(oracle_context)
        assert reset_result == RunStatus.SUCCESS

        analyze_result = analyze_query(oracle_context)
        assert analyze_result == RunStatus.SUCCESS

        # Build system prompt
        prompt_result = build_system_prompt(oracle_context)
        assert prompt_result == RunStatus.SUCCESS

        # Verify prompt was built - it's stored as messages[0]["content"]
        messages = oracle_context.blackboard._data.get("messages", [])
        assert len(messages) > 0
        assert messages[0]["role"] == "system"
        system_prompt = messages[0]["content"]
        assert len(system_prompt) > 0

        # If real prompts are deployed, check for signal content
        if "signal" in system_prompt.lower():
            assert any(
                marker in system_prompt
                for marker in ["need_turn", "context_sufficient", "<signal"]
            )


# =============================================================================
# US3 - Budget and Loop Enforcement
# Tasks covered: T034 from tasks-expanded-us3.md
#
# Acceptance Criteria Mapping:
# - US3-AC1: Agent at 29/30 turns gets one final turn
# - US3-AC2: Agent at 30/30 turns forced completion with partial answer
# - US3-AC3: 3 consecutive same reason triggers stuck detection
# - US3-AC4: BERT fallback (or force_completion) on stuck
# - FR-007: Configurable max turn limits (default 30)
# - SC-004: Agent completes within turn budget 99% of time
# =============================================================================


@pytest.fixture
def budget_blackboard():
    """Create a blackboard configured for budget testing."""
    bb = TypedBlackboard(scope_name="budget-test")
    # Initialize budget-related fields
    bb._data["turn"] = 0
    bb._data["accumulated_content"] = ""
    bb._data["messages"] = [{"role": "user", "content": "test query"}]
    bb._data["tool_calls"] = []
    bb._data["force_complete"] = False
    bb._data["iteration_warning_emitted"] = False
    bb._data["iteration_exceeded_emitted"] = False
    bb._data["consecutive_same_reason"] = 0
    bb._data["loop_detected"] = False
    bb._data["loop_warning_emitted"] = False
    return bb


@pytest.fixture
def budget_context(budget_blackboard):
    """Create a TickContext for budget testing."""
    return TickContext(blackboard=budget_blackboard)


class TestBudgetEnforcementIntegration:
    """Integration tests for US3: Budget and Loop Enforcement."""

    @pytest.fixture
    def mock_oracle_config_5_turns(self):
        """Oracle config with max_turns=5 for testing."""
        config = MagicMock()
        config.max_turns = 5
        config.iteration_warning_threshold = 0.70
        config.token_warning_threshold = 0.80
        config.context_warning_threshold = 0.70
        config.loop_threshold = 3
        return config

    def test_max_turns_5_forces_completion_at_turn_5(
        self, mock_oracle_config_5_turns, budget_context
    ):
        """Agent should be forced to complete at turn 5 when max_turns=5.

        This is the US3 Independent Test: Set max turns to 5 and verify
        agent is forced to respond after 5 turns even if requesting more.
        """
        from backend.src.bt.conditions.budget import is_over_budget
        from backend.src.bt.actions.budget_actions import force_completion

        # Set turn to 5 (at max_turns limit with max=5)
        budget_context.blackboard._data["turn"] = 5
        budget_context.blackboard._data["accumulated_content"] = "Partial response so far..."

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_oracle_config_5_turns
        ), patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_oracle_config_5_turns
        ):
            # Should detect over budget
            result = is_over_budget(budget_context)
            assert result == RunStatus.SUCCESS

            # Should force completion
            result = force_completion(budget_context)
            assert result == RunStatus.SUCCESS

            # Should have truncation notice
            accumulated = budget_context.blackboard._data["accumulated_content"]
            assert "[Response truncated due to budget limit]" in accumulated

            # Should have force_complete flag set
            assert budget_context.blackboard._data.get("force_complete") is True

    def test_max_turns_5_allows_turn_4(
        self, mock_oracle_config_5_turns, budget_context
    ):
        """Agent should be allowed one more turn at turn 4 when max_turns=5.

        Maps to: US3-AC1 (agent at 29/30 turns gets one more)
        """
        from backend.src.bt.conditions.budget import (
            is_over_budget,
            is_at_budget_limit,
            turns_remaining,
        )

        # Set turn to 4 (last turn when max=5)
        budget_context.blackboard._data["turn"] = 4

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_oracle_config_5_turns
        ):
            # Should NOT be over budget
            assert is_over_budget(budget_context) == RunStatus.FAILURE

            # SHOULD be at budget limit (last turn)
            assert is_at_budget_limit(budget_context) == RunStatus.SUCCESS

            # Should have turns remaining (1 turn left)
            assert turns_remaining(budget_context) == RunStatus.SUCCESS

    def test_budget_warning_emitted_at_70_percent(
        self, mock_oracle_config_5_turns, budget_context
    ):
        """Budget warning should be emitted at 70% (turn 3 when max=5).

        70% of 5 = 3.5, rounds down to 3.
        """
        from backend.src.bt.actions.budget_actions import emit_budget_warning

        # Set turn to 4 (which is >= 70% threshold of 3)
        budget_context.blackboard._data["turn"] = 4
        budget_context.blackboard._data["iteration_warning_emitted"] = False

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_oracle_config_5_turns
        ):
            result = emit_budget_warning(budget_context)
            assert result == RunStatus.SUCCESS

            # Should have marked warning as emitted
            assert budget_context.blackboard._data.get("iteration_warning_emitted") is True

    def test_budget_warning_only_emitted_once(
        self, mock_oracle_config_5_turns, budget_context
    ):
        """Budget warning should only be emitted once per session."""
        from backend.src.bt.actions.budget_actions import emit_budget_warning

        budget_context.blackboard._data["turn"] = 4
        budget_context.blackboard._data["iteration_warning_emitted"] = True  # Already emitted

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_oracle_config_5_turns
        ):
            # Call should succeed but not emit again
            result = emit_budget_warning(budget_context)
            assert result == RunStatus.SUCCESS

            # Flag should still be True
            assert budget_context.blackboard._data.get("iteration_warning_emitted") is True

    def test_stuck_loop_forces_completion(
        self, mock_oracle_config_5_turns, budget_context
    ):
        """Stuck loop should force completion even before max turns.

        Maps to: US3-AC3, US3-AC4 (3 consecutive same reason triggers stuck/fallback)
        """
        from backend.src.bt.conditions.loop_detection import is_stuck_loop
        from backend.src.bt.actions.budget_actions import force_completion

        # Set up stuck loop scenario: 3+ consecutive same-reason signals
        budget_context.blackboard._data["turn"] = 2  # Only turn 2, but stuck
        budget_context.blackboard._data["consecutive_same_reason"] = 3  # Stuck!
        budget_context.blackboard._data["loop_detected"] = False
        budget_context.blackboard._data["accumulated_content"] = "Working on it..."

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_oracle_config_5_turns
        ), patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_oracle_config_5_turns
        ):
            # Should detect stuck loop
            assert is_stuck_loop(budget_context) == RunStatus.SUCCESS

            # Should force completion
            result = force_completion(budget_context)
            assert result == RunStatus.SUCCESS
            assert "[Response truncated" in budget_context.blackboard._data["accumulated_content"]

    def test_tool_pattern_loop_forces_completion(
        self, mock_oracle_config_5_turns, budget_context
    ):
        """Tool pattern loop (loop_detected flag) should also trigger stuck."""
        from backend.src.bt.conditions.loop_detection import is_stuck_loop

        # Set up tool pattern loop (via existing detection)
        budget_context.blackboard._data["consecutive_same_reason"] = 0  # No signal loop
        budget_context.blackboard._data["loop_detected"] = True  # Tool pattern loop

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_oracle_config_5_turns
        ):
            # Should detect stuck loop from tool pattern
            assert is_stuck_loop(budget_context) == RunStatus.SUCCESS

    def test_two_consecutive_signals_not_stuck(
        self, mock_oracle_config_5_turns, budget_context
    ):
        """2 consecutive same-reason signals should NOT trigger stuck."""
        from backend.src.bt.conditions.loop_detection import is_stuck_loop

        budget_context.blackboard._data["consecutive_same_reason"] = 2  # Not enough
        budget_context.blackboard._data["loop_detected"] = False

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=mock_oracle_config_5_turns
        ):
            assert is_stuck_loop(budget_context) == RunStatus.FAILURE


class TestEndToEndBudgetEnforcement:
    """End-to-end tests simulating full agent loop with budget."""

    def test_agent_stops_at_max_turns_even_with_tool_calls(self):
        """Agent should stop at max turns even if still making tool calls.

        Maps to: US3-AC2 (forced completion at max turns)
        """
        from backend.src.bt.conditions.budget import is_over_budget

        config = MagicMock()
        config.max_turns = 5

        bb = TypedBlackboard(scope_name="test")
        bb._data["turn"] = 5
        bb._data["tool_calls"] = [
            {"id": "call_1", "function": {"name": "search", "arguments": "{}"}}
        ]
        bb._data["accumulated_content"] = "Still working..."

        ctx = TickContext(blackboard=bb)

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ):
            # Even with pending tool calls, should detect over budget
            assert is_over_budget(ctx) == RunStatus.SUCCESS
            assert len(bb._data["tool_calls"]) > 0  # Tool calls still present

    def test_force_complete_flag_stops_loop(self):
        """force_complete flag should cause loop to exit."""
        bb = TypedBlackboard(scope_name="test")
        bb._data["force_complete"] = True

        # Simulating the BT condition check in oracle-agent.lua:
        # BT.condition("is-force-complete", { expression = "bb.force_complete == true" })
        is_force_complete = bb._data.get("force_complete") == True
        assert is_force_complete is True

    def test_full_budget_enforcement_flow(self):
        """Full flow: reach budget -> force completion -> exit.

        Simulates what happens in oracle-agent.lua:
        1. is_over_budget returns SUCCESS
        2. force_completion sets flag and adds truncation notice
        3. emit_done would complete the response
        """
        from backend.src.bt.conditions.budget import is_over_budget
        from backend.src.bt.actions.budget_actions import force_completion

        config = MagicMock()
        config.max_turns = 5
        config.iteration_warning_threshold = 0.70

        bb = TypedBlackboard(scope_name="test")
        bb._data["turn"] = 5
        bb._data["accumulated_content"] = "Here's what I found so far..."
        bb._data["messages"] = [{"role": "user", "content": "Complex query"}]
        bb._data["force_complete"] = False

        ctx = TickContext(blackboard=bb)

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ), patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ):
            # Step 1: Check budget
            assert is_over_budget(ctx) == RunStatus.SUCCESS

            # Step 2: Force completion
            result = force_completion(ctx)
            assert result == RunStatus.SUCCESS

            # Step 3: Verify state
            assert bb._data["force_complete"] is True
            assert "[Response truncated" in bb._data["accumulated_content"]

    def test_custom_loop_threshold(self):
        """Loop threshold should be configurable."""
        from backend.src.bt.conditions.loop_detection import is_stuck_loop

        # Test with threshold=5
        config = MagicMock()
        config.loop_threshold = 5

        bb = TypedBlackboard(scope_name="test")
        bb._data["consecutive_same_reason"] = 4  # Below threshold
        bb._data["loop_detected"] = False

        ctx = TickContext(blackboard=bb)

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ):
            # 4 < 5, should NOT be stuck
            assert is_stuck_loop(ctx) == RunStatus.FAILURE

            # Now set to 5
            bb._data["consecutive_same_reason"] = 5
            assert is_stuck_loop(ctx) == RunStatus.SUCCESS


class TestBudgetWithSignals:
    """Tests combining budget enforcement with signal processing."""

    def test_stuck_signal_combined_with_loop_detection(self, budget_context):
        """Stuck signal from agent combined with consecutive reason tracking.

        When agent emits a stuck signal AND has been repeating the same reason,
        the system should force completion.
        """
        from backend.src.bt.conditions.loop_detection import is_stuck_loop
        from backend.src.bt.conditions.signals import signal_type_is

        config = MagicMock()
        config.loop_threshold = 3

        # Agent has been repeating the same reason and now signals stuck
        budget_context.blackboard._data["consecutive_same_reason"] = 3
        budget_context.blackboard._data["loop_detected"] = False
        budget_context.blackboard._data["last_signal"] = {
            "type": "stuck",
            "confidence": 0.9,
            "fields": {"attempted": ["search"], "blocker": "No results"}
        }

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ):
            # Both conditions should indicate stuck
            assert is_stuck_loop(budget_context) == RunStatus.SUCCESS
            assert signal_type_is(budget_context, "stuck") == RunStatus.SUCCESS

    def test_need_turn_signal_increments_consecutive_counter(self, budget_context):
        """Processing need_turn signals should update consecutive counter.

        This tests the integration between signal processing (US2) and
        loop detection (US3).
        """
        from backend.src.bt.actions.signal_actions import (
            parse_response_signal,
            update_signal_state,
        )

        # Simulate 3 turns with same need_turn reason
        for i in range(3):
            budget_context.blackboard._data["_signal_parsed_this_turn"] = False
            budget_context.blackboard._data["accumulated_content"] = f"""
Turn {i+1} response.

<signal type="need_turn">
  <reason>API rate limited, retrying</reason>
  <confidence>0.8</confidence>
</signal>"""

            parse_response_signal(budget_context)
            update_signal_state(budget_context)

        # After 3 turns with same reason, counter should be 3
        assert budget_context.blackboard._data["consecutive_same_reason"] == 3


# =============================================================================
# US5 - BERT Fallback for Edge Cases
# Tasks covered: T049 from tasks-expanded-us5.md
#
# Acceptance Criteria Mapping:
# - US5-AC-1: Given agent response with no signal, when 3 turns pass, BERT fallback activates
# - US5-AC-2: Given signal with confidence < 0.3, BERT fallback is consulted
# - US5-AC-3: Given explicit `stuck` signal, BERT attempts to identify alternative strategy
# - FR-019: BERT fallback activates when no signal for 3+ turns
# - FR-020: BERT fallback activates when signal confidence < 0.3
# - FR-021: BERT fallback activates on explicit `stuck` signal
# - FR-022: System functions with heuristic defaults when BERT unavailable
# =============================================================================


@pytest.fixture
def fallback_blackboard():
    """Create a blackboard configured for fallback testing."""
    bb = TypedBlackboard(scope_name="fallback-test")
    # Initialize fallback-related fields
    bb._data["query"] = "Complex question"
    bb._data["accumulated_content"] = ""
    bb._data["messages"] = []
    bb._data["turns_without_signal"] = 0
    bb._data["last_signal"] = None
    bb._data["tool_results"] = []
    bb._data["fallback_classification"] = None
    bb._data["_pending_chunks"] = []
    return bb


@pytest.fixture
def fallback_context(fallback_blackboard):
    """Create a TickContext for fallback testing."""
    return TickContext(blackboard=fallback_blackboard)


class TestFallbackTriggerIntegration:
    """Integration tests for US5 fallback triggering."""

    def test_three_turns_without_signals_triggers_fallback(self, fallback_context):
        """3 turns without signals triggers fallback classification (US5-AC-1, FR-019).

        This is the US5 Independent Test.
        """
        from backend.src.bt.conditions.fallback import needs_fallback
        from backend.src.bt.actions.fallback_actions import trigger_fallback

        # Set up state: 3 turns without signal
        fallback_context.blackboard._data["turns_without_signal"] = 3
        fallback_context.blackboard._data["last_signal"] = None
        fallback_context.blackboard._data["accumulated_content"] = ""
        fallback_context.blackboard._data["tool_results"] = []
        fallback_context.blackboard._data["query"] = "Test question"

        # Should trigger fallback
        assert needs_fallback(fallback_context) is True

        # Trigger and verify classification is stored
        result = trigger_fallback(fallback_context)
        assert result == RunStatus.SUCCESS

        classification = fallback_context.blackboard._data.get("fallback_classification")
        assert classification is not None
        assert classification["action"] in ["continue", "force_response", "retry_with_hint", "escalate"]
        assert 0.0 <= classification["confidence"] <= 1.0

    def test_low_confidence_signal_triggers_fallback(self, fallback_context):
        """Signal with confidence < 0.3 triggers fallback (US5-AC-2, FR-020)."""
        from backend.src.bt.conditions.fallback import needs_fallback

        # Set up state: low confidence signal
        fallback_context.blackboard._data["turns_without_signal"] = 0
        fallback_context.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "confidence": 0.25,  # Below 0.3 threshold
            "fields": {"reason": "Not sure what to do"},
        }

        # Should trigger fallback
        assert needs_fallback(fallback_context) is True

    def test_stuck_signal_triggers_immediate_fallback(self, fallback_context):
        """Explicit stuck signal triggers fallback immediately (US5-AC-3, FR-021)."""
        from backend.src.bt.conditions.fallback import needs_fallback
        from backend.src.bt.actions.fallback_actions import trigger_fallback

        # Set up state: stuck signal (even on first occurrence)
        fallback_context.blackboard._data["turns_without_signal"] = 0  # Recent signal
        fallback_context.blackboard._data["last_signal"] = {
            "type": "stuck",
            "confidence": 0.9,  # High confidence stuck
            "fields": {
                "attempted": ["search_code", "read_file"],
                "blocker": "Cannot find file",
            },
        }
        fallback_context.blackboard._data["query"] = "Find the auth middleware"
        fallback_context.blackboard._data["tool_results"] = [
            {"name": "search_code", "success": False, "error": "No results"},
        ]

        # Should trigger fallback
        assert needs_fallback(fallback_context) is True

        # Trigger and verify
        result = trigger_fallback(fallback_context)
        assert result == RunStatus.SUCCESS

        classification = fallback_context.blackboard._data.get("fallback_classification")
        assert classification is not None

    def test_normal_state_does_not_trigger_fallback(self, fallback_context):
        """Normal operation should NOT trigger fallback."""
        from backend.src.bt.conditions.fallback import needs_fallback

        # Set up normal state
        fallback_context.blackboard._data["turns_without_signal"] = 1  # Recent signal
        fallback_context.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "confidence": 0.85,  # High confidence
            "fields": {"reason": "Need more context"},
        }

        # Should NOT trigger fallback
        assert needs_fallback(fallback_context) is False


class TestFallbackActionIntegration:
    """Integration tests for fallback action application."""

    def test_force_response_injects_system_message(self, fallback_context):
        """FORCE_RESPONSE action should inject system message."""
        from backend.src.bt.actions.fallback_actions import apply_heuristic_classification

        # Set up state with FORCE_RESPONSE classification
        fallback_context.blackboard._data["fallback_classification"] = {
            "action": "force_response",
            "confidence": 0.8,
            "hint": None,
            "reason": "Test force response",
        }
        fallback_context.blackboard._data["messages"] = []

        # Apply classification
        result = apply_heuristic_classification(fallback_context)
        assert result == RunStatus.SUCCESS

        # Verify system message was injected
        messages = fallback_context.blackboard._data["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "final response" in messages[0]["content"].lower()

        # Classification should be cleared
        assert fallback_context.blackboard._data.get("fallback_classification") is None

    def test_retry_with_hint_injects_guidance(self, fallback_context):
        """RETRY_WITH_HINT action should inject hint message."""
        from backend.src.bt.actions.fallback_actions import apply_heuristic_classification

        # Set up state with RETRY_WITH_HINT classification
        fallback_context.blackboard._data["fallback_classification"] = {
            "action": "retry_with_hint",
            "confidence": 0.6,
            "hint": "Try searching in the vault instead of code.",
            "reason": "No progress",
        }
        fallback_context.blackboard._data["messages"] = []

        # Apply classification
        result = apply_heuristic_classification(fallback_context)
        assert result == RunStatus.SUCCESS

        # Verify hint message was injected
        messages = fallback_context.blackboard._data["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "vault" in messages[0]["content"].lower()

    def test_escalate_emits_to_frontend(self, fallback_context):
        """ESCALATE action should emit system chunk and add message."""
        from backend.src.bt.actions.fallback_actions import apply_heuristic_classification

        # Set up state with ESCALATE classification
        fallback_context.blackboard._data["fallback_classification"] = {
            "action": "escalate",
            "confidence": 0.7,
            "hint": None,
            "reason": "High failure rate",
        }
        fallback_context.blackboard._data["messages"] = []
        fallback_context.blackboard._data["_pending_chunks"] = []

        # Apply classification
        result = apply_heuristic_classification(fallback_context)
        assert result == RunStatus.SUCCESS

        # Verify system message was added to messages
        messages = fallback_context.blackboard._data["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

        # Verify chunk was emitted to frontend
        chunks = fallback_context.blackboard._data.get("_pending_chunks", [])
        assert len(chunks) == 1
        assert chunks[0]["type"] == "system"
        assert chunks[0]["severity"] == "warning"

    def test_continue_does_nothing(self, fallback_context):
        """CONTINUE action should not modify messages."""
        from backend.src.bt.actions.fallback_actions import apply_heuristic_classification

        # Set up state with CONTINUE classification
        fallback_context.blackboard._data["fallback_classification"] = {
            "action": "continue",
            "confidence": 0.5,
            "hint": None,
            "reason": "No clear trigger",
        }
        initial_messages = [{"role": "user", "content": "test"}]
        fallback_context.blackboard._data["messages"] = initial_messages.copy()

        # Apply classification
        result = apply_heuristic_classification(fallback_context)
        assert result == RunStatus.SUCCESS

        # Messages should be unchanged (same length)
        messages = fallback_context.blackboard._data["messages"]
        assert len(messages) == 1


class TestFallbackFullFlow:
    """Full flow integration tests for fallback mechanism."""

    def test_three_turns_no_signal_full_flow(self, fallback_context):
        """Full flow: 3 turns without signal -> trigger -> apply -> message injected."""
        from backend.src.bt.conditions.fallback import needs_fallback
        from backend.src.bt.actions.fallback_actions import (
            trigger_fallback,
            apply_heuristic_classification,
        )

        # Set up state: 3 turns without signal, no content
        # Use a long query with tool usage to avoid simple query logic
        fallback_context.blackboard._data["turns_without_signal"] = 3
        fallback_context.blackboard._data["last_signal"] = None
        fallback_context.blackboard._data["accumulated_content"] = ""
        fallback_context.blackboard._data["tool_results"] = [{"name": "search_code", "success": True}]  # Has tool use
        fallback_context.blackboard._data["query"] = "I need to understand how the complex authentication middleware interacts with the session management and authorization subsystems in this project"
        fallback_context.blackboard._data["messages"] = []

        # Step 1: Check if fallback needed
        assert needs_fallback(fallback_context) is True

        # Step 2: Trigger fallback classification
        result = trigger_fallback(fallback_context)
        assert result == RunStatus.SUCCESS

        classification = fallback_context.blackboard._data.get("fallback_classification")
        assert classification is not None
        # Without content and 3+ turns, should be RETRY_WITH_HINT
        assert classification["action"] == "retry_with_hint"

        # Step 3: Apply the classification
        result = apply_heuristic_classification(fallback_context)
        assert result == RunStatus.SUCCESS

        # Verify message was injected
        messages = fallback_context.blackboard._data["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "Guidance" in messages[0]["content"]

        # Classification should be cleared
        assert fallback_context.blackboard._data.get("fallback_classification") is None

    def test_substantial_content_forces_response(self, fallback_context):
        """Substantial content accumulated -> force response."""
        from backend.src.bt.conditions.fallback import needs_fallback
        from backend.src.bt.actions.fallback_actions import (
            trigger_fallback,
            apply_heuristic_classification,
        )

        # Set up state: content accumulated but no signal
        fallback_context.blackboard._data["turns_without_signal"] = 3
        fallback_context.blackboard._data["last_signal"] = None
        fallback_context.blackboard._data["accumulated_content"] = "A" * 600  # > 500 chars
        fallback_context.blackboard._data["tool_results"] = []
        fallback_context.blackboard._data["query"] = "Explain the architecture"
        fallback_context.blackboard._data["messages"] = []

        # Trigger and apply
        assert needs_fallback(fallback_context) is True
        trigger_fallback(fallback_context)

        classification = fallback_context.blackboard._data.get("fallback_classification")
        assert classification["action"] == "force_response"

        apply_heuristic_classification(fallback_context)

        messages = fallback_context.blackboard._data["messages"]
        assert len(messages) == 1
        assert "final response" in messages[0]["content"].lower()

    def test_high_tool_failure_escalates(self, fallback_context):
        """High tool failure rate should escalate."""
        from backend.src.bt.conditions.fallback import needs_fallback
        from backend.src.bt.actions.fallback_actions import (
            trigger_fallback,
            apply_heuristic_classification,
        )

        # Set up state: many tool failures
        fallback_context.blackboard._data["turns_without_signal"] = 3
        fallback_context.blackboard._data["last_signal"] = None
        fallback_context.blackboard._data["accumulated_content"] = ""
        fallback_context.blackboard._data["tool_results"] = [
            {"name": "search_code", "success": False},
            {"name": "read_file", "success": False},
            {"name": "get_repo_map", "success": False},
        ]
        fallback_context.blackboard._data["query"] = "Debug the auth issue"
        fallback_context.blackboard._data["messages"] = []
        fallback_context.blackboard._data["_pending_chunks"] = []

        # Trigger and apply
        assert needs_fallback(fallback_context) is True
        trigger_fallback(fallback_context)

        classification = fallback_context.blackboard._data.get("fallback_classification")
        assert classification["action"] == "escalate"

        apply_heuristic_classification(fallback_context)

        # Should have emitted system chunk
        chunks = fallback_context.blackboard._data.get("_pending_chunks", [])
        assert len(chunks) == 1
        assert chunks[0]["severity"] == "warning"


class TestFallbackWithBudgetIntegration:
    """Tests combining fallback with budget enforcement."""

    def test_fallback_before_budget_limit(self, fallback_context):
        """Fallback should trigger even before reaching budget limit."""
        from backend.src.bt.conditions.fallback import needs_fallback
        from backend.src.bt.conditions.budget import is_over_budget

        config = MagicMock()
        config.max_turns = 30

        # Turn 3, not at budget limit but 3 turns without signal
        fallback_context.blackboard._data["turn"] = 3
        fallback_context.blackboard._data["turns_without_signal"] = 3
        fallback_context.blackboard._data["last_signal"] = None

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ):
            # Should trigger fallback
            assert needs_fallback(fallback_context) is True

            # Should NOT be over budget
            assert is_over_budget(fallback_context) == RunStatus.FAILURE

    def test_fallback_and_stuck_signal_combined(self, fallback_context):
        """Stuck signal should trigger both fallback and signal conditions."""
        from backend.src.bt.conditions.fallback import needs_fallback
        from backend.src.bt.conditions.signals import signal_type_is
        from backend.src.bt.conditions.loop_detection import is_stuck_loop

        config = MagicMock()
        config.loop_threshold = 3

        # Set up stuck state
        fallback_context.blackboard._data["turns_without_signal"] = 0
        fallback_context.blackboard._data["last_signal"] = {
            "type": "stuck",
            "confidence": 0.9,
            "fields": {"blocker": "Cannot proceed"},
        }
        fallback_context.blackboard._data["consecutive_same_reason"] = 0
        fallback_context.blackboard._data["loop_detected"] = False

        with patch(
            "backend.src.services.config.get_oracle_config",
            return_value=config
        ):
            # Fallback should trigger on stuck signal
            assert needs_fallback(fallback_context) is True

            # Signal condition should match
            assert signal_type_is(fallback_context, "stuck") == RunStatus.SUCCESS

            # But loop detection should NOT trigger (no consecutive reasons)
            assert is_stuck_loop(fallback_context) == RunStatus.FAILURE
