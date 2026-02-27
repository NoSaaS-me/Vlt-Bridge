"""Tests for query analysis action and context needs conditions.

Part of feature 020-bt-oracle-agent (US1 - Intelligent Context Selection).
Tasks covered: T018 from tasks-expanded-us1.md

Test Coverage:
- analyze_query action with various query types
- All context_needs condition functions
- Query classification to context needs flow

Acceptance Scenarios Tested:
- AS1.1: Weather query -> needs_web = True
- AS1.2: Code query -> needs_code = True
- AS1.3: Documentation query -> needs_vault = True
- AS1.4: Conversational query -> no context needed
"""

import pytest

from backend.src.bt.state.base import RunStatus
from backend.src.bt.state.blackboard import TypedBlackboard
from backend.src.bt.core.context import TickContext
from backend.src.bt.actions.query_analysis import analyze_query
from backend.src.bt.conditions.context_needs import (
    has_query_classification,
    needs_code_context,
    needs_vault_context,
    needs_web_context,
    is_conversational,
    any_context_needed,
    query_type_is,
    classification_confidence_above,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_blackboard() -> TypedBlackboard:
    """Create a basic blackboard for testing."""
    return TypedBlackboard(scope_name="test")


@pytest.fixture
def ctx_with_query():
    """Create a TickContext with query in blackboard."""
    def _create(query: str):
        bb = TypedBlackboard(scope_name="test")
        bb._data["query"] = query
        return TickContext(blackboard=bb)
    return _create


@pytest.fixture
def ctx_with_classification():
    """Create a TickContext with a pre-set classification."""
    def _create(
        query_type: str = "conversational",
        needs_code: bool = False,
        needs_vault: bool = False,
        needs_web: bool = False,
        confidence: float = 0.8,
    ):
        bb = TypedBlackboard(scope_name="test")
        bb._data["query_classification"] = {
            "query_type": query_type,
            "needs_code": needs_code,
            "needs_vault": needs_vault,
            "needs_web": needs_web,
            "confidence": confidence,
            "keywords_matched": [],
        }
        # Also set individual flags
        bb._data["needs_code"] = needs_code
        bb._data["needs_vault"] = needs_vault
        bb._data["needs_web"] = needs_web
        return TickContext(blackboard=bb)
    return _create


# =============================================================================
# analyze_query Action Tests
# =============================================================================


class TestAnalyzeQuery:
    """Test suite for analyze_query action."""

    def test_weather_query_classified_as_research(self, ctx_with_query):
        """Weather query should route to web search (AS1.1)."""
        ctx = ctx_with_query("What's the weather in Paris?")

        result = analyze_query(ctx)

        assert result == RunStatus.SUCCESS
        classification = ctx.blackboard._data["query_classification"]
        assert classification["query_type"] == "research"
        assert classification["needs_web"] is True
        assert classification["needs_code"] is False
        assert classification["needs_vault"] is False

    def test_code_query_classified_as_code(self, ctx_with_query):
        """Code question should route to code search (AS1.2)."""
        ctx = ctx_with_query("How does the auth middleware work?")

        result = analyze_query(ctx)

        assert result == RunStatus.SUCCESS
        classification = ctx.blackboard._data["query_classification"]
        assert classification["query_type"] == "code"
        assert classification["needs_code"] is True
        assert classification["needs_web"] is False

    def test_documentation_query_classified_as_documentation(self, ctx_with_query):
        """Documentation question should route to vault search (AS1.3)."""
        ctx = ctx_with_query("What did we decide about caching?")

        result = analyze_query(ctx)

        assert result == RunStatus.SUCCESS
        classification = ctx.blackboard._data["query_classification"]
        assert classification["query_type"] == "documentation"
        assert classification["needs_vault"] is True
        assert classification["needs_code"] is False
        assert classification["needs_web"] is False

    def test_thanks_classified_as_conversational(self, ctx_with_query):
        """Simple acknowledgment needs no tools (AS1.4)."""
        ctx = ctx_with_query("Thanks, that helps!")

        result = analyze_query(ctx)

        assert result == RunStatus.SUCCESS
        classification = ctx.blackboard._data["query_classification"]
        assert classification["query_type"] == "conversational"
        assert classification["needs_code"] is False
        assert classification["needs_vault"] is False
        assert classification["needs_web"] is False

    def test_missing_query_returns_failure(self, basic_blackboard):
        """No query in blackboard should fail."""
        ctx = TickContext(blackboard=basic_blackboard)

        result = analyze_query(ctx)

        assert result == RunStatus.FAILURE

    def test_empty_query_returns_failure(self, ctx_with_query):
        """Empty query string should fail."""
        ctx = ctx_with_query("")

        result = analyze_query(ctx)

        assert result == RunStatus.FAILURE

    def test_no_blackboard_returns_failure(self):
        """No blackboard should fail."""
        ctx = TickContext()

        result = analyze_query(ctx)

        assert result == RunStatus.FAILURE

    def test_dict_query_format(self):
        """Query as dict should work."""
        bb = TypedBlackboard(scope_name="test")
        bb._data["query"] = {"question": "What's the weather?"}
        ctx = TickContext(blackboard=bb)

        result = analyze_query(ctx)

        assert result == RunStatus.SUCCESS
        classification = ctx.blackboard._data["query_classification"]
        assert classification["needs_web"] is True

    def test_sets_individual_flags(self, ctx_with_query):
        """analyze_query should set individual needs_* flags."""
        ctx = ctx_with_query("Where is the login function?")

        result = analyze_query(ctx)

        assert result == RunStatus.SUCCESS
        # Individual flags should be set
        assert ctx.blackboard._data["needs_code"] is True
        assert ctx.blackboard._data["needs_vault"] is False
        assert ctx.blackboard._data["needs_web"] is False


# =============================================================================
# Context Needs Condition Tests
# =============================================================================


class TestHasQueryClassification:
    """Test suite for has_query_classification condition."""

    def test_returns_success_with_classification(self, ctx_with_classification):
        """Returns SUCCESS when classification exists."""
        ctx = ctx_with_classification(query_type="code")

        result = has_query_classification(ctx)

        assert result == RunStatus.SUCCESS

    def test_returns_failure_without_classification(self, basic_blackboard):
        """Returns FAILURE when no classification."""
        ctx = TickContext(blackboard=basic_blackboard)

        result = has_query_classification(ctx)

        assert result == RunStatus.FAILURE

    def test_returns_failure_no_blackboard(self):
        """Returns FAILURE when no blackboard."""
        ctx = TickContext()

        result = has_query_classification(ctx)

        assert result == RunStatus.FAILURE


class TestNeedsCodeContext:
    """Test suite for needs_code_context condition."""

    def test_returns_success_when_needs_code_true(self, ctx_with_classification):
        """Returns SUCCESS when needs_code is True."""
        ctx = ctx_with_classification(query_type="code", needs_code=True)

        result = needs_code_context(ctx)

        assert result == RunStatus.SUCCESS

    def test_returns_failure_when_needs_code_false(self, ctx_with_classification):
        """Returns FAILURE when needs_code is False."""
        ctx = ctx_with_classification(query_type="research", needs_code=False)

        result = needs_code_context(ctx)

        assert result == RunStatus.FAILURE

    def test_returns_failure_without_classification(self, basic_blackboard):
        """Returns FAILURE when no classification."""
        ctx = TickContext(blackboard=basic_blackboard)

        result = needs_code_context(ctx)

        assert result == RunStatus.FAILURE


class TestNeedsVaultContext:
    """Test suite for needs_vault_context condition."""

    def test_returns_success_when_needs_vault_true(self, ctx_with_classification):
        """Returns SUCCESS when needs_vault is True."""
        ctx = ctx_with_classification(query_type="documentation", needs_vault=True)

        result = needs_vault_context(ctx)

        assert result == RunStatus.SUCCESS

    def test_returns_failure_when_needs_vault_false(self, ctx_with_classification):
        """Returns FAILURE when needs_vault is False."""
        ctx = ctx_with_classification(query_type="code", needs_vault=False)

        result = needs_vault_context(ctx)

        assert result == RunStatus.FAILURE


class TestNeedsWebContext:
    """Test suite for needs_web_context condition."""

    def test_returns_success_when_needs_web_true(self, ctx_with_classification):
        """Returns SUCCESS when needs_web is True."""
        ctx = ctx_with_classification(query_type="research", needs_web=True)

        result = needs_web_context(ctx)

        assert result == RunStatus.SUCCESS

    def test_returns_failure_when_needs_web_false(self, ctx_with_classification):
        """Returns FAILURE when needs_web is False."""
        ctx = ctx_with_classification(query_type="code", needs_web=False)

        result = needs_web_context(ctx)

        assert result == RunStatus.FAILURE


class TestIsConversational:
    """Test suite for is_conversational condition."""

    def test_returns_success_for_conversational(self, ctx_with_classification):
        """Returns SUCCESS when query_type is conversational."""
        ctx = ctx_with_classification(query_type="conversational")

        result = is_conversational(ctx)

        assert result == RunStatus.SUCCESS

    def test_returns_failure_for_non_conversational(self, ctx_with_classification):
        """Returns FAILURE when query_type is not conversational."""
        ctx = ctx_with_classification(query_type="code")

        result = is_conversational(ctx)

        assert result == RunStatus.FAILURE


class TestAnyContextNeeded:
    """Test suite for any_context_needed condition."""

    def test_returns_success_when_code_needed(self, ctx_with_classification):
        """Returns SUCCESS when needs_code is True."""
        ctx = ctx_with_classification(query_type="code", needs_code=True)

        result = any_context_needed(ctx)

        assert result == RunStatus.SUCCESS

    def test_returns_success_when_vault_needed(self, ctx_with_classification):
        """Returns SUCCESS when needs_vault is True."""
        ctx = ctx_with_classification(query_type="documentation", needs_vault=True)

        result = any_context_needed(ctx)

        assert result == RunStatus.SUCCESS

    def test_returns_success_when_web_needed(self, ctx_with_classification):
        """Returns SUCCESS when needs_web is True."""
        ctx = ctx_with_classification(query_type="research", needs_web=True)

        result = any_context_needed(ctx)

        assert result == RunStatus.SUCCESS

    def test_returns_failure_when_no_context_needed(self, ctx_with_classification):
        """Returns FAILURE when no context needed (conversational)."""
        ctx = ctx_with_classification(
            query_type="conversational",
            needs_code=False,
            needs_vault=False,
            needs_web=False,
        )

        result = any_context_needed(ctx)

        assert result == RunStatus.FAILURE


class TestQueryTypeIs:
    """Test suite for query_type_is condition."""

    def test_returns_success_on_match(self, ctx_with_classification):
        """Returns SUCCESS when query_type matches."""
        ctx = ctx_with_classification(query_type="code")

        result = query_type_is(ctx, "code")

        assert result == RunStatus.SUCCESS

    def test_returns_failure_on_mismatch(self, ctx_with_classification):
        """Returns FAILURE when query_type doesn't match."""
        ctx = ctx_with_classification(query_type="code")

        result = query_type_is(ctx, "research")

        assert result == RunStatus.FAILURE

    def test_case_insensitive_matching(self, ctx_with_classification):
        """Matching should be case-insensitive."""
        ctx = ctx_with_classification(query_type="CODE")

        result = query_type_is(ctx, "code")

        assert result == RunStatus.SUCCESS


class TestClassificationConfidenceAbove:
    """Test suite for classification_confidence_above condition."""

    def test_returns_success_above_threshold(self, ctx_with_classification):
        """Returns SUCCESS when confidence >= threshold."""
        ctx = ctx_with_classification(confidence=0.8)

        result = classification_confidence_above(ctx, 0.5)

        assert result == RunStatus.SUCCESS

    def test_returns_failure_below_threshold(self, ctx_with_classification):
        """Returns FAILURE when confidence < threshold."""
        ctx = ctx_with_classification(confidence=0.3)

        result = classification_confidence_above(ctx, 0.5)

        assert result == RunStatus.FAILURE

    def test_default_threshold_is_05(self, ctx_with_classification):
        """Default threshold should be 0.5."""
        ctx = ctx_with_classification(confidence=0.6)

        result = classification_confidence_above(ctx)

        assert result == RunStatus.SUCCESS


# =============================================================================
# Integration: Query -> Classification -> Condition Flow
# =============================================================================


class TestQueryClassificationConditionFlow:
    """Integration tests for the full query classification to condition flow."""

    def test_weather_query_flow(self, ctx_with_query):
        """Weather query should pass needs_web_context."""
        ctx = ctx_with_query("What's the weather in Paris?")

        # Step 1: Analyze query
        analyze_result = analyze_query(ctx)
        assert analyze_result == RunStatus.SUCCESS

        # Step 2: Check conditions
        assert needs_web_context(ctx) == RunStatus.SUCCESS
        assert needs_code_context(ctx) == RunStatus.FAILURE
        assert needs_vault_context(ctx) == RunStatus.FAILURE
        assert is_conversational(ctx) == RunStatus.FAILURE

    def test_code_query_flow(self, ctx_with_query):
        """Code query should pass needs_code_context."""
        ctx = ctx_with_query("Where is the login function?")

        # Step 1: Analyze query
        analyze_result = analyze_query(ctx)
        assert analyze_result == RunStatus.SUCCESS

        # Step 2: Check conditions
        assert needs_code_context(ctx) == RunStatus.SUCCESS
        assert needs_web_context(ctx) == RunStatus.FAILURE
        assert is_conversational(ctx) == RunStatus.FAILURE

    def test_documentation_query_flow(self, ctx_with_query):
        """Documentation query should pass needs_vault_context."""
        ctx = ctx_with_query("Why did we choose this architecture?")

        # Step 1: Analyze query
        analyze_result = analyze_query(ctx)
        assert analyze_result == RunStatus.SUCCESS

        # Step 2: Check conditions
        assert needs_vault_context(ctx) == RunStatus.SUCCESS
        assert needs_web_context(ctx) == RunStatus.FAILURE
        assert is_conversational(ctx) == RunStatus.FAILURE

    def test_conversational_query_flow(self, ctx_with_query):
        """Conversational query should pass is_conversational."""
        ctx = ctx_with_query("Ok thanks!")

        # Step 1: Analyze query
        analyze_result = analyze_query(ctx)
        assert analyze_result == RunStatus.SUCCESS

        # Step 2: Check conditions
        assert is_conversational(ctx) == RunStatus.SUCCESS
        assert any_context_needed(ctx) == RunStatus.FAILURE
        assert needs_code_context(ctx) == RunStatus.FAILURE
        assert needs_vault_context(ctx) == RunStatus.FAILURE
        assert needs_web_context(ctx) == RunStatus.FAILURE
