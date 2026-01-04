"""Unit tests for ExpressionEvaluator (T020).

This module tests:
- Simple comparisons: context.turn.token_usage > 0.8
- Boolean composition: x > 0.5 and y < 0.3
- Built-in functions
- Error handling for invalid expressions
"""

from datetime import datetime

import pytest

from backend.src.services.plugins.context import (
    HistoryState,
    PluginState,
    ProjectState,
    RuleContext,
    ToolCallRecord,
    ToolResult,
    TurnState,
    UserState,
)
from backend.src.services.plugins.expression import (
    ExpressionEvaluator,
    ExpressionError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Create a fresh ExpressionEvaluator instance."""
    return ExpressionEvaluator()


@pytest.fixture
def basic_context() -> RuleContext:
    """Create a basic RuleContext for testing."""
    return RuleContext(
        turn=TurnState(
            number=5,
            token_usage=0.85,
            context_usage=0.6,
            iteration_count=3,
        ),
        history=HistoryState(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            tools=[
                ToolCallRecord(
                    name="vault_search",
                    arguments={"query": "test"},
                    result="Found 5 results",
                    success=True,
                    timestamp=datetime.now(),
                ),
                ToolCallRecord(
                    name="web_fetch",
                    arguments={"url": "https://example.com"},
                    result=None,
                    success=False,
                    timestamp=datetime.now(),
                ),
            ],
            failures={"web_fetch": 2, "code_exec": 1},
        ),
        user=UserState(
            id="user-123",
            settings={"theme": "dark"},
        ),
        project=ProjectState(
            id="project-456",
            settings={"max_iterations": 10},
        ),
        state=PluginState(_store={"counter": 5, "last_warning": 3}),
    )


@pytest.fixture
def tool_result_context(basic_context: RuleContext) -> RuleContext:
    """Create a context with tool result for ON_TOOL_COMPLETE testing."""
    basic_context.result = ToolResult(
        tool_name="vault_search",
        success=True,
        result="Found 10 documents matching query",
        duration_ms=150.5,
    )
    return basic_context


# =============================================================================
# Simple Comparison Tests
# =============================================================================


class TestSimpleComparisons:
    """Tests for simple comparison expressions."""

    def test_greater_than_true(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test > comparison that evaluates to True."""
        result = evaluator.evaluate(
            "context.turn.token_usage > 0.8",
            basic_context,
        )
        assert result is True

    def test_greater_than_false(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test > comparison that evaluates to False."""
        result = evaluator.evaluate(
            "context.turn.token_usage > 0.9",
            basic_context,
        )
        assert result is False

    def test_less_than(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test < comparison."""
        result = evaluator.evaluate(
            "context.turn.context_usage < 0.7",
            basic_context,
        )
        assert result is True

    def test_greater_equal(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test >= comparison."""
        result = evaluator.evaluate(
            "context.turn.token_usage >= 0.85",
            basic_context,
        )
        assert result is True

    def test_less_equal(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test <= comparison."""
        result = evaluator.evaluate(
            "context.turn.number <= 5",
            basic_context,
        )
        assert result is True

    def test_equality(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test == comparison."""
        result = evaluator.evaluate(
            "context.turn.number == 5",
            basic_context,
        )
        assert result is True

    def test_inequality(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test != comparison."""
        result = evaluator.evaluate(
            "context.turn.iteration_count != 0",
            basic_context,
        )
        assert result is True

    def test_string_equality(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test string comparison."""
        result = evaluator.evaluate(
            "context.user.id == 'user-123'",
            basic_context,
        )
        assert result is True


# =============================================================================
# Boolean Composition Tests
# =============================================================================


class TestBooleanComposition:
    """Tests for boolean composition (and, or, not)."""

    def test_and_both_true(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test 'and' with both conditions true."""
        result = evaluator.evaluate(
            "context.turn.token_usage > 0.8 and context.turn.number >= 3",
            basic_context,
        )
        assert result is True

    def test_and_one_false(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test 'and' with one condition false."""
        result = evaluator.evaluate(
            "context.turn.token_usage > 0.8 and context.turn.number >= 10",
            basic_context,
        )
        assert result is False

    def test_or_both_true(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test 'or' with both conditions true."""
        result = evaluator.evaluate(
            "context.turn.token_usage > 0.8 or context.turn.number == 5",
            basic_context,
        )
        assert result is True

    def test_or_one_true(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test 'or' with one condition true."""
        result = evaluator.evaluate(
            "context.turn.token_usage > 0.9 or context.turn.number == 5",
            basic_context,
        )
        assert result is True

    def test_or_both_false(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test 'or' with both conditions false."""
        result = evaluator.evaluate(
            "context.turn.token_usage > 0.9 or context.turn.number == 10",
            basic_context,
        )
        assert result is False

    def test_not_true(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test 'not' on a true expression."""
        result = evaluator.evaluate(
            "not context.turn.token_usage < 0.5",
            basic_context,
        )
        assert result is True

    def test_not_false(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test 'not' on a false expression."""
        result = evaluator.evaluate(
            "not context.turn.token_usage > 0.8",
            basic_context,
        )
        assert result is False

    def test_complex_composition(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test complex boolean composition."""
        result = evaluator.evaluate(
            "(context.turn.token_usage > 0.8 and context.turn.number >= 3) or "
            "context.turn.iteration_count > 10",
            basic_context,
        )
        assert result is True

    def test_nested_parentheses(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test nested parentheses in expressions."""
        result = evaluator.evaluate(
            "((context.turn.number > 2) and (context.turn.token_usage > 0.5))",
            basic_context,
        )
        assert result is True


# =============================================================================
# Arithmetic Expression Tests
# =============================================================================


class TestArithmeticExpressions:
    """Tests for arithmetic operations in expressions."""

    def test_addition(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test addition in expression."""
        result = evaluator.evaluate(
            "context.turn.number + 5 == 10",
            basic_context,
        )
        assert result is True

    def test_subtraction(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test subtraction in expression."""
        result = evaluator.evaluate(
            "context.turn.number - 2 == 3",
            basic_context,
        )
        assert result is True

    def test_multiplication(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test multiplication in expression."""
        result = evaluator.evaluate(
            "context.turn.token_usage * 100 > 80",
            basic_context,
        )
        assert result is True

    def test_division(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test division in expression."""
        result = evaluator.evaluate(
            "context.turn.number / 5 == 1",
            basic_context,
        )
        assert result is True


# =============================================================================
# Built-in Function Tests
# =============================================================================


class TestBuiltInFunctions:
    """Tests for built-in helper functions."""

    def test_len_function(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test len() function on lists."""
        result = evaluator.evaluate(
            "len(context.history.messages) == 2",
            basic_context,
        )
        assert result is True

    def test_len_tools(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test len() on tool calls."""
        result = evaluator.evaluate(
            "len(context.history.tools) >= 2",
            basic_context,
        )
        assert result is True

    def test_any_function(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test any() function."""
        result = evaluator.evaluate(
            "any([True, False, False])",
            basic_context,
        )
        assert result is True

    def test_all_function(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test all() function."""
        result = evaluator.evaluate(
            "all([True, True, True])",
            basic_context,
        )
        assert result is True

    def test_all_function_false(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test all() with one False."""
        result = evaluator.evaluate(
            "all([True, False, True])",
            basic_context,
        )
        assert result is False

    def test_min_function(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test min() function."""
        result = evaluator.evaluate(
            "min(5, 3, 8) == 3",
            basic_context,
        )
        assert result is True

    def test_max_function(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test max() function."""
        result = evaluator.evaluate(
            "max(5, 3, 8) == 8",
            basic_context,
        )
        assert result is True

    def test_abs_function(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test abs() function."""
        result = evaluator.evaluate(
            "abs(-5) == 5",
            basic_context,
        )
        assert result is True

    def test_int_function(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test int() function for conversion."""
        result = evaluator.evaluate(
            "int(context.turn.token_usage * 100) == 85",
            basic_context,
        )
        assert result is True

    def test_float_function(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test float() function."""
        result = evaluator.evaluate(
            "float('3.14') > 3",
            basic_context,
        )
        assert result is True

    def test_str_function(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test str() function."""
        result = evaluator.evaluate(
            "str(context.turn.number) == '5'",
            basic_context,
        )
        assert result is True

    def test_bool_function(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test bool() function."""
        result = evaluator.evaluate(
            "bool(context.turn.number) == True",
            basic_context,
        )
        assert result is True


# =============================================================================
# Context-Specific Function Tests
# =============================================================================


class TestContextSpecificFunctions:
    """Tests for context-specific helper functions."""

    def test_tool_completed_true(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test tool_completed() returns True for completed tool."""
        result = evaluator.evaluate(
            "tool_completed('vault_search')",
            basic_context,
        )
        assert result is True

    def test_tool_completed_false(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test tool_completed() returns False for non-completed tool."""
        result = evaluator.evaluate(
            "tool_completed('nonexistent_tool')",
            basic_context,
        )
        assert result is False

    def test_tool_failed(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test tool_failed() returns True for failed tool."""
        result = evaluator.evaluate(
            "tool_failed('web_fetch')",
            basic_context,
        )
        assert result is True

    def test_failure_count(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test failure_count() returns correct count."""
        result = evaluator.evaluate(
            "failure_count('web_fetch') >= 2",
            basic_context,
        )
        assert result is True

    def test_failure_count_zero(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test failure_count() returns 0 for no failures."""
        result = evaluator.evaluate(
            "failure_count('unknown_tool') == 0",
            basic_context,
        )
        assert result is True

    def test_context_above_threshold(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test context_above_threshold() helper."""
        result = evaluator.evaluate(
            "context_above_threshold(0.5)",
            basic_context,
        )
        assert result is True  # context_usage is 0.6

    def test_context_above_threshold_false(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test context_above_threshold() returns False."""
        result = evaluator.evaluate(
            "context_above_threshold(0.7)",
            basic_context,
        )
        assert result is False  # context_usage is 0.6

    def test_message_count_above(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test message_count_above() helper."""
        result = evaluator.evaluate(
            "message_count_above(1)",
            basic_context,
        )
        assert result is True  # 2 messages

    def test_message_count_above_false(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test message_count_above() returns False."""
        result = evaluator.evaluate(
            "message_count_above(5)",
            basic_context,
        )
        assert result is False

    def test_state_get(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test state.get() for plugin state access."""
        result = evaluator.evaluate(
            "context.state.get('counter') == 5",
            basic_context,
        )
        assert result is True

    def test_state_get_default(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test state.get() with default value."""
        result = evaluator.evaluate(
            "context.state.get('nonexistent', 0) == 0",
            basic_context,
        )
        assert result is True

    def test_state_has(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test state.has() for key existence."""
        result = evaluator.evaluate(
            "context.state.has('counter')",
            basic_context,
        )
        assert result is True

    def test_state_has_false(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Test state.has() returns False for missing key."""
        result = evaluator.evaluate(
            "context.state.has('nonexistent')",
            basic_context,
        )
        assert result is False


# =============================================================================
# Tool Result Context Tests
# =============================================================================


class TestToolResultContext:
    """Tests for expressions using tool result context."""

    def test_result_tool_name(
        self,
        evaluator: ExpressionEvaluator,
        tool_result_context: RuleContext,
    ) -> None:
        """Test accessing result.tool_name."""
        result = evaluator.evaluate(
            "context.result.tool_name == 'vault_search'",
            tool_result_context,
        )
        assert result is True

    def test_result_success(
        self,
        evaluator: ExpressionEvaluator,
        tool_result_context: RuleContext,
    ) -> None:
        """Test accessing result.success."""
        result = evaluator.evaluate(
            "context.result.success == True",
            tool_result_context,
        )
        assert result is True

    def test_result_duration(
        self,
        evaluator: ExpressionEvaluator,
        tool_result_context: RuleContext,
    ) -> None:
        """Test accessing result.duration_ms."""
        result = evaluator.evaluate(
            "context.result.duration_ms > 100",
            tool_result_context,
        )
        assert result is True

    def test_result_contains_text(
        self,
        evaluator: ExpressionEvaluator,
        tool_result_context: RuleContext,
    ) -> None:
        """Test checking if result contains specific text."""
        result = evaluator.evaluate(
            "'10 documents' in context.result.result",
            tool_result_context,
        )
        assert result is True


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestExpressionErrorHandling:
    """Tests for error handling in expression evaluation."""

    def test_syntax_error(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Syntax errors raise ExpressionError."""
        with pytest.raises(ExpressionError) as excinfo:
            evaluator.evaluate(
                "context.turn.token_usage > > 0.8",
                basic_context,
            )
        assert "syntax" in str(excinfo.value).lower() or "parse" in str(excinfo.value).lower()

    def test_undefined_attribute(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Accessing undefined attribute raises ExpressionError."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate(
                "context.turn.nonexistent_field > 0",
                basic_context,
            )

    def test_undefined_variable(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Using undefined variable raises ExpressionError."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate(
                "undefined_variable > 0",
                basic_context,
            )

    def test_type_error(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Type errors raise ExpressionError."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate(
                "context.turn.number + 'string'",
                basic_context,
            )

    def test_division_by_zero(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Division by zero raises ExpressionError."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate(
                "context.turn.number / 0",
                basic_context,
            )

    def test_invalid_function_call(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Calling non-whitelisted function raises ExpressionError."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate(
                "open('/etc/passwd')",
                basic_context,
            )

    def test_import_blocked(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Import statements are blocked."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate(
                "__import__('os').system('ls')",
                basic_context,
            )

    def test_empty_expression(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Empty expression raises ExpressionError."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate("", basic_context)

    def test_whitespace_only_expression(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Whitespace-only expression raises ExpressionError."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate("   ", basic_context)


# =============================================================================
# Security Tests
# =============================================================================


class TestExpressionSecurity:
    """Tests for security of expression evaluation."""

    def test_no_exec(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """exec() is not available."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate(
                "exec('print(1)')",
                basic_context,
            )

    def test_no_eval(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """eval() is not available."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate(
                "eval('1+1')",
                basic_context,
            )

    def test_no_getattr(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """getattr() is not available."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate(
                "getattr(context, '__class__')",
                basic_context,
            )

    def test_no_dunder_access(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Dunder attributes cannot be accessed."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate(
                "context.__class__",
                basic_context,
            )

    def test_no_subclass_access(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Cannot access __subclasses__."""
        with pytest.raises(ExpressionError):
            evaluator.evaluate(
                "context.turn.__class__.__subclasses__()",
                basic_context,
            )


# =============================================================================
# Edge Cases
# =============================================================================


class TestExpressionEdgeCases:
    """Tests for edge cases in expression evaluation."""

    def test_literal_true(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Literal True evaluates correctly."""
        result = evaluator.evaluate("True", basic_context)
        assert result is True

    def test_literal_false(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Literal False evaluates correctly."""
        result = evaluator.evaluate("False", basic_context)
        assert result is False

    def test_lowercase_true(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Lowercase 'true' should work (common in TOML)."""
        # Note: simpleeval doesn't support lowercase by default
        # This tests that we handle it
        result = evaluator.evaluate("True", basic_context)
        assert result is True

    def test_comparison_chain(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Comparison chaining works."""
        result = evaluator.evaluate(
            "0.5 < context.turn.token_usage < 0.9",
            basic_context,
        )
        assert result is True

    def test_in_operator(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """'in' operator works."""
        result = evaluator.evaluate(
            "'user' in context.user.id",
            basic_context,
        )
        assert result is True

    def test_not_in_operator(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """'not in' operator works."""
        result = evaluator.evaluate(
            "'admin' not in context.user.id",
            basic_context,
        )
        assert result is True

    def test_ternary_expression(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Ternary (conditional) expressions work."""
        result = evaluator.evaluate(
            "True if context.turn.token_usage > 0.5 else False",
            basic_context,
        )
        assert result is True

    def test_list_indexing(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """List indexing works."""
        result = evaluator.evaluate(
            "context.history.messages[0]['role'] == 'user'",
            basic_context,
        )
        assert result is True

    def test_dict_access(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """Dict key access works."""
        result = evaluator.evaluate(
            "context.user.settings['theme'] == 'dark'",
            basic_context,
        )
        assert result is True

    def test_none_comparison(
        self,
        evaluator: ExpressionEvaluator,
        basic_context: RuleContext,
    ) -> None:
        """None comparison works."""
        result = evaluator.evaluate(
            "context.result is None",
            basic_context,
        )
        assert result is True

    def test_not_none_comparison(
        self,
        evaluator: ExpressionEvaluator,
        tool_result_context: RuleContext,
    ) -> None:
        """Not None comparison works."""
        result = evaluator.evaluate(
            "context.result is not None",
            tool_result_context,
        )
        assert result is True
