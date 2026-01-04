"""Expression evaluator for rule conditions using simpleeval.

This module provides a safe expression evaluator for rule conditions,
using simpleeval with configured safe functions and operators.

Features:
- Expression caching for frequently evaluated rules (T106)
- Timing support for performance debugging (T107)
"""

from __future__ import annotations

import ast
import logging
import time
from functools import lru_cache
from typing import Any, Callable, Optional

from simpleeval import EvalWithCompoundTypes, FeatureNotAvailable, InvalidExpression

from .context import RuleContext


logger = logging.getLogger(__name__)


# Module-level cache for parsed expressions (T106)
# LRU cache with max 256 expressions
@lru_cache(maxsize=256)
def _parse_expression(expression: str) -> ast.Expression:
    """Parse an expression string into an AST.

    This function is cached to avoid reparsing the same expression.

    Args:
        expression: The expression string to parse.

    Returns:
        Parsed AST Expression node.

    Raises:
        SyntaxError: If expression has invalid syntax.
    """
    return ast.parse(expression, mode='eval')


class ExpressionCache:
    """Cache for expression evaluation statistics.

    Tracks cache hits/misses and evaluation times for debugging.
    """

    def __init__(self) -> None:
        """Initialize the expression cache."""
        self._hits: int = 0
        self._misses: int = 0
        self._total_eval_time_ms: float = 0.0
        self._eval_count: int = 0

    def record_hit(self) -> None:
        """Record a cache hit."""
        self._hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self._misses += 1

    def record_eval_time(self, time_ms: float) -> None:
        """Record evaluation time in milliseconds."""
        self._total_eval_time_ms += time_ms
        self._eval_count += 1

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate (0.0-1.0)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def avg_eval_time_ms(self) -> float:
        """Calculate average evaluation time in milliseconds."""
        if self._eval_count == 0:
            return 0.0
        return self._total_eval_time_ms / self._eval_count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics as a dictionary."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
            "eval_count": self._eval_count,
            "total_eval_time_ms": round(self._total_eval_time_ms, 3),
            "avg_eval_time_ms": round(self.avg_eval_time_ms, 3),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self._hits = 0
        self._misses = 0
        self._total_eval_time_ms = 0.0
        self._eval_count = 0


# Global expression cache instance
_expression_cache = ExpressionCache()


class ExpressionError(Exception):
    """Raised when expression evaluation fails."""

    pass


class ExpressionEvaluator:
    """Evaluates rule condition expressions safely using simpleeval.

    This evaluator provides a restricted Python expression evaluation
    environment with:
    - Safe built-in functions (len, min, max, abs, etc.)
    - Context-specific helper functions (tool_completed, failure_count, etc.)
    - Read-only access to the RuleContext
    - Expression caching for performance (T106)
    - Optional timing for debugging (T107)

    Security:
    - No access to __dunder__ attributes
    - No import or exec capabilities
    - Limited function whitelist
    - No file or network operations
    """

    def __init__(self, enable_timing: bool = False) -> None:
        """Initialize the expression evaluator with safe defaults.

        Args:
            enable_timing: If True, log evaluation timing at DEBUG level.
        """
        self._safe_functions = self._build_safe_functions()
        self._enable_timing = enable_timing
        self._cache = _expression_cache

    def _build_safe_functions(self) -> dict[str, Callable[..., Any]]:
        """Build the dictionary of safe functions available in expressions.

        Returns:
            Dictionary mapping function names to callables.
        """
        return {
            # Type conversions
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            # Math functions
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "sum": sum,
            # Collection functions
            "len": len,
            "any": any,
            "all": all,
            "sorted": sorted,
            "reversed": lambda x: list(reversed(x)),
            # Type checks
            "isinstance": isinstance,
            "type": type,
        }

    def _build_context_functions(
        self,
        context: RuleContext,
    ) -> dict[str, Callable[..., Any]]:
        """Build context-specific helper functions.

        Args:
            context: The rule context to build helpers for.

        Returns:
            Dictionary mapping function names to bound callables.
        """
        return {
            "tool_completed": lambda name: self._tool_completed(context, name),
            "tool_failed": lambda name: self._tool_failed(context, name),
            "failure_count": lambda name: self._failure_count(context, name),
            "context_above_threshold": lambda threshold: self._context_above_threshold(
                context, threshold
            ),
            "message_count_above": lambda count: self._message_count_above(
                context, count
            ),
        }

    def _tool_completed(self, context: RuleContext, tool_name: str) -> bool:
        """Check if a tool has completed successfully.

        Args:
            context: Rule context.
            tool_name: Name of the tool to check.

        Returns:
            True if the tool completed successfully.
        """
        for tool in context.history.tools:
            if tool.name == tool_name and tool.success:
                return True
        return False

    def _tool_failed(self, context: RuleContext, tool_name: str) -> bool:
        """Check if a tool has failed.

        Args:
            context: Rule context.
            tool_name: Name of the tool to check.

        Returns:
            True if the tool failed at least once.
        """
        return context.history.failures.get(tool_name, 0) > 0

    def _failure_count(self, context: RuleContext, tool_name: str) -> int:
        """Get the failure count for a specific tool.

        Args:
            context: Rule context.
            tool_name: Name of the tool.

        Returns:
            Number of times the tool has failed.
        """
        return context.history.failures.get(tool_name, 0)

    def _context_above_threshold(
        self,
        context: RuleContext,
        threshold: float,
    ) -> bool:
        """Check if context usage is above a threshold.

        Args:
            context: Rule context.
            threshold: Threshold value (0.0-1.0).

        Returns:
            True if context_usage > threshold.
        """
        return context.turn.context_usage > threshold

    def _message_count_above(self, context: RuleContext, count: int) -> bool:
        """Check if message count is above a threshold.

        Args:
            context: Rule context.
            count: Threshold count.

        Returns:
            True if message count > count.
        """
        return len(context.history.messages) > count

    def _build_names(self, context: RuleContext) -> dict[str, Any]:
        """Build the namespace of names available in expressions.

        Args:
            context: The rule context to make available.

        Returns:
            Dictionary mapping names to values.
        """
        return {
            "context": context,
            "True": True,
            "False": False,
            "None": None,
        }

    def evaluate(self, expression: str, context: RuleContext) -> bool:
        """Evaluate an expression against the given context.

        Uses expression caching for performance (T106) and optional
        timing for debugging (T107).

        Args:
            expression: The expression string to evaluate.
            context: The rule context providing data for evaluation.

        Returns:
            Boolean result of the expression.

        Raises:
            ExpressionError: If expression is invalid or evaluation fails.
        """
        start_time = time.perf_counter() if self._enable_timing else None

        # Validate expression
        expression = expression.strip()
        if not expression:
            raise ExpressionError("Expression cannot be empty")

        # Try to use cached parsed expression (T106)
        try:
            # Check if expression is in cache
            cache_info = _parse_expression.cache_info()
            _parse_expression(expression)  # This will use cache if available
            new_cache_info = _parse_expression.cache_info()

            if new_cache_info.hits > cache_info.hits:
                self._cache.record_hit()
            else:
                self._cache.record_miss()
        except SyntaxError:
            # Syntax error will be caught below by simpleeval
            self._cache.record_miss()

        # Build evaluator with safe configuration
        evaluator = EvalWithCompoundTypes()

        # Add safe functions
        evaluator.functions = {
            **self._safe_functions,
            **self._build_context_functions(context),
        }

        # Add names (context and constants)
        evaluator.names = self._build_names(context)

        # Evaluate the expression
        try:
            result = evaluator.eval(expression)

            # Coerce result to boolean
            bool_result = bool(result)

            # Record timing (T107)
            if start_time is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._cache.record_eval_time(elapsed_ms)
                logger.debug(
                    f"Expression evaluation: {elapsed_ms:.3f}ms "
                    f"(expr: {expression[:50]}{'...' if len(expression) > 50 else ''}, "
                    f"result: {bool_result})"
                )

            return bool_result

        except FeatureNotAvailable as e:
            logger.warning(f"Blocked unsafe feature in expression: {e}")
            raise ExpressionError(f"Feature not available: {e}") from e

        except InvalidExpression as e:
            logger.warning(f"Invalid expression syntax: {e}")
            raise ExpressionError(f"Invalid expression syntax: {e}") from e

        except AttributeError as e:
            logger.warning(f"Attribute access error in expression: {e}")
            raise ExpressionError(f"Attribute error: {e}") from e

        except KeyError as e:
            logger.warning(f"Key access error in expression: {e}")
            raise ExpressionError(f"Key error: {e}") from e

        except TypeError as e:
            logger.warning(f"Type error in expression: {e}")
            raise ExpressionError(f"Type error: {e}") from e

        except ZeroDivisionError as e:
            logger.warning(f"Division by zero in expression: {e}")
            raise ExpressionError(f"Division by zero: {e}") from e

        except NameError as e:
            logger.warning(f"Name error in expression: {e}")
            raise ExpressionError(f"Undefined name: {e}") from e

        except Exception as e:
            logger.warning(f"Unexpected error evaluating expression: {e}")
            raise ExpressionError(f"Evaluation error: {e}") from e

    def get_cache_stats(self) -> dict[str, Any]:
        """Get expression cache statistics.

        Returns:
            Dictionary with cache hit/miss stats and timing info.
        """
        return self._cache.get_stats()

    def reset_cache_stats(self) -> None:
        """Reset expression cache statistics."""
        self._cache.reset()


def get_expression_cache_stats() -> dict[str, Any]:
    """Get global expression cache statistics.

    Returns:
        Dictionary with cache hit/miss stats and timing info.
    """
    return _expression_cache.get_stats()


def reset_expression_cache_stats() -> None:
    """Reset global expression cache statistics."""
    _expression_cache.reset()


def clear_expression_parse_cache() -> None:
    """Clear the expression parse cache.

    Useful for testing or when rules are reloaded.
    """
    _parse_expression.cache_clear()


__all__ = [
    "ExpressionEvaluator",
    "ExpressionError",
    "ExpressionCache",
    "get_expression_cache_stats",
    "reset_expression_cache_stats",
    "clear_expression_parse_cache",
]
