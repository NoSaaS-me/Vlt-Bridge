"""Decorator nodes that modify child behavior.

This module implements decorators that wrap a single child and modify
its behavior in various ways:
- Inverter: Flip SUCCESS/FAILURE
- Succeeder: Always return SUCCESS
- Failer: Always return FAILURE
- UntilFail: Repeat until child fails
- UntilSuccess: Repeat until child succeeds
- Cooldown: Rate limiting (suppress for N ticks/ms)
- Guard: Conditional execution gating
- Retry: Retry on failure
- Timeout: Fail if child takes too long

Based on Honorbuddy decorator patterns (research.md Section 7.2).
"""

from __future__ import annotations

import time
from typing import Callable, Optional, TYPE_CHECKING
import logging

from .types import RunStatus, TickContext
from .node import Decorator, BehaviorNode

if TYPE_CHECKING:
    from ..context import RuleContext


logger = logging.getLogger(__name__)


class Inverter(Decorator):
    """Inverts the child's result.

    Converts SUCCESS to FAILURE and vice versa.
    RUNNING is passed through unchanged.

    Example:
        >>> # Check if file does NOT exist
        >>> inverter = Inverter(ConditionNode("file_exists"))
        >>> # Returns SUCCESS if file doesn't exist
    """

    def __init__(
        self,
        child: Optional[BehaviorNode] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the inverter.

        Args:
            child: Node to invert.
            name: Optional name for debugging.
        """
        super().__init__(child=child, name=name or "Inverter")

    def _tick(self, context: TickContext) -> RunStatus:
        """Tick child and invert its result.

        Args:
            context: The tick context.

        Returns:
            Inverted status (SUCCESS<->FAILURE, RUNNING unchanged).
        """
        if not self._child:
            return RunStatus.FAILURE

        status = self._child.tick(context)

        if status == RunStatus.SUCCESS:
            return RunStatus.FAILURE
        elif status == RunStatus.FAILURE:
            return RunStatus.SUCCESS
        else:
            return RunStatus.RUNNING


class Succeeder(Decorator):
    """Always returns SUCCESS regardless of child result.

    Useful for optional actions that shouldn't cause failure.
    RUNNING is passed through to support async operations.

    Example:
        >>> # Try to log but don't fail if logging fails
        >>> succeeder = Succeeder(ActionNode(log_action))
    """

    def __init__(
        self,
        child: Optional[BehaviorNode] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the succeeder.

        Args:
            child: Node to wrap.
            name: Optional name for debugging.
        """
        super().__init__(child=child, name=name or "Succeeder")

    def _tick(self, context: TickContext) -> RunStatus:
        """Tick child and return SUCCESS unless running.

        Args:
            context: The tick context.

        Returns:
            SUCCESS or RUNNING.
        """
        if not self._child:
            return RunStatus.SUCCESS

        status = self._child.tick(context)

        if status == RunStatus.RUNNING:
            return RunStatus.RUNNING
        return RunStatus.SUCCESS


class Failer(Decorator):
    """Always returns FAILURE regardless of child result.

    Useful for testing or forcing failure paths.
    RUNNING is passed through to support async operations.
    """

    def __init__(
        self,
        child: Optional[BehaviorNode] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the failer.

        Args:
            child: Node to wrap.
            name: Optional name for debugging.
        """
        super().__init__(child=child, name=name or "Failer")

    def _tick(self, context: TickContext) -> RunStatus:
        """Tick child and return FAILURE unless running.

        Args:
            context: The tick context.

        Returns:
            FAILURE or RUNNING.
        """
        if not self._child:
            return RunStatus.FAILURE

        status = self._child.tick(context)

        if status == RunStatus.RUNNING:
            return RunStatus.RUNNING
        return RunStatus.FAILURE


class UntilFail(Decorator):
    """Repeats child until it fails.

    Returns RUNNING while child succeeds.
    Returns SUCCESS when child finally fails.

    Example:
        >>> # Keep processing items until queue empty
        >>> until_fail = UntilFail(ActionNode(process_item))
    """

    def __init__(
        self,
        child: Optional[BehaviorNode] = None,
        name: Optional[str] = None,
        max_iterations: int = 0,
    ) -> None:
        """Initialize the until-fail decorator.

        Args:
            child: Node to repeat.
            name: Optional name for debugging.
            max_iterations: Maximum iterations (0 = unlimited).
        """
        super().__init__(child=child, name=name or "UntilFail")
        self._max_iterations = max_iterations
        self._iteration_count = 0

    def _tick(self, context: TickContext) -> RunStatus:
        """Tick child repeatedly until failure.

        Args:
            context: The tick context.

        Returns:
            RUNNING while succeeding, SUCCESS when failed.
        """
        if not self._child:
            return RunStatus.SUCCESS

        status = self._child.tick(context)

        if status == RunStatus.FAILURE:
            self._iteration_count = 0
            return RunStatus.SUCCESS

        if status == RunStatus.RUNNING:
            return RunStatus.RUNNING

        # Child succeeded
        self._iteration_count += 1

        # Check max iterations
        if self._max_iterations > 0 and self._iteration_count >= self._max_iterations:
            self._iteration_count = 0
            return RunStatus.SUCCESS

        # Reset child and return RUNNING to continue loop
        self._child.reset()
        return RunStatus.RUNNING

    def reset(self) -> None:
        """Reset decorator and iteration count."""
        super().reset()
        self._iteration_count = 0


class UntilSuccess(Decorator):
    """Repeats child until it succeeds.

    Returns RUNNING while child fails.
    Returns SUCCESS when child finally succeeds.

    Example:
        >>> # Keep retrying connection until success
        >>> until_success = UntilSuccess(ActionNode(connect))
    """

    def __init__(
        self,
        child: Optional[BehaviorNode] = None,
        name: Optional[str] = None,
        max_iterations: int = 0,
    ) -> None:
        """Initialize the until-success decorator.

        Args:
            child: Node to repeat.
            name: Optional name for debugging.
            max_iterations: Maximum iterations (0 = unlimited).
        """
        super().__init__(child=child, name=name or "UntilSuccess")
        self._max_iterations = max_iterations
        self._iteration_count = 0

    def _tick(self, context: TickContext) -> RunStatus:
        """Tick child repeatedly until success.

        Args:
            context: The tick context.

        Returns:
            RUNNING while failing, SUCCESS when succeeded.
        """
        if not self._child:
            return RunStatus.SUCCESS

        status = self._child.tick(context)

        if status == RunStatus.SUCCESS:
            self._iteration_count = 0
            return RunStatus.SUCCESS

        if status == RunStatus.RUNNING:
            return RunStatus.RUNNING

        # Child failed
        self._iteration_count += 1

        # Check max iterations
        if self._max_iterations > 0 and self._iteration_count >= self._max_iterations:
            self._iteration_count = 0
            return RunStatus.FAILURE

        # Reset child and return RUNNING to continue loop
        self._child.reset()
        return RunStatus.RUNNING

    def reset(self) -> None:
        """Reset decorator and iteration count."""
        super().reset()
        self._iteration_count = 0


class Cooldown(Decorator):
    """Rate-limits child execution.

    After child completes (success or failure), prevents re-execution
    for a specified duration. During cooldown, returns FAILURE.

    Supports both tick-based and time-based cooldowns:
    - cooldown_ticks: Minimum ticks between executions
    - cooldown_ms: Minimum milliseconds between executions

    Example:
        >>> # Only allow notification every 5 seconds
        >>> cooldown = Cooldown(
        ...     ActionNode(notify),
        ...     cooldown_ms=5000,
        ... )
    """

    def __init__(
        self,
        child: Optional[BehaviorNode] = None,
        name: Optional[str] = None,
        cooldown_ticks: int = 0,
        cooldown_ms: float = 0.0,
    ) -> None:
        """Initialize the cooldown decorator.

        Args:
            child: Node to rate-limit.
            name: Optional name for debugging.
            cooldown_ticks: Minimum ticks between executions.
            cooldown_ms: Minimum milliseconds between executions.
        """
        super().__init__(child=child, name=name or "Cooldown")
        self._cooldown_ticks = cooldown_ticks
        self._cooldown_ms = cooldown_ms

        # Tracking state
        self._last_complete_tick: int = -1
        self._last_complete_time_ms: float = 0.0
        self._is_cooling_down: bool = False

    def _tick(self, context: TickContext) -> RunStatus:
        """Tick child if not in cooldown.

        Args:
            context: The tick context.

        Returns:
            Child status or FAILURE if cooling down.
        """
        if not self._child:
            return RunStatus.FAILURE

        current_tick = context.frame_id
        current_time_ms = time.perf_counter() * 1000

        # Check if still cooling down
        if self._is_cooling_down:
            # Check tick-based cooldown
            if self._cooldown_ticks > 0:
                ticks_elapsed = current_tick - self._last_complete_tick
                if ticks_elapsed < self._cooldown_ticks:
                    logger.debug(
                        f"{self._name}: Cooling down "
                        f"({ticks_elapsed}/{self._cooldown_ticks} ticks)"
                    )
                    return RunStatus.FAILURE

            # Check time-based cooldown
            if self._cooldown_ms > 0:
                time_elapsed = current_time_ms - self._last_complete_time_ms
                if time_elapsed < self._cooldown_ms:
                    logger.debug(
                        f"{self._name}: Cooling down "
                        f"({time_elapsed:.0f}/{self._cooldown_ms:.0f} ms)"
                    )
                    return RunStatus.FAILURE

            # Cooldown complete
            self._is_cooling_down = False

        # Execute child
        status = self._child.tick(context)

        # Start cooldown on completion (not RUNNING)
        if status != RunStatus.RUNNING:
            self._last_complete_tick = current_tick
            self._last_complete_time_ms = current_time_ms
            self._is_cooling_down = True
            logger.debug(f"{self._name}: Child completed, starting cooldown")

        return status

    def reset(self) -> None:
        """Reset decorator but preserve cooldown state."""
        super().reset()
        # Note: We don't reset cooldown state, as that would defeat its purpose

    def force_reset_cooldown(self) -> None:
        """Forcibly reset cooldown state."""
        self._is_cooling_down = False
        self._last_complete_tick = -1
        self._last_complete_time_ms = 0.0

    def debug_info(self) -> dict:
        """Include cooldown state in debug info."""
        info = super().debug_info()
        info["is_cooling_down"] = self._is_cooling_down
        info["cooldown_ticks"] = self._cooldown_ticks
        info["cooldown_ms"] = self._cooldown_ms
        return info


# Type alias for condition functions
ConditionFn = Callable[["RuleContext"], bool]


class Guard(Decorator):
    """Conditional gate that controls child execution.

    Evaluates a condition before ticking the child:
    - If condition is True: tick child and return its status
    - If condition is False: return FAILURE without ticking child

    The condition can be:
    - A callable taking RuleContext and returning bool
    - An expression string evaluated via ExpressionEvaluator

    Example:
        >>> # Only execute action if token usage is high
        >>> guard = Guard(
        ...     ActionNode(notify),
        ...     condition=lambda ctx: ctx.turn.token_usage > 0.8,
        ... )
    """

    def __init__(
        self,
        child: Optional[BehaviorNode] = None,
        name: Optional[str] = None,
        condition: Optional[ConditionFn] = None,
        expression: Optional[str] = None,
    ) -> None:
        """Initialize the guard decorator.

        Args:
            child: Node to conditionally execute.
            name: Optional name for debugging.
            condition: Callable condition (takes RuleContext, returns bool).
            expression: Expression string (evaluated via ExpressionEvaluator).

        Note: Either condition or expression should be provided, not both.
        If both provided, condition takes precedence.
        """
        super().__init__(child=child, name=name or "Guard")
        self._condition = condition
        self._expression = expression
        self._evaluator = None

    def _tick(self, context: TickContext) -> RunStatus:
        """Evaluate condition and tick child if True.

        Args:
            context: The tick context.

        Returns:
            Child status if condition True, FAILURE otherwise.
        """
        if not self._child:
            return RunStatus.FAILURE

        # Evaluate condition
        condition_result = self._evaluate_condition(context)

        if not condition_result:
            logger.debug(f"{self._name}: Condition failed, returning FAILURE")
            return RunStatus.FAILURE

        # Condition passed, tick child
        return self._child.tick(context)

    def _evaluate_condition(self, context: TickContext) -> bool:
        """Evaluate the guard condition.

        Args:
            context: The tick context.

        Returns:
            True if condition passes, False otherwise.
        """
        # Try callable condition first
        if self._condition:
            try:
                return self._condition(context.rule_context)
            except Exception as e:
                logger.warning(
                    f"{self._name}: Condition callable failed: {e}"
                )
                return False

        # Try expression string
        if self._expression:
            try:
                # Lazy import to avoid circular dependency
                if self._evaluator is None:
                    from ..expression import ExpressionEvaluator
                    self._evaluator = ExpressionEvaluator()

                return self._evaluator.evaluate(
                    self._expression,
                    context.rule_context,
                )
            except Exception as e:
                logger.warning(
                    f"{self._name}: Expression evaluation failed: {e}"
                )
                return False

        # No condition configured - always pass
        logger.warning(f"{self._name}: No condition configured, passing by default")
        return True

    def debug_info(self) -> dict:
        """Include condition info in debug."""
        info = super().debug_info()
        info["has_condition"] = self._condition is not None
        info["expression"] = self._expression
        return info


class Retry(Decorator):
    """Retries child on failure.

    If child fails, resets it and tries again up to max_attempts times.
    Returns FAILURE only after all attempts exhausted.

    Example:
        >>> # Retry action up to 3 times
        >>> retry = Retry(ActionNode(flaky_action), max_attempts=3)
    """

    def __init__(
        self,
        child: Optional[BehaviorNode] = None,
        name: Optional[str] = None,
        max_attempts: int = 3,
    ) -> None:
        """Initialize the retry decorator.

        Args:
            child: Node to retry.
            name: Optional name for debugging.
            max_attempts: Maximum retry attempts (default 3).
        """
        super().__init__(child=child, name=name or "Retry")
        self._max_attempts = max_attempts
        self._attempt_count = 0

    def _tick(self, context: TickContext) -> RunStatus:
        """Tick child, retrying on failure.

        Args:
            context: The tick context.

        Returns:
            SUCCESS if child succeeds, FAILURE after max attempts.
        """
        if not self._child:
            return RunStatus.FAILURE

        status = self._child.tick(context)

        if status == RunStatus.SUCCESS:
            self._attempt_count = 0
            return RunStatus.SUCCESS

        if status == RunStatus.RUNNING:
            return RunStatus.RUNNING

        # Child failed
        self._attempt_count += 1
        logger.debug(
            f"{self._name}: Attempt {self._attempt_count}/{self._max_attempts} failed"
        )

        if self._attempt_count >= self._max_attempts:
            self._attempt_count = 0
            return RunStatus.FAILURE

        # Reset child and return RUNNING to retry
        self._child.reset()
        return RunStatus.RUNNING

    def reset(self) -> None:
        """Reset decorator and attempt count."""
        super().reset()
        self._attempt_count = 0


class Timeout(Decorator):
    """Fails child if it runs too long.

    If child returns RUNNING for too many ticks or too long,
    returns FAILURE instead.

    Supports both tick-based and time-based timeouts.

    Example:
        >>> # Fail if action takes more than 10 ticks
        >>> timeout = Timeout(ActionNode(slow_action), timeout_ticks=10)
    """

    def __init__(
        self,
        child: Optional[BehaviorNode] = None,
        name: Optional[str] = None,
        timeout_ticks: int = 0,
        timeout_ms: float = 0.0,
    ) -> None:
        """Initialize the timeout decorator.

        Args:
            child: Node to time out.
            name: Optional name for debugging.
            timeout_ticks: Maximum ticks in RUNNING state.
            timeout_ms: Maximum milliseconds in RUNNING state.
        """
        super().__init__(child=child, name=name or "Timeout")
        self._timeout_ticks = timeout_ticks
        self._timeout_ms = timeout_ms

        # Tracking state
        self._running_start_tick: int = -1
        self._running_start_time_ms: float = 0.0
        self._is_running: bool = False

    def _tick(self, context: TickContext) -> RunStatus:
        """Tick child with timeout enforcement.

        Args:
            context: The tick context.

        Returns:
            Child status or FAILURE on timeout.
        """
        if not self._child:
            return RunStatus.FAILURE

        current_tick = context.frame_id
        current_time_ms = time.perf_counter() * 1000

        # Tick child
        status = self._child.tick(context)

        if status != RunStatus.RUNNING:
            # Reset tracking when child completes
            self._is_running = False
            return status

        # Child is running - check timeout
        if not self._is_running:
            # First RUNNING tick
            self._is_running = True
            self._running_start_tick = current_tick
            self._running_start_time_ms = current_time_ms
            return RunStatus.RUNNING

        # Check tick-based timeout
        if self._timeout_ticks > 0:
            ticks_running = current_tick - self._running_start_tick
            if ticks_running >= self._timeout_ticks:
                logger.warning(
                    f"{self._name}: Timeout after {ticks_running} ticks"
                )
                self._is_running = False
                self._child.reset()
                return RunStatus.FAILURE

        # Check time-based timeout
        if self._timeout_ms > 0:
            time_running = current_time_ms - self._running_start_time_ms
            if time_running >= self._timeout_ms:
                logger.warning(
                    f"{self._name}: Timeout after {time_running:.0f} ms"
                )
                self._is_running = False
                self._child.reset()
                return RunStatus.FAILURE

        return RunStatus.RUNNING

    def reset(self) -> None:
        """Reset decorator and timeout tracking."""
        super().reset()
        self._is_running = False
        self._running_start_tick = -1
        self._running_start_time_ms = 0.0


class Repeat(Decorator):
    """Repeats child a fixed number of times.

    Executes child N times, returning RUNNING until all complete.
    Returns SUCCESS after all iterations succeed.
    Returns FAILURE immediately if any iteration fails.

    Example:
        >>> # Execute action exactly 3 times
        >>> repeat = Repeat(ActionNode(action), count=3)
    """

    def __init__(
        self,
        child: Optional[BehaviorNode] = None,
        name: Optional[str] = None,
        count: int = 1,
    ) -> None:
        """Initialize the repeat decorator.

        Args:
            child: Node to repeat.
            name: Optional name for debugging.
            count: Number of iterations.
        """
        super().__init__(child=child, name=name or "Repeat")
        self._count = count
        self._current_iteration = 0

    def _tick(self, context: TickContext) -> RunStatus:
        """Tick child repeatedly.

        Args:
            context: The tick context.

        Returns:
            RUNNING until complete, then SUCCESS or FAILURE.
        """
        if not self._child:
            return RunStatus.SUCCESS

        if self._current_iteration >= self._count:
            self._current_iteration = 0
            return RunStatus.SUCCESS

        status = self._child.tick(context)

        if status == RunStatus.FAILURE:
            self._current_iteration = 0
            return RunStatus.FAILURE

        if status == RunStatus.RUNNING:
            return RunStatus.RUNNING

        # Child succeeded - next iteration
        self._current_iteration += 1
        self._child.reset()

        if self._current_iteration >= self._count:
            self._current_iteration = 0
            return RunStatus.SUCCESS

        return RunStatus.RUNNING

    def reset(self) -> None:
        """Reset decorator and iteration count."""
        super().reset()
        self._current_iteration = 0


__all__ = [
    "Inverter",
    "Succeeder",
    "Failer",
    "UntilFail",
    "UntilSuccess",
    "Cooldown",
    "Guard",
    "Retry",
    "Timeout",
    "Repeat",
    "ConditionFn",
]
