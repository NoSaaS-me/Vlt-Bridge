"""Leaf nodes that perform actual work in the behavior tree.

This module implements terminal nodes (no children) that:
- Evaluate conditions (ConditionNode)
- Execute actions (ActionNode)
- Wait for time (WaitNode)
- Run Lua scripts (ScriptNode)
- Provide constant values (SuccessNode, FailureNode)

Leaf nodes interact with RuleContext to make decisions or trigger effects.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, TYPE_CHECKING

from .types import RunStatus, TickContext
from .node import Leaf

if TYPE_CHECKING:
    from ..context import RuleContext
    from ..rule import RuleAction


logger = logging.getLogger(__name__)


class SuccessNode(Leaf):
    """Always returns SUCCESS.

    Useful as a placeholder or for testing.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the success node."""
        super().__init__(name=name or "SuccessNode")

    def _tick(self, context: TickContext) -> RunStatus:
        """Always return SUCCESS."""
        return RunStatus.SUCCESS


class FailureNode(Leaf):
    """Always returns FAILURE.

    Useful as a placeholder or for testing.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the failure node."""
        super().__init__(name=name or "FailureNode")

    def _tick(self, context: TickContext) -> RunStatus:
        """Always return FAILURE."""
        return RunStatus.FAILURE


class RunningNode(Leaf):
    """Always returns RUNNING.

    Useful for testing frame locking behavior.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the running node."""
        super().__init__(name=name or "RunningNode")

    def _tick(self, context: TickContext) -> RunStatus:
        """Always return RUNNING."""
        return RunStatus.RUNNING


# Type alias for condition callables
ConditionCallable = Callable[["RuleContext"], bool]


class ConditionNode(Leaf):
    """Evaluates a condition and returns SUCCESS/FAILURE.

    Conditions can be specified as:
    - A callable taking RuleContext and returning bool
    - An expression string (evaluated via ExpressionEvaluator)

    Returns SUCCESS if condition is True, FAILURE if False.
    Never returns RUNNING (conditions are synchronous).

    Example:
        >>> # Using callable
        >>> node = ConditionNode(
        ...     condition=lambda ctx: ctx.turn.token_usage > 0.8
        ... )

        >>> # Using expression
        >>> node = ConditionNode(
        ...     expression="context.turn.token_usage > 0.8"
        ... )
    """

    def __init__(
        self,
        name: Optional[str] = None,
        condition: Optional[ConditionCallable] = None,
        expression: Optional[str] = None,
    ) -> None:
        """Initialize the condition node.

        Args:
            name: Optional name for debugging.
            condition: Callable condition (RuleContext -> bool).
            expression: Expression string for ExpressionEvaluator.

        Note: Either condition or expression should be provided.
        """
        super().__init__(name=name or "ConditionNode")
        self._condition = condition
        self._expression = expression
        self._evaluator = None

    @property
    def expression(self) -> Optional[str]:
        """Get the expression string."""
        return self._expression

    def _tick(self, context: TickContext) -> RunStatus:
        """Evaluate condition and return result.

        Args:
            context: The tick context.

        Returns:
            SUCCESS if condition True, FAILURE otherwise.
        """
        result = self._evaluate(context)
        return RunStatus.from_bool(result)

    def _evaluate(self, context: TickContext) -> bool:
        """Evaluate the condition.

        Args:
            context: The tick context.

        Returns:
            Boolean result of condition.
        """
        # Try callable first
        if self._condition:
            try:
                return self._condition(context.rule_context)
            except Exception as e:
                logger.warning(f"{self._name}: Condition error: {e}")
                return False

        # Try expression
        if self._expression:
            try:
                if self._evaluator is None:
                    from ..expression import ExpressionEvaluator
                    self._evaluator = ExpressionEvaluator()

                return self._evaluator.evaluate(
                    self._expression,
                    context.rule_context,
                )
            except Exception as e:
                logger.warning(f"{self._name}: Expression error: {e}")
                return False

        # No condition - default to True
        logger.debug(f"{self._name}: No condition, returning True")
        return True

    def debug_info(self) -> dict:
        """Include condition info in debug."""
        info = super().debug_info()
        info["has_condition"] = self._condition is not None
        info["expression"] = self._expression
        return info


# Type alias for action callables
ActionCallable = Callable[["RuleContext"], bool]


class ActionNode(Leaf):
    """Executes an action and returns SUCCESS/FAILURE.

    Actions can be specified as:
    - A callable taking RuleContext and returning bool (success/failure)
    - A RuleAction object (dispatched via ActionDispatcher)

    Returns SUCCESS if action succeeds, FAILURE if it fails.
    For async actions, returns RUNNING until complete.

    Example:
        >>> # Using callable
        >>> node = ActionNode(
        ...     action=lambda ctx: do_something(ctx)
        ... )

        >>> # Using RuleAction
        >>> node = ActionNode(
        ...     rule_action=RuleAction(type=ActionType.LOG, message="Hello")
        ... )
    """

    def __init__(
        self,
        name: Optional[str] = None,
        action: Optional[ActionCallable] = None,
        rule_action: Optional["RuleAction"] = None,
        dispatcher: Optional[Any] = None,
    ) -> None:
        """Initialize the action node.

        Args:
            name: Optional name for debugging.
            action: Callable action (RuleContext -> bool).
            rule_action: RuleAction to dispatch.
            dispatcher: ActionDispatcher for rule_action execution.
        """
        super().__init__(name=name or "ActionNode")
        self._action = action
        self._rule_action = rule_action
        self._dispatcher = dispatcher

    def set_dispatcher(self, dispatcher: Any) -> None:
        """Set the action dispatcher for rule actions.

        Args:
            dispatcher: ActionDispatcher instance.
        """
        self._dispatcher = dispatcher

    def _tick(self, context: TickContext) -> RunStatus:
        """Execute action and return result.

        Args:
            context: The tick context.

        Returns:
            SUCCESS if action succeeds, FAILURE otherwise.
        """
        success = self._execute(context)
        return RunStatus.from_bool(success)

    def _execute(self, context: TickContext) -> bool:
        """Execute the action.

        Args:
            context: The tick context.

        Returns:
            True if action succeeded.
        """
        # Try callable first
        if self._action:
            try:
                return self._action(context.rule_context)
            except Exception as e:
                logger.warning(f"{self._name}: Action error: {e}")
                return False

        # Try rule action
        if self._rule_action:
            if self._dispatcher is None:
                logger.warning(f"{self._name}: No dispatcher for rule_action")
                return False

            try:
                return self._dispatcher.dispatch(
                    self._rule_action,
                    context.rule_context,
                )
            except Exception as e:
                logger.warning(f"{self._name}: Dispatch error: {e}")
                return False

        # No action - treat as success
        logger.debug(f"{self._name}: No action, returning True")
        return True

    def debug_info(self) -> dict:
        """Include action info in debug."""
        info = super().debug_info()
        info["has_action"] = self._action is not None
        info["has_rule_action"] = self._rule_action is not None
        return info


class WaitNode(Leaf):
    """Waits for a specified number of ticks or duration.

    Returns RUNNING until wait completes, then SUCCESS.
    Useful for introducing delays or throttling.

    Example:
        >>> # Wait 5 ticks
        >>> node = WaitNode(wait_ticks=5)

        >>> # Wait 1 second
        >>> node = WaitNode(wait_ms=1000)
    """

    def __init__(
        self,
        name: Optional[str] = None,
        wait_ticks: int = 0,
        wait_ms: float = 0.0,
    ) -> None:
        """Initialize the wait node.

        Args:
            name: Optional name for debugging.
            wait_ticks: Number of ticks to wait.
            wait_ms: Milliseconds to wait.
        """
        super().__init__(name=name or "WaitNode")
        self._wait_ticks = wait_ticks
        self._wait_ms = wait_ms

        # State tracking
        self._start_tick: int = -1
        self._start_time_ms: float = 0.0
        self._waiting: bool = False

    def _tick(self, context: TickContext) -> RunStatus:
        """Check if wait is complete.

        Args:
            context: The tick context.

        Returns:
            RUNNING while waiting, SUCCESS when done.
        """
        current_tick = context.frame_id
        current_time_ms = time.perf_counter() * 1000

        # Start waiting
        if not self._waiting:
            self._waiting = True
            self._start_tick = current_tick
            self._start_time_ms = current_time_ms

            # Immediate completion if no wait
            if self._wait_ticks <= 0 and self._wait_ms <= 0:
                self._waiting = False
                return RunStatus.SUCCESS

            return RunStatus.RUNNING

        # Check tick-based wait
        if self._wait_ticks > 0:
            ticks_elapsed = current_tick - self._start_tick
            if ticks_elapsed < self._wait_ticks:
                return RunStatus.RUNNING

        # Check time-based wait
        if self._wait_ms > 0:
            time_elapsed = current_time_ms - self._start_time_ms
            if time_elapsed < self._wait_ms:
                return RunStatus.RUNNING

        # Wait complete
        self._waiting = False
        return RunStatus.SUCCESS

    def reset(self) -> None:
        """Reset wait state."""
        super().reset()
        self._waiting = False
        self._start_tick = -1
        self._start_time_ms = 0.0

    def debug_info(self) -> dict:
        """Include wait state in debug."""
        info = super().debug_info()
        info["wait_ticks"] = self._wait_ticks
        info["wait_ms"] = self._wait_ms
        info["waiting"] = self._waiting
        return info


class ScriptNode(Leaf):
    """Executes a Lua script and maps result to RunStatus.

    Scripts run in the LuaSandbox with access to the RuleContext.
    Script return values are interpreted as:
    - nil: FAILURE
    - false: FAILURE
    - true: SUCCESS
    - dict/table: SUCCESS (table can contain action details)

    Example:
        >>> node = ScriptNode(script='''
        ...     if context.turn.token_usage > 0.8 then
        ...         return true
        ...     end
        ...     return false
        ... ''')
    """

    def __init__(
        self,
        name: Optional[str] = None,
        script: Optional[str] = None,
        script_path: Optional[str] = None,
        sandbox: Optional[Any] = None,
    ) -> None:
        """Initialize the script node.

        Args:
            name: Optional name for debugging.
            script: Inline Lua script code.
            script_path: Path to Lua script file.
            sandbox: LuaSandbox instance for execution.
        """
        super().__init__(name=name or "ScriptNode")
        self._script = script
        self._script_path = script_path
        self._sandbox = sandbox
        self._script_content: Optional[str] = None

    def set_sandbox(self, sandbox: Any) -> None:
        """Set the Lua sandbox for script execution.

        Args:
            sandbox: LuaSandbox instance.
        """
        self._sandbox = sandbox

    def _tick(self, context: TickContext) -> RunStatus:
        """Execute script and return result.

        Args:
            context: The tick context.

        Returns:
            SUCCESS/FAILURE based on script result.
        """
        if self._sandbox is None:
            logger.warning(f"{self._name}: No sandbox configured")
            return RunStatus.FAILURE

        script = self._get_script()
        if not script:
            logger.warning(f"{self._name}: No script to execute")
            return RunStatus.FAILURE

        try:
            result = self._sandbox.execute(script, context.rule_context)
            return self._map_result(result)

        except Exception as e:
            logger.warning(f"{self._name}: Script execution error: {e}")
            return RunStatus.FAILURE

    def _get_script(self) -> Optional[str]:
        """Get the script content.

        Returns:
            Script content string or None.
        """
        # Return inline script
        if self._script:
            return self._script

        # Load from file (cached)
        if self._script_path:
            if self._script_content is None:
                try:
                    with open(self._script_path, "r", encoding="utf-8") as f:
                        self._script_content = f.read()
                except Exception as e:
                    logger.error(f"{self._name}: Failed to load script: {e}")
                    return None
            return self._script_content

        return None

    def _map_result(self, result: Any) -> RunStatus:
        """Map script result to RunStatus.

        Args:
            result: Script return value.

        Returns:
            Corresponding RunStatus.
        """
        if result is None:
            return RunStatus.FAILURE

        if isinstance(result, bool):
            return RunStatus.from_bool(result)

        if isinstance(result, dict):
            # Non-empty dict = success
            return RunStatus.SUCCESS

        # Any other truthy value = success
        return RunStatus.from_bool(bool(result))

    def debug_info(self) -> dict:
        """Include script info in debug."""
        info = super().debug_info()
        info["has_script"] = self._script is not None
        info["script_path"] = self._script_path
        return info


class BlackboardCondition(Leaf):
    """Checks a condition on the blackboard.

    Reads a key from the blackboard and evaluates a condition.
    Useful for cross-node communication.

    Example:
        >>> node = BlackboardCondition(
        ...     key="target_found",
        ...     expected=True,
        ... )
    """

    def __init__(
        self,
        name: Optional[str] = None,
        key: str = "",
        expected: Any = True,
        comparison: str = "eq",
    ) -> None:
        """Initialize the blackboard condition.

        Args:
            name: Optional name for debugging.
            key: Blackboard key to check.
            expected: Expected value.
            comparison: Comparison type ("eq", "ne", "gt", "lt", "ge", "le").
        """
        super().__init__(name=name or "BlackboardCondition")
        self._key = key
        self._expected = expected
        self._comparison = comparison

    def _tick(self, context: TickContext) -> RunStatus:
        """Check blackboard condition.

        Args:
            context: The tick context.

        Returns:
            SUCCESS if condition met, FAILURE otherwise.
        """
        if context.blackboard is None:
            logger.warning(f"{self._name}: No blackboard in context")
            return RunStatus.FAILURE

        if not self._key:
            logger.warning(f"{self._name}: No key specified")
            return RunStatus.FAILURE

        value = context.blackboard.get(self._key)
        result = self._compare(value, self._expected)
        return RunStatus.from_bool(result)

    def _compare(self, value: Any, expected: Any) -> bool:
        """Compare value against expected.

        Args:
            value: Actual value from blackboard.
            expected: Expected value.

        Returns:
            Comparison result.
        """
        try:
            if self._comparison == "eq":
                return value == expected
            elif self._comparison == "ne":
                return value != expected
            elif self._comparison == "gt":
                return value > expected
            elif self._comparison == "lt":
                return value < expected
            elif self._comparison == "ge":
                return value >= expected
            elif self._comparison == "le":
                return value <= expected
            else:
                logger.warning(f"{self._name}: Unknown comparison: {self._comparison}")
                return False
        except (TypeError, ValueError):
            return False


class BlackboardSet(Leaf):
    """Sets a value in the blackboard.

    Always returns SUCCESS after setting the value.

    Example:
        >>> node = BlackboardSet(key="target_found", value=True)
    """

    def __init__(
        self,
        name: Optional[str] = None,
        key: str = "",
        value: Any = None,
    ) -> None:
        """Initialize the blackboard set node.

        Args:
            name: Optional name for debugging.
            key: Blackboard key to set.
            value: Value to set.
        """
        super().__init__(name=name or "BlackboardSet")
        self._key = key
        self._value = value

    def _tick(self, context: TickContext) -> RunStatus:
        """Set blackboard value.

        Args:
            context: The tick context.

        Returns:
            SUCCESS after setting.
        """
        if context.blackboard is None:
            logger.warning(f"{self._name}: No blackboard in context")
            return RunStatus.FAILURE

        if not self._key:
            logger.warning(f"{self._name}: No key specified")
            return RunStatus.FAILURE

        context.blackboard.set(self._key, self._value)
        return RunStatus.SUCCESS


class LogNode(Leaf):
    """Logs a message and returns SUCCESS.

    Useful for debugging tree execution.

    Example:
        >>> node = LogNode(message="Reached decision point", level="debug")
    """

    def __init__(
        self,
        name: Optional[str] = None,
        message: str = "",
        level: str = "debug",
    ) -> None:
        """Initialize the log node.

        Args:
            name: Optional name for debugging.
            message: Message to log.
            level: Log level (debug, info, warning, error).
        """
        super().__init__(name=name or "LogNode")
        self._message = message
        self._level = level.lower()

    def _tick(self, context: TickContext) -> RunStatus:
        """Log message and return SUCCESS.

        Args:
            context: The tick context.

        Returns:
            SUCCESS.
        """
        msg = f"[BT] {self._name}: {self._message}"

        if self._level == "debug":
            logger.debug(msg)
        elif self._level == "info":
            logger.info(msg)
        elif self._level == "warning":
            logger.warning(msg)
        elif self._level == "error":
            logger.error(msg)
        else:
            logger.debug(msg)

        return RunStatus.SUCCESS


__all__ = [
    "SuccessNode",
    "FailureNode",
    "RunningNode",
    "ConditionNode",
    "ActionNode",
    "WaitNode",
    "ScriptNode",
    "BlackboardCondition",
    "BlackboardSet",
    "LogNode",
    "ConditionCallable",
    "ActionCallable",
]
