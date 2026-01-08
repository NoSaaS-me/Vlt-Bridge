"""
BT Decorator Nodes - Nodes that modify the behavior of exactly one child.

Implements Timeout, Retry, Guard, Cooldown, Inverter, AlwaysSucceed, AlwaysFail
decorator nodes per contracts/nodes.yaml specification.

Tasks covered: 2.2.1-2.2.6 from tasks.md

Error codes:
- E6001: Timeout exceeded (from errors.yaml)

Part of the BT Universal Runtime (spec 019).
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING, Union

from ..state.base import NodeType, RunStatus
from ..state.contracts import NodeContract
from .base import BehaviorNode

if TYPE_CHECKING:
    from ..core.context import TickContext

logger = logging.getLogger(__name__)


class DecoratorNode(BehaviorNode):
    """Base class for nodes with exactly one child.

    Decorators modify the behavior of their single child node. Different
    decorator types implement different modifications:
    - Timeout: Fails child if it runs too long
    - Retry: Retries child on failure
    - Guard: Only ticks child if condition passes
    - Cooldown: Prevents child from running too frequently
    - Inverter: Inverts child result
    - AlwaysSucceed/AlwaysFail: Forces completion status

    From nodes.yaml:
    - child_count: "1"
    - node_type: DECORATOR

    Subclasses must implement _tick() to define their specific behavior.
    """

    def __init__(
        self,
        id: str,
        child: BehaviorNode,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a decorator node with one child.

        Args:
            id: Unique identifier within the tree.
            child: The single child node to decorate.
            name: Human-readable name (defaults to id).
            metadata: Arbitrary metadata for debugging.
        """
        super().__init__(id=id, name=name, metadata=metadata)
        self._add_child(child)

    @property
    def node_type(self) -> NodeType:
        """Decorators have exactly 1 child."""
        return NodeType.DECORATOR

    @property
    def child(self) -> BehaviorNode:
        """Get the decorated child node."""
        return self._children[0]


class Timeout(DecoratorNode):
    """Fails child if it runs too long.

    From nodes.yaml Timeout specification:
    - Track time since child started RUNNING
    - If timeout exceeded: reset child, return FAILURE
    - Write _timeout_triggered to blackboard

    Config:
        timeout_ms: Maximum RUNNING time in milliseconds

    Outputs:
        _timeout_triggered: bool - True if timeout was triggered

    Example:
        >>> timeout = Timeout("llm-timeout", llm_call, timeout_ms=30000)
        >>> status = timeout.tick(ctx)
        >>> # If llm_call is RUNNING for >30 seconds, returns FAILURE
    """

    def __init__(
        self,
        id: str,
        child: BehaviorNode,
        timeout_ms: int,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize timeout decorator.

        Args:
            id: Unique identifier.
            child: Node to decorate.
            timeout_ms: Maximum RUNNING time in milliseconds.
            name: Human-readable name.
            metadata: Debugging metadata.

        Raises:
            ValueError: If timeout_ms <= 0.
        """
        super().__init__(id=id, child=child, name=name, metadata=metadata)

        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {timeout_ms}")

        self._timeout_ms = timeout_ms
        self._child_running_since: Optional[datetime] = None

    @classmethod
    def contract(cls) -> NodeContract:
        """Timeout outputs timeout trigger flag."""
        return NodeContract(
            description="Fails child if it runs too long"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Tick child with timeout enforcement.

        Args:
            ctx: Execution context.

        Returns:
            - Child's status if completed in time
            - FAILURE if timeout exceeded (E6001 logged)
        """
        # Tick the child
        ctx.push_path(self.child.id)
        try:
            status = self.child.tick(ctx)
        finally:
            ctx.pop_path()

        if status == RunStatus.RUNNING:
            # Track when child started running
            if self._child_running_since is None:
                self._child_running_since = datetime.now(timezone.utc)

            # Check timeout
            elapsed_ms = (
                datetime.now(timezone.utc) - self._child_running_since
            ).total_seconds() * 1000

            if elapsed_ms >= self._timeout_ms:
                # Timeout exceeded
                logger.warning(
                    f"[E6001] Timeout '{self._id}': child '{self.child.id}' "
                    f"exceeded {self._timeout_ms}ms timeout (elapsed: {elapsed_ms:.0f}ms)"
                )

                # Reset child and mark timeout
                self.child.reset()
                self._child_running_since = None
                ctx.blackboard.set_internal("_timeout_triggered", True)

                return RunStatus.FAILURE
            else:
                # Still within timeout
                return RunStatus.RUNNING
        else:
            # Child completed (SUCCESS or FAILURE)
            self._child_running_since = None
            ctx.blackboard.set_internal("_timeout_triggered", False)
            return status

    def reset(self) -> None:
        """Reset timeout tracking."""
        super().reset()
        self._child_running_since = None


class Retry(DecoratorNode):
    """Retries child on failure.

    From nodes.yaml Retry specification:
    - On child FAILURE: increment retry count, reset child, try again
    - On max retries: return FAILURE
    - On child SUCCESS: return SUCCESS

    Config:
        max_retries: Maximum number of retry attempts
        backoff_ms: Delay between retries (optional)

    State:
        _retry_count: Current retry count (reset_to: 0)

    Example:
        >>> retry = Retry("retry-api", api_call, max_retries=3, backoff_ms=1000)
        >>> status = retry.tick(ctx)
        >>> # If api_call fails, retries up to 3 times with 1s delay
    """

    def __init__(
        self,
        id: str,
        child: BehaviorNode,
        max_retries: int,
        backoff_ms: int = 0,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize retry decorator.

        Args:
            id: Unique identifier.
            child: Node to decorate.
            max_retries: Maximum retry attempts.
            backoff_ms: Delay between retries in milliseconds.
            name: Human-readable name.
            metadata: Debugging metadata.

        Raises:
            ValueError: If max_retries < 0.
        """
        super().__init__(id=id, child=child, name=name, metadata=metadata)

        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")

        self._max_retries = max_retries
        self._backoff_ms = backoff_ms
        self._retry_count: int = 0
        self._last_retry_at: Optional[datetime] = None

    @classmethod
    def contract(cls) -> NodeContract:
        """Retry has no specific outputs."""
        return NodeContract(
            description="Retries child on failure"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Tick child with retry logic.

        Args:
            ctx: Execution context.

        Returns:
            - SUCCESS: Child succeeded
            - RUNNING: Child running or waiting for backoff
            - FAILURE: Max retries exhausted
        """
        # Check if we're in backoff period
        if self._last_retry_at is not None and self._backoff_ms > 0:
            elapsed_ms = (
                datetime.now(timezone.utc) - self._last_retry_at
            ).total_seconds() * 1000

            if elapsed_ms < self._backoff_ms:
                # Still in backoff period
                return RunStatus.RUNNING

        # Tick the child
        ctx.push_path(self.child.id)
        try:
            status = self.child.tick(ctx)
        finally:
            ctx.pop_path()

        if status == RunStatus.SUCCESS:
            # Child succeeded
            return RunStatus.SUCCESS

        elif status == RunStatus.FAILURE:
            # Child failed - check if we can retry
            if self._retry_count < self._max_retries:
                self._retry_count += 1
                self._last_retry_at = datetime.now(timezone.utc)

                logger.debug(
                    f"Retry '{self._id}': child failed, "
                    f"attempt {self._retry_count}/{self._max_retries}"
                )

                # Reset child for retry
                self.child.reset()

                # If no backoff, immediately return RUNNING to retry next tick
                if self._backoff_ms > 0:
                    return RunStatus.RUNNING
                else:
                    # No backoff - could retry immediately but return RUNNING
                    # to give other nodes a chance
                    return RunStatus.RUNNING
            else:
                # Max retries exhausted
                logger.warning(
                    f"Retry '{self._id}': max retries ({self._max_retries}) exhausted"
                )
                return RunStatus.FAILURE

        else:
            # RUNNING
            return RunStatus.RUNNING

    def reset(self) -> None:
        """Reset retry count."""
        super().reset()
        self._retry_count = 0
        self._last_retry_at = None


# Type for Guard conditions
GuardCondition = Union[str, Callable[["TickContext"], bool]]


class Guard(DecoratorNode):
    """Only ticks child if condition passes.

    From nodes.yaml Guard specification:
    - Evaluate condition (Lua expression or Python callable)
    - If true: tick child, return child status
    - If false: return FAILURE (child not ticked)

    Config:
        condition: Lua expression (str) or Python callable returning bool

    Example (Python callable):
        >>> guard = Guard("check-budget", llm_call, condition=lambda ctx: ctx.budget > 0)

    Example (Lua expression - requires Lua runtime):
        >>> guard = Guard("check-budget", llm_call, condition="bb.budget > 0")
    """

    def __init__(
        self,
        id: str,
        child: BehaviorNode,
        condition: GuardCondition,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize guard decorator.

        Args:
            id: Unique identifier.
            child: Node to decorate.
            condition: Guard condition - string (Lua) or callable.
            name: Human-readable name.
            metadata: Debugging metadata.
        """
        super().__init__(id=id, child=child, name=name, metadata=metadata)
        self._condition = condition

    @classmethod
    def contract(cls) -> NodeContract:
        """Guard has no specific outputs."""
        return NodeContract(
            description="Only ticks child if condition passes"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Evaluate condition and tick child if true.

        Args:
            ctx: Execution context.

        Returns:
            - Child's status if condition true
            - FAILURE if condition false (child not ticked)
        """
        # Evaluate condition
        condition_result = self._evaluate_condition(ctx)

        if not condition_result:
            # Condition false - don't tick child
            logger.debug(f"Guard '{self._id}': condition false, child not ticked")
            return RunStatus.FAILURE

        # Condition true - tick child
        ctx.push_path(self.child.id)
        try:
            return self.child.tick(ctx)
        finally:
            ctx.pop_path()

    def _evaluate_condition(self, ctx: "TickContext") -> bool:
        """Evaluate the guard condition.

        Args:
            ctx: Execution context.

        Returns:
            True if condition passes, False otherwise.
        """
        if callable(self._condition):
            # Python callable
            try:
                result = self._condition(ctx)
                return bool(result)
            except Exception as e:
                logger.error(
                    f"Guard '{self._id}': condition evaluation failed: {e}"
                )
                return False

        elif isinstance(self._condition, str):
            # Lua expression - would need LuaSandbox integration
            # For now, try simple Python eval as fallback
            try:
                # Create evaluation context with blackboard access
                eval_context = {
                    "bb": ctx.blackboard.snapshot() if ctx.blackboard else {},
                    "ctx": ctx,
                }
                result = eval(self._condition, {"__builtins__": {}}, eval_context)
                return bool(result)
            except Exception as e:
                logger.error(
                    f"Guard '{self._id}': string condition evaluation failed: {e}"
                )
                return False

        else:
            logger.error(
                f"Guard '{self._id}': unsupported condition type: {type(self._condition)}"
            )
            return False


class Cooldown(DecoratorNode):
    """Prevents child from running too frequently.

    From nodes.yaml Cooldown specification:
    - Track last completion time
    - If cooldown elapsed: tick child
    - If cooldown not elapsed: return FAILURE

    Config:
        cooldown_ms: Minimum time between child executions in milliseconds

    State:
        _last_completion_at: When child last completed (reset_to: None)

    Example:
        >>> cooldown = Cooldown("rate-limit", api_call, cooldown_ms=5000)
        >>> # api_call can only run once every 5 seconds
    """

    def __init__(
        self,
        id: str,
        child: BehaviorNode,
        cooldown_ms: int,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize cooldown decorator.

        Args:
            id: Unique identifier.
            child: Node to decorate.
            cooldown_ms: Minimum time between executions in milliseconds.
            name: Human-readable name.
            metadata: Debugging metadata.

        Raises:
            ValueError: If cooldown_ms <= 0.
        """
        super().__init__(id=id, child=child, name=name, metadata=metadata)

        if cooldown_ms <= 0:
            raise ValueError(f"cooldown_ms must be positive, got {cooldown_ms}")

        self._cooldown_ms = cooldown_ms
        self._last_completion_at: Optional[datetime] = None

    @classmethod
    def contract(cls) -> NodeContract:
        """Cooldown has no specific outputs."""
        return NodeContract(
            description="Prevents child from running too frequently"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Tick child if cooldown has elapsed.

        Args:
            ctx: Execution context.

        Returns:
            - FAILURE if cooldown not elapsed
            - Child's status if cooldown elapsed
        """
        # Check cooldown
        if self._last_completion_at is not None:
            elapsed_ms = (
                datetime.now(timezone.utc) - self._last_completion_at
            ).total_seconds() * 1000

            if elapsed_ms < self._cooldown_ms:
                # Still in cooldown
                logger.debug(
                    f"Cooldown '{self._id}': {self._cooldown_ms - elapsed_ms:.0f}ms remaining"
                )
                return RunStatus.FAILURE

        # Cooldown elapsed (or first run) - tick child
        ctx.push_path(self.child.id)
        try:
            status = self.child.tick(ctx)
        finally:
            ctx.pop_path()

        # Track completion time if child finished
        if status.is_complete():
            self._last_completion_at = datetime.now(timezone.utc)

        return status

    def reset(self) -> None:
        """Reset cooldown timer."""
        super().reset()
        self._last_completion_at = None


class Inverter(DecoratorNode):
    """Inverts child result (SUCCESS <-> FAILURE).

    From nodes.yaml Inverter specification:
    - Tick child
    - SUCCESS becomes FAILURE, FAILURE becomes SUCCESS
    - RUNNING stays RUNNING

    Example:
        >>> inverter = Inverter("not-found", check_exists)
        >>> # Returns SUCCESS if check_exists returns FAILURE
    """

    @classmethod
    def contract(cls) -> NodeContract:
        """Inverter has no specific outputs."""
        return NodeContract(
            description="Inverts child result (SUCCESS <-> FAILURE)"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Tick child and invert result.

        Args:
            ctx: Execution context.

        Returns:
            - FAILURE if child returns SUCCESS
            - SUCCESS if child returns FAILURE
            - RUNNING if child returns RUNNING
        """
        ctx.push_path(self.child.id)
        try:
            status = self.child.tick(ctx)
        finally:
            ctx.pop_path()

        if status == RunStatus.SUCCESS:
            return RunStatus.FAILURE
        elif status == RunStatus.FAILURE:
            return RunStatus.SUCCESS
        else:
            # RUNNING or FRESH stays as-is
            return status


class AlwaysSucceed(DecoratorNode):
    """Converts any completion to SUCCESS.

    From nodes.yaml AlwaysSucceed specification:
    - Tick child
    - Both SUCCESS and FAILURE become SUCCESS
    - RUNNING stays RUNNING

    Useful for optional operations that shouldn't fail the parent.

    Example:
        >>> always_succeed = AlwaysSucceed("optional-log", log_action)
        >>> # Even if log_action fails, returns SUCCESS
    """

    @classmethod
    def contract(cls) -> NodeContract:
        """AlwaysSucceed has no specific outputs."""
        return NodeContract(
            description="Converts any completion to SUCCESS"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Tick child and convert completion to SUCCESS.

        Args:
            ctx: Execution context.

        Returns:
            - SUCCESS if child returns SUCCESS or FAILURE
            - RUNNING if child returns RUNNING
        """
        ctx.push_path(self.child.id)
        try:
            status = self.child.tick(ctx)
        finally:
            ctx.pop_path()

        if status.is_complete():
            return RunStatus.SUCCESS
        else:
            return status


class AlwaysFail(DecoratorNode):
    """Converts any completion to FAILURE.

    From nodes.yaml AlwaysFail specification:
    - Tick child
    - Both SUCCESS and FAILURE become FAILURE
    - RUNNING stays RUNNING

    Useful for testing or forcing failure conditions.

    Example:
        >>> always_fail = AlwaysFail("force-retry", action)
        >>> # Even if action succeeds, returns FAILURE
    """

    @classmethod
    def contract(cls) -> NodeContract:
        """AlwaysFail has no specific outputs."""
        return NodeContract(
            description="Converts any completion to FAILURE"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Tick child and convert completion to FAILURE.

        Args:
            ctx: Execution context.

        Returns:
            - FAILURE if child returns SUCCESS or FAILURE
            - RUNNING if child returns RUNNING
        """
        ctx.push_path(self.child.id)
        try:
            status = self.child.tick(ctx)
        finally:
            ctx.pop_path()

        if status.is_complete():
            return RunStatus.FAILURE
        else:
            return status


__all__ = [
    "DecoratorNode",
    "Timeout",
    "Retry",
    "Guard",
    "GuardCondition",
    "Cooldown",
    "Inverter",
    "AlwaysSucceed",
    "AlwaysFail",
]
