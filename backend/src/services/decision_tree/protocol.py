"""DecisionTree Protocol for pluggable control flow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from src.models.agent_state import AgentState
    from src.models.settings import AgentConfig


@runtime_checkable
class DecisionTree(Protocol):
    """
    Protocol for pluggable decision trees in OracleAgent.

    Implementations can customize:
    - When to continue vs terminate
    - What warnings to emit
    - How to process tool results

    Use @decision_tree("name") decorator to register implementations.
    """

    def should_continue(self, state: "AgentState") -> Tuple[bool, str]:
        """
        Determine if the agent loop should continue.

        Args:
            state: Current agent state

        Returns:
            Tuple of (should_continue, reason)
            - If False, reason explains why termination occurred
            - If True, reason is empty or informational
        """
        ...

    def on_turn_start(self, state: "AgentState") -> "AgentState":
        """
        Hook called at the start of each turn.

        Use to:
        - Emit warnings (via yielding system chunks in agent)
        - Update state with turn-specific data

        Args:
            state: Current agent state

        Returns:
            Updated state (may be same instance if frozen)
        """
        ...

    def on_tool_result(self, state: "AgentState", result: dict) -> "AgentState":
        """
        Hook called after each tool execution.

        Use to:
        - Track action history for no-progress detection
        - Update token counts
        - Modify state based on tool output

        Args:
            state: Current agent state
            result: Tool execution result dict

        Returns:
            Updated state
        """
        ...

    def get_config(self) -> "AgentConfig":
        """Return the agent configuration for this tree."""
        ...
