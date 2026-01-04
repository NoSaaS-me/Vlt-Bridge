"""Default DecisionTree implementation for Oracle agent turn control.

This module provides the standard termination logic per FR-004:
- Iteration limits
- Token budget tracking
- Timeout detection
- No-progress detection (5 minutes of continuous identical actions)
- Consecutive error tracking
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .registry import decision_tree

if TYPE_CHECKING:
    from src.models.agent_state import AgentState
    from src.models.settings import AgentConfig


@decision_tree("default")
class DefaultDecisionTree:
    """Default decision tree with standard termination logic.

    Implements termination conditions per FR-004:
    1. User cancellation (handled externally)
    2. Model finish_reason="stop" (no tool calls - handled externally)
    3. Max iterations reached
    4. Token budget exceeded
    5. Timeout exceeded
    6. No-progress (5 minutes of continuous identical actions)
    7. Error limit (3 consecutive errors)
    """

    def __init__(self, config: "AgentConfig"):
        """Initialize with agent configuration.

        Args:
            config: AgentConfig with limits and thresholds
        """
        self._config = config
        self._consecutive_errors = 0
        self._warning_emitted_iteration = False
        self._warning_emitted_token = False

    def should_continue(self, state: "AgentState") -> Tuple[bool, str]:
        """Determine if the agent loop should continue.

        Args:
            state: Current agent state

        Returns:
            Tuple of (should_continue, reason)
        """
        # Check termination conditions in priority order

        # Already terminated
        if state.is_terminal:
            return False, state.termination_reason or "terminated"

        # Max iterations
        if state.is_over_iteration_limit:
            return False, "max_iterations"

        # Token budget
        if state.is_over_token_limit:
            return False, "token_budget"

        # Timeout
        if state.is_timed_out:
            return False, "timeout"

        # No-progress detection (time-based)
        if self._detect_no_progress(state):
            return False, "no_progress"

        # Consecutive errors
        if self._consecutive_errors >= 3:
            return False, "error_limit"

        return True, ""

    def on_turn_start(self, state: "AgentState") -> "AgentState":
        """Hook called at the start of each turn.

        Checks warning thresholds and returns state with updated warnings.

        Args:
            state: Current agent state

        Returns:
            Updated state (unchanged since frozen)
        """
        # Warning checks happen here but actual emission happens in oracle_agent
        # Just return state unchanged - warning emission handled by oracle_agent
        return state

    def on_tool_result(self, state: "AgentState", result: dict) -> "AgentState":
        """Process tool result and update state.

        Tracks action history for no-progress detection and handles errors.

        Args:
            state: Current agent state
            result: Tool execution result dict

        Returns:
            Updated state with new action in history and loop timing
        """
        # Track errors
        if result.get("error") or result.get("is_error"):
            self._consecutive_errors += 1
        else:
            self._consecutive_errors = 0

        # Create action signature for no-progress detection
        tool_name = result.get("tool_name", "unknown")
        arguments = result.get("arguments", {})
        signature = self._action_signature(tool_name, arguments)

        # Update recent_actions (keep last 3)
        recent = state.recent_actions[-2:] + (signature,) if len(state.recent_actions) >= 2 else state.recent_actions + (signature,)

        # Update loop timing
        loop_start = state.loop_start_time
        if len(recent) >= 2 and recent[-1] == recent[-2]:
            # Same action as previous - start or continue loop timer
            if loop_start is None:
                loop_start = time.time()
        else:
            # Different action - reset loop timer
            loop_start = None

        # Return new state with updated recent_actions and loop timing
        return replace(state, recent_actions=recent, loop_start_time=loop_start)

    def get_config(self) -> "AgentConfig":
        """Return the agent configuration for this tree."""
        return self._config

    def get_warning_state(self, state: "AgentState") -> dict:
        """Get current warning state for system message emission.

        Args:
            state: Current agent state

        Returns:
            Dict with warning info if warnings should be emitted
        """
        warnings = {}

        # Iteration warning (emit once at threshold)
        if state.is_near_iteration_limit and not self._warning_emitted_iteration:
            self._warning_emitted_iteration = True
            warnings["iteration"] = {
                "type": "limit_warning",
                "current_value": state.turn,
                "limit_value": self._config.max_iterations,
                "percent": state.iteration_percent,
            }

        # Token warning (emit once at threshold)
        if state.is_near_token_limit and not self._warning_emitted_token:
            self._warning_emitted_token = True
            warnings["token"] = {
                "type": "limit_warning",
                "current_value": state.tokens_used,
                "limit_value": self._config.token_budget,
                "percent": state.token_percent,
            }

        return warnings

    def _detect_no_progress(self, state: "AgentState") -> bool:
        """Detect if the same action has been repeated for too long.

        Time-based detection: Only terminate after loop_detection_window_seconds
        of continuous identical actions. This is much less aggressive than
        count-based detection (3 consecutive) and better matches how frontier
        AI companies handle loop detection.

        Args:
            state: Current agent state (includes loop_start_time)

        Returns:
            True if no progress detected (loop running too long)
        """
        # Need at least 2 actions to detect a loop
        if len(state.recent_actions) < 2:
            return False

        # Check if we're currently in a loop
        if state.loop_start_time is None:
            return False

        # Only terminate after the configured window has passed
        window = self._config.loop_detection_window_seconds
        return state.loop_duration_seconds >= window

    def _action_signature(self, tool_name: str, arguments: dict) -> str:
        """Create comparable signature from tool call.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments dict

        Returns:
            String signature for comparison
        """
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"

    def check_within_turn_loops(
        self,
        tool_calls: List[Dict[str, any]],
    ) -> Optional[Dict[str, any]]:
        """Check for duplicate tool calls within a single turn.

        This detects when the model requests the same tool call multiple times
        in a single response, which may indicate a loop or confusion.

        Args:
            tool_calls: List of tool calls from the model, each with:
                - name: Tool name
                - arguments: Dict of arguments

        Returns:
            Dict with pattern info if duplicates found, None otherwise.
            Pattern info includes:
                - duplicate_count: Number of duplicate calls
                - total_calls: Total number of calls
                - duplicate_rate: Percentage of duplicates (0-100)
                - duplicated_tools: List of tool names that were duplicated
                - warning: Human-readable warning message
        """
        if not tool_calls or len(tool_calls) < 2:
            return None

        # Build signatures and track duplicates
        seen_signatures: Dict[str, str] = {}  # signature -> tool_name
        duplicates: List[str] = []

        for call in tool_calls:
            name = call.get("name", "unknown")
            arguments = call.get("arguments", {})
            signature = self._action_signature(name, arguments)

            if signature in seen_signatures:
                duplicates.append(name)
            else:
                seen_signatures[signature] = name

        if not duplicates:
            return None

        # Calculate duplicate rate
        duplicate_count = len(duplicates)
        total_calls = len(tool_calls)
        duplicate_rate = (duplicate_count / total_calls) * 100

        # Deduplicate the tool names for reporting
        duplicated_tools = list(set(duplicates))

        return {
            "duplicate_count": duplicate_count,
            "total_calls": total_calls,
            "duplicate_rate": duplicate_rate,
            "duplicated_tools": duplicated_tools,
            "warning": (
                f"Detected {duplicate_count} duplicate tool calls out of {total_calls} "
                f"({duplicate_rate:.0f}%): {', '.join(duplicated_tools)}"
            ),
        }
