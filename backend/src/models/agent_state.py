"""Agent state for Oracle turn control."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Any

from ..models.settings import AgentConfig


@dataclass(frozen=True, kw_only=True)
class AgentState:
    """Immutable state during query execution.

    This dataclass tracks the execution state of an Oracle agent query,
    including iteration counts, token usage, and timing information.
    The frozen=True ensures immutability for safe state transitions.
    """
    user_id: str
    project_id: str
    config: AgentConfig

    # Tracking
    turn: int = 0
    tokens_used: int = 0
    start_time: float = field(default_factory=time.time)
    recent_actions: tuple[str, ...] = field(default_factory=tuple)

    # Loop detection timing
    loop_start_time: Optional[float] = None

    # Termination
    termination_reason: Optional[str] = None

    # Extensions for future modules (DeepResearcher, etc.)
    extensions: dict[str, Any] = field(default_factory=dict)

    # Derived properties
    @property
    def is_terminal(self) -> bool:
        """Return True if agent has terminated."""
        return self.termination_reason is not None

    @property
    def elapsed_seconds(self) -> float:
        """Return seconds elapsed since query started."""
        return time.time() - self.start_time

    @property
    def iteration_percent(self) -> float:
        """Return percentage of max iterations used."""
        return (self.turn / self.config.max_iterations) * 100

    @property
    def token_percent(self) -> float:
        """Return percentage of token budget used."""
        return (self.tokens_used / self.config.token_budget) * 100

    @property
    def is_near_iteration_limit(self) -> bool:
        """Return True if approaching iteration limit (soft warning threshold)."""
        return self.iteration_percent >= self.config.soft_warning_percent

    @property
    def is_near_token_limit(self) -> bool:
        """Return True if approaching token budget limit (warning threshold)."""
        return self.token_percent >= self.config.token_warning_percent

    @property
    def is_over_iteration_limit(self) -> bool:
        """Return True if at or over the max iteration limit."""
        return self.turn >= self.config.max_iterations

    @property
    def is_over_token_limit(self) -> bool:
        """Return True if at or over the token budget."""
        return self.tokens_used >= self.config.token_budget

    @property
    def is_timed_out(self) -> bool:
        """Return True if query has exceeded timeout."""
        return self.elapsed_seconds >= self.config.timeout_seconds

    @property
    def loop_duration_seconds(self) -> float:
        """Return how long the current loop has been running, or 0 if not in loop."""
        if self.loop_start_time is None:
            return 0.0
        return time.time() - self.loop_start_time
