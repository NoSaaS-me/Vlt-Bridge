"""Pydantic models for Oracle agent signals.

Signals enable agent self-reflection and BT control flow decisions.
The agent emits XML signals that are parsed into these typed structures.

Part of feature 020-bt-oracle-agent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class SignalType(str, Enum):
    """Signal type enumeration.

    Signals communicate agent internal state to the BT runtime.
    """

    NEED_TURN = "need_turn"
    """Agent needs more iterations to complete the task."""

    CONTEXT_SUFFICIENT = "context_sufficient"
    """Agent has gathered enough information to answer."""

    STUCK = "stuck"
    """Agent cannot proceed without external help."""

    NEED_CAPABILITY = "need_capability"
    """Agent needs a tool or capability that is unavailable."""

    PARTIAL_ANSWER = "partial_answer"
    """Agent can provide an answer but with known limitations."""

    DELEGATION_RECOMMENDED = "delegation_recommended"
    """Task would benefit from delegation to a subagent."""


# =============================================================================
# Type-Specific Field Models
# =============================================================================


class NeedTurnFields(BaseModel):
    """Fields for need_turn signal.

    Emitted when agent needs to continue working.
    """

    reason: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Why another turn is needed (be specific)",
    )
    expected_turns: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Estimated additional turns needed",
    )


class ContextSufficientFields(BaseModel):
    """Fields for context_sufficient signal.

    Emitted when agent has gathered enough information.
    """

    sources_found: int = Field(
        ...,
        ge=0,
        description="Number of relevant sources gathered",
    )
    source_types: Optional[List[str]] = Field(
        None,
        description="Types of sources found (code, docs, web)",
    )

    @field_validator("source_types")
    @classmethod
    def validate_source_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate source types are from known set."""
        if v is None:
            return v
        valid_types = {"code", "docs", "web", "thread", "vault", "repomap"}
        for source_type in v:
            if source_type.lower() not in valid_types:
                # Warn but don't reject - allow unknown types
                pass
        return v


class StuckFields(BaseModel):
    """Fields for stuck signal.

    Emitted when agent cannot proceed.
    """

    attempted: List[str] = Field(
        ...,
        min_length=1,
        description="Tools/approaches already tried",
    )
    blocker: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="What prevents progress",
    )
    suggestions: Optional[List[str]] = Field(
        None,
        description="What might help if available",
    )


class NeedCapabilityFields(BaseModel):
    """Fields for need_capability signal.

    Emitted when agent needs an unavailable tool.
    """

    capability: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="What capability is needed",
    )
    reason: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Why it's needed for this task",
    )
    workaround: Optional[str] = Field(
        None,
        max_length=500,
        description="Partial solution without capability",
    )


class PartialAnswerFields(BaseModel):
    """Fields for partial_answer signal.

    Emitted when answering with known limitations.
    """

    missing: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="What information is missing",
    )
    caveat: Optional[str] = Field(
        None,
        max_length=500,
        description="Important caveat for user",
    )


class DelegationRecommendedFields(BaseModel):
    """Fields for delegation_recommended signal.

    Emitted when task would benefit from subagent.
    """

    reason: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Why delegation is recommended",
    )
    scope: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="What should be delegated",
    )
    estimated_tokens: Optional[int] = Field(
        None,
        ge=100,
        le=100000,
        description="Estimated token budget needed",
    )
    subagent_type: Optional[str] = Field(
        None,
        max_length=50,
        description="Recommended subagent type (research, librarian, etc.)",
    )


# Type alias for signal fields union
SignalFields = Union[
    NeedTurnFields,
    ContextSufficientFields,
    StuckFields,
    NeedCapabilityFields,
    PartialAnswerFields,
    DelegationRecommendedFields,
]

# Mapping of signal type to expected fields model
SIGNAL_FIELDS_MAP: Dict[SignalType, type] = {
    SignalType.NEED_TURN: NeedTurnFields,
    SignalType.CONTEXT_SUFFICIENT: ContextSufficientFields,
    SignalType.STUCK: StuckFields,
    SignalType.NEED_CAPABILITY: NeedCapabilityFields,
    SignalType.PARTIAL_ANSWER: PartialAnswerFields,
    SignalType.DELEGATION_RECOMMENDED: DelegationRecommendedFields,
}


# =============================================================================
# Main Signal Model
# =============================================================================


class Signal(BaseModel):
    """Structured agent self-reflection extracted from LLM response.

    Signals are immutable once parsed. They are stored in:
    - Blackboard (runtime) - for BT condition evaluation
    - ANS Events (events table) - for audit/debugging
    - Exchange metadata (context_nodes) - for session continuity

    Example XML that produces this model:
        <signal type="need_turn">
          <reason>Found backup API, need to test response</reason>
          <confidence>0.85</confidence>
          <expected_turns>1</expected_turns>
        </signal>
    """

    type: SignalType = Field(
        ...,
        description="Signal category",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent confidence in signal (0.0-1.0)",
    )
    fields: Dict[str, Any] = Field(
        ...,
        description="Type-specific fields",
    )
    raw_xml: str = Field(
        ...,
        min_length=10,
        description="Original XML for debugging",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When signal was parsed",
    )

    @model_validator(mode="after")
    def validate_fields_for_type(self) -> "Signal":
        """Validate that fields match the expected schema for signal type."""
        expected_model = SIGNAL_FIELDS_MAP.get(self.type)
        if expected_model is None:
            raise ValueError(f"Unknown signal type: {self.type}")

        try:
            # Validate fields against expected model
            expected_model.model_validate(self.fields)
        except Exception as e:
            raise ValueError(
                f"Invalid fields for signal type {self.type}: {e}"
            ) from e

        return self

    @property
    def typed_fields(self) -> SignalFields:
        """Get fields as strongly-typed model instance."""
        expected_model = SIGNAL_FIELDS_MAP[self.type]
        return expected_model.model_validate(self.fields)

    def is_terminal(self) -> bool:
        """Check if this signal indicates task completion.

        Terminal signals mean agent is done (successfully or not).
        Non-terminal signals mean agent wants to continue.
        """
        return self.type in {
            SignalType.CONTEXT_SUFFICIENT,
            SignalType.STUCK,
            SignalType.PARTIAL_ANSWER,
        }

    def is_continuation(self) -> bool:
        """Check if this signal requests continuation."""
        return self.type in {
            SignalType.NEED_TURN,
            SignalType.NEED_CAPABILITY,
            SignalType.DELEGATION_RECOMMENDED,
        }


# =============================================================================
# Agent Signal State (for blackboard tracking)
# =============================================================================


class AgentSignalState(BaseModel):
    """Tracking of signals across agent loop iterations.

    Stored in blackboard for BT condition evaluation.
    """

    signals_emitted: List[Signal] = Field(
        default_factory=list,
        description="All signals emitted this session",
    )
    last_signal: Optional[Signal] = Field(
        None,
        description="Most recent signal",
    )
    consecutive_same_reason: int = Field(
        0,
        ge=0,
        description="Count of consecutive need_turn signals with same reason",
    )
    turns_without_signal: int = Field(
        0,
        ge=0,
        description="Turns with no signal emitted",
    )

    def record_signal(self, signal: Signal) -> None:
        """Record a new signal and update tracking state.

        This method mutates the state - call after parsing a signal.
        """
        self.signals_emitted.append(signal)

        # Track consecutive same reason for loop detection
        if signal.type == SignalType.NEED_TURN:
            if self.last_signal and self.last_signal.type == SignalType.NEED_TURN:
                # Compare reasons
                old_reason = self.last_signal.fields.get("reason", "")
                new_reason = signal.fields.get("reason", "")
                if old_reason == new_reason:
                    self.consecutive_same_reason += 1
                else:
                    self.consecutive_same_reason = 1
            else:
                self.consecutive_same_reason = 1
        else:
            self.consecutive_same_reason = 0

        self.last_signal = signal
        self.turns_without_signal = 0

    def record_turn_without_signal(self) -> None:
        """Record that a turn completed without signal emission."""
        self.turns_without_signal += 1
        self.consecutive_same_reason = 0

    def is_stuck_loop(self, threshold: int = 3) -> bool:
        """Check if agent is in a stuck loop.

        Returns True if agent has emitted the same need_turn reason
        for `threshold` or more consecutive turns.
        """
        return self.consecutive_same_reason >= threshold

    def needs_fallback(self, turns_threshold: int = 3) -> bool:
        """Check if fallback classification should activate.

        Returns True if no signal has been emitted for `turns_threshold`
        consecutive turns.
        """
        return self.turns_without_signal >= turns_threshold


__all__ = [
    "SignalType",
    "Signal",
    "NeedTurnFields",
    "ContextSufficientFields",
    "StuckFields",
    "NeedCapabilityFields",
    "PartialAnswerFields",
    "DelegationRecommendedFields",
    "SignalFields",
    "AgentSignalState",
    "SIGNAL_FIELDS_MAP",
]
