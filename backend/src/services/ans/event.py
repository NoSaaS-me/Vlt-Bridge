"""Event types and data structures for the Agent Notification System."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


class Severity(Enum):
    """Event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def value_int(self) -> int:
        """Return integer value for sorting (lower = less severe)."""
        order = {
            "debug": 0,
            "info": 1,
            "warning": 2,
            "error": 3,
            "critical": 4,
        }
        return order[self.value]

    def __ge__(self, other: "Severity") -> bool:
        """Greater than or equal comparison."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value_int >= other.value_int

    def __gt__(self, other: "Severity") -> bool:
        """Greater than comparison."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value_int > other.value_int

    def __le__(self, other: "Severity") -> bool:
        """Less than or equal comparison."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value_int <= other.value_int

    def __lt__(self, other: "Severity") -> bool:
        """Less than comparison."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value_int < other.value_int


class EventType:
    """Hierarchical event type constants.

    Event types follow the pattern: category.subcategory.action
    """

    # Tool events
    TOOL_CALL_PENDING = "tool.call.pending"
    TOOL_CALL_SUCCESS = "tool.call.success"
    TOOL_CALL_FAILURE = "tool.call.failure"
    TOOL_CALL_TIMEOUT = "tool.call.timeout"
    TOOL_BATCH_COMPLETE = "tool.batch.complete"

    # Budget events
    BUDGET_TOKEN_WARNING = "budget.token.warning"
    BUDGET_TOKEN_EXCEEDED = "budget.token.exceeded"
    BUDGET_ITERATION_WARNING = "budget.iteration.warning"
    BUDGET_ITERATION_EXCEEDED = "budget.iteration.exceeded"
    BUDGET_TIMEOUT_WARNING = "budget.timeout.warning"

    # Agent events
    AGENT_TURN_START = "agent.turn.start"
    AGENT_TURN_END = "agent.turn.end"
    AGENT_LOOP_DETECTED = "agent.loop.detected"
    AGENT_SELF_NOTIFY = "agent.self.notify"
    AGENT_SELF_REMIND = "agent.self.remind"

    # Proactive context events (014-ans-enhancements)
    CONTEXT_APPROACHING_LIMIT = "context.approaching_limit"
    SESSION_RESUMED = "session.resumed"
    SOURCE_STALE = "source.stale"
    TASK_CHECKPOINT = "task.checkpoint"

    # Future events (placeholders)
    SUBAGENT_COMPLETE = "subagent.complete"
    SUBAGENT_FAILED = "subagent.failed"
    CLI_EVENT = "cli.event"


@dataclass
class Event:
    """An occurrence in the system that may trigger a notification.

    Attributes:
        id: Unique event identifier.
        type: Hierarchical event type (e.g., "tool.call.failure").
        source: Component that generated the event.
        severity: Event severity level.
        timestamp: When the event occurred (UTC).
        payload: Event-specific data.
        dedupe_key: Key for deduplication (derived if not set).
    """

    type: str
    source: str
    severity: Severity
    payload: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dedupe_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Generate dedupe_key if not provided."""
        if self.dedupe_key is None:
            # Default dedupe key: type + source + payload hash
            self.dedupe_key = f"{self.type}:{self.source}"

    def matches_type(self, event_type: str) -> bool:
        """Check if this event matches a type pattern (supports wildcards).

        Args:
            event_type: Event type pattern. Supports exact match or wildcard suffix.
                       Examples: "tool.call.failure" (exact) or "tool.*" (prefix match)

        Returns:
            True if this event's type matches the pattern.
        """
        if event_type == self.type:
            return True
        # Support prefix matching with wildcard: "tool.*" matches "tool.call.failure"
        if event_type.endswith(".*"):
            prefix = event_type[:-2]
            return self.type.startswith(prefix + ".")
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "id": str(self.id),
            "type": self.type,
            "source": self.source,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "dedupe_key": self.dedupe_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data.get("id"), str) else data.get("id", uuid4()),
            type=data["type"],
            source=data["source"],
            severity=Severity(data["severity"]) if isinstance(data.get("severity"), str) else data["severity"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.now(timezone.utc)),
            payload=data.get("payload", {}),
            dedupe_key=data.get("dedupe_key"),
        )
