"""Rule context definitions for the Oracle Plugin System.

This module defines the read-only context API available to rule conditions
and scripts during evaluation.

Also provides RuleContextBuilder (T042-T048) for constructing RuleContext
from agent state with database-backed persistence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..database import DatabaseService
    from ..ans.event import Event

logger = logging.getLogger(__name__)


@dataclass
class TurnState:
    """Current turn information.

    Provides information about the current agent turn being executed.

    Attributes:
        number: Turn number (1-indexed).
        token_usage: Token budget usage as a ratio (0.0-1.0).
        context_usage: Context window usage as a ratio (0.0-1.0).
        iteration_count: Current iteration in turn.
    """

    number: int
    token_usage: float
    context_usage: float
    iteration_count: int

    def __post_init__(self) -> None:
        """Validate turn state values."""
        if self.number < 1:
            raise ValueError(f"Turn number must be >= 1, got {self.number}")
        if not 0.0 <= self.token_usage <= 1.0:
            raise ValueError(f"Token usage must be 0.0-1.0, got {self.token_usage}")
        if not 0.0 <= self.context_usage <= 1.0:
            raise ValueError(f"Context usage must be 0.0-1.0, got {self.context_usage}")
        if self.iteration_count < 0:
            raise ValueError(f"Iteration count must be >= 0, got {self.iteration_count}")


@dataclass
class ToolCallRecord:
    """Record of a tool call.

    Captures details about a single tool invocation.

    Attributes:
        name: Tool name.
        arguments: Arguments passed to the tool.
        result: Tool result (if available).
        success: Whether the tool call succeeded.
        timestamp: When the tool was called.
    """

    name: str
    arguments: dict[str, Any]
    result: Optional[str]
    success: bool
    timestamp: datetime


@dataclass
class HistoryState:
    """Historical information about the conversation.

    Provides access to recent messages and tool call history.

    Attributes:
        messages: Recent messages (list of dicts with role, content).
        tools: Recent tool calls.
        failures: Tool name to failure count mapping.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[ToolCallRecord] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def total_tool_calls(self) -> int:
        """Return the total number of tool calls."""
        return len(self.tools)

    @property
    def total_failures(self) -> int:
        """Return the total number of failures across all tools."""
        return sum(self.failures.values())

    def get_failures_for_tool(self, tool_name: str) -> int:
        """Get the number of failures for a specific tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            Number of failures for the specified tool.
        """
        return self.failures.get(tool_name, 0)


@dataclass
class UserState:
    """User information (read-only).

    Provides access to user identity and settings snapshot.

    Attributes:
        id: User identifier.
        settings: User settings snapshot as a dictionary.
    """

    id: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectState:
    """Project information (read-only).

    Provides access to project identity and settings snapshot.

    Attributes:
        id: Project identifier.
        settings: Project settings snapshot as a dictionary.
    """

    id: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginState:
    """Plugin-scoped persistent state.

    Provides read access to plugin state during rule evaluation.
    Write access (set) is only available in actions, not conditions.

    Attributes:
        _store: Internal storage dictionary.
    """

    _store: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from plugin state.

        Args:
            key: State key to retrieve.
            default: Default value if key not found.

        Returns:
            The value associated with the key, or default if not found.
        """
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in plugin state.

        Args:
            key: State key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        return key in self._store

    def keys(self) -> list[str]:
        """Return all keys in the plugin state.

        Returns:
            List of state keys.
        """
        return list(self._store.keys())


@dataclass
class EventData:
    """Data from the triggering event.

    Captures information about the ANS event that triggered the rule.

    Attributes:
        type: Event type string.
        source: Component that generated the event.
        severity: Event severity level.
        payload: Event-specific payload data.
        timestamp: When the event occurred.
    """

    type: str
    source: str
    severity: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class ToolResult:
    """Result from a tool execution.

    Available in ON_TOOL_COMPLETE hook for accessing tool output.

    Attributes:
        tool_name: Name of the tool that was executed.
        success: Whether the tool succeeded.
        result: The tool's output/result.
        error: Error message if the tool failed.
        duration_ms: Execution time in milliseconds.
    """

    tool_name: str
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class RuleContext:
    """Context available to rule evaluation.

    This is the complete context passed to rule conditions and scripts
    for evaluation. All state objects are read-only during condition
    evaluation.

    Attributes:
        turn: Current turn information.
        history: Historical conversation information.
        user: User information.
        project: Project information.
        state: Plugin-scoped state.
        event: The triggering event (if applicable).
        result: Tool result (for ON_TOOL_COMPLETE hook).
    """

    turn: TurnState
    history: HistoryState
    user: UserState
    project: ProjectState
    state: PluginState
    event: Optional[EventData] = None
    result: Optional[ToolResult] = None

    @classmethod
    def create_minimal(
        cls,
        user_id: str,
        project_id: str,
        turn_number: int = 1,
    ) -> "RuleContext":
        """Create a minimal context for testing or simple scenarios.

        Args:
            user_id: User identifier.
            project_id: Project identifier.
            turn_number: Turn number (default 1).

        Returns:
            A RuleContext with minimal default values.
        """
        return cls(
            turn=TurnState(
                number=turn_number,
                token_usage=0.0,
                context_usage=0.0,
                iteration_count=0,
            ),
            history=HistoryState(),
            user=UserState(id=user_id),
            project=ProjectState(id=project_id),
            state=PluginState(),
        )


class RuleContextBuilder:
    """Builder for constructing RuleContext from agent state (T042-T048).

    This class provides a clean API for building RuleContext instances
    from various agent state components. It handles:

    - TurnState: Current turn number, token/context usage, iteration count (T043)
    - HistoryState: Messages and tool call history (T044)
    - UserState: User ID and settings snapshot (T045)
    - ProjectState: Project ID and settings snapshot (T046)
    - PluginState: Persistent plugin state from database (T048)
    - EventData: Triggering event information

    Example:
        >>> builder = RuleContextBuilder(database_service=db)
        >>> context = builder.build(
        ...     turn_number=5,
        ...     token_usage=0.75,
        ...     user_id="user-123",
        ...     project_id="proj-456",
        ...     event=some_event,
        ... )
    """

    def __init__(
        self,
        database_service: Optional["DatabaseService"] = None,
        plugin_id: Optional[str] = None,
    ):
        """Initialize the context builder.

        Args:
            database_service: Optional DatabaseService for loading user settings
                            and plugin state. If not provided, settings/state
                            will be empty.
            plugin_id: Optional plugin ID for loading plugin-specific state.
        """
        self._database_service = database_service
        self._plugin_id = plugin_id

    def build(
        self,
        turn_number: int = 0,
        token_usage: float = 0.0,
        context_usage: float = 0.0,
        iteration_count: int = 0,
        messages: Optional[list[dict[str, Any]]] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        user_id: str = "",
        project_id: str = "",
        event: Optional["Event"] = None,
    ) -> RuleContext:
        """Build a RuleContext from the provided agent state.

        Args:
            turn_number: Current turn number (1-indexed, 0 defaults to 1).
            token_usage: Token budget usage ratio (0.0-1.0).
            context_usage: Context window usage ratio (0.0-1.0).
            iteration_count: Current iteration within the turn.
            messages: List of message dicts with 'role' and 'content'.
            tool_calls: List of tool call dicts with name, arguments, result, status.
            user_id: User identifier.
            project_id: Project identifier.
            event: The ANS event that triggered rule evaluation.

        Returns:
            A fully populated RuleContext.
        """
        # Build TurnState (T043)
        turn = self._build_turn_state(
            turn_number=turn_number,
            token_usage=token_usage,
            context_usage=context_usage,
            iteration_count=iteration_count,
        )

        # Build HistoryState (T044)
        history = self._build_history_state(
            messages=messages,
            tool_calls=tool_calls,
        )

        # Build UserState (T045)
        user = self._build_user_state(user_id=user_id)

        # Build ProjectState (T046)
        project = self._build_project_state(
            user_id=user_id,
            project_id=project_id,
        )

        # Build PluginState (T048)
        state = self._build_plugin_state(
            user_id=user_id,
            project_id=project_id,
        )

        # Build EventData
        event_data = self._build_event_data(event)

        return RuleContext(
            turn=turn,
            history=history,
            user=user,
            project=project,
            state=state,
            event=event_data,
        )

    def _build_turn_state(
        self,
        turn_number: int,
        token_usage: float,
        context_usage: float,
        iteration_count: int,
    ) -> TurnState:
        """Build TurnState from agent loop state (T043).

        Args:
            turn_number: Current turn number (0 defaults to 1).
            token_usage: Token usage ratio.
            context_usage: Context usage ratio.
            iteration_count: Iteration count.

        Returns:
            Populated TurnState.
        """
        # Ensure turn_number is at least 1
        if turn_number < 1:
            turn_number = 1

        # Clamp usage values to valid range
        token_usage = max(0.0, min(1.0, token_usage))
        context_usage = max(0.0, min(1.0, context_usage))

        # Ensure iteration_count is non-negative
        iteration_count = max(0, iteration_count)

        return TurnState(
            number=turn_number,
            token_usage=token_usage,
            context_usage=context_usage,
            iteration_count=iteration_count,
        )

    def _build_history_state(
        self,
        messages: Optional[list[dict[str, Any]]],
        tool_calls: Optional[list[dict[str, Any]]],
    ) -> HistoryState:
        """Build HistoryState from collected tool calls (T044).

        Args:
            messages: List of message dicts.
            tool_calls: List of tool call dicts.

        Returns:
            Populated HistoryState.
        """
        # Convert tool call dicts to ToolCallRecord objects
        tools: list[ToolCallRecord] = []
        failures: dict[str, int] = {}

        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name", "unknown")
                arguments = tc.get("arguments", {})
                result = tc.get("result")
                status = tc.get("status", "unknown")
                success = status == "success"

                # Parse timestamp if present
                timestamp_str = tc.get("timestamp")
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                    except (ValueError, TypeError):
                        timestamp = datetime.now(timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)

                record = ToolCallRecord(
                    name=name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    result=result,
                    success=success,
                    timestamp=timestamp,
                )
                tools.append(record)

                # Track failures
                if not success:
                    failures[name] = failures.get(name, 0) + 1

        return HistoryState(
            messages=messages or [],
            tools=tools,
            failures=failures,
        )

    def _build_user_state(self, user_id: str) -> UserState:
        """Build UserState from user settings (T045).

        Args:
            user_id: User identifier.

        Returns:
            Populated UserState with settings snapshot.
        """
        settings: dict[str, Any] = {}

        if self._database_service and user_id:
            try:
                from ..user_settings import UserSettingsService
                user_settings_service = UserSettingsService(self._database_service)
                model_settings = user_settings_service.get_settings(user_id)

                # Convert ModelSettings to dict snapshot
                settings = {
                    "oracle_model": model_settings.oracle_model,
                    "oracle_provider": model_settings.oracle_provider.value,
                    "subagent_model": model_settings.subagent_model,
                    "subagent_provider": model_settings.subagent_provider.value,
                    "thinking_enabled": model_settings.thinking_enabled,
                    "chat_center_mode": model_settings.chat_center_mode,
                    "librarian_timeout": model_settings.librarian_timeout,
                    "max_context_nodes": model_settings.max_context_nodes,
                }
            except Exception as e:
                logger.warning(f"Failed to load user settings for {user_id}: {e}")

        return UserState(id=user_id, settings=settings)

    def _build_project_state(
        self,
        user_id: str,
        project_id: str,
    ) -> ProjectState:
        """Build ProjectState from project settings (T046).

        Args:
            user_id: User identifier.
            project_id: Project identifier.

        Returns:
            Populated ProjectState with settings snapshot.
        """
        settings: dict[str, Any] = {}

        if self._database_service and user_id and project_id:
            try:
                # Load project settings from database
                conn = self._database_service.connect()
                try:
                    cursor = conn.execute(
                        """
                        SELECT name, description, settings_json
                        FROM projects
                        WHERE user_id = ? AND project_id = ?
                        """,
                        (user_id, project_id),
                    )
                    row = cursor.fetchone()
                    if row:
                        import json
                        settings = {
                            "name": row["name"],
                            "description": row["description"],
                        }
                        if row["settings_json"]:
                            try:
                                settings.update(json.loads(row["settings_json"]))
                            except (json.JSONDecodeError, TypeError):
                                pass
                finally:
                    conn.close()
            except Exception as e:
                logger.warning(f"Failed to load project settings for {project_id}: {e}")

        return ProjectState(id=project_id, settings=settings)

    def _build_plugin_state(
        self,
        user_id: str,
        project_id: str,
    ) -> PluginState:
        """Build PluginState from database persistence (T048).

        Args:
            user_id: User identifier.
            project_id: Project identifier.

        Returns:
            Populated PluginState with persisted values.
        """
        store: dict[str, Any] = {}

        if self._database_service and self._plugin_id and user_id and project_id:
            try:
                from .state import PluginStateService
                state_service = PluginStateService(self._database_service)
                store = state_service.get_all(
                    user_id=user_id,
                    project_id=project_id,
                    plugin_id=self._plugin_id,
                )
            except Exception as e:
                logger.warning(f"Failed to load plugin state for {self._plugin_id}: {e}")

        return PluginState(_store=store)

    def _build_event_data(self, event: Optional["Event"]) -> Optional[EventData]:
        """Build EventData from ANS event.

        Args:
            event: The triggering Event, or None.

        Returns:
            EventData if event provided, None otherwise.
        """
        if event is None:
            return None

        return EventData(
            type=event.type,
            source=event.source,
            severity=event.severity.value if hasattr(event.severity, 'value') else str(event.severity),
            payload=event.payload,
            timestamp=event.timestamp,
        )


__all__ = [
    "TurnState",
    "ToolCallRecord",
    "HistoryState",
    "UserState",
    "ProjectState",
    "PluginState",
    "EventData",
    "ToolResult",
    "RuleContext",
    "RuleContextBuilder",
]
