"""Unit tests for RuleContextBuilder and PluginState persistence (T040, T041).

This module tests:
- RuleContextBuilder builds context from agent state
- Default values for missing data
- All state components (TurnState, HistoryState, UserState, ProjectState, PluginState)
- PluginState get/set persistence with user/project scoping
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from backend.src.services.ans.event import Event, EventType, Severity
from backend.src.services.database import DatabaseService
from backend.src.services.plugins.context import (
    EventData,
    HistoryState,
    PluginState,
    ProjectState,
    RuleContext,
    ToolCallRecord,
    TurnState,
    UserState,
    RuleContextBuilder,
)
from backend.src.services.plugins.state import PluginStateService


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db() -> DatabaseService:
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DatabaseService(db_path)
        db.initialize()
        yield db


@pytest.fixture
def plugin_state_service(temp_db: DatabaseService) -> PluginStateService:
    """Create a PluginStateService with temporary database."""
    return PluginStateService(temp_db)


@pytest.fixture
def context_builder(temp_db: DatabaseService) -> RuleContextBuilder:
    """Create a RuleContextBuilder with temporary database."""
    return RuleContextBuilder(database_service=temp_db)


@pytest.fixture
def sample_event() -> Event:
    """Create a sample event for testing."""
    return Event(
        type=EventType.AGENT_TURN_START,
        source="test_source",
        severity=Severity.INFO,
        payload={"key": "value"},
    )


@pytest.fixture
def sample_tool_calls() -> list:
    """Create sample tool call data."""
    return [
        {
            "name": "search_code",
            "arguments": {"query": "test query"},
            "result": "Found 5 results",
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "name": "read_file",
            "arguments": {"path": "/test/file.py"},
            "result": None,
            "status": "failed",
            "error": "File not found",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    ]


@pytest.fixture
def sample_messages() -> list:
    """Create sample message data."""
    return [
        {"role": "user", "content": "Hello, find the bug"},
        {"role": "assistant", "content": "Let me search for that..."},
        {"role": "user", "content": "Thanks!"},
    ]


# =============================================================================
# T040: RuleContextBuilder Tests
# =============================================================================


class TestRuleContextBuilderBasic:
    """Basic tests for RuleContextBuilder."""

    def test_builder_with_defaults(self, context_builder: RuleContextBuilder) -> None:
        """RuleContextBuilder.build() returns valid RuleContext with defaults."""
        context = context_builder.build(
            user_id="test-user",
            project_id="test-project",
        )

        assert isinstance(context, RuleContext)
        assert context.turn.number == 1  # Default turn number
        assert context.turn.token_usage == 0.0
        assert context.turn.context_usage == 0.0
        assert context.turn.iteration_count == 0
        assert context.user.id == "test-user"
        assert context.project.id == "test-project"
        assert context.event is None
        assert context.result is None

    def test_builder_with_turn_info(self, context_builder: RuleContextBuilder) -> None:
        """RuleContextBuilder populates TurnState from agent state."""
        context = context_builder.build(
            turn_number=5,
            token_usage=0.75,
            context_usage=0.5,
            iteration_count=3,
            user_id="test-user",
            project_id="test-project",
        )

        assert context.turn.number == 5
        assert context.turn.token_usage == 0.75
        assert context.turn.context_usage == 0.5
        assert context.turn.iteration_count == 3

    def test_builder_with_event(
        self,
        context_builder: RuleContextBuilder,
        sample_event: Event,
    ) -> None:
        """RuleContextBuilder populates EventData from event."""
        context = context_builder.build(
            user_id="test-user",
            project_id="test-project",
            event=sample_event,
        )

        assert context.event is not None
        assert isinstance(context.event, EventData)
        assert context.event.type == EventType.AGENT_TURN_START
        assert context.event.source == "test_source"
        assert context.event.severity == "info"
        assert context.event.payload == {"key": "value"}

    def test_builder_clamps_usage_values(
        self,
        context_builder: RuleContextBuilder,
    ) -> None:
        """RuleContextBuilder clamps token/context usage to 0.0-1.0."""
        context = context_builder.build(
            token_usage=1.5,  # Over 1.0
            context_usage=-0.1,  # Under 0.0
            user_id="test-user",
            project_id="test-project",
        )

        assert context.turn.token_usage == 1.0  # Clamped to max
        assert context.turn.context_usage == 0.0  # Clamped to min


class TestRuleContextBuilderTurnState:
    """Tests for TurnState population (T043)."""

    def test_turn_state_from_agent_loop(
        self,
        context_builder: RuleContextBuilder,
    ) -> None:
        """TurnState correctly populated from agent loop state."""
        context = context_builder.build(
            turn_number=10,
            token_usage=0.85,
            context_usage=0.6,
            iteration_count=7,
            user_id="test-user",
            project_id="test-project",
        )

        assert context.turn.number == 10
        assert context.turn.token_usage == 0.85
        assert context.turn.context_usage == 0.6
        assert context.turn.iteration_count == 7

    def test_turn_number_minimum(
        self,
        context_builder: RuleContextBuilder,
    ) -> None:
        """Turn number defaults to 1 if 0 or negative."""
        context = context_builder.build(
            turn_number=0,
            user_id="test-user",
            project_id="test-project",
        )

        assert context.turn.number == 1  # Minimum is 1


class TestRuleContextBuilderHistoryState:
    """Tests for HistoryState population (T044)."""

    def test_history_from_tool_calls(
        self,
        context_builder: RuleContextBuilder,
        sample_tool_calls: list,
    ) -> None:
        """HistoryState correctly populated from tool call history."""
        context = context_builder.build(
            tool_calls=sample_tool_calls,
            user_id="test-user",
            project_id="test-project",
        )

        assert len(context.history.tools) == 2
        assert context.history.tools[0].name == "search_code"
        assert context.history.tools[0].success is True
        assert context.history.tools[1].name == "read_file"
        assert context.history.tools[1].success is False

    def test_history_failure_counting(
        self,
        context_builder: RuleContextBuilder,
        sample_tool_calls: list,
    ) -> None:
        """HistoryState correctly counts failures per tool."""
        context = context_builder.build(
            tool_calls=sample_tool_calls,
            user_id="test-user",
            project_id="test-project",
        )

        assert context.history.total_failures == 1
        assert context.history.get_failures_for_tool("read_file") == 1
        assert context.history.get_failures_for_tool("search_code") == 0

    def test_history_from_messages(
        self,
        context_builder: RuleContextBuilder,
        sample_messages: list,
    ) -> None:
        """HistoryState includes message history."""
        context = context_builder.build(
            messages=sample_messages,
            user_id="test-user",
            project_id="test-project",
        )

        assert len(context.history.messages) == 3
        assert context.history.messages[0]["role"] == "user"
        assert context.history.messages[0]["content"] == "Hello, find the bug"

    def test_empty_history(
        self,
        context_builder: RuleContextBuilder,
    ) -> None:
        """HistoryState is empty when no data provided."""
        context = context_builder.build(
            user_id="test-user",
            project_id="test-project",
        )

        assert len(context.history.tools) == 0
        assert len(context.history.messages) == 0
        assert context.history.total_failures == 0
        assert context.history.total_tool_calls == 0


class TestRuleContextBuilderUserState:
    """Tests for UserState population (T045)."""

    def test_user_state_basic(
        self,
        context_builder: RuleContextBuilder,
    ) -> None:
        """UserState correctly populated with user ID."""
        context = context_builder.build(
            user_id="user-123",
            project_id="test-project",
        )

        assert context.user.id == "user-123"
        assert isinstance(context.user.settings, dict)

    def test_user_state_empty_id(
        self,
        context_builder: RuleContextBuilder,
    ) -> None:
        """UserState handles empty user_id."""
        context = context_builder.build(
            user_id="",
            project_id="test-project",
        )

        assert context.user.id == ""

    def test_user_settings_loaded(
        self,
        temp_db: DatabaseService,
    ) -> None:
        """UserState settings loaded from database when user exists."""
        # First create user settings in database
        from backend.src.services.user_settings import UserSettingsService
        user_settings_service = UserSettingsService(temp_db)
        user_settings_service.update_settings(
            user_id="settings-user",
            thinking_enabled=True,
            chat_center_mode=True,
        )

        builder = RuleContextBuilder(database_service=temp_db)
        context = builder.build(
            user_id="settings-user",
            project_id="test-project",
        )

        # Settings should be a snapshot dict
        assert context.user.id == "settings-user"
        assert isinstance(context.user.settings, dict)


class TestRuleContextBuilderProjectState:
    """Tests for ProjectState population (T046)."""

    def test_project_state_basic(
        self,
        context_builder: RuleContextBuilder,
    ) -> None:
        """ProjectState correctly populated with project ID."""
        context = context_builder.build(
            user_id="test-user",
            project_id="project-abc",
        )

        assert context.project.id == "project-abc"
        assert isinstance(context.project.settings, dict)

    def test_project_state_default(
        self,
        context_builder: RuleContextBuilder,
    ) -> None:
        """ProjectState uses 'default' when project_id is empty."""
        context = context_builder.build(
            user_id="test-user",
            project_id="",
        )

        assert context.project.id == ""


class TestRuleContextBuilderPluginState:
    """Tests for PluginState population (T048)."""

    def test_plugin_state_empty_initially(
        self,
        context_builder: RuleContextBuilder,
    ) -> None:
        """PluginState is empty when no persisted state exists."""
        context = context_builder.build(
            user_id="test-user",
            project_id="test-project",
        )

        assert isinstance(context.state, PluginState)
        assert len(context.state.keys()) == 0

    def test_plugin_state_with_persisted_data(
        self,
        temp_db: DatabaseService,
        plugin_state_service: PluginStateService,
    ) -> None:
        """PluginState loaded from database when persisted state exists."""
        # First persist some state
        plugin_state_service.set(
            user_id="state-user",
            project_id="state-project",
            plugin_id="test-plugin",
            key="counter",
            value=42,
        )
        plugin_state_service.set(
            user_id="state-user",
            project_id="state-project",
            plugin_id="test-plugin",
            key="enabled",
            value=True,
        )

        builder = RuleContextBuilder(
            database_service=temp_db,
            plugin_id="test-plugin",
        )
        context = builder.build(
            user_id="state-user",
            project_id="state-project",
        )

        assert context.state.get("counter") == 42
        assert context.state.get("enabled") is True


# =============================================================================
# T041: PluginStateService Persistence Tests
# =============================================================================


class TestPluginStateServiceBasic:
    """Basic tests for PluginStateService."""

    def test_set_and_get(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """PluginStateService can set and get values."""
        plugin_state_service.set(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="my_key",
            value="my_value",
        )

        result = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="my_key",
        )

        assert result == "my_value"

    def test_get_nonexistent_returns_default(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """Getting nonexistent key returns default value."""
        result = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="nonexistent",
            default="fallback",
        )

        assert result == "fallback"

    def test_get_nonexistent_returns_none(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """Getting nonexistent key returns None if no default."""
        result = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="nonexistent",
        )

        assert result is None

    def test_set_overwrites_existing(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """Setting a key that exists overwrites the value."""
        plugin_state_service.set(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="counter",
            value=1,
        )

        plugin_state_service.set(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="counter",
            value=2,
        )

        result = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="counter",
        )

        assert result == 2


class TestPluginStateServiceScoping:
    """Tests for user/project scoping."""

    def test_user_scoping(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """State is scoped to user - different users have separate state."""
        plugin_state_service.set(
            user_id="user-a",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="value",
            value="A's value",
        )
        plugin_state_service.set(
            user_id="user-b",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="value",
            value="B's value",
        )

        result_a = plugin_state_service.get(
            user_id="user-a",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="value",
        )
        result_b = plugin_state_service.get(
            user_id="user-b",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="value",
        )

        assert result_a == "A's value"
        assert result_b == "B's value"

    def test_project_scoping(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """State is scoped to project - different projects have separate state."""
        plugin_state_service.set(
            user_id="user-1",
            project_id="proj-a",
            plugin_id="plugin-1",
            key="value",
            value="Project A",
        )
        plugin_state_service.set(
            user_id="user-1",
            project_id="proj-b",
            plugin_id="plugin-1",
            key="value",
            value="Project B",
        )

        result_a = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-a",
            plugin_id="plugin-1",
            key="value",
        )
        result_b = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-b",
            plugin_id="plugin-1",
            key="value",
        )

        assert result_a == "Project A"
        assert result_b == "Project B"

    def test_plugin_scoping(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """State is scoped to plugin - different plugins have separate state."""
        plugin_state_service.set(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-a",
            key="value",
            value="Plugin A",
        )
        plugin_state_service.set(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-b",
            key="value",
            value="Plugin B",
        )

        result_a = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-a",
            key="value",
        )
        result_b = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-b",
            key="value",
        )

        assert result_a == "Plugin A"
        assert result_b == "Plugin B"


class TestPluginStateServiceComplexValues:
    """Tests for complex value types."""

    def test_store_dict(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """Can store and retrieve dict values."""
        plugin_state_service.set(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="config",
            value={"enabled": True, "threshold": 0.75},
        )

        result = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="config",
        )

        assert result == {"enabled": True, "threshold": 0.75}

    def test_store_list(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """Can store and retrieve list values."""
        plugin_state_service.set(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="items",
            value=["item1", "item2", "item3"],
        )

        result = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="items",
        )

        assert result == ["item1", "item2", "item3"]

    def test_store_boolean(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """Can store and retrieve boolean values."""
        plugin_state_service.set(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="flag",
            value=True,
        )

        result = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="flag",
        )

        assert result is True

    def test_store_null(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """Can store and retrieve null/None values."""
        plugin_state_service.set(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="nothing",
            value=None,
        )

        result = plugin_state_service.get(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
            key="nothing",
        )

        assert result is None


class TestPluginStateServiceGetAll:
    """Tests for get_all method."""

    def test_get_all_returns_all_keys(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """get_all returns all keys for plugin."""
        plugin_state_service.set("user-1", "proj-1", "plugin-1", "key1", "value1")
        plugin_state_service.set("user-1", "proj-1", "plugin-1", "key2", "value2")
        plugin_state_service.set("user-1", "proj-1", "plugin-1", "key3", "value3")

        result = plugin_state_service.get_all(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
        )

        assert result == {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }

    def test_get_all_empty(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """get_all returns empty dict when no state exists."""
        result = plugin_state_service.get_all(
            user_id="no-user",
            project_id="no-proj",
            plugin_id="no-plugin",
        )

        assert result == {}

    def test_get_all_only_returns_matching_scope(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """get_all only returns keys matching the exact scope."""
        # Set values in different scopes
        plugin_state_service.set("user-1", "proj-1", "plugin-1", "key1", "v1")
        plugin_state_service.set("user-1", "proj-1", "plugin-2", "key2", "v2")
        plugin_state_service.set("user-1", "proj-2", "plugin-1", "key3", "v3")
        plugin_state_service.set("user-2", "proj-1", "plugin-1", "key4", "v4")

        result = plugin_state_service.get_all("user-1", "proj-1", "plugin-1")

        # Should only return key1
        assert result == {"key1": "v1"}


class TestPluginStateServiceClear:
    """Tests for clear method."""

    def test_clear_removes_all_keys(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """clear removes all keys for the specified scope."""
        plugin_state_service.set("user-1", "proj-1", "plugin-1", "key1", "v1")
        plugin_state_service.set("user-1", "proj-1", "plugin-1", "key2", "v2")

        plugin_state_service.clear(
            user_id="user-1",
            project_id="proj-1",
            plugin_id="plugin-1",
        )

        result = plugin_state_service.get_all("user-1", "proj-1", "plugin-1")
        assert result == {}

    def test_clear_only_affects_matching_scope(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """clear only affects the specified scope."""
        plugin_state_service.set("user-1", "proj-1", "plugin-1", "key1", "v1")
        plugin_state_service.set("user-1", "proj-1", "plugin-2", "key2", "v2")

        plugin_state_service.clear("user-1", "proj-1", "plugin-1")

        # plugin-1 should be empty
        result1 = plugin_state_service.get_all("user-1", "proj-1", "plugin-1")
        assert result1 == {}

        # plugin-2 should still have data
        result2 = plugin_state_service.get_all("user-1", "proj-1", "plugin-2")
        assert result2 == {"key2": "v2"}


class TestPluginStateServiceThreadSafety:
    """Tests for thread-safe state access."""

    def test_concurrent_writes(
        self,
        plugin_state_service: PluginStateService,
    ) -> None:
        """Multiple writes from different 'threads' don't conflict."""
        import threading

        errors = []

        def write_values(thread_id: int):
            try:
                for i in range(10):
                    plugin_state_service.set(
                        user_id=f"user-{thread_id}",
                        project_id="proj-1",
                        plugin_id="plugin-1",
                        key=f"key-{i}",
                        value=f"value-{thread_id}-{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_values, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify each thread's data is intact
        for thread_id in range(5):
            for i in range(10):
                result = plugin_state_service.get(
                    user_id=f"user-{thread_id}",
                    project_id="proj-1",
                    plugin_id="plugin-1",
                    key=f"key-{i}",
                )
                assert result == f"value-{thread_id}-{i}"
