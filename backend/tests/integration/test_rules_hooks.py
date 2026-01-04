"""Integration tests for rule hook point firing (T032).

This module tests:
- Rules fire at correct hook points
- Multiple rules on same hook
- End-to-end event -> rule -> action flow
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator, List

import pytest

from backend.src.services.ans.bus import EventBus, reset_event_bus
from backend.src.services.ans.event import Event, EventType, Severity

from backend.src.services.plugins.rule import (
    ActionType,
    HookPoint,
    Rule,
    RuleAction,
)
from backend.src.services.plugins.context import (
    EventData,
    HistoryState,
    PluginState,
    ProjectState,
    RuleContext,
    ToolResult,
    TurnState,
    UserState,
)
from backend.src.services.plugins.loader import RuleLoader
from backend.src.services.plugins.expression import ExpressionEvaluator
from backend.src.services.plugins.actions import ActionDispatcher
from backend.src.services.plugins.engine import (
    RuleEngine,
    EVENT_TO_HOOK,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clean_event_bus():
    """Reset the global event bus before each test."""
    reset_event_bus()
    yield
    reset_event_bus()


@pytest.fixture
def event_bus() -> EventBus:
    """Create a fresh EventBus instance."""
    return EventBus()


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Create a fresh ExpressionEvaluator instance."""
    return ExpressionEvaluator()


@pytest.fixture
def dispatcher(event_bus: EventBus) -> ActionDispatcher:
    """Create an ActionDispatcher with the event bus."""
    return ActionDispatcher(event_bus=event_bus)


@pytest.fixture
def all_hooks_rules() -> str:
    """Create rules for every hook point."""
    rules = []

    # Generate a rule for each hook point
    hook_configs = [
        ("on_query_start", "query-start-hook", "Query started"),
        ("on_turn_start", "turn-start-hook", "Turn started"),
        ("on_turn_end", "turn-end-hook", "Turn ended"),
        ("on_tool_call", "tool-call-hook", "Tool about to be called"),
        ("on_tool_complete", "tool-complete-hook", "Tool completed"),
        ("on_tool_failure", "tool-failure-hook", "Tool failed"),
        ("on_session_end", "session-end-hook", "Session ended"),
    ]

    for trigger, rule_id, message in hook_configs:
        rule = f'''
[rule]
id = "{rule_id}"
name = "{rule_id.replace('-', ' ').title()}"
description = "Test rule for {trigger}"
trigger = "{trigger}"
priority = 100

[condition]
expression = "True"

[action]
type = "notify_self"
message = "{message}"
category = "test"
'''
        rules.append((f"{rule_id}.toml", rule))

    return rules


@pytest.fixture
def multi_rule_same_hook() -> List[tuple]:
    """Create multiple rules on the same hook point."""
    return [
        ("first.toml", '''
[rule]
id = "first-rule"
name = "First Rule"
description = "First rule on turn start"
trigger = "on_turn_start"
priority = 300

[condition]
expression = "True"

[action]
type = "notify_self"
message = "First rule fired"
category = "first"
'''),
        ("second.toml", '''
[rule]
id = "second-rule"
name = "Second Rule"
description = "Second rule on turn start"
trigger = "on_turn_start"
priority = 200

[condition]
expression = "context.turn.token_usage > 0.5"

[action]
type = "notify_self"
message = "Second rule fired"
category = "second"
'''),
        ("third.toml", '''
[rule]
id = "third-rule"
name = "Third Rule"
description = "Third rule on turn start"
trigger = "on_turn_start"
priority = 100

[condition]
expression = "context.turn.number >= 1"

[action]
type = "notify_self"
message = "Third rule fired"
category = "third"
'''),
    ]


@pytest.fixture
def temp_all_hooks_dir(all_hooks_rules) -> Generator[Path, None, None]:
    """Create a temp directory with rules for all hook points."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rules_dir = Path(tmpdir)
        for filename, content in all_hooks_rules:
            (rules_dir / filename).write_text(content)
        yield rules_dir


@pytest.fixture
def temp_multi_rule_dir(multi_rule_same_hook) -> Generator[Path, None, None]:
    """Create a temp directory with multiple rules on same hook."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rules_dir = Path(tmpdir)
        for filename, content in multi_rule_same_hook:
            (rules_dir / filename).write_text(content)
        yield rules_dir


@pytest.fixture
def basic_context() -> RuleContext:
    """Create a basic RuleContext for testing."""
    return RuleContext(
        turn=TurnState(
            number=3,
            token_usage=0.75,
            context_usage=0.4,
            iteration_count=2,
        ),
        history=HistoryState(),
        user=UserState(id="test-user"),
        project=ProjectState(id="test-project"),
        state=PluginState(),
    )


# =============================================================================
# T032: Hook Point Firing Tests
# =============================================================================


class TestRulesFireAtCorrectHookPoints:
    """Test that rules fire at their designated hook points."""

    def test_query_start_hook_fires(
        self,
        temp_all_hooks_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        basic_context: RuleContext,
    ) -> None:
        """QUERY_START event triggers on_query_start rules."""
        loader = RuleLoader(temp_all_hooks_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            context_builder=lambda e: basic_context,
            auto_subscribe=False,
        )

        event = Event(
            type=EventType.QUERY_START,
            source="oracle_agent",
            severity=Severity.INFO,
            payload={"query": "test query"},
        )

        result = engine.evaluate_hook(HookPoint.ON_QUERY_START, event, context=basic_context)

        assert result.first_match is not None
        assert result.first_match.id == "query-start-hook"
        assert any("Query started" in n["message"] for n in dispatcher.pending_notifications)

    def test_turn_start_hook_fires(
        self,
        temp_all_hooks_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        basic_context: RuleContext,
    ) -> None:
        """AGENT_TURN_START event triggers on_turn_start rules."""
        loader = RuleLoader(temp_all_hooks_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            auto_subscribe=False,
        )

        event = Event(
            type=EventType.AGENT_TURN_START,
            source="oracle_agent",
            severity=Severity.INFO,
            payload={"turn": 1},
        )

        result = engine.evaluate_hook(HookPoint.ON_TURN_START, event, context=basic_context)

        assert result.first_match is not None
        assert result.first_match.id == "turn-start-hook"

    def test_turn_end_hook_fires(
        self,
        temp_all_hooks_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        basic_context: RuleContext,
    ) -> None:
        """AGENT_TURN_END event triggers on_turn_end rules."""
        loader = RuleLoader(temp_all_hooks_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            auto_subscribe=False,
        )

        event = Event(
            type=EventType.AGENT_TURN_END,
            source="oracle_agent",
            severity=Severity.INFO,
        )

        result = engine.evaluate_hook(HookPoint.ON_TURN_END, event, context=basic_context)

        assert result.first_match is not None
        assert result.first_match.id == "turn-end-hook"

    def test_tool_call_hook_fires(
        self,
        temp_all_hooks_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        basic_context: RuleContext,
    ) -> None:
        """TOOL_CALL_PENDING event triggers on_tool_call rules."""
        loader = RuleLoader(temp_all_hooks_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            auto_subscribe=False,
        )

        event = Event(
            type=EventType.TOOL_CALL_PENDING,
            source="tool_executor",
            severity=Severity.INFO,
            payload={"tool_name": "vault_search"},
        )

        result = engine.evaluate_hook(HookPoint.ON_TOOL_CALL, event, context=basic_context)

        assert result.first_match is not None
        assert result.first_match.id == "tool-call-hook"

    def test_tool_complete_hook_fires(
        self,
        temp_all_hooks_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        basic_context: RuleContext,
    ) -> None:
        """TOOL_CALL_SUCCESS event triggers on_tool_complete rules."""
        loader = RuleLoader(temp_all_hooks_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            auto_subscribe=False,
        )

        # Add tool result to context
        basic_context.result = ToolResult(
            tool_name="vault_search",
            success=True,
            result="Found 5 documents",
        )

        event = Event(
            type=EventType.TOOL_CALL_SUCCESS,
            source="tool_executor",
            severity=Severity.INFO,
            payload={"tool_name": "vault_search"},
        )

        result = engine.evaluate_hook(HookPoint.ON_TOOL_COMPLETE, event, context=basic_context)

        assert result.first_match is not None
        assert result.first_match.id == "tool-complete-hook"

    def test_tool_failure_hook_fires(
        self,
        temp_all_hooks_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        basic_context: RuleContext,
    ) -> None:
        """TOOL_CALL_FAILURE event triggers on_tool_failure rules."""
        loader = RuleLoader(temp_all_hooks_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            auto_subscribe=False,
        )

        event = Event(
            type=EventType.TOOL_CALL_FAILURE,
            source="tool_executor",
            severity=Severity.ERROR,
            payload={"tool_name": "web_fetch", "error": "Connection timeout"},
        )

        result = engine.evaluate_hook(HookPoint.ON_TOOL_FAILURE, event, context=basic_context)

        assert result.first_match is not None
        assert result.first_match.id == "tool-failure-hook"

    def test_session_end_hook_fires(
        self,
        temp_all_hooks_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        basic_context: RuleContext,
    ) -> None:
        """SESSION_END event triggers on_session_end rules."""
        loader = RuleLoader(temp_all_hooks_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            auto_subscribe=False,
        )

        event = Event(
            type=EventType.SESSION_END,
            source="oracle_agent",
            severity=Severity.INFO,
            payload={"reason": "normal"},
        )

        result = engine.evaluate_hook(HookPoint.ON_SESSION_END, event, context=basic_context)

        assert result.first_match is not None
        assert result.first_match.id == "session-end-hook"


class TestMultipleRulesOnSameHook:
    """Test multiple rules firing on the same hook point."""

    def test_all_matching_rules_fire(
        self,
        temp_multi_rule_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        basic_context: RuleContext,
    ) -> None:
        """All matching rules on a hook point fire their actions."""
        loader = RuleLoader(temp_multi_rule_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            auto_subscribe=False,
        )

        event = Event(
            type=EventType.AGENT_TURN_START,
            source="oracle_agent",
            severity=Severity.INFO,
        )

        result = engine.evaluate_hook(HookPoint.ON_TURN_START, event, context=basic_context)

        # All 3 rules should have matched (context satisfies all conditions)
        matched_count = sum(1 for r in result.results if r.matched)
        assert matched_count == 3

        # Check notifications in priority order
        notifications = dispatcher.pending_notifications
        assert len(notifications) == 3

        # Notifications should be in priority order (highest first)
        assert notifications[0]["category"] == "first"   # priority 300
        assert notifications[1]["category"] == "second"  # priority 200
        assert notifications[2]["category"] == "third"   # priority 100

    def test_partial_matching_rules(
        self,
        temp_multi_rule_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
    ) -> None:
        """Only matching rules fire, non-matching rules are skipped."""
        loader = RuleLoader(temp_multi_rule_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            auto_subscribe=False,
        )

        # Context with low token usage (below 0.5 threshold for second rule)
        context = RuleContext(
            turn=TurnState(number=1, token_usage=0.3, context_usage=0.1, iteration_count=0),
            history=HistoryState(),
            user=UserState(id="test-user"),
            project=ProjectState(id="test-project"),
            state=PluginState(),
        )

        event = Event(
            type=EventType.AGENT_TURN_START,
            source="oracle_agent",
            severity=Severity.INFO,
        )

        result = engine.evaluate_hook(HookPoint.ON_TURN_START, event, context=context)

        # Only first and third should match (second requires token_usage > 0.5)
        matched_count = sum(1 for r in result.results if r.matched)
        assert matched_count == 2

        notifications = dispatcher.pending_notifications
        assert len(notifications) == 2

        categories = [n["category"] for n in notifications]
        assert "first" in categories
        assert "third" in categories
        assert "second" not in categories


class TestEndToEndEventFlow:
    """Test complete event -> rule -> action flow through EventBus."""

    def test_event_bus_emission_triggers_rules(
        self,
        temp_multi_rule_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        basic_context: RuleContext,
    ) -> None:
        """Events emitted to EventBus trigger rule evaluation."""
        loader = RuleLoader(temp_multi_rule_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            context_builder=lambda e: basic_context,
            auto_subscribe=True,
        )

        engine.start()

        # Emit event through the bus
        event = Event(
            type=EventType.AGENT_TURN_START,
            source="oracle_agent",
            severity=Severity.INFO,
        )
        event_bus.emit(event)

        # Rules should have fired
        notifications = dispatcher.pending_notifications
        assert len(notifications) == 3

        engine.stop()

    def test_multiple_sequential_events(
        self,
        temp_all_hooks_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        basic_context: RuleContext,
    ) -> None:
        """Multiple sequential events each trigger their rules."""
        loader = RuleLoader(temp_all_hooks_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            context_builder=lambda e: basic_context,
            auto_subscribe=True,
        )

        engine.start()

        # Simulate agent lifecycle events
        events = [
            Event(type=EventType.QUERY_START, source="test", severity=Severity.INFO),
            Event(type=EventType.AGENT_TURN_START, source="test", severity=Severity.INFO),
            Event(type=EventType.TOOL_CALL_PENDING, source="test", severity=Severity.INFO),
            Event(type=EventType.TOOL_CALL_SUCCESS, source="test", severity=Severity.INFO),
            Event(type=EventType.AGENT_TURN_END, source="test", severity=Severity.INFO),
            Event(type=EventType.SESSION_END, source="test", severity=Severity.INFO),
        ]

        for event in events:
            event_bus.emit(event)

        # Each event should have triggered its corresponding rule
        notifications = dispatcher.pending_notifications
        assert len(notifications) == 6

        messages = [n["message"] for n in notifications]
        assert "Query started" in messages
        assert "Turn started" in messages
        assert "Tool about to be called" in messages
        assert "Tool completed" in messages
        assert "Turn ended" in messages
        assert "Session ended" in messages

        engine.stop()


class TestHookEventMapping:
    """Test the mapping between events and hook points."""

    def test_all_hook_points_have_event_mapping(self) -> None:
        """Every hook point has at least one event that triggers it."""
        triggered_hooks = set(EVENT_TO_HOOK.values())

        for hook in HookPoint:
            assert hook in triggered_hooks, f"HookPoint {hook} has no event mapping"

    def test_timeout_maps_to_tool_failure(self) -> None:
        """TOOL_CALL_TIMEOUT maps to ON_TOOL_FAILURE hook."""
        assert EVENT_TO_HOOK[EventType.TOOL_CALL_TIMEOUT] == HookPoint.ON_TOOL_FAILURE

    def test_event_data_in_context(
        self,
        temp_all_hooks_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
    ) -> None:
        """Event data is available in context for rule evaluation."""
        loader = RuleLoader(temp_all_hooks_dir)

        captured_contexts: List[RuleContext] = []

        def capturing_builder(event: Event) -> RuleContext:
            ctx = RuleContext(
                turn=TurnState(number=1, token_usage=0.5, context_usage=0.3, iteration_count=0),
                history=HistoryState(),
                user=UserState(id="test-user"),
                project=ProjectState(id="test-project"),
                state=PluginState(),
                event=EventData(
                    type=event.type,
                    source=event.source,
                    severity=event.severity.value,
                    payload=event.payload,
                    timestamp=event.timestamp,
                ),
            )
            captured_contexts.append(ctx)
            return ctx

        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            context_builder=capturing_builder,
            auto_subscribe=True,
        )

        engine.start()

        event = Event(
            type=EventType.QUERY_START,
            source="test_source",
            severity=Severity.WARNING,
            payload={"custom_data": "test_value"},
        )
        event_bus.emit(event)

        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx.event is not None
        assert ctx.event.type == EventType.QUERY_START
        assert ctx.event.source == "test_source"
        assert ctx.event.severity == "warning"
        assert ctx.event.payload["custom_data"] == "test_value"

        engine.stop()
