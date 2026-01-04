"""Unit tests for RuleEngine event subscription and rule evaluation (T031).

This module tests:
- RuleEngine subscribes to correct events
- Rule evaluation on event
- Priority ordering
- Hook point mapping
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

import pytest

from backend.src.services.ans.bus import EventBus, reset_event_bus
from backend.src.services.ans.event import Event, EventType, Severity

from backend.src.services.plugins.rule import (
    ActionType,
    HookPoint,
    Priority,
    Rule,
    RuleAction,
)
from backend.src.services.plugins.context import (
    HistoryState,
    PluginState,
    ProjectState,
    RuleContext,
    TurnState,
    UserState,
)
from backend.src.services.plugins.loader import RuleLoader
from backend.src.services.plugins.expression import ExpressionEvaluator
from backend.src.services.plugins.actions import ActionDispatcher
from backend.src.services.plugins.engine import (
    RuleEngine,
    RuleEvaluationResult,
    HookEvaluationResult,
    EVENT_TO_HOOK,
    SUBSCRIBED_EVENTS,
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
def basic_context() -> RuleContext:
    """Create a basic RuleContext for testing."""
    return RuleContext(
        turn=TurnState(
            number=5,
            token_usage=0.85,
            context_usage=0.6,
            iteration_count=3,
        ),
        history=HistoryState(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        ),
        user=UserState(id="user-123"),
        project=ProjectState(id="project-456"),
        state=PluginState(),
    )


@pytest.fixture
def query_start_rule_toml() -> str:
    """Rule that fires on QUERY_START hook."""
    return '''
[rule]
id = "query-start-rule"
name = "Query Start Rule"
description = "Fires on query start"
trigger = "on_query_start"
priority = 100

[condition]
expression = "True"

[action]
type = "notify_self"
message = "Query started"
category = "info"
'''


@pytest.fixture
def turn_start_rule_toml() -> str:
    """Rule that fires on TURN_START hook."""
    return '''
[rule]
id = "turn-start-rule"
name = "Turn Start Rule"
description = "Fires on turn start"
trigger = "on_turn_start"
priority = 100

[condition]
expression = "context.turn.number > 1"

[action]
type = "notify_self"
message = "Turn {{ context.turn.number }} started"
category = "info"
'''


@pytest.fixture
def high_priority_rule_toml() -> str:
    """High priority rule."""
    return '''
[rule]
id = "high-priority-rule"
name = "High Priority Rule"
description = "High priority rule"
trigger = "on_turn_start"
priority = 900

[condition]
expression = "True"

[action]
type = "log"
message = "High priority"
level = "info"
'''


@pytest.fixture
def low_priority_rule_toml() -> str:
    """Low priority rule."""
    return '''
[rule]
id = "low-priority-rule"
name = "Low Priority Rule"
description = "Low priority rule"
trigger = "on_turn_start"
priority = 50

[condition]
expression = "True"

[action]
type = "log"
message = "Low priority"
level = "info"
'''


@pytest.fixture
def token_budget_rule_toml() -> str:
    """Rule that checks token budget."""
    return '''
[rule]
id = "token-budget-warning"
name = "Token Budget Warning"
description = "Warn when token usage exceeds 80%"
trigger = "on_turn_start"
priority = 100

[condition]
expression = "context.turn.token_usage > 0.8"

[action]
type = "notify_self"
message = "Token budget at {{ (context.turn.token_usage * 100) | int }}%"
category = "warning"
priority = "high"
'''


@pytest.fixture
def temp_rules_dir(
    query_start_rule_toml: str,
    turn_start_rule_toml: str,
) -> Generator[Path, None, None]:
    """Create a temporary directory with test rule files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rules_dir = Path(tmpdir)
        (rules_dir / "query_start.toml").write_text(query_start_rule_toml)
        (rules_dir / "turn_start.toml").write_text(turn_start_rule_toml)
        yield rules_dir


@pytest.fixture
def temp_rules_dir_with_priorities(
    high_priority_rule_toml: str,
    low_priority_rule_toml: str,
    token_budget_rule_toml: str,
) -> Generator[Path, None, None]:
    """Create a temp directory with rules of different priorities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rules_dir = Path(tmpdir)
        (rules_dir / "high.toml").write_text(high_priority_rule_toml)
        (rules_dir / "low.toml").write_text(low_priority_rule_toml)
        (rules_dir / "budget.toml").write_text(token_budget_rule_toml)
        yield rules_dir


@pytest.fixture
def engine_with_rules(
    temp_rules_dir: Path,
    evaluator: ExpressionEvaluator,
    dispatcher: ActionDispatcher,
    event_bus: EventBus,
) -> RuleEngine:
    """Create a RuleEngine with loaded rules."""
    loader = RuleLoader(temp_rules_dir)
    return RuleEngine(
        loader=loader,
        evaluator=evaluator,
        dispatcher=dispatcher,
        event_bus=event_bus,
        auto_subscribe=False,  # Don't auto-subscribe for unit tests
    )


@pytest.fixture
def engine_with_priorities(
    temp_rules_dir_with_priorities: Path,
    evaluator: ExpressionEvaluator,
    dispatcher: ActionDispatcher,
    event_bus: EventBus,
) -> RuleEngine:
    """Create a RuleEngine with priority-ordered rules."""
    loader = RuleLoader(temp_rules_dir_with_priorities)
    return RuleEngine(
        loader=loader,
        evaluator=evaluator,
        dispatcher=dispatcher,
        event_bus=event_bus,
        auto_subscribe=False,
    )


# =============================================================================
# T031: Event Subscription Tests
# =============================================================================


class TestRuleEngineEventSubscription:
    """Tests for RuleEngine event subscription."""

    def test_engine_subscribes_to_events_on_start(
        self,
        temp_rules_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
    ) -> None:
        """RuleEngine subscribes to correct events on start()."""
        loader = RuleLoader(temp_rules_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            auto_subscribe=True,
        )

        assert not engine.is_running

        engine.start()

        assert engine.is_running
        assert event_bus.handler_count > 0

        engine.stop()
        assert not engine.is_running

    def test_engine_event_to_hook_mapping(self) -> None:
        """EVENT_TO_HOOK contains correct mappings."""
        # Verify key event types are mapped
        assert EVENT_TO_HOOK[EventType.QUERY_START] == HookPoint.ON_QUERY_START
        assert EVENT_TO_HOOK[EventType.AGENT_TURN_START] == HookPoint.ON_TURN_START
        assert EVENT_TO_HOOK[EventType.AGENT_TURN_END] == HookPoint.ON_TURN_END
        assert EVENT_TO_HOOK[EventType.TOOL_CALL_PENDING] == HookPoint.ON_TOOL_CALL
        assert EVENT_TO_HOOK[EventType.TOOL_CALL_SUCCESS] == HookPoint.ON_TOOL_COMPLETE
        assert EVENT_TO_HOOK[EventType.TOOL_CALL_FAILURE] == HookPoint.ON_TOOL_FAILURE
        assert EVENT_TO_HOOK[EventType.SESSION_END] == HookPoint.ON_SESSION_END

    def test_subscribed_events_list(self) -> None:
        """SUBSCRIBED_EVENTS contains expected event types."""
        assert EventType.QUERY_START in SUBSCRIBED_EVENTS
        assert EventType.AGENT_TURN_START in SUBSCRIBED_EVENTS
        assert EventType.SESSION_END in SUBSCRIBED_EVENTS

    def test_engine_loads_rules_on_init(
        self,
        engine_with_rules: RuleEngine,
    ) -> None:
        """RuleEngine loads rules during initialization."""
        assert engine_with_rules.rule_count == 2

    def test_engine_organizes_rules_by_hook(
        self,
        engine_with_rules: RuleEngine,
    ) -> None:
        """RuleEngine organizes rules by hook point."""
        query_start_rules = engine_with_rules.get_rules_for_hook(HookPoint.ON_QUERY_START)
        turn_start_rules = engine_with_rules.get_rules_for_hook(HookPoint.ON_TURN_START)

        assert len(query_start_rules) == 1
        assert len(turn_start_rules) == 1
        assert query_start_rules[0].id == "query-start-rule"
        assert turn_start_rules[0].id == "turn-start-rule"

    def test_engine_handles_event_emission(
        self,
        temp_rules_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        basic_context: RuleContext,
    ) -> None:
        """RuleEngine handles events emitted to the bus."""
        loader = RuleLoader(temp_rules_dir)

        # Track events received
        received_events: List[Event] = []

        def context_builder(event: Event) -> RuleContext:
            received_events.append(event)
            return basic_context

        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            context_builder=context_builder,
            auto_subscribe=True,
        )

        engine.start()

        # Emit a QUERY_START event
        event = Event(
            type=EventType.QUERY_START,
            source="test",
            severity=Severity.INFO,
            payload={"query": "test query"},
        )
        event_bus.emit(event)

        # Check that event was processed
        assert len(received_events) == 1
        assert received_events[0].type == EventType.QUERY_START

        engine.stop()


# =============================================================================
# T031: Rule Evaluation Tests
# =============================================================================


class TestRuleEngineEvaluation:
    """Tests for rule evaluation on event."""

    def test_evaluate_hook_returns_result(
        self,
        engine_with_rules: RuleEngine,
        basic_context: RuleContext,
    ) -> None:
        """evaluate_hook returns HookEvaluationResult."""
        event = Event(
            type=EventType.AGENT_TURN_START,
            source="test",
            severity=Severity.INFO,
        )

        result = engine_with_rules.evaluate_hook(
            HookPoint.ON_TURN_START,
            event,
            context=basic_context,
        )

        assert isinstance(result, HookEvaluationResult)
        assert result.hook == HookPoint.ON_TURN_START
        assert result.event == event
        assert len(result.results) == 1

    def test_evaluate_rule_condition_match(
        self,
        engine_with_rules: RuleEngine,
        basic_context: RuleContext,
    ) -> None:
        """Rule condition matching is evaluated correctly."""
        # basic_context has turn.number = 5, which matches "context.turn.number > 1"
        event = Event(
            type=EventType.AGENT_TURN_START,
            source="test",
            severity=Severity.INFO,
        )

        result = engine_with_rules.evaluate_hook(
            HookPoint.ON_TURN_START,
            event,
            context=basic_context,
        )

        assert len(result.results) == 1
        assert result.results[0].matched is True
        assert result.first_match is not None
        assert result.first_match.id == "turn-start-rule"

    def test_evaluate_rule_condition_no_match(
        self,
        engine_with_rules: RuleEngine,
    ) -> None:
        """Rule condition that doesn't match returns matched=False."""
        # Create context with turn.number = 1, which doesn't match "context.turn.number > 1"
        context = RuleContext(
            turn=TurnState(number=1, token_usage=0.5, context_usage=0.3, iteration_count=0),
            history=HistoryState(),
            user=UserState(id="user-123"),
            project=ProjectState(id="project-456"),
            state=PluginState(),
        )

        event = Event(
            type=EventType.AGENT_TURN_START,
            source="test",
            severity=Severity.INFO,
        )

        result = engine_with_rules.evaluate_hook(
            HookPoint.ON_TURN_START,
            event,
            context=context,
        )

        assert len(result.results) == 1
        assert result.results[0].matched is False
        assert result.first_match is None

    def test_evaluate_action_executed_on_match(
        self,
        engine_with_rules: RuleEngine,
        dispatcher: ActionDispatcher,
        basic_context: RuleContext,
    ) -> None:
        """Action is executed when rule condition matches."""
        event = Event(
            type=EventType.AGENT_TURN_START,
            source="test",
            severity=Severity.INFO,
        )

        result = engine_with_rules.evaluate_hook(
            HookPoint.ON_TURN_START,
            event,
            context=basic_context,
        )

        assert result.results[0].action_executed is True

        # Check that notification was queued
        notifications = dispatcher.pending_notifications
        assert len(notifications) == 1
        assert "Turn 5 started" in notifications[0]["message"]


# =============================================================================
# T031: Priority Ordering Tests
# =============================================================================


class TestRuleEnginePriorityOrdering:
    """Tests for priority-ordered rule execution."""

    def test_rules_sorted_by_priority_highest_first(
        self,
        engine_with_priorities: RuleEngine,
    ) -> None:
        """Rules are sorted by priority with highest first."""
        rules = engine_with_priorities.get_rules_for_hook(HookPoint.ON_TURN_START)

        # Should have 3 rules: high (900), token-budget (100), low (50)
        assert len(rules) == 3
        assert rules[0].id == "high-priority-rule"
        assert rules[0].priority == 900
        assert rules[1].id == "token-budget-warning"
        assert rules[1].priority == 100
        assert rules[2].id == "low-priority-rule"
        assert rules[2].priority == 50

    def test_first_match_is_highest_priority(
        self,
        engine_with_priorities: RuleEngine,
        basic_context: RuleContext,
    ) -> None:
        """first_match in result is the highest priority matching rule."""
        event = Event(
            type=EventType.AGENT_TURN_START,
            source="test",
            severity=Severity.INFO,
        )

        result = engine_with_priorities.evaluate_hook(
            HookPoint.ON_TURN_START,
            event,
            context=basic_context,
        )

        # All 3 rules should be evaluated
        assert len(result.results) == 3

        # First match should be high priority
        assert result.first_match is not None
        assert result.first_match.id == "high-priority-rule"

    def test_all_matching_rules_evaluated(
        self,
        engine_with_priorities: RuleEngine,
        basic_context: RuleContext,
    ) -> None:
        """All matching rules are evaluated, not just the first."""
        event = Event(
            type=EventType.AGENT_TURN_START,
            source="test",
            severity=Severity.INFO,
        )

        result = engine_with_priorities.evaluate_hook(
            HookPoint.ON_TURN_START,
            event,
            context=basic_context,
        )

        # All 3 rules should match and have actions executed
        matched_count = sum(1 for r in result.results if r.matched)
        assert matched_count == 3

        executed_count = sum(1 for r in result.results if r.action_executed)
        assert executed_count == 3


# =============================================================================
# Rule Management Tests
# =============================================================================


class TestRuleEngineRuleManagement:
    """Tests for rule enable/disable and reload."""

    def test_disable_rule(
        self,
        engine_with_rules: RuleEngine,
    ) -> None:
        """Disabled rules are excluded from evaluation."""
        # Disable the turn-start-rule
        disabled = engine_with_rules.disable_rule("turn-start-rule")
        assert disabled is True

        rules = engine_with_rules.get_rules_for_hook(HookPoint.ON_TURN_START)
        assert len(rules) == 0

    def test_enable_rule(
        self,
        engine_with_rules: RuleEngine,
    ) -> None:
        """Re-enabling a rule includes it in evaluation."""
        engine_with_rules.disable_rule("turn-start-rule")
        rules_before = engine_with_rules.get_rules_for_hook(HookPoint.ON_TURN_START)
        assert len(rules_before) == 0

        enabled = engine_with_rules.enable_rule("turn-start-rule")
        assert enabled is True

        rules_after = engine_with_rules.get_rules_for_hook(HookPoint.ON_TURN_START)
        assert len(rules_after) == 1

    def test_reload_rules(
        self,
        temp_rules_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
    ) -> None:
        """reload_rules re-reads rules from loader."""
        loader = RuleLoader(temp_rules_dir)
        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            auto_subscribe=False,
        )

        initial_count = engine.rule_count
        assert initial_count == 2

        # Add a new rule file
        new_rule = '''
[rule]
id = "new-rule"
name = "New Rule"
description = "Dynamically added"
trigger = "on_session_end"

[condition]
expression = "True"

[action]
type = "log"
message = "Session ended"
'''
        (temp_rules_dir / "new_rule.toml").write_text(new_rule)

        reloaded_count = engine.reload_rules()
        assert reloaded_count == 3

    def test_core_rules_cannot_be_disabled(
        self,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
    ) -> None:
        """Core rules cannot be disabled."""
        core_rule_toml = '''
[rule]
id = "core-rule"
name = "Core Rule"
description = "Cannot be disabled"
trigger = "on_turn_start"
core = true

[condition]
expression = "True"

[action]
type = "log"
message = "Core rule"
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_dir = Path(tmpdir)
            (rules_dir / "core.toml").write_text(core_rule_toml)

            loader = RuleLoader(rules_dir)
            engine = RuleEngine(
                loader=loader,
                evaluator=evaluator,
                dispatcher=dispatcher,
                event_bus=event_bus,
                auto_subscribe=False,
            )

            # Try to disable core rule
            disabled = engine.disable_rule("core-rule")
            assert disabled is False

            # Rule should still be present
            rules = engine.get_rules_for_hook(HookPoint.ON_TURN_START)
            assert len(rules) == 1


# =============================================================================
# Context Builder Tests
# =============================================================================


class TestRuleEngineContextBuilder:
    """Tests for custom context building."""

    def test_custom_context_builder_used(
        self,
        temp_rules_dir: Path,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
    ) -> None:
        """Custom context builder is called for each event."""
        loader = RuleLoader(temp_rules_dir)

        builder_calls = []

        def custom_builder(event: Event) -> RuleContext:
            builder_calls.append(event)
            return RuleContext.create_minimal("test-user", "test-project", turn_number=10)

        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            context_builder=custom_builder,
            auto_subscribe=False,
        )

        event = Event(
            type=EventType.AGENT_TURN_START,
            source="test",
            severity=Severity.INFO,
        )

        engine.evaluate_hook(HookPoint.ON_TURN_START, event)

        assert len(builder_calls) == 1
        assert builder_calls[0] == event


# =============================================================================
# Notification Handling Tests
# =============================================================================


class TestRuleEngineNotifications:
    """Tests for notification accumulation."""

    def test_pending_notifications_accessible(
        self,
        engine_with_rules: RuleEngine,
        basic_context: RuleContext,
    ) -> None:
        """Pending notifications are accessible via the engine."""
        event = Event(
            type=EventType.AGENT_TURN_START,
            source="test",
            severity=Severity.INFO,
        )

        # Evaluate to trigger notification
        engine_with_rules.evaluate_hook(HookPoint.ON_TURN_START, event, context=basic_context)

        notifications = engine_with_rules.pending_notifications
        assert len(notifications) == 1

    def test_clear_notifications(
        self,
        engine_with_rules: RuleEngine,
        basic_context: RuleContext,
    ) -> None:
        """clear_notifications returns and clears pending."""
        event = Event(
            type=EventType.AGENT_TURN_START,
            source="test",
            severity=Severity.INFO,
        )

        engine_with_rules.evaluate_hook(HookPoint.ON_TURN_START, event, context=basic_context)

        cleared = engine_with_rules.clear_notifications()
        assert len(cleared) == 1

        remaining = engine_with_rules.pending_notifications
        assert len(remaining) == 0
