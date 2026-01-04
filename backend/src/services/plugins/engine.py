"""Rule Engine for event-driven rule evaluation and execution.

This module provides the RuleEngine class which:
- Subscribes to ANS EventBus for lifecycle events
- Maps events to HookPoints
- Evaluates rules in priority order
- Executes actions via ActionDispatcher

The RuleEngine is the central orchestrator that connects:
- EventBus (ANS) -> Rule triggers
- RuleLoader -> Rule definitions
- ExpressionEvaluator -> Condition evaluation
- ActionDispatcher -> Action execution

Performance features (T106, T107):
- Expression caching for frequently evaluated rules
- Optional timing logs for debugging
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..ans.bus import EventBus
from ..ans.event import Event, EventType, Severity

from .rule import HookPoint, Rule, RuleAction, ActionType, Priority, InjectionPoint
from .context import EventData, RuleContext
from .loader import RuleLoader
from .expression import ExpressionEvaluator, ExpressionError
from .actions import ActionDispatcher
from .lua_sandbox import (
    LuaSandbox,
    LuaSandboxError,
    LuaExecutionError,
    LuaTimeoutError,
)


logger = logging.getLogger(__name__)


# Mapping from ANS EventType to HookPoint
EVENT_TO_HOOK: Dict[str, HookPoint] = {
    EventType.QUERY_START: HookPoint.ON_QUERY_START,
    EventType.AGENT_TURN_START: HookPoint.ON_TURN_START,
    EventType.AGENT_TURN_END: HookPoint.ON_TURN_END,
    EventType.TOOL_CALL_PENDING: HookPoint.ON_TOOL_CALL,
    EventType.TOOL_CALL_SUCCESS: HookPoint.ON_TOOL_COMPLETE,
    EventType.TOOL_CALL_FAILURE: HookPoint.ON_TOOL_FAILURE,
    EventType.TOOL_CALL_TIMEOUT: HookPoint.ON_TOOL_FAILURE,
    EventType.SESSION_END: HookPoint.ON_SESSION_END,
}

# Event types that the RuleEngine subscribes to
SUBSCRIBED_EVENTS = list(EVENT_TO_HOOK.keys())


@dataclass
class RuleEvaluationResult:
    """Result of evaluating a rule.

    Attributes:
        rule: The rule that was evaluated.
        matched: Whether the rule condition matched.
        action_executed: Whether the action was executed.
        error: Error message if evaluation or execution failed.
        evaluation_time_ms: Time taken to evaluate in milliseconds (T107).
    """

    rule: Rule
    matched: bool
    action_executed: bool = False
    error: Optional[str] = None
    evaluation_time_ms: Optional[float] = None


@dataclass
class HookEvaluationResult:
    """Result of evaluating all rules for a hook point.

    Attributes:
        hook: The hook point that was triggered.
        event: The event that triggered the hook.
        results: Results for each rule evaluated.
        first_match: The first matching rule (if any).
        total_time_ms: Total time to evaluate all rules in milliseconds (T107).
    """

    hook: HookPoint
    event: Event
    results: List[RuleEvaluationResult] = field(default_factory=list)
    first_match: Optional[Rule] = None
    total_time_ms: Optional[float] = None


# Type alias for context builder callback
ContextBuilder = Callable[[Event], Optional[RuleContext]]


class RuleEngine:
    """Engine for event-driven rule evaluation and execution.

    The RuleEngine subscribes to ANS events and evaluates rules when
    matching events are received. Rules are organized by hook point
    and evaluated in priority order (highest first).

    Example:
        from pathlib import Path
        from services.ans.bus import EventBus
        from services.plugins.engine import RuleEngine
        from services.plugins.loader import RuleLoader
        from services.plugins.expression import ExpressionEvaluator
        from services.plugins.actions import ActionDispatcher

        # Create components
        event_bus = EventBus()
        loader = RuleLoader(Path("rules/"))
        evaluator = ExpressionEvaluator()
        dispatcher = ActionDispatcher(event_bus=event_bus)

        # Create engine with context builder
        def build_context(event: Event) -> RuleContext:
            return RuleContext.create_minimal("user1", "project1")

        engine = RuleEngine(
            loader=loader,
            evaluator=evaluator,
            dispatcher=dispatcher,
            event_bus=event_bus,
            context_builder=build_context,
        )

        # Start listening
        engine.start()

        # ... events from event_bus will trigger rules ...

        engine.stop()
    """

    def __init__(
        self,
        loader: RuleLoader,
        evaluator: ExpressionEvaluator,
        dispatcher: ActionDispatcher,
        event_bus: EventBus,
        context_builder: Optional[ContextBuilder] = None,
        auto_subscribe: bool = True,
        lua_timeout_seconds: float = 5.0,
        lua_max_memory_mb: int = 100,
    ) -> None:
        """Initialize the rule engine.

        Args:
            loader: RuleLoader for loading rules from TOML files.
            evaluator: ExpressionEvaluator for condition evaluation.
            dispatcher: ActionDispatcher for executing actions.
            event_bus: EventBus for receiving events.
            context_builder: Callback to build RuleContext from events.
                            If None, a minimal context is created.
            auto_subscribe: Whether to subscribe to events on start().
            lua_timeout_seconds: Timeout for Lua script execution (default 5.0).
            lua_max_memory_mb: Memory limit for Lua scripts in MB (default 100).
        """
        self._loader = loader
        self._evaluator = evaluator
        self._dispatcher = dispatcher
        self._event_bus = event_bus
        self._context_builder = context_builder
        self._auto_subscribe = auto_subscribe

        # Lua sandbox for script execution
        self._lua_sandbox = LuaSandbox(
            timeout_seconds=lua_timeout_seconds,
            max_memory_mb=lua_max_memory_mb,
        )

        # Rules organized by hook point
        self._rules_by_hook: Dict[HookPoint, List[Rule]] = {}

        # Subscription state
        self._subscribed = False
        self._subscribed_handlers: Dict[str, Callable[[Event], None]] = {}

        # Disabled rules (by qualified_id)
        self._disabled_rules: set[str] = set()

        # Load and organize rules
        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from the loader and organize by hook point."""
        try:
            rules = self._loader.load_all(skip_invalid=True)
            self._organize_rules(rules)
            logger.info(f"RuleEngine loaded {len(rules)} rules")
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            self._rules_by_hook = {}

    def _organize_rules(self, rules: List[Rule]) -> None:
        """Organize rules by hook point and sort by priority.

        Args:
            rules: List of rules to organize.
        """
        # Clear existing rules
        self._rules_by_hook = {hook: [] for hook in HookPoint}

        # Group by hook point
        for rule in rules:
            if rule.enabled:
                self._rules_by_hook[rule.trigger].append(rule)

        # Sort each group by priority (highest first)
        for hook in HookPoint:
            self._rules_by_hook[hook].sort(key=lambda r: r.priority, reverse=True)

        # Log rule counts per hook
        for hook in HookPoint:
            count = len(self._rules_by_hook[hook])
            if count > 0:
                logger.debug(f"Hook {hook.value}: {count} rules")

    def reload_rules(self) -> int:
        """Reload rules from the loader.

        Returns:
            Number of rules loaded.
        """
        self._load_rules()
        return sum(len(rules) for rules in self._rules_by_hook.values())

    def get_rules_for_hook(self, hook: HookPoint) -> List[Rule]:
        """Get all enabled rules for a hook point (sorted by priority).

        Args:
            hook: The hook point to get rules for.

        Returns:
            List of rules sorted by priority (highest first).
        """
        return [
            r for r in self._rules_by_hook.get(hook, [])
            if r.qualified_id not in self._disabled_rules
        ]

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule by its qualified ID.

        Args:
            rule_id: Qualified rule ID to disable.

        Returns:
            True if rule was found and disabled.
        """
        # Find the rule to check if it's core
        for rules in self._rules_by_hook.values():
            for rule in rules:
                if rule.qualified_id == rule_id:
                    if rule.core:
                        logger.warning(f"Cannot disable core rule: {rule_id}")
                        return False
                    self._disabled_rules.add(rule_id)
                    logger.info(f"Disabled rule: {rule_id}")
                    return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a previously disabled rule.

        Args:
            rule_id: Qualified rule ID to enable.

        Returns:
            True if rule was found and enabled.
        """
        if rule_id in self._disabled_rules:
            self._disabled_rules.discard(rule_id)
            logger.info(f"Enabled rule: {rule_id}")
            return True
        return False

    def load_disabled_rules_for_user(self, user_id: str) -> None:
        """Load disabled rules from user settings.

        This method should be called when initializing a RuleEngine
        for a specific user session (T066).

        Args:
            user_id: User identifier to load disabled rules for.
        """
        try:
            from ..user_settings import UserSettingsService
            settings_service = UserSettingsService()
            disabled = settings_service.get_disabled_rules(user_id)
            self._disabled_rules = set(disabled)
            logger.info(f"Loaded {len(disabled)} disabled rules for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to load disabled rules for user {user_id}: {e}")
            self._disabled_rules = set()

    def set_disabled_rules(self, disabled_rules: set[str]) -> None:
        """Set the disabled rules set directly.

        Args:
            disabled_rules: Set of qualified rule IDs to disable.
        """
        self._disabled_rules = disabled_rules
        logger.debug(f"Set {len(disabled_rules)} disabled rules")

    def start(self) -> None:
        """Start the rule engine by subscribing to events.

        This subscribes to all events that map to hook points.
        """
        if self._subscribed:
            logger.warning("RuleEngine already started")
            return

        if not self._auto_subscribe:
            logger.info("RuleEngine started (auto_subscribe=False)")
            return

        # Subscribe to each event type
        for event_type in SUBSCRIBED_EVENTS:
            handler = self._create_handler(event_type)
            self._event_bus.subscribe(event_type, handler)
            self._subscribed_handlers[event_type] = handler
            logger.debug(f"Subscribed to {event_type}")

        self._subscribed = True
        logger.info(f"RuleEngine started, subscribed to {len(SUBSCRIBED_EVENTS)} event types")

    def stop(self) -> None:
        """Stop the rule engine by unsubscribing from events."""
        if not self._subscribed:
            return

        for event_type, handler in self._subscribed_handlers.items():
            self._event_bus.unsubscribe(event_type, handler)

        self._subscribed_handlers.clear()
        self._subscribed = False
        logger.info("RuleEngine stopped")

    def _create_handler(self, event_type: str) -> Callable[[Event], None]:
        """Create an event handler for a specific event type.

        Args:
            event_type: The event type to handle.

        Returns:
            Event handler function.
        """
        def handler(event: Event) -> None:
            self._on_event(event)

        return handler

    def _on_event(self, event: Event) -> None:
        """Handle an incoming event.

        Maps the event to a hook point and evaluates matching rules.

        Args:
            event: The event to process.
        """
        # Map event type to hook point
        hook = EVENT_TO_HOOK.get(event.type)
        if hook is None:
            logger.debug(f"No hook mapping for event type: {event.type}")
            return

        # Evaluate rules for this hook
        self.evaluate_hook(hook, event)

    def evaluate_hook(
        self,
        hook: HookPoint,
        event: Event,
        context: Optional[RuleContext] = None,
    ) -> HookEvaluationResult:
        """Evaluate all rules for a hook point.

        Rules are evaluated in priority order. For each matching rule,
        its action is executed via the ActionDispatcher.

        Includes timing measurement for performance debugging (T107).

        Args:
            hook: The hook point being triggered.
            event: The event that triggered the hook.
            context: Optional pre-built context. If None, uses context_builder.

        Returns:
            HookEvaluationResult with details of all evaluations.
        """
        start_time = time.perf_counter()
        result = HookEvaluationResult(hook=hook, event=event)

        # Get rules for this hook (already sorted by priority)
        rules = self.get_rules_for_hook(hook)
        if not rules:
            logger.debug(f"No rules for hook {hook.value}")
            result.total_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        # Build context if not provided
        if context is None:
            context = self._build_context(event)
            if context is None:
                logger.warning(f"Could not build context for hook {hook.value}")
                result.total_time_ms = (time.perf_counter() - start_time) * 1000
                return result

        # Evaluate rules in priority order
        logger.debug(f"Evaluating {len(rules)} rules for hook {hook.value}")
        for rule in rules:
            eval_result = self._evaluate_rule(rule, context)
            result.results.append(eval_result)

            if eval_result.matched and result.first_match is None:
                result.first_match = rule

        # Record total timing (T107)
        total_time_ms = (time.perf_counter() - start_time) * 1000
        result.total_time_ms = total_time_ms

        # Log summary at INFO level if any rules matched, DEBUG otherwise
        matched_count = sum(1 for r in result.results if r.matched)
        if matched_count > 0:
            logger.info(
                f"Hook {hook.value}: evaluated {len(rules)} rules in {total_time_ms:.3f}ms, "
                f"{matched_count} matched"
            )
        else:
            logger.debug(
                f"Hook {hook.value}: evaluated {len(rules)} rules in {total_time_ms:.3f}ms, "
                f"none matched"
            )

        return result

    def _evaluate_rule(
        self,
        rule: Rule,
        context: RuleContext,
    ) -> RuleEvaluationResult:
        """Evaluate a single rule against the context.

        Supports both expression-based conditions (simpleeval) and
        Lua script-based evaluation. For script rules, the script
        return value can define the action to execute.

        Includes timing measurement for performance debugging (T107).

        Args:
            rule: The rule to evaluate.
            context: The context for evaluation.

        Returns:
            RuleEvaluationResult with evaluation details.
        """
        start_time = time.perf_counter()
        result = RuleEvaluationResult(rule=rule, matched=False)

        # Handle Lua scripts (T057)
        if rule.script:
            script_result = self._evaluate_script_rule(rule, context)
            # Add timing to script result
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            script_result.evaluation_time_ms = elapsed_ms
            logger.debug(
                f"Rule {rule.id} (script) evaluated in {elapsed_ms:.3f}ms "
                f"(matched: {script_result.matched})"
            )
            return script_result

        # Evaluate expression condition
        if not rule.condition:
            result.error = "Rule has no condition or script"
            result.evaluation_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        try:
            matched = self._evaluator.evaluate(rule.condition, context)
        except ExpressionError as e:
            logger.warning(f"Rule {rule.id} condition error: {e}")
            result.error = str(e)
            result.evaluation_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        result.matched = matched

        if not matched:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result.evaluation_time_ms = elapsed_ms
            logger.debug(
                f"Rule {rule.id} evaluated in {elapsed_ms:.3f}ms (not matched)"
            )
            return result

        # Execute action
        if rule.action:
            logger.info(f"Rule {rule.id} matched, executing action {rule.action.type.value}")
            try:
                success = self._dispatcher.dispatch(rule.action, context)
                result.action_executed = success
                if not success:
                    result.error = "Action execution failed"
            except Exception as e:
                logger.error(f"Rule {rule.id} action error: {e}")
                result.error = str(e)

        # Record final timing (T107)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result.evaluation_time_ms = elapsed_ms
        logger.debug(
            f"Rule {rule.id} evaluated in {elapsed_ms:.3f}ms "
            f"(matched: {result.matched}, action: {result.action_executed})"
        )

        return result

    def _evaluate_script_rule(
        self,
        rule: Rule,
        context: RuleContext,
    ) -> RuleEvaluationResult:
        """Evaluate a rule that uses a Lua script (T057).

        Lua scripts can return:
        - nil/None: Rule did not match (no action executed)
        - true: Rule matched, execute the rule's defined action
        - dict/table: Rule matched, action is defined in the returned table

        Args:
            rule: The rule with a script path.
            context: The context for evaluation.

        Returns:
            RuleEvaluationResult with evaluation details.
        """
        result = RuleEvaluationResult(rule=rule, matched=False)

        # Load script content from file
        script_content = self._load_script(rule.script, rule.source_path)
        if script_content is None:
            result.error = f"Failed to load script: {rule.script}"
            logger.error(f"Rule {rule.id}: {result.error}")
            return result

        # Execute script in sandbox (T058 - error handling)
        try:
            script_result = self._lua_sandbox.execute(script_content, context)
        except LuaTimeoutError as e:
            result.error = f"Script timeout: {e}"
            logger.warning(f"Rule {rule.id} script timeout: {e}")
            return result
        except LuaExecutionError as e:
            result.error = f"Script error: {e}"
            logger.warning(f"Rule {rule.id} script error: {e}")
            return result
        except LuaSandboxError as e:
            result.error = f"Sandbox error: {e}"
            logger.error(f"Rule {rule.id} sandbox error: {e}")
            return result
        except Exception as e:
            result.error = f"Unexpected script error: {e}"
            logger.error(f"Rule {rule.id} unexpected error: {e}")
            return result

        # Interpret script result
        if script_result is None:
            # Script returned nil - rule did not match
            logger.debug(f"Rule {rule.id} script returned nil (no match)")
            return result

        # Script returned something - rule matched
        result.matched = True

        # Determine action to execute
        action = self._resolve_script_action(rule, script_result)
        if action is None:
            logger.debug(f"Rule {rule.id} matched but no action to execute")
            return result

        # Execute the action
        logger.info(f"Rule {rule.id} script matched, executing action {action.type.value}")
        try:
            success = self._dispatcher.dispatch(action, context)
            result.action_executed = success
            if not success:
                result.error = "Action execution failed"
        except Exception as e:
            logger.error(f"Rule {rule.id} action error: {e}")
            result.error = str(e)

        return result

    def _load_script(
        self,
        script_path: str,
        rule_source_path: str,
    ) -> Optional[str]:
        """Load Lua script content from file.

        Args:
            script_path: Relative path to the script file.
            rule_source_path: Path where the rule TOML was loaded from.

        Returns:
            Script content as string, or None if loading failed.
        """
        try:
            # Resolve script path relative to rule source or loader base
            if rule_source_path:
                rule_dir = Path(rule_source_path).parent
                full_path = rule_dir / script_path
            else:
                full_path = self._loader.rules_dir / script_path

            # Normalize and resolve path
            full_path = full_path.resolve()

            # Read script content
            if not full_path.exists():
                logger.error(f"Script file not found: {full_path}")
                return None

            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()

        except Exception as e:
            logger.error(f"Failed to load script {script_path}: {e}")
            return None

    def _resolve_script_action(
        self,
        rule: Rule,
        script_result: Any,
    ) -> Optional[RuleAction]:
        """Resolve the action to execute from a script result.

        Script can return:
        - True (bool): Use the rule's defined action
        - dict with 'type' key: Parse as action definition

        Args:
            rule: The rule that was evaluated.
            script_result: Return value from the Lua script.

        Returns:
            RuleAction to execute, or None if no valid action.
        """
        # If script returned True, use the rule's defined action
        if script_result is True:
            return rule.action

        # If script returned a dict, try to parse as action
        if isinstance(script_result, dict):
            return self._parse_action_from_dict(script_result)

        # Other truthy values - use rule's defined action
        if script_result:
            return rule.action

        return None

    def _parse_action_from_dict(self, action_dict: Dict[str, Any]) -> Optional[RuleAction]:
        """Parse a dict (from Lua table) into a RuleAction.

        Expected format:
        {
            "type": "notify_self",  # Required
            "message": "...",       # For notify_self
            "priority": "high",     # Optional
            "category": "warning",  # Optional
            ...
        }

        Args:
            action_dict: Dictionary with action configuration.

        Returns:
            RuleAction if valid, None otherwise.
        """
        try:
            action_type_str = action_dict.get("type")
            if not action_type_str:
                logger.warning("Script returned dict without 'type' key")
                return None

            # Parse action type
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                logger.warning(f"Invalid action type from script: {action_type_str}")
                return None

            # Parse priority
            priority = Priority.NORMAL
            priority_str = action_dict.get("priority")
            if priority_str:
                try:
                    priority = Priority(priority_str)
                except ValueError:
                    logger.warning(f"Invalid priority from script: {priority_str}")

            # Parse injection point
            deliver_at = InjectionPoint.TURN_START
            deliver_at_str = action_dict.get("deliver_at")
            if deliver_at_str:
                try:
                    deliver_at = InjectionPoint(deliver_at_str)
                except ValueError:
                    logger.warning(f"Invalid deliver_at from script: {deliver_at_str}")

            return RuleAction(
                type=action_type,
                message=action_dict.get("message"),
                category=action_dict.get("category"),
                priority=priority,
                deliver_at=deliver_at,
                level=action_dict.get("level", "info"),
                key=action_dict.get("key"),
                value=action_dict.get("value"),
                event_type=action_dict.get("event_type"),
                payload=action_dict.get("payload"),
            )

        except Exception as e:
            logger.error(f"Failed to parse action from script result: {e}")
            return None

    def _build_context(self, event: Event) -> Optional[RuleContext]:
        """Build RuleContext from an event.

        Uses the context_builder callback if provided, otherwise
        creates a minimal context.

        Args:
            event: The event to build context from.

        Returns:
            RuleContext or None if building fails.
        """
        if self._context_builder:
            try:
                return self._context_builder(event)
            except Exception as e:
                logger.error(f"Context builder error: {e}")
                return None

        # Create minimal context with event data
        event_data = EventData(
            type=event.type,
            source=event.source,
            severity=event.severity.value,
            payload=event.payload,
            timestamp=event.timestamp,
        )

        return RuleContext(
            turn=context_module.TurnState(
                number=1,
                token_usage=0.0,
                context_usage=0.0,
                iteration_count=0,
            ),
            history=context_module.HistoryState(),
            user=context_module.UserState(id="unknown"),
            project=context_module.ProjectState(id="unknown"),
            state=context_module.PluginState(),
            event=event_data,
        )

    @property
    def is_running(self) -> bool:
        """Check if the engine is running (subscribed to events)."""
        return self._subscribed

    @property
    def rule_count(self) -> int:
        """Get total number of enabled rules."""
        return sum(len(rules) for rules in self._rules_by_hook.values())

    @property
    def pending_notifications(self) -> List[Dict[str, Any]]:
        """Get pending notifications from the dispatcher."""
        return self._dispatcher.pending_notifications

    def clear_notifications(self) -> List[Dict[str, Any]]:
        """Clear and return pending notifications."""
        return self._dispatcher.clear_notifications()


# Import context module for minimal context building
from . import context as context_module


__all__ = [
    "RuleEngine",
    "RuleEvaluationResult",
    "HookEvaluationResult",
    "ContextBuilder",
    "EVENT_TO_HOOK",
    "SUBSCRIBED_EVENTS",
]
