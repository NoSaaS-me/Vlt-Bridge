"""Oracle Plugin System - Rule Engine and Plugin Architecture.

This package provides the rule engine and plugin architecture for the Oracle agent.
It enables reactive and proactive agent behaviors through TOML-based rule definitions
and Lua scripting.

The plugin system uses a tiered complexity model:
- 80% of use cases: TOML rule definitions with simpleeval expressions
- 20% of use cases: Lua scripts via lupa for complex logic

Components:
- rule.py: Rule dataclass and related models
- loader.py: RuleLoader for TOML discovery and validation
- engine.py: RuleEngine for event subscription and rule evaluation
- expression.py: ExpressionEvaluator using simpleeval
- lua_sandbox.py: LuaSandbox for Lua script execution via lupa
- actions.py: ActionDispatcher for executing rule actions
- context.py: RuleContext builder for rule evaluation

Subdirectories:
- rules/: Built-in rule TOML files
- scripts/: Lua scripts for complex rule logic

Example usage:
    from services.plugins import Rule, RuleAction, ActionType, HookPoint
    from services.plugins import RuleContext, TurnState
    from services.plugins import RuleLoader, ExpressionEvaluator, ActionDispatcher

    # Load rules from TOML files
    loader = RuleLoader(Path("rules/"))
    rules = loader.load_all()

    # Evaluate conditions
    evaluator = ExpressionEvaluator()
    for rule in rules:
        if evaluator.evaluate(rule.condition, context):
            dispatcher.dispatch(rule.action, context)
"""

# Rule definitions
from .rule import (
    ActionType,
    HookPoint,
    InjectionPoint,
    Priority,
    Rule,
    RuleAction,
)

# Context definitions
from .context import (
    EventData,
    HistoryState,
    PluginState,
    ProjectState,
    RuleContext,
    RuleContextBuilder,
    ToolCallRecord,
    ToolResult,
    TurnState,
    UserState,
)

# State persistence
from .state import (
    PluginStateService,
    get_plugin_state_service,
    reset_plugin_state_service,
)

# Loader
from .loader import (
    RuleLoader,
    RuleLoadError,
)

# Plugin types
from .plugin import (
    Plugin,
    PluginSetting,
)

# Plugin loader
from .plugin_loader import (
    PluginLoader,
    PluginLoadError,
    PluginDependencyError,
    DEFAULT_CAPABILITIES,
)

# Expression evaluator
from .expression import (
    ExpressionEvaluator,
    ExpressionError,
)

# Action dispatcher
from .actions import (
    ActionDispatcher,
    ActionError,
)

# Rule engine
from .engine import (
    RuleEngine,
    RuleEvaluationResult,
    HookEvaluationResult,
    ContextBuilder,
    EVENT_TO_HOOK,
    SUBSCRIBED_EVENTS,
)

# Lua sandbox
from .lua_sandbox import (
    LuaSandbox,
    LuaSandboxError,
    LuaExecutionError,
    LuaTimeoutError,
    LuaMemoryError,
    execute_script,
)

__all__ = [
    # Enums
    "HookPoint",
    "ActionType",
    "Priority",
    "InjectionPoint",
    # Rule types
    "Rule",
    "RuleAction",
    # Plugin types
    "Plugin",
    "PluginSetting",
    # Context types
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
    # State persistence
    "PluginStateService",
    "get_plugin_state_service",
    "reset_plugin_state_service",
    # Rule loader
    "RuleLoader",
    "RuleLoadError",
    # Plugin loader
    "PluginLoader",
    "PluginLoadError",
    "PluginDependencyError",
    "DEFAULT_CAPABILITIES",
    # Expression evaluator
    "ExpressionEvaluator",
    "ExpressionError",
    # Action dispatcher
    "ActionDispatcher",
    "ActionError",
    # Rule engine
    "RuleEngine",
    "RuleEvaluationResult",
    "HookEvaluationResult",
    "ContextBuilder",
    "EVENT_TO_HOOK",
    "SUBSCRIBED_EVENTS",
    # Lua sandbox
    "LuaSandbox",
    "LuaSandboxError",
    "LuaExecutionError",
    "LuaTimeoutError",
    "LuaMemoryError",
    "execute_script",
]
