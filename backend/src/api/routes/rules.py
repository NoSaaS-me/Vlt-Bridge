"""HTTP API routes for rule and plugin management.

This module provides endpoints for listing, viewing, toggling, and testing rules
and plugins in the Oracle Plugin System.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ...models.rule import (
    ErrorResponse,
    HookPointEnum,
    RuleDetail,
    RuleInfo,
    RuleListResponse,
    RuleTestRequest,
    RuleTestResponse,
    RuleToggleRequest,
    RuleAction as RuleActionModel,
    ActionTypeEnum,
    PriorityEnum,
    InjectionPointEnum,
    # Plugin models
    PluginInfo,
    PluginDetail,
    PluginListResponse,
    PluginSettingSchema,
    PluginSettingsUpdateRequest,
)
from ...services.plugins.rule import Rule, HookPoint, ActionType, Priority, InjectionPoint
from ...services.plugins.loader import RuleLoader
from ...services.plugins.plugin import Plugin, PluginSetting
from ...services.plugins.plugin_loader import PluginLoader, PluginLoadError, PluginDependencyError
from ...services.plugins.expression import ExpressionEvaluator, ExpressionError
from ...services.plugins.context import RuleContext
from ...services.plugins.state import get_plugin_state_service
from ...services.user_settings import UserSettingsService
from ..middleware import AuthContext, require_auth_context

logger = logging.getLogger(__name__)

router = APIRouter()

# Default directories - relative to backend/src/services/plugins/
PLUGINS_BASE_DIR = Path(__file__).parent.parent.parent / "services" / "plugins"
RULES_DIR = PLUGINS_BASE_DIR / "rules"
PLUGINS_DIR = PLUGINS_BASE_DIR / "plugins"


def _get_rule_loader() -> RuleLoader:
    """Get the rule loader instance."""
    return RuleLoader(RULES_DIR)


def _get_plugin_loader() -> PluginLoader:
    """Get the plugin loader instance."""
    return PluginLoader(PLUGINS_DIR)


def _plugin_to_info(plugin: Plugin) -> PluginInfo:
    """Convert internal Plugin to PluginInfo response model."""
    return PluginInfo(
        id=plugin.id,
        name=plugin.name,
        version=plugin.version,
        description=plugin.description,
        rule_count=plugin.rule_count,
        enabled=plugin.enabled,
    )


def _plugin_setting_to_schema(setting: PluginSetting) -> PluginSettingSchema:
    """Convert internal PluginSetting to PluginSettingSchema response model."""
    return PluginSettingSchema(
        name=setting.name,
        type=setting.type,
        default=setting.default,
        description=setting.description,
        min_value=setting.min_value,
        max_value=setting.max_value,
        options=setting.options,
    )


def _plugin_to_detail(plugin: Plugin, disabled_rules: set[str]) -> PluginDetail:
    """Convert internal Plugin to PluginDetail response model."""
    # Convert rules
    rule_infos = [_rule_to_info(rule, disabled_rules) for rule in plugin.rules]

    # Convert settings
    settings_schema = {
        setting_id: _plugin_setting_to_schema(setting)
        for setting_id, setting in plugin.settings.items()
    }

    return PluginDetail(
        id=plugin.id,
        name=plugin.name,
        version=plugin.version,
        description=plugin.description,
        rule_count=plugin.rule_count,
        enabled=plugin.enabled,
        rules=rule_infos,
        requires=plugin.requires,
        settings_schema=settings_schema,
        source_dir=plugin.source_dir,
    )


def _rule_to_info(rule: Rule, disabled_rules: set[str]) -> RuleInfo:
    """Convert internal Rule to RuleInfo response model."""
    is_disabled = rule.qualified_id in disabled_rules
    return RuleInfo(
        id=rule.qualified_id,
        name=rule.name,
        description=rule.description or None,
        trigger=HookPointEnum(rule.trigger.value),
        enabled=rule.enabled and not is_disabled,
        core=rule.core,
        priority=rule.priority,
        plugin_id=rule.plugin_id,
    )


def _rule_to_detail(rule: Rule, disabled_rules: set[str]) -> RuleDetail:
    """Convert internal Rule to RuleDetail response model."""
    is_disabled = rule.qualified_id in disabled_rules

    # Convert action if present
    action_model = None
    if rule.action:
        action_model = RuleActionModel(
            type=ActionTypeEnum(rule.action.type.value),
            message=rule.action.message,
            category=rule.action.category,
            priority=PriorityEnum(rule.action.priority.value),
            deliver_at=InjectionPointEnum(rule.action.deliver_at.value),
        )

    return RuleDetail(
        id=rule.qualified_id,
        name=rule.name,
        description=rule.description or None,
        trigger=HookPointEnum(rule.trigger.value),
        enabled=rule.enabled and not is_disabled,
        core=rule.core,
        priority=rule.priority,
        plugin_id=rule.plugin_id,
        version=rule.version,
        condition=rule.condition,
        script=rule.script,
        action=action_model,
        source_path=rule.source_path,
    )


@router.get("/api/rules", response_model=RuleListResponse)
async def list_rules(
    plugin_id: Optional[str] = Query(None, description="Filter by plugin ID"),
    trigger: Optional[HookPointEnum] = Query(None, description="Filter by hook point trigger"),
    enabled_only: bool = Query(False, description="Only return enabled rules"),
    auth: AuthContext = Depends(require_auth_context),
):
    """List all registered rules with their enabled/disabled status.

    Returns all rules loaded from TOML files, with user-specific disabled
    status applied.
    """
    user_id = auth.user_id

    try:
        # Load all rules
        loader = _get_rule_loader()
        rules = loader.load_all(skip_invalid=True)

        # Get user's disabled rules
        settings_service = UserSettingsService()
        disabled_rules = set(settings_service.get_disabled_rules(user_id))

        # Convert to response models
        rule_infos = [_rule_to_info(rule, disabled_rules) for rule in rules]

        # Apply filters
        if plugin_id is not None:
            rule_infos = [r for r in rule_infos if r.plugin_id == plugin_id]

        if trigger is not None:
            rule_infos = [r for r in rule_infos if r.trigger == trigger]

        if enabled_only:
            rule_infos = [r for r in rule_infos if r.enabled]

        return RuleListResponse(
            rules=rule_infos,
            total=len(rule_infos),
        )

    except Exception as e:
        logger.error(f"Failed to list rules: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": str(e)},
        )


@router.get("/api/rules/{rule_id}", response_model=RuleDetail)
async def get_rule(
    rule_id: str,
    auth: AuthContext = Depends(require_auth_context),
):
    """Get detailed information about a specific rule.

    Args:
        rule_id: The qualified rule ID (e.g., "plugin-id:rule-name" or "rule-name")
    """
    user_id = auth.user_id

    try:
        # Load all rules
        loader = _get_rule_loader()
        rules = loader.load_all(skip_invalid=True)

        # Find the requested rule
        target_rule = None
        for rule in rules:
            if rule.qualified_id == rule_id:
                target_rule = rule
                break

        if target_rule is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"Rule not found: {rule_id}"},
            )

        # Get user's disabled rules
        settings_service = UserSettingsService()
        disabled_rules = set(settings_service.get_disabled_rules(user_id))

        return _rule_to_detail(target_rule, disabled_rules)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get rule {rule_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": str(e)},
        )


@router.post("/api/rules/{rule_id}/toggle", response_model=RuleInfo)
async def toggle_rule(
    rule_id: str,
    request: RuleToggleRequest,
    auth: AuthContext = Depends(require_auth_context),
):
    """Toggle a rule enabled/disabled.

    Core rules cannot be disabled. Attempting to disable a core rule will
    return a 400 error.

    Args:
        rule_id: The qualified rule ID
        request: Toggle request with enabled state
    """
    user_id = auth.user_id

    try:
        # Load all rules
        loader = _get_rule_loader()
        rules = loader.load_all(skip_invalid=True)

        # Find the requested rule
        target_rule = None
        for rule in rules:
            if rule.qualified_id == rule_id:
                target_rule = rule
                break

        if target_rule is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"Rule not found: {rule_id}"},
            )

        # Check if core rule
        if target_rule.core and not request.enabled:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "cannot_disable_core_rule",
                    "detail": f"Cannot disable core rule: {rule_id}",
                },
            )

        # Get user settings service
        settings_service = UserSettingsService()
        disabled_rules = set(settings_service.get_disabled_rules(user_id))

        # Update disabled state
        if request.enabled:
            # Remove from disabled list
            disabled_rules.discard(rule_id)
        else:
            # Add to disabled list
            disabled_rules.add(rule_id)

        # Save updated disabled rules
        settings_service.set_disabled_rules(user_id, list(disabled_rules))

        logger.info(f"User {user_id} toggled rule {rule_id} to enabled={request.enabled}")

        return _rule_to_info(target_rule, disabled_rules)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle rule {rule_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": str(e)},
        )


@router.post("/api/rules/{rule_id}/test", response_model=RuleTestResponse)
async def test_rule(
    rule_id: str,
    request: Optional[RuleTestRequest] = None,
    auth: AuthContext = Depends(require_auth_context),
):
    """Test a rule manually with mock context.

    This endpoint evaluates the rule's condition without executing
    the action. Useful for testing rule configurations.

    Args:
        rule_id: The qualified rule ID
        request: Optional context overrides for testing
    """
    user_id = auth.user_id

    try:
        # Load all rules
        loader = _get_rule_loader()
        rules = loader.load_all(skip_invalid=True)

        # Find the requested rule
        target_rule = None
        for rule in rules:
            if rule.qualified_id == rule_id:
                target_rule = rule
                break

        if target_rule is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"Rule not found: {rule_id}"},
            )

        # Check if rule has a condition (not script-based)
        if not target_rule.condition:
            return RuleTestResponse(
                condition_result=False,
                action_would_execute=False,
                evaluation_time_ms=0.0,
                error="Rule uses a Lua script, which cannot be tested via API",
            )

        # Create minimal test context
        context = RuleContext.create_minimal(user_id, "test-project")

        # Apply context overrides if provided
        if request and request.context_override:
            # Override turn state values if provided
            overrides = request.context_override
            if "turn_number" in overrides:
                context.turn.number = overrides["turn_number"]
            if "token_usage" in overrides:
                context.turn.token_usage = overrides["token_usage"]
            if "context_usage" in overrides:
                context.turn.context_usage = overrides["context_usage"]
            if "iteration_count" in overrides:
                context.turn.iteration_count = overrides["iteration_count"]

        # Evaluate the condition
        evaluator = ExpressionEvaluator()

        start_time = time.perf_counter()
        error_msg = None
        condition_result = False

        try:
            condition_result = evaluator.evaluate(target_rule.condition, context)
        except ExpressionError as e:
            error_msg = str(e)
        except Exception as e:
            error_msg = f"Unexpected error: {e}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RuleTestResponse(
            condition_result=condition_result,
            action_would_execute=condition_result and target_rule.action is not None,
            evaluation_time_ms=round(elapsed_ms, 3),
            error=error_msg,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test rule {rule_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": str(e)},
        )


# ============================================================================
# Plugin Endpoints (T081-T084)
# ============================================================================


@router.get("/api/plugins", response_model=PluginListResponse)
async def list_plugins(
    auth: AuthContext = Depends(require_auth_context),
):
    """List all registered plugins.

    Returns all plugins loaded from the plugins directory.
    """
    try:
        # Load all plugins
        loader = _get_plugin_loader()
        plugins = loader.load_all(skip_invalid=True)

        # Convert to response models
        plugin_infos = [_plugin_to_info(plugin) for plugin in plugins]

        return PluginListResponse(plugins=plugin_infos)

    except Exception as e:
        logger.error(f"Failed to list plugins: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": str(e)},
        )


@router.get("/api/plugins/{plugin_id}", response_model=PluginDetail)
async def get_plugin(
    plugin_id: str,
    auth: AuthContext = Depends(require_auth_context),
):
    """Get detailed information about a specific plugin.

    Args:
        plugin_id: The plugin identifier.
    """
    user_id = auth.user_id

    try:
        # Load all plugins
        loader = _get_plugin_loader()
        plugins = loader.load_all(skip_invalid=True)

        # Find the requested plugin
        target_plugin = None
        for plugin in plugins:
            if plugin.id == plugin_id:
                target_plugin = plugin
                break

        if target_plugin is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"Plugin not found: {plugin_id}"},
            )

        # Get user's disabled rules
        settings_service = UserSettingsService()
        disabled_rules = set(settings_service.get_disabled_rules(user_id))

        return _plugin_to_detail(target_plugin, disabled_rules)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get plugin {plugin_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": str(e)},
        )


@router.get("/api/plugins/{plugin_id}/settings")
async def get_plugin_settings(
    plugin_id: str,
    auth: AuthContext = Depends(require_auth_context),
) -> dict[str, Any]:
    """Get user's settings for a specific plugin.

    Returns the effective settings (user overrides merged with defaults).

    Args:
        plugin_id: The plugin identifier.
    """
    user_id = auth.user_id

    try:
        # Load all plugins to find the target
        loader = _get_plugin_loader()
        plugins = loader.load_all(skip_invalid=True)

        target_plugin = None
        for plugin in plugins:
            if plugin.id == plugin_id:
                target_plugin = plugin
                break

        if target_plugin is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"Plugin not found: {plugin_id}"},
            )

        # Get user overrides
        settings_service = UserSettingsService()
        user_overrides = settings_service.get_plugin_settings(user_id, plugin_id)

        # Get effective settings (merged with defaults)
        effective_settings = target_plugin.get_all_settings(user_overrides)

        return effective_settings

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get plugin settings for {plugin_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": str(e)},
        )


@router.put("/api/plugins/{plugin_id}/settings")
async def update_plugin_settings(
    plugin_id: str,
    request: PluginSettingsUpdateRequest,
    auth: AuthContext = Depends(require_auth_context),
) -> dict[str, Any]:
    """Update user's settings for a specific plugin.

    Args:
        plugin_id: The plugin identifier.
        request: New setting values.

    Returns:
        The updated effective settings.
    """
    user_id = auth.user_id

    try:
        # Load all plugins to find the target and validate settings
        loader = _get_plugin_loader()
        plugins = loader.load_all(skip_invalid=True)

        target_plugin = None
        for plugin in plugins:
            if plugin.id == plugin_id:
                target_plugin = plugin
                break

        if target_plugin is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"Plugin not found: {plugin_id}"},
            )

        # Validate the provided settings
        validation_errors = []
        for setting_id, value in request.settings.items():
            if setting_id not in target_plugin.settings:
                validation_errors.append(f"Unknown setting: {setting_id}")
                continue

            setting = target_plugin.settings[setting_id]
            is_valid, error = setting.validate_value(value)
            if not is_valid:
                validation_errors.append(f"{setting_id}: {error}")

        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_settings",
                    "detail": "; ".join(validation_errors),
                },
            )

        # Save the settings
        settings_service = UserSettingsService()

        # Get existing settings and merge
        existing = settings_service.get_plugin_settings(user_id, plugin_id)
        existing.update(request.settings)

        settings_service.set_plugin_settings(user_id, plugin_id, existing)

        # Return the effective settings
        effective_settings = target_plugin.get_all_settings(existing)

        logger.info(f"User {user_id} updated settings for plugin {plugin_id}")

        return effective_settings

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update plugin settings for {plugin_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": str(e)},
        )


@router.get("/api/plugins/{plugin_id}/state")
async def get_plugin_state(
    plugin_id: str,
    project_id: str = Query("default", description="Project ID for scoped state"),
    auth: AuthContext = Depends(require_auth_context),
) -> dict[str, Any]:
    """Get plugin-scoped persistent state.

    State is scoped by user, project, and plugin.

    Args:
        plugin_id: The plugin identifier.
        project_id: The project ID for state scoping.
    """
    user_id = auth.user_id

    try:
        # Verify plugin exists
        loader = _get_plugin_loader()
        plugins = loader.load_all(skip_invalid=True)

        plugin_exists = any(p.id == plugin_id for p in plugins)
        if not plugin_exists:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"Plugin not found: {plugin_id}"},
            )

        # Get plugin state
        state_service = get_plugin_state_service()
        state = state_service.get_all(user_id, project_id, plugin_id)

        return state

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get plugin state for {plugin_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": str(e)},
        )


@router.delete("/api/plugins/{plugin_id}/state", status_code=204)
async def clear_plugin_state(
    plugin_id: str,
    project_id: str = Query("default", description="Project ID for scoped state"),
    auth: AuthContext = Depends(require_auth_context),
):
    """Clear all plugin-scoped persistent state.

    Args:
        plugin_id: The plugin identifier.
        project_id: The project ID for state scoping.
    """
    user_id = auth.user_id

    try:
        # Verify plugin exists
        loader = _get_plugin_loader()
        plugins = loader.load_all(skip_invalid=True)

        plugin_exists = any(p.id == plugin_id for p in plugins)
        if not plugin_exists:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"Plugin not found: {plugin_id}"},
            )

        # Clear plugin state
        state_service = get_plugin_state_service()
        state_service.clear(user_id, project_id, plugin_id)

        logger.info(f"User {user_id} cleared state for plugin {plugin_id} in project {project_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear plugin state for {plugin_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": str(e)},
        )


__all__ = ["router"]
