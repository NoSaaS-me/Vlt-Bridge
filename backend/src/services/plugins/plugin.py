"""Plugin definitions for the Oracle Plugin System.

This module defines the Plugin and PluginSetting dataclasses for
packaging multiple rules with shared configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .rule import Rule


@dataclass
class PluginSetting:
    """A configurable plugin setting.

    Plugin settings allow users to customize plugin behavior without
    modifying the TOML files. Settings are defined in the plugin manifest
    and can be overridden per-user.

    Attributes:
        name: Human-readable setting name.
        type: Data type ("integer", "float", "string", "boolean").
        default: Default value for the setting.
        description: What this setting controls.
        min_value: Minimum value for numeric types.
        max_value: Maximum value for numeric types.
        options: Valid options for enum/select types.
    """

    name: str
    type: str  # "integer", "float", "string", "boolean"
    default: Any
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[list[str]] = None

    def validate(self) -> list[str]:
        """Validate the setting configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Validate type
        valid_types = {"integer", "float", "string", "boolean"}
        if self.type not in valid_types:
            errors.append(f"Invalid setting type: {self.type}. Must be one of: {valid_types}")

        # Validate default value matches type
        if self.type == "integer" and not isinstance(self.default, int):
            errors.append(f"Default value must be an integer, got: {type(self.default).__name__}")
        elif self.type == "float" and not isinstance(self.default, (int, float)):
            errors.append(f"Default value must be a number, got: {type(self.default).__name__}")
        elif self.type == "string" and not isinstance(self.default, str):
            errors.append(f"Default value must be a string, got: {type(self.default).__name__}")
        elif self.type == "boolean" and not isinstance(self.default, bool):
            errors.append(f"Default value must be a boolean, got: {type(self.default).__name__}")

        # Validate numeric constraints
        if self.type in ("integer", "float"):
            if self.min_value is not None and self.default < self.min_value:
                errors.append(f"Default value {self.default} is below min_value {self.min_value}")
            if self.max_value is not None and self.default > self.max_value:
                errors.append(f"Default value {self.default} is above max_value {self.max_value}")

        # Validate options
        if self.options is not None:
            if self.type != "string":
                errors.append("Options can only be specified for string type settings")
            elif self.default not in self.options:
                errors.append(f"Default value '{self.default}' is not in options: {self.options}")

        return errors

    def validate_value(self, value: Any) -> tuple[bool, str]:
        """Validate a user-provided value against this setting's constraints.

        Args:
            value: The value to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Type validation
        if self.type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Value must be an integer, got: {type(value).__name__}"
        elif self.type == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"Value must be a number, got: {type(value).__name__}"
        elif self.type == "string":
            if not isinstance(value, str):
                return False, f"Value must be a string, got: {type(value).__name__}"
        elif self.type == "boolean":
            if not isinstance(value, bool):
                return False, f"Value must be a boolean, got: {type(value).__name__}"

        # Range validation for numeric types
        if self.type in ("integer", "float"):
            if self.min_value is not None and value < self.min_value:
                return False, f"Value {value} is below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Value {value} is above maximum {self.max_value}"

        # Options validation for string types
        if self.options is not None and value not in self.options:
            return False, f"Value '{value}' is not in allowed options: {self.options}"

        return True, ""


@dataclass
class Plugin:
    """A plugin definition with multiple rules.

    Plugins package related rules together with shared configuration
    and settings. They are loaded from directories containing a
    manifest.toml file.

    Attributes:
        id: Unique identifier (kebab-case).
        name: Display name.
        version: Semantic version string.
        description: What the plugin provides.
        rules: Rules this plugin provides.
        requires: Required capabilities (e.g., "vault_search").
        settings: User-configurable parameters.
        source_dir: Directory where plugin was loaded.
        enabled: Whether plugin is currently active.
    """

    # Identity
    id: str
    name: str
    version: str
    description: str

    # Rules
    rules: list[Rule] = field(default_factory=list)

    # Dependencies
    requires: list[str] = field(default_factory=list)

    # Configuration
    settings: dict[str, PluginSetting] = field(default_factory=dict)

    # Source
    source_dir: str = ""

    # State
    enabled: bool = True

    def validate(self) -> list[str]:
        """Validate the plugin configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Validate ID format (kebab-case)
        if not re.match(r"^[a-z0-9-]+$", self.id):
            errors.append(f"Plugin ID must be kebab-case: {self.id}")

        # Validate version format (semantic versioning)
        if not re.match(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$", self.version):
            errors.append(f"Version must be semantic versioning format: {self.version}")

        # Validate settings
        for setting_id, setting in self.settings.items():
            setting_errors = setting.validate()
            for error in setting_errors:
                errors.append(f"Setting '{setting_id}': {error}")

        # Validate rules (but rules validate themselves during loading)
        for rule in self.rules:
            if rule.plugin_id != self.id:
                errors.append(
                    f"Rule '{rule.id}' has mismatched plugin_id: "
                    f"expected '{self.id}', got '{rule.plugin_id}'"
                )

        return errors

    @property
    def rule_count(self) -> int:
        """Return the number of rules in this plugin."""
        return len(self.rules)

    def get_setting_value(
        self,
        setting_id: str,
        user_overrides: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Get the effective value for a setting.

        Args:
            setting_id: The setting identifier.
            user_overrides: Optional user-specific setting overrides.

        Returns:
            The effective setting value (user override or default).

        Raises:
            KeyError: If setting_id is not defined.
        """
        if setting_id not in self.settings:
            raise KeyError(f"Unknown setting: {setting_id}")

        setting = self.settings[setting_id]

        # Check for user override
        if user_overrides and setting_id in user_overrides:
            user_value = user_overrides[setting_id]
            is_valid, error = setting.validate_value(user_value)
            if is_valid:
                return user_value
            # Fall back to default if user override is invalid

        return setting.default

    def get_all_settings(
        self,
        user_overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Get all effective setting values.

        Args:
            user_overrides: Optional user-specific setting overrides.

        Returns:
            Dictionary of setting_id -> effective value.
        """
        result = {}
        for setting_id in self.settings:
            result[setting_id] = self.get_setting_value(setting_id, user_overrides)
        return result


__all__ = [
    "Plugin",
    "PluginSetting",
]
