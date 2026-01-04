"""Rule loader for TOML-based rule definitions.

This module provides TOML file discovery, parsing, and validation
for the Oracle Plugin System.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import toml

from .rule import (
    ActionType,
    HookPoint,
    InjectionPoint,
    Priority,
    Rule,
    RuleAction,
)


logger = logging.getLogger(__name__)


class RuleLoadError(Exception):
    """Raised when rule loading or validation fails."""

    pass


class RuleLoader:
    """Loads rules from TOML files.

    This loader discovers and parses TOML rule definitions from a directory,
    validating them against the Rule schema and returning Rule instances.

    Example:
        loader = RuleLoader(Path("rules/"))
        rules = loader.load_all()
        for rule in rules:
            print(f"Loaded rule: {rule.id}")
    """

    def __init__(self, rules_dir: Path) -> None:
        """Initialize the rule loader.

        Args:
            rules_dir: Directory containing .toml rule files.
        """
        self.rules_dir = rules_dir

    def load_all(self, skip_invalid: bool = False) -> list[Rule]:
        """Load all rules from the rules directory.

        Args:
            skip_invalid: If True, skip invalid files instead of raising errors.

        Returns:
            List of loaded Rule instances.

        Raises:
            RuleLoadError: If a rule file is invalid and skip_invalid is False.
        """
        rules: list[Rule] = []

        if not self.rules_dir.exists():
            logger.warning(f"Rules directory does not exist: {self.rules_dir}")
            return rules

        # Find all .toml files
        toml_files = sorted(self.rules_dir.glob("*.toml"))

        for toml_file in toml_files:
            try:
                rule = self.load_rule(toml_file)
                rules.append(rule)
                logger.debug(f"Loaded rule: {rule.id} from {toml_file}")
            except RuleLoadError as e:
                if skip_invalid:
                    logger.warning(f"Skipping invalid rule file {toml_file}: {e}")
                else:
                    raise

        logger.info(f"Loaded {len(rules)} rules from {self.rules_dir}")
        return rules

    def load_rule(self, path: Path) -> Rule:
        """Load a single rule from a TOML file.

        Args:
            path: Path to the TOML file.

        Returns:
            Loaded Rule instance.

        Raises:
            RuleLoadError: If the file cannot be loaded or is invalid.
        """
        # Read and parse TOML
        try:
            content = path.read_text(encoding="utf-8")
            data = toml.loads(content)
        except FileNotFoundError:
            raise RuleLoadError(f"Rule file not found: {path}")
        except toml.TomlDecodeError as e:
            raise RuleLoadError(f"TOML parse error in {path}: {e}")

        # Validate structure
        if "rule" not in data:
            raise RuleLoadError(f"Missing [rule] section in {path}")

        # Parse the rule
        try:
            rule = self._parse_rule(data, str(path))
        except Exception as e:
            raise RuleLoadError(f"Error parsing rule from {path}: {e}")

        # Validate the rule
        errors = rule.validate()
        if errors:
            raise RuleLoadError(
                f"Validation errors in {path}:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return rule

    def _parse_rule(self, data: dict[str, Any], source_path: str) -> Rule:
        """Parse a Rule from TOML data.

        Args:
            data: Parsed TOML data.
            source_path: Path to the source file (for reference).

        Returns:
            Rule instance.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If field values are invalid.
        """
        rule_data = data["rule"]

        # Required fields
        rule_id = rule_data["id"]
        name = rule_data.get("name", rule_id)
        description = rule_data.get("description", "")

        # Trigger (convert string to enum)
        trigger_str = rule_data.get("trigger", "on_turn_start")
        trigger = HookPoint(trigger_str)

        # Optional fields with defaults
        version = rule_data.get("version", "1.0.0")
        priority = rule_data.get("priority", 100)
        enabled = rule_data.get("enabled", True)
        core = rule_data.get("core", False)
        plugin_id = rule_data.get("plugin_id")

        # Condition - either expression or script
        condition_data = data.get("condition", {})
        condition = condition_data.get("expression")
        script = condition_data.get("script")

        # Action
        action_data = data.get("action", {})
        action = self._parse_action(action_data) if action_data else None

        return Rule(
            id=rule_id,
            name=name,
            description=description,
            version=version,
            trigger=trigger,
            condition=condition,
            script=script,
            action=action,
            priority=priority,
            enabled=enabled,
            core=core,
            plugin_id=plugin_id,
            source_path=source_path,
        )

    def _parse_action(self, data: dict[str, Any]) -> RuleAction:
        """Parse a RuleAction from TOML data.

        Args:
            data: Action section of TOML data.

        Returns:
            RuleAction instance.

        Raises:
            ValueError: If action type is invalid.
        """
        # Action type (required)
        action_type_str = data.get("type", "log")
        action_type = ActionType(action_type_str)

        # Common fields
        message = data.get("message")
        category = data.get("category")

        # Priority (convert string to enum)
        priority_str = data.get("priority", "normal")
        priority = Priority(priority_str)

        # Injection point (convert string to enum)
        deliver_at_str = data.get("deliver_at", "turn_start")
        deliver_at = InjectionPoint(deliver_at_str)

        # Log-specific
        level = data.get("level", "info")

        # set_state-specific
        key = data.get("key")
        value = data.get("value")

        # emit_event-specific
        event_type = data.get("event_type")
        payload = data.get("payload")

        return RuleAction(
            type=action_type,
            message=message,
            category=category,
            priority=priority,
            deliver_at=deliver_at,
            level=level,
            key=key,
            value=value,
            event_type=event_type,
            payload=payload,
        )


__all__ = [
    "RuleLoader",
    "RuleLoadError",
]
