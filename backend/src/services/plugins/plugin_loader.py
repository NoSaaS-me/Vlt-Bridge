"""Plugin loader for manifest-based plugin discovery and validation.

This module provides plugin discovery, manifest parsing, and dependency
validation for the Oracle Plugin System.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import toml

from .plugin import Plugin, PluginSetting
from .rule import Rule
from .loader import RuleLoader, RuleLoadError


logger = logging.getLogger(__name__)


class PluginLoadError(Exception):
    """Raised when plugin loading or validation fails."""

    pass


class PluginDependencyError(Exception):
    """Raised when plugin dependencies are not satisfied."""

    pass


# Default available capabilities in the system
DEFAULT_CAPABILITIES = frozenset({
    "vault_search",
    "vault_read",
    "vault_write",
    "web_search",
    "code_search",
    "thread_search",
})


class PluginLoader:
    """Loads plugins from directories with manifest.toml files.

    This loader discovers plugins in a directory, parses their manifests,
    loads their rules, and validates dependencies.

    Example:
        loader = PluginLoader(Path("plugins/"))
        plugins = loader.load_all()
        for plugin in plugins:
            print(f"Loaded plugin: {plugin.id} with {plugin.rule_count} rules")

    Attributes:
        plugins_dir: Directory containing plugin subdirectories.
        available_capabilities: Set of capabilities available in the system.
    """

    def __init__(
        self,
        plugins_dir: Path,
        available_capabilities: Optional[frozenset[str]] = None,
    ) -> None:
        """Initialize the plugin loader.

        Args:
            plugins_dir: Directory containing plugin subdirectories.
            available_capabilities: Set of capabilities available in the system.
                                   If None, uses DEFAULT_CAPABILITIES.
        """
        self.plugins_dir = plugins_dir
        self.available_capabilities = available_capabilities or DEFAULT_CAPABILITIES

    def load_all(self, skip_invalid: bool = False) -> list[Plugin]:
        """Load all plugins from the plugins directory.

        Args:
            skip_invalid: If True, skip invalid plugins instead of raising errors.

        Returns:
            List of loaded Plugin instances.

        Raises:
            PluginLoadError: If a plugin is invalid and skip_invalid is False.
            PluginDependencyError: If dependencies are not satisfied and skip_invalid is False.
        """
        plugins: list[Plugin] = []

        if not self.plugins_dir.exists():
            logger.warning(f"Plugins directory does not exist: {self.plugins_dir}")
            return plugins

        # Scan for subdirectories with manifest.toml
        for item in sorted(self.plugins_dir.iterdir()):
            if not item.is_dir():
                continue

            manifest_path = item / "manifest.toml"
            if not manifest_path.exists():
                continue

            try:
                plugin = self.load_plugin(item)
                plugins.append(plugin)
                logger.debug(
                    f"Loaded plugin: {plugin.id} v{plugin.version} "
                    f"with {plugin.rule_count} rules"
                )
            except (PluginLoadError, PluginDependencyError) as e:
                if skip_invalid:
                    logger.warning(f"Skipping invalid plugin in {item}: {e}")
                else:
                    raise

        logger.info(f"Loaded {len(plugins)} plugins from {self.plugins_dir}")
        return plugins

    def load_plugin(self, plugin_dir: Path) -> Plugin:
        """Load a single plugin from a directory.

        Args:
            plugin_dir: Directory containing manifest.toml.

        Returns:
            Loaded Plugin instance.

        Raises:
            PluginLoadError: If the plugin cannot be loaded or is invalid.
            PluginDependencyError: If dependencies are not satisfied.
        """
        manifest_path = plugin_dir / "manifest.toml"

        # Read and parse manifest
        try:
            content = manifest_path.read_text(encoding="utf-8")
            data = toml.loads(content)
        except FileNotFoundError:
            raise PluginLoadError(f"Manifest not found: {manifest_path}")
        except toml.TomlDecodeError as e:
            raise PluginLoadError(f"TOML parse error in {manifest_path}: {e}")

        # Validate manifest structure
        if "plugin" not in data:
            raise PluginLoadError(f"Missing [plugin] section in {manifest_path}")

        # Parse the plugin
        try:
            plugin = self._parse_manifest(data, plugin_dir)
        except Exception as e:
            raise PluginLoadError(f"Error parsing manifest from {manifest_path}: {e}")

        # Load rules from the plugin's rules directory
        rules_dir = plugin_dir / "rules"
        if rules_dir.exists():
            try:
                rule_loader = RuleLoader(rules_dir)
                rules = rule_loader.load_all(skip_invalid=False)

                # Set plugin_id on each rule
                for rule in rules:
                    rule.plugin_id = plugin.id

                plugin.rules = rules
            except RuleLoadError as e:
                raise PluginLoadError(f"Error loading rules for plugin {plugin.id}: {e}")

        # Validate the complete plugin
        errors = plugin.validate()
        if errors:
            raise PluginLoadError(
                f"Validation errors in {manifest_path}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        # Validate dependencies
        self._validate_dependencies(plugin)

        return plugin

    def _parse_manifest(self, data: dict[str, Any], plugin_dir: Path) -> Plugin:
        """Parse a Plugin from manifest TOML data.

        Args:
            data: Parsed TOML data.
            plugin_dir: Directory of the plugin.

        Returns:
            Plugin instance (without rules, which are loaded separately).

        Raises:
            KeyError: If required fields are missing.
            ValueError: If field values are invalid.
        """
        plugin_data = data["plugin"]

        # Required fields
        plugin_id = plugin_data["id"]
        name = plugin_data.get("name", plugin_id)
        version = plugin_data.get("version", "1.0.0")
        description = plugin_data.get("description", "")

        # Capabilities/dependencies
        capabilities_data = data.get("capabilities", {})
        requires = capabilities_data.get("requires", [])

        # Parse settings
        settings: dict[str, PluginSetting] = {}
        settings_data = data.get("settings", {})

        for setting_id, setting_config in settings_data.items():
            # Skip non-dict entries (could be comments or other TOML artifacts)
            if not isinstance(setting_config, dict):
                continue

            setting = self._parse_setting(setting_id, setting_config)
            settings[setting_id] = setting

        return Plugin(
            id=plugin_id,
            name=name,
            version=version,
            description=description,
            rules=[],  # Rules loaded separately
            requires=requires,
            settings=settings,
            source_dir=str(plugin_dir),
            enabled=True,
        )

    def _parse_setting(self, setting_id: str, data: dict[str, Any]) -> PluginSetting:
        """Parse a PluginSetting from manifest TOML data.

        Args:
            setting_id: Setting identifier.
            data: Setting configuration from TOML.

        Returns:
            PluginSetting instance.
        """
        # Required fields
        setting_type = data.get("type", "string")
        default = data.get("default")
        description = data.get("description", "")
        name = data.get("name", setting_id)

        # Optional constraints
        min_value = data.get("min")
        max_value = data.get("max")
        options = data.get("options")

        # Type coercion for default value
        if default is None:
            # Set type-appropriate default
            if setting_type == "integer":
                default = 0
            elif setting_type == "float":
                default = 0.0
            elif setting_type == "string":
                default = ""
            elif setting_type == "boolean":
                default = False

        return PluginSetting(
            name=name,
            type=setting_type,
            default=default,
            description=description,
            min_value=min_value,
            max_value=max_value,
            options=options,
        )

    def _validate_dependencies(self, plugin: Plugin) -> None:
        """Validate that plugin dependencies are satisfied.

        Args:
            plugin: Plugin to validate.

        Raises:
            PluginDependencyError: If any required capability is not available.
        """
        missing = set(plugin.requires) - self.available_capabilities

        if missing:
            raise PluginDependencyError(
                f"Plugin '{plugin.id}' requires capabilities not available: {sorted(missing)}. "
                f"Available capabilities: {sorted(self.available_capabilities)}"
            )

    def scan_plugins(self) -> list[str]:
        """Scan for plugin directories without loading them.

        Returns:
            List of plugin directory names that have manifest.toml files.
        """
        plugin_dirs = []

        if not self.plugins_dir.exists():
            return plugin_dirs

        for item in sorted(self.plugins_dir.iterdir()):
            if not item.is_dir():
                continue

            manifest_path = item / "manifest.toml"
            if manifest_path.exists():
                plugin_dirs.append(item.name)

        return plugin_dirs


__all__ = [
    "PluginLoader",
    "PluginLoadError",
    "PluginDependencyError",
    "DEFAULT_CAPABILITIES",
]
