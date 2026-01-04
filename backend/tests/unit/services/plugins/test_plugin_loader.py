"""Unit tests for PluginLoader manifest parsing and dependency validation.

T073: Unit test for PluginLoader manifest parsing
T074: Unit test for Plugin dependency validation
"""

import pytest
import tempfile
from pathlib import Path

from backend.src.services.plugins.plugin import Plugin, PluginSetting
from backend.src.services.plugins.plugin_loader import (
    PluginLoader,
    PluginLoadError,
    PluginDependencyError,
    DEFAULT_CAPABILITIES,
)


class TestPluginSetting:
    """Tests for PluginSetting dataclass."""

    def test_valid_integer_setting(self):
        """Test valid integer setting."""
        setting = PluginSetting(
            name="Max Retries",
            type="integer",
            default=3,
            description="Maximum number of retries",
            min_value=1,
            max_value=10,
        )
        errors = setting.validate()
        assert errors == []

    def test_valid_float_setting(self):
        """Test valid float setting."""
        setting = PluginSetting(
            name="Threshold",
            type="float",
            default=0.8,
            description="Threshold for triggering",
            min_value=0.0,
            max_value=1.0,
        )
        errors = setting.validate()
        assert errors == []

    def test_valid_string_setting_with_options(self):
        """Test valid string setting with options."""
        setting = PluginSetting(
            name="Mode",
            type="string",
            default="normal",
            description="Operation mode",
            options=["fast", "normal", "thorough"],
        )
        errors = setting.validate()
        assert errors == []

    def test_valid_boolean_setting(self):
        """Test valid boolean setting."""
        setting = PluginSetting(
            name="Enabled",
            type="boolean",
            default=True,
            description="Enable this feature",
        )
        errors = setting.validate()
        assert errors == []

    def test_invalid_type(self):
        """Test invalid setting type."""
        setting = PluginSetting(
            name="Invalid",
            type="invalid_type",
            default="value",
            description="Invalid type",
        )
        errors = setting.validate()
        assert len(errors) == 1
        assert "Invalid setting type" in errors[0]

    def test_type_mismatch_integer(self):
        """Test type mismatch - expected integer, got string."""
        setting = PluginSetting(
            name="Count",
            type="integer",
            default="not_an_int",
            description="Count",
        )
        errors = setting.validate()
        assert len(errors) == 1
        assert "must be an integer" in errors[0]

    def test_type_mismatch_boolean(self):
        """Test type mismatch - expected boolean, got string."""
        setting = PluginSetting(
            name="Flag",
            type="boolean",
            default="true",
            description="Flag",
        )
        errors = setting.validate()
        assert len(errors) == 1
        assert "must be a boolean" in errors[0]

    def test_default_below_min(self):
        """Test default value below minimum."""
        setting = PluginSetting(
            name="Count",
            type="integer",
            default=0,
            description="Count",
            min_value=1,
        )
        errors = setting.validate()
        assert len(errors) == 1
        assert "below min_value" in errors[0]

    def test_default_above_max(self):
        """Test default value above maximum."""
        setting = PluginSetting(
            name="Count",
            type="integer",
            default=100,
            description="Count",
            max_value=10,
        )
        errors = setting.validate()
        assert len(errors) == 1
        assert "above max_value" in errors[0]

    def test_default_not_in_options(self):
        """Test default value not in options list."""
        setting = PluginSetting(
            name="Mode",
            type="string",
            default="invalid_mode",
            description="Mode",
            options=["fast", "normal", "slow"],
        )
        errors = setting.validate()
        assert len(errors) == 1
        assert "not in options" in errors[0]

    def test_validate_value_integer(self):
        """Test value validation for integer setting."""
        setting = PluginSetting(
            name="Count",
            type="integer",
            default=5,
            description="Count",
            min_value=1,
            max_value=10,
        )

        # Valid value
        valid, error = setting.validate_value(5)
        assert valid is True
        assert error == ""

        # Invalid type
        valid, error = setting.validate_value("5")
        assert valid is False
        assert "must be an integer" in error

        # Below minimum
        valid, error = setting.validate_value(0)
        assert valid is False
        assert "below minimum" in error

        # Above maximum
        valid, error = setting.validate_value(100)
        assert valid is False
        assert "above maximum" in error


class TestPlugin:
    """Tests for Plugin dataclass."""

    def test_valid_plugin(self):
        """Test valid plugin configuration."""
        plugin = Plugin(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
            description="A test plugin",
            rules=[],
            requires=["vault_search"],
            settings={},
            source_dir="/test/plugins/test-plugin",
        )
        errors = plugin.validate()
        assert errors == []

    def test_invalid_id_format(self):
        """Test invalid plugin ID (not kebab-case)."""
        plugin = Plugin(
            id="Test_Plugin",  # Invalid: contains uppercase and underscore
            name="Test Plugin",
            version="1.0.0",
            description="A test plugin",
        )
        errors = plugin.validate()
        assert len(errors) == 1
        assert "kebab-case" in errors[0]

    def test_invalid_version_format(self):
        """Test invalid version format (not semantic versioning)."""
        plugin = Plugin(
            id="test-plugin",
            name="Test Plugin",
            version="v1.0",  # Invalid: not semver
            description="A test plugin",
        )
        errors = plugin.validate()
        assert len(errors) == 1
        assert "semantic versioning" in errors[0]

    def test_valid_semver_with_prerelease(self):
        """Test valid semantic version with prerelease tag."""
        plugin = Plugin(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0-beta.1",
            description="A test plugin",
        )
        errors = plugin.validate()
        assert errors == []

    def test_get_setting_value_default(self):
        """Test getting setting value with defaults."""
        plugin = Plugin(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
            description="A test plugin",
            settings={
                "threshold": PluginSetting(
                    name="Threshold",
                    type="float",
                    default=0.8,
                    description="Threshold",
                ),
            },
        )

        # Get default value
        value = plugin.get_setting_value("threshold")
        assert value == 0.8

    def test_get_setting_value_with_override(self):
        """Test getting setting value with user override."""
        plugin = Plugin(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
            description="A test plugin",
            settings={
                "threshold": PluginSetting(
                    name="Threshold",
                    type="float",
                    default=0.8,
                    description="Threshold",
                ),
            },
        )

        # Get overridden value
        value = plugin.get_setting_value("threshold", {"threshold": 0.5})
        assert value == 0.5

    def test_get_setting_value_invalid_override(self):
        """Test that invalid override falls back to default."""
        plugin = Plugin(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
            description="A test plugin",
            settings={
                "threshold": PluginSetting(
                    name="Threshold",
                    type="float",
                    default=0.8,
                    description="Threshold",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
        )

        # Invalid override (too high) falls back to default
        value = plugin.get_setting_value("threshold", {"threshold": 2.0})
        assert value == 0.8

    def test_get_setting_value_unknown_setting(self):
        """Test getting unknown setting raises KeyError."""
        plugin = Plugin(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
            description="A test plugin",
            settings={},
        )

        with pytest.raises(KeyError):
            plugin.get_setting_value("unknown")


class TestPluginLoaderManifestParsing:
    """T073: Unit tests for PluginLoader manifest parsing."""

    def test_load_valid_manifest(self, tmp_path: Path):
        """Test loading a valid plugin manifest."""
        # Create plugin directory with manifest
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        manifest = """
[plugin]
id = "test-plugin"
name = "Test Plugin"
version = "1.0.0"
description = "A test plugin for testing"

[capabilities]
requires = ["vault_search"]

[settings.threshold]
name = "Threshold"
type = "float"
default = 0.8
description = "The threshold for triggering"
min = 0.0
max = 1.0

[settings.enabled]
type = "boolean"
default = true
description = "Enable the feature"
"""
        (plugin_dir / "manifest.toml").write_text(manifest)

        # Load the plugin
        loader = PluginLoader(tmp_path)
        plugin = loader.load_plugin(plugin_dir)

        assert plugin.id == "test-plugin"
        assert plugin.name == "Test Plugin"
        assert plugin.version == "1.0.0"
        assert plugin.description == "A test plugin for testing"
        assert plugin.requires == ["vault_search"]
        assert "threshold" in plugin.settings
        assert plugin.settings["threshold"].default == 0.8
        assert "enabled" in plugin.settings
        assert plugin.settings["enabled"].default is True

    def test_load_manifest_with_rules(self, tmp_path: Path):
        """Test loading a plugin manifest that also has rules."""
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        manifest = """
[plugin]
id = "test-plugin"
name = "Test Plugin"
version = "1.0.0"
description = "A test plugin"
"""
        (plugin_dir / "manifest.toml").write_text(manifest)

        # Create rules directory with a rule
        rules_dir = plugin_dir / "rules"
        rules_dir.mkdir()

        rule_toml = """
[rule]
id = "test-rule"
name = "Test Rule"
trigger = "on_turn_start"
priority = 100

[condition]
expression = "true"

[action]
type = "notify_self"
message = "Test message"
"""
        (rules_dir / "test-rule.toml").write_text(rule_toml)

        # Load the plugin
        loader = PluginLoader(tmp_path)
        plugin = loader.load_plugin(plugin_dir)

        assert plugin.id == "test-plugin"
        assert plugin.rule_count == 1
        assert plugin.rules[0].id == "test-rule"
        assert plugin.rules[0].plugin_id == "test-plugin"

    def test_load_manifest_missing_plugin_section(self, tmp_path: Path):
        """Test error when manifest is missing [plugin] section."""
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        manifest = """
[settings.threshold]
type = "float"
default = 0.5
"""
        (plugin_dir / "manifest.toml").write_text(manifest)

        loader = PluginLoader(tmp_path)
        with pytest.raises(PluginLoadError) as exc_info:
            loader.load_plugin(plugin_dir)

        assert "Missing [plugin] section" in str(exc_info.value)

    def test_load_manifest_invalid_toml(self, tmp_path: Path):
        """Test error when manifest has invalid TOML syntax."""
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        manifest = """
[plugin]
id = "test-plugin"
name = "Missing closing quote
"""
        (plugin_dir / "manifest.toml").write_text(manifest)

        loader = PluginLoader(tmp_path)
        with pytest.raises(PluginLoadError) as exc_info:
            loader.load_plugin(plugin_dir)

        assert "TOML parse error" in str(exc_info.value)

    def test_load_manifest_missing_id(self, tmp_path: Path):
        """Test error when manifest is missing required plugin.id."""
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        manifest = """
[plugin]
name = "Test Plugin"
version = "1.0.0"
"""
        (plugin_dir / "manifest.toml").write_text(manifest)

        loader = PluginLoader(tmp_path)
        with pytest.raises(PluginLoadError) as exc_info:
            loader.load_plugin(plugin_dir)

        # The exact error message depends on implementation
        assert "error" in str(exc_info.value).lower()

    def test_load_manifest_invalid_plugin_id(self, tmp_path: Path):
        """Test validation error for invalid plugin ID format."""
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        manifest = """
[plugin]
id = "Test_Plugin"
name = "Test Plugin"
version = "1.0.0"
"""
        (plugin_dir / "manifest.toml").write_text(manifest)

        loader = PluginLoader(tmp_path)
        with pytest.raises(PluginLoadError) as exc_info:
            loader.load_plugin(plugin_dir)

        assert "kebab-case" in str(exc_info.value)

    def test_load_all_plugins(self, tmp_path: Path):
        """Test loading all plugins from a directory."""
        # Create two plugin directories
        for name in ["plugin-one", "plugin-two"]:
            plugin_dir = tmp_path / name
            plugin_dir.mkdir()

            manifest = f"""
[plugin]
id = "{name}"
name = "{name.title()}"
version = "1.0.0"
"""
            (plugin_dir / "manifest.toml").write_text(manifest)

        # Also create a non-plugin directory (no manifest)
        (tmp_path / "not-a-plugin").mkdir()

        loader = PluginLoader(tmp_path)
        plugins = loader.load_all()

        assert len(plugins) == 2
        plugin_ids = {p.id for p in plugins}
        assert plugin_ids == {"plugin-one", "plugin-two"}

    def test_load_all_skip_invalid(self, tmp_path: Path):
        """Test that load_all with skip_invalid=True skips broken plugins."""
        # Create one valid plugin
        valid_dir = tmp_path / "valid-plugin"
        valid_dir.mkdir()
        (valid_dir / "manifest.toml").write_text("""
[plugin]
id = "valid-plugin"
name = "Valid Plugin"
version = "1.0.0"
""")

        # Create one invalid plugin
        invalid_dir = tmp_path / "invalid-plugin"
        invalid_dir.mkdir()
        (invalid_dir / "manifest.toml").write_text("""
[plugin]
id = "Invalid_Plugin"
name = "Invalid"
""")

        loader = PluginLoader(tmp_path)
        plugins = loader.load_all(skip_invalid=True)

        assert len(plugins) == 1
        assert plugins[0].id == "valid-plugin"

    def test_load_all_raises_on_invalid(self, tmp_path: Path):
        """Test that load_all with skip_invalid=False raises on broken plugins."""
        # Create one invalid plugin
        invalid_dir = tmp_path / "invalid-plugin"
        invalid_dir.mkdir()
        (invalid_dir / "manifest.toml").write_text("""
[plugin]
id = "Invalid_Plugin"
name = "Invalid"
""")

        loader = PluginLoader(tmp_path)
        with pytest.raises(PluginLoadError):
            loader.load_all(skip_invalid=False)

    def test_scan_plugins(self, tmp_path: Path):
        """Test scanning for plugin directories without loading."""
        for name in ["plugin-one", "plugin-two", "plugin-three"]:
            plugin_dir = tmp_path / name
            plugin_dir.mkdir()
            (plugin_dir / "manifest.toml").write_text(f"""
[plugin]
id = "{name}"
""")

        # Add non-plugin directory
        (tmp_path / "not-a-plugin").mkdir()

        loader = PluginLoader(tmp_path)
        plugin_names = loader.scan_plugins()

        assert len(plugin_names) == 3
        assert set(plugin_names) == {"plugin-one", "plugin-two", "plugin-three"}


class TestPluginLoaderDependencyValidation:
    """T074: Unit tests for Plugin dependency validation."""

    def test_satisfied_dependencies(self, tmp_path: Path):
        """Test plugin loads when all dependencies are satisfied."""
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        manifest = """
[plugin]
id = "test-plugin"
name = "Test Plugin"
version = "1.0.0"

[capabilities]
requires = ["vault_search", "vault_read"]
"""
        (plugin_dir / "manifest.toml").write_text(manifest)

        # All required capabilities are in DEFAULT_CAPABILITIES
        loader = PluginLoader(tmp_path)
        plugin = loader.load_plugin(plugin_dir)

        assert plugin.id == "test-plugin"
        assert plugin.requires == ["vault_search", "vault_read"]

    def test_missing_dependency(self, tmp_path: Path):
        """Test error when plugin has unsatisfied dependencies."""
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        manifest = """
[plugin]
id = "test-plugin"
name = "Test Plugin"
version = "1.0.0"

[capabilities]
requires = ["vault_search", "magic_capability"]
"""
        (plugin_dir / "manifest.toml").write_text(manifest)

        loader = PluginLoader(tmp_path)
        with pytest.raises(PluginDependencyError) as exc_info:
            loader.load_plugin(plugin_dir)

        assert "magic_capability" in str(exc_info.value)
        assert "not available" in str(exc_info.value)

    def test_custom_capabilities(self, tmp_path: Path):
        """Test loading with custom available capabilities."""
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        manifest = """
[plugin]
id = "test-plugin"
name = "Test Plugin"
version = "1.0.0"

[capabilities]
requires = ["custom_capability"]
"""
        (plugin_dir / "manifest.toml").write_text(manifest)

        # Load with custom capabilities
        custom_caps = frozenset(["custom_capability", "another_capability"])
        loader = PluginLoader(tmp_path, available_capabilities=custom_caps)
        plugin = loader.load_plugin(plugin_dir)

        assert plugin.id == "test-plugin"

    def test_no_dependencies(self, tmp_path: Path):
        """Test plugin with no dependency requirements."""
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        manifest = """
[plugin]
id = "test-plugin"
name = "Test Plugin"
version = "1.0.0"
"""
        (plugin_dir / "manifest.toml").write_text(manifest)

        loader = PluginLoader(tmp_path)
        plugin = loader.load_plugin(plugin_dir)

        assert plugin.id == "test-plugin"
        assert plugin.requires == []

    def test_all_default_capabilities_available(self):
        """Test that all default capabilities are defined."""
        expected = {
            "vault_search",
            "vault_read",
            "vault_write",
            "web_search",
            "code_search",
            "thread_search",
        }
        assert DEFAULT_CAPABILITIES == expected
