"""Unit tests for RuleLoader TOML parsing and validation (T019, T021).

This module tests:
- TOML file loading and parsing
- Rule validation from TOML data
- Directory discovery of .toml files
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from backend.src.services.plugins.rule import (
    ActionType,
    HookPoint,
    InjectionPoint,
    Priority,
    Rule,
    RuleAction,
)
from backend.src.services.plugins.loader import RuleLoader, RuleLoadError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_rule_toml() -> str:
    """Return a valid rule TOML string."""
    return '''
[rule]
id = "token-budget-warning"
name = "Token Budget Warning"
description = "Warn when token usage exceeds 80%"
version = "1.0.0"
trigger = "on_turn_start"
priority = 100
enabled = true
core = true

[condition]
expression = "context.turn.token_usage > 0.8"

[action]
type = "notify_self"
message = "Token budget at {{ (context.turn.token_usage * 100) | int }}%."
category = "warning"
priority = "high"
deliver_at = "turn_start"
'''


@pytest.fixture
def minimal_rule_toml() -> str:
    """Return a minimal valid rule TOML string."""
    return '''
[rule]
id = "minimal-rule"
name = "Minimal Rule"
description = "A minimal test rule"
trigger = "on_turn_start"

[condition]
expression = "true"

[action]
type = "log"
message = "Rule fired"
level = "info"
'''


@pytest.fixture
def invalid_id_toml() -> str:
    """Return a TOML with invalid rule ID (not kebab-case)."""
    return '''
[rule]
id = "Invalid_ID"
name = "Invalid ID Rule"
description = "Rule with invalid ID"
trigger = "on_turn_start"

[condition]
expression = "true"

[action]
type = "log"
message = "This should fail"
'''


@pytest.fixture
def no_condition_toml() -> str:
    """Return a TOML missing both condition and script."""
    return '''
[rule]
id = "no-condition"
name = "No Condition Rule"
description = "Rule without condition or script"
trigger = "on_turn_start"

[action]
type = "log"
message = "This should fail"
'''


@pytest.fixture
def both_condition_and_script_toml() -> str:
    """Return a TOML with both condition and script (invalid)."""
    return '''
[rule]
id = "both-condition-script"
name = "Both Condition and Script"
description = "Rule with both condition and script"
trigger = "on_turn_start"

[condition]
expression = "true"
script = "scripts/test.lua"

[action]
type = "log"
message = "This should fail"
'''


@pytest.fixture
def missing_action_toml() -> str:
    """Return a TOML missing action section."""
    return '''
[rule]
id = "missing-action"
name = "Missing Action"
description = "Rule without action"
trigger = "on_turn_start"

[condition]
expression = "true"
'''


@pytest.fixture
def invalid_priority_toml() -> str:
    """Return a TOML with invalid priority (out of range)."""
    return '''
[rule]
id = "invalid-priority"
name = "Invalid Priority"
description = "Rule with invalid priority"
trigger = "on_turn_start"
priority = 5000

[condition]
expression = "true"

[action]
type = "log"
message = "Test"
'''


@pytest.fixture
def set_state_action_toml() -> str:
    """Return a TOML with set_state action."""
    return '''
[rule]
id = "set-state-rule"
name = "Set State Rule"
description = "Rule that sets state"
trigger = "on_turn_start"

[condition]
expression = "context.turn.number == 1"

[action]
type = "set_state"
key = "last_turn"
value = "{{ context.turn.number }}"
'''


@pytest.fixture
def emit_event_action_toml() -> str:
    """Return a TOML with emit_event action."""
    return '''
[rule]
id = "emit-event-rule"
name = "Emit Event Rule"
description = "Rule that emits an event"
trigger = "on_turn_start"

[condition]
expression = "context.turn.token_usage > 0.9"

[action]
type = "emit_event"
event_type = "custom.budget.critical"
payload = { token_usage = "{{ context.turn.token_usage }}", turn = "{{ context.turn.number }}" }
'''


@pytest.fixture
def temp_rules_dir(
    valid_rule_toml: str,
    minimal_rule_toml: str,
) -> Generator[Path, None, None]:
    """Create a temporary directory with test rule files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rules_dir = Path(tmpdir)

        # Write valid rule files
        (rules_dir / "token_budget.toml").write_text(valid_rule_toml)
        (rules_dir / "minimal.toml").write_text(minimal_rule_toml)

        # Write a non-TOML file (should be ignored)
        (rules_dir / "readme.md").write_text("# Rules\n\nThis file should be ignored.")

        yield rules_dir


@pytest.fixture
def temp_rules_dir_with_invalid(
    valid_rule_toml: str,
    invalid_id_toml: str,
) -> Generator[Path, None, None]:
    """Create a temp directory with both valid and invalid rule files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rules_dir = Path(tmpdir)

        (rules_dir / "valid.toml").write_text(valid_rule_toml)
        (rules_dir / "invalid.toml").write_text(invalid_id_toml)

        yield rules_dir


# =============================================================================
# T019: RuleLoader TOML Parsing Tests
# =============================================================================


class TestRuleLoaderTomlParsing:
    """Tests for loading rules from TOML files."""

    def test_load_valid_rule(self, valid_rule_toml: str) -> None:
        """Load a valid rule from TOML content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(valid_rule_toml)
            f.flush()

            loader = RuleLoader(Path(f.name).parent)
            rule = loader.load_rule(Path(f.name))

            assert rule.id == "token-budget-warning"
            assert rule.name == "Token Budget Warning"
            assert rule.description == "Warn when token usage exceeds 80%"
            assert rule.version == "1.0.0"
            assert rule.trigger == HookPoint.ON_TURN_START
            assert rule.priority == 100
            assert rule.enabled is True
            assert rule.core is True
            assert rule.condition == "context.turn.token_usage > 0.8"
            assert rule.script is None
            assert rule.action is not None
            assert rule.action.type == ActionType.NOTIFY_SELF
            assert "Token budget" in rule.action.message
            assert rule.action.category == "warning"
            assert rule.action.priority == Priority.HIGH
            assert rule.action.deliver_at == InjectionPoint.TURN_START

    def test_load_minimal_rule(self, minimal_rule_toml: str) -> None:
        """Load a minimal rule with defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(minimal_rule_toml)
            f.flush()

            loader = RuleLoader(Path(f.name).parent)
            rule = loader.load_rule(Path(f.name))

            assert rule.id == "minimal-rule"
            assert rule.version == "1.0.0"  # default
            assert rule.priority == 100  # default
            assert rule.enabled is True  # default
            assert rule.core is False  # default
            assert rule.action.type == ActionType.LOG
            assert rule.action.level == "info"

    def test_load_rule_with_script(self) -> None:
        """Load a rule with script instead of expression."""
        toml_content = '''
[rule]
id = "script-rule"
name = "Script Rule"
description = "Rule with Lua script"
trigger = "on_tool_complete"

[condition]
script = "scripts/check_result.lua"

[action]
type = "notify_self"
message = "Script condition met"
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            f.flush()

            loader = RuleLoader(Path(f.name).parent)
            rule = loader.load_rule(Path(f.name))

            assert rule.condition is None
            assert rule.script == "scripts/check_result.lua"

    def test_load_rule_sets_source_path(self, valid_rule_toml: str) -> None:
        """Loading a rule sets its source_path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(valid_rule_toml)
            f.flush()

            loader = RuleLoader(Path(f.name).parent)
            rule = loader.load_rule(Path(f.name))

            assert rule.source_path == f.name

    def test_load_invalid_toml_syntax(self) -> None:
        """Loading invalid TOML syntax raises RuleLoadError."""
        invalid_toml = "not valid toml {{{"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(invalid_toml)
            f.flush()

            loader = RuleLoader(Path(f.name).parent)
            with pytest.raises(RuleLoadError) as excinfo:
                loader.load_rule(Path(f.name))

            assert "TOML" in str(excinfo.value) or "parse" in str(excinfo.value).lower()

    def test_load_missing_rule_section(self) -> None:
        """Loading TOML without [rule] section raises RuleLoadError."""
        no_rule_section = '''
[action]
type = "log"
message = "No rule section"
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(no_rule_section)
            f.flush()

            loader = RuleLoader(Path(f.name).parent)
            with pytest.raises(RuleLoadError) as excinfo:
                loader.load_rule(Path(f.name))

            assert "rule" in str(excinfo.value).lower()

    def test_load_missing_required_fields(self) -> None:
        """Loading TOML with missing required fields raises RuleLoadError."""
        missing_fields = '''
[rule]
id = "incomplete"
# missing name, description, trigger

[action]
type = "log"
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(missing_fields)
            f.flush()

            loader = RuleLoader(Path(f.name).parent)
            with pytest.raises(RuleLoadError):
                loader.load_rule(Path(f.name))

    def test_load_nonexistent_file(self) -> None:
        """Loading nonexistent file raises appropriate error."""
        loader = RuleLoader(Path("/tmp"))
        with pytest.raises((RuleLoadError, FileNotFoundError)):
            loader.load_rule(Path("/tmp/nonexistent.toml"))


# =============================================================================
# T019: Directory Discovery Tests
# =============================================================================


class TestRuleLoaderDiscovery:
    """Tests for discovering rules from a directory."""

    def test_discover_all_toml_files(self, temp_rules_dir: Path) -> None:
        """Discover all .toml files in directory."""
        loader = RuleLoader(temp_rules_dir)
        rules = loader.load_all()

        assert len(rules) == 2
        rule_ids = {r.id for r in rules}
        assert "token-budget-warning" in rule_ids
        assert "minimal-rule" in rule_ids

    def test_ignore_non_toml_files(self, temp_rules_dir: Path) -> None:
        """Non-TOML files are ignored during discovery."""
        loader = RuleLoader(temp_rules_dir)
        rules = loader.load_all()

        # readme.md should be ignored
        for rule in rules:
            assert not rule.source_path.endswith(".md")

    def test_empty_directory(self) -> None:
        """Loading from empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = RuleLoader(Path(tmpdir))
            rules = loader.load_all()

            assert rules == []

    def test_skip_invalid_files_in_directory(
        self,
        temp_rules_dir_with_invalid: Path,
    ) -> None:
        """Invalid files are skipped during bulk loading."""
        loader = RuleLoader(temp_rules_dir_with_invalid)

        # load_all should skip invalid files and return valid ones
        rules = loader.load_all(skip_invalid=True)

        assert len(rules) == 1
        assert rules[0].id == "token-budget-warning"

    def test_raise_on_invalid_files(
        self,
        temp_rules_dir_with_invalid: Path,
    ) -> None:
        """By default, invalid files raise errors."""
        loader = RuleLoader(temp_rules_dir_with_invalid)

        with pytest.raises(RuleLoadError):
            loader.load_all(skip_invalid=False)


# =============================================================================
# T021: Rule Validation Tests
# =============================================================================


class TestRuleValidation:
    """Tests for Rule validation (T021)."""

    def test_valid_rule_has_no_errors(self) -> None:
        """A properly configured rule validates successfully."""
        rule = Rule(
            id="test-rule",
            name="Test Rule",
            description="A test rule",
            trigger=HookPoint.ON_TURN_START,
            condition="context.turn.token_usage > 0.5",
            action=RuleAction(
                type=ActionType.LOG,
                message="Test message",
            ),
        )

        errors = rule.validate()
        assert errors == []

    def test_invalid_id_format(self) -> None:
        """Rule ID must be kebab-case."""
        rule = Rule(
            id="Invalid_ID",
            name="Test",
            description="Test",
            trigger=HookPoint.ON_TURN_START,
            condition="true",
            action=RuleAction(type=ActionType.LOG, message="Test"),
        )

        errors = rule.validate()
        assert any("kebab-case" in e for e in errors)

    def test_id_with_uppercase_rejected(self) -> None:
        """Uppercase letters in ID are rejected."""
        rule = Rule(
            id="MyRule",
            name="Test",
            description="Test",
            trigger=HookPoint.ON_TURN_START,
            condition="true",
            action=RuleAction(type=ActionType.LOG, message="Test"),
        )

        errors = rule.validate()
        assert len(errors) > 0

    def test_id_with_numbers_allowed(self) -> None:
        """Numbers in kebab-case ID are allowed."""
        rule = Rule(
            id="rule-v2-test",
            name="Test",
            description="Test",
            trigger=HookPoint.ON_TURN_START,
            condition="true",
            action=RuleAction(type=ActionType.LOG, message="Test"),
        )

        errors = rule.validate()
        # Should only fail validation if there are other issues
        assert not any("kebab-case" in e for e in errors)

    def test_must_have_condition_or_script(self) -> None:
        """Rule must have either condition or script."""
        rule = Rule(
            id="no-condition",
            name="Test",
            description="Test",
            trigger=HookPoint.ON_TURN_START,
            condition=None,
            script=None,
            action=RuleAction(type=ActionType.LOG, message="Test"),
        )

        errors = rule.validate()
        assert any("condition" in e.lower() or "script" in e.lower() for e in errors)

    def test_cannot_have_both_condition_and_script(self) -> None:
        """Rule cannot have both condition and script."""
        rule = Rule(
            id="both-set",
            name="Test",
            description="Test",
            trigger=HookPoint.ON_TURN_START,
            condition="true",
            script="scripts/test.lua",
            action=RuleAction(type=ActionType.LOG, message="Test"),
        )

        errors = rule.validate()
        assert any("both" in e.lower() for e in errors)

    def test_priority_must_be_in_range(self) -> None:
        """Priority must be within 1-1000 range."""
        rule = Rule(
            id="out-of-range",
            name="Test",
            description="Test",
            trigger=HookPoint.ON_TURN_START,
            condition="true",
            priority=5000,
            action=RuleAction(type=ActionType.LOG, message="Test"),
        )

        errors = rule.validate()
        assert any("priority" in e.lower() for e in errors)

    def test_priority_zero_invalid(self) -> None:
        """Priority of 0 is invalid."""
        rule = Rule(
            id="zero-priority",
            name="Test",
            description="Test",
            trigger=HookPoint.ON_TURN_START,
            condition="true",
            priority=0,
            action=RuleAction(type=ActionType.LOG, message="Test"),
        )

        errors = rule.validate()
        assert any("priority" in e.lower() for e in errors)

    def test_must_have_action(self) -> None:
        """Rule must have an action."""
        rule = Rule(
            id="no-action",
            name="Test",
            description="Test",
            trigger=HookPoint.ON_TURN_START,
            condition="true",
            action=None,
        )

        errors = rule.validate()
        assert any("action" in e.lower() for e in errors)


# =============================================================================
# RuleAction Validation Tests
# =============================================================================


class TestRuleActionValidation:
    """Tests for RuleAction validation."""

    def test_notify_self_requires_message(self) -> None:
        """notify_self action requires a message."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message=None,
        )

        errors = action.validate()
        assert any("message" in e.lower() for e in errors)

    def test_notify_self_with_message_valid(self) -> None:
        """notify_self action with message is valid."""
        action = RuleAction(
            type=ActionType.NOTIFY_SELF,
            message="Test notification",
        )

        errors = action.validate()
        assert errors == []

    def test_log_requires_valid_level(self) -> None:
        """log action requires valid log level."""
        action = RuleAction(
            type=ActionType.LOG,
            message="Test",
            level="invalid_level",
        )

        errors = action.validate()
        assert any("level" in e.lower() for e in errors)

    def test_log_valid_levels(self) -> None:
        """log action accepts valid log levels."""
        for level in ["debug", "info", "warning", "error"]:
            action = RuleAction(
                type=ActionType.LOG,
                message="Test",
                level=level,
            )
            errors = action.validate()
            assert errors == []

    def test_set_state_requires_key(self) -> None:
        """set_state action requires a key."""
        action = RuleAction(
            type=ActionType.SET_STATE,
            key=None,
            value="test",
        )

        errors = action.validate()
        assert any("key" in e.lower() for e in errors)

    def test_set_state_with_key_valid(self) -> None:
        """set_state action with key is valid."""
        action = RuleAction(
            type=ActionType.SET_STATE,
            key="my_key",
            value="my_value",
        )

        errors = action.validate()
        assert errors == []

    def test_emit_event_requires_event_type(self) -> None:
        """emit_event action requires event_type."""
        action = RuleAction(
            type=ActionType.EMIT_EVENT,
            event_type=None,
        )

        errors = action.validate()
        assert any("event_type" in e.lower() for e in errors)

    def test_emit_event_with_event_type_valid(self) -> None:
        """emit_event action with event_type is valid."""
        action = RuleAction(
            type=ActionType.EMIT_EVENT,
            event_type="custom.event",
            payload={"key": "value"},
        )

        errors = action.validate()
        assert errors == []


# =============================================================================
# Action Type TOML Loading Tests
# =============================================================================


class TestActionTypeLoading:
    """Tests for loading different action types from TOML."""

    def test_load_set_state_action(self, set_state_action_toml: str) -> None:
        """Load rule with set_state action."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(set_state_action_toml)
            f.flush()

            loader = RuleLoader(Path(f.name).parent)
            rule = loader.load_rule(Path(f.name))

            assert rule.action.type == ActionType.SET_STATE
            assert rule.action.key == "last_turn"
            assert "context.turn.number" in rule.action.value

    def test_load_emit_event_action(self, emit_event_action_toml: str) -> None:
        """Load rule with emit_event action."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(emit_event_action_toml)
            f.flush()

            loader = RuleLoader(Path(f.name).parent)
            rule = loader.load_rule(Path(f.name))

            assert rule.action.type == ActionType.EMIT_EVENT
            assert rule.action.event_type == "custom.budget.critical"
            assert rule.action.payload is not None
            assert "token_usage" in rule.action.payload


# =============================================================================
# Edge Cases
# =============================================================================


class TestRuleLoaderEdgeCases:
    """Tests for edge cases in rule loading."""

    def test_load_rule_with_all_triggers(self) -> None:
        """All HookPoint values can be loaded from TOML."""
        triggers = [
            ("on_query_start", "query-start"),
            ("on_turn_start", "turn-start"),
            ("on_turn_end", "turn-end"),
            ("on_tool_call", "tool-call"),
            ("on_tool_complete", "tool-complete"),
            ("on_tool_failure", "tool-failure"),
            ("on_session_end", "session-end"),
        ]

        for trigger, rule_id in triggers:
            toml_content = f'''
[rule]
id = "trigger-{rule_id}"
name = "Trigger Test"
description = "Test {trigger}"
trigger = "{trigger}"

[condition]
expression = "true"

[action]
type = "log"
message = "Test"
'''
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                f.write(toml_content)
                f.flush()

                loader = RuleLoader(Path(f.name).parent)
                rule = loader.load_rule(Path(f.name))

                assert rule.trigger.value == trigger

    def test_load_rule_with_all_priorities(self) -> None:
        """All Priority values can be loaded from TOML."""
        priorities = ["low", "normal", "high", "critical"]

        for priority in priorities:
            toml_content = f'''
[rule]
id = "priority-{priority}"
name = "Priority Test"
description = "Test {priority}"
trigger = "on_turn_start"

[condition]
expression = "true"

[action]
type = "notify_self"
message = "Test"
priority = "{priority}"
'''
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                f.write(toml_content)
                f.flush()

                loader = RuleLoader(Path(f.name).parent)
                rule = loader.load_rule(Path(f.name))

                assert rule.action.priority.value == priority

    def test_load_rule_with_all_injection_points(self) -> None:
        """All InjectionPoint values can be loaded from TOML."""
        injection_points = [
            ("turn_start", "turn-start"),
            ("after_tool", "after-tool"),
            ("immediate", "immediate"),
        ]

        for point, rule_id in injection_points:
            toml_content = f'''
[rule]
id = "injection-{rule_id}"
name = "Injection Test"
description = "Test {point}"
trigger = "on_turn_start"

[condition]
expression = "true"

[action]
type = "notify_self"
message = "Test"
deliver_at = "{point}"
'''
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                f.write(toml_content)
                f.flush()

                loader = RuleLoader(Path(f.name).parent)
                rule = loader.load_rule(Path(f.name))

                assert rule.action.deliver_at.value == point

    def test_qualified_id_with_plugin(self) -> None:
        """Rule qualified_id includes plugin prefix when set."""
        rule = Rule(
            id="test-rule",
            name="Test",
            description="Test",
            trigger=HookPoint.ON_TURN_START,
            condition="true",
            action=RuleAction(type=ActionType.LOG, message="Test"),
            plugin_id="my-plugin",
        )

        assert rule.qualified_id == "my-plugin:test-rule"

    def test_qualified_id_without_plugin(self) -> None:
        """Rule qualified_id is just the ID when no plugin."""
        rule = Rule(
            id="test-rule",
            name="Test",
            description="Test",
            trigger=HookPoint.ON_TURN_START,
            condition="true",
            action=RuleAction(type=ActionType.LOG, message="Test"),
        )

        assert rule.qualified_id == "test-rule"
