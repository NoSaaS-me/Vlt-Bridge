"""Unit tests for the ANS SubscriberLoader and subscriber configuration."""

import logging
from pathlib import Path
from textwrap import dedent

import pytest

from backend.src.services.ans.subscriber import (
    InjectionPoint,
    Priority,
    Subscriber,
    SubscriberConfig,
    SubscriberLoader,
    ValidationResult,
)


@pytest.fixture
def temp_subscribers_dir(tmp_path: Path) -> Path:
    """Create a temporary subscribers directory."""
    subscribers_dir = tmp_path / "subscribers"
    subscribers_dir.mkdir()
    return subscribers_dir


@pytest.fixture
def temp_templates_dir(tmp_path: Path) -> Path:
    """Create a temporary templates directory."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    return templates_dir


def write_toml(path: Path, content: str) -> None:
    """Write TOML content to a file."""
    path.write_text(dedent(content).strip())


# =============================================================================
# T073: Test valid subscriber discovery
# =============================================================================


class TestValidSubscriberDiscovery:
    """Tests for valid subscriber TOML discovery and loading."""

    def test_discover_single_subscriber(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """A single valid TOML file is discovered and loaded."""
        # Create template
        (temp_templates_dir / "test.toon.j2").write_text("test template")

        # Create subscriber TOML
        write_toml(
            temp_subscribers_dir / "test_subscriber.toml",
            """
            [subscriber]
            id = "test_sub"
            name = "Test Subscriber"
            description = "A test subscriber"
            version = "1.0.0"

            [events]
            types = ["test.event.one", "test.event.two"]
            severity_filter = "info"

            [output]
            priority = "normal"
            inject_at = "after_tool"
            template = "test.toon.j2"
            core = false
            """,
        )

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()

        assert len(subscribers) == 1
        assert "test_sub" in subscribers

        sub = subscribers["test_sub"]
        assert sub.id == "test_sub"
        assert sub.name == "Test Subscriber"
        assert sub.event_types == ["test.event.one", "test.event.two"]
        assert sub.priority == Priority.NORMAL
        assert sub.inject_at == InjectionPoint.AFTER_TOOL
        assert sub.core is False

    def test_discover_multiple_subscribers(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """Multiple valid TOML files are all discovered and loaded."""
        # Create templates
        (temp_templates_dir / "alpha.toon.j2").write_text("alpha template")
        (temp_templates_dir / "beta.toon.j2").write_text("beta template")
        (temp_templates_dir / "gamma.toon.j2").write_text("gamma template")

        # Create multiple subscribers
        for name, priority in [("alpha", "low"), ("beta", "normal"), ("gamma", "high")]:
            write_toml(
                temp_subscribers_dir / f"{name}.toml",
                f"""
                [subscriber]
                id = "{name}_sub"
                name = "{name.title()} Subscriber"

                [events]
                types = ["{name}.event"]

                [output]
                priority = "{priority}"
                template = "{name}.toon.j2"
                """,
            )

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()

        assert len(subscribers) == 3
        assert "alpha_sub" in subscribers
        assert "beta_sub" in subscribers
        assert "gamma_sub" in subscribers

        assert subscribers["alpha_sub"].priority == Priority.LOW
        assert subscribers["beta_sub"].priority == Priority.NORMAL
        assert subscribers["gamma_sub"].priority == Priority.HIGH

    def test_all_priority_values(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """All valid priority enum values are correctly parsed."""
        (temp_templates_dir / "test.toon.j2").write_text("test")

        for priority in ["low", "normal", "high", "critical"]:
            write_toml(
                temp_subscribers_dir / f"priority_{priority}.toml",
                f"""
                [subscriber]
                id = "priority_{priority}"
                name = "Priority {priority.title()}"

                [events]
                types = ["test.event"]

                [output]
                priority = "{priority}"
                template = "test.toon.j2"
                """,
            )

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()

        assert len(subscribers) == 4
        assert subscribers["priority_low"].priority == Priority.LOW
        assert subscribers["priority_normal"].priority == Priority.NORMAL
        assert subscribers["priority_high"].priority == Priority.HIGH
        assert subscribers["priority_critical"].priority == Priority.CRITICAL

    def test_all_injection_point_values(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """All valid inject_at enum values are correctly parsed."""
        (temp_templates_dir / "test.toon.j2").write_text("test")

        for inject_at in ["immediate", "turn_start", "after_tool", "turn_end"]:
            write_toml(
                temp_subscribers_dir / f"inject_{inject_at}.toml",
                f"""
                [subscriber]
                id = "inject_{inject_at}"
                name = "Inject {inject_at}"

                [events]
                types = ["test.event"]

                [output]
                inject_at = "{inject_at}"
                template = "test.toon.j2"
                """,
            )

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()

        assert len(subscribers) == 4
        assert subscribers["inject_immediate"].inject_at == InjectionPoint.IMMEDIATE
        assert subscribers["inject_turn_start"].inject_at == InjectionPoint.TURN_START
        assert subscribers["inject_after_tool"].inject_at == InjectionPoint.AFTER_TOOL
        assert subscribers["inject_turn_end"].inject_at == InjectionPoint.TURN_END

    def test_batching_config_parsed(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """Batching configuration is correctly parsed."""
        (temp_templates_dir / "test.toon.j2").write_text("test")

        write_toml(
            temp_subscribers_dir / "batched.toml",
            """
            [subscriber]
            id = "batched_sub"
            name = "Batched Subscriber"

            [events]
            types = ["batch.event"]

            [batching]
            window_ms = 5000
            max_size = 25
            dedupe_key = "event_id"
            dedupe_window_ms = 10000

            [output]
            template = "test.toon.j2"
            """,
        )

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()

        sub = subscribers["batched_sub"]
        assert sub.config.batching.window_ms == 5000
        assert sub.config.batching.max_size == 25
        assert sub.config.batching.dedupe_key == "event_id"
        assert sub.config.batching.dedupe_window_ms == 10000

    def test_core_subscriber_flag(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """Core subscriber flag is correctly parsed."""
        (temp_templates_dir / "test.toon.j2").write_text("test")

        write_toml(
            temp_subscribers_dir / "core_sub.toml",
            """
            [subscriber]
            id = "core_sub"
            name = "Core Subscriber"

            [events]
            types = ["core.event"]

            [output]
            template = "test.toon.j2"
            core = true
            """,
        )

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()

        assert subscribers["core_sub"].core is True

    def test_empty_directory_returns_empty_dict(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """Empty subscribers directory returns empty dictionary."""
        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()

        assert subscribers == {}

    def test_nonexistent_directory_returns_empty_dict(
        self, tmp_path: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-existent subscribers directory returns empty dict with warning."""
        nonexistent = tmp_path / "does_not_exist"

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=nonexistent,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        assert subscribers == {}
        assert "not found" in caplog.text

    def test_wildcard_event_matching(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """Subscriber with wildcard event type matches correctly."""
        (temp_templates_dir / "test.toon.j2").write_text("test")

        write_toml(
            temp_subscribers_dir / "wildcard.toml",
            """
            [subscriber]
            id = "wildcard_sub"
            name = "Wildcard Subscriber"

            [events]
            types = ["tool.*"]

            [output]
            template = "test.toon.j2"
            """,
        )

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        subscribers = loader.load_all()
        sub = subscribers["wildcard_sub"]

        assert sub.matches_event("tool.call.failure") is True
        assert sub.matches_event("tool.call.success") is True
        assert sub.matches_event("agent.loop.detected") is False


# =============================================================================
# T074: Test invalid subscriber handling (graceful skip)
# =============================================================================


class TestInvalidSubscriberHandling:
    """Tests for graceful handling of invalid subscriber configurations."""

    def test_missing_subscriber_id_skipped(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subscriber without ID is skipped with warning."""
        write_toml(
            temp_subscribers_dir / "no_id.toml",
            """
            [subscriber]
            name = "No ID Subscriber"

            [events]
            types = ["test.event"]

            [output]
            template = "test.toon.j2"
            """,
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        assert len(subscribers) == 0
        assert "subscriber.id" in caplog.text

    def test_missing_subscriber_name_skipped(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subscriber without name is skipped with warning."""
        write_toml(
            temp_subscribers_dir / "no_name.toml",
            """
            [subscriber]
            id = "no_name_sub"

            [events]
            types = ["test.event"]

            [output]
            template = "test.toon.j2"
            """,
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        assert len(subscribers) == 0
        assert "subscriber.name" in caplog.text

    def test_missing_event_types_skipped(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subscriber without event types is skipped with warning."""
        write_toml(
            temp_subscribers_dir / "no_events.toml",
            """
            [subscriber]
            id = "no_events_sub"
            name = "No Events"

            [events]

            [output]
            template = "test.toon.j2"
            """,
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        assert len(subscribers) == 0
        assert "events.types" in caplog.text

    def test_missing_template_field_skipped(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subscriber without template field is skipped with warning."""
        write_toml(
            temp_subscribers_dir / "no_template.toml",
            """
            [subscriber]
            id = "no_template_sub"
            name = "No Template"

            [events]
            types = ["test.event"]

            [output]
            priority = "normal"
            """,
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        assert len(subscribers) == 0
        assert "output.template" in caplog.text

    def test_invalid_priority_value_skipped(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subscriber with invalid priority value is skipped with warning."""
        write_toml(
            temp_subscribers_dir / "bad_priority.toml",
            """
            [subscriber]
            id = "bad_priority_sub"
            name = "Bad Priority"

            [events]
            types = ["test.event"]

            [output]
            priority = "super_high"
            template = "test.toon.j2"
            """,
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        assert len(subscribers) == 0
        assert "output.priority" in caplog.text
        assert "super_high" in caplog.text

    def test_invalid_inject_at_value_skipped(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subscriber with invalid inject_at value is skipped with warning."""
        write_toml(
            temp_subscribers_dir / "bad_inject.toml",
            """
            [subscriber]
            id = "bad_inject_sub"
            name = "Bad Inject"

            [events]
            types = ["test.event"]

            [output]
            inject_at = "whenever"
            template = "test.toon.j2"
            """,
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        assert len(subscribers) == 0
        assert "output.inject_at" in caplog.text
        assert "whenever" in caplog.text

    def test_invalid_severity_filter_skipped(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subscriber with invalid severity_filter is skipped with warning."""
        write_toml(
            temp_subscribers_dir / "bad_severity.toml",
            """
            [subscriber]
            id = "bad_severity_sub"
            name = "Bad Severity"

            [events]
            types = ["test.event"]
            severity_filter = "extreme"

            [output]
            template = "test.toon.j2"
            """,
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        assert len(subscribers) == 0
        assert "severity_filter" in caplog.text
        assert "extreme" in caplog.text

    def test_invalid_toml_syntax_skipped(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """File with invalid TOML syntax is skipped with warning."""
        # Write invalid TOML
        (temp_subscribers_dir / "broken.toml").write_text(
            "this is not valid [ toml {syntax"
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        assert len(subscribers) == 0
        assert "Invalid TOML syntax" in caplog.text or "broken.toml" in caplog.text

    def test_valid_subscribers_still_load_with_invalid_present(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Valid subscribers are loaded even when invalid ones are present."""
        (temp_templates_dir / "valid.toon.j2").write_text("valid template")

        # Create one valid subscriber
        write_toml(
            temp_subscribers_dir / "valid.toml",
            """
            [subscriber]
            id = "valid_sub"
            name = "Valid Subscriber"

            [events]
            types = ["valid.event"]

            [output]
            template = "valid.toon.j2"
            """,
        )

        # Create one invalid subscriber (missing required fields)
        write_toml(
            temp_subscribers_dir / "invalid.toml",
            """
            [subscriber]
            id = "invalid_sub"
            # Missing name

            [events]
            # Missing types

            [output]
            template = "test.toon.j2"
            """,
        )

        # Create another with bad syntax
        (temp_subscribers_dir / "broken.toml").write_text("invalid toml {{{")

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        # Only the valid one should be loaded
        assert len(subscribers) == 1
        assert "valid_sub" in subscribers
        assert subscribers["valid_sub"].name == "Valid Subscriber"

    def test_non_boolean_core_value_skipped(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subscriber with non-boolean core value is skipped."""
        write_toml(
            temp_subscribers_dir / "bad_core.toml",
            """
            [subscriber]
            id = "bad_core_sub"
            name = "Bad Core"

            [events]
            types = ["test.event"]

            [output]
            template = "test.toon.j2"
            core = "yes"
            """,
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        assert len(subscribers) == 0
        assert "output.core" in caplog.text


# =============================================================================
# T075: Test missing template handling
# =============================================================================


class TestMissingTemplateHandling:
    """Tests for handling subscribers with missing templates."""

    def test_missing_template_logs_warning(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subscriber with missing template file logs warning but still loads."""
        # Do NOT create the template file
        write_toml(
            temp_subscribers_dir / "missing_template.toml",
            """
            [subscriber]
            id = "missing_template_sub"
            name = "Missing Template Subscriber"

            [events]
            types = ["test.event"]

            [output]
            template = "nonexistent.toon.j2"
            """,
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        # Subscriber should still load (graceful degradation)
        assert len(subscribers) == 1
        assert "missing_template_sub" in subscribers

        # But warning should be logged about the template
        assert "Template not found" in caplog.text
        assert "nonexistent.toon.j2" in caplog.text
        assert "fallback" in caplog.text.lower()

    def test_existing_template_no_warning(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subscriber with existing template does not log template warning."""
        # Create the template file
        (temp_templates_dir / "exists.toon.j2").write_text("template content")

        write_toml(
            temp_subscribers_dir / "has_template.toml",
            """
            [subscriber]
            id = "has_template_sub"
            name = "Has Template Subscriber"

            [events]
            types = ["test.event"]

            [output]
            template = "exists.toon.j2"
            """,
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        assert len(subscribers) == 1
        assert "Template not found" not in caplog.text

    def test_mixed_template_availability(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Multiple subscribers with some having templates, some missing."""
        # Create only one template
        (temp_templates_dir / "present.toon.j2").write_text("present template")

        write_toml(
            temp_subscribers_dir / "with_template.toml",
            """
            [subscriber]
            id = "with_template"
            name = "With Template"

            [events]
            types = ["test.event"]

            [output]
            template = "present.toon.j2"
            """,
        )

        write_toml(
            temp_subscribers_dir / "without_template.toml",
            """
            [subscriber]
            id = "without_template"
            name = "Without Template"

            [events]
            types = ["test.event"]

            [output]
            template = "missing.toon.j2"
            """,
        )

        with caplog.at_level(logging.WARNING):
            loader = SubscriberLoader(
                subscribers_dir=temp_subscribers_dir,
                templates_dir=temp_templates_dir,
            )
            subscribers = loader.load_all()

        # Both should load
        assert len(subscribers) == 2
        assert "with_template" in subscribers
        assert "without_template" in subscribers

        # But only one template warning
        assert caplog.text.count("Template not found") == 1
        assert "missing.toon.j2" in caplog.text


# =============================================================================
# Additional validation tests
# =============================================================================


class TestSubscriberConfigValidation:
    """Tests for SubscriberConfig.validate_toml static method."""

    def test_valid_minimal_config(self) -> None:
        """Minimal valid configuration passes validation."""
        data = {
            "subscriber": {"id": "test", "name": "Test"},
            "events": {"types": ["test.event"]},
            "output": {"template": "test.toon.j2"},
        }

        result = SubscriberConfig.validate_toml(data)

        assert result.is_valid is True
        assert result.errors == []

    def test_valid_full_config(self) -> None:
        """Full configuration with all fields passes validation."""
        data = {
            "subscriber": {
                "id": "test",
                "name": "Test",
                "description": "Test description",
                "version": "2.0.0",
            },
            "events": {"types": ["test.event"], "severity_filter": "warning"},
            "batching": {
                "window_ms": 1000,
                "max_size": 5,
                "dedupe_key": "id",
                "dedupe_window_ms": 2000,
            },
            "output": {
                "priority": "high",
                "inject_at": "turn_start",
                "template": "test.toon.j2",
                "core": True,
            },
        }

        result = SubscriberConfig.validate_toml(data)

        assert result.is_valid is True
        assert result.errors == []

    def test_multiple_validation_errors(self) -> None:
        """Multiple validation errors are all reported."""
        data = {
            "subscriber": {},  # Missing id and name
            "events": {},  # Missing types
            "output": {},  # Missing template
        }

        result = SubscriberConfig.validate_toml(data)

        assert result.is_valid is False
        assert len(result.errors) >= 4
        assert any("subscriber.id" in e for e in result.errors)
        assert any("subscriber.name" in e for e in result.errors)
        assert any("events.types" in e for e in result.errors)
        assert any("output.template" in e for e in result.errors)

    def test_type_validation_for_event_types(self) -> None:
        """Event types must be a list of strings."""
        # Not a list
        data1 = {
            "subscriber": {"id": "test", "name": "Test"},
            "events": {"types": "single.event"},  # Should be a list
            "output": {"template": "test.toon.j2"},
        }
        result1 = SubscriberConfig.validate_toml(data1)
        assert result1.is_valid is False
        assert any("events.types" in e for e in result1.errors)

        # List with non-strings
        data2 = {
            "subscriber": {"id": "test", "name": "Test"},
            "events": {"types": ["valid.event", 123]},  # Contains non-string
            "output": {"template": "test.toon.j2"},
        }
        result2 = SubscriberConfig.validate_toml(data2)
        assert result2.is_valid is False
        assert any("strings" in e for e in result2.errors)


class TestSubscriberLoaderHelpers:
    """Tests for SubscriberLoader helper methods."""

    def test_get_subscriber_returns_loaded(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """get_subscriber returns loaded subscriber by ID."""
        (temp_templates_dir / "test.toon.j2").write_text("test")

        write_toml(
            temp_subscribers_dir / "test.toml",
            """
            [subscriber]
            id = "my_sub"
            name = "My Sub"

            [events]
            types = ["test"]

            [output]
            template = "test.toon.j2"
            """,
        )

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        loader.load_all()

        sub = loader.get_subscriber("my_sub")
        assert sub is not None
        assert sub.id == "my_sub"

    def test_get_subscriber_returns_none_for_unknown(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """get_subscriber returns None for unknown ID."""
        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        loader.load_all()

        sub = loader.get_subscriber("nonexistent")
        assert sub is None

    def test_get_all_subscribers(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """get_all_subscribers returns list of all loaded subscribers."""
        (temp_templates_dir / "test.toon.j2").write_text("test")

        for i in range(3):
            write_toml(
                temp_subscribers_dir / f"sub_{i}.toml",
                f"""
                [subscriber]
                id = "sub_{i}"
                name = "Sub {i}"

                [events]
                types = ["test"]

                [output]
                template = "test.toon.j2"
                """,
            )

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        loader.load_all()

        all_subs = loader.get_all_subscribers()
        assert len(all_subs) == 3

    def test_get_subscribers_for_event(
        self, temp_subscribers_dir: Path, temp_templates_dir: Path
    ) -> None:
        """get_subscribers_for_event returns matching subscribers."""
        (temp_templates_dir / "test.toon.j2").write_text("test")

        write_toml(
            temp_subscribers_dir / "tool_sub.toml",
            """
            [subscriber]
            id = "tool_sub"
            name = "Tool Sub"

            [events]
            types = ["tool.call.failure"]

            [output]
            template = "test.toon.j2"
            """,
        )

        write_toml(
            temp_subscribers_dir / "budget_sub.toml",
            """
            [subscriber]
            id = "budget_sub"
            name = "Budget Sub"

            [events]
            types = ["budget.warning"]

            [output]
            template = "test.toon.j2"
            """,
        )

        loader = SubscriberLoader(
            subscribers_dir=temp_subscribers_dir,
            templates_dir=temp_templates_dir,
        )
        loader.load_all()

        tool_subs = loader.get_subscribers_for_event("tool.call.failure")
        assert len(tool_subs) == 1
        assert tool_subs[0].id == "tool_sub"

        budget_subs = loader.get_subscribers_for_event("budget.warning")
        assert len(budget_subs) == 1
        assert budget_subs[0].id == "budget_sub"

        no_subs = loader.get_subscribers_for_event("unknown.event")
        assert len(no_subs) == 0
