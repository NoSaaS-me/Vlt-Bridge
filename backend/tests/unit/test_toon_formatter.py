"""Unit tests for the ANS ToonFormatter (T020)."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from uuid import uuid4

import pytest

from backend.src.services.ans.accumulator import Notification
from backend.src.services.ans.event import Event, Severity
from backend.src.services.ans.subscriber import InjectionPoint, Priority
from backend.src.services.ans.toon_formatter import (
    ToonFormatter,
    get_toon_formatter,
    TOON_AVAILABLE,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_templates_dir(tmp_path: Path) -> Path:
    """Create a temporary templates directory."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    return templates_dir


@pytest.fixture
def formatter(temp_templates_dir: Path) -> ToonFormatter:
    """Create a ToonFormatter with temp templates directory."""
    return ToonFormatter(templates_dir=temp_templates_dir)


@pytest.fixture
def sample_event() -> Event:
    """Create a sample event."""
    return Event(
        type="tool.call.failure",
        source="oracle_agent",
        severity=Severity.ERROR,
        payload={
            "tool_name": "bash",
            "error_type": "timeout",
            "error_message": "Command timed out",
        },
    )


@pytest.fixture
def sample_notification(sample_event: Event) -> Notification:
    """Create a sample notification with one event."""
    return Notification(
        id=uuid4(),
        subscriber_id="test_sub",
        content="",
        priority=Priority.NORMAL,
        inject_at=InjectionPoint.AFTER_TOOL,
        timestamp=datetime.now(timezone.utc),
        events=[sample_event],
    )


def write_template(templates_dir: Path, name: str, content: str) -> Path:
    """Write a template file."""
    path = templates_dir / name
    path.write_text(dedent(content).strip())
    return path


# =============================================================================
# Basic Initialization Tests
# =============================================================================


class TestToonFormatterInitialization:
    """Tests for ToonFormatter initialization."""

    def test_init_with_custom_templates_dir(self, temp_templates_dir: Path) -> None:
        """Formatter initializes with custom templates directory."""
        formatter = ToonFormatter(templates_dir=temp_templates_dir)

        assert formatter.templates_dir == temp_templates_dir

    def test_init_default_templates_dir(self) -> None:
        """Formatter uses default templates directory if not specified."""
        formatter = ToonFormatter()

        assert formatter.templates_dir.name == "templates"
        assert formatter.templates_dir.exists()

    def test_jinja2_environment_configured(
        self, temp_templates_dir: Path
    ) -> None:
        """Jinja2 environment is properly configured."""
        formatter = ToonFormatter(templates_dir=temp_templates_dir)

        assert formatter.env.autoescape is False  # TOON is plain text
        assert formatter.env.trim_blocks is True
        assert formatter.env.lstrip_blocks is True

    def test_custom_filters_registered(self, formatter: ToonFormatter) -> None:
        """Custom filters are registered with the environment."""
        assert "toon_encode" in formatter.env.filters
        assert "format_time" in formatter.env.filters
        assert "severity_prefix" in formatter.env.filters


# =============================================================================
# Template Existence Tests
# =============================================================================


class TestTemplateExistence:
    """Tests for template existence checking."""

    def test_template_exists_returns_true(
        self, temp_templates_dir: Path, formatter: ToonFormatter
    ) -> None:
        """template_exists returns True for existing template."""
        write_template(temp_templates_dir, "exists.j2", "content")

        assert formatter.template_exists("exists.j2") is True

    def test_template_exists_returns_false(self, formatter: ToonFormatter) -> None:
        """template_exists returns False for non-existent template."""
        assert formatter.template_exists("does_not_exist.j2") is False

    def test_list_templates_returns_j2_files(
        self, temp_templates_dir: Path, formatter: ToonFormatter
    ) -> None:
        """list_templates returns all .j2 files."""
        write_template(temp_templates_dir, "one.j2", "one")
        write_template(temp_templates_dir, "two.j2", "two")
        (temp_templates_dir / "not_template.txt").write_text("ignored")

        templates = formatter.list_templates()

        assert len(templates) == 2
        assert "one.j2" in templates
        assert "two.j2" in templates
        assert "not_template.txt" not in templates

    def test_list_templates_empty_dir(self, formatter: ToonFormatter) -> None:
        """list_templates returns empty list for empty directory."""
        templates = formatter.list_templates()

        assert templates == []

    def test_list_templates_nonexistent_dir(self, tmp_path: Path) -> None:
        """list_templates returns empty list for non-existent directory."""
        nonexistent = tmp_path / "does_not_exist"
        formatter = ToonFormatter(templates_dir=nonexistent)

        templates = formatter.list_templates()

        assert templates == []


# =============================================================================
# Notification Formatting Tests
# =============================================================================


class TestNotificationFormatting:
    """Tests for formatting notifications with templates."""

    def test_format_notification_simple_template(
        self,
        temp_templates_dir: Path,
        formatter: ToonFormatter,
        sample_notification: Notification,
    ) -> None:
        """Simple template renders correctly."""
        write_template(
            temp_templates_dir,
            "simple.j2",
            """
            notification: {{ notification.subscriber_id }}
            event_count: {{ count }}
            """,
        )

        result = formatter.format_notification(sample_notification, "simple.j2")

        assert "notification: test_sub" in result
        assert "event_count: 1" in result

    def test_format_notification_accesses_event(
        self,
        temp_templates_dir: Path,
        formatter: ToonFormatter,
        sample_notification: Notification,
    ) -> None:
        """Template can access single event."""
        write_template(
            temp_templates_dir,
            "event.j2",
            """
            tool: {{ event.payload.tool_name }}
            error: {{ event.payload.error_type }}
            """,
        )

        result = formatter.format_notification(sample_notification, "event.j2")

        assert "tool: bash" in result
        assert "error: timeout" in result

    def test_format_notification_accesses_events_list(
        self, temp_templates_dir: Path, formatter: ToonFormatter
    ) -> None:
        """Template can iterate over events list."""
        events = [
            Event(
                type="tool.call.failure",
                source="test",
                severity=Severity.ERROR,
                payload={"tool_name": f"tool_{i}", "error": "error"},
            )
            for i in range(3)
        ]
        notification = Notification(
            subscriber_id="test",
            events=events,
        )

        write_template(
            temp_templates_dir,
            "list.j2",
            """
            tools:
            {% for e in events %}
            - {{ e.payload.tool_name }}
            {% endfor %}
            """,
        )

        result = formatter.format_notification(notification, "list.j2")

        assert "tool_0" in result
        assert "tool_1" in result
        assert "tool_2" in result

    def test_format_notification_strips_whitespace(
        self, temp_templates_dir: Path, formatter: ToonFormatter, sample_notification: Notification
    ) -> None:
        """Result is stripped of leading/trailing whitespace."""
        write_template(
            temp_templates_dir,
            "whitespace.j2",
            """

            content here

            """,
        )

        result = formatter.format_notification(sample_notification, "whitespace.j2")

        assert result == "content here"


# =============================================================================
# Custom Filter Tests
# =============================================================================


class TestCustomFilters:
    """Tests for custom Jinja2 filters."""

    def test_format_time_filter(
        self,
        temp_templates_dir: Path,
        formatter: ToonFormatter,
    ) -> None:
        """format_time filter formats datetime correctly."""
        notification = Notification(
            subscriber_id="test",
            timestamp=datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc),
            events=[],
        )

        write_template(
            temp_templates_dir,
            "time.j2",
            "time: {{ timestamp | format_time }}",
        )

        result = formatter.format_notification(notification, "time.j2")

        assert result == "time: 10:30:45"

    def test_format_time_filter_non_datetime(
        self, temp_templates_dir: Path, formatter: ToonFormatter
    ) -> None:
        """format_time filter handles non-datetime values."""
        notification = Notification(
            subscriber_id="test",
            events=[],
        )

        write_template(
            temp_templates_dir,
            "time_str.j2",
            "{{ 'not a date' | format_time }}",
        )

        result = formatter.format_notification(notification, "time_str.j2")

        assert result == "not a date"

    def test_severity_prefix_filter_critical(self, formatter: ToonFormatter) -> None:
        """severity_prefix returns correct prefix for critical."""
        prefix = formatter._severity_prefix("critical")
        assert prefix == "!critical:"

    def test_severity_prefix_filter_error(self, formatter: ToonFormatter) -> None:
        """severity_prefix returns correct prefix for error."""
        prefix = formatter._severity_prefix("error")
        assert prefix == "!error:"

    def test_severity_prefix_filter_warning(self, formatter: ToonFormatter) -> None:
        """severity_prefix returns correct prefix for warning."""
        prefix = formatter._severity_prefix("warning")
        assert prefix == "!warning:"

    def test_severity_prefix_filter_info(self, formatter: ToonFormatter) -> None:
        """severity_prefix returns empty string for info."""
        prefix = formatter._severity_prefix("info")
        assert prefix == ""

    def test_severity_prefix_filter_debug(self, formatter: ToonFormatter) -> None:
        """severity_prefix returns empty string for debug."""
        prefix = formatter._severity_prefix("debug")
        assert prefix == ""

    def test_severity_prefix_filter_unknown(self, formatter: ToonFormatter) -> None:
        """severity_prefix returns empty string for unknown severity."""
        prefix = formatter._severity_prefix("unknown")
        assert prefix == ""


# =============================================================================
# Fallback Formatting Tests
# =============================================================================


class TestFallbackFormatting:
    """Tests for fallback formatting when templates fail."""

    def test_fallback_for_missing_template(
        self,
        formatter: ToonFormatter,
        sample_notification: Notification,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Missing template uses fallback format."""
        with caplog.at_level(logging.WARNING):
            result = formatter.format_notification(
                sample_notification, "nonexistent.j2"
            )

        assert "tool.call.failure" in result
        assert "Template not found" in caplog.text

    def test_fallback_for_template_error(
        self,
        temp_templates_dir: Path,
        formatter: ToonFormatter,
        sample_notification: Notification,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Template syntax error uses fallback format."""
        write_template(
            temp_templates_dir,
            "broken.j2",
            "{{ undefined_variable.missing }}",
        )

        with caplog.at_level(logging.ERROR):
            result = formatter.format_notification(sample_notification, "broken.j2")

        assert "tool.call.failure" in result
        assert "Template error" in caplog.text or "error" in caplog.text.lower()

    def test_fallback_single_event(
        self, formatter: ToonFormatter, sample_event: Event
    ) -> None:
        """Fallback format for single event shows type and payload."""
        notification = Notification(
            subscriber_id="test",
            events=[sample_event],
        )

        result = formatter._fallback_format(notification)

        assert "tool.call.failure" in result
        assert "bash" in result or "timeout" in result

    def test_fallback_multiple_events(self, formatter: ToonFormatter) -> None:
        """Fallback format for multiple events shows count and event types."""
        events = [
            Event(type=f"test.event.{i}", source="test", severity=Severity.INFO)
            for i in range(3)
        ]
        notification = Notification(
            subscriber_id="test",
            events=events,
        )

        result = formatter._fallback_format(notification)

        # Check for structured format with count
        assert "count: 3" in result
        assert "subscriber: test" in result
        assert "events:" in result
        assert "test.event.0" in result
        assert "test.event.1" in result
        assert "test.event.2" in result

    def test_fallback_no_events(self, formatter: ToonFormatter) -> None:
        """Fallback format for no events shows source and status."""
        notification = Notification(
            subscriber_id="empty_sub",
            events=[],
        )

        result = formatter._fallback_format(notification)

        assert "empty_sub" in result
        assert "no events" in result or "status:" in result


# =============================================================================
# Event Formatting Tests
# =============================================================================


class TestEventFormatting:
    """Tests for formatting individual events."""

    def test_format_event(self, formatter: ToonFormatter, sample_event: Event) -> None:
        """format_event returns type: payload format."""
        result = formatter.format_event(sample_event)

        assert result.startswith("tool.call.failure:")
        assert "bash" in result or "timeout" in result

    def test_format_events_batch_empty(self, formatter: ToonFormatter) -> None:
        """format_events_batch returns empty string for empty list."""
        result = formatter.format_events_batch([])

        assert result == ""

    def test_format_events_batch_single(
        self, formatter: ToonFormatter, sample_event: Event
    ) -> None:
        """format_events_batch with one event calls format_event."""
        result = formatter.format_events_batch([sample_event])

        assert result == formatter.format_event(sample_event)

    def test_format_events_batch_multiple(self, formatter: ToonFormatter) -> None:
        """format_events_batch with multiple events uses toon_encode."""
        events = [
            Event(
                type="test",
                source="test",
                severity=Severity.INFO,
                payload={"key": f"value_{i}"},
            )
            for i in range(3)
        ]

        result = formatter.format_events_batch(events)

        # Result should contain all payloads
        assert "value_0" in result
        assert "value_1" in result
        assert "value_2" in result


# =============================================================================
# Template Caching Tests
# =============================================================================


class TestTemplateCaching:
    """Tests for template caching behavior."""

    def test_template_cached(
        self,
        temp_templates_dir: Path,
        formatter: ToonFormatter,
        sample_notification: Notification,
    ) -> None:
        """Template is cached after first load."""
        write_template(temp_templates_dir, "cached.j2", "cached content")

        # First call loads template
        formatter.format_notification(sample_notification, "cached.j2")

        assert "cached.j2" in formatter._template_cache

        # Second call uses cache
        formatter.format_notification(sample_notification, "cached.j2")

        # Still only one entry
        assert len(formatter._template_cache) == 1


# =============================================================================
# Real Template Tests
# =============================================================================


class TestRealTemplates:
    """Tests using the real templates from the ANS."""

    def test_tool_failure_template_exists(self) -> None:
        """The tool_failure.toon.j2 template exists."""
        formatter = ToonFormatter()

        assert formatter.template_exists("tool_failure.toon.j2")

    def test_budget_warning_template_exists(self) -> None:
        """The budget_warning.toon.j2 template exists."""
        formatter = ToonFormatter()

        assert formatter.template_exists("budget_warning.toon.j2")

    def test_budget_exceeded_template_exists(self) -> None:
        """The budget_exceeded.toon.j2 template exists."""
        formatter = ToonFormatter()

        assert formatter.template_exists("budget_exceeded.toon.j2")

    def test_loop_detected_template_exists(self) -> None:
        """The loop_detected.toon.j2 template exists."""
        formatter = ToonFormatter()

        assert formatter.template_exists("loop_detected.toon.j2")

    def test_tool_failure_template_renders(self) -> None:
        """tool_failure.toon.j2 renders correctly with sample data."""
        formatter = ToonFormatter()

        event = Event(
            type="tool.call.failure",
            source="oracle_agent",
            severity=Severity.ERROR,
            payload={
                "tool_name": "bash",
                "error_type": "timeout",
                "error_message": "Command timed out after 30s",
            },
        )
        notification = Notification(
            subscriber_id="tool_failure",
            events=[event],
        )

        result = formatter.format_notification(notification, "tool_failure.toon.j2")

        assert "tool_fail" in result
        assert "bash" in result
        assert "timeout" in result

    def test_tool_failure_template_batch(self) -> None:
        """tool_failure.toon.j2 handles multiple events."""
        formatter = ToonFormatter()

        events = [
            Event(
                type="tool.call.failure",
                source="oracle_agent",
                severity=Severity.ERROR,
                payload={
                    "tool_name": f"tool_{i}",
                    "error_type": "error",
                    "error_message": f"Error {i}",
                },
            )
            for i in range(3)
        ]
        notification = Notification(
            subscriber_id="tool_failure",
            events=events,
        )

        result = formatter.format_notification(notification, "tool_failure.toon.j2")

        assert "tool_fails[3]" in result
        assert "tool_0" in result
        assert "tool_1" in result
        assert "tool_2" in result


# =============================================================================
# Global Singleton Tests
# =============================================================================


class TestGlobalFormatter:
    """Tests for the global ToonFormatter singleton."""

    def test_get_toon_formatter_returns_singleton(self) -> None:
        """get_toon_formatter returns the same instance."""
        formatter1 = get_toon_formatter()
        formatter2 = get_toon_formatter()

        assert formatter1 is formatter2

    def test_get_toon_formatter_uses_default_dir(self) -> None:
        """get_toon_formatter uses default templates directory."""
        formatter = get_toon_formatter()

        assert formatter.templates_dir.name == "templates"


# =============================================================================
# TOON Availability Tests
# =============================================================================


class TestToonAvailability:
    """Tests for TOON library availability."""

    def test_toon_available_flag(self) -> None:
        """TOON_AVAILABLE flag reflects library availability."""
        # This test just ensures the flag is a boolean
        assert isinstance(TOON_AVAILABLE, bool)

    def test_toon_encode_filter_works(
        self, temp_templates_dir: Path, formatter: ToonFormatter
    ) -> None:
        """toon_encode filter works (with or without python-toon)."""
        notification = Notification(
            subscriber_id="test",
            events=[],
        )

        write_template(
            temp_templates_dir,
            "encode.j2",
            "{{ {'key': 'value'} | toon_encode }}",
        )

        result = formatter.format_notification(notification, "encode.j2")

        # Should contain key/value in some format
        assert "key" in result
        assert "value" in result
