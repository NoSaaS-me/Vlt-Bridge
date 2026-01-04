"""TOON formatting with Jinja2 templates for the Agent Notification System."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from jinja2 import Environment, FileSystemLoader, TemplateError, TemplateNotFound

from .event import Event

if TYPE_CHECKING:
    from .accumulator import Notification


logger = logging.getLogger(__name__)


# Try to import python-toon, fall back gracefully
try:
    from toon import encode as toon_encode
    TOON_AVAILABLE = True
except ImportError:
    TOON_AVAILABLE = False
    logger.warning("python-toon not installed, falling back to simple format")

    def toon_encode(data: Any) -> str:
        """Fallback encoder when TOON is not available."""
        if isinstance(data, list):
            return "\n".join(str(item) for item in data)
        return str(data)


class ToonFormatter:
    """Formats notifications using Jinja2 templates and TOON encoding."""

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize the formatter.

        Args:
            templates_dir: Directory containing Jinja2 templates.
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"

        self.templates_dir = templates_dir

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=False,  # TOON is plain text
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters
        self.env.filters["toon_encode"] = toon_encode
        self.env.filters["format_time"] = self._format_time
        self.env.filters["severity_prefix"] = self._severity_prefix

        # Template cache
        self._template_cache: dict[str, Any] = {}

    def _format_time(self, dt: Any) -> str:
        """Format datetime for TOON output."""
        if hasattr(dt, "strftime"):
            return dt.strftime("%H:%M:%S")
        return str(dt)

    def _severity_prefix(self, severity: str) -> str:
        """Get prefix for severity level."""
        prefixes = {
            "critical": "!critical:",
            "error": "!error:",
            "warning": "!warning:",
            "info": "",
            "debug": "",
        }
        return prefixes.get(severity, "")

    def format_notification(
        self, notification: Notification, template_name: str
    ) -> str:
        """Format a notification using its template.

        Args:
            notification: The notification to format.
            template_name: Name of the template file.

        Returns:
            Formatted TOON string.
        """
        try:
            template = self._get_template(template_name)

            # Prepare template context
            context = {
                "notification": notification,
                "events": notification.events,
                "event": notification.events[0] if notification.events else None,
                "count": len(notification.events),
                "timestamp": notification.timestamp,
                "toon_encode": toon_encode,
            }

            content = template.render(**context)
            return content.strip()

        except TemplateNotFound:
            logger.warning(f"Template not found: {template_name}")
            return self._fallback_format(notification)
        except TemplateError as e:
            logger.error(f"Template error for {template_name}: {e}")
            return self._fallback_format(notification)
        except Exception as e:
            logger.error(f"Unexpected error formatting notification: {e}")
            return self._fallback_format(notification)

    def _get_template(self, template_name: str) -> Any:
        """Get a template from cache or load it."""
        if template_name not in self._template_cache:
            self._template_cache[template_name] = self.env.get_template(template_name)
        return self._template_cache[template_name]

    def _fallback_format(self, notification: Notification) -> str:
        """Fallback format when template fails.

        Provides a useful generic message when template rendering fails,
        ensuring the agent still receives notification information.
        """
        if not notification.events:
            return (
                f"!warning: system notification\n"
                f"source: {notification.subscriber_id}\n"
                f"status: no events to report"
            )

        if len(notification.events) == 1:
            event = notification.events[0]
            # Include severity prefix for visibility
            severity_prefix = self._severity_prefix(event.severity.value)
            lines = [
                f"{severity_prefix}system notification",
                f"type: {event.type}",
                f"source: {event.source}",
            ]
            # Add key payload fields
            if event.payload:
                for key, value in list(event.payload.items())[:5]:  # Limit to 5 fields
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)

        # Multiple events - tabular format with severity grouping
        lines = [
            "!info: system notifications",
            f"count: {len(notification.events)}",
            f"subscriber: {notification.subscriber_id}",
            "events:",
        ]
        for event in notification.events[:10]:  # Limit to 10 events in fallback
            severity_prefix = self._severity_prefix(event.severity.value)
            lines.append(f"  - {severity_prefix}{event.type}")
        if len(notification.events) > 10:
            lines.append(f"  ... and {len(notification.events) - 10} more")
        return "\n".join(lines)

    def format_event(self, event: Event) -> str:
        """Format a single event to TOON string."""
        return f"{event.type}: {toon_encode(event.payload)}"

    def format_events_batch(self, events: list[Event]) -> str:
        """Format multiple events into a batched TOON string."""
        if not events:
            return ""

        if len(events) == 1:
            return self.format_event(events[0])

        # Create tabular format if events have similar structure
        try:
            payloads = [event.payload for event in events]
            return toon_encode(payloads)
        except Exception:
            # Fallback to simple format
            return "\n".join(self.format_event(e) for e in events)

    def template_exists(self, template_name: str) -> bool:
        """Check if a template file exists."""
        template_path = self.templates_dir / template_name
        return template_path.exists()

    def list_templates(self) -> list[str]:
        """List all available template files."""
        if not self.templates_dir.exists():
            return []
        return [f.name for f in self.templates_dir.glob("*.j2")]


# Global singleton instance
_formatter: Optional[ToonFormatter] = None


def get_toon_formatter() -> ToonFormatter:
    """Get the global ToonFormatter instance."""
    global _formatter
    if _formatter is None:
        _formatter = ToonFormatter()
    return _formatter
