"""Subscriber configuration and loading for the Agent Notification System."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback


logger = logging.getLogger(__name__)


# Valid enum values for schema validation
VALID_PRIORITIES = {"low", "normal", "high", "critical"}
VALID_INJECTION_POINTS = {"immediate", "turn_start", "after_tool", "turn_end"}
VALID_SEVERITY_FILTERS = {"debug", "info", "warning", "error", "critical"}


class Priority(str, Enum):
    """Notification priority levels."""

    CRITICAL = "critical"  # Inject immediately, bypass batching
    HIGH = "high"          # Inject at next opportunity
    NORMAL = "normal"      # Standard batching behavior
    LOW = "low"            # Aggregate at turn end


class InjectionPoint(str, Enum):
    """When to inject notifications into the agent context."""

    IMMEDIATE = "immediate"    # Insert now (critical only)
    TURN_START = "turn_start"  # Before agent gets control
    AFTER_TOOL = "after_tool"  # Between tool result and next LLM call
    TURN_END = "turn_end"      # Summary before yielding


@dataclass
class ValidationResult:
    """Result of validating a subscriber configuration."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class BatchingConfig:
    """Batching configuration for a subscriber."""

    window_ms: int = 2000
    max_size: int = 10
    dedupe_key: Optional[str] = None
    dedupe_window_ms: int = 5000


@dataclass
class SubscriberConfig:
    """Configuration loaded from a subscriber TOML file."""

    id: str
    name: str
    description: str
    version: str
    event_types: list[str]
    severity_filter: str = "info"
    template: str = ""
    priority: Priority = Priority.NORMAL
    inject_at: InjectionPoint = InjectionPoint.AFTER_TOOL
    core: bool = False
    batching: BatchingConfig = field(default_factory=BatchingConfig)

    @classmethod
    def from_toml(cls, toml_data: dict[str, Any]) -> "SubscriberConfig":
        """Create a SubscriberConfig from parsed TOML data."""
        subscriber = toml_data.get("subscriber", {})
        events = toml_data.get("events", {})
        batching_data = toml_data.get("batching", {})
        output = toml_data.get("output", {})

        batching = BatchingConfig(
            window_ms=batching_data.get("window_ms", 2000),
            max_size=batching_data.get("max_size", 10),
            dedupe_key=batching_data.get("dedupe_key"),
            dedupe_window_ms=batching_data.get("dedupe_window_ms", 5000),
        )

        return cls(
            id=subscriber.get("id", ""),
            name=subscriber.get("name", ""),
            description=subscriber.get("description", ""),
            version=subscriber.get("version", "1.0.0"),
            event_types=events.get("types", []),
            severity_filter=events.get("severity_filter", "info"),
            template=output.get("template", ""),
            priority=Priority(output.get("priority", "normal")),
            inject_at=InjectionPoint(output.get("inject_at", "after_tool")),
            core=output.get("core", False),
            batching=batching,
        )

    @staticmethod
    def validate_toml(toml_data: dict[str, Any]) -> ValidationResult:
        """
        Validate TOML data against the subscriber schema.

        Required fields:
        - subscriber.id (string)
        - subscriber.name (string)
        - events.types (list of strings)
        - output.template (string)

        Optional fields with enum validation:
        - output.priority: low, normal, high, critical
        - output.inject_at: immediate, turn_start, after_tool, turn_end
        - events.severity_filter: debug, info, warning, error, critical
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check required sections
        subscriber = toml_data.get("subscriber", {})
        events = toml_data.get("events", {})
        output = toml_data.get("output", {})

        # Validate required fields
        if not subscriber.get("id"):
            errors.append("Missing required field: subscriber.id")
        elif not isinstance(subscriber.get("id"), str):
            errors.append("Field subscriber.id must be a string")

        if not subscriber.get("name"):
            errors.append("Missing required field: subscriber.name")
        elif not isinstance(subscriber.get("name"), str):
            errors.append("Field subscriber.name must be a string")

        if not events.get("types"):
            errors.append("Missing required field: events.types")
        elif not isinstance(events.get("types"), list):
            errors.append("Field events.types must be a list")
        elif not all(isinstance(t, str) for t in events.get("types", [])):
            errors.append("Field events.types must contain only strings")

        if not output.get("template"):
            errors.append("Missing required field: output.template")
        elif not isinstance(output.get("template"), str):
            errors.append("Field output.template must be a string")

        # Validate enum fields
        priority = output.get("priority")
        if priority is not None and priority not in VALID_PRIORITIES:
            errors.append(
                f"Invalid value for output.priority: '{priority}'. "
                f"Must be one of: {', '.join(sorted(VALID_PRIORITIES))}"
            )

        inject_at = output.get("inject_at")
        if inject_at is not None and inject_at not in VALID_INJECTION_POINTS:
            errors.append(
                f"Invalid value for output.inject_at: '{inject_at}'. "
                f"Must be one of: {', '.join(sorted(VALID_INJECTION_POINTS))}"
            )

        severity_filter = events.get("severity_filter")
        if severity_filter is not None and severity_filter not in VALID_SEVERITY_FILTERS:
            errors.append(
                f"Invalid value for events.severity_filter: '{severity_filter}'. "
                f"Must be one of: {', '.join(sorted(VALID_SEVERITY_FILTERS))}"
            )

        # Validate optional field types
        if "core" in output and not isinstance(output.get("core"), bool):
            errors.append("Field output.core must be a boolean")

        # Validate batching section if present
        batching = toml_data.get("batching", {})
        if batching:
            if "window_ms" in batching and not isinstance(batching.get("window_ms"), int):
                errors.append("Field batching.window_ms must be an integer")
            if "max_size" in batching and not isinstance(batching.get("max_size"), int):
                errors.append("Field batching.max_size must be an integer")
            if "dedupe_window_ms" in batching and not isinstance(batching.get("dedupe_window_ms"), int):
                errors.append("Field batching.dedupe_window_ms must be an integer")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


@dataclass
class Subscriber:
    """A loaded subscriber ready to process events."""

    config: SubscriberConfig
    enabled: bool = True

    @property
    def id(self) -> str:
        """Get the subscriber ID."""
        return self.config.id

    @property
    def name(self) -> str:
        """Get the subscriber name."""
        return self.config.name

    @property
    def event_types(self) -> list[str]:
        """Get the event types this subscriber listens to."""
        return self.config.event_types

    @property
    def priority(self) -> Priority:
        """Get the notification priority level."""
        return self.config.priority

    @property
    def inject_at(self) -> InjectionPoint:
        """Get the injection point for notifications."""
        return self.config.inject_at

    @property
    def core(self) -> bool:
        """Check if this is a core subscriber (cannot be disabled)."""
        return self.config.core

    def matches_event(self, event_type: str) -> bool:
        """Check if this subscriber handles the given event type."""
        for subscribed_type in self.event_types:
            if subscribed_type == event_type:
                return True
            # Support wildcard matching: "tool.*" matches "tool.call.failure"
            if subscribed_type.endswith(".*"):
                prefix = subscribed_type[:-2]
                if event_type.startswith(prefix + "."):
                    return True
        return False


class SubscriberLoader:
    """Loads subscriber configurations from TOML files."""

    def __init__(
        self,
        subscribers_dir: Optional[Path] = None,
        templates_dir: Optional[Path] = None,
    ):
        """
        Initialize loader with subscriber and templates directory paths.

        Args:
            subscribers_dir: Directory containing subscriber TOML files.
                            Defaults to ./subscribers relative to this module.
            templates_dir: Directory containing Jinja2 template files.
                          Defaults to ./templates relative to this module.
        """
        if subscribers_dir is None:
            # Default to the subscribers directory relative to this module
            subscribers_dir = Path(__file__).parent / "subscribers"
        if templates_dir is None:
            # Default to the templates directory relative to this module
            templates_dir = Path(__file__).parent / "templates"

        self.subscribers_dir = subscribers_dir
        self.templates_dir = templates_dir
        self._subscribers: dict[str, Subscriber] = {}

    def load_all(self) -> dict[str, Subscriber]:
        """
        Load all subscriber configs from the subscribers directory.

        Uses glob-based discovery to find all *.toml files in the subscribers
        directory. Each file is validated against the subscriber schema before
        loading. Invalid configs are skipped with warnings logged.

        Returns:
            Dictionary mapping subscriber IDs to loaded Subscriber objects.
        """
        self._subscribers.clear()

        if not self.subscribers_dir.exists():
            logger.warning(f"Subscribers directory not found: {self.subscribers_dir}")
            return self._subscribers

        # Glob-based discovery of all TOML files (T069)
        toml_files = list(self.subscribers_dir.glob("*.toml"))
        logger.info(f"Discovered {len(toml_files)} subscriber config files in {self.subscribers_dir}")

        for toml_file in sorted(toml_files):  # Sort for deterministic order
            try:
                subscriber = self._load_subscriber(toml_file)
                if subscriber:
                    self._subscribers[subscriber.id] = subscriber
                    logger.info(f"Loaded subscriber: {subscriber.id} from {toml_file.name}")
            except Exception as e:
                # T071: Log warning and skip invalid config files
                logger.warning(f"Failed to load subscriber from {toml_file.name}: {e}")
                continue

        logger.info(f"Successfully loaded {len(self._subscribers)} subscribers")
        return self._subscribers

    def _load_subscriber(self, toml_file: Path) -> Optional[Subscriber]:
        """
        Load a single subscriber from a TOML file.

        Performs validation against the subscriber schema and verifies
        that the referenced Jinja2 template exists.

        Args:
            toml_file: Path to the subscriber TOML configuration file.

        Returns:
            A Subscriber object if valid, None otherwise.
        """
        try:
            with open(toml_file, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            # T071: Log warning for TOML parse errors
            logger.warning(
                f"Invalid TOML syntax in {toml_file.name}: {e}"
            )
            return None
        except Exception as e:
            logger.warning(f"Error reading {toml_file.name}: {e}")
            return None

        # T070: Validate TOML against subscriber schema
        validation = SubscriberConfig.validate_toml(data)
        if not validation.is_valid:
            # T071: Log warning with specific field errors
            error_details = "; ".join(validation.errors)
            logger.warning(
                f"Invalid subscriber config in {toml_file.name}: {error_details}"
            )
            return None

        # Log any validation warnings
        for warning in validation.warnings:
            logger.warning(f"Subscriber {toml_file.name}: {warning}")

        try:
            config = SubscriberConfig.from_toml(data)
        except ValueError as e:
            # Handle enum conversion errors
            logger.warning(f"Failed to parse config from {toml_file.name}: {e}")
            return None

        # T072: Verify Jinja2 template exists for each subscriber
        if config.template:
            template_path = self.templates_dir / config.template
            if not template_path.exists():
                logger.warning(
                    f"Template not found for subscriber '{config.id}': "
                    f"{config.template} (expected at {template_path}). "
                    f"Subscriber will use fallback formatting."
                )
                # Note: Subscriber still loads, just with a warning about the template

        return Subscriber(config=config)

    def get_subscriber(self, subscriber_id: str) -> Optional[Subscriber]:
        """Get a loaded subscriber by ID."""
        return self._subscribers.get(subscriber_id)

    def get_all_subscribers(self) -> list[Subscriber]:
        """Get all loaded subscribers."""
        return list(self._subscribers.values())

    def get_subscribers_for_event(self, event_type: str) -> list[Subscriber]:
        """Get all subscribers that handle the given event type."""
        return [
            sub for sub in self._subscribers.values()
            if sub.enabled and sub.matches_event(event_type)
        ]
