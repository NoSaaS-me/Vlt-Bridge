"""Signal parser for Oracle agent responses.

Extracts and parses XML signals emitted by the Oracle agent.
Signals are always at the end of the response, on their own line.

Performance target: <10ms for 10KB response (per research.md)

Part of feature 020-bt-oracle-agent.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..models.signals import Signal, SignalType, SIGNAL_FIELDS_MAP

logger = logging.getLogger(__name__)


# =============================================================================
# Regex Patterns
# =============================================================================

# Main signal pattern - matches <signal type="...">...</signal>
# Captures: (1) signal type, (2) inner content
# Uses re.DOTALL to match across lines
# Pattern allows optional whitespace and optional confidence attribute
SIGNAL_PATTERN = re.compile(
    r'<signal\s+type=["\']([^"\']+)["\'](?:\s+[^>]*)?\s*>\s*(.*?)\s*</signal>',
    re.DOTALL | re.IGNORECASE,
)

# Pattern to extract individual field elements from signal content
# Captures: (1) field name, (2) field value
FIELD_PATTERN = re.compile(
    r'<(\w+)>(.*?)</\1>',
    re.DOTALL,
)

# Confidence field specifically (may appear as attribute or element)
CONFIDENCE_ATTR_PATTERN = re.compile(
    r'confidence=["\']([0-9.]+)["\']',
    re.IGNORECASE,
)


# =============================================================================
# Parser Functions
# =============================================================================


def parse_signal(response: str) -> Optional[Signal]:
    """Parse signal from Oracle agent response.

    Extracts the XML signal block from the end of the response and
    parses it into a Signal model.

    Algorithm:
    1. Search for <signal type="...">...</signal> pattern
    2. Extract signal type from attribute
    3. Parse inner fields from XML elements
    4. Extract confidence (element or attribute)
    5. Validate fields against signal type schema
    6. Return Signal model or None if no signal found

    Args:
        response: Full LLM response text

    Returns:
        Parsed Signal model, or None if no signal found or parse error

    Example:
        >>> response = '''Here's what I found.
        ...
        ... <signal type="need_turn">
        ...   <reason>Need to verify the API response</reason>
        ...   <confidence>0.85</confidence>
        ... </signal>'''
        >>> signal = parse_signal(response)
        >>> signal.type
        SignalType.NEED_TURN
        >>> signal.confidence
        0.85
    """
    if not response or not isinstance(response, str):
        return None

    # Step 1: Find signal block
    match = SIGNAL_PATTERN.search(response)
    if not match:
        logger.debug("No signal pattern found in response")
        return None

    signal_type_str = match.group(1).lower()
    inner_content = match.group(2)
    raw_xml = match.group(0)

    # Step 2: Parse signal type
    try:
        signal_type = SignalType(signal_type_str)
    except ValueError:
        logger.warning(f"Unknown signal type: {signal_type_str}")
        return None

    # Step 3: Parse inner fields
    fields = _parse_fields(inner_content)

    # Step 4: Extract confidence
    confidence = _extract_confidence(inner_content, raw_xml)

    # Step 5: Validate fields against expected schema
    expected_model = SIGNAL_FIELDS_MAP.get(signal_type)
    if expected_model is None:
        logger.warning(f"No field model for signal type: {signal_type}")
        return None

    # Check required fields exist
    try:
        # This validates the fields match the expected schema
        expected_model.model_validate(fields)
    except Exception as e:
        logger.warning(f"Signal fields validation failed: {e}")
        # Continue anyway - we'll let Signal model do final validation
        pass

    # Step 6: Create Signal model
    try:
        signal = Signal(
            type=signal_type,
            confidence=confidence,
            fields=fields,
            raw_xml=raw_xml,
            timestamp=datetime.now(timezone.utc),
        )
        logger.debug(f"Parsed signal: type={signal_type}, confidence={confidence}")
        return signal
    except Exception as e:
        logger.warning(f"Failed to create Signal model: {e}")
        return None


def strip_signal(response: str) -> str:
    """Remove signal XML from response text.

    Returns the response with the signal block removed and
    whitespace normalized.

    Args:
        response: Full LLM response text

    Returns:
        Response text with signal removed and trailing whitespace stripped

    Example:
        >>> response = '''Here's the answer.
        ...
        ... <signal type="context_sufficient">
        ...   <sources_found>3</sources_found>
        ...   <confidence>0.9</confidence>
        ... </signal>'''
        >>> clean = strip_signal(response)
        >>> clean
        "Here's the answer."
    """
    if not response or not isinstance(response, str):
        return response or ""

    # Remove signal block
    cleaned = SIGNAL_PATTERN.sub("", response)

    # Normalize whitespace: collapse multiple newlines, strip trailing
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()

    return cleaned


def parse_and_strip(response: str) -> Tuple[Optional[Signal], str]:
    """Parse signal and return both signal and cleaned response.

    Convenience function that combines parse_signal and strip_signal.

    Args:
        response: Full LLM response text

    Returns:
        Tuple of (Signal or None, cleaned response text)

    Example:
        >>> response = "Answer here. <signal type='stuck'>...</signal>"
        >>> signal, clean = parse_and_strip(response)
        >>> signal is not None
        True
        >>> "<signal" in clean
        False
    """
    signal = parse_signal(response)
    cleaned = strip_signal(response)
    return signal, cleaned


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_fields(content: str) -> Dict[str, Any]:
    """Parse field elements from signal inner content.

    Handles:
    - Simple string fields: <reason>text</reason>
    - Numeric fields: <expected_turns>2</expected_turns>
    - List fields: <attempted>["tool1", "tool2"]</attempted>
    - Boolean fields (as strings): <success>true</success>

    Args:
        content: Inner XML content between signal tags

    Returns:
        Dictionary of field name -> parsed value
    """
    fields: Dict[str, Any] = {}

    for match in FIELD_PATTERN.finditer(content):
        field_name = match.group(1).lower()
        field_value = match.group(2).strip()

        # Skip confidence - handled separately
        if field_name == "confidence":
            continue

        # Parse value based on content
        fields[field_name] = _parse_field_value(field_value)

    return fields


def _parse_field_value(value: str) -> Any:
    """Parse a field value string into appropriate Python type.

    Parsing rules:
    1. JSON array -> list (handles ["a", "b"] format)
    2. Integer string -> int
    3. Float string -> float (only if has decimal point)
    4. "true"/"false" -> bool
    5. Otherwise -> string

    Args:
        value: Raw string value from XML

    Returns:
        Parsed Python value
    """
    value = value.strip()

    # Empty string
    if not value:
        return ""

    # JSON array (used for lists like attempted tools)
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON, return as string
            pass

    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Integer (no decimal point)
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        try:
            return int(value)
        except ValueError:
            pass

    # Float (has decimal point)
    if "." in value:
        try:
            return float(value)
        except ValueError:
            pass

    # Default: string
    return value


def _extract_confidence(content: str, raw_xml: str) -> float:
    """Extract confidence value from signal.

    Confidence can appear as:
    1. Element: <confidence>0.85</confidence>
    2. Attribute: <signal type="..." confidence="0.85">

    Default: 0.5 if not found

    Args:
        content: Inner XML content
        raw_xml: Full signal XML (for attribute check)

    Returns:
        Confidence value (0.0-1.0)
    """
    # Try element first
    for match in FIELD_PATTERN.finditer(content):
        if match.group(1).lower() == "confidence":
            try:
                conf = float(match.group(2).strip())
                return max(0.0, min(1.0, conf))  # Clamp to 0-1
            except ValueError:
                pass

    # Try attribute
    attr_match = CONFIDENCE_ATTR_PATTERN.search(raw_xml)
    if attr_match:
        try:
            conf = float(attr_match.group(1))
            return max(0.0, min(1.0, conf))
        except ValueError:
            pass

    # Default confidence
    logger.debug("No confidence found, using default 0.5")
    return 0.5


__all__ = [
    "parse_signal",
    "strip_signal",
    "parse_and_strip",
    "SIGNAL_PATTERN",
]
