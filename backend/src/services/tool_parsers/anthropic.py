"""Anthropic-style XML tool call parser.

Handles Anthropic's tool calling format which may use variations like:
    <function_calls>
    <invoke name="tool_name">
    <parameter name="param_name">value</parameter>
    </invoke>
    </function_calls>

Or standalone invokes without function_calls wrapper.
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from .base import ParsedToolCall, ToolCallParser

logger = logging.getLogger(__name__)


class AnthropicXMLParser(ToolCallParser):
    """Parser for Anthropic's XML-style tool calling format."""

    # Pattern for invoke elements (may appear standalone)
    INVOKE_PATTERN = re.compile(
        r'<invoke\s+name=["\']([^"\']+)["\'][^>]*?>\s*(.*?)\s*</invoke>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern for parameter elements
    PARAMETER_PATTERN = re.compile(
        r'<parameter\s+name=["\']([^"\']+)["\'][^>]*?>([^<]*)</parameter>',
        re.DOTALL | re.IGNORECASE
    )

    @property
    def name(self) -> str:
        """Return parser name."""
        return "AnthropicXMLParser"

    def can_parse(self, content: str) -> bool:
        """Check if content contains Anthropic-style invoke elements.

        Args:
            content: Text to check

        Returns:
            True if content matches Anthropic format
        """
        return bool(self.INVOKE_PATTERN.search(content))

    def parse(self, content: str) -> Tuple[List[ParsedToolCall], str]:
        """Parse Anthropic XML function calls.

        This parser handles standalone <invoke> elements that may not be
        wrapped in a <function_calls> block.

        Args:
            content: Text containing tool calls

        Returns:
            Tuple of (parsed_calls, cleaned_content)
        """
        tool_calls: List[ParsedToolCall] = []
        cleaned_content = content

        # Find all invoke elements (standalone or in blocks)
        for invoke_match in self.INVOKE_PATTERN.finditer(content):
            tool_name = invoke_match.group(1)
            params_content = invoke_match.group(2)
            raw_xml = invoke_match.group(0)

            # Extract parameters
            arguments = self._parse_parameters(params_content)

            tool_calls.append(ParsedToolCall(
                name=tool_name,
                arguments=arguments,
                raw_xml=raw_xml
            ))

            # Remove this invoke block from content
            cleaned_content = cleaned_content.replace(raw_xml, '')

        # Clean up extra whitespace
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content.strip())

        logger.debug(f"[{self.name}] Parsed {len(tool_calls)} tool call(s)")
        return tool_calls, cleaned_content

    def _parse_parameters(self, params_content: str) -> Dict[str, Any]:
        """Extract and parse parameter values.

        Args:
            params_content: XML content containing parameters

        Returns:
            Dictionary of parameter names to parsed values
        """
        arguments: Dict[str, Any] = {}

        for param_match in self.PARAMETER_PATTERN.finditer(params_content):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()

            # Parse value with type inference
            arguments[param_name] = self._parse_value(param_value)

        return arguments

    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse parameter value with type inference.

        Args:
            value: String value to parse

        Returns:
            Parsed value as appropriate Python type
        """
        # Handle booleans
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Handle integers
        if value.isdigit():
            return int(value)

        # Try JSON parsing for complex types
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Keep as string if JSON parsing fails
            return value
