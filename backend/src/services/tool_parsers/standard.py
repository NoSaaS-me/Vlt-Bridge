"""Standard XML tool call parser.

Handles the most common format used by models that support XML-style tool calling:

    <function_calls>
    <invoke name="tool_name">
    <parameter name="param_name">value</parameter>
    </invoke>
    </function_calls>
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from .base import ParsedToolCall, ToolCallParser

logger = logging.getLogger(__name__)


class StandardXMLParser(ToolCallParser):
    """Parser for standard <function_calls> format."""

    # Pattern for function_calls blocks
    FUNCTION_CALLS_PATTERN = re.compile(
        r'<function_calls>\s*(.*?)\s*</function_calls>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern for invoke elements with name attribute
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
        return "StandardXMLParser"

    def can_parse(self, content: str) -> bool:
        """Check if content contains standard function_calls blocks.

        Args:
            content: Text to check

        Returns:
            True if content matches standard format
        """
        return bool(self.FUNCTION_CALLS_PATTERN.search(content))

    def parse(self, content: str) -> Tuple[List[ParsedToolCall], str]:
        """Parse standard XML function calls.

        Args:
            content: Text containing tool calls

        Returns:
            Tuple of (parsed_calls, cleaned_content)
        """
        tool_calls: List[ParsedToolCall] = []
        cleaned_content = content

        # Find all function_calls blocks
        for fc_match in self.FUNCTION_CALLS_PATTERN.finditer(content):
            block_content = fc_match.group(1)
            raw_xml = fc_match.group(0)
            cleaned_content = cleaned_content.replace(raw_xml, '')

            # Find all invoke elements within this block
            for invoke_match in self.INVOKE_PATTERN.finditer(block_content):
                tool_name = invoke_match.group(1)
                params_content = invoke_match.group(2)

                # Extract parameters
                arguments = self._parse_parameters(params_content)

                tool_calls.append(ParsedToolCall(
                    name=tool_name,
                    arguments=arguments,
                    raw_xml=raw_xml
                ))

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
