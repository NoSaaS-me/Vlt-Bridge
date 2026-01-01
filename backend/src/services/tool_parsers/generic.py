"""Generic XML tool call parser with flexible matching.

This is the fallback parser that handles various XML formats with more
permissive matching. It tries to handle any XML-like structure that resembles
function calling, including:
- Tags with prefixes: <prefix:function_calls>
- Tags with special characters
- Variations in tag naming
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from .base import ParsedToolCall, ToolCallParser

logger = logging.getLogger(__name__)


class GenericXMLParser(ToolCallParser):
    """Generic parser for various XML-style formats.

    This parser uses more flexible regex patterns to match XML-like
    structures that may not conform to standard formats. It should be
    used as a last resort after more specific parsers fail.
    """

    # Pattern for function_calls blocks - matches any prefix
    FUNCTION_CALLS_PATTERN = re.compile(
        r'<[^>]*?function_calls[^>]*?>\s*(.*?)\s*</[^>]*?function_calls[^>]*?>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern for invoke elements - matches any prefix
    INVOKE_PATTERN = re.compile(
        r'<[^>]*?invoke\s+name=["\']([^"\']+)["\'][^>]*?>\s*(.*?)\s*</[^>]*?invoke[^>]*?>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern for parameter elements - matches any prefix
    PARAMETER_PATTERN = re.compile(
        r'<[^>]*?parameter\s+name=["\']([^"\']+)["\'][^>]*?>([^<]*)</[^>]*?parameter[^>]*?>',
        re.DOTALL | re.IGNORECASE
    )

    @property
    def name(self) -> str:
        """Return parser name."""
        return "GenericXMLParser"

    def can_parse(self, content: str) -> bool:
        """Check if content has any XML-like function call structure.

        Args:
            content: Text to check

        Returns:
            True if content contains any invoke or function_calls tags
        """
        # Look for any variation of invoke or function_calls
        return bool(
            self.FUNCTION_CALLS_PATTERN.search(content) or
            self.INVOKE_PATTERN.search(content)
        )

    def parse(self, content: str) -> Tuple[List[ParsedToolCall], str]:
        """Parse generic XML function calls.

        This parser tries to extract tool calls from any XML-like format.

        Args:
            content: Text containing tool calls

        Returns:
            Tuple of (parsed_calls, cleaned_content)
        """
        tool_calls: List[ParsedToolCall] = []
        cleaned_content = content

        # First try to find function_calls blocks
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

        # Also check for standalone invoke elements
        if not tool_calls:
            for invoke_match in self.INVOKE_PATTERN.finditer(content):
                tool_name = invoke_match.group(1)
                params_content = invoke_match.group(2)
                raw_xml = invoke_match.group(0)

                arguments = self._parse_parameters(params_content)

                tool_calls.append(ParsedToolCall(
                    name=tool_name,
                    arguments=arguments,
                    raw_xml=raw_xml
                ))

                cleaned_content = cleaned_content.replace(raw_xml, '')

        # Clean up extra whitespace
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content.strip())

        if tool_calls:
            logger.info(
                f"[{self.name}] Parsed {len(tool_calls)} tool call(s) using fallback parser"
            )
        else:
            # Log when no tool calls found even with generic parser
            if any(keyword in content.lower() for keyword in ['<function', '<invoke', 'tool_call']):
                logger.warning(
                    f"[{self.name}] Content looks like it has tool calls but none parsed"
                )
                logger.warning(f"[{self.name}] Content sample: {content[:500]}")

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

        # Handle negative integers
        if value.startswith('-') and value[1:].isdigit():
            return int(value)

        # Try JSON parsing for complex types
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Keep as string if JSON parsing fails
            return value
