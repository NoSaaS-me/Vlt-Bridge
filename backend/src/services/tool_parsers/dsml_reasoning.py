"""DSML Reasoning parser for DeepSeek tool calls in reasoning field.

DeepSeek sometimes outputs tool calls in the `reasoning` field using a custom
DSML (DeepSeek Markup Language) format with ASCII pipe characters:

    </|DSML|function_calls>
    |DSML|invoke name="tool_name">
        |DSML|parameter name="param">value</|DSML|parameter>
    </|DSML|invoke>
    </|DSML|function_calls>

This format differs from the standard DeepSeek XML format which uses Unicode
fullwidth vertical bars (｜). This parser handles both variants.

The key insight is that DeepSeek's reasoning field may contain tool calls that
need to be parsed and executed, even though the API response shows:
- finish_reason=stop
- content=""
- tool_calls={}
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from .base import ParsedToolCall, ToolCallParser

logger = logging.getLogger(__name__)


class DSMLReasoningParser(ToolCallParser):
    """Parser for DSML tool calls in reasoning field.

    This parser handles DeepSeek's custom markup in the reasoning field,
    supporting both ASCII pipe (|DSML|) and Unicode fullwidth vertical bar (｜DSML｜).
    """

    # ASCII pipe variant patterns - the format from logs
    # Note: Tags may be malformed with missing opening < or closing >
    ASCII_FUNCTION_CALLS_PATTERN = re.compile(
        r'<?\|DSML\|function_calls[^>]*>?\s*(.*?)\s*<?\/?[\|｜]DSML[\|｜]function_calls[^>]*>?',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern for invoke elements - handles malformed tags
    ASCII_INVOKE_PATTERN = re.compile(
        r'<?\|DSML\|invoke\s+name=["\']?([^"\'>\s]+)["\']?[^>]*>?\s*(.*?)\s*<?\/?[\|｜]DSML[\|｜]invoke[^>]*>?',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern for parameter elements - handles malformed tags
    ASCII_PARAMETER_PATTERN = re.compile(
        r'<?\|DSML\|parameter\s+name=["\']?([^"\'>\s]+)["\']?[^>]*>?\s*([^<]*)\s*<?\/?[\|｜]DSML[\|｜]parameter[^>]*>?',
        re.DOTALL | re.IGNORECASE
    )

    # Alternative pattern: newline-separated format from logs
    # |DSML|invoke
    # </|DSML|parameter>
    # limit
    # 10
    NEWLINE_INVOKE_PATTERN = re.compile(
        r'\|DSML\|invoke[^<\n]*\n',
        re.IGNORECASE
    )

    # Detect DSML markers
    DSML_MARKER_PATTERN = re.compile(r'[\|｜]DSML[\|｜]', re.IGNORECASE)

    @property
    def name(self) -> str:
        """Return parser name."""
        return "DSMLReasoningParser"

    def can_parse(self, content: str) -> bool:
        """Check if content contains DSML markers.

        Args:
            content: Text to check (typically reasoning field content)

        Returns:
            True if content contains DSML markers
        """
        return bool(self.DSML_MARKER_PATTERN.search(content))

    def parse(self, content: str) -> Tuple[List[ParsedToolCall], str]:
        """Parse DSML tool calls from content.

        This handles both well-formed and malformed DSML markup.

        Args:
            content: Text containing DSML tool calls

        Returns:
            Tuple of (parsed_calls, cleaned_content)
        """
        tool_calls: List[ParsedToolCall] = []
        cleaned_content = content

        logger.debug(f"[{self.name}] Parsing content ({len(content)} chars)")
        logger.debug(f"[{self.name}] Content sample: {content[:500]}")

        # Try structured parsing first
        tool_calls = self._parse_structured(content)

        if tool_calls:
            # Remove parsed XML from content
            for match in self.ASCII_FUNCTION_CALLS_PATTERN.finditer(content):
                cleaned_content = cleaned_content.replace(match.group(0), '')
            for match in self.ASCII_INVOKE_PATTERN.finditer(content):
                cleaned_content = cleaned_content.replace(match.group(0), '')
            cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content.strip())
            logger.info(f"[{self.name}] Parsed {len(tool_calls)} tool call(s) via structured parsing")
            return tool_calls, cleaned_content

        # Try lenient parsing for malformed content
        tool_calls = self._parse_lenient(content)

        if tool_calls:
            # For lenient parsing, we can't easily clean the content
            # Just return empty content since it was reasoning anyway
            logger.info(f"[{self.name}] Parsed {len(tool_calls)} tool call(s) via lenient parsing")
            return tool_calls, ""

        logger.debug(f"[{self.name}] No DSML tool calls found in content")
        return [], content

    def _parse_structured(self, content: str) -> List[ParsedToolCall]:
        """Parse well-formed DSML content.

        Args:
            content: DSML content

        Returns:
            List of parsed tool calls
        """
        tool_calls: List[ParsedToolCall] = []

        # Try function_calls blocks first
        for fc_match in self.ASCII_FUNCTION_CALLS_PATTERN.finditer(content):
            block_content = fc_match.group(1)
            raw_xml = fc_match.group(0)

            for invoke_match in self.ASCII_INVOKE_PATTERN.finditer(block_content):
                tool_name = invoke_match.group(1)
                params_content = invoke_match.group(2)

                arguments = self._parse_parameters(params_content)

                tool_calls.append(ParsedToolCall(
                    name=tool_name,
                    arguments=arguments,
                    raw_xml=raw_xml
                ))

        # Try standalone invokes if no function_calls blocks found
        if not tool_calls:
            for invoke_match in self.ASCII_INVOKE_PATTERN.finditer(content):
                tool_name = invoke_match.group(1)
                params_content = invoke_match.group(2)
                raw_xml = invoke_match.group(0)

                arguments = self._parse_parameters(params_content)

                tool_calls.append(ParsedToolCall(
                    name=tool_name,
                    arguments=arguments,
                    raw_xml=raw_xml
                ))

        return tool_calls

    def _parse_lenient(self, content: str) -> List[ParsedToolCall]:
        """Parse malformed DSML content using heuristics.

        This handles cases where DSML tags are broken across lines or
        missing proper XML structure.

        Args:
            content: Potentially malformed DSML content

        Returns:
            List of parsed tool calls
        """
        tool_calls: List[ParsedToolCall] = []

        # Look for tool name patterns
        # Pattern: |DSML|invoke name="tool_name"> or similar
        tool_name_pattern = re.compile(
            r'[\|｜]DSML[\|｜]invoke[^"\']*name\s*=\s*["\']?([^"\'>\s\n]+)["\']?',
            re.IGNORECASE
        )

        tool_names = tool_name_pattern.findall(content)

        if not tool_names:
            # Try finding just tool name after invoke
            simple_pattern = re.compile(
                r'[\|｜]DSML[\|｜]invoke\s+([a-z_][a-z0-9_]*)',
                re.IGNORECASE
            )
            tool_names = simple_pattern.findall(content)

        # Look for parameter patterns
        # Pattern: |DSML|parameter name="param_name">value
        # or: |DSML|parameter name="param_name" string="false">value
        param_pattern = re.compile(
            r'[\|｜]DSML[\|｜]parameter\s+name\s*=\s*["\']?([^"\'>\s]+)["\']?'
            r'(?:\s+[a-z]+\s*=\s*["\']?[^"\'>\s]*["\']?)*'
            r'\s*>?\s*([^<\n｜|]+)',
            re.IGNORECASE
        )

        params = {}
        for match in param_pattern.finditer(content):
            param_name = match.group(1)
            param_value = match.group(2).strip()
            if param_value:
                params[param_name] = self._parse_value(param_value)

        # If we found both tool name and params, create a tool call
        if tool_names and params:
            for tool_name in tool_names:
                tool_calls.append(ParsedToolCall(
                    name=tool_name,
                    arguments=params.copy(),
                    raw_xml=content[:500]  # Store sample for debugging
                ))
        elif tool_names:
            # Found tool name but no params - still create call with empty args
            for tool_name in tool_names:
                tool_calls.append(ParsedToolCall(
                    name=tool_name,
                    arguments={},
                    raw_xml=content[:500]
                ))

        return tool_calls

    def _parse_parameters(self, params_content: str) -> Dict[str, Any]:
        """Extract parameters from DSML content.

        Args:
            params_content: Content containing parameter elements

        Returns:
            Dictionary of parameter names to values
        """
        arguments: Dict[str, Any] = {}

        for match in self.ASCII_PARAMETER_PATTERN.finditer(params_content):
            param_name = match.group(1)
            param_value = match.group(2).strip()

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
        value = value.strip()

        # Handle empty values
        if not value:
            return ""

        # Handle booleans
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Handle integers
        if value.lstrip('-').isdigit():
            return int(value)

        # Handle floats
        try:
            if '.' in value:
                return float(value)
        except ValueError:
            pass

        # Try JSON parsing for complex types
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Return as string
        return value
