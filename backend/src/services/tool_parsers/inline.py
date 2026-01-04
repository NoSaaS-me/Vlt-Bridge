"""Parser for inline tool calls in the format: tool_name{json}.

Some models output tool calls directly in the content without XML wrapping,
using the format: tool_name{"arg1": "value1", "arg2": "value2"}

This parser detects and extracts these inline tool invocations.

Examples:
    vault_search{"query": "oracle agent", "limit": 10}
    get_repo_map{}
    search_code{"query": "authentication", "limit": 5}
"""

import json
import logging
import re
from typing import List, Tuple

from .base import ParsedToolCall, ToolCallParser

logger = logging.getLogger(__name__)

# Pattern for inline tool calls: tool_name{json_object}
# Matches: tool_name{ ... } where tool_name is alphanumeric with underscores
# and the braces contain valid JSON
INLINE_TOOL_PATTERN = re.compile(
    r'([a-z_][a-z0-9_]*)\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
    re.IGNORECASE | re.DOTALL
)

# Known tool names to help distinguish tool calls from regular JSON
KNOWN_TOOLS = {
    'vault_search', 'vault_read', 'vault_write', 'vault_list_files',
    'search_code', 'get_repo_map', 'thread_list', 'thread_read', 'thread_create',
    'thread_push', 'thread_seek', 'web_search', 'fetch_url',
    'read_file', 'write_file', 'list_files', 'execute_command',
}


class InlineToolParser(ToolCallParser):
    """Parser for inline tool call format: tool_name{json}.

    This parser handles cases where models output tool calls as plain text
    in the format tool_name{"arg": "value"} instead of using XML or proper
    function calling.
    """

    @property
    def name(self) -> str:
        return "InlineToolParser"

    def can_parse(self, content: str) -> bool:
        """Check if content contains inline tool call patterns.

        Looks for patterns like tool_name{json} and verifies the tool name
        is a known tool to avoid false positives.

        Args:
            content: The text content to check

        Returns:
            True if inline tool calls are detected
        """
        # Quick check for common patterns
        stripped = content.strip()

        # Check if content starts with a known tool name followed by {
        for tool_name in KNOWN_TOOLS:
            if stripped.startswith(f"{tool_name}{{") or stripped.startswith(f"{tool_name} {{"):
                return True
            # Also check for tool name on its own line
            if f"\n{tool_name}{{" in content or f"\n{tool_name} {{" in content:
                return True

        # Check for generic pattern but verify it looks like a tool call
        match = INLINE_TOOL_PATTERN.search(content)
        if match:
            potential_tool = match.group(1).lower()
            # Check if it's a known tool or follows tool naming conventions
            if potential_tool in KNOWN_TOOLS:
                return True
            # Additional heuristic: underscore-separated names are likely tools
            if '_' in potential_tool and len(potential_tool) > 3:
                return True

        return False

    def parse(self, content: str) -> Tuple[List[ParsedToolCall], str]:
        """Extract inline tool calls from content.

        Finds all instances of tool_name{json} and converts them to
        ParsedToolCall objects.

        Args:
            content: The text content containing inline tool calls

        Returns:
            Tuple of (parsed_calls, cleaned_content)
        """
        parsed_calls: List[ParsedToolCall] = []
        cleaned_content = content

        # Find all inline tool calls
        for match in INLINE_TOOL_PATTERN.finditer(content):
            tool_name = match.group(1)
            json_str = match.group(2)
            full_match = match.group(0)

            # Skip if not a known tool (avoid false positives)
            if tool_name.lower() not in KNOWN_TOOLS:
                # Still include if it follows tool naming pattern
                if '_' not in tool_name or len(tool_name) <= 3:
                    continue

            try:
                # Parse the JSON arguments
                arguments = json.loads(json_str)

                if isinstance(arguments, dict):
                    parsed_calls.append(ParsedToolCall(
                        name=tool_name,
                        arguments=arguments,
                        raw_xml=full_match  # Store original for debugging
                    ))

                    # Remove this tool call from content
                    cleaned_content = cleaned_content.replace(full_match, '', 1)

                    logger.debug(f"Parsed inline tool call: {tool_name}({json.dumps(arguments)[:100]})")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON in inline tool call '{tool_name}': {e}")
                continue

        # Clean up whitespace in cleaned content
        cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
        cleaned_content = cleaned_content.strip()

        if parsed_calls:
            logger.info(f"InlineToolParser found {len(parsed_calls)} tool call(s)")

        return parsed_calls, cleaned_content
