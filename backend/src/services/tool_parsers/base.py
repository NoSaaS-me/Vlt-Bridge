"""Base classes for tool call parsing strategy pattern.

This module defines the abstract interface and data structures for parsing
tool calls from different LLM model formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import json


@dataclass
class ParsedToolCall:
    """Intermediate representation of a parsed tool call.

    This dataclass provides a clean abstraction over the various XML formats
    that different models use, and provides conversion to the standard OpenAI
    function calling format.

    Attributes:
        name: The name of the tool/function to call
        arguments: Dictionary of parameter names to values
        raw_xml: Original XML string for debugging purposes
    """
    name: str
    arguments: Dict[str, Any]
    raw_xml: str

    def to_openai_format(self, call_id: str) -> Dict[str, Any]:
        """Convert to OpenAI function calling format.

        Args:
            call_id: Unique identifier for this tool call

        Returns:
            Dictionary matching OpenAI's tool_calls structure with:
            - id: unique call identifier
            - type: "function"
            - function: object with name and JSON-encoded arguments
        """
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments)
            }
        }


class ToolCallParser(ABC):
    """Abstract base class for tool call parsers.

    Each concrete parser implements parsing for a specific LLM model's
    tool calling format. Parsers are organized in a chain of responsibility
    pattern, where each parser checks if it can handle the content before
    attempting to parse.
    """

    @abstractmethod
    def can_parse(self, content: str) -> bool:
        """Check if this parser can handle the given content.

        This method should perform a lightweight check (e.g., regex match)
        to determine if the content matches this parser's expected format.

        Args:
            content: The text content to check

        Returns:
            True if this parser can handle the content, False otherwise
        """
        pass

    @abstractmethod
    def parse(self, content: str) -> Tuple[List[ParsedToolCall], str]:
        """Extract tool calls from content and return cleaned content.

        This method performs the actual parsing, extracting all tool calls
        and removing the XML markup from the content.

        Args:
            content: The text content containing tool calls

        Returns:
            Tuple of (parsed_calls, cleaned_content) where:
            - parsed_calls: List of ParsedToolCall objects
            - cleaned_content: Original content with XML removed and whitespace normalized
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Parser name for logging and debugging.

        Returns:
            Human-readable name identifying this parser
        """
        pass
