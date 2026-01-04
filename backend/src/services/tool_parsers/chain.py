"""Parser chain for trying multiple tool call parsers in sequence.

This module implements the Chain of Responsibility pattern for tool call parsing.
Each parser is tried in priority order until one successfully parses the content.
"""

import logging
import uuid
from typing import Any, Dict, List, Tuple

from .anthropic import AnthropicXMLParser
from .deepseek import DeepSeekXMLParser
from .generic import GenericXMLParser
from .inline import InlineToolParser
from .standard import StandardXMLParser

logger = logging.getLogger(__name__)


class ToolCallParserChain:
    """Chains multiple parsers together, trying each in priority order.

    The chain tries parsers from most specific to most generic:
    1. InlineToolParser (tool_name{json} format - check first as it's common with DeepSeek)
    2. DeepSeek (most distinctive XML markers)
    3. Standard (common XML format)
    4. Anthropic (may overlap with standard)
    5. Generic (fallback, most permissive)

    When a parser successfully identifies content it can handle (via can_parse),
    it is used to extract tool calls. The chain stops at the first successful parse.
    """

    def __init__(self):
        """Initialize the parser chain with all available parsers."""
        self.parsers = [
            InlineToolParser(),        # tool_name{json} format - common with DeepSeek V3
            DeepSeekXMLParser(),       # DeepSeek XML format
            StandardXMLParser(),       # Common XML format
            AnthropicXMLParser(),      # Standalone invokes
            GenericXMLParser(),        # Fallback - most permissive
        ]

    def parse(self, content: str) -> Tuple[List[Dict[str, Any]], str]:
        """Try each parser until one succeeds.

        The method iterates through parsers in priority order. For each parser,
        it first checks if the parser can handle the content. If so, it parses
        the content and converts the results to OpenAI format.

        Args:
            content: Text that may contain XML-style tool calls

        Returns:
            Tuple of (tool_calls, cleaned_content) where:
            - tool_calls: List of dicts in OpenAI function calling format
            - cleaned_content: Original content with XML removed

        Example:
            >>> chain = ToolCallParserChain()
            >>> calls, cleaned = chain.parse(content_with_xml)
            >>> calls[0]
            {
                'id': 'xml_call_abc123',
                'type': 'function',
                'function': {
                    'name': 'search_code',
                    'arguments': '{"query": "auth"}'
                }
            }
        """
        for parser in self.parsers:
            if parser.can_parse(content):
                logger.debug(f"Using {parser.name} to parse tool calls")

                # Parse using this parser
                parsed_calls, cleaned = parser.parse(content)

                if not parsed_calls:
                    # Parser matched but found no calls, try next parser
                    continue

                # Convert to OpenAI format
                tool_calls = []
                for call in parsed_calls:
                    call_id = f"xml_call_{uuid.uuid4().hex[:8]}"
                    tool_calls.append(call.to_openai_format(call_id))

                logger.info(
                    f"[{parser.name}] Successfully parsed {len(tool_calls)} tool call(s)",
                    extra={"tool_names": [tc["function"]["name"] for tc in tool_calls]},
                )

                return tool_calls, cleaned

        # No parser found tool calls
        return [], content
