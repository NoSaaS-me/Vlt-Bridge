"""Tool call parsers for different LLM model formats.

This package provides a Strategy Pattern implementation for parsing tool calls
from various LLM models that use XML-style function calling instead of proper
OpenAI-compatible function calling.

Main Components:
- ParsedToolCall: Intermediate representation of a tool call
- ToolCallParser: Abstract base class for parsers
- ToolCallParserChain: Chain of Responsibility for trying parsers

Supported Formats:
- Standard XML: <function_calls><invoke>...</invoke></function_calls>
- Anthropic: Standalone <invoke> elements
- DeepSeek: <｜DSML｜function_calls> with special markers
- Generic: Fallback for any XML-like format

Usage:
    from .tool_parsers import ToolCallParserChain

    chain = ToolCallParserChain()
    tool_calls, cleaned_content = chain.parse(model_output)
"""

from .base import ParsedToolCall, ToolCallParser
from .chain import ToolCallParserChain

__all__ = [
    "ParsedToolCall",
    "ToolCallParser",
    "ToolCallParserChain",
]
