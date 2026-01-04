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
- DeepSeek Unicode: <｜DSML｜function_calls> with special markers
- DeepSeek ASCII: |DSML| format in reasoning field
- Generic: Fallback for any XML-like format

Usage:
    from .tool_parsers import ToolCallParserChain

    chain = ToolCallParserChain()
    tool_calls, cleaned_content = chain.parse(model_output)

For parsing reasoning field specifically:
    from .tool_parsers import DSMLReasoningParser

    parser = DSMLReasoningParser()
    if parser.can_parse(reasoning_content):
        calls, _ = parser.parse(reasoning_content)
"""

from .base import ParsedToolCall, ToolCallParser
from .chain import ToolCallParserChain
from .dsml_reasoning import DSMLReasoningParser

__all__ = [
    "ParsedToolCall",
    "ToolCallParser",
    "ToolCallParserChain",
    "DSMLReasoningParser",
]
