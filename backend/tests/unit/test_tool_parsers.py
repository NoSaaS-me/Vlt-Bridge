"""Unit tests for tool call parsers.

Tests each parser independently and the chain behavior.
"""

import pytest

from src.services.tool_parsers import ToolCallParserChain
from src.services.tool_parsers.anthropic import AnthropicXMLParser
from src.services.tool_parsers.deepseek import DeepSeekXMLParser
from src.services.tool_parsers.generic import GenericXMLParser
from src.services.tool_parsers.standard import StandardXMLParser


class TestStandardXMLParser:
    """Test the standard <function_calls> parser."""

    def test_can_parse_standard_format(self):
        """Should detect standard function_calls blocks."""
        parser = StandardXMLParser()
        content = """
        <function_calls>
        <invoke name="search_code">
        <parameter name="query">authentication</parameter>
        </invoke>
        </function_calls>
        """
        assert parser.can_parse(content) is True

    def test_cannot_parse_without_function_calls(self):
        """Should not detect content without function_calls."""
        parser = StandardXMLParser()
        content = "Just some regular text"
        assert parser.can_parse(content) is False

    def test_parse_single_tool_call(self):
        """Should parse a single tool call correctly."""
        parser = StandardXMLParser()
        content = """
        <function_calls>
        <invoke name="search_code">
        <parameter name="query">authentication</parameter>
        <parameter name="limit">5</parameter>
        </invoke>
        </function_calls>
        """

        calls, cleaned = parser.parse(content)

        assert len(calls) == 1
        assert calls[0].name == "search_code"
        assert calls[0].arguments == {"query": "authentication", "limit": 5}
        assert cleaned.strip() == ""

    def test_parse_multiple_tool_calls(self):
        """Should parse multiple tool calls in one block."""
        parser = StandardXMLParser()
        content = """
        <function_calls>
        <invoke name="search_code">
        <parameter name="query">auth</parameter>
        </invoke>
        <invoke name="read_file">
        <parameter name="path">src/auth.py</parameter>
        </invoke>
        </function_calls>
        """

        calls, cleaned = parser.parse(content)

        assert len(calls) == 2
        assert calls[0].name == "search_code"
        assert calls[1].name == "read_file"

    def test_parse_boolean_parameters(self):
        """Should correctly parse boolean values."""
        parser = StandardXMLParser()
        content = """
        <function_calls>
        <invoke name="search_code">
        <parameter name="case_sensitive">true</parameter>
        <parameter name="regex">false</parameter>
        </invoke>
        </function_calls>
        """

        calls, _ = parser.parse(content)

        assert calls[0].arguments["case_sensitive"] is True
        assert calls[0].arguments["regex"] is False

    def test_parse_json_parameters(self):
        """Should parse JSON objects in parameters."""
        parser = StandardXMLParser()
        content = """
        <function_calls>
        <invoke name="search_code">
        <parameter name="filters">{"type": "function", "name": "auth"}</parameter>
        </invoke>
        </function_calls>
        """

        calls, _ = parser.parse(content)

        assert calls[0].arguments["filters"] == {"type": "function", "name": "auth"}


class TestAnthropicXMLParser:
    """Test the Anthropic standalone invoke parser."""

    def test_can_parse_invoke(self):
        """Should detect invoke elements."""
        parser = AnthropicXMLParser()
        content = """
        <invoke name="search_code">
        <parameter name="query">test</parameter>
        </invoke>
        """
        assert parser.can_parse(content) is True

    def test_parse_standalone_invoke(self):
        """Should parse invoke without function_calls wrapper."""
        parser = AnthropicXMLParser()
        content = """
        Here's what I'll do:
        <invoke name="search_code">
        <parameter name="query">authentication</parameter>
        </invoke>
        """

        calls, cleaned = parser.parse(content)

        assert len(calls) == 1
        assert calls[0].name == "search_code"
        assert "Here's what I'll do:" in cleaned
        assert "<invoke" not in cleaned


class TestDeepSeekXMLParser:
    """Test the DeepSeek ｜DSML｜ parser."""

    def test_can_parse_deepseek_markers(self):
        """Should detect DeepSeek's special markers."""
        parser = DeepSeekXMLParser()
        content = """
        <｜DSML｜function_calls>
        <｜DSML｜invoke name="search_code">
        <｜DSML｜parameter name="query">test</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜function_calls>
        """
        assert parser.can_parse(content) is True

    def test_cannot_parse_without_markers(self):
        """Should not detect content without ｜DSML｜ markers."""
        parser = DeepSeekXMLParser()
        content = "<function_calls><invoke name=\"test\"></invoke></function_calls>"
        assert parser.can_parse(content) is False

    def test_parse_deepseek_format(self):
        """Should parse DeepSeek's full format."""
        parser = DeepSeekXMLParser()
        content = """
        <｜DSML｜function_calls>
        <｜DSML｜invoke name="search_code">
        <｜DSML｜parameter name="query">authentication</｜DSML｜parameter>
        <｜DSML｜parameter name="limit">10</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜function_calls>
        """

        calls, cleaned = parser.parse(content)

        assert len(calls) == 1
        assert calls[0].name == "search_code"
        assert calls[0].arguments["query"] == "authentication"
        assert calls[0].arguments["limit"] == 10


class TestGenericXMLParser:
    """Test the generic fallback parser."""

    def test_can_parse_prefixed_tags(self):
        """Should detect tags with arbitrary prefixes."""
        parser = GenericXMLParser()
        content = """
        <custom:function_calls>
        <custom:invoke name="test">
        </custom:invoke>
        </custom:function_calls>
        """
        assert parser.can_parse(content) is True

    def test_parse_with_prefixes(self):
        """Should parse tags with prefixes."""
        parser = GenericXMLParser()
        content = """
        <x:function_calls>
        <x:invoke name="search_code">
        <x:parameter name="query">test</x:parameter>
        </x:invoke>
        </x:function_calls>
        """

        calls, _ = parser.parse(content)

        assert len(calls) == 1
        assert calls[0].name == "search_code"

    def test_parse_malformed_spacing(self):
        """Should handle irregular spacing."""
        parser = GenericXMLParser()
        content = """
        <function_calls  >
        <invoke   name="search_code"  >
        <parameter  name="query" >test</parameter>
        </invoke>
        </function_calls>
        """

        calls, _ = parser.parse(content)

        assert len(calls) == 1
        assert calls[0].name == "search_code"


class TestToolCallParserChain:
    """Test the parser chain orchestration."""

    def test_chain_tries_parsers_in_order(self):
        """Should try parsers from most specific to generic."""
        chain = ToolCallParserChain()

        # Should use DeepSeek parser first (most specific)
        deepseek_content = """
        <｜DSML｜function_calls>
        <｜DSML｜invoke name="search">
        <｜DSML｜parameter name="q">test</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜function_calls>
        """

        calls, _ = chain.parse(deepseek_content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "search"

    def test_chain_returns_openai_format(self):
        """Should convert parsed calls to OpenAI format."""
        chain = ToolCallParserChain()
        content = """
        <function_calls>
        <invoke name="search_code">
        <parameter name="query">auth</parameter>
        </invoke>
        </function_calls>
        """

        calls, _ = chain.parse(content)

        assert len(calls) == 1
        call = calls[0]
        assert "id" in call
        assert call["type"] == "function"
        assert "function" in call
        assert call["function"]["name"] == "search_code"
        assert '"query": "auth"' in call["function"]["arguments"]

    def test_chain_handles_no_tool_calls(self):
        """Should return empty list when no tool calls found."""
        chain = ToolCallParserChain()
        content = "Just some regular text without any tool calls"

        calls, cleaned = chain.parse(content)

        assert calls == []
        assert cleaned == content

    def test_chain_cleans_content(self):
        """Should remove XML from content."""
        chain = ToolCallParserChain()
        content = """
        I'll search for that.
        <function_calls>
        <invoke name="search">
        <parameter name="query">test</parameter>
        </invoke>
        </function_calls>
        Done!
        """

        calls, cleaned = chain.parse(content)

        assert len(calls) == 1
        assert "<function_calls" not in cleaned
        assert "<invoke" not in cleaned
        assert "I'll search for that." in cleaned
        assert "Done!" in cleaned

    def test_chain_handles_multiple_formats(self):
        """Should handle different formats in same content."""
        chain = ToolCallParserChain()

        # Standard format
        standard = "<function_calls><invoke name=\"test\"><parameter name=\"q\">x</parameter></invoke></function_calls>"
        calls, _ = chain.parse(standard)
        assert len(calls) == 1

        # Anthropic standalone
        anthropic = "<invoke name=\"test\"><parameter name=\"q\">x</parameter></invoke>"
        calls, _ = chain.parse(anthropic)
        assert len(calls) == 1

        # DeepSeek
        deepseek = "<｜DSML｜invoke name=\"test\"><｜DSML｜parameter name=\"q\">x</｜DSML｜parameter></｜DSML｜invoke>"
        calls, _ = chain.parse(deepseek)
        assert len(calls) == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_parameters(self):
        """Should handle tool calls with no parameters."""
        parser = StandardXMLParser()
        content = """
        <function_calls>
        <invoke name="get_current_time">
        </invoke>
        </function_calls>
        """

        calls, _ = parser.parse(content)

        assert len(calls) == 1
        assert calls[0].arguments == {}

    def test_special_characters_in_values(self):
        """Should handle special characters in parameter values."""
        parser = StandardXMLParser()
        content = """
        <function_calls>
        <invoke name="search">
        <parameter name="query">function with "quotes" and 'apostrophes'</parameter>
        </invoke>
        </function_calls>
        """

        calls, _ = parser.parse(content)

        assert len(calls) == 1
        assert "quotes" in calls[0].arguments["query"]
        assert "apostrophes" in calls[0].arguments["query"]

    def test_numeric_string_values(self):
        """Should parse numeric strings correctly."""
        parser = StandardXMLParser()
        content = """
        <function_calls>
        <invoke name="test">
        <parameter name="port">8080</parameter>
        <parameter name="version">1.2.3</parameter>
        </invoke>
        </function_calls>
        """

        calls, _ = parser.parse(content)

        assert calls[0].arguments["port"] == 8080  # Should be int
        assert calls[0].arguments["version"] == "1.2.3"  # Should be string

    def test_whitespace_handling(self):
        """Should normalize whitespace in cleaned content."""
        chain = ToolCallParserChain()
        content = """
        Text before.


        <function_calls>
        <invoke name="test">
        </invoke>
        </function_calls>


        Text after.
        """

        _, cleaned = chain.parse(content)

        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in cleaned
