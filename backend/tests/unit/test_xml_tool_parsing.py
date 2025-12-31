"""Unit tests for XML-style tool call parsing (oracle_agent._parse_xml_tool_calls).

Some models (like DeepSeek) don't properly support OpenAI function calling and
instead output XML-style tool invocations in their text response. This module
tests the parsing logic that extracts those pseudo-tool-calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from backend.src.services.oracle_agent import _parse_xml_tool_calls


class TestParseXmlToolCalls:
    """Tests for _parse_xml_tool_calls function."""

    def test_parse_basic_function_call(self):
        """Test parsing a basic XML function call."""
        content = """I'll search for that information.
<function_calls>
<invoke name="vault_search">
<parameter name="query">oracle agent model</parameter>
</invoke>
</function_calls>
"""
        tool_calls, cleaned = _parse_xml_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "vault_search"

        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["query"] == "oracle agent model"

        assert "function_calls" not in cleaned
        assert "invoke" not in cleaned
        assert "I'll search" in cleaned

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls in one block."""
        content = """<function_calls>
<invoke name="vault_search">
<parameter name="query">authentication</parameter>
<parameter name="limit">5</parameter>
</invoke>
<invoke name="search_code">
<parameter name="query">jwt validation</parameter>
</invoke>
</function_calls>"""

        tool_calls, cleaned = _parse_xml_tool_calls(content)

        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "vault_search"
        assert tool_calls[1]["function"]["name"] == "search_code"

        args0 = json.loads(tool_calls[0]["function"]["arguments"])
        assert args0["query"] == "authentication"
        assert args0["limit"] == 5  # Should be parsed as int

    def test_parse_boolean_values(self):
        """Test parsing boolean parameter values."""
        content = """<function_calls>
<invoke name="vault_create_index">
<parameter name="folder">docs</parameter>
<parameter name="include_summaries">true</parameter>
</invoke>
</function_calls>"""

        tool_calls, cleaned = _parse_xml_tool_calls(content)

        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["include_summaries"] is True

    def test_no_xml_returns_empty(self):
        """Test that normal content without XML returns empty list."""
        content = "This is just regular text without any XML tool calls."
        
        tool_calls, cleaned = _parse_xml_tool_calls(content)
        
        assert len(tool_calls) == 0
        assert cleaned == content

    def test_parse_deepseek_style(self):
        """Test parsing DeepSeek-style XML output (the actual bug case)."""
        content = """<function_calls>
<invoke name="vault_search">
<parameter name="query" string="true">oracle agent model system information</parameter>
</invoke>
</function_calls>"""

        tool_calls, cleaned = _parse_xml_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "vault_search"
        
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert "query" in args

    def test_tool_call_has_id(self):
        """Test that parsed tool calls get unique IDs."""
        content = """<function_calls>
<invoke name="search_code">
<parameter name="query">test</parameter>
</invoke>
</function_calls>"""

        tool_calls, _ = _parse_xml_tool_calls(content)

        assert len(tool_calls) == 1
        assert "id" in tool_calls[0]
        assert tool_calls[0]["id"].startswith("xml_call_")
        assert tool_calls[0]["type"] == "function"

    def test_multiple_blocks_parsed(self):
        """Test that multiple function_calls blocks are all parsed."""
        content = """First thought...
<function_calls>
<invoke name="vault_search">
<parameter name="query">first search</parameter>
</invoke>
</function_calls>

Second thought...
<function_calls>
<invoke name="search_code">
<parameter name="query">second search</parameter>
</invoke>
</function_calls>
"""

        tool_calls, cleaned = _parse_xml_tool_calls(content)

        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "vault_search"
        assert tool_calls[1]["function"]["name"] == "search_code"
        
        # Both blocks should be removed
        assert "function_calls" not in cleaned
        assert "First thought" in cleaned
        assert "Second thought" in cleaned
