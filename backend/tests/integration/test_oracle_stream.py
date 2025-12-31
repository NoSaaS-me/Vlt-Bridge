"""Integration tests for Oracle streaming API (009-oracle-agent T016).

These tests verify the Oracle Agent streaming functionality through the
FastAPI endpoints. They require:
- Backend server running (or use pytest-httpx for mocking)
- OpenRouter API key configured (or mocked)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from backend.src.models.oracle import OracleRequest, OracleStreamChunk
from backend.src.services.oracle_agent import OracleAgent, OracleAgentError


@pytest.fixture
def mock_openrouter_response() -> Dict[str, Any]:
    """Mock successful OpenRouter API response."""
    return {
        "id": "gen-test-123",
        "model": "anthropic/claude-sonnet-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Based on my analysis, the answer is 42.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }


@pytest.fixture
def mock_tool_call_response() -> Dict[str, Any]:
    """Mock OpenRouter response with tool calls."""
    return {
        "id": "gen-test-456",
        "model": "anthropic/claude-sonnet-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "vault_search",
                                "arguments": '{"query": "authentication", "limit": 5}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


@pytest.fixture
def mock_final_response() -> Dict[str, Any]:
    """Mock final response after tool execution."""
    return {
        "id": "gen-test-789",
        "model": "anthropic/claude-sonnet-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Based on the vault search results, here is the authentication documentation.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        },
    }


class TestOracleAgentNonStreaming:
    """Tests for non-streaming Oracle Agent queries."""

    @pytest.mark.asyncio
    async def test_simple_query_returns_content_chunks(
        self, mock_openrouter_response: Dict[str, Any]
    ) -> None:
        """Agent returns content and done chunks for simple query."""
        with patch("httpx.AsyncClient") as mock_client_class:
            # Setup mock - json() returns a dict, not a coroutine
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openrouter_response
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Create agent and query
            agent = OracleAgent(api_key="test-key", model="test-model")
            chunks: List[OracleStreamChunk] = []

            async for chunk in agent.query(
                question="What is the answer?",
                user_id="test-user",
                stream=False,
            ):
                chunks.append(chunk)

            # Verify chunks
            chunk_types = [c.type for c in chunks]
            assert "thinking" in chunk_types  # Initial thinking
            assert "content" in chunk_types  # Answer content
            assert "done" in chunk_types  # Completion

            # Verify content
            content_chunk = next(c for c in chunks if c.type == "content")
            assert "42" in content_chunk.content

            # Verify done chunk has metadata
            done_chunk = next(c for c in chunks if c.type == "done")
            assert done_chunk.model_used == "test-model"

    @pytest.mark.asyncio
    async def test_query_with_tool_calls(
        self,
        mock_tool_call_response: Dict[str, Any],
        mock_final_response: Dict[str, Any],
    ) -> None:
        """Agent handles tool calls and returns tool_call/tool_result chunks."""
        call_count = 0

        def get_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Use MagicMock for json() to return dict, not coroutine
            mock_response = MagicMock()
            if call_count == 1:
                mock_response.json.return_value = mock_tool_call_response
            else:
                mock_response.json.return_value = mock_final_response
            mock_response.raise_for_status = MagicMock()
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = get_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Mock tool executor
            with patch(
                "backend.src.services.oracle_agent._get_tool_executor"
            ) as mock_te:
                mock_executor = MagicMock()
                mock_executor.get_tool_schemas.return_value = []
                mock_executor.execute = AsyncMock(
                    return_value='{"query": "authentication", "results": [], "count": 0}'
                )
                mock_te.return_value = mock_executor

                agent = OracleAgent(api_key="test-key")
                chunks: List[OracleStreamChunk] = []

                async for chunk in agent.query(
                    question="Find auth docs",
                    user_id="test-user",
                    stream=False,
                ):
                    chunks.append(chunk)

                # Verify tool-related chunks
                chunk_types = [c.type for c in chunks]
                assert "tool_call" in chunk_types
                assert "tool_result" in chunk_types
                assert "done" in chunk_types

                # Verify tool executor was called
                mock_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_error_returns_error_chunk(self) -> None:
        """Agent returns error chunk on API failure."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_class:
            # Use MagicMock for the response (not AsyncMock)
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.text = "Rate limited"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Rate limited", request=MagicMock(), response=mock_response
            )

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            agent = OracleAgent(api_key="test-key")
            chunks: List[OracleStreamChunk] = []

            async for chunk in agent.query(
                question="Test query",
                user_id="test-user",
                stream=False,
            ):
                chunks.append(chunk)

            # Should have error chunk
            error_chunks = [c for c in chunks if c.type == "error"]
            assert len(error_chunks) >= 1
            assert "429" in error_chunks[-1].error

    @pytest.mark.asyncio
    async def test_timeout_returns_error_chunk(self) -> None:
        """Agent returns error chunk on timeout."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            agent = OracleAgent(api_key="test-key")
            chunks: List[OracleStreamChunk] = []

            async for chunk in agent.query(
                question="Test query",
                user_id="test-user",
                stream=False,
            ):
                chunks.append(chunk)

            # Should have error chunk
            error_chunks = [c for c in chunks if c.type == "error"]
            assert len(error_chunks) >= 1
            assert "timeout" in error_chunks[-1].error.lower()


class TestOracleAgentStreaming:
    """Tests for streaming Oracle Agent queries."""

    @pytest.mark.asyncio
    async def test_streaming_yields_content_incrementally(self) -> None:
        """Streaming mode yields content chunks as they arrive."""

        async def mock_aiter_lines():
            # Simulate SSE stream
            lines = [
                'data: {"id":"gen-1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
                'data: {"id":"gen-1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
                'data: {"id":"gen-1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
                'data: {"id":"gen-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
            for line in lines:
                yield line

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = AsyncMock()
            mock_response.aiter_lines = mock_aiter_lines
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            agent = OracleAgent(api_key="test-key")
            chunks: List[OracleStreamChunk] = []

            async for chunk in agent.query(
                question="Say hello",
                user_id="test-user",
                stream=True,
            ):
                chunks.append(chunk)

            # Verify streaming content
            content_chunks = [c for c in chunks if c.type == "content"]
            assert len(content_chunks) == 2  # "Hello" and " world"
            assert content_chunks[0].content == "Hello"
            assert content_chunks[1].content == " world"

    @pytest.mark.asyncio
    async def test_streaming_tool_calls_yield_visibility_chunks(self) -> None:
        """Streaming tool calls yield tool_call and tool_result chunks."""

        async def mock_aiter_lines_with_tools():
            lines = [
                'data: {"id":"gen-1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"vault_search","arguments":""}}]},"finish_reason":null}]}',
                'data: {"id":"gen-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"query\\": \\"test\\"}"}}]},"finish_reason":null}]}',
                'data: {"id":"gen-1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
            for line in lines:
                yield line

        async def mock_aiter_lines_final():
            lines = [
                'data: {"id":"gen-2","choices":[{"index":0,"delta":{"role":"assistant","content":"Found results."},"finish_reason":null}]}',
                'data: {"id":"gen-2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
            for line in lines:
                yield line

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            if call_count == 1:
                mock_response.aiter_lines = mock_aiter_lines_with_tools
            else:
                mock_response.aiter_lines = mock_aiter_lines_final
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Mock tool executor
            with patch(
                "backend.src.services.oracle_agent._get_tool_executor"
            ) as mock_te:
                mock_executor = MagicMock()
                mock_executor.get_tool_schemas.return_value = []
                mock_executor.execute = AsyncMock(
                    return_value='{"results": []}'
                )
                mock_te.return_value = mock_executor

                agent = OracleAgent(api_key="test-key")
                chunks: List[OracleStreamChunk] = []

                async for chunk in agent.query(
                    question="Search vault",
                    user_id="test-user",
                    stream=True,
                ):
                    chunks.append(chunk)

                # Verify tool visibility chunks
                chunk_types = [c.type for c in chunks]
                assert "tool_call" in chunk_types
                assert "tool_result" in chunk_types


class TestOracleAgentSourceExtraction:
    """Tests for source citation extraction."""

    @pytest.mark.asyncio
    async def test_code_search_results_become_sources(
        self, mock_final_response: Dict[str, Any]
    ) -> None:
        """search_code results are extracted as source citations."""
        # First call returns tool call, second returns final answer
        call_count = 0
        tool_call_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search_code",
                                    "arguments": '{"query": "auth"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        def get_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Use MagicMock for json() to return dict, not coroutine
            mock_response = MagicMock()
            if call_count == 1:
                mock_response.json.return_value = tool_call_response
            else:
                mock_response.json.return_value = mock_final_response
            mock_response.raise_for_status = MagicMock()
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = get_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with patch(
                "backend.src.services.oracle_agent._get_tool_executor"
            ) as mock_te:
                mock_executor = MagicMock()
                mock_executor.get_tool_schemas.return_value = []
                mock_executor.execute = AsyncMock(
                    return_value=json.dumps({
                        "results": [
                            {
                                "file_path": "src/auth.py",
                                "line_start": 10,
                                "content": "def authenticate():",
                                "score": 0.95,
                            }
                        ]
                    })
                )
                mock_te.return_value = mock_executor

                agent = OracleAgent(api_key="test-key")
                chunks: List[OracleStreamChunk] = []

                async for chunk in agent.query(
                    question="Find auth code",
                    user_id="test-user",
                    stream=False,
                ):
                    chunks.append(chunk)

                # Find source chunks
                source_chunks = [c for c in chunks if c.type == "source"]
                assert len(source_chunks) >= 1

                source = source_chunks[0].source
                assert source.path == "src/auth.py"
                assert source.source_type == "code"
                assert source.line == 10


class TestOracleAgentMaxTurns:
    """Tests for agent loop termination."""

    @pytest.mark.asyncio
    async def test_max_turns_limit_prevents_infinite_loop(self) -> None:
        """Agent stops after MAX_TURNS if no completion."""
        # Always return tool calls to simulate infinite loop
        tool_call_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "vault_list",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            # Use MagicMock for json() to return dict, not coroutine
            mock_response = MagicMock()
            mock_response.json.return_value = tool_call_response
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with patch(
                "backend.src.services.oracle_agent._get_tool_executor"
            ) as mock_te:
                mock_executor = MagicMock()
                mock_executor.get_tool_schemas.return_value = []
                mock_executor.execute = AsyncMock(return_value='{"notes": []}')
                mock_te.return_value = mock_executor

                # Reduce MAX_TURNS for faster test
                agent = OracleAgent(api_key="test-key")
                original_max = agent.MAX_TURNS
                agent.MAX_TURNS = 3

                try:
                    chunks: List[OracleStreamChunk] = []
                    async for chunk in agent.query(
                        question="Loop forever",
                        user_id="test-user",
                        stream=False,
                    ):
                        chunks.append(chunk)

                    # Should have error about max turns
                    error_chunks = [c for c in chunks if c.type == "error"]
                    assert len(error_chunks) >= 1
                    assert "Maximum" in error_chunks[-1].error

                finally:
                    agent.MAX_TURNS = original_max
