"""Unit tests for LibrarianAgent (009-oracle-agent T047).

Tests verify:
- Summarization with source citation
- Summary caching in proper folder structure
- Index creation with wikilinks
- Streaming block responses
- Cache hit/miss behavior
- Model selection from settings (not hardcoded)
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.src.services.librarian_agent import (
    LibrarianAgent,
    LibrarianAgentError,
    get_librarian_agent,
)
from backend.src.models.librarian import LibrarianStreamChunk


# --- Test Fixtures ---


@pytest.fixture
def mock_openrouter_response():
    """Mock LLM response with summary."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "## Summary\n\n- Point 1 from [[doc1]]\n- Point 2 from [[doc2]]",
                }
            }
        ],
        "usage": {"total_tokens": 150},
    }


@pytest.fixture
def mock_tool_executor():
    """Mock ToolExecutor for vault operations."""
    mock = MagicMock()
    # Default: vault_read returns not found (cache miss)
    mock.execute = AsyncMock(
        return_value=json.dumps({"error": "File not found"})
    )
    return mock


@pytest.fixture
def content_items():
    """Sample content items for summarization."""
    return [
        {
            "source_type": "vault",
            "path": "notes/doc1.md",
            "content": "First document content about authentication.",
        },
        {
            "source_type": "code",
            "path": "src/auth.py",
            "content": "class AuthService:\n    def authenticate(self): pass",
        },
        {
            "source_type": "thread",
            "path": "thread-123",
            "content": "Decision: Use JWT for session management.",
        },
    ]


@pytest.fixture
def librarian_agent(mock_tool_executor):
    """Create a LibrarianAgent with mocked dependencies."""
    agent = LibrarianAgent(
        api_key="test-api-key",
        model="anthropic/claude-sonnet-4",
        user_id="test-user",
        project_id="test-project",
        tool_executor=mock_tool_executor,
    )
    return agent


# --- Test Classes ---


class TestLibrarianInit:
    """Tests for LibrarianAgent initialization."""

    def test_init_with_required_params(self):
        """Agent initializes with required parameters."""
        agent = LibrarianAgent(
            api_key="test-key",
        )
        assert agent.api_key == "test-key"
        assert agent.model == LibrarianAgent.DEFAULT_MODEL
        assert agent.user_id is None

    def test_init_with_all_params(self):
        """Agent initializes with all parameters."""
        agent = LibrarianAgent(
            api_key="test-key",
            model="custom/model",
            user_id="user-123",
            project_id="project-456",
        )
        assert agent.api_key == "test-key"
        assert agent.model == "custom/model"
        assert agent.user_id == "user-123"
        assert agent.project_id == "project-456"

    def test_default_model_is_not_hardcoded_in_init(self):
        """Model can be set from settings, not hardcoded."""
        # Agent should accept any model string
        custom_model = "deepseek/deepseek-chat"
        agent = LibrarianAgent(
            api_key="test-key",
            model=custom_model,
        )
        assert agent.model == custom_model


class TestLibrarianCachePath:
    """Tests for cache path generation."""

    def test_cache_path_structure(self, librarian_agent):
        """Cache path follows oracle-cache/summaries/{type}/{date}/{id}.md pattern."""
        cache_path = librarian_agent._get_cache_path(
            source_type="vault",
            source_id="test-summary",
            date_str="2025-01-15",
        )

        assert cache_path.startswith("oracle-cache/summaries/")
        assert "/vault/" in cache_path
        assert "/2025-01-15/" in cache_path
        assert cache_path.endswith(".md")

    def test_cache_path_sanitizes_source_id(self, librarian_agent):
        """Source ID is sanitized for filesystem safety."""
        cache_path = librarian_agent._get_cache_path(
            source_type="vault",
            source_id="unsafe/path\\with:special<chars>",
        )

        # Should not contain unsafe characters
        assert "/" not in cache_path.split("/")[-1].replace(".md", "")
        assert "\\" not in cache_path
        assert ":" not in cache_path.split("/")[-1]
        assert "<" not in cache_path
        assert ">" not in cache_path

    def test_cache_path_uses_today_by_default(self, librarian_agent):
        """Cache path uses today's date when not specified."""
        cache_path = librarian_agent._get_cache_path(
            source_type="code",
            source_id="test",
        )

        today = datetime.now().strftime("%Y-%m-%d")
        assert today in cache_path

    def test_generate_cache_key_is_deterministic(self, librarian_agent):
        """Same inputs produce same cache key."""
        content = [
            {"path": "a.md", "content": "Content A"},
            {"path": "b.md", "content": "Content B"},
        ]

        key1 = librarian_agent._generate_cache_key("Task", content)
        key2 = librarian_agent._generate_cache_key("Task", content)

        assert key1 == key2
        assert len(key1) == 16  # SHA256 truncated to 16 chars

    def test_generate_cache_key_differs_for_different_inputs(self, librarian_agent):
        """Different inputs produce different cache keys."""
        content1 = [{"path": "a.md", "content": "Content A"}]
        content2 = [{"path": "b.md", "content": "Content B"}]

        key1 = librarian_agent._generate_cache_key("Task", content1)
        key2 = librarian_agent._generate_cache_key("Task", content2)

        assert key1 != key2


class TestLibrarianSummarization:
    """Tests for content summarization."""

    @pytest.mark.asyncio
    async def test_summarize_returns_streaming_chunks(
        self, librarian_agent, content_items, mock_openrouter_response
    ):
        """Summarization yields LibrarianStreamChunk objects."""
        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            chunks = []
            async for chunk in librarian_agent.summarize(
                task="Summarize the content",
                content=content_items,
            ):
                chunks.append(chunk)

            # Should have at least one chunk
            assert len(chunks) > 0

            # All chunks should be LibrarianStreamChunk
            for chunk in chunks:
                assert isinstance(chunk, LibrarianStreamChunk)
                assert chunk.type in [
                    "thinking",
                    "summary",
                    "cache_hit",
                    "index",
                    "error",
                    "done",
                ]

    @pytest.mark.asyncio
    async def test_summarize_yields_thinking_chunks(
        self, librarian_agent, content_items, mock_openrouter_response
    ):
        """Summarization emits thinking chunks during processing."""
        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            chunks = []
            async for chunk in librarian_agent.summarize(
                task="Summarize",
                content=content_items,
            ):
                chunks.append(chunk)

            # Should have thinking chunks
            thinking_chunks = [c for c in chunks if c.type == "thinking"]
            assert len(thinking_chunks) >= 1
            # Thinking chunks should have content
            for tc in thinking_chunks:
                assert tc.content is not None
                assert len(tc.content) > 0

    @pytest.mark.asyncio
    async def test_summarize_yields_summary_chunk_with_citations(
        self, librarian_agent, content_items, mock_openrouter_response
    ):
        """Summary chunk includes wikilink citations."""
        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            chunks = []
            async for chunk in librarian_agent.summarize(
                task="Summarize",
                content=content_items,
            ):
                chunks.append(chunk)

            # Find summary chunk
            summary_chunks = [c for c in chunks if c.type == "summary"]
            assert len(summary_chunks) == 1

            summary = summary_chunks[0]
            assert summary.content is not None
            # Summary should include wikilinks (from mock response)
            assert "[[" in summary.content
            assert "]]" in summary.content

    @pytest.mark.asyncio
    async def test_summarize_yields_done_chunk_at_end(
        self, librarian_agent, content_items, mock_openrouter_response
    ):
        """Done chunk is always emitted at the end."""
        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            chunks = []
            async for chunk in librarian_agent.summarize(
                task="Summarize",
                content=content_items,
            ):
                chunks.append(chunk)

            # Last chunk should be done
            assert len(chunks) > 0
            assert chunks[-1].type == "done"

    @pytest.mark.asyncio
    async def test_summarize_empty_content_yields_error(self, librarian_agent):
        """Summarizing empty content yields error chunk."""
        chunks = []
        async for chunk in librarian_agent.summarize(
            task="Summarize nothing",
            content=[],
        ):
            chunks.append(chunk)

        # Should have an error chunk
        error_chunks = [c for c in chunks if c.type == "error"]
        assert len(error_chunks) == 1
        assert "No content" in error_chunks[0].content

    @pytest.mark.asyncio
    async def test_summarize_respects_token_limit(
        self, librarian_agent, content_items, mock_openrouter_response
    ):
        """Summarization passes max_tokens to LLM."""
        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            max_tokens = 500
            async for _ in librarian_agent.summarize(
                task="Summarize",
                content=content_items,
                max_summary_tokens=max_tokens,
            ):
                pass

            # Verify the API was called with the token limit
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            assert call_args is not None
            request_body = call_args.kwargs.get("json", {})
            assert request_body.get("max_tokens") == max_tokens


class TestLibrarianCaching:
    """Tests for summary caching."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_summary(
        self, librarian_agent, content_items, mock_tool_executor
    ):
        """When cache exists, return it without LLM call."""
        # Setup cache hit response
        cached_content = """---
created: 2025-01-15T10:00:00Z
cache_key: test123
sources:
  - notes/doc1.md
---

# Cached Summary

This is the cached content from [[doc1]].
"""
        mock_tool_executor.execute = AsyncMock(
            return_value=json.dumps({
                "path": "oracle-cache/summaries/vault/2025-01-15/test.md",
                "title": "Cached Summary",
                "content": cached_content,
            })
        )

        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            # LLM should NOT be called if cache hit
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            chunks = []
            async for chunk in librarian_agent.summarize(
                task="Summarize",
                content=content_items,
            ):
                chunks.append(chunk)

            # Should have a cache_hit chunk
            cache_hit_chunks = [c for c in chunks if c.type == "cache_hit"]
            assert len(cache_hit_chunks) == 1
            assert "Cached Summary" in cache_hit_chunks[0].content

            # LLM should not have been called
            mock_client.return_value.__aenter__.return_value.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_llm(
        self, librarian_agent, content_items, mock_tool_executor, mock_openrouter_response
    ):
        """When no cache, call LLM and save result."""
        # Cache miss (default mock behavior)
        mock_tool_executor.execute = AsyncMock(
            return_value=json.dumps({"error": "File not found"})
        )

        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            chunks = []
            async for chunk in librarian_agent.summarize(
                task="Summarize",
                content=content_items,
            ):
                chunks.append(chunk)

            # LLM should have been called
            mock_client.return_value.__aenter__.return_value.post.assert_called()

            # Should have summary chunk (not cache_hit)
            summary_chunks = [c for c in chunks if c.type == "summary"]
            cache_hit_chunks = [c for c in chunks if c.type == "cache_hit"]
            assert len(summary_chunks) == 1
            assert len(cache_hit_chunks) == 0

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(
        self, librarian_agent, content_items, mock_tool_executor, mock_openrouter_response
    ):
        """force_refresh=True ignores cache even if it exists."""
        # Setup cache hit response
        mock_tool_executor.execute = AsyncMock(
            return_value=json.dumps({
                "content": "# Old cached content",
            })
        )

        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            chunks = []
            async for chunk in librarian_agent.summarize(
                task="Summarize",
                content=content_items,
                force_refresh=True,
            ):
                chunks.append(chunk)

            # LLM should be called despite cache existing
            mock_client.return_value.__aenter__.return_value.post.assert_called()

            # Should NOT have cache_hit chunk
            cache_hit_chunks = [c for c in chunks if c.type == "cache_hit"]
            assert len(cache_hit_chunks) == 0

    @pytest.mark.asyncio
    async def test_cached_summary_has_frontmatter(
        self, librarian_agent, content_items, mock_tool_executor, mock_openrouter_response
    ):
        """Cached summary includes YAML frontmatter with metadata."""
        # Track what gets written
        write_calls = []

        async def track_execute(name, arguments, user_id):
            if name == "vault_write":
                write_calls.append(arguments)
                return json.dumps({"path": arguments.get("path"), "version": 1})
            return json.dumps({"error": "File not found"})

        mock_tool_executor.execute = track_execute

        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            async for _ in librarian_agent.summarize(
                task="Summarize for frontmatter test",
                content=content_items,
            ):
                pass

            # Check what was written
            assert len(write_calls) >= 1
            written_body = write_calls[0].get("body", "")

            # Should have YAML frontmatter
            assert written_body.startswith("---")
            assert "---" in written_body[3:]  # Closing frontmatter marker
            # Should have required metadata
            assert "created:" in written_body
            assert "sources:" in written_body
            assert "cache_key:" in written_body


class TestLibrarianOrganization:
    """Tests for vault organization with indexes."""

    @pytest.mark.asyncio
    async def test_organize_creates_index_with_wikilinks(
        self, librarian_agent, mock_tool_executor
    ):
        """Organize creates index.md with wikilinks to all notes."""
        # Setup mock responses
        async def mock_execute(name, arguments, user_id):
            if name == "vault_list":
                return json.dumps({
                    "folder": "docs",
                    "notes": [
                        {"path": "docs/note1.md"},
                        {"path": "docs/note2.md"},
                    ],
                })
            elif name == "vault_read":
                path = arguments.get("path", "")
                if "note1" in path:
                    return json.dumps({
                        "title": "Note 1",
                        "content": "Content of note 1",
                    })
                elif "note2" in path:
                    return json.dumps({
                        "title": "Note 2",
                        "content": "Content of note 2",
                    })
            elif name == "vault_write":
                return json.dumps({
                    "path": arguments.get("path"),
                    "version": 1,
                })
            return json.dumps({"error": "Unknown operation"})

        mock_tool_executor.execute = mock_execute

        chunks = []
        async for chunk in librarian_agent.organize(
            folder="docs",
            create_index=True,
        ):
            chunks.append(chunk)

        # Should have an index chunk
        index_chunks = [c for c in chunks if c.type == "index"]
        assert len(index_chunks) == 1

        index_content = index_chunks[0].content
        # Should contain wikilinks
        assert "[[" in index_content
        assert "]]" in index_content
        # Should reference the notes
        assert "Note 1" in index_content or "note1" in index_content
        assert "Note 2" in index_content or "note2" in index_content

    @pytest.mark.asyncio
    async def test_organize_without_user_id_yields_error(self):
        """Organize requires user_id, yields error if missing."""
        agent = LibrarianAgent(
            api_key="test-key",
            user_id=None,  # No user ID
        )

        chunks = []
        async for chunk in agent.organize(folder="docs"):
            chunks.append(chunk)

        # Should have an error chunk
        error_chunks = [c for c in chunks if c.type == "error"]
        assert len(error_chunks) == 1
        assert "User ID" in error_chunks[0].content

    @pytest.mark.asyncio
    async def test_organize_empty_folder_yields_done(
        self, librarian_agent, mock_tool_executor
    ):
        """Organizing empty folder yields done without index creation."""
        mock_tool_executor.execute = AsyncMock(
            return_value=json.dumps({
                "folder": "empty",
                "notes": [],
            })
        )

        chunks = []
        async for chunk in librarian_agent.organize(
            folder="empty",
            create_index=True,
        ):
            chunks.append(chunk)

        # Should have a done chunk
        done_chunks = [c for c in chunks if c.type == "done"]
        assert len(done_chunks) == 1

        # Should not have an index chunk (no notes to index)
        index_chunks = [c for c in chunks if c.type == "index"]
        assert len(index_chunks) == 0


class TestLibrarianModelSelection:
    """Tests for model selection from settings."""

    @pytest.mark.asyncio
    async def test_uses_model_passed_in_constructor(
        self, content_items, mock_tool_executor, mock_openrouter_response
    ):
        """Agent uses model passed to constructor, not hardcoded default."""
        custom_model = "anthropic/claude-3-haiku"
        agent = LibrarianAgent(
            api_key="test-key",
            model=custom_model,
            user_id="test-user",
            tool_executor=mock_tool_executor,
        )

        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            async for _ in agent.summarize(
                task="Summarize",
                content=content_items,
            ):
                pass

            # Verify the custom model was used
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            request_body = call_args.kwargs.get("json", {})
            assert request_body.get("model") == custom_model

    @pytest.mark.asyncio
    async def test_default_model_used_when_none_specified(
        self, content_items, mock_tool_executor, mock_openrouter_response
    ):
        """Falls back to DEFAULT_MODEL when model not specified."""
        agent = LibrarianAgent(
            api_key="test-key",
            model=None,  # Not specified
            user_id="test-user",
            tool_executor=mock_tool_executor,
        )

        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            async for _ in agent.summarize(
                task="Summarize",
                content=content_items,
            ):
                pass

            # Verify DEFAULT_MODEL was used
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            request_body = call_args.kwargs.get("json", {})
            assert request_body.get("model") == LibrarianAgent.DEFAULT_MODEL


class TestLibrarianErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_llm_error_yields_error_chunk(
        self, librarian_agent, content_items, mock_tool_executor
    ):
        """LLM API errors yield error chunks, don't raise."""
        with patch(
            "backend.src.services.librarian_agent.httpx.AsyncClient"
        ) as mock_client:
            # Simulate API error
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("API connection failed")
            )

            chunks = []
            # Should not raise - errors are yielded as chunks
            async for chunk in librarian_agent.summarize(
                task="Summarize",
                content=content_items,
            ):
                chunks.append(chunk)

            # Should have an error chunk
            error_chunks = [c for c in chunks if c.type == "error"]
            assert len(error_chunks) >= 1
            assert "failed" in error_chunks[0].content.lower()

    @pytest.mark.asyncio
    async def test_vault_list_error_yields_error_chunk(
        self, librarian_agent, mock_tool_executor
    ):
        """Vault list errors yield error chunks in organize."""
        mock_tool_executor.execute = AsyncMock(
            return_value=json.dumps({"error": "Permission denied"})
        )

        chunks = []
        async for chunk in librarian_agent.organize(folder="protected"):
            chunks.append(chunk)

        # Should have an error chunk
        error_chunks = [c for c in chunks if c.type == "error"]
        assert len(error_chunks) >= 1


class TestLibrarianFactoryFunction:
    """Tests for get_librarian_agent factory function."""

    def test_factory_creates_agent_with_all_params(self):
        """Factory creates agent with all provided parameters."""
        agent = get_librarian_agent(
            api_key="test-key",
            model="custom/model",
            project_id="project-123",
            user_id="user-456",
        )

        assert isinstance(agent, LibrarianAgent)
        assert agent.api_key == "test-key"
        assert agent.model == "custom/model"
        assert agent.project_id == "project-123"
        assert agent.user_id == "user-456"

    def test_factory_creates_agent_with_minimal_params(self):
        """Factory creates agent with only required parameters."""
        agent = get_librarian_agent(api_key="test-key")

        assert isinstance(agent, LibrarianAgent)
        assert agent.api_key == "test-key"
        assert agent.model == LibrarianAgent.DEFAULT_MODEL


class TestLibrarianAgentError:
    """Tests for LibrarianAgentError exception."""

    def test_error_has_message(self):
        """Error includes message."""
        error = LibrarianAgentError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"

    def test_error_has_details(self):
        """Error includes optional details dict."""
        details = {"key": "value", "count": 42}
        error = LibrarianAgentError("Error with details", details=details)
        assert error.details == details

    def test_error_details_default_empty(self):
        """Error details default to empty dict."""
        error = LibrarianAgentError("Simple error")
        assert error.details == {}


class TestLibrarianIndexGeneration:
    """Tests for index content generation."""

    def test_generate_index_content_has_header(self, librarian_agent):
        """Generated index has proper header."""
        notes = [
            {"path": "docs/note1.md", "title": "Note 1", "summary": "Summary 1"},
        ]

        content = librarian_agent._generate_index_content(
            folder="docs",
            notes=notes,
        )

        # Should have a title header
        assert content.startswith("#")
        assert "Docs" in content or "docs" in content.lower()

    def test_generate_index_content_has_wikilinks(self, librarian_agent):
        """Generated index includes wikilinks to all notes."""
        notes = [
            {"path": "docs/note1.md", "title": "Note 1", "summary": "Summary 1"},
            {"path": "docs/note2.md", "title": "Note 2", "summary": "Summary 2"},
        ]

        content = librarian_agent._generate_index_content(
            folder="docs",
            notes=notes,
        )

        # Should have wikilinks
        assert "[[Note 1]]" in content
        assert "[[Note 2]]" in content

    def test_generate_index_content_includes_summaries(self, librarian_agent):
        """Generated index includes note summaries."""
        notes = [
            {"path": "docs/note1.md", "title": "Note 1", "summary": "This is a summary"},
        ]

        content = librarian_agent._generate_index_content(
            folder="docs",
            notes=notes,
        )

        # Summary should appear
        assert "summary" in content.lower()

    def test_generate_index_content_includes_note_count(self, librarian_agent):
        """Generated index shows note count."""
        notes = [
            {"path": "docs/note1.md", "title": "Note 1", "summary": ""},
            {"path": "docs/note2.md", "title": "Note 2", "summary": ""},
            {"path": "docs/note3.md", "title": "Note 3", "summary": ""},
        ]

        content = librarian_agent._generate_index_content(
            folder="docs",
            notes=notes,
        )

        # Should mention note count
        assert "3" in content

    def test_generate_index_content_sorted_alphabetically(self, librarian_agent):
        """Notes are sorted alphabetically in index."""
        notes = [
            {"path": "docs/zebra.md", "title": "Zebra", "summary": ""},
            {"path": "docs/apple.md", "title": "Apple", "summary": ""},
            {"path": "docs/mango.md", "title": "Mango", "summary": ""},
        ]

        content = librarian_agent._generate_index_content(
            folder="docs",
            notes=notes,
        )

        # Apple should come before Mango, Mango before Zebra
        apple_pos = content.find("Apple")
        mango_pos = content.find("Mango")
        zebra_pos = content.find("Zebra")

        assert apple_pos < mango_pos < zebra_pos
