"""Tests for Tavily service."""
import pytest
from unittest.mock import AsyncMock, patch

from backend.src.services.tavily_service import (
    TavilySearchService,
    TavilySearchResult,
    TavilySearchResponse,
)


class TestTavilySearchService:
    """Tests for TavilySearchService."""

    def test_is_configured_without_key(self):
        """Should return False when no API key."""
        service = TavilySearchService()
        with patch.dict("os.environ", {}, clear=True):
            assert not service.is_configured()

    def test_is_configured_with_key(self):
        """Should return True when API key provided."""
        service = TavilySearchService(api_key="test-key")
        assert service.is_configured()

    def test_is_configured_with_env_key(self):
        """Should return True when API key in environment."""
        service = TavilySearchService()
        with patch.dict("os.environ", {"TAVILY_API_KEY": "env-test-key"}):
            assert service.is_configured()

    def test_client_raises_without_key(self):
        """Should raise ValueError when accessing client without API key."""
        service = TavilySearchService()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="TAVILY_API_KEY not configured"):
                _ = service.client

    @pytest.mark.asyncio
    async def test_search_returns_structured_response(self):
        """Should return properly structured TavilySearchResponse."""
        service = TavilySearchService(api_key="test-key")

        mock_response = {
            "query": "test query",
            "results": [
                {
                    "url": "http://example.com",
                    "title": "Example Title",
                    "content": "Example content",
                    "score": 0.95,
                    "raw_content": None,
                }
            ],
            "answer": "Test answer",
        }

        with patch.object(service, '_client') as mock_client:
            mock_client.search = AsyncMock(return_value=mock_response)
            service._client = mock_client

            response = await service.search("test query", include_answer=True)

            assert isinstance(response, TavilySearchResponse)
            assert response.query == "test query"
            assert response.answer == "Test answer"
            assert len(response.results) == 1
            assert response.results[0].url == "http://example.com"
            assert response.results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_search_parallel_deduplication(self):
        """Should deduplicate URLs across queries."""
        service = TavilySearchService(api_key="test-key")

        # Mock the search method
        with patch.object(service, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = [
                TavilySearchResponse(
                    query="query1",
                    results=[
                        TavilySearchResult(url="http://a.com", title="A", content="", score=0.9),
                        TavilySearchResult(url="http://b.com", title="B", content="", score=0.8),
                    ]
                ),
                TavilySearchResponse(
                    query="query2",
                    results=[
                        TavilySearchResult(url="http://b.com", title="B", content="", score=0.85),
                        TavilySearchResult(url="http://c.com", title="C", content="", score=0.7),
                    ]
                ),
            ]

            responses = await service.search_parallel(["query1", "query2"], deduplicate=True)

            # Second response should not have b.com (deduplicated)
            assert len(responses) == 2
            assert len(responses[0].results) == 2  # a.com, b.com
            assert len(responses[1].results) == 1  # only c.com
            assert responses[1].results[0].url == "http://c.com"

    @pytest.mark.asyncio
    async def test_search_parallel_no_deduplication(self):
        """Should keep duplicate URLs when deduplicate=False."""
        service = TavilySearchService(api_key="test-key")

        with patch.object(service, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = [
                TavilySearchResponse(
                    query="query1",
                    results=[
                        TavilySearchResult(url="http://a.com", title="A", content="", score=0.9),
                    ]
                ),
                TavilySearchResponse(
                    query="query2",
                    results=[
                        TavilySearchResult(url="http://a.com", title="A", content="", score=0.85),
                    ]
                ),
            ]

            responses = await service.search_parallel(["query1", "query2"], deduplicate=False)

            # Both responses should have their results intact
            assert len(responses) == 2
            assert len(responses[0].results) == 1
            assert len(responses[1].results) == 1

    @pytest.mark.asyncio
    async def test_search_parallel_handles_failures(self):
        """Should continue with valid responses when some queries fail."""
        service = TavilySearchService(api_key="test-key")

        with patch.object(service, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = [
                TavilySearchResponse(
                    query="query1",
                    results=[
                        TavilySearchResult(url="http://a.com", title="A", content="", score=0.9),
                    ]
                ),
                Exception("Search failed"),
                TavilySearchResponse(
                    query="query3",
                    results=[
                        TavilySearchResult(url="http://c.com", title="C", content="", score=0.7),
                    ]
                ),
            ]

            responses = await service.search_parallel(["query1", "query2", "query3"])

            # Should have 2 valid responses, failed query excluded
            assert len(responses) == 2
            assert responses[0].query == "query1"
            assert responses[1].query == "query3"

    @pytest.mark.asyncio
    async def test_get_search_context_format(self):
        """Should return formatted context string."""
        service = TavilySearchService(api_key="test-key")

        with patch.object(service, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = TavilySearchResponse(
                query="test query",
                results=[
                    TavilySearchResult(
                        url="http://example.com",
                        title="Example",
                        content="Example content",
                        score=0.9
                    ),
                ],
                answer="Summary answer"
            )

            context = await service.get_search_context("test query")

            assert "## Summary" in context
            assert "Summary answer" in context
            assert "## Sources" in context
            assert "### [1] Example" in context
            assert "URL: http://example.com" in context
            assert "Score: 0.90" in context
            assert "Example content" in context


class TestTavilySearchResult:
    """Tests for TavilySearchResult dataclass."""

    def test_dataclass_creation(self):
        """Should create TavilySearchResult with all fields."""
        result = TavilySearchResult(
            url="http://example.com",
            title="Test Title",
            content="Test content",
            score=0.85,
            raw_content="Full content"
        )

        assert result.url == "http://example.com"
        assert result.title == "Test Title"
        assert result.content == "Test content"
        assert result.score == 0.85
        assert result.raw_content == "Full content"

    def test_dataclass_optional_raw_content(self):
        """Should allow None for raw_content."""
        result = TavilySearchResult(
            url="http://example.com",
            title="Test",
            content="Content",
            score=0.5
        )

        assert result.raw_content is None


class TestTavilySearchResponse:
    """Tests for TavilySearchResponse dataclass."""

    def test_dataclass_creation(self):
        """Should create TavilySearchResponse with all fields."""
        results = [
            TavilySearchResult(url="http://a.com", title="A", content="", score=0.9)
        ]
        response = TavilySearchResponse(
            query="test",
            results=results,
            answer="Test answer"
        )

        assert response.query == "test"
        assert len(response.results) == 1
        assert response.answer == "Test answer"

    def test_dataclass_optional_answer(self):
        """Should allow None for answer."""
        response = TavilySearchResponse(
            query="test",
            results=[]
        )

        assert response.answer is None
