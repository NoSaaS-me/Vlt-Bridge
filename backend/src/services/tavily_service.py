"""Tavily Search Service for Deep Research."""

from dataclasses import dataclass
from typing import List, Literal, Optional
import asyncio
import logging
import os

from tavily import AsyncTavilyClient

logger = logging.getLogger(__name__)


@dataclass
class TavilySearchResult:
    """Single search result from Tavily."""
    url: str
    title: str
    content: str  # Snippet or full content
    score: float  # Relevance score 0-1
    raw_content: Optional[str] = None  # Full page content if requested


@dataclass
class TavilySearchResponse:
    """Response from Tavily search."""
    query: str
    results: List[TavilySearchResult]
    answer: Optional[str] = None  # Tavily's AI answer if requested


class TavilySearchService:
    """Service for Tavily web search."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key from param or environment."""
        self.api_key = api_key
        self._client: Optional[AsyncTavilyClient] = None

    @property
    def client(self) -> AsyncTavilyClient:
        """Lazy-load the async client."""
        if self._client is None:
            if not self.api_key:
                self.api_key = os.getenv("TAVILY_API_KEY")
            if not self.api_key:
                raise ValueError("TAVILY_API_KEY not configured")
            self._client = AsyncTavilyClient(api_key=self.api_key)
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_answer: bool = False,
        include_raw_content: bool = False,
    ) -> TavilySearchResponse:
        """Execute a single search query."""
        try:
            response = await self.client.search(
                query=query,
                max_results=max_results,
                topic=topic,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
            )

            results = [
                TavilySearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    raw_content=r.get("raw_content"),
                )
                for r in response.get("results", [])
            ]

            return TavilySearchResponse(
                query=query,
                results=results,
                answer=response.get("answer"),
            )
        except Exception as e:
            logger.error(f"Tavily search failed for '{query}': {e}")
            raise

    async def search_parallel(
        self,
        queries: List[str],
        max_results_per_query: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        deduplicate: bool = True,
    ) -> List[TavilySearchResponse]:
        """Execute multiple queries in parallel with optional deduplication."""
        tasks = [
            self.search(q, max_results_per_query, topic)
            for q in queries
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = []
        seen_urls = set()

        for resp in responses:
            if isinstance(resp, Exception):
                logger.warning(f"Query failed: {resp}")
                continue

            if deduplicate:
                # Filter out duplicate URLs
                unique_results = []
                for result in resp.results:
                    if result.url not in seen_urls:
                        seen_urls.add(result.url)
                        unique_results.append(result)
                resp.results = unique_results

            valid_responses.append(resp)

        return valid_responses

    async def get_search_context(
        self,
        query: str,
        max_tokens: int = 4000,
        topic: Literal["general", "news", "finance"] = "general",
    ) -> str:
        """Get formatted context from search results for LLM consumption."""
        response = await self.search(
            query=query,
            max_results=10,
            topic=topic,
            include_answer=True,
        )

        context_parts = []

        if response.answer:
            context_parts.append(f"## Summary\n{response.answer}\n")

        context_parts.append("## Sources\n")
        for i, result in enumerate(response.results, 1):
            context_parts.append(
                f"### [{i}] {result.title}\n"
                f"URL: {result.url}\n"
                f"Score: {result.score:.2f}\n\n"
                f"{result.content}\n\n"
            )

        return "\n".join(context_parts)

    def is_configured(self) -> bool:
        """Check if Tavily is properly configured."""
        return bool(self.api_key or os.getenv("TAVILY_API_KEY"))


# Singleton instance
_tavily_service: Optional[TavilySearchService] = None


def get_tavily_service(api_key: Optional[str] = None) -> TavilySearchService:
    """Get or create the Tavily service singleton."""
    global _tavily_service
    if _tavily_service is None or api_key:
        _tavily_service = TavilySearchService(api_key=api_key)
    return _tavily_service
