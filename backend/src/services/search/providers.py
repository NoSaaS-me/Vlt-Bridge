"""Concrete SearchProvider implementations.

Each class wraps an existing backend service and normalises its output to
the SearchResult TypedDict expected by the deep research bridge.

Adding a new provider:
    1. Implement SearchProvider here (or in its own file)
    2. Register it in factory.py
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from .base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class NullSearchProvider(SearchProvider):
    """No-op provider — returns empty results. Used when search is disabled."""

    async def search(
        self,
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True,
    ) -> list[SearchResult]:
        logger.debug("NullSearchProvider: search disabled, returning empty results")
        return []


class TavilyProvider(SearchProvider):
    """Wraps TavilySearchService."""

    def __init__(self, api_key: str) -> None:
        from ..tavily_service import TavilySearchService
        self._svc = TavilySearchService(api_key=api_key)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True,
    ) -> list[SearchResult]:
        resp = await self._svc.search(
            query=query,
            max_results=max_results,
            topic=topic,
            include_raw_content=include_raw_content,
        )
        return [
            SearchResult(
                url=r.url,
                title=r.title,
                content=r.content,
                raw_content=r.raw_content,
                score=r.score,
            )
            for r in resp.results
        ]


class OpenRouterProvider(SearchProvider):
    """Wraps OpenRouterSearchService (Perplexity/sonar via OpenRouter)."""

    def __init__(self, api_key: str) -> None:
        from ..openrouter_search import OpenRouterSearchService
        self._svc = OpenRouterSearchService(api_key=api_key)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True,
    ) -> list[SearchResult]:
        resp = await self._svc.search(query=query, max_results=max_results, topic=topic)
        return [
            SearchResult(
                url=r.url,
                title=r.title,
                content=r.content,
                raw_content=r.content,  # OpenRouter doesn't return separate raw_content
                score=r.score,
            )
            for r in resp.results
        ]


# ---------------------------------------------------------------------------
# Future provider stubs — implement and register in factory.py when ready
# ---------------------------------------------------------------------------

class ZAIProvider(SearchProvider):
    """Z.AI search/reader provider (stub — not yet implemented)."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def search(
        self,
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True,
    ) -> list[SearchResult]:
        raise NotImplementedError("ZAIProvider is not yet implemented")


class ProxySearchProvider(SearchProvider):
    """Custom proxy-based search provider (stub — not yet implemented)."""

    def __init__(self, endpoint: str, api_key: Optional[str] = None) -> None:
        self._endpoint = endpoint
        self._api_key = api_key

    async def search(
        self,
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True,
    ) -> list[SearchResult]:
        raise NotImplementedError("ProxySearchProvider is not yet implemented")
