"""Base SearchProvider protocol and shared types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional, TypedDict


class SearchResult(TypedDict):
    """Normalised result format — matches what open_deep_research expects from Tavily."""
    url: str
    title: str
    content: str           # snippet / summary
    raw_content: Optional[str]  # full page text if available
    score: float           # relevance 0-1


class SearchProvider(ABC):
    """Abstract search provider.

    Implementors must return results in SearchResult format so the
    open_deep_research bridge can consume them uniformly.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True,
    ) -> list[SearchResult]:
        """Execute a single query and return normalised results."""
        ...
