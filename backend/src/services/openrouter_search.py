"""OpenRouter Search Service for Deep Research.

Uses Perplexity models via OpenRouter for web search capabilities.
This provides an alternative to Tavily when users don't have a Tavily API key.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional
import asyncio
import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterSearchResult:
    """Single search result extracted from Perplexity response."""
    url: str
    title: str
    content: str  # Snippet or extracted content
    score: float  # Estimated relevance score 0-1


@dataclass
class OpenRouterSearchResponse:
    """Response from OpenRouter search."""
    query: str
    results: List[OpenRouterSearchResult]
    answer: Optional[str] = None  # Full response from Perplexity


class OpenRouterSearchService:
    """Service for web search via OpenRouter using Perplexity models.

    Uses Perplexity's online models which have built-in web search capability.
    The model searches the web, synthesizes results, and provides citations.
    """

    # Perplexity models with web search capability via OpenRouter
    SEARCH_MODEL = "perplexity/sonar"  # Cost-effective with web search
    SEARCH_MODEL_PRO = "perplexity/sonar-pro"  # Higher quality

    def __init__(self, api_key: Optional[str] = None, use_pro: bool = False):
        """Initialize with API key.

        Args:
            api_key: OpenRouter API key
            use_pro: Whether to use the pro model (higher quality, more expensive)
        """
        self.api_key = api_key
        self.model = self.SEARCH_MODEL_PRO if use_pro else self.SEARCH_MODEL
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-load the async client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("OpenRouter API key not configured")
            self._client = httpx.AsyncClient(
                base_url="https://openrouter.ai/api/v1",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://vlt-bridge.local",
                    "X-Title": "Vlt Bridge Deep Research",
                },
                timeout=60.0,
            )
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
    ) -> OpenRouterSearchResponse:
        """Execute a single search query using Perplexity via OpenRouter.

        Args:
            query: Search query
            max_results: Target number of results to extract
            topic: Topic category (used to adjust prompt)

        Returns:
            OpenRouterSearchResponse with extracted results
        """
        # Build a prompt that encourages structured output with citations
        topic_context = {
            "general": "",
            "news": "Focus on recent news and current events. ",
            "finance": "Focus on financial data, markets, and business news. ",
        }

        system_prompt = f"""You are a research assistant. {topic_context.get(topic, '')}
Search the web and provide accurate, well-cited information.

For each piece of information, include the source URL in brackets like [Source: URL].
Structure your response with clear sections and citations."""

        user_prompt = f"""Search for information about: {query}

Provide {max_results} key findings with sources. For each finding:
1. State the key information
2. Include the source URL in brackets [Source: https://...]

Be thorough and cite your sources."""

        try:
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4000,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract the response content
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse the response to extract sources
            results = self._extract_sources(content, max_results)

            return OpenRouterSearchResponse(
                query=query,
                results=results,
                answer=content,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter search failed for '{query}': {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter search failed for '{query}': {e}")
            raise

    def _extract_sources(self, content: str, max_results: int) -> List[OpenRouterSearchResult]:
        """Extract source URLs and content from Perplexity response.

        Perplexity includes citations in various formats:
        - [Source: URL]
        - [1]: URL
        - (Source: URL)
        - URLs inline in text
        """
        results = []

        # Pattern to match URLs
        url_pattern = r'https?://[^\s\]\)\>\"\'\,]+'

        # Pattern to match citations like [Source: URL] or [1]: URL
        citation_patterns = [
            r'\[Source:\s*(https?://[^\]]+)\]',
            r'\[\d+\]:\s*(https?://[^\s]+)',
            r'\(Source:\s*(https?://[^\)]+)\)',
            r'\[([^\]]+)\]\((https?://[^\)]+)\)',  # Markdown links
        ]

        seen_urls = set()

        # First, try to extract structured citations
        for pattern in citation_patterns:
            for match in re.finditer(pattern, content):
                # Get the URL (last group is typically the URL)
                groups = match.groups()
                url = groups[-1] if groups else None
                if url and url not in seen_urls:
                    seen_urls.add(url)

                    # Extract context around the citation
                    start = max(0, match.start() - 200)
                    end = min(len(content), match.end() + 200)
                    context = content[start:end].strip()

                    # Clean up context
                    context = re.sub(r'\s+', ' ', context)

                    results.append(OpenRouterSearchResult(
                        url=url,
                        title=self._extract_title_from_url(url),
                        content=context[:500],
                        score=0.8,  # Default relevance
                    ))

                    if len(results) >= max_results:
                        break

            if len(results) >= max_results:
                break

        # If we didn't find structured citations, look for raw URLs
        if len(results) < max_results:
            for match in re.finditer(url_pattern, content):
                url = match.group()
                # Clean up URL (remove trailing punctuation)
                url = re.sub(r'[.,;:!?\)\]]+$', '', url)

                if url not in seen_urls and self._is_valid_source_url(url):
                    seen_urls.add(url)

                    # Extract context
                    start = max(0, match.start() - 150)
                    end = min(len(content), match.end() + 150)
                    context = content[start:end].strip()
                    context = re.sub(r'\s+', ' ', context)

                    results.append(OpenRouterSearchResult(
                        url=url,
                        title=self._extract_title_from_url(url),
                        content=context[:500],
                        score=0.6,
                    ))

                    if len(results) >= max_results:
                        break

        return results

    def _extract_title_from_url(self, url: str) -> str:
        """Extract a reasonable title from a URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)

            # Get domain as fallback
            domain = parsed.netloc.replace('www.', '')

            # Try to get title from path
            path = parsed.path.strip('/')
            if path:
                # Take the last path segment
                segment = path.split('/')[-1]
                # Clean up
                segment = segment.replace('-', ' ').replace('_', ' ')
                segment = re.sub(r'\.[a-z]+$', '', segment)  # Remove extension
                if len(segment) > 5:
                    return f"{segment.title()} - {domain}"

            return domain
        except Exception:
            return url[:50]

    def _is_valid_source_url(self, url: str) -> bool:
        """Check if URL is a valid source (not an image, asset, etc.)."""
        invalid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.css', '.js']
        invalid_domains = ['cdn.', 'static.', 'assets.', 'images.']

        url_lower = url.lower()

        for ext in invalid_extensions:
            if url_lower.endswith(ext):
                return False

        for domain in invalid_domains:
            if domain in url_lower:
                return False

        return True

    async def search_parallel(
        self,
        queries: List[str],
        max_results_per_query: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        deduplicate: bool = True,
    ) -> List[OpenRouterSearchResponse]:
        """Execute multiple queries in parallel with optional deduplication.

        Args:
            queries: List of search queries
            max_results_per_query: Max results per query
            topic: Topic category
            deduplicate: Whether to remove duplicate URLs across queries

        Returns:
            List of search responses
        """
        # Run searches in parallel with concurrency limit
        semaphore = asyncio.Semaphore(3)  # Limit concurrent requests

        async def search_with_limit(query: str) -> OpenRouterSearchResponse:
            async with semaphore:
                return await self.search(query, max_results_per_query, topic)

        tasks = [search_with_limit(q) for q in queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = []
        seen_urls = set()

        for resp in responses:
            if isinstance(resp, Exception):
                logger.warning(f"Query failed: {resp}")
                continue

            if deduplicate:
                unique_results = []
                for result in resp.results:
                    if result.url not in seen_urls:
                        seen_urls.add(result.url)
                        unique_results.append(result)
                resp.results = unique_results

            valid_responses.append(resp)

        return valid_responses

    def is_configured(self) -> bool:
        """Check if OpenRouter is properly configured."""
        return bool(self.api_key)

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance
_openrouter_search_service: Optional[OpenRouterSearchService] = None


def get_openrouter_search_service(api_key: Optional[str] = None) -> OpenRouterSearchService:
    """Get or create the OpenRouter search service singleton."""
    global _openrouter_search_service
    if _openrouter_search_service is None or api_key:
        _openrouter_search_service = OpenRouterSearchService(api_key=api_key)
    return _openrouter_search_service
