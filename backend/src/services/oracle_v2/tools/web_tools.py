"""Web tools for the Oracle CodeAct agent — web search and fetch."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import httpx

logger = logging.getLogger(__name__)

_NO_SEARCH_MSG = (
    "Web search not configured. Set up a search provider in Settings."
)


def _run_async(coro) -> object:
    """Run a coroutine synchronously (eval_fn runs in a sync thread)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an event loop — use run_coroutine_threadsafe from another thread
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=60)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _format_tavily_results(response) -> str:
    """Format TavilySearchResponse into a readable string."""
    lines = [f"Search results for: {response.query}\n"]
    if response.answer:
        lines.append(f"Summary: {response.answer}\n")
    for i, r in enumerate(response.results, 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    URL: {r.url}")
        lines.append(f"    {r.content[:300]}\n")
    return "\n".join(lines)


def _format_openrouter_results(response) -> str:
    """Format OpenRouterSearchResponse into a readable string."""
    lines = [f"Search results for: {response.query}\n"]
    if response.answer:
        lines.append(f"Answer: {response.answer[:1000]}\n")
    for i, r in enumerate(response.results, 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    URL: {r.url}")
        lines.append(f"    {r.content[:300]}\n")
    return "\n".join(lines)


def make_web_tools(
    search_provider: str,
    tavily_api_key: str = "",
    openrouter_api_key: str = "",
) -> list[Callable]:
    """Build web tool callables bound to the configured search provider.

    Args:
        search_provider: "tavily", "openrouter", or "none".
        tavily_api_key: Tavily API key (required if provider="tavily").
        openrouter_api_key: OpenRouter API key (required if provider="openrouter").

    Returns:
        [web_search, web_fetch]
    """
    # Lazy-init search service to avoid import errors when deps missing
    _tavily_svc = None
    _openrouter_svc = None

    def _get_tavily():
        nonlocal _tavily_svc
        if _tavily_svc is None:
            from backend.src.services.tavily_service import TavilySearchService
            _tavily_svc = TavilySearchService(api_key=tavily_api_key or None)
        return _tavily_svc

    def _get_openrouter():
        nonlocal _openrouter_svc
        if _openrouter_svc is None:
            from backend.src.services.openrouter_search import OpenRouterSearchService
            _openrouter_svc = OpenRouterSearchService(api_key=openrouter_api_key or None)
        return _openrouter_svc

    def web_search(query: str, num_results: int = 5) -> str:
        """Search the web for information.

        Args:
            query: Search query string.
            num_results: Number of results to return (default 5).

        Returns:
            Formatted search results with titles, URLs, and snippets.
        """
        if search_provider == "none" or not search_provider:
            return _NO_SEARCH_MSG
        try:
            if search_provider == "tavily":
                if not tavily_api_key:
                    return _NO_SEARCH_MSG
                svc = _get_tavily()
                response = _run_async(svc.search(query, max_results=num_results))
                return _format_tavily_results(response)
            elif search_provider == "openrouter":
                if not openrouter_api_key:
                    return _NO_SEARCH_MSG
                svc = _get_openrouter()
                response = _run_async(svc.search(query, max_results=num_results))
                return _format_openrouter_results(response)
            else:
                return _NO_SEARCH_MSG
        except Exception as exc:
            logger.warning("web_search failed: %s", exc)
            return f"Web search failed: {exc}"

    def web_fetch(url: str) -> str:
        """Fetch and extract text content from a URL.

        Args:
            url: Full URL to fetch (must start with http:// or https://).

        Returns:
            Extracted text content (truncated to 8000 chars).
        """
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://"
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                html = resp.text

            try:
                import trafilatura
                text = trafilatura.extract(html) or html
            except ImportError:
                text = html

            if len(text) > 8000:
                return text[:8000] + " [truncated]"
            return text
        except Exception as exc:
            logger.warning("web_fetch failed for %r: %s", url, exc)
            return f"Failed to fetch {url}: {exc}"

    return [web_search, web_fetch]
