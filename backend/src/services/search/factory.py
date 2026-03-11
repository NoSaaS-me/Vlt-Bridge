"""SearchProvider factory.

Maps the app's search_provider setting strings to concrete provider instances.
This is the single file to edit when adding a new provider.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import SearchProvider
from .providers import NullSearchProvider, OpenRouterProvider, TavilyProvider

logger = logging.getLogger(__name__)

# Human-readable labels shown in the deep-research progress stream
PROVIDER_LABELS: dict[str, str] = {
    "tavily": "Tavily web search",
    "openrouter": "OpenRouter (Perplexity) web search",
    "zai": "Z.AI search/reader",
    "proxy": "custom proxy search",
    "none": "no web search — configure a search provider in Settings",
}


def get_search_provider(
    provider_name: str,
    *,
    tavily_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    # Future kwargs: zai_api_key, proxy_endpoint, proxy_api_key, …
) -> SearchProvider:
    """Return a configured SearchProvider for the given provider_name.

    Falls back to NullSearchProvider if the requested provider lacks
    required credentials rather than raising, so deep research still
    runs (without web results) instead of hard-erroring.
    """
    match provider_name:
        case "tavily":
            if not tavily_api_key:
                logger.warning("Tavily selected but no API key — falling back to null search")
                return NullSearchProvider()
            return TavilyProvider(api_key=tavily_api_key)

        case "openrouter":
            if not openrouter_api_key:
                logger.warning("OpenRouter search selected but no API key — falling back to null search")
                return NullSearchProvider()
            return OpenRouterProvider(api_key=openrouter_api_key)

        case "zai":
            # TODO: implement ZAIProvider and wire api_key here
            logger.warning("ZAI provider not yet implemented — falling back to null search")
            return NullSearchProvider()

        case "proxy":
            # TODO: implement ProxySearchProvider and wire endpoint/key here
            logger.warning("Proxy provider not yet implemented — falling back to null search")
            return NullSearchProvider()

        case _:
            return NullSearchProvider()
