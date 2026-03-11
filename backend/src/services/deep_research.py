"""Deep Research service — web-grounded multi-step research via open_deep_research.

Architecture:
  - Wraps the open_deep_research LangGraph pipeline.
  - Injects a SearchProvider via monkey-patching `open_deep_research.utils.tavily_search_async`
    so any configured search backend (Tavily, OpenRouter/Perplexity, future Z.AI / proxy…)
    is used transparently. The graph always sees `search_api: "tavily"` but the actual
    HTTP requests go through our SearchProvider abstraction.
  - All LLM calls route through OpenRouter (OPENAI_BASE_URL + OPENAI_API_KEY env vars).
  - Streams OracleStreamChunk objects — same format as RLMOracleWrapper.

Adding a new search backend:
    1. Implement SearchProvider in services/search/providers.py
    2. Register it in services/search/factory.py
    — no changes needed here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncGenerator, Literal, Optional

from .search import SearchProvider, SearchResult, get_search_provider, PROVIDER_LABELS

logger = logging.getLogger(__name__)

# Serialise concurrent deep-research runs so env-var injection and the
# monkey-patch are both safe. Deep research is expensive (30-90 s), so
# serialising is acceptable.
_DEEP_RESEARCH_LOCK = asyncio.Lock()

# LangGraph node names → progress labels shown to the user
_NODE_LABELS: dict[str, str] = {
    "generate_queries": "🔍 Generating research queries...",
    "research_supervisor": "🗂️ Planning research strategy...",
    "researcher": "🌐 Searching and gathering sources...",
    "compress_notes": "🗜️ Compressing research notes...",
    "write_final_sections": "✍️ Writing final report...",
    "write_report": "✍️ Writing final report...",
}


class DeepResearchWrapper:
    """Async generator wrapper around open_deep_research, yields OracleStreamChunks."""

    def __init__(
        self,
        openrouter_api_key: str,
        model: str = "google/gemini-2.0-flash",
        search_provider: str = "none",
        tavily_api_key: Optional[str] = None,
        openrouter_search_api_key: Optional[str] = None,
    ) -> None:
        self._openrouter_key = openrouter_api_key
        self._model = model
        self._search_provider_name = search_provider
        self._provider: SearchProvider = get_search_provider(
            search_provider,
            tavily_api_key=tavily_api_key,
            openrouter_api_key=openrouter_search_api_key or openrouter_api_key,
        )
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    async def process_query(
        self,
        query: str,
        context_id: Optional[str] = None,
    ) -> AsyncGenerator:
        async with _DEEP_RESEARCH_LOCK:
            saved_env = _patch_env(
                OPENAI_API_KEY=self._openrouter_key,
                OPENAI_BASE_URL="https://openrouter.ai/api/v1",
            )
            saved_fn = _patch_search_async(self._provider)
            try:
                async for chunk in self._run_graph(query):
                    yield chunk
            finally:
                _restore_search_async(saved_fn)
                _restore_env(saved_env)

    async def _run_graph(self, query: str) -> AsyncGenerator:
        from ...models.oracle import OracleStreamChunk

        try:
            from open_deep_research.deep_researcher import deep_researcher_builder
            from langgraph.checkpoint.memory import MemorySaver
            import uuid
        except ImportError as exc:
            yield OracleStreamChunk(type="error", error=f"open-deep-research not installed: {exc}")
            return

        graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
        model_str = f"openai:{self._model}"
        provider_label = PROVIDER_LABELS.get(self._search_provider_name, self._search_provider_name)

        config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "allow_clarification": False,
                # Always "tavily" — our monkey-patch intercepts the call
                "search_api": "tavily",
                "research_model": model_str,
                "summarization_model": model_str,
                "compression_model": model_str,
                "final_report_model": model_str,
                "max_concurrent_research_units": 3,
                "max_researcher_iterations": 4,
                "max_react_tool_calls": 8,
            }
        }

        yield OracleStreamChunk(
            type="progress",
            content=f"🔬 Starting deep research using {provider_label}...\n",
        )

        final_report: Optional[str] = None
        seen_nodes: set[str] = set()

        try:
            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": query}]},
                config,
                stream_mode="updates",
            ):
                if self._cancelled:
                    yield OracleStreamChunk(type="error", error="Deep research cancelled.")
                    return

                if not chunk:
                    continue

                node_name = next(iter(chunk))
                state_delta = chunk[node_name]

                if "final_report" in state_delta and state_delta["final_report"]:
                    final_report = state_delta["final_report"]
                    continue

                if node_name not in seen_nodes:
                    seen_nodes.add(node_name)
                    label = _NODE_LABELS.get(node_name, f"⚙️ {node_name}...")
                    yield OracleStreamChunk(type="progress", content=f"{label}\n")

        except Exception as exc:
            logger.exception("Deep research graph failed")
            yield OracleStreamChunk(type="error", error=f"Deep research failed: {exc}")
            return

        if not final_report:
            yield OracleStreamChunk(
                type="error",
                error="Deep research completed but produced no report.",
            )
            return

        chunk_size = 80
        for i in range(0, len(final_report), chunk_size):
            yield OracleStreamChunk(type="content", content=final_report[i : i + chunk_size])
            if self._cancelled:
                break

        yield OracleStreamChunk(type="done", metadata={"mode": "deep_research"})


# ---------------------------------------------------------------------------
# Monkey-patch helpers
# ---------------------------------------------------------------------------

def _patch_search_async(provider: SearchProvider):
    """Replace open_deep_research.utils.tavily_search_async with a bridge
    that routes to *provider*. Returns the original function for restore."""
    import open_deep_research.utils as _odr_utils

    original = _odr_utils.tavily_search_async

    async def _bridge(
        search_queries: list[str],
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True,
        config=None,
    ) -> list[dict]:
        """Drop-in replacement that routes to our SearchProvider."""
        async def _one(query: str) -> dict:
            try:
                results: list[SearchResult] = await provider.search(
                    query=query,
                    max_results=max_results,
                    topic=topic,
                    include_raw_content=include_raw_content,
                )
            except Exception as exc:
                logger.warning("Search provider failed for %r: %s", query, exc)
                results = []
            return {
                "query": query,
                "results": [
                    {
                        "url": r["url"],
                        "title": r["title"],
                        "content": r["content"],
                        "raw_content": r.get("raw_content"),
                        "score": r.get("score", 0.5),
                    }
                    for r in results
                ],
            }

        return await asyncio.gather(*[_one(q) for q in search_queries])

    _odr_utils.tavily_search_async = _bridge
    return original


def _restore_search_async(original) -> None:
    import open_deep_research.utils as _odr_utils
    _odr_utils.tavily_search_async = original


# ---------------------------------------------------------------------------
# Env helpers — safe under _DEEP_RESEARCH_LOCK
# ---------------------------------------------------------------------------

def _patch_env(**kwargs: str) -> dict[str, Optional[str]]:
    saved: dict[str, Optional[str]] = {}
    for key, value in kwargs.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = value
    return saved


def _restore_env(saved: dict[str, Optional[str]]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
