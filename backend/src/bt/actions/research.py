"""
Research Actions - BT action functions for the Deep Research subtree.

These functions are called by the research.lua and single-researcher.lua trees.
Each function receives a TickContext and returns a RunStatus.

Migration from: backend/src/services/research/behaviors.py
Target: All action functions needed by research Lua trees

Part of the BT Universal Runtime (spec 019).
Tasks covered: Phase 6.1 Research Migration
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..state.base import RunStatus

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard

logger = logging.getLogger(__name__)


# =============================================================================
# Blackboard Helpers (same pattern as oracle.py)
# =============================================================================


def bb_get(bb: "TypedBlackboard", key: str, default: Any = None) -> Any:
    """Get value from blackboard without schema validation."""
    value = bb._lookup(key)
    return value if value is not None else default


def bb_set(bb: "TypedBlackboard", key: str, value: Any) -> None:
    """Set value in blackboard without schema validation."""
    bb._data[key] = value
    bb._writes.add(key)


# =============================================================================
# Constants
# =============================================================================

# Research configuration defaults
DEFAULT_MAX_SOURCES = 10
DEFAULT_MAX_TOOL_CALLS = 10
DEFAULT_MAX_CONCURRENT = 5

# Model defaults
DEFAULT_PLANNING_MODEL = "deepseek/deepseek-chat"
DEFAULT_RESEARCH_MODEL = "deepseek/deepseek-chat"
DEFAULT_COMPRESSION_MODEL = "deepseek/deepseek-chat"
DEFAULT_REPORT_MODEL = "deepseek/deepseek-chat"


# =============================================================================
# Phase 1: Initialization Actions
# =============================================================================


def init_research(ctx: "TickContext") -> RunStatus:
    """Initialize research state and generate research_id.

    Sets up initial blackboard state for a new research task.
    """
    bb = ctx.blackboard
    if bb is None:
        logger.error("init_research: No blackboard available")
        return RunStatus.FAILURE

    query = bb_get(bb, "query", "")
    depth = bb_get(bb, "depth", "standard")

    # Generate research ID
    words = query.lower().split()[:3]
    slug = "-".join(
        "".join(c for c in word if c.isalnum())
        for word in words
    )[:30]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    unique = uuid.uuid4().hex[:6]
    research_id = f"{timestamp}-{slug}-{unique}"

    bb_set(bb, "research_id", research_id)
    bb_set(bb, "started_at", datetime.now(timezone.utc).isoformat())
    bb_set(bb, "status", "planning")

    # Initialize artifact containers
    bb_set(bb, "brief", None)
    bb_set(bb, "researchers", [])
    bb_set(bb, "sources", [])
    bb_set(bb, "findings", [])
    bb_set(bb, "report", None)

    # Initialize progress tracking
    bb_set(bb, "progress_phase", "initializing")
    bb_set(bb, "progress_pct", 0)
    bb_set(bb, "progress_message", "Starting research...")
    bb_set(bb, "sources_found", 0)

    # Initialize config based on depth
    config = _get_config_for_depth(depth)
    bb_set(bb, "config", config)

    # Initialize pending chunks for streaming
    bb_set(bb, "_pending_chunks", [])

    logger.info(f"Research initialized: {research_id}")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def _get_config_for_depth(depth: str) -> Dict[str, Any]:
    """Get research configuration for depth level."""
    if depth == "quick":
        return {
            "max_concurrent_researchers": 1,
            "max_tool_calls_per_researcher": 5,
            "max_sources": 5,
            "planning_model": DEFAULT_PLANNING_MODEL,
            "research_model": DEFAULT_RESEARCH_MODEL,
            "compression_model": DEFAULT_COMPRESSION_MODEL,
            "report_model": DEFAULT_REPORT_MODEL,
        }
    elif depth == "thorough":
        return {
            "max_concurrent_researchers": 5,
            "max_tool_calls_per_researcher": 15,
            "max_sources": 30,
            "planning_model": DEFAULT_PLANNING_MODEL,
            "research_model": DEFAULT_RESEARCH_MODEL,
            "compression_model": DEFAULT_COMPRESSION_MODEL,
            "report_model": DEFAULT_REPORT_MODEL,
        }
    else:  # standard
        return {
            "max_concurrent_researchers": 3,
            "max_tool_calls_per_researcher": 10,
            "max_sources": 10,
            "planning_model": DEFAULT_PLANNING_MODEL,
            "research_model": DEFAULT_RESEARCH_MODEL,
            "compression_model": DEFAULT_COMPRESSION_MODEL,
            "report_model": DEFAULT_REPORT_MODEL,
        }


def emit_start_event(ctx: "TickContext") -> RunStatus:
    """Emit research.start event to ANS."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    research_id = bb_get(bb, "research_id")
    user_id = bb_get(bb, "user_id")
    query = bb_get(bb, "query")

    try:
        from src.services.ans.bus import get_event_bus
        from src.services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="research.start",
            source="research_bt",
            severity=Severity.INFO,
            payload={
                "research_id": research_id,
                "user_id": user_id,
                "query_preview": str(query)[:100] if query else "",
            }
        ))
    except Exception as e:
        logger.warning(f"Failed to emit research.start event: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Phase 2: Brief Generation Actions
# =============================================================================


def set_phase(ctx: "TickContext") -> RunStatus:
    """Set research phase and emit progress update."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    # Get args from node definition
    args = ctx.node_args or {}
    phase = args.get("phase", "unknown")
    pct = args.get("pct", 0)
    message = args.get("message")

    # Update progress state
    bb_set(bb, "progress_phase", phase)
    bb_set(bb, "progress_pct", pct)

    if message:
        bb_set(bb, "progress_message", message)
    else:
        # Generate dynamic message based on sources found
        sources_found = bb_get(bb, "sources_found", 0)
        bb_set(bb, "progress_message", f"Found {sources_found} sources")

    # Emit progress chunk for streaming
    _add_pending_chunk(bb, {
        "type": "research_progress",
        "research_id": bb_get(bb, "research_id"),
        "phase": phase,
        "pct": pct,
        "message": bb_get(bb, "progress_message"),
        "sources_found": bb_get(bb, "sources_found", 0),
    })

    ctx.mark_progress()
    return RunStatus.SUCCESS


def generate_brief(ctx: "TickContext") -> RunStatus:
    """Generate research brief from query via LLM.

    This is an LLM call action - it should be invoked within an llm_call node.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    query = bb_get(bb, "query", "")
    config = bb_get(bb, "config", {})

    try:
        # Load prompt template
        from src.services.prompt_loader import PromptLoader
        prompts = PromptLoader()

        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prompt = prompts.load(
            "research/brief.md",
            {
                "query": query,
                "current_date": current_date,
            }
        )

        # Call LLM service
        from src.services.research.llm_service import ResearchLLMService
        user_id = bb_get(bb, "user_id", "")
        llm = ResearchLLMService(user_id=user_id)

        # Run async LLM call
        loop = asyncio.get_event_loop()
        brief_data = loop.run_until_complete(
            llm.generate_json(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3,
            )
        )

        # Parse into brief structure
        brief = {
            "original_query": query,
            "refined_question": brief_data.get("refined_question", query),
            "scope": brief_data.get("scope", "General research"),
            "subtopics": brief_data.get("subtopics", [query]),
            "constraints": brief_data.get("constraints"),
            "language": brief_data.get("language", "en"),
        }

        bb_set(bb, "brief", brief)
        logger.info(f"Generated brief with {len(brief['subtopics'])} subtopics")

        ctx.mark_progress()
        return RunStatus.SUCCESS

    except Exception as e:
        logger.error(f"Failed to generate brief: {e}")
        return RunStatus.FAILURE


def validate_brief(ctx: "TickContext") -> RunStatus:
    """Check if brief is valid."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    brief = bb_get(bb, "brief")

    if brief is None:
        return RunStatus.FAILURE

    if not isinstance(brief, dict):
        return RunStatus.FAILURE

    if not brief.get("subtopics"):
        return RunStatus.FAILURE

    ctx.mark_progress()
    return RunStatus.SUCCESS


def create_fallback_brief(ctx: "TickContext") -> RunStatus:
    """Create minimal brief from query when LLM fails."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    query = bb_get(bb, "query", "Unknown query")

    brief = {
        "original_query": query,
        "refined_question": query,
        "scope": "General research",
        "subtopics": [query],
        "constraints": None,
        "language": "en",
    }

    bb_set(bb, "brief", brief)
    logger.info("Created fallback brief")

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Phase 3: Planning Actions
# =============================================================================


def plan_subtopics(ctx: "TickContext") -> RunStatus:
    """Create researcher assignments from brief subtopics."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    brief = bb_get(bb, "brief")
    config = bb_get(bb, "config", {})

    if not brief or not brief.get("subtopics"):
        logger.error("No brief or subtopics for planning")
        return RunStatus.FAILURE

    max_concurrent = config.get("max_concurrent_researchers", DEFAULT_MAX_CONCURRENT)
    max_tool_calls = config.get("max_tool_calls_per_researcher", DEFAULT_MAX_TOOL_CALLS)

    subtopics = brief["subtopics"][:max_concurrent]

    researchers = [
        {
            "subtopic": subtopic,
            "max_tool_calls": max_tool_calls,
            "sources": [],
            "tool_calls": 0,
            "completed": False,
            "error": None,
        }
        for subtopic in subtopics
    ]

    bb_set(bb, "researchers", researchers)
    logger.info(f"Planned {len(researchers)} researchers")

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Phase 4: Parallel Research Actions
# =============================================================================


def init_researcher(ctx: "TickContext") -> RunStatus:
    """Initialize researcher-local state."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    subtopic = bb_get(bb, "subtopic", "")
    max_tool_calls = bb_get(bb, "max_tool_calls", DEFAULT_MAX_TOOL_CALLS)

    bb_set(bb, "researcher_sources", [])
    bb_set(bb, "researcher_queries", [])
    bb_set(bb, "researcher_error", None)
    bb_set(bb, "researcher_completed", False)
    bb_set(bb, "tool_calls_made", 0)

    logger.debug(f"Initialized researcher for: {subtopic}")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def emit_researcher_event(ctx: "TickContext") -> RunStatus:
    """Emit researcher lifecycle event."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    args = ctx.node_args or {}
    event_type = args.get("event_type", "unknown")

    subtopic = bb_get(bb, "subtopic", "")
    researcher_index = bb_get(bb, "researcher_index", 0)
    research_id = bb_get(bb, "research_id")

    try:
        from src.services.ans.bus import get_event_bus
        from src.services.ans.event import Event, Severity

        bus = get_event_bus()
        severity = Severity.ERROR if event_type == "error" else Severity.INFO

        bus.emit(Event(
            type=f"research.researcher.{event_type}",
            source="research_bt",
            severity=severity,
            payload={
                "research_id": research_id,
                "researcher_index": researcher_index,
                "subtopic": subtopic,
                "error": bb_get(bb, "researcher_error") if event_type == "error" else None,
            }
        ))
    except Exception as e:
        logger.warning(f"Failed to emit researcher event: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def generate_search_queries(ctx: "TickContext") -> RunStatus:
    """Generate targeted search queries for subtopic."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    subtopic = bb_get(bb, "subtopic", "")
    brief = bb_get(bb, "brief", {})

    current_year = datetime.now(timezone.utc).year

    # Generate multiple query variants
    queries = [
        subtopic,
        f"{subtopic} {current_year}",
        f"{subtopic} latest research",
    ]

    # Add constraints if available
    constraints = brief.get("constraints")
    if constraints:
        queries.append(f"{subtopic} {constraints}")

    # Limit to 4 queries
    queries = queries[:4]

    bb_set(bb, "researcher_queries", queries)
    logger.debug(f"Generated {len(queries)} queries for: {subtopic}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def has_tavily(ctx: "TickContext") -> RunStatus:
    """Check if Tavily search is available."""
    api_key = os.getenv("TAVILY_API_KEY")

    if api_key:
        ctx.mark_progress()
        return RunStatus.SUCCESS

    return RunStatus.FAILURE


def has_openrouter_search(ctx: "TickContext") -> RunStatus:
    """Check if OpenRouter search is available."""
    api_key = os.getenv("OPENROUTER_API_KEY")

    if api_key:
        ctx.mark_progress()
        return RunStatus.SUCCESS

    return RunStatus.FAILURE


def search_tavily(ctx: "TickContext") -> RunStatus:
    """Execute Tavily search."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    queries = bb_get(bb, "researcher_queries", [])

    try:
        from src.services.tavily_service import get_tavily_service

        tavily = get_tavily_service()
        if not tavily:
            return RunStatus.FAILURE

        # Run async search
        loop = asyncio.get_event_loop()
        search_responses = loop.run_until_complete(
            tavily.search_parallel(
                queries=queries,
                max_results_per_query=3,
                deduplicate=True,
            )
        )

        # Convert to standardized search results
        search_results = []
        for response in search_responses:
            for result in response.results:
                search_results.append({
                    "url": result.url,
                    "title": result.title,
                    "content": result.content[:500] if result.content else "",
                    "raw_content": result.raw_content or result.content,
                    "score": result.score,
                    "source_type": "web",
                })

        bb_set(bb, "search_results", search_results)
        bb_set(bb, "tool_calls_made", (bb_get(bb, "tool_calls_made", 0) or 0) + 1)

        logger.debug(f"Tavily search returned {len(search_results)} results")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return RunStatus.FAILURE


def search_openrouter(ctx: "TickContext") -> RunStatus:
    """Execute OpenRouter Perplexity search."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    queries = bb_get(bb, "researcher_queries", [])

    try:
        from src.services.openrouter_search import get_openrouter_search_service

        openrouter = get_openrouter_search_service()
        if not openrouter:
            return RunStatus.FAILURE

        # Run async search
        loop = asyncio.get_event_loop()
        search_responses = loop.run_until_complete(
            openrouter.search_parallel(
                queries=queries,
                max_results_per_query=3,
                deduplicate=True,
            )
        )

        # Convert to standardized search results
        search_results = []
        for response in search_responses:
            for result in response.results:
                search_results.append({
                    "url": result.url,
                    "title": result.title,
                    "content": result.content[:500] if result.content else "",
                    "raw_content": result.content,
                    "score": result.score,
                    "source_type": "web",
                })

        bb_set(bb, "search_results", search_results)
        bb_set(bb, "tool_calls_made", (bb_get(bb, "tool_calls_made", 0) or 0) + 1)

        logger.debug(f"OpenRouter search returned {len(search_results)} results")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    except Exception as e:
        logger.error(f"OpenRouter search failed: {e}")
        return RunStatus.FAILURE


def search_llm_fallback(ctx: "TickContext") -> RunStatus:
    """Use LLM knowledge as search fallback when no search provider available."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    subtopic = bb_get(bb, "subtopic", "")

    # Create a synthetic source from LLM
    search_results = [{
        "url": "llm://knowledge",
        "title": f"LLM Knowledge: {subtopic}",
        "content": f"Research based on LLM knowledge about: {subtopic}",
        "raw_content": None,
        "score": 0.5,
        "source_type": "llm",
    }]

    bb_set(bb, "search_results", search_results)
    logger.warning(f"Using LLM fallback for: {subtopic} (no search provider)")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def convert_search_results(ctx: "TickContext") -> RunStatus:
    """Convert raw search results to ResearchSource objects."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    search_results = bb_get(bb, "search_results", [])

    sources = []
    for i, result in enumerate(search_results, 1):
        sources.append({
            "id": i,
            "url": result.get("url", ""),
            "title": result.get("title", "Unknown"),
            "source_type": result.get("source_type", "web"),
            "relevance_score": result.get("score", 0.5),
            "content_summary": result.get("content", "")[:500],
            "key_quotes": [],
            "accessed_at": datetime.now(timezone.utc).isoformat(),
            "raw_content": result.get("raw_content"),
        })

    bb_set(bb, "researcher_sources", sources)
    logger.debug(f"Converted {len(sources)} search results to sources")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def researcher_has_sources(ctx: "TickContext") -> RunStatus:
    """Check if researcher has found sources."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    sources = bb_get(bb, "researcher_sources", [])

    if sources:
        ctx.mark_progress()
        return RunStatus.SUCCESS

    return RunStatus.FAILURE


def extract_findings(ctx: "TickContext") -> RunStatus:
    """Extract key quotes and findings from sources via LLM."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    subtopic = bb_get(bb, "subtopic", "")
    sources = bb_get(bb, "researcher_sources", [])

    if not sources:
        ctx.mark_progress()
        return RunStatus.SUCCESS

    try:
        # Build content for extraction
        sources_text = "\n\n".join([
            f"## Source [{s['id']}]: {s['title']}\nURL: {s['url']}\n\n{s['content_summary']}"
            for s in sources
        ])

        prompt = f"""Analyze these sources for the subtopic: "{subtopic}"

{sources_text}

For each source, identify:
1. Key quotes (exact text from the source)
2. Main findings relevant to the subtopic
3. Relevance score (0.0-1.0)

Respond in JSON format:
```json
{{
    "sources": [
        {{
            "id": 1,
            "key_quotes": ["quote1", "quote2"],
            "relevance": 0.8
        }}
    ]
}}
```"""

        from src.services.research.llm_service import ResearchLLMService
        user_id = bb_get(bb, "user_id", "")
        llm = ResearchLLMService(user_id=user_id)

        loop = asyncio.get_event_loop()
        extracted = loop.run_until_complete(
            llm.generate_json(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.2,
            )
        )

        # Update sources with extracted info
        source_map = {s["id"]: s for s in sources}
        for item in extracted.get("sources", []):
            source_id = item.get("id")
            if source_id in source_map:
                source = source_map[source_id]
                source["key_quotes"] = item.get("key_quotes", [])
                source["relevance_score"] = item.get("relevance", source["relevance_score"])

        bb_set(bb, "researcher_sources", sources)
        logger.debug(f"Extracted findings for {len(sources)} sources")

    except Exception as e:
        logger.warning(f"Failed to extract findings: {e}")
        # Continue with raw sources

    ctx.mark_progress()
    return RunStatus.SUCCESS


def mark_researcher_complete(ctx: "TickContext") -> RunStatus:
    """Update researcher state to completed."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    bb_set(bb, "researcher_completed", True)

    # Update the researcher in parent's researchers array
    researcher_index = bb_get(bb, "researcher_index", 0)
    researchers = bb_get(bb, "researchers", [])
    sources = bb_get(bb, "researcher_sources", [])

    if researcher_index < len(researchers):
        researchers[researcher_index]["completed"] = True
        researchers[researcher_index]["sources"] = sources
        bb_set(bb, "researchers", researchers)

    logger.debug(f"Researcher {researcher_index} completed with {len(sources)} sources")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def update_researcher_progress(ctx: "TickContext") -> RunStatus:
    """Update parent research progress based on researcher completion."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    researchers = bb_get(bb, "researchers", [])
    completed_count = sum(1 for r in researchers if r.get("completed"))
    total_count = len(researchers)

    if total_count > 0:
        # Calculate progress (20% to 50% range for research phase)
        progress = 20 + (30 * completed_count / total_count)
        bb_set(bb, "progress_pct", progress)

    # Update sources count
    total_sources = sum(len(r.get("sources", [])) for r in researchers)
    bb_set(bb, "sources_found", total_sources)

    ctx.mark_progress()
    return RunStatus.SUCCESS


def capture_researcher_error(ctx: "TickContext") -> RunStatus:
    """Capture error details for this researcher."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    # Get error from context if available
    error = ctx.last_error or "Unknown researcher error"
    bb_set(bb, "researcher_error", str(error))

    logger.warning(f"Researcher error captured: {error}")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def mark_researcher_failed(ctx: "TickContext") -> RunStatus:
    """Mark researcher as failed but completed (prevents retry loops)."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    bb_set(bb, "researcher_completed", True)

    researcher_index = bb_get(bb, "researcher_index", 0)
    researchers = bb_get(bb, "researchers", [])
    error = bb_get(bb, "researcher_error", "Unknown error")

    if researcher_index < len(researchers):
        researchers[researcher_index]["completed"] = True
        researchers[researcher_index]["error"] = error
        bb_set(bb, "researchers", researchers)

    logger.warning(f"Researcher {researcher_index} marked as failed: {error}")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def aggregate_sources(ctx: "TickContext") -> RunStatus:
    """Collect all sources from researchers into unified list."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    researchers = bb_get(bb, "researchers", [])

    all_sources = []
    seen_urls = set()
    source_id = 1

    for researcher in researchers:
        for source in researcher.get("sources", []):
            url = source.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                source["id"] = source_id
                all_sources.append(source)
                source_id += 1

    bb_set(bb, "sources", all_sources)
    bb_set(bb, "sources_found", len(all_sources))

    logger.info(f"Aggregated {len(all_sources)} unique sources from {len(researchers)} researchers")
    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Phase 5: Compression Actions
# =============================================================================


def has_sources(ctx: "TickContext") -> RunStatus:
    """Check if research has found any sources."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    sources = bb_get(bb, "sources", [])

    if sources:
        ctx.mark_progress()
        return RunStatus.SUCCESS

    return RunStatus.FAILURE


def compress_findings(ctx: "TickContext") -> RunStatus:
    """Synthesize findings from all sources via LLM."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    brief = bb_get(bb, "brief", {})
    sources = bb_get(bb, "sources", [])
    researchers = bb_get(bb, "researchers", [])

    if not sources:
        bb_set(bb, "findings", [])
        ctx.mark_progress()
        return RunStatus.SUCCESS

    try:
        # Build findings text
        sections = []
        for researcher in researchers:
            subtopic = researcher.get("subtopic", "Unknown")
            section = f"## Subtopic: {subtopic}\n\n"

            if researcher.get("error"):
                section += f"*Error: {researcher['error']}*\n"
            else:
                for source in researcher.get("sources", []):
                    section += f"### [{source['id']}] {source['title']}\n"
                    section += f"URL: {source['url']}\n"
                    section += f"Relevance: {source.get('relevance_score', 0):.2f}\n\n"
                    section += f"{source.get('content_summary', '')}\n\n"

                    key_quotes = source.get("key_quotes", [])
                    if key_quotes:
                        section += "Key quotes:\n"
                        for quote in key_quotes:
                            section += f"> {quote}\n"
                        section += "\n"

            sections.append(section)

        findings_text = "\n".join(sections)

        brief_text = f"""
Original Query: {brief.get('original_query', '')}
Refined Question: {brief.get('refined_question', '')}
Scope: {brief.get('scope', '')}
"""

        # Load prompt template
        from src.services.prompt_loader import PromptLoader
        prompts = PromptLoader()

        prompt = prompts.load(
            "research/compress.md",
            {
                "brief": brief_text,
                "findings": findings_text,
            }
        )

        # Generate compressed findings
        from src.services.research.llm_service import ResearchLLMService
        user_id = bb_get(bb, "user_id", "")
        llm = ResearchLLMService(user_id=user_id)

        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(
            llm.generate(
                prompt=prompt,
                max_tokens=5000,
                temperature=0.3,
            )
        )

        # Parse compressed findings
        findings = _parse_compressed_findings(response)
        bb_set(bb, "findings", findings)

        logger.info(f"Compressed into {len(findings)} findings")

    except Exception as e:
        logger.error(f"Failed to compress findings: {e}")
        # Create basic findings from sources
        findings = [
            {
                "claim": s.get("content_summary", "")[:200],
                "source_ids": [s.get("id", 0)],
                "confidence": s.get("relevance_score", 0.5),
            }
            for s in sources[:10]
        ]
        bb_set(bb, "findings", findings)

    ctx.mark_progress()
    return RunStatus.SUCCESS


def _parse_compressed_findings(response: str) -> List[Dict[str, Any]]:
    """Parse LLM response into findings list."""
    findings = []

    # Look for patterns like "- ... [sources: 1, 2]"
    pattern = re.compile(
        r"-\s*(.+?)\s*\[sources?:\s*([\d,\s]+)\]",
        re.IGNORECASE
    )

    for match in pattern.finditer(response):
        claim = match.group(1).strip()
        source_ids_str = match.group(2)

        source_ids = [
            int(s.strip())
            for s in source_ids_str.split(",")
            if s.strip().isdigit()
        ]

        if claim and source_ids:
            findings.append({
                "claim": claim,
                "source_ids": source_ids,
                "confidence": 0.8,
            })

    # If pattern matching failed, create basic findings
    if not findings:
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                claim = line.lstrip("-* ").strip()
                if len(claim) > 20:
                    findings.append({
                        "claim": claim[:500],
                        "source_ids": [1],
                        "confidence": 0.6,
                    })

    return findings[:20]


def create_empty_findings(ctx: "TickContext") -> RunStatus:
    """Create empty findings when no sources available."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    bb_set(bb, "findings", [])
    logger.info("Created empty findings (no sources)")

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Phase 6: Report Generation Actions
# =============================================================================


def generate_report(ctx: "TickContext") -> RunStatus:
    """Generate final research report with citations via LLM."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    brief = bb_get(bb, "brief", {})
    findings = bb_get(bb, "findings", [])
    sources = bb_get(bb, "sources", [])

    try:
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        brief_text = f"""
Original Query: {brief.get('original_query', '')}
Refined Question: {brief.get('refined_question', '')}
Scope: {brief.get('scope', '')}
Subtopics: {', '.join(brief.get('subtopics', []))}
"""

        # Format findings
        findings_lines = []
        for i, finding in enumerate(findings, 1):
            source_refs = ", ".join(str(s) for s in finding.get("source_ids", []))
            findings_lines.append(f"{i}. {finding.get('claim', '')} [sources: {source_refs}]")
        findings_text = "\n".join(findings_lines)

        # Format sources
        sources_lines = []
        for source in sources:
            sources_lines.append(f"[{source.get('id', 0)}] {source.get('title', '')}. {source.get('url', '')}")
        sources_text = "\n".join(sources_lines)

        # Load prompt template
        from src.services.prompt_loader import PromptLoader
        prompts = PromptLoader()

        prompt = prompts.load(
            "research/report.md",
            {
                "current_date": current_date,
                "brief": brief_text,
                "compressed_findings": findings_text,
                "sources": sources_text,
                "language": brief.get("language", "en"),
            }
        )

        # Generate report
        from src.services.research.llm_service import ResearchLLMService
        user_id = bb_get(bb, "user_id", "")
        llm = ResearchLLMService(user_id=user_id)

        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(
            llm.generate(
                prompt=prompt,
                max_tokens=15000,
                temperature=0.5,
            )
        )

        # Parse report
        report = _parse_report(response, brief, findings, sources)
        bb_set(bb, "report", report)

        logger.info(f"Generated report: {report.get('title', 'Unknown')}")

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        # Create minimal report
        query = brief.get("original_query", "Unknown")
        report = {
            "title": f"Research: {query[:50]}",
            "executive_summary": f"Research on: {query}\n\nGeneration failed: {e}",
            "sections": [],
            "recommendations": None,
            "limitations": None,
            "sources": sources,
            "comprehensiveness": 0,
            "analytical_depth": 0,
            "source_diversity": 0,
            "citation_density": 0,
        }
        bb_set(bb, "report", report)

    ctx.mark_progress()
    return RunStatus.SUCCESS


def _parse_report(
    response: str,
    brief: Dict[str, Any],
    findings: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Parse LLM response into report structure."""
    lines = response.split("\n")

    title = brief.get("original_query", "Research")[:100]
    executive_summary = ""
    sections = []
    recommendations = []
    limitations = []
    current_section = None

    in_executive = False
    in_recommendations = False
    in_limitations = False

    for line in lines:
        line_stripped = line.strip()

        # Parse title (H1)
        if line_stripped.startswith("# "):
            title = line_stripped[2:].strip()
            continue

        # Detect section headers
        if line_stripped.startswith("## "):
            heading = line_stripped[3:].strip().lower()

            if "executive" in heading or "summary" in heading:
                in_executive = True
                in_recommendations = False
                in_limitations = False
                current_section = None
            elif "recommendation" in heading:
                in_executive = False
                in_recommendations = True
                in_limitations = False
                current_section = None
            elif "limitation" in heading:
                in_executive = False
                in_recommendations = False
                in_limitations = True
                current_section = None
            elif "reference" in heading or "source" in heading:
                in_executive = False
                in_recommendations = False
                in_limitations = False
                current_section = None
            else:
                in_executive = False
                in_recommendations = False
                in_limitations = False
                current_section = {
                    "heading": line_stripped[3:].strip(),
                    "content": "",
                    "source_ids": [],
                }
                sections.append(current_section)
            continue

        # Accumulate content
        if in_executive:
            executive_summary += line + "\n"
        elif in_recommendations and line_stripped:
            if line_stripped.startswith(("-", "*")) or line_stripped[0].isdigit():
                rec = line_stripped.lstrip("-*0123456789. ").strip()
                if rec:
                    recommendations.append(rec)
        elif in_limitations and line_stripped:
            if line_stripped.startswith(("-", "*")) or line_stripped[0].isdigit():
                lim = line_stripped.lstrip("-*0123456789. ").strip()
                if lim:
                    limitations.append(lim)
        elif current_section is not None:
            current_section["content"] += line + "\n"

            # Extract source references
            for match in re.finditer(r"\[(\d+)\]", line):
                source_id = int(match.group(1))
                if source_id not in current_section["source_ids"]:
                    current_section["source_ids"].append(source_id)

    # Calculate quality metrics
    total_claims = sum(len(s.get("content", "").split(".")) for s in sections)
    cited_claims = sum(len(s.get("source_ids", [])) for s in sections)
    citation_density = cited_claims / max(total_claims, 1)

    source_types = set(s.get("source_type", "web") for s in sources)
    source_diversity = len(source_types) / 5.0

    return {
        "title": title,
        "executive_summary": executive_summary.strip(),
        "sections": sections,
        "recommendations": recommendations if recommendations else None,
        "limitations": limitations if limitations else None,
        "sources": sources,
        "comprehensiveness": min(len(sections) / 5.0, 1.0),
        "analytical_depth": min(len(findings) / 10.0, 1.0),
        "source_diversity": source_diversity,
        "citation_density": min(citation_density, 1.0),
    }


# =============================================================================
# Phase 7: Vault Persistence Actions
# =============================================================================


def should_persist(ctx: "TickContext") -> RunStatus:
    """Check if research should be persisted to vault."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    save_to_vault = bb_get(bb, "save_to_vault", True)
    vault_path = bb_get(bb, "vault_path")
    report = bb_get(bb, "report")

    if save_to_vault and vault_path and report:
        ctx.mark_progress()
        return RunStatus.SUCCESS

    return RunStatus.FAILURE


def persist_to_vault(ctx: "TickContext") -> RunStatus:
    """Save research artifacts to user vault."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    vault_path = bb_get(bb, "vault_path")
    research_id = bb_get(bb, "research_id")
    report = bb_get(bb, "report")
    brief = bb_get(bb, "brief", {})
    researchers = bb_get(bb, "researchers", [])
    started_at = bb_get(bb, "started_at")

    if not vault_path or not report:
        logger.warning("Cannot persist: missing vault_path or report")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    try:
        from src.services.research.vault_persister import ResearchVaultPersister

        # Build a minimal state-like object for the persister
        # The persister expects ResearchState, but we can use a dict
        from src.models.research import (
            ResearchState, ResearchRequest, ResearchDepth, ResearchStatus,
            ResearchBrief, ResearchReport, ResearchSource, ResearcherState,
        )

        # Convert dict sources to ResearchSource objects
        sources = [
            ResearchSource(
                id=s.get("id", 0),
                url=s.get("url", ""),
                title=s.get("title", "Unknown"),
                source_type=s.get("source_type", "web"),
                relevance_score=s.get("relevance_score", 0.5),
                content_summary=s.get("content_summary", ""),
                key_quotes=s.get("key_quotes", []),
            )
            for s in report.get("sources", [])
        ]

        # Convert report dict to ResearchReport
        report_obj = ResearchReport(
            title=report.get("title", "Research"),
            executive_summary=report.get("executive_summary", ""),
            sections=report.get("sections", []),
            recommendations=report.get("recommendations"),
            limitations=report.get("limitations"),
            sources=sources,
            comprehensiveness=report.get("comprehensiveness", 0),
            analytical_depth=report.get("analytical_depth", 0),
            source_diversity=report.get("source_diversity", 0),
            citation_density=report.get("citation_density", 0),
        )

        # Convert brief dict to ResearchBrief
        brief_obj = ResearchBrief(
            original_query=brief.get("original_query", ""),
            refined_question=brief.get("refined_question", ""),
            scope=brief.get("scope", ""),
            subtopics=brief.get("subtopics", []),
            constraints=brief.get("constraints"),
            language=brief.get("language", "en"),
        )

        # Create request
        query = brief.get("original_query", "")
        depth_str = bb_get(bb, "depth", "standard")
        depth = ResearchDepth(depth_str) if depth_str in ["quick", "standard", "thorough"] else ResearchDepth.STANDARD

        request = ResearchRequest(
            query=query,
            depth=depth,
            save_to_vault=True,
        )

        # Convert researcher dicts to ResearcherState
        researcher_states = [
            ResearcherState(
                subtopic=r.get("subtopic", ""),
                sources=[
                    ResearchSource(
                        id=s.get("id", 0),
                        url=s.get("url", ""),
                        title=s.get("title", "Unknown"),
                        source_type=s.get("source_type", "web"),
                        relevance_score=s.get("relevance_score", 0.5),
                        content_summary=s.get("content_summary", ""),
                        key_quotes=s.get("key_quotes", []),
                    )
                    for s in r.get("sources", [])
                ],
                completed=r.get("completed", False),
                error=r.get("error"),
            )
            for r in researchers
        ]

        # Create state
        state = ResearchState(
            research_id=research_id,
            user_id=bb_get(bb, "user_id", ""),
            request=request,
            status=ResearchStatus.COMPLETED,
            brief=brief_obj,
            researchers=researcher_states,
            all_sources=sources,
            report=report_obj,
            completed_at=datetime.now(timezone.utc),
        )

        # Persist
        persister = ResearchVaultPersister(vault_path)
        index_path = persister.persist(state)

        bb_set(bb, "output_vault_path", str(index_path))
        logger.info(f"Research persisted to: {index_path}")

    except Exception as e:
        logger.error(f"Failed to persist research: {e}")
        # Don't fail - persistence is optional

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Phase 8: Finalization Actions
# =============================================================================


def emit_complete_event(ctx: "TickContext") -> RunStatus:
    """Emit research.complete event to ANS."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    research_id = bb_get(bb, "research_id")
    sources_found = bb_get(bb, "sources_found", 0)

    try:
        from src.services.ans.bus import get_event_bus
        from src.services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="research.complete",
            source="research_bt",
            severity=Severity.INFO,
            payload={
                "research_id": research_id,
                "sources_found": sources_found,
            }
        ))
    except Exception as e:
        logger.warning(f"Failed to emit research.complete event: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def finalize(ctx: "TickContext") -> RunStatus:
    """Set output status and cleanup."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    bb_set(bb, "status", "completed")
    bb_set(bb, "completed_at", datetime.now(timezone.utc).isoformat())

    # Final progress chunk
    _add_pending_chunk(bb, {
        "type": "research_complete",
        "research_id": bb_get(bb, "research_id"),
        "status": "completed",
        "sources_found": bb_get(bb, "sources_found", 0),
        "vault_path": bb_get(bb, "output_vault_path"),
    })

    logger.info(f"Research finalized: {bb_get(bb, 'research_id')}")
    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Utility Actions
# =============================================================================


def noop(ctx: "TickContext") -> RunStatus:
    """No-operation action for control flow."""
    ctx.mark_progress()
    return RunStatus.SUCCESS


def _add_pending_chunk(bb: Any, chunk: Dict[str, Any]) -> None:
    """Add a chunk to the pending chunks list for streaming."""
    chunks = bb_get(bb, "_pending_chunks") or []
    chunks.append(chunk)
    bb_set(bb, "_pending_chunks", chunks)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Initialization
    "init_research",
    "emit_start_event",
    # Brief generation
    "set_phase",
    "generate_brief",
    "validate_brief",
    "create_fallback_brief",
    # Planning
    "plan_subtopics",
    # Research
    "init_researcher",
    "emit_researcher_event",
    "generate_search_queries",
    "has_tavily",
    "has_openrouter_search",
    "search_tavily",
    "search_openrouter",
    "search_llm_fallback",
    "convert_search_results",
    "researcher_has_sources",
    "extract_findings",
    "mark_researcher_complete",
    "update_researcher_progress",
    "capture_researcher_error",
    "mark_researcher_failed",
    "aggregate_sources",
    # Compression
    "has_sources",
    "compress_findings",
    "create_empty_findings",
    # Report
    "generate_report",
    # Persistence
    "should_persist",
    "persist_to_vault",
    # Finalization
    "emit_complete_event",
    "finalize",
    # Utility
    "noop",
]
