"""Behavior tree nodes for Deep Research orchestration.

Each behavior represents a step in the research workflow:
1. GenerateBriefBehavior - Transform query into ResearchBrief
2. PlanSubtopicsBehavior - Extract subtopics from brief
3. ResearcherBehavior - Single researcher for one subtopic
4. ParallelResearchersBehavior - Run multiple researchers in parallel
5. CompressFindingsBehavior - Synthesize all findings
6. GenerateReportBehavior - Create final report
7. PersistToVaultBehavior - Save to vault

All behaviors follow the async pattern and update ResearchState in place.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncGenerator, List, Optional

from ...models.research import (
    ResearchBrief,
    ResearchConfig,
    ResearchFinding,
    ResearchProgress,
    ResearchReport,
    ResearchSource,
    ResearchState,
    ResearchStatus,
    ResearcherState,
    SourceType,
)
from ..openrouter_search import OpenRouterSearchService, get_openrouter_search_service
from ..prompt_loader import PromptLoader
from ..tavily_service import TavilySearchService, get_tavily_service
from .llm_service import ResearchLLMService
from .vault_persister import ResearchVaultPersister
from typing import Literal

SearchProvider = Literal["tavily", "openrouter", "none"]

logger = logging.getLogger(__name__)


class ResearchBehavior(ABC):
    """Base class for research behavior nodes."""

    @abstractmethod
    async def run(
        self,
        state: ResearchState,
    ) -> ResearchState:
        """Execute the behavior and update state.

        Args:
            state: Current research state

        Returns:
            Updated research state
        """
        pass

    def get_progress_message(self) -> str:
        """Get a message describing what this behavior is doing."""
        return self.__class__.__name__


class GenerateBriefBehavior(ResearchBehavior):
    """Generate a research brief from the user query.

    Uses the brief.md prompt to transform a natural language query
    into a structured ResearchBrief with refined question, scope,
    and subtopics.
    """

    def __init__(
        self,
        llm_service: ResearchLLMService,
        prompt_loader: Optional[PromptLoader] = None,
    ):
        """Initialize the behavior.

        Args:
            llm_service: LLM service for generation
            prompt_loader: Optional prompt loader (creates default if not provided)
        """
        self.llm = llm_service
        self.prompts = prompt_loader or PromptLoader()

    async def run(self, state: ResearchState) -> ResearchState:
        """Generate brief and update state."""
        logger.info(f"Generating research brief for: {state.request.query[:100]}")

        try:
            # Load and render the brief prompt
            current_date = datetime.utcnow().strftime("%Y-%m-%d")

            prompt = self.prompts.load(
                "research/brief.md",
                {
                    "query": state.request.query,
                    "current_date": current_date,
                }
            )

            # Generate structured brief using JSON mode
            brief_data = await self.llm.generate_json(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3,
            )

            # Parse into ResearchBrief model
            state.brief = ResearchBrief(
                original_query=state.request.query,
                refined_question=brief_data.get("refined_question", state.request.query),
                scope=brief_data.get("scope", "General research"),
                subtopics=brief_data.get("subtopics", [state.request.query]),
                constraints=brief_data.get("constraints"),
                language=brief_data.get("language", "en"),
            )

            state.status = ResearchStatus.PLANNING
            logger.info(
                f"Generated brief with {len(state.brief.subtopics)} subtopics",
                extra={"subtopics": state.brief.subtopics}
            )

        except Exception as e:
            logger.error(f"Failed to generate brief: {e}")
            state.status = ResearchStatus.FAILED
            # Create minimal brief to allow continuation
            state.brief = ResearchBrief(
                original_query=state.request.query,
                refined_question=state.request.query,
                scope="General research",
                subtopics=[state.request.query],
                language="en",
            )

        return state

    def get_progress_message(self) -> str:
        return "Generating research brief..."


class PlanSubtopicsBehavior(ResearchBehavior):
    """Plan the research by creating researcher states for each subtopic.

    Takes the subtopics from the brief and creates ResearcherState
    instances to be processed by parallel researchers.
    """

    def __init__(self, config: Optional[ResearchConfig] = None):
        """Initialize the behavior.

        Args:
            config: Research configuration (uses defaults if not provided)
        """
        self.config = config

    async def run(self, state: ResearchState) -> ResearchState:
        """Create researcher states for each subtopic."""
        if not state.brief:
            logger.error("No brief available for planning")
            state.status = ResearchStatus.FAILED
            return state

        # Get config based on research depth
        config = self.config or ResearchConfig.for_depth(state.request.depth)

        # Limit subtopics to max concurrent researchers
        subtopics = state.brief.subtopics[:config.max_concurrent_researchers]

        logger.info(
            f"Planning research with {len(subtopics)} subtopics",
            extra={"max_concurrent": config.max_concurrent_researchers}
        )

        # Create researcher state for each subtopic
        state.researchers = [
            ResearcherState(
                subtopic=subtopic,
                max_tool_calls=config.max_tool_calls_per_researcher,
            )
            for subtopic in subtopics
        ]

        state.status = ResearchStatus.RESEARCHING
        return state

    def get_progress_message(self) -> str:
        return "Planning research subtopics..."


class ResearcherBehavior(ResearchBehavior):
    """A single researcher that investigates one subtopic.

    Uses Tavily or OpenRouter (Perplexity) search to find sources and extracts
    relevant information. Each researcher operates independently and can be run
    in parallel.
    """

    def __init__(
        self,
        llm_service: ResearchLLMService,
        tavily_service: Optional[TavilySearchService] = None,
        openrouter_search: Optional[OpenRouterSearchService] = None,
        search_provider: SearchProvider = "none",
        prompt_loader: Optional[PromptLoader] = None,
        tavily_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ):
        """Initialize the researcher.

        Args:
            llm_service: LLM service for content extraction
            tavily_service: Optional Tavily search service
            openrouter_search: Optional OpenRouter search service
            search_provider: Which search provider to use
            prompt_loader: Optional prompt loader
            tavily_api_key: Optional Tavily API key
            openrouter_api_key: Optional OpenRouter API key
        """
        self.llm = llm_service
        self.search_provider = search_provider
        self.prompts = prompt_loader or PromptLoader()

        # Initialize the appropriate search service
        if search_provider == "tavily":
            self.tavily = tavily_service or get_tavily_service(tavily_api_key)
            self.openrouter_search = None
        elif search_provider == "openrouter":
            self.tavily = None
            self.openrouter_search = openrouter_search or get_openrouter_search_service(openrouter_api_key)
        else:
            self.tavily = None
            self.openrouter_search = None

    async def run_single(
        self,
        state: ResearchState,
        researcher: ResearcherState,
    ) -> ResearcherState:
        """Run research for a single subtopic.

        Args:
            state: Overall research state (for context)
            researcher: The specific researcher state to run

        Returns:
            Updated researcher state with sources
        """
        logger.info(f"Researching subtopic: {researcher.subtopic} via {self.search_provider}")

        try:
            # Generate search queries based on subtopic and brief
            queries = await self._generate_search_queries(state, researcher)

            # Execute searches using the appropriate provider
            if self.search_provider == "tavily" and self.tavily:
                search_responses = await self.tavily.search_parallel(
                    queries=queries,
                    max_results_per_query=3,
                    deduplicate=True,
                )
                # Convert Tavily results to ResearchSource objects
                source_id = 1
                for response in search_responses:
                    for result in response.results:
                        source = ResearchSource(
                            id=source_id,
                            url=result.url,
                            title=result.title,
                            source_type=SourceType.WEB,
                            relevance_score=result.score,
                            content_summary=result.content[:500] if result.content else "",
                            raw_content=result.raw_content or result.content,
                        )
                        researcher.sources.append(source)
                        source_id += 1

            elif self.search_provider == "openrouter" and self.openrouter_search:
                search_responses = await self.openrouter_search.search_parallel(
                    queries=queries,
                    max_results_per_query=3,
                    deduplicate=True,
                )
                # Convert OpenRouter results to ResearchSource objects
                source_id = 1
                for response in search_responses:
                    for result in response.results:
                        source = ResearchSource(
                            id=source_id,
                            url=result.url,
                            title=result.title,
                            source_type=SourceType.WEB,
                            relevance_score=result.score,
                            content_summary=result.content[:500] if result.content else "",
                            raw_content=result.content,
                        )
                        researcher.sources.append(source)
                        source_id += 1

            else:
                logger.warning(f"No search provider configured for subtopic: {researcher.subtopic}")
                researcher.error = "No search provider configured"

            # Extract key information from sources
            if researcher.sources:
                await self._extract_findings(state, researcher)

            researcher.completed = True
            logger.info(
                f"Completed research for: {researcher.subtopic}",
                extra={"sources_found": len(researcher.sources), "provider": self.search_provider}
            )

        except Exception as e:
            logger.error(f"Research failed for {researcher.subtopic}: {e}", exc_info=True)
            researcher.error = str(e)
            researcher.completed = True  # Mark as done even on failure

        return researcher

    async def _generate_search_queries(
        self,
        state: ResearchState,
        researcher: ResearcherState,
    ) -> List[str]:
        """Generate search queries for a subtopic.

        Uses the LLM to generate targeted search queries based on
        the subtopic and research context.
        """
        # Simple query generation - can be enhanced with LLM
        base_query = researcher.subtopic
        current_year = datetime.utcnow().year

        # Generate multiple query variants
        queries = [
            base_query,
            f"{base_query} {current_year}",
            f"{base_query} latest research",
        ]

        # If we have constraints, add them
        if state.brief and state.brief.constraints:
            queries.append(f"{base_query} {state.brief.constraints}")

        return queries[:4]  # Limit to 4 queries

    async def _extract_findings(
        self,
        state: ResearchState,
        researcher: ResearcherState,
    ) -> None:
        """Extract key findings from gathered sources.

        Uses the LLM to analyze sources and extract key quotes
        and findings relevant to the subtopic.
        """
        if not researcher.sources:
            return

        # Build content for extraction
        sources_text = "\n\n".join([
            f"## Source [{s.id}]: {s.title}\nURL: {s.url}\n\n{s.content_summary}"
            for s in researcher.sources
        ])

        prompt = f"""Analyze these sources for the subtopic: "{researcher.subtopic}"

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

        try:
            extracted = await self.llm.generate_json(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.2,
            )

            # Update sources with extracted info
            source_map = {s.id: s for s in researcher.sources}
            for item in extracted.get("sources", []):
                source_id = item.get("id")
                if source_id in source_map:
                    source = source_map[source_id]
                    source.key_quotes = item.get("key_quotes", [])
                    source.relevance_score = item.get("relevance", source.relevance_score)

        except Exception as e:
            logger.warning(f"Failed to extract findings: {e}")
            # Continue with raw sources

    async def run(self, state: ResearchState) -> ResearchState:
        """Run all researchers sequentially (use ParallelResearchersBehavior for parallel)."""
        for researcher in state.researchers:
            if not researcher.completed:
                await self.run_single(state, researcher)

                # Add sources to overall state
                for source in researcher.sources:
                    state.add_source(source)

        return state

    def get_progress_message(self) -> str:
        return "Researching subtopics..."


class ParallelResearchersBehavior(ResearchBehavior):
    """Run multiple researchers in parallel.

    Coordinates parallel execution of ResearcherBehavior instances,
    one for each subtopic in the research plan.
    """

    def __init__(
        self,
        llm_service: ResearchLLMService,
        tavily_service: Optional[TavilySearchService] = None,
        openrouter_search: Optional[OpenRouterSearchService] = None,
        search_provider: SearchProvider = "none",
        prompt_loader: Optional[PromptLoader] = None,
        tavily_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        max_concurrent: int = 5,
    ):
        """Initialize parallel researchers.

        Args:
            llm_service: LLM service for research
            tavily_service: Optional Tavily service
            openrouter_search: Optional OpenRouter search service
            search_provider: Which search provider to use
            prompt_loader: Optional prompt loader
            tavily_api_key: Optional Tavily API key
            openrouter_api_key: Optional OpenRouter API key
            max_concurrent: Maximum concurrent researchers
        """
        self.researcher = ResearcherBehavior(
            llm_service=llm_service,
            tavily_service=tavily_service,
            openrouter_search=openrouter_search,
            search_provider=search_provider,
            prompt_loader=prompt_loader,
            tavily_api_key=tavily_api_key,
            openrouter_api_key=openrouter_api_key,
        )
        self.max_concurrent = max_concurrent
        self.search_provider = search_provider

    async def run(self, state: ResearchState) -> ResearchState:
        """Run all researchers in parallel."""
        if not state.researchers:
            logger.warning("No researchers to run")
            return state

        logger.info(
            f"Starting {len(state.researchers)} parallel researchers",
            extra={"max_concurrent": self.max_concurrent}
        )

        # Create tasks for all researchers
        tasks = [
            self.researcher.run_single(state, researcher)
            for researcher in state.researchers
            if not researcher.completed
        ]

        # Run in parallel with semaphore for limiting concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_with_limit(task):
            async with semaphore:
                return await task

        # Execute all tasks
        results = await asyncio.gather(
            *[run_with_limit(t) for t in tasks],
            return_exceptions=True,
        )

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Researcher task failed: {result}")
            elif isinstance(result, ResearcherState):
                # Add sources to overall state
                for source in result.sources:
                    state.add_source(source)

        # Update total searches count
        state.total_searches = sum(
            len(r.sources) for r in state.researchers
        )

        logger.info(
            f"Parallel research completed",
            extra={
                "total_sources": len(state.all_sources),
                "total_searches": state.total_searches,
            }
        )

        return state

    def get_progress_message(self) -> str:
        return "Running parallel research..."


class CompressFindingsBehavior(ResearchBehavior):
    """Compress and synthesize findings from all researchers.

    Takes the raw findings from all researchers and synthesizes
    them into a coherent set of compressed findings, removing
    duplicates and organizing by theme.
    """

    def __init__(
        self,
        llm_service: ResearchLLMService,
        prompt_loader: Optional[PromptLoader] = None,
    ):
        """Initialize the compression behavior.

        Args:
            llm_service: LLM service for synthesis
            prompt_loader: Optional prompt loader
        """
        self.llm = llm_service
        self.prompts = prompt_loader or PromptLoader()

    async def run(self, state: ResearchState) -> ResearchState:
        """Compress findings from all researchers."""
        logger.info(f"Compressing findings from {len(state.researchers)} researchers")

        state.status = ResearchStatus.COMPRESSING

        if not state.researchers or not state.all_sources:
            logger.warning("No findings to compress")
            return state

        try:
            # Build findings summary from researchers
            findings_text = self._build_findings_text(state)

            # Build brief summary
            brief_text = ""
            if state.brief:
                brief_text = f"""
Original Query: {state.brief.original_query}
Refined Question: {state.brief.refined_question}
Scope: {state.brief.scope}
"""

            # Load and render compression prompt
            prompt = self.prompts.load(
                "research/compress.md",
                {
                    "brief": brief_text,
                    "findings": findings_text,
                }
            )

            # Generate compressed findings
            response = await self.llm.generate(
                prompt=prompt,
                max_tokens=5000,
                temperature=0.3,
            )

            # Parse compressed findings into structured format
            state.compressed_findings = self._parse_compressed_findings(response)

            logger.info(
                f"Compressed into {len(state.compressed_findings)} findings",
            )

        except Exception as e:
            logger.error(f"Failed to compress findings: {e}")
            # Create basic findings from sources
            state.compressed_findings = [
                ResearchFinding(
                    claim=s.content_summary[:200],
                    source_ids=[s.id],
                    confidence=s.relevance_score,
                )
                for s in state.all_sources[:10]
            ]

        return state

    def _build_findings_text(self, state: ResearchState) -> str:
        """Build text representation of all findings."""
        sections = []

        for researcher in state.researchers:
            section = f"## Subtopic: {researcher.subtopic}\n\n"

            if researcher.error:
                section += f"*Error: {researcher.error}*\n"
            else:
                for source in researcher.sources:
                    section += f"### [{source.id}] {source.title}\n"
                    section += f"URL: {source.url}\n"
                    section += f"Relevance: {source.relevance_score:.2f}\n\n"
                    section += f"{source.content_summary}\n\n"

                    if source.key_quotes:
                        section += "Key quotes:\n"
                        for quote in source.key_quotes:
                            section += f"> {quote}\n"
                        section += "\n"

            sections.append(section)

        return "\n".join(sections)

    def _parse_compressed_findings(self, response: str) -> List[ResearchFinding]:
        """Parse LLM response into ResearchFinding objects."""
        findings = []

        # Look for patterns like "Finding: ... [sources: 1, 2]"
        finding_pattern = re.compile(
            r"-\s*(.+?)\s*\[sources?:\s*([\d,\s]+)\]",
            re.IGNORECASE
        )

        for match in finding_pattern.finditer(response):
            claim = match.group(1).strip()
            source_ids_str = match.group(2)

            # Parse source IDs
            source_ids = [
                int(s.strip())
                for s in source_ids_str.split(",")
                if s.strip().isdigit()
            ]

            if claim and source_ids:
                findings.append(
                    ResearchFinding(
                        claim=claim,
                        source_ids=source_ids,
                        confidence=0.8,  # Default confidence
                    )
                )

        # If pattern matching failed, create basic findings
        if not findings:
            lines = response.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("-") or line.startswith("*"):
                    claim = line.lstrip("-* ").strip()
                    if len(claim) > 20:  # Minimum claim length
                        findings.append(
                            ResearchFinding(
                                claim=claim[:500],
                                source_ids=[1],  # Default source
                                confidence=0.6,
                            )
                        )

        return findings[:20]  # Limit to 20 findings

    def get_progress_message(self) -> str:
        return "Synthesizing findings..."


class GenerateReportBehavior(ResearchBehavior):
    """Generate the final research report.

    Takes compressed findings and produces a well-structured
    markdown report with citations and quality metrics.
    """

    def __init__(
        self,
        llm_service: ResearchLLMService,
        prompt_loader: Optional[PromptLoader] = None,
    ):
        """Initialize the report generator.

        Args:
            llm_service: LLM service for generation
            prompt_loader: Optional prompt loader
        """
        self.llm = llm_service
        self.prompts = prompt_loader or PromptLoader()

    async def run(self, state: ResearchState) -> ResearchState:
        """Generate the final research report."""
        logger.info("Generating research report")

        state.status = ResearchStatus.GENERATING

        if not state.brief:
            logger.error("No brief available for report generation")
            state.status = ResearchStatus.FAILED
            return state

        try:
            # Build context for report generation
            current_date = datetime.utcnow().strftime("%Y-%m-%d")

            brief_text = f"""
Original Query: {state.brief.original_query}
Refined Question: {state.brief.refined_question}
Scope: {state.brief.scope}
Subtopics: {', '.join(state.brief.subtopics)}
"""

            findings_text = self._format_findings(state)
            sources_text = self._format_sources(state)

            # Load and render report prompt
            prompt = self.prompts.load(
                "research/report.md",
                {
                    "current_date": current_date,
                    "brief": brief_text,
                    "compressed_findings": findings_text,
                    "sources": sources_text,
                    "language": state.brief.language,
                }
            )

            # Generate report
            response = await self.llm.generate(
                prompt=prompt,
                max_tokens=15000,
                temperature=0.5,
            )

            # Parse report into structured format
            state.report = self._parse_report(response, state)

            state.status = ResearchStatus.COMPLETED
            state.completed_at = datetime.utcnow()

            logger.info(
                "Report generation completed",
                extra={
                    "title": state.report.title if state.report else "Unknown",
                    "sections": len(state.report.sections) if state.report else 0,
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            state.status = ResearchStatus.FAILED
            # Create minimal report
            state.report = ResearchReport(
                title=f"Research: {state.request.query[:50]}",
                executive_summary=f"Research on: {state.request.query}\n\nGeneration failed: {e}",
                sections=[],
                sources=state.all_sources,
            )

        return state

    def _format_findings(self, state: ResearchState) -> str:
        """Format compressed findings for the report prompt."""
        if not state.compressed_findings:
            return "No findings available."

        lines = []
        for i, finding in enumerate(state.compressed_findings, 1):
            source_refs = ", ".join(str(s) for s in finding.source_ids)
            lines.append(f"{i}. {finding.claim} [sources: {source_refs}]")

        return "\n".join(lines)

    def _format_sources(self, state: ResearchState) -> str:
        """Format sources for the report prompt."""
        if not state.all_sources:
            return "No sources available."

        lines = []
        for source in state.all_sources:
            lines.append(
                f"[{source.id}] {source.title}. {source.url}"
            )

        return "\n".join(lines)

    def _parse_report(
        self,
        response: str,
        state: ResearchState,
    ) -> ResearchReport:
        """Parse LLM response into ResearchReport."""
        lines = response.split("\n")

        title = state.request.query[:100]
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
            if line_stripped.startswith("# ") and not title:
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
                    # Skip references section (we have structured sources)
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
                if line_stripped.startswith(("-", "*", "1", "2", "3")):
                    rec = line_stripped.lstrip("-*0123456789. ").strip()
                    if rec:
                        recommendations.append(rec)
            elif in_limitations and line_stripped:
                if line_stripped.startswith(("-", "*", "1", "2", "3")):
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

        source_types = set(s.source_type for s in state.all_sources)
        source_diversity = len(source_types) / 5.0  # 5 possible types

        return ResearchReport(
            title=title,
            executive_summary=executive_summary.strip(),
            sections=sections,
            recommendations=recommendations if recommendations else None,
            limitations=limitations if limitations else None,
            sources=state.all_sources,
            comprehensiveness=min(len(sections) / 5.0, 1.0),
            analytical_depth=min(len(state.compressed_findings) / 10.0, 1.0),
            source_diversity=source_diversity,
            citation_density=min(citation_density, 1.0),
        )

    def get_progress_message(self) -> str:
        return "Generating report..."


class PersistToVaultBehavior(ResearchBehavior):
    """Persist research results to the vault.

    Saves the complete research project to the user's vault
    including index, brief, report, sources, and methodology.
    """

    def __init__(self, vault_path: str):
        """Initialize the persistence behavior.

        Args:
            vault_path: Path to the user's vault directory
        """
        self.vault_path = vault_path

    async def run(self, state: ResearchState) -> ResearchState:
        """Persist research to vault."""
        if not state.request.save_to_vault:
            logger.info("Vault persistence disabled for this research")
            return state

        if not state.report:
            logger.error("No report to persist")
            return state

        logger.info(f"Persisting research to vault: {self.vault_path}")

        try:
            persister = ResearchVaultPersister(self.vault_path)
            index_path = persister.persist(state)

            state.vault_folder = str(index_path)
            logger.info(f"Research persisted to: {index_path}")

        except Exception as e:
            logger.error(f"Failed to persist research: {e}")
            # Don't fail the research - persistence is optional

        return state

    def get_progress_message(self) -> str:
        return "Saving to vault..."


__all__ = [
    "ResearchBehavior",
    "GenerateBriefBehavior",
    "PlanSubtopicsBehavior",
    "ResearcherBehavior",
    "ParallelResearchersBehavior",
    "CompressFindingsBehavior",
    "GenerateReportBehavior",
    "PersistToVaultBehavior",
]
