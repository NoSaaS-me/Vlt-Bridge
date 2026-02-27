"""Research Orchestrator for Deep Research feature.

Coordinates the entire research workflow using behavior tree pattern:
1. Generate Brief - Transform query into structured research plan
2. Plan Subtopics - Create researcher assignments
3. Parallel Research - Execute multiple researchers concurrently
4. Compress Findings - Synthesize and deduplicate findings
5. Generate Report - Create final cited report
6. Persist to Vault - Save research artifacts

The orchestrator provides both synchronous and streaming interfaces
for integration with Oracle and the frontend.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional

from ...models.research import (
    ResearchConfig,
    ResearchProgress,
    ResearchRequest,
    ResearchState,
    ResearchStatus,
)
from ..openrouter_search import OpenRouterSearchService, get_openrouter_search_service
from ..prompt_loader import PromptLoader
from ..tavily_service import TavilySearchService, get_tavily_service
from ..user_settings import UserSettingsService
from .behaviors import (
    CompressFindingsBehavior,
    GenerateBriefBehavior,
    GenerateReportBehavior,
    ParallelResearchersBehavior,
    PersistToVaultBehavior,
    PlanSubtopicsBehavior,
)
from .llm_service import ResearchLLMService
from typing import Literal

SearchProvider = Literal["tavily", "openrouter", "none"]

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Orchestrates the deep research workflow.

    The orchestrator manages the execution of research behaviors in sequence,
    handling state transitions and progress updates. It supports both blocking
    and streaming modes for flexibility in integration.

    Example:
        ```python
        orchestrator = ResearchOrchestrator(
            user_id="user-123",
            vault_path="/path/to/vault",
        )

        # Blocking mode
        state = await orchestrator.run_research(request)
        print(state.report.executive_summary)

        # Streaming mode
        async for progress in orchestrator.run_research_streaming(request):
            print(f"Progress: {progress.progress_pct}%")
        ```
    """

    def __init__(
        self,
        user_id: str,
        vault_path: Optional[str] = None,
        user_settings: Optional[UserSettingsService] = None,
        search_provider: SearchProvider = "none",
        tavily_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        config: Optional[ResearchConfig] = None,
    ):
        """Initialize the research orchestrator.

        Args:
            user_id: User identifier for settings and attribution
            vault_path: Path to user's vault (None to skip persistence)
            user_settings: Optional user settings service
            search_provider: Which search provider to use ('tavily', 'openrouter', 'none')
            tavily_api_key: Optional Tavily API key (required if provider is 'tavily')
            openrouter_api_key: Optional OpenRouter API key (required if provider is 'openrouter')
            config: Optional research configuration override
        """
        self.user_id = user_id
        self.vault_path = vault_path
        self.user_settings = user_settings or UserSettingsService()
        self.search_provider = search_provider
        self.tavily_api_key = tavily_api_key
        self.openrouter_api_key = openrouter_api_key
        self.config = config

        # Services will be initialized per-research to allow fresh configuration
        self._llm_service: Optional[ResearchLLMService] = None
        self._tavily_service: Optional[TavilySearchService] = None
        self._openrouter_search: Optional[OpenRouterSearchService] = None
        self._prompt_loader: Optional[PromptLoader] = None

    def _init_services(self) -> None:
        """Initialize or reinitialize services."""
        self._llm_service = ResearchLLMService(
            user_id=self.user_id,
            user_settings=self.user_settings,
        )

        # Initialize the appropriate search service based on provider
        if self.search_provider == "tavily":
            self._tavily_service = get_tavily_service(self.tavily_api_key)
            self._openrouter_search = None
            logger.info("Using Tavily for web search")
        elif self.search_provider == "openrouter":
            self._tavily_service = None
            self._openrouter_search = get_openrouter_search_service(self.openrouter_api_key)
            logger.info("Using OpenRouter (Perplexity) for web search")
        else:
            self._tavily_service = None
            self._openrouter_search = None
            logger.warning("No search provider configured - research will use LLM knowledge only")

        self._prompt_loader = PromptLoader()

    def _create_initial_state(
        self,
        request: ResearchRequest,
    ) -> ResearchState:
        """Create initial research state from request.

        Args:
            request: The research request

        Returns:
            Initialized ResearchState
        """
        research_id = self._generate_research_id(request)

        return ResearchState(
            research_id=research_id,
            user_id=self.user_id,
            request=request,
            status=ResearchStatus.PLANNING,
        )

    def _generate_research_id(self, request: ResearchRequest) -> str:
        """Generate a unique research ID.

        Creates a human-readable ID based on the query and timestamp.

        Args:
            request: The research request

        Returns:
            Unique research identifier
        """
        # Extract first few words from query for readability
        words = request.query.lower().split()[:3]
        slug = "-".join(
            "".join(c for c in word if c.isalnum())
            for word in words
        )[:30]

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M")
        unique = uuid.uuid4().hex[:6]

        return f"{timestamp}-{slug}-{unique}"

    async def run_research(
        self,
        request: ResearchRequest,
    ) -> ResearchState:
        """Run the complete research workflow.

        Executes all research behaviors in sequence and returns
        the final state with completed report.

        Args:
            request: The research request

        Returns:
            Final ResearchState with report

        Raises:
            Exception: If research fails critically
        """
        logger.info(
            f"Starting research: {request.query[:100]}",
            extra={"depth": request.depth.value}
        )

        # Initialize services
        self._init_services()

        # Create initial state
        state = self._create_initial_state(request)

        # Get config for this depth
        config = self.config or ResearchConfig.for_depth(request.depth)

        try:
            # 1. Generate Brief
            brief_behavior = GenerateBriefBehavior(
                llm_service=self._llm_service,
                prompt_loader=self._prompt_loader,
            )
            state = await brief_behavior.run(state)

            if state.status == ResearchStatus.FAILED:
                logger.error("Research failed at brief generation")
                return state

            # 2. Plan Subtopics
            plan_behavior = PlanSubtopicsBehavior(config=config)
            state = await plan_behavior.run(state)

            if state.status == ResearchStatus.FAILED:
                logger.error("Research failed at planning")
                return state

            # 3. Parallel Research
            research_behavior = ParallelResearchersBehavior(
                llm_service=self._llm_service,
                tavily_service=self._tavily_service,
                openrouter_search=self._openrouter_search,
                search_provider=self.search_provider,
                prompt_loader=self._prompt_loader,
                max_concurrent=config.max_concurrent_researchers,
            )
            state = await research_behavior.run(state)

            # 4. Compress Findings
            compress_behavior = CompressFindingsBehavior(
                llm_service=self._llm_service,
                prompt_loader=self._prompt_loader,
            )
            state = await compress_behavior.run(state)

            # 5. Generate Report
            report_behavior = GenerateReportBehavior(
                llm_service=self._llm_service,
                prompt_loader=self._prompt_loader,
            )
            state = await report_behavior.run(state)

            # 6. Persist to Vault (if configured)
            if self.vault_path and request.save_to_vault:
                persist_behavior = PersistToVaultBehavior(
                    vault_path=self.vault_path,
                )
                state = await persist_behavior.run(state)

            logger.info(
                f"Research completed: {state.research_id}",
                extra={
                    "status": state.status.value,
                    "sources": len(state.all_sources),
                    "findings": len(state.compressed_findings),
                }
            )

        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
            state.status = ResearchStatus.FAILED

        return state

    async def run_research_streaming(
        self,
        request: ResearchRequest,
    ) -> AsyncGenerator[ResearchProgress, None]:
        """Run research with progress streaming.

        Yields progress updates as the research proceeds through
        each phase of the workflow.

        Args:
            request: The research request

        Yields:
            ResearchProgress updates at each phase

        Returns:
            Final state is available via the last progress.research_id
        """
        logger.info(
            f"Starting streaming research: {request.query[:100]}",
            extra={"depth": request.depth.value}
        )

        # Initialize services
        self._init_services()

        # Create initial state
        state = self._create_initial_state(request)

        # Get config for this depth
        config = self.config or ResearchConfig.for_depth(request.depth)

        # Yield initial progress
        yield ResearchProgress(
            research_id=state.research_id,
            status=state.status,
            phase="initializing",
            progress_pct=0,
            message="Starting research...",
        )

        try:
            # 1. Generate Brief
            yield ResearchProgress(
                research_id=state.research_id,
                status=ResearchStatus.PLANNING,
                phase="brief",
                progress_pct=5,
                message="Generating research brief...",
            )

            brief_behavior = GenerateBriefBehavior(
                llm_service=self._llm_service,
                prompt_loader=self._prompt_loader,
            )
            state = await brief_behavior.run(state)

            if state.status == ResearchStatus.FAILED:
                yield state.get_progress()
                return

            yield ResearchProgress(
                research_id=state.research_id,
                status=state.status,
                phase="brief",
                progress_pct=10,
                message=f"Brief generated with {len(state.brief.subtopics)} subtopics",
            )

            # 2. Plan Subtopics
            plan_behavior = PlanSubtopicsBehavior(config=config)
            state = await plan_behavior.run(state)

            yield ResearchProgress(
                research_id=state.research_id,
                status=state.status,
                phase="planning",
                progress_pct=15,
                message=f"Planned {len(state.researchers)} research threads",
            )

            # 3. Parallel Research
            yield ResearchProgress(
                research_id=state.research_id,
                status=state.status,
                phase="researching",
                progress_pct=20,
                message="Starting parallel research...",
            )

            research_behavior = ParallelResearchersBehavior(
                llm_service=self._llm_service,
                tavily_service=self._tavily_service,
                openrouter_search=self._openrouter_search,
                search_provider=self.search_provider,
                prompt_loader=self._prompt_loader,
                max_concurrent=config.max_concurrent_researchers,
            )
            state = await research_behavior.run(state)

            yield ResearchProgress(
                research_id=state.research_id,
                status=state.status,
                phase="researching",
                progress_pct=50,
                sources_found=len(state.all_sources),
                message=f"Found {len(state.all_sources)} sources via {self.search_provider}",
            )

            # 4. Compress Findings
            yield ResearchProgress(
                research_id=state.research_id,
                status=ResearchStatus.COMPRESSING,
                phase="compressing",
                progress_pct=60,
                sources_found=len(state.all_sources),
                message="Synthesizing findings...",
            )

            compress_behavior = CompressFindingsBehavior(
                llm_service=self._llm_service,
                prompt_loader=self._prompt_loader,
            )
            state = await compress_behavior.run(state)

            yield ResearchProgress(
                research_id=state.research_id,
                status=state.status,
                phase="compressing",
                progress_pct=70,
                sources_found=len(state.all_sources),
                message=f"Synthesized {len(state.compressed_findings)} key findings",
            )

            # 5. Generate Report
            yield ResearchProgress(
                research_id=state.research_id,
                status=ResearchStatus.GENERATING,
                phase="generating",
                progress_pct=75,
                sources_found=len(state.all_sources),
                message="Generating report...",
            )

            report_behavior = GenerateReportBehavior(
                llm_service=self._llm_service,
                prompt_loader=self._prompt_loader,
            )
            state = await report_behavior.run(state)

            yield ResearchProgress(
                research_id=state.research_id,
                status=state.status,
                phase="generating",
                progress_pct=90,
                sources_found=len(state.all_sources),
                message="Report generated",
            )

            # 6. Persist to Vault
            if self.vault_path and request.save_to_vault:
                yield ResearchProgress(
                    research_id=state.research_id,
                    status=state.status,
                    phase="saving",
                    progress_pct=95,
                    sources_found=len(state.all_sources),
                    message="Saving to vault...",
                )

                persist_behavior = PersistToVaultBehavior(
                    vault_path=self.vault_path,
                )
                state = await persist_behavior.run(state)

            # Final progress
            yield ResearchProgress(
                research_id=state.research_id,
                status=state.status,
                phase="completed",
                progress_pct=100,
                sources_found=len(state.all_sources),
                message="Research completed",
            )

            logger.info(
                f"Streaming research completed: {state.research_id}",
                extra={
                    "status": state.status.value,
                    "sources": len(state.all_sources),
                }
            )

        except Exception as e:
            logger.error(f"Streaming research failed: {e}", exc_info=True)
            state.status = ResearchStatus.FAILED
            yield ResearchProgress(
                research_id=state.research_id,
                status=ResearchStatus.FAILED,
                phase="failed",
                progress_pct=0,
                message=f"Research failed: {str(e)}",
            )


# Factory function for easy instantiation
def create_research_orchestrator(
    user_id: str,
    vault_path: Optional[str] = None,
    search_provider: SearchProvider = "none",
    tavily_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
) -> ResearchOrchestrator:
    """Create a research orchestrator instance.

    Args:
        user_id: User identifier
        vault_path: Optional path to user's vault
        search_provider: Which search provider to use ('tavily', 'openrouter', 'none')
        tavily_api_key: Optional Tavily API key (required if provider is 'tavily')
        openrouter_api_key: Optional OpenRouter API key (required if provider is 'openrouter')

    Returns:
        Configured ResearchOrchestrator
    """
    return ResearchOrchestrator(
        user_id=user_id,
        vault_path=vault_path,
        search_provider=search_provider,
        tavily_api_key=tavily_api_key,
        openrouter_api_key=openrouter_api_key,
    )


__all__ = ["ResearchOrchestrator", "create_research_orchestrator"]
