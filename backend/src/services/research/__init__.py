"""Deep Research services.

This package provides the orchestration layer for Deep Research:
- ResearchOrchestrator: Main coordinator for research workflow
- Behavior nodes: Individual steps in the research behavior tree
- LLM service: Simplified LLM interface for research
- Vault persister: Saves research to user's vault
"""

from .behaviors import (
    CompressFindingsBehavior,
    GenerateBriefBehavior,
    GenerateReportBehavior,
    ParallelResearchersBehavior,
    PersistToVaultBehavior,
    PlanSubtopicsBehavior,
    ResearchBehavior,
    ResearcherBehavior,
)
from .llm_service import ResearchLLMService
from .orchestrator import ResearchOrchestrator, create_research_orchestrator
from .vault_persister import ResearchVaultPersister

__all__ = [
    # Orchestrator
    "ResearchOrchestrator",
    "create_research_orchestrator",
    # Behaviors
    "ResearchBehavior",
    "GenerateBriefBehavior",
    "PlanSubtopicsBehavior",
    "ResearcherBehavior",
    "ParallelResearchersBehavior",
    "CompressFindingsBehavior",
    "GenerateReportBehavior",
    "PersistToVaultBehavior",
    # Services
    "ResearchLLMService",
    "ResearchVaultPersister",
]
