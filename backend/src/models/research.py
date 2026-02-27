"""Models for Deep Research feature."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResearchStatus(str, Enum):
    """Status of a research project."""

    PLANNING = "planning"
    RESEARCHING = "researching"
    COMPRESSING = "compressing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchDepth(str, Enum):
    """Depth level for research."""

    QUICK = "quick"  # 2-3 sources, 1 researcher
    STANDARD = "standard"  # 5-8 sources, 3 researchers
    THOROUGH = "thorough"  # 10+ sources, 5 researchers


class SourceType(str, Enum):
    """Type of research source."""

    WEB = "web"
    ACADEMIC = "academic"
    CODE = "code"
    VAULT = "vault"
    GITHUB = "github"


# Pydantic Models for API


class ResearchBrief(BaseModel):
    """The structured research brief generated from user query."""

    original_query: str
    refined_question: str
    scope: str = Field(description="What is in/out of scope")
    subtopics: List[str] = Field(description="Key subtopics to research")
    constraints: Optional[str] = None
    language: str = "en"


class ResearchSource(BaseModel):
    """A single source found during research."""

    id: int
    url: str
    title: str
    source_type: SourceType = SourceType.WEB
    relevance_score: float = Field(ge=0, le=1)
    content_summary: str
    key_quotes: List[str] = Field(default_factory=list)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    raw_content: Optional[str] = None


class ResearchFinding(BaseModel):
    """A synthesized finding from research."""

    claim: str
    source_ids: List[int]  # References to ResearchSource.id
    confidence: float = Field(ge=0, le=1)
    category: Optional[str] = None


class ResearchReport(BaseModel):
    """The final research report."""

    title: str
    executive_summary: str
    sections: List[Dict[str, Any]]  # [{heading, content, source_ids}]
    recommendations: Optional[List[str]] = None
    limitations: Optional[List[str]] = None
    sources: List[ResearchSource]

    # Quality metrics
    comprehensiveness: float = Field(ge=0, le=1, default=0.0)
    analytical_depth: float = Field(ge=0, le=1, default=0.0)
    source_diversity: float = Field(ge=0, le=1, default=0.0)
    citation_density: float = Field(ge=0, le=1, default=0.0)


class ResearchRequest(BaseModel):
    """Request to start a deep research task."""

    query: str
    depth: ResearchDepth = ResearchDepth.STANDARD
    save_to_vault: bool = True
    output_folder: Optional[str] = None
    max_sources: int = Field(default=10, ge=1, le=50)
    include_code: bool = False
    include_vault: bool = False


class ResearchProgress(BaseModel):
    """Progress update for ongoing research."""

    research_id: str
    status: ResearchStatus
    phase: str
    progress_pct: float = Field(ge=0, le=100)
    sources_found: int = 0
    current_subtopic: Optional[str] = None
    message: Optional[str] = None


# Dataclasses for Internal State


@dataclass
class ResearcherState:
    """State for a single researcher agent."""

    subtopic: str
    messages: List[Any] = field(default_factory=list)
    sources: List[ResearchSource] = field(default_factory=list)
    tool_calls: int = 0
    max_tool_calls: int = 10
    completed: bool = False
    error: Optional[str] = None


@dataclass
class ResearchState:
    """Overall state for a research project."""

    research_id: str
    user_id: str
    request: ResearchRequest
    status: ResearchStatus = ResearchStatus.PLANNING

    # Research artifacts
    brief: Optional[ResearchBrief] = None
    researchers: List[ResearcherState] = field(default_factory=list)
    all_sources: List[ResearchSource] = field(default_factory=list)
    compressed_findings: List[ResearchFinding] = field(default_factory=list)
    report: Optional[ResearchReport] = None

    # Metrics
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_tokens: int = 0
    total_searches: int = 0

    # Vault integration
    vault_folder: Optional[str] = None

    def add_source(self, source: ResearchSource) -> None:
        """Add source with deduplication."""
        existing_urls = {s.url for s in self.all_sources}
        if source.url not in existing_urls:
            source.id = len(self.all_sources) + 1
            self.all_sources.append(source)

    def get_progress(self) -> ResearchProgress:
        """Get current progress."""
        phase_progress = {
            ResearchStatus.PLANNING: 10,
            ResearchStatus.RESEARCHING: 50,
            ResearchStatus.COMPRESSING: 70,
            ResearchStatus.GENERATING: 90,
            ResearchStatus.COMPLETED: 100,
            ResearchStatus.FAILED: 0,
        }

        active_researcher = None
        for r in self.researchers:
            if not r.completed:
                active_researcher = r.subtopic
                break

        return ResearchProgress(
            research_id=self.research_id,
            status=self.status,
            phase=self.status.value,
            progress_pct=phase_progress.get(self.status, 0),
            sources_found=len(self.all_sources),
            current_subtopic=active_researcher,
        )


# Configuration


@dataclass
class ResearchConfig:
    """Configuration for research behavior."""

    # Model settings
    planning_model: str = "openrouter:anthropic/claude-sonnet-4"
    research_model: str = "openrouter:anthropic/claude-sonnet-4"
    compression_model: str = "openrouter:anthropic/claude-sonnet-4"
    report_model: str = "openrouter:anthropic/claude-sonnet-4"

    # Limits
    max_concurrent_researchers: int = 5
    max_tool_calls_per_researcher: int = 10
    max_sources: int = 20
    max_retries: int = 3

    # Token budgets
    brief_max_tokens: int = 2000
    researcher_max_tokens: int = 10000
    compression_max_tokens: int = 5000
    report_max_tokens: int = 15000

    # Depth presets
    @classmethod
    def for_depth(cls, depth: ResearchDepth) -> "ResearchConfig":
        """Get config for a specific depth level."""
        if depth == ResearchDepth.QUICK:
            return cls(
                max_concurrent_researchers=1,
                max_tool_calls_per_researcher=5,
                max_sources=5,
            )
        elif depth == ResearchDepth.THOROUGH:
            return cls(
                max_concurrent_researchers=5,
                max_tool_calls_per_researcher=15,
                max_sources=30,
            )
        return cls()  # Standard
