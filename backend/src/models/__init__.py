"""Pydantic models for data validation and serialization."""

from .auth import JWTPayload, TokenResponse
from .index import IndexHealth, Tag, Wikilink
from .note import Note, NoteCreate, NoteMetadata, NoteSummary, NoteUpdate
from .research import (
    ResearchBrief,
    ResearchConfig,
    ResearchDepth,
    ResearcherState,
    ResearchFinding,
    ResearchProgress,
    ResearchReport,
    ResearchRequest,
    ResearchSource,
    ResearchState,
    ResearchStatus,
    SourceType,
)
from .search import SearchRequest, SearchResult
from .user import GHProfile, User

__all__ = [
    "User",
    "GHProfile",
    "Note",
    "NoteMetadata",
    "NoteCreate",
    "NoteUpdate",
    "NoteSummary",
    "Wikilink",
    "Tag",
    "IndexHealth",
    "SearchResult",
    "SearchRequest",
    "TokenResponse",
    "JWTPayload",
    # Research models
    "ResearchStatus",
    "ResearchDepth",
    "SourceType",
    "ResearchBrief",
    "ResearchSource",
    "ResearchFinding",
    "ResearchReport",
    "ResearchRequest",
    "ResearchProgress",
    "ResearcherState",
    "ResearchState",
    "ResearchConfig",
]

