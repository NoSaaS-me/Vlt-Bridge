"""Pydantic models for Librarian Agent (009-oracle-agent).

The Librarian is a specialized subagent that handles content summarization,
vault organization, and caching of summaries for the Oracle agent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ContentItem(BaseModel):
    """A single piece of content to be summarized or processed.

    Attributes:
        path: Path to the source (file path, note path, thread ID, or URL)
        content: The actual content text
        source_type: Type of source this content came from
        metadata: Optional additional metadata about the content
    """
    path: str = Field(..., description="Path to the source (file, note, thread, or URL)")
    content: str = Field(..., description="The actual content text to process")
    source_type: Literal["vault", "thread", "code", "web"] = Field(
        ..., description="Type of knowledge source"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata about the content"
    )


class SummarizeRequest(BaseModel):
    """Request to summarize content.

    Attributes:
        task: Description of what to summarize or focus on
        content: List of content items to summarize
        max_tokens: Maximum tokens for the summary output
        force_refresh: Bypass cache and regenerate summary
    """
    task: str = Field(..., min_length=1, description="Description of what to summarize")
    content: List[ContentItem] = Field(..., min_length=1, description="Content items to summarize")
    max_tokens: int = Field(
        1000, ge=100, le=4000, description="Maximum tokens for summary"
    )
    force_refresh: bool = Field(
        False, description="Bypass cache and regenerate summary"
    )


class OrganizeRequest(BaseModel):
    """Request to organize vault content.

    Attributes:
        folder: Target folder to organize
        create_index: Whether to create/update an index.md file
        recursive: Whether to include subfolders
        task: Optional specific organization instructions
    """
    folder: str = Field(..., description="Folder path to organize")
    create_index: bool = Field(True, description="Create/update index.md for folder")
    recursive: bool = Field(False, description="Include subfolders in organization")
    task: Optional[str] = Field(None, description="Specific organization instructions")


class LibrarianStreamChunk(BaseModel):
    """Streaming chunk from Librarian to Oracle.

    The Librarian streams progress back to the Oracle using these chunks,
    allowing real-time updates during long-running operations.

    Chunk types:
    - thinking: Processing/analyzing content
    - summary: Summary content being generated
    - index: Index page content
    - cache_hit: Summary retrieved from cache
    - done: Operation completed successfully
    - error: Error occurred

    Attributes:
        type: Type of chunk (determines which other fields are populated)
        content: Text content for thinking/summary/index chunks
        sources: Paths cited in summaries
        cache_path: Where summary was cached (for summary/cache_hit)
        metadata: Additional chunk metadata
    """
    type: Literal["thinking", "summary", "index", "cache_hit", "done", "error"] = Field(
        ..., description="Chunk type"
    )
    content: Optional[str] = Field(
        None, description="Text content (thinking, summary, index chunks)"
    )
    sources: Optional[List[str]] = Field(
        None, description="Source paths cited in the content"
    )
    cache_path: Optional[str] = Field(
        None, description="Vault path where summary was cached"
    )
    error: Optional[str] = Field(
        None, description="Error message (for error chunks)"
    )
    tokens_used: Optional[int] = Field(
        None, description="Total tokens used (done chunk only)"
    )
    model_used: Optional[str] = Field(
        None, description="Model used (done chunk only)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )


class SummaryResult(BaseModel):
    """Final result of a summarization operation.

    Attributes:
        summary: The generated summary text
        sources: List of source paths that were summarized
        cache_path: Vault path where summary was cached
        from_cache: Whether the summary was retrieved from cache
        token_count: Approximate token count of the summary
        created_at: When the summary was created/retrieved
    """
    summary: str = Field(..., description="The generated summary text")
    sources: List[str] = Field(
        default_factory=list, description="Source paths that were summarized"
    )
    cache_path: Optional[str] = Field(
        None, description="Vault path where summary was cached"
    )
    from_cache: bool = Field(False, description="Whether summary was from cache")
    token_count: int = Field(0, description="Approximate token count")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )


class OrganizeResult(BaseModel):
    """Final result of a vault organization operation.

    Attributes:
        index_path: Path to the created/updated index file
        files_organized: Number of files that were organized
        files_moved: List of file moves (old_path -> new_path)
        index_content: Content of the generated index file
        wikilinks_created: Count of wikilinks added
    """
    index_path: Optional[str] = Field(
        None, description="Path to created/updated index file"
    )
    files_organized: int = Field(0, description="Number of files organized")
    files_moved: List[Dict[str, str]] = Field(
        default_factory=list, description="File moves: [{old_path, new_path}]"
    )
    index_content: Optional[str] = Field(
        None, description="Content of generated index file"
    )
    wikilinks_created: int = Field(0, description="Number of wikilinks added")


class CachedSummaryMetadata(BaseModel):
    """Frontmatter metadata for cached summaries in the vault.

    Stored as YAML frontmatter in oracle-cache/summaries/.../*.md files.

    Attributes:
        created: When the summary was created
        sources: List of source paths that were summarized
        token_count: Approximate token count of the summary
        cache_key: Unique key for cache invalidation
        task: The summarization task that generated this
        source_type: Type of content that was summarized
    """
    created: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    sources: List[str] = Field(
        default_factory=list, description="Source paths summarized"
    )
    token_count: int = Field(0, description="Summary token count")
    cache_key: str = Field(..., description="Unique cache key for invalidation")
    task: Optional[str] = Field(None, description="Task description")
    source_type: Literal["vault", "thread", "code", "web", "mixed"] = Field(
        "mixed", description="Primary source type"
    )


__all__ = [
    "ContentItem",
    "SummarizeRequest",
    "OrganizeRequest",
    "LibrarianStreamChunk",
    "SummaryResult",
    "OrganizeResult",
    "CachedSummaryMetadata",
]
