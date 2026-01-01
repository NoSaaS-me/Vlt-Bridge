"""HTTP API routes for search operations."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ...models.index import Tag
from ...models.search import SearchResult
from ...services.database import DatabaseService
from ...services.indexer import IndexerService
from ..middleware import AuthContext, get_auth_context

router = APIRouter()


class BacklinkResult(BaseModel):
    """Result from backlinks query."""

    note_path: str
    title: str


class WikilinkResolution(BaseModel):
    """Result from wikilink resolution."""

    link_text: str
    target_path: Optional[str] = None
    is_resolved: bool


@router.get("/api/search", response_model=list[SearchResult])
async def search_notes(
    q: str = Query(..., min_length=1, max_length=256),
    auth: AuthContext = Depends(get_auth_context),
):
    """Full-text search across all notes."""
    user_id = auth.user_id
    indexer_service = IndexerService()
    
    try:
        results = indexer_service.search_notes(user_id, q, limit=50)
        
        search_results = []
        for result in results:
            # Use snippet from search results
            snippet = result.get("snippet", "")
            
            updated = result.get("updated")
            if isinstance(updated, str):
                updated = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            elif not isinstance(updated, datetime):
                updated = datetime.now()
            
            search_results.append(
                SearchResult(
                    note_path=result["path"],
                    title=result["title"],
                    snippet=snippet,
                    score=result.get("score", 0.0),
                    updated=updated,
                )
            )
        
        return search_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/api/backlinks/{path:path}", response_model=list[BacklinkResult])
async def get_backlinks(path: str, auth: AuthContext = Depends(get_auth_context)):
    """Get all notes that link to this note."""
    user_id = auth.user_id
    indexer_service = IndexerService()
    
    try:
        # URL decode the path
        note_path = unquote(path)
        
        backlinks = indexer_service.get_backlinks(user_id, note_path)
        
        return [
            BacklinkResult(
                note_path=backlink["path"],
                title=backlink["title"],
            )
            for backlink in backlinks
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backlinks: {str(e)}")


@router.get("/api/tags", response_model=list[Tag])
async def get_tags(auth: AuthContext = Depends(get_auth_context)):
    """Get all tags with usage counts."""
    user_id = auth.user_id
    indexer_service = IndexerService()

    try:
        tags = indexer_service.get_tags(user_id)

        return [
            Tag(tag_name=tag["tag"], count=tag["count"])
            for tag in tags
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}")


@router.get("/api/wikilinks/resolve", response_model=WikilinkResolution)
async def resolve_wikilink(
    link: str = Query(..., min_length=1, max_length=256, description="Wikilink text to resolve"),
    context: Optional[str] = Query(None, description="Optional context note path for same-folder preference"),
    auth: AuthContext = Depends(get_auth_context),
):
    """Resolve a wikilink to its target note path using slug-based matching."""
    user_id = auth.user_id
    indexer_service = IndexerService()

    try:
        # Decode the link text if needed
        link_text = unquote(link)
        context_path = unquote(context) if context else ""

        result = indexer_service.resolve_single_wikilink(user_id, link_text, context_path)

        return WikilinkResolution(
            link_text=result["link_text"],
            target_path=result["target_path"],
            is_resolved=result["is_resolved"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve wikilink: {str(e)}")


__all__ = ["router", "BacklinkResult", "WikilinkResolution"]

