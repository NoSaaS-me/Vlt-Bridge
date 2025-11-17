"""HTTP API routes for search operations."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...models.index import Tag
from ...models.search import SearchResult
from ...services.database import DatabaseService
from ...services.indexer import IndexerService

router = APIRouter()


class BacklinkResult(BaseModel):
    """Result from backlinks query."""

    note_path: str
    title: str


def get_user_id() -> str:
    """Return the current user ID. For now, hardcoded to 'local-dev'."""
    return "local-dev"


@router.get("/api/search", response_model=list[SearchResult])
async def search_notes(q: str = Query(..., min_length=1, max_length=256)):
    """Full-text search across all notes."""
    user_id = get_user_id()
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
async def get_backlinks(path: str):
    """Get all notes that link to this note."""
    user_id = get_user_id()
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
async def get_tags():
    """Get all tags with usage counts."""
    user_id = get_user_id()
    indexer_service = IndexerService()
    
    try:
        tags = indexer_service.get_tags(user_id)
        
        return [
            Tag(tag_name=tag["tag"], count=tag["count"])
            for tag in tags
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}")


__all__ = ["router", "BacklinkResult"]

