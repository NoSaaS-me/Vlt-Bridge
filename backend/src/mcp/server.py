"""FastMCP server exposing vault and indexing tools."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from pydantic import Field

from ..services import IndexerService, VaultNote, VaultService

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "obsidian-docs-viewer",
    instructions=(
        "Multi-tenant vault tools. STDIO uses user_id 'local-dev'; HTTP mode must validate each "
        "request with JWT.sub. Note paths must be relative '.md' ≤256 chars without '..' or '\\'. "
        "Frontmatter is YAML: tags are string arrays and 'version' is reserved. Notes must be ≤1 MiB; "
        "writes refresh created/updated timestamps and synchronously update the search index; deletes "
        "clear index rows and backlinks. Wikilinks use [[...]] slug matching (prefer same folder, else "
        "lexicographic). Search ranking = bm25(title*3, body*1) + recency bonus (+1 ≤7d, +0.5 ≤30d)."
    ),
)

vault_service = VaultService()
indexer_service = IndexerService()


def _current_user_id() -> str:
    """Resolve the acting user ID (local mode defaults to local-dev)."""
    return os.getenv("LOCAL_USER_ID", "local-dev")


def _note_to_response(note: VaultNote) -> Dict[str, Any]:
    return {
        "path": note["path"],
        "title": note["title"],
        "metadata": dict(note.get("metadata") or {}),
        "body": note.get("body", ""),
    }


@mcp.tool(name="list_notes", description="List notes in the vault (optionally scoped to a folder).")
def list_notes(
    folder: Optional[str] = Field(
        default=None,
        description="Optional relative folder (trim '/' ; no '..' or '\\').",
    ),
) -> List[Dict[str, Any]]:
    start_time = time.time()
    user_id = _current_user_id()
    
    notes = vault_service.list_notes(user_id, folder=folder)
    
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "MCP tool called",
        extra={
            "tool_name": "list_notes",
            "user_id": user_id,
            "folder": folder or "(root)",
            "result_count": len(notes),
            "duration_ms": f"{duration_ms:.2f}"
        }
    )
    
    return [
        {
            "path": entry["path"],
            "title": entry["title"],
            "last_modified": entry["last_modified"].isoformat(),
        }
        for entry in notes
    ]


@mcp.tool(name="read_note", description="Read a Markdown note with metadata and body.")
def read_note(
    path: str = Field(..., description="Relative '.md' path ≤256 chars (no '..' or '\\')."),
) -> Dict[str, Any]:
    start_time = time.time()
    user_id = _current_user_id()
    
    note = vault_service.read_note(user_id, path)
    
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "MCP tool called",
        extra={
            "tool_name": "read_note",
            "user_id": user_id,
            "note_path": path,
            "duration_ms": f"{duration_ms:.2f}"
        }
    )
    
    return _note_to_response(note)


@mcp.tool(
    name="write_note",
    description="Create or update a note. Automatically updates frontmatter timestamps and search index.",
)
def write_note(
    path: str = Field(..., description="Relative '.md' path ≤256 chars (no '..' or '\\')."),
    body: str = Field(..., description="Markdown body ≤1 MiB."),
    title: Optional[str] = Field(
        default=None,
        description="Optional title override; otherwise frontmatter/H1/filename is used.",
    ),
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional frontmatter dict (tags arrays of strings; 'version' reserved).",
    ),
) -> Dict[str, Any]:
    start_time = time.time()
    user_id = _current_user_id()
    
    note = vault_service.write_note(
        user_id,
        path,
        title=title,
        metadata=metadata,
        body=body,
    )
    indexer_service.index_note(user_id, note)
    
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "MCP tool called",
        extra={
            "tool_name": "write_note",
            "user_id": user_id,
            "note_path": path,
            "duration_ms": f"{duration_ms:.2f}"
        }
    )
    
    return {"status": "ok", "path": path}


@mcp.tool(name="delete_note", description="Delete a note and remove it from the index.")
def delete_note(
    path: str = Field(..., description="Relative '.md' path ≤256 chars (no '..' or '\\')."),
) -> Dict[str, str]:
    start_time = time.time()
    user_id = _current_user_id()
    
    vault_service.delete_note(user_id, path)
    indexer_service.delete_note_index(user_id, path)
    
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "MCP tool called",
        extra={
            "tool_name": "delete_note",
            "user_id": user_id,
            "note_path": path,
            "duration_ms": f"{duration_ms:.2f}"
        }
    )
    
    return {"status": "ok"}


@mcp.tool(
    name="search_notes",
    description="Full-text search with snippets and recency-aware scoring.",
)
def search_notes(
    query: str = Field(..., description="Non-empty search query (bm25 + recency)."),
    limit: int = Field(50, ge=1, le=100, description="Result cap between 1 and 100."),
) -> List[Dict[str, Any]]:
    start_time = time.time()
    user_id = _current_user_id()
    
    results = indexer_service.search_notes(user_id, query, limit=limit)
    
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "MCP tool called",
        extra={
            "tool_name": "search_notes",
            "user_id": user_id,
            "query": query,
            "limit": limit,
            "result_count": len(results),
            "duration_ms": f"{duration_ms:.2f}"
        }
    )
    
    return [
        {
            "path": row["path"],
            "title": row["title"],
            "snippet": row["snippet"],
        }
        for row in results
    ]


@mcp.tool(name="get_backlinks", description="List notes that reference the target note.")
def get_backlinks(
    path: str = Field(..., description="Relative '.md' path ≤256 chars (no '..' or '\\')."),
) -> List[Dict[str, Any]]:
    user_id = _current_user_id()
    backlinks = indexer_service.get_backlinks(user_id, path)
    return backlinks


@mcp.tool(name="get_tags", description="List tags and associated note counts.")
def get_tags() -> List[Dict[str, Any]]:
    user_id = _current_user_id()
    return indexer_service.get_tags(user_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
