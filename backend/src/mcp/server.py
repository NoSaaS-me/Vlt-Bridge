"""FastMCP server exposing vault and indexing tools."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from pydantic import Field

from ..services import IndexerService, VaultNote, VaultService

mcp = FastMCP(
    "obsidian-docs-viewer",
    instructions="Interact with a multi-tenant Obsidian-like documentation vault.",
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
        description="Optional folder path (relative) to filter results. Example: 'api'.",
    ),
) -> List[Dict[str, Any]]:
    user_id = _current_user_id()
    notes = vault_service.list_notes(user_id, folder=folder)
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
    path: str = Field(..., description="Relative path to the note (must include .md)"),
) -> Dict[str, Any]:
    user_id = _current_user_id()
    note = vault_service.read_note(user_id, path)
    return _note_to_response(note)


@mcp.tool(
    name="write_note",
    description="Create or update a note. Automatically updates frontmatter timestamps and search index.",
)
def write_note(
    path: str = Field(..., description="Relative note path (includes .md)"),
    body: str = Field(..., description="Markdown body content"),
    title: Optional[str] = Field(
        default=None,
        description="Optional title override. Defaults to frontmatter title or first heading.",
    ),
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional frontmatter metadata dictionary (tags, project, etc.).",
    ),
) -> Dict[str, Any]:
    user_id = _current_user_id()
    note = vault_service.write_note(
        user_id,
        path,
        title=title,
        metadata=metadata,
        body=body,
    )
    indexer_service.index_note(user_id, note)
    return {"status": "ok", "path": path}


@mcp.tool(name="delete_note", description="Delete a note and remove it from the index.")
def delete_note(
    path: str = Field(..., description="Relative note path (includes .md)"),
) -> Dict[str, str]:
    user_id = _current_user_id()
    vault_service.delete_note(user_id, path)
    indexer_service.delete_note_index(user_id, path)
    return {"status": "ok"}


@mcp.tool(
    name="search_notes",
    description="Full-text search with snippets and recency-aware scoring.",
)
def search_notes(
    query: str = Field(..., description="Search query (minimum 1 character)"),
    limit: int = Field(50, ge=1, le=100, description="Maximum number of results to return."),
) -> List[Dict[str, Any]]:
    user_id = _current_user_id()
    results = indexer_service.search_notes(user_id, query, limit=limit)
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
    path: str = Field(..., description="Target note path (includes .md)"),
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
