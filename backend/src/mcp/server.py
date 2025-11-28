"""FastMCP server exposing vault and indexing tools."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent
from pydantic import Field

# Load environment variables from .env file
load_dotenv()

from ..services import IndexerService, VaultNote, VaultService
from ..services.auth import AuthError, AuthService
from ..services.config import get_config, PROJECT_ROOT

try:
    from fastmcp.server.http import _current_http_request  # type: ignore
except ImportError:  # pragma: no cover
    _current_http_request = None

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "obsidian-docs-viewer",
    instructions=(
        "Multi-tenant vault tools. STDIO uses user_id 'local-dev'; HTTP mode must validate each "
        "request with JWT.sub. Note paths must be relative '.md' files under 256 chars without '..' or '\\'. "
        "Frontmatter is YAML: tags are string arrays and 'version' is reserved. Notes must be <=1 MiB; "
        "writes refresh created/updated timestamps and synchronously update the search index; deletes "
        "clear index rows and backlinks. Wikilinks use [[...]] slug matching (prefer same folder, else "
        "lexicographic). Search ranking = bm25(title*3, body*1) + recency bonus (+1 if <=7d, +0.5 if <=30d)."
    ),
)

vault_service = VaultService()
indexer_service = IndexerService()
auth_service = AuthService()


@mcp.resource("ui://widget/note.html", mime_type="text/html+skybridge")
def widget_resource() -> str:
    """Return the widget HTML bundle."""
    # Locate widget.html relative to project root
    # In Docker: /app/frontend/dist/widget.html
    # Local: frontend/dist/widget.html
    # We use PROJECT_ROOT from config
    
    widget_path = PROJECT_ROOT / "frontend" / "dist" / "widget.html"
    
    logger.info(f"Reading widget from: {widget_path}")
    
    if not widget_path.exists():
        logger.error(f"Widget path does not exist: {widget_path}")
        return "Widget build not found. Please run 'npm run build' in frontend directory."
        
    try:
        html_content = widget_path.read_text(encoding="utf-8")
        logger.info(f"Widget content length: {len(html_content)}")
        if not html_content.strip():
            logger.error("Widget file is empty!")
            return "Widget build file is empty."
            
        # Replace relative asset paths with absolute URLs for ChatGPT iframe
        config = get_config()
        base_url = config.hf_space_url.rstrip("/")
        logger.info(f"Injecting base URL: {base_url}")
        
        # Inject API_BASE_URL global for the widget to use
        html_content = html_content.replace(
            '<head>', 
            f'<head><script>window.API_BASE_URL = "{base_url}";</script>'
        )
        
        # Vite builds usually output /assets/...
        html_content = html_content.replace('src="/assets/', f'src="{base_url}/assets/')
        html_content = html_content.replace('href="/assets/', f'href="{base_url}/assets/')
        
        return html_content
    except Exception as e:
        logger.exception(f"Failed to read widget file: {e}")
        return f"Server error reading widget: {e}"


def _current_user_id() -> str:
    """Resolve the acting user ID (local mode defaults to local-dev)."""
    # HTTP transport (hosted) uses Authorization headers
    if _current_http_request is not None:
        try:
            request = _current_http_request.get()  # type: ignore[call-arg]
        except LookupError:
            request = None
        if request is not None:
            header = request.headers.get("Authorization")
            
            # Check for No-Auth mode if header is missing
            if not header:
                config = get_config()
                if config.enable_noauth_mcp:
                    return "demo-user"
                raise PermissionError("Authorization header required")
                
            scheme, _, token = header.partition(" ")
            if scheme.lower() != "bearer" or not token:
                raise PermissionError("Authorization header must be 'Bearer <token>'")
            try:
                payload = auth_service.validate_jwt(token)
            except AuthError as exc:
                raise PermissionError(exc.message) from exc
            os.environ.setdefault("LOCAL_USER_ID", payload.sub)
            return payload.sub

    # STDIO / local fall back
    return os.getenv("LOCAL_USER_ID", "local-dev")


def _note_to_response(note: VaultNote) -> Dict[str, Any]:
    return {
        "path": note["path"],
        "title": note["title"],
        "metadata": dict(note.get("metadata") or {}),
        "body": note.get("body", ""),
    }


@mcp.tool(
    name="list_notes",
    description="List notes in the vault (optionally scoped to a folder).",
)
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
            "duration_ms": f"{duration_ms:.2f}",
        },
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
    path: str = Field(
        ..., description="Relative '.md' path ≤256 chars (no '..' or '\\')."
    ),
) -> dict:
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
            "duration_ms": f"{duration_ms:.2f}",
        },
    )

    structured_note = {
        "title": note["title"],
        "note_path": note["path"],
        "body": note["body"],
        "metadata": note["metadata"],
        "updated": note["modified"].isoformat(),
    }

    return ToolResult(
        content=[TextContent(type="text", text=f"Read note: {note['title']}")],
        structured_content={"note": structured_note},
        meta={
            "openai/outputTemplate": "ui://widget/note.html",
            "openai/resultCanProduceWidget": True,
            "openai/toolInvocation/invoking": f"Opening {note['title']}...",
            "openai/toolInvocation/invoked": f"Loaded {note['title']}"
        }
    )


@mcp.tool(
    name="write_note",
    description="Create or update a note. Automatically updates frontmatter timestamps and search index.",
)
def write_note(
    path: str = Field(
        ..., description="Relative '.md' path ≤256 chars (no '..' or '\\')."
    ),
    body: str = Field(..., description="Markdown body ≤1 MiB."),
    title: Optional[str] = Field(
        default=None,
        description="Optional title override; otherwise frontmatter/H1/filename is used.",
    ),
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional frontmatter dict (tags arrays of strings; 'version' reserved).",
    ),
) -> dict:
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
            "duration_ms": f"{duration_ms:.2f}",
        },
    )

    structured_note = {
        "title": note["title"],
        "note_path": note["path"],
        "body": note["body"],
        "metadata": note["metadata"],
        "updated": note["modified"].isoformat(),
    }

    return ToolResult(
        content=[TextContent(type="text", text=f"Successfully saved note: {path}")],
        structured_content={"note": structured_note},
        meta={
            "openai/outputTemplate": "ui://widget/note.html",
            "openai/resultCanProduceWidget": True,
            "openai/toolInvocation/invoking": f"Saving {path}...",
            "openai/toolInvocation/invoked": f"Saved {path}"
        }
    )


@mcp.tool(name="delete_note", description="Delete a note and remove it from the index.")
def delete_note(
    path: str = Field(
        ..., description="Relative '.md' path ≤256 chars (no '..' or '\\')."
    ),
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
            "duration_ms": f"{duration_ms:.2f}",
        },
    )

    return {"status": "ok"}


@mcp.tool(
    name="search_notes",
    description="Full-text search with snippets and recency-aware scoring.",
    meta={
        "openai/outputTemplate": "ui://widget/note.html",
        "openai/toolInvocation/invoking": "Searching...",
        "openai/toolInvocation/invoked": "Search complete."
    }
)
def search_notes(
    query: str = Field(..., description="Non-empty search query (bm25 + recency)."),
    limit: int = Field(50, ge=1, le=100, description="Result cap between 1 and 100."),
) -> ToolResult:
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
            "duration_ms": f"{duration_ms:.2f}",
        },
    )

    # Structure results for the widget
    structured_results = []
    for r in results:
        structured_results.append({
            "title": r["title"],
            "note_path": r["path"],
            "snippet": r["snippet"],
            "score": r["score"],
            "updated": r["updated"] if isinstance(r["updated"], str) else r["updated"].isoformat()
        })

    return ToolResult(
        content=[TextContent(type="text", text=f"Found {len(results)} notes matching '{query}'.")],
        structured_content={"results": structured_results},
        meta={
            "openai/outputTemplate": "ui://widget/note.html",
            "openai/resultCanProduceWidget": True,
            "openai/toolInvocation/invoking": f"Searching for '{query}'...",
            "openai/toolInvocation/invoked": f"Found {len(results)} results."
        }
    )



@mcp.tool(
    name="get_backlinks", description="List notes that reference the target note."
)
def get_backlinks(
    path: str = Field(
        ..., description="Relative '.md' path ≤256 chars (no '..' or '\\')."
    ),
) -> List[Dict[str, Any]]:
    user_id = _current_user_id()
    backlinks = indexer_service.get_backlinks(user_id, path)
    return backlinks


@mcp.tool(name="get_tags", description="List tags and associated note counts.")
def get_tags() -> List[Dict[str, Any]]:
    user_id = _current_user_id()
    return indexer_service.get_tags(user_id)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower() or "stdio"

    # Configure HTTP transport with custom port if specified
    if transport == "http":
        port = int(os.getenv("MCP_PORT", "8001"))
        host = os.getenv("MCP_HOST", "127.0.0.1")
        logger.info(
            "Starting MCP server",
            extra={"transport": transport, "host": host, "port": port},
        )
        mcp.run(transport=transport, host=host, port=port)
    else:
        logger.info("Starting MCP server", extra={"transport": transport})
        mcp.run(transport=transport)
