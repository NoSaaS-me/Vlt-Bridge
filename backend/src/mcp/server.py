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
    # Check for deprecated noauth mode and log warning
    config = get_config()
    if config.enable_noauth_mcp:
        logger.warning(
            "ENABLE_NOAUTH_MCP is enabled. This setting is DEPRECATED and should ONLY be used "
            "in isolated development environments, NEVER in production. It will be removed in a future version."
        )

    # HTTP transport (hosted) uses Authorization headers
    if _current_http_request is not None:
        try:
            request = _current_http_request.get()  # type: ignore[call-arg]
        except LookupError:
            request = None
        if request is not None:
            header = request.headers.get("Authorization")

            if not header:
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
        content=[TextContent(type="text", text=f"Read note: {note['title']}\n\n{note['body']}")],
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


# ============================================================================
# Oracle & Code Intelligence Tools (Vlt Oracle Feature)
# ============================================================================

from ..services.oracle_bridge import OracleBridge, OracleBridgeError

oracle_bridge = OracleBridge()


@mcp.tool(
    name="ask_oracle",
    description=(
        "Ask a question about the codebase. Searches across documentation (markdown vault), "
        "code (functions, classes, imports), and development history (past decisions, debugging sessions). "
        "Returns a synthesized answer with source citations. Use for conceptual questions, "
        "understanding code behavior, or finding how things work."
    ),
)
async def ask_oracle(
    question: str = Field(
        ...,
        description="Natural language question about the codebase",
        min_length=1,
        max_length=2000,
    ),
    sources: Optional[List[str]] = Field(
        default=None,
        description="Knowledge sources to query. Omit to query all sources.",
    ),
    explain: bool = Field(
        default=False,
        description="Include retrieval traces for debugging",
    ),
) -> Dict[str, Any]:
    """Ask Oracle a question about the codebase."""
    start_time = time.time()
    user_id = _current_user_id()

    try:
        # Call vlt oracle via bridge
        result = await oracle_bridge.ask_oracle(
            question=question,
            sources=sources,
            explain=explain,
        )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "MCP tool called",
            extra={
                "tool_name": "ask_oracle",
                "user_id": user_id,
                "question_length": len(question),
                "sources": sources or "all",
                "duration_ms": f"{duration_ms:.2f}",
            },
        )

        return result

    except OracleBridgeError as exc:
        logger.error(f"Oracle query failed: {exc.message}", extra=exc.details)
        return {
            "error": exc.message,
            "details": exc.details,
            "answer": "Sorry, I encountered an error while processing your question.",
            "sources": [],
        }


@mcp.tool(
    name="search_code",
    description=(
        "Search code using hybrid retrieval (semantic vector search + exact keyword match). "
        "Returns ranked code chunks with context. Use when you need to find specific code patterns, "
        "function implementations, or explore the codebase."
    ),
)
async def search_code(
    query: str = Field(
        ...,
        description="Search query - can be natural language or code pattern",
        min_length=1,
        max_length=500,
    ),
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to return",
    ),
    language: Optional[str] = Field(
        default=None,
        description="Filter by programming language",
    ),
    file_pattern: Optional[str] = Field(
        default=None,
        description="Glob pattern to filter files (e.g., 'src/**/*.py')",
    ),
) -> Dict[str, Any]:
    """Search code using hybrid retrieval."""
    start_time = time.time()
    user_id = _current_user_id()

    try:
        result = await oracle_bridge.search_code(
            query=query,
            limit=limit,
            language=language,
            file_pattern=file_pattern,
        )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "MCP tool called",
            extra={
                "tool_name": "search_code",
                "user_id": user_id,
                "query": query,
                "limit": limit,
                "duration_ms": f"{duration_ms:.2f}",
            },
        )

        return result

    except OracleBridgeError as exc:
        logger.error(f"Code search failed: {exc.message}", extra=exc.details)
        return {
            "error": exc.message,
            "details": exc.details,
            "results": [],
            "total_matches": 0,
        }


@mcp.tool(
    name="find_definition",
    description=(
        "Find where a symbol (function, class, variable) is defined. Uses code intelligence "
        "(ctags/SCIP) for exact lookup, not fuzzy search. Returns precise file:line location. "
        "Use when you need to navigate to a symbol's definition."
    ),
)
async def find_definition(
    symbol: str = Field(
        ...,
        description="Symbol name to find (e.g., 'UserService', 'authenticate', 'MAX_RETRIES')",
        min_length=1,
        max_length=256,
    ),
    scope: Optional[str] = Field(
        default=None,
        description="Optional file path to narrow search (e.g., 'src/services/')",
    ),
    kind: Optional[str] = Field(
        default=None,
        description="Symbol kind to filter",
    ),
) -> Dict[str, Any]:
    """Find where a symbol is defined."""
    start_time = time.time()
    user_id = _current_user_id()

    try:
        result = await oracle_bridge.find_definition(
            symbol=symbol,
            scope=scope,
            kind=kind,
        )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "MCP tool called",
            extra={
                "tool_name": "find_definition",
                "user_id": user_id,
                "symbol": symbol,
                "duration_ms": f"{duration_ms:.2f}",
            },
        )

        # Transform oracle response to match expected schema
        # This is a workaround until vlt-cli has dedicated definition command
        return {
            "found": "answer" in result and len(result.get("sources", [])) > 0,
            "definitions": result.get("sources", []),
            "lookup_method": "semantic",  # Since we're using oracle as workaround
        }

    except OracleBridgeError as exc:
        logger.error(f"Find definition failed: {exc.message}", extra=exc.details)
        return {
            "error": exc.message,
            "details": exc.details,
            "found": False,
            "definitions": [],
            "lookup_method": "error",
        }


@mcp.tool(
    name="find_references",
    description=(
        "Find all usages/references of a symbol. Uses call graph and code intelligence to find "
        "where a function is called, where a class is instantiated, etc. Returns all locations "
        "with context. Use when you need to understand impact of changes or find usage patterns."
    ),
)
async def find_references(
    symbol: str = Field(
        ...,
        description="Symbol name to find references for",
        min_length=1,
        max_length=256,
    ),
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum references to return",
    ),
    include_definition: bool = Field(
        default=False,
        description="Include the definition in results",
    ),
    reference_type: str = Field(
        default="all",
        description="Type of references to find",
    ),
) -> Dict[str, Any]:
    """Find all references to a symbol."""
    start_time = time.time()
    user_id = _current_user_id()

    try:
        result = await oracle_bridge.find_references(
            symbol=symbol,
            limit=limit,
            include_definition=include_definition,
            reference_type=reference_type,
        )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "MCP tool called",
            extra={
                "tool_name": "find_references",
                "user_id": user_id,
                "symbol": symbol,
                "duration_ms": f"{duration_ms:.2f}",
            },
        )

        # Transform oracle response to match expected schema
        sources = result.get("sources", [])
        return {
            "found": len(sources) > 0,
            "references": sources,
            "total_count": len(sources),
            "lookup_method": "semantic",  # Since we're using oracle as workaround
        }

    except OracleBridgeError as exc:
        logger.error(f"Find references failed: {exc.message}", extra=exc.details)
        return {
            "error": exc.message,
            "details": exc.details,
            "found": False,
            "references": [],
            "total_count": 0,
            "lookup_method": "error",
        }


@mcp.tool(
    name="get_repo_map",
    description=(
        "Get codebase structure overview. Returns an Aider-style repository map showing files, "
        "classes, functions, and their signatures. Prioritizes important symbols using graph centrality. "
        "Use when you need to understand the overall codebase structure or navigate to relevant areas."
    ),
)
async def get_repo_map(
    scope: Optional[str] = Field(
        default=None,
        description="Subdirectory to focus on (e.g., 'src/api/'). Omit for entire repo.",
    ),
    max_tokens: int = Field(
        default=4000,
        ge=1000,
        le=16000,
        description="Maximum tokens for the map (controls pruning)",
    ),
    include_signatures: bool = Field(
        default=True,
        description="Include function/method signatures",
    ),
    include_docstrings: bool = Field(
        default=False,
        description="Include brief docstrings",
    ),
) -> Dict[str, Any]:
    """Get repository structure map."""
    start_time = time.time()
    user_id = _current_user_id()

    try:
        result = await oracle_bridge.get_repo_map(
            scope=scope,
            max_tokens=max_tokens,
            include_signatures=include_signatures,
            include_docstrings=include_docstrings,
        )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "MCP tool called",
            extra={
                "tool_name": "get_repo_map",
                "user_id": user_id,
                "scope": scope or "entire_repo",
                "duration_ms": f"{duration_ms:.2f}",
            },
        )

        return result

    except OracleBridgeError as exc:
        logger.error(f"Get repo map failed: {exc.message}", extra=exc.details)
        return {
            "error": exc.message,
            "details": exc.details,
            "map": "Error generating repository map",
            "stats": {
                "files_included": 0,
                "symbols_included": 0,
                "symbols_total": 0,
                "token_count": 0,
            },
            "scope_applied": scope,
        }


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
