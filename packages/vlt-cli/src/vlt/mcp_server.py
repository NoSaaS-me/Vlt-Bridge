"""Vlt Unified MCP Server.

Exposes the full vlt capability surface to AI agents via Model Context Protocol.
Runs as a STDIO server, spawned by Claude Desktop/Code on demand.

Usage:
    vlt-mcp                    # STDIO (default) — used by Claude Desktop/Code
    vlt-mcp --http             # HTTP transport (for remote clients)
    vlt-mcp --check            # Health check, print JSON status, exit 0

Configuration (via profile .env or environment variables):
    VLT_PROFILE          Override the active profile (default: "default")
    VLT_ORACLE_ENABLED   Enable/disable oracle tools (default: true)

Claude Code global registration (one-time setup):
    claude mcp add --scope user vlt vlt-mcp
"""

from __future__ import annotations

import json
import logging
import sys

logger = logging.getLogger(__name__)


def create_server():
    """Create and configure the MCP server with all tool groups registered."""
    from fastmcp import FastMCP

    mcp = FastMCP(
        "vlt",
        instructions=(
            "VLT: Persistent cognitive state, code intelligence, and documentation for AI agents.\n\n"

            "QUICK START:\n"
            "1. Call vlt_status — see projects, daemon status, backend health.\n"
            "2. Call vlt_project_detect — auto-detect project from working directory.\n\n"

            "TOOL GROUPS & WORKFLOWS:\n\n"

            "THREADS (reasoning memory):\n"
            "  vlt_thread_create → vlt_thread_push → vlt_thread_read\n"
            "  Push is <50ms — use liberally to offload context.\n"
            "  vlt_thread_seek searches across all threads (semantic + keyword fallback).\n"
            "  vlt_thread_list shows all threads in a project.\n\n"

            "CODE INTELLIGENCE:\n"
            "  vlt_code_init(project_id, path) → indexes a codebase (runs async).\n"
            "  vlt_code_status(project_id) → poll until indexed=true before searching.\n"
            "  vlt_code_search(query, project_id) → hybrid BM25 search over indexed code.\n"
            "  vlt_code_map(project_id) → compact repo structure overview.\n"
            "  vlt_code_lookup(symbol, project_id) → find where a symbol is defined.\n"
            "  IMPORTANT: You must call vlt_code_init and wait for indexing to complete\n"
            "  before vlt_code_search/map/lookup will work.\n\n"

            "VAULT NOTES (Markdown docs):\n"
            "  vlt_note_write/read/search/list/backlinks — CRUD for Markdown notes.\n"
            "  Requires the Document-MCP backend to be running.\n"
            "  Paths must include .md extension (e.g. 'docs/api-design.md').\n\n"

            "ORACLE (AI-powered codebase Q&A):\n"
            "  1. vlt_oracle_status — check if oracle is enabled + configured.\n"
            "     If guidance field is set, oracle is not ready — follow the guidance.\n"
            "  2. vlt_oracle_query(query) — ask questions about your codebase.\n"
            "     Pass context_id from a previous response for multi-turn conversation.\n"
            "     Timeout: 60s. Returns a single answer (not streamed).\n\n"

            "CONNECTORS (external services — email, APIs, etc.):\n"
            "  1. connector_list — discover available connectors and their actions.\n"
            "  2. connector_actions(connector) — get parameter schemas for a connector's actions.\n"
            "  3. connector_call(connector, action, params) — execute an action.\n"
            "  Use 'composio:appname' prefix for Composio integrations (e.g. 'composio:gmail').\n"
            "  IMPORTANT: params must be a JSON string, not a dict.\n\n"

            "CRONBAN (scheduled tasks):\n"
            "  cronban_sessions — list active Claude sessions (pick a target).\n"
            "  cronban_create — schedule recurring or one-off prompt injections.\n"
            "  cronban_list/fire/pause/resume/delete — manage scheduled triggers.\n"
        ),
    )

    # Core tools — always available
    from vlt.mcp.thread_tools import register_thread_tools
    from vlt.mcp.meta_tools import register_meta_tools

    register_thread_tools(mcp)
    register_meta_tools(mcp)

    # Optional tool groups — loaded when their modules exist
    try:
        from vlt.mcp.code_tools import register_code_tools
        register_code_tools(mcp)
    except ImportError:
        logger.debug("code_tools not available")

    try:
        from vlt.mcp.vault_tools import register_vault_tools
        register_vault_tools(mcp)
    except ImportError:
        logger.debug("vault_tools not available")

    try:
        from vlt.mcp.oracle_tools import register_oracle_tools
        register_oracle_tools(mcp)
    except ImportError:
        logger.debug("oracle_tools not available")

    try:
        from vlt.mcp.cronban_tools import register_cronban_tools
        register_cronban_tools(mcp)
    except ImportError:
        logger.debug("cronban_tools not available")

    try:
        from vlt.mcp.connector_tools import register_connector_tools
        register_connector_tools(mcp)
    except ImportError:
        logger.debug("connector_tools not available")

    try:
        from vlt.mcp.artifact_tools import register_artifact_tools
        register_artifact_tools(mcp)
    except ImportError:
        logger.debug("artifact_tools not available")

    return mcp


def _health_check() -> None:
    """Run startup health check and print JSON status to stdout, then exit."""
    status: dict = {"status": "ok", "checks": {}}

    # Check DB connection
    try:
        from vlt.config import get_settings
        settings = get_settings()
        db_path = settings.get_db_path()
        status["checks"]["settings"] = "ok"
        status["checks"]["db_path"] = str(db_path)
        status["checks"]["db_exists"] = db_path.exists()
    except Exception as e:
        status["checks"]["settings"] = f"error: {e}"
        status["status"] = "degraded"

    # Check oracle_enabled
    try:
        from vlt.config import get_settings
        settings = get_settings()
        oracle_enabled = getattr(settings, "oracle_enabled", True)
        status["checks"]["oracle_enabled"] = oracle_enabled
    except Exception as e:
        status["checks"]["oracle_enabled"] = f"error: {e}"

    # Check daemon (sync HTTP check — avoids event loop in CLI context)
    try:
        import httpx
        from vlt.config import get_settings
        _s = get_settings()
        r = httpx.get(f"{_s.daemon_url}/health", timeout=1.0)
        status["checks"]["daemon_running"] = r.status_code == 200
    except Exception:
        status["checks"]["daemon_running"] = False

    print(json.dumps(status, indent=2))
    sys.exit(0)


def main() -> None:
    """Entry point for the vlt-mcp command."""
    # Configure logging to stderr — stdout is reserved for STDIO MCP transport
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if "--check" in sys.argv:
        _health_check()

    transport = "stdio"
    if "--http" in sys.argv:
        transport = "http"

    server = create_server()
    server.run(transport=transport)


if __name__ == "__main__":
    main()
