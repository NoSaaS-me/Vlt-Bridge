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
            "TOOL GROUPS:\n"
            "- vlt_thread_*: Reasoning chain memory. Log decisions, track progress, search past reasoning. "
            "Thread push is <50ms — use liberally to offload context.\n"
            "- vlt_code_*: Code intelligence. Semantic search, repo map, symbol lookup. "
            "Call vlt_code_init first on any new project.\n"
            "- vlt_note_*: Markdown vault notes. Requires Document-MCP backend running.\n"
            "- vlt_oracle_*: AI-powered multi-source answers about your codebase.\n"
            "- vlt_status, vlt_project_detect: Health check and project auto-detection.\n\n"
            "QUICK START: Call vlt_status to see what's available. "
            "Call vlt_project_detect to find your current project context automatically."
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
