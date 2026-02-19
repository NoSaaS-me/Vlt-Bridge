"""Vlt Unified MCP Server — tool modules.

Each sub-module exposes a register_*_tools(mcp: FastMCP) function
that decorates all tools in that group onto the server instance.
"""

from typing import Any


def _ok(**data: Any) -> dict:
    """Return a structured success response."""
    return {"status": "ok", **data}


def _err(code: str, message: str) -> dict:
    """Return a structured error response."""
    return {"status": "error", "code": code, "message": message}


__all__ = ["_ok", "_err"]
