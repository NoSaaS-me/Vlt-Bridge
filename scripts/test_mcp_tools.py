#!/usr/bin/env python3
"""Exercise all MCP tools exposed by the hosted server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import Any, Dict

from fastmcp.client import Client, StreamableHttpTransport
from fastmcp.client.client import ToolError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test MCP HTTP tools end-to-end")
    parser.add_argument(
        "--url",
        default=os.environ.get("MCP_URL", "https://bigwolfe-document-mcp.hf.space/mcp"),
        help="Hosted MCP endpoint base URL",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("MCP_TOKEN"),
        help="Bearer token for authentication (or set MCP_TOKEN env variable)",
    )
    parser.add_argument(
        "--note",
        default=f"mcp-test-{uuid.uuid4().hex}.md",
        help="Temporary note path to create and delete during the test",
    )
    return parser


async def exercise_tools(url: str, token: str, note_path: str) -> Dict[str, Any]:
    transport = StreamableHttpTransport(url=url, headers={"Authorization": f"Bearer {token}"})
    async with Client(transport, name="multi-tenant-audit") as client:
        results: Dict[str, Any] = {}

        tools = await client.list_tools()
        results["list_tools"] = [tool.name for tool in tools]

        results["list_notes_before"] = await client.call_tool("list_notes", {})

        # Read the welcome note if it exists
        try:
            results["read_note"] = await client.call_tool(
                "read_note", {"path": "Welcome.md"}
            )
        except ToolError as exc:
            results["read_note_error"] = str(exc)

        # Create a temporary note
        body = "# MCP Test\n\nThis note was created by the MCP HTTP audit script."
        results["write_note"] = await client.call_tool(
            "write_note",
            {
                "path": note_path,
                "body": body,
                "title": "MCP Test",
                "metadata": {"tags": ["mcp", "audit"]},
            },
        )

        # Search for the word "MCP"
        results["search_notes"] = await client.call_tool(
            "search_notes", {"query": "MCP", "limit": 10}
        )

        # Fetch backlinks for the welcome note (if present)
        try:
            results["get_backlinks"] = await client.call_tool(
                "get_backlinks", {"path": "Welcome.md"}
            )
        except ToolError as exc:
            results["get_backlinks_error"] = str(exc)

        # Fetch tags
        results["get_tags"] = await client.call_tool("get_tags", {})

        # Delete the temporary note and show list afterwards
        results["delete_note"] = await client.call_tool(
            "delete_note", {"path": note_path}
        )
        results["list_notes_after"] = await client.call_tool("list_notes", {})

        return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.token:
        parser.error("Bearer token must be provided via --token or MCP_TOKEN env variable")

    try:
        results = asyncio.run(exercise_tools(args.url, args.token, args.note))
    except Exception as exc:  # pragma: no cover
        print(f"Error exercising MCP tools: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    json.dump(results, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
