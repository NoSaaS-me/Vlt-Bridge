"""Vault note tools for the vlt MCP server.

HTTP proxy to the Document-MCP backend. All five tools require the backend
to be running. On ConnectError or Timeout, they return a structured
VAULT_UNAVAILABLE error rather than propagating an exception.

Tools registered:
    vlt_note_write      — create or update a Markdown note
    vlt_note_read       — read a note by path
    vlt_note_search     — full-text search across all notes
    vlt_note_list       — list notes in a folder
    vlt_note_backlinks  — find notes that link to a given note
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _get_client():
    """Build an httpx.Client configured for the Document-MCP backend."""
    import httpx
    from vlt.config import get_settings

    settings = get_settings()
    vault_url = getattr(settings, "vault_url", None) or getattr(settings, "sync_url", None)
    sync_token = getattr(settings, "sync_token", None)

    if not vault_url:
        return None, None

    client = httpx.Client(
        base_url=vault_url,
        headers={"Authorization": f"Bearer {sync_token}"} if sync_token else {},
        timeout=10.0,
    )
    return client, vault_url


def _unavailable(vault_url: Optional[str]) -> dict:
    from vlt.mcp import _err

    location = vault_url or "unknown"
    return _err(
        "VAULT_UNAVAILABLE",
        f"Document-MCP backend unreachable at {location}. Start with ./start-dev.sh",
    )


def register_vault_tools(mcp) -> None:
    """Register all vault note tools onto a FastMCP server instance."""

    @mcp.tool()
    def vlt_note_write(
        path: str,
        content: str,
        project_id: Optional[str] = None,
    ) -> dict:
        """Create or update a Markdown note in the vault.

        Args:
            path: Note path relative to vault root (e.g. "docs/api-design.md").
            content: Full Markdown content to write.
            project_id: Project identifier (optional, uses default if omitted).

        Returns:
            {status, path, version, size_bytes}
        """
        from vlt.mcp import _ok, _err

        client, vault_url = _get_client()
        if client is None:
            return _err("VAULT_NOT_CONFIGURED", "vault_url not set. Configure VLT_VAULT_URL.")

        try:
            import httpx

            params = {}
            if project_id:
                params["project_id"] = project_id

            encoded = _encode_path(path)
            # Try update (PUT) first; fall back to create (POST) on 404
            r = client.put(
                f"/api/notes/{encoded}",
                json={"body": content},
                params=params,
            )
            if r.status_code == 404:
                # Note doesn't exist yet — create it
                r = client.post(
                    "/api/notes",
                    json={"note_path": path, "body": content},
                    params=params,
                )
            r.raise_for_status()
            data = r.json()
            return _ok(
                path=data.get("note_path", path),
                version=data.get("version"),
                size_bytes=data.get("size_bytes"),
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            return _unavailable(vault_url)
        except httpx.HTTPStatusError as e:
            return _err("VAULT_ERROR", f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            logger.exception("vlt_note_write failed")
            return _err("INTERNAL_ERROR", str(e))
        finally:
            client.close()

    @mcp.tool()
    def vlt_note_read(
        path: str,
        project_id: Optional[str] = None,
    ) -> dict:
        """Read a Markdown note from the vault.

        Args:
            path: Note path relative to vault root.
            project_id: Project identifier (optional).

        Returns:
            {status, path, content, version, title, tags, updated_at}
        """
        from vlt.mcp import _ok, _err

        client, vault_url = _get_client()
        if client is None:
            return _err("VAULT_NOT_CONFIGURED", "vault_url not set. Configure VLT_VAULT_URL.")

        try:
            import httpx

            params = {}
            if project_id:
                params["project_id"] = project_id

            encoded = _encode_path(path)
            r = client.get(f"/api/notes/{encoded}", params=params)
            if r.status_code == 404:
                return _err("NOTE_NOT_FOUND", f"Note not found: {path}")
            r.raise_for_status()
            data = r.json()
            return _ok(
                path=data.get("note_path", path),
                content=data.get("body", ""),
                version=data.get("version"),
                title=data.get("title"),
                tags=data.get("tags", []),
                updated_at=data.get("updated"),
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            return _unavailable(vault_url)
        except httpx.HTTPStatusError as e:
            return _err("VAULT_ERROR", f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            logger.exception("vlt_note_read failed")
            return _err("INTERNAL_ERROR", str(e))
        finally:
            client.close()

    @mcp.tool()
    def vlt_note_search(
        query: str,
        project_id: Optional[str] = None,
        limit: int = 10,
    ) -> dict:
        """Full-text search across all notes in the vault.

        Args:
            query: Search query (BM25 ranking with title 3x weight).
            project_id: Project identifier (optional).
            limit: Maximum results to return. Default: 10.

        Returns:
            {status, results: [{path, title, snippet, score}], total}
        """
        from vlt.mcp import _ok, _err

        client, vault_url = _get_client()
        if client is None:
            return _err("VAULT_NOT_CONFIGURED", "vault_url not set. Configure VLT_VAULT_URL.")

        try:
            import httpx

            params: dict = {"q": query, "limit": limit}
            if project_id:
                params["project_id"] = project_id

            r = client.get("/api/search", params=params)
            r.raise_for_status()
            data = r.json()
            raw = data if isinstance(data, list) else data.get("results", data)
            # Normalize backend field names to stable MCP contract
            results = [
                {
                    "path": item.get("note_path", item.get("path")),
                    "title": item.get("title"),
                    "snippet": item.get("snippet", ""),
                    "score": item.get("score"),
                }
                for item in raw
            ]
            return _ok(results=results, total=len(results))
        except (httpx.ConnectError, httpx.TimeoutException):
            return _unavailable(vault_url)
        except httpx.HTTPStatusError as e:
            return _err("VAULT_ERROR", f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            logger.exception("vlt_note_search failed")
            return _err("INTERNAL_ERROR", str(e))
        finally:
            client.close()

    @mcp.tool()
    def vlt_note_list(
        folder: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> dict:
        """List notes in the vault, optionally filtered to a folder.

        Args:
            folder: Folder path prefix to filter (optional).
            project_id: Project identifier (optional).

        Returns:
            {status, notes: [{path, title, updated_at}], total}
        """
        from vlt.mcp import _ok, _err

        client, vault_url = _get_client()
        if client is None:
            return _err("VAULT_NOT_CONFIGURED", "vault_url not set. Configure VLT_VAULT_URL.")

        try:
            import httpx

            params: dict = {}
            if folder:
                params["folder"] = folder
            if project_id:
                params["project_id"] = project_id

            r = client.get("/api/notes", params=params)
            r.raise_for_status()
            data = r.json()
            raw = data if isinstance(data, list) else data.get("notes", [])
            # Normalize backend field names to stable MCP contract
            notes = [
                {
                    "path": item.get("note_path", item.get("path")),
                    "title": item.get("title"),
                    "updated_at": item.get("updated", item.get("updated_at")),
                }
                for item in raw
            ]
            return _ok(notes=notes, total=len(notes))
        except (httpx.ConnectError, httpx.TimeoutException):
            return _unavailable(vault_url)
        except httpx.HTTPStatusError as e:
            return _err("VAULT_ERROR", f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            logger.exception("vlt_note_list failed")
            return _err("INTERNAL_ERROR", str(e))
        finally:
            client.close()

    @mcp.tool()
    def vlt_note_backlinks(
        path: str,
        project_id: Optional[str] = None,
    ) -> dict:
        """Find all notes that contain a wikilink pointing to the given note.

        Args:
            path: Target note path.
            project_id: Project identifier (optional).

        Returns:
            {status, path, backlinks: [{source_path, link_text, is_resolved}], total}
        """
        from vlt.mcp import _ok, _err

        client, vault_url = _get_client()
        if client is None:
            return _err("VAULT_NOT_CONFIGURED", "vault_url not set. Configure VLT_VAULT_URL.")

        try:
            import httpx

            params: dict = {}
            if project_id:
                params["project_id"] = project_id

            encoded = _encode_path(path)
            r = client.get(f"/api/backlinks/{encoded}", params=params)
            if r.status_code == 404:
                return _ok(path=path, backlinks=[], total=0)
            r.raise_for_status()
            data = r.json()
            raw = data if isinstance(data, list) else data.get("backlinks", [])
            # Normalize backend field names to stable MCP contract
            backlinks = [
                {
                    "source_path": item.get("note_path", item.get("source_path")),
                    "title": item.get("title"),
                }
                for item in raw
            ]
            return _ok(path=path, backlinks=backlinks, total=len(backlinks))
        except (httpx.ConnectError, httpx.TimeoutException):
            return _unavailable(vault_url)
        except httpx.HTTPStatusError as e:
            return _err("VAULT_ERROR", f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            logger.exception("vlt_note_backlinks failed")
            return _err("INTERNAL_ERROR", str(e))
        finally:
            client.close()


def _encode_path(path: str) -> str:
    """URL-encode a note path for use in route path segments."""
    from urllib.parse import quote

    return quote(path, safe="")
