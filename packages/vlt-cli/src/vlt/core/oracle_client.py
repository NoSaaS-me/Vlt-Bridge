"""Oracle HTTP client for backend API (thin client mode).

This module provides an HTTP client for Oracle queries that allows the CLI
to use the backend server for Oracle context retrieval instead of local processing.

When the backend is available:
- Uses shared context tree with web UI
- Leverages server-side CodeRAG and vault indexing
- Centralized LLM calls (no local API keys needed)

When offline or backend unavailable:
- Falls back to local OracleOrchestrator
- Uses local indexes and API keys

Configuration:
- VLT_VAULT_URL: Backend server URL (default: http://localhost:8000)
- VLT_SYNC_TOKEN: Authentication token for backend
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Response models (aligned with backend API)

class SourceReference(BaseModel):
    """Source citation from oracle retrieval."""
    path: str
    source_type: str  # "code", "vault", "thread", "repomap"
    line: Optional[int] = None
    snippet: Optional[str] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class OracleStreamChunk(BaseModel):
    """Server-sent event chunk for streaming responses."""
    type: str  # "thinking", "content", "source", "tool_call", "tool_result", "done", "error"
    content: Optional[str] = None
    source: Optional[SourceReference] = None
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OracleResponse(BaseModel):
    """Non-streaming oracle response."""
    answer: str
    sources: List[SourceReference] = Field(default_factory=list)
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    retrieval_traces: Optional[Dict[str, Any]] = None


class ContextNode(BaseModel):
    """A node in the context tree (represents a conversation turn)."""
    id: str
    root_id: str
    parent_id: Optional[str] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    created_at: datetime
    question: str = ""
    answer: str = ""
    tokens_used: int = 0
    model_used: Optional[str] = None
    label: Optional[str] = None
    is_checkpoint: bool = False
    is_root: bool = False
    child_count: int = 0


class ContextTree(BaseModel):
    """Context tree representing conversation history."""
    root_id: str
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    current_node_id: str
    created_at: datetime
    last_activity: datetime
    node_count: int = 1
    max_nodes: int = 30
    label: Optional[str] = None


class ContextTreesResponse(BaseModel):
    """Response from GET /api/oracle/context/trees."""
    trees: List[ContextTree] = Field(default_factory=list)
    active_tree: Optional[ContextTree] = None


class ContextTreeData(BaseModel):
    """Full tree data including nodes."""
    trees: List[ContextTree] = Field(default_factory=list)
    active_tree: Optional[ContextTree] = None
    nodes: Dict[str, ContextNode] = Field(default_factory=dict)
    path_to_head: List[str] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    """A message in conversation history."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[datetime] = None
    sources: Optional[List[SourceReference]] = None


class OracleClient:
    """HTTP client for Oracle backend API.

    This client enables thin-client mode where the CLI delegates Oracle queries
    to the backend server instead of running them locally. This allows:

    1. Shared context with web UI (same conversation tree)
    2. Server-side indexing and retrieval
    3. Centralized API key management

    Example usage:
        client = OracleClient()

        if client.is_available():
            # Use backend
            async for chunk in client.query_stream("How does auth work?"):
                print(chunk.content)
        else:
            # Fall back to local orchestrator
            ...
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """Initialize Oracle client.

        Args:
            base_url: Backend server URL. If None, loaded from settings.
            token: Authentication token. If None, loaded from settings.
            timeout: Request timeout in seconds.
        """
        from ..config import settings

        self.base_url = base_url or settings.vault_url
        self.token = token or settings.sync_token
        self.timeout = timeout

        # Remove trailing slash from base URL
        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

        logger.debug(f"OracleClient initialized with base_url={self.base_url}")

    def _headers(self) -> Dict[str, str]:
        """Get request headers with auth token."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def is_available(self, timeout: float = 2.0) -> bool:
        """Check if backend is reachable.

        Performs a quick health check against the backend's system info endpoint.

        Args:
            timeout: Timeout for health check in seconds.

        Returns:
            True if backend is available and responding.
        """
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(
                    f"{self.base_url}/api/system/info",
                    headers=self._headers(),
                )
                return response.status_code == 200
        except httpx.TimeoutException:
            logger.debug("Backend health check timed out")
            return False
        except httpx.RequestError as e:
            logger.debug(f"Backend unavailable: {e}")
            return False
        except Exception as e:
            logger.debug(f"Backend check failed: {e}")
            return False

    async def is_available_async(self, timeout: float = 2.0) -> bool:
        """Async version of is_available."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/system/info",
                    headers=self._headers(),
                )
                return response.status_code == 200
        except Exception:
            return False

    async def query(
        self,
        question: str,
        sources: Optional[List[str]] = None,
        explain: bool = False,
        model: Optional[str] = None,
        thinking: bool = False,
        max_tokens: int = 16000,
        context_id: Optional[str] = None,
    ) -> OracleResponse:
        """Query oracle (non-streaming).

        Args:
            question: Natural language question.
            sources: Filter sources: ["vault", "code", "threads"]. None = all.
            explain: Include retrieval traces for debugging.
            model: Override LLM model.
            thinking: Enable extended thinking mode.
            max_tokens: Maximum context tokens.
            context_id: ID of context tree to use (for conversation continuity).

        Returns:
            OracleResponse with answer and sources.

        Raises:
            httpx.HTTPError: On network or API errors.
            ValueError: If no auth token configured.
        """
        if not self.token:
            raise ValueError("No auth token configured. Run: vlt login")

        payload = {
            "question": question,
            "sources": sources,
            "explain": explain,
            "model": model,
            "thinking": thinking,
            "max_tokens": max_tokens,
            "context_id": context_id,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/oracle",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()

            return OracleResponse(
                answer=data.get("answer", ""),
                sources=[SourceReference(**s) for s in data.get("sources", [])],
                tokens_used=data.get("tokens_used"),
                model_used=data.get("model_used"),
                retrieval_traces=data.get("retrieval_traces"),
            )

    async def query_stream(
        self,
        question: str,
        sources: Optional[List[str]] = None,
        explain: bool = False,
        model: Optional[str] = None,
        thinking: bool = False,
        max_tokens: int = 16000,
        context_id: Optional[str] = None,
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """Query oracle with streaming response (SSE).

        Streams Server-Sent Events from the backend Oracle API.

        Args:
            question: Natural language question.
            sources: Filter sources: ["vault", "code", "threads"]. None = all.
            explain: Include retrieval traces for debugging.
            model: Override LLM model.
            thinking: Enable extended thinking mode.
            max_tokens: Maximum context tokens.
            context_id: ID of context tree to use (for conversation continuity).

        Yields:
            OracleStreamChunk objects with type, content, sources, etc.

        Raises:
            httpx.HTTPError: On network or API errors.
            ValueError: If no auth token configured.
        """
        if not self.token:
            yield OracleStreamChunk(
                type="error",
                error="No auth token configured. Run: vlt login"
            )
            return

        payload = {
            "question": question,
            "sources": sources,
            "explain": explain,
            "model": model,
            "thinking": thinking,
            "max_tokens": max_tokens,
            "context_id": context_id,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/oracle/stream",
                    json=payload,
                    headers={
                        **self._headers(),
                        "Accept": "text/event-stream",
                    },
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield OracleStreamChunk(
                            type="error",
                            error=f"HTTP {response.status_code}: {error_text.decode()[:200]}"
                        )
                        return

                    # Parse SSE stream
                    async for line in response.aiter_lines():
                        if not line or line.startswith(":"):
                            # Skip empty lines and comments
                            continue

                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            if data_str.strip():
                                try:
                                    data = json.loads(data_str)
                                    chunk = OracleStreamChunk(**data)
                                    yield chunk

                                    # Stop on done or error
                                    if chunk.type in ("done", "error"):
                                        return
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse SSE data: {e}")
                                    continue

        except httpx.TimeoutException:
            yield OracleStreamChunk(
                type="error",
                error=f"Request timed out after {self.timeout}s"
            )
        except httpx.RequestError as e:
            yield OracleStreamChunk(
                type="error",
                error=f"Network error: {str(e)}"
            )
        except Exception as e:
            logger.exception("Error in query_stream")
            yield OracleStreamChunk(
                type="error",
                error=f"Unexpected error: {str(e)}"
            )

    async def cancel_query(self) -> bool:
        """Cancel the active Oracle query for this user.

        Returns:
            True if a session was cancelled, False if no active session.
        """
        if not self.token:
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/oracle/cancel",
                    headers=self._headers(),
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("status") == "cancelled"
                return False
        except Exception as e:
            logger.debug(f"Failed to cancel query: {e}")
            return False

    # =========================================================================
    # Conversation History
    # =========================================================================

    async def get_history(self) -> List[ConversationMessage]:
        """Get conversation history from backend.

        Returns:
            List of conversation messages.
        """
        if not self.token:
            return []

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/oracle/history",
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()

                messages = []
                for msg in data.get("messages", []):
                    messages.append(ConversationMessage(
                        role=msg.get("role"),
                        content=msg.get("content"),
                        timestamp=msg.get("timestamp"),
                        sources=[SourceReference(**s) for s in msg.get("sources", [])] if msg.get("sources") else None,
                    ))
                return messages
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []

    async def clear_history(self) -> bool:
        """Clear conversation history.

        Returns:
            True if history was cleared successfully.
        """
        if not self.token:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.delete(
                    f"{self.base_url}/api/oracle/history",
                    headers=self._headers(),
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return False

    # =========================================================================
    # Context Tree Operations
    # =========================================================================

    async def get_trees(self) -> ContextTreesResponse:
        """Get all context trees for the user.

        Returns:
            ContextTreesResponse with trees and active_tree.
        """
        if not self.token:
            return ContextTreesResponse()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/oracle/context/trees",
                    headers=self._headers(),
                )
                if response.status_code == 404:
                    # Endpoint not implemented yet
                    return ContextTreesResponse()
                response.raise_for_status()
                data = response.json()
                return ContextTreesResponse(**data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return ContextTreesResponse()
            raise
        except Exception as e:
            logger.error(f"Failed to get trees: {e}")
            return ContextTreesResponse()

    async def get_tree(self, root_id: str) -> Optional[ContextTreeData]:
        """Get a specific context tree with all its nodes.

        Args:
            root_id: Root ID of the tree.

        Returns:
            ContextTreeData with full tree and node info.
        """
        if not self.token:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/oracle/context/trees/{root_id}",
                    headers=self._headers(),
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return ContextTreeData(**response.json())
        except Exception as e:
            logger.error(f"Failed to get tree {root_id}: {e}")
            return None

    async def create_tree(self, label: Optional[str] = None) -> Optional[ContextTree]:
        """Create a new context tree (start new conversation).

        Args:
            label: Optional label for the tree.

        Returns:
            New ContextTree if created, None on error.
        """
        if not self.token:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/oracle/context/trees",
                    json={"label": label} if label else {},
                    headers=self._headers(),
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return ContextTree(**response.json())
        except Exception as e:
            logger.error(f"Failed to create tree: {e}")
            return None

    async def delete_tree(self, root_id: str) -> bool:
        """Delete a context tree.

        Args:
            root_id: Root ID of the tree to delete.

        Returns:
            True if deleted, False otherwise.
        """
        if not self.token:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.delete(
                    f"{self.base_url}/api/oracle/context/trees/{root_id}",
                    headers=self._headers(),
                )
                return response.status_code in (200, 204)
        except Exception as e:
            logger.error(f"Failed to delete tree {root_id}: {e}")
            return False

    async def activate_tree(self, root_id: str) -> bool:
        """Set a tree as the active context tree.

        Args:
            root_id: Root ID of the tree to activate.

        Returns:
            True if successful, False otherwise.
        """
        if not self.token:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/oracle/context/trees/{root_id}/activate",
                    headers=self._headers(),
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to activate tree {root_id}: {e}")
            return False

    async def checkout(self, node_id: str) -> Optional[ContextTree]:
        """Switch to a different node in the context tree.

        Args:
            node_id: ID of node to switch to.

        Returns:
            Updated ContextTree if successful, None on error.
        """
        if not self.token:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/oracle/context/nodes/{node_id}/checkout",
                    headers=self._headers(),
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return ContextTree(**response.json())
        except Exception as e:
            logger.error(f"Failed to checkout node: {e}")
            return None

    async def label_node(self, node_id: str, label: str) -> Optional[ContextNode]:
        """Set a label on a context node.

        Args:
            node_id: ID of the node to label.
            label: Label text.

        Returns:
            Updated ContextNode if successful, None on error.
        """
        if not self.token:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.put(
                    f"{self.base_url}/api/oracle/context/nodes/{node_id}/label",
                    json={"label": label},
                    headers=self._headers(),
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return ContextNode(**response.json())
        except Exception as e:
            logger.error(f"Failed to label node {node_id}: {e}")
            return None

    async def set_checkpoint(self, node_id: str, is_checkpoint: bool) -> Optional[ContextNode]:
        """Set checkpoint status on a context node.

        Checkpoints are protected from pruning.

        Args:
            node_id: ID of the node.
            is_checkpoint: Whether this node should be a checkpoint.

        Returns:
            Updated ContextNode if successful, None on error.
        """
        if not self.token:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.put(
                    f"{self.base_url}/api/oracle/context/nodes/{node_id}/checkpoint",
                    json={"is_checkpoint": is_checkpoint},
                    headers=self._headers(),
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return ContextNode(**response.json())
        except Exception as e:
            logger.error(f"Failed to set checkpoint on node {node_id}: {e}")
            return None

    async def prune_tree(self, root_id: str) -> Dict[str, Any]:
        """Prune old non-checkpoint nodes from a tree.

        Args:
            root_id: Root ID of the tree to prune.

        Returns:
            Prune result with nodes_removed count.
        """
        if not self.token:
            return {"nodes_removed": 0}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/oracle/context/trees/{root_id}/prune",
                    headers=self._headers(),
                )
                if response.status_code == 404:
                    return {"nodes_removed": 0}
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to prune tree {root_id}: {e}")
            return {"nodes_removed": 0}

    async def get_active_context(self) -> Optional[ContextTree]:
        """Get the current active context tree.

        Returns:
            Active ContextTree if any, None otherwise.
        """
        response = await self.get_trees()
        return response.active_tree

    async def get_context_id(self) -> Optional[str]:
        """Get the ID of the active context for queries.

        Returns:
            Root ID of active tree if any, None otherwise.
        """
        active = await self.get_active_context()
        return active.root_id if active else None


# Synchronous wrapper for CLI use
def oracle_query_sync(
    question: str,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    sources: Optional[List[str]] = None,
    explain: bool = False,
    model: Optional[str] = None,
    thinking: bool = False,
    max_tokens: int = 16000,
) -> OracleResponse:
    """Synchronous wrapper for oracle query.

    Convenience function for use in CLI commands.
    """
    client = OracleClient(base_url=base_url, token=token)
    return asyncio.run(client.query(
        question=question,
        sources=sources,
        explain=explain,
        model=model,
        thinking=thinking,
        max_tokens=max_tokens,
    ))
