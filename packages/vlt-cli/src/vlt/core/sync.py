"""Thread Sync Client - Syncs threads to Document-MCP backend."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .identity import load_oracle_config

logger = logging.getLogger(__name__)


class SyncQueueItem(BaseModel):
    """Queued entry for retry when sync fails."""
    thread_id: str
    project_id: str
    name: str
    entry: Dict[str, Any]  # entry_id, sequence_id, content, author, timestamp
    attempts: int = 0
    last_attempt: Optional[str] = None
    error: Optional[str] = None


class SyncResult(BaseModel):
    """Result of a sync operation."""
    thread_id: str
    synced_count: int
    last_synced_sequence: int
    success: bool
    error: Optional[str] = None


class SummarizeResult(BaseModel):
    """Result of a server-side summarization request."""
    thread_id: str
    summary: str
    model: Optional[str] = None
    tokens_used: int = 0
    success: bool
    error: Optional[str] = None


class ThreadSyncClient:
    """Client for syncing threads to Document-MCP backend."""

    QUEUE_FILE = Path.home() / ".vlt" / "sync_queue.json"
    MAX_RETRIES = 3

    def __init__(self, vault_url: Optional[str] = None, sync_token: Optional[str] = None):
        """Initialize sync client."""
        # Try to load from identity config
        oracle_config = load_oracle_config()
        self.vault_url = vault_url or (oracle_config.vault_url if oracle_config else "http://localhost:8000")
        self.sync_token = sync_token

        # Load token from config if not provided
        if not self.sync_token:
            from ..config import settings
            self.sync_token = settings.sync_token

    async def sync_entries(
        self,
        thread_id: str,
        project_id: str,
        name: str,
        entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Sync thread entries to the backend.

        Args:
            thread_id: Thread identifier
            project_id: Project identifier
            name: Thread display name
            entries: List of entry dicts with entry_id, sequence_id, content, author, timestamp

        Returns:
            Response with thread_id, synced_count, last_synced_sequence

        Raises:
            httpx.HTTPError on network failure
        """
        if not self.sync_token:
            logger.warning("No sync token configured, skipping sync")
            raise ValueError("No sync token configured. Run 'vlt config set sync.token <token>'")

        url = f"{self.vault_url}/api/threads/sync"

        payload = {
            "thread_id": thread_id,
            "project_id": project_id,
            "name": name,
            "status": "active",
            "entries": entries,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.sync_token}"},
            )
            response.raise_for_status()
            return response.json()

    async def get_sync_status(self, thread_id: str) -> Dict[str, Any]:
        """Get sync status for a thread from the backend."""
        if not self.sync_token:
            raise ValueError("No sync token configured")

        url = f"{self.vault_url}/api/threads/{thread_id}/status"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {self.sync_token}"},
            )
            response.raise_for_status()
            return response.json()

    async def sync_thread_full(
        self,
        thread_id: str,
        project_id: str,
        name: str,
        db: Session,
    ) -> SyncResult:
        """
        Sync a complete thread with all its entries to the backend.

        This method fetches all entries from the local database and sends them
        to the server. The server uses upsert logic, so it's idempotent.

        Args:
            thread_id: Thread identifier
            project_id: Project identifier
            name: Thread display name
            db: Database session to fetch entries from

        Returns:
            SyncResult with sync status
        """
        if not self.sync_token:
            return SyncResult(
                thread_id=thread_id,
                synced_count=0,
                last_synced_sequence=-1,
                success=False,
                error="No sync token configured. Run: vlt config set-key <token>",
            )

        # Import here to avoid circular imports
        from .models import Node

        # Get all nodes for this thread
        nodes = db.scalars(
            select(Node)
            .where(Node.thread_id == thread_id)
            .order_by(Node.sequence_id)
        ).all()

        if not nodes:
            return SyncResult(
                thread_id=thread_id,
                synced_count=0,
                last_synced_sequence=-1,
                success=True,
                error=None,
            )

        # Format entries for sync
        entries = [
            {
                "entry_id": node.id,
                "sequence_id": node.sequence_id,
                "content": node.content,
                "author": node.author,
                "timestamp": node.timestamp.isoformat() if node.timestamp else datetime.utcnow().isoformat(),
            }
            for node in nodes
        ]

        try:
            result = await self.sync_entries(
                thread_id=thread_id,
                project_id=project_id,
                name=name,
                entries=entries,
            )
            return SyncResult(
                thread_id=thread_id,
                synced_count=result.get("synced_count", len(entries)),
                last_synced_sequence=result.get("last_synced_sequence", nodes[-1].sequence_id),
                success=True,
                error=None,
            )
        except Exception as e:
            logger.error(f"Failed to sync thread {thread_id}: {e}")
            return SyncResult(
                thread_id=thread_id,
                synced_count=0,
                last_synced_sequence=-1,
                success=False,
                error=str(e),
            )

    async def request_summarize(
        self,
        thread_id: str,
        current_summary: Optional[str] = None,
    ) -> SummarizeResult:
        """
        Request server-side summarization for a thread.

        The server handles all LLM operations. This method just requests
        the summarization and returns the result.

        Args:
            thread_id: Thread identifier
            current_summary: Existing summary for incremental updates (optional)

        Returns:
            SummarizeResult with summary and metadata
        """
        if not self.sync_token:
            return SummarizeResult(
                thread_id=thread_id,
                summary="",
                success=False,
                error="No sync token configured. Run: vlt config set-key <token>",
            )

        url = f"{self.vault_url}/api/threads/{thread_id}/summarize"
        headers = {
            "Authorization": f"Bearer {self.sync_token}",
            "Content-Type": "application/json",
        }

        body = {}
        if current_summary:
            body["current_summary"] = current_summary

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()

                return SummarizeResult(
                    thread_id=thread_id,
                    summary=data.get("summary", ""),
                    model=data.get("model"),
                    tokens_used=data.get("tokens_used", 0),
                    success=data.get("success", True),
                    error=data.get("error"),
                )
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text[:200] if e.response.text else str(e)
            logger.error(f"Server summarization failed: {e.response.status_code} - {error_detail}")
            return SummarizeResult(
                thread_id=thread_id,
                summary="",
                success=False,
                error=f"HTTP {e.response.status_code}: {error_detail}",
            )
        except Exception as e:
            logger.error(f"Server summarization failed: {e}")
            return SummarizeResult(
                thread_id=thread_id,
                summary="",
                success=False,
                error=str(e),
            )

    def _load_queue(self) -> List[SyncQueueItem]:
        """Load sync queue from file."""
        if not self.QUEUE_FILE.exists():
            return []

        try:
            with open(self.QUEUE_FILE, "r") as f:
                data = json.load(f)
                return [SyncQueueItem(**item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load sync queue: {e}")
            return []

    def _save_queue(self, queue: List[SyncQueueItem]) -> None:
        """Save sync queue to file."""
        self.QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(self.QUEUE_FILE, "w") as f:
            json.dump([item.model_dump() for item in queue], f, indent=2)

    def queue_entry(
        self,
        thread_id: str,
        project_id: str,
        name: str,
        entry: Dict[str, Any],
        error: str,
    ) -> None:
        """Add an entry to the sync queue for retry."""
        queue = self._load_queue()

        # Check if entry already queued
        for item in queue:
            if item.entry.get("entry_id") == entry.get("entry_id"):
                item.attempts += 1
                item.last_attempt = datetime.utcnow().isoformat()
                item.error = error
                self._save_queue(queue)
                return

        # Add new item
        queue.append(SyncQueueItem(
            thread_id=thread_id,
            project_id=project_id,
            name=name,
            entry=entry,
            attempts=1,
            last_attempt=datetime.utcnow().isoformat(),
            error=error,
        ))
        self._save_queue(queue)
        logger.info(f"Queued entry {entry.get('entry_id')} for retry")

    async def retry_queue(self) -> Dict[str, int]:
        """Retry all queued entries. Returns counts of success/failure."""
        queue = self._load_queue()
        if not queue:
            return {"success": 0, "failed": 0, "skipped": 0}

        success = 0
        failed = 0
        skipped = 0
        remaining = []

        for item in queue:
            if item.attempts >= self.MAX_RETRIES:
                skipped += 1
                remaining.append(item)  # Keep for manual review
                continue

            try:
                await self.sync_entries(
                    thread_id=item.thread_id,
                    project_id=item.project_id,
                    name=item.name,
                    entries=[item.entry],
                )
                success += 1
                logger.info(f"Successfully synced queued entry {item.entry.get('entry_id')}")
            except Exception as e:
                item.attempts += 1
                item.last_attempt = datetime.utcnow().isoformat()
                item.error = str(e)
                remaining.append(item)
                failed += 1

        self._save_queue(remaining)
        return {"success": success, "failed": failed, "skipped": skipped}

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        queue = self._load_queue()
        return {
            "pending": len(queue),
            "items": [item.model_dump() for item in queue],
        }


# Helper for sync from thread push
async def sync_thread_entry(
    thread_id: str,
    project_id: str,
    name: str,
    entry_id: str,
    sequence_id: int,
    content: str,
    author: str = "user",
    timestamp: Optional[str] = None,
) -> bool:
    """
    Sync a single thread entry. Returns True on success.
    Queues for retry on failure.
    """
    client = ThreadSyncClient()

    entry = {
        "entry_id": entry_id,
        "sequence_id": sequence_id,
        "content": content,
        "author": author,
        "timestamp": timestamp or datetime.utcnow().isoformat(),
    }

    try:
        await client.sync_entries(
            thread_id=thread_id,
            project_id=project_id,
            name=name,
            entries=[entry],
        )
        logger.debug(f"Synced entry {entry_id} to backend")
        return True
    except ValueError as e:
        # No token configured, skip silently
        logger.debug(f"Sync skipped: {e}")
        return False
    except Exception as e:
        # Queue for retry
        logger.warning(f"Sync failed, queuing for retry: {e}")
        client.queue_entry(
            thread_id=thread_id,
            project_id=project_id,
            name=name,
            entry=entry,
            error=str(e),
        )
        return False


async def sync_all_threads(db: Session, project_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Sync local threads to the server.

    This is used by the librarian to ensure all threads are available
    on the server before requesting summarization.

    Args:
        db: Database session
        project_id: Optional project ID to filter threads. If None, syncs ALL threads.

    Returns:
        Dict with sync statistics:
        {
            "threads_synced": int,
            "threads_failed": int,
            "total_entries": int,
            "errors": list of error messages,
            "project_id": str or None
        }
    """
    from .models import Thread

    client = ThreadSyncClient()

    if not client.sync_token:
        return {
            "threads_synced": 0,
            "threads_failed": 0,
            "total_entries": 0,
            "errors": ["No sync token configured. Run: vlt config set-key <token>"],
            "project_id": project_id,
        }

    # Get threads, optionally filtered by project_id
    query = select(Thread)
    if project_id:
        query = query.where(Thread.project_id == project_id)
    threads = db.scalars(query).all()

    stats = {
        "threads_synced": 0,
        "threads_failed": 0,
        "total_entries": 0,
        "errors": [],
        "project_id": project_id,
    }

    for thread in threads:
        result = await client.sync_thread_full(
            thread_id=thread.id,
            project_id=thread.project_id,
            name=thread.id,  # Use thread ID as name
            db=db,
        )

        if result.success:
            stats["threads_synced"] += 1
            stats["total_entries"] += result.synced_count
        else:
            stats["threads_failed"] += 1
            if result.error:
                stats["errors"].append(f"{thread.id}: {result.error}")

    return stats


async def sync_and_summarize_thread(
    thread_id: str,
    project_id: str,
    db: Session,
    current_summary: Optional[str] = None,
) -> SummarizeResult:
    """
    Sync a thread to the server and request summarization.

    This is the recommended way to get a summary for a thread when
    server-side summarization is configured. It:
    1. Syncs the thread to ensure the server has all entries
    2. Requests server-side summarization
    3. Returns the summary result

    Args:
        thread_id: Thread identifier
        project_id: Project identifier
        db: Database session
        current_summary: Existing summary for incremental updates

    Returns:
        SummarizeResult with summary and metadata
    """
    client = ThreadSyncClient()

    # First, sync the thread
    sync_result = await client.sync_thread_full(
        thread_id=thread_id,
        project_id=project_id,
        name=thread_id,
        db=db,
    )

    if not sync_result.success:
        return SummarizeResult(
            thread_id=thread_id,
            summary=current_summary or "",
            success=False,
            error=f"Sync failed: {sync_result.error}",
        )

    # Then request summarization
    return await client.request_summarize(
        thread_id=thread_id,
        current_summary=current_summary,
    )
