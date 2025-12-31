"""
VLT Daemon Client - Client for CLI to communicate with daemon.

This client provides a simple interface for CLI commands to talk to the daemon.
All methods have short timeouts and fallback gracefully if the daemon isn't running.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DaemonStatus(BaseModel):
    """Status information from daemon."""
    running: bool
    uptime_seconds: float = 0.0
    backend_url: str = ""
    backend_connected: bool = False
    queue_size: int = 0
    error: Optional[str] = None


class EnqueueResult(BaseModel):
    """Result of enqueue operation."""
    success: bool
    queued: bool = False
    message: str = ""
    queue_size: int = 0
    error: Optional[str] = None


class SummarizeResult(BaseModel):
    """Result of summarization request."""
    success: bool
    summary: str = ""
    model: Optional[str] = None
    tokens_used: int = 0
    error: Optional[str] = None


class RetryResult(BaseModel):
    """Result of retry operation."""
    success: bool
    synced: int = 0
    failed: int = 0
    skipped: int = 0
    error: Optional[str] = None


class DaemonClient:
    """
    Client for communicating with the vlt daemon.

    All operations are designed to fail fast with short timeouts.
    If the daemon isn't running, methods return appropriate fallback values.
    """

    # Short timeout for health checks
    HEALTH_TIMEOUT = 0.5

    # Normal timeout for operations
    OPERATION_TIMEOUT = 5.0

    def __init__(self, url: str = "http://127.0.0.1:8765"):
        """
        Initialize daemon client.

        Args:
            url: Daemon URL (default: http://127.0.0.1:8765)
        """
        self.url = url.rstrip("/")

    async def is_running(self) -> bool:
        """
        Check if the daemon is running.

        Uses a short timeout to fail fast if daemon isn't available.

        Returns:
            True if daemon is running and healthy, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=self.HEALTH_TIMEOUT) as client:
                response = await client.get(f"{self.url}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def get_status(self) -> DaemonStatus:
        """
        Get detailed daemon status.

        Returns:
            DaemonStatus with running state and details.
        """
        try:
            async with httpx.AsyncClient(timeout=self.OPERATION_TIMEOUT) as client:
                response = await client.get(f"{self.url}/health")
                if response.status_code == 200:
                    data = response.json()
                    return DaemonStatus(
                        running=True,
                        uptime_seconds=data.get("uptime_seconds", 0.0),
                        backend_url=data.get("backend_url", ""),
                        backend_connected=data.get("backend_connected", False),
                        queue_size=data.get("queue_size", 0),
                    )
                else:
                    return DaemonStatus(
                        running=False,
                        error=f"Unexpected status code: {response.status_code}",
                    )
        except httpx.ConnectError:
            return DaemonStatus(running=False, error="Daemon not running")
        except httpx.TimeoutException:
            return DaemonStatus(running=False, error="Connection timeout")
        except Exception as e:
            return DaemonStatus(running=False, error=str(e))

    async def enqueue_sync(
        self,
        thread_id: str,
        project_id: str,
        name: str,
        entry: Dict[str, Any],
    ) -> EnqueueResult:
        """
        Queue a sync entry via the daemon.

        The daemon will try to sync immediately if the backend is connected,
        otherwise it queues for background retry.

        Args:
            thread_id: Thread identifier
            project_id: Project identifier
            name: Thread display name
            entry: Entry dict with entry_id, sequence_id, content, author, timestamp

        Returns:
            EnqueueResult with success state and details.
        """
        try:
            async with httpx.AsyncClient(timeout=self.OPERATION_TIMEOUT) as client:
                response = await client.post(
                    f"{self.url}/sync/enqueue",
                    json={
                        "thread_id": thread_id,
                        "project_id": project_id,
                        "name": name,
                        "entry": entry,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    return EnqueueResult(
                        success=True,
                        queued=data.get("queued", False),
                        message=data.get("message", ""),
                        queue_size=data.get("queue_size", 0),
                    )
                else:
                    return EnqueueResult(
                        success=False,
                        error=f"HTTP {response.status_code}: {response.text[:100]}",
                    )
        except httpx.ConnectError:
            return EnqueueResult(success=False, error="Daemon not running")
        except httpx.TimeoutException:
            return EnqueueResult(success=False, error="Connection timeout")
        except Exception as e:
            return EnqueueResult(success=False, error=str(e))

    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get sync queue status from daemon.

        Returns:
            Dict with pending count and items, or empty dict on error.
        """
        try:
            async with httpx.AsyncClient(timeout=self.OPERATION_TIMEOUT) as client:
                response = await client.get(f"{self.url}/sync/status")
                if response.status_code == 200:
                    return response.json()
                return {"pending": 0, "items": [], "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"pending": 0, "items": [], "error": str(e)}

    async def retry_sync(self) -> RetryResult:
        """
        Request daemon to retry all queued sync entries.

        Returns:
            RetryResult with counts and status.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:  # Longer timeout for retry
                response = await client.post(f"{self.url}/sync/retry")
                if response.status_code == 200:
                    data = response.json()
                    return RetryResult(
                        success=True,
                        synced=data.get("success", 0),
                        failed=data.get("failed", 0),
                        skipped=data.get("skipped", 0),
                    )
                else:
                    return RetryResult(
                        success=False,
                        error=f"HTTP {response.status_code}: {response.text[:100]}",
                    )
        except httpx.ConnectError:
            return RetryResult(success=False, error="Daemon not running")
        except httpx.TimeoutException:
            return RetryResult(success=False, error="Operation timeout")
        except Exception as e:
            return RetryResult(success=False, error=str(e))

    async def request_summarize(
        self,
        thread_id: str,
        current_summary: Optional[str] = None,
    ) -> SummarizeResult:
        """
        Request summarization via daemon.

        The daemon proxies this to the backend server.

        Args:
            thread_id: Thread identifier
            current_summary: Existing summary for incremental updates (optional)

        Returns:
            SummarizeResult with summary and metadata.
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:  # Longer timeout for LLM
                response = await client.post(
                    f"{self.url}/summarize/{thread_id}",
                    json={"current_summary": current_summary} if current_summary else {},
                )
                if response.status_code == 200:
                    data = response.json()
                    return SummarizeResult(
                        success=data.get("success", False),
                        summary=data.get("summary", ""),
                        model=data.get("model"),
                        tokens_used=data.get("tokens_used", 0),
                        error=data.get("error"),
                    )
                else:
                    return SummarizeResult(
                        success=False,
                        error=f"HTTP {response.status_code}: {response.text[:100]}",
                    )
        except httpx.ConnectError:
            return SummarizeResult(success=False, error="Daemon not running")
        except httpx.TimeoutException:
            return SummarizeResult(success=False, error="Operation timeout")
        except Exception as e:
            return SummarizeResult(success=False, error=str(e))


# Convenience function for sync checking
def is_daemon_running(url: str = "http://127.0.0.1:8765") -> bool:
    """
    Synchronous check if daemon is running.

    This is a blocking call with short timeout for CLI startup checks.

    Args:
        url: Daemon URL

    Returns:
        True if daemon is running, False otherwise.
    """
    import asyncio

    try:
        # Try to get an existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, DaemonClient(url).is_running())
                return future.result(timeout=1.0)
        else:
            return loop.run_until_complete(DaemonClient(url).is_running())
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(DaemonClient(url).is_running())
    except Exception:
        return False
