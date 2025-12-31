"""
VLT Daemon Package - Background sync service for vlt-cli.

The daemon provides:
- Persistent HTTP connection to backend (no connection overhead per CLI call)
- Fast CLI responses (queue and return immediately)
- Background sync with retry
- Server-side summarization requests

Key principle: All functionality works with fallback to direct calls if daemon isn't running.

Components:
- server.py: FastAPI server running on localhost
- client.py: Client for CLI to communicate with daemon
- manager.py: Process lifecycle management (start/stop/status)
"""

from .client import DaemonClient
from .manager import DaemonManager

__all__ = ["DaemonClient", "DaemonManager"]
