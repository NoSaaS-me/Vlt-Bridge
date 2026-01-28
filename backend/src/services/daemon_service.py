"""Service for managing Vlt Daemon instances and sessions."""

from typing import Dict, Optional, List
import uuid
from datetime import datetime

class DaemonService:
    """Manages the lifecycle and state of connected Vlt Daemons."""
    
    def __init__(self):
        # In-memory store for now, will move to DB later
        self._daemons: Dict[str, Dict] = {}
        self._sessions: Dict[str, Dict] = {}

    async def register_daemon(self, daemon_id: str, capabilities: Dict) -> Dict:
        """Register a new daemon connection."""
        self._daemons[daemon_id] = {
            "id": daemon_id,
            "capabilities": capabilities,
            "status": "ONLINE",
            "last_seen": datetime.utcnow()
        }
        return self._daemons[daemon_id]

    async def get_active_daemons(self) -> List[Dict]:
        """List all online daemons."""
        return list(self._daemons.values())
