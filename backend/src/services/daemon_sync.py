"""Service to sync task updates from Daemon to Server."""

from typing import Dict
from ..models.session import AgentSession

class DaemonSyncService:
    async def handle_task_update(self, lease_id: str, update: Dict):
        """
        Processes an update from a daemon.
        update: {"output": "...", "status": "RUNNING"}
        """
        # In real impl, store to DB and broadcast via WebSocket
        print(f"Sync received for lease {lease_id}: {update.get('status')}")
        
        # Broadcast to clients subscribed to this session
        # await client_manager.broadcast(session_id, update)
