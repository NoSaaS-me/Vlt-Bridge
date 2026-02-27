"""Service for managing Agent Sessions and Leases."""

from typing import Dict, Optional, List
import uuid
from datetime import datetime, timedelta
from ..models.session import AgentSession, SessionLease, SessionCreate

class SessionService:
    def __init__(self):
        # In-memory for MVP
        self._sessions: Dict[str, AgentSession] = {}
        self._leases: Dict[str, SessionLease] = {}

    async def create_session(self, create: SessionCreate, owner_id: str) -> AgentSession:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = AgentSession(
            id=session_id,
            project_id=create.project_id,
            metadata=create.metadata,
            status="CREATED"
        )
        self._sessions[session_id] = session
        return session

    async def acquire_lease(self, session_id: str, daemon_id: str) -> Optional[SessionLease]:
        """Grant a lease to a daemon if available."""
        # Check if already leased
        for lease in self._leases.values():
            if lease.session_id == session_id and lease.active and lease.expires_at > datetime.utcnow():
                if lease.daemon_id == daemon_id:
                    return lease # Renew
                return None # Taken

        lease_id = str(uuid.uuid4())
        lease = SessionLease(
            id=lease_id,
            session_id=session_id,
            daemon_id=daemon_id,
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )
        self._leases[lease_id] = lease
        
        # Update session
        if session_id in self._sessions:
            self._sessions[session_id].daemon_id = daemon_id
            
        return lease
