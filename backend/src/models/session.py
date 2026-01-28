"""Session and Lease models."""

from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field

class SessionLease(BaseModel):
    """Authority granting a Daemon exclusive control over a Session."""
    
    id: str = Field(..., description="Lease UUID")
    session_id: str = Field(..., description="Target Session UUID")
    daemon_id: str = Field(..., description="Authorized Daemon UUID")
    granted_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    active: bool = True

class AgentSession(BaseModel):
    """Persistent execution context for an agent."""
    
    id: str = Field(..., description="Session UUID")
    project_id: str = Field(..., description="Project context")
    daemon_id: Optional[str] = Field(None, description="Currently assigned daemon")
    status: str = Field("CREATED", description="CREATED | RUNNING | PAUSED | COMPLETED | FAILED")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict, description="Task config, model settings")

class SessionCreate(BaseModel):
    """Payload to create a new session."""
    project_id: str
    metadata: Dict
