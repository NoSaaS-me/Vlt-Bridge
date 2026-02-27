"""Daemon entity model."""

from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field

class Daemon(BaseModel):
    """Represents a running instance of the Vlt Daemon."""
    
    id: str = Field(..., description="Unique UUID for the daemon instance")
    hostname: str = Field(..., description="Hostname or IP of the machine")
    status: str = Field("ONLINE", description="ONLINE | OFFLINE | BUSY")
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    capabilities: Dict = Field(default_factory=dict, description="Hardware specs (GPU, RAM)")
    owner_id: str = Field(..., description="User ID who owns this daemon")

class DaemonCreate(BaseModel):
    """Payload for registering a daemon."""
    id: str
    hostname: str
    capabilities: Optional[Dict] = {}
