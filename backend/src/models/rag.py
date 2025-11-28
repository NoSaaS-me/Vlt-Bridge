"""Pydantic models for RAG chat feature."""

from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]

class SourceReference(BaseModel):
    """Metadata about a note used to generate a response."""
    path: str = Field(..., description="Relative path in vault")
    title: str = Field(..., description="Note title")
    snippet: str = Field(..., description="Relevant text excerpt (max 500 chars)")
    score: Optional[float] = Field(None, description="Relevance score (0.0-1.0)")

class NoteWritten(BaseModel):
    """Metadata about a note created or updated by the agent."""
    path: str = Field(..., description="Path to created/updated note")
    title: str = Field(..., description="Note title")
    action: Literal["created", "updated"] = Field(..., description="Action performed")

class ChatMessage(BaseModel):
    """Represents a single message in the conversation."""
    role: Role = Field(..., description="Message author")
    content: str = Field(..., max_length=10000, description="Message text")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Creation time")
    sources: Optional[List[SourceReference]] = Field(None, description="Referenced notes (assistant only)")
    notes_written: Optional[List[NoteWritten]] = Field(None, description="Notes created (Phase 2)")

class ChatRequest(BaseModel):
    """Request payload for the RAG chat endpoint."""
    messages: List[ChatMessage] = Field(..., min_length=1, description="Conversation history")

class ChatResponse(BaseModel):
    """Response payload from the RAG chat endpoint."""
    answer: str = Field(..., description="AI-generated response")
    sources: List[SourceReference] = Field(default_factory=list, description="Notes used in response")
    notes_written: List[NoteWritten] = Field(default_factory=list, description="Notes created (Phase 2)")

class StatusResponse(BaseModel):
    """RAG index status."""
    status: Literal["ready", "building", "error"]
    doc_count: int
    last_updated: Optional[datetime] = None

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
