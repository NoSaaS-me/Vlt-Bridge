"""Pydantic models for Oracle feature."""

from enum import Enum
from typing import List, Optional, Literal, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field


class StreamEventType(str, Enum):
    """SSE stream event types for Oracle responses."""
    STATUS = "status"
    THINKING = "thinking"
    CONTENT = "content"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SOURCES = "sources"
    ERROR = "error"
    DONE = "done"
    SYSTEM = "system"


class SourceReference(BaseModel):
    """Source citation from oracle retrieval."""
    path: str = Field(..., description="File path, note path, or thread ID")
    source_type: Literal["code", "vault", "thread", "repomap"] = Field(
        ..., description="Type of knowledge source"
    )
    line: Optional[int] = Field(None, description="Line number for code sources")
    snippet: Optional[str] = Field(None, description="Relevant excerpt (max 500 chars)")
    score: Optional[float] = Field(None, description="Relevance score (0.0-1.0)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class OracleRequest(BaseModel):
    """Request payload for oracle queries."""
    question: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    sources: Optional[List[Literal["vault", "code", "threads"]]] = Field(
        None, description="Knowledge sources to query (None = all)"
    )
    explain: bool = Field(False, description="Include retrieval traces for debugging")
    model: Optional[str] = Field(None, description="Override LLM model (e.g., 'anthropic/claude-3.5-sonnet')")
    thinking: bool = Field(False, description="Enable thinking mode (append :thinking suffix)")
    max_tokens: int = Field(16000, ge=1000, le=100000, description="Maximum tokens for context assembly")
    context_id: Optional[str] = Field(None, description="Context ID to continue conversation (None = create new)")
    project_id: Optional[str] = Field(None, description="Project ID for context scoping")


class OracleStreamChunk(BaseModel):
    """Server-sent event chunk for streaming responses."""
    type: Literal["thinking", "content", "source", "tool_call", "tool_result", "done", "error", "system", "context_update"] = Field(
        ..., description="Chunk type"
    )
    content: Optional[str] = Field(None, description="Text content for thinking/content chunks")
    source: Optional[SourceReference] = Field(None, description="Source citation for source chunks")
    tool_call: Optional[Dict[str, Any]] = Field(None, description="Tool call info (tool_call chunks)")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID for associating results with calls")
    tool_result: Optional[str] = Field(None, description="Tool result (tool_result chunks)")
    tokens_used: Optional[int] = Field(None, description="Total tokens used (done chunk only)")
    model_used: Optional[str] = Field(None, description="Model used (done chunk only)")
    context_id: Optional[str] = Field(None, description="Context ID for conversation persistence (done chunk)")
    error: Optional[str] = Field(None, description="Error message (error chunk only)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    # Context window tracking fields (for context_update and done chunks)
    context_tokens: Optional[int] = Field(None, description="Current tokens used in context window")
    max_context_tokens: Optional[int] = Field(None, description="Maximum context window size for the model")


class OracleResponse(BaseModel):
    """Response payload for non-streaming oracle queries."""
    answer: str = Field(..., description="Synthesized answer")
    sources: List[SourceReference] = Field(default_factory=list, description="Source citations")
    tokens_used: Optional[int] = Field(None, description="Total tokens used")
    model_used: Optional[str] = Field(None, description="Model that generated the response")
    context_id: Optional[str] = Field(None, description="Context ID for conversation persistence")
    retrieval_traces: Optional[Dict[str, Any]] = Field(None, description="Retrieval debug info (if explain=True)")


class ConversationMessage(BaseModel):
    """A single message in conversation history."""
    role: Literal["user", "assistant", "system"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    sources: Optional[List[SourceReference]] = Field(None, description="Sources for assistant messages")


class ConversationHistoryResponse(BaseModel):
    """Response payload for conversation history."""
    messages: List[ConversationMessage] = Field(default_factory=list, description="Conversation history")
    session_id: Optional[str] = Field(None, description="Session identifier")
    compressed: bool = Field(False, description="Whether history has been compressed")
    token_count: Optional[int] = Field(None, description="Approximate token count")
