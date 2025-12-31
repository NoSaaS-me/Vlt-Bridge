"""Oracle API endpoints - Multi-source intelligent context retrieval.

This module provides the Oracle Agent API which uses OpenRouter function calling
for autonomous tool execution. The Oracle can search code, read documentation,
query development threads, and search the web to answer questions.

Updated for 009-oracle-agent: Uses OracleAgent instead of OracleBridge subprocess.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ..middleware import AuthContext, get_auth_context
from ...models.oracle import (
    OracleRequest,
    OracleResponse,
    OracleStreamChunk,
    ConversationHistoryResponse,
    ConversationMessage,
    SourceReference,
)
from ...services.oracle_agent import OracleAgent, OracleAgentError
from ...services.oracle_bridge import OracleBridge, OracleBridgeError
from ...services.user_settings import UserSettingsService, get_user_settings_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/oracle", tags=["oracle"])

# Singleton oracle bridge instance (kept for fallback/deprecation period)
_oracle_bridge: OracleBridge | None = None

# Active Oracle sessions for cancellation support
# Maps user_id to active OracleAgent instance
_active_sessions: Dict[str, OracleAgent] = {}


def get_oracle_bridge() -> OracleBridge:
    """Get or create the oracle bridge instance (deprecated, use OracleAgent)."""
    global _oracle_bridge
    if _oracle_bridge is None:
        _oracle_bridge = OracleBridge()
    return _oracle_bridge


@router.post("", response_model=OracleResponse)
async def query_oracle(
    request: OracleRequest,
    auth: AuthContext = Depends(get_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
):
    """
    Query the oracle with a natural language question (non-streaming).

    Uses the OracleAgent with OpenRouter function calling for autonomous
    tool execution. This endpoint collects the full response before returning.

    **Request Body:**
    - `question`: Natural language question (required)
    - `sources`: List of sources to query ("vault", "code", "threads") - null means all
    - `explain`: Include retrieval traces for debugging (default: false)
    - `model`: Override LLM model (e.g., "anthropic/claude-sonnet-4")
    - `thinking`: Enable thinking mode for extended reasoning (default: false)
    - `max_tokens`: Maximum context tokens (default: 16000)

    **Response:**
    - `answer`: Synthesized answer
    - `sources`: List of source citations with paths and snippets
    - `tokens_used`: Total tokens consumed
    - `model_used`: Model that generated the response
    - `retrieval_traces`: Debug information (if explain=True)
    """
    # Get user's OpenRouter API key
    openrouter_api_key = settings_service.get_openrouter_api_key(auth.user_id)

    if not openrouter_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OpenRouter API key not configured. Please add your API key in Settings.",
        )

    try:
        logger.info(f"Oracle query from user {auth.user_id}: {request.question[:100]}")

        # Get user's model settings for Oracle and Librarian
        oracle_model = request.model or settings_service.get_oracle_model(auth.user_id)
        subagent_model = settings_service.get_subagent_model(auth.user_id)

        logger.debug(f"Using oracle_model={oracle_model}, subagent_model={subagent_model}")

        # Create OracleAgent with context service integration
        agent = OracleAgent(
            api_key=openrouter_api_key,
            model=oracle_model,
            subagent_model=subagent_model,
            project_id=request.project_id or "default",
            user_id=auth.user_id,
        )

        # Collect all chunks from the stream
        content_parts = []
        sources = []
        tokens_used = None
        model_used = None
        context_id = None

        async for chunk in agent.query(
            question=request.question,
            user_id=auth.user_id,
            stream=False,  # Non-streaming mode
            thinking=request.thinking,
            max_tokens=request.max_tokens,
            project_id=request.project_id,
            context_id=request.context_id,
        ):
            if chunk.type == "content" and chunk.content:
                content_parts.append(chunk.content)
            elif chunk.type == "source" and chunk.source:
                sources.append(chunk.source)
            elif chunk.type == "done":
                tokens_used = chunk.tokens_used
                model_used = chunk.model_used
                context_id = chunk.context_id
            elif chunk.type == "error":
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=chunk.error or "Oracle query failed",
                )

        return OracleResponse(
            answer="".join(content_parts),
            sources=sources,
            tokens_used=tokens_used,
            model_used=model_used,
            context_id=context_id,
            retrieval_traces=None,  # TODO: Implement if explain=True
        )

    except OracleAgentError as e:
        logger.error(f"Oracle agent error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Oracle error: {e.message}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Oracle query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Oracle query failed: {str(e)}",
        )


@router.post("/stream")
async def query_oracle_stream(
    request: OracleRequest,
    auth: AuthContext = Depends(get_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
):
    """
    Query the oracle with streaming response (Server-Sent Events).

    Uses the OracleAgent with OpenRouter function calling for autonomous
    tool execution. The agent can search code, read documentation, query
    threads, and search the web to gather context before answering.

    The response streams as Server-Sent Events (SSE) with the following chunk types:
    - `thinking`: Progress updates during retrieval
    - `tool_call`: Tool being invoked (with id, name, arguments)
    - `tool_result`: Result from tool execution
    - `content`: Answer text chunks
    - `source`: Source citations
    - `done`: Final chunk with metadata (tokens_used, model_used)
    - `error`: Error occurred

    **Request Body:** Same as non-streaming endpoint

    **Response:** SSE stream of JSON objects

    **Example chunks:**
    ```json
    data: {"type": "tool_call", "tool_call": {"name": "search_code", "arguments": "..."}}
    data: {"type": "content", "content": "Based on the code..."}
    ```
    """
    # Get user's OpenRouter API key
    openrouter_api_key = settings_service.get_openrouter_api_key(auth.user_id)

    if not openrouter_api_key:
        # Return error if no API key configured
        async def error_generator():
            error_chunk = OracleStreamChunk(
                type="error",
                error="OpenRouter API key not configured. Please add your API key in Settings."
            )
            yield json.dumps(error_chunk.model_dump(exclude_none=True))

        return EventSourceResponse(error_generator())

    # Cancel any existing session for this user
    if auth.user_id in _active_sessions:
        logger.info(f"Cancelling existing session for user {auth.user_id}")
        _active_sessions[auth.user_id].cancel()

    # Get user's model settings for Oracle and Librarian
    oracle_model = request.model or settings_service.get_oracle_model(auth.user_id)
    subagent_model = settings_service.get_subagent_model(auth.user_id)

    logger.debug(f"Stream using oracle_model={oracle_model}, subagent_model={subagent_model}")

    # Create OracleAgent with user's settings and context integration
    agent = OracleAgent(
        api_key=openrouter_api_key,
        model=oracle_model,
        subagent_model=subagent_model,
        project_id=request.project_id or "default",
        user_id=auth.user_id,
    )

    # Register the agent for cancellation support
    _active_sessions[auth.user_id] = agent

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from OracleAgent stream."""
        chunk_counter = 0
        try:
            logger.info(f"Oracle Agent query from user {auth.user_id}: {request.question[:100]}")

            async for chunk in agent.query(
                question=request.question,
                user_id=auth.user_id,
                stream=True,
                thinking=request.thinking,
                max_tokens=request.max_tokens,
                project_id=request.project_id,
                context_id=request.context_id,
            ):
                chunk_counter += 1
                chunk_json = chunk.model_dump(exclude_none=True)
                # Debug logging to trace chunk duplication issue
                logger.debug(
                    f"[SSE #{chunk_counter}] type={chunk.type} "
                    f"content_preview={str(chunk.content)[:50] if chunk.content else 'N/A'}"
                )
                # OracleAgent yields OracleStreamChunk objects directly
                yield json.dumps(chunk_json)

        except OracleAgentError as e:
            logger.error(f"Oracle agent error: {e.message}")
            error_chunk = OracleStreamChunk(
                type="error",
                error=f"Oracle error: {e.message}"
            )
            yield json.dumps(error_chunk.model_dump(exclude_none=True))

        except Exception as e:
            logger.exception("Oracle streaming failed")
            error_chunk = OracleStreamChunk(
                type="error",
                error=f"Streaming error: {str(e)}"
            )
            yield json.dumps(error_chunk.model_dump(exclude_none=True))

        finally:
            # Clean up session when done
            if auth.user_id in _active_sessions and _active_sessions[auth.user_id] is agent:
                del _active_sessions[auth.user_id]

    return EventSourceResponse(event_generator())


@router.post("/cancel")
async def cancel_oracle_session(
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Cancel the active Oracle session for this user.

    This endpoint immediately cancels any running Oracle query for the authenticated user.
    The agent will stop at the next checkpoint and yield a cancellation error.

    **Response:**
    - `{"status": "cancelled"}`: Successfully cancelled an active session
    - `{"status": "no_active_session"}`: No active session to cancel
    """
    session_id = auth.user_id

    if session_id in _active_sessions:
        agent = _active_sessions[session_id]
        agent.cancel()
        del _active_sessions[session_id]
        logger.info(f"Cancelled Oracle session for user {session_id}")
        return {"status": "cancelled"}

    logger.debug(f"No active Oracle session to cancel for user {session_id}")
    return {"status": "no_active_session"}


@router.get("/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    auth: AuthContext = Depends(get_auth_context),
    oracle: OracleBridge = Depends(get_oracle_bridge),
):
    """
    Get conversation history for the current user.

    Returns the conversation history maintained across oracle queries.
    History is stored in memory and limited to the last 50 messages.

    **Response:**
    - `messages`: List of conversation messages
    - `session_id`: User ID (used as session identifier)
    - `compressed`: Whether history has been compressed (always false for now)
    - `token_count`: Approximate token count (null for now)
    """
    try:
        history = oracle.get_conversation_history(auth.user_id)

        # Transform to ConversationMessage objects
        messages = []
        for msg in history:
            messages.append(
                ConversationMessage(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg.get("timestamp"),
                    sources=[
                        SourceReference(**source) if isinstance(source, dict) else source
                        for source in msg.get("sources", [])
                    ] if msg.get("sources") else None,
                )
            )

        return ConversationHistoryResponse(
            messages=messages,
            session_id=auth.user_id,
            compressed=False,
            token_count=None,  # TODO: Calculate token count
        )

    except Exception as e:
        logger.exception("Failed to get conversation history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get history: {str(e)}",
        )


@router.delete("/history")
async def clear_conversation_history(
    auth: AuthContext = Depends(get_auth_context),
    oracle: OracleBridge = Depends(get_oracle_bridge),
):
    """
    Clear conversation history for the current user.

    Removes all stored conversation messages for the authenticated user.

    **Response:**
    ```json
    {"status": "ok", "message": "Conversation history cleared"}
    ```
    """
    try:
        oracle.clear_conversation_history(auth.user_id)
        logger.info(f"Cleared conversation history for user: {auth.user_id}")

        return {
            "status": "ok",
            "message": "Conversation history cleared",
        }

    except Exception as e:
        logger.exception("Failed to clear conversation history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear history: {str(e)}",
        )
