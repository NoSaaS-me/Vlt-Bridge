"""Oracle API endpoints - Multi-source intelligent context retrieval."""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

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
from ...services.oracle_bridge import OracleBridge, OracleBridgeError
from ...services.user_settings import UserSettingsService, get_user_settings_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/oracle", tags=["oracle"])

# Singleton oracle bridge instance
_oracle_bridge: OracleBridge | None = None


def get_oracle_bridge() -> OracleBridge:
    """Get or create the oracle bridge instance."""
    global _oracle_bridge
    if _oracle_bridge is None:
        _oracle_bridge = OracleBridge()
    return _oracle_bridge


@router.post("", response_model=OracleResponse)
async def query_oracle(
    request: OracleRequest,
    auth: AuthContext = Depends(get_auth_context),
    oracle: OracleBridge = Depends(get_oracle_bridge),
):
    """
    Query the oracle with a natural language question (non-streaming).

    The oracle queries multiple knowledge sources (vault, code, threads) and
    returns a synthesized answer with source citations.

    **Request Body:**
    - `question`: Natural language question (required)
    - `sources`: List of sources to query ("vault", "code", "threads") - null means all
    - `explain`: Include retrieval traces for debugging (default: false)
    - `model`: Override LLM model (e.g., "anthropic/claude-3.5-sonnet")
    - `thinking`: Enable thinking mode for extended reasoning (default: false)
    - `max_tokens`: Maximum context tokens (default: 16000)

    **Response:**
    - `answer`: Synthesized answer
    - `sources`: List of source citations with paths and snippets
    - `tokens_used`: Total tokens consumed
    - `model_used`: Model that generated the response
    - `retrieval_traces`: Debug information (if explain=True)
    """
    try:
        logger.info(f"Oracle query from user {auth.user_id}: {request.question[:100]}")

        # Call oracle bridge (non-streaming)
        result = await oracle.ask_oracle(
            question=request.question,
            sources=request.sources,
            explain=request.explain,
            project=None,  # Auto-detect project
            max_tokens=request.max_tokens,
        )

        # Transform response to match our schema
        return OracleResponse(
            answer=result.get("answer", ""),
            sources=[
                SourceReference(**source) if isinstance(source, dict) else source
                for source in result.get("sources", [])
            ],
            tokens_used=result.get("tokens_used"),
            model_used=result.get("model_used"),
            retrieval_traces=result.get("retrieval_traces") if request.explain else None,
        )

    except OracleBridgeError as e:
        logger.error(f"Oracle bridge error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Oracle error: {e.message}",
        )
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
    oracle: OracleBridge = Depends(get_oracle_bridge),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
):
    """
    Query the oracle with streaming response (Server-Sent Events).

    The response streams as Server-Sent Events (SSE) with the following chunk types:
    - `thinking`: Progress updates during retrieval
    - `content`: Answer text chunks
    - `source`: Source citations
    - `done`: Final chunk with metadata (tokens_used, model_used)
    - `error`: Error occurred

    **Request Body:** Same as non-streaming endpoint

    **Response:** SSE stream of JSON objects

    **Example chunk:**
    ```json
    data: {"type": "content", "content": "Based on the code..."}
    ```
    """
    # Get user's OpenRouter API key
    openrouter_api_key = settings_service.get_openrouter_api_key(auth.user_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from oracle stream."""
        try:
            logger.info(f"Oracle streaming query from user {auth.user_id}: {request.question[:100]}")

            async for chunk in oracle.ask_oracle_stream(
                user_id=auth.user_id,
                question=request.question,
                sources=request.sources,
                explain=request.explain,
                model=request.model,
                thinking=request.thinking,
                project=None,  # Auto-detect project
                max_tokens=request.max_tokens,
                openrouter_api_key=openrouter_api_key,
            ):
                # Validate chunk against schema
                try:
                    validated_chunk = OracleStreamChunk(**chunk)
                    # Yield as SSE data
                    yield json.dumps(validated_chunk.model_dump(exclude_none=True))
                except Exception as e:
                    logger.error(f"Invalid chunk from oracle: {e}")
                    # Send error chunk
                    error_chunk = OracleStreamChunk(
                        type="error",
                        error=f"Invalid response format: {str(e)}"
                    )
                    yield json.dumps(error_chunk.model_dump(exclude_none=True))
                    break

        except OracleBridgeError as e:
            logger.error(f"Oracle bridge error: {e.message}")
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

    return EventSourceResponse(event_generator())


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
