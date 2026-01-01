from fastapi import APIRouter, Depends, HTTPException, Query, status
from ..middleware import AuthContext, get_auth_context
from ...models.project import DEFAULT_PROJECT_ID
from ...models.rag import ChatRequest, ChatResponse, StatusResponse
from ...services.rag_index import RAGIndexService
import logging
import traceback
import sys

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])

def get_rag_service() -> RAGIndexService:
    return RAGIndexService()

@router.get("/status", response_model=StatusResponse)
async def get_status(
    project_id: str = Query(DEFAULT_PROJECT_ID, description="Project ID (default: 'default')"),
    auth: AuthContext = Depends(get_auth_context),
    rag_service: RAGIndexService = Depends(get_rag_service)
):
    """Get the status of the RAG index for a project."""
    return rag_service.get_status(auth.user_id, project_id)

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    project_id: str = Query(DEFAULT_PROJECT_ID, description="Project ID (default: 'default')"),
    auth: AuthContext = Depends(get_auth_context),
    rag_service: RAGIndexService = Depends(get_rag_service)
):
    """
    Chat with the vault RAG agent for a project.
    """
    try:
        return await rag_service.chat(auth.user_id, request.messages, project_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log full traceback
        logger.exception("RAG Chat failed")
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")

