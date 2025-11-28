from fastapi import APIRouter, Depends, HTTPException, status
from ..middleware import AuthContext, get_auth_context
from ...models.rag import ChatRequest, ChatResponse, StatusResponse
from ...services.rag_index import RAGIndexService

router = APIRouter(prefix="/api/rag", tags=["rag"])

def get_rag_service() -> RAGIndexService:
    return RAGIndexService()

@router.get("/status", response_model=StatusResponse)
async def get_status(
    auth: AuthContext = Depends(get_auth_context),
    rag_service: RAGIndexService = Depends(get_rag_service)
):
    """Get the status of the RAG index."""
    return rag_service.get_status(auth.user_id)

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    auth: AuthContext = Depends(get_auth_context),
    rag_service: RAGIndexService = Depends(get_rag_service)
):
    """
    Chat with the vault RAG agent.
    """
    try:
        return rag_service.chat(auth.user_id, request.messages)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")