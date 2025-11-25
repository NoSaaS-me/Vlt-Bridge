from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated

from backend.src.models.graph import GraphData
from backend.src.api.middleware import AuthContext, get_auth_context
from backend.src.services.indexer import IndexerService
from backend.src.services.database import DatabaseService

router = APIRouter()

@router.get("/api/graph", response_model=GraphData)
async def get_graph_data(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    indexer_service: Annotated[IndexerService, Depends(lambda: IndexerService(DatabaseService()))]
) -> GraphData:
    """Retrieve graph visualization data."""
    try:
        return indexer_service.get_graph_data(auth.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch graph data: {str(e)}")
