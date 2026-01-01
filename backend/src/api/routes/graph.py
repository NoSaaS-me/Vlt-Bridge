from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Annotated

from ...models.graph import GraphData
from ...models.project import DEFAULT_PROJECT_ID
from ..middleware import AuthContext, require_auth_context
from ...services.indexer import IndexerService
from ...services.database import DatabaseService

router = APIRouter()

@router.get("/api/graph", response_model=GraphData)
async def get_graph_data(
    auth: Annotated[AuthContext, Depends(require_auth_context)],
    project_id: str = Query(DEFAULT_PROJECT_ID, description="Project ID (default: 'default')"),
    indexer_service: Annotated[IndexerService, Depends(lambda: IndexerService(DatabaseService()))] = None
) -> GraphData:
    """Retrieve graph visualization data for a project."""
    try:
        if indexer_service is None:
            indexer_service = IndexerService(DatabaseService())
        return indexer_service.get_graph_data(auth.user_id, project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch graph data: {str(e)}")