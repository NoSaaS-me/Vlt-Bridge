import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from backend.src.api.main import app
from backend.src.services.indexer import IndexerService
from backend.src.models.graph import GraphData
from backend.src.api.middleware import AuthContext, get_auth_context

client = TestClient(app)

@pytest.fixture
def mock_indexer():
    with patch("backend.src.api.routes.graph.IndexerService") as mock:
        yield mock

def test_get_graph_data_success(mock_indexer):
    """Test successful retrieval of graph data."""
    # Setup mock return value
    mock_instance = mock_indexer.return_value
    mock_instance.get_graph_data.return_value = {
        "nodes": [
            {"id": "note1.md", "label": "Note 1", "val": 1, "group": "root"},
            {"id": "folder/note2.md", "label": "Note 2", "val": 2, "group": "folder"}
        ],
        "links": [
            {"source": "note1.md", "target": "folder/note2.md"}
        ]
    }

    # Override dependencies
    app.dependency_overrides[IndexerService] = lambda: mock_instance
    
    mock_auth = Mock(spec=AuthContext)
    mock_auth.user_id = "test-user"
    app.dependency_overrides[get_auth_context] = lambda: mock_auth

    response = client.get("/api/graph")
    
    # Clean up overrides
    app.dependency_overrides = {}

    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "links" in data
    assert len(data["nodes"]) == 2
    assert len(data["links"]) == 1
    assert data["nodes"][0]["id"] == "note1.md"

def test_get_graph_data_error(mock_indexer):
    """Test error handling when service fails."""
    mock_instance = mock_indexer.return_value
    mock_instance.get_graph_data.side_effect = Exception("Database error")

    app.dependency_overrides[IndexerService] = lambda: mock_instance
    
    mock_auth = Mock(spec=AuthContext)
    mock_auth.user_id = "test-user"
    app.dependency_overrides[get_auth_context] = lambda: mock_auth

    response = client.get("/api/graph")
    
    # Clean up overrides
    app.dependency_overrides = {}

    assert response.status_code == 500
    assert "Failed to fetch graph data" in response.json()["detail"]
