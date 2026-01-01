"""Unit tests for RAG Index Service."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from backend.src.services.rag_index import RAGIndexService
from backend.src.services.config import AppConfig

@pytest.fixture
def mock_config(tmp_path):
    config = MagicMock(spec=AppConfig)
    config.google_api_key = "fake-key"
    config.llamaindex_persist_dir = tmp_path / "data" / "llamaindex"
    return config

@pytest.fixture
def rag_service(mock_config):
    with patch("backend.src.services.rag_index.get_config", return_value=mock_config):
        with patch("backend.src.services.rag_index.VaultService") as mock_vault:
            service = RAGIndexService()
            service.vault_service = mock_vault.return_value
            yield service

def test_get_persist_dir(rag_service):
    user_id = "test-user"
    persist_dir = rag_service.get_persist_dir(user_id)
    assert user_id in persist_dir
    assert "default" in persist_dir  # Now includes project_id
    assert Path(persist_dir).exists()


def test_get_persist_dir_with_project(rag_service):
    user_id = "test-user"
    project_id = "my-project"
    persist_dir = rag_service.get_persist_dir(user_id, project_id)
    assert user_id in persist_dir
    assert project_id in persist_dir
    assert Path(persist_dir).exists()

@patch("backend.src.services.rag_index.load_index_from_storage")
@patch("backend.src.services.rag_index.StorageContext")
def test_get_or_build_index_existing(mock_storage_context, mock_load, rag_service):
    """Test loading an existing index."""
    user_id = "test-user"
    mock_index = MagicMock()
    mock_load.return_value = mock_index
    
    # Assume load succeeds
    index = rag_service.get_or_build_index(user_id)
    
    assert index == mock_index
    mock_load.assert_called_once()
    # Should NOT build
    rag_service.vault_service.list_notes.assert_not_called()

@patch("backend.src.services.rag_index.VectorStoreIndex")
@patch("backend.src.services.rag_index.load_index_from_storage")
def test_get_or_build_index_new(mock_load, mock_vector_store, rag_service):
    """Test building a new index when load fails."""
    user_id = "test-user"
    mock_load.side_effect = Exception("No index")
    
    mock_index = MagicMock()
    mock_vector_store.from_documents.return_value = mock_index
    
    # Mock vault data
    rag_service.vault_service.list_notes.return_value = [{"path": "note1.md"}]
    rag_service.vault_service.read_note.return_value = {
        "title": "Note 1",
        "body": "Content",
        "metadata": {}
    }
    
    index = rag_service.get_or_build_index(user_id)

    # Now includes project_id='default' parameter
    rag_service.vault_service.list_notes.assert_called_with(user_id, project_id='default')

@patch("backend.src.services.rag_index.os.path.exists")
def test_get_status(mock_exists, rag_service):
    user_id = "test-user"
    mock_exists.return_value = True
    status = rag_service.get_status(user_id)
    assert status.status == "ready"
    
    mock_exists.return_value = False
    status = rag_service.get_status(user_id)
    assert status.status == "building"

@patch("backend.src.services.rag_index.load_index_from_storage")
@patch("backend.src.services.rag_index.StorageContext")
def test_chat(mock_storage, mock_load, rag_service):
    user_id = "test-user"
    
    # Mock Index and ChatEngine
    mock_index = MagicMock()
    mock_chat_engine = MagicMock()
    mock_index.as_chat_engine.return_value = mock_chat_engine
    mock_load.return_value = mock_index
    
    # Mock Response
    mock_response = MagicMock()
    mock_response.__str__.return_value = "AI Answer"
    
    # Mock Source Nodes
    mock_node = MagicMock()
    mock_node.metadata = {"path": "note.md", "title": "Note"}
    mock_node.get_content.return_value = "Snippet content"
    mock_node.score = 0.9
    mock_response.source_nodes = [mock_node]
    
    mock_chat_engine.chat.return_value = mock_response
    
    from backend.src.models.rag import ChatMessage
    messages = [ChatMessage(role="user", content="Question")]

    # chat is now async, so we need to await it
    import asyncio
    response = asyncio.get_event_loop().run_until_complete(rag_service.chat(user_id, messages))

    assert response.answer == "AI Answer"
    assert len(response.sources) == 1
    assert response.sources[0].path == "note.md"
    mock_chat_engine.chat.assert_called()
