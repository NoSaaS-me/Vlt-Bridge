"""RAG Index Service using LlamaIndex and Gemini."""

import logging
import os
from pathlib import Path
from typing import Optional, List

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document,
    Settings
)
from llama_index.llms.google_genai import Gemini
from llama_index.embeddings.google_genai import GeminiEmbedding
from llama_index.core.base.response.schema import Response as LlamaResponse
from llama_index.core.llms import ChatMessage as LlamaChatMessage, MessageRole

from .config import get_config
from .vault import VaultService
from ..models.rag import ChatMessage, ChatResponse, SourceReference, StatusResponse

logger = logging.getLogger(__name__)

class RAGIndexService:
    """Service for managing LlamaIndex vector stores."""

    def __init__(self):
        self.vault_service = VaultService()
        self.config = get_config()
        self._setup_gemini()

    def _setup_gemini(self):
        """Configure global LlamaIndex settings for Gemini."""
        if not self.config.google_api_key:
            logger.warning("GOOGLE_API_KEY not set. RAG features will fail.")
            return

        # Set up Gemini
        try:
            # Configure global settings
            Settings.llm = Gemini(
                model="models/gemini-1.5-flash", 
                api_key=self.config.google_api_key
            )
            Settings.embed_model = GeminiEmbedding(
                model_name="models/embedding-001", 
                api_key=self.config.google_api_key
            )
        except Exception as e:
            logger.error(f"Failed to setup Gemini: {e}")

    def get_persist_dir(self, user_id: str) -> str:
        """Get persistence directory for a user's index."""
        user_dir = self.config.llamaindex_persist_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return str(user_dir)

    def get_or_build_index(self, user_id: str) -> VectorStoreIndex:
        """Load existing index or build a new one from vault notes."""
        persist_dir = self.get_persist_dir(user_id)
        
        # check if index files exist (docstore.json, index_store.json etc)
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            logger.info(f"Loaded existing index for user {user_id}")
            return index
        except Exception:
            logger.info(f"No valid index found for {user_id}, building new one...")
            return self.build_index(user_id)

    def build_index(self, user_id: str) -> VectorStoreIndex:
        """Build a new index from the user's vault."""
        if not self.config.google_api_key:
            raise ValueError("GOOGLE_API_KEY required to build index")

        # Read notes from VaultService
        notes = self.vault_service.list_notes(user_id)
        documents = []
        
        for note_summary in notes:
            path = note_summary["path"]
            try:
                note = self.vault_service.read_note(user_id, path)
                # Create Document
                metadata = {
                    "path": path,
                    "title": note["title"],
                    **note.get("metadata", {})
                }
                doc = Document(
                    text=note["body"],
                    metadata=metadata,
                    id_=path # Use path as ID for stability
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to index note {path}: {e}")

        logger.info(f"Indexing {len(documents)} documents for {user_id}")
        
        index = VectorStoreIndex.from_documents(documents)
        
        # Persist
        persist_dir = self.get_persist_dir(user_id)
        index.storage_context.persist(persist_dir=persist_dir)
        logger.info(f"Persisted index to {persist_dir}")
        
        return index

    def rebuild_index(self, user_id: str) -> VectorStoreIndex:
        """Force rebuild of index."""
        return self.build_index(user_id)

    def get_status(self, user_id: str) -> StatusResponse:
        """Get index status."""
        persist_dir = self.get_persist_dir(user_id)
        doc_store_path = os.path.join(persist_dir, "docstore.json")
        
        if os.path.exists(doc_store_path):
            return StatusResponse(status="ready", doc_count=0, last_updated=None)
        
        return StatusResponse(status="building", doc_count=0, last_updated=None)

    def chat(self, user_id: str, messages: List[ChatMessage]) -> ChatResponse:
        """Run RAG chat query with history."""
        if not self.config.google_api_key:
            raise ValueError("Google API Key is not configured. Please set GOOGLE_API_KEY in settings or env.")

        index = self.get_or_build_index(user_id)
        
        if not messages:
             raise ValueError("No messages provided")
             
        last_message = messages[-1]
        if last_message.role != "user":
            raise ValueError("Last message must be from user")
            
        query_text = last_message.content
        
        # Convert history (excluding last message)
        history = []
        for m in messages[:-1]:
            role = MessageRole.USER if m.role == "user" else MessageRole.ASSISTANT
            history.append(LlamaChatMessage(role=role, content=m.content))
            
        # Use chat engine with context mode
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            system_prompt=(
                "You are a helpful assistant for a documentation vault. "
                "Answer questions based on the provided context. "
                "If the answer is not in the context, say you don't know. "
                "Always cite your sources."
            )
        )
        
        response = chat_engine.chat(query_text, chat_history=history)
        
        return self._format_response(response)

    def _format_response(self, response: LlamaResponse) -> ChatResponse:
        """Convert LlamaIndex response to ChatResponse."""
        sources = []
        for node in response.source_nodes:
            metadata = node.metadata
            sources.append(SourceReference(
                path=metadata.get("path", "unknown"),
                title=metadata.get("title", "Untitled"),
                snippet=node.get_content()[:500], # Truncate snippet
                score=node.score
            ))
            
        return ChatResponse(
            answer=str(response),
            sources=sources
        )