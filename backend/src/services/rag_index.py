"""RAG Index Service using LlamaIndex and Gemini."""

import logging
import os
import threading
from pathlib import Path
from typing import Optional, List

# Configure logger first so it can be used in try/except
logger = logging.getLogger(__name__)

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document,
    Settings
)

# Try to import Gemini, handle missing dependency gracefully
try:
    from llama_index.llms.google_genai import GoogleGenAI as Gemini
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding as GeminiEmbedding
except ImportError as e:
    Gemini = None
    GeminiEmbedding = None
    logger.warning(f"Could not import google_genai modules: {e}")

from llama_index.core.base.response.schema import Response as LlamaResponse
from llama_index.core.llms import ChatMessage as LlamaChatMessage, MessageRole

from .config import get_config
from .vault import VaultService
from ..models.rag import ChatMessage, ChatResponse, SourceReference, StatusResponse

class RAGIndexService:
    """Service for managing LlamaIndex vector stores."""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RAGIndexService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
            
        self.vault_service = VaultService()
        self.config = get_config()
        self._index_lock = threading.Lock() # Per-instance lock for index ops
        self._setup_gemini()
        self._initialized = True

    def _setup_gemini(self):
        """Configure global LlamaIndex settings for Gemini."""
        if not Gemini or not GeminiEmbedding:
            logger.error("Google GenAI modules not loaded. RAG setup skipped.")
            return

        api_key = self.config.google_api_key
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set. RAG features will fail.")
            return
            
        # Log key status (masked)
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        logger.info(f"Configuring Gemini with API key: {masked_key}")

        # Set up Gemini
        try:
            # Configure global settings
            Settings.llm = Gemini(
                model="gemini-2.0-flash", 
                api_key=self.config.google_api_key
            )
            Settings.embed_model = GeminiEmbedding(
                model_name="models/text-embedding-004", 
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
        with self._index_lock:
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
        if not notes:
            # Handle empty vault (Fix #8)
            logger.info(f"No notes found for {user_id}, creating empty index")
            index = VectorStoreIndex.from_documents([])
            # Persist empty index to avoid rebuilding every time?
            # LlamaIndex might not persist empty index well.
            # Let's just return it.
            return index

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
        
        doc_count = 0
        status = "building"
        
        if os.path.exists(doc_store_path):
            status = "ready"
            try:
                # Simple line count or file size check to avoid loading whole JSON
                # Actually, docstore.json is a dict.
                # Let's just load it if it's small, or stat it.
                # For MVP, just checking existence is "ready".
                # To get count, we can try loading keys.
                import json
                with open(doc_store_path, 'r') as f:
                    data = json.load(f)
                    doc_count = len(data.get("docstore/data", {}))
            except Exception:
                logger.warning(f"Failed to read docstore for status: {doc_store_path}")
                
        return StatusResponse(status=status, doc_count=doc_count, last_updated=None)

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