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
from llama_index.core.tools import FunctionTool

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
from ..models.rag import ChatMessage, ChatResponse, SourceReference, StatusResponse, NoteWritten

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

    def _create_note_tool(self, user_id: str):
        """Create a tool for writing new notes."""
        def create_note(title: str, content: str, folder: str = "agent-notes") -> str:
            """
            Create a new Markdown note in the vault.
            
            Args:
                title: The title of the note.
                content: The markdown content of the note.
                folder: The folder to place the note in (default: agent-notes).
            """
            # Sanitize folder path to prevent escaping agent-notes (simple check)
            # Actually, spec says "constrained to agent-notes/".
            # But user might want to organize within agent-notes/.
            safe_folder = folder if folder.startswith("agent-notes") else f"agent-notes/{folder}"
            safe_folder = safe_folder.strip("/")
            
            path = f"{safe_folder}/{title}.md"
            
            try:
                self.vault_service.write_note(
                    user_id,
                    path,
                    title=title,
                    body=content,
                    metadata={"created_by": "gemini-agent"}
                )
                return f"Note created successfully at {path}"
            except Exception as e:
                return f"Failed to create note: {e}"

        return FunctionTool.from_defaults(fn=create_note)

    def _move_note_tool(self, user_id: str):
        """Create a tool for moving notes."""
        def move_note(path: str, target_folder: str) -> str:
            """
            Move an existing note to a new folder.
            
            Args:
                path: The current path of the note (e.g. "agent-notes/My Note.md").
                target_folder: The destination folder (e.g. "agent-notes/archive").
            """
            if not path.endswith(".md"):
                path += ".md"
                
            filename = os.path.basename(path)
            new_path = f"{target_folder}/{filename}"
            
            try:
                self.vault_service.move_note(user_id, path, new_path)
                return f"Note moved from {path} to {new_path}"
            except Exception as e:
                return f"Failed to move note: {e}"

        return FunctionTool.from_defaults(fn=move_note)

    def _create_folder_tool(self, user_id: str):
        """Create a tool for creating new folders."""
        def create_folder(folder: str) -> str:
            """
            Create a new folder in the vault.
            
            Args:
                folder: The path of the folder to create (e.g. "agent-notes/archive").
            """
            safe_folder = folder.strip("/")
            
            try:
                # Create a placeholder to ensure directory exists
                placeholder = f"{safe_folder}/.placeholder.md"
                self.vault_service.write_note(
                    user_id,
                    placeholder,
                    title="Folder Placeholder",
                    body="# Folder\nCreated by agent.",
                    metadata={"created_by": "gemini-agent"}
                )
                return f"Folder created successfully at {safe_folder}"
            except Exception as e:
                return f"Failed to create folder: {e}"

        return FunctionTool.from_defaults(fn=create_folder)

    async def chat(self, user_id: str, messages: list[ChatMessage]) -> ChatResponse:
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
            
        tools = [
            self._create_note_tool(user_id),
            self._move_note_tool(user_id),
            self._create_folder_tool(user_id)
        ]
        
        from llama_index.core.tools import QueryEngineTool, ToolMetadata
        
        query_tool = QueryEngineTool(
            query_engine=index.as_query_engine(),
            metadata=ToolMetadata(
                name="vault_search",
                description="Search information in the documentation vault."
            )
        )
        
        all_tools = tools + [query_tool]
        
        # Use AgentRunner (factory for best agent type)
        from llama_index.core.agent import AgentRunner
        agent = AgentRunner.from_llm(
            llm=Settings.llm,
            tools=all_tools,
            chat_history=history,
            verbose=True,
            system_prompt="You are a documentation assistant. Use vault_search to find info. You can create notes and folders."
        )
        
        response = await agent.achat(query_text)
        
        return self._format_response(response)

    def _format_response(self, response: LlamaResponse) -> ChatResponse:
        """Convert LlamaIndex response to ChatResponse."""
        sources = []
        notes_written = []
        
        # Handle source nodes (RAG retrieval)
        if hasattr(response, "source_nodes"):
            for node_with_score in response.source_nodes:
                node = node_with_score.node
                metadata = node.metadata
                sources.append(SourceReference(
                    path=metadata.get("path", "unknown"),
                    title=metadata.get("title", "Untitled"),
                    snippet=node.get_content()[:500], # Truncate snippet
                    score=node_with_score.score
                ))
        
        # Handle tool outputs (Agent actions)
        if hasattr(response, "sources"):
            for tool_output in response.sources:
                if tool_output.tool_name == "create_note":
                    args = tool_output.raw_input
                    if args and "title" in args:
                        notes_written.append(NoteWritten(
                            path=f"agent-notes/{args['title']}.md", 
                            title=args["title"],
                            action="created"
                        ))
                elif tool_output.tool_name == "move_note":
                    args = tool_output.raw_input
                    if args and "path" in args:
                        notes_written.append(NoteWritten(
                            path=args.get("target_folder", "") + "/" + os.path.basename(args["path"]),
                            title=os.path.basename(args["path"]),
                            action="updated"
                        ))
                elif tool_output.tool_name == "create_folder":
                    pass

        return ChatResponse(
            answer=str(response),
            sources=sources,
            notes_written=notes_written
        )
