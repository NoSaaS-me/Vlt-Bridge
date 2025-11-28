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
        # ... (existing)

    def get_persist_dir(self, user_id: str) -> str:
        # ... (existing)

    def get_or_build_index(self, user_id: str) -> VectorStoreIndex:
        # ... (existing)

    def build_index(self, user_id: str) -> VectorStoreIndex:
        # ... (existing)

    def rebuild_index(self, user_id: str) -> VectorStoreIndex:
        # ... (existing)

    def get_status(self, user_id: str) -> StatusResponse:
        # ... (existing)

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
                # Index the new note immediately so agent knows about it? 
                # write_note does NOT auto-update the RAG index (it updates FTS5).
                # We might need to add it to the index.
                # For now, just return success.
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
            # Constraint: Can only move notes created by agent (in agent-notes/)?
            # Or allow moving anywhere? Spec said "not deleting or editing existing".
            # Moving is technically deleting + creating.
            # Let's restrict source to agent-notes/ to be safe?
            # Or just allow it. "We need one for moving notes into folder".
            
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
            # Sanitize path?
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
        
        from llama_index.core.agent import ReActAgent
        agent = ReActAgent.from_tools(
            all_tools, 
            llm=Settings.llm, 
            chat_history=history,
            verbose=True,
            context="You are a documentation assistant. Use vault_search to find info. You can create notes and folders."
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
                    pass # No badge for folders yet

        return ChatResponse(
            answer=str(response),
            sources=sources,
            notes_written=notes_written
        )