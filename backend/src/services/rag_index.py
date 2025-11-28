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
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
except ImportError as e:
    GoogleGenAI = None
    GoogleGenAIEmbedding = None
    logger.warning(f"Could not import google_genai modules: {e}")

from llama_index.core.base.response.schema import Response as LlamaResponse
from llama_index.core.llms import ChatMessage as LlamaChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

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

        # Import here to avoid circular dependency
        from .indexer import IndexerService
        from .database import DatabaseService
        self.indexer_service = IndexerService(DatabaseService())
        self._index_lock = threading.Lock() # Per-instance lock for index ops
        self._setup_gemini()
        self._initialized = True

    def _setup_gemini(self):
        """Configure global LlamaIndex settings for Gemini."""
        if not GoogleGenAI or not GoogleGenAIEmbedding:
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
            # Using 2.5 Flash-Lite: 15 RPM quota, good balance for multi-tool agents
            # Backup from 2.5-flash (10 RPM, better quality but lower quota)
            Settings.llm = GoogleGenAI(
                model="gemini-2.5-flash-lite",
                api_key=self.config.google_api_key
            )
            Settings.embed_model = GoogleGenAIEmbedding(
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
                written_note = self.vault_service.write_note(
                    user_id,
                    path,
                    title=title,
                    body=content,
                    metadata={"created_by": "gemini-agent"}
                )

                # Index the note immediately so graph view updates
                try:
                    self.indexer_service.index_note(user_id, written_note)
                    logger.info(f"[RAG] Indexed note after creation: {path}")
                except Exception as idx_err:
                    logger.warning(f"[RAG] Failed to index note {path}: {idx_err}")
                    # Continue even if indexing fails

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

    def _list_notes_tool(self, user_id: str):
        """Create a tool for listing all notes in the vault."""
        def list_notes(folder: str = "") -> str:
            """
            List all notes in the vault, optionally filtered by folder.
            Use this to discover what notes exist before reading or referencing them.

            Args:
                folder: Optional folder path to filter results (e.g. "agent-notes"). Leave empty to list all notes.

            Returns:
                A formatted list of note paths and titles.
            """
            try:
                notes = self.vault_service.list_notes(user_id)

                # Filter by folder if specified
                if folder:
                    folder_normalized = folder.strip("/")
                    notes = [n for n in notes if n["path"].startswith(folder_normalized)]

                if not notes:
                    return f"No notes found{' in folder: ' + folder if folder else ''}."

                # Format as list with paths and titles
                result = f"Found {len(notes)} note(s):\n\n"
                for note in notes[:100]:  # Limit to 100 to avoid huge responses
                    result += f"- **{note['title']}** (`{note['path']}`)\n"

                if len(notes) > 100:
                    result += f"\n... and {len(notes) - 100} more notes."

                return result
            except Exception as e:
                return f"Failed to list notes: {e}"

        return FunctionTool.from_defaults(fn=list_notes)

    def _read_note_tool(self, user_id: str):
        """Create a tool for reading a specific note by path."""
        def read_note(path: str) -> str:
            """
            Read the full content of a specific note by its path.
            Use this when you need the complete content of a note, not just search snippets.

            Args:
                path: The path to the note (e.g. "architecture/API Design.md" or "agent-notes/Meeting Notes.md").

            Returns:
                The full markdown content of the note including title and body.
            """
            try:
                note = self.vault_service.read_note(user_id, path)
                return f"# {note['title']}\n\n{note['body']}"
            except Exception as e:
                return f"Failed to read note at '{path}': {e}"

        return FunctionTool.from_defaults(fn=read_note)

    def _update_note_tool(self, user_id: str):
        """Create a tool for updating an existing note."""
        def update_note(path: str, content: str) -> str:
            """
            Update an existing note's content. Use this to edit or append to existing notes.

            Args:
                path: The path to the note to update (e.g. "agent-notes/Summary.md").
                content: The new markdown content for the note body.

            Returns:
                Confirmation message with the updated path.
            """
            try:
                # Read existing note to get title and metadata
                existing = self.vault_service.read_note(user_id, path)

                # Update with new content, preserving title
                updated_note = self.vault_service.write_note(
                    user_id,
                    path,
                    title=existing["title"],
                    body=content,
                    metadata={**existing.get("metadata", {}), "updated_by": "gemini-agent"}
                )

                # Index the note immediately so graph view updates
                try:
                    self.indexer_service.index_note(user_id, updated_note)
                    logger.info(f"[RAG] Indexed note after update: {path}")
                except Exception as idx_err:
                    logger.warning(f"[RAG] Failed to index note {path}: {idx_err}")

                return f"Note updated successfully at {path}"
            except Exception as e:
                return f"Failed to update note at '{path}': {e}"

        return FunctionTool.from_defaults(fn=update_note)

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

        # Create memory with chat history for context persistence
        memory = ChatMemoryBuffer.from_defaults(token_limit=8000)

        # Load previous messages into memory
        for m in messages[:-1]:  # Exclude last message (the current query)
            role = MessageRole.USER if m.role == "user" else MessageRole.ASSISTANT
            memory.put(LlamaChatMessage(role=role, content=m.content))

        # Define all available tools
        tools = [
            # Read/Browse tools (use these first to understand the vault)
            self._list_notes_tool(user_id),
            self._read_note_tool(user_id),

            # Write/Modify tools
            self._create_note_tool(user_id),
            self._update_note_tool(user_id),

            # Organization tools
            self._move_note_tool(user_id),
            self._create_folder_tool(user_id)
        ]

        from llama_index.core.tools import QueryEngineTool, ToolMetadata

        # RAG search tool for semantic queries
        query_tool = QueryEngineTool(
            query_engine=index.as_query_engine(),
            metadata=ToolMetadata(
                name="vault_search",
                description="Semantic search across vault content. Use when you need to find notes by topic/concept, not by exact path or title."
            )
        )

        all_tools = tools + [query_tool]
        
        # Use FunctionAgent (new in 0.14.x, replaces FunctionCallingAgent)
        try:
            from llama_index.core.agent import FunctionAgent
        except ImportError:
            # Fallback for older versions just in case, or log error
            logger.error("Could not import FunctionAgent. Check llama-index-core version.")
            raise

        # Create FunctionAgent with tools (0.14.x pattern)
        agent = FunctionAgent(
            tools=all_tools,
            llm=Settings.llm,
            verbose=True,
            system_prompt="""You are an autonomous documentation assistant with full access to a markdown vault.

CORE BEHAVIORS:
1. **Be Proactive**: Take initiative. Don't ask for information you can discover using tools.
2. **Use Tools First**: Before asking the user for details, use list_notes, read_note, or vault_search to gather information.
3. **Make Reasonable Decisions**: When creating content, use your judgment to generate appropriate titles, structure, and content based on context.
4. **Multi-Step Thinking**: Break complex tasks into steps and execute them autonomously.
5. **Remember Context**: Pay attention to the conversation history and use information from previous messages.

AVAILABLE TOOLS:
- list_notes: Discover what notes exist (use this FIRST for browsing tasks)
- read_note: Read complete note content by path
- vault_search: Semantic search for finding notes by topic/concept
- create_note: Create new notes in agent-notes/ folder
- update_note: Edit existing notes
- move_note, create_folder: Organize notes

WORKFLOW EXAMPLES:

User: "Create an index of all notes"
→ 1. Use list_notes() to get all notes
→ 2. Use read_note() on key notes to get summaries
→ 3. Generate index content autonomously
→ 4. Use create_note() with a sensible title like "Note Index"

User: "Summarize the ChatGPT integration"
→ 1. Use vault_search("ChatGPT integration") to find relevant notes
→ 2. Read the full content if needed
→ 3. Provide summary directly (don't ask for permission)

User: "Create a note based on this chat"
→ 1. Review conversation history
→ 2. Generate appropriate title (e.g., "Chat Summary - [Date]")
→ 3. Create structured content from the discussion
→ 4. Use create_note() immediately

NEVER:
- Ask repeatedly for the same information
- Ask "what should I name it?" when context provides clear answers
- Say "I cannot do X" when you have tools that enable X
- Request permission for routine operations (creating notes, searching, etc.)

ALWAYS:
- Execute tasks end-to-end autonomously
- Provide clear confirmation of what you did
- Use wikilink syntax [[Note Name]] when referencing notes
- Put new notes in agent-notes/ folder unless specified otherwise"""
        )

        # Use .run() method with memory for context persistence (0.14.x pattern)
        response = await agent.run(user_msg=query_text, memory=memory)

        logger.info(f"[RAG] Agent returned response type: {type(response)}")
        logger.info(f"[RAG] Response string: {str(response)[:200]}")

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

        # Handle tool calls (Agent actions) - LlamaIndex 0.14.x uses 'tool_calls'
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"[RAG] Processing {len(response.tool_calls)} tool calls")
            for tool_call in response.tool_calls:
                tool_name = tool_call.tool_name
                tool_kwargs = tool_call.tool_kwargs
                logger.info(f"[RAG] Tool: {tool_name}, args: {tool_kwargs}")

                if tool_name == "create_note":
                    if "title" in tool_kwargs:
                        notes_written.append(NoteWritten(
                            path=f"agent-notes/{tool_kwargs['title']}.md",
                            title=tool_kwargs["title"],
                            action="created"
                        ))
                elif tool_name == "update_note":
                    if "path" in tool_kwargs:
                        notes_written.append(NoteWritten(
                            path=tool_kwargs["path"],
                            title=os.path.basename(tool_kwargs["path"]).replace(".md", ""),
                            action="updated"
                        ))
                elif tool_name == "move_note":
                    if "path" in tool_kwargs:
                        notes_written.append(NoteWritten(
                            path=tool_kwargs.get("target_folder", "") + "/" + os.path.basename(tool_kwargs["path"]),
                            title=os.path.basename(tool_kwargs["path"]),
                            action="updated"
                        ))
                elif tool_name == "create_folder":
                    pass

        logger.info(f"[RAG] Response formatted: {len(sources)} sources, {len(notes_written)} notes_written")
        return ChatResponse(
            answer=str(response),
            sources=sources,
            notes_written=notes_written
        )
