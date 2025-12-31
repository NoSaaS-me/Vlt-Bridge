"""Tool Executor - Dispatches tool calls to service implementations.

This service routes tool calls from the Oracle Agent to the appropriate
backend services (VaultService, IndexerService, ThreadService, OracleBridge).
It also handles schema loading and filtering by agent scope.

Timeout Configuration:
    The executor supports configurable timeouts for tool execution to prevent
    hanging operations. Timeouts can be configured at three levels:

    1. Class-level defaults (DEFAULT_TIMEOUT, TOOL_TIMEOUTS)
    2. Constructor parameter (default_timeout)
    3. Per-call override (timeout parameter in execute())

    Tool-specific timeouts are defined in TOOL_TIMEOUTS for operations that
    may naturally take longer (e.g., network requests, large searches).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.thread import ThreadEntry
from .database import DatabaseService
from .indexer import IndexerService
from .librarian_service import LibrarianService, get_librarian_service
from .oracle_bridge import OracleBridge
from .thread_service import ThreadService
from .user_settings import UserSettingsService, get_user_settings_service
from .vault import VaultService

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executes tool calls by routing to appropriate backend services.

    Supports both Oracle and Librarian agent scopes with different tool sets.
    Tool schemas are loaded from backend/prompts/tools.json with fallback
    to specs/009-oracle-agent/contracts/tools.json.

    Attributes:
        DEFAULT_TIMEOUT: Default timeout for tool execution (30 seconds)
        TOOL_TIMEOUTS: Per-tool timeout overrides for operations that need
            longer execution times (network requests, large searches, etc.)
    """

    # Default timeout for all tools (seconds)
    DEFAULT_TIMEOUT: float = 30.0

    # Tool-specific timeout overrides (seconds)
    # These are for operations that may naturally take longer than the default
    TOOL_TIMEOUTS: Dict[str, float] = {
        # Network operations - may have latency
        "web_search": 60.0,
        "web_fetch": 60.0,
        # Code search - may scan large repositories
        "search_code": 30.0,
        "find_definition": 30.0,
        "find_references": 30.0,
        "get_repo_map": 45.0,  # May process entire repo structure
        # Vault operations - local filesystem, should be fast
        "vault_read": 10.0,
        "vault_write": 10.0,
        "vault_search": 15.0,
        "vault_list": 10.0,
        "vault_move": 10.0,
        "vault_create_index": 20.0,  # May process multiple files
        # Thread operations - database operations
        "thread_push": 10.0,
        "thread_read": 10.0,
        "thread_seek": 15.0,
        "thread_list": 10.0,
        # Meta operations - may spawn subagents
        # Subagents can run for extended periods (e.g., summarizing many files, web research)
        # 20 minutes allows for large vault operations and web research without premature timeout
        "delegate_librarian": 1200.0,  # 20 minutes for large summarizations and web research
    }

    def __init__(
        self,
        vault_service: Optional[VaultService] = None,
        indexer_service: Optional[IndexerService] = None,
        thread_service: Optional[ThreadService] = None,
        oracle_bridge: Optional[OracleBridge] = None,
        db_service: Optional[DatabaseService] = None,
        user_settings_service: Optional[UserSettingsService] = None,
        librarian_service: Optional[LibrarianService] = None,
        default_timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize the tool executor with service dependencies.

        Args:
            vault_service: VaultService instance (created if None)
            indexer_service: IndexerService instance (created if None)
            thread_service: ThreadService instance (created if None)
            oracle_bridge: OracleBridge instance (created if None)
            db_service: DatabaseService instance for indexer (created if None)
            user_settings_service: UserSettingsService for accessing user model preferences
            librarian_service: LibrarianService for subagent summarization
            default_timeout: Override the default timeout for all tools (seconds).
                Individual tool timeouts from TOOL_TIMEOUTS still apply unless
                overridden at call time. If None, uses DEFAULT_TIMEOUT (30s).
        """
        self._db = db_service or DatabaseService()
        self.vault = vault_service or VaultService()
        self.indexer = indexer_service or IndexerService(self._db)
        self.threads = thread_service or ThreadService(self._db)
        self.oracle_bridge = oracle_bridge or OracleBridge()
        self.user_settings = user_settings_service or get_user_settings_service()
        self.librarian = librarian_service or get_librarian_service()

        # Instance-level default timeout (can override class default)
        self._default_timeout = default_timeout if default_timeout is not None else self.DEFAULT_TIMEOUT

        # Tool registry mapping tool names to handler methods
        self._tools: Dict[str, Any] = {
            # Code tools
            "search_code": self._search_code,
            "find_definition": self._find_definition,
            "find_references": self._find_references,
            "get_repo_map": self._get_repo_map,
            # Vault tools
            "vault_read": self._vault_read,
            "vault_write": self._vault_write,
            "vault_search": self._vault_search,
            "vault_list": self._vault_list,
            "vault_move": self._vault_move,
            "vault_create_index": self._vault_create_index,
            # Thread tools
            "thread_push": self._thread_push,
            "thread_read": self._thread_read,
            "thread_seek": self._thread_seek,
            "thread_list": self._thread_list,
            # Web tools
            "web_search": self._web_search,
            "web_fetch": self._web_fetch,
            # Meta tools
            "delegate_librarian": self._delegate_librarian,
        }

        # Cache for tool schemas
        self._schema_cache: Optional[Dict[str, Any]] = None

    def get_timeout(
        self,
        tool_name: str,
        override: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> float:
        """
        Get the timeout value for a specific tool.

        Timeout resolution order (first non-None wins):
        1. Per-call override parameter
        2. User settings (for delegate_librarian)
        3. Tool-specific timeout from TOOL_TIMEOUTS
        4. Instance-level default timeout (set in constructor)
        5. Class-level DEFAULT_TIMEOUT

        Args:
            tool_name: Name of the tool
            override: Optional per-call timeout override
            user_id: Optional user ID for per-user settings lookup

        Returns:
            Timeout value in seconds
        """
        if override is not None:
            return override

        # Check user settings for delegate_librarian
        if tool_name == "delegate_librarian" and user_id:
            user_timeout = self.user_settings.get_librarian_timeout(user_id)
            if user_timeout:
                return float(user_timeout)

        if tool_name in self.TOOL_TIMEOUTS:
            return self.TOOL_TIMEOUTS[tool_name]
        return self._default_timeout

    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        user_id: str,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Execute a tool call and return the result as a JSON string.

        Args:
            name: Tool name to execute
            arguments: Tool arguments dictionary
            user_id: User ID for scoped operations
            timeout: Optional timeout override in seconds. If not provided,
                uses tool-specific timeout from TOOL_TIMEOUTS or the default.

        Returns:
            JSON string containing the tool result or error.
            On timeout, returns a JSON object with an error field containing
            a descriptive message about the timeout, including the tool name
            and timeout duration to help agents understand and potentially
            retry with adjusted parameters.
        """
        if name not in self._tools:
            logger.warning(f"Unknown tool requested: {name}")
            return json.dumps({"error": f"Unknown tool: {name}"})

        handler = self._tools[name]
        actual_timeout = self.get_timeout(name, timeout, user_id)

        try:
            logger.info(
                f"Executing tool: {name}",
                extra={
                    "user_id": user_id,
                    "tool": name,
                    "args_keys": list(arguments.keys()),
                    "timeout": actual_timeout,
                },
            )

            # Wrap the handler call with timeout protection
            result = await asyncio.wait_for(
                handler(user_id, **arguments),
                timeout=actual_timeout,
            )
            return json.dumps(result, default=str)

        except asyncio.TimeoutError:
            # Log timeout event for debugging
            logger.warning(
                f"Tool {name} timed out after {actual_timeout}s",
                extra={
                    "user_id": user_id,
                    "tool": name,
                    "timeout": actual_timeout,
                    "args_keys": list(arguments.keys()),
                },
            )
            # Return a descriptive error that helps agents understand what happened
            return json.dumps({
                "error": f"Tool '{name}' timed out after {actual_timeout} seconds. "
                         f"The operation took too long to complete. Consider: "
                         f"(1) reducing the scope of the request (e.g., smaller limit), "
                         f"(2) breaking into smaller operations, or "
                         f"(3) retrying if this was a transient issue.",
                "timeout": actual_timeout,
                "tool": name,
                "timed_out": True,
            })
        except FileNotFoundError as e:
            logger.warning(f"Tool {name} file not found: {e}")
            return json.dumps({"error": f"File not found: {str(e)}"})
        except ValueError as e:
            logger.warning(f"Tool {name} validation error: {e}")
            return json.dumps({"error": f"Invalid arguments: {str(e)}"})
        except Exception as e:
            logger.exception(f"Tool {name} execution failed: {e}")
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

    async def execute_batch(
        self,
        tool_calls: List[Dict[str, Any]],
        user_id: str,
        timeout: Optional[float] = None,
        include_call_ids: bool = False,
    ) -> List[Any]:
        """
        Execute multiple tool calls in parallel.

        This method enables efficient parallel execution of independent tool calls,
        which is particularly useful when an LLM requests multiple tools at once.

        Args:
            tool_calls: List of tool call dictionaries, each containing:
                - name: Tool name to execute
                - arguments: Tool arguments dictionary
                - id (optional): Tool call ID for correlation
            user_id: User ID for scoped operations
            timeout: Optional timeout override in seconds (applies to each tool)
            include_call_ids: If True, returns tuples of (call_id, result_json).
                If False (default), returns just the result JSON strings.

        Returns:
            When include_call_ids=False: List of JSON strings (same as execute())
            When include_call_ids=True: List of (call_id, result_json) tuples

        Example:
            >>> results = await executor.execute_batch([
            ...     {"name": "search_code", "arguments": {"query": "auth"}, "id": "call_1"},
            ...     {"name": "vault_list", "arguments": {}, "id": "call_2"},
            ... ], user_id="user-123")
            >>> # Returns: ['{"results": [...]}', '{"notes": [...]}']
            >>>
            >>> results = await executor.execute_batch([...], "user", include_call_ids=True)
            >>> # Returns: [("call_1", '{"results": ...}'), ("call_2", '{"notes": ...}')]
        """
        if not tool_calls:
            return []

        async def execute_single(call: Dict[str, Any], index: int) -> tuple:
            """Execute a single tool call and return (call_id, result_json)."""
            tool_name = call.get("name", "")
            arguments = call.get("arguments", {})
            call_id = call.get("id", f"batch_{index}")

            result_json = await self.execute(tool_name, arguments, user_id, timeout)
            return (call_id, result_json)

        # Execute all tools in parallel using asyncio.gather
        tasks = [execute_single(call, i) for i, call in enumerate(tool_calls)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            call = tool_calls[i]
            call_id = call.get("id", f"batch_{i}")

            if isinstance(result, Exception):
                # Convert exception to error JSON
                error_json = json.dumps({"error": str(result)})
                if include_call_ids:
                    processed_results.append((call_id, error_json))
                else:
                    processed_results.append(error_json)
            else:
                # result is (call_id, result_json) tuple
                if include_call_ids:
                    processed_results.append(result)
                else:
                    processed_results.append(result[1])  # Just the result_json

        return processed_results

    def get_tool_schemas(self, agent: str = "oracle") -> List[Dict[str, Any]]:
        """
        Get OpenRouter-compatible tool schemas filtered by agent scope.

        Args:
            agent: Agent type ("oracle" or "librarian")

        Returns:
            List of tool definitions in OpenRouter function calling format
        """
        if self._schema_cache is None:
            self._schema_cache = self._load_tool_schemas()

        tools_data = self._schema_cache.get("tools", [])

        # Filter tools by agent scope and format for OpenRouter
        filtered_tools = []
        for tool in tools_data:
            agent_scope = tool.get("agent_scope", ["oracle"])
            if agent in agent_scope:
                # Return only type and function fields (OpenRouter format)
                filtered_tools.append({
                    "type": tool.get("type", "function"),
                    "function": tool.get("function", {}),
                })

        logger.debug(f"Loaded {len(filtered_tools)} tools for agent '{agent}'")
        return filtered_tools

    def _load_tool_schemas(self) -> Dict[str, Any]:
        """
        Load tool schemas from JSON file.

        Tries backend/prompts/tools.json first, falls back to
        specs/009-oracle-agent/contracts/tools.json.

        Returns:
            Parsed JSON data containing tool definitions
        """
        # Primary location: backend/prompts/tools.json
        prompts_tools = Path(__file__).parent.parent.parent / "prompts" / "tools.json"

        # Fallback location: specs/009-oracle-agent/contracts/tools.json
        specs_tools = (
            Path(__file__).parent.parent.parent.parent
            / "specs"
            / "009-oracle-agent"
            / "contracts"
            / "tools.json"
        )

        tools_file = prompts_tools if prompts_tools.exists() else specs_tools

        if not tools_file.exists():
            logger.error(f"Tool schemas not found at {prompts_tools} or {specs_tools}")
            return {"tools": []}

        try:
            with open(tools_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded tool schemas from {tools_file}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool schemas: {e}")
            return {"tools": []}

    # =========================================================================
    # Code Tool Implementations
    # =========================================================================

    async def _search_code(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search the codebase using hybrid retrieval."""
        result = await self.oracle_bridge.search_code(
            query=query,
            limit=limit,
            language=language,
        )
        return result

    async def _find_definition(
        self,
        user_id: str,
        symbol: str,
        scope: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Find where a symbol is defined."""
        # Not yet fully implemented in OracleBridge
        return {"error": f"Tool not yet implemented: find_definition"}

    async def _find_references(
        self,
        user_id: str,
        symbol: str,
        limit: int = 20,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Find all usages of a symbol."""
        # Not yet fully implemented in OracleBridge
        return {"error": f"Tool not yet implemented: find_references"}

    async def _get_repo_map(
        self,
        user_id: str,
        scope: Optional[str] = None,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get repository structure map.

        Returns a hierarchical representation of the repository structure
        suitable for LLM context. Uses OracleBridge for full CodeRAG integration
        with PageRank-based symbol importance, falling back to a lightweight
        filesystem-based map if CodeRAG isn't available.

        Args:
            user_id: User ID for scoped operations
            scope: Optional subdirectory to focus on (e.g., "src/api/")
            max_tokens: Maximum tokens for the map output
            **kwargs: Additional arguments (project, include_signatures, etc.)

        Returns:
            Dict with map_text, token_count, files_included, symbols_included, etc.
        """
        project = kwargs.get("project")

        try:
            # Try OracleBridge first (uses vlt coderag map --json)
            result = await self.oracle_bridge.get_repo_map(
                scope=scope,
                max_tokens=max_tokens,
                include_signatures=kwargs.get("include_signatures", True),
                include_docstrings=kwargs.get("include_docstrings", False),
                project=project,
            )

            # Check if result contains an actual map or an error
            if "map_text" in result:
                return result
            elif "error" in result:
                logger.warning(f"OracleBridge get_repo_map failed: {result.get('error')}")
                # Fall through to filesystem fallback
            else:
                return result

        except Exception as e:
            logger.warning(f"OracleBridge get_repo_map error: {e}, using filesystem fallback")

        # Fallback: Generate lightweight filesystem-based map
        return await self._generate_filesystem_map(scope, max_tokens)

    async def _generate_filesystem_map(
        self,
        scope: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """Generate a lightweight filesystem-based repository map.

        This is a fallback when CodeRAG isn't available. It provides a basic
        tree view of the repository structure without symbol extraction.

        Args:
            scope: Optional subdirectory to focus on
            max_tokens: Maximum approximate tokens for output

        Returns:
            Dict with map_text and metadata
        """
        from collections import defaultdict

        # Determine base path
        base_path = Path.cwd()
        if scope:
            scoped_path = base_path / scope
            if scoped_path.exists():
                base_path = scoped_path

        # File type categories for summary
        file_types: Dict[str, int] = defaultdict(int)
        key_files: List[str] = []
        tree_lines: List[str] = []

        # Important file patterns to highlight
        key_patterns = {
            "README.md", "README", "CLAUDE.md",
            "package.json", "pyproject.toml", "Cargo.toml", "go.mod",
            "setup.py", "setup.cfg", "requirements.txt",
            "Makefile", "Dockerfile", "docker-compose.yml",
            "vlt.toml", ".env.example",
            "main.py", "app.py", "index.ts", "index.js", "main.go", "main.rs",
        }

        # Directories to skip
        skip_dirs = {
            ".git", ".venv", "venv", "node_modules", "__pycache__",
            ".next", ".nuxt", "dist", "build", ".cache", ".pytest_cache",
            "target", ".tox", "htmlcov", ".mypy_cache", ".ruff_cache",
            "Ai-notes", ".specify",
        }

        # Extensions to include
        include_extensions = {
            ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java",
            ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
            ".md", ".json", ".yaml", ".yml", ".toml", ".sh", ".sql",
        }

        def estimate_tokens(text: str) -> int:
            """Rough token estimate: 1 token ~ 4 characters."""
            return len(text) // 4

        def should_include(path: Path) -> bool:
            """Check if file should be included in map."""
            return path.suffix.lower() in include_extensions

        def walk_tree(path: Path, prefix: str = "", depth: int = 0) -> None:
            """Walk directory tree and collect structure."""
            nonlocal tree_lines, file_types, key_files

            if depth > 6:  # Limit depth
                return

            try:
                entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except PermissionError:
                return

            dirs = [e for e in entries if e.is_dir() and e.name not in skip_dirs]
            files = [e for e in entries if e.is_file() and should_include(e)]

            for i, entry in enumerate(dirs + files):
                is_last = i == len(dirs) + len(files) - 1
                connector = "└── " if is_last else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")

                # Check token budget
                if estimate_tokens("\n".join(tree_lines)) > max_tokens * 0.8:
                    if not any("... (truncated)" in line for line in tree_lines):
                        tree_lines.append(f"{prefix}{connector}... (truncated)")
                    return

                if entry.is_dir():
                    tree_lines.append(f"{prefix}{connector}{entry.name}/")
                    walk_tree(entry, new_prefix, depth + 1)
                else:
                    # Track file types
                    ext = entry.suffix.lower() or "no-ext"
                    file_types[ext] += 1

                    # Check for key files
                    if entry.name in key_patterns:
                        key_files.append(str(entry.relative_to(base_path)))

                    # Add to tree with optional line count
                    try:
                        line_count = sum(1 for _ in entry.open("r", errors="ignore"))
                        tree_lines.append(f"{prefix}{connector}{entry.name} ({line_count} lines)")
                    except Exception:
                        tree_lines.append(f"{prefix}{connector}{entry.name}")

        # Build the tree
        tree_lines.append(f"{base_path.name}/")
        walk_tree(base_path)

        # Build summary section
        summary_lines = ["# Repository Map (filesystem)\n"]

        if scope:
            summary_lines.append(f"Scope: {scope}\n")

        # Key files section
        if key_files:
            summary_lines.append("\n## Key Files")
            for kf in key_files[:10]:  # Limit to 10
                summary_lines.append(f"- {kf}")

        # File types summary
        if file_types:
            summary_lines.append("\n## File Types")
            sorted_types = sorted(file_types.items(), key=lambda x: -x[1])[:10]
            for ext, count in sorted_types:
                summary_lines.append(f"- {ext}: {count} files")

        # Directory tree
        summary_lines.append("\n## Structure")
        summary_lines.append("```")
        summary_lines.extend(tree_lines)
        summary_lines.append("```")

        map_text = "\n".join(summary_lines)
        token_count = estimate_tokens(map_text)

        return {
            "map_text": map_text,
            "token_count": token_count,
            "max_tokens": max_tokens,
            "files_included": sum(file_types.values()),
            "symbols_included": 0,  # Filesystem map doesn't extract symbols
            "symbols_total": 0,
            "scope": scope,
            "source": "filesystem",  # Indicate this is a fallback
            "key_files": key_files[:10],
            "file_types": dict(sorted(file_types.items(), key=lambda x: -x[1])[:10]),
        }

    # =========================================================================
    # Vault Tool Implementations
    # =========================================================================

    async def _vault_read(
        self,
        user_id: str,
        path: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Read a markdown note from the vault."""
        note = self.vault.read_note(user_id, path)
        return {
            "path": path,
            "title": note.get("title", ""),
            "content": note.get("body", ""),
            "metadata": note.get("metadata", {}),
        }

    async def _vault_write(
        self,
        user_id: str,
        path: str,
        body: str,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create or update a markdown note in the vault."""
        note = self.vault.write_note(
            user_id,
            path,
            title=title,
            body=body,
        )
        # Index the note after writing
        self.indexer.index_note(user_id, note)
        return {
            "status": "ok",
            "path": path,
            "title": note.get("title", ""),
        }

    async def _vault_search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search the documentation vault using full-text search."""
        results = self.indexer.search_notes(user_id, query, limit=limit)
        return {
            "query": query,
            "results": results,
            "count": len(results),
        }

    async def _vault_list(
        self,
        user_id: str,
        folder: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """List notes in a vault folder."""
        notes = self.vault.list_notes(user_id, folder=folder)
        return {
            "folder": folder or "/",
            "notes": notes,
            "count": len(notes),
        }

    async def _vault_move(
        self,
        user_id: str,
        old_path: str,
        new_path: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Move or rename a note (Librarian tool)."""
        # Not yet implemented - would need to update wikilinks
        return {"error": f"Tool not yet implemented: vault_move"}

    async def _vault_create_index(
        self,
        user_id: str,
        folder: str,
        title: Optional[str] = None,
        include_summaries: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create an index.md file for a folder (Librarian tool)."""
        # Not yet implemented
        return {"error": f"Tool not yet implemented: vault_create_index"}

    # =========================================================================
    # Thread Tool Implementations
    # =========================================================================

    async def _thread_push(
        self,
        user_id: str,
        thread_id: str,
        content: str,
        entry_type: str = "thought",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Record a thought, decision, or finding to long-term memory."""
        # Create entry with unique ID
        entry_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Get current max sequence for the thread
        thread = self.threads.get_thread(user_id, thread_id, include_entries=False)
        if thread is None:
            # Create the thread first with a project_id derived from user context
            # For now, use a placeholder project
            project_id = kwargs.get("project_id", "default")
            self.threads.create_or_update_thread(
                user_id=user_id,
                thread_id=thread_id,
                project_id=project_id,
                name=thread_id,
                status="active",
            )
            next_sequence = 0
        else:
            # Get highest sequence from existing entries
            sync_status = self.threads.get_sync_status(user_id, thread_id)
            next_sequence = (sync_status.last_synced_sequence + 1) if sync_status else 0

        # Create thread entry
        entry = ThreadEntry(
            entry_id=entry_id,
            sequence_id=next_sequence,
            content=f"[{entry_type}] {content}",
            author="oracle",
            timestamp=timestamp,
        )

        # Add entry to thread
        synced_count, last_seq = self.threads.add_entries(user_id, thread_id, [entry])

        return {
            "status": "ok",
            "thread_id": thread_id,
            "entry_id": entry_id,
            "sequence_id": next_sequence,
        }

    async def _thread_read(
        self,
        user_id: str,
        thread_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Read a thread to get context and summary of past work."""
        thread = self.threads.get_thread(
            user_id,
            thread_id,
            include_entries=True,
            entries_limit=limit,
        )

        if thread is None:
            return {
                "error": f"Thread not found: {thread_id}",
                "thread_id": thread_id,
            }

        # Convert entries to serializable format
        entries = []
        if thread.entries:
            entries = [
                {
                    "entry_id": e.entry_id,
                    "content": e.content,
                    "author": e.author,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in thread.entries
            ]

        return {
            "thread_id": thread.thread_id,
            "project_id": thread.project_id,
            "name": thread.name,
            "status": thread.status,
            "entries": entries,
            "entry_count": len(entries),
        }

    async def _thread_seek(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search across all threads for relevant past context."""
        response = self.threads.search_threads(
            user_id,
            query,
            project_id=kwargs.get("project_id"),
            limit=limit,
        )

        # Convert results to serializable format
        results = [
            {
                "thread_id": r.thread_id,
                "entry_id": r.entry_id,
                "content": r.content,
                "author": r.author,
                "timestamp": r.timestamp.isoformat(),
                "score": r.score,
            }
            for r in response.results
        ]

        return {
            "query": query,
            "results": results,
            "total": response.total,
        }

    async def _thread_list(
        self,
        user_id: str,
        status: str = "active",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """List all threads for the current project."""
        # Map status filter
        status_filter = None if status == "all" else status

        response = self.threads.list_threads(
            user_id,
            project_id=kwargs.get("project_id"),
            status=status_filter,
            limit=kwargs.get("limit", 50),
            offset=kwargs.get("offset", 0),
        )

        # Convert threads to serializable format
        threads = [
            {
                "thread_id": t.thread_id,
                "project_id": t.project_id,
                "name": t.name,
                "status": t.status,
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat(),
            }
            for t in response.threads
        ]

        return {
            "threads": threads,
            "total": response.total,
        }

    # =========================================================================
    # Web Tool Implementations
    # =========================================================================

    async def _web_search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Search the web using DuckDuckGo.

        Args:
            user_id: User ID for logging/scoping
            query: Search query string
            limit: Maximum number of results (default: 5)

        Returns:
            Dictionary with query, results list, and count
        """
        try:
            from duckduckgo_search import DDGS
            from duckduckgo_search.exceptions import DuckDuckGoSearchException
        except ImportError:
            logger.error("duckduckgo-search package not installed")
            return {"error": "Web search unavailable: duckduckgo-search package not installed"}

        if not query or not query.strip():
            return {"error": "Search query cannot be empty"}

        # Clamp limit to reasonable bounds
        limit = max(1, min(limit, 20))

        try:
            # Run DuckDuckGo search synchronously in a thread pool
            # (the library is not async-native)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._execute_ddg_search(query, limit)
            )

            # Check for error from sync execution
            if isinstance(results, dict) and "error" in results:
                return results

            formatted_results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]

            logger.info(
                f"Web search completed: query='{query[:50]}...', results={len(formatted_results)}",
                extra={"user_id": user_id},
            )

            return {
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results),
            }

        except Exception as e:
            logger.exception(f"Unexpected error in web search: {e}")
            return {"error": f"Web search failed: {str(e)}"}

    def _execute_ddg_search(self, query: str, limit: int) -> Any:
        """
        Execute DuckDuckGo search synchronously.

        This is run in a thread pool to avoid blocking the async event loop.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of search results or dict with error
        """
        try:
            from duckduckgo_search import DDGS
            from duckduckgo_search.exceptions import DuckDuckGoSearchException
        except ImportError:
            return {"error": "duckduckgo-search package not installed"}

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=limit))
            return results
        except DuckDuckGoSearchException as e:
            error_msg = str(e)
            if "ratelimit" in error_msg.lower():
                logger.warning(f"DuckDuckGo rate limit hit: {e}")
                return {"error": "Search rate limited. Please try again in a few seconds."}
            logger.error(f"DuckDuckGo search error: {e}")
            return {"error": f"Search failed: {error_msg}"}
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return {"error": f"Search failed: {str(e)}"}

    async def _web_fetch(
        self,
        user_id: str,
        url: str,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Fetch and extract readable content from a URL.

        Uses httpx for HTTP requests and trafilatura for content extraction.
        Trafilatura removes ads, navigation, and other non-content elements.

        Args:
            user_id: User ID for logging/scoping
            url: URL to fetch
            max_tokens: Approximate maximum token limit (uses 4 chars per token heuristic)

        Returns:
            Dictionary with url, content, truncated flag, and extracted_at timestamp
        """
        try:
            import httpx
        except ImportError:
            logger.error("httpx package not installed")
            return {"error": "Web fetch unavailable: httpx package not installed"}

        try:
            import trafilatura
        except ImportError:
            logger.error("trafilatura package not installed")
            return {"error": "Web fetch unavailable: trafilatura package not installed"}

        # Validate URL
        if not url or not url.strip():
            return {"error": "URL cannot be empty"}

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return {"error": f"Invalid URL scheme: {url}. Must start with http:// or https://"}

        # Clamp max_tokens to reasonable bounds
        max_tokens = max(100, min(max_tokens, 10000))
        char_limit = max_tokens * 4  # Approximate 4 chars per token

        try:
            # Fetch the URL with timeout and redirect following
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, connect=10.0),
            ) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; OracleAgent/1.0; +https://example.com/bot)"
                    },
                )
                response.raise_for_status()

            # Check content type - only process HTML
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return {
                    "error": f"Unsupported content type: {content_type}. Only HTML pages are supported.",
                    "url": url,
                }

            # Extract main content using trafilatura
            # Run in executor since trafilatura can be CPU-intensive
            loop = asyncio.get_event_loop()
            extracted = await loop.run_in_executor(
                None,
                lambda: trafilatura.extract(
                    response.text,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                )
            )

            if not extracted:
                # Try with fallback extraction
                extracted = await loop.run_in_executor(
                    None,
                    lambda: trafilatura.extract(
                        response.text,
                        include_comments=False,
                        include_tables=True,
                        no_fallback=False,
                        favor_recall=True,
                    )
                )

            if not extracted:
                return {
                    "error": "Could not extract readable content from the page. The page may be dynamically rendered or have no main content.",
                    "url": url,
                }

            # Truncate to character limit
            truncated = len(extracted) > char_limit
            content = extracted[:char_limit]

            # Try to extract title
            title = None
            try:
                metadata = trafilatura.extract_metadata(response.text)
                if metadata and metadata.title:
                    title = metadata.title
            except Exception:
                pass  # Title extraction is optional

            logger.info(
                f"Web fetch completed: url='{url[:50]}...', chars={len(content)}, truncated={truncated}",
                extra={"user_id": user_id},
            )

            result = {
                "url": str(response.url),  # May differ from input URL due to redirects
                "content": content,
                "truncated": truncated,
                "extracted_at": datetime.utcnow().isoformat(),
            }

            if title:
                result["title"] = title

            if str(response.url) != url:
                result["redirected_from"] = url

            return result

        except httpx.TimeoutException:
            logger.warning(f"Web fetch timeout: {url}")
            return {"error": f"Request to {url} timed out after 30 seconds"}
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.warning(f"Web fetch HTTP error: {url} returned {status}")
            if status == 404:
                return {"error": f"Page not found (404): {url}"}
            elif status == 403:
                return {"error": f"Access forbidden (403): {url}"}
            elif status == 429:
                return {"error": f"Rate limited (429): {url}. Try again later."}
            elif status >= 500:
                return {"error": f"Server error ({status}): {url}. The site may be down."}
            else:
                return {"error": f"HTTP {status}: {url}"}
        except httpx.ConnectError as e:
            logger.warning(f"Web fetch connection error: {url} - {e}")
            return {"error": f"Could not connect to {url}. Check the URL or try again later."}
        except httpx.InvalidURL:
            return {"error": f"Invalid URL format: {url}"}
        except Exception as e:
            logger.exception(f"Unexpected error in web fetch: {e}")
            return {"error": f"Failed to fetch {url}: {str(e)}"}

    # =========================================================================
    # Meta Tool Implementations
    # =========================================================================

    async def _delegate_librarian(
        self,
        user_id: str,
        task: str,
        task_type: str = "summarize",
        files: Optional[List[str]] = None,
        folder: Optional[str] = None,
        content: Optional[str] = None,
        content_items: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 1000,
        force_refresh: bool = False,
        create_index: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Delegate a summarization or organization task to the Librarian subagent.

        The Librarian uses the user's configured subagent model (from settings)
        to process content that would be too large for the Oracle to handle directly.

        Task types:
        - "summarize": Summarize content with caching (uses LibrarianAgent)
        - "organize": Organize vault folder with index creation (uses LibrarianAgent)
        - "summarize_content": Legacy - Summarize provided content (uses LibrarianService)
        - "summarize_folder": Legacy - Summarize all notes in a folder (uses LibrarianService)
        - "summarize_search_results": Legacy - Summarize search results (uses LibrarianService)

        Args:
            user_id: User ID for scoped operations
            task: Task description or legacy task type
            task_type: "summarize" or "organize" for new agent, or legacy task type
            files: Optional list of file paths to process
            folder: Optional folder path for folder-based operations
            content: Optional content to summarize directly (legacy)
            content_items: List of {path, content, source_type} dicts for new agent
            max_tokens: Maximum tokens for summary output
            force_refresh: Bypass cache and regenerate
            create_index: For organize - whether to create index.md
            **kwargs: Additional arguments (search_results, etc.)

        Returns:
            Dict with summary, model used, and success status
        """
        import os
        from ..models.thread import ThreadEntry

        # Get user's subagent model from settings
        subagent_model = self.user_settings.get_subagent_model(user_id)
        subagent_provider = self.user_settings.get_subagent_provider(user_id)

        logger.info(
            f"Delegating to Librarian: task_type={task_type}, task={task[:50]}..., model={subagent_model}",
            extra={"user_id": user_id},
        )

        # Route to new LibrarianAgent for "summarize" and "organize" task types
        if task_type in ("summarize", "organize"):
            return await self._delegate_librarian_agent(
                user_id=user_id,
                task=task,
                task_type=task_type,
                content_items=content_items,
                files=files,
                folder=folder,
                max_tokens=max_tokens,
                force_refresh=force_refresh,
                create_index=create_index,
                **kwargs,
            )

        # Legacy task types using LibrarianService
        try:
            if task_type == "summarize_content" or task == "summarize_content":
                # Direct content summarization
                if not content:
                    return {"error": "No content provided for summarization"}

                # Create a synthetic thread entry for the librarian
                entries = [
                    ThreadEntry(
                        entry_id="synthetic",
                        sequence_id=0,
                        content=content,
                        author="oracle",
                        timestamp=datetime.utcnow(),
                    )
                ]

                result = await self.librarian.summarize_thread(
                    user_id=user_id,
                    entries=entries,
                    current_summary=None,
                    model_override=subagent_model,
                    provider_override=subagent_provider,
                )

                return {
                    "task": task_type or task,
                    "summary": result.get("summary", ""),
                    "model": result.get("model"),
                    "tokens_used": result.get("tokens_used", 0),
                    "success": result.get("success", False),
                    "error": result.get("error"),
                }

            elif task_type == "summarize_folder" or task == "summarize_folder":
                # Summarize all notes in a folder
                if not folder:
                    return {"error": "No folder path provided for folder summarization"}

                # List notes in folder
                notes = self.vault.list_notes(user_id, folder=folder)
                if not notes:
                    return {
                        "task": task_type or task,
                        "summary": f"No notes found in folder: {folder}",
                        "success": True,
                    }

                # Read and combine note contents
                combined_content = []
                for note in notes[:20]:  # Limit to 20 notes
                    try:
                        note_data = self.vault.read_note(user_id, note.get("path", ""))
                        title = note_data.get("title", note.get("path", "Untitled"))
                        body = note_data.get("body", "")[:2000]  # Limit per note
                        combined_content.append(f"## {title}\n{body}")
                    except Exception as e:
                        logger.warning(f"Failed to read note {note.get('path')}: {e}")

                if not combined_content:
                    return {
                        "task": task_type or task,
                        "summary": f"Could not read any notes in folder: {folder}",
                        "success": False,
                    }

                entries = [
                    ThreadEntry(
                        entry_id="folder-summary",
                        sequence_id=0,
                        content="\n\n".join(combined_content),
                        author="oracle",
                        timestamp=datetime.utcnow(),
                    )
                ]

                result = await self.librarian.summarize_thread(
                    user_id=user_id,
                    entries=entries,
                    current_summary=f"Summarizing {len(notes)} notes from folder: {folder}",
                    model_override=subagent_model,
                    provider_override=subagent_provider,
                )

                return {
                    "task": task_type or task,
                    "folder": folder,
                    "notes_processed": len(combined_content),
                    "summary": result.get("summary", ""),
                    "model": result.get("model"),
                    "tokens_used": result.get("tokens_used", 0),
                    "success": result.get("success", False),
                    "error": result.get("error"),
                }

            elif task_type == "summarize_search_results" or task == "summarize_search_results":
                # Summarize search results passed in kwargs
                search_results = kwargs.get("search_results", [])
                if not search_results:
                    return {"error": "No search results provided for summarization"}

                # Format search results for summarization
                formatted_results = []
                for i, r in enumerate(search_results[:15], 1):  # Limit to 15 results
                    path = r.get("path", r.get("file_path", "unknown"))
                    snippet = r.get("snippet", r.get("content", ""))[:500]
                    score = r.get("score", "N/A")
                    formatted_results.append(
                        f"**Result {i}** (score: {score})\n"
                        f"Path: {path}\n"
                        f"```\n{snippet}\n```"
                    )

                entries = [
                    ThreadEntry(
                        entry_id="search-summary",
                        sequence_id=0,
                        content="\n\n".join(formatted_results),
                        author="oracle",
                        timestamp=datetime.utcnow(),
                    )
                ]

                result = await self.librarian.summarize_thread(
                    user_id=user_id,
                    entries=entries,
                    current_summary=f"Summarizing {len(search_results)} search results",
                    model_override=subagent_model,
                    provider_override=subagent_provider,
                )

                return {
                    "task": task_type or task,
                    "results_processed": len(search_results),
                    "summary": result.get("summary", ""),
                    "model": result.get("model"),
                    "tokens_used": result.get("tokens_used", 0),
                    "success": result.get("success", False),
                    "error": result.get("error"),
                }

            else:
                return {
                    "error": f"Unknown task type: {task_type}. "
                             f"Supported: summarize, organize, summarize_content, summarize_folder, summarize_search_results"
                }

        except Exception as e:
            logger.exception(f"Librarian delegation failed: {e}")
            return {
                "task": task_type or task,
                "success": False,
                "error": f"Librarian failed: {str(e)}",
            }

    async def _delegate_librarian_agent(
        self,
        user_id: str,
        task: str,
        task_type: str,
        content_items: Optional[List[Dict[str, Any]]] = None,
        files: Optional[List[str]] = None,
        folder: Optional[str] = None,
        max_tokens: int = 1000,
        force_refresh: bool = False,
        create_index: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Delegate to the new LibrarianAgent for summarization with caching.

        The LibrarianAgent provides:
        - Content summarization with vault caching
        - Vault folder organization with index creation
        - Wikilink-based indexes for Obsidian graph integration

        Args:
            user_id: User ID for vault access
            task: Description of what to do
            task_type: "summarize" or "organize"
            content_items: List of {path, content, source_type} dicts
            files: Legacy file paths to read and summarize
            folder: Folder path for organization
            max_tokens: Maximum tokens for summary
            force_refresh: Bypass cache
            create_index: Whether to create index.md

        Returns:
            For summarize: {summary, sources, cache_path, from_cache, success}
            For organize: {index_path, files_organized, wikilinks_created, success}
        """
        import os

        # Get API key from user settings or environment
        api_key = self.user_settings.get_openrouter_api_key(user_id) or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return {"error": "OpenRouter API key not configured for Librarian", "success": False}

        # Get model from user settings (subagent model)
        settings = self.user_settings.get_settings(user_id)
        model = kwargs.get("model") or settings.subagent_model

        # Import and create librarian agent
        from .librarian_agent import LibrarianAgent

        librarian = LibrarianAgent(
            api_key=api_key,
            model=model,
            project_id=kwargs.get("project_id"),
            user_id=user_id,
            tool_executor=self,  # Pass self for vault operations
        )

        if task_type == "summarize":
            return await self._execute_librarian_summarize(
                librarian=librarian,
                task=task,
                content_items=content_items,
                files=files,
                user_id=user_id,
                max_tokens=max_tokens,
                force_refresh=force_refresh,
            )
        elif task_type == "organize":
            return await self._execute_librarian_organize(
                librarian=librarian,
                task=task,
                folder=folder,
                create_index=create_index,
            )
        else:
            return {"error": f"Unknown task_type: {task_type}. Use 'summarize' or 'organize'.", "success": False}

    async def _execute_librarian_summarize(
        self,
        librarian: Any,
        task: str,
        content_items: Optional[List[Dict[str, Any]]],
        files: Optional[List[str]],
        user_id: str,
        max_tokens: int,
        force_refresh: bool,
    ) -> Dict[str, Any]:
        """Execute summarization through the LibrarianAgent.

        Args:
            librarian: LibrarianAgent instance
            task: Summarization task description
            content_items: Content items to summarize
            files: Legacy file paths (converted to content)
            user_id: User ID for vault access
            max_tokens: Maximum summary tokens
            force_refresh: Bypass cache

        Returns:
            Summary result dict with success flag
        """
        content = content_items

        # Handle legacy 'files' parameter by reading content
        if content is None and files:
            content = []
            for file_path in files:
                try:
                    result = await self._vault_read(user_id, file_path)
                    if "error" not in result:
                        content.append({
                            "path": file_path,
                            "content": result.get("content", ""),
                            "source_type": "vault",
                        })
                except Exception as e:
                    logger.warning(f"Failed to read {file_path} for summarization: {e}")

        if not content:
            return {"error": "No content provided for summarization", "success": False}

        # Collect streaming chunks into final result
        result = {
            "task": "summarize",
            "summary": "",
            "sources": [],
            "cache_path": None,
            "from_cache": False,
            "success": True,
        }

        try:
            async for chunk in librarian.summarize(
                task=task,
                content=content,
                max_summary_tokens=max_tokens,
                force_refresh=force_refresh,
            ):
                if chunk.type == "summary":
                    result["summary"] = chunk.content or ""
                    result["sources"] = chunk.sources or []
                    result["cache_path"] = chunk.cache_path
                elif chunk.type == "cache_hit":
                    result["summary"] = chunk.content or ""
                    result["sources"] = chunk.sources or []
                    result["cache_path"] = chunk.cache_path
                    result["from_cache"] = True
                elif chunk.type == "error":
                    return {"error": chunk.content or "Summarization failed", "success": False}
                elif chunk.type == "done":
                    if chunk.metadata:
                        result["from_cache"] = chunk.metadata.get("from_cache", False)
                        if not result["cache_path"]:
                            result["cache_path"] = chunk.metadata.get("cache_path")

            return result

        except Exception as e:
            logger.exception(f"LibrarianAgent summarization failed: {e}")
            return {"error": f"Summarization failed: {str(e)}", "success": False}

    async def _execute_librarian_organize(
        self,
        librarian: Any,
        task: str,
        folder: Optional[str],
        create_index: bool,
    ) -> Dict[str, Any]:
        """Execute vault organization through the LibrarianAgent.

        Args:
            librarian: LibrarianAgent instance
            task: Organization task description
            folder: Folder to organize
            create_index: Whether to create index.md

        Returns:
            Organization result dict with success flag
        """
        if not folder:
            return {"error": "Folder path required for organization task", "success": False}

        # Collect streaming chunks into final result
        result = {
            "task": "organize",
            "folder": folder,
            "index_path": None,
            "files_organized": 0,
            "wikilinks_created": 0,
            "index_content": None,
            "success": True,
        }

        try:
            async for chunk in librarian.organize(
                folder=folder,
                create_index=create_index,
                task=task,
            ):
                if chunk.type == "index":
                    result["index_content"] = chunk.content
                    result["wikilinks_created"] = len(chunk.sources or [])
                elif chunk.type == "error":
                    return {"error": chunk.content or "Organization failed", "success": False}
                elif chunk.type == "done":
                    if chunk.metadata:
                        result["index_path"] = chunk.metadata.get("index_path")
                        result["files_organized"] = chunk.metadata.get("files_organized", 0)
                        result["wikilinks_created"] = chunk.metadata.get("wikilinks_created", 0)

            return result

        except Exception as e:
            logger.exception(f"LibrarianAgent organization failed: {e}")
            return {"error": f"Organization failed: {str(e)}", "success": False}


# Singleton instance for dependency injection
_tool_executor: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    """Get or create the tool executor singleton."""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor()
    return _tool_executor


__all__ = ["ToolExecutor", "get_tool_executor"]
