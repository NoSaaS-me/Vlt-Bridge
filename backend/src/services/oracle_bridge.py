"""Oracle Bridge Service - Bridges Document-MCP to vlt-cli oracle and coderag.

This bridge can operate in two modes:
1. Direct import mode (preferred): Import vlt modules directly from packages/vlt-cli
2. Subprocess mode (fallback): Call vlt CLI via subprocess

Direct import is faster and allows better integration. Subprocess mode is used
when vlt-cli is not available as a Python package in the same environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from .thread_retriever import ThreadRetriever, get_thread_retriever

logger = logging.getLogger(__name__)

# Try to add packages/vlt-cli to Python path for direct import
_PACKAGES_DIR = Path(__file__).parent.parent.parent.parent / "packages" / "vlt-cli" / "src"
if _PACKAGES_DIR.exists() and str(_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGES_DIR))
    logger.info(f"Added vlt-cli to Python path: {_PACKAGES_DIR}")

# Try direct imports from vlt-cli
_DIRECT_IMPORT_AVAILABLE = False
try:
    from vlt.core.oracle import OracleOrchestrator
    from vlt.core.schemas import OracleQuery
    from vlt.core.coderag.indexer import CodeRAGIndexer
    from vlt.core.coderag.repomap import generate_repo_map
    from vlt.core.retrievers.hybrid import hybrid_retrieve
    _DIRECT_IMPORT_AVAILABLE = True
    logger.info("Direct vlt-cli import available - using direct mode")
except ImportError as e:
    logger.warning(f"Direct vlt-cli import not available, using subprocess mode: {e}")


class OracleBridgeError(Exception):
    """Raised when oracle bridge operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


def _find_vlt_command() -> str:
    """Find vlt CLI executable, checking common locations.

    Priority order:
    1. VLT_CLI_PATH environment variable
    2. packages/vlt-cli/.venv/bin/vlt (local venv)
    3. 'vlt' in PATH
    """
    # 1. Check env var
    env_path = os.getenv("VLT_CLI_PATH")
    if env_path and os.path.isfile(env_path):
        logger.info(f"Using vlt from VLT_CLI_PATH: {env_path}")
        return env_path

    # 2. Check packages/vlt-cli/.venv/bin/vlt
    project_root = Path(__file__).parent.parent.parent.parent
    vlt_venv_path = project_root / "packages" / "vlt-cli" / ".venv" / "bin" / "vlt"
    if vlt_venv_path.exists():
        logger.info(f"Using vlt from local venv: {vlt_venv_path}")
        return str(vlt_venv_path)

    # 3. Fall back to PATH
    return "vlt"


class OracleBridge:
    """
    Bridge service that integrates Document-MCP with vlt-cli oracle and coderag.

    Uses subprocess calls to vlt CLI for oracle and code intelligence operations.
    This approach ensures we use the production vlt-cli implementation without
    coupling to internal vlt modules.
    """

    def __init__(self, vlt_command: Optional[str] = None):
        """
        Initialize the oracle bridge.

        Args:
            vlt_command: Path to vlt CLI executable (auto-detected if None)
        """
        self.vlt_command = vlt_command or _find_vlt_command()
        # Store conversation history per user session
        # Key: user_id, Value: list of conversation messages
        self._conversation_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # Cache for vlt command availability check
        self._vlt_available: Optional[bool] = None
        # Cache for CodeRAG initialization status
        self._coderag_initialized: Optional[bool] = None

    def _check_vlt_available(self) -> bool:
        """Check if vlt command is available.

        Returns:
            True if vlt is available, False otherwise
        """
        if self._vlt_available is not None:
            return self._vlt_available

        try:
            # Use --help since vlt doesn't have --version
            result = subprocess.run(
                [self.vlt_command, "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # --help returns 0 on success
            self._vlt_available = result.returncode == 0
            if self._vlt_available:
                logger.info(f"vlt CLI available at: {self.vlt_command}")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning(f"Failed to check vlt availability: {e}")
            self._vlt_available = False

        if not self._vlt_available:
            logger.warning(f"vlt command not available: {self.vlt_command}")

        return self._vlt_available

    def _check_coderag_initialized(self, project: Optional[str] = None) -> bool:
        """Check if CodeRAG index exists for the project.

        Args:
            project: Project ID to check (auto-detected if None)

        Returns:
            True if CodeRAG is initialized, False otherwise
        """
        # Skip check if vlt is not available
        if not self._check_vlt_available():
            return False

        try:
            args = ["coderag", "status"]
            if project:
                args.extend(["--project", project])
            args.append("--json")

            logger.debug(f"Checking CodeRAG status with args: {args}")
            result = subprocess.run(
                [self.vlt_command] + args,
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.debug(f"CodeRAG status result: returncode={result.returncode}, stdout={result.stdout[:200] if result.stdout else 'empty'}, stderr={result.stderr[:200] if result.stderr else 'empty'}")

            if result.returncode != 0:
                logger.warning(f"CodeRAG status check failed: returncode={result.returncode}, stderr={result.stderr}")
                self._coderag_initialized = False
                return False

            # Try to parse the status response
            try:
                status = json.loads(result.stdout)
                # Check if index exists based on status response
                # Status can be "ready", "indexing", etc.
                # Chunks can be in chunks_created or index_stats.chunks_count
                is_ready = status.get("status") == "ready"
                has_chunks = (
                    status.get("chunks_created", 0) > 0 or
                    status.get("index_stats", {}).get("chunks_count", 0) > 0
                )
                self._coderag_initialized = is_ready or has_chunks
                logger.info(f"CodeRAG initialized check: status={status.get('status')}, chunks={has_chunks}, result={self._coderag_initialized}")
                return self._coderag_initialized
            except json.JSONDecodeError:
                # If output isn't JSON, check for success indicators in text
                self._coderag_initialized = "ready" in result.stdout.lower()
                return self._coderag_initialized

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning(f"Failed to check CodeRAG status: {e}")
            self._coderag_initialized = False
            return False

    def _run_vlt_command(
        self,
        args: List[str],
        timeout: int = 60,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a vlt CLI command and parse JSON output.

        Args:
            args: Command arguments (after 'vlt')
            timeout: Command timeout in seconds
            env_vars: Optional environment variables to pass to the subprocess

        Returns:
            Parsed JSON response from vlt

        Raises:
            OracleBridgeError: If command fails or returns invalid JSON
        """
        # Check vlt availability before running
        if not self._check_vlt_available():
            return {
                "error": True,
                "message": "vlt CLI is not available. Please install vlt-cli or ensure it's in PATH.",
                "suggestion": "Run 'pip install vlt-cli' or check your PATH configuration.",
            }

        cmd = [self.vlt_command] + args + ["--json"]

        # Build environment with optional additional vars
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
            logger.debug(f"Passing environment variables: {list(env_vars.keys())}")

        try:
            logger.info(f"Running vlt command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
                env=env,
            )

            # Validate output before parsing
            stdout = result.stdout.strip()
            if not stdout:
                logger.warning(f"vlt command returned empty output: {' '.join(cmd)}")
                return {
                    "error": True,
                    "message": "vlt command returned empty response",
                    "command": ' '.join(args),
                }

            # Parse JSON output
            try:
                return json.loads(stdout)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse vlt JSON output: {stdout[:500]}")
                # Return structured error instead of raising
                return {
                    "error": True,
                    "message": f"Invalid JSON response from vlt: {str(e)}",
                    "raw_output": stdout[:1000] if len(stdout) > 1000 else stdout,
                    "command": ' '.join(args),
                }

        except subprocess.TimeoutExpired as e:
            logger.error(f"vlt command timeout: {' '.join(cmd)}")
            return {
                "error": True,
                "message": f"vlt command timeout after {timeout}s",
                "command": ' '.join(args),
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"vlt command failed: {e.stderr}")
            # Try to parse stderr as JSON (some vlt errors are JSON formatted)
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            try:
                error_data = json.loads(error_msg)
                error_msg = error_data.get("message", error_msg)
            except json.JSONDecodeError:
                pass
            return {
                "error": True,
                "message": f"vlt command failed: {error_msg}",
                "command": ' '.join(args),
                "returncode": e.returncode,
            }

        except FileNotFoundError:
            logger.error(f"vlt command not found: {self.vlt_command}")
            self._vlt_available = False
            return {
                "error": True,
                "message": "vlt CLI not found. Please install vlt-cli.",
                "suggestion": "Run 'pip install vlt-cli' or add vlt to your PATH.",
            }

        except OSError as e:
            logger.error(f"OS error running vlt: {e}")
            return {
                "error": True,
                "message": f"Failed to run vlt command: {str(e)}",
            }

    async def ask_oracle_stream(
        self,
        user_id: str,
        question: str,
        sources: Optional[List[str]] = None,
        explain: bool = False,
        model: Optional[str] = None,
        thinking: bool = False,
        project: Optional[str] = None,
        max_tokens: int = 16000,
        openrouter_api_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Ask Oracle a question with streaming response.

        Args:
            user_id: User ID for conversation history tracking
            question: Natural language question
            sources: Knowledge sources to query (vault, code, threads) - None means all
            explain: Include retrieval traces
            model: Override LLM model to use
            thinking: Enable thinking mode (append :thinking suffix to model)
            project: Project ID (auto-detected if None)
            max_tokens: Maximum tokens for context assembly
            openrouter_api_key: User's OpenRouter API key (passed as env var)

        Yields:
            Streaming chunks as dictionaries
        """
        # Build command args
        args = ["oracle", question]

        if project:
            args.extend(["--project", project])

        if sources:
            for source in sources:
                args.extend(["--source", source])

        if explain:
            args.append("--explain")

        if model:
            # Apply thinking suffix if requested AND model supports it
            actual_model = model
            if thinking:
                model_lower = model.lower()
                supports_thinking = (
                    "-r1" in model_lower
                    or "/r1" in model_lower
                    or "/o1" in model_lower
                    or "/o3" in model_lower
                    or "claude-3.7-sonnet" in model_lower
                    or ("qwen" in model_lower and "thinking" in model_lower)
                )
                if supports_thinking:
                    actual_model = f"{model}:thinking"
                else:
                    logger.warning(
                        f"Thinking mode requested but model '{model}' does not support :thinking suffix."
                    )
            args.extend(["--model", actual_model])

        args.extend(["--max-tokens", str(max_tokens)])

        # For streaming, we'll use subprocess with line-buffered output
        # Note: This assumes vlt CLI supports --stream flag
        # If not available yet, we fall back to simulating streaming
        cmd = [self.vlt_command] + args

        try:
            logger.info(f"Running streaming vlt command: {' '.join(cmd)}")

            # Store question in conversation history
            self._add_to_history(user_id, "user", question)

            # Build environment with API key if provided
            env = os.environ.copy()
            if openrouter_api_key:
                env["VLT_OPENROUTER_API_KEY"] = openrouter_api_key
                logger.debug("Passing OpenRouter API key to vlt subprocess")

            # Start subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Simulate streaming chunks
            # TODO: Replace with actual streaming when vlt CLI supports it
            yield {"type": "thinking", "content": "Searching knowledge sources..."}
            await asyncio.sleep(0.1)

            # Retrieve thread context using direct database query
            thread_retriever = get_thread_retriever()
            thread_context, thread_citations = thread_retriever.get_context_for_query(
                user_id=user_id,
                query=question,
                project_id=project,
                max_tokens=max_tokens // 4,  # Reserve 1/4 of context for threads
            )

            if thread_context:
                yield {"type": "thinking", "content": f"Found {len(thread_citations)} relevant thread entries..."}
                await asyncio.sleep(0.1)

            yield {"type": "thinking", "content": "Retrieving relevant context..."}
            await asyncio.sleep(0.1)

            yield {"type": "thinking", "content": "Analyzing code and documentation..."}
            await asyncio.sleep(0.1)

            # Wait for process to complete
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"vlt command failed: {error_msg}")
                yield {
                    "type": "error",
                    "error": f"Oracle query failed: {error_msg}"
                }
                return

            # Parse JSON output
            try:
                result = json.loads(stdout.decode())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse vlt JSON output: {stdout.decode()}")
                yield {
                    "type": "error",
                    "error": "Invalid response format from oracle"
                }
                return

            # Stream the answer content
            answer = result.get("answer", "")

            # Split answer into chunks for streaming effect
            chunk_size = 50
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield {"type": "content", "content": chunk}
                await asyncio.sleep(0.01)  # Small delay for streaming effect

            # Stream sources from subprocess result
            sources_list = result.get("sources", [])
            for source in sources_list:
                yield {"type": "source", "source": source}
                await asyncio.sleep(0.01)

            # Stream thread citations (from direct DB query)
            for citation in thread_citations:
                yield {"type": "source", "source": citation.model_dump()}
                await asyncio.sleep(0.01)

            # Combine all sources for history
            all_sources = sources_list + [c.model_dump() for c in thread_citations]

            # Store answer in conversation history
            self._add_to_history(
                user_id,
                "assistant",
                answer,
                sources=all_sources
            )

            # Done chunk
            yield {
                "type": "done",
                "tokens_used": result.get("tokens_used"),
                "model_used": result.get("model_used"),
            }

        except asyncio.TimeoutError:
            logger.error("vlt command timeout")
            yield {
                "type": "error",
                "error": "Oracle query timeout"
            }
        except Exception as e:
            logger.exception(f"Oracle streaming failed: {e}")
            yield {
                "type": "error",
                "error": f"Oracle error: {str(e)}"
            }

    async def ask_oracle(
        self,
        question: str,
        sources: Optional[List[str]] = None,
        explain: bool = False,
        project: Optional[str] = None,
        max_tokens: int = 16000,
    ) -> Dict[str, Any]:
        """
        Ask Oracle a question about the codebase.

        Args:
            question: Natural language question
            sources: Knowledge sources to query (vault, code, threads) - None means all
            explain: Include retrieval traces
            project: Project ID (auto-detected if None)
            max_tokens: Maximum tokens for context assembly

        Returns:
            Oracle response with answer and sources
        """
        args = ["oracle", question]

        if project:
            args.extend(["--project", project])

        if sources:
            for source in sources:
                args.extend(["--source", source])

        if explain:
            args.append("--explain")

        args.extend(["--max-tokens", str(max_tokens)])

        return self._run_vlt_command(args, timeout=90)

    async def search_code(
        self,
        query: str,
        limit: int = 10,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None,
        project: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search code using hybrid retrieval (vector + BM25).

        Args:
            query: Search query
            limit: Maximum results to return
            language: Filter by programming language
            file_pattern: File pattern filter (not directly supported - would need implementation)
            project: Project ID (auto-detected if None)
            openrouter_api_key: User's OpenRouter API key for vector search (passed as env var)

        Returns:
            Search results with code chunks
        """
        # Check if CodeRAG is initialized before searching
        if not self._check_coderag_initialized(project):
            logger.warning(f"CodeRAG not initialized for project: {project or 'default'}")
            return {
                "error": True,
                "message": "CodeRAG index not initialized. Please run 'vlt coderag init' first.",
                "suggestion": "Initialize the code index with: vlt coderag init --project <project>",
                "results": [],
            }

        args = ["coderag", "search", query, "--limit", str(limit)]

        if project:
            args.extend(["--project", project])

        if language:
            args.extend(["--language", language])

        # Note: file_pattern filtering would need to be implemented in vlt-cli
        # or filtered post-retrieval
        if file_pattern:
            logger.warning(f"file_pattern filtering not yet supported: {file_pattern}")

        # Pass API key as environment variable for vector search
        env_vars = {}
        if openrouter_api_key:
            env_vars["VLT_OPENROUTER_API_KEY"] = openrouter_api_key
            logger.debug("Passing OpenRouter API key for vector search")

        return self._run_vlt_command(args, timeout=60, env_vars=env_vars if env_vars else None)

    async def find_definition(
        self,
        symbol: str,
        scope: Optional[str] = None,
        kind: Optional[str] = None,
        project: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find where a symbol is defined using direct code search.

        Uses search_code to find definitions without invoking oracle
        (avoiding circular agent calls).

        Args:
            symbol: Symbol name to find
            scope: Optional file path to narrow search
            kind: Symbol kind filter (function, class, method, variable, constant)
            project: Project ID (auto-detected if None)
            openrouter_api_key: API key for vector search

        Returns:
            Search results for the symbol definition
        """
        # Build a simple search query for the symbol
        # The hybrid retrieval system (BM25 + vector + graph) naturally ranks
        # definitions higher through:
        # - BM25 field weighting (qualified_name, signature fields)
        # - Vector semantic similarity to definition contexts
        # - Graph PageRank (definitions are highly connected nodes)
        query = symbol

        if kind:
            # Optionally narrow by kind (e.g., "VaultService class")
            query = f"{symbol} {kind}"

        # Use direct code search instead of oracle
        return await self.search_code(
            query=query,
            limit=10,
            language=None,
            openrouter_api_key=openrouter_api_key,
        )

    async def find_references(
        self,
        symbol: str,
        limit: int = 20,
        include_definition: bool = False,
        reference_type: str = "all",
        project: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find all references to a symbol using direct code search.

        Uses search_code to find usage without invoking oracle
        (avoiding circular agent calls).

        Args:
            symbol: Symbol name to find references for
            limit: Maximum references to return
            include_definition: Include the definition in results
            reference_type: Type of references (calls, imports, inherits, all)
            project: Project ID (auto-detected if None)
            openrouter_api_key: API key for vector search

        Returns:
            Search results for the symbol usage
        """
        # Build a targeted search query for references
        query = symbol

        # Add context keywords based on reference type
        if reference_type == "calls":
            query += " call invoke"
        elif reference_type == "imports":
            query += " import from require"
        elif reference_type == "inherits":
            query += " extends implements inherit"

        # Use direct code search instead of oracle
        return await self.search_code(
            query=query,
            limit=limit,
            language=None,
            openrouter_api_key=openrouter_api_key,
        )

    async def get_repo_map(
        self,
        scope: Optional[str] = None,
        max_tokens: int = 4000,
        include_signatures: bool = True,
        include_docstrings: bool = False,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get repository structure map.

        Args:
            scope: Subdirectory to focus on
            max_tokens: Maximum tokens for the map
            include_signatures: Include function/method signatures
            include_docstrings: Include docstrings
            project: Project ID (auto-detected if None)

        Returns:
            Repository map with stats
        """
        args = ["coderag", "map"]

        if project:
            args.extend(["--project", project])

        # Note: These options may need to be added to vlt-cli coderag map command
        # For now, log warnings if they're used
        if scope:
            logger.warning(f"scope filtering not yet supported in vlt coderag map: {scope}")

        if max_tokens != 4000:
            logger.warning(f"max_tokens configuration not yet supported: {max_tokens}")

        if not include_signatures:
            logger.warning("Signatures are always included in current implementation")

        if include_docstrings:
            logger.warning("Docstring inclusion not yet configurable")

        return self._run_vlt_command(args, timeout=60)

    def _add_to_history(
        self,
        user_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add a message to conversation history.

        Args:
            user_id: User ID
            role: Message role (user, assistant, system)
            content: Message content
            sources: Optional source citations
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if sources:
            message["sources"] = sources

        self._conversation_history[user_id].append(message)

        # Keep only last 50 messages to prevent unbounded growth
        if len(self._conversation_history[user_id]) > 50:
            self._conversation_history[user_id] = self._conversation_history[user_id][-50:]

    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.

        Args:
            user_id: User ID

        Returns:
            List of conversation messages
        """
        return self._conversation_history.get(user_id, [])

    def clear_conversation_history(self, user_id: str) -> None:
        """
        Clear conversation history for a user.

        Args:
            user_id: User ID
        """
        if user_id in self._conversation_history:
            del self._conversation_history[user_id]
            logger.info(f"Cleared conversation history for user: {user_id}")

    # =========================================================================
    # CodeRAG Index Management Methods
    # =========================================================================

    def get_coderag_status(self, project_id: str) -> Dict[str, Any]:
        """
        Get CodeRAG index status for a project.

        Invokes `vlt coderag status --project <project_id> --json` to retrieve
        the current status of the code index.

        Args:
            project_id: Project identifier

        Returns:
            Dict containing:
            - project_id: Project identifier
            - status: Index status (not_initialized, indexing, ready, failed, stale)
            - file_count: Number of indexed files
            - chunk_count: Number of code chunks
            - last_indexed_at: Last successful index timestamp
            - error_message: Error details if status is failed
            - active_job: Current indexing job details if any
        """
        args = ["coderag", "status", "--project", project_id]
        return self._run_vlt_command(args, timeout=30)

    def init_coderag(
        self,
        project_id: str,
        target_path: str,
        force: bool = False,
        background: bool = True,
    ) -> Dict[str, Any]:
        """
        Initialize or re-index CodeRAG for a project.

        Invokes `vlt coderag init --project <project_id> --path <path>` to
        trigger indexing. By default, indexing runs in the background.

        Args:
            project_id: Project to associate with index
            target_path: Directory path to index
            force: Force re-index even if index exists
            background: Run indexing in background (via daemon)

        Returns:
            Dict containing:
            - job_id: Identifier for tracking the job
            - status: Whether job is queued or started
            - message: Human-readable status message
        """
        args = ["coderag", "init", "--project", project_id, "--path", target_path]

        if force:
            args.append("--force")

        if background:
            args.append("--background")

        return self._run_vlt_command(args, timeout=60)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get detailed status of an indexing job.

        Invokes `vlt coderag job <job_id> --json` to retrieve job status.

        Args:
            job_id: Job identifier (UUID)

        Returns:
            Dict containing:
            - job_id: Job identifier
            - project_id: Associated project ID
            - status: Current job status (pending, running, completed, failed, cancelled)
            - progress_percent: Completion percentage (0-100)
            - files_total: Total files to process
            - files_processed: Files completed
            - chunks_created: Code chunks generated
            - started_at: Processing start time
            - completed_at: Processing end time (if finished)
            - error_message: Error details if failed
            - duration_seconds: Elapsed time in seconds
        """
        args = ["coderag", "job", job_id]
        return self._run_vlt_command(args, timeout=30)

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel an indexing job.

        Invokes `vlt coderag cancel <job_id> --json` to cancel a pending
        or running job.

        Args:
            job_id: Job identifier (UUID)

        Returns:
            Dict containing:
            - status: "cancelled" if successful
            - message: Confirmation message
        """
        args = ["coderag", "cancel", job_id]
        return self._run_vlt_command(args, timeout=30)
