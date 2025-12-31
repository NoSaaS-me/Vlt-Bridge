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


class OracleBridge:
    """
    Bridge service that integrates Document-MCP with vlt-cli oracle and coderag.

    Uses subprocess calls to vlt CLI for oracle and code intelligence operations.
    This approach ensures we use the production vlt-cli implementation without
    coupling to internal vlt modules.
    """

    def __init__(self, vlt_command: str = "vlt"):
        """
        Initialize the oracle bridge.

        Args:
            vlt_command: Path to vlt CLI executable (default: "vlt" from PATH)
        """
        self.vlt_command = vlt_command
        # Store conversation history per user session
        # Key: user_id, Value: list of conversation messages
        self._conversation_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def _run_vlt_command(
        self,
        args: List[str],
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """
        Run a vlt CLI command and parse JSON output.

        Args:
            args: Command arguments (after 'vlt')
            timeout: Command timeout in seconds

        Returns:
            Parsed JSON response from vlt

        Raises:
            OracleBridgeError: If command fails or returns invalid JSON
        """
        cmd = [self.vlt_command] + args + ["--json"]

        try:
            logger.info(f"Running vlt command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )

            # Parse JSON output
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse vlt JSON output: {result.stdout}")
                raise OracleBridgeError(
                    "Invalid JSON response from vlt",
                    {"stdout": result.stdout, "stderr": result.stderr}
                ) from e

        except subprocess.TimeoutExpired as e:
            logger.error(f"vlt command timeout: {' '.join(cmd)}")
            raise OracleBridgeError(
                f"vlt command timeout after {timeout}s",
                {"command": cmd}
            ) from e

        except subprocess.CalledProcessError as e:
            logger.error(f"vlt command failed: {e.stderr}")
            raise OracleBridgeError(
                f"vlt command failed: {e.stderr}",
                {"command": cmd, "returncode": e.returncode, "stderr": e.stderr}
            ) from e

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
            # Apply thinking suffix if requested
            actual_model = f"{model}:thinking" if thinking else model
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

            # Stream sources
            sources_list = result.get("sources", [])
            for source in sources_list:
                yield {"type": "source", "source": source}
                await asyncio.sleep(0.01)

            # Store answer in conversation history
            self._add_to_history(
                user_id,
                "assistant",
                answer,
                sources=sources_list
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
    ) -> Dict[str, Any]:
        """
        Search code using hybrid retrieval (vector + BM25).

        Args:
            query: Search query
            limit: Maximum results to return
            language: Filter by programming language
            file_pattern: File pattern filter (not directly supported - would need implementation)
            project: Project ID (auto-detected if None)

        Returns:
            Search results with code chunks
        """
        args = ["coderag", "search", query, "--limit", str(limit)]

        if project:
            args.extend(["--project", project])

        if language:
            args.extend(["--language", language])

        # Note: file_pattern filtering would need to be implemented in vlt-cli
        # or filtered post-retrieval
        if file_pattern:
            logger.warning(f"file_pattern filtering not yet supported: {file_pattern}")

        return self._run_vlt_command(args, timeout=60)

    async def find_definition(
        self,
        symbol: str,
        scope: Optional[str] = None,
        kind: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find where a symbol is defined.

        Note: This requires vlt-cli to expose a `coderag definition` command,
        which may need to be implemented. For now, we use oracle with a
        structured query.

        Args:
            symbol: Symbol name to find
            scope: Optional file path to narrow search
            kind: Symbol kind filter (function, class, method, variable, constant)
            project: Project ID (auto-detected if None)

        Returns:
            Definition locations
        """
        # Build a structured query for the oracle
        query_parts = [f"Where is {symbol} defined?"]

        if kind:
            query_parts.append(f"Looking for a {kind}.")

        if scope:
            query_parts.append(f"In scope: {scope}")

        question = " ".join(query_parts)

        args = ["oracle", question, "--source", "code"]

        if project:
            args.extend(["--project", project])

        # This is a workaround - ideally vlt-cli would have `coderag definition`
        return self._run_vlt_command(args, timeout=60)

    async def find_references(
        self,
        symbol: str,
        limit: int = 20,
        include_definition: bool = False,
        reference_type: str = "all",
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find all references to a symbol.

        Note: This requires vlt-cli to expose a `coderag references` command.
        For now, we use oracle with a structured query.

        Args:
            symbol: Symbol name to find references for
            limit: Maximum references to return
            include_definition: Include the definition in results
            reference_type: Type of references (calls, imports, inherits, all)
            project: Project ID (auto-detected if None)

        Returns:
            Reference locations
        """
        # Build a structured query
        ref_desc = {
            "calls": "What calls",
            "imports": "What imports",
            "inherits": "What inherits from",
            "all": "Where is",
        }.get(reference_type, "Where is")

        question = f"{ref_desc} {symbol} used?"

        args = ["oracle", question, "--source", "code"]

        if project:
            args.extend(["--project", project])

        # This is a workaround - ideally vlt-cli would have `coderag references`
        return self._run_vlt_command(args, timeout=60)

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
