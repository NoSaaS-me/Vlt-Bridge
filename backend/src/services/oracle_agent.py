"""Oracle Agent - Main AI agent with tool calling (009-oracle-agent).

This replaces the subprocess-based OracleBridge with a proper agent implementation
that uses OpenRouter function calling for tool execution.

Context persistence is handled by OracleContextService, which:
- Loads existing context based on user_id + project_id
- Saves exchanges after each response
- Handles compression when approaching token budget
- Builds message history from stored exchanges
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from ..models.oracle import OracleStreamChunk, SourceReference
from ..models.oracle_context import (
    ContextStatus,
    ExchangeRole,
    OracleContext,
    OracleExchange,
    ToolCall,
    ToolCallStatus,
)
from .oracle_context_service import OracleContextService, get_context_service
from .context_tree_service import ContextTreeService, get_context_tree_service

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Result of a single tool execution for parallel handling."""
    call_id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    error: Optional[str] = None
    success: bool = True


# Lazy imports to avoid circular dependencies
_tool_executor = None
_prompt_loader = None


def _get_tool_executor():
    """Get ToolExecutor instance lazily."""
    global _tool_executor
    if _tool_executor is None:
        from .tool_executor import ToolExecutor
        _tool_executor = ToolExecutor()
    return _tool_executor


def _get_prompt_loader():
    """Get PromptLoader instance lazily."""
    global _prompt_loader
    if _prompt_loader is None:
        from .prompt_loader import PromptLoader
        _prompt_loader = PromptLoader()
    return _prompt_loader


def _parse_xml_tool_calls(content: str) -> Tuple[List[Dict[str, Any]], str]:
    """Parse XML-style function calls from content.

    Some models (like DeepSeek) don't properly support OpenAI function calling
    and instead output XML-style tool invocations in their text response.

    This function extracts those pseudo-tool-calls and returns them in the
    standard OpenAI tool_calls format so the agent loop can process them.

    Supported formats:
        <function_calls>
        <invoke name="tool_name">
        <parameter name="param_name">value</parameter>
        </invoke>
        </function_calls>

        Also handles: <function_calls>, <invoke>, <parameter>

    Args:
        content: The text content that may contain XML function calls

    Returns:
        Tuple of (tool_calls_list, cleaned_content) where:
        - tool_calls_list: List of tool calls in OpenAI format
        - cleaned_content: Content with XML blocks removed
    """
    tool_calls: List[Dict[str, Any]] = []

    # Match various XML function call formats
    # Pattern for <function_calls> or <function_calls> blocks
    function_calls_pattern = re.compile(
        r'<(?:antml:)?function_calls>\s*(.*?)\s*</(?:antml:)?function_calls>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern for individual <invoke> or <invoke> elements
    invoke_pattern = re.compile(
        r'<(?:antml:)?invoke\s+name=["\']([^"\']+)["\']\s*>\s*(.*?)\s*</(?:antml:)?invoke>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern for <parameter> or <parameter> elements
    param_pattern = re.compile(
        r'<(?:antml:)?parameter\s+name=["\']([^"\']+)["\'](?:\s+[^>]*)?>([^<]*)</(?:antml:)?parameter>',
        re.DOTALL | re.IGNORECASE
    )

    cleaned_content = content

    # Find all function_calls blocks
    for fc_match in function_calls_pattern.finditer(content):
        block_content = fc_match.group(1)
        cleaned_content = cleaned_content.replace(fc_match.group(0), '')

        # Find all invoke elements within this block
        for invoke_match in invoke_pattern.finditer(block_content):
            tool_name = invoke_match.group(1)
            params_content = invoke_match.group(2)

            # Extract parameters
            arguments: Dict[str, Any] = {}
            for param_match in param_pattern.finditer(params_content):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()

                # Try to parse as JSON, otherwise keep as string
                try:
                    # Handle boolean and numeric values
                    if param_value.lower() in ('true', 'false'):
                        arguments[param_name] = param_value.lower() == 'true'
                    elif param_value.isdigit():
                        arguments[param_name] = int(param_value)
                    else:
                        try:
                            arguments[param_name] = json.loads(param_value)
                        except json.JSONDecodeError:
                            arguments[param_name] = param_value
                except Exception:
                    arguments[param_name] = param_value

            # Create tool call in OpenAI format
            tool_call = {
                "id": f"xml_call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            }
            tool_calls.append(tool_call)

    # Also check for standalone invoke elements (not wrapped in function_calls)
    if not tool_calls:
        for invoke_match in invoke_pattern.finditer(content):
            tool_name = invoke_match.group(1)
            params_content = invoke_match.group(2)
            cleaned_content = cleaned_content.replace(invoke_match.group(0), '')

            arguments: Dict[str, Any] = {}
            for param_match in param_pattern.finditer(params_content):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()

                try:
                    if param_value.lower() in ('true', 'false'):
                        arguments[param_name] = param_value.lower() == 'true'
                    elif param_value.isdigit():
                        arguments[param_name] = int(param_value)
                    else:
                        try:
                            arguments[param_name] = json.loads(param_value)
                        except json.JSONDecodeError:
                            arguments[param_name] = param_value
                except Exception:
                    arguments[param_name] = param_value

            tool_call = {
                "id": f"xml_call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            }
            tool_calls.append(tool_call)

    # Clean up extra whitespace in the cleaned content
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content.strip())

    if tool_calls:
        logger.info(
            f"[XML PARSE] Parsed {len(tool_calls)} XML-style tool call(s) from content",
            extra={"tool_names": [tc["function"]["name"] for tc in tool_calls]},
        )
        for i, tc in enumerate(tool_calls):
            logger.info(f"[XML PARSE #{i+1}] name={tc['function']['name']} args={tc['function']['arguments'][:500]}")
    else:
        # Log when we DON'T find XML tool calls - to help debug
        if "<function" in content.lower() or "<invoke" in content.lower() or "tool_call" in content.lower():
            logger.warning(f"[XML PARSE] Content looks like it might have tool calls but none were parsed!")
            logger.warning(f"[XML PARSE] Content sample: {content[:1000]}")

    return tool_calls, cleaned_content


class OracleAgentError(Exception):
    """Raised when Oracle agent operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class OracleAgent:
    """AI project manager agent with tool calling.

    The Oracle answers questions about codebases by using tools to search code,
    read documentation, query development threads, and search the web.

    Supports cancellation via the cancel() method, which sets a cancellation flag
    and cancels any active asyncio tasks. Check _cancelled at key points during
    long-running operations.

    Auto-delegation: When search results are large or have many near-equal scores,
    the Oracle can automatically delegate to the Librarian subagent for summarization.
    """

    OPENROUTER_BASE = "https://openrouter.ai/api/v1"
    MAX_TURNS = 30  # Increased from 15 to allow complex multi-step queries
    DEFAULT_MODEL = "anthropic/claude-sonnet-4"
    DEFAULT_SUBAGENT_MODEL = "deepseek/deepseek-chat"

    # Thresholds for auto-delegation to Librarian
    DELEGATION_THRESHOLDS = {
        "vault_search_results": 6,      # >6 results with similar scores
        "search_code_results": 6,       # >6 code search results with similar scores
        "vault_list_files": 10,         # >10 files in listing
        "thread_read_entries": 20,      # >20 entries in thread
        "token_estimate": 4000,         # >4000 tokens in result
        "score_similarity": 0.1,        # Scores within 0.1 considered "similar"
    }

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        subagent_model: Optional[str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context_service: Optional[OracleContextService] = None,
        tree_service: Optional[ContextTreeService] = None,
    ):
        """Initialize the Oracle agent.

        Args:
            api_key: OpenRouter API key
            model: Model to use (default: anthropic/claude-sonnet-4)
            subagent_model: Model for Librarian subagent (from user settings)
            project_id: Project context for tool scoping
            user_id: User ID for context tracking
            context_service: OracleContextService for persistence (uses singleton if None)
            tree_service: ContextTreeService for tree-based persistence (uses singleton if None)
        """
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.subagent_model = subagent_model or self.DEFAULT_SUBAGENT_MODEL
        self.project_id = project_id or "default"
        self.user_id = user_id
        self._context: Optional[OracleContext] = None
        self._collected_sources: List[SourceReference] = []
        self._collected_tool_calls: List[ToolCall] = []
        self._context_service = context_service or get_context_service()
        self._tree_service = tree_service or get_context_tree_service()

        # Tree-based context tracking
        self._current_tree_root_id: Optional[str] = None
        self._current_node_id: Optional[str] = None

        # Cancellation support
        self._cancelled = False
        self._active_tasks: List[asyncio.Task] = []

    def cancel(self) -> None:
        """Cancel all running operations.

        Sets the cancellation flag and cancels any active asyncio tasks.
        The agent loop will stop at the next checkpoint.
        """
        logger.info(f"Cancelling Oracle agent for user {self.user_id}")
        self._cancelled = True
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
        self._active_tasks.clear()

    def is_cancelled(self) -> bool:
        """Check if the agent has been cancelled."""
        return self._cancelled

    def reset_cancellation(self) -> None:
        """Reset cancellation state for reuse."""
        self._cancelled = False
        self._active_tasks.clear()

    async def query(
        self,
        question: str,
        user_id: str,
        stream: bool = True,
        thinking: bool = False,
        max_tokens: int = 4000,
        project_id: Optional[str] = None,
        context_id: Optional[str] = None,
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """Run agent loop, yielding streaming chunks.

        Args:
            question: User's question
            user_id: User identifier
            stream: Whether to stream response
            thinking: Enable thinking/reasoning mode
            max_tokens: Maximum tokens in response
            project_id: Project ID for context scoping (overrides init value)
            context_id: Node ID from context tree (for conversation continuity)

        Yields:
            OracleStreamChunk objects for each piece of the response
        """
        # Reset cancellation state for new query
        self.reset_cancellation()
        self.user_id = user_id
        self._collected_sources = []
        self._collected_tool_calls = []

        # Use provided project_id or fall back to init value
        effective_project_id = project_id or self.project_id or "default"

        # Check cancellation at start
        if self._cancelled:
            yield OracleStreamChunk(type="error", error="Cancelled by user")
            return

        # Load context from tree service (new tree-based system)
        try:
            if context_id:
                # Load from existing node's tree
                node = self._tree_service.get_node(user_id, context_id)
                if node:
                    self._current_tree_root_id = node.root_id
                    self._current_node_id = context_id
                    logger.debug(
                        f"Loaded tree context from node {context_id} "
                        f"(tree: {node.root_id})"
                    )
                else:
                    logger.warning(f"Context node {context_id} not found, creating new tree")
                    context_id = None

            if not context_id:
                # Get or create active tree for this user/project
                active_tree_id = self._tree_service.get_active_tree_id(user_id, effective_project_id)
                if active_tree_id:
                    tree = self._tree_service.get_tree(user_id, active_tree_id)
                    if tree:
                        self._current_tree_root_id = tree.root_id
                        self._current_node_id = tree.current_node_id
                        logger.debug(
                            f"Using active tree {tree.root_id} "
                            f"(HEAD: {tree.current_node_id})"
                        )
                else:
                    # Create new tree if none exists
                    tree = self._tree_service.create_tree(
                        user_id=user_id,
                        project_id=effective_project_id,
                        max_nodes=30,
                    )
                    self._current_tree_root_id = tree.root_id
                    self._current_node_id = tree.current_node_id
                    logger.info(f"Created new tree {tree.root_id} for {user_id}/{effective_project_id}")

        except Exception as e:
            logger.error(f"Failed to load tree context: {e}")
            # Continue without context persistence
            self._current_tree_root_id = None
            self._current_node_id = None

        # Also load legacy context for backwards compatibility
        try:
            self._context = self._context_service.get_or_create_context(
                user_id=user_id,
                project_id=effective_project_id,
                token_budget=max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to load legacy context: {e}")
            self._context = None

        # Get services
        tool_executor = _get_tool_executor()
        prompt_loader = _get_prompt_loader()

        # Fetch vault file list to include in system prompt
        vault_files: List[str] = []
        try:
            notes = tool_executor.vault.list_notes(user_id)
            # Extract paths and limit to 100 files to avoid overwhelming the prompt
            vault_files = [note.get("path", "") for note in notes[:100]]
            logger.debug(f"Loaded {len(vault_files)} vault files for system prompt")
        except Exception as e:
            logger.warning(f"Failed to load vault files for system prompt: {e}")

        # Build initial messages
        system_prompt = prompt_loader.load(
            "oracle/system.md",
            {
                "project_id": effective_project_id,
                "user_id": user_id,
                "vault_files": vault_files,
            },
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Add context history from tree nodes (primary source)
        if self._current_tree_root_id and self._current_node_id:
            try:
                # Get all nodes in tree
                nodes = self._tree_service.get_nodes(user_id, self._current_tree_root_id)
                node_map = {n.id: n for n in nodes}

                # Build path from root to current node
                path_nodes = []
                current_id = self._current_node_id
                while current_id and current_id in node_map:
                    path_nodes.insert(0, node_map[current_id])
                    current_id = node_map[current_id].parent_id

                # Add conversation history from path (skip empty root nodes)
                for node in path_nodes:
                    if node.is_root and not node.question and not node.answer:
                        continue
                    if node.question:
                        messages.append({"role": "user", "content": node.question})
                    if node.answer:
                        messages.append({"role": "assistant", "content": node.answer})

                logger.debug(
                    f"Loaded {len(path_nodes)} nodes from tree, "
                    f"added {len(messages) - 1} context messages"
                )
            except Exception as e:
                logger.error(f"Failed to load tree history: {e}")

        # Fallback: Add legacy context history if tree is empty
        elif self._context and self._context.recent_exchanges:
            # Add compressed summary as system context if available
            if self._context.compressed_summary:
                messages.append({
                    "role": "system",
                    "content": f"<conversation_summary>\n{self._context.compressed_summary}\n</conversation_summary>",
                })

            # Add recent exchanges to message history
            for exchange in self._context.recent_exchanges:
                if exchange.role == ExchangeRole.USER:
                    messages.append({"role": "user", "content": exchange.content})
                elif exchange.role == ExchangeRole.ASSISTANT:
                    messages.append({"role": "assistant", "content": exchange.content})

        # Add current question
        messages.append({"role": "user", "content": question})

        # Get tool definitions
        tools = tool_executor.get_tool_schemas(agent="oracle")

        # Yield thinking chunk to indicate we're starting
        yield OracleStreamChunk(
            type="thinking",
            content="Analyzing question and gathering context...",
        )

        # Track the original question for context saving
        self._current_question = question

        # Track accumulated content across turns for max-turns fallback
        accumulated_content = ""
        accumulated_thinking = ""

        # Agent loop
        for turn in range(self.MAX_TURNS):
            # Check cancellation before each turn
            if self._cancelled:
                logger.info(f"Agent cancelled at turn {turn + 1}")
                yield OracleStreamChunk(type="error", error="Cancelled by user")
                return

            logger.debug(f"Agent turn {turn + 1}/{self.MAX_TURNS}")

            async for chunk in self._agent_turn(
                messages=messages,
                tools=tools,
                stream=stream,
                thinking=thinking,
                max_tokens=max_tokens,
                user_id=user_id,
            ):
                # Check cancellation during streaming
                if self._cancelled:
                    logger.info("Agent cancelled during streaming")
                    yield OracleStreamChunk(type="error", error="Cancelled by user")
                    return

                # Track content and thinking for max-turns fallback
                if chunk.type == "content" and chunk.content:
                    accumulated_content += chunk.content
                elif chunk.type == "thinking" and chunk.content:
                    accumulated_thinking += chunk.content + "\n"

                yield chunk

                # Check if we're done
                if chunk.type == "done":
                    return

                # If we got an error, stop
                if chunk.type == "error":
                    return

        # Max turns reached - yield accumulated content before error
        logger.warning(f"Max turns ({self.MAX_TURNS}) reached with accumulated content: {len(accumulated_content)} chars")

        if accumulated_content:
            # Yield what we have so far before the warning
            yield OracleStreamChunk(
                type="content",
                content=f"\n\n---\n*Note: Response incomplete due to turn limit ({self.MAX_TURNS} turns). Here is what was gathered:*\n\n",
            )
        elif accumulated_thinking:
            # If no content but we have thinking, summarize what was found
            yield OracleStreamChunk(
                type="content",
                content=f"*Response incomplete after {self.MAX_TURNS} turns. Last reasoning:\n{accumulated_thinking[-500:]}*",
            )

        # Save what we have as a partial exchange
        if accumulated_content or accumulated_thinking:
            context_id = self._save_exchange(
                question=question,
                answer=accumulated_content or f"*Partial response - thinking: {accumulated_thinking[:500]}*",
            )
        else:
            context_id = None

        yield OracleStreamChunk(
            type="done",
            tokens_used=None,
            model_used=self.model,
            context_id=context_id,
        )

    async def _agent_turn(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        stream: bool,
        thinking: bool,
        max_tokens: int,
        user_id: str,
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """Execute one turn of the agent loop.

        Args:
            messages: Conversation messages so far
            tools: Available tool definitions
            stream: Whether to stream response
            thinking: Enable thinking mode
            max_tokens: Max tokens for response
            user_id: User identifier

        Yields:
            Response chunks from this turn
        """
        # Apply thinking suffix if requested
        model = self.model
        if thinking and not model.endswith(":thinking"):
            model = f"{model}:thinking"

        try:
            request_body = {
                "model": model,
                "messages": messages,
                "tools": tools if tools else None,
                "tool_choice": "auto" if tools else None,
                "parallel_tool_calls": True,
                "stream": stream,
                "max_tokens": max_tokens,
            }

            # Log the request
            logger.info(f"=== OPENROUTER REQUEST ===")
            logger.info(f"[REQUEST] model={model} stream={stream} max_tokens={max_tokens}")
            logger.info(f"[REQUEST] tools_count={len(tools) if tools else 0}")
            logger.info(f"[REQUEST] messages_count={len(messages)}")
            # Log last user message
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    logger.info(f"[REQUEST] last_user_msg={msg.get('content', '')[:200]}")
                    break

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.OPENROUTER_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://vlt.ai",
                        "X-Title": "Vlt Oracle",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                )
                response.raise_for_status()

                if stream:
                    async for chunk in self._process_stream(response, messages, user_id):
                        yield chunk
                else:
                    data = response.json()
                    async for chunk in self._process_response(data, messages, user_id):
                        yield chunk

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error: {e.response.status_code} - {e.response.text}")
            yield OracleStreamChunk(
                type="error",
                error=f"API error: {e.response.status_code}",
            )
        except httpx.TimeoutException:
            logger.error("OpenRouter API timeout")
            yield OracleStreamChunk(
                type="error",
                error="Request timeout - please try again",
            )
        except Exception as e:
            logger.exception(f"Agent turn failed: {e}")
            yield OracleStreamChunk(
                type="error",
                error=f"Agent error: {str(e)}",
            )

    async def _process_stream(
        self,
        response: httpx.Response,
        messages: List[Dict[str, Any]],
        user_id: str,
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """Process streaming response from OpenRouter.

        Args:
            response: HTTP response with SSE stream
            messages: Conversation messages (mutated with assistant response)
            user_id: User identifier

        Yields:
            Parsed stream chunks
        """
        content_buffer = ""
        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}
        finish_reason = None
        chunk_count = 0

        logger.info("=== STARTING SSE STREAM PROCESSING ===")

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue

            data_str = line[6:]  # Remove "data: " prefix
            if data_str == "[DONE]":
                logger.info("=== SSE STREAM [DONE] ===")
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                logger.warning(f"[RAW SSE] Failed to parse: {data_str[:500]}")
                continue

            chunk_count += 1
            choices = data.get("choices", [])
            if not choices:
                logger.debug(f"[RAW SSE #{chunk_count}] No choices: {json.dumps(data)[:500]}")
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            # CRITICAL: Only update finish_reason if it has a truthy value
            # Otherwise later chunks with finish_reason=None will overwrite it!
            if choice.get("finish_reason"):
                finish_reason = choice.get("finish_reason")

            # Log EVERY chunk with full details
            logger.info(f"[RAW SSE #{chunk_count}] finish_reason={finish_reason} delta_keys={list(delta.keys())} delta={json.dumps(delta)[:1000]}")

            # Handle reasoning/thinking traces (from models like DeepSeek)
            # Track what we've yielded to avoid duplication
            yielded_reasoning_text = None
            if "reasoning" in delta and delta["reasoning"]:
                yielded_reasoning_text = delta["reasoning"]
                yield OracleStreamChunk(
                    type="thinking",
                    content=yielded_reasoning_text,
                )

            # Handle reasoning_details (alternative format)
            # Only yield if different from what we already yielded from "reasoning"
            if "reasoning_details" in delta and delta["reasoning_details"]:
                for detail in delta["reasoning_details"]:
                    if isinstance(detail, dict) and detail.get("text"):
                        detail_text = detail["text"]
                        # Skip if this is the same as what we already yielded
                        if detail_text != yielded_reasoning_text:
                            yield OracleStreamChunk(
                                type="thinking",
                                content=detail_text,
                            )

            # Handle content
            if "content" in delta and delta["content"]:
                content_buffer += delta["content"]
                yield OracleStreamChunk(
                    type="content",
                    content=delta["content"],
                )

            # Handle tool calls
            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    if "id" in tc and tc["id"]:
                        tool_calls_buffer[idx]["id"] = tc["id"]

                    if "function" in tc:
                        if "name" in tc["function"]:
                            tool_calls_buffer[idx]["function"]["name"] = tc["function"]["name"]
                        if "arguments" in tc["function"]:
                            tool_calls_buffer[idx]["function"]["arguments"] += tc["function"]["arguments"]

        # Log final state
        logger.info(f"=== SSE STREAM ENDED after {chunk_count} chunks ===")
        logger.info(f"[FINAL STATE] finish_reason={finish_reason}")
        logger.info(f"[FINAL STATE] content_buffer_len={len(content_buffer)}")
        logger.info(f"[FINAL STATE] tool_calls_buffer={json.dumps(tool_calls_buffer)[:2000]}")
        if content_buffer:
            logger.info(f"[FINAL STATE] content_preview={content_buffer[:500]}")

        # Process finish
        if finish_reason == "tool_calls" and tool_calls_buffer:
            # Convert buffer to list
            tool_calls = [tool_calls_buffer[i] for i in sorted(tool_calls_buffer.keys())]

            # Add assistant message with tool calls
            assistant_msg = {"role": "assistant", "content": content_buffer or None}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # Execute tools and yield results
            async for chunk in self._execute_tools(tool_calls, messages, user_id):
                yield chunk

        elif finish_reason == "stop" or (content_buffer and not tool_calls_buffer):
            # Check if the model output XML-style tool calls in content
            # (Some models like DeepSeek don't support proper function calling)
            xml_tool_calls, cleaned_content = _parse_xml_tool_calls(content_buffer)

            if xml_tool_calls:
                # Model used XML-style tool calls instead of proper function calling
                logger.warning(
                    f"Model {self.model} output {len(xml_tool_calls)} XML-style tool call(s) "
                    "instead of using proper function calling. Parsing and executing."
                )

                # Add assistant message with the parsed tool calls
                assistant_msg = {"role": "assistant", "content": cleaned_content or None}
                assistant_msg["tool_calls"] = xml_tool_calls
                messages.append(assistant_msg)

                # Execute tools and yield results
                async for chunk in self._execute_tools(xml_tool_calls, messages, user_id):
                    yield chunk
            else:
                # Final response without tool calls
                messages.append({"role": "assistant", "content": content_buffer})

                # Save the exchange to persistent context
                context_id = self._save_exchange(
                    question=getattr(self, '_current_question', ''),
                    answer=content_buffer,
                )

                # Yield sources
                for source in self._collected_sources:
                    yield OracleStreamChunk(
                        type="source",
                        source=source,
                    )

                # Done with context_id for frontend reference
                yield OracleStreamChunk(
                    type="done",
                    tokens_used=None,  # Could extract from response headers
                    model_used=self.model,
                    context_id=context_id,
                )

    async def _process_response(
        self,
        data: Dict[str, Any],
        messages: List[Dict[str, Any]],
        user_id: str,
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """Process non-streaming response from OpenRouter.

        Args:
            data: Parsed JSON response
            messages: Conversation messages (mutated with assistant response)
            user_id: User identifier

        Yields:
            Response chunks
        """
        choices = data.get("choices", [])
        if not choices:
            yield OracleStreamChunk(
                type="error",
                error="No response from model",
            )
            return

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")

        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        if finish_reason == "tool_calls" and tool_calls:
            # Add assistant message
            messages.append(message)

            # Execute tools
            async for chunk in self._execute_tools(tool_calls, messages, user_id):
                yield chunk

        else:
            # Check for XML-style tool calls in content
            # (Some models like DeepSeek don't support proper function calling)
            xml_tool_calls, cleaned_content = _parse_xml_tool_calls(content)

            if xml_tool_calls:
                # Model used XML-style tool calls instead of proper function calling
                logger.warning(
                    f"Model {self.model} output {len(xml_tool_calls)} XML-style tool call(s) "
                    "instead of using proper function calling. Parsing and executing."
                )

                # Yield the cleaned content if any
                if cleaned_content:
                    yield OracleStreamChunk(
                        type="content",
                        content=cleaned_content,
                    )

                # Add assistant message with the parsed tool calls
                assistant_msg = {"role": "assistant", "content": cleaned_content or None}
                assistant_msg["tool_calls"] = xml_tool_calls
                messages.append(assistant_msg)

                # Execute tools
                async for chunk in self._execute_tools(xml_tool_calls, messages, user_id):
                    yield chunk
            else:
                # Final response without tool calls
                if content:
                    yield OracleStreamChunk(
                        type="content",
                        content=content,
                    )

                messages.append({"role": "assistant", "content": content})

                # Save the exchange to persistent context
                usage = data.get("usage", {})
                context_id = self._save_exchange(
                    question=getattr(self, '_current_question', ''),
                    answer=content,
                    tokens_used=usage.get("total_tokens"),
                )

                # Yield sources
                for source in self._collected_sources:
                    yield OracleStreamChunk(
                        type="source",
                        source=source,
                    )

                # Done - only when no XML tool calls were found
                yield OracleStreamChunk(
                    type="done",
                    tokens_used=usage.get("total_tokens"),
                    model_used=self.model,
                    context_id=context_id,
                )

    async def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        user_id: str,
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """Execute tool calls in parallel and add results to messages.

        Implements T026 (parallel execution) and T027 (error handling):
        - Runs multiple tool calls concurrently using asyncio.gather
        - Continues with other tools if one fails (return_exceptions=True)
        - Returns error result for failed tools but allows agent loop to continue
        - Preserves order of results for consistent message flow

        Args:
            tool_calls: List of tool calls from the model
            messages: Conversation messages (mutated with tool results)
            user_id: User identifier

        Yields:
            Tool call and result chunks
        """
        if not tool_calls:
            return

        tool_executor = _get_tool_executor()

        # Parse all tool calls first
        parsed_calls: List[Tuple[str, str, Dict[str, Any], str]] = []
        for call in tool_calls:
            call_id = call.get("id", str(uuid.uuid4()))
            function = call.get("function", {})
            name = function.get("name", "unknown")
            arguments_str = function.get("arguments", "{}")

            # Parse arguments
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {}

            parsed_calls.append((call_id, name, arguments, arguments_str))

        # Yield thinking update about tool execution
        tool_names = [name for _, name, _, _ in parsed_calls]
        if len(tool_names) == 1:
            thinking_msg = f"Executing tool: {tool_names[0]}"
        else:
            thinking_msg = f"Executing {len(tool_names)} tools: {', '.join(tool_names)}"
        yield OracleStreamChunk(
            type="thinking",
            content=thinking_msg,
        )

        # Yield all tool call notifications first (pending status)
        for call_id, name, arguments, arguments_str in parsed_calls:
            yield OracleStreamChunk(
                type="tool_call",
                tool_call={
                    "id": call_id,
                    "name": name,
                    "arguments": arguments_str,  # Full arguments for frontend display
                    "status": "pending",
                },
            )

        # Execute all tools in parallel
        async def execute_single_tool(
            call_id: str,
            name: str,
            arguments: Dict[str, Any],
        ) -> ToolExecutionResult:
            """Execute a single tool and return structured result."""
            try:
                result = await tool_executor.execute(
                    name=name,
                    arguments=arguments,
                    user_id=user_id,
                )
                return ToolExecutionResult(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                    result=result,
                    success=True,
                )
            except Exception as e:
                logger.exception(f"Tool execution failed: {name}")
                return ToolExecutionResult(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                    error=str(e),
                    success=False,
                )

        # Create tasks for parallel execution
        tasks = [
            execute_single_tool(call_id, name, arguments)
            for call_id, name, arguments, _ in parsed_calls
        ]

        # Execute all tools concurrently, continue even if some fail
        results: List[ToolExecutionResult] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        # Process results in order (preserves message ordering)
        for i, result in enumerate(results):
            # Handle case where asyncio.gather returns an exception object
            if isinstance(result, Exception):
                call_id, name, arguments, _ = parsed_calls[i]
                logger.exception(f"Unexpected exception in tool {name}: {result}")
                result = ToolExecutionResult(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                    error=f"Unexpected error: {str(result)}",
                    success=False,
                )

            if result.success and result.result is not None:
                # Success case
                yield OracleStreamChunk(
                    type="tool_result",
                    tool_call_id=result.call_id,  # Associate result with tool call
                    tool_result=(
                        result.result[:2000]  # Allow more content for frontend display
                        if len(result.result) > 2000
                        else result.result
                    ),
                )

                # Yield thinking update about tool completion
                result_preview = result.result[:100] + "..." if len(result.result) > 100 else result.result
                yield OracleStreamChunk(
                    type="thinking",
                    content=f"✓ {result.name} returned: {result_preview}",
                )

                # Extract sources from successful result
                self._extract_sources_from_result(result.name, result.result)

                # Collect tool call for context persistence
                self._collected_tool_calls.append(
                    ToolCall(
                        id=result.call_id,
                        name=result.name,
                        arguments=result.arguments,
                        result=result.result[:2000] if len(result.result) > 2000 else result.result,
                        status=ToolCallStatus.SUCCESS,
                    )
                )

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.result,
                })
            else:
                # Error case - T027: provide error but let agent continue
                error_content = self._format_tool_error(
                    result.name,
                    result.error or "Unknown error",
                    result.arguments,
                )

                yield OracleStreamChunk(
                    type="tool_result",
                    tool_call_id=result.call_id,  # Associate error with tool call
                    tool_result=error_content,
                )

                # Yield thinking update about tool error
                yield OracleStreamChunk(
                    type="thinking",
                    content=f"✗ {result.name} failed: {result.error or 'Unknown error'}",
                )

                # Collect failed tool call for context persistence
                self._collected_tool_calls.append(
                    ToolCall(
                        id=result.call_id,
                        name=result.name,
                        arguments=result.arguments,
                        result=result.error,
                        status=ToolCallStatus.ERROR,
                    )
                )

                # Add error result to messages so agent can handle it
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": error_content,
                })

    def _format_tool_error(
        self,
        tool_name: str,
        error: str,
        arguments: Dict[str, Any],
    ) -> str:
        """Format a tool error message for the agent.

        Provides structured error information that helps the agent:
        - Understand what failed
        - Decide whether to retry with different parameters
        - Choose an alternative approach

        Args:
            tool_name: Name of the failed tool
            error: Error message
            arguments: Arguments that were passed to the tool

        Returns:
            JSON-formatted error message
        """
        return json.dumps({
            "error": True,
            "tool": tool_name,
            "message": error,
            "suggestion": self._get_error_suggestion(tool_name, error),
            "failed_arguments": arguments,
        })

    def _get_error_suggestion(self, tool_name: str, error: str) -> str:
        """Generate a suggestion for handling tool errors.

        Provides context-aware suggestions to help the agent recover from errors.

        Args:
            tool_name: Name of the failed tool
            error: Error message

        Returns:
            Suggestion string for the agent
        """
        error_lower = error.lower()

        # File not found errors
        if "not found" in error_lower or "does not exist" in error_lower:
            if "vault" in tool_name:
                return "The note may not exist. Try vault_list to see available notes, or vault_search to find related content."
            elif "thread" in tool_name:
                return "The thread may not exist. Try thread_list to see available threads."
            else:
                return "The requested resource was not found. Try searching for alternatives."

        # Permission/access errors
        if "permission" in error_lower or "access denied" in error_lower:
            return "Access was denied. The user may not have permission to access this resource."

        # Timeout errors
        if "timeout" in error_lower:
            return "The operation timed out. Try with more specific parameters or a smaller scope."

        # Invalid arguments
        if "invalid" in error_lower or "validation" in error_lower:
            return "The arguments were invalid. Check the parameter format and try again."

        # Network errors
        if "network" in error_lower or "connection" in error_lower:
            return "A network error occurred. This may be temporary - consider retrying."

        # Tool-specific suggestions
        suggestions = {
            "search_code": "Try a different search query or use vault_search for documentation.",
            "vault_read": "Use vault_list to verify the note path exists.",
            "vault_search": "Try broader search terms or check if the vault has content.",
            "thread_read": "Use thread_list to verify the thread exists.",
            "thread_seek": "Try different search terms or use thread_list first.",
            "web_search": "Try a different query or check if web search is available.",
            "web_fetch": "Verify the URL is accessible or try a different source.",
        }

        return suggestions.get(tool_name, "Consider trying an alternative approach or asking the user for clarification.")

    def _extract_sources_from_result(self, tool_name: str, result: str) -> None:
        """Extract source references from tool results.

        Args:
            tool_name: Name of the tool that was executed
            result: JSON result string from the tool
        """
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return

        # Handle different tool result formats
        if tool_name == "search_code":
            results = data.get("results", [])
            for r in results[:5]:  # Limit sources
                self._collected_sources.append(
                    SourceReference(
                        path=r.get("file_path", r.get("path", "")),
                        source_type="code",
                        line=r.get("line_start"),
                        snippet=r.get("content", "")[:500],
                        score=r.get("score"),
                    )
                )

        elif tool_name == "vault_search":
            results = data.get("results", [])
            for r in results[:5]:
                self._collected_sources.append(
                    SourceReference(
                        path=r.get("path", ""),
                        source_type="vault",
                        snippet=r.get("snippet", "")[:500],
                        score=r.get("score"),
                    )
                )

        elif tool_name == "vault_read":
            path = data.get("path", "")
            if path:
                self._collected_sources.append(
                    SourceReference(
                        path=path,
                        source_type="vault",
                        snippet=data.get("content", "")[:500],
                    )
                )

        elif tool_name in ("thread_read", "thread_seek"):
            results = data.get("results", data.get("entries", []))
            for r in results[:5]:
                self._collected_sources.append(
                    SourceReference(
                        path=f"thread:{r.get('thread_id', '')}",
                        source_type="thread",
                        snippet=r.get("content", "")[:500],
                        score=r.get("score"),
                    )
                )

    def _should_delegate_to_librarian(
        self,
        tool_name: str,
        tool_result: Dict[str, Any],
    ) -> bool:
        """
        Determine if results should be delegated to Librarian for summarization.

        Auto-delegation occurs when:
        - vault_search/search_code returns >6 results with similar scores (within 0.1)
        - vault_list returns >10 files
        - thread_read returns >20 entries
        - Any result exceeds ~4000 tokens

        Args:
            tool_name: Name of the tool that produced the result
            tool_result: Parsed JSON result from the tool

        Returns:
            True if the result should be summarized by Librarian
        """
        thresholds = self.DELEGATION_THRESHOLDS

        # Check for errors - don't delegate error responses
        if "error" in tool_result:
            return False

        # Estimate token count (rough: 4 chars per token)
        result_str = json.dumps(tool_result)
        estimated_tokens = len(result_str) // 4
        if estimated_tokens > thresholds["token_estimate"]:
            logger.debug(
                f"Auto-delegation: {tool_name} result exceeds {thresholds['token_estimate']} tokens "
                f"(estimated: {estimated_tokens})"
            )
            return True

        # Check tool-specific thresholds
        if tool_name in ("vault_search", "search_code"):
            results = tool_result.get("results", [])
            if len(results) > thresholds["vault_search_results"]:
                # Check if scores are "near-equal" (within threshold)
                scores = [r.get("score", 0) for r in results if r.get("score") is not None]
                if len(scores) >= 2:
                    # Check if the spread is small (many similar scores)
                    max_score = max(scores)
                    min_score = min(scores)
                    if (max_score - min_score) <= thresholds["score_similarity"]:
                        logger.debug(
                            f"Auto-delegation: {tool_name} has {len(results)} results "
                            f"with similar scores (spread: {max_score - min_score:.3f})"
                        )
                        return True
                    # Also delegate if most results are within threshold of each other
                    near_equal_count = sum(
                        1 for s in scores
                        if abs(s - scores[0]) <= thresholds["score_similarity"]
                    )
                    if near_equal_count > thresholds["vault_search_results"]:
                        logger.debug(
                            f"Auto-delegation: {tool_name} has {near_equal_count} near-equal results"
                        )
                        return True

        elif tool_name == "vault_list":
            notes = tool_result.get("notes", [])
            if len(notes) > thresholds["vault_list_files"]:
                logger.debug(
                    f"Auto-delegation: vault_list has {len(notes)} files "
                    f"(threshold: {thresholds['vault_list_files']})"
                )
                return True

        elif tool_name == "thread_read":
            entries = tool_result.get("entries", [])
            if len(entries) > thresholds["thread_read_entries"]:
                logger.debug(
                    f"Auto-delegation: thread_read has {len(entries)} entries "
                    f"(threshold: {thresholds['thread_read_entries']})"
                )
                return True

        return False

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string (rough: 4 chars per token)."""
        return len(text) // 4

    def _save_exchange(
        self,
        question: str,
        answer: str,
        tokens_used: Optional[int] = None,
    ) -> Optional[str]:
        """Save the question and answer exchange to persistent context.

        This is called after a successful response to persist the conversation
        for future context loading. Saves to BOTH tree-based and legacy systems.

        Args:
            question: User's question
            answer: Assistant's full response
            tokens_used: Total tokens consumed (if known)

        Returns:
            Node ID if saved successfully (for tree), or legacy context ID
        """
        result_id = None

        # SAVE TO TREE-BASED SYSTEM (primary)
        if self._current_tree_root_id and self._current_node_id and self.user_id:
            try:
                # Create new node as child of current HEAD
                new_node = self._tree_service.create_node(
                    user_id=self.user_id,
                    root_id=self._current_tree_root_id,
                    parent_id=self._current_node_id,
                    question=question,
                    answer=answer,
                    tool_calls=self._collected_tool_calls if self._collected_tool_calls else None,
                    tokens_used=tokens_used or self._estimate_tokens(question + answer),
                    model_used=self.model,
                )

                # Update current node to the new one (new HEAD)
                self._current_node_id = new_node.id
                result_id = new_node.id

                logger.info(
                    f"Saved exchange to tree node {new_node.id} "
                    f"(tree: {self._current_tree_root_id})"
                )

            except Exception as e:
                logger.error(f"Failed to save exchange to tree: {e}")

        # SAVE TO LEGACY SYSTEM (for backwards compatibility)
        if self._context:
            try:
                # Create user exchange
                user_exchange = OracleExchange(
                    id=str(uuid.uuid4()),
                    role=ExchangeRole.USER,
                    content=question,
                    timestamp=datetime.now(timezone.utc),
                    token_count=self._estimate_tokens(question),
                )

                # Add user exchange first
                self._context_service.add_exchange(
                    user_id=self._context.user_id,
                    project_id=self._context.project_id,
                    exchange=user_exchange,
                )

                # Extract mentioned files and symbols from sources
                mentioned_files = []
                mentioned_symbols = []
                for source in self._collected_sources:
                    if source.path:
                        mentioned_files.append(source.path)
                    # Note: symbols would need parsing from content

                # Create assistant exchange with tool calls
                assistant_exchange = OracleExchange(
                    id=str(uuid.uuid4()),
                    role=ExchangeRole.ASSISTANT,
                    content=answer,
                    tool_calls=self._collected_tool_calls if self._collected_tool_calls else None,
                    timestamp=datetime.now(timezone.utc),
                    token_count=tokens_used or self._estimate_tokens(answer),
                    mentioned_files=mentioned_files[:20],  # Limit to 20
                )

                # Add assistant exchange
                self._context = self._context_service.add_exchange(
                    user_id=self._context.user_id,
                    project_id=self._context.project_id,
                    exchange=assistant_exchange,
                    model_used=self.model,
                )

                logger.debug(
                    f"Saved exchange to legacy context {self._context.id} "
                    f"(total exchanges: {len(self._context.recent_exchanges)})"
                )

                # Use legacy ID as fallback if tree save failed
                if not result_id:
                    result_id = self._context.id

            except Exception as e:
                logger.error(f"Failed to save exchange to legacy context: {e}")
                if not result_id and self._context:
                    result_id = self._context.id

        return result_id


# Singleton instance
_oracle_agent: Optional[OracleAgent] = None


def get_oracle_agent(
    api_key: str,
    model: Optional[str] = None,
    subagent_model: Optional[str] = None,
    project_id: Optional[str] = None,
) -> OracleAgent:
    """Get or create an OracleAgent instance.

    Note: This creates a new instance each time since agents are stateful
    per request. Use for dependency injection in routes.

    Args:
        api_key: OpenRouter API key
        model: Model override
        subagent_model: Model for Librarian subagent (from user settings)
        project_id: Project context

    Returns:
        Configured OracleAgent
    """
    return OracleAgent(
        api_key=api_key,
        model=model,
        subagent_model=subagent_model,
        project_id=project_id,
    )


__all__ = ["OracleAgent", "OracleAgentError", "get_oracle_agent"]
