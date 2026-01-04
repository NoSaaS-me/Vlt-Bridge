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
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from ..models.agent_state import AgentState
from ..models.oracle import OracleStreamChunk, SourceReference
from ..models.settings import AgentConfig
from .decision_tree import DefaultDecisionTree
from .user_settings import UserSettingsService
from ..models.oracle_context import (
    ExchangeRole,
    OracleContext,
    OracleExchange,
    ToolCall,
    ToolCallStatus,
)
from .oracle_context_service import OracleContextService, get_context_service
from .context_tree_service import ContextTreeService, get_context_tree_service
from .tool_parsers import ToolCallParserChain, DSMLReasoningParser
from .model_capabilities import (
    get_model_capability,
    should_use_reasoning_param,
    should_use_thinking_suffix,
    requires_reasoning_passback,
    ReasoningApproach,
)

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

    This function uses the ToolCallParserChain to try multiple parsers in
    priority order, extracting tool calls and returning them in the standard
    OpenAI tool_calls format.

    Args:
        content: The text content that may contain XML function calls

    Returns:
        Tuple of (tool_calls_list, cleaned_content) where:
        - tool_calls_list: List of tool calls in OpenAI format
        - cleaned_content: Content with XML blocks removed

    Note:
        This function is now a thin wrapper around ToolCallParserChain.
        See tool_parsers/ package for parser implementations.
    """
    parser_chain = ToolCallParserChain()
    return parser_chain.parse(content)


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
        self._working_node_id: Optional[str] = None  # For incremental saves

        # Cancellation support
        self._cancelled = False
        self._active_tasks: List[asyncio.Task] = []

        # Agent config (set per-query in stream_response, defaults for safety)
        self._agent_config: Optional[AgentConfig] = None

        # DeepSeek reasoning_content tracking (must be passed back during tool calls)
        # This is reset at the start of each turn but persists across tool calls within a turn
        self._turn_reasoning_content: Optional[str] = None

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
        # Reset DeepSeek reasoning_content tracking for new query
        # This persists across tool calls within a query but resets for new questions
        self._turn_reasoning_content = None

        # Use provided project_id or fall back to init value
        effective_project_id = project_id or self.project_id or "default"
        # Update instance variable so tool injection uses correct project
        self.project_id = effective_project_id

        # T029: Load user's AgentConfig from user_settings
        settings_service = UserSettingsService()
        agent_config = settings_service.get_agent_config(user_id)
        # Store config on instance for _execute_tools to access
        self._agent_config = agent_config
        logger.info(
            f"[TURN_CONTROL] Loaded AgentConfig for {user_id}: "
            f"max_iterations={agent_config.max_iterations}, "
            f"token_budget={agent_config.token_budget}, "
            f"timeout_seconds={agent_config.timeout_seconds}, "
            f"max_tool_calls_per_turn={agent_config.max_tool_calls_per_turn}, "
            f"max_parallel_tools={agent_config.max_parallel_tools}"
        )

        # T030: Create initial AgentState for this query
        agent_state = AgentState(
            user_id=user_id,
            project_id=effective_project_id,
            config=agent_config,
            turn=0,
            tokens_used=0,
            start_time=time.time(),
        )

        # T031: Create decision tree with config
        decision_tree = DefaultDecisionTree(agent_config)

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

        # Fetch threads for this project to include in system prompt
        threads: List[Dict[str, Any]] = []
        try:
            thread_response = tool_executor.threads.list_threads(
                user_id,
                project_id=effective_project_id,
                status="active",
                limit=50,
            )
            threads = [
                {
                    "thread_id": t.thread_id,
                    "name": t.name,
                    "entry_count": None,  # Could be expensive to fetch
                }
                for t in thread_response.threads
            ]
            logger.info(f"[THREADS] Loaded {len(threads)} threads for project {effective_project_id}")
        except Exception as e:
            logger.warning(f"Failed to load threads for system prompt: {e}")

        # Build initial messages
        system_prompt = prompt_loader.load(
            "oracle/system.md",
            {
                "project_id": effective_project_id,
                "user_id": user_id,
                "vault_files": vault_files,
                "threads": threads,
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
        exchange_saved = False  # Track if we've already saved
        termination_reason: Optional[str] = None  # T037: Track termination reason

        # Create a "working node" for incremental saves during multi-turn execution
        # This allows us to save tool progress even if the agent halts mid-execution
        self._working_node_id: Optional[str] = None
        if self._current_tree_root_id and self._current_node_id and self.user_id:
            logger.info(
                f"[WORKING_NODE] Attempting to create working node: "
                f"tree={self._current_tree_root_id[:8]}, parent={self._current_node_id[:8]}"
            )
            try:
                working_node = self._tree_service.create_node(
                    user_id=self.user_id,
                    root_id=self._current_tree_root_id,
                    parent_id=self._current_node_id,
                    question=question,
                    answer="",  # Will be updated incrementally
                    tool_calls=None,
                    tokens_used=0,
                    model_used=self.model,
                )
                self._working_node_id = working_node.id
                self._current_node_id = working_node.id  # Update HEAD to working node
                logger.info(
                    f"[WORKING_NODE] Successfully created working node {working_node.id[:8]} "
                    f"for incremental saves"
                )
            except Exception as e:
                logger.error(
                    f"[WORKING_NODE] FAILED to create working node: {e}. "
                    f"Incremental saves will fall back to legacy context.",
                    exc_info=True
                )
                self._working_node_id = None
        else:
            logger.warning(
                f"[WORKING_NODE] Cannot create working node - missing prereqs: "
                f"tree_root={self._current_tree_root_id}, current_node={self._current_node_id}, "
                f"user={self.user_id}"
            )

        try:
            # Agent loop - T031: Use config.max_iterations instead of MAX_TURNS
            for turn in range(agent_config.max_iterations):
                # T030: Update state for this turn (1-based for display)
                agent_state = replace(agent_state, turn=turn + 1)

                # T032: Call decision tree at turn start
                agent_state = decision_tree.on_turn_start(agent_state)

                # T035: Check timeout before turn
                if agent_state.is_timed_out:
                    termination_reason = "timeout"
                    # T063: Log timeout termination event
                    logger.info(
                        f"[TERMINATION:timeout] Agent terminated at turn {turn + 1}: "
                        f"elapsed {agent_state.elapsed_seconds:.1f}s exceeds limit of {agent_config.timeout_seconds}s"
                    )
                    yield OracleStreamChunk(
                        type="system",
                        system_type="limit_reached",
                        system_message=f"Query timed out after {agent_state.elapsed_seconds:.0f} seconds",
                    )
                    break

                # Check if should continue via decision tree
                should_continue, reason = decision_tree.should_continue(agent_state)
                if not should_continue:
                    termination_reason = reason
                    # T063: Log specific termination events
                    if reason == "max_iterations":
                        logger.info(
                            f"[TERMINATION:max_iterations] Agent terminated at turn {turn + 1}: "
                            f"reached limit of {agent_config.max_iterations} iterations"
                        )
                    elif reason == "token_budget":
                        logger.info(
                            f"[TERMINATION:token_budget] Agent terminated at turn {turn + 1}: "
                            f"used {agent_state.tokens_used} tokens (budget: {agent_config.token_budget})"
                        )
                    elif reason == "no_progress":
                        logger.info(
                            f"[TERMINATION:no_progress] Agent terminated at turn {turn + 1}: "
                            f"detected 3 consecutive identical actions"
                        )
                    elif reason == "error_limit":
                        logger.info(
                            f"[TERMINATION:error_limit] Agent terminated at turn {turn + 1}: "
                            f"reached 3 consecutive errors"
                        )
                    else:
                        logger.warning(f"[TERMINATION:{reason}] Agent terminated at turn {turn + 1}: {reason}")
                    # Yield appropriate system chunk based on reason
                    system_type: Optional[str] = None
                    if "iteration" in reason or "max" in reason:
                        system_type = "limit_reached"
                    elif "token" in reason:
                        system_type = "limit_reached"
                    elif "progress" in reason:
                        system_type = "no_progress"
                    elif "error" in reason:
                        system_type = "error_limit"
                    else:
                        system_type = "limit_reached"
                    yield OracleStreamChunk(
                        type="system",
                        system_type=system_type,
                        system_message=f"Agent terminated: {reason}",
                    )
                    break

                # T033: Check for warnings and emit system chunks
                warnings = decision_tree.get_warning_state(agent_state)
                for warn_type, warn_info in warnings.items():
                    yield OracleStreamChunk(
                        type="system",
                        system_type="limit_warning",
                        system_message=(
                            f"Approaching {warn_type} limit: "
                            f"{warn_info['percent']:.0f}% used "
                            f"({warn_info['current_value']}/{warn_info['limit_value']})"
                        ),
                    )

                # Check cancellation before each turn
                if self._cancelled:
                    termination_reason = "cancelled"
                    # T063: Log cancellation termination event
                    logger.info(f"[TERMINATION:cancelled] Agent cancelled by user at turn {turn + 1}")
                    yield OracleStreamChunk(type="error", error="Cancelled by user")
                    return

                logger.debug(
                    f"Agent turn {turn + 1}/{agent_config.max_iterations} "
                    f"(tokens: {agent_state.tokens_used}, elapsed: {agent_state.elapsed_seconds:.1f}s)"
                )

                # Track tokens accumulated this turn for T034
                turn_content_start = len(accumulated_content)

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
                        termination_reason = "cancelled"
                        # T063: Log cancellation during streaming
                        logger.info(f"[TERMINATION:cancelled] Agent cancelled during streaming at turn {turn + 1}")
                        yield OracleStreamChunk(type="error", error="Cancelled by user")
                        return

                    # Track content and thinking for max-turns fallback
                    if chunk.type == "content" and chunk.content:
                        accumulated_content += chunk.content
                    elif chunk.type == "thinking" and chunk.content:
                        accumulated_thinking += chunk.content + "\n"

                    yield chunk

                    # Check if we're done - mark as saved since _agent_turn handles it
                    if chunk.type == "done":
                        exchange_saved = True
                        return

                    # If we got an error, stop
                    if chunk.type == "error":
                        termination_reason = "error"
                        return

                # T034: Update tokens after turn (estimate: 4 chars per token)
                turn_content = len(accumulated_content) - turn_content_start
                estimated_turn_tokens = turn_content // 4
                agent_state = replace(
                    agent_state,
                    tokens_used=agent_state.tokens_used + estimated_turn_tokens
                )

            else:
                # Loop completed without break - max iterations reached via loop exhaustion
                termination_reason = "max_iterations"
                # T063: Log max_iterations termination (loop exhaustion case)
                logger.info(
                    f"[TERMINATION:max_iterations] Agent loop exhausted at turn {agent_config.max_iterations}: "
                    f"accumulated {len(accumulated_content)} chars of content"
                )

            # T036/T037: Max turns reached or other termination - save content
            logger.debug(
                f"[TURN_CONTROL] Post-loop termination handling: "
                f"reason={termination_reason}, content_chars={len(accumulated_content)}"
            )

            if accumulated_content:
                # Yield what we have so far before the warning
                yield OracleStreamChunk(
                    type="content",
                    content=(
                        f"\n\n---\n*Note: Response incomplete due to "
                        f"{termination_reason or 'turn limit'} "
                        f"({agent_config.max_iterations} turns). "
                        f"Here is what was gathered:*\n\n"
                    ),
                )
            elif accumulated_thinking:
                # If no content but we have thinking, summarize what was found
                yield OracleStreamChunk(
                    type="content",
                    content=(
                        f"*Response incomplete after {agent_config.max_iterations} turns "
                        f"({termination_reason or 'limit reached'}). "
                        f"Last reasoning:\n{accumulated_thinking[-500:]}*"
                    ),
                )

            # T036: Save what we have as a partial exchange on any termination
            if accumulated_content or accumulated_thinking:
                saved_context_id = self._save_exchange(
                    question=question,
                    answer=accumulated_content or f"*Partial response - thinking: {accumulated_thinking[:500]}*",
                )
                exchange_saved = True
            else:
                saved_context_id = None

            # T037: Add termination reason to done chunk
            yield OracleStreamChunk(
                type="done",
                tokens_used=agent_state.tokens_used,
                model_used=self.model,
                context_id=saved_context_id,
                metadata={
                    "termination_reason": termination_reason,
                    "turns_used": agent_state.turn,
                    "max_turns": agent_config.max_iterations,
                } if termination_reason else None,
            )

        finally:
            # T036: Ensure we save something if the connection was dropped mid-response
            # This is the LAST CHANCE to save data before the generator exits
            logger.info(
                f"[FINALLY] Entering finally block: exchange_saved={exchange_saved}, "
                f"content_len={len(accumulated_content)}, thinking_len={len(accumulated_thinking)}, "
                f"tool_calls={len(self._collected_tool_calls)}, reason={termination_reason}"
            )

            if not exchange_saved:
                # Always try to save something if we haven't already
                has_content = accumulated_content or accumulated_thinking or self._collected_tool_calls
                if has_content or question:
                    logger.info(
                        f"[FINALLY] Saving partial exchange due to early termination "
                        f"(content={len(accumulated_content)}, thinking={len(accumulated_thinking)}, "
                        f"tools={len(self._collected_tool_calls)}, reason={termination_reason})"
                    )

                    # Build answer from available content
                    if accumulated_content:
                        answer = accumulated_content
                    elif self._collected_tool_calls:
                        # Format tool calls into answer if no content yet
                        tool_summary = []
                        for tc in self._collected_tool_calls:
                            tool_summary.append(f"- {tc.name}: {tc.status.value}")
                            if tc.result:
                                preview = tc.result[:150] + "..." if len(tc.result) > 150 else tc.result
                                tool_summary.append(f"  {preview}")
                        answer = f"*Response interrupted during tool execution*\n\n**Completed tools:**\n" + "\n".join(tool_summary)
                    elif accumulated_thinking:
                        answer = f"*Response interrupted*\n\n**Last reasoning:**\n{accumulated_thinking[:500]}"
                    else:
                        answer = "*Response interrupted before content was generated*"

                    try:
                        saved_id = self._save_exchange(
                            question=question,
                            answer=answer,
                        )
                        if saved_id:
                            logger.info(f"[FINALLY] Successfully saved partial exchange: {saved_id[:8]}")
                        else:
                            logger.error("[FINALLY] _save_exchange returned None - save may have failed")
                    except Exception as e:
                        logger.error(f"[FINALLY] Failed to save partial exchange: {e}", exc_info=True)

                        # Last resort: Try direct legacy save
                        try:
                            if self._context and self._context_service:
                                logger.info("[FINALLY] Attempting emergency legacy save")
                                emergency_exchange = OracleExchange(
                                    id=str(uuid.uuid4()),
                                    role=ExchangeRole.ASSISTANT,
                                    content=answer,
                                    tool_calls=self._collected_tool_calls if self._collected_tool_calls else None,
                                    timestamp=datetime.now(timezone.utc),
                                    token_count=len(answer) // 4,
                                )
                                self._context_service.add_exchange(
                                    user_id=self._context.user_id,
                                    project_id=self._context.project_id,
                                    exchange=emergency_exchange,
                                    model_used=self.model,
                                )
                                logger.info("[FINALLY] Emergency legacy save succeeded")
                        except Exception as e2:
                            logger.error(f"[FINALLY] CRITICAL: Emergency save also failed: {e2}")
                else:
                    logger.warning("[FINALLY] No content to save (no content, thinking, tools, or question)")
            else:
                logger.info("[FINALLY] Exchange already saved, skipping")

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
        # NOTE: Do NOT reset _turn_reasoning_content here!
        # DeepSeek requires reasoning_content to be passed back on subsequent API calls
        # during tool invocation. The reset happens at query() start, not per-turn.

        # Handle reasoning mode based on model capability
        model = self.model
        capability = get_model_capability(model)

        if thinking:
            if should_use_thinking_suffix(model):
                # Gemini-style: append suffix
                if not model.endswith(":thinking"):
                    model = f"{model}:thinking"
            # Note: reasoning_param handled in request body below

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

            # Add reasoning parameter for Claude models when thinking is enabled
            if thinking and should_use_reasoning_param(self.model):
                # Get reasoning effort from user settings
                settings_service = UserSettingsService()
                effort = settings_service.get_reasoning_effort(user_id)
                request_body["reasoning"] = {
                    "enabled": True,
                    "effort": effort,  # "low", "medium", or "high"
                }
                logger.info(f"[REASONING] Enabled reasoning for {self.model} with effort={effort}")

            # Add DeepSeek reasoning_content passback if captured from previous API call
            # DeepSeek requires this to be passed back during tool invocation turns
            if self._turn_reasoning_content and requires_reasoning_passback(self.model):
                request_body["reasoning_content"] = self._turn_reasoning_content
                logger.debug(
                    f"[DEEPSEEK] Passing back reasoning_content ({len(self._turn_reasoning_content)} chars)"
                )

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
            status_code = e.response.status_code
            response_text = e.response.text[:500] if e.response.text else "No details"
            logger.error(f"OpenRouter API error: {status_code} - {response_text}")

            # Provide specific error messages for common status codes
            if status_code == 400:
                # Log detailed request info for debugging
                total_content_length = sum(len(str(m.get("content", ""))) for m in messages)
                tool_results_count = sum(1 for m in messages if m.get("role") == "tool")
                logger.warning(
                    f"[API_400] Request details: {len(messages)} messages, "
                    f"~{total_content_length} chars, {tool_results_count} tool results, "
                    f"response: {response_text}"
                )
                # Don't assume cause - could be malformed request, rate limit, model issue
                error_msg = (
                    f"API returned 400 Bad Request. "
                    f"Details: {response_text[:200]}"
                )
            elif status_code == 429:
                error_msg = "Rate limited - please wait a moment and try again"
            elif status_code == 503:
                error_msg = "Service temporarily unavailable - please try again"
            else:
                error_msg = f"API error: {status_code}"

            yield OracleStreamChunk(
                type="error",
                error=error_msg,
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
        reasoning_buffer = ""  # Accumulate reasoning/thinking content for DSML parsing
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
                reasoning_buffer += delta["reasoning"]  # Accumulate for DSML parsing
                yield OracleStreamChunk(
                    type="thinking",
                    content=yielded_reasoning_text,
                )

            # Handle DeepSeek reasoning_content (MUST be captured for passback)
            # This is different from "reasoning" - it's a separate field that DeepSeek
            # requires to be passed back on subsequent API calls during tool invocation
            if "reasoning_content" in delta and delta["reasoning_content"]:
                reasoning_content = delta["reasoning_content"]
                # Capture for passback on next API call
                self._turn_reasoning_content = reasoning_content
                # Accumulate for DSML parsing (avoid double-counting)
                if reasoning_content != yielded_reasoning_text:
                    reasoning_buffer += reasoning_content
                # Also yield as thinking for display (if not already yielded from "reasoning")
                if reasoning_content != yielded_reasoning_text:
                    yield OracleStreamChunk(
                        type="thinking",
                        content=reasoning_content,
                    )
                    yielded_reasoning_text = reasoning_content

            # Handle reasoning_details (alternative format)
            # Only yield if different from what we already yielded from "reasoning"
            if "reasoning_details" in delta and delta["reasoning_details"]:
                for detail in delta["reasoning_details"]:
                    if isinstance(detail, dict) and detail.get("text"):
                        detail_text = detail["text"]
                        # Skip if this is the same as what we already yielded
                        if detail_text != yielded_reasoning_text:
                            reasoning_buffer += detail_text  # Accumulate for DSML parsing
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
                        if "name" in tc["function"] and tc["function"]["name"]:
                            tool_calls_buffer[idx]["function"]["name"] = tc["function"]["name"]
                        if "arguments" in tc["function"] and tc["function"]["arguments"]:
                            tool_calls_buffer[idx]["function"]["arguments"] += tc["function"]["arguments"]

        # Log final state
        logger.info(f"=== SSE STREAM ENDED after {chunk_count} chunks ===")
        logger.info(f"[FINAL STATE] finish_reason={finish_reason}")
        logger.info(f"[FINAL STATE] content_buffer_len={len(content_buffer)}")
        logger.info(f"[FINAL STATE] reasoning_buffer_len={len(reasoning_buffer)}")
        logger.info(f"[FINAL STATE] tool_calls_buffer={json.dumps(tool_calls_buffer)[:2000]}")
        if content_buffer:
            logger.info(f"[FINAL STATE] content_preview={content_buffer[:500]}")
        if reasoning_buffer:
            logger.info(f"[FINAL STATE] reasoning_preview={reasoning_buffer[:500]}")

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

            # CRITICAL: If no XML tool calls in content, check reasoning buffer for DSML tool calls
            # DeepSeek sometimes outputs tool calls in the reasoning field with custom DSML format
            if not xml_tool_calls and reasoning_buffer:
                dsml_parser = DSMLReasoningParser()
                if dsml_parser.can_parse(reasoning_buffer):
                    logger.warning(
                        f"[DSML] Found DSML markers in reasoning buffer ({len(reasoning_buffer)} chars). "
                        "Checking for tool calls..."
                    )
                    dsml_tool_calls, _ = _parse_xml_tool_calls(reasoning_buffer)
                    if dsml_tool_calls:
                        logger.warning(
                            f"[DSML] Found {len(dsml_tool_calls)} tool call(s) in reasoning field! "
                            "DeepSeek is outputting tool calls in reasoning instead of content."
                        )
                        xml_tool_calls = dsml_tool_calls
                        # Keep content as-is since tool calls were in reasoning

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

        # Check for DeepSeek reasoning_content in message (non-streaming response)
        # This MUST be captured for passback on subsequent API calls during tool invocation
        if "reasoning_content" in message and message["reasoning_content"]:
            reasoning_content = message["reasoning_content"]
            self._turn_reasoning_content = reasoning_content
            # Yield as thinking for display
            yield OracleStreamChunk(
                type="thinking",
                content=reasoning_content,
            )
            logger.debug(
                f"[DEEPSEEK] Captured reasoning_content from non-streaming response "
                f"({len(reasoning_content)} chars)"
            )

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

            # CRITICAL: If no XML tool calls in content, check reasoning_content for DSML tool calls
            # DeepSeek sometimes outputs tool calls in the reasoning field with custom DSML format
            reasoning_content = message.get("reasoning_content", "")
            if not xml_tool_calls and reasoning_content:
                dsml_parser = DSMLReasoningParser()
                if dsml_parser.can_parse(reasoning_content):
                    logger.warning(
                        f"[DSML] Found DSML markers in reasoning_content ({len(reasoning_content)} chars). "
                        "Checking for tool calls..."
                    )
                    dsml_tool_calls, _ = _parse_xml_tool_calls(reasoning_content)
                    if dsml_tool_calls:
                        logger.warning(
                            f"[DSML] Found {len(dsml_tool_calls)} tool call(s) in reasoning_content! "
                            "DeepSeek is outputting tool calls in reasoning instead of content."
                        )
                        xml_tool_calls = dsml_tool_calls
                        # Keep content as-is since tool calls were in reasoning

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

        # Get config limits (with safe defaults matching settings.py)
        max_tool_calls = 100  # Default (must match AgentConfig.max_tool_calls_per_turn default)
        max_parallel = 3  # Default
        tool_timeout = 30  # Default per-tool timeout in seconds
        if self._agent_config:
            max_tool_calls = self._agent_config.max_tool_calls_per_turn
            max_parallel = self._agent_config.max_parallel_tools
            tool_timeout = self._agent_config.tool_timeout_seconds

        # Log total tool call count
        logger.info(
            f"[TOOL_CALLS] Received {len(tool_calls)} tool calls "
            f"(limit: {max_tool_calls}, parallel: {max_parallel})"
        )

        # T012: Enforce max_tool_calls_per_turn limit
        if len(tool_calls) > max_tool_calls:
            truncated_count = len(tool_calls) - max_tool_calls
            logger.warning(
                f"[TOOL_LIMIT] Truncating tool calls from {len(tool_calls)} to {max_tool_calls} "
                f"({truncated_count} calls dropped)"
            )
            # Truncate the list to enforce the limit
            tool_calls = tool_calls[:max_tool_calls]

            # Emit system warning to user
            yield OracleStreamChunk(
                type="system",
                system_type="tool_limit",
                system_message=(
                    f"Tool call limit reached: executing {max_tool_calls} of "
                    f"{max_tool_calls + truncated_count} requested tool calls"
                ),
            )

        # Parse all tool calls first
        parsed_calls: List[Tuple[str, str, Dict[str, Any], str]] = []
        for call in tool_calls:
            call_id = call.get("id", str(uuid.uuid4()))
            function = call.get("function", {})
            # Handle case where name is None or empty (defensive parsing)
            name = function.get("name") or "unknown"
            arguments_str = function.get("arguments") or "{}"

            # Parse arguments
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Malformed JSON in tool call arguments for {name}: {arguments_str[:200]}",
                    exc_info=True
                )
                # Set arguments to empty dict but mark as malformed
                # The tool will fail with a clear error about missing required arguments
                arguments = {
                    "_json_parse_error": f"Malformed JSON arguments: {str(e)}",
                    "_raw_arguments": arguments_str[:500],
                }

            parsed_calls.append((call_id, name, arguments, arguments_str))

        # Within-turn deduplication: detect and filter duplicate tool calls
        seen_signatures: set = set()
        unique_calls: List[Tuple[str, str, Dict[str, Any], str]] = []
        duplicate_count = 0

        for call_id, name, arguments, arguments_str in parsed_calls:
            # Create signature for deduplication (sorted JSON for consistent comparison)
            signature = f"{name}:{json.dumps(arguments, sort_keys=True)}"
            if signature in seen_signatures:
                duplicate_count += 1
                logger.warning(f"[TOOL_DEDUP] Skipping duplicate tool call: {name}")
                # Yield warning to user
                yield OracleStreamChunk(
                    type="thinking",
                    content=f"Skipping duplicate call to {name}",
                )
            else:
                seen_signatures.add(signature)
                unique_calls.append((call_id, name, arguments, arguments_str))

        if duplicate_count > 0:
            logger.info(f"[TOOL_DEDUP] Filtered {duplicate_count} duplicate tool calls")

        # Check if >50% of calls were duplicates - emit system warning
        total_parsed = len(parsed_calls)
        if total_parsed > 0 and duplicate_count > total_parsed // 2:
            logger.warning(
                f"[TOOL_DEDUP] High duplicate rate: {duplicate_count}/{total_parsed} "
                f"({100 * duplicate_count / total_parsed:.0f}%) tool calls were duplicates"
            )
            yield OracleStreamChunk(
                type="system",
                system_type="no_progress",
                system_message=f"Model requested {duplicate_count} duplicate tool calls out of {total_parsed}",
            )

        # Replace parsed_calls with unique_calls for execution
        parsed_calls = unique_calls

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

        # Tools that should receive project_id context for scoping
        PROJECT_SCOPED_TOOLS = {
            # Thread tools
            "thread_list", "thread_read", "thread_seek", "thread_push",
            # Code tools
            "search_code", "coderag_status",
            # Vault tools
            "vault_read", "vault_write", "vault_search", "vault_list",
            "vault_move", "vault_create_index",
        }

        # Create semaphore to limit parallel tool execution
        semaphore = asyncio.Semaphore(max_parallel)
        logger.debug(f"[TOOL_PARALLEL] Created semaphore with limit {max_parallel}")

        # Execute all tools with controlled parallelism
        async def execute_single_tool(
            call_id: str,
            name: str,
            arguments: Dict[str, Any],
        ) -> ToolExecutionResult:
            """Execute a single tool and return structured result.

            Uses semaphore to limit concurrent executions to max_parallel_tools.
            Applies per-tool timeout from tool_timeout_seconds config.
            """
            import time as _time
            _tool_start = _time.perf_counter()
            async with semaphore:
                _acquired = _time.perf_counter()
                logger.info(f"[TOOL_PROFILING] Tool {name} (id={call_id[:8]}) acquired semaphore after {(_acquired - _tool_start)*1000:.1f}ms")
                try:
                    # Inject project_id for tools that need project context
                    if name in PROJECT_SCOPED_TOOLS and "project_id" not in arguments:
                        arguments = {**arguments, "project_id": self.project_id}
                        logger.info(f"[PROJECT_SCOPE] Injected project_id={self.project_id} for tool {name}")
                    elif name in PROJECT_SCOPED_TOOLS:
                        logger.info(f"[PROJECT_SCOPE] Tool {name} already has project_id={arguments.get('project_id')}")

                    # Add per-tool timeout to prevent single slow tools from blocking
                    async with asyncio.timeout(tool_timeout):
                        result = await tool_executor.execute(
                            name=name,
                            arguments=arguments,
                            user_id=user_id,
                        )
                    _elapsed = _time.perf_counter() - _tool_start
                    logger.info(f"[TOOL_PROFILING] Tool {name} (id={call_id[:8]}) completed in {_elapsed*1000:.1f}ms")
                    return ToolExecutionResult(
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        result=result,
                        success=True,
                    )
                except asyncio.TimeoutError:
                    _elapsed = _time.perf_counter() - _tool_start
                    logger.warning(f"[TOOL_TIMEOUT] Tool {name} (id={call_id[:8]}) timed out after {_elapsed:.1f}s (limit: {tool_timeout}s)")
                    return ToolExecutionResult(
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        error=f"Tool execution timed out after {tool_timeout} seconds",
                        success=False,
                    )
                except Exception as e:
                    _elapsed = _time.perf_counter() - _tool_start
                    logger.exception(f"[TOOL_PROFILING] Tool {name} (id={call_id[:8]}) FAILED after {_elapsed*1000:.1f}ms: {e}")
                    return ToolExecutionResult(
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        error=str(e),
                        success=False,
                    )

        # Prepare tool calls for batched execution
        # Separate JSON parse failures, think tools, and valid tool calls
        valid_calls: List[Tuple[str, str, Dict[str, Any]]] = []
        parse_failures: List[ToolExecutionResult] = []
        think_tool_results: List[Tuple[str, str, Dict[str, Any]]] = []  # (call_id, name, arguments)

        for call_id, name, arguments, _ in parsed_calls:
            # Check if this tool call had a JSON parsing error
            if "_json_parse_error" in arguments:
                # Create immediate failure result for JSON parse errors
                parse_failures.append(ToolExecutionResult(
                    call_id=call_id,
                    name=name,
                    arguments={},
                    error=arguments["_json_parse_error"],
                    success=False,
                ))
            elif name == "think":
                # Handle think tool specially - don't send to tool_executor
                think_tool_results.append((call_id, name, arguments))
            else:
                # Normal tool call - add to execution queue
                valid_calls.append((call_id, name, arguments))

        # Handle think tools inline (before parallel execution of other tools)
        # Think tool should yield a thinking chunk and add result to messages
        for call_id, name, arguments in think_tool_results:
            thought = arguments.get("thought", "")
            logger.info(f"[THINK_TOOL] Processing think tool call: {thought[:100]}...")

            # Yield thinking chunk for frontend display
            yield OracleStreamChunk(
                type="thinking",
                content=f"\U0001F4AD {thought}",
            )

            # Add to messages so model sees it on next API call
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps({
                    "recorded": True,
                    "thought": thought,
                    "note": "Use this reasoning in your next steps",
                }),
            })

            # Track think tool call for context persistence
            self._collected_tool_calls.append(
                ToolCall(
                    id=call_id,
                    name="think",
                    arguments={"thought": thought},
                    result="recorded",
                    status=ToolCallStatus.SUCCESS,
                )
            )

            # Yield tool result chunk to update UI status
            yield OracleStreamChunk(
                type="tool_result",
                tool_call_id=call_id,
                tool_result=json.dumps({"recorded": True, "thought_length": len(thought)}),
            )

        # Calculate totals for logging and progress tracking
        total_calls = len(parsed_calls)
        logger.info(
            f"[TOOL_STREAMING] Preparing {len(valid_calls)} valid tools for progressive streaming "
            f"(max_parallel={max_parallel}, {len(parse_failures)} parse failures)"
        )

        # Start timing for overall execution
        import time as _time
        _total_start = _time.perf_counter()

        # First, yield parse failures immediately (they don't need execution)
        for failure_idx, failure in enumerate(parse_failures):
            async for chunk in self._yield_tool_result(failure, failure_idx, total_calls, messages):
                yield chunk

        # Create all tasks upfront with index tracking for progressive streaming
        all_tasks: List[asyncio.Task] = []
        task_to_index: Dict[asyncio.Task, int] = {}
        task_to_info: Dict[asyncio.Task, Tuple[str, str]] = {}

        for idx, (call_id, name, arguments) in enumerate(valid_calls):
            task = asyncio.create_task(execute_single_tool(call_id, name, arguments))
            all_tasks.append(task)
            task_to_index[task] = idx
            task_to_info[task] = (call_id, name)

        # Track completed results for final message ordering
        # Results are yielded immediately in completion order but added to
        # messages in original order for consistent model context
        completed_results: Dict[int, ToolExecutionResult] = {}
        total_valid = len(valid_calls)
        completed_count = 0

        logger.info(f"[TOOL_STREAMING] Starting progressive execution of {total_valid} tools")

        # Stream results as they complete using asyncio.wait with FIRST_COMPLETED
        # This provides both immediate streaming AND heartbeat support for long-running tools
        HEARTBEAT_INTERVAL = 5.0  # seconds
        _stream_start = _time.perf_counter()
        pending: set = set(all_tasks)

        while pending:
            # Wait for any task to complete, or timeout for heartbeat
            done, pending = await asyncio.wait(
                pending,
                timeout=HEARTBEAT_INTERVAL,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if done:
                # Process completed tasks immediately
                for task in done:
                    try:
                        tool_result = task.result()
                        completed_count += 1
                        remaining = total_valid - completed_count

                        # Get original index from our tracking dict
                        original_idx = task_to_index.get(task)
                        if original_idx is None:
                            # Fallback: find by call_id
                            for idx, (cid, _, _) in enumerate(valid_calls):
                                if cid == tool_result.call_id and idx not in completed_results:
                                    original_idx = idx
                                    break

                        if original_idx is not None:
                            completed_results[original_idx] = tool_result
                            logger.info(
                                f"[TOOL_STREAMING] Result {completed_count}/{total_valid} completed "
                                f"({remaining} remaining): {tool_result.name} (id={tool_result.call_id[:8]}, order={original_idx})"
                            )
                        else:
                            # Emergency fallback
                            original_idx = completed_count - 1
                            completed_results[original_idx] = tool_result
                            logger.warning(
                                f"[TOOL_STREAMING] Could not determine original index for {tool_result.name}"
                            )

                        # Yield result immediately (out of order for better responsiveness)
                        # Include order metadata so frontend can reorder if needed
                        if tool_result.success and tool_result.result is not None:
                            yield OracleStreamChunk(
                                type="tool_result",
                                tool_call_id=tool_result.call_id,
                                tool_result=(
                                    tool_result.result[:2000]
                                    if len(tool_result.result) > 2000
                                    else tool_result.result
                                ),
                                metadata={"order": original_idx},
                            )

                            # Yield thinking update about tool completion
                            result_preview = (
                                tool_result.result[:100] + "..."
                                if len(tool_result.result) > 100
                                else tool_result.result
                            )
                            yield OracleStreamChunk(
                                type="thinking",
                                content=f"✓ {tool_result.name} returned: {result_preview}",
                            )

                            # Extract sources from successful result
                            self._extract_sources_from_result(tool_result.name, tool_result.result)

                            # Collect tool call for context persistence
                            self._collected_tool_calls.append(
                                ToolCall(
                                    id=tool_result.call_id,
                                    name=tool_result.name,
                                    arguments=tool_result.arguments,
                                    result=(
                                        tool_result.result[:2000]
                                        if len(tool_result.result) > 2000
                                        else tool_result.result
                                    ),
                                    status=ToolCallStatus.SUCCESS,
                                )
                            )

                            # INCREMENTAL SAVE: Save after each tool completion
                            # This ensures we don't lose progress if cancelled mid-execution
                            self._save_tool_progress()
                        else:
                            # Error case
                            error_content = self._format_tool_error(
                                tool_result.name,
                                tool_result.error or "Unknown error",
                                tool_result.arguments,
                            )

                            yield OracleStreamChunk(
                                type="tool_result",
                                tool_call_id=tool_result.call_id,
                                tool_result=error_content,
                                metadata={"order": original_idx},
                            )

                            yield OracleStreamChunk(
                                type="thinking",
                                content=f"✗ {tool_result.name} failed: {tool_result.error or 'Unknown error'}",
                            )

                            # Collect failed tool call for context persistence
                            self._collected_tool_calls.append(
                                ToolCall(
                                    id=tool_result.call_id,
                                    name=tool_result.name,
                                    arguments=tool_result.arguments,
                                    result=tool_result.error,
                                    status=ToolCallStatus.ERROR,
                                )
                            )

                            # INCREMENTAL SAVE: Save even failed tools
                            self._save_tool_progress()

                    except Exception as e:
                        # Handle unexpected exceptions from task
                        cid, n = task_to_info.get(task, ("unknown", "unknown"))
                        logger.exception(f"[TOOL_STREAMING] Task {n} raised exception: {e}")
                        completed_count += 1

            elif pending:
                # Timeout with no completions - emit heartbeat status chunk
                elapsed = _time.perf_counter() - _stream_start
                pending_tools = [
                    task_to_info.get(t, ("?", "unknown"))[1]
                    for t in pending
                ]
                status_msg = f"Still processing {len(pending)} tool(s): {', '.join(pending_tools[:3])}"
                if len(pending_tools) > 3:
                    status_msg += f" (+{len(pending_tools) - 3} more)"
                status_msg += f" ({elapsed:.0f}s elapsed)"

                logger.debug(f"[TOOL_STREAMING] Emitting heartbeat: {status_msg}")
                yield OracleStreamChunk(
                    type="status",
                    message=status_msg,
                )

        # Add tool results to messages in CORRECT ORDER for consistent model context
        # This ensures the model sees results in the same order it requested the tools
        logger.info(f"[TOOL_STREAMING] Adding {len(completed_results)} results to messages in original order")
        for idx, (call_id, name, arguments) in enumerate(valid_calls):
            tool_result = completed_results.get(idx)
            if tool_result is None:
                logger.error(f"[TOOL_STREAMING] Missing result for index {idx} ({name})")
                continue

            if tool_result.success and tool_result.result is not None:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result.call_id,
                    "content": tool_result.result,
                })
            else:
                error_content = self._format_tool_error(
                    tool_result.name,
                    tool_result.error or "Unknown error",
                    tool_result.arguments,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result.call_id,
                    "content": error_content,
                })

        _total_elapsed = _time.perf_counter() - _total_start
        logger.info(
            f"[TOOL_STREAMING] All {total_calls} tool results streamed in {_total_elapsed:.3f}s"
        )

        # Incremental save after tool execution round
        # This persists tool progress even if the agent halts mid-execution
        self._save_tool_progress()

    async def _yield_tool_result(
        self,
        result: ToolExecutionResult,
        result_idx: int,
        total_results: int,
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """Yield chunks for a single tool result.

        Extracts the result-yielding logic into a reusable async generator
        to support batched execution with yield points.

        Args:
            result: The tool execution result to yield
            result_idx: Index of this result (0-based, for logging)
            total_results: Total number of results expected
            messages: Conversation messages (mutated with tool result)

        Yields:
            OracleStreamChunk for tool_result and thinking updates
        """
        if result.success and result.result is not None:
            # Success case
            logger.info(
                f"[TOOL_PROFILING] Yielding result #{result_idx+1}/{total_results}: "
                f"{result.name} (id={result.call_id[:8]})"
            )
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
            result_preview = (
                result.result[:100] + "..."
                if len(result.result) > 100
                else result.result
            )
            yield OracleStreamChunk(
                type="thinking",
                content=f"✓ {result.name} returned: {result_preview}",
            )
            logger.info(
                f"[TOOL_PROFILING] Yielded thinking for result #{result_idx+1}/{total_results}: "
                f"{result.name}"
            )

            # Extract sources from successful result
            self._extract_sources_from_result(result.name, result.result)

            # Collect tool call for context persistence
            self._collected_tool_calls.append(
                ToolCall(
                    id=result.call_id,
                    name=result.name,
                    arguments=result.arguments,
                    result=(
                        result.result[:2000]
                        if len(result.result) > 2000
                        else result.result
                    ),
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
        - Understand what failed and why
        - Categorize the error (configuration vs user input vs runtime)
        - Decide whether to retry with different parameters
        - Choose an alternative approach

        The error includes:
        - error: The raw error message
        - category: One of configuration_error, user_input_error, resource_error, network_error, etc.
        - suggestion: Context-specific advice on how to recover
        - failed_arguments: The arguments that were passed (for debugging)

        Args:
            tool_name: Name of the failed tool
            error: Error message from tool execution
            arguments: Arguments that were passed to the tool

        Returns:
            JSON-formatted error message with structured information
        """
        # Try to parse if error is already JSON (from tool_executor)
        error_data = {}
        if isinstance(error, str) and error.startswith('{'):
            try:
                error_data = json.loads(error)
            except json.JSONDecodeError:
                pass

        # If we successfully parsed JSON from tool_executor, use it
        if error_data and "category" in error_data:
            # Merge in the suggestion from oracle agent level
            if "suggestion" not in error_data:
                error_data["suggestion"] = self._get_error_suggestion(tool_name, error)
            error_data["failed_arguments"] = arguments
            return json.dumps(error_data)

        # Fallback: format as a simple error with oracle-level suggestion
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

    def _save_tool_progress(self) -> bool:
        """Save incremental tool progress to the working node.

        This is called after each tool execution round to persist progress.
        If the agent halts mid-execution (e.g., user refresh), this ensures
        that completed tool calls are preserved.

        This method is designed to be safe - it will not raise exceptions
        even if the save fails, to avoid interrupting the agent loop.

        Returns:
            True if save succeeded, False otherwise
        """
        # Log current state for debugging
        logger.info(
            f"[SAVE_PROGRESS] Attempting save: working_node_id={self._working_node_id}, "
            f"user_id={self.user_id}, tool_calls={len(self._collected_tool_calls)}"
        )

        if not self._working_node_id:
            logger.warning("[SAVE_PROGRESS] No working node ID - cannot save to tree, will try legacy")
            return self._save_tool_progress_legacy()

        if not self.user_id:
            logger.error("[SAVE_PROGRESS] No user_id - cannot save")
            return False

        if not self._collected_tool_calls:
            logger.debug("[SAVE_PROGRESS] No tool calls to save")
            return True  # Nothing to save is success

        try:
            import time as _time
            _save_start = _time.perf_counter()

            # Estimate tokens from tool results
            total_result_chars = sum(len(tc.result or "") for tc in self._collected_tool_calls)
            estimated_tokens = total_result_chars // 4  # Rough estimate: 4 chars per token

            # Update the working node with current tool progress
            result = self._tree_service.update_node_content(
                user_id=self.user_id,
                node_id=self._working_node_id,
                tool_calls=self._collected_tool_calls,
                tokens_used=estimated_tokens,
            )

            # Check if update actually succeeded (returns None if node not found)
            if result is None:
                logger.error(
                    f"[SAVE_PROGRESS] update_node_content returned None - "
                    f"node {self._working_node_id} may not exist. Trying legacy save."
                )
                return self._save_tool_progress_legacy()

            _save_elapsed = _time.perf_counter() - _save_start
            logger.info(
                f"[SAVE_PROGRESS] Successfully saved {len(self._collected_tool_calls)} tools "
                f"to node {self._working_node_id[:8]} in {_save_elapsed*1000:.1f}ms"
            )
            return True

        except Exception as e:
            # Log and try fallback to legacy save
            logger.error(f"[SAVE_PROGRESS] Tree save failed: {e}, trying legacy save")
            return self._save_tool_progress_legacy()

    def _save_tool_progress_legacy(self) -> bool:
        """Fallback: Save tool progress to legacy context system.

        This is used when tree-based saves fail or working node doesn't exist.

        Returns:
            True if save succeeded, False otherwise
        """
        if not self._context or not self._collected_tool_calls:
            logger.warning("[SAVE_PROGRESS_LEGACY] No context or no tool calls to save")
            return False

        try:
            # Build a summary of tool calls for legacy save
            tool_summary = []
            for tc in self._collected_tool_calls:
                tool_summary.append(f"- {tc.name}: {tc.status.value}")
                if tc.result:
                    tool_summary.append(f"  Result: {tc.result[:200]}...")

            summary_content = f"[Tool Progress]\n" + "\n".join(tool_summary)

            # Create a partial exchange in legacy system
            partial_exchange = OracleExchange(
                id=str(uuid.uuid4()),
                role=ExchangeRole.ASSISTANT,
                content=summary_content,
                tool_calls=self._collected_tool_calls,
                timestamp=datetime.now(timezone.utc),
                token_count=len(summary_content) // 4,
            )

            self._context = self._context_service.add_exchange(
                user_id=self._context.user_id,
                project_id=self._context.project_id,
                exchange=partial_exchange,
                model_used=self.model,
            )

            logger.info(
                f"[SAVE_PROGRESS_LEGACY] Saved {len(self._collected_tool_calls)} "
                f"tool calls to legacy context {self._context.id}"
            )
            return True

        except Exception as e:
            logger.error(f"[SAVE_PROGRESS_LEGACY] Failed to save to legacy context: {e}")
            return False

    def _save_exchange(
        self,
        question: str,
        answer: str,
        tokens_used: Optional[int] = None,
    ) -> Optional[str]:
        """Save the question and answer exchange to persistent context.

        This is called after a successful response to persist the conversation
        for future context loading. Saves to BOTH tree-based and legacy systems.

        IMPORTANT: This method implements defensive saving - if tree save fails,
        it falls back to legacy. If both fail, it logs detailed error info.

        Args:
            question: User's question
            answer: Assistant's full response
            tokens_used: Total tokens consumed (if known)

        Returns:
            Node ID if saved successfully (for tree), or legacy context ID
        """
        result_id = None
        tree_save_success = False
        legacy_save_success = False

        # Log what we're trying to save
        logger.info(
            f"[SAVE_EXCHANGE] Starting save: question_len={len(question)}, "
            f"answer_len={len(answer)}, tool_calls={len(self._collected_tool_calls)}, "
            f"working_node={self._working_node_id}, tree_root={self._current_tree_root_id}"
        )

        # SAVE TO TREE-BASED SYSTEM (primary)
        if self._current_tree_root_id and self.user_id:
            try:
                # Check if we have a working node to update (created at query start)
                if self._working_node_id:
                    # Update existing working node with final answer
                    updated_node = self._tree_service.update_node_content(
                        user_id=self.user_id,
                        node_id=self._working_node_id,
                        answer=answer,
                        tool_calls=self._collected_tool_calls if self._collected_tool_calls else None,
                        tokens_used=tokens_used or self._estimate_tokens(question + answer),
                    )

                    # Verify the update succeeded
                    if updated_node is not None:
                        result_id = self._working_node_id
                        tree_save_success = True
                        logger.info(
                            f"[SAVE_EXCHANGE] Successfully updated working node {self._working_node_id[:8]} "
                            f"(tree: {self._current_tree_root_id[:8]})"
                        )
                    else:
                        logger.error(
                            f"[SAVE_EXCHANGE] update_node_content returned None for "
                            f"node {self._working_node_id} - node may have been deleted!"
                        )

                # Fallback if working node update failed or didn't exist
                if not tree_save_success and self._current_node_id:
                    logger.info(
                        f"[SAVE_EXCHANGE] Creating new node as fallback "
                        f"(working_node failed or missing)"
                    )
                    # Fallback: Create new node as child of current HEAD
                    # (This path is used when working node creation failed at query start)
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
                    tree_save_success = True

                    logger.info(
                        f"[SAVE_EXCHANGE] Created fallback node {new_node.id[:8]} "
                        f"(tree: {self._current_tree_root_id[:8]})"
                    )

            except Exception as e:
                logger.error(f"[SAVE_EXCHANGE] Tree save failed: {e}", exc_info=True)
                tree_save_success = False

        # SAVE TO LEGACY SYSTEM (always attempt, as backup)
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

                legacy_save_success = True
                logger.info(
                    f"[SAVE_EXCHANGE] Legacy save succeeded: context={self._context.id}, "
                    f"total_exchanges={len(self._context.recent_exchanges)}"
                )

                # Use legacy ID as fallback if tree save failed
                if not result_id:
                    result_id = self._context.id

            except Exception as e:
                logger.error(f"[SAVE_EXCHANGE] Legacy save failed: {e}", exc_info=True)
                legacy_save_success = False
                if not result_id and self._context:
                    result_id = self._context.id
        else:
            logger.warning("[SAVE_EXCHANGE] No legacy context available for backup save")

        # Final summary log
        if not tree_save_success and not legacy_save_success:
            logger.error(
                f"[SAVE_EXCHANGE] CRITICAL: Both tree and legacy saves failed! "
                f"Data may be lost. Question: {question[:100]}..., Answer: {answer[:100]}..."
            )
        else:
            logger.info(
                f"[SAVE_EXCHANGE] Completed: tree={tree_save_success}, legacy={legacy_save_success}, "
                f"result_id={result_id[:8] if result_id else None}"
            )

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
