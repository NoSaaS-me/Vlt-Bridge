"""Oracle Agent - Main AI agent with tool calling (009-oracle-agent).

DEPRECATED: This module is replaced by OracleBTWrapper (020-bt-oracle-agent).
The BT-controlled Oracle uses the behavior tree runtime for execution control,
with LLM calls routed through OpenRouterClient.

To use the new implementation:
    from backend.src.bt.wrappers import OracleBTWrapper

    wrapper = OracleBTWrapper(
        user_id="user-id",
        api_key="openrouter-api-key",
        project_id="project-id",
        model="deepseek/deepseek-chat",
    )

    async for chunk in wrapper.process_query(query="Hello", context_id=None):
        print(chunk.type, chunk.content)

This legacy implementation is kept for reference but is no longer used.

---

Original docstring:

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
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from ..models.oracle import OracleStreamChunk, SourceReference, StreamEventType
from ..models.settings import ModelInfo
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
from .tool_parsers import ToolCallParserChain

# ANS imports for event emission
from .ans.bus import get_event_bus, EventBus
from .ans.event import Event, EventType, Severity
from .ans.accumulator import NotificationAccumulator
from .ans.subscriber import SubscriberLoader
from .ans.toon_formatter import get_toon_formatter
from .ans.persistence import CrossSessionPersistenceService, get_persistence_service
from .ans.deferred import reset_deferred_queue, get_deferred_queue

# Plugin system imports for rule engine
from .plugins.engine import RuleEngine
from .plugins.loader import RuleLoader
from .plugins.expression import ExpressionEvaluator
from .plugins.actions import ActionDispatcher
from .plugins.context import (
    EventData,
    HistoryState,
    PluginState,
    ProjectState,
    RuleContext,
    ToolCallRecord,
    TurnState,
    UserState,
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


# Default context window sizes for common models (fallback when API doesn't provide)
DEFAULT_MODEL_CONTEXT_SIZES = {
    # DeepSeek models
    "deepseek/deepseek-chat": 64000,
    "deepseek/deepseek-coder": 64000,
    "deepseek/deepseek-v3": 64000,
    "deepseek/deepseek-r1": 64000,
    # Anthropic models
    "anthropic/claude-3-opus": 200000,
    "anthropic/claude-3-sonnet": 200000,
    "anthropic/claude-3-haiku": 200000,
    "anthropic/claude-sonnet-4": 200000,
    "anthropic/claude-3.5-sonnet": 200000,
    "anthropic/claude-3.7-sonnet": 200000,
    # Google models
    "gemini-2.0-flash-exp": 1000000,
    "gemini-1.5-pro": 2000000,
    "gemini-1.5-flash": 1000000,
    "google/gemini-pro": 128000,
    "google/gemini-flash-1.5": 1000000,
    # OpenAI models
    "openai/gpt-4-turbo": 128000,
    "openai/gpt-4": 8192,
    "openai/gpt-3.5-turbo": 16384,
    "openai/o1": 200000,
    "openai/o1-mini": 128000,
    # Meta Llama models
    "meta-llama/llama-3-70b": 8192,
    "meta-llama/llama-3.1-70b": 131072,
    "meta-llama/llama-3.3-70b": 131072,
    # Mistral models
    "mistralai/mistral-large": 128000,
    "mistralai/mixtral-8x7b": 32000,
    # Qwen models
    "qwen/qwen-2.5-72b": 131072,
}

# Default fallback for unknown models
DEFAULT_CONTEXT_SIZE = 64000


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string.

    Uses a simple heuristic of ~4 characters per token for English text.
    This is reasonably accurate for most LLM tokenizers.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Average ~4 characters per token for English
    # Adjust slightly for code/JSON which tends to have more tokens per char
    return max(1, len(text) // 4)


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    """Estimate tokens for a conversation message.

    Accounts for message structure overhead (role, separators, etc.)

    Args:
        message: A conversation message dict with 'role' and 'content'

    Returns:
        Estimated token count including overhead
    """
    content = message.get("content") or ""
    role = message.get("role", "")

    # Base content tokens
    tokens = estimate_tokens(content)

    # Add overhead for message structure (~4 tokens for role + formatting)
    tokens += 4

    # Tool calls add significant overhead
    if "tool_calls" in message and message["tool_calls"]:
        for tc in message["tool_calls"]:
            func = tc.get("function", {})
            tokens += estimate_tokens(func.get("name", ""))
            tokens += estimate_tokens(func.get("arguments", ""))
            tokens += 10  # Overhead for tool call structure

    return tokens


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for a list of conversation messages.

    Args:
        messages: List of conversation messages

    Returns:
        Total estimated token count
    """
    total = 0
    for msg in messages:
        total += estimate_message_tokens(msg)
    # Add conversation overhead (beginning/end markers, etc.)
    total += 3
    return total


def get_model_context_size(model_id: str) -> int:
    """Get the context window size for a model.

    First checks the default sizes dict, then falls back to a reasonable default.

    Args:
        model_id: The model identifier (e.g., 'deepseek/deepseek-chat')

    Returns:
        Context window size in tokens
    """
    # Try exact match first
    if model_id in DEFAULT_MODEL_CONTEXT_SIZES:
        return DEFAULT_MODEL_CONTEXT_SIZES[model_id]

    # Try matching base model (without :free, :thinking suffixes)
    base_model = model_id.split(":")[0]
    if base_model in DEFAULT_MODEL_CONTEXT_SIZES:
        return DEFAULT_MODEL_CONTEXT_SIZES[base_model]

    # Try partial match (for models like 'deepseek/deepseek-chat:free')
    for key, size in DEFAULT_MODEL_CONTEXT_SIZES.items():
        if key in model_id or model_id.startswith(key):
            return size

    # Check for model family patterns
    model_lower = model_id.lower()
    if "claude" in model_lower:
        return 200000
    elif "gemini" in model_lower:
        return 1000000
    elif "deepseek" in model_lower:
        return 64000
    elif "gpt-4" in model_lower:
        return 128000
    elif "llama" in model_lower:
        return 131072
    elif "mistral" in model_lower:
        return 128000
    elif "qwen" in model_lower:
        return 131072

    return DEFAULT_CONTEXT_SIZE


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


# Patterns that indicate tool call syntax starting
# Used by streaming to detect when to stop yielding content
_TOOL_CALL_PREFIXES = (
    "<function_calls",
    "<invoke",
    "<｜DSML｜",
    "ALERT:",  # Common model prefix before tool calls
)

# Pattern for inline tool calls like: tool_name{"arg": "value"}
# or tool_name{json...}
_INLINE_TOOL_CALL_PATTERN = re.compile(
    r'^([a-z_][a-z0-9_]*)\s*\{',
    re.IGNORECASE
)

def _is_tool_call_content(content: str) -> bool:
    """Check if content looks like it contains tool call syntax.

    This is used during streaming to detect when the model is outputting
    tool calls as plain text instead of using proper function calling.

    Args:
        content: The accumulated content buffer

    Returns:
        True if content appears to contain tool call syntax
    """
    # Check for XML-style prefixes
    stripped = content.strip()
    for prefix in _TOOL_CALL_PREFIXES:
        if stripped.startswith(prefix):
            return True

    # Check for inline tool calls (tool_name{json})
    if _INLINE_TOOL_CALL_PATTERN.match(stripped):
        return True

    return False


def _extract_safe_content(content: str) -> Tuple[str, str]:
    """Extract content that is safe to show to users, separating tool call syntax.

    Some models output a mix of readable content and tool call syntax.
    This function separates them so we can show readable content while
    still parsing and executing tool calls.

    Args:
        content: The accumulated content buffer

    Returns:
        Tuple of (safe_content, tool_call_content) where:
        - safe_content: Content that should be shown to the user
        - tool_call_content: Content that contains tool call syntax
    """
    # Check for common patterns where content precedes tool calls
    patterns_to_split = [
        "\n<function_calls",
        "\n<invoke",
        "\n<｜DSML｜",
        "\n\nALERT:",
        "\n\n<function_calls",
        "\n\n<invoke",
    ]

    for pattern in patterns_to_split:
        if pattern in content:
            idx = content.index(pattern)
            return content[:idx].strip(), content[idx:].strip()

    # Check if the entire content is tool call syntax
    if _is_tool_call_content(content):
        return "", content

    return content, ""


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
        self._collected_system_messages: List[str] = []
        self._context_service = context_service or get_context_service()
        self._tree_service = tree_service or get_context_tree_service()

        # Tree-based context tracking
        self._current_tree_root_id: Optional[str] = None
        self._current_node_id: Optional[str] = None

        # Cancellation support
        self._cancelled = False
        self._active_tasks: List[asyncio.Task] = []

        # Loop detection tracking
        self._recent_tool_patterns: List[str] = []  # Track recent tool call patterns
        self._loop_detection_window = 6  # Number of recent patterns to track
        self._loop_threshold = 3  # Number of repetitions to trigger loop detection
        self._loop_already_warned = False  # Prevent repeated warnings for same loop

        # ANS (Agent Notification System) initialization
        self._event_bus = get_event_bus()
        self._accumulator = NotificationAccumulator()
        self._toon_formatter = get_toon_formatter()
        self._subscriber_loader = SubscriberLoader()

        # Load and register subscribers
        subscribers = self._subscriber_loader.load_all()
        self._accumulator.register_subscribers(list(subscribers.values()))

        # Budget tracking state (reset per query)
        self._iteration_warning_emitted = False
        self._iteration_exceeded_emitted = False
        self._token_warning_emitted = False
        self._token_exceeded_emitted = False
        self._total_tokens_used = 0
        self._max_tokens_budget = 0  # Set per query

        # Budget thresholds (configurable)
        self.ITERATION_WARNING_THRESHOLD = 0.70  # 70% of MAX_TURNS
        self.TOKEN_WARNING_THRESHOLD = 0.80  # 80% of max_tokens

        # Context window tracking (for UI display)
        self._context_tokens = 0  # Current tokens in context
        self._max_context_tokens = DEFAULT_CONTEXT_SIZE  # Model's max context
        self._last_context_update_tokens = 0  # Last sent update (avoid spam)

        # Context limit warning (014-ans-enhancements)
        self._context_limit_warning_emitted = False
        self.CONTEXT_WARNING_THRESHOLD = 0.70  # 70% of model context window

        # Cross-session persistence (014-ans-enhancements Feature 3)
        self._persistence_service = get_persistence_service()

        # Plugin System - RuleEngine (015-oracle-plugin-system Phase 4)
        self._rule_engine: Optional[RuleEngine] = None
        self._init_rule_engine()

    def _init_rule_engine(self) -> None:
        """Initialize the RuleEngine for plugin-based rule evaluation.

        Sets up the RuleEngine with:
        - RuleLoader pointing to the rules directory
        - ExpressionEvaluator for condition evaluation
        - ActionDispatcher for action execution
        - EventBus subscription for lifecycle events
        """
        try:
            # Rules directory relative to this file
            rules_dir = Path(__file__).parent / "plugins" / "rules"

            # Only initialize if rules directory exists
            if not rules_dir.exists():
                logger.info(f"Rules directory does not exist, skipping RuleEngine init: {rules_dir}")
                return

            loader = RuleLoader(rules_dir)
            evaluator = ExpressionEvaluator()
            dispatcher = ActionDispatcher(
                event_bus=self._event_bus,
                state_setter=self._set_plugin_state,
            )

            self._rule_engine = RuleEngine(
                loader=loader,
                evaluator=evaluator,
                dispatcher=dispatcher,
                event_bus=self._event_bus,
                context_builder=self._build_rule_context,
                auto_subscribe=True,
            )

            self._rule_engine.start()
            logger.info(f"RuleEngine initialized with {self._rule_engine.rule_count} rules")

        except Exception as e:
            logger.error(f"Failed to initialize RuleEngine: {e}")
            self._rule_engine = None

    def _set_plugin_state(self, key: str, value: Any) -> None:
        """Set a value in plugin-scoped state.

        Used by ActionDispatcher for set_state actions.
        State is persisted per user/project in the context.

        Args:
            key: State key to set.
            value: Value to store.
        """
        # For now, log the state change. Full persistence can be added later.
        logger.debug(f"Plugin state set: {key} = {value}")
        # TODO: Persist to plugin_state table when database schema is ready

    def _build_rule_context(self, event: Event) -> RuleContext:
        """Build a RuleContext from the current agent state.

        Called by RuleEngine when evaluating rules. Creates a snapshot
        of the agent's current state for rule condition evaluation.

        Args:
            event: The event that triggered rule evaluation.

        Returns:
            RuleContext populated with current agent state.
        """
        # Build turn state
        turn_number = getattr(self, '_current_turn', 1)
        token_usage = 0.0
        if self._max_tokens_budget > 0:
            token_usage = min(1.0, self._total_tokens_used / self._max_tokens_budget)
        context_usage = 0.0
        if self._max_context_tokens > 0:
            context_usage = min(1.0, self._context_tokens / self._max_context_tokens)

        turn = TurnState(
            number=turn_number,
            token_usage=token_usage,
            context_usage=context_usage,
            iteration_count=getattr(self, '_iteration_count', 0),
        )

        # Build history state from collected data
        messages = []
        if hasattr(self, '_context') and self._context:
            for exchange in self._context.recent_exchanges:
                messages.append({
                    "role": exchange.role.value,
                    "content": exchange.content,
                })

        tools = []
        failures: Dict[str, int] = {}
        for tc in self._collected_tool_calls:
            record = ToolCallRecord(
                name=tc.name,
                arguments=tc.arguments,
                result=tc.result,
                success=tc.status == ToolCallStatus.SUCCESS,
                timestamp=datetime.now(timezone.utc),
            )
            tools.append(record)
            if tc.status == ToolCallStatus.FAILED:
                failures[tc.name] = failures.get(tc.name, 0) + 1

        history = HistoryState(
            messages=messages,
            tools=tools,
            failures=failures,
        )

        # User and project state
        user = UserState(
            id=self.user_id or "unknown",
            settings={},  # Could load from user_settings in future
        )

        project = ProjectState(
            id=self.project_id,
            settings={},  # Could load from project settings in future
        )

        # Plugin state (empty for now, can load from DB later)
        state = PluginState()

        # Event data
        event_data = EventData(
            type=event.type,
            source=event.source,
            severity=event.severity.value,
            payload=event.payload,
            timestamp=event.timestamp,
        )

        return RuleContext(
            turn=turn,
            history=history,
            user=user,
            project=project,
            state=state,
            event=event_data,
        )

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

    def _detect_loop(self, tool_calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect if the agent is stuck in a repetitive loop.

        Tracks recent tool call patterns and detects when the same pattern
        is repeated multiple times, indicating the agent may be stuck.

        Args:
            tool_calls: List of tool calls from current turn

        Returns:
            Dict with pattern info if loop detected, None otherwise.
            Contains: pattern (str), count (int), suggestion (str)
        """
        if not tool_calls:
            return None

        # Create a pattern signature from tool names and key arguments
        pattern_parts = []
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "unknown")
            args_str = function.get("arguments", "{}")
            try:
                args = json.loads(args_str)
                # Include key identifying arguments in pattern
                key_args = []
                for key in ["path", "query", "thread_id", "file_path"]:
                    if key in args:
                        key_args.append(f"{key}={args[key][:50] if isinstance(args[key], str) else args[key]}")
                pattern_parts.append(f"{name}({','.join(key_args)})")
            except json.JSONDecodeError:
                pattern_parts.append(name)

        current_pattern = "|".join(sorted(pattern_parts))

        # Add to recent patterns
        self._recent_tool_patterns.append(current_pattern)

        # Keep only the window size
        if len(self._recent_tool_patterns) > self._loop_detection_window:
            self._recent_tool_patterns = self._recent_tool_patterns[-self._loop_detection_window:]

        # Count occurrences of current pattern in recent history
        pattern_count = self._recent_tool_patterns.count(current_pattern)

        if pattern_count >= self._loop_threshold and not self._loop_already_warned:
            self._loop_already_warned = True
            logger.warning(
                f"Loop detected: pattern '{current_pattern}' repeated {pattern_count}x"
            )

            # Generate human-readable pattern description
            tool_names = [call.get("function", {}).get("name", "unknown") for call in tool_calls]
            if len(tool_names) == 1:
                pattern_desc = f"Same tool '{tool_names[0]}' called repeatedly"
            else:
                pattern_desc = f"Same tool sequence ({', '.join(tool_names)}) repeated"

            return {
                "pattern": pattern_desc,
                "count": pattern_count,
                "suggestion": "Try a different approach or ask for clarification",
            }

        return None

    def _reset_loop_detection(self) -> None:
        """Reset loop detection state for new query."""
        self._recent_tool_patterns.clear()
        self._loop_already_warned = False

    def _should_emit_context_update(self, new_tokens: int) -> bool:
        """Check if we should emit a context update based on token change.

        Only emits updates when tokens change by at least 500 or 2% of max context,
        whichever is smaller. This prevents spamming updates on every chunk.

        Args:
            new_tokens: New total context token count

        Returns:
            True if we should emit an update
        """
        threshold = min(500, self._max_context_tokens // 50)
        return abs(new_tokens - self._last_context_update_tokens) >= threshold

    def _create_context_update_chunk(self) -> OracleStreamChunk:
        """Create a context update chunk with current token counts.

        Returns:
            OracleStreamChunk with context_update type
        """
        self._last_context_update_tokens = self._context_tokens
        return OracleStreamChunk(
            type="context_update",
            context_tokens=self._context_tokens,
            max_context_tokens=self._max_context_tokens,
        )

    def _reset_budget_tracking(self, max_tokens: int) -> None:
        """Reset budget tracking state for new query.

        Args:
            max_tokens: Maximum tokens budget for this query
        """
        self._iteration_warning_emitted = False
        self._iteration_exceeded_emitted = False
        self._token_warning_emitted = False
        self._token_exceeded_emitted = False
        self._total_tokens_used = 0
        self._max_tokens_budget = max_tokens
        self._context_limit_warning_emitted = False  # Reset context limit flag

    def _check_iteration_budget(self, turn: int) -> None:
        """Check iteration budget and emit warning event at 70% threshold.

        Args:
            turn: Current turn number (0-indexed)
        """
        warning_threshold_turn = int(self.MAX_TURNS * self.ITERATION_WARNING_THRESHOLD)

        # Check for iteration warning at 70% threshold
        if not self._iteration_warning_emitted and (turn + 1) >= warning_threshold_turn:
            self._iteration_warning_emitted = True
            percent = int(((turn + 1) / self.MAX_TURNS) * 100)
            logger.warning(f"Iteration budget warning: {turn + 1}/{self.MAX_TURNS} ({percent}%)")

            self._event_bus.emit(Event(
                type=EventType.BUDGET_ITERATION_WARNING,
                source="oracle_agent",
                severity=Severity.WARNING,
                payload={
                    "budget_type": "iteration",
                    "current": turn + 1,
                    "max": self.MAX_TURNS,
                    "percent": percent,
                    "message": f"Iteration {turn + 1} of {self.MAX_TURNS} - approaching limit",
                }
            ))

    def _check_token_budget(self, tokens_used: int) -> None:
        """Check token budget and emit warning event at 80% threshold.

        Args:
            tokens_used: Tokens used in last response
        """
        if self._max_tokens_budget <= 0:
            return

        self._total_tokens_used += tokens_used
        token_percent = self._total_tokens_used / self._max_tokens_budget

        # Check for token warning at 80% threshold
        if not self._token_warning_emitted and token_percent >= self.TOKEN_WARNING_THRESHOLD:
            self._token_warning_emitted = True
            percent = int(token_percent * 100)
            logger.warning(f"Token budget warning: {self._total_tokens_used}/{self._max_tokens_budget} ({percent}%)")

            self._event_bus.emit(Event(
                type=EventType.BUDGET_TOKEN_WARNING,
                source="oracle_agent",
                severity=Severity.WARNING,
                payload={
                    "budget_type": "token",
                    "current": self._total_tokens_used,
                    "max": self._max_tokens_budget,
                    "percent": percent,
                    "message": f"Used {self._total_tokens_used} of {self._max_tokens_budget} tokens",
                }
            ))

    def _emit_iteration_exceeded(self) -> None:
        """Emit iteration exceeded event when MAX_TURNS is reached."""
        if self._iteration_exceeded_emitted:
            return

        self._iteration_exceeded_emitted = True
        logger.error(f"Iteration budget exceeded: {self.MAX_TURNS}/{self.MAX_TURNS}")

        self._event_bus.emit(Event(
            type=EventType.BUDGET_ITERATION_EXCEEDED,
            source="oracle_agent",
            severity=Severity.ERROR,
            payload={
                "budget_type": "iteration",
                "current": self.MAX_TURNS,
                "max": self.MAX_TURNS,
                "percent": 100,
                "message": f"Maximum iterations ({self.MAX_TURNS}) reached - stopping",
            }
        ))

    def _emit_token_exceeded(self) -> None:
        """Emit token exceeded event when token budget is reached."""
        if self._token_exceeded_emitted or self._max_tokens_budget <= 0:
            return

        self._token_exceeded_emitted = True
        logger.error(f"Token budget exceeded: {self._total_tokens_used}/{self._max_tokens_budget}")

        self._event_bus.emit(Event(
            type=EventType.BUDGET_TOKEN_EXCEEDED,
            source="oracle_agent",
            severity=Severity.ERROR,
            payload={
                "budget_type": "token",
                "current": self._total_tokens_used,
                "max": self._max_tokens_budget,
                "percent": 100,
                "message": f"Token limit ({self._max_tokens_budget}) exceeded - stopping",
            }
        ))

    def _check_context_limit(self) -> None:
        """Check context window usage and emit warning at 70% threshold.

        This proactively notifies the agent when it's approaching the model's
        context window limit, allowing it to adjust strategy (e.g., summarize,
        compress history, or wrap up the task).

        Only emits once per query session to avoid notification spam.
        """
        if self._context_limit_warning_emitted:
            return

        if self._max_context_tokens <= 0:
            return

        usage_percent = self._context_tokens / self._max_context_tokens
        if usage_percent >= self.CONTEXT_WARNING_THRESHOLD:
            self._context_limit_warning_emitted = True
            percent = int(usage_percent * 100)
            remaining_tokens = self._max_context_tokens - self._context_tokens

            logger.warning(
                f"Context window approaching limit: {self._context_tokens}/{self._max_context_tokens} ({percent}%)"
            )

            self._event_bus.emit(Event(
                type=EventType.CONTEXT_APPROACHING_LIMIT,
                source="oracle_agent",
                severity=Severity.WARNING,
                payload={
                    "current_tokens": self._context_tokens,
                    "max_tokens": self._max_context_tokens,
                    "percent": percent,
                    "remaining_tokens": remaining_tokens,
                    "message": f"Context window at {percent}% ({remaining_tokens} tokens remaining)",
                }
            ))

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
        self._reset_loop_detection()
        self._reset_budget_tracking(max_tokens)
        reset_deferred_queue()  # Reset deferred delivery queue for new query
        self.user_id = user_id
        self._collected_sources = []
        self._collected_tool_calls = []
        self._collected_system_messages = []

        # Use provided project_id or fall back to init value
        effective_project_id = project_id or self.project_id or "default"
        # Update instance variable so tool injection uses correct project
        self.project_id = effective_project_id

        # Emit QUERY_START event (015-oracle-plugin-system Phase 4)
        self._event_bus.emit(Event(
            type=EventType.QUERY_START,
            source="oracle_agent",
            severity=Severity.INFO,
            payload={
                "query": question[:200],  # Truncate for event payload
                "user_id": user_id,
                "project_id": effective_project_id,
                "context_id": context_id,
            }
        ))

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

        # Load and inject cross-session notifications (014-ans-enhancements Feature 3)
        cross_session_notifications: List[str] = []
        try:
            pending_notifications = self._persistence_service.get_pending(
                user_id=user_id,
                project_id=effective_project_id,
                tree_id=self._current_tree_root_id,
            )

            for notification in pending_notifications:
                # Use formatted content if available, otherwise create basic format
                if notification.formatted_content:
                    content = notification.formatted_content
                else:
                    content = f"[{notification.severity.upper()}] {notification.event_type}: {json.dumps(notification.payload)}"

                cross_session_notifications.append(content)

                # Mark as delivered
                self._persistence_service.mark_delivered(notification.id)

            if cross_session_notifications:
                logger.info(
                    f"Loaded {len(cross_session_notifications)} cross-session notifications "
                    f"for user {user_id} project {effective_project_id}"
                )

        except Exception as e:
            logger.error(f"Failed to load cross-session notifications: {e}")

        # Get services
        tool_executor = _get_tool_executor()
        prompt_loader = _get_prompt_loader()

        # Fetch vault file list to include in system prompt
        vault_files: List[str] = []
        try:
            notes = tool_executor.vault.list_notes(user_id, project_id=effective_project_id)
            # Extract paths and limit to 100 files to avoid overwhelming the prompt
            vault_files = [note.get("path", "") for note in notes[:100]]
            logger.debug(f"Loaded {len(vault_files)} vault files for project {effective_project_id}")
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
                # Include system messages for each node (T041)
                system_messages_loaded = 0
                for node in path_nodes:
                    if node.is_root and not node.question and not node.answer:
                        continue
                    if node.question:
                        messages.append({"role": "user", "content": node.question})
                    # Add system messages (notifications) that occurred during this exchange
                    if hasattr(node, 'system_messages') and node.system_messages:
                        for sys_msg in node.system_messages:
                            messages.append({"role": "system", "content": sys_msg})
                            system_messages_loaded += 1
                    if node.answer:
                        messages.append({"role": "assistant", "content": node.answer})

                logger.debug(
                    f"Loaded {len(path_nodes)} nodes from tree, "
                    f"added {len(messages) - 1} context messages "
                    f"(including {system_messages_loaded} system messages)"
                )

                # Emit SESSION_RESUMED event if we loaded meaningful context (014-ans-enhancements)
                # Count nodes with actual Q&A content (not just empty root nodes)
                content_nodes = [n for n in path_nodes if n.question or n.answer]
                if content_nodes:
                    # Find the last question for context in the notification
                    last_question = None
                    for node in reversed(content_nodes):
                        if node.question:
                            last_question = node.question[:100]  # Truncate for notification
                            break

                    self._event_bus.emit(Event(
                        type=EventType.SESSION_RESUMED,
                        source="oracle_agent",
                        severity=Severity.INFO,
                        payload={
                            "tree_id": self._current_tree_root_id,
                            "nodes_loaded": len(content_nodes),
                            "last_question": last_question,
                            "message": f"Resumed session with {len(content_nodes)} previous exchanges",
                        }
                    ))

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

        # Inject cross-session notifications as system messages (014-ans-enhancements Feature 3)
        if cross_session_notifications:
            for notification_content in cross_session_notifications:
                messages.append({
                    "role": "system",
                    "content": f"<cross_session_notification>\n{notification_content}\n</cross_session_notification>",
                })
                # Also collect for persistence in the current exchange
                self._collected_system_messages.append(notification_content)

        # Add current question
        messages.append({"role": "user", "content": question})

        # Get tool definitions
        tools = tool_executor.get_tool_schemas(agent="oracle")

        # Initialize context window tracking
        self._max_context_tokens = get_model_context_size(self.model)
        self._context_tokens = estimate_messages_tokens(messages)
        # Add estimated tokens for tool definitions (~50 tokens per tool)
        self._context_tokens += len(tools) * 50
        self._last_context_update_tokens = 0

        logger.debug(
            f"Initial context: {self._context_tokens}/{self._max_context_tokens} tokens "
            f"({100 * self._context_tokens // self._max_context_tokens}%)"
        )

        # Yield initial context update
        yield OracleStreamChunk(
            type="context_update",
            context_tokens=self._context_tokens,
            max_context_tokens=self._max_context_tokens,
        )

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

        try:
            # Agent loop
            for turn in range(self.MAX_TURNS):
                # Check cancellation before each turn
                if self._cancelled:
                    logger.info(f"Agent cancelled at turn {turn + 1}")
                    yield OracleStreamChunk(type="error", error="Cancelled by user")
                    return

                # Check iteration budget and emit warning at 70% threshold (T047)
                self._check_iteration_budget(turn)

                # Drain and yield turn_start notifications (T049 - budget warnings)
                async for notification_chunk in self._drain_and_yield_turn_start_notifications(messages):
                    yield notification_chunk

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

                    # Check if we're done - mark as saved since _agent_turn handles it
                    if chunk.type == "done":
                        exchange_saved = True
                        return

                    # If we got an error, stop
                    if chunk.type == "error":
                        return

            # Max turns reached - emit iteration exceeded event (T048)
            self._emit_iteration_exceeded()

            # Drain and yield immediate notifications for exceeded events (T050)
            async for notification_chunk in self._drain_and_yield_immediate_notifications(messages):
                yield notification_chunk

            # Log and yield accumulated content before error
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
                exchange_saved = True
            else:
                context_id = None

            yield OracleStreamChunk(
                type="done",
                tokens_used=None,
                model_used=self.model,
                context_id=context_id,
            )

        finally:
            # Ensure we save something if the connection was dropped mid-response
            if not exchange_saved and (accumulated_content or accumulated_thinking or question):
                logger.info(f"Saving partial exchange due to early termination (content={len(accumulated_content)}, thinking={len(accumulated_thinking)})")
                try:
                    self._save_exchange(
                        question=question,
                        answer=accumulated_content or accumulated_thinking[:500] or "*Response interrupted*",
                    )
                except Exception as e:
                    logger.error(f"Failed to save partial exchange in finally block: {e}")

            # Emit SESSION_END event (015-oracle-plugin-system Phase 4)
            self._event_bus.emit(Event(
                type=EventType.SESSION_END,
                source="oracle_agent",
                severity=Severity.INFO,
                payload={
                    "user_id": user_id,
                    "project_id": effective_project_id,
                    "exchange_saved": exchange_saved,
                    "accumulated_content_length": len(accumulated_content),
                    "cancelled": self._cancelled,
                }
            ))

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
        # Apply thinking suffix if requested AND model supports it
        model = self.model
        if thinking and not model.endswith(":thinking"):
            # Only apply :thinking to models that actually support it
            # Based on OpenRouter's supported models:
            # - Models with -r1, /r1 (DeepSeek R1, etc.)
            # - Models with /o1, /o3 (OpenAI reasoning models)
            # - Claude 3.7 Sonnet (anthropic/claude-3.7-sonnet)
            # - Qwen thinking variants
            model_lower = model.lower()
            supports_thinking = (
                "-r1" in model_lower
                or "/r1" in model_lower
                or "/o1" in model_lower
                or "/o3" in model_lower
                or "claude-3.7-sonnet" in model_lower
                or "qwen" in model_lower and "thinking" in model_lower
            )
            if supports_thinking:
                model = f"{model}:thinking"
            elif thinking:
                # Log warning that thinking was requested but model doesn't support it
                logger.warning(
                    f"Thinking mode requested but model '{model}' does not support :thinking suffix. "
                    "Using model without thinking mode."
                )

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
        content_buffer = ""  # All content received
        reasoning_buffer = ""  # All reasoning/thinking received
        yielded_content_len = 0  # How much content we've already yielded to user
        yielded_reasoning_len = 0  # How much reasoning we've already yielded
        tool_call_detected = False  # True when we detect tool call syntax starting
        reasoning_tool_call_detected = False  # Tool calls found in reasoning stream
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
            # IMPORTANT: Some models output tool calls in reasoning instead of content!
            # We must buffer and check reasoning for tool call patterns too.
            reasoning_chunk = None
            if "reasoning" in delta and delta["reasoning"]:
                reasoning_chunk = delta["reasoning"]
            elif "reasoning_details" in delta and delta["reasoning_details"]:
                # Alternative format - combine all text
                for detail in delta["reasoning_details"]:
                    if isinstance(detail, dict) and detail.get("text"):
                        if reasoning_chunk is None:
                            reasoning_chunk = ""
                        reasoning_chunk += detail["text"]

            if reasoning_chunk:
                reasoning_buffer += reasoning_chunk

                # Check if reasoning contains tool call syntax
                if not reasoning_tool_call_detected:
                    if _is_tool_call_content(reasoning_buffer):
                        reasoning_tool_call_detected = True
                        logger.info(f"[STREAM] Detected tool call syntax in reasoning, suppressing output")
                        # Yield any safe reasoning before the tool call marker
                        safe_reasoning, _ = _extract_safe_content(reasoning_buffer)
                        if safe_reasoning and len(safe_reasoning) > yielded_reasoning_len:
                            remaining = safe_reasoning[yielded_reasoning_len:]
                            if remaining.strip():
                                yield OracleStreamChunk(
                                    type="thinking",
                                    content=remaining,
                                )
                            yielded_reasoning_len = len(safe_reasoning)
                    else:
                        # Safe to yield reasoning - but keep a buffer for safety
                        safe_threshold = 30  # Keep last 30 chars buffered for reasoning
                        if len(reasoning_buffer) > yielded_reasoning_len + safe_threshold:
                            to_yield = reasoning_buffer[yielded_reasoning_len:-safe_threshold]
                            if to_yield:
                                yield OracleStreamChunk(
                                    type="thinking",
                                    content=to_yield,
                                )
                                yielded_reasoning_len = len(reasoning_buffer) - safe_threshold

            # Handle content with tool call detection
            # Buffer content and only yield what's safe to show to users
            if "content" in delta and delta["content"]:
                content_buffer += delta["content"]

                # If we haven't detected tool call syntax yet, check now
                if not tool_call_detected:
                    # Check if the buffer looks like tool call syntax starting
                    if _is_tool_call_content(content_buffer):
                        tool_call_detected = True
                        logger.info(f"[STREAM] Detected tool call syntax in content, suppressing output")
                    else:
                        # Safe to yield - but be conservative about what we show
                        # Keep a buffer of last few chars in case tool call starts
                        safe_threshold = 20  # Keep last 20 chars buffered
                        if len(content_buffer) > yielded_content_len + safe_threshold:
                            # Extract safe content and any tool call markers
                            safe_content, tool_content = _extract_safe_content(content_buffer)

                            if tool_content:
                                # Found tool call markers, stop yielding
                                tool_call_detected = True
                                logger.info(f"[STREAM] Found tool call markers in buffer, suppressing")
                                # Yield any safe content before the tool call
                                remaining_safe = safe_content[yielded_content_len:]
                                if remaining_safe:
                                    yield OracleStreamChunk(
                                        type="content",
                                        content=remaining_safe,
                                    )
                                    yielded_content_len = len(safe_content)
                            else:
                                # No tool call markers, yield up to safe threshold
                                to_yield = content_buffer[yielded_content_len:-safe_threshold] if safe_threshold else content_buffer[yielded_content_len:]
                                if to_yield:
                                    yield OracleStreamChunk(
                                        type="content",
                                        content=to_yield,
                                    )
                                    yielded_content_len = len(content_buffer) - safe_threshold

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
        logger.info(f"[FINAL STATE] content_buffer_len={len(content_buffer)} yielded={yielded_content_len}")
        logger.info(f"[FINAL STATE] reasoning_buffer_len={len(reasoning_buffer)} yielded={yielded_reasoning_len}")
        logger.info(f"[FINAL STATE] tool_call_detected={tool_call_detected} reasoning_tool_call_detected={reasoning_tool_call_detected}")
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

        elif finish_reason == "stop" or content_buffer or reasoning_buffer:
            # Check if the model output XML-style tool calls in content OR reasoning
            # (Some models like DeepSeek output tool calls in reasoning stream!)
            xml_tool_calls, cleaned_content = _parse_xml_tool_calls(content_buffer)

            # If no tool calls in content, check reasoning buffer
            if not xml_tool_calls and reasoning_buffer:
                logger.info(f"[STREAM] No tool calls in content, checking reasoning buffer...")
                xml_tool_calls, cleaned_reasoning = _parse_xml_tool_calls(reasoning_buffer)
                if xml_tool_calls:
                    logger.warning(
                        f"Model {self.model} output {len(xml_tool_calls)} tool call(s) "
                        "in REASONING stream instead of content. Extracting and executing."
                    )
                    # Yield any safe reasoning that wasn't shown
                    if cleaned_reasoning and len(cleaned_reasoning) > yielded_reasoning_len:
                        remaining = cleaned_reasoning[yielded_reasoning_len:]
                        if remaining.strip():
                            yield OracleStreamChunk(
                                type="thinking",
                                content=remaining,
                            )

            if xml_tool_calls:
                # Model used XML-style tool calls instead of proper function calling
                logger.warning(
                    f"Model {self.model} output {len(xml_tool_calls)} XML-style tool call(s) "
                    "instead of using proper function calling. Parsing and executing."
                )

                # Yield any remaining safe content that wasn't yielded during streaming
                # The cleaned_content is what's left after removing tool call syntax
                if cleaned_content and len(cleaned_content) > yielded_content_len:
                    remaining = cleaned_content[yielded_content_len:]
                    if remaining.strip():
                        yield OracleStreamChunk(
                            type="content",
                            content=remaining,
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
                # Yield any remaining buffered content that wasn't yielded during streaming
                if len(content_buffer) > yielded_content_len:
                    remaining = content_buffer[yielded_content_len:]
                    if remaining.strip():
                        yield OracleStreamChunk(
                            type="content",
                            content=remaining,
                        )
                messages.append({"role": "assistant", "content": content_buffer})

                # Update context token count with assistant response
                self._context_tokens += estimate_tokens(content_buffer)
                self._check_context_limit()  # Proactive context warning (014-ans-enhancements)

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

                # Emit final context update before done
                yield OracleStreamChunk(
                    type="context_update",
                    context_tokens=self._context_tokens,
                    max_context_tokens=self._max_context_tokens,
                )

                # Done with context_id for frontend reference
                yield OracleStreamChunk(
                    type="done",
                    tokens_used=None,  # Could extract from response headers
                    model_used=self.model,
                    context_id=context_id,
                    context_tokens=self._context_tokens,
                    max_context_tokens=self._max_context_tokens,
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

                # Update context token count with assistant response
                self._context_tokens += estimate_tokens(content)
                self._check_context_limit()  # Proactive context warning (014-ans-enhancements)

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

                # Emit final context update before done
                yield OracleStreamChunk(
                    type="context_update",
                    context_tokens=self._context_tokens,
                    max_context_tokens=self._max_context_tokens,
                )

                # Done - only when no XML tool calls were found
                yield OracleStreamChunk(
                    type="done",
                    tokens_used=usage.get("total_tokens"),
                    model_used=self.model,
                    context_id=context_id,
                    context_tokens=self._context_tokens,
                    max_context_tokens=self._max_context_tokens,
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

        # Loop detection check (T053, T054)
        loop_info = self._detect_loop(tool_calls)
        if loop_info:
            # Emit agent.loop.detected event
            self._event_bus.emit(Event(
                type=EventType.AGENT_LOOP_DETECTED,
                source="oracle_agent",
                severity=Severity.WARNING,
                payload={
                    "pattern": loop_info["pattern"],
                    "count": loop_info["count"],
                    "suggestion": loop_info["suggestion"],
                }
            ))

            # Yield system notification about the loop
            loop_content = f"!loop: {loop_info['pattern']} repeated {loop_info['count']}x - {loop_info['suggestion']}"
            # Collect for persistence (T040)
            self._collected_system_messages.append(loop_content)
            # CRITICAL FIX: Inject into messages so LLM sees the notification
            messages.append({
                "role": "system",
                "content": f"<system_notification>\n{loop_content}\n</system_notification>",
            })
            yield OracleStreamChunk(
                type="system",
                content=loop_content,
            )

        tool_executor = _get_tool_executor()

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

        # Execute all tools in parallel
        async def execute_single_tool(
            call_id: str,
            name: str,
            arguments: Dict[str, Any],
        ) -> ToolExecutionResult:
            """Execute a single tool and return structured result."""
            try:
                # Inject project_id for tools that need project context
                if name in PROJECT_SCOPED_TOOLS and "project_id" not in arguments:
                    arguments = {**arguments, "project_id": self.project_id}
                    logger.info(f"[PROJECT_SCOPE] Injected project_id={self.project_id} for tool {name}")
                elif name in PROJECT_SCOPED_TOOLS:
                    logger.info(f"[PROJECT_SCOPE] Tool {name} already has project_id={arguments.get('project_id')}")

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
        # Skip tools with JSON parse errors - they'll be handled as immediate failures
        tasks = []
        failed_parse_results = []

        for call_id, name, arguments, _ in parsed_calls:
            # Check if this tool call had a JSON parsing error
            if "_json_parse_error" in arguments:
                # Create immediate failure result for JSON parse errors
                failed_parse_results.append(ToolExecutionResult(
                    call_id=call_id,
                    name=name,
                    arguments={},
                    error=arguments["_json_parse_error"],
                    success=False,
                ))
            else:
                # Normal tool call - add to execution queue
                tasks.append(execute_single_tool(call_id, name, arguments))

        # Execute all tools concurrently, continue even if some fail
        executed_results: List[ToolExecutionResult] = await asyncio.gather(
            *tasks, return_exceptions=True
        ) if tasks else []

        # Merge failed parse results with executed results (preserving order)
        # This ensures JSON parse errors are reported in the correct position
        results: List[ToolExecutionResult] = []
        exec_idx = 0
        for call_id, name, arguments, _ in parsed_calls:
            if "_json_parse_error" in arguments:
                # Find the corresponding failed parse result
                for fail_result in failed_parse_results:
                    if fail_result.call_id == call_id:
                        results.append(fail_result)
                        break
            else:
                # Use the next executed result
                results.append(executed_results[exec_idx])
                exec_idx += 1

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

                # Emit tool.call.success event (T029)
                self._event_bus.emit(Event(
                    type=EventType.TOOL_CALL_SUCCESS,
                    source="oracle_agent",
                    severity=Severity.INFO,
                    payload={
                        "tool_name": result.name,
                        "call_id": result.call_id,
                        "result_length": len(result.result),
                    }
                ))

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

                # Update context tokens and emit update if significant change
                self._context_tokens += estimate_tokens(result.result) + 10  # +10 for message overhead
                self._check_context_limit()  # Proactive context warning (014-ans-enhancements)
                if self._should_emit_context_update(self._context_tokens):
                    yield self._create_context_update_chunk()
            else:
                # Error case - T027: provide error but let agent continue
                error_content = self._format_tool_error(
                    result.name,
                    result.error or "Unknown error",
                    result.arguments,
                )

                # Emit tool.call.failure event (T027)
                self._event_bus.emit(Event(
                    type=EventType.TOOL_CALL_FAILURE,
                    source="oracle_agent",
                    severity=Severity.ERROR,
                    payload={
                        "tool_name": result.name,
                        "error_type": "execution_error",
                        "error_message": result.error or "Unknown error",
                        "call_id": result.call_id,
                    }
                ))

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

                # Update context tokens for error result
                self._context_tokens += estimate_tokens(error_content) + 10
                self._check_context_limit()  # Proactive context warning (014-ans-enhancements)

        # Drain notifications after tool execution (T030-T033)
        # Get list of tool names that were executed for deferred delivery
        executed_tool_names = [name for _, name, _, _ in parsed_calls]
        async for notification_chunk in self._drain_and_yield_notifications(messages, executed_tool_names):
            yield notification_chunk

    async def _drain_and_yield_notifications(
        self,
        messages: List[Dict[str, Any]],
        tool_names: Optional[List[str]] = None,
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """Drain notifications from the accumulator and yield as system chunks.

        This processes events that have been accumulated during tool execution
        and formats them into TOON-formatted system messages for the agent.

        Args:
            messages: Conversation messages list (mutated to add system notifications
                     so the LLM can see them in the next turn).
            tool_names: List of tool names that were just executed (for deferred delivery)

        Yields:
            OracleStreamChunk objects with type="system" containing formatted notifications.
        """
        # Drain pending events ONCE before iterating subscribers
        # (draining inside the loop would empty the list on first iteration)
        pending = self._event_bus.drain_pending()

        # Process events through subscribers to create notifications
        for subscriber in self._subscriber_loader.get_all_subscribers():
            if not subscriber.enabled:
                continue

            # Check each pending event against this subscriber
            for event in pending:
                if subscriber.matches_event(event.type):
                    notification = self._accumulator.accumulate(event, subscriber)
                    if notification:
                        # Format immediately if returned (critical priority)
                        template_name = subscriber.config.template
                        notification.content = self._toon_formatter.format_notification(
                            notification, template_name
                        )

        # Drain after_tool notifications
        notifications = self._accumulator.drain_after_tool()

        # Also drain deferred after_tool notifications for each executed tool
        if tool_names:
            for tool_name in tool_names:
                deferred_notifications = self._accumulator.drain_deferred_after_tool(tool_name)
                notifications.extend(deferred_notifications)

        for notification in notifications:
            # Format content if not already done
            if not notification.content:
                subscriber = self._subscriber_loader.get_subscriber(notification.subscriber_id)
                if subscriber:
                    template_name = subscriber.config.template
                    notification.content = self._toon_formatter.format_notification(
                        notification, template_name
                    )

            if notification.content:
                # Collect for persistence (T040)
                self._collected_system_messages.append(notification.content)
                # CRITICAL FIX: Inject into messages so LLM sees the notification
                messages.append({
                    "role": "system",
                    "content": f"<system_notification>\n{notification.content}\n</system_notification>",
                })
                yield OracleStreamChunk(
                    type="system",
                    content=notification.content,
                )

    async def _drain_and_yield_turn_start_notifications(
        self,
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """Drain turn_start notifications and yield as system chunks.

        This is called at the start of each agent turn to inject budget
        warnings and other turn_start priority notifications (T049).

        Args:
            messages: Conversation messages list (mutated to add system notifications
                     so the LLM can see them in the next turn).

        Yields:
            OracleStreamChunk objects with type="system" containing formatted notifications.
        """
        # Drain pending events ONCE before iterating subscribers
        # (draining inside the loop would empty the list on first iteration)
        pending = self._event_bus.drain_pending()

        # Process events through subscribers to create notifications
        for subscriber in self._subscriber_loader.get_all_subscribers():
            if not subscriber.enabled:
                continue

            # Check each pending event against this subscriber
            for event in pending:
                if subscriber.matches_event(event.type):
                    notification = self._accumulator.accumulate(event, subscriber)
                    if notification:
                        # Format immediately if returned (critical priority)
                        template_name = subscriber.config.template
                        notification.content = self._toon_formatter.format_notification(
                            notification, template_name
                        )

        # Drain turn_start notifications
        notifications = self._accumulator.drain_turn_start()

        # Also drain deferred turn_start notifications
        deferred_notifications = self._accumulator.drain_deferred_turn_start()
        notifications.extend(deferred_notifications)

        for notification in notifications:
            # Format content if not already done
            if not notification.content:
                subscriber = self._subscriber_loader.get_subscriber(notification.subscriber_id)
                if subscriber:
                    template_name = subscriber.config.template
                    notification.content = self._toon_formatter.format_notification(
                        notification, template_name
                    )

            if notification.content:
                # Collect for persistence
                self._collected_system_messages.append(notification.content)
                # CRITICAL FIX: Inject into messages so LLM sees the notification
                messages.append({
                    "role": "system",
                    "content": f"<system_notification>\n{notification.content}\n</system_notification>",
                })
                yield OracleStreamChunk(
                    type="system",
                    content=notification.content,
                )

    async def _drain_and_yield_immediate_notifications(
        self,
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """Drain immediate (critical) notifications and yield as system chunks.

        This is called when critical events occur (like budget exceeded)
        to immediately inject notifications (T050).

        Args:
            messages: Conversation messages list (mutated to add system notifications
                     so the LLM can see them in the next turn).

        Yields:
            OracleStreamChunk objects with type="system" containing formatted notifications.
        """
        # Drain pending events ONCE before iterating subscribers
        # (draining inside the loop would empty the list on first iteration)
        pending = self._event_bus.drain_pending()

        # Process events through subscribers to create notifications
        for subscriber in self._subscriber_loader.get_all_subscribers():
            if not subscriber.enabled:
                continue

            # Check each pending event against this subscriber
            for event in pending:
                if subscriber.matches_event(event.type):
                    notification = self._accumulator.accumulate(event, subscriber)
                    if notification:
                        # Format immediately if returned (critical priority)
                        template_name = subscriber.config.template
                        notification.content = self._toon_formatter.format_notification(
                            notification, template_name
                        )

        # Drain immediate notifications (critical priority)
        notifications = self._accumulator.drain_immediate()
        for notification in notifications:
            # Format content if not already done
            if not notification.content:
                subscriber = self._subscriber_loader.get_subscriber(notification.subscriber_id)
                if subscriber:
                    template_name = subscriber.config.template
                    notification.content = self._toon_formatter.format_notification(
                        notification, template_name
                    )

            if notification.content:
                # Collect for persistence
                self._collected_system_messages.append(notification.content)
                # CRITICAL FIX: Inject into messages so LLM sees the notification
                messages.append({
                    "role": "system",
                    "content": f"<system_notification>\n{notification.content}\n</system_notification>",
                })
                yield OracleStreamChunk(
                    type="system",
                    content=notification.content,
                )

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
                # Include system messages collected during this exchange (T040)
                new_node = self._tree_service.create_node(
                    user_id=self.user_id,
                    root_id=self._current_tree_root_id,
                    parent_id=self._current_node_id,
                    question=question,
                    answer=answer,
                    tool_calls=self._collected_tool_calls if self._collected_tool_calls else None,
                    tokens_used=tokens_used or self._estimate_tokens(question + answer),
                    model_used=self.model,
                    system_messages=self._collected_system_messages if self._collected_system_messages else None,
                )

                # Update current node to the new one (new HEAD)
                self._current_node_id = new_node.id
                result_id = new_node.id

                logger.info(
                    f"Saved exchange to tree node {new_node.id} "
                    f"(tree: {self._current_tree_root_id}, system_messages: {len(self._collected_system_messages)})"
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
