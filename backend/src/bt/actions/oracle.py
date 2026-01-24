"""
Oracle Actions - BT action functions for the Oracle agent.

These functions are called by the oracle-agent.lua tree definition.
Each function receives a TickContext and returns a RunStatus.

Migration from: backend/src/services/oracle_agent.py
Target: All action functions needed by oracle-agent.lua tree

Part of the BT Universal Runtime (spec 019).
Tasks covered: 5.1.3, 5.1.4 from tasks.md
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..state.base import RunStatus

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard

logger = logging.getLogger(__name__)


# =============================================================================
# Blackboard Helpers
# =============================================================================

# The TypedBlackboard requires schemas for .get() and .set() operations.
# For oracle actions, we use direct _data access for simple values.


def bb_get(bb: "TypedBlackboard", key: str, default: Any = None) -> Any:
    """Get value from blackboard without schema validation.

    Uses _lookup which traverses scope chain.
    """
    value = bb._lookup(key)
    return value if value is not None else default


def bb_set(bb: "TypedBlackboard", key: str, value: Any) -> None:
    """Set value in blackboard without schema validation.

    For internal oracle state that doesn't need Pydantic validation.
    Now uses _store() method to ensure proper scope propagation.
    """
    # Use _store() instead of direct _data access to enable scope propagation
    if hasattr(bb, '_store'):
        bb._store(key, value)
    else:
        # Fallback for TypedBlackboard without _store
        bb._data[key] = value
        bb._writes.add(key)


# =============================================================================
# Constants from oracle_agent.py
# =============================================================================

MAX_TURNS = 30
ITERATION_WARNING_THRESHOLD = 0.70
TOKEN_WARNING_THRESHOLD = 0.80
CONTEXT_WARNING_THRESHOLD = 0.70

# Default model context sizes
DEFAULT_MODEL_CONTEXT_SIZES = {
    "deepseek/deepseek-chat": 64000,
    "deepseek/deepseek-r1": 64000,
    "anthropic/claude-3-opus": 200000,
    "anthropic/claude-sonnet-4": 200000,
    "gemini-2.0-flash-exp": 1000000,
    "openai/gpt-4-turbo": 128000,
    "openai/o1": 200000,
    "meta-llama/llama-3.3-70b": 131072,
}
DEFAULT_CONTEXT_SIZE = 128000

# Tools that need project_id injected
PROJECT_SCOPED_TOOLS = {
    "search_code",
    "get_repo_map",
    "search_symbol",
    "get_coderag_status",
}


# =============================================================================
# Phase 1: Initialization Actions
# =============================================================================


def reset_state(ctx: "TickContext") -> RunStatus:
    """Reset cancellation, loop detection, budget tracking, and deferred queue.

    Sets up initial blackboard state for a new query.
    Corresponds to oracle_agent.py lines 924-932.
    """
    bb = ctx.blackboard
    if bb is None:
        logger.error("reset_state: No blackboard available")
        return RunStatus.FAILURE

    logger.debug("reset_state: Initializing blackboard state for new query")

    # Reset cancellation flag
    bb_set(bb,"cancelled", False)

    # Reset loop detection - CRITICAL: Must be reset at start of each query
    bb_set(bb,"recent_tool_patterns", [])
    bb_set(bb,"loop_detected", False)
    bb_set(bb,"loop_already_warned", False)
    bb_set(bb,"loop_warning", "")
    logger.debug("reset_state: Loop detection state reset (recent_tool_patterns=[], loop_detected=False)")

    # Reset budget tracking
    bb_set(bb,"turn", 0)
    bb_set(bb,"tokens_used", 0)
    bb_set(bb,"iteration_warning_emitted", False)
    bb_set(bb,"iteration_exceeded_emitted", False)
    bb_set(bb,"token_warning_emitted", False)
    bb_set(bb,"token_exceeded_emitted", False)
    bb_set(bb,"context_tokens", 0)

    # Reset output state
    bb_set(bb,"accumulated_content", "")
    bb_set(bb,"reasoning_content", "")
    bb_set(bb,"partial_response", "")
    bb_set(bb,"collected_sources", [])
    bb_set(bb,"system_messages", [])
    bb_set(bb,"tool_calls", [])
    bb_set(bb,"tool_results", [])
    bb_set(bb,"pending_notifications", [])

    # Initialize messages list
    bb_set(bb,"messages", [])

    # Reset query classification (US1 - Intelligent Context Selection)
    bb_set(bb, "query_classification", None)
    bb_set(bb, "needs_code", False)
    bb_set(bb, "needs_vault", False)
    bb_set(bb, "needs_web", False)

    # ==========================================================================
    # Signal state tracking (T023 - US2: Agent Self-Reflection via Signals)
    # Per data-model.md AgentSignalState entity
    # ==========================================================================
    bb_set(bb, "last_signal", None)              # Most recent parsed signal
    bb_set(bb, "signals_emitted", [])            # All signals this session
    bb_set(bb, "consecutive_same_reason", 0)     # Loop detection counter
    bb_set(bb, "turns_without_signal", 0)        # Fallback trigger counter
    bb_set(bb, "_signal_parsed_this_turn", False)  # Per-turn parse flag
    bb_set(bb, "_prev_signal", None)             # For consecutive comparison

    # ==========================================================================
    # Additional state keys (found during audit)
    # ==========================================================================
    bb_set(bb, "_exchange_saved", False)          # Track if exchange was saved
    bb_set(bb, "thinking_enabled", False)         # LLM thinking mode (extended thinking)
    bb_set(bb, "_pending_chunks", [])             # SSE chunks pending delivery
    bb_set(bb, "_last_streamed_len", 0)           # For incremental streaming

    logger.debug("Oracle state reset for new query (with signal tracking)")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def emit_query_start(ctx: "TickContext") -> RunStatus:
    """Emit QUERY_START event for plugin system.

    Corresponds to oracle_agent.py lines 939-950.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    # Get query info from blackboard
    user_id = bb_get(bb,"user_id")
    project_id = bb_get(bb,"project_id")
    query = bb_get(bb,"query")
    model = bb_get(bb,"model")

    # Emit event to ANS event bus
    try:
        from src.services.ans.bus import get_event_bus
        from src.services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="oracle.query.start",
            source="oracle_bt",
            severity=Severity.INFO,
            payload={
                "user_id": user_id,
                "project_id": project_id,
                "query_preview": str(query)[:100] if query else "",
                "model": model,
            }
        ))
    except Exception as e:
        logger.warning(f"Failed to emit QUERY_START event: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Phase 2: Context Loading Actions
# =============================================================================


def load_tree_node(ctx: "TickContext") -> RunStatus:
    """Load existing tree node from context_id.

    Corresponds to oracle_agent.py lines 957-1001.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    context_id = bb_get(bb, "context_id")
    user_id = bb_get(bb, "user_id")
    if not context_id:
        return RunStatus.FAILURE

    if not user_id:
        logger.error("load_tree_node: user_id required")
        return RunStatus.FAILURE

    try:
        from src.services.context_tree_service import ContextTreeService

        tree_service = ContextTreeService()
        # get_node requires (user_id, node_id) signature
        node = tree_service.get_node(user_id, context_id)

        if node is None:
            logger.warning(f"Context node not found: {context_id}")
            return RunStatus.FAILURE

        # Set tree context in blackboard (ContextNode uses root_id, not tree_id)
        bb_set(bb, "tree_root_id", node.root_id)
        bb_set(bb, "current_node_id", context_id)

        logger.debug(f"Loaded tree context: root={node.root_id}, node={context_id}")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    except Exception as e:
        logger.error(f"Failed to load tree node: {e}")
        return RunStatus.FAILURE


def get_or_create_tree(ctx: "TickContext") -> RunStatus:
    """Get active tree or create new one for user/project.

    Corresponds to oracle_agent.py lines 957-1001 (fallback path).
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    user_id = bb_get(bb, "user_id")
    project_id = bb_get(bb, "project_id") or "default"

    if not user_id:
        logger.error("get_or_create_tree: user_id required")
        return RunStatus.FAILURE

    try:
        from src.services.context_tree_service import ContextTreeService

        tree_service = ContextTreeService()

        # Try to get existing active tree ID
        active_tree_id = tree_service.get_active_tree_id(user_id, project_id)

        if active_tree_id:
            # Get the full tree object
            tree = tree_service.get_tree(user_id, active_tree_id)
            if tree:
                logger.debug(f"Using existing tree: {tree.root_id}")
            else:
                # Tree ID exists but tree not found - create new
                tree = tree_service.create_tree(user_id, project_id)
                logger.debug(f"Created new tree (previous not found): {tree.root_id}")
        else:
            # Create new tree
            tree = tree_service.create_tree(user_id, project_id)
            logger.debug(f"Created new tree: {tree.root_id}")

        # Set tree context in blackboard (ContextTree uses root_id and current_node_id)
        bb_set(bb, "tree_root_id", tree.root_id)
        bb_set(bb, "current_node_id", tree.current_node_id)

        ctx.mark_progress()
        return RunStatus.SUCCESS

    except Exception as e:
        logger.error(f"Failed to get/create tree: {e}")
        # Non-critical - can continue without tree context
        return RunStatus.SUCCESS


def load_legacy_context(ctx: "TickContext") -> RunStatus:
    """Load OracleContextService context (fallback).

    Corresponds to oracle_agent.py lines 1002-1011.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    user_id = bb_get(bb,"user_id")
    project_id = bb_get(bb,"project_id")

    try:
        from src.services.oracle_context_service import OracleContextService

        context_service = OracleContextService()
        context = context_service.get_or_create_context(user_id, project_id or "default")

        bb_set(bb,"context", context)
        logger.debug("Loaded legacy context")

    except Exception as e:
        logger.warning(f"Failed to load legacy context: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def load_cross_session_notifications(ctx: "TickContext") -> RunStatus:
    """Load ANS notifications persisted across sessions.

    Corresponds to oracle_agent.py lines 1013-1042.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    user_id = bb_get(bb,"user_id")

    try:
        from src.services.ans.persistence import get_notification_persistence

        persistence = get_notification_persistence()
        notifications = persistence.get_pending(user_id)

        if notifications:
            bb_set(bb,"pending_notifications", notifications)
            logger.debug(f"Loaded {len(notifications)} cross-session notifications")

            # Mark as delivered
            for notification in notifications:
                persistence.mark_delivered(notification.id)

    except Exception as e:
        logger.warning(f"Failed to load cross-session notifications: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Phase 3: Message Building Actions
# =============================================================================


def build_system_prompt(ctx: "TickContext") -> RunStatus:
    """Load vault files/threads, render system prompt via prompt_composer.

    US1 Enhancement: Uses query classification for dynamic prompt composition.
    The prompt_composer selects segments based on query_type:
    - Always includes: base.md, signals.md, tools-reference.md
    - Conditionally includes: code-analysis.md, documentation.md, research.md

    Corresponds to oracle_agent.py lines 1043-1183.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    messages = bb_get(bb, "messages") or []
    user_id = bb_get(bb, "user_id")
    project_id = bb_get(bb, "project_id")
    classification = bb_get(bb, "query_classification")

    # Build system prompt using composer (US1 - dynamic composition)
    try:
        from src.services.prompt_composer import PromptComposer
        from src.models.query_classification import QueryClassification, QueryType

        # Extract query_type from classification, defaulting to "conversational"
        query_type_str = "conversational"
        if classification and isinstance(classification, dict):
            query_type_str = classification.get("query_type", "conversational")

        # Convert to QueryType enum
        try:
            query_type = QueryType(query_type_str)
        except ValueError:
            query_type = QueryType.CONVERSATIONAL

        # Build classification object
        classification_obj = QueryClassification.from_type(
            query_type,
            confidence=classification.get("confidence", 1.0) if classification else 1.0,
            keywords_matched=classification.get("keywords_matched", []) if classification else [],
        )

        # Compose prompt
        composer = PromptComposer()
        context = {
            "user_id": str(user_id or "unknown"),
            "project_id": str(project_id or "default"),
        }
        result = composer.compose(classification_obj, context=context)
        system_content = result.content

        logger.debug(
            f"Prompt composed: segments={result.segments_included}, "
            f"tokens~={result.token_estimate}, warnings={result.warnings}"
        )

    except ImportError as e:
        # Fallback to legacy prompt building if composer unavailable
        logger.warning(f"prompt_composer not available, using legacy: {e}")
        system_content = _build_system_content_legacy(user_id, project_id)
    except Exception as e:
        logger.warning(f"Prompt composition failed, using legacy: {e}")
        system_content = _build_system_content_legacy(user_id, project_id)

    # Add system message
    messages.insert(0, {
        "role": "system",
        "content": system_content
    })

    bb_set(bb, "messages", messages)
    ctx.mark_progress()
    return RunStatus.SUCCESS


def _build_system_content_legacy(user_id: Optional[str], project_id: Optional[str]) -> str:
    """Build system prompt content from template (legacy fallback).

    Used when prompt_composer is unavailable.
    """
    # Try to load template
    template_path = Path(__file__).parent.parent.parent / "services" / "prompts" / "oracle" / "system.md"

    if template_path.exists():
        try:
            template = template_path.read_text()
            # Basic template substitution
            template = template.replace("{{user_id}}", str(user_id or "unknown"))
            template = template.replace("{{project_id}}", str(project_id or "default"))
            return template
        except Exception as e:
            logger.warning(f"Failed to load system template: {e}")

    # Fallback default prompt
    return """You are Oracle, an AI assistant for software development.
You have access to tools for searching code, documentation, and memory.
Use tools when needed to answer questions accurately.
Be concise and helpful."""


def add_tree_history(ctx: "TickContext") -> RunStatus:
    """Walk root to current node, add exchanges to messages.

    Corresponds to oracle_agent.py lines 1094-1127.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    messages = bb_get(bb, "messages") or []
    user_id = bb_get(bb, "user_id")
    tree_root_id = bb_get(bb, "tree_root_id")
    current_node_id = bb_get(bb, "current_node_id")

    if not tree_root_id or not current_node_id:
        logger.debug("add_tree_history: No tree context, skipping history load")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    if not user_id:
        logger.warning("add_tree_history: user_id required for tree history")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    try:
        from src.services.context_tree_service import ContextTreeService

        tree_service = ContextTreeService()
        # get_nodes requires (user_id, root_id) signature
        nodes = tree_service.get_nodes(user_id, tree_root_id)

        if not nodes:
            logger.debug(f"add_tree_history: No nodes found for tree {tree_root_id}")
            ctx.mark_progress()
            return RunStatus.SUCCESS

        # Build path from root to current
        node_map = {n.id: n for n in nodes}
        path_nodes = []
        current_id = current_node_id

        while current_id and current_id in node_map:
            path_nodes.insert(0, node_map[current_id])
            current_id = node_map[current_id].parent_id

        # Add exchanges from path to messages (skip root node which has empty Q&A)
        for node in path_nodes:
            # Skip root/empty nodes
            if not node.question and not node.answer:
                continue
            if node.question:
                messages.append({
                    "role": "user",
                    "content": node.question
                })
            if node.answer:
                messages.append({
                    "role": "assistant",
                    "content": node.answer
                })

        bb_set(bb, "messages", messages)
        bb_set(bb, "message_history", path_nodes)
        logger.debug(f"add_tree_history: Added {len(path_nodes)} exchanges from tree history")

    except Exception as e:
        logger.warning(f"Failed to add tree history: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def inject_notifications(ctx: "TickContext") -> RunStatus:
    """Inject cross-session notifications as system messages.

    Corresponds to oracle_agent.py notification injection.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    messages = bb_get(bb,"messages") or []
    pending = bb_get(bb,"pending_notifications") or []

    for notification in pending:
        content = notification.get("content") or str(notification)
        messages.append({
            "role": "system",
            "content": f"[System Notification] {content}"
        })

    if pending:
        bb_set(bb,"messages", messages)
        bb_set(bb,"pending_notifications", [])  # Clear after injection
        logger.debug(f"Injected {len(pending)} notifications")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def add_user_question(ctx: "TickContext") -> RunStatus:
    """Add current user question to messages.

    Corresponds to oracle_agent.py user message addition.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    messages = bb_get(bb,"messages") or []
    query = bb_get(bb,"query")

    if query:
        # Handle different query formats
        if isinstance(query, str):
            question = query
        elif hasattr(query, "question"):
            question = query.question
        elif isinstance(query, dict):
            question = query.get("question", str(query))
        else:
            question = str(query)

        messages.append({
            "role": "user",
            "content": question
        })
        bb_set(bb,"messages", messages)
        logger.debug(f"Added user question: {question[:50]}...")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def get_tool_schemas(ctx: "TickContext") -> RunStatus:
    """Get tool definitions from tool_executor.

    Corresponds to oracle_agent.py line 1185.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    try:
        from src.services.tool_executor import ToolExecutor

        executor = ToolExecutor()
        schemas = executor.get_tool_schemas(agent="oracle")
        bb_set(bb,"tools", schemas)
        logger.debug(f"Loaded {len(schemas)} tool schemas")

    except Exception as e:
        logger.warning(f"Failed to get tool schemas: {e}")
        bb_set(bb,"tools", [])

    ctx.mark_progress()
    return RunStatus.SUCCESS


def init_context_tracking(ctx: "TickContext") -> RunStatus:
    """Set max_context_tokens from model, estimate initial tokens.

    Corresponds to oracle_agent.py lines 1187-1204.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    model = bb_get(bb,"model") or "default"
    max_tokens = bb_get(bb,"max_tokens") or 4096

    # Get context size for model
    max_context = DEFAULT_MODEL_CONTEXT_SIZES.get(model, DEFAULT_CONTEXT_SIZE)
    bb_set(bb,"max_context_tokens", max_context)

    # Estimate initial context usage
    messages = bb_get(bb,"messages") or []
    estimated_tokens = _estimate_tokens(messages)
    bb_set(bb,"context_tokens", estimated_tokens)

    logger.debug(f"Context tracking: {estimated_tokens}/{max_context} tokens")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def _estimate_tokens(messages: List[Dict[str, str]]) -> int:
    """Estimate token count for messages.

    Handles messages where content may be None (e.g., assistant messages
    with tool_calls but no text content).
    """
    total = 0
    for msg in messages:
        content = msg.get("content")
        # Handle None content (e.g., tool_call messages have content: null)
        if content is None:
            content = ""
        # Rough estimate: 4 chars per token
        total += len(content) // 4
    return total


def yield_context_update(ctx: "TickContext") -> RunStatus:
    """Yield initial context_update chunk to frontend.

    Corresponds to oracle_agent.py context update emission.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    context_tokens = bb_get(bb,"context_tokens") or 0
    max_context = bb_get(bb,"max_context_tokens") or DEFAULT_CONTEXT_SIZE

    # Store chunk for streaming bridge to pick up
    chunk = {
        "type": "context_update",
        "context_tokens": context_tokens,
        "max_context_tokens": max_context,
        "percentage": (context_tokens / max_context * 100) if max_context > 0 else 0
    }

    _add_pending_chunk(bb, chunk)
    ctx.mark_progress()
    return RunStatus.SUCCESS


def _add_pending_chunk(bb: Any, chunk: Dict[str, Any]) -> None:
    """Add a chunk to the pending chunks list for streaming."""
    chunks = bb_get(bb,"_pending_chunks") or []
    chunks.append(chunk)
    bb_set(bb,"_pending_chunks", chunks)


# =============================================================================
# Phase 4: Agent Loop Actions
# =============================================================================


def check_iteration_budget(ctx: "TickContext") -> RunStatus:
    """Emit warning at 70% of max turns.

    Corresponds to oracle_agent.py lines 762-787.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    turn = bb_get(bb,"turn") or 0
    warning_emitted = bb_get(bb,"iteration_warning_emitted") or False

    # Check warning threshold
    threshold_turn = int(MAX_TURNS * ITERATION_WARNING_THRESHOLD)
    if turn >= threshold_turn and not warning_emitted:
        bb_set(bb,"iteration_warning_emitted", True)

        try:
            from ...services.ans.bus import get_event_bus
            from ...services.ans.event import Event, Severity

            bus = get_event_bus()
            bus.emit(Event(
                type="budget.iteration.warning",
                source="oracle_bt",
                severity=Severity.WARNING,
                payload={
                    "turn": turn,
                    "max_turns": MAX_TURNS,
                    "percentage": (turn / MAX_TURNS * 100)
                }
            ))
        except Exception as e:
            logger.warning(f"Failed to emit iteration warning: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def drain_turn_start_notifications(ctx: "TickContext") -> RunStatus:
    """Drain and yield turn_start notifications.

    Corresponds to oracle_agent.py lines 2239-2303.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    try:
        from src.services.ans.accumulator import NotificationAccumulator

        accumulator = NotificationAccumulator()
        notifications = accumulator.drain("turn_start")

        for notification in notifications:
            _add_pending_chunk(bb, {
                "type": "system",
                "content": notification.get("content", ""),
                "severity": notification.get("severity", "info")
            })

    except Exception as e:
        logger.debug(f"No turn_start notifications: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def increment_turn(ctx: "TickContext") -> RunStatus:
    """Increment turn counter."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    turn = (bb_get(bb,"turn") or 0) + 1
    bb_set(bb,"turn", turn)
    logger.debug(f"Turn: {turn}/{MAX_TURNS}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def clear_tool_calls(ctx: "TickContext") -> RunStatus:
    """Clear tool calls after processing."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    bb_set(bb,"tool_calls", [])
    ctx.mark_progress()
    return RunStatus.SUCCESS


def save_exchange(ctx: "TickContext") -> RunStatus:
    """Save Q&A to tree and legacy context.

    Corresponds to oracle_agent.py lines 2620-2730.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    # Check if already saved this exchange
    if bb_get(bb, "_exchange_saved"):
        logger.debug("save_exchange: Exchange already saved, skipping")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    query = bb_get(bb, "query")
    accumulated = bb_get(bb, "accumulated_content") or ""
    user_id = bb_get(bb, "user_id")
    tree_root_id = bb_get(bb, "tree_root_id")
    current_node_id = bb_get(bb, "current_node_id")
    tool_results = bb_get(bb, "tool_results") or []
    system_messages = bb_get(bb, "system_messages") or []
    model = bb_get(bb, "model")
    tokens_used = bb_get(bb, "tokens_used") or 0

    # Get question text
    if isinstance(query, str):
        question = query
    elif hasattr(query, "question"):
        question = query.question
    elif isinstance(query, dict):
        question = query.get("question", str(query))
    else:
        question = str(query)

    # Save to tree context
    if tree_root_id and current_node_id and user_id:
        try:
            from src.services.context_tree_service import ContextTreeService
            from src.models.oracle_context import ToolCall, ToolCallStatus

            tree_service = ContextTreeService()

            # Convert tool_results to ToolCall objects
            tool_call_objects = []
            for tc in tool_results:
                if isinstance(tc, dict):
                    # Map success boolean to ToolCallStatus enum
                    # tool_results contains: call_id, name, result/error, success (bool)
                    # ToolCallStatus enum has: PENDING, SUCCESS, ERROR
                    if tc.get("status"):
                        # If status is explicitly set, use it
                        status = ToolCallStatus(tc.get("status"))
                    elif tc.get("success", False):
                        status = ToolCallStatus.SUCCESS
                    else:
                        status = ToolCallStatus.ERROR

                    tool_call_objects.append(ToolCall(
                        id=tc.get("call_id", ""),
                        name=tc.get("name", ""),
                        arguments=tc.get("arguments", {}),
                        result=tc.get("result"),
                        status=status,
                        duration_ms=tc.get("duration_ms"),
                    ))

            # create_node signature: (user_id, root_id, parent_id, question, answer, ...)
            new_node = tree_service.create_node(
                user_id=user_id,
                root_id=tree_root_id,
                parent_id=current_node_id,
                question=question,
                answer=accumulated,
                tool_calls=tool_call_objects if tool_call_objects else None,
                tokens_used=tokens_used,
                model_used=model,
                system_messages=system_messages if system_messages else None,
            )
            bb_set(bb, "current_node_id", new_node.id)
            bb_set(bb, "_exchange_saved", True)
            logger.debug(f"Saved exchange to tree: {new_node.id}")

        except Exception as e:
            logger.warning(f"Failed to save to tree: {e}", exc_info=True)

    # Save to legacy context (for backwards compatibility)
    try:
        from src.services.oracle_context_service import OracleContextService
        from src.models.oracle_context import OracleExchange, ExchangeRole

        context_service = OracleContextService()
        project_id = bb_get(bb, "project_id") or "default"

        if user_id:
            # Create user exchange
            user_exchange = OracleExchange(
                id=str(uuid.uuid4()),
                role=ExchangeRole.USER,
                content=question,
                timestamp=datetime.now(timezone.utc),
            )
            context_service.add_exchange(user_id, project_id, user_exchange, model)

            # Create assistant exchange
            assistant_exchange = OracleExchange(
                id=str(uuid.uuid4()),
                role=ExchangeRole.ASSISTANT,
                content=accumulated,
                timestamp=datetime.now(timezone.utc),
                token_count=tokens_used,
            )
            context_service.add_exchange(user_id, project_id, assistant_exchange, model)

    except Exception as e:
        logger.warning(f"Failed to save to legacy context: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def yield_sources(ctx: "TickContext") -> RunStatus:
    """Yield collected sources to frontend."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    sources = bb_get(bb,"collected_sources") or []

    if sources:
        _add_pending_chunk(bb, {
            "type": "sources",
            "sources": sources
        })
        logger.debug(f"Yielded {len(sources)} sources")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def emit_done(ctx: "TickContext") -> RunStatus:
    """Yield done chunk to complete response."""
    bb = ctx.blackboard
    if bb is None:
        logger.error("emit_done: No blackboard available")
        return RunStatus.FAILURE

    # Debug: check blackboard state
    accumulated_raw = bb_get(bb, "accumulated_content")
    logger.info(f"emit_done: accumulated_content raw value = {repr(accumulated_raw)[:100]}, type = {type(accumulated_raw)}")

    accumulated = accumulated_raw or ""
    turn = bb_get(bb, "turn") or 0
    tool_calls = bb_get(bb, "tool_calls") or []

    logger.info(
        f"emit_done called: turn={turn}, accumulated_len={len(accumulated)}, "
        f"tool_calls={len(tool_calls)}"
    )

    _add_pending_chunk(bb, {
        "type": "done",
        "accumulated_content": accumulated,
        "turn": turn,
    })

    logger.info("Emitted done chunk to pending_chunks")
    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Tool Execution Actions
# =============================================================================


def detect_loop(ctx: "TickContext") -> RunStatus:
    """Check for repeated tool call patterns.

    Corresponds to oracle_agent.py lines 649-713.

    IMPORTANT: This function handles BOTH formats of tool_calls:
    1. Raw format from LLM: {"function": {"name": "...", "arguments": {...}}}
    2. Parsed format after parse_tool_calls: {"name": "...", "arguments": {...}}

    The detect_loop action runs AFTER parse_tool_calls in execute-tools.lua,
    so we must handle the parsed format where "name" is at the top level.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    tool_calls = bb_get(bb,"tool_calls") or []
    recent_patterns = bb_get(bb,"recent_tool_patterns") or []
    loop_already_warned = bb_get(bb,"loop_already_warned") or False
    turn = bb_get(bb,"turn") or 0

    # CRITICAL DEBUG: Log every call to detect_loop
    logger.info(
        f"detect_loop ENTRY: turn={turn}, tool_calls_count={len(tool_calls)}, "
        f"recent_patterns_count={len(recent_patterns)}, loop_already_warned={loop_already_warned}"
    )

    if not tool_calls or loop_already_warned:
        # Ensure loop_detected is False when there are no tool calls
        # to prevent stale state from previous iterations
        if not tool_calls:
            bb_set(bb, "loop_detected", False)
        logger.info(f"detect_loop EARLY_EXIT: no tool_calls or already warned")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Build pattern signature
    pattern_parts = []
    for call in tool_calls:
        # Handle BOTH raw and parsed format:
        # Raw: {"function": {"name": "...", "arguments": {...}}}
        # Parsed: {"name": "...", "arguments": {...}}
        if "function" in call:
            # Raw format from LLM
            func = call.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})
        else:
            # Parsed format (after parse_tool_calls)
            name = call.get("name", "")
            args = call.get("arguments", {})

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        # Include key args in pattern
        key_args = ["path", "query", "thread_id", "file_path"]
        arg_str = ",".join(
            f"{k}={args.get(k)}" for k in key_args if k in args
        )
        pattern_parts.append(f"{name}({arg_str})")

    current_pattern = "|".join(sorted(pattern_parts))

    # Skip empty patterns (would cause false positive loop detection)
    if not current_pattern or current_pattern == "()" or all(p == "()" for p in pattern_parts):
        logger.debug(f"detect_loop: Skipping empty pattern from {len(tool_calls)} tool calls")
        bb_set(bb, "loop_detected", False)
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Add to recent patterns (window size 10)
    # Increased from 6 to reduce false positives
    recent_patterns.append(current_pattern)
    if len(recent_patterns) > 10:
        recent_patterns = recent_patterns[-10:]
    bb_set(bb,"recent_tool_patterns", recent_patterns)

    # Check for loop (5+ occurrences to avoid false positives)
    # Was 3, but this triggered too easily on legitimate tool use
    count = recent_patterns.count(current_pattern)
    loop_threshold = 5

    # Debug logging to help diagnose loop detection issues
    logger.debug(
        f"detect_loop: pattern='{current_pattern}', count={count}/{loop_threshold}, "
        f"recent_patterns_len={len(recent_patterns)}, patterns={recent_patterns}"
    )

    if count >= loop_threshold:
        bb_set(bb,"loop_detected", True)
        bb_set(bb,"loop_warning", f"Detected loop: pattern repeated {count} times")
        logger.warning(f"Loop detected: {current_pattern}, count={count}, all_patterns={recent_patterns}")
    else:
        bb_set(bb,"loop_detected", False)

    ctx.mark_progress()
    return RunStatus.SUCCESS


def emit_loop_event(ctx: "TickContext") -> RunStatus:
    """Emit AGENT_LOOP_DETECTED event, inject warning."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    loop_warning = bb_get(bb,"loop_warning") or "Loop detected"

    try:
        from src.services.ans.bus import get_event_bus
        from src.services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="agent.loop.detected",
            source="oracle_bt",
            severity=Severity.WARNING,
            payload={"warning": loop_warning}
        ))
    except Exception as e:
        logger.warning(f"Failed to emit loop event: {e}")

    # Mark as warned to prevent duplicate warnings
    bb_set(bb,"loop_already_warned", True)

    # Inject system notification
    messages = bb_get(bb,"messages") or []
    messages.append({
        "role": "system",
        "content": f"[Warning] {loop_warning}. Please try a different approach."
    })
    bb_set(bb,"messages", messages)

    ctx.mark_progress()
    return RunStatus.SUCCESS


def yield_loop_warning(ctx: "TickContext") -> RunStatus:
    """Yield loop warning chunk to frontend."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    loop_warning = bb_get(bb,"loop_warning") or "Loop detected"

    _add_pending_chunk(bb, {
        "type": "system",
        "content": loop_warning,
        "severity": "warning"
    })

    ctx.mark_progress()
    return RunStatus.SUCCESS


def parse_tool_calls(ctx: "TickContext") -> RunStatus:
    """Extract call_id, name, arguments from each call."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    tool_calls = bb_get(bb,"tool_calls") or []
    parsed_calls = []

    for call in tool_calls:
        call_id = call.get("id", f"call_{len(parsed_calls)}")
        func = call.get("function", {})
        name = func.get("name") or ""  # Handle None values from API
        args = func.get("arguments", {})

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        parsed_calls.append({
            "id": call_id,
            "name": name,
            "arguments": args,
            "status": "pending"
        })

    bb_set(bb,"tool_calls", parsed_calls)
    logger.debug(f"Parsed {len(parsed_calls)} tool calls")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def yield_tool_pending(ctx: "TickContext") -> RunStatus:
    """Yield tool_call chunks with status=pending."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    tool_calls = bb_get(bb,"tool_calls") or []

    for call in tool_calls:
        _add_pending_chunk(bb, {
            "type": "tool_call",
            "call_id": call.get("id"),
            "name": call.get("name"),
            "status": "pending"
        })

    ctx.mark_progress()
    return RunStatus.SUCCESS


def inject_project_context(ctx: "TickContext") -> RunStatus:
    """Add project_id to PROJECT_SCOPED_TOOLS."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    project_id = bb_get(bb,"project_id")
    if not project_id:
        ctx.mark_progress()
        return RunStatus.SUCCESS

    tool_calls = bb_get(bb,"tool_calls") or []

    for call in tool_calls:
        if call.get("name") in PROJECT_SCOPED_TOOLS:
            args = call.get("arguments", {})
            args["project_id"] = project_id
            call["arguments"] = args

    bb_set(bb,"tool_calls", tool_calls)
    ctx.mark_progress()
    return RunStatus.SUCCESS


def execute_single_tool(ctx: "TickContext") -> RunStatus:
    """Execute tool and collect result.

    This action is called inside a for_each loop with current_tool set.
    Note: ForEach sets item_key with underscore prefix (_current_tool),
    so we must read from _current_tool, not current_tool.
    """
    import asyncio

    bb = ctx.blackboard
    if bb is None:
        logger.error("execute_single_tool: No blackboard available")
        return RunStatus.FAILURE

    # ForEach sets the iteration variable with underscore prefix
    # e.g., item_key="current_tool" -> stored as "_current_tool"
    current_tool = bb_get(bb, "_current_tool")
    if not current_tool:
        logger.warning(
            f"execute_single_tool: _current_tool not found in blackboard. "
            f"Available keys: {list(bb._data.keys()) if hasattr(bb, '_data') else 'N/A'}"
        )
        return RunStatus.FAILURE

    call_id = current_tool.get("id")
    name = current_tool.get("name") or ""  # Defensive: handle None
    args = current_tool.get("arguments", {})
    user_id = bb_get(bb, "user_id") or "anonymous"

    try:
        from src.services.tool_executor import ToolExecutor

        executor = ToolExecutor()
        # ToolExecutor.execute is async, need to run it in the event loop
        # Use asyncio.get_event_loop().run_until_complete() for sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is already running, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, executor.execute(name, args, user_id))
                result = future.result(timeout=120)  # 2 minute timeout
        else:
            result = loop.run_until_complete(executor.execute(name, args, user_id))

        # Store result
        tool_results = bb_get(bb,"tool_results") or []
        tool_results.append({
            "call_id": call_id,
            "name": name,
            "result": result,
            "success": True
        })
        bb_set(bb,"tool_results", tool_results)

        # Update tool status
        current_tool["status"] = "success"
        current_tool["result"] = result

        logger.debug(f"Tool {name} executed successfully")
        ctx.mark_progress()
        return RunStatus.SUCCESS

    except Exception as e:
        # Store failure
        tool_results = bb_get(bb,"tool_results") or []
        tool_results.append({
            "call_id": call_id,
            "name": name,
            "error": str(e),
            "success": False
        })
        bb_set(bb,"tool_results", tool_results)

        # Update tool status
        current_tool["status"] = "error"
        current_tool["error"] = str(e)

        logger.error(f"Tool {name} failed: {e}")
        ctx.mark_progress()
        return RunStatus.SUCCESS  # Continue on failure per tree config


def process_tool_results(ctx: "TickContext") -> RunStatus:
    """Yield result chunks, emit events, extract sources."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    tool_results = bb_get(bb,"tool_results") or []

    for result in tool_results:
        call_id = result.get("call_id")
        name = result.get("name")
        success = result.get("success", False)

        # Yield chunk
        if success:
            _add_pending_chunk(bb, {
                "type": "tool_result",
                "call_id": call_id,
                "name": name,
                "status": "success",
                "result": result.get("result")
            })

            # Emit success event
            _emit_tool_event(name, call_id, "success", result.get("result"))

            # Extract sources if present
            _extract_sources(bb, name, result.get("result"))
        else:
            _add_pending_chunk(bb, {
                "type": "tool_result",
                "call_id": call_id,
                "name": name,
                "status": "error",
                "error": result.get("error")
            })

            # Emit failure event
            _emit_tool_event(name, call_id, "failure", result.get("error"))

    ctx.mark_progress()
    return RunStatus.SUCCESS


def _emit_tool_event(name: str, call_id: str, status: str, data: Any) -> None:
    """Emit tool call event to ANS."""
    try:
        from src.services.ans.bus import get_event_bus
        from src.services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type=f"tool.call.{status}",
            source="oracle_bt",
            severity=Severity.INFO if status == "success" else Severity.ERROR,
            payload={
                "tool": name,
                "call_id": call_id,
                "data": str(data)[:500] if data else None
            }
        ))
    except Exception as e:
        logger.debug(f"Failed to emit tool event: {e}")


def _extract_sources(bb: Any, tool_name: str, result: Any) -> None:
    """Extract sources from tool result."""
    sources = bb_get(bb,"collected_sources") or []

    if result is None:
        return

    # Handle different result types
    if isinstance(result, dict):
        if "sources" in result:
            sources.extend(result["sources"])
        elif "path" in result:
            sources.append({"type": tool_name, "path": result["path"]})
    elif isinstance(result, list):
        for item in result[:10]:  # Limit sources
            if isinstance(item, dict) and "path" in item:
                sources.append({"type": tool_name, "path": item["path"]})

    bb_set(bb,"collected_sources", sources)


def update_context_tokens(ctx: "TickContext") -> RunStatus:
    """Update context_tokens estimate after tool results."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    messages = bb_get(bb,"messages") or []
    tool_results = bb_get(bb,"tool_results") or []

    # Re-estimate with tool results included
    base_tokens = _estimate_tokens(messages)
    result_tokens = sum(
        len(str(r.get("result", ""))) // 4
        for r in tool_results
    )

    total = base_tokens + result_tokens
    bb_set(bb,"context_tokens", total)

    # Check context warning threshold
    max_context = bb_get(bb,"max_context_tokens") or DEFAULT_CONTEXT_SIZE
    if total / max_context >= CONTEXT_WARNING_THRESHOLD:
        logger.warning(f"Context approaching limit: {total}/{max_context}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def add_tool_results_to_messages(ctx: "TickContext") -> RunStatus:
    """Append tool results to conversation messages."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    messages = bb_get(bb,"messages") or []
    tool_results = bb_get(bb,"tool_results") or []
    tool_calls = bb_get(bb,"tool_calls") or []

    if not tool_results:
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Add assistant message with tool calls
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": tc.get("name"),
                    "arguments": json.dumps(tc.get("arguments", {}))
                }
            }
            for tc in tool_calls
        ]
    })

    # Add tool result messages
    for result in tool_results:
        content = result.get("result") if result.get("success") else f"Error: {result.get('error')}"
        if not isinstance(content, str):
            content = json.dumps(content)

        messages.append({
            "role": "tool",
            "tool_call_id": result.get("call_id"),
            "name": result.get("name"),
            "content": content
        })

    bb_set(bb,"messages", messages)
    bb_set(bb,"tool_results", [])  # Clear for next iteration

    ctx.mark_progress()
    return RunStatus.SUCCESS


def drain_after_tool_notifications(ctx: "TickContext") -> RunStatus:
    """Drain and yield after_tool notifications."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    try:
        from src.services.ans.accumulator import NotificationAccumulator

        accumulator = NotificationAccumulator()
        notifications = accumulator.drain("after_tool")

        for notification in notifications:
            _add_pending_chunk(bb, {
                "type": "system",
                "content": notification.get("content", ""),
                "severity": notification.get("severity", "info")
            })

    except Exception as e:
        logger.debug(f"No after_tool notifications: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# LLM Related Actions
# =============================================================================


def build_llm_request(ctx: "TickContext") -> RunStatus:
    """Build request with model, messages, tools, max_tokens."""
    bb = ctx.blackboard
    if bb is None:
        logger.error("build_llm_request: No blackboard!")
        return RunStatus.FAILURE

    # Request is already prepared in blackboard
    # This action is a hook for any last-minute modifications

    model = bb_get(bb,"model")
    # Only add :thinking suffix if user has thinking_enabled AND model supports it
    # Note: thinking_enabled should be set in blackboard during init
    thinking_enabled = bb_get(bb, "thinking_enabled")
    if model and thinking_enabled and ":thinking" not in model:
        # Some models support thinking traces via OpenRouter suffix
        thinking_models = ["deepseek", "o1"]  # Claude doesn't use :thinking suffix
        if any(m in model.lower() for m in thinking_models):
            bb_set(bb, "model", f"{model}:thinking")
            logger.debug(f"Enabled thinking mode for model: {model}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def on_llm_chunk(ctx: "TickContext") -> RunStatus:
    """Callback for streaming LLM chunks.

    Called by LLMCallNode during streaming.
    Filters out signal XML tags from the streamed content.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    partial = bb_get(bb,"partial_response") or ""

    # Filter out signal tags from streaming content
    # Signal tags look like: <signal type="...">...</signal>
    # We need to detect the start of a potential signal and not stream it
    clean_content = partial

    # Check for start of signal tag
    signal_start = partial.find("<signal")
    if signal_start != -1:
        # Only emit content before the signal tag
        clean_content = partial[:signal_start]
    else:
        # Also check for partial signal tag at the end (e.g., "<sig" or "<sign")
        # This prevents streaming partial tags that look like: "<", "<s", "<si", etc.
        for i in range(1, min(8, len(partial) + 1)):  # "<signal" is 7 chars
            suffix = partial[-i:] if len(partial) >= i else partial
            if "<signal".startswith(suffix) and suffix.startswith("<"):
                clean_content = partial[:-i]
                break

    # Track what we've already streamed to avoid duplicates
    last_streamed_len = bb_get(bb, "_last_streamed_len") or 0

    # Only emit new content
    if len(clean_content) > last_streamed_len:
        new_content = clean_content[last_streamed_len:]
        bb_set(bb, "_last_streamed_len", len(clean_content))

        # Yield streaming chunk with only new content
        _add_pending_chunk(bb, {
            "type": "content",
            "content": new_content,
            "streaming": True
        })

    ctx.mark_progress()
    return RunStatus.SUCCESS


def update_token_budget(ctx: "TickContext") -> RunStatus:
    """Update tokens_used, check budget warnings."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    llm_response = bb_get(bb,"llm_response")
    if llm_response and hasattr(llm_response, "usage"):
        usage = llm_response.usage
        tokens_used = (bb_get(bb,"tokens_used") or 0) + usage.get("total_tokens", 0)
        bb_set(bb,"tokens_used", tokens_used)

        # Check warning threshold
        max_tokens = bb_get(bb,"max_tokens") or 0
        if max_tokens > 0 and tokens_used / max_tokens >= TOKEN_WARNING_THRESHOLD:
            if not bb_get(bb,"token_warning_emitted"):
                bb_set(bb,"token_warning_emitted", True)
                logger.warning(f"Token budget warning: {tokens_used}/{max_tokens}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def extract_xml_tool_calls(ctx: "TickContext") -> RunStatus:
    """Parse XML tool syntax from content/reasoning.

    Some models return tool calls as XML instead of native format.
    Corresponds to oracle_agent.py XML parsing logic.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    content = bb_get(bb,"accumulated_content") or ""
    reasoning = bb_get(bb,"reasoning_content") or ""

    # Check both content and reasoning for XML tool calls
    combined = f"{content}\n{reasoning}"

    # Pattern for XML tool calls: <tool_call><name>...</name><arguments>...</arguments></tool_call>
    pattern = r"<tool_call>\s*<name>([^<]+)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>"
    matches = re.findall(pattern, combined, re.DOTALL)

    if matches:
        tool_calls = []
        for i, (name, args_str) in enumerate(matches):
            try:
                args = json.loads(args_str.strip())
            except json.JSONDecodeError:
                args = {"raw": args_str.strip()}

            tool_calls.append({
                "id": f"xml_call_{i}",
                "function": {
                    "name": name.strip(),
                    "arguments": args
                }
            })

        bb_set(bb,"tool_calls", tool_calls)
        logger.debug(f"Extracted {len(tool_calls)} XML tool calls")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def accumulate_content(ctx: "TickContext") -> RunStatus:
    """Add LLM response to accumulated_content."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    llm_response = bb_get(bb, "llm_response")
    logger.info(f"accumulate_content: llm_response type={type(llm_response)}, has_content={hasattr(llm_response, 'content') if llm_response else False}")

    if llm_response and hasattr(llm_response, "content"):
        content_to_add = llm_response.content if llm_response.content is not None else ""
        logger.info(f"accumulate_content: content_to_add length={len(content_to_add)}")

        accumulated = bb_get(bb, "accumulated_content") or ""
        logger.info(f"accumulate_content: BEFORE accumulated length={len(accumulated)}")

        accumulated += content_to_add
        bb_set(bb, "accumulated_content", accumulated)

        logger.info(f"accumulate_content: AFTER accumulated length={len(accumulated)}")
    else:
        logger.warning(f"accumulate_content: Could not accumulate - llm_response={llm_response is not None}, has_content={hasattr(llm_response, 'content') if llm_response else 'N/A'}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Error and Completion Actions
# =============================================================================


def emit_cancelled(ctx: "TickContext") -> RunStatus:
    """Yield cancelled error chunk."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    _add_pending_chunk(bb, {
        "type": "error",
        "error": "cancelled",
        "message": "Request was cancelled"
    })

    ctx.mark_progress()
    return RunStatus.SUCCESS


def handle_max_turns_exceeded(ctx: "TickContext") -> RunStatus:
    """Emit iteration_exceeded event, save partial, yield done."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    # Emit event
    emit_iteration_exceeded(ctx)

    # Drain immediate notifications
    drain_immediate_notifications(ctx)

    # Save partial exchange
    save_partial_exchange(ctx)

    # Emit done with warning
    emit_done_with_warning(ctx)

    return RunStatus.SUCCESS


def emit_iteration_exceeded(ctx: "TickContext") -> RunStatus:
    """Emit BUDGET_ITERATION_EXCEEDED event."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    turn = bb_get(bb,"turn") or 0

    try:
        from src.services.ans.bus import get_event_bus
        from src.services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="budget.iteration.exceeded",
            source="oracle_bt",
            severity=Severity.ERROR,
            payload={
                "turn": turn,
                "max_turns": MAX_TURNS
            }
        ))
    except Exception as e:
        logger.warning(f"Failed to emit iteration exceeded: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def drain_immediate_notifications(ctx: "TickContext") -> RunStatus:
    """Drain and yield immediate notifications."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    try:
        from src.services.ans.accumulator import NotificationAccumulator

        accumulator = NotificationAccumulator()
        notifications = accumulator.drain("immediate")

        for notification in notifications:
            _add_pending_chunk(bb, {
                "type": "system",
                "content": notification.get("content", ""),
                "severity": notification.get("severity", "error")
            })

    except Exception as e:
        logger.debug(f"No immediate notifications: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


def save_partial_exchange(ctx: "TickContext") -> RunStatus:
    """Save partial exchange for recovery."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    accumulated = bb_get(bb,"accumulated_content") or ""
    if accumulated:
        # Mark as partial
        bb_set(bb,"accumulated_content", accumulated + "\n[Response truncated due to turn limit]")
        save_exchange(ctx)

    ctx.mark_progress()
    return RunStatus.SUCCESS


def emit_done_with_warning(ctx: "TickContext") -> RunStatus:
    """Yield done chunk with max_turns warning."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    _add_pending_chunk(bb, {
        "type": "done",
        "accumulated_content": bb_get(bb,"accumulated_content") or "",
        "turn": bb_get(bb,"turn") or 0,
        "warning": "max_turns_exceeded"
    })

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Finalization Actions
# =============================================================================


def finalize_response(ctx: "TickContext") -> RunStatus:
    """Final cleanup and metrics."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    # Log metrics
    turn = bb_get(bb,"turn") or 0
    tokens_used = bb_get(bb,"tokens_used") or 0
    context_tokens = bb_get(bb,"context_tokens") or 0

    logger.info(
        f"Oracle response complete: turns={turn}, "
        f"tokens_used={tokens_used}, context={context_tokens}"
    )

    ctx.mark_progress()
    return RunStatus.SUCCESS


def save_partial_if_needed(ctx: "TickContext") -> RunStatus:
    """Save partial exchange if connection dropped."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    accumulated = bb_get(bb,"accumulated_content") or ""

    # Only save if we have content but haven't saved yet
    if accumulated and not bb_get(bb,"_exchange_saved"):
        logger.info("Saving partial exchange due to early termination")
        save_exchange(ctx)

    ctx.mark_progress()
    return RunStatus.SUCCESS


def emit_session_end(ctx: "TickContext") -> RunStatus:
    """Emit SESSION_END event for plugin system."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    user_id = bb_get(bb,"user_id")
    project_id = bb_get(bb,"project_id")
    turn = bb_get(bb,"turn") or 0

    try:
        from src.services.ans.bus import get_event_bus
        from src.services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="oracle.session.end",
            source="oracle_bt",
            severity=Severity.INFO,
            payload={
                "user_id": user_id,
                "project_id": project_id,
                "turns": turn
            }
        ))
    except Exception as e:
        logger.warning(f"Failed to emit SESSION_END: {e}")

    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Utility Actions
# =============================================================================


def noop(ctx: "TickContext") -> RunStatus:
    """No-operation action for tree control flow."""
    ctx.mark_progress()
    return RunStatus.SUCCESS


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Phase 1: Initialization
    "reset_state",
    "emit_query_start",
    # Phase 2: Context Loading
    "load_tree_node",
    "get_or_create_tree",
    "load_legacy_context",
    "load_cross_session_notifications",
    # Phase 3: Message Building
    "build_system_prompt",
    "add_tree_history",
    "inject_notifications",
    "add_user_question",
    "get_tool_schemas",
    "init_context_tracking",
    "yield_context_update",
    # Phase 4: Agent Loop
    "check_iteration_budget",
    "drain_turn_start_notifications",
    "increment_turn",
    "clear_tool_calls",
    "save_exchange",
    "yield_sources",
    "emit_done",
    # Tool Execution
    "detect_loop",
    "emit_loop_event",
    "yield_loop_warning",
    "parse_tool_calls",
    "yield_tool_pending",
    "inject_project_context",
    "execute_single_tool",
    "process_tool_results",
    "update_context_tokens",
    "add_tool_results_to_messages",
    "drain_after_tool_notifications",
    # LLM
    "build_llm_request",
    "on_llm_chunk",
    "update_token_budget",
    "extract_xml_tool_calls",
    "accumulate_content",
    # Error and Completion
    "emit_cancelled",
    "handle_max_turns_exceeded",
    "emit_iteration_exceeded",
    "drain_immediate_notifications",
    "save_partial_exchange",
    "emit_done_with_warning",
    # Finalization
    "finalize_response",
    "save_partial_if_needed",
    "emit_session_end",
    # Utility
    "noop",
]
