"""Oracle V2 graph nodes: memory_loader_node, memory_writer_node.

LangGraph supports async nodes natively — these are defined as async functions.
Both nodes read graphiti, user_id, and project_id from config["configurable"]
so they never depend on state fields that would be persisted in the checkpoint.

Phase 6 planner_node will be added here once US4 work begins.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..memory.loader import load_recalled_facts
from ..memory.writer import write_episode

logger = logging.getLogger(__name__)


async def memory_loader_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Load relevant facts from Graphiti before the agent turn.

    Reads:
        - state["messages"]    — last user message used as semantic search query
        - config["configurable"]["graphiti"]   — Graphiti instance or None
        - config["configurable"]["user_id"]    — scoping key
        - config["configurable"]["project_id"] — scoping key

    Returns:
        {"recalled_facts": [list of fact strings]}

    Gracefully returns empty list when graphiti is None or unavailable.
    """
    configurable = config.get("configurable", {})
    graphiti = configurable.get("graphiti")
    user_id: str = configurable.get("user_id", state.get("user_id", ""))
    project_id: str = configurable.get("project_id", state.get("project_id", ""))

    # Extract last user message as the semantic search query
    query = _last_user_message(state)
    if not query:
        return {"recalled_facts": []}

    try:
        facts = await load_recalled_facts(
            graphiti=graphiti,
            query=query,
            user_id=user_id,
            project_id=project_id,
        )
    except Exception as exc:
        logger.warning(
            "memory_loader_node: graphiti search failed (degrading gracefully): %s", exc
        )
        return {"recalled_facts": []}

    logger.debug(
        "memory_loader_node: recalled %d facts for user=%s project=%s",
        len(facts),
        user_id,
        project_id,
    )
    return {"recalled_facts": facts}


async def memory_writer_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Store turn summary to Graphiti after the agent turn (fire-and-forget).

    Assembles episode_body from the last assistant message in state["messages"].
    Fires write_episode as asyncio.create_task() — does NOT block the response.

    Reads:
        - state["messages"]    — last assistant message used as episode body
        - config["configurable"]["graphiti"]   — Graphiti instance or None
        - config["configurable"]["user_id"]    — scoping key
        - config["configurable"]["project_id"] — scoping key
        - config["configurable"]["thread_id"]  — LangGraph thread_id
        - config["configurable"]["turn_index"] — optional turn counter (default 0)

    Returns:
        {}  (no state mutation)
    """
    configurable = config.get("configurable", {})
    graphiti = configurable.get("graphiti")

    if graphiti is None:
        return {}

    user_id: str = configurable.get("user_id", state.get("user_id", ""))
    project_id: str = configurable.get("project_id", state.get("project_id", ""))
    thread_id: str = configurable.get("thread_id", "unknown")
    turn_index: int = configurable.get("turn_index", 0)

    episode_body = _last_assistant_message(state)
    if not episode_body:
        return {}

    # Fire and forget — do not await, do not block the response stream
    try:
        asyncio.create_task(
            write_episode(
                graphiti=graphiti,
                episode_body=episode_body,
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
                turn_index=turn_index,
            )
        )
    except Exception as exc:
        logger.warning("memory_writer_node: failed to schedule write_episode task: %s", exc)

    logger.debug(
        "memory_writer_node: fired write_episode task for thread=%s turn=%d",
        thread_id,
        turn_index,
    )
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_user_message(state: dict[str, Any]) -> str:
    """Return the content of the last user/human message in state, or ''."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        # Support both LangChain BaseMessage objects and plain dicts
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Handle multi-part message content
                parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
                return " ".join(p for p in parts if p)
    return ""


def _last_assistant_message(state: dict[str, Any]) -> str:
    """Return the content of the last AI/assistant message in state, or ''."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("ai", "assistant"):
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
                return " ".join(p for p in parts if p)
    return ""


async def planner_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Generate a multi-step plan for complex tasks using the configured LLM.

    Reads:
        - state["messages"]  — last user message as the task description
        - config["configurable"]["api_key"]      — OpenRouter API key
        - config["configurable"]["oracle_model"] — model name for planning

    Returns:
        {"plan": [list of step strings], "plan_step": 0}

    Gracefully returns empty plan if the model call fails or dependencies absent.
    """
    configurable = config.get("configurable", {})
    api_key: str = configurable.get("api_key", "")
    oracle_model: str = configurable.get("oracle_model", "anthropic/claude-sonnet-4-6")

    query = _last_user_message(state)
    if not query:
        return {"plan": [], "plan_step": 0}

    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import]
    except ImportError:
        logger.debug("planner_node: langchain_openai not available, skipping planner")
        return {"plan": [], "plan_step": 0}

    try:
        planner_model = ChatOpenAI(
            model=oracle_model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=2000,
            streaming=False,
        )
        system_prompt = (
            "You are a task decomposer for a code search agent. "
            "Break the user's request into ordered, concrete steps. "
            "Output only a numbered list, one step per line, no explanations."
        )
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore[import]
        response = await planner_model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ])
        raw_plan = getattr(response, "content", "") or ""
        plan_steps = [
            line.lstrip("0123456789. )-").strip()
            for line in raw_plan.strip().splitlines()
            if line.strip()
        ]
        logger.debug("planner_node: generated %d steps for query=%r", len(plan_steps), query[:60])
        return {"plan": plan_steps, "plan_step": 0}
    except Exception as exc:
        logger.warning("planner_node: planning call failed (degrading to no plan): %s", exc)
        return {"plan": [], "plan_step": 0}


__all__ = ["memory_loader_node", "memory_writer_node", "planner_node"]
