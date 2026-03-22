"""Memory tools for explicit remember/recall operations.

These tools are injected into the Oracle REPL so the agent can explicitly
store and retrieve cross-session facts via Graphiti.

The eval_fn runs in a ThreadPoolExecutor (sync context), so async calls
to Graphiti must bridge via asyncio.run_coroutine_threadsafe().

Usage pattern (inside make_memory_tools factory):
    tools = make_memory_tools(graphiti, user_id, project_id)
    # → [remember, recall]  — bound callables with correct closures
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from ...memory.loader import load_recalled_facts
from ...memory.writer import write_episode

logger = logging.getLogger(__name__)

# Sentinel thread_id / turn_index for explicit remember() calls
# (not part of an automated turn summary — still stored as an episode)
_EXPLICIT_THREAD_ID = "explicit"
_EXPLICIT_TURN_INDEX = 0


def make_memory_tools(
    graphiti: Optional[Any],
    user_id: str,
    project_id: str,
) -> list[Callable]:
    """Factory: returns [remember, recall] bound to graphiti/user/project.

    Args:
        graphiti: Graphiti instance or None (tools degrade gracefully when None).
        user_id: Authenticated user identifier (captured in closure).
        project_id: Project identifier (captured in closure).

    Returns:
        [remember, recall] — list of two sync callables for injection into REPL.
    """

    def _get_loop() -> asyncio.AbstractEventLoop:
        """Return the running event loop (raises RuntimeError if none)."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            raise RuntimeError(
                "No running event loop found. memory_tools must be called "
                "from within an async context (e.g. via run_coroutine_threadsafe)."
            )

    def remember(fact: str) -> str:
        """Store a named fact to cross-session memory.

        The fact will be recalled in future sessions when relevant queries are made.
        Graphiti extracts semantic knowledge from the episode text, so natural
        language descriptions work best (e.g. 'The rate limiter lives in
        src/api/middleware/rate_limit.py and uses a sliding window algorithm').

        Args:
            fact: The fact to remember as a natural language string.

        Returns:
            Confirmation string, or error message if storage failed.
        """
        if graphiti is None:
            return "Memory not available — Graphiti is not connected."

        try:
            loop = _get_loop()
            future = asyncio.run_coroutine_threadsafe(
                write_episode(
                    graphiti=graphiti,
                    episode_body=fact,
                    user_id=user_id,
                    project_id=project_id,
                    thread_id=_EXPLICIT_THREAD_ID,
                    turn_index=_EXPLICIT_TURN_INDEX,
                ),
                loop,
            )
            future.result(timeout=10)
            return f"Remembered: {fact[:120]}{'...' if len(fact) > 120 else ''}"
        except TimeoutError:
            return "Error: memory write timed out after 10s."
        except Exception as exc:
            logger.warning("remember() failed: %s", exc, exc_info=True)
            return f"Error storing fact: {exc}"

    def recall(query: str) -> str:
        """Retrieve facts from cross-session memory relevant to query.

        Performs semantic search over the project knowledge graph and returns
        currently-valid facts matching the query.

        Args:
            query: Natural language search query to find relevant remembered facts.

        Returns:
            Formatted list of relevant facts, or 'No relevant facts found.'
        """
        if graphiti is None:
            return "No facts available — Graphiti is not connected."

        try:
            loop = _get_loop()
            future = asyncio.run_coroutine_threadsafe(
                load_recalled_facts(
                    graphiti=graphiti,
                    query=query,
                    user_id=user_id,
                    project_id=project_id,
                    limit=10,
                ),
                loop,
            )
            facts = future.result(timeout=10)
        except TimeoutError:
            return "Error: memory recall timed out after 10s."
        except Exception as exc:
            logger.warning("recall() failed: %s", exc, exc_info=True)
            return f"Error retrieving facts: {exc}"

        if not facts:
            return "No relevant facts found."

        lines = [f"Recalled {len(facts)} fact(s) for query {query!r}:\n"]
        for i, fact in enumerate(facts, 1):
            lines.append(f"{i}. {fact}")
        return "\n".join(lines)

    return [remember, recall]


__all__ = ["make_memory_tools"]
