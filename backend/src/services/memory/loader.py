"""Memory loader: retrieve relevant facts from Graphiti before each turn."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .client import group_id_for

logger = logging.getLogger(__name__)


async def load_recalled_facts(
    graphiti: Optional[Any],
    query: str,
    user_id: str,
    project_id: str,
    limit: int = 10,
) -> list[str]:
    """Load relevant facts from Graphiti memory for the given query.

    Performs a semantic search over the project-scoped knowledge graph and
    returns currently-valid fact strings (edges where invalid_at is None).

    Args:
        graphiti: Graphiti instance or None (graceful degradation if None).
        query: The user's current question — used for semantic search.
        user_id: Authenticated user identifier.
        project_id: Project identifier.
        limit: Maximum number of facts to return (default 10).

    Returns:
        List of fact strings. Empty list if graphiti is None or on error.
    """
    if graphiti is None:
        return []

    gid = group_id_for(user_id, project_id)

    try:
        results = await graphiti.search(query=query, group_id=gid, limit=limit)
    except Exception:
        logger.warning(
            "Graphiti search failed for user=%s project=%s query=%r",
            user_id,
            project_id,
            query,
            exc_info=True,
        )
        return []

    facts: list[str] = []
    for edge in results:
        # Only include facts that are still valid (not superseded/invalidated)
        if getattr(edge, "invalid_at", None) is None:
            fact = getattr(edge, "fact", None)
            if fact:
                facts.append(fact)

    return facts


__all__ = ["load_recalled_facts"]
