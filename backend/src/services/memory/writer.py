"""Memory writer: persist turn summaries to Graphiti after each turn."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .client import group_id_for

logger = logging.getLogger(__name__)


async def write_episode(
    graphiti: Optional[Any],
    episode_body: str,
    user_id: str,
    project_id: str,
    thread_id: str,
    turn_index: int,
) -> None:
    """Add a conversation episode to Graphiti memory (fire-and-forget safe).

    Designed to be called via asyncio.create_task() — errors are logged but
    never propagated so they cannot affect the main response stream.

    Episode name is deterministic: f"oracle-turn-{thread_id}-{turn_index}"
    Facts extracted from the episode are stored under the project-scoped group_id.

    Args:
        graphiti: Graphiti instance or None (silent no-op if None).
        episode_body: Stringified turn summary to store as an episode.
        user_id: Authenticated user identifier.
        project_id: Project identifier.
        thread_id: LangGraph thread_id (Oracle conversation thread).
        turn_index: Zero-based turn counter within the thread.
    """
    if graphiti is None:
        return

    name = f"oracle-turn-{thread_id}-{turn_index}"
    gid = group_id_for(user_id, project_id)

    try:
        await graphiti.add_episode(
            name=name,
            episode_body=episode_body,
            group_id=gid,
            source="message",
            reference_time=datetime.now(timezone.utc),
        )
        logger.debug("Wrote Graphiti episode %r (group=%s)", name, gid)
    except Exception:
        logger.warning(
            "Failed to write Graphiti episode %r (group=%s) — non-fatal",
            name,
            gid,
            exc_info=True,
        )


__all__ = ["write_episode"]
