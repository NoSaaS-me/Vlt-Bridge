"""Graphiti client helpers for Oracle V2 memory layer."""

from __future__ import annotations

from typing import Any, Optional


def get_graphiti(app_state: Any) -> Optional[Any]:
    """Return app.state.graphiti or None if unavailable.

    Gracefully handles missing attribute (e.g. during testing or if
    FalkorDB was unreachable at startup).

    Args:
        app_state: FastAPI application state object (app.state).

    Returns:
        Graphiti instance or None.
    """
    return getattr(app_state, "graphiti", None)


def group_id_for(user_id: str, project_id: str) -> str:
    """Return the project-scoped Graphiti group_id.

    Facts stored under this group_id are scoped to a specific
    (user, project) pair — no cross-project leakage.

    Args:
        user_id: Authenticated user identifier.
        project_id: Project identifier.

    Returns:
        f"{user_id}:{project_id}"
    """
    return f"{user_id}:{project_id}"


def user_group_id(user_id: str) -> str:
    """Return the user-scoped Graphiti group_id (preferences, cross-project facts).

    Args:
        user_id: Authenticated user identifier.

    Returns:
        f"{user_id}:_user"
    """
    return f"{user_id}:_user"


__all__ = ["get_graphiti", "group_id_for", "user_group_id"]
