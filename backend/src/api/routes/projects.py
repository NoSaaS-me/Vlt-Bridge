"""HTTP API routes for project management."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...models.project import (
    Project,
    ProjectCreate,
    ProjectUpdate,
    ProjectList,
    ProjectStats,
    DEFAULT_PROJECT_ID,
)
from ...services.project_service import ProjectService
from ...services.config import get_config
from ..middleware import AuthContext, get_auth_context

router = APIRouter(prefix="/api/projects", tags=["projects"])

DEMO_USER_ID = "demo-user"


def _ensure_write_allowed(user_id: str) -> None:
    # Allow writes in local development mode
    config = get_config()
    if config.enable_local_mode:
        return

    if user_id == DEMO_USER_ID:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "demo_read_only",
                "message": "Demo mode is read-only. Sign in with GitHub to manage projects.",
            },
        )


def get_project_service() -> ProjectService:
    return ProjectService()


@router.get("", response_model=ProjectList)
async def list_projects(
    auth: AuthContext = Depends(get_auth_context),
    project_service: ProjectService = Depends(get_project_service),
):
    """List all projects for the current user."""
    projects = project_service.list_projects(auth.user_id)
    return ProjectList(projects=projects, total=len(projects))


@router.post("", response_model=Project, status_code=201)
async def create_project(
    data: ProjectCreate,
    auth: AuthContext = Depends(get_auth_context),
    project_service: ProjectService = Depends(get_project_service),
):
    """Create a new project."""
    _ensure_write_allowed(auth.user_id)

    try:
        project = project_service.create_project(auth.user_id, data)
        return project
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")


@router.get("/{project_id}", response_model=Project)
async def get_project(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    project_service: ProjectService = Depends(get_project_service),
):
    """Get a specific project."""
    project = project_service.get_project(auth.user_id, project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return project


@router.put("/{project_id}", response_model=Project)
async def update_project(
    project_id: str,
    data: ProjectUpdate,
    auth: AuthContext = Depends(get_auth_context),
    project_service: ProjectService = Depends(get_project_service),
):
    """Update a project."""
    _ensure_write_allowed(auth.user_id)

    try:
        project = project_service.update_project(auth.user_id, project_id, data)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        return project
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    project_service: ProjectService = Depends(get_project_service),
):
    """Delete a project and all its data."""
    _ensure_write_allowed(auth.user_id)

    try:
        success = project_service.delete_project(auth.user_id, project_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")


@router.get("/{project_id}/stats", response_model=ProjectStats)
async def get_project_stats(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    project_service: ProjectService = Depends(get_project_service),
):
    """Get detailed statistics for a project."""
    stats = project_service.get_project_stats(auth.user_id, project_id)
    if not stats:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return stats


@router.post("/{project_id}/set-default", response_model=dict)
async def set_default_project(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    project_service: ProjectService = Depends(get_project_service),
):
    """Set a project as the user's default."""
    _ensure_write_allowed(auth.user_id)

    success = project_service.set_default_project(auth.user_id, project_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    return {"message": f"Default project set to {project_id}"}


@router.get("/default/id", response_model=dict)
async def get_default_project_id(
    auth: AuthContext = Depends(get_auth_context),
    project_service: ProjectService = Depends(get_project_service),
):
    """Get the user's default project ID."""
    project_id = project_service.get_default_project_id(auth.user_id)
    return {"project_id": project_id}


__all__ = ["router"]
