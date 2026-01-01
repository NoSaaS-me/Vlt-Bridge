"""Service for managing user projects."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from ..models.project import (
    Project,
    ProjectCreate,
    ProjectUpdate,
    ProjectStats,
    generate_project_slug,
    DEFAULT_PROJECT_ID,
)
from .database import DatabaseService
from .config import get_config

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProjectService:
    """Service for CRUD operations on user projects."""

    def __init__(self, db_service: Optional[DatabaseService] = None):
        """Initialize with optional database service."""
        self.db = db_service or DatabaseService()
        self.config = get_config()

    def list_projects(self, user_id: str) -> List[Project]:
        """
        List all projects for a user.

        Args:
            user_id: User identifier

        Returns:
            List of Project objects with note/thread counts
        """
        conn = self.db.connect()
        try:
            # Get projects with note counts
            cursor = conn.execute(
                """
                SELECT
                    p.user_id,
                    p.project_id,
                    p.name,
                    p.description,
                    p.created_at,
                    p.updated_at,
                    COALESCE(nm.note_count, 0) as note_count,
                    COALESCE(t.thread_count, 0) as thread_count
                FROM projects p
                LEFT JOIN (
                    SELECT user_id, project_id, COUNT(*) as note_count
                    FROM note_metadata
                    GROUP BY user_id, project_id
                ) nm ON p.user_id = nm.user_id AND p.project_id = nm.project_id
                LEFT JOIN (
                    SELECT user_id, project_id, COUNT(*) as thread_count
                    FROM threads
                    GROUP BY user_id, project_id
                ) t ON p.user_id = t.user_id AND p.project_id = t.project_id
                WHERE p.user_id = ?
                ORDER BY p.updated_at DESC
                """,
                (user_id,)
            )
            rows = cursor.fetchall()

            projects = []
            for row in rows:
                created_at = row["created_at"]
                updated_at = row["updated_at"]

                # Parse timestamps
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if isinstance(updated_at, str):
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

                projects.append(Project(
                    id=row["project_id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    description=row["description"],
                    created_at=created_at,
                    updated_at=updated_at,
                    note_count=row["note_count"],
                    thread_count=row["thread_count"],
                ))

            return projects

        except Exception as e:
            logger.error(f"Failed to list projects for user {user_id}: {e}")
            return []
        finally:
            conn.close()

    def get_project(self, user_id: str, project_id: str) -> Optional[Project]:
        """
        Get a specific project.

        Args:
            user_id: User identifier
            project_id: Project identifier

        Returns:
            Project object or None if not found
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                """
                SELECT
                    p.user_id,
                    p.project_id,
                    p.name,
                    p.description,
                    p.created_at,
                    p.updated_at,
                    COALESCE(nm.note_count, 0) as note_count,
                    COALESCE(t.thread_count, 0) as thread_count
                FROM projects p
                LEFT JOIN (
                    SELECT user_id, project_id, COUNT(*) as note_count
                    FROM note_metadata
                    WHERE user_id = ? AND project_id = ?
                    GROUP BY user_id, project_id
                ) nm ON p.user_id = nm.user_id AND p.project_id = nm.project_id
                LEFT JOIN (
                    SELECT user_id, project_id, COUNT(*) as thread_count
                    FROM threads
                    WHERE user_id = ? AND project_id = ?
                    GROUP BY user_id, project_id
                ) t ON p.user_id = t.user_id AND p.project_id = t.project_id
                WHERE p.user_id = ? AND p.project_id = ?
                """,
                (user_id, project_id, user_id, project_id, user_id, project_id)
            )
            row = cursor.fetchone()

            if not row:
                return None

            created_at = row["created_at"]
            updated_at = row["updated_at"]

            # Parse timestamps
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

            return Project(
                id=row["project_id"],
                user_id=row["user_id"],
                name=row["name"],
                description=row["description"],
                created_at=created_at,
                updated_at=updated_at,
                note_count=row["note_count"],
                thread_count=row["thread_count"],
            )

        except Exception as e:
            logger.error(f"Failed to get project {project_id} for user {user_id}: {e}")
            return None
        finally:
            conn.close()

    def create_project(self, user_id: str, data: ProjectCreate) -> Project:
        """
        Create a new project.

        Args:
            user_id: User identifier
            data: Project creation data

        Returns:
            Created Project object

        Raises:
            ValueError: If project ID already exists
        """
        # Generate project ID from name if not provided
        project_id = data.id or generate_project_slug(data.name)

        # Ensure unique project ID
        if self.get_project(user_id, project_id):
            raise ValueError(f"Project with ID '{project_id}' already exists")

        now_iso = _utcnow_iso()
        conn = self.db.connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO projects (user_id, project_id, name, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, project_id, data.name, data.description, now_iso, now_iso)
                )

            # Create project vault directory
            vault_root = self.config.vault_base_path
            project_vault = vault_root / user_id / project_id
            project_vault.mkdir(parents=True, exist_ok=True)

            logger.info(f"Created project {project_id} for user {user_id}")

            return Project(
                id=project_id,
                user_id=user_id,
                name=data.name,
                description=data.description,
                created_at=datetime.fromisoformat(now_iso),
                updated_at=datetime.fromisoformat(now_iso),
                note_count=0,
                thread_count=0,
            )

        except Exception as e:
            logger.error(f"Failed to create project for user {user_id}: {e}")
            raise
        finally:
            conn.close()

    def update_project(self, user_id: str, project_id: str, data: ProjectUpdate) -> Optional[Project]:
        """
        Update an existing project.

        Args:
            user_id: User identifier
            project_id: Project identifier
            data: Project update data

        Returns:
            Updated Project object or None if not found
        """
        # Get current project
        current = self.get_project(user_id, project_id)
        if not current:
            return None

        # Apply updates
        new_name = data.name if data.name is not None else current.name
        new_description = data.description if data.description is not None else current.description
        now_iso = _utcnow_iso()

        conn = self.db.connect()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE projects
                    SET name = ?, description = ?, updated_at = ?
                    WHERE user_id = ? AND project_id = ?
                    """,
                    (new_name, new_description, now_iso, user_id, project_id)
                )

            logger.info(f"Updated project {project_id} for user {user_id}")

            return Project(
                id=project_id,
                user_id=user_id,
                name=new_name,
                description=new_description,
                created_at=current.created_at,
                updated_at=datetime.fromisoformat(now_iso),
                note_count=current.note_count,
                thread_count=current.thread_count,
            )

        except Exception as e:
            logger.error(f"Failed to update project {project_id} for user {user_id}: {e}")
            raise
        finally:
            conn.close()

    def delete_project(self, user_id: str, project_id: str) -> bool:
        """
        Delete a project and all its data.

        Args:
            user_id: User identifier
            project_id: Project identifier

        Returns:
            True if deleted, False if not found

        Note:
            This deletes all notes, threads, and context trees for the project.
            The 'default' project cannot be deleted.
        """
        if project_id == DEFAULT_PROJECT_ID:
            raise ValueError("Cannot delete the default project")

        # Check if project exists
        if not self.get_project(user_id, project_id):
            return False

        conn = self.db.connect()
        try:
            with conn:
                # Delete all related data (cascade)
                conn.execute(
                    "DELETE FROM note_metadata WHERE user_id = ? AND project_id = ?",
                    (user_id, project_id)
                )
                conn.execute(
                    "DELETE FROM note_fts WHERE user_id = ? AND project_id = ?",
                    (user_id, project_id)
                )
                conn.execute(
                    "DELETE FROM note_tags WHERE user_id = ? AND project_id = ?",
                    (user_id, project_id)
                )
                conn.execute(
                    "DELETE FROM note_links WHERE user_id = ? AND project_id = ?",
                    (user_id, project_id)
                )
                conn.execute(
                    "DELETE FROM index_health WHERE user_id = ? AND project_id = ?",
                    (user_id, project_id)
                )
                conn.execute(
                    "DELETE FROM threads WHERE user_id = ? AND project_id = ?",
                    (user_id, project_id)
                )
                conn.execute(
                    "DELETE FROM context_trees WHERE user_id = ? AND project_id = ?",
                    (user_id, project_id)
                )
                conn.execute(
                    "DELETE FROM context_nodes WHERE user_id = ? AND project_id = ?",
                    (user_id, project_id)
                )
                conn.execute(
                    "DELETE FROM projects WHERE user_id = ? AND project_id = ?",
                    (user_id, project_id)
                )

            # Delete project vault directory
            vault_root = self.config.vault_base_path
            project_vault = vault_root / user_id / project_id
            if project_vault.exists():
                import shutil
                shutil.rmtree(project_vault, ignore_errors=True)

            # Delete project RAG index directory
            rag_dir = self.config.llamaindex_persist_dir / user_id / project_id
            if rag_dir.exists():
                import shutil
                shutil.rmtree(rag_dir, ignore_errors=True)

            logger.info(f"Deleted project {project_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete project {project_id} for user {user_id}: {e}")
            raise
        finally:
            conn.close()

    def get_or_create_default(self, user_id: str) -> Project:
        """
        Get the default project, creating it if it doesn't exist.

        Args:
            user_id: User identifier

        Returns:
            Default Project object
        """
        project = self.get_project(user_id, DEFAULT_PROJECT_ID)
        if project:
            return project

        # Create default project
        return self.create_project(
            user_id,
            ProjectCreate(
                id=DEFAULT_PROJECT_ID,
                name="Default Project",
                description="Your default project"
            )
        )

    def get_project_stats(self, user_id: str, project_id: str) -> Optional[ProjectStats]:
        """
        Get detailed statistics for a project.

        Args:
            user_id: User identifier
            project_id: Project identifier

        Returns:
            ProjectStats object or None if project not found
        """
        conn = self.db.connect()
        try:
            # Check project exists
            cursor = conn.execute(
                "SELECT project_id FROM projects WHERE user_id = ? AND project_id = ?",
                (user_id, project_id)
            )
            if not cursor.fetchone():
                return None

            # Get note count and last update
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count, MAX(updated) as last_update
                FROM note_metadata
                WHERE user_id = ? AND project_id = ?
                """,
                (user_id, project_id)
            )
            note_row = cursor.fetchone()
            note_count = note_row["count"] if note_row else 0
            last_note_update = note_row["last_update"] if note_row else None

            # Get thread count and last update
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count, MAX(updated_at) as last_update
                FROM threads
                WHERE user_id = ? AND project_id = ?
                """,
                (user_id, project_id)
            )
            thread_row = cursor.fetchone()
            thread_count = thread_row["count"] if thread_row else 0
            last_thread_update = thread_row["last_update"] if thread_row else None

            # Parse timestamps
            if last_note_update and isinstance(last_note_update, str):
                last_note_update = datetime.fromisoformat(last_note_update.replace("Z", "+00:00"))
            if last_thread_update and isinstance(last_thread_update, str):
                last_thread_update = datetime.fromisoformat(last_thread_update.replace("Z", "+00:00"))

            return ProjectStats(
                project_id=project_id,
                note_count=note_count,
                thread_count=thread_count,
                last_note_update=last_note_update,
                last_thread_update=last_thread_update,
            )

        except Exception as e:
            logger.error(f"Failed to get stats for project {project_id}: {e}")
            return None
        finally:
            conn.close()

    def set_default_project(self, user_id: str, project_id: str) -> bool:
        """
        Set a project as the user's default.

        Args:
            user_id: User identifier
            project_id: Project identifier

        Returns:
            True if successful, False if project doesn't exist
        """
        # Verify project exists
        if not self.get_project(user_id, project_id):
            return False

        conn = self.db.connect()
        try:
            now_iso = _utcnow_iso()
            with conn:
                conn.execute(
                    """
                    INSERT INTO user_settings (user_id, default_project_id, created, updated)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        default_project_id = excluded.default_project_id,
                        updated = excluded.updated
                    """,
                    (user_id, project_id, now_iso, now_iso)
                )

            logger.info(f"Set default project to {project_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to set default project for user {user_id}: {e}")
            return False
        finally:
            conn.close()

    def get_default_project_id(self, user_id: str) -> str:
        """
        Get the user's default project ID.

        Args:
            user_id: User identifier

        Returns:
            Project ID (defaults to 'default' if not set)
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                "SELECT default_project_id FROM user_settings WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row and row["default_project_id"]:
                return row["default_project_id"]
            return DEFAULT_PROJECT_ID

        except Exception:
            return DEFAULT_PROJECT_ID
        finally:
            conn.close()


def get_project_service() -> ProjectService:
    """Get instance of ProjectService."""
    return ProjectService()


__all__ = ["ProjectService", "get_project_service"]
