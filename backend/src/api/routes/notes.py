"""HTTP API routes for note operations."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Query

from ...models.note import Note, NoteSummary, NoteUpdate, NoteCreate
from ...services.database import DatabaseService
from ...services.indexer import IndexerService
from ...services.vault import VaultService
from ..middleware import AuthContext, get_auth_context

router = APIRouter()


class ConflictError(Exception):
    """Raised when optimistic concurrency check fails."""

    def __init__(self, message: str = "Version conflict detected"):
        self.message = message
        super().__init__(self.message)


@router.get("/api/notes", response_model=list[NoteSummary])
async def list_notes(
    folder: Optional[str] = Query(None, description="Optional folder filter"),
    auth: AuthContext = Depends(get_auth_context),
):
    """List all notes in the vault."""
    user_id = auth.user_id
    vault_service = VaultService()
    
    try:
        notes = vault_service.list_notes(user_id, folder=folder)
        
        summaries = []
        for note in notes:
            # list_notes returns {path, title, last_modified}
            updated = note.get("last_modified")
            if not isinstance(updated, datetime):
                updated = datetime.now()
            
            summaries.append(
                NoteSummary(
                    note_path=note["path"],
                    title=note["title"],
                    updated=updated,
                )
            )
        return summaries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list notes: {str(e)}")


@router.post("/api/notes", response_model=Note, status_code=201)
async def create_note(create: NoteCreate, auth: AuthContext = Depends(get_auth_context)):
    """Create a new note."""
    user_id = auth.user_id
    vault_service = VaultService()
    indexer_service = IndexerService()
    db_service = DatabaseService()
    
    try:
        note_path = create.note_path

        # Check if note already exists
        try:
            vault_service.read_note(user_id, note_path)
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "note_already_exists",
                    "message": f"A note with the name '{note_path}' already exists. Please choose a different name.",
                }
            )
        except FileNotFoundError:
            pass  # Good, note doesn't exist
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        
        # Prepare metadata
        metadata = create.metadata.model_dump() if create.metadata else {}
        if create.title:
            metadata["title"] = create.title
        
        # Write note to vault
        written_note = vault_service.write_note(
            user_id,
            note_path,
            body=create.body,
            metadata=metadata,
            title=create.title
        )
        
        # Index the note
        new_version = indexer_service.index_note(user_id, written_note)
        
        # Update index health
        conn = db_service.connect()
        try:
            with conn:
                indexer_service.update_index_health(conn, user_id)
        finally:
            conn.close()
        
        # Return created note
        created = written_note["metadata"].get("created")
        updated_ts = written_note["metadata"].get("updated")

        # Parse created timestamp
        try:
            if isinstance(created, str):
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            elif isinstance(created, datetime):
                pass  # Already a datetime
            else:
                created = datetime.now()
        except (ValueError, TypeError):
            created = datetime.now()

        # Parse updated timestamp
        try:
            if isinstance(updated_ts, str):
                updated_ts = datetime.fromisoformat(updated_ts.replace("Z", "+00:00"))
            elif isinstance(updated_ts, datetime):
                pass  # Already a datetime
            else:
                updated_ts = created
        except (ValueError, TypeError):
            updated_ts = created
        
        return Note(
            user_id=user_id,
            note_path=note_path,
            version=new_version,
            title=written_note["title"],
            metadata=written_note["metadata"],
            body=written_note["body"],
            created=created,
            updated=updated_ts,
            size_bytes=written_note.get("size_bytes", len(written_note["body"].encode("utf-8"))),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create note: {str(e)}")


@router.get("/api/notes/{path:path}", response_model=Note)
async def get_note(path: str, auth: AuthContext = Depends(get_auth_context)):
    """Get a specific note by path."""
    user_id = auth.user_id
    vault_service = VaultService()
    db_service = DatabaseService()
    
    try:
        # URL decode the path
        note_path = unquote(path)
        
        # Read note from vault
        note_data = vault_service.read_note(user_id, note_path)
        
        # Get version from index
        conn = db_service.connect()
        try:
            cursor = conn.execute(
                "SELECT version FROM note_metadata WHERE user_id = ? AND note_path = ?",
                (user_id, note_path),
            )
            row = cursor.fetchone()
            version = row["version"] if row else 1
        finally:
            conn.close()
        
        # Parse metadata
        metadata = note_data.get("metadata", {})
        created = metadata.get("created")
        updated = metadata.get("updated")

        # Parse created timestamp
        try:
            if isinstance(created, str):
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            elif isinstance(created, datetime):
                pass  # Already a datetime
            else:
                created = datetime.now()
        except (ValueError, TypeError):
            created = datetime.now()

        # Parse updated timestamp
        try:
            if isinstance(updated, str):
                updated = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            elif isinstance(updated, datetime):
                pass  # Already a datetime
            else:
                updated = created
        except (ValueError, TypeError):
            updated = created
        
        return Note(
            user_id=user_id,
            note_path=note_path,
            version=version,
            title=note_data["title"],
            metadata=metadata,
            body=note_data["body"],
            created=created,
            updated=updated,
            size_bytes=note_data.get("size_bytes", len(note_data["body"].encode("utf-8"))),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Note not found: {path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read note: {str(e)}")


@router.put("/api/notes/{path:path}", response_model=Note)
async def update_note(
    path: str,
    update: NoteUpdate,
    auth: AuthContext = Depends(get_auth_context),
):
    """Update a note with optimistic concurrency control."""
    user_id = auth.user_id
    vault_service = VaultService()
    indexer_service = IndexerService()
    db_service = DatabaseService()
    
    try:
        # URL decode the path
        note_path = unquote(path)
        
        # Check version if provided
        if update.if_version is not None:
            conn = db_service.connect()
            try:
                cursor = conn.execute(
                    "SELECT version FROM note_metadata WHERE user_id = ? AND note_path = ?",
                    (user_id, note_path),
                )
                row = cursor.fetchone()
                current_version = row["version"] if row else 0
                
                if current_version != update.if_version:
                    raise ConflictError(
                        f"Version conflict: expected {update.if_version}, got {current_version}"
                    )
            finally:
                conn.close()
        
        # Prepare metadata
        metadata = update.metadata.model_dump() if update.metadata else {}
        if update.title:
            metadata["title"] = update.title
        
        # Write note to vault
        written_note = vault_service.write_note(
            user_id, 
            note_path, 
            body=update.body, 
            metadata=metadata,
            title=update.title
        )
        
        # Index the note
        new_version = indexer_service.index_note(user_id, written_note)
        
        # Update index health
        conn = db_service.connect()
        try:
            with conn:
                indexer_service.update_index_health(conn, user_id)
        finally:
            conn.close()
        
        # Return updated note
        created = written_note["metadata"].get("created")
        updated_ts = written_note["metadata"].get("updated")

        # Parse created timestamp
        try:
            if isinstance(created, str):
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            elif isinstance(created, datetime):
                pass  # Already a datetime
            else:
                created = datetime.now()
        except (ValueError, TypeError):
            created = datetime.now()

        # Parse updated timestamp
        try:
            if isinstance(updated_ts, str):
                updated_ts = datetime.fromisoformat(updated_ts.replace("Z", "+00:00"))
            elif isinstance(updated_ts, datetime):
                pass  # Already a datetime
            else:
                updated_ts = created
        except (ValueError, TypeError):
            updated_ts = created
        
        return Note(
            user_id=user_id,
            note_path=note_path,
            version=new_version,
            title=written_note["title"],
            metadata=written_note["metadata"],
            body=written_note["body"],
            created=created,
            updated=updated_ts,
            size_bytes=written_note.get("size_bytes", len(written_note["body"].encode("utf-8"))),
        )
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Note not found: {path}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update note: {str(e)}")


from pydantic import BaseModel


class NoteMoveRequest(BaseModel):
    """Request payload for moving/renaming a note."""
    new_path: str


@router.patch("/api/notes/{path:path}", response_model=Note)
async def move_note(path: str, move_request: NoteMoveRequest):
    """Move or rename a note to a new path."""
    user_id = get_user_id()
    vault_service = VaultService()
    indexer_service = IndexerService()
    db_service = DatabaseService()

    try:
        # URL decode the old path
        old_path = unquote(path)
        new_path = move_request.new_path

        # Move the note in the vault
        moved_note = vault_service.move_note(user_id, old_path, new_path)

        # Delete old note index entries
        conn = db_service.connect()
        try:
            with conn:
                # Delete from all index tables
                conn.execute("DELETE FROM note_metadata WHERE user_id = ? AND note_path = ?", (user_id, old_path))
                conn.execute("DELETE FROM note_links WHERE user_id = ? AND source_path = ?", (user_id, old_path))
                conn.execute("DELETE FROM note_tags WHERE user_id = ? AND note_path = ?", (user_id, old_path))
        finally:
            conn.close()

        # Index the note at new location
        new_version = indexer_service.index_note(user_id, moved_note)

        # Update index health
        conn = db_service.connect()
        try:
            with conn:
                indexer_service.update_index_health(conn, user_id)
        finally:
            conn.close()

        # Parse metadata
        metadata = moved_note.get("metadata", {})
        created = metadata.get("created")
        updated = metadata.get("updated")

        # Parse created timestamp
        try:
            if isinstance(created, str):
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            elif isinstance(created, datetime):
                pass  # Already a datetime
            else:
                created = datetime.now()
        except (ValueError, TypeError):
            created = datetime.now()

        # Parse updated timestamp
        try:
            if isinstance(updated, str):
                updated = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            elif isinstance(updated, datetime):
                pass  # Already a datetime
            else:
                updated = created
        except (ValueError, TypeError):
            updated = created

        return Note(
            user_id=user_id,
            note_path=new_path,
            version=new_version,
            title=moved_note["title"],
            metadata=metadata,
            body=moved_note["body"],
            created=created,
            updated=updated,
            size_bytes=moved_note.get("size_bytes", len(moved_note["body"].encode("utf-8"))),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Note not found: {path}")
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to move note: {str(e)}")


__all__ = ["router", "ConflictError"]

