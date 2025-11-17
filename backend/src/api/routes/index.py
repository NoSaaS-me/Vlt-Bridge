"""HTTP API routes for index operations."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...models.index import IndexHealth
from ...services.database import DatabaseService
from ...services.indexer import IndexerService
from ...services.vault import VaultService

router = APIRouter()


class RebuildResponse(BaseModel):
    """Response from index rebuild."""

    status: str
    notes_indexed: int


def get_user_id() -> str:
    """Return the current user ID. For now, hardcoded to 'local-dev'."""
    return "local-dev"


@router.get("/api/index/health", response_model=IndexHealth)
async def get_index_health():
    """Get index health statistics."""
    user_id = get_user_id()
    db_service = DatabaseService()
    
    try:
        conn = db_service.connect()
        try:
            cursor = conn.execute(
                """
                SELECT note_count, last_full_rebuild, last_incremental_update
                FROM index_health
                WHERE user_id = ?
                """,
                (user_id,),
            )
            row = cursor.fetchone()
            
            if not row:
                # Initialize if not exists
                return IndexHealth(
                    user_id=user_id,
                    note_count=0,
                    last_full_rebuild=None,
                    last_incremental_update=None,
                )
            
            last_full_rebuild = row["last_full_rebuild"]
            last_incremental_update = row["last_incremental_update"]
            
            if last_full_rebuild and isinstance(last_full_rebuild, str):
                last_full_rebuild = datetime.fromisoformat(last_full_rebuild.replace("Z", "+00:00"))
            
            if last_incremental_update and isinstance(last_incremental_update, str):
                last_incremental_update = datetime.fromisoformat(last_incremental_update.replace("Z", "+00:00"))
            
            return IndexHealth(
                user_id=user_id,
                note_count=row["note_count"],
                last_full_rebuild=last_full_rebuild,
                last_incremental_update=last_incremental_update,
            )
        finally:
            conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get index health: {str(e)}")


@router.post("/api/index/rebuild", response_model=RebuildResponse)
async def rebuild_index():
    """Rebuild the entire index from scratch."""
    user_id = get_user_id()
    vault_service = VaultService()
    indexer_service = IndexerService()
    
    try:
        # Get all notes
        notes = vault_service.list_notes(user_id)
        
        # Clear existing index entries
        db_service = DatabaseService()
        conn = db_service.connect()
        try:
            with conn:
                conn.execute("DELETE FROM note_metadata WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM note_fts WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM note_tags WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM note_links WHERE user_id = ?", (user_id,))
        finally:
            conn.close()
        
        # Re-index all notes
        indexed_count = 0
        for note in notes:
            try:
                note_data = vault_service.read_note(user_id, note["path"])
                indexer_service.index_note(user_id, note_data)
                indexed_count += 1
            except Exception as e:
                print(f"Failed to index {note['path']}: {e}")
        
        # Update index health
        conn = db_service.connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO index_health (user_id, note_count, last_full_rebuild, last_incremental_update)
                    VALUES (?, ?, datetime('now'), datetime('now'))
                    ON CONFLICT(user_id) DO UPDATE SET
                        note_count = excluded.note_count,
                        last_full_rebuild = excluded.last_full_rebuild,
                        last_incremental_update = excluded.last_incremental_update
                    """,
                    (user_id, indexed_count),
                )
        finally:
            conn.close()
        
        return RebuildResponse(
            status="completed",
            notes_indexed=indexed_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")


__all__ = ["router", "RebuildResponse"]

