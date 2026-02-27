"""HTTP API routes for asset (non-markdown file) operations."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from ...models.asset import AssetMetadata, AssetMoveRequest, AssetSearchResult, AssetSummary, AssetUploadResponse
from ...models.project import DEFAULT_PROJECT_ID
from ...services.asset_indexer import AssetIndexer
from ...services.asset_vault import AssetVaultService, MAX_ASSET_BYTES, validate_asset_path
from ...services.auth import AuthService
from ..middleware import AuthContext, require_auth_context


_auth_service = AuthService()


def _require_asset_auth(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
    token: Optional[str] = Query(None, description="JWT token (for <img>/<video> src URLs)"),
) -> AuthContext:
    """Auth dependency that accepts Bearer header OR ?token= query param.

    <img src>, <video src>, and <audio src> elements cannot send Authorization
    headers, so we also accept a ?token= query parameter for media serving.
    """
    raw = token or authorization
    if not raw:
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "message": "Authorization required"},
        )
    # Strip "Bearer " prefix if present (header form)
    if raw.lower().startswith("bearer "):
        raw = raw[7:]
    try:
        payload = _auth_service.validate_jwt(raw)
    except Exception:
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "message": "Invalid or expired token"},
        )
    return AuthContext(user_id=payload.sub, token=raw, payload=payload)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_vault() -> AssetVaultService:
    return AssetVaultService()


def _get_indexer() -> AssetIndexer:
    return AssetIndexer()


@router.post("/api/assets/upload", response_model=AssetUploadResponse, status_code=201)
async def upload_asset(
    file: UploadFile,
    path: str = Form(..., description="Relative asset path (e.g. 'images/photo.jpg')"),
    project_id: str = Form(DEFAULT_PROJECT_ID, description="Project ID"),
    auth: AuthContext = Depends(require_auth_context),
):
    """Upload a file asset to the vault.

    Validates the path, writes the file, and triggers background OCR for
    PDFs and images.
    """
    user_id = auth.user_id

    # Validate path
    is_valid, msg = validate_asset_path(path)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    # Read file bytes (memory-limited)
    data = await file.read()
    if len(data) > MAX_ASSET_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(data)} bytes (max {MAX_ASSET_BYTES} bytes)",
        )

    # Detect MIME type from upload or fallback to extension
    mime_type = file.content_type
    if not mime_type or mime_type == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(path)
        if guessed:
            mime_type = guessed
    if not mime_type:
        mime_type = "application/octet-stream"

    vault = _get_vault()
    indexer = _get_indexer()

    try:
        write_result = vault.write_asset(user_id, project_id, path, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to write asset %s for user %s", path, user_id)
        raise HTTPException(status_code=500, detail="Failed to store asset")

    full_path = vault.get_full_path(user_id, project_id, path)
    ocr_status = indexer.index_asset(
        user_id=user_id,
        project_id=project_id,
        asset_path=path,
        mime_type=mime_type,
        file_size=len(data),
        full_path=full_path,
    )

    return AssetUploadResponse(
        asset_path=path,
        mime_type=mime_type,
        file_size=len(data),
        ocr_status=ocr_status,
        created=write_result["created"],
    )


@router.get("/api/assets", response_model=list[AssetSummary])
async def list_assets(
    folder: Optional[str] = Query(None, description="Optional folder filter"),
    project_id: str = Query(DEFAULT_PROJECT_ID, description="Project ID"),
    auth: AuthContext = Depends(require_auth_context),
):
    """List all assets in the project vault, optionally scoped to a folder.

    Returns AssetSummary objects. For accurate OCR status, fetch metadata per-asset.
    """
    user_id = auth.user_id
    vault = _get_vault()
    indexer = _get_indexer()

    try:
        assets = vault.list_assets(user_id, project_id, folder=folder)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to list assets for user %s", user_id)
        raise HTTPException(status_code=500, detail="Failed to list assets")

    # Merge real ocr_status from index where available
    enriched: list[AssetSummary] = []
    for asset in assets:
        meta = indexer.get_asset_metadata(user_id, project_id, asset.asset_path)
        ocr_status = meta["ocr_status"] if meta else asset.ocr_status  # type: ignore[assignment]
        updated = meta["updated"] if meta else asset.updated
        enriched.append(
            AssetSummary(
                asset_path=asset.asset_path,
                mime_type=asset.mime_type,
                file_size=asset.file_size,
                ocr_status=ocr_status,
                updated=updated,
            )
        )
    return enriched


@router.get("/api/assets/{asset_path:path}/metadata", response_model=AssetMetadata)
async def get_asset_metadata(
    asset_path: str,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    auth: AuthContext = Depends(require_auth_context),
):
    """Get metadata (including OCR status and text) for an asset."""
    user_id = auth.user_id

    is_valid, msg = validate_asset_path(asset_path)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    vault = _get_vault()
    indexer = _get_indexer()

    # Verify file exists
    try:
        full_path = vault.get_full_path(user_id, project_id, asset_path)
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")
        stat = full_path.stat()
    except (ValueError, HTTPException):
        raise
    except Exception as exc:
        logger.exception("Failed to stat asset %s for user %s", asset_path, user_id)
        raise HTTPException(status_code=500, detail="Failed to get asset info")

    meta = indexer.get_asset_metadata(user_id, project_id, asset_path)
    if meta:
        return AssetMetadata(
            asset_path=asset_path,
            mime_type=meta["mime_type"],
            file_size=meta["file_size"],
            ocr_status=meta["ocr_status"],  # type: ignore[arg-type]
            created=meta["created"],
            updated=meta["updated"],
            ocr_text=meta.get("ocr_text"),
        )

    # Fallback: not yet indexed — derive from filesystem
    from datetime import datetime, timezone
    mime_type, _ = mimetypes.guess_type(asset_path)
    mime_type = mime_type or "application/octet-stream"
    now_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    return AssetMetadata(
        asset_path=asset_path,
        mime_type=mime_type,
        file_size=stat.st_size,
        ocr_status="skipped",
        created=now_iso,
        updated=now_iso,
    )


@router.get("/api/assets/{asset_path:path}")
async def get_asset(
    asset_path: str,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    auth: AuthContext = Depends(_require_asset_auth),
):
    """Stream the raw asset file with appropriate content type."""
    user_id = auth.user_id

    is_valid, msg = validate_asset_path(asset_path)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    vault = _get_vault()
    try:
        full_path = vault.get_full_path(user_id, project_id, asset_path)
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")
    except (ValueError, HTTPException):
        raise
    except Exception as exc:
        logger.exception("Failed to resolve asset path %s for user %s", asset_path, user_id)
        raise HTTPException(status_code=500, detail="Failed to resolve asset")

    mime_type, _ = mimetypes.guess_type(asset_path)
    mime_type = mime_type or "application/octet-stream"
    filename = Path(asset_path).name

    return FileResponse(
        path=str(full_path),
        media_type=mime_type,
        filename=filename,
    )


@router.put("/api/assets/{asset_path:path}", response_model=AssetSummary)
async def update_asset(
    asset_path: str,
    request: Request,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    auth: AuthContext = Depends(require_auth_context),
):
    """Update the content of an existing text asset (HTML, CSV, TXT).

    Accepts raw body bytes with any text/* content type. Does not re-trigger OCR.
    """
    user_id = auth.user_id

    is_valid, msg = validate_asset_path(asset_path)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    data = await request.body()
    if len(data) > MAX_ASSET_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(data)} bytes (max {MAX_ASSET_BYTES} bytes)",
        )

    vault = _get_vault()
    indexer = _get_indexer()

    # Verify file exists before allowing update
    try:
        full_path = vault.get_full_path(user_id, project_id, asset_path)
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")
    except (ValueError, HTTPException):
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to resolve asset")

    try:
        vault.write_asset(user_id, project_id, asset_path, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Failed to update asset %s for user %s", asset_path, user_id)
        raise HTTPException(status_code=500, detail="Failed to update asset")

    mime_type, _ = mimetypes.guess_type(asset_path)
    mime_type = mime_type or "application/octet-stream"
    stat = full_path.stat()

    # Update index size/timestamp; preserve existing ocr_text and status
    meta = indexer.get_asset_metadata(user_id, project_id, asset_path)
    ocr_status = meta["ocr_status"] if meta else "skipped"  # type: ignore[assignment]

    from datetime import datetime, timezone
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return AssetSummary(
        asset_path=asset_path,
        mime_type=mime_type,
        file_size=len(data),
        ocr_status=ocr_status,
        updated=updated,
    )


@router.delete("/api/assets/{asset_path:path}", status_code=204)
async def delete_asset(
    asset_path: str,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    auth: AuthContext = Depends(require_auth_context),
):
    """Delete an asset and remove it from the index."""
    user_id = auth.user_id

    is_valid, msg = validate_asset_path(asset_path)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    vault = _get_vault()
    indexer = _get_indexer()

    try:
        vault.delete_asset(user_id, project_id, asset_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to delete asset %s for user %s", asset_path, user_id)
        raise HTTPException(status_code=500, detail="Failed to delete asset")

    indexer.delete_asset_index(user_id, project_id, asset_path)


@router.patch("/api/assets/{asset_path:path}", response_model=AssetSummary)
async def move_asset(
    asset_path: str,
    body: AssetMoveRequest,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    auth: AuthContext = Depends(require_auth_context),
):
    """Move or rename an asset to a new path within the same project."""
    user_id = auth.user_id

    is_valid, msg = validate_asset_path(asset_path)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid source path: {msg}")

    is_valid_new, msg_new = validate_asset_path(body.new_path)
    if not is_valid_new:
        raise HTTPException(status_code=400, detail=f"Invalid destination path: {msg_new}")

    vault = _get_vault()
    indexer = _get_indexer()

    try:
        vault.move_asset(user_id, project_id, asset_path, body.new_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")
    except FileExistsError:
        raise HTTPException(status_code=409, detail=f"Destination already exists: {body.new_path}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to move asset %s for user %s", asset_path, user_id)
        raise HTTPException(status_code=500, detail="Failed to move asset")

    # Update index: re-index under new path, remove old index entry
    new_full_path = vault.get_full_path(user_id, project_id, body.new_path)
    stat = new_full_path.stat()
    mime_type, _ = mimetypes.guess_type(body.new_path)
    mime_type = mime_type or "application/octet-stream"

    indexer.delete_asset_index(user_id, project_id, asset_path)
    ocr_status = indexer.index_asset(
        user_id=user_id,
        project_id=project_id,
        asset_path=body.new_path,
        mime_type=mime_type,
        file_size=stat.st_size,
        full_path=new_full_path,
    )

    from datetime import datetime, timezone
    updated = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    return AssetSummary(
        asset_path=body.new_path,
        mime_type=mime_type,
        file_size=stat.st_size,
        ocr_status=ocr_status,
        updated=updated,
    )
