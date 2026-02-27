"""AssetIndexer - SQLite-backed indexing and OCR for asset files.

Supports background OCR via asyncio tasks for PDF and image files.
OCR requires pytesseract and pdf2image; if unavailable, ocr_status is 'failed'.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .database import DatabaseService
from ..models.asset import AssetSearchResult, OcrStatus
from ..models.project import DEFAULT_PROJECT_ID
from .indexer import _prepare_match_query  # re-use token sanitizer

logger = logging.getLogger(__name__)

# Check optional OCR dependencies once at import time
try:
    import pytesseract  # type: ignore
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False
    logger.warning(
        "pytesseract not installed — OCR will be unavailable. "
        "Install with: uv pip install pytesseract"
    )

try:
    from pdf2image import convert_from_path  # type: ignore
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    _PDF2IMAGE_AVAILABLE = False
    logger.warning(
        "pdf2image not installed — PDF OCR will be unavailable. "
        "Install with: uv pip install pdf2image"
    )

try:
    from PIL import Image  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_ocr_able(mime_type: str) -> bool:
    """Return True if this MIME type supports OCR extraction."""
    return mime_type.startswith("image/") or mime_type == "application/pdf"


def _do_ocr(full_path: Path, mime_type: str) -> Optional[str]:
    """Synchronous OCR. Returns extracted text or None on failure/unsupported.

    - PDF: convert pages to images via pdf2image, then OCR each page.
    - Image: run pytesseract directly.
    """
    if not _TESSERACT_AVAILABLE or not _PIL_AVAILABLE:
        logger.warning("OCR dependencies missing — cannot OCR %s", full_path)
        return None

    try:
        if mime_type == "application/pdf":
            if not _PDF2IMAGE_AVAILABLE:
                logger.warning("pdf2image missing — cannot OCR PDF %s", full_path)
                return None
            pages = convert_from_path(str(full_path))
            page_texts = [pytesseract.image_to_string(page) for page in pages]
            return "\n\n".join(page_texts).strip() or None

        if mime_type.startswith("image/"):
            text = pytesseract.image_to_string(Image.open(str(full_path)))
            return text.strip() or None

    except Exception as exc:
        logger.warning("OCR failed for %s: %s", full_path, exc)
        return None

    return None


class AssetIndexer:
    """Manage SQLite-backed asset metadata, FTS, and background OCR."""

    def __init__(self, db_service: DatabaseService | None = None) -> None:
        self.db_service = db_service or DatabaseService()

    def index_asset(
        self,
        user_id: str,
        project_id: str,
        asset_path: str,
        mime_type: str,
        file_size: int,
        full_path: Path,
    ) -> OcrStatus:
        """Insert or update asset_metadata for the given asset.

        Schedules background OCR if the file type supports it and OCR libs
        are available.  Returns the initial ocr_status.
        """
        now_iso = _utcnow_iso()

        if _is_ocr_able(mime_type) and (_TESSERACT_AVAILABLE or mime_type == "application/pdf"):
            initial_status: OcrStatus = "pending"
        else:
            initial_status = "skipped"

        conn = self.db_service.connect()
        try:
            with conn:
                # Check existing created timestamp to preserve it on update
                existing = conn.execute(
                    "SELECT created FROM asset_metadata WHERE user_id = ? AND project_id = ? AND asset_path = ?",
                    (user_id, project_id, asset_path),
                ).fetchone()
                created_at = existing["created"] if existing else now_iso

                conn.execute(
                    """
                    INSERT INTO asset_metadata
                        (user_id, project_id, asset_path, mime_type, file_size, ocr_status, created, updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, project_id, asset_path) DO UPDATE SET
                        mime_type = excluded.mime_type,
                        file_size = excluded.file_size,
                        ocr_status = excluded.ocr_status,
                        updated = excluded.updated
                    """,
                    (user_id, project_id, asset_path, mime_type, file_size, initial_status, created_at, now_iso),
                )

                # Update FTS with filename (OCR text is empty until OCR completes)
                self._update_asset_fts(conn, user_id, project_id, asset_path, ocr_text=None)
        finally:
            conn.close()

        # Schedule background OCR if warranted
        if initial_status == "pending":
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        self._run_ocr_background(user_id, project_id, asset_path, full_path, mime_type)
                    )
                else:
                    logger.debug("No running event loop for OCR task on %s", asset_path)
            except RuntimeError:
                logger.debug("Could not schedule OCR background task for %s", asset_path)

        logger.info(
            "Asset indexed",
            extra={
                "user_id": user_id,
                "project_id": project_id,
                "asset_path": asset_path,
                "ocr_status": initial_status,
            },
        )
        return initial_status

    async def _run_ocr_background(
        self,
        user_id: str,
        project_id: str,
        asset_path: str,
        full_path: Path,
        mime_type: str,
    ) -> None:
        """Async wrapper: runs OCR in a thread, updates DB with result."""
        conn = self.db_service.connect()
        try:
            with conn:
                conn.execute(
                    "UPDATE asset_metadata SET ocr_status = 'running', updated = ? "
                    "WHERE user_id = ? AND project_id = ? AND asset_path = ?",
                    (_utcnow_iso(), user_id, project_id, asset_path),
                )
        finally:
            conn.close()

        try:
            ocr_text = await asyncio.to_thread(_do_ocr, full_path, mime_type)
        except Exception as exc:
            logger.warning("OCR background task failed for %s: %s", asset_path, exc)
            ocr_text = None

        final_status: OcrStatus = "done" if ocr_text is not None else "failed"
        conn = self.db_service.connect()
        try:
            with conn:
                conn.execute(
                    "UPDATE asset_metadata SET ocr_text = ?, ocr_status = ?, updated = ? "
                    "WHERE user_id = ? AND project_id = ? AND asset_path = ?",
                    (ocr_text, final_status, _utcnow_iso(), user_id, project_id, asset_path),
                )
                self._update_asset_fts(conn, user_id, project_id, asset_path, ocr_text=ocr_text)
        finally:
            conn.close()

        logger.info(
            "OCR complete",
            extra={
                "user_id": user_id,
                "project_id": project_id,
                "asset_path": asset_path,
                "ocr_status": final_status,
                "ocr_chars": len(ocr_text) if ocr_text else 0,
            },
        )

    def _update_asset_fts(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        project_id: str,
        asset_path: str,
        ocr_text: Optional[str],
    ) -> None:
        """Update or insert the FTS row for an asset (filename + ocr_text)."""
        from pathlib import Path as _Path
        filename = _Path(asset_path).name

        # Delete old FTS row
        conn.execute(
            "DELETE FROM asset_fts WHERE user_id = ? AND project_id = ? AND asset_path = ?",
            (user_id, project_id, asset_path),
        )
        # Re-insert with latest data
        conn.execute(
            "INSERT INTO asset_fts (user_id, project_id, asset_path, filename, ocr_text) VALUES (?, ?, ?, ?, ?)",
            (user_id, project_id, asset_path, filename, ocr_text or ""),
        )

    def get_asset_metadata(
        self, user_id: str, project_id: str, asset_path: str
    ) -> Optional[dict]:
        """Fetch asset metadata from DB. Returns None if not found."""
        conn = self.db_service.connect()
        try:
            row = conn.execute(
                "SELECT * FROM asset_metadata WHERE user_id = ? AND project_id = ? AND asset_path = ?",
                (user_id, project_id, asset_path),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def delete_asset_index(
        self, user_id: str, project_id: str, asset_path: str
    ) -> None:
        """Remove index entries for a deleted asset."""
        conn = self.db_service.connect()
        try:
            with conn:
                conn.execute(
                    "DELETE FROM asset_metadata WHERE user_id = ? AND project_id = ? AND asset_path = ?",
                    (user_id, project_id, asset_path),
                )
                conn.execute(
                    "DELETE FROM asset_fts WHERE user_id = ? AND project_id = ? AND asset_path = ?",
                    (user_id, project_id, asset_path),
                )
        finally:
            conn.close()

    def search_assets(
        self,
        user_id: str,
        project_id: str,
        query: str,
        limit: int = 20,
    ) -> list[AssetSearchResult]:
        """FTS5 search over asset filename and OCR text."""
        sanitized_query = _prepare_match_query(query)

        conn = self.db_service.connect()
        try:
            rows = conn.execute(
                """
                SELECT
                    f.asset_path,
                    m.mime_type,
                    f.filename,
                    snippet(asset_fts, 3, '<mark>', '</mark>', '...', 32) AS snippet,
                    bm25(asset_fts, 3.0, 1.0) AS score
                FROM asset_fts f
                JOIN asset_metadata m USING (user_id, project_id, asset_path)
                WHERE f.user_id = ? AND f.project_id = ? AND asset_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (user_id, project_id, sanitized_query, limit),
            ).fetchall()
        finally:
            conn.close()

        results: list[AssetSearchResult] = []
        for row in rows:
            if isinstance(row, sqlite3.Row):
                results.append(
                    AssetSearchResult(
                        asset_path=row["asset_path"],
                        mime_type=row["mime_type"],
                        filename=row["filename"],
                        snippet=row["snippet"] or "",
                        score=float(row["score"]),
                    )
                )
            else:
                results.append(
                    AssetSearchResult(
                        asset_path=row[0],
                        mime_type=row[1],
                        filename=row[2],
                        snippet=row[3] or "",
                        score=float(row[4]),
                    )
                )
        return results


__all__ = ["AssetIndexer"]
