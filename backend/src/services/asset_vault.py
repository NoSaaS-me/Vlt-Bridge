"""AssetVault - filesystem storage for binary and text asset files.

Assets are non-markdown files: PDFs, images, audio, video, HTML, CSV, TXT.
Stored in the same vault directory structure as notes but tracked separately.
"""

from __future__ import annotations

import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import logging

from ..models.asset import AssetSummary
from .config import AppConfig, get_config
from ..models.project import DEFAULT_PROJECT_ID

logger = logging.getLogger(__name__)

MAX_ASSET_BYTES = 200 * 1024 * 1024  # 200 MB

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".mp3", ".wav", ".ogg", ".flac",
    ".mp4", ".webm", ".mov",
    ".html", ".htm", ".csv", ".txt",
}

INVALID_PATH_CHARS = {'<', '>', ':', '"', '|', '?', '*', '\\'}


def validate_asset_path(asset_path: str) -> tuple[bool, str]:
    """Validate a relative asset path.

    Returns (is_valid, message). Message is empty when valid.
    """
    if not asset_path or len(asset_path) > 256:
        return False, "Path must be 1-256 characters"
    if ".." in asset_path:
        return False, "Path must not contain '..'"
    if "\\" in asset_path:
        return False, "Path must use Unix separators (/)"
    if asset_path.startswith("/"):
        return False, "Path must be relative (no leading /)"
    if any(char in INVALID_PATH_CHARS for char in asset_path):
        return False, "Path contains invalid characters"
    suffix = Path(asset_path).suffix.lower()
    if not suffix:
        return False, "Path must have a file extension"
    if suffix not in ALLOWED_EXTENSIONS:
        return False, f"Extension '{suffix}' not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
    return True, ""


class AssetVaultService:
    """Service for managing asset (non-markdown) file storage."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.vault_root = self.config.vault_base_path
        self.vault_root.mkdir(parents=True, exist_ok=True)

    def _asset_root(self, user_id: str, project_id: str = DEFAULT_PROJECT_ID) -> Path:
        """Return the vault directory for a user/project. Mirrors VaultService._vault_path pattern."""
        path = (self.vault_root / user_id / project_id).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _asset_file_path(
        self, user_id: str, project_id: str, asset_path: str
    ) -> Path:
        """Resolve and security-check an asset path within the user's project vault.

        Raises ValueError if the resolved path escapes the vault root.
        """
        root = self._asset_root(user_id, project_id)
        full_path = (root / asset_path).resolve()
        try:
            common = os.path.commonpath([str(root), str(full_path)])
        except ValueError as exc:
            raise ValueError(f"Path escapes vault root: {asset_path}") from exc
        if common != str(root):
            raise ValueError(f"Path escapes vault root: {asset_path}")
        return full_path

    def write_asset(
        self,
        user_id: str,
        project_id: str,
        asset_path: str,
        data: bytes,
    ) -> dict:
        """Write asset bytes to the vault. Creates parent directories as needed.

        Returns a dict with path, file_size, and timestamps.
        """
        is_valid, msg = validate_asset_path(asset_path)
        if not is_valid:
            raise ValueError(msg)

        full_path = self._asset_file_path(user_id, project_id, asset_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)

        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.info(
            "Asset written",
            extra={
                "user_id": user_id,
                "project_id": project_id,
                "asset_path": asset_path,
                "file_size": len(data),
            },
        )
        return {
            "path": asset_path,
            "file_size": len(data),
            "created": now_iso,
            "updated": now_iso,
        }

    def read_asset(self, user_id: str, project_id: str, asset_path: str) -> bytes:
        """Read raw bytes for an asset file."""
        is_valid, msg = validate_asset_path(asset_path)
        if not is_valid:
            raise ValueError(msg)

        full_path = self._asset_file_path(user_id, project_id, asset_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Asset not found: {asset_path}")
        return full_path.read_bytes()

    def delete_asset(self, user_id: str, project_id: str, asset_path: str) -> None:
        """Delete an asset file from the vault."""
        is_valid, msg = validate_asset_path(asset_path)
        if not is_valid:
            raise ValueError(msg)

        full_path = self._asset_file_path(user_id, project_id, asset_path)
        try:
            full_path.unlink()
            logger.info(
                "Asset deleted",
                extra={"user_id": user_id, "project_id": project_id, "asset_path": asset_path},
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Asset not found: {asset_path}") from exc

    def move_asset(
        self,
        user_id: str,
        project_id: str,
        old_path: str,
        new_path: str,
    ) -> None:
        """Move or rename an asset to a new path within the same project."""
        for p, label in [(old_path, "source"), (new_path, "destination")]:
            is_valid, msg = validate_asset_path(p)
            if not is_valid:
                raise ValueError(f"Invalid {label} path: {msg}")

        old_full = self._asset_file_path(user_id, project_id, old_path)
        new_full = self._asset_file_path(user_id, project_id, new_path)

        if not old_full.exists():
            raise FileNotFoundError(f"Source asset not found: {old_path}")
        if new_full.exists():
            raise FileExistsError(f"Destination asset already exists: {new_path}")

        new_full.parent.mkdir(parents=True, exist_ok=True)
        old_full.rename(new_full)
        logger.info(
            "Asset moved",
            extra={
                "user_id": user_id,
                "project_id": project_id,
                "old_path": old_path,
                "new_path": new_path,
            },
        )

    def get_full_path(self, user_id: str, project_id: str, asset_path: str) -> Path:
        """Return the absolute filesystem path for an asset (for FileResponse)."""
        return self._asset_file_path(user_id, project_id, asset_path)

    def list_assets(
        self,
        user_id: str,
        project_id: str = DEFAULT_PROJECT_ID,
        folder: Optional[str] = None,
    ) -> list[AssetSummary]:
        """List assets in the vault, optionally scoped to a folder.

        Returns AssetSummary objects with MIME type detected from extension.
        OCR status defaults to 'skipped' — use AssetIndexer for real status.
        """
        root = self._asset_root(user_id, project_id)

        if folder:
            cleaned = folder.strip().strip("/")
            if ".." in cleaned or "\\" in cleaned:
                raise ValueError("Folder path contains invalid characters")
            search_root = (root / cleaned).resolve() if cleaned else root
            try:
                if os.path.commonpath([str(root), str(search_root)]) != str(root):
                    raise ValueError("Folder path escapes vault root")
            except ValueError as exc:
                raise ValueError("Folder path escapes vault root") from exc
            if not search_root.exists():
                return []
        else:
            search_root = root

        results: list[AssetSummary] = []
        for file_path in search_root.rglob("*"):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                continue
            try:
                rel = file_path.relative_to(root).as_posix()
                stat = file_path.stat()
                mime_type, _ = mimetypes.guess_type(str(file_path))
                mime_type = mime_type or "application/octet-stream"
                updated = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(
                    timespec="seconds"
                )
                results.append(
                    AssetSummary(
                        asset_path=rel,
                        mime_type=mime_type,
                        file_size=stat.st_size,
                        ocr_status="skipped",
                        updated=updated,
                    )
                )
            except (OSError, ValueError) as exc:
                logger.debug("Skipping asset %s: %s", file_path, exc)

        return sorted(results, key=lambda a: a.asset_path.lower())


__all__ = ["AssetVaultService", "validate_asset_path", "MAX_ASSET_BYTES", "ALLOWED_EXTENSIONS"]
