"""Filesystem vault management."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

import frontmatter

from .config import AppConfig, get_config

INVALID_PATH_CHARS = {'<', '>', ':', '"', '|', '?', '*'}
MAX_NOTE_BYTES = 1_048_576
H1_PATTERN = re.compile(r"^\s*#\s+(.+)$", re.MULTILINE)

VaultNote = Dict[str, Any]


def validate_note_path(note_path: str) -> Tuple[bool, str]:
    """
    Validate a relative Markdown path.

    Returns (is_valid, message). Message is empty when valid.
    """
    if not note_path or len(note_path) > 256:
        return False, "Path must be 1-256 characters"
    if not note_path.endswith(".md"):
        return False, "Path must end with .md"
    if ".." in note_path:
        return False, "Path must not contain '..'"
    if "\\" in note_path:
        return False, "Path must use Unix separators (/)"
    if note_path.startswith("/"):
        return False, "Path must be relative (no leading /)"
    if any(char in INVALID_PATH_CHARS for char in note_path):
        return False, "Path contains invalid characters"
    return True, ""


def sanitize_path(user_id: str, vault_root: Path, note_path: str) -> Path:
    """
    Sanitize and resolve a note path within the vault.

    Raises ValueError if the resolved path escapes the vault root.
    """
    vault = (vault_root / user_id).resolve()
    full_path = (vault / note_path).resolve()
    if not str(full_path).startswith(str(vault)):
        raise ValueError(f"Path escapes vault root: {note_path}")
    return full_path


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_frontmatter(metadata: Dict[str, Any]) -> Dict[str, Any]:
    reserved = {"version"}
    for key in metadata.keys():
        if key in reserved:
            raise ValueError(f"Field '{key}' is reserved and cannot be set in frontmatter")
    tags = metadata.get("tags")
    if tags is not None:
        if not isinstance(tags, list):
            raise ValueError("Field 'tags' must be an array")
        if not all(isinstance(tag, str) for tag in tags):
            raise ValueError("All tags must be strings")
    return metadata


def _validate_note_body(body: str) -> None:
    body_bytes = body.encode("utf-8")
    if len(body_bytes) > MAX_NOTE_BYTES:
        raise ValueError("Note exceeds 1 MiB limit")


def _derive_title(note_path: str, metadata: Dict[str, Any], body: str) -> str:
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    match = H1_PATTERN.search(body or "")
    if match:
        return match.group(1).strip()
    stem = Path(note_path).stem
    title_from_filename = stem.replace("-", " ").replace("_", " ").strip()
    return title_from_filename or stem


class VaultService:
    """Service for managing vault directories and basic path validation."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.vault_root = self.config.vault_base_path
        self.vault_root.mkdir(parents=True, exist_ok=True)

    def initialize_vault(self, user_id: str) -> Path:
        """Ensure a user's vault directory exists and return its path."""
        path = (self.vault_root / user_id).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_note_path(self, user_id: str, note_path: str) -> Path:
        """
        Validate and resolve a note path inside a user's vault.

        Raises ValueError for invalid paths.
        """
        is_valid, message = validate_note_path(note_path)
        if not is_valid:
            raise ValueError(message)
        return sanitize_path(user_id, self.vault_root, note_path)

    def read_note(self, user_id: str, note_path: str) -> VaultNote:
        """Read a Markdown note, returning metadata, body, and derived title."""
        base = self.initialize_vault(user_id)
        absolute_path = self.resolve_note_path(user_id, note_path)
        if not absolute_path.exists():
            raise FileNotFoundError(f"Note not found: {note_path}")
        post = frontmatter.load(absolute_path)
        metadata = dict(post.metadata or {})
        body = post.content or ""
        return self._build_note_payload(note_path, metadata, body, absolute_path)

    def write_note(
        self,
        user_id: str,
        note_path: str,
        *,
        title: str | None = None,
        metadata: Dict[str, Any] | None = None,
        body: str,
    ) -> VaultNote:
        """Create or update a note with validated metadata and content."""
        absolute_path = self.resolve_note_path(user_id, note_path)
        body = body or ""
        _validate_note_body(body)

        metadata_dict: Dict[str, Any] = dict(metadata or {})
        _validate_frontmatter(metadata_dict)

        existing_created: str | None = None
        if absolute_path.exists():
            try:
                current = frontmatter.load(absolute_path)
                current_created = current.metadata.get("created")
                if isinstance(current_created, str):
                    existing_created = current_created
            except Exception:
                existing_created = None

        effective_title = title or metadata_dict.get("title")
        if not effective_title:
            effective_title = _derive_title(note_path, metadata_dict, body)
        metadata_dict["title"] = effective_title

        now_iso = _utcnow_iso()
        metadata_dict.setdefault("created", existing_created or metadata_dict.get("created") or now_iso)
        metadata_dict["updated"] = now_iso

        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(body, **metadata_dict)
        absolute_path.write_text(frontmatter.dumps(post), encoding="utf-8")
        return self._build_note_payload(note_path, metadata_dict, body, absolute_path)

    def delete_note(self, user_id: str, note_path: str) -> None:
        """Delete a note from the vault."""
        absolute_path = self.resolve_note_path(user_id, note_path)
        try:
            absolute_path.unlink()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Note not found: {note_path}") from exc

    def list_notes(self, user_id: str, folder: str | None = None) -> List[Dict[str, Any]]:
        """List notes (optionally scoped to a folder) with titles and timestamps."""
        base = self.initialize_vault(user_id).resolve()

        if folder:
            cleaned = folder.strip().strip("/")
            if "\\" in cleaned or ".." in cleaned:
                raise ValueError("Folder path contains invalid characters")
            folder_path = (base / cleaned).resolve() if cleaned else base
            if not str(folder_path).startswith(str(base)):
                raise ValueError("Folder path escapes vault root")
            if not folder_path.exists():
                return []
            if folder_path.is_file():
                files = [folder_path] if folder_path.suffix == ".md" else []
            else:
                files = list(folder_path.rglob("*.md"))
        else:
            files = list(base.rglob("*.md"))

        results: List[Dict[str, Any]] = []
        for file_path in files:
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(base).as_posix()
            try:
                post = frontmatter.load(file_path)
                metadata = dict(post.metadata or {})
                body = post.content or ""
                title = _derive_title(relative_path, metadata, body)
            except Exception:
                title = Path(relative_path).stem
            stat = file_path.stat()
            results.append(
                {
                    "path": relative_path,
                    "title": title,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                }
            )

        return sorted(results, key=lambda item: item["path"].lower())

    def _build_note_payload(
        self, note_path: str, metadata: Dict[str, Any], body: str, absolute_path: Path
    ) -> VaultNote:
        stat = absolute_path.stat()
        title = _derive_title(note_path, metadata, body)
        return {
            "path": note_path,
            "title": title,
            "metadata": metadata,
            "body": body,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            "absolute_path": absolute_path,
        }


__all__ = ["VaultService", "validate_note_path", "sanitize_path"]
