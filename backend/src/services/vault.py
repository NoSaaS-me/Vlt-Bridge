"""Filesystem vault management."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .config import AppConfig, get_config

INVALID_PATH_CHARS = {'<', '>', ':', '"', '|', '?', '*'}


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


__all__ = ["VaultService", "validate_note_path", "sanitize_path"]
