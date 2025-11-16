"""Service layer for business logic and external integrations."""

from .auth import AuthError, AuthService
from .config import AppConfig, get_config, reload_config
from .database import DatabaseService, init_database
from .indexer import IndexerService, normalize_slug, normalize_tag
from .vault import VaultNote, VaultService, sanitize_path, validate_note_path

__all__ = [
    "AppConfig",
    "get_config",
    "reload_config",
    "DatabaseService",
    "init_database",
    "AuthService",
    "AuthError",
    "VaultService",
    "VaultNote",
    "sanitize_path",
    "validate_note_path",
    "IndexerService",
    "normalize_slug",
    "normalize_tag",
]
