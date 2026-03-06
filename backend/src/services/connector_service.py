"""Service for per-user connector configuration in SQLite, with encrypted secrets."""
from __future__ import annotations

import logging
from typing import Optional

from cryptography.fernet import Fernet
from fastapi import Depends

from .database import DatabaseService, get_db_service
from vlt_connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# OAuth2 token keys that must always be encrypted regardless of credential_fields
_ALWAYS_SECRET_KEYS: frozenset[str] = frozenset({
    "__oauth_access_token",
    "__oauth_refresh_token",
})


class ConnectorService:
    def __init__(self, db_service: Optional[DatabaseService] = None):
        self.db = db_service or DatabaseService()
        self._fernet: Optional[Fernet] = None  # lazy init

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _get_fernet(self) -> Fernet:
        if self._fernet is None:
            from vlt_connectors.encryption import get_fernet
            from .config import get_config
            cfg = get_config()
            enc_key = cfg.connector_encryption_key or ""
            jwt_key = cfg.jwt_secret_key or ""
            self._fernet = get_fernet(
                encryption_key=enc_key if enc_key else None,
                jwt_secret=jwt_key if jwt_key else None,
            )
        return self._fernet

    def _get_secret_field_names(self, connector_name: str) -> set[str]:
        """Return config_key names that are secret=True for this connector."""
        from vlt_connectors.registry import get_registry
        connector = get_registry().get(connector_name)
        if connector is None:
            return set()
        return {f.name for f in connector.credential_fields if f.secret}

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def get_config(self, user_id: str, connector_name: str) -> dict[str, str]:
        """Return all config key/value pairs for a connector, including __enabled.

        Secret fields are decrypted transparently. Legacy plaintext values
        (no ``v1:`` prefix) pass through unchanged.
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                "SELECT config_key, config_value FROM connector_configs"
                " WHERE user_id=? AND connector_name=?",
                (user_id, connector_name),
            )
            raw = {row["config_key"]: (row["config_value"] or "") for row in cursor.fetchall()}
        finally:
            conn.close()

        secret_fields = self._get_secret_field_names(connector_name) | _ALWAYS_SECRET_KEYS
        fernet = self._get_fernet()

        result: dict[str, str] = {}
        for key, value in raw.items():
            # Decrypt if this key is secret. For regular credential fields we skip __ prefixed
            # keys (internal control flags), but _ALWAYS_SECRET_KEYS are __ prefixed by design
            # so we treat them as secret unconditionally.
            is_secret = (
                (key in secret_fields and not key.startswith("__"))
                or key in _ALWAYS_SECRET_KEYS
            )
            if is_secret and value:
                from vlt_connectors.encryption import decrypt_value
                try:
                    result[key] = decrypt_value(value, fernet)
                except Exception:
                    logger.warning(
                        "Failed to decrypt credential key '%s' for connector '%s'; "
                        "returning raw value (may be legacy plaintext).",
                        key, connector_name,
                    )
                    result[key] = value
            else:
                result[key] = value
        return result

    def set_config(self, user_id: str, connector_name: str, updates: dict[str, str]) -> None:
        """Upsert config keys for a connector.

        Secret fields are encrypted before storage. Empty values for secret
        fields are stored as-is (empty string) so callers can clear a credential.
        """
        secret_fields = self._get_secret_field_names(connector_name) | _ALWAYS_SECRET_KEYS
        fernet = self._get_fernet()

        conn = self.db.connect()
        try:
            for key, value in updates.items():
                # Encrypt regular secret credential fields (not __ internal flags),
                # plus the OAuth token keys which are always secret.
                should_encrypt = (
                    (key in secret_fields and not key.startswith("__"))
                    or key in _ALWAYS_SECRET_KEYS
                )
                if should_encrypt and value:
                    from vlt_connectors.encryption import encrypt_value
                    stored_value = encrypt_value(value, fernet)
                else:
                    stored_value = value

                conn.execute(
                    """
                    INSERT INTO connector_configs (user_id, connector_name, config_key, config_value)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (user_id, connector_name, config_key)
                    DO UPDATE SET config_value=excluded.config_value, updated_at=datetime('now')
                    """,
                    (user_id, connector_name, key, stored_value),
                )
            conn.commit()
        finally:
            conn.close()

    def is_enabled(self, user_id: str, connector_name: str) -> bool:
        config = self.get_config(user_id, connector_name)
        return config.get("__enabled", "false").lower() == "true"

    def get_credentials(self, user_id: str, connector_name: str) -> dict[str, str]:
        """Return decrypted config minus any keys starting with __ (internal control keys)."""
        return {k: v for k, v in self.get_config(user_id, connector_name).items() if not k.startswith("__")}

    def is_configured(self, user_id: str, connector: BaseConnector) -> bool:
        """True if all secret credential fields have non-empty values."""
        creds = self.get_credentials(user_id, connector.name)
        secret_fields = [f.name for f in connector.credential_fields if f.secret]
        return all(creds.get(f, "").strip() for f in secret_fields)


def get_connector_service(db: DatabaseService = Depends(get_db_service)) -> ConnectorService:
    return ConnectorService(db_service=db)
