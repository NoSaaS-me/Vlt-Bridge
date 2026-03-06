"""Connector credential encryption using Fernet (AES-128-CBC + HMAC)."""
from __future__ import annotations
import base64
import os
import logging
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

logger = logging.getLogger(__name__)
_PREFIX = "v1:"


def _derive_key_from_jwt_secret(jwt_secret: str) -> bytes:
    """Derive a Fernet key from JWT_SECRET_KEY via HKDF."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"vlt-connector-credentials",
        info=b"connector-encryption-v1",
    )
    raw = hkdf.derive(jwt_secret.encode())
    return base64.urlsafe_b64encode(raw)


def get_fernet(encryption_key: str | None = None, jwt_secret: str | None = None) -> Fernet:
    """
    Get a Fernet instance.
    Priority: encryption_key env var > derive from jwt_secret > error.
    """
    key = encryption_key or os.environ.get("CONNECTOR_ENCRYPTION_KEY")
    if not key:
        fallback = jwt_secret or os.environ.get("JWT_SECRET_KEY")
        if fallback:
            logger.warning(
                "CONNECTOR_ENCRYPTION_KEY not set; deriving from JWT_SECRET_KEY. "
                "Set CONNECTOR_ENCRYPTION_KEY for better security."
            )
            key = _derive_key_from_jwt_secret(fallback).decode()
        else:
            raise RuntimeError(
                "Neither CONNECTOR_ENCRYPTION_KEY nor JWT_SECRET_KEY is set. "
                "Cannot encrypt connector credentials."
            )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_value(value: str, fernet: Fernet) -> str:
    """Encrypt a credential value. Returns 'v1:<fernet_token>'."""
    token = fernet.encrypt(value.encode()).decode()
    return f"{_PREFIX}{token}"


def decrypt_value(stored: str, fernet: Fernet) -> str:
    """
    Decrypt a stored credential value.
    Handles 'v1:<token>' format and legacy plaintext (no prefix).
    """
    if not stored.startswith(_PREFIX):
        # Legacy plaintext — return as-is (will be re-encrypted on next write)
        return stored
    token = stored[len(_PREFIX):]
    try:
        return fernet.decrypt(token.encode()).decode()
    except InvalidToken:
        logger.error("Failed to decrypt connector credential — wrong key?")
        raise


def generate_key() -> str:
    """Generate a new Fernet key. Print and store as CONNECTOR_ENCRYPTION_KEY."""
    return Fernet.generate_key().decode()
