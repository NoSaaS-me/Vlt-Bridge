"""Daemon-side credential cache for service connector access."""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_TTL = 300  # 5 minutes


class CredentialCache:
    """
    In-memory credential cache for daemon/CLI service connector access.

    Fetches decrypted credentials from the backend via
    GET /api/connectors/{connector_name}/credentials and caches them
    for _TTL seconds to avoid hammering the backend on every use.

    Usage::

        cache = CredentialCache(
            backend_url="http://localhost:8000",
            auth_token="<daemon-jwt>",
        )
        creds = cache.fetch("huggingface_inference", user_id="local-dev")
        api_key = creds["api_key"]
    """

    def __init__(self, backend_url: str, auth_token: str) -> None:
        self._backend_url = backend_url.rstrip("/")
        self._auth_token = auth_token
        # key -> (credentials_dict, expires_at_monotonic)
        self._cache: dict[str, tuple[dict[str, str], float]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, connector_name: str, user_id: str) -> str:
        return f"{user_id}:{connector_name}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, connector_name: str, user_id: str) -> Optional[dict[str, str]]:
        """Return cached credentials if still valid, otherwise None."""
        key = self._cache_key(connector_name, user_id)
        entry = self._cache.get(key)
        if entry is not None and time.monotonic() < entry[1]:
            return entry[0]
        return None

    def fetch(self, connector_name: str, user_id: str) -> dict[str, str]:
        """
        Return credentials for *connector_name* / *user_id*, using cache.

        Raises:
            ValueError: connector not found or not configured on the backend.
            RuntimeError: unexpected HTTP error from the backend.
        """
        cached = self.get(connector_name, user_id)
        if cached is not None:
            return cached

        resp = httpx.get(
            f"{self._backend_url}/api/connectors/{connector_name}/credentials",
            headers={"Authorization": f"Bearer {self._auth_token}"},
            timeout=5.0,
        )
        if resp.status_code == 404:
            raise ValueError(
                f"Connector '{connector_name}' not found or not configured"
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Backend returned {resp.status_code} fetching credentials for '{connector_name}'"
            )

        creds: dict[str, str] = resp.json().get("credentials", {})
        key = self._cache_key(connector_name, user_id)
        self._cache[key] = (creds, time.monotonic() + _TTL)
        logger.debug(
            "Cached credentials for %s/%s (TTL %ds)", connector_name, user_id, _TTL
        )
        return creds

    def invalidate(self, connector_name: str, user_id: str) -> None:
        """Evict cached credentials so the next fetch hits the backend."""
        key = self._cache_key(connector_name, user_id)
        self._cache.pop(key, None)
