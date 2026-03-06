"""OAuth2 authorization code flow support for connectors."""
from __future__ import annotations

import base64
import hashlib
import os
import secrets
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlencode


@dataclass
class OAuthConfig:
    """OAuth2 configuration for a connector."""
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    scopes: list[str]
    pkce: bool = False
    # How to inject the access token into API requests
    auth_header_template: str = "Bearer {access_token}"


class OAuthHelpers:
    """Stateless OAuth2 helpers: state generation, PKCE, URL building."""

    @staticmethod
    def generate_state() -> str:
        """Generate a cryptographically random state parameter."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def generate_pkce_pair() -> tuple[str, str]:
        """Generate PKCE code_verifier and code_challenge (S256 method)."""
        verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()
        ).rstrip(b"=").decode()
        return verifier, challenge

    @staticmethod
    def build_authorize_url(
        config: OAuthConfig,
        redirect_uri: str,
        state: str,
        code_challenge: Optional[str] = None,
    ) -> str:
        """Build the provider's authorization URL."""
        params: dict[str, str] = {
            "client_id": config.client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(config.scopes),
            "state": state,
            "response_type": "code",
        }
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"
        return f"{config.authorize_url}?{urlencode(params)}"

    @staticmethod
    def is_token_expired(expires_at_iso: str, buffer_seconds: int = 60) -> bool:
        """Return True if the token expires within buffer_seconds."""
        try:
            import datetime
            exp = datetime.datetime.fromisoformat(expires_at_iso.replace("Z", "+00:00"))
            now = datetime.datetime.now(datetime.timezone.utc)
            return (exp - now).total_seconds() < buffer_seconds
        except Exception:
            return True  # conservative: treat unparseable as expired


class OAuthActionConnector:
    """
    Mixin for ActionConnectors that use OAuth2.

    Subclasses declare `get_oauth_config()` and optionally `parse_webhook()`.
    The ActionConnector invoke() should call `_build_auth_header()` with the
    access token retrieved from credentials before making API calls.
    """
    auth_type: str = "oauth2"  # signals to frontend to show Connect button

    def get_oauth_config(self) -> OAuthConfig:
        """Return OAuth2 config. Subclasses must implement."""
        raise NotImplementedError

    def _build_auth_header(self, access_token: str) -> str:
        config = self.get_oauth_config()
        return config.auth_header_template.format(access_token=access_token)
