"""OAuth2 authorization code flow endpoints for connectors."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from ..middleware import AuthContext, require_auth_context
from ...services.connector_service import ConnectorService, get_connector_service
from ...services.database import DatabaseService, get_db_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/connectors", tags=["connector-oauth"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_oauth_env(connector_name: str) -> dict[str, str]:
    """
    Resolve OAuth2 client credentials from environment variables.
    Convention: {CONNECTOR_NAME_UPPER}_CLIENT_ID / _CLIENT_SECRET
    e.g., GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET
    """
    prefix = connector_name.upper().replace("-", "_")
    client_id = os.environ.get(f"{prefix}_CLIENT_ID", "")
    client_secret = os.environ.get(f"{prefix}_CLIENT_SECRET", "")
    return {"client_id": client_id, "client_secret": client_secret}


def _get_redirect_uri(request: Request, connector_name: str) -> str:
    """Build the OAuth2 callback URI, respecting reverse-proxy headers."""
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{scheme}://{host}/api/connectors/{connector_name}/oauth/callback"


def _get_connector_or_404(connector_name: str):
    from ...connectors.registry import get_registry
    connector = get_registry().get(connector_name)
    if not connector:
        raise HTTPException(status_code=404, detail=f"Connector '{connector_name}' not found")
    return connector


# ---------------------------------------------------------------------------
# Authorize — redirect user to provider consent screen
# ---------------------------------------------------------------------------

@router.get("/{connector_name}/oauth/authorize")
async def oauth_authorize(
    connector_name: str,
    request: Request,
    auth: AuthContext = Depends(require_auth_context),
    db: DatabaseService = Depends(get_db_service),
) -> RedirectResponse:
    """
    Start OAuth2 authorization flow.
    Generates a state token, optionally a PKCE pair, stores them in oauth_states,
    then redirects the user to the provider's consent screen.
    """
    connector = _get_connector_or_404(connector_name)

    auth_type = getattr(connector, "auth_type", "api_key")
    if auth_type != "oauth2":
        raise HTTPException(
            status_code=400,
            detail=f"Connector '{connector_name}' does not use OAuth2",
        )

    from vlt_connectors.oauth import OAuthHelpers

    oauth_cfg = connector.get_oauth_config()

    state = OAuthHelpers.generate_state()
    pkce_verifier: Optional[str] = None
    code_challenge: Optional[str] = None
    if oauth_cfg.pkce:
        pkce_verifier, code_challenge = OAuthHelpers.generate_pkce_pair()

    redirect_uri = _get_redirect_uri(request, connector_name)
    authorize_url = OAuthHelpers.build_authorize_url(
        oauth_cfg,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
    )

    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    conn = db.connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO oauth_states
               (state, user_id, connector_name, created_at, expires_at, pkce_verifier)
               VALUES (?, ?, ?, datetime('now'), ?, ?)""",
            (state, auth.user_id, connector_name, expires_at, pkce_verifier),
        )
        # Purge expired states
        conn.execute("DELETE FROM oauth_states WHERE expires_at < datetime('now')")
        conn.commit()
    finally:
        conn.close()

    return RedirectResponse(url=authorize_url, status_code=302)


# ---------------------------------------------------------------------------
# Callback — exchange auth code, store tokens
# ---------------------------------------------------------------------------

@router.get("/{connector_name}/oauth/callback")
async def oauth_callback(
    connector_name: str,
    request: Request,
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
    db: DatabaseService = Depends(get_db_service),
    svc: ConnectorService = Depends(get_connector_service),
) -> RedirectResponse:
    """
    OAuth2 callback: validate state, exchange auth code for tokens, store encrypted.
    Redirects to frontend with ?connected=<name> or ?oauth_error=<msg>.
    """
    FRONTEND_REDIRECT = "/"

    if error:
        msg = error_description or error
        logger.warning("OAuth2 provider error for connector '%s': %s", connector_name, msg)
        return RedirectResponse(url=f"{FRONTEND_REDIRECT}?oauth_error={error}", status_code=302)

    if not code or not state:
        return RedirectResponse(url=f"{FRONTEND_REDIRECT}?oauth_error=missing_params", status_code=302)

    # ------------------------------------------------------------------
    # Validate state and look up user
    # ------------------------------------------------------------------
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT user_id, expires_at, pkce_verifier FROM oauth_states WHERE state=? AND connector_name=?",
            (state, connector_name),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

        expires_at_str = row["expires_at"]
        user_id = row["user_id"]

        expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > expires_at:
            conn.execute("DELETE FROM oauth_states WHERE state=?", (state,))
            conn.commit()
            raise HTTPException(status_code=400, detail="OAuth state expired")

        # Read PKCE verifier from the dedicated column
        pkce_verifier: Optional[str] = row["pkce_verifier"]

        # Delete used state row
        conn.execute("DELETE FROM oauth_states WHERE state=?", (state,))
        conn.commit()
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # Exchange authorization code for tokens
    # ------------------------------------------------------------------
    connector = _get_connector_or_404(connector_name)
    oauth_cfg = connector.get_oauth_config()
    oauth_env = _get_oauth_env(connector_name)
    redirect_uri = _get_redirect_uri(request, connector_name)

    token_payload: dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": oauth_env["client_id"],
        "client_secret": oauth_env["client_secret"],
    }
    if pkce_verifier:
        token_payload["code_verifier"] = pkce_verifier

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                oauth_cfg.token_url,
                data=token_payload,
                headers={"Accept": "application/json"},
            )
    except httpx.RequestError as exc:
        logger.error("Token exchange network error for '%s': %s", connector_name, exc)
        return RedirectResponse(
            url=f"{FRONTEND_REDIRECT}?oauth_error=network_error",
            status_code=302,
        )

    if resp.status_code >= 400:
        logger.error(
            "Token exchange failed for '%s': HTTP %d — %s",
            connector_name, resp.status_code, resp.text[:200],
        )
        return RedirectResponse(
            url=f"{FRONTEND_REDIRECT}?oauth_error=token_exchange_failed",
            status_code=302,
        )

    token_data = resp.json()
    access_token: str = token_data.get("access_token", "")
    refresh_token: str = token_data.get("refresh_token", "")
    expires_in = token_data.get("expires_in")
    scope: str = token_data.get("scope", "")

    if not access_token:
        logger.error("Token exchange for '%s' returned no access_token", connector_name)
        return RedirectResponse(
            url=f"{FRONTEND_REDIRECT}?oauth_error=no_access_token",
            status_code=302,
        )

    token_expires_at: str = ""
    if expires_in:
        token_expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
        ).isoformat()

    # ------------------------------------------------------------------
    # Persist tokens (encrypted via ConnectorService / _ALWAYS_SECRET_KEYS)
    # ------------------------------------------------------------------
    updates: dict[str, str] = {
        "__oauth_access_token": access_token,
        "__oauth_expires_at": token_expires_at,
        "__oauth_scope": scope,
        "__enabled": "true",
    }
    if refresh_token:
        updates["__oauth_refresh_token"] = refresh_token

    svc.set_config(user_id, connector_name, updates)
    logger.info("OAuth2 tokens stored for user '%s', connector '%s'", user_id, connector_name)

    return RedirectResponse(
        url=f"{FRONTEND_REDIRECT}?connected={connector_name}",
        status_code=302,
    )


# ---------------------------------------------------------------------------
# Status — check OAuth connection
# ---------------------------------------------------------------------------

@router.get("/{connector_name}/oauth/status")
async def oauth_status(
    connector_name: str,
    auth: AuthContext = Depends(require_auth_context),
    svc: ConnectorService = Depends(get_connector_service),
) -> dict:
    """Return OAuth2 connection status for a connector."""
    _get_connector_or_404(connector_name)

    cfg = svc.get_config(auth.user_id, connector_name)
    access_token = cfg.get("__oauth_access_token", "")
    expires_at = cfg.get("__oauth_expires_at", "")
    scope = cfg.get("__oauth_scope", "")
    enabled = cfg.get("__enabled", "false").lower() == "true"

    connected = bool(access_token) and enabled

    # Determine if expired
    token_expired = False
    if connected and expires_at:
        from vlt_connectors.oauth import OAuthHelpers
        token_expired = OAuthHelpers.is_token_expired(expires_at)

    return {
        "connector": connector_name,
        "connected": connected and not token_expired,
        "expires_at": expires_at,
        "scope": scope,
        "token_expired": token_expired,
    }


# ---------------------------------------------------------------------------
# Revoke — disconnect OAuth connector
# ---------------------------------------------------------------------------

@router.post("/{connector_name}/oauth/revoke")
async def oauth_revoke(
    connector_name: str,
    auth: AuthContext = Depends(require_auth_context),
    db: DatabaseService = Depends(get_db_service),
    svc: ConnectorService = Depends(get_connector_service),
) -> dict:
    """Disconnect an OAuth2 connector by deleting stored tokens."""
    _get_connector_or_404(connector_name)

    # Delete all __oauth_* keys and disable the connector
    oauth_keys = [
        "__oauth_access_token",
        "__oauth_refresh_token",
        "__oauth_expires_at",
        "__oauth_scope",
    ]

    conn = db.connect()
    try:
        conn.execute(
            f"""DELETE FROM connector_configs
                WHERE user_id=? AND connector_name=?
                  AND config_key IN ({','.join('?' * len(oauth_keys))})""",
            (auth.user_id, connector_name, *oauth_keys),
        )
        conn.commit()
    finally:
        conn.close()

    # Mark disabled
    svc.set_config(auth.user_id, connector_name, {"__enabled": "false"})

    logger.info("OAuth2 revoked for user '%s', connector '%s'", auth.user_id, connector_name)
    return {"connector": connector_name, "revoked": True}
