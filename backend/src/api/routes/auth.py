"""OAuth and authentication routes with GitHub OAuth as primary login."""

from __future__ import annotations

import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode, quote

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from ...models.auth import TokenResponse
from ...models.user import GHProfile, User
from ...services.auth import AuthError, AuthService
from ...services.config import get_config
from ...services.github_service import get_github_service, GitHubError
from ...services.seed import ensure_welcome_note
from ...services.vault import VaultService
from ..middleware import AuthContext, require_auth_context

logger = logging.getLogger(__name__)

router = APIRouter()

OAUTH_STATE_TTL_SECONDS = 300
# GitHub OAuth states for primary login (no user_id yet)
github_login_states: dict[str, float] = {}  # state -> timestamp
# GitHub OAuth states for connecting GitHub to existing account (has user_id)
github_connect_states: dict[str, tuple[float, str]] = {}  # state -> (timestamp, user_id)

auth_service = AuthService()


def _create_github_login_state() -> str:
    """Generate a state token for GitHub login and store it with a timestamp."""
    now = time.time()
    # Garbage collect expired states
    expired = [
        state
        for state, ts in github_login_states.items()
        if now - ts > OAUTH_STATE_TTL_SECONDS
    ]
    for state in expired:
        github_login_states.pop(state, None)

    state = secrets.token_urlsafe(32)
    github_login_states[state] = now
    return state


def _consume_github_login_state(state: str | None) -> None:
    """Validate and remove the GitHub login state token; raise if invalid."""
    if not state or state not in github_login_states:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")
    # Remove to prevent reuse
    del github_login_states[state]


def get_base_url(request: Request) -> str:
    """
    Get the base URL for OAuth redirects.

    Uses the actual request URL scheme and hostname from FastAPI's request.url.
    HF Spaces doesn't set X-Forwarded-Host, but the 'host' header is correct.
    """
    # Get scheme from X-Forwarded-Proto or request
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = forwarded_proto if forwarded_proto else str(request.url.scheme)

    # Get hostname from request URL (this comes from the 'host' header)
    hostname = str(request.url.hostname)

    # Check for port (but HF Spaces uses standard 443 for HTTPS)
    port = request.url.port
    if port and port not in (80, 443):
        base_url = f"{scheme}://{hostname}:{port}"
    else:
        base_url = f"{scheme}://{hostname}"

    logger.info(
        f"OAuth base URL detected: {base_url}",
        extra={
            "scheme": scheme,
            "hostname": hostname,
            "port": port,
            "request_url": str(request.url),
        },
    )

    return base_url


@router.get("/auth/login")
async def login(request: Request):
    """Redirect to GitHub OAuth authorization page for primary login."""
    try:
        github_service = get_github_service()
    except GitHubError as e:
        logger.error(f"GitHub service initialization error: {e}")
        raise HTTPException(
            status_code=501,
            detail="GitHub OAuth not configured. Set GITHUB_OAUTH_CLIENT_ID and GITHUB_OAUTH_CLIENT_SECRET environment variables.",
        )

    # Get base URL from request
    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/auth/callback"

    state = _create_github_login_state()

    # Get GitHub OAuth URL
    try:
        oauth_url = github_service.get_oauth_url(state=state, redirect_uri=redirect_uri)
    except GitHubError as e:
        logger.error(f"Failed to get GitHub OAuth URL: {e}")
        raise HTTPException(
            status_code=501,
            detail="GitHub OAuth not configured. Set GITHUB_OAUTH_CLIENT_ID and GITHUB_OAUTH_CLIENT_SECRET environment variables.",
        )

    logger.info(
        "Initiating GitHub OAuth login flow",
        extra={
            "redirect_uri": redirect_uri,
            "state": state,
        },
    )

    return RedirectResponse(url=oauth_url, status_code=302)


@router.get("/auth/callback")
async def callback(
    request: Request,
    code: str = Query(..., description="OAuth authorization code"),
    state: Optional[str] = Query(
        None, description="State parameter for CSRF protection"
    ),
):
    """Handle OAuth callback from GitHub for primary login."""
    github_service = get_github_service()

    # Get base URL from request (must match the one sent to GitHub)
    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/auth/callback"

    # Validate state token to prevent CSRF and replay attacks
    _consume_github_login_state(state)

    logger.info(
        "GitHub OAuth login callback received",
        extra={
            "redirect_uri": redirect_uri,
            "state": state,
            "code_length": len(code) if code else 0,
        },
    )

    try:
        # Exchange code for token using GitHub service
        access_token, github_username = await github_service.exchange_code_for_token(
            code=code,
            redirect_uri=redirect_uri,
        )

        # Use GitHub username (prefixed with "gh-") as user_id
        user_id = f"gh-{github_username}"

        # Store the GitHub token for the user (for github_read/github_search tools)
        github_service.store_token(user_id, access_token, github_username)

        # Ensure the user has an initialized vault with a welcome note
        try:
            created = ensure_welcome_note(user_id)
            logger.info(
                "Ensured welcome note for user",
                extra={"user_id": user_id, "created": created},
            )
        except Exception as seed_exc:
            logger.exception(
                "Failed to seed welcome note for user",
                extra={"user_id": user_id},
            )

        # Create JWT for our application
        import jwt
        from datetime import timedelta

        payload = {
            "sub": user_id,
            "username": github_username,
            "github_username": github_username,
            "exp": datetime.now(timezone.utc) + timedelta(days=7),
            "iat": datetime.now(timezone.utc),
        }

        try:
            jwt_secret = auth_service._require_secret()
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message)

        jwt_token = jwt.encode(payload, jwt_secret, algorithm="HS256")

        logger.info(
            "GitHub OAuth login successful",
            extra={
                "github_username": github_username,
                "user_id": user_id,
            },
        )

        # Redirect to frontend with token in URL hash
        redirect_url = f"{base_url}/#token={jwt_token}"
        logger.info(f"Redirecting to frontend: {redirect_url}")
        return RedirectResponse(url=redirect_url, status_code=302)

    except GitHubError as e:
        logger.error(f"GitHub OAuth error: {e}")
        return RedirectResponse(
            url=f"{base_url}/#login=error&message={quote(e.message)}",
            status_code=302,
        )
    except Exception as e:
        logger.exception(f"Unexpected error during GitHub OAuth login: {e}")
        return RedirectResponse(
            url=f"{base_url}/#login=error&message={quote('Login failed')}",
            status_code=302,
        )


@router.post("/api/tokens", response_model=TokenResponse)
async def create_api_token(auth: AuthContext = Depends(require_auth_context)):
    """Issue a new JWT for the authenticated user."""
    token, expires_at = auth_service.issue_token_response(auth.user_id)
    return TokenResponse(token=token, token_type="bearer", expires_at=expires_at)


@router.get("/api/me", response_model=User)
async def get_current_user(auth: AuthContext = Depends(require_auth_context)):
    """Return profile metadata for the authenticated user."""
    user_id = auth.user_id
    vault_service = VaultService()
    vault_path = vault_service.initialize_vault(user_id)

    # Attempt to derive a stable "created" timestamp from the vault directory
    try:
        stat = vault_path.stat()
        created_dt = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
    except Exception:
        created_dt = datetime.now(timezone.utc)

    profile: Optional[GHProfile] = None
    if user_id.startswith("gh-"):
        username = user_id[len("gh-") :]
        profile = GHProfile(
            username=username,
            name=username.replace("-", " ").title(),
            avatar_url=f"https://github.com/{username}.png",
        )
    elif user_id not in {"local-dev", "demo-user"}:
        # Fallback for other user types
        profile = GHProfile(
            username=user_id,
            avatar_url=f"https://api.dicebear.com/7.x/initials/svg?seed={user_id}",
        )

    return User(
        user_id=user_id,
        gh_profile=profile,
        vault_path=str(vault_path),
        created=created_dt,
    )


# =========================================================================
# GitHub OAuth Routes (for re-connecting/refreshing GitHub token)
# =========================================================================


def _create_github_connect_state(user_id: str) -> str:
    """Generate a state token for GitHub connect and store it with user_id."""
    now = time.time()
    # Garbage collect expired states
    expired = [
        state
        for state, (ts, _) in github_connect_states.items()
        if now - ts > OAUTH_STATE_TTL_SECONDS
    ]
    for state in expired:
        github_connect_states.pop(state, None)

    state = secrets.token_urlsafe(32)
    github_connect_states[state] = (now, user_id)
    return state


def _consume_github_connect_state(state: str | None) -> str:
    """Validate and remove the GitHub connect state token; return user_id or raise."""
    if not state or state not in github_connect_states:
        raise HTTPException(status_code=400, detail="Invalid or expired GitHub OAuth state.")
    ts, user_id = github_connect_states.pop(state)
    return user_id


@router.get("/api/auth/github")
async def github_reconnect(
    request: Request,
    token: Optional[str] = Query(None, description="JWT token for authentication (required for browser navigation)"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Initiate GitHub OAuth flow to reconnect/refresh GitHub token.

    This endpoint is for users who are already logged in but need to
    reconnect their GitHub account (e.g., token expired, or to update permissions).

    Accepts authentication via either:
    - Authorization header (for programmatic API calls)
    - token query parameter (for browser navigation which cannot set headers)

    This dual approach is necessary because browser navigation (window.location.href)
    cannot include custom headers like Authorization.

    On error, redirects back to settings page with error message in URL hash
    instead of returning JSON (since this is a browser navigation).
    """
    # Get base URL early for error redirects
    base_url = get_base_url(request)

    def error_redirect(message: str) -> RedirectResponse:
        """Redirect to settings with error message in hash."""
        encoded_message = quote(message)
        return RedirectResponse(
            url=f"{base_url}/settings#github=error&message={encoded_message}",
            status_code=302,
        )

    try:
        github_service = get_github_service()
    except GitHubError as e:
        logger.error(f"GitHub service initialization error: {e}")
        return error_redirect(e.message)

    # Try to get user_id from either token source
    user_id = None

    # First try Authorization header
    if authorization:
        scheme, _, header_token = authorization.partition(" ")
        if scheme.lower() == "bearer" and header_token:
            try:
                payload = auth_service.validate_jwt(header_token)
                user_id = payload.sub
            except AuthError as exc:
                logger.warning(f"Invalid Authorization header: {exc.message}")

    # Fall back to query parameter token
    if not user_id and token:
        try:
            payload = auth_service.validate_jwt(token)
            user_id = payload.sub
        except AuthError as exc:
            logger.warning(f"Invalid token query parameter: {exc.message}")

    # If neither worked, redirect with error (not JSON since this is browser navigation)
    if not user_id:
        return error_redirect("Authentication required. Please sign in first.")

    try:
        # Get redirect URI for OAuth callback
        redirect_uri = f"{base_url}/api/auth/github/callback"

        # Create state with user_id for linking after callback
        state = _create_github_connect_state(user_id)

        # Get OAuth URL - this may raise GitHubError if not configured
        oauth_url = github_service.get_oauth_url(state=state, redirect_uri=redirect_uri)

        logger.info(
            "Initiating GitHub OAuth reconnect flow",
            extra={
                "user_id": user_id,
                "redirect_uri": redirect_uri,
            },
        )

        return RedirectResponse(url=oauth_url, status_code=302)

    except GitHubError as e:
        logger.error(f"GitHub OAuth error: {e}")
        return error_redirect(e.message)


@router.get("/api/auth/github/callback")
async def github_reconnect_callback(
    request: Request,
    code: str = Query(..., description="GitHub OAuth authorization code"),
    state: Optional[str] = Query(None, description="State parameter for CSRF protection"),
):
    """Handle GitHub OAuth callback for reconnecting GitHub token."""
    github_service = get_github_service()

    # Validate state and get user_id
    user_id = _consume_github_connect_state(state)

    # Get redirect URI (must match the one sent to GitHub)
    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/auth/github/callback"

    logger.info(
        "GitHub OAuth reconnect callback received",
        extra={
            "user_id": user_id,
            "redirect_uri": redirect_uri,
            "code_length": len(code) if code else 0,
        },
    )

    try:
        # Exchange code for token
        access_token, github_username = await github_service.exchange_code_for_token(
            code=code,
            redirect_uri=redirect_uri,
        )

        # Store token for user
        github_service.store_token(user_id, access_token, github_username)

        logger.info(
            f"GitHub reconnected successfully for user {user_id} (GitHub: {github_username})"
        )

        # Redirect back to settings page with success message
        return RedirectResponse(
            url=f"{base_url}/settings#github=connected",
            status_code=302,
        )

    except GitHubError as e:
        logger.error(f"GitHub OAuth error: {e}")
        return RedirectResponse(
            url=f"{base_url}/settings#github=error&message={quote(e.message)}",
            status_code=302,
        )
    except Exception as e:
        logger.exception(f"Unexpected error during GitHub OAuth: {e}")
        return RedirectResponse(
            url=f"{base_url}/settings#github=error&message={quote('OAuth failed')}",
            status_code=302,
        )


@router.get("/api/auth/github/status")
async def github_status(auth: AuthContext = Depends(require_auth_context)):
    """Get GitHub connection status for current user."""
    github_service = get_github_service()

    username = github_service.get_github_username(auth.user_id)
    connected = username is not None

    result = {
        "connected": connected,
        "username": username,
    }

    # If connected, try to get rate limit info
    if connected:
        rate_limit = await github_service.get_rate_limit_status(auth.user_id)
        if "core" in rate_limit:
            result["rate_limit"] = {
                "remaining": rate_limit["core"].get("remaining"),
                "limit": rate_limit["core"].get("limit"),
                "reset": rate_limit["core"].get("reset"),
            }

    return result


@router.delete("/api/auth/github")
async def github_disconnect(auth: AuthContext = Depends(require_auth_context)):
    """Disconnect GitHub account from user."""
    github_service = get_github_service()

    username = github_service.get_github_username(auth.user_id)
    github_service.disconnect(auth.user_id)

    logger.info(
        f"GitHub disconnected for user {auth.user_id} (was: {username})"
    )

    return {"status": "disconnected", "previous_username": username}


__all__ = ["router"]
