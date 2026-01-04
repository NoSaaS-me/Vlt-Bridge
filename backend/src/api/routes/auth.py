"""OAuth and authentication routes for Hugging Face integration."""

from __future__ import annotations

import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from ...models.auth import TokenResponse
from ...models.user import HFProfile, User
from ...services.auth import AuthError, AuthService
from ...services.config import get_config
from ...services.github_service import get_github_service, GitHubError
from ...services.seed import ensure_welcome_note
from ...services.vault import VaultService
from ..middleware import AuthContext, require_auth_context

logger = logging.getLogger(__name__)

router = APIRouter()

OAUTH_STATE_TTL_SECONDS = 300
oauth_states: dict[str, float] = {}
github_oauth_states: dict[str, tuple[float, str]] = {}  # state -> (timestamp, user_id)

auth_service = AuthService()


def _create_oauth_state() -> str:
    """Generate a state token and store it with a timestamp."""
    now = time.time()
    # Garbage collect expired states
    expired = [
        state
        for state, ts in oauth_states.items()
        if now - ts > OAUTH_STATE_TTL_SECONDS
    ]
    for state in expired:
        oauth_states.pop(state, None)

    state = secrets.token_urlsafe(32)
    oauth_states[state] = now
    return state


def _consume_oauth_state(state: str | None) -> None:
    """Validate and remove the state token; raise if invalid."""
    if not state or state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")
    # Remove to prevent reuse
    del oauth_states[state]


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
    """Redirect to Hugging Face OAuth authorization page."""
    config = get_config()

    if not config.hf_oauth_client_id:
        raise HTTPException(
            status_code=501,
            detail="OAuth not configured. Set HF_OAUTH_CLIENT_ID and HF_OAUTH_CLIENT_SECRET environment variables.",
        )

    # Get base URL from request (handles HF Spaces proxy)
    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/auth/callback"

    state = _create_oauth_state()

    # Construct HF OAuth URL
    oauth_base = "https://huggingface.co/oauth/authorize"
    params = {
        "client_id": config.hf_oauth_client_id,
        "redirect_uri": redirect_uri,
        "scope": "openid profile email",
        "response_type": "code",
        "state": state,
    }

    auth_url = f"{oauth_base}?{urlencode(params)}"
    logger.info(
        "Initiating OAuth flow",
        extra={
            "redirect_uri": redirect_uri,
            "auth_url": auth_url,
            "client_id": config.hf_oauth_client_id[:8] + "...",
            "state": state,
        },
    )

    return RedirectResponse(url=auth_url, status_code=302)


@router.get("/auth/callback")
async def callback(
    request: Request,
    code: str = Query(..., description="OAuth authorization code"),
    state: Optional[str] = Query(
        None, description="State parameter for CSRF protection"
    ),
):
    """Handle OAuth callback from Hugging Face."""
    config = get_config()

    if not config.hf_oauth_client_id or not config.hf_oauth_client_secret:
        raise HTTPException(status_code=501, detail="OAuth not configured")

    # Get base URL from request (must match the one sent to HF)
    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/auth/callback"

    # Validate state token to prevent CSRF and replay attacks
    _consume_oauth_state(state)

    logger.info(
        "OAuth callback received",
        extra={
            "redirect_uri": redirect_uri,
            "state": state,
            "code_length": len(code) if code else 0,
        },
    )

    try:
        # Exchange authorization code for access token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://huggingface.co/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": config.hf_oauth_client_id,
                    "client_secret": config.hf_oauth_client_secret,
                },
            )

            if token_response.status_code != 200:
                logger.error(f"Token exchange failed: {token_response.text}")
                raise HTTPException(
                    status_code=400,
                    detail="Failed to exchange authorization code for token",
                )

            token_data = token_response.json()
            access_token = token_data.get("access_token")

            if not access_token:
                raise HTTPException(
                    status_code=400, detail="No access token in response"
                )

            # Get user profile from HF
            user_response = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if user_response.status_code != 200:
                logger.error(f"User profile fetch failed: {user_response.text}")
                raise HTTPException(
                    status_code=400, detail="Failed to fetch user profile"
                )

            user_data = user_response.json()
            username = user_data.get("name")
            email = user_data.get("email")

            if not username:
                raise HTTPException(
                    status_code=400, detail="No username in user profile"
                )

            # Create JWT for our application
            import jwt
            from datetime import datetime, timedelta, timezone

            user_id = username  # Use HF username as user_id

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

            payload = {
                "sub": user_id,
                "username": username,
                "email": email,
                "exp": datetime.now(timezone.utc) + timedelta(days=7),
                "iat": datetime.now(timezone.utc),
            }

            try:
                jwt_secret = auth_service._require_secret()
            except AuthError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.message)

            jwt_token = jwt.encode(payload, jwt_secret, algorithm="HS256")

            logger.info(
                "OAuth successful",
                extra={
                    "username": username,
                    "user_id": user_id,
                    "email": email,
                },
            )

            # Redirect to frontend with token in URL hash
            frontend_url = base_url
            redirect_url = f"{frontend_url}/#token={jwt_token}"
            logger.info(f"Redirecting to frontend: {redirect_url}")
            return RedirectResponse(url=redirect_url, status_code=302)

    except httpx.HTTPError as e:
        logger.exception(f"HTTP error during OAuth: {e}")
        raise HTTPException(
            status_code=500, detail="OAuth flow failed due to network error"
        )
    except Exception as e:
        logger.exception(f"Unexpected error during OAuth: {e}")
        raise HTTPException(status_code=500, detail="OAuth flow failed")


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

    profile: Optional[HFProfile] = None
    if user_id.startswith("hf-"):
        username = user_id[len("hf-") :]
        profile = HFProfile(
            username=username,
            name=username.replace("-", " ").title(),
            avatar_url=f"https://api.dicebear.com/7.x/initials/svg?seed={username}",
        )
    elif user_id not in {"local-dev", "demo-user"}:
        profile = HFProfile(username=user_id)

    return User(
        user_id=user_id,
        hf_profile=profile,
        vault_path=str(vault_path),
        created=created_dt,
    )


# =========================================================================
# GitHub OAuth Routes
# =========================================================================


def _create_github_oauth_state(user_id: str) -> str:
    """Generate a state token for GitHub OAuth and store it with user_id."""
    now = time.time()
    # Garbage collect expired states
    expired = [
        state
        for state, (ts, _) in github_oauth_states.items()
        if now - ts > OAUTH_STATE_TTL_SECONDS
    ]
    for state in expired:
        github_oauth_states.pop(state, None)

    state = secrets.token_urlsafe(32)
    github_oauth_states[state] = (now, user_id)
    return state


def _consume_github_oauth_state(state: str | None) -> str:
    """Validate and remove the GitHub state token; return user_id or raise."""
    if not state or state not in github_oauth_states:
        raise HTTPException(status_code=400, detail="Invalid or expired GitHub OAuth state.")
    ts, user_id = github_oauth_states.pop(state)
    return user_id


@router.get("/api/auth/github")
async def github_login(
    request: Request,
    token: Optional[str] = Query(None, description="JWT token for authentication (required for browser navigation)"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Initiate GitHub OAuth flow to connect GitHub account.

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
        from urllib.parse import quote
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
        state = _create_github_oauth_state(user_id)

        # Get OAuth URL - this may raise GitHubError if not configured
        oauth_url = github_service.get_oauth_url(state=state, redirect_uri=redirect_uri)

        logger.info(
            "Initiating GitHub OAuth flow",
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
async def github_callback(
    request: Request,
    code: str = Query(..., description="GitHub OAuth authorization code"),
    state: Optional[str] = Query(None, description="State parameter for CSRF protection"),
):
    """Handle GitHub OAuth callback."""
    github_service = get_github_service()

    # Validate state and get user_id
    user_id = _consume_github_oauth_state(state)

    # Get redirect URI (must match the one sent to GitHub)
    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/auth/github/callback"

    logger.info(
        "GitHub OAuth callback received",
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
            f"GitHub connected successfully for user {user_id} (GitHub: {github_username})"
        )

        # Redirect back to settings page with success message
        return RedirectResponse(
            url=f"{base_url}/settings#github=connected",
            status_code=302,
        )

    except GitHubError as e:
        logger.error(f"GitHub OAuth error: {e}")
        return RedirectResponse(
            url=f"{base_url}/settings#github=error&message={e.message}",
            status_code=302,
        )
    except Exception as e:
        logger.exception(f"Unexpected error during GitHub OAuth: {e}")
        return RedirectResponse(
            url=f"{base_url}/settings#github=error&message=OAuth+failed",
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
