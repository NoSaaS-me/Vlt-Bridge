"""OAuth and authentication routes for Hugging Face integration."""

from __future__ import annotations

import logging
import secrets
import time
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from ...services.config import get_config
from ...services.seed import ensure_welcome_note

logger = logging.getLogger(__name__)

router = APIRouter()

OAUTH_STATE_TTL_SECONDS = 300
oauth_states: dict[str, float] = {}


def _create_oauth_state() -> str:
    """Generate a state token and store it with a timestamp."""
    now = time.time()
    # Garbage collect expired states
    expired = [state for state, ts in oauth_states.items() if now - ts > OAUTH_STATE_TTL_SECONDS]
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


class UserInfo(BaseModel):
    """Current user information."""

    user_id: str
    username: str
    email: Optional[str] = None


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
        }
    )
    
    return base_url


@router.get("/auth/login")
async def login(request: Request):
    """Redirect to Hugging Face OAuth authorization page."""
    config = get_config()
    
    if not config.hf_oauth_client_id:
        raise HTTPException(
            status_code=501,
            detail="OAuth not configured. Set HF_OAUTH_CLIENT_ID and HF_OAUTH_CLIENT_SECRET environment variables."
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
        }
    )
    
    return RedirectResponse(url=auth_url, status_code=302)


@router.get("/auth/callback")
async def callback(
    request: Request,
    code: str = Query(..., description="OAuth authorization code"),
    state: Optional[str] = Query(None, description="State parameter for CSRF protection"),
):
    """Handle OAuth callback from Hugging Face."""
    config = get_config()
    
    if not config.hf_oauth_client_id or not config.hf_oauth_client_secret:
        raise HTTPException(
            status_code=501,
            detail="OAuth not configured"
        )
    
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
        }
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
                    detail="Failed to exchange authorization code for token"
                )
            
            token_data = token_response.json()
            access_token = token_data.get("access_token")
            
            if not access_token:
                raise HTTPException(
                    status_code=400,
                    detail="No access token in response"
                )
            
            # Get user profile from HF
            user_response = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if user_response.status_code != 200:
                logger.error(f"User profile fetch failed: {user_response.text}")
                raise HTTPException(
                    status_code=400,
                    detail="Failed to fetch user profile"
                )
            
            user_data = user_response.json()
            username = user_data.get("name")
            email = user_data.get("email")
            
            if not username:
                raise HTTPException(
                    status_code=400,
                    detail="No username in user profile"
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
            
            jwt_token = jwt.encode(payload, config.jwt_secret_key, algorithm="HS256")
            
            logger.info(
                "OAuth successful",
                extra={
                    "username": username,
                    "user_id": user_id,
                    "email": email,
                }
            )
            
            # Redirect to frontend with token in URL hash
            frontend_url = base_url
            redirect_url = f"{frontend_url}/#token={jwt_token}"
            logger.info(f"Redirecting to frontend: {redirect_url}")
            return RedirectResponse(url=redirect_url, status_code=302)
    
    except httpx.HTTPError as e:
        logger.exception(f"HTTP error during OAuth: {e}")
        raise HTTPException(
            status_code=500,
            detail="OAuth flow failed due to network error"
        )
    except Exception as e:
        logger.exception(f"Unexpected error during OAuth: {e}")
        raise HTTPException(
            status_code=500,
            detail="OAuth flow failed"
        )


@router.get("/api/me", response_model=UserInfo)
async def get_current_user(request: Request):
    """Get current authenticated user information."""
    # Extract user_id from request state (set by auth middleware)
    from ..middleware.auth import get_user_id
    
    try:
        user_id = get_user_id()
        
        # In a real app, we might fetch more user details from DB
        # For now, just return the user_id
        return UserInfo(
            user_id=user_id,
            username=user_id,  # username is same as user_id in our system
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )


__all__ = ["router"]

