"""OAuth and authentication routes for Hugging Face integration."""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from ...services.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter()


class UserInfo(BaseModel):
    """Current user information."""

    user_id: str
    username: str
    email: Optional[str] = None


@router.get("/auth/login")
async def login():
    """Redirect to Hugging Face OAuth authorization page."""
    config = get_config()
    
    if not config.hf_oauth_client_id:
        raise HTTPException(
            status_code=501,
            detail="OAuth not configured. Set HF_OAUTH_CLIENT_ID and HF_OAUTH_CLIENT_SECRET environment variables."
        )
    
    # Construct HF OAuth URL
    base_url = "https://huggingface.co/oauth/authorize"
    params = {
        "client_id": config.hf_oauth_client_id,
        "redirect_uri": f"{config.hf_space_url}/auth/callback",
        "scope": "openid profile email",
        "response_type": "code",
        "state": "random_state_token",  # In production, use a secure random state
    }
    
    auth_url = f"{base_url}?{urlencode(params)}"
    logger.info(f"Redirecting to HF OAuth: {auth_url}")
    
    return RedirectResponse(url=auth_url)


@router.get("/auth/callback")
async def callback(
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
    
    try:
        # Exchange authorization code for access token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://huggingface.co/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": f"{config.hf_space_url}/auth/callback",
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
            payload = {
                "sub": user_id,
                "username": username,
                "email": email,
                "exp": datetime.now(timezone.utc) + timedelta(days=7),
                "iat": datetime.now(timezone.utc),
            }
            
            jwt_token = jwt.encode(payload, config.jwt_secret_key, algorithm="HS256")
            
            logger.info(f"OAuth successful for user: {username}")
            
            # Redirect to frontend with token in URL hash
            return RedirectResponse(url=f"/#token={jwt_token}")
    
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

