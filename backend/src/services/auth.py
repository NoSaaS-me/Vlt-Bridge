"""Authentication helpers (JWT + HF OAuth placeholder)."""

from __future__ import annotations

import abc
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List

import jwt
from fastapi import status

from ..models.auth import JWTPayload
from .config import AppConfig, get_config


class AuthError(Exception):
    """Domain-specific authentication error."""

    def __init__(
        self,
        error: str,
        message: str,
        *,
        status_code: int = status.HTTP_401_UNAUTHORIZED,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error = error
        self.message = message
        self.status_code = status_code
        self.detail = detail or {}


class TokenValidator(abc.ABC):
    """Abstract base class for token validation strategies."""

    @abc.abstractmethod
    def validate(self, token: str) -> Optional[JWTPayload]:
        """
        Validate the token and return payload if valid, or None if this validator
        does not recognize the token (allow fallthrough).
        Raises AuthError if token is recognized but invalid/expired.
        """
        pass


class StaticTokenValidator(TokenValidator):
    """Validates against a configured static token (e.g. local dev or service token)."""

    def __init__(self, static_token: Optional[str], user_id: str):
        self.static_token = static_token
        self.user_id = user_id

    def validate(self, token: str) -> Optional[JWTPayload]:
        if self.static_token and token == self.static_token:
            # Return a long-lived payload for the static user
            now = datetime.now(timezone.utc)
            return JWTPayload(
                sub=self.user_id,
                iat=int(now.timestamp()),
                exp=int((now + timedelta(days=365)).timestamp()),
            )
        return None


class JWTValidator(TokenValidator):
    """Validates standard JWT tokens signed by the application secret."""

    def __init__(self, config: AppConfig, algorithm: str = "HS256"):
        self.config = config
        self.algorithm = algorithm

    def _require_secret(self) -> str:
        secret = self.config.jwt_secret_key
        if not secret:
            # If strictly in dev mode, allow a fallback, otherwise fail
            # logic moved from old AuthService
            env = os.getenv("ENVIRONMENT", "").lower()
            is_dev = env in ("development", "dev")
            if is_dev and self.config.enable_local_mode and self.config.local_dev_token:
                 return "local-dev-secret-key-123"

            raise AuthError(
                "missing_jwt_secret",
                "JWT secret is not configured.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return secret

    def validate(self, token: str) -> Optional[JWTPayload]:
        try:
            secret = self._require_secret()
            decoded = jwt.decode(token, secret, algorithms=[self.algorithm])
            return JWTPayload(**decoded)
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("token_expired", "Token expired") from exc
        except jwt.DecodeError:
            # Token is malformed (not a JWT) - return None to allow other validators
            # or fall through to generic "Invalid credentials"
            return None
        except jwt.InvalidTokenError as exc:
            # Other JWT errors (e.g. invalid signature, bad audience)
            raise AuthError("invalid_token", f"Invalid token: {exc}") from exc


class AuthService:
    """Issue and validate tokens using configured strategies."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        algorithm: str = "HS256",
        token_ttl_days: int = 90,
    ) -> None:
        self.config = config or get_config()
        self.algorithm = algorithm
        self.token_ttl_days = token_ttl_days
        
        # Initialize strategies
        self.validators: List[TokenValidator] = []
        
        # 1. Local Dev Token (Highest priority)
        if self.config.enable_local_mode:
            self.validators.append(
                StaticTokenValidator(self.config.local_dev_token, "demo-user")
            )
            
        # 2. ChatGPT Service Token
        if self.config.chatgpt_service_token:
            self.validators.append(
                StaticTokenValidator(self.config.chatgpt_service_token, "demo-user")
            )
            
        # 3. JWT Validator (Standard)
        self.validators.append(JWTValidator(self.config, algorithm))

    def validate_jwt(self, token: str) -> JWTPayload:
        """
        Validate a token against all registered strategies.
        Returns the first successful payload.
        Raises AuthError if no validator accepts it or if validation explicitly fails.
        """
        last_error = None
        
        for validator in self.validators:
            try:
                payload = validator.validate(token)
                if payload:
                    return payload
            except AuthError as e:
                # Validator recognized the token type but rejected it (e.g. expired)
                # Stop chain and raise immediately
                raise e
            except Exception as e:
                # Unexpected error, capture and continue
                last_error = e
        
        # If we get here, no validator returned a payload.
        # If the JWT validator raised an exception (e.g. malformed), it usually raises AuthError.
        # If it didn't (e.g. because secret was missing and it fell through?), we raise generic.
        if last_error:
             raise AuthError("invalid_token", f"Token validation failed: {last_error}")
        
        raise AuthError("invalid_token", "Invalid authentication credentials")

    # ... methods for creating tokens remain similar ...
    def _require_secret(self) -> str:
        # Delegate to JWT validator logic or duplicate simple check for issuance
        # Re-implement simple check for issuance context
        secret = self.config.jwt_secret_key
        if not secret:
             # Allow fallback for issuance in dev mode
            env = os.getenv("ENVIRONMENT", "").lower()
            is_dev = env in ("development", "dev")
            if is_dev and self.config.enable_local_mode:
                 return "local-dev-secret-key-123"
            raise AuthError("missing_jwt_secret", "JWT secret not configured", status_code=500)
        return secret

    def _build_payload(
        self, user_id: str, expires_in: Optional[timedelta] = None
    ) -> JWTPayload:
        now = datetime.now(timezone.utc)
        lifetime = expires_in or timedelta(days=self.token_ttl_days)
        return JWTPayload(
            sub=user_id,
            iat=int(now.timestamp()),
            exp=int((now + lifetime).timestamp()),
        )

    def create_jwt(
        self, user_id: str, *, expires_in: Optional[timedelta] = None
    ) -> str:
        """Create a signed JWT for the given user."""
        payload = self._build_payload(user_id, expires_in)
        return jwt.encode(
            payload.model_dump(),
            self._require_secret(),
            algorithm=self.algorithm,
        )

    def issue_token_response(
        self, user_id: str, *, expires_in: Optional[timedelta] = None
    ) -> tuple[str, datetime]:
        """Return token string and expiry timestamp (helper for API routes)."""
        payload = self._build_payload(user_id, expires_in)
        token = jwt.encode(
            payload.model_dump(),
            self._require_secret(),
            algorithm=self.algorithm,
        )
        expires_at = datetime.fromtimestamp(payload.exp, tz=timezone.utc)
        return token, expires_at

    def exchange_hf_oauth_code(self, code: str) -> Dict[str, Any]:
        """Placeholder for Hugging Face OAuth code exchange."""
        raise NotImplementedError("HF OAuth integration not implemented yet")


__all__ = ["AuthService", "AuthError", "TokenValidator", "StaticTokenValidator", "JWTValidator"]
