"""Security headers middleware for FastAPI/Starlette applications."""

from __future__ import annotations

from typing import Dict, Optional

from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers to all HTTP responses.

    This middleware injects security-related HTTP headers to protect against
    common web vulnerabilities including clickjacking, MIME-type sniffing,
    XSS attacks, and information leakage.

    Default headers applied:
    - X-Content-Type-Options: nosniff (prevent MIME-type sniffing)
    - X-Frame-Options: DENY (prevent clickjacking)
    - X-XSS-Protection: 1; mode=block (enable XSS filter)
    - Referrer-Policy: strict-origin-when-cross-origin (limit referrer info)
    - Permissions-Policy: geolocation=(), microphone=(), camera=() (restrict features)
    - Content-Security-Policy: restrictive policy for Document-MCP application

    Args:
        app: The ASGI application
        csp_policy: Custom Content-Security-Policy directive (optional)
        frame_options: X-Frame-Options value - DENY, SAMEORIGIN, or custom (optional)
        enable_hsts: Whether to enable Strict-Transport-Security (default: False)
        hsts_max_age: Max age for HSTS header in seconds (default: 31536000 / 1 year)
    """

    def __init__(
        self,
        app,
        csp_policy: Optional[str] = None,
        frame_options: str = "DENY",
        enable_hsts: bool = False,
        hsts_max_age: int = 31536000,
    ):
        super().__init__(app)
        self.csp_policy = csp_policy or self._default_csp_policy()
        self.frame_options = frame_options
        self.enable_hsts = enable_hsts
        self.hsts_max_age = hsts_max_age

    def _default_csp_policy(self) -> str:
        """
        Generate default Content-Security-Policy for Document-MCP application.

        This policy is designed to work with the React frontend while maintaining security:
        - Allows scripts from self and inline (required for Vite/React)
        - Allows styles from self and inline (required for styled components)
        - Restricts frames, objects, and base-uri
        - Upgrades insecure requests when possible
        """
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "object-src 'none'; "
            "upgrade-insecure-requests"
        )

    def _get_security_headers(self) -> Dict[str, str]:
        """Build the dictionary of security headers to apply."""
        headers = {
            # Prevent MIME-type sniffing
            "X-Content-Type-Options": "nosniff",
            # Prevent clickjacking
            "X-Frame-Options": self.frame_options,
            # Enable XSS filter (legacy but still useful for older browsers)
            "X-XSS-Protection": "1; mode=block",
            # Control referrer information
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Restrict browser features
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            # Content Security Policy
            "Content-Security-Policy": self.csp_policy,
        }

        # Only add HSTS in production (when enabled)
        if self.enable_hsts:
            headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains"
            )

        return headers

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request and inject security headers into the response.

        Args:
            request: The incoming HTTP request
            call_next: Callable to invoke the next middleware/endpoint

        Returns:
            Response with security headers added
        """
        # Process the request through the rest of the middleware/endpoint stack
        response = await call_next(request)

        # Add security headers to the response
        security_headers = self._get_security_headers()
        for header_name, header_value in security_headers.items():
            # Use MutableHeaders to safely modify response headers
            # Only set if not already present (allow endpoints to override)
            if header_name not in response.headers:
                response.headers[header_name] = header_value

        return response


__all__ = ["SecurityHeadersMiddleware"]
