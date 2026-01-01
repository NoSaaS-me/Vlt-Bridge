"""Unit tests for SecurityHeadersMiddleware."""

import pytest
from unittest.mock import Mock, AsyncMock
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.applications import Starlette
from starlette.testclient import TestClient
from starlette.routing import Route

from backend.src.api.middleware.security_headers import SecurityHeadersMiddleware


@pytest.fixture
def mock_app():
    """Create a simple Starlette app for testing."""
    async def homepage(request):
        return Response("OK", status_code=200)

    async def json_endpoint(request):
        return JSONResponse({"message": "success"})

    async def error_endpoint(request):
        return Response("Error", status_code=500)

    async def empty_response(request):
        return Response("", status_code=204)

    app = Starlette(routes=[
        Route("/", homepage),
        Route("/json", json_endpoint),
        Route("/error", error_endpoint),
        Route("/empty", empty_response),
    ])

    return app


def test_middleware_adds_default_headers(mock_app):
    """Test that middleware adds all expected default security headers."""
    # Add middleware with defaults
    mock_app.add_middleware(SecurityHeadersMiddleware)
    client = TestClient(mock_app)

    response = client.get("/")

    # Verify all expected headers are present
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Permissions-Policy"] == "geolocation=(), microphone=(), camera=()"
    assert "Content-Security-Policy" in response.headers

    # HSTS should not be present by default
    assert "Strict-Transport-Security" not in response.headers


def test_middleware_default_csp_policy(mock_app):
    """Test that default CSP policy is correctly set."""
    mock_app.add_middleware(SecurityHeadersMiddleware)
    client = TestClient(mock_app)

    response = client.get("/")

    csp = response.headers["Content-Security-Policy"]

    # Verify key directives are present
    assert "default-src 'self'" in csp
    assert "script-src 'self' 'unsafe-inline' 'unsafe-eval'" in csp
    assert "style-src 'self' 'unsafe-inline'" in csp
    assert "img-src 'self' data: https:" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp
    assert "upgrade-insecure-requests" in csp


def test_middleware_custom_csp_policy(mock_app):
    """Test that custom CSP policy is respected."""
    custom_csp = "default-src 'none'; script-src 'self'"
    mock_app.add_middleware(SecurityHeadersMiddleware, csp_policy=custom_csp)
    client = TestClient(mock_app)

    response = client.get("/")

    assert response.headers["Content-Security-Policy"] == custom_csp


def test_middleware_custom_frame_options(mock_app):
    """Test that custom X-Frame-Options value is respected."""
    mock_app.add_middleware(SecurityHeadersMiddleware, frame_options="SAMEORIGIN")
    client = TestClient(mock_app)

    response = client.get("/")

    assert response.headers["X-Frame-Options"] == "SAMEORIGIN"


def test_middleware_hsts_enabled(mock_app):
    """Test that HSTS header is added when enabled."""
    mock_app.add_middleware(SecurityHeadersMiddleware, enable_hsts=True)
    client = TestClient(mock_app)

    response = client.get("/")

    assert "Strict-Transport-Security" in response.headers
    assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"


def test_middleware_hsts_custom_max_age(mock_app):
    """Test that custom HSTS max-age is respected."""
    custom_max_age = 86400  # 1 day
    mock_app.add_middleware(
        SecurityHeadersMiddleware,
        enable_hsts=True,
        hsts_max_age=custom_max_age
    )
    client = TestClient(mock_app)

    response = client.get("/")

    assert response.headers["Strict-Transport-Security"] == f"max-age={custom_max_age}; includeSubDomains"


def test_middleware_hsts_disabled_by_default(mock_app):
    """Test that HSTS is not added when disabled (default)."""
    mock_app.add_middleware(SecurityHeadersMiddleware, enable_hsts=False)
    client = TestClient(mock_app)

    response = client.get("/")

    assert "Strict-Transport-Security" not in response.headers


def test_middleware_on_json_response(mock_app):
    """Test that headers are added to JSON responses."""
    mock_app.add_middleware(SecurityHeadersMiddleware)
    client = TestClient(mock_app)

    response = client.get("/json")

    # Verify headers are present
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in response.headers

    # Verify JSON content is intact
    assert response.json() == {"message": "success"}


def test_middleware_on_error_response(mock_app):
    """Test that headers are added to error responses."""
    mock_app.add_middleware(SecurityHeadersMiddleware)
    client = TestClient(mock_app)

    response = client.get("/error")

    # Verify headers are present even on error
    assert response.status_code == 500
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in response.headers


def test_middleware_on_empty_response(mock_app):
    """Test that headers are added to empty responses (204 No Content)."""
    mock_app.add_middleware(SecurityHeadersMiddleware)
    client = TestClient(mock_app)

    response = client.get("/empty")

    # Verify headers are present even on empty response
    assert response.status_code == 204
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in response.headers


def test_middleware_does_not_override_existing_headers(mock_app):
    """Test that middleware does not override headers set by endpoints."""
    # Create an app with endpoint that sets custom header
    async def custom_header_endpoint(request):
        return Response(
            "OK",
            headers={"X-Frame-Options": "ALLOW-FROM https://example.com"}
        )

    app = Starlette(routes=[Route("/custom", custom_header_endpoint)])
    app.add_middleware(SecurityHeadersMiddleware, frame_options="DENY")
    client = TestClient(app)

    response = client.get("/custom")

    # Endpoint's custom header should be preserved
    assert response.headers["X-Frame-Options"] == "ALLOW-FROM https://example.com"
    # Other headers should still be added
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_middleware_all_headers_present_with_custom_config(mock_app):
    """Test that all headers are present when using custom configuration."""
    custom_csp = "default-src 'self'; script-src 'self'"
    mock_app.add_middleware(
        SecurityHeadersMiddleware,
        csp_policy=custom_csp,
        frame_options="SAMEORIGIN",
        enable_hsts=True,
        hsts_max_age=3600
    )
    client = TestClient(mock_app)

    response = client.get("/")

    # Verify all expected headers with custom values
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Permissions-Policy"] == "geolocation=(), microphone=(), camera=()"
    assert response.headers["Content-Security-Policy"] == custom_csp
    assert response.headers["Strict-Transport-Security"] == "max-age=3600; includeSubDomains"


def test_middleware_initialization_parameters():
    """Test that middleware stores initialization parameters correctly."""
    custom_csp = "default-src 'none'"
    middleware = SecurityHeadersMiddleware(
        Mock(),  # app
        csp_policy=custom_csp,
        frame_options="SAMEORIGIN",
        enable_hsts=True,
        hsts_max_age=7200
    )

    assert middleware.csp_policy == custom_csp
    assert middleware.frame_options == "SAMEORIGIN"
    assert middleware.enable_hsts is True
    assert middleware.hsts_max_age == 7200


def test_middleware_get_security_headers_method():
    """Test the _get_security_headers method directly."""
    middleware = SecurityHeadersMiddleware(
        Mock(),
        frame_options="DENY",
        enable_hsts=False
    )

    headers = middleware._get_security_headers()

    # Verify returned dictionary structure
    assert isinstance(headers, dict)
    assert "X-Content-Type-Options" in headers
    assert "X-Frame-Options" in headers
    assert "X-XSS-Protection" in headers
    assert "Referrer-Policy" in headers
    assert "Permissions-Policy" in headers
    assert "Content-Security-Policy" in headers
    assert "Strict-Transport-Security" not in headers  # HSTS disabled


def test_middleware_get_security_headers_with_hsts():
    """Test _get_security_headers includes HSTS when enabled."""
    middleware = SecurityHeadersMiddleware(
        Mock(),
        enable_hsts=True,
        hsts_max_age=86400
    )

    headers = middleware._get_security_headers()

    assert "Strict-Transport-Security" in headers
    assert headers["Strict-Transport-Security"] == "max-age=86400; includeSubDomains"


@pytest.mark.asyncio
async def test_middleware_dispatch_method():
    """Test the dispatch method directly."""
    # Create a mock request
    mock_request = Mock(spec=Request)

    # Create a mock call_next that returns a response
    async def mock_call_next(request):
        return Response("OK", status_code=200)

    # Create middleware instance
    middleware = SecurityHeadersMiddleware(Mock())

    # Call dispatch
    response = await middleware.dispatch(mock_request, mock_call_next)

    # Verify response has security headers
    assert "X-Content-Type-Options" in response.headers
    assert "X-Frame-Options" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.asyncio
async def test_middleware_dispatch_preserves_existing_headers():
    """Test that dispatch preserves headers already set in response."""
    mock_request = Mock(spec=Request)

    # Create a call_next that returns response with pre-set header
    async def mock_call_next(request):
        return Response(
            "OK",
            headers={"X-Frame-Options": "CUSTOM-VALUE"}
        )

    middleware = SecurityHeadersMiddleware(Mock(), frame_options="DENY")
    response = await middleware.dispatch(mock_request, mock_call_next)

    # Pre-existing header should be preserved
    assert response.headers["X-Frame-Options"] == "CUSTOM-VALUE"
    # Other headers should still be added
    assert response.headers["X-Content-Type-Options"] == "nosniff"
