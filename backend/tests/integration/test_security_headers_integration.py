"""Integration tests for security headers in HTTP responses."""

import pytest
from starlette.testclient import TestClient

from backend.src.api.main import app


# Expected security headers that should be present in all responses
EXPECTED_SECURITY_HEADERS = [
    "X-Content-Type-Options",
    "X-Frame-Options",
    "X-XSS-Protection",
    "Referrer-Policy",
    "Permissions-Policy",
    "Content-Security-Policy",
]


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.mark.integration
def test_security_headers_on_health_endpoint(client):
    """Test that security headers are present on the /health endpoint."""
    response = client.get("/health")

    # Verify successful response
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

    # Verify all security headers are present
    for header in EXPECTED_SECURITY_HEADERS:
        assert header in response.headers, f"Missing security header: {header}"

    # Verify specific header values
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Permissions-Policy"] == "geolocation=(), microphone=(), camera=()"

    # Verify CSP is present and non-empty
    assert len(response.headers["Content-Security-Policy"]) > 0

    # HSTS should not be present by default (requires explicit configuration)
    assert "Strict-Transport-Security" not in response.headers


@pytest.mark.integration
def test_security_headers_on_api_notes_endpoint(client):
    """Test that security headers are present on /api/notes endpoint."""
    # This endpoint requires authentication, but we're testing that headers
    # are present even if we don't provide valid auth
    response = client.get("/api/notes")

    # The response may be 401 (unauthorized) or 200 (if running in local mode)
    # Either way, security headers should be present
    assert response.status_code in [200, 401]

    # Verify all security headers are present
    for header in EXPECTED_SECURITY_HEADERS:
        assert header in response.headers, f"Missing security header: {header}"

    # Verify specific header values
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"


@pytest.mark.integration
def test_security_headers_on_404_error(client):
    """Test that security headers are present on 404 error responses."""
    response = client.get("/api/nonexistent-endpoint")

    # Verify 404 error
    assert response.status_code == 404

    # Verify all security headers are present even on error
    for header in EXPECTED_SECURITY_HEADERS:
        assert header in response.headers, f"Missing security header on 404: {header}"

    # Verify specific header values
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Content-Security-Policy"] is not None


@pytest.mark.integration
def test_security_headers_on_mcp_endpoint(client):
    """Test that security headers are present on /mcp endpoint."""
    # The MCP endpoint expects specific JSON-RPC payload, but we're just
    # checking that security headers are present regardless of payload validity
    response = client.get("/mcp")

    # Response status may vary (200, 400, etc.) depending on payload
    # but headers should always be present
    assert response.status_code in [200, 400, 401, 405]

    # Verify all security headers are present
    for header in EXPECTED_SECURITY_HEADERS:
        assert header in response.headers, f"Missing security header on MCP endpoint: {header}"


@pytest.mark.integration
def test_security_headers_on_post_request(client):
    """Test that security headers are present on POST requests."""
    # Try to create a note without proper authentication
    # We expect this to fail, but headers should still be present
    response = client.post(
        "/api/notes/test-note.md",
        json={"content": "Test content"}
    )

    # Response status may vary (401, 403, etc.) but headers should be present
    assert response.status_code in [200, 401, 403, 422]

    # Verify all security headers are present
    for header in EXPECTED_SECURITY_HEADERS:
        assert header in response.headers, f"Missing security header on POST: {header}"

    # Verify specific header values
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


@pytest.mark.integration
def test_csp_policy_contains_expected_directives(client):
    """Test that CSP policy contains expected security directives."""
    response = client.get("/health")

    assert response.status_code == 200

    csp = response.headers.get("Content-Security-Policy", "")

    # Verify key CSP directives are present
    assert "default-src" in csp
    assert "script-src" in csp
    assert "style-src" in csp
    assert "frame-ancestors" in csp
    assert "object-src 'none'" in csp

    # Verify restrictive directives
    assert "frame-ancestors 'none'" in csp


@pytest.mark.integration
def test_headers_consistency_across_endpoints(client):
    """Test that security headers are consistent across different endpoints."""
    endpoints = [
        "/health",
        "/api/notes",  # May require auth but should still have headers
    ]

    header_sets = []

    for endpoint in endpoints:
        response = client.get(endpoint)
        # Collect the set of security headers present
        headers_present = {
            header for header in EXPECTED_SECURITY_HEADERS
            if header in response.headers
        }
        header_sets.append(headers_present)

    # All endpoints should have the same set of security headers
    first_set = header_sets[0]
    for i, header_set in enumerate(header_sets[1:], 1):
        assert header_set == first_set, (
            f"Endpoint {endpoints[i]} has different security headers than {endpoints[0]}"
        )


@pytest.mark.integration
def test_security_headers_values_match_configuration(client):
    """Test that security header values match expected configuration."""
    response = client.get("/health")

    assert response.status_code == 200

    # Test specific header values match what middleware should set
    headers = response.headers

    # These should match the defaults in SecurityHeadersMiddleware
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["X-XSS-Protection"] == "1; mode=block"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert headers["Permissions-Policy"] == "geolocation=(), microphone=(), camera=()"

    # CSP should be non-empty
    assert len(headers["Content-Security-Policy"]) > 0


@pytest.mark.integration
def test_no_hsts_header_by_default(client):
    """Test that HSTS header is not present by default (only for HTTPS in production)."""
    endpoints = ["/health", "/api/notes"]

    for endpoint in endpoints:
        response = client.get(endpoint)
        # HSTS should be disabled by default since it's only appropriate for HTTPS
        assert "Strict-Transport-Security" not in response.headers, (
            f"HSTS header should not be present by default on {endpoint}"
        )
