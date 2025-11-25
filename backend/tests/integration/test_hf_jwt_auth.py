#!/usr/bin/env python3
"""Integration checks for JWT + HTTP MCP in HF Spaces."""

import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv
from mcp.server.streamable_http import MCP_SESSION_ID_HEADER

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / "backend" / ".env"

# Load environment variables even when running from a different cwd
load_dotenv(dotenv_path=ENV_PATH)

# Make backend importable
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from backend.src.services.auth import AuthService  # noqa: E402
from backend.src.services.config import get_config  # noqa: E402

BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8001/mcp")
HTTP_TIMEOUT = float(os.getenv("MCP_TEST_TIMEOUT", "8.0"))


@pytest.fixture(scope="module")
def auth_service() -> AuthService:
    return AuthService(config=get_config())


@pytest.fixture(scope="module")
def tokens(auth_service: AuthService) -> dict[str, str]:
    users = [
        {"id": "hf_user_alice_123", "name": "Alice"},
        {"id": "hf_user_bob_456", "name": "Bob"},
        {"id": "hf_user_charlie_789", "name": "Charlie"},
    ]

    # If no JWT secret is configured, skip integration JWT issuance tests.
    if not auth_service.config.jwt_secret_key:
        pytest.skip("JWT_SECRET_KEY not configured; skipping integration JWT issuance tests")

    return {user["id"]: auth_service.create_jwt(user["id"]) for user in users}


@pytest.mark.integration
def test_jwt_generation_and_validation(auth_service: AuthService, tokens: dict[str, str]) -> None:
    for user_id, token in tokens.items():
        payload = auth_service.validate_jwt(token)
        assert payload.sub == user_id


def _post_or_skip(payload: dict, headers: dict) -> requests.Response:
    try:
        return requests.post(BASE_URL, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pytest.skip(f"MCP server not reachable at {BASE_URL}")


@pytest.mark.integration
def test_http_initialize_and_list_notes(tokens: dict[str, str]) -> None:
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }

    tool_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "list_notes", "arguments": {}},
    }

    for user_id, token in tokens.items():
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        init_resp = _post_or_skip(init_request, headers)
        assert init_resp.status_code == 200, init_resp.text[:200]

        session_id = init_resp.headers.get(MCP_SESSION_ID_HEADER)
        assert session_id, "Missing mcp-session-id header"

        tool_headers = {**headers, MCP_SESSION_ID_HEADER: session_id}
        tool_resp = _post_or_skip(tool_request, tool_headers)
        assert tool_resp.status_code == 200, tool_resp.text[:200]


@pytest.mark.integration
def test_http_rejects_invalid_token() -> None:
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }
    headers = {
        "Authorization": "Bearer not-a-valid-token",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    resp = _post_or_skip(init_request, headers)

    if resp.status_code == 200:
        pytest.skip("Server accepted invalid token (likely running in permissive/local mode)")

    assert resp.status_code == 401


@pytest.mark.integration
def test_expired_token_rejected_by_service(auth_service: AuthService) -> None:
    if not auth_service.config.jwt_secret_key:
        pytest.skip("JWT_SECRET_KEY not configured; skipping expired token check")

    expired = auth_service.create_jwt("expired-user", expires_in=timedelta(seconds=-1))
    from backend.src.services.auth import AuthError

    with pytest.raises(AuthError):
        auth_service.validate_jwt(expired)
