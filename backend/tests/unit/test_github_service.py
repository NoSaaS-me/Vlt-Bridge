"""Unit tests for GitHub service."""

import base64
import os
import pytest
import sqlite3
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from backend.src.services.github_service import (
    GitHubService,
    GitHubError,
    GitHubRateLimitError,
    GitHubAuthError,
    GitHubNotFoundError,
    get_github_service,
)
from backend.src.services.database import DatabaseService


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database with required schema."""
    db_path = tmp_path / "test.db"
    db = DatabaseService(db_path)
    db.initialize()
    return db


@pytest.fixture
def github_service(temp_db):
    """Create a GitHubService instance with temp database."""
    return GitHubService(db_service=temp_db)


class TestGitHubServiceTokenManagement:
    """Tests for OAuth token storage and retrieval."""

    def test_store_and_retrieve_token(self, github_service):
        """Test storing and retrieving a GitHub token."""
        user_id = "test-user"
        token = "ghp_test_token_123"
        username = "octocat"

        github_service.store_token(user_id, token, username)

        # Verify token retrieval
        retrieved = github_service.get_token(user_id)
        assert retrieved == token

        # Verify username retrieval
        retrieved_username = github_service.get_github_username(user_id)
        assert retrieved_username == username

    def test_get_token_nonexistent_user(self, github_service):
        """Test getting token for non-existent user returns None."""
        result = github_service.get_token("nonexistent")
        assert result is None

    def test_get_username_nonexistent_user(self, github_service):
        """Test getting username for non-existent user returns None."""
        result = github_service.get_github_username("nonexistent")
        assert result is None

    def test_disconnect(self, github_service):
        """Test disconnecting GitHub clears token and username."""
        user_id = "test-user"
        github_service.store_token(user_id, "token", "octocat")

        # Verify connected
        assert github_service.get_token(user_id) is not None

        # Disconnect
        github_service.disconnect(user_id)

        # Verify disconnected
        assert github_service.get_token(user_id) is None
        assert github_service.get_github_username(user_id) is None


class TestGitHubServiceOAuthConfig:
    """Tests for OAuth configuration."""

    def test_get_oauth_url_without_config(self, github_service):
        """Test OAuth URL generation fails without config."""
        with pytest.raises(GitHubError) as exc_info:
            github_service.get_oauth_url("state123", "http://localhost/callback")
        assert "not configured" in str(exc_info.value)

    def test_get_oauth_url_with_config(self, github_service, monkeypatch):
        """Test OAuth URL generation with proper config."""
        monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "test-client-id")
        monkeypatch.setenv("GITHUB_OAUTH_CLIENT_SECRET", "test-secret")

        url = github_service.get_oauth_url("state123", "http://localhost/callback")

        assert "github.com/login/oauth/authorize" in url
        assert "client_id=test-client-id" in url
        assert "state=state123" in url
        assert "redirect_uri=http://localhost/callback" in url
        assert "scope=repo" in url


class TestGitHubServiceFileReading:
    """Tests for file reading functionality."""

    @pytest.mark.asyncio
    async def test_read_file_raw_fallback_public_repo(self, github_service):
        """Test reading from public repo via raw URL when no auth."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "print('Hello, World!')"

        with patch.object(github_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await github_service.read_file(
                user_id="user1",
                repo="octocat/Hello-World",
                path="README.md",
                branch="main",
            )

            assert result["content"] == "print('Hello, World!')"
            assert result["repo"] == "octocat/Hello-World"
            assert result["path"] == "README.md"
            assert result["source"] == "raw"
            assert result["from_cache"] is False

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, github_service):
        """Test reading non-existent file raises GitHubError."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(github_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # When no auth token, it raises GitHubError (not GitHubNotFoundError)
            # because the file might be private
            with pytest.raises(GitHubError) as exc_info:
                await github_service.read_file(
                    user_id="user1",
                    repo="octocat/Hello-World",
                    path="nonexistent.md",
                )
            assert "private or not exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_read_file_api_with_auth(self, github_service):
        """Test reading file via API when authenticated."""
        # Store a token first
        github_service.store_token("user1", "ghp_token", "octocat")

        # Mock API response
        content = "Hello from API!"
        encoded_content = base64.b64encode(content.encode()).decode()
        api_response = {
            "type": "file",
            "encoding": "base64",
            "content": encoded_content,
            "size": len(content),
            "sha": "abc123",
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json = MagicMock(return_value=api_response)

        with patch.object(github_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await github_service.read_file(
                user_id="user1",
                repo="octocat/Hello-World",
                path="README.md",
            )

            assert result["content"] == content
            assert result["sha"] == "abc123"
            assert result["source"] == "api"


class TestGitHubServiceCaching:
    """Tests for file caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, github_service):
        """Test that cached content is returned."""
        # Manually populate cache
        github_service._update_cache(
            repo="octocat/test",
            path="file.py",
            ref="main",
            content="cached content",
            etag="etag123",
        )

        result = await github_service.read_file(
            user_id="user1",
            repo="octocat/test",
            path="file.py",
            branch="main",
            use_cache=True,
        )

        assert result["content"] == "cached content"
        assert result["from_cache"] is True
        assert result["source"] == "cache"

    @pytest.mark.asyncio
    async def test_cache_bypass(self, github_service):
        """Test that cache can be bypassed."""
        # Populate cache
        github_service._update_cache(
            repo="octocat/test",
            path="file.py",
            ref="main",
            content="old content",
        )

        # Mock fresh fetch
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "new content"

        with patch.object(github_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await github_service.read_file(
                user_id="user1",
                repo="octocat/test",
                path="file.py",
                branch="main",
                use_cache=False,
            )

            assert result["content"] == "new content"
            assert result["from_cache"] is False


class TestGitHubServiceCodeSearch:
    """Tests for code search functionality."""

    @pytest.mark.asyncio
    async def test_search_code_requires_auth(self, github_service):
        """Test that search fails gracefully without auth."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_response.json = MagicMock(return_value={"message": "Unauthorized"})

        with patch.object(github_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(GitHubError):
                await github_service.search_code(
                    user_id="user1",
                    query="test function",
                )

    @pytest.mark.asyncio
    async def test_search_code_with_results(self, github_service):
        """Test successful code search."""
        github_service.store_token("user1", "ghp_token", "octocat")

        search_response = {
            "total_count": 2,
            "incomplete_results": False,
            "items": [
                {
                    "path": "src/main.py",
                    "repository": {"full_name": "octocat/project"},
                    "html_url": "https://github.com/octocat/project/blob/main/src/main.py",
                    "sha": "abc123",
                    "score": 1.5,
                },
                {
                    "path": "tests/test_main.py",
                    "repository": {"full_name": "octocat/project"},
                    "html_url": "https://github.com/octocat/project/blob/main/tests/test_main.py",
                    "sha": "def456",
                    "score": 1.2,
                },
            ],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json = MagicMock(return_value=search_response)

        with patch.object(github_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await github_service.search_code(
                user_id="user1",
                query="def main",
                repo="octocat/project",
            )

            assert result["total_count"] == 2
            assert len(result["results"]) == 2
            assert result["results"][0]["path"] == "src/main.py"
            assert result["incomplete"] is False


class TestGitHubServiceRateLimiting:
    """Tests for rate limit handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, github_service):
        """Test that rate limit is properly detected and raised."""
        github_service.store_token("user1", "ghp_token", "octocat")

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "1234567890",
        }
        mock_response.json = MagicMock(return_value={"message": "Rate limit exceeded"})

        with patch.object(github_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(GitHubRateLimitError) as exc_info:
                await github_service._api_request(
                    "GET",
                    "/repos/test/test/contents/file.py",
                    user_id="user1",
                )

            assert exc_info.value.reset_time == 1234567890


class TestGitHubServiceSingleton:
    """Tests for singleton pattern."""

    def test_get_github_service_returns_singleton(self):
        """Test that get_github_service returns same instance."""
        # Reset singleton for test
        import backend.src.services.github_service as module
        module._github_service = None

        service1 = get_github_service()
        service2 = get_github_service()

        assert service1 is service2

        # Clean up
        module._github_service = None
