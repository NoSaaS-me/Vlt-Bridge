"""GitHub integration service for Oracle agent.

Provides file reading and code search from GitHub repositories.
Handles OAuth token management, API rate limiting, and caching.
Falls back to raw URL fetching for public repositories without auth.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .database import DatabaseService

logger = logging.getLogger(__name__)


# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com"
GITHUB_OAUTH_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_OAUTH_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_OAUTH_USER_URL = "https://api.github.com/user"

# Rate limiting configuration
RATE_LIMIT_REMAINING_THRESHOLD = 10  # Warn when this few requests remain
RATE_LIMIT_BACKOFF_SECONDS = 60  # Wait time when rate limited

# Cache configuration
CACHE_TTL_SECONDS = 300  # 5 minutes for file content cache
SEARCH_CACHE_TTL_SECONDS = 60  # 1 minute for search results


class GitHubError(Exception):
    """Base exception for GitHub service errors."""

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response


class GitHubRateLimitError(GitHubError):
    """Raised when GitHub API rate limit is exceeded."""

    def __init__(self, reset_time: int, message: str = None):
        self.reset_time = reset_time
        self.reset_datetime = datetime.fromtimestamp(reset_time, tz=timezone.utc)
        msg = message or f"GitHub API rate limit exceeded. Resets at {self.reset_datetime.isoformat()}"
        super().__init__(msg, status_code=403)


class GitHubAuthError(GitHubError):
    """Raised when GitHub authentication fails."""

    def __init__(self, message: str = "GitHub authentication failed"):
        super().__init__(message, status_code=401)


class GitHubNotFoundError(GitHubError):
    """Raised when a GitHub resource is not found."""

    def __init__(self, resource: str):
        super().__init__(f"GitHub resource not found: {resource}", status_code=404)


class GitHubService:
    """
    Service for interacting with GitHub API.

    Provides:
    - File reading from repositories (with fallback to raw URLs)
    - Code search across repositories
    - OAuth token management
    - Rate limiting awareness
    - Response caching

    Usage:
        service = GitHubService()
        content = await service.read_file(
            user_id="user-123",
            repo="owner/repo",
            path="src/main.py",
            branch="main"
        )
    """

    def __init__(self, db_service: Optional[DatabaseService] = None):
        """Initialize the GitHub service.

        Args:
            db_service: Database service for token storage and caching.
                        If None, a default instance is created.
        """
        self.db = db_service or DatabaseService()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._rate_limit_remaining: Optional[int] = None
        self._rate_limit_reset: Optional[int] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
            )
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    def _get_oauth_config(self) -> Tuple[str, str]:
        """Get GitHub OAuth configuration from environment.

        Returns:
            Tuple of (client_id, client_secret)

        Raises:
            GitHubError: If OAuth is not configured
        """
        client_id = os.environ.get("GITHUB_OAUTH_CLIENT_ID")
        client_secret = os.environ.get("GITHUB_OAUTH_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise GitHubError(
                "GitHub OAuth not configured. Set GITHUB_OAUTH_CLIENT_ID and GITHUB_OAUTH_CLIENT_SECRET.",
                status_code=501,
            )

        return client_id, client_secret

    def get_oauth_url(self, state: str, redirect_uri: str) -> str:
        """Generate GitHub OAuth authorization URL.

        Args:
            state: CSRF protection state token
            redirect_uri: OAuth callback URL

        Returns:
            Full OAuth authorization URL
        """
        client_id, _ = self._get_oauth_config()

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": "repo read:user",  # repo for private repos, read:user for profile
            "state": state,
        }

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{GITHUB_OAUTH_AUTHORIZE_URL}?{query_string}"

    async def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str,
    ) -> Tuple[str, str]:
        """Exchange OAuth code for access token.

        Args:
            code: Authorization code from OAuth callback
            redirect_uri: Same redirect URI used in authorization

        Returns:
            Tuple of (access_token, github_username)

        Raises:
            GitHubAuthError: If token exchange fails
        """
        client_id, client_secret = self._get_oauth_config()
        client = await self._get_client()

        try:
            # Exchange code for token
            token_response = await client.post(
                GITHUB_OAUTH_TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                headers={"Accept": "application/json"},
            )

            if token_response.status_code != 200:
                logger.error(f"GitHub token exchange failed: {token_response.text}")
                raise GitHubAuthError("Failed to exchange authorization code for token")

            token_data = token_response.json()

            if "error" in token_data:
                logger.error(f"GitHub OAuth error: {token_data}")
                raise GitHubAuthError(
                    token_data.get("error_description", token_data.get("error"))
                )

            access_token = token_data.get("access_token")
            if not access_token:
                raise GitHubAuthError("No access token in response")

            # Get user profile to get username
            user_response = await client.get(
                GITHUB_OAUTH_USER_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )

            if user_response.status_code != 200:
                logger.error(f"Failed to fetch GitHub user: {user_response.text}")
                raise GitHubAuthError("Failed to fetch GitHub user profile")

            user_data = user_response.json()
            username = user_data.get("login")

            if not username:
                raise GitHubAuthError("No username in GitHub user profile")

            return access_token, username

        except httpx.HTTPError as e:
            logger.exception(f"HTTP error during GitHub OAuth: {e}")
            raise GitHubAuthError(f"Network error during authentication: {e}")

    def store_token(self, user_id: str, token: str, username: str) -> None:
        """Store GitHub access token for a user.

        The token is stored encrypted in the database (simple base64 for now,
        should use proper encryption in production).

        Args:
            user_id: User identifier
            token: GitHub access token
            username: GitHub username
        """
        conn = self.db.connect()
        try:
            # Simple encoding (in production, use proper encryption)
            encoded_token = base64.b64encode(token.encode()).decode()
            now = datetime.now(timezone.utc).isoformat()

            with conn:
                conn.execute(
                    """
                    UPDATE user_settings
                    SET github_token_encrypted = ?, github_username = ?, updated = ?
                    WHERE user_id = ?
                    """,
                    (encoded_token, username, now, user_id),
                )

                # Create row if it doesn't exist
                if conn.total_changes == 0:
                    conn.execute(
                        """
                        INSERT INTO user_settings (
                            user_id, github_token_encrypted, github_username,
                            oracle_model, oracle_provider, subagent_model, subagent_provider,
                            thinking_enabled, chat_center_mode, librarian_timeout, max_context_nodes,
                            created, updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            encoded_token,
                            username,
                            "gemini-2.0-flash-exp",
                            "google",
                            "gemini-2.0-flash-exp",
                            "google",
                            0,
                            0,
                            1200,
                            30,
                            now,
                            now,
                        ),
                    )

            logger.info(f"Stored GitHub token for user {user_id} ({username})")
        finally:
            conn.close()

    def get_token(self, user_id: str) -> Optional[str]:
        """Get stored GitHub token for a user.

        Args:
            user_id: User identifier

        Returns:
            Decrypted token or None if not set
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                "SELECT github_token_encrypted FROM user_settings WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()

            if row and row["github_token_encrypted"]:
                # Simple decoding (in production, use proper decryption)
                return base64.b64decode(row["github_token_encrypted"]).decode()
            return None
        finally:
            conn.close()

    def get_github_username(self, user_id: str) -> Optional[str]:
        """Get stored GitHub username for a user.

        Args:
            user_id: User identifier

        Returns:
            GitHub username or None if not connected
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                "SELECT github_username FROM user_settings WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            return row["github_username"] if row else None
        finally:
            conn.close()

    def disconnect(self, user_id: str) -> None:
        """Disconnect GitHub from user account.

        Args:
            user_id: User identifier
        """
        conn = self.db.connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            with conn:
                conn.execute(
                    """
                    UPDATE user_settings
                    SET github_token_encrypted = NULL, github_username = NULL, updated = ?
                    WHERE user_id = ?
                    """,
                    (now, user_id),
                )
            logger.info(f"Disconnected GitHub for user {user_id}")
        finally:
            conn.close()

    def _update_rate_limits(self, headers: httpx.Headers) -> None:
        """Update rate limit tracking from response headers."""
        if "x-ratelimit-remaining" in headers:
            self._rate_limit_remaining = int(headers["x-ratelimit-remaining"])
        if "x-ratelimit-reset" in headers:
            self._rate_limit_reset = int(headers["x-ratelimit-reset"])

        if self._rate_limit_remaining is not None and self._rate_limit_remaining < RATE_LIMIT_REMAINING_THRESHOLD:
            logger.warning(
                f"GitHub rate limit low: {self._rate_limit_remaining} requests remaining"
            )

    async def _api_request(
        self,
        method: str,
        path: str,
        user_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        require_auth: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """Make a GitHub API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (without base URL)
            user_id: User ID for token lookup
            params: Query parameters
            require_auth: If True, raise error when no token available

        Returns:
            Tuple of (status_code, response_json)

        Raises:
            GitHubRateLimitError: When rate limit is exceeded
            GitHubAuthError: When auth is required but not available
        """
        client = await self._get_client()

        # Build headers
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "VltBridge-OracleAgent/1.0",
        }

        # Add auth if available
        token = self.get_token(user_id) if user_id else None
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif require_auth:
            raise GitHubAuthError("GitHub authentication required but no token available")

        # Make request
        url = f"{GITHUB_API_BASE}{path}"
        response = await client.request(method, url, params=params, headers=headers)

        # Track rate limits
        self._update_rate_limits(response.headers)

        # Handle rate limiting
        if response.status_code == 403:
            if "x-ratelimit-remaining" in response.headers and int(response.headers["x-ratelimit-remaining"]) == 0:
                reset_time = int(response.headers.get("x-ratelimit-reset", time.time() + 60))
                raise GitHubRateLimitError(reset_time)

        # Parse response
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = {"raw_content": response.text}

        return response.status_code, data

    def _get_cache_key(self, repo: str, path: str, ref: str) -> str:
        """Generate cache key for a file."""
        return hashlib.sha256(f"{repo}/{path}@{ref}".encode()).hexdigest()[:16]

    def _check_cache(
        self,
        user_id: str,
        repo: str,
        path: str,
        ref: str,
    ) -> Optional[Tuple[str, str]]:
        """Check cache for file content.

        Returns:
            Tuple of (content, etag) or None if not cached or expired
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                """
                SELECT content, etag, cached_at
                FROM github_cache
                WHERE repo = ? AND path = ? AND ref = ?
                """,
                (repo, path, ref),
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Check TTL
            cached_at = datetime.fromisoformat(row["cached_at"])
            age_seconds = (datetime.now(timezone.utc) - cached_at.replace(tzinfo=timezone.utc)).total_seconds()

            if age_seconds > CACHE_TTL_SECONDS:
                return None

            return row["content"], row["etag"]
        finally:
            conn.close()

    def _update_cache(
        self,
        repo: str,
        path: str,
        ref: str,
        content: str,
        etag: Optional[str] = None,
    ) -> None:
        """Update cache with file content."""
        conn = self.db.connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO github_cache (repo, path, ref, content, etag, cached_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (repo, path, ref, content, etag, now),
                )
        finally:
            conn.close()

    async def read_file(
        self,
        user_id: str,
        repo: str,
        path: str,
        branch: str = "main",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Read a file from a GitHub repository.

        Attempts to use the GitHub API with authentication if available.
        Falls back to raw URL for public repositories.

        Args:
            user_id: User ID for token lookup
            repo: Repository in "owner/repo" format
            path: Path to file within repository
            branch: Branch or ref (default: "main")
            use_cache: Whether to use cached content (default: True)

        Returns:
            Dict with:
                - content: File content as string
                - path: Full path
                - repo: Repository
                - branch: Branch/ref used
                - size: File size in bytes (if available)
                - sha: Git SHA (if available from API)
                - from_cache: Whether content came from cache
                - source: "api" or "raw" depending on how it was fetched
        """
        # Normalize inputs
        repo = repo.strip()
        path = path.strip().lstrip("/")
        branch = branch.strip() or "main"

        # Check cache first
        if use_cache:
            cached = self._check_cache(user_id, repo, path, branch)
            if cached:
                content, _ = cached
                logger.debug(f"Cache hit for {repo}/{path}@{branch}")
                return {
                    "content": content,
                    "path": path,
                    "repo": repo,
                    "branch": branch,
                    "from_cache": True,
                    "source": "cache",
                }

        # Try API first (works for private repos with auth)
        token = self.get_token(user_id)
        if token:
            try:
                result = await self._read_file_api(user_id, repo, path, branch)
                # Cache the result
                self._update_cache(repo, path, branch, result["content"], result.get("sha"))
                return result
            except GitHubNotFoundError:
                raise  # Re-raise not found errors
            except GitHubAuthError:
                # Token might be invalid, fall through to public fetch
                logger.warning(f"GitHub API auth failed for {user_id}, falling back to raw URL")
            except GitHubRateLimitError:
                # Rate limited, try raw URL fallback
                logger.warning("GitHub API rate limited, falling back to raw URL")

        # Fallback to raw URL (public repos only)
        try:
            result = await self._read_file_raw(repo, path, branch)
            # Cache the result
            self._update_cache(repo, path, branch, result["content"])
            return result
        except Exception as e:
            if token:
                raise GitHubError(f"Failed to read file from both API and raw URL: {e}")
            raise GitHubError(
                f"Cannot access {repo}/{path}. File may be private or not exist. "
                "Connect GitHub account for private repo access."
            )

    async def _read_file_api(
        self,
        user_id: str,
        repo: str,
        path: str,
        ref: str,
    ) -> Dict[str, Any]:
        """Read file using GitHub API."""
        api_path = f"/repos/{repo}/contents/{path}"
        params = {"ref": ref}

        status_code, data = await self._api_request(
            "GET",
            api_path,
            user_id=user_id,
            params=params,
            require_auth=True,
        )

        if status_code == 404:
            raise GitHubNotFoundError(f"{repo}/{path}@{ref}")

        if status_code == 401:
            raise GitHubAuthError()

        if status_code != 200:
            raise GitHubError(
                f"GitHub API error: {data.get('message', 'Unknown error')}",
                status_code=status_code,
                response=data,
            )

        # Handle file vs directory
        if isinstance(data, list):
            raise GitHubError(f"Path is a directory, not a file: {path}")

        if data.get("type") != "file":
            raise GitHubError(f"Path is not a file: {path} (type: {data.get('type')})")

        # Decode content (base64 encoded by GitHub)
        encoding = data.get("encoding", "base64")
        if encoding == "base64":
            content = base64.b64decode(data.get("content", "")).decode("utf-8")
        else:
            content = data.get("content", "")

        return {
            "content": content,
            "path": path,
            "repo": repo,
            "branch": ref,
            "size": data.get("size"),
            "sha": data.get("sha"),
            "from_cache": False,
            "source": "api",
        }

    async def _read_file_raw(
        self,
        repo: str,
        path: str,
        ref: str,
    ) -> Dict[str, Any]:
        """Read file from raw.githubusercontent.com (public repos only)."""
        client = await self._get_client()

        url = f"{GITHUB_RAW_BASE}/{repo}/{ref}/{path}"
        response = await client.get(
            url,
            headers={"User-Agent": "VltBridge-OracleAgent/1.0"},
        )

        if response.status_code == 404:
            raise GitHubNotFoundError(f"{repo}/{path}@{ref}")

        if response.status_code != 200:
            raise GitHubError(
                f"Failed to fetch file from raw URL: HTTP {response.status_code}",
                status_code=response.status_code,
            )

        content = response.text

        return {
            "content": content,
            "path": path,
            "repo": repo,
            "branch": ref,
            "size": len(content),
            "from_cache": False,
            "source": "raw",
        }

    async def search_code(
        self,
        user_id: str,
        query: str,
        repo: Optional[str] = None,
        language: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search code across GitHub.

        Args:
            user_id: User ID for token lookup
            query: Search query
            repo: Limit to specific repository (owner/repo format)
            language: Filter by programming language
            limit: Maximum results to return (default: 10)

        Returns:
            Dict with:
                - query: Original query
                - total_count: Total matches found
                - results: List of matching code items
                - incomplete: Whether results are incomplete
        """
        # Build query string
        q_parts = [query]
        if repo:
            q_parts.append(f"repo:{repo}")
        if language:
            q_parts.append(f"language:{language}")

        full_query = " ".join(q_parts)

        # Make search request
        params = {
            "q": full_query,
            "per_page": min(limit, 100),
        }

        try:
            status_code, data = await self._api_request(
                "GET",
                "/search/code",
                user_id=user_id,
                params=params,
            )

            if status_code == 403:
                # Check for rate limit
                if "rate limit" in str(data).lower():
                    raise GitHubRateLimitError(
                        int(time.time()) + 60,
                        "Code search rate limited. Please try again later.",
                    )
                raise GitHubError(
                    f"GitHub search forbidden: {data.get('message', 'Unknown error')}",
                    status_code=403,
                )

            if status_code == 401:
                raise GitHubAuthError()

            if status_code != 200:
                raise GitHubError(
                    f"GitHub search failed: {data.get('message', 'Unknown error')}",
                    status_code=status_code,
                    response=data,
                )

            # Format results
            results = []
            for item in data.get("items", [])[:limit]:
                results.append({
                    "path": item.get("path"),
                    "repo": item.get("repository", {}).get("full_name"),
                    "url": item.get("html_url"),
                    "sha": item.get("sha"),
                    "score": item.get("score"),
                    # Note: Content is not returned by search API
                    # Would need separate read_file call to get content
                })

            return {
                "query": full_query,
                "total_count": data.get("total_count", 0),
                "results": results,
                "incomplete": data.get("incomplete_results", False),
            }

        except GitHubRateLimitError:
            raise
        except GitHubAuthError:
            # Without auth, search may be more limited
            raise GitHubError(
                "GitHub code search requires authentication. Please connect your GitHub account."
            )

    async def get_rate_limit_status(self, user_id: str) -> Dict[str, Any]:
        """Get current rate limit status.

        Args:
            user_id: User ID for token lookup

        Returns:
            Dict with rate limit information
        """
        try:
            status_code, data = await self._api_request(
                "GET",
                "/rate_limit",
                user_id=user_id,
            )

            if status_code != 200:
                return {
                    "error": "Failed to get rate limit status",
                    "authenticated": self.get_token(user_id) is not None,
                }

            return {
                "core": data.get("rate", {}),
                "search": data.get("resources", {}).get("search", {}),
                "authenticated": self.get_token(user_id) is not None,
            }

        except Exception as e:
            logger.error(f"Failed to get rate limit status: {e}")
            return {
                "error": str(e),
                "authenticated": self.get_token(user_id) is not None,
            }


# Singleton instance
_github_service: Optional[GitHubService] = None


def get_github_service() -> GitHubService:
    """Get or create the GitHub service singleton."""
    global _github_service
    if _github_service is None:
        _github_service = GitHubService()
    return _github_service


__all__ = [
    "GitHubService",
    "GitHubError",
    "GitHubRateLimitError",
    "GitHubAuthError",
    "GitHubNotFoundError",
    "get_github_service",
]
