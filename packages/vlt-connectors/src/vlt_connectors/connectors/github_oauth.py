"""GitHub OAuth2 connector — uses GitHub's OAuth App authorization flow."""
from __future__ import annotations
import os
from typing import Any, ClassVar
import httpx

from ..base import ActionConnector
from ..models import ConnectorAction, ConnectorParam, CredentialField
from ..oauth import OAuthActionConnector, OAuthConfig

GITHUB_API_BASE = "https://api.github.com"


class GitHubOAuthConnector(OAuthActionConnector, ActionConnector):
    """GitHub connector using OAuth2 App authorization (no PAT required)."""
    name = "github"
    display_name = "GitHub"
    description = (
        "GitHub repository, issue, and pull request management. "
        "Uses OAuth2 — click Connect to authorize with your GitHub account."
    )
    connector_type: ClassVar[str] = "action"

    # credential_fields intentionally empty — OAuth2 tokens are managed automatically
    credential_fields: ClassVar[list[CredentialField]] = []

    actions: ClassVar[list[ConnectorAction]] = [
        ConnectorAction(
            name="list_repos",
            description="List the authenticated user's repositories.",
            params=[
                ConnectorParam(name="per_page", description="Results per page (max 100)", required=False, default="30"),
                ConnectorParam(name="sort", description="Sort by: created, updated, pushed, full_name", required=False, default="updated"),
            ],
        ),
        ConnectorAction(
            name="list_issues",
            description="List open issues for a repository.",
            params=[
                ConnectorParam(name="owner", description="Repository owner", required=True),
                ConnectorParam(name="repo", description="Repository name", required=True),
                ConnectorParam(name="state", description="Issue state: open, closed, all", required=False, default="open"),
                ConnectorParam(name="per_page", description="Results per page (max 100)", required=False, default="20"),
            ],
        ),
        ConnectorAction(
            name="create_issue",
            description="Create a new issue in a GitHub repository.",
            params=[
                ConnectorParam(name="owner", description="Repository owner", required=True),
                ConnectorParam(name="repo", description="Repository name", required=True),
                ConnectorParam(name="title", description="Issue title", required=True),
                ConnectorParam(name="body", description="Issue body (Markdown)", required=False, default=""),
                ConnectorParam(name="labels", description="Comma-separated label names", required=False, default=""),
            ],
        ),
        ConnectorAction(
            name="get_user",
            description="Get the authenticated GitHub user's profile.",
            params=[],
        ),
        ConnectorAction(
            name="search_repos",
            description="Search GitHub repositories by query string.",
            params=[
                ConnectorParam(name="q", description="Search query (GitHub search syntax)", required=True),
                ConnectorParam(name="per_page", description="Results per page (max 100)", required=False, default="10"),
                ConnectorParam(name="sort", description="Sort by: stars, forks, updated", required=False, default="stars"),
            ],
        ),
    ]

    def get_oauth_config(self) -> OAuthConfig:
        return OAuthConfig(
            client_id=os.environ.get("GITHUB_CLIENT_ID", ""),
            client_secret=os.environ.get("GITHUB_CLIENT_SECRET", ""),
            authorize_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            scopes=["repo", "user:email", "read:org"],
            pkce=False,  # GitHub OAuth Apps do not support PKCE
            auth_header_template="Authorization: token {access_token}",
        )

    def _get_access_token(self, credentials: dict[str, str]) -> str:
        token = credentials.get("__oauth_access_token", "").strip()
        if not token:
            raise ValueError(
                "GitHub is not connected. Please click 'Connect' in the Connectors settings."
            )
        return token

    async def invoke(self, action: str, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        token = self._get_access_token(credentials)
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        async with httpx.AsyncClient(timeout=15.0, base_url=GITHUB_API_BASE) as client:
            if action == "list_repos":
                resp = await client.get(
                    "/user/repos",
                    params={"per_page": params.get("per_page", 30), "sort": params.get("sort", "updated")},
                    headers=headers,
                )

            elif action == "list_issues":
                resp = await client.get(
                    f"/repos/{params['owner']}/{params['repo']}/issues",
                    params={"state": params.get("state", "open"), "per_page": params.get("per_page", 20)},
                    headers=headers,
                )

            elif action == "create_issue":
                body: dict[str, Any] = {"title": params["title"]}
                if params.get("body"):
                    body["body"] = params["body"]
                if params.get("labels"):
                    body["labels"] = [lbl.strip() for lbl in params["labels"].split(",") if lbl.strip()]
                resp = await client.post(
                    f"/repos/{params['owner']}/{params['repo']}/issues",
                    json=body,
                    headers=headers,
                )

            elif action == "get_user":
                resp = await client.get("/user", headers=headers)

            elif action == "search_repos":
                resp = await client.get(
                    "/search/repositories",
                    params={
                        "q": params["q"],
                        "per_page": params.get("per_page", 10),
                        "sort": params.get("sort", "stars"),
                    },
                    headers=headers,
                )

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("message", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            return {"success": False, "error": f"GitHub API error {resp.status_code}: {detail}"}

        return {"success": True, "data": resp.json()}
