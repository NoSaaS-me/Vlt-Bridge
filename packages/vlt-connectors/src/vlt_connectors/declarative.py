"""
Declarative connector loader: TOML files → ActionConnector instances.

TOML schema example:

    [connector]
    name = "github"
    display_name = "GitHub"
    description = "GitHub repository and issue management"
    auth_type = "api_key"      # "api_key" | "oauth2"

    [[credential_fields]]
    name = "api_key"
    label = "Personal Access Token"
    secret = true
    placeholder = "ghp_..."

    [[actions]]
    name = "list_repos"
    description = "List authenticated user's repositories"
    method = "GET"
    url = "https://api.github.com/user/repos"
    # auth_header: how to inject credentials (default: "Authorization: Bearer {api_key}")
    auth_header = "Authorization: token {api_key}"

    [[actions.params]]
    name = "per_page"
    description = "Results per page (max 100)"
    required = false
    default = "30"

    [[actions]]
    name = "create_issue"
    description = "Create a new issue in a repository"
    method = "POST"
    url = "https://api.github.com/repos/{owner}/{repo}/issues"

    [[actions.params]]
    name = "owner"
    description = "Repository owner"
    required = true

    [[actions.params]]
    name = "repo"
    description = "Repository name"
    required = true

    [[actions.params]]
    name = "title"
    description = "Issue title"
    required = true

    [[actions.params]]
    name = "body"
    description = "Issue body (Markdown)"
    required = false
"""
from __future__ import annotations
import tomllib
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

import httpx

from .base import ActionConnector
from .models import CredentialField, ConnectorAction, ConnectorParam

logger = logging.getLogger(__name__)


def _build_url(template: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Substitute {path_param} variables in URL template.
    Returns (url_with_substitutions, remaining_params).
    """
    used: set[str] = set()

    def replace(m: re.Match) -> str:
        key = m.group(1)
        used.add(key)
        return str(params.get(key, m.group(0)))

    url = re.sub(r'\{(\w+)\}', replace, template)
    remaining = {k: v for k, v in params.items() if k not in used}
    return url, remaining


class DeclarativeConnector(ActionConnector):
    """
    ActionConnector built from a TOML definition.
    Handles api_key auth with templated HTTP actions.
    """
    connector_type: ClassVar[str] = "action"

    def __init__(self, spec: dict) -> None:
        conn_spec = spec["connector"]
        self.name = conn_spec["name"]
        self.display_name = conn_spec["display_name"]
        self.description = conn_spec.get("description", "")
        self.auth_type = conn_spec.get("auth_type", "api_key")
        self._spec = spec

        # Parse credential_fields
        self.credential_fields = [
            CredentialField(
                name=f["name"],
                label=f.get("label", f["name"]),
                secret=f.get("secret", False),
                placeholder=f.get("placeholder", ""),
                field_type=f.get("field_type", "text"),
                options=f.get("options", []),
            )
            for f in spec.get("credential_fields", [])
        ]

        # Parse actions
        self.actions = [
            ConnectorAction(
                name=a["name"],
                description=a.get("description", ""),
                params=[
                    ConnectorParam(
                        name=p["name"],
                        description=p.get("description", ""),
                        required=p.get("required", True),
                        default=p.get("default"),
                    )
                    for p in a.get("params", [])
                ],
            )
            for a in spec.get("actions", [])
        ]

        # Action metadata (method, url, auth_header) — kept separate from the model
        self._action_meta: dict[str, dict] = {
            a["name"]: a for a in spec.get("actions", [])
        }

    async def invoke(self, action: str, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        meta = self._action_meta.get(action)
        if not meta:
            return {"success": False, "error": f"Unknown action '{action}'"}

        method = meta.get("method", "GET").upper()
        url_template = meta.get("url", "")
        auth_header_template = meta.get("auth_header", "Authorization: Bearer {api_key}")

        # Substitute path params into URL; remaining params go to query/body
        url, remaining = _build_url(url_template, params)

        # Build auth header
        auth_header_value = auth_header_template.format(**credentials)
        header_name, _, header_value = auth_header_value.partition(": ")
        headers = {header_name: header_value, "Accept": "application/json"}

        # Make request
        async with httpx.AsyncClient(timeout=15.0) as client:
            if method == "GET":
                resp = await client.get(url, params=remaining, headers=headers)
            elif method in ("POST", "PUT", "PATCH"):
                resp = await client.request(method, url, json=remaining or None, headers=headers)
            elif method == "DELETE":
                resp = await client.delete(url, headers=headers)
            else:
                return {"success": False, "error": f"Unsupported HTTP method: {method}"}

        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:300]
            return {"success": False, "error": f"HTTP {resp.status_code}", "detail": detail}

        try:
            return {"success": True, "data": resp.json()}
        except Exception:
            return {"success": True, "data": resp.text}


def load_connector_from_toml(path: Path) -> DeclarativeConnector:
    """Load a DeclarativeConnector from a TOML file."""
    with open(path, "rb") as f:
        spec = tomllib.load(f)
    return DeclarativeConnector(spec)


def load_connectors_from_dir(directory: Path) -> list[DeclarativeConnector]:
    """Load all .toml connectors from a directory."""
    connectors: list[DeclarativeConnector] = []
    for path in sorted(directory.glob("*.toml")):
        try:
            c = load_connector_from_toml(path)
            connectors.append(c)
            logger.debug("Loaded declarative connector: %s from %s", c.name, path.name)
        except Exception as e:
            logger.warning("Failed to load connector from %s: %s", path, e)
    return connectors
