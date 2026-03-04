"""Mailgun email connector."""
from __future__ import annotations

from typing import Any
import httpx

from .base import BaseConnector
from ..models.connectors import ConnectorAction, ConnectorParam, CredentialField

MAILGUN_API_BASE = "https://api.mailgun.net/v3"


class MailgunConnector(BaseConnector):
    name = "mailgun"
    display_name = "Mailgun"
    description = "Send transactional emails via the Mailgun API."
    credential_fields = [
        CredentialField(name="api_key", label="API Key", secret=True, placeholder="key-..."),
        CredentialField(name="domain", label="Mailgun Domain", secret=False, placeholder="mg.yourdomain.com"),
        CredentialField(name="from_address", label="Default From Address", secret=False, placeholder="noreply@mg.yourdomain.com"),
    ]
    actions = [
        ConnectorAction(
            name="send_email",
            description="Send an email via Mailgun.",
            params=[
                ConnectorParam(name="to", description="Recipient email address", required=True),
                ConnectorParam(name="subject", description="Email subject line", required=True),
                ConnectorParam(name="body", description="Plain-text email body", required=True),
                ConnectorParam(name="from_address", description="Override sender address (optional)", required=False),
                ConnectorParam(name="reply_to", description="Reply-to address (optional)", required=False),
            ],
        )
    ]

    async def invoke(self, action: str, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        if action != "send_email":
            raise ValueError(f"Unknown action: {action}")

        api_key = credentials.get("api_key", "").strip()
        domain = credentials.get("domain", "").strip()
        if not api_key:
            raise ValueError("api_key is required in connector credentials")
        if not domain:
            raise ValueError("domain is required in connector credentials")

        to = params.get("to", "").strip()
        subject = params.get("subject", "").strip()
        body = params.get("body", "").strip()
        if not to:
            raise ValueError("params.to is required")
        if not subject:
            raise ValueError("params.subject is required")
        if not body:
            raise ValueError("params.body is required")

        from_address = (
            params.get("from_address")
            or credentials.get("from_address")
            or f"noreply@{domain}"
        )

        data: dict[str, str] = {
            "from": from_address,
            "to": to,
            "subject": subject,
            "text": body,
        }
        if params.get("reply_to"):
            data["h:Reply-To"] = params["reply_to"]

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{MAILGUN_API_BASE}/{domain}/messages",
                auth=("api", api_key),
                data=data,
            )

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("message", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            return {"success": False, "error": f"Mailgun error {resp.status_code}: {detail}"}

        body_json = resp.json()
        return {
            "success": True,
            "message_id": body_json.get("id", ""),
            "message": body_json.get("message", "Queued"),
        }
