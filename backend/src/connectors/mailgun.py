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
    description = "Send and receive emails via the Mailgun API. Supports inbox reading for inbound messages stored by a Mailgun Route."
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
        ),
        ConnectorAction(
            name="list_inbox",
            description="List inbound emails stored by Mailgun (requires a Store route to be configured). Messages are retained for 3 days.",
            params=[
                ConnectorParam(name="limit", description="Max number of messages to return (default 10)", required=False, default="10"),
            ],
        ),
        ConnectorAction(
            name="read_message",
            description="Read the full content of a stored inbound message.",
            params=[
                ConnectorParam(name="storage_url", description="The storage URL returned by list_inbox", required=True),
            ],
        ),
    ]

    async def invoke(self, action: str, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        if action == "list_inbox":
            return await self._list_inbox(params, credentials)
        if action == "read_message":
            return await self._read_message(params, credentials)
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

    async def _list_inbox(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        api_key = credentials.get("api_key", "").strip()
        domain = credentials.get("domain", "").strip()
        if not api_key:
            raise ValueError("api_key is required in connector credentials")
        if not domain:
            raise ValueError("domain is required in connector credentials")

        limit = int(params.get("limit") or 10)

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{MAILGUN_API_BASE}/{domain}/events",
                auth=("api", api_key),
                params={"event": "stored", "limit": limit},
            )

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("message", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            return {"success": False, "error": f"Mailgun error {resp.status_code}: {detail}"}

        items = resp.json().get("items", [])
        messages = []
        for item in items:
            headers = item.get("message", {}).get("headers", {})
            storage = item.get("storage", {})
            messages.append({
                "subject": headers.get("subject", "(no subject)"),
                "from": headers.get("from", ""),
                "to": headers.get("to", ""),
                "date": item.get("timestamp", ""),
                "storage_url": storage.get("url", ""),
                "storage_key": storage.get("key", ""),
            })

        return {"success": True, "messages": messages, "count": len(messages)}

    async def _read_message(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        api_key = credentials.get("api_key", "").strip()
        storage_url = params.get("storage_url", "").strip()
        if not api_key:
            raise ValueError("api_key is required in connector credentials")
        if not storage_url:
            raise ValueError("params.storage_url is required")

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(storage_url, auth=("api", api_key))

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("message", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            return {"success": False, "error": f"Mailgun error {resp.status_code}: {detail}"}

        msg = resp.json()
        return {
            "success": True,
            "subject": msg.get("subject", ""),
            "from": msg.get("from", ""),
            "to": msg.get("To", ""),
            "date": msg.get("Date", ""),
            "body_plain": msg.get("body-plain", ""),
            "body_html": msg.get("body-html", ""),
            "message_id": msg.get("Message-Id", ""),
        }
