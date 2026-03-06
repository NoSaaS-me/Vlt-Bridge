"""
Connector Webhook Receiver.

External services (Mailgun, Stripe, etc.) POST to:
  POST /api/webhooks/connector/{listener_id}

The backend verifies HMAC, parses payload, applies pattern filter,
then forwards a rendered prompt to the vlt daemon for Cronban execution.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request

from ...services.connector_service import ConnectorService
from ...services.database import DatabaseService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])

DAEMON_URL = "http://127.0.0.1:8765"


def _render_event_context(event) -> str:
    lines = [
        f"## Webhook Event: {event.connector} / {event.event_type}",
        f"**Received at**: {event.timestamp}",
        "",
    ]
    for key, value in event.fields.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


def _render_template(template: str, fields: dict[str, str]) -> str:
    result = template
    for key, value in fields.items():
        result = result.replace(f"{{{{{key}}}}}", value)
    return result


@router.post("/connector/{listener_id}")
async def receive_connector_webhook(listener_id: str, request: Request):
    """
    Receive a connector webhook. No API auth — validation is connector-specific (HMAC).
    """
    from vlt_connectors.registry import get_registry
    from vlt_connectors.webhook_models import PatternFilter, match_pattern

    body_bytes = await request.body()
    headers = dict(request.headers)

    # 1. Look up the listener from the daemon
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{DAEMON_URL}/api/cronban/webhooks/{listener_id}")
        except httpx.ConnectError:
            raise HTTPException(503, "vlt daemon is not running")
        if r.status_code == 404:
            raise HTTPException(404, "Webhook listener not found")
        if r.status_code != 200:
            raise HTTPException(502, f"Daemon error: {r.status_code}")
        listener = r.json()

    connector_name: Optional[str] = listener.get("connector_name")
    if not connector_name:
        raise HTTPException(400, "Listener is not bound to a connector")

    backend_user_id: Optional[str] = listener.get("backend_user_id")
    if not backend_user_id:
        raise HTTPException(400, "Listener has no backend user binding")

    # 2. Get connector + credentials
    registry = get_registry()
    connector = registry.get(connector_name)
    if not connector:
        raise HTTPException(400, f"Connector '{connector_name}' not found in registry")

    connector_service = ConnectorService(DatabaseService())
    credentials = connector_service.get_credentials(backend_user_id, connector_name)

    # 3. Verify webhook signature
    if not connector.verify_webhook(headers, body_bytes, credentials):
        raise HTTPException(401, "Webhook signature verification failed")

    # 4. Parse into ConnectorEvent
    event = connector.parse_webhook(headers, body_bytes, credentials)
    if not event:
        return {"received": True, "matched": False, "reason": "unparseable"}

    # 5. Apply pattern filter
    pattern_json: Optional[str] = listener.get("pattern_filter_json")
    if pattern_json:
        try:
            pattern = PatternFilter.model_validate_json(pattern_json)
            if not match_pattern(pattern, event.fields):
                return {"received": True, "matched": False, "reason": "filter_no_match"}
        except Exception as exc:
            logger.warning("Pattern filter parse error for listener %s: %s", listener_id, exc)

    # 6. Render prompt template with event variables
    append_message = _render_event_context(event)
    # Apply template substitution if skill/prompt_text uses {{vars}}
    if listener.get("prompt_text"):
        rendered_prompt = _render_template(listener["prompt_text"], event.fields)
        append_message = rendered_prompt + "\n\n" + append_message

    # 7. Forward to daemon
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{DAEMON_URL}/api/cronban/webhooks/{listener_id}/fire",
            json={"append_message": append_message},
        )
        if r.status_code >= 400:
            logger.error("Daemon fire failed: %s %s", r.status_code, r.text[:200])
            raise HTTPException(502, f"Daemon failed to fire: {r.status_code}")

    return {"received": True, "matched": True, "fired": True, "event_type": event.event_type}
