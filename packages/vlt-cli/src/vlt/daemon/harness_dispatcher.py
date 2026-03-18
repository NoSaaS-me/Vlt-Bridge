"""Bidirectional harness dispatcher.

When the artifact backend process writes a JSON line that contains a ``_type``
field, the stdout reader hands it here instead of treating it as a normal
request response.  The dispatcher routes it to the appropriate service and,
where relevant, sends a response back on the process's stdin.

Supported message types
-----------------------
``connector_call``
    The backend wants to invoke a connector action that the daemon proxies to
    the main backend service.  Requires ``_id``, ``connector``, ``action``, and
    ``params`` fields.  The manifest must declare the connector+action pair.
    The cost tracker is checked before every call.

``storage``
    Read/write the artifact's .vlt/storage/ key-value store.  Requires ``_id``,
    ``action`` ("get" | "set" | "list" | "delete"), and ``key`` (except for
    "list").  "set" also requires ``value``.

``event``
    Fire-and-forget: emit an IPC event on the artifact event bus.
    Requires ``event_type`` and ``payload``.  No response is sent.

``notification``
    Fire-and-forget: log a notification with the given ``severity`` and
    ``message``.  No response is sent.  (Full ANS integration is future work.)
"""

import asyncio
import json
import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def dispatch_backend_message(
    artifact_id: str,
    message: dict,
    send_fn=None,
) -> None:
    """Dispatch a backend-initiated message and optionally send a response.

    Args:
        artifact_id: ID of the artifact whose backend sent the message.
        message: Parsed JSON dict from the backend's stdout.
        send_fn: Async callable(artifact_id, data_dict) to send response back
                 to the backend's stdin. If None, uses _send_to_backend from
                 artifact_service.
    """
    # Resolve send function
    if send_fn is None:
        from vlt.daemon.artifact_service import _send_to_backend
        send_fn = _send_to_backend

    # Get manifest from artifact process info
    from vlt.daemon.artifact_service import _artifact_processes, get_artifact
    proc_info = _artifact_processes.get(artifact_id, {})
    manifest = proc_info.get("manifest")
    if not manifest:
        artifact = get_artifact(artifact_id)
        manifest = artifact.get("manifest", {}) if artifact else {}

    msg_type = message.get("_type", "")

    if msg_type == "connector_call":
        await _handle_connector_call(artifact_id, message, send_fn, manifest)
    elif msg_type == "storage":
        await _handle_storage(artifact_id, message, send_fn)
    elif msg_type == "event":
        await _handle_event(artifact_id, message)
    elif msg_type == "notification":
        _handle_notification(artifact_id, message)
    else:
        log.warning(f"Artifact {artifact_id}: unknown _type '{msg_type}' — ignoring")


# ---------------------------------------------------------------------------
# Helper: write response back to the backend process's stdin
# ---------------------------------------------------------------------------

async def _send_response(send_fn, artifact_id: str, payload: dict) -> None:
    """Send a response back to the backend process via the send function."""
    try:
        await send_fn(artifact_id, payload)
    except Exception as e:
        log.error(f"Failed to send response to backend {artifact_id}: {e}")


# ---------------------------------------------------------------------------
# connector_call
# ---------------------------------------------------------------------------

async def _handle_connector_call(
    artifact_id: str,
    message: dict,
    send_fn,
    manifest: dict,
) -> None:
    """Validate, cost-check, and proxy a connector call.

    Expected message fields:
        _id        Correlation ID echoed in the response.
        connector  Connector name declared in manifest.connectors.
        action     Action name declared for that connector.
        params     Arbitrary dict forwarded to the connector.
    """
    msg_id = message.get("_id", "")
    connector = message.get("connector", "")
    action = message.get("action", "")
    params = message.get("params", {})

    # --- Manifest validation ------------------------------------------------
    declared_connectors: list[dict] | list[str] | None = manifest.get("connectors")

    if declared_connectors is not None:
        # Support both list-of-str (legacy) and list-of-dict formats:
        #   ["openrouter"]  OR  [{"connector": "openrouter", "actions": [...]}]
        allowed_entry = None
        for entry in declared_connectors:
            if isinstance(entry, str):
                if entry == connector:
                    allowed_entry = entry
                    break
            elif isinstance(entry, dict):
                if entry.get("connector") == connector:
                    allowed_entry = entry
                    break

        if allowed_entry is None:
            err = (
                f"Connector '{connector}' is not declared in this artifact's "
                "manifest.connectors."
            )
            log.warning(f"Artifact {artifact_id}: {err}")
            await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "connector_response", "error": err})
            return

        # If the entry specifies allowed actions, validate the requested action.
        if isinstance(allowed_entry, dict):
            allowed_actions = allowed_entry.get("actions")
            if allowed_actions is not None and action not in allowed_actions:
                err = (
                    f"Action '{action}' is not declared for connector '{connector}' "
                    "in this artifact's manifest.connectors."
                )
                log.warning(f"Artifact {artifact_id}: {err}")
                await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "connector_response", "error": err})
                return

    # --- Cost limit check ---------------------------------------------------
    from vlt.daemon.cost_tracker import check_cost_limit, record_cost

    allowed, reason = check_cost_limit(artifact_id, connector, estimated_cost=0.0)
    if not allowed:
        log.warning(f"Artifact {artifact_id}: cost limit exceeded for '{connector}': {reason}")
        await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "connector_response", "error": reason})
        return

    # --- Proxy the call through the daemon endpoint -------------------------
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"http://127.0.0.1:8765/api/artifacts/{artifact_id}/connectors/call",
                json={"connector": connector, "action": action, "params": params},
            )

        if resp.status_code >= 400:
            err = f"Connector call failed (HTTP {resp.status_code}): {resp.text[:400]}"
            log.error(f"Artifact {artifact_id}: {err}")
            await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "connector_response", "error": err})
            return

        result = resp.json()

        # Record cost if the connector reports it
        cost_usd = 0.0
        if isinstance(result, dict):
            cost_usd = float(result.get("cost_usd", 0.0) or 0.0)
        record_cost(artifact_id, connector, cost_usd)

        log.debug(
            f"Artifact {artifact_id}: connector_call {connector}.{action} → "
            f"HTTP {resp.status_code}, cost=${cost_usd:.6f}"
        )
        await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "connector_response", "result": result})

    except httpx.RequestError as exc:
        err = f"Connector proxy request failed: {exc}"
        log.error(f"Artifact {artifact_id}: {err}")
        await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "connector_response", "error": err})
    except Exception as exc:
        err = f"Unexpected error during connector call: {exc}"
        log.exception(f"Artifact {artifact_id}: {err}")
        await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "connector_response", "error": err})


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

async def _handle_storage(
    artifact_id: str,
    message: dict,
    send_fn,
) -> None:
    """Handle a storage read/write operation.

    Expected message fields:
        _id     Correlation ID.
        action  One of: "get", "set", "list", "delete".
        key     Storage key (required for get/set/delete).
        value   Value to store (required for set).

    Responses:
        { "_id": ..., "_type": "storage_response", "result": <value> }
        { "_id": ..., "_type": "storage_response", "error": "..." }
    """
    msg_id = message.get("_id", "")
    action = message.get("action", "")
    key = message.get("key", "")

    try:
        from vlt.daemon.artifact_service import get_artifact

        artifact = get_artifact(artifact_id)
        if not artifact:
            await _send_response(send_fn, artifact_id, {
                "_id": msg_id, "_type": "storage_response",
                "error": f"Artifact {artifact_id} not found",
            })
            return

        disk_path = Path(artifact["disk_path"]) / ".vlt" / "storage"
        disk_path.mkdir(parents=True, exist_ok=True)

        if action == "get":
            file_path = disk_path / f"{key}.json"
            if not file_path.exists():
                result = None
            else:
                result = json.loads(file_path.read_text())
            await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "storage_response", "result": result})

        elif action == "set":
            value = message.get("value")
            file_path = disk_path / f"{key}.json"
            file_path.write_text(json.dumps(value))
            log.debug(f"Artifact {artifact_id}: storage set '{key}'")
            await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "storage_response", "result": "ok"})

        elif action == "list":
            keys = [p.stem for p in disk_path.glob("*.json")]
            await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "storage_response", "result": keys})

        elif action == "delete":
            file_path = disk_path / f"{key}.json"
            existed = file_path.exists()
            if existed:
                file_path.unlink()
            log.debug(f"Artifact {artifact_id}: storage delete '{key}' (existed={existed})")
            await _send_response(send_fn, artifact_id, {
                "_id": msg_id, "_type": "storage_response",
                "result": {"deleted": existed},
            })

        else:
            err = f"Unknown storage action '{action}'. Valid: get, set, list, delete."
            log.warning(f"Artifact {artifact_id}: {err}")
            await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "storage_response", "error": err})

    except Exception as exc:
        err = f"Storage operation failed: {exc}"
        log.exception(f"Artifact {artifact_id}: {err}")
        await _send_response(send_fn, artifact_id, {"_id": msg_id, "_type": "storage_response", "error": err})


# ---------------------------------------------------------------------------
# event
# ---------------------------------------------------------------------------

async def _handle_event(artifact_id: str, message: dict) -> None:
    """Emit an IPC event on behalf of the backend process (fire-and-forget).

    Expected message fields:
        event_type  Event identifier string.
        payload     Arbitrary dict forwarded to subscribers.
    """
    event_type = message.get("event_type", "")
    payload = message.get("payload", {})

    if not event_type:
        log.warning(f"Artifact {artifact_id}: 'event' message missing event_type — ignoring")
        return

    try:
        from vlt.daemon.artifact_event_bus import get_event_bus
        bus = get_event_bus()
        recipients = await bus.emit(artifact_id, event_type, payload)
        log.debug(
            f"Artifact {artifact_id}: emitted '{event_type}' → {len(recipients)} recipient(s)"
        )
    except Exception as exc:
        log.error(f"Artifact {artifact_id}: event emission failed: {exc}")


# ---------------------------------------------------------------------------
# notification
# ---------------------------------------------------------------------------

def _handle_notification(artifact_id: str, message: dict) -> None:
    """Log a notification from the backend process (fire-and-forget).

    Expected message fields:
        severity  "info" | "warning" | "error"
        message   Human-readable notification text.

    Full ANS integration is future work.
    """
    severity = message.get("severity", "info").lower()
    text = message.get("message", "")

    if severity == "error":
        log.error(f"[Artifact {artifact_id} notification] {text}")
    elif severity == "warning":
        log.warning(f"[Artifact {artifact_id} notification] {text}")
    else:
        log.info(f"[Artifact {artifact_id} notification] {text}")
