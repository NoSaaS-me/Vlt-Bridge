"""Artifact REST + WebSocket endpoints for the daemon.

Provides CRUD, state machine transitions, backend proxy, frontend serving,
storage, testing, screenshots, and import/export.
"""

import io
import json
import logging
import mimetypes
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, WebSocket
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from vlt.daemon import artifact_service as svc

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateArtifactRequest(BaseModel):
    name: str
    description: Optional[str] = None
    type: str = "ephemeral"
    project_id: str = "default"
    template: Optional[str] = None


class UpdateArtifactRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    manifest: Optional[dict] = None


class TransitionStateRequest(BaseModel):
    target_state: str
    actor: str = "user"


class BackendCallRequest(BaseModel):
    action: str
    params: dict = {}


class StorageSetRequest(BaseModel):
    value: object


class EmitEventRequest(BaseModel):
    event_type: str
    payload: dict = {}


class ConnectorCallRequest(BaseModel):
    connector: str
    action: str
    params: dict = {}
    instance_id: str = "default"


# ============================================================================
# CRUD Endpoints
# ============================================================================

@router.post("")
def create_artifact(req: CreateArtifactRequest):
    """Create a new artifact."""
    # Default user_id for local mode
    user_id = "local-dev"
    try:
        artifact = svc.create_artifact(
            user_id=user_id,
            project_id=req.project_id,
            name=req.name,
            description=req.description,
            artifact_type=req.type,
            template=req.template,
        )
        return artifact
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/templates")
def list_templates():
    """List available artifact templates."""
    return svc.list_templates()


@router.get("")
def list_artifacts(
    project_id: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
):
    """List all artifacts, optionally filtered."""
    return svc.list_artifacts(project_id=project_id, state=state)


@router.get("/{artifact_id}")
def get_artifact(artifact_id: str):
    """Get artifact details."""
    artifact = svc.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return artifact


@router.put("/{artifact_id}")
def update_artifact(artifact_id: str, req: UpdateArtifactRequest):
    """Update artifact metadata or manifest."""
    kwargs = {k: v for k, v in req.model_dump().items() if v is not None}
    result = svc.update_artifact(artifact_id, **kwargs)
    if not result:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return result


@router.delete("/{artifact_id}")
def delete_artifact(artifact_id: str):
    """Delete an artifact."""
    if not svc.delete_artifact(artifact_id):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return Response(status_code=204)


# ============================================================================
# State Machine
# ============================================================================

@router.post("/{artifact_id}/state")
def transition_state(artifact_id: str, req: TransitionStateRequest):
    """Transition artifact to a new state."""
    try:
        result = svc.transition_state(artifact_id, req.target_state, req.actor)
        if not result:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return result
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ============================================================================
# Backend Process Management
# ============================================================================

@router.post("/{artifact_id}/backend/start")
async def start_backend(artifact_id: str):
    """Start the artifact's backend process."""
    try:
        return await svc.start_backend(artifact_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{artifact_id}/backend/stop")
async def stop_backend(artifact_id: str):
    """Stop the artifact's backend process."""
    return await svc.stop_backend(artifact_id)


@router.post("/{artifact_id}/backend/call")
async def call_backend(artifact_id: str, req: BackendCallRequest):
    """Proxy a call to the artifact's backend."""
    try:
        return await svc.call_backend(artifact_id, req.action, req.params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Frontend Serving
# ============================================================================

VLT_BRIDGE_INJECT = """<script src="/vlt-bridge.js"></script>
<script>
(function(){
  var aid='%ARTIFACT_ID%';
  var ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/vlt/api/artifacts/ws/'+aid+'/hmr');
  ws.onmessage=function(e){
    var m=JSON.parse(e.data);
    if(m.type==='css_update'){
      document.querySelectorAll('link[rel="stylesheet"]').forEach(function(l){
        l.href=l.href.split('?')[0]+'?v='+Date.now();
      });
    }
    if(m.type==='will_reload'&&window.__vlt_save_state){
      var s=window.__vlt_save_state();
      ws.send(JSON.stringify({type:'state_saved',state:s}));
    }
    if(m.type==='reload'){location.reload();}
    if(m.type==='state_restore'&&window.__vlt_load_state&&m.state){
      window.__vlt_load_state(m.state);
    }
  };
})();
</script>
"""


@router.get("/{artifact_id}/frontend/{file_path:path}")
def serve_frontend(artifact_id: str, file_path: str):
    """Serve static files from the artifact's frontend directory."""
    # Auto-start watcher when frontend is first accessed
    try:
        from vlt.daemon.artifact_watcher import start_watcher, _watchers
        if artifact_id not in _watchers:
            artifact_for_watch = svc.get_artifact(artifact_id)
            if artifact_for_watch:
                import asyncio
                loop = asyncio.get_event_loop()
                start_watcher(artifact_id, artifact_for_watch["disk_path"], loop)
    except Exception:
        pass

    artifact = svc.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    full_path = Path(artifact["disk_path"]) / "frontend" / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Security: ensure path is within artifact frontend dir
    frontend_dir = Path(artifact["disk_path"]) / "frontend"
    try:
        full_path.resolve().relative_to(frontend_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal denied")

    content = full_path.read_bytes()
    mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

    # Inject VltBridge + HMR into index.html
    if file_path == "index.html" or file_path.endswith("/index.html"):
        text = content.decode("utf-8", errors="replace")
        inject = VLT_BRIDGE_INJECT.replace("%ARTIFACT_ID%", artifact_id)
        # Skip bridge script if already included by the artifact
        if "vlt-bridge.js" in text:
            # Only inject HMR script, strip the duplicate bridge include
            inject = inject.split("\n", 1)[1] if "\n" in inject else ""
        if "</head>" in text:
            text = text.replace("</head>", inject + "</head>")
        else:
            text = inject + text
        content = text.encode("utf-8")
        mime_type = "text/html"

    return Response(content=content, media_type=mime_type)


# ============================================================================
# Storage (artifact-scoped key-value)
# ============================================================================

@router.get("/{artifact_id}/storage")
def storage_list(artifact_id: str):
    """List all storage keys."""
    keys = svc.storage_list(artifact_id)
    return {"keys": keys}


@router.get("/{artifact_id}/storage/{key}")
def storage_get(artifact_id: str, key: str):
    """Get a storage value."""
    value = svc.storage_get(artifact_id, key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Key not found: {key}")
    return {"key": key, "value": value}


@router.put("/{artifact_id}/storage/{key}")
def storage_set(artifact_id: str, key: str, req: StorageSetRequest):
    """Set a storage value."""
    if not svc.storage_set(artifact_id, key, req.value):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return {"key": key, "stored": True}


# ============================================================================
# Testing
# ============================================================================

@router.post("/{artifact_id}/test")
async def run_tests(artifact_id: str):
    """Run the artifact's test command."""
    try:
        return await svc.run_tests(artifact_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Screenshots & Vision Review
# ============================================================================

@router.post("/{artifact_id}/screenshot")
async def capture_screenshot(artifact_id: str):
    """Capture a screenshot of the artifact's frontend."""
    try:
        return await svc.capture_screenshot(artifact_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{artifact_id}/screenshot/{filename}")
def get_screenshot(artifact_id: str, filename: str):
    """Retrieve a previously captured screenshot."""
    artifact = svc.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    screenshot_path = Path(artifact["disk_path"]) / ".vlt" / "screenshots" / filename
    if not screenshot_path.exists():
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return Response(content=screenshot_path.read_bytes(), media_type="image/png")


@router.post("/{artifact_id}/review")
async def review_artifact(artifact_id: str):
    """Run vision model review on artifact screenshot."""
    try:
        return await svc.review_artifact(artifact_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Import / Export
# ============================================================================

@router.get("/{artifact_id}/export")
def export_artifact(artifact_id: str):
    """Export artifact as a zip file."""
    artifact = svc.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    disk_path = Path(artifact["disk_path"])
    if not disk_path.exists():
        raise HTTPException(status_code=404, detail="Artifact directory not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in disk_path.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(disk_path)
                # Exclude runtime state, git, and deps
                parts = rel.parts
                if any(p in (".git", ".vlt", ".deps", "__pycache__") for p in parts):
                    continue
                zf.write(file_path, str(rel))

    buf.seek(0)
    filename = f"{artifact['name'].replace(' ', '_')}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/import")
async def import_artifact(
    file: UploadFile = File(...),
    project_id: str = Query("default"),
):
    """Import artifact from a zip file."""
    content = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")

    # Validate manifest exists
    if "manifest.json" not in zf.namelist():
        raise HTTPException(status_code=400, detail="Zip must contain manifest.json")

    try:
        manifest = json.loads(zf.read("manifest.json"))
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid manifest.json: {e}")

    # Create artifact
    name = file.filename.replace(".zip", "").replace("_", " ") if file.filename else "Imported Artifact"
    user_id = "local-dev"
    artifact = svc.create_artifact(
        user_id=user_id,
        project_id=project_id,
        name=name,
    )

    # Extract files over the default scaffold
    disk_path = Path(artifact["disk_path"])
    for member in zf.namelist():
        if member.endswith("/"):
            continue
        target = disk_path / member
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(member))

    # Update manifest in DB
    svc.update_artifact(artifact["id"], manifest=manifest)

    return svc.get_artifact(artifact["id"])


# ============================================================================
# WebSocket: HMR (Hot Module Replacement)
# ============================================================================

# Connected HMR clients: artifact_id -> set of WebSocket
_hmr_clients: dict[str, set] = {}


@router.websocket("/ws/{artifact_id}/hmr")  # Note: registered under /api/artifacts prefix
async def artifact_hmr_ws(websocket: WebSocket, artifact_id: str):
    """HMR WebSocket for artifact iframe hot reload."""
    await websocket.accept()
    _hmr_clients.setdefault(artifact_id, set()).add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle state_saved messages from iframe
            try:
                msg = json.loads(data)
                if msg.get("type") == "state_saved":
                    # Store temporarily for restore after reload
                    artifact = svc.get_artifact(artifact_id)
                    if artifact:
                        state_path = Path(artifact["disk_path"]) / ".vlt" / "hmr_state.json"
                        state_path.write_text(json.dumps(msg.get("state")))
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    finally:
        clients = _hmr_clients.get(artifact_id, set())
        clients.discard(websocket)
        if not clients:
            _hmr_clients.pop(artifact_id, None)


async def broadcast_hmr(artifact_id: str, message: dict):
    """Send HMR message to all connected iframe clients."""
    clients = _hmr_clients.get(artifact_id, set())
    dead = set()
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    clients -= dead


# ============================================================================
# WebSocket: State Changes
# ============================================================================

_state_clients: dict[str, set] = {}


@router.websocket("/ws/{artifact_id}/state")
async def artifact_state_ws(websocket: WebSocket, artifact_id: str):
    """State change WebSocket for artifact sidebar."""
    await websocket.accept()
    _state_clients.setdefault(artifact_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except Exception:
        pass
    finally:
        clients = _state_clients.get(artifact_id, set())
        clients.discard(websocket)
        if not clients:
            _state_clients.pop(artifact_id, None)


async def broadcast_state(artifact_id: str, message: dict):
    """Send state change to all connected sidebar clients."""
    clients = _state_clients.get(artifact_id, set())
    dead = set()
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    clients -= dead


# ============================================================================
# WebSocket: Backend Logs
# ============================================================================

_log_clients: dict[str, set] = {}


@router.websocket("/ws/{artifact_id}/logs")
async def artifact_logs_ws(websocket: WebSocket, artifact_id: str):
    """Backend log streaming WebSocket."""
    await websocket.accept()
    _log_clients.setdefault(artifact_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        clients = _log_clients.get(artifact_id, set())
        clients.discard(websocket)
        if not clients:
            _log_clients.pop(artifact_id, None)


# ============================================================================
# Events: Artifact-to-Artifact IPC
# ============================================================================

_event_clients: dict[str, set] = {}


@router.post("/{artifact_id}/events/emit")
async def emit_event(artifact_id: str, req: EmitEventRequest):
    """Emit an event from source artifact to all subscribers."""
    from vlt.daemon.artifact_event_bus import get_event_bus
    bus = get_event_bus()
    recipients = await bus.emit(artifact_id, req.event_type, req.payload)
    log.info(f"Artifact {artifact_id} emitted '{req.event_type}' → {recipients}")
    return {"delivered_to": recipients}


@router.websocket("/ws/{artifact_id}/events")
async def artifact_events_ws(websocket: WebSocket, artifact_id: str):
    """Events WebSocket for pushing IPC events to artifact frontends."""
    await websocket.accept()
    _event_clients.setdefault(artifact_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except Exception:
        pass
    finally:
        clients = _event_clients.get(artifact_id, set())
        clients.discard(websocket)
        if not clients:
            _event_clients.pop(artifact_id, None)


async def broadcast_events(artifact_id: str, message: dict):
    """Push an IPC event to all connected event WebSocket clients for an artifact."""
    clients = _event_clients.get(artifact_id, set())
    dead = set()
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    clients -= dead


# ============================================================================
# Connector Proxy
# ============================================================================

@router.post("/{artifact_id}/connectors/call")
async def proxy_connector_call(artifact_id: str, req: ConnectorCallRequest):
    """Proxy a connector call through the backend after validating manifest.connectors.

    1. Fetch artifact manifest.
    2. Verify the connector is declared in manifest.connectors (if the list is present).
    3. Forward to backend POST /api/connectors/{connector}/invoke with instance_id.
    """
    import httpx
    from vlt.config import get_settings

    artifact = svc.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Validate connector is declared in manifest (if connectors list is defined)
    manifest = artifact.get("manifest") or {}
    declared_connectors = manifest.get("connectors")
    if declared_connectors is not None:
        # Support both list-of-str and list-of-dict formats
        allowed = any(
            (entry == req.connector if isinstance(entry, str)
             else entry.get("connector") == req.connector)
            for entry in declared_connectors
        )
        if not allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Connector '{req.connector}' is not declared in this artifact's manifest.connectors",
            )

    # Try local vlt-connectors registry first (AI model connectors live here)
    try:
        from vlt_connectors.registry import get_registry
        registry = get_registry()
        connector_obj = registry.get_action(req.connector)
        if connector_obj:
            credentials = _resolve_connector_credentials(req.connector, req.instance_id)
            log.info(f"Invoking connector '{req.connector}' action '{req.action}' locally (keys: {list(credentials.keys())})")
            result = await connector_obj.invoke(req.action, req.params, credentials)
            return result
        else:
            log.debug(f"Connector '{req.connector}' not found in local registry, falling through to backend")
    except ImportError as e:
        log.debug(f"vlt_connectors not available: {e}")
    except Exception as e:
        log.error(f"Local connector '{req.connector}' error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Connector '{req.connector}' error: {e}")

    # Fall back to backend proxy for connectors registered there
    settings = get_settings()
    vault_url = getattr(settings, "vault_url", None) or getattr(settings, "sync_url", None)
    sync_token = getattr(settings, "sync_token", None)

    if not vault_url or not sync_token:
        raise HTTPException(
            status_code=503,
            detail="Backend not configured. Set VLT_VAULT_URL and VLT_SYNC_TOKEN.",
        )

    headers = {"Authorization": f"Bearer {sync_token}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{vault_url}/api/connectors/{req.connector}/invoke",
                json={"action": req.action, "params": req.params, "instance_id": req.instance_id},
                headers=headers,
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Backend request failed: {exc}")

    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Connector '{req.connector}' not found")
    if resp.status_code == 403:
        raise HTTPException(status_code=403, detail=resp.json().get("detail", "Forbidden"))
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:400])

    return resp.json()


def _resolve_connector_credentials(connector_name: str, instance_id: str = "default") -> dict:
    """Resolve API credentials for a connector from user settings or connector_configs."""
    import sqlite3

    credentials = {}

    # Try user_settings table in backend DB for known AI provider keys
    key_mapping = {
        "openrouter": "openrouter_api_key",
        "zai": "glm_api_key",
        "zai_vision": "glm_api_key",
        "elevenlabs": "elevenlabs_api_key",
    }
    settings_key = key_mapping.get(connector_name)
    if settings_key:
        try:
            # Backend DB is at data/index.db
            from pathlib import Path
            db_path = Path("/mnt/sda1/Projects/00Tooling/Vlt-Bridge/data/index.db")
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(f"SELECT {settings_key} FROM user_settings LIMIT 1")
                row = cursor.fetchone()
                if row and row[0]:
                    credentials["api_key"] = row[0]
                conn.close()
        except Exception as e:
            log.debug(f"Could not read {settings_key} from user_settings: {e}")

    # Also check connector_configs table in daemon DB
    if not credentials.get("api_key"):
        try:
            from vlt.config import settings as vlt_settings
            db_path = vlt_settings.get_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT config_key, config_value FROM connector_configs "
                "WHERE connector_name = ? AND instance_id = ?",
                (connector_name, instance_id),
            )
            for key, value in cursor.fetchall():
                if not key.startswith("__"):
                    credentials[key] = value
            conn.close()
        except Exception:
            pass

    return credentials
