"""Artifact lifecycle management service.

Handles creation, state transitions, backend process management,
and disk operations for artifact sandbox plugins.
"""

import asyncio
import json
import logging
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

from sqlalchemy.orm import Session

from vlt.core.models import Artifact, ArtifactState, ARTIFACT_STATE_TRANSITIONS
from vlt.db import SessionLocal

log = logging.getLogger(__name__)

# Default manifest for new artifacts
DEFAULT_MANIFEST = {
    "frontend": {"entry": "frontend/index.html", "deps": []},
    "backend": None,
    "connectors": [],
    "mcp_tools": [],
    "events": {"emits": [], "subscribes": []},
    "quotas": {"max_cpu_seconds": 60, "max_memory_mb": 512, "max_storage_mb": 50},
    "tests": None,
}

# Artifact backend processes:
#   artifact_id -> {proc, cwd, restart_count, reader_task, pending_responses, manifest}
#
# pending_responses: req_id -> asyncio.Future
#   call_backend() registers a Future here before writing to stdin.
#   _backend_stdout_reader() resolves it when the matching response arrives.
_artifact_processes: dict[str, dict] = {}


def _get_vault_base() -> Path:
    """Get the vault base directory (where user vaults live)."""
    # Try backend's VAULT_BASE_PATH first, fall back to default
    vault_base = Path("/mnt/sda1/Projects/00Tooling/Vlt-Bridge/data/vaults")
    if not vault_base.exists():
        from vlt.config import settings
        db_path = Path(settings.get_db_path())
        vault_base = db_path.parent / "vaults"
    vault_base.mkdir(parents=True, exist_ok=True)
    return vault_base


def _get_artifact_dir(user_id: str, artifact_id: str, project_id: str = "default", name: str = "") -> Path:
    """Get the disk path for an artifact inside the user's vault.

    Structure: data/vaults/{user_id}/{project_id}/Artifacts/{name}_{id}/
    This makes artifacts visible in the main vault file browser.
    """
    # Sanitize name for filesystem
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in (name or "")).strip().replace(" ", "-")
    folder_name = f"{safe_name}_{artifact_id}" if safe_name else artifact_id

    base = _get_vault_base() / user_id / project_id / "Artifacts" / folder_name
    return base


def _get_templates_dir() -> Path:
    """Get the directory containing artifact templates."""
    return Path(__file__).parent / "artifact_templates"


def list_templates() -> list[dict]:
    """List available artifact templates."""
    templates_dir = _get_templates_dir()
    if not templates_dir.exists():
        return []
    result = []
    for d in sorted(templates_dir.iterdir()):
        if d.is_dir() and (d / "manifest.json").exists():
            manifest = json.loads((d / "manifest.json").read_text())
            result.append({
                "name": d.name,
                "description": manifest.get("description", d.name.replace("_", " ").title()),
                "has_backend": manifest.get("backend") is not None,
                "connectors": [c.get("connector", c) if isinstance(c, dict) else c
                               for c in manifest.get("connectors", [])],
            })
    return result


def create_artifact(
    user_id: str,
    project_id: str,
    name: str,
    description: str | None = None,
    artifact_type: str = "ephemeral",
    template: str | None = None,
) -> dict:
    """Create a new artifact with directory structure and git init."""
    artifact_id = str(uuid.uuid4())[:8]
    disk_path = _get_artifact_dir(user_id, artifact_id, project_id=project_id, name=name)

    # Check for template
    template_dir = None
    if template:
        template_dir = _get_templates_dir() / template
        if not template_dir.exists() or not (template_dir / "manifest.json").exists():
            raise ValueError(f"Template '{template}' not found")

    if template_dir:
        # Copy entire template to artifact dir
        shutil.copytree(str(template_dir), str(disk_path))
        # Ensure .vlt directories exist
        (disk_path / ".vlt" / "storage" / "content").mkdir(parents=True, exist_ok=True)
        (disk_path / ".vlt" / "screenshots").mkdir(parents=True, exist_ok=True)
        (disk_path / ".vlt" / "costs").mkdir(parents=True, exist_ok=True)
        # Read manifest from template
        manifest = json.loads((disk_path / "manifest.json").read_text())
    else:
        # Create default directory structure
        (disk_path / "frontend").mkdir(parents=True, exist_ok=True)
        (disk_path / "backend").mkdir(parents=True, exist_ok=True)
        (disk_path / "tests").mkdir(parents=True, exist_ok=True)
        (disk_path / ".vlt" / "storage").mkdir(parents=True, exist_ok=True)
        (disk_path / ".vlt" / "screenshots").mkdir(parents=True, exist_ok=True)

        # Write default index.html
        (disk_path / "frontend" / "index.html").write_text(
            "<!DOCTYPE html>\n<html>\n<head>\n  <meta charset=\"utf-8\">\n"
            "  <title>{name}</title>\n  <link rel=\"stylesheet\" href=\"style.css\">\n"
            "</head>\n<body>\n  <h1>{name}</h1>\n  <p>Edit this file to get started.</p>\n"
            "  <script src=\"app.js\"></script>\n</body>\n</html>\n".format(name=name)
        )
        (disk_path / "frontend" / "style.css").write_text(
            "body { font-family: system-ui, sans-serif; padding: 1rem; }\n"
        )
        (disk_path / "frontend" / "app.js").write_text(
            "// VltBridge is auto-injected. Use VltBridge.storage, VltBridge.notes, etc.\n"
            "console.log('Artifact loaded:', document.title);\n"
        )
        manifest = {**DEFAULT_MANIFEST}

    # Write manifest (potentially overwritten with template values)
    (disk_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Write README.md so the vault file browser picks up this directory
    readme = f"# {name}\n\n{description or 'Artifact'}\n\n"
    readme += f"- **Type**: {artifact_type}\n"
    readme += f"- **ID**: {artifact_id}\n"
    readme += f"- **Created**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
    if manifest.get("pipeline"):
        stages = manifest["pipeline"].get("stages", [])
        readme += f"- **Pipeline**: {len(stages)} stages\n"
        for s in stages:
            readme += f"  - {s.get('type', '?')} ({s.get('connector', '?')}/{s.get('model', '?')})\n"
    (disk_path / "README.md").write_text(readme)

    # Git init
    try:
        subprocess.run(
            ["git", "init"], cwd=str(disk_path),
            capture_output=True, timeout=5
        )
        subprocess.run(
            ["git", "add", "."], cwd=str(disk_path),
            capture_output=True, timeout=5
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial artifact creation"],
            cwd=str(disk_path), capture_output=True, timeout=5,
            env={**__import__("os").environ, "GIT_AUTHOR_NAME": "vlt", "GIT_COMMITTER_NAME": "vlt",
                 "GIT_AUTHOR_EMAIL": "vlt@local", "GIT_COMMITTER_EMAIL": "vlt@local"}
        )
    except Exception as e:
        log.warning(f"Git init failed for artifact {artifact_id}: {e}")

    # Create DB record
    now = datetime.now(timezone.utc).isoformat()
    state_history = [{"state": "draft", "at": now, "by": "user"}]

    db = SessionLocal()
    try:
        artifact = Artifact(
            id=artifact_id,
            user_id=user_id,
            project_id=project_id,
            name=name,
            description=description,
            type=artifact_type,
            state=ArtifactState.DRAFT,
            state_history_json=json.dumps(state_history),
            manifest_json=json.dumps(manifest),
            disk_path=str(disk_path),
            created_at=now,
            updated_at=now,
        )
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        return _artifact_to_dict(artifact)
    finally:
        db.close()


def get_artifact(artifact_id: str) -> dict | None:
    """Get a single artifact by ID."""
    db = SessionLocal()
    try:
        artifact = db.get(Artifact, artifact_id)
        return _artifact_to_dict(artifact) if artifact else None
    finally:
        db.close()


def list_artifacts(
    user_id: str | None = None,
    project_id: str | None = None,
    state: str | None = None,
) -> list[dict]:
    """List artifacts with optional filters."""
    db = SessionLocal()
    try:
        q = db.query(Artifact)
        if user_id:
            q = q.filter(Artifact.user_id == user_id)
        if project_id:
            q = q.filter(Artifact.project_id == project_id)
        if state:
            q = q.filter(Artifact.state == ArtifactState(state))
        return [_artifact_to_dict(a) for a in q.order_by(Artifact.updated_at.desc()).all()]
    finally:
        db.close()


def update_artifact(artifact_id: str, **kwargs) -> dict | None:
    """Update artifact metadata or manifest."""
    db = SessionLocal()
    try:
        artifact = db.get(Artifact, artifact_id)
        if not artifact:
            return None

        for key, value in kwargs.items():
            if key == "manifest":
                artifact.manifest_json = json.dumps(value)
                # Also write to disk
                manifest_path = Path(artifact.disk_path) / "manifest.json"
                manifest_path.write_text(json.dumps(value, indent=2))
            elif hasattr(artifact, key):
                setattr(artifact, key, value)

        artifact.updated_at = datetime.now(timezone.utc).isoformat()
        db.commit()
        db.refresh(artifact)
        return _artifact_to_dict(artifact)
    finally:
        db.close()


def delete_artifact(artifact_id: str) -> bool:
    """Delete artifact: stop backend, remove disk dir, delete DB record."""
    db = SessionLocal()
    try:
        artifact = db.get(Artifact, artifact_id)
        if not artifact:
            return False

        # Stop backend if running
        if artifact_id in _artifact_processes:
            asyncio.ensure_future(_stop_backend_process(artifact_id))

        # Remove disk directory
        disk_path = Path(artifact.disk_path)
        if disk_path.exists():
            shutil.rmtree(disk_path)

        db.delete(artifact)
        db.commit()
        return True
    finally:
        db.close()


def transition_state(
    artifact_id: str,
    target_state: str,
    actor: str = "user",
) -> dict | None:
    """Transition artifact to a new state, validating the state machine graph."""
    db = SessionLocal()
    try:
        artifact = db.get(Artifact, artifact_id)
        if not artifact:
            return None

        current = artifact.state
        target = ArtifactState(target_state)

        # Validate transition
        valid_targets = ARTIFACT_STATE_TRANSITIONS.get(current, [])
        if target not in valid_targets:
            raise ValueError(
                f"Invalid state transition: {current.value} → {target.value}. "
                f"Valid targets: {[s.value for s in valid_targets]}"
            )

        now = datetime.now(timezone.utc).isoformat()
        history = json.loads(artifact.state_history_json)
        history.append({"state": target.value, "at": now, "by": actor})

        artifact.state = target
        artifact.state_history_json = json.dumps(history)
        artifact.updated_at = now
        artifact.version += 1
        db.commit()

        # Git commit on state transition
        disk_path = Path(artifact.disk_path)
        try:
            subprocess.run(
                ["git", "add", "-A"], cwd=str(disk_path),
                capture_output=True, timeout=5
            )
            subprocess.run(
                ["git", "commit", "-m", f"state: {current.value} → {target.value}", "--allow-empty"],
                cwd=str(disk_path), capture_output=True, timeout=5,
                env={**__import__("os").environ, "GIT_AUTHOR_NAME": "vlt", "GIT_COMMITTER_NAME": "vlt",
                     "GIT_AUTHOR_EMAIL": "vlt@local", "GIT_COMMITTER_EMAIL": "vlt@local"}
            )
        except Exception as e:
            log.warning(f"Git commit failed for artifact {artifact_id}: {e}")

        db.refresh(artifact)
        return _artifact_to_dict(artifact)
    finally:
        db.close()


# ============================================================================
# Backend Process Management
# ============================================================================

async def start_backend(artifact_id: str) -> dict:
    """Start an artifact's Python backend process."""
    artifact = get_artifact(artifact_id)
    if not artifact:
        raise ValueError(f"Artifact {artifact_id} not found")

    manifest = artifact["manifest"]
    if not manifest.get("backend"):
        raise ValueError(f"Artifact {artifact_id} has no backend configured")

    # Check if already running
    if artifact_id in _artifact_processes:
        proc = _artifact_processes[artifact_id].get("proc")
        if proc and proc.returncode is None:
            return {"status": "running", "pid": proc.pid}

    disk_path = Path(artifact["disk_path"])
    backend_dir = disk_path / "backend"

    # Install deps if requirements.txt exists
    reqs = backend_dir / "requirements.txt"
    if reqs.exists():
        deps_dir = backend_dir / ".deps"
        deps_dir.mkdir(exist_ok=True)
        try:
            proc = await asyncio.create_subprocess_exec(
                "uv", "pip", "install", "-r", str(reqs),
                "--target", str(deps_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=60)
        except Exception as e:
            log.warning(f"Dependency install failed for {artifact_id}: {e}")

    # Build environment
    import os
    env = {**os.environ}
    deps_dir = backend_dir / ".deps"
    if deps_dir.exists():
        pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{deps_dir}:{pythonpath}" if pythonpath else str(deps_dir)

    # Find the harness script
    harness_path = Path(__file__).parent / "artifact_harness.py"

    proc = await asyncio.create_subprocess_exec(
        "python", str(harness_path), str(backend_dir),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(backend_dir),
        env=env,
    )

    _artifact_processes[artifact_id] = {
        "proc": proc,
        "cwd": str(backend_dir),
        "restart_count": 0,
        "pending_responses": {},  # req_id -> asyncio.Future
        "manifest": manifest,
    }

    # Start reader task — sole consumer of proc.stdout
    reader_task = asyncio.create_task(_backend_stdout_reader(artifact_id, proc))
    _artifact_processes[artifact_id]["reader_task"] = reader_task

    log.info(f"Started backend for artifact {artifact_id}, pid={proc.pid}")

    # Register event subscriptions from manifest
    _register_event_subscriptions(artifact_id, manifest)

    return {"status": "running", "pid": proc.pid}


async def stop_backend(artifact_id: str, save_state: bool = False) -> dict:
    """Stop an artifact's backend process."""
    if artifact_id not in _artifact_processes:
        return {"status": "not_running"}

    proc_info = _artifact_processes[artifact_id]
    proc = proc_info.get("proc")

    if proc and proc.returncode is None:
        if save_state:
            await _request_state_save(artifact_id)
        await _stop_backend_process(artifact_id)

    # Unregister all event subscriptions
    from vlt.daemon.artifact_event_bus import get_event_bus
    get_event_bus().unsubscribe(artifact_id)

    return {"status": "stopped"}


async def _send_to_backend(artifact_id: str, data: dict) -> None:
    """Write a JSON line to the backend's stdin.

    Used by both call_backend() and the harness dispatcher to send replies to
    backend-initiated connector/storage calls.
    """
    proc_info = _artifact_processes.get(artifact_id)
    if not proc_info:
        raise ValueError(f"Backend not running for artifact {artifact_id}")
    proc = proc_info["proc"]
    if proc.returncode is not None:
        raise ValueError(f"Backend process exited for artifact {artifact_id}")
    line = json.dumps(data) + "\n"
    proc.stdin.write(line.encode())
    await proc.stdin.drain()


async def call_backend(artifact_id: str, action: str, params: dict, timeout: float = 30.0) -> dict:
    """Send a request to the artifact backend and await the response.

    Uses an ID-based Future so that _backend_stdout_reader() is the SOLE reader
    of proc.stdout.  No direct readline() here — the reader resolves our Future
    when the matching response line arrives.
    """
    if artifact_id not in _artifact_processes:
        raise ValueError(f"Backend not running for artifact {artifact_id}")

    proc_info = _artifact_processes[artifact_id]
    proc = proc_info["proc"]
    if proc.returncode is not None:
        raise ValueError(f"Backend process exited for artifact {artifact_id}")

    req_id = uuid.uuid4().hex[:8]
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    proc_info["pending_responses"][req_id] = future

    try:
        await _send_to_backend(artifact_id, {"id": req_id, "action": action, "params": params})
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            raise ValueError(f"Backend call '{action}' timed out after {timeout}s")
    finally:
        # Always clean up the pending entry regardless of success/failure/cancel
        proc_info["pending_responses"].pop(req_id, None)


def _register_event_subscriptions(artifact_id: str, manifest: dict):
    """Register event bus subscriptions declared in manifest.events.subscribes."""
    subscribes = manifest.get("events", {}).get("subscribes", [])
    if not subscribes:
        return

    from vlt.daemon.artifact_event_bus import get_event_bus
    bus = get_event_bus()

    async def _event_callback(recipient_id: str, event_type: str, source_id: str, payload: dict):
        """Forward event to backend handle() and broadcast to WS clients."""
        # Notify the backend process
        if artifact_id in _artifact_processes:
            try:
                await call_backend(
                    artifact_id,
                    "__event",
                    {"event_type": event_type, "source": source_id, "payload": payload},
                )
            except Exception as e:
                log.warning(f"Event delivery to backend {artifact_id} failed: {e}")

        # Broadcast to events WebSocket clients
        try:
            from vlt.daemon.artifact_routes import broadcast_events
            await broadcast_events(
                artifact_id,
                {
                    "type": "vlt_event",
                    "event_type": event_type,
                    "source": source_id,
                    "payload": payload,
                },
            )
        except Exception as e:
            log.warning(f"Event WS broadcast for {artifact_id} failed: {e}")

    for event_type in subscribes:
        bus.subscribe(artifact_id, event_type, _event_callback)
        log.debug(f"Artifact {artifact_id} subscribed to event '{event_type}'")


async def _stop_backend_process(artifact_id: str):
    """Terminate a backend process and clean up."""
    if artifact_id not in _artifact_processes:
        return

    proc_info = _artifact_processes.pop(artifact_id, None)
    if not proc_info:
        return

    proc = proc_info.get("proc")
    if proc and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()

    reader = proc_info.get("reader_task")
    if reader:
        reader.cancel()


async def _request_state_save(artifact_id: str):
    """Ask backend to serialize state before restart."""
    try:
        result = await asyncio.wait_for(
            call_backend(artifact_id, "__save_state", {}),
            timeout=5,
        )
        if result.get("state"):
            artifact = get_artifact(artifact_id)
            if artifact:
                state_path = Path(artifact["disk_path"]) / ".vlt" / "hot_state.json"
                state_path.write_text(json.dumps(result["state"]))
    except Exception as e:
        log.warning(f"State save failed for {artifact_id}: {e}")


async def _backend_stdout_reader(artifact_id: str, proc):
    """Sole reader of the backend process stdout.

    Dispatches each line to one of:
      - pending_responses Future  (lines with "id" matching a call_backend() request)
      - harness_dispatcher        (lines with "_type" — backend-initiated calls/events)
      - log handler               (lines with "log" key or plain text)

    This eliminates the race condition where call_backend() and this reader
    competed for the same readline().
    """
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break

            raw = line.decode().strip()
            if not raw:
                continue

            # Parse JSON
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.debug(f"Artifact {artifact_id} non-JSON stdout: {raw}")
                continue

            # ------------------------------------------------------------------
            # 1. Backend-initiated outbound message (_type field present)
            # ------------------------------------------------------------------
            msg_type = data.get("_type")
            if msg_type:
                if msg_type == "log":
                    level = data.get("level", "info")
                    getattr(log, level, log.info)(
                        f"Artifact {artifact_id}: {data.get('message', raw)}"
                    )
                elif msg_type in ("event", "notification"):
                    log.debug(f"Artifact {artifact_id} {msg_type}: {raw}")
                    # Forward to WS clients if routing is set up
                    try:
                        from vlt.daemon.artifact_routes import broadcast_events
                        await broadcast_events(artifact_id, data)
                    except Exception as e:
                        log.debug(f"Event broadcast skipped for {artifact_id}: {e}")
                else:
                    # connector_call, storage, or custom — delegate to dispatcher
                    try:
                        from vlt.daemon.harness_dispatcher import dispatch_backend_message
                        await dispatch_backend_message(artifact_id, data, _send_to_backend)
                    except ImportError:
                        log.warning(
                            f"harness_dispatcher not available — dropping {msg_type} "
                            f"from {artifact_id}"
                        )
                    except Exception as e:
                        log.error(
                            f"Dispatcher error for {artifact_id} ({msg_type}): {e}"
                        )
                continue

            # ------------------------------------------------------------------
            # 2. Response to a call_backend() request (has "id" field)
            # ------------------------------------------------------------------
            resp_id = data.get("id")
            if resp_id:
                proc_info = _artifact_processes.get(artifact_id)
                if proc_info:
                    future = proc_info["pending_responses"].get(resp_id)
                    if future and not future.done():
                        future.set_result(data)
                        continue
                    elif future:
                        log.warning(
                            f"Artifact {artifact_id}: received response for already-done "
                            f"request id={resp_id}"
                        )
                        continue
                # Unknown id — fall through to log
                log.debug(f"Artifact {artifact_id}: unmatched response id={resp_id}")
                continue

            # ------------------------------------------------------------------
            # 2b. Skip harness startup "ready" signal — not a response to any request
            # ------------------------------------------------------------------
            if data.get("status") == "ready" and len(data) == 1:
                log.info(f"Artifact {artifact_id} backend ready")
                continue

            # ------------------------------------------------------------------
            # 3. Legacy response without id (backwards compatibility)
            #    Try to match to the oldest pending Future, if any.
            # ------------------------------------------------------------------
            proc_info = _artifact_processes.get(artifact_id)
            if proc_info:
                pending = proc_info["pending_responses"]
                if pending:
                    # Resolve the oldest pending request (FIFO)
                    oldest_id = next(iter(pending))
                    future = pending[oldest_id]
                    if not future.done():
                        log.debug(
                            f"Artifact {artifact_id}: legacy (no-id) response matched to "
                            f"oldest pending id={oldest_id}"
                        )
                        future.set_result(data)
                        continue

            # ------------------------------------------------------------------
            # 4. Diagnostic / log messages
            # ------------------------------------------------------------------
            if "log" in data:
                log.info(f"Artifact {artifact_id}: {data['log']}")
            else:
                log.debug(f"Artifact {artifact_id} stdout (unrouted): {raw}")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.error(f"Backend reader error for {artifact_id}: {e}")
    finally:
        # Fail all pending futures so callers don't hang indefinitely
        proc_info = _artifact_processes.get(artifact_id)
        if proc_info:
            for req_id, future in list(proc_info.get("pending_responses", {}).items()):
                if not future.done():
                    future.set_exception(
                        ValueError(f"Backend process for {artifact_id} exited unexpectedly")
                    )
            proc_info["pending_responses"].clear()

        if artifact_id in _artifact_processes:
            restart_count = (proc_info or {}).get("restart_count", 0)

            # Auto-restart if appropriate
            artifact = get_artifact(artifact_id)
            if artifact and artifact["state"] in ("building", "testing", "deployed") and restart_count < 3:
                log.info(f"Auto-restarting backend for {artifact_id} (attempt {restart_count + 1})")
                _artifact_processes.pop(artifact_id, None)
                await asyncio.sleep(2)
                try:
                    await start_backend(artifact_id)
                    _artifact_processes[artifact_id]["restart_count"] = restart_count + 1
                except Exception as e:
                    log.error(f"Auto-restart failed for {artifact_id}: {e}")
                    transition_state(artifact_id, "error", actor="harness")
            elif artifact_id in _artifact_processes:
                _artifact_processes.pop(artifact_id, None)
                if restart_count >= 3:
                    log.error(f"Max restarts exceeded for {artifact_id}")
                    transition_state(artifact_id, "error", actor="harness")


# ============================================================================
# Test Runner
# ============================================================================

async def run_tests(artifact_id: str) -> dict:
    """Run artifact test command from manifest."""
    artifact = get_artifact(artifact_id)
    if not artifact:
        raise ValueError(f"Artifact {artifact_id} not found")

    manifest = artifact["manifest"]
    tests_config = manifest.get("tests")
    if not tests_config or not tests_config.get("command"):
        return {"passed": True, "exit_code": 0, "stdout": "No tests configured", "stderr": "", "duration_ms": 0}

    import time
    start = time.monotonic()
    timeout = tests_config.get("timeout", 30)

    try:
        proc = await asyncio.create_subprocess_shell(
            tests_config["command"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=artifact["disk_path"],
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        duration_ms = int((time.monotonic() - start) * 1000)

        return {
            "passed": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
            "duration_ms": duration_ms,
        }
    except asyncio.TimeoutError:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "passed": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Test timed out after {timeout}s",
            "duration_ms": duration_ms,
        }


# ============================================================================
# Storage (artifact-scoped key-value)
# ============================================================================

def storage_get(artifact_id: str, key: str) -> dict | None:
    """Get a value from artifact-scoped storage."""
    artifact = get_artifact(artifact_id)
    if not artifact:
        return None
    path = Path(artifact["disk_path"]) / ".vlt" / "storage" / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def storage_set(artifact_id: str, key: str, value) -> bool:
    """Set a value in artifact-scoped storage."""
    artifact = get_artifact(artifact_id)
    if not artifact:
        return False
    path = Path(artifact["disk_path"]) / ".vlt" / "storage" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return True


def storage_list(artifact_id: str) -> list[str]:
    """List all keys in artifact-scoped storage."""
    artifact = get_artifact(artifact_id)
    if not artifact:
        return []
    storage_dir = Path(artifact["disk_path"]) / ".vlt" / "storage"
    if not storage_dir.exists():
        return []
    return [p.stem for p in storage_dir.glob("*.json")]


# ============================================================================
# Screenshot Capture
# ============================================================================

async def capture_screenshot(artifact_id: str) -> dict:
    """Capture a screenshot of the artifact's frontend via headless Playwright."""
    artifact = get_artifact(artifact_id)
    if not artifact:
        raise ValueError(f"Artifact {artifact_id} not found")

    from vlt.config import settings
    daemon_port = getattr(settings, "daemon_port", 8765)
    url = f"http://localhost:{daemon_port}/api/artifacts/{artifact_id}/frontend/index.html"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    screenshot_dir = Path(artifact["disk_path"]) / ".vlt" / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = screenshot_dir / f"{timestamp}.png"

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        log.warning("Playwright not installed — skipping screenshot capture")
        return {"error": "Playwright not installed. Install with: pip install playwright && playwright install chromium"}

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1280, "height": 720})
            await page.goto(url, wait_until="networkidle", timeout=15000)
            await page.screenshot(path=str(screenshot_path))
            await browser.close()

        return {
            "path": str(screenshot_path.relative_to(Path(artifact["disk_path"]))),
            "absolute_path": str(screenshot_path),
            "width": 1280,
            "height": 720,
        }
    except Exception as e:
        log.error(f"Screenshot capture failed for {artifact_id}: {e}")
        return {"error": str(e)}


# ============================================================================
# Vision Model Review
# ============================================================================

async def review_artifact(artifact_id: str) -> dict:
    """Run vision model review: screenshot → vision model describes → oracle decides pass/fail."""
    artifact = get_artifact(artifact_id)
    if not artifact:
        raise ValueError(f"Artifact {artifact_id} not found")

    # 1. Capture screenshot
    screenshot_result = await capture_screenshot(artifact_id)
    if "error" in screenshot_result:
        log.warning(f"Vision review skipped for {artifact_id}: {screenshot_result['error']}")
        return {"skipped": True, "reason": screenshot_result["error"]}

    # 2. Discover vision model
    vision_model_info = await _discover_vision_model()
    if not vision_model_info:
        log.warning(f"No vision model configured — skipping review for {artifact_id}")
        return {
            "skipped": True,
            "reason": "Please configure a vision model in Settings > Models",
        }

    # 3. Send screenshot to vision model for description
    import base64
    screenshot_path = screenshot_result["absolute_path"]
    with open(screenshot_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    description = artifact.get("description", artifact.get("name", "artifact"))
    vision_prompt = (
        f"Describe what you see in this screenshot of an artifact called '{artifact['name']}'. "
        f"The artifact is supposed to: {description}. "
        f"Describe the layout, content, and visual elements. "
        f"Note any issues, errors, or things that look incomplete."
    )

    try:
        vision_response = await _call_model_with_image(
            vision_model_info, vision_prompt, image_b64
        )
    except Exception as e:
        log.error(f"Vision model call failed: {e}")
        return {"skipped": True, "reason": f"Vision model error: {e}"}

    # 4. Pass assessment to primary model for pass/fail decision
    # For now, return the vision assessment and let the state machine decide
    return {
        "skipped": False,
        "assessment": vision_response,
        "screenshot": screenshot_result["path"],
        "vision_model": vision_model_info.get("model"),
    }


async def _discover_vision_model() -> dict | None:
    """Find a configured vision-capable model. Returns {provider, model, api_key, base_url} or None."""
    import httpx

    # Check backend settings for user's configured vision model
    try:
        from vlt.config import settings
        vault_url = settings.vault_url
        resp = httpx.get(f"{vault_url}/api/settings/models", timeout=5.0)
        if resp.status_code == 200:
            model_settings = resp.json()
            vision_model = model_settings.get("vision_model")
            vision_provider = model_settings.get("vision_provider")
            if vision_model and vision_provider:
                return {
                    "model": vision_model,
                    "provider": vision_provider,
                }
    except Exception as e:
        log.debug(f"Could not fetch vision model settings: {e}")

    # Fallback: check if Gemini is configured (server-wide env var)
    import os
    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        return {
            "model": "gemini-2.0-flash-exp",
            "provider": "google",
        }

    return None


async def _call_model_with_image(model_info: dict, prompt: str, image_b64: str) -> str:
    """Call a vision model with text + image. Returns the model's response text."""
    import httpx

    provider = model_info["provider"]
    model = model_info["model"]

    if provider == "google":
        import os
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}}
                ]
            }]
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    elif provider == "openrouter":
        from vlt.config import settings
        vault_url = settings.vault_url
        # Get the API key from backend
        key_resp = httpx.get(f"{vault_url}/api/settings/models", timeout=5.0)
        openrouter_key = ""
        if key_resp.status_code == 200:
            # We can't get the actual key from the settings endpoint (it's masked)
            # Fall back to env var
            import os
            openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }]
        }
        headers = {"Authorization": f"Bearer {openrouter_key}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    elif provider == "glm":
        # GLM vision models use OpenAI-compatible API
        from vlt.config import settings
        import os
        vault_url = settings.vault_url
        glm_key = os.environ.get("GLM_API_KEY", "")
        url = "https://api.z.ai/api/coding/paas/v4/chat/completions"
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }]
        }
        headers = {"Authorization": f"Bearer {glm_key}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    raise ValueError(f"Unsupported vision provider: {provider}")


# ============================================================================
# Helpers
# ============================================================================

def _artifact_to_dict(artifact: Artifact) -> dict:
    """Convert Artifact ORM object to dict."""
    return {
        "id": artifact.id,
        "user_id": artifact.user_id,
        "project_id": artifact.project_id,
        "name": artifact.name,
        "description": artifact.description,
        "type": artifact.type,
        "state": artifact.state.value if isinstance(artifact.state, ArtifactState) else artifact.state,
        "state_history": json.loads(artifact.state_history_json),
        "manifest": json.loads(artifact.manifest_json),
        "thread_id": artifact.thread_id,
        "disk_path": artifact.disk_path,
        "version": artifact.version,
        "created_at": artifact.created_at,
        "updated_at": artifact.updated_at,
    }
