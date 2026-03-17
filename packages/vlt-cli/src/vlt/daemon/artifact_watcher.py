"""Artifact file watcher with debounced hot reload.

Watches active artifact directories for file changes using watchdog.
Classifies changes (CSS-only, JS/HTML, Python, manifest) and coordinates
reload: CSS hot-swap, frontend full reload, backend state-aware restart.
Applies state machine demotion policies on code changes.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

log = logging.getLogger(__name__)

# Single shared observer for all artifact watchers
_observer: Optional[Observer] = None
_watchers: dict[str, "ArtifactWatcher"] = {}


def _get_observer() -> Observer:
    """Get or create the shared watchdog observer."""
    global _observer
    if _observer is None or not _observer.is_alive():
        _observer = Observer()
        _observer.daemon = True
        _observer.start()
    return _observer


class ArtifactChangeEvent:
    """Classified file change event for an artifact."""
    __slots__ = ("artifact_id", "css_only", "frontend_changed",
                 "backend_changed", "manifest_changed", "changed_files")

    def __init__(self, artifact_id: str):
        self.artifact_id = artifact_id
        self.css_only = False
        self.frontend_changed = False
        self.backend_changed = False
        self.manifest_changed = False
        self.changed_files: list[str] = []


class _ArtifactEventHandler(FileSystemEventHandler):
    """Watchdog handler that collects changes and debounces."""

    def __init__(self, watcher: "ArtifactWatcher"):
        super().__init__()
        self.watcher = watcher

    def on_any_event(self, event: FileSystemEvent):
        if event.is_directory:
            return
        src = event.src_path
        # Ignore hidden/internal directories
        if "/.git/" in src or "/.vlt/" in src or "/__pycache__/" in src or "/.deps/" in src:
            return
        self.watcher._on_file_change(src)


class ArtifactWatcher:
    """Watches a single artifact directory for file changes with debounce."""

    DEBOUNCE_SECONDS = 0.5

    def __init__(self, artifact_id: str, artifact_path: str, loop: asyncio.AbstractEventLoop):
        self.artifact_id = artifact_id
        self.path = Path(artifact_path)
        self.loop = loop
        self._pending_changes: set[str] = set()
        self._debounce_handle: Optional[asyncio.TimerHandle] = None
        self._handler = _ArtifactEventHandler(self)
        self._watch = None

    def start(self):
        """Start watching the artifact directory."""
        observer = _get_observer()
        self._watch = observer.schedule(self._handler, str(self.path), recursive=True)
        log.info(f"Started watching artifact {self.artifact_id} at {self.path}")

    def stop(self):
        """Stop watching."""
        if self._watch and _observer and _observer.is_alive():
            try:
                _observer.unschedule(self._watch)
            except Exception:
                pass
        self._watch = None
        if self._debounce_handle:
            self._debounce_handle.cancel()
            self._debounce_handle = None
        log.info(f"Stopped watching artifact {self.artifact_id}")

    def _on_file_change(self, file_path: str):
        """Called by watchdog handler on any file change."""
        self._pending_changes.add(file_path)
        # Reset debounce timer (thread-safe via loop.call_soon_threadsafe)
        self.loop.call_soon_threadsafe(self._reset_debounce)

    def _reset_debounce(self):
        """Reset the debounce timer."""
        if self._debounce_handle:
            self._debounce_handle.cancel()
        self._debounce_handle = self.loop.call_later(
            self.DEBOUNCE_SECONDS,
            lambda: asyncio.ensure_future(self._flush())
        )

    async def _flush(self):
        """Process all pending changes after debounce window."""
        if not self._pending_changes:
            return

        changes = self._pending_changes.copy()
        self._pending_changes.clear()

        event = self._classify_changes(changes)
        if not event.changed_files:
            return

        log.info(
            f"Artifact {self.artifact_id} changes: "
            f"css_only={event.css_only} frontend={event.frontend_changed} "
            f"backend={event.backend_changed} manifest={event.manifest_changed} "
            f"files={event.changed_files}"
        )

        await reload_artifact(event)

    def _classify_changes(self, file_paths: set[str]) -> ArtifactChangeEvent:
        """Classify a set of file changes into categories."""
        event = ArtifactChangeEvent(self.artifact_id)
        has_css = False
        has_non_css_frontend = False

        for fp in file_paths:
            try:
                rel = str(Path(fp).relative_to(self.path))
            except ValueError:
                continue

            event.changed_files.append(rel)

            if rel == "manifest.json":
                event.manifest_changed = True
            elif rel.startswith("frontend/"):
                if rel.endswith(".css"):
                    has_css = True
                else:
                    has_non_css_frontend = True
                event.frontend_changed = True
            elif rel.startswith("backend/"):
                event.backend_changed = True

        event.css_only = has_css and not has_non_css_frontend and not event.backend_changed

        return event


# ============================================================================
# Reload Coordinator
# ============================================================================

async def reload_artifact(event: ArtifactChangeEvent):
    """Coordinate reload based on what changed."""
    from vlt.daemon import artifact_service as svc
    from vlt.daemon.artifact_routes import broadcast_hmr, broadcast_state

    artifact = svc.get_artifact(event.artifact_id)
    if not artifact:
        return

    # 1. Apply state demotion policy
    await _apply_state_demotion(event, artifact)

    # 2. Backend restart (if Python changed)
    if event.backend_changed and event.artifact_id in svc._artifact_processes:
        log.info(f"Restarting backend for artifact {event.artifact_id}")
        await svc.stop_backend(event.artifact_id, save_state=True)
        try:
            await svc.start_backend(event.artifact_id)
        except Exception as e:
            log.error(f"Backend restart failed: {e}")
            await broadcast_state(event.artifact_id, {
                "type": "error", "message": f"Backend restart failed: {e}"
            })

    # 3. Frontend reload
    if event.frontend_changed:
        if event.css_only:
            css_files = [f for f in event.changed_files if f.endswith(".css")]
            await broadcast_hmr(event.artifact_id, {
                "type": "css_update", "files": css_files
            })
        else:
            await broadcast_hmr(event.artifact_id, {"type": "will_reload"})
            await asyncio.sleep(0.1)  # Brief pause for state save
            await broadcast_hmr(event.artifact_id, {"type": "reload"})

    # 4. Manifest reload
    if event.manifest_changed:
        try:
            manifest_path = Path(artifact["disk_path"]) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            svc.update_artifact(event.artifact_id, manifest=manifest)
        except Exception as e:
            log.warning(f"Manifest reload failed: {e}")

    # 5. Auto-run tests if configured
    manifest = artifact["manifest"]
    if manifest.get("tests", {}) and manifest["tests"].get("command"):
        test_result = await svc.run_tests(event.artifact_id)
        await broadcast_state(event.artifact_id, {
            "type": "test_result",
            "passed": test_result["passed"],
            "summary": f"{'PASS' if test_result['passed'] else 'FAIL'} "
                       f"(exit {test_result['exit_code']}, {test_result['duration_ms']}ms)",
        })

        # Advance state machine based on test results
        if test_result["passed"] and artifact["state"] in ("building", "testing"):
            try:
                target = "testing" if artifact["state"] == "building" else "reviewing"
                svc.transition_state(event.artifact_id, target, actor="harness")
                refreshed = svc.get_artifact(event.artifact_id)
                if refreshed:
                    await broadcast_state(event.artifact_id, {
                        "type": "state_change",
                        "from": artifact["state"],
                        "to": refreshed["state"],
                        "by": "harness",
                    })
                    # Auto-trigger vision review when entering "reviewing" state
                    if target == "reviewing":
                        await _run_vision_review(event.artifact_id)
            except ValueError:
                pass  # Invalid transition — stay in current state

        # Inject test results into Claude Code session if active
        await _inject_test_results(event.artifact_id, test_result)


async def _apply_state_demotion(event: ArtifactChangeEvent, artifact: dict):
    """Apply state machine demotion policies on code changes."""
    from vlt.daemon import artifact_service as svc
    from vlt.daemon.artifact_routes import broadcast_state

    current_state = artifact["state"]
    demote_to = None

    if current_state == "approved":
        demote_to = "testing"
    elif current_state == "deployed":
        # Set update_pending — don't auto-demote deployed artifacts
        demote_to = "update_pending"
    elif current_state == "error":
        demote_to = "building"

    if demote_to:
        try:
            svc.transition_state(event.artifact_id, demote_to, actor="watcher")
            log.info(f"State demotion: {current_state} -> {demote_to} for {event.artifact_id}")
            await broadcast_state(event.artifact_id, {
                "type": "state_change",
                "from": current_state,
                "to": demote_to,
                "by": "watcher",
            })
        except ValueError as e:
            log.warning(f"State demotion failed: {e}")


async def _inject_test_results(artifact_id: str, test_result: dict):
    """Inject test results into any Claude Code session working on this artifact."""
    try:
        from vlt.daemon import artifact_service as svc
        artifact = svc.get_artifact(artifact_id)
        if not artifact:
            return

        # Check for active sessions matching this artifact's project
        import sys
        server_module = sys.modules.get("__main__")
        if not server_module:
            return

        injection_queues = getattr(server_module, "_injection_queues", None)
        if not injection_queues:
            # Try the server module directly
            try:
                from vlt.daemon.server import _injection_queues as iq
                injection_queues = iq
            except ImportError:
                return

        status = "PASSED" if test_result["passed"] else "FAILED"
        msg = (
            f"[Artifact Test Result] {artifact['name']} — {status}\n"
            f"Exit code: {test_result['exit_code']} | Duration: {test_result['duration_ms']}ms\n"
        )
        if not test_result["passed"] and test_result.get("stderr"):
            msg += f"Stderr:\n{test_result['stderr'][:500]}\n"

        # Inject into all sessions for the same project
        from vlt.db import SessionLocal
        from vlt.core.models import AgentSession
        db = SessionLocal()
        try:
            sessions = db.query(AgentSession).filter(
                AgentSession.project_id == artifact["project_id"],
                AgentSession.status.in_(["idle", "thinking"]),
            ).all()
            for session in sessions:
                injection_queues.setdefault(session.id, []).append(msg)
        finally:
            db.close()

    except Exception as e:
        log.debug(f"Test result injection failed (non-critical): {e}")


async def _run_vision_review(artifact_id: str):
    """Run vision model review and advance state based on result."""
    from vlt.daemon import artifact_service as svc
    from vlt.daemon.artifact_routes import broadcast_state

    try:
        result = await svc.review_artifact(artifact_id)

        if result.get("skipped"):
            # No vision model — skip review, advance directly to approved
            log.info(f"Vision review skipped for {artifact_id}: {result.get('reason')}")
            await broadcast_state(artifact_id, {
                "type": "error",
                "message": result.get("reason", "Vision review skipped"),
            })
            try:
                svc.transition_state(artifact_id, "approved", actor="harness")
                await broadcast_state(artifact_id, {
                    "type": "state_change", "from": "reviewing",
                    "to": "approved", "by": "harness",
                })
            except ValueError:
                pass
        else:
            # Got an assessment — for now auto-approve (full two-step review is Phase 2)
            log.info(f"Vision review for {artifact_id}: {result.get('assessment', '')[:100]}")
            try:
                svc.transition_state(artifact_id, "approved", actor="vision-review")
                await broadcast_state(artifact_id, {
                    "type": "state_change", "from": "reviewing",
                    "to": "approved", "by": "vision-review",
                })
            except ValueError:
                pass

    except Exception as e:
        log.error(f"Vision review failed for {artifact_id}: {e}")
        await broadcast_state(artifact_id, {
            "type": "error", "message": f"Vision review error: {e}",
        })


# ============================================================================
# Watcher Lifecycle Management
# ============================================================================

def start_watcher(artifact_id: str, artifact_path: str, loop: asyncio.AbstractEventLoop):
    """Start watching an artifact directory."""
    if artifact_id in _watchers:
        return  # Already watching

    watcher = ArtifactWatcher(artifact_id, artifact_path, loop)
    watcher.start()
    _watchers[artifact_id] = watcher


def stop_watcher(artifact_id: str):
    """Stop watching an artifact directory."""
    watcher = _watchers.pop(artifact_id, None)
    if watcher:
        watcher.stop()


def stop_all_watchers():
    """Stop all artifact watchers (called on daemon shutdown)."""
    for watcher in _watchers.values():
        watcher.stop()
    _watchers.clear()
    global _observer
    if _observer and _observer.is_alive():
        _observer.stop()
        _observer = None
