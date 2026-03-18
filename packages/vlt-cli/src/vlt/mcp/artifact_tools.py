"""Artifact sandbox tools for the vlt MCP server.

Exposes artifact lifecycle management so AI agents can create, inspect,
update source files, run tests, and transition state for artifact sandboxes.

Tools registered:
    vlt_artifact_create    — create a new artifact sandbox
    vlt_artifact_list      — list artifacts, optionally filtered by project
    vlt_artifact_update    — write source files to an artifact's disk path
    vlt_artifact_state     — transition artifact state machine
    vlt_artifact_test      — run the artifact's test command
    vlt_artifact_screenshot — (Phase 7) capture a screenshot of the artifact frontend

Dynamic tool registration from deployed artifacts requires an MCP server restart
(Phase 2 feature, per spec decision R5). Each deployed artifact's custom tools
will be discoverable only after the agent restarts its vlt-mcp process.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DAEMON_BASE = None  # resolved lazily via settings


def _daemon_url() -> str:
    """Return the daemon base URL from settings (e.g. http://127.0.0.1:8765)."""
    global _DAEMON_BASE
    if _DAEMON_BASE is None:
        try:
            from vlt.config import get_settings
            _DAEMON_BASE = get_settings().daemon_url
        except Exception:
            _DAEMON_BASE = "http://127.0.0.1:8765"
    return _DAEMON_BASE


def register_artifact_tools(mcp) -> None:
    """Register artifact management tools onto a FastMCP server instance."""

    @mcp.tool()
    def vlt_artifact_create(
        name: str,
        description: str = "",
        project_id: str = "default",
        type: str = "ephemeral",
        template: str = "",
    ) -> dict:
        """Create a new artifact sandbox (frontend + optional backend).

        Args:
            name: Human-readable artifact name (e.g. "My Dashboard").
            description: Optional description of what the artifact does.
            project_id: Project to associate the artifact with (default: "default").
            type: "ephemeral" (temp) or "persistent" (permanent). Default: "ephemeral".
            template: Template to initialize from (e.g. "text_factory"). Empty = blank.
                      Use vlt_artifact_list_templates() to see available templates.

        Returns:
            {status, artifact_id, name, project_id, state, disk_path, template}
        """
        import httpx
        from vlt.mcp import _ok, _err

        try:
            body: dict = {
                "name": name,
                "description": description or None,
                "project_id": project_id,
                "type": type,
            }
            if template:
                body["template"] = template

            resp = httpx.post(
                f"{_daemon_url()}/api/artifacts",
                json=body,
                timeout=10.0,
            )
            if resp.status_code >= 400:
                return _err("API_ERROR", f"Daemon returned HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            return _ok(
                artifact_id=data["id"],
                name=data["name"],
                project_id=data["project_id"],
                state=data["state"],
                disk_path=data["disk_path"],
                type=data["type"],
                template=template or "blank",
            )
        except Exception as e:
            logger.exception("vlt_artifact_create failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_artifact_list(project_id: str = "") -> dict:
        """List all artifacts, optionally filtered by project.

        Args:
            project_id: Filter to this project slug. Lists all projects if empty.

        Returns:
            {status, artifacts: [{id, name, project_id, state, type, version, updated_at}], total}
        """
        import httpx
        from vlt.mcp import _ok, _err

        try:
            params: dict = {}
            if project_id:
                params["project_id"] = project_id

            resp = httpx.get(
                f"{_daemon_url()}/api/artifacts",
                params=params,
                timeout=10.0,
            )
            if resp.status_code >= 400:
                return _err("API_ERROR", f"Daemon returned HTTP {resp.status_code}: {resp.text[:300]}")

            items = resp.json()
            artifacts = [
                {
                    "id": a["id"],
                    "name": a["name"],
                    "project_id": a["project_id"],
                    "state": a["state"],
                    "type": a["type"],
                    "version": a["version"],
                    "updated_at": a["updated_at"],
                }
                for a in items
            ]
            return _ok(artifacts=artifacts, total=len(artifacts))
        except Exception as e:
            logger.exception("vlt_artifact_list failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_artifact_update(artifact_id: str, files: str = "{}") -> dict:
        """Write source files to an artifact sandbox.

        Files are written directly to the artifact's disk path. Parent directories
        are created automatically.

        Args:
            artifact_id: Artifact UUID to update.
            files: JSON string mapping relative paths to file contents.
                   Example: '{"frontend/index.html": "<html>…</html>",
                              "backend/main.py": "def handle(): ..."}'

        Returns:
            {status, artifact_id, files_written: [path, ...], disk_path}
        """
        import httpx
        from vlt.mcp import _ok, _err

        # Parse files JSON
        try:
            files_dict: dict[str, str] = json.loads(files) if files else {}
        except json.JSONDecodeError as e:
            return _err("INVALID_FILES", f"files must be a valid JSON string: {e}")

        if not isinstance(files_dict, dict):
            return _err("INVALID_FILES", "files must be a JSON object mapping path → content")

        # Fetch artifact to get disk_path
        try:
            resp = httpx.get(
                f"{_daemon_url()}/api/artifacts/{artifact_id}",
                timeout=10.0,
            )
            if resp.status_code == 404:
                return _err("NOT_FOUND", f"Artifact '{artifact_id}' not found.")
            if resp.status_code >= 400:
                return _err("API_ERROR", f"Daemon returned HTTP {resp.status_code}: {resp.text[:300]}")
            artifact_data = resp.json()
        except Exception as e:
            logger.exception("vlt_artifact_update: failed to fetch artifact")
            return _err("INTERNAL_ERROR", str(e))

        disk_path = Path(artifact_data["disk_path"])

        # Write each file
        written: list[str] = []
        errors: list[str] = []
        for rel_path, content in files_dict.items():
            # Security: prevent path traversal
            try:
                target = (disk_path / rel_path).resolve()
                if not str(target).startswith(str(disk_path.resolve())):
                    errors.append(f"{rel_path}: path traversal rejected")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(rel_path)
            except Exception as exc:
                errors.append(f"{rel_path}: {exc}")

        result = _ok(
            artifact_id=artifact_id,
            files_written=written,
            disk_path=str(disk_path),
        )
        if errors:
            result["errors"] = errors
        return result

    @mcp.tool()
    def vlt_artifact_state(artifact_id: str, target_state: str) -> dict:
        """Transition an artifact through its state machine.

        Valid state chain: draft → building → testing → reviewing → approved → deployed
        Error recovery: error → draft | building
        Deployed updates: deployed → update_pending → building

        Args:
            artifact_id: Artifact UUID.
            target_state: One of draft, building, testing, reviewing, approved,
                          deployed, error, update_pending.

        Returns:
            {status, artifact_id, state, previous_state}
        """
        import httpx
        from vlt.mcp import _ok, _err

        try:
            resp = httpx.post(
                f"{_daemon_url()}/api/artifacts/{artifact_id}/state",
                json={"target_state": target_state, "actor": "agent"},
                timeout=10.0,
            )
            if resp.status_code == 404:
                return _err("NOT_FOUND", f"Artifact '{artifact_id}' not found.")
            if resp.status_code == 422:
                return _err("INVALID_TRANSITION", resp.json().get("detail", "Invalid state transition"))
            if resp.status_code >= 400:
                return _err("API_ERROR", f"Daemon returned HTTP {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            history = data.get("state_history", [])
            previous = history[-2]["state"] if len(history) >= 2 else None
            return _ok(
                artifact_id=data["id"],
                state=data["state"],
                previous_state=previous,
                version=data["version"],
            )
        except Exception as e:
            logger.exception("vlt_artifact_state failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_artifact_test(artifact_id: str) -> dict:
        """Run the artifact's test command and return the results.

        Executes the test suite configured for the artifact. Results include
        pass/fail status, exit code, stdout, stderr, and duration.

        Args:
            artifact_id: Artifact UUID.

        Returns:
            {status, artifact_id, passed, exit_code, stdout, stderr, duration_ms}
        """
        import httpx
        from vlt.mcp import _ok, _err

        try:
            resp = httpx.post(
                f"{_daemon_url()}/api/artifacts/{artifact_id}/test",
                timeout=120.0,  # tests can take a while
            )
            if resp.status_code == 404:
                return _err("NOT_FOUND", f"Artifact '{artifact_id}' not found.")
            if resp.status_code >= 400:
                return _err("API_ERROR", f"Daemon returned HTTP {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            return _ok(
                artifact_id=artifact_id,
                passed=data.get("passed", False),
                exit_code=data.get("exit_code", -1),
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", ""),
                duration_ms=data.get("duration_ms", 0),
            )
        except Exception as e:
            logger.exception("vlt_artifact_test failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_artifact_call(
        artifact_id: str,
        action: str,
        params: str = "{}",
    ) -> dict:
        """Call an action on an artifact's backend process.

        The backend must be running (use vlt_artifact_state to transition to
        "building", then the backend starts automatically; or start it manually).

        Common actions for text_factory artifacts:
            get_config      — returns pipeline stage configuration
            test            — run the pipeline once on a topic: {"topic": "..."}
            generate        — run pipeline and add result to content queue
            update_config   — update pipeline stages: {"stages": [...]}
            get_queue       — list content queue items
            get_history     — list test run history
            approve         — approve a queue item: {"item_id": "cnt_..."}
            reject          — reject a queue item: {"item_id": "cnt_...", "reason": "..."}
            save_runbook    — save current pipeline as named config: {"name": "..."}
            list_runbooks   — list saved runbooks
            run_runbook     — execute a saved runbook: {"name": "...", "topic": "..."}

        Args:
            artifact_id: Artifact UUID.
            action: Backend action name.
            params: JSON string of action parameters (default: "{}").

        Returns:
            {status, artifact_id, action, result: <action-specific data>}
        """
        import httpx
        from vlt.mcp import _ok, _err

        try:
            params_dict = json.loads(params) if params else {}
        except json.JSONDecodeError as e:
            return _err("INVALID_PARAMS", f"params must be valid JSON: {e}")

        # Pipeline actions need longer timeouts
        slow_actions = {"test", "generate", "run_runbook"}
        timeout = 120.0 if action in slow_actions else 30.0

        try:
            resp = httpx.post(
                f"{_daemon_url()}/api/artifacts/{artifact_id}/backend/call",
                json={"action": action, "params": params_dict},
                timeout=timeout,
            )
            if resp.status_code == 400:
                detail = resp.text[:300]
                try:
                    detail = resp.json().get("detail", detail)
                except Exception:
                    pass
                # Check for "backend not running" specifically
                if "not running" in detail.lower():
                    return _err(
                        "BACKEND_NOT_RUNNING",
                        f"Backend not running for artifact {artifact_id}. "
                        "Transition to 'building' state first: "
                        "vlt_artifact_state(artifact_id, 'building')"
                    )
                return _err("ACTION_ERROR", detail)
            if resp.status_code >= 400:
                return _err("API_ERROR", f"HTTP {resp.status_code}: {resp.text[:300]}")

            data = resp.json()

            # If the harness returned an error (exception in handle())
            if isinstance(data, dict) and "error" in data and "result" not in data:
                return _err("BACKEND_ERROR", data["error"])

            result = data.get("result", data)
            return _ok(artifact_id=artifact_id, action=action, result=result)

        except httpx.TimeoutException:
            return _err("TIMEOUT", f"Backend call '{action}' timed out after {timeout}s")
        except Exception as e:
            logger.exception("vlt_artifact_call failed")
            return _err("INTERNAL_ERROR", str(e))

    @mcp.tool()
    def vlt_artifact_screenshot(artifact_id: str) -> dict:
        """Capture a screenshot of the artifact's frontend.

        NOTE: Screenshots require the vision/browser harness (Phase 7).
        This feature is not yet implemented. The artifact frontend is served at:
            /vlt/api/artifacts/{id}/frontend/index.html

        Args:
            artifact_id: Artifact UUID.

        Returns:
            {status, message} — always returns a Phase 7 notice until implemented.
        """
        from vlt.mcp import _err

        return _err(
            "NOT_IMPLEMENTED",
            (
                f"Screenshots for artifact '{artifact_id}' require the vision review harness "
                "(Phase 7 — not yet implemented). "
                "The artifact frontend is accessible at "
                f"/vlt/api/artifacts/{artifact_id}/frontend/index.html via the daemon proxy."
            ),
        )
