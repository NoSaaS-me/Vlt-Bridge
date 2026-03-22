"""Cronban scheduling tools for the vlt MCP server.

Exposes CronTrigger operations via MCP so AI agents can schedule, monitor,
and cancel time-based tasks without needing the full CLI or web UI.

Agents can list running sessions (with last model output for identification),
pick a target session, and schedule cron triggers that inject prompts into it.

Tools registered:
    cronban_sessions  — list live sessions with last assistant output snippet
    cronban_list      — list CronTriggers (with status, next fire time, fire count)
    cronban_create    — create a new CronTrigger targeting a session or CWD
    cronban_fire      — fire a CronTrigger immediately
    cronban_pause     — pause an active CronTrigger
    cronban_resume    — resume a paused CronTrigger
    cronban_delete    — permanently delete a CronTrigger
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Max chars of last assistant output to include in session listing.
_SNIPPET_CHARS = 280


def _get_settings():
    from vlt.config import get_settings
    return get_settings()


def _last_assistant_snippet(daemon_url: str, session_id: str) -> Optional[str]:
    """Read the last assistant message text from a session transcript (up to _SNIPPET_CHARS chars)."""
    try:
        import httpx
        r = httpx.get(
            f"{daemon_url}/api/sessions/{session_id}/transcript",
            params={"types": "assistant"},
            timeout=3.0,
        )
        if r.status_code != 200:
            return None
        entries = r.json().get("entries", [])
        if not entries:
            return None
        msg = entries[-1].get("message", {})
        content = msg.get("content", [])
        text = ""
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
        elif isinstance(content, str):
            text = content
        text = text.strip()
        return text[:_SNIPPET_CHARS] if text else None
    except Exception:
        return None


def register_cronban_tools(mcp) -> None:
    """Register all Cronban tools onto a FastMCP server instance."""

    from vlt.mcp import _ok, _err

    @mcp.tool()
    def cronban_sessions(
        project_id: Optional[str] = None,
        include_dead: bool = False,
    ) -> dict:
        """List Claude agent sessions with their last model output.

        Use this to find a target_session_id for cronban_create. The last_output
        field shows the tail of the last assistant message (truncated to 280 chars)
        so you can identify sessions by what they were working on.

        Args:
            project_id: Filter to a specific project slug (optional).
            include_dead: Include terminated sessions (default False).

        Returns:
            {status, sessions: [{id, name, cwd, status, model, last_activity,
             last_output (truncated to 280 chars)}], total}
        """
        try:
            import httpx
            settings = _get_settings()
            daemon_url = settings.daemon_url

            params: dict = {}
            if include_dead:
                params["include_all"] = "true"
            if project_id:
                params["project_id"] = project_id

            r = httpx.get(f"{daemon_url}/api/sessions", params=params, timeout=5.0)
            r.raise_for_status()
            sessions_raw = r.json()

            sessions = []
            for s in sessions_raw[:15]:  # cap at 15 to keep response fast
                snippet = _last_assistant_snippet(daemon_url, s["id"])
                sessions.append({
                    "id": s["id"],
                    "name": s.get("name"),
                    "cwd": s.get("cwd"),
                    "status": s.get("status"),
                    "model": s.get("model"),
                    "last_activity": s.get("last_activity"),
                    "last_output": snippet,
                })

            return _ok(sessions=sessions, total=len(sessions))
        except Exception as e:
            logger.exception("cronban_sessions failed")
            return _err("CRONBAN_SESSIONS_FAILED", str(e))

    @mcp.tool()
    def cronban_list(
        project_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict:
        """List CronTriggers.

        Returns cron schedules with their next fire time, fire count, and last
        fired timestamp so you can see what's scheduled and when.

        Args:
            project_id: Filter to a specific project slug (optional).
            status: Filter by status: "active" or "paused" (optional).

        Returns:
            {status, triggers: [...], total}
        """
        try:
            import httpx
            settings = _get_settings()
            daemon_url = settings.daemon_url

            params: dict = {}
            if project_id:
                params["project_id"] = project_id

            r = httpx.get(f"{daemon_url}/api/cronban/crons", params=params, timeout=5.0)
            r.raise_for_status()
            triggers = r.json()

            if status:
                triggers = [t for t in triggers if t.get("status") == status]

            return _ok(triggers=triggers, total=len(triggers))
        except Exception as e:
            logger.exception("cronban_list failed")
            return _err("CRONBAN_LIST_FAILED", str(e))

    @mcp.tool()
    def cronban_create(
        title: str,
        prompt_text: str,
        cron_expression: Optional[str] = None,
        fire_in: Optional[str] = None,
        fire_at: Optional[str] = None,
        target_session_id: Optional[str] = None,
        target_cwd: Optional[str] = None,
        create_new_session: bool = False,
        skill_id: Optional[str] = None,
        project_id: Optional[str] = None,
        timezone: str = "UTC",
        max_fires: Optional[int] = None,
    ) -> dict:
        """Create a scheduled task — recurring or one-off.

        SCHEDULING (provide EXACTLY ONE — passing multiple causes undefined behavior):
          - cron_expression: recurring (e.g. "0 9 * * 1-5" = weekdays at 9am UTC).
            Respects the timezone parameter.
          - fire_in: one-off relative offset (e.g. "1:30" = 90 min, "2:00:30" = 2h 30s).
          - fire_at: one-off absolute ISO datetime. MUST include timezone suffix
            (e.g. "2026-03-10T14:30:00Z"). Without a suffix, behavior is undefined.

        TARGET (provide EXACTLY ONE):
          - target_session_id: inject prompt into an existing session (preferred).
            Use cronban_sessions() to find session IDs.
          - target_cwd: spawn a new session in this directory on each fire.
            Requires create_new_session=True.

        Args:
            title: Human-readable label for this schedule.
            prompt_text: Message to inject into the session when the trigger fires.
            cron_expression: Recurring 5-field cron (e.g. "*/30 * * * *").
            fire_in: One-off offset: "hh:mm" or "hh:mm:ss" from now.
            fire_at: One-off ISO datetime with timezone (e.g. "2026-03-10T14:30:00Z").
            target_session_id: Inject into this session (from cronban_sessions).
            target_cwd: Working directory for new sessions (requires create_new_session=True).
            create_new_session: Spawn fresh session on each fire. Default: False.
            skill_id: Use a saved Skill prompt instead of prompt_text (optional).
            project_id: Associate with a project slug (optional).
            timezone: IANA timezone for cron_expression only (default "UTC").
            max_fires: Stop after this many fires (auto-completes). None = unlimited.

        Returns:
            {status, trigger_id, title, next_fire_at, cron_expression, fire_once, max_fires}
        """
        try:
            import httpx
            settings = _get_settings()
            daemon_url = settings.daemon_url

            payload: dict = {
                "title": title,
                "prompt_text": prompt_text,
                "create_new_session": create_new_session,
                "timezone": timezone,
                "status": "active",
            }
            if fire_in:
                payload["fire_in"] = fire_in
            elif fire_at:
                payload["fire_at"] = fire_at
            elif cron_expression:
                payload["cron_expression"] = cron_expression

            if target_session_id:
                payload["target_session_id"] = target_session_id
            if target_cwd:
                payload["target_cwd"] = target_cwd
            if skill_id:
                payload["skill_id"] = skill_id
            if project_id:
                payload["project_id"] = project_id
            if max_fires is not None:
                payload["max_fires"] = max_fires

            r = httpx.post(
                f"{daemon_url}/api/cronban/crons",
                json=payload,
                timeout=5.0,
            )
            r.raise_for_status()
            data = r.json()
            return _ok(
                trigger_id=data["id"],
                title=data["title"],
                cron_expression=data.get("cron_expression"),
                next_fire_at=data.get("next_fire_at"),
                fire_once=data.get("fire_once", False),
                max_fires=data.get("max_fires"),
                status=data.get("status"),
            )
        except Exception as e:
            logger.exception("cronban_create failed")
            return _err("CRONBAN_CREATE_FAILED", str(e))

    @mcp.tool()
    def cronban_fire(trigger_id: str) -> dict:
        """Fire a CronTrigger immediately (outside of its schedule).

        Useful for testing or one-off manual executions. Does not affect the
        scheduled next fire time.

        Args:
            trigger_id: UUID of the CronTrigger to fire.

        Returns:
            {status, fired, session_id, log_id}
        """
        try:
            import httpx
            settings = _get_settings()
            daemon_url = settings.daemon_url

            r = httpx.post(
                f"{daemon_url}/api/cronban/crons/{trigger_id}/fire",
                timeout=10.0,
            )
            r.raise_for_status()
            data = r.json()
            return _ok(
                fired=data.get("fired", True),
                session_id=data.get("session_id"),
                log_id=data.get("log_id"),
            )
        except Exception as e:
            logger.exception("cronban_fire failed")
            return _err("CRONBAN_FIRE_FAILED", str(e))

    @mcp.tool()
    def cronban_pause(trigger_id: str) -> dict:
        """Pause an active CronTrigger.

        No more fires will occur while paused. The schedule is retained and
        can be re-activated with ``cronban_resume``.

        Args:
            trigger_id: UUID of the CronTrigger to pause.

        Returns:
            {status, trigger_id, new_status}
        """
        try:
            import httpx
            settings = _get_settings()
            daemon_url = settings.daemon_url

            r = httpx.patch(
                f"{daemon_url}/api/cronban/crons/{trigger_id}",
                json={"status": "paused"},
                timeout=5.0,
            )
            r.raise_for_status()
            data = r.json()
            return _ok(trigger_id=trigger_id, new_status=data.get("status", "paused"))
        except Exception as e:
            logger.exception("cronban_pause failed")
            return _err("CRONBAN_PAUSE_FAILED", str(e))

    @mcp.tool()
    def cronban_resume(trigger_id: str) -> dict:
        """Resume a paused CronTrigger.

        Reactivates the schedule and recalculates the next fire time from now.

        Args:
            trigger_id: UUID of the CronTrigger to resume.

        Returns:
            {status, trigger_id, new_status, next_fire_at}
        """
        try:
            import httpx
            settings = _get_settings()
            daemon_url = settings.daemon_url

            r = httpx.patch(
                f"{daemon_url}/api/cronban/crons/{trigger_id}",
                json={"status": "active"},
                timeout=5.0,
            )
            r.raise_for_status()
            data = r.json()
            return _ok(
                trigger_id=trigger_id,
                new_status=data.get("status", "active"),
                next_fire_at=data.get("next_fire_at"),
            )
        except Exception as e:
            logger.exception("cronban_resume failed")
            return _err("CRONBAN_RESUME_FAILED", str(e))

    @mcp.tool()
    def cronban_delete(trigger_id: str) -> dict:
        """Permanently delete a CronTrigger.

        Removes the trigger and all associated logs. This cannot be undone.
        Use ``cronban_pause`` if you want to keep the schedule but stop firing.

        Args:
            trigger_id: UUID of the CronTrigger to delete.

        Returns:
            {status, trigger_id, deleted}
        """
        try:
            import httpx
            settings = _get_settings()
            daemon_url = settings.daemon_url

            r = httpx.delete(
                f"{daemon_url}/api/cronban/crons/{trigger_id}",
                timeout=5.0,
            )
            r.raise_for_status()
            return _ok(trigger_id=trigger_id, deleted=True)
        except Exception as e:
            logger.exception("cronban_delete failed")
            return _err("CRONBAN_DELETE_FAILED", str(e))
