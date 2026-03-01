"""Cronban scheduling tools for the vlt MCP server.

Exposes Cronban operations via MCP so AI agents can schedule, monitor, and
cancel time-based or gate-based tasks without needing the full CLI or web UI.

Tools registered:
    cronban_list      — list entries (optionally filtered by project/type)
    cronban_schedule  — create a new cron-type scheduled entry
    cronban_cancel    — pause an active entry
    cronban_status    — get detailed entry status (last fire + gate result)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def register_cronban_tools(mcp) -> None:
    """Register all Cronban tools onto a FastMCP server instance."""

    @mcp.tool()
    def cronban_list(
        project_id: Optional[str] = None,
        entry_type: Optional[str] = None,
    ) -> dict:
        """List Cronban scheduled entries.

        Returns active (and paused) scheduling entries visible to the agent.
        Eval text is never included — only the ``has_eval`` flag is returned.

        Args:
            project_id: Filter to a specific project slug (optional).
            entry_type: Filter by type: "cron", "gate", or "webhook" (optional).

        Returns:
            {status, entries: [...], total}
        """
        from vlt.mcp import _ok, _err

        try:
            import httpx
            from vlt.config import get_settings
            settings = get_settings()
            daemon_url = settings.daemon_url

            params: dict = {}
            if project_id:
                params["project_id"] = project_id
            if entry_type:
                params["entry_type"] = entry_type

            r = httpx.get(
                f"{daemon_url}/api/cronban/entries",
                params=params,
                timeout=5.0,
            )
            r.raise_for_status()
            entries = r.json()
            return _ok(entries=entries, total=len(entries))
        except Exception as e:
            logger.exception("cronban_list failed")
            return _err("CRONBAN_LIST_FAILED", str(e))

    @mcp.tool()
    def cronban_schedule(
        title: str,
        cron_expression: str,
        prompt_text: str,
        project_id: Optional[str] = None,
        target_cwd: Optional[str] = None,
        create_new_session: bool = True,
        timezone: str = "UTC",
    ) -> dict:
        """Create a new cron-based scheduled entry.

        Schedules a task that fires at the given cron expression and injects
        ``prompt_text`` into a Claude session. The entry is immediately active.

        Args:
            title: Human-readable label for the schedule.
            cron_expression: Standard 5-field cron (e.g. "0 9 * * 1-5" for
                             weekdays at 09:00 UTC).
            prompt_text: The message/instruction to inject when the entry fires.
            project_id: Associate with a project slug (optional).
            target_cwd: Working directory for new sessions (optional).
            create_new_session: If True, spawn a fresh Claude session on each
                                fire. Default True.
            timezone: IANA timezone string, e.g. "America/New_York". Default UTC.

        Returns:
            {status, entry_id, next_fire_at, title}
        """
        from vlt.mcp import _ok, _err

        try:
            import httpx
            from vlt.config import get_settings
            settings = get_settings()
            daemon_url = settings.daemon_url

            payload: dict = {
                "title": title,
                "entry_type": "cron",
                "cron_expression": cron_expression,
                "prompt_text": prompt_text,
                "create_new_session": create_new_session,
                "timezone": timezone,
            }
            if project_id:
                payload["project_id"] = project_id
            if target_cwd:
                payload["target_cwd"] = target_cwd

            r = httpx.post(
                f"{daemon_url}/api/cronban/entries",
                json=payload,
                timeout=5.0,
            )
            r.raise_for_status()
            data = r.json()
            return _ok(
                entry_id=data["id"],
                title=data["title"],
                next_fire_at=data.get("next_fire_at"),
                status=data.get("status"),
            )
        except Exception as e:
            logger.exception("cronban_schedule failed")
            return _err("CRONBAN_SCHEDULE_FAILED", str(e))

    @mcp.tool()
    def cronban_cancel(entry_id: str) -> dict:
        """Pause an active Cronban entry.

        Sets the entry status to "paused". The entry is retained in the DB
        and can be re-activated via the web UI or API. No more fires will
        occur while paused.

        Args:
            entry_id: UUID of the entry to pause.

        Returns:
            {status, entry_id, new_status}
        """
        from vlt.mcp import _ok, _err

        try:
            import httpx
            from vlt.config import get_settings
            settings = get_settings()
            daemon_url = settings.daemon_url

            r = httpx.patch(
                f"{daemon_url}/api/cronban/entries/{entry_id}",
                json={"status": "paused"},
                timeout=5.0,
            )
            r.raise_for_status()
            data = r.json()
            return _ok(entry_id=entry_id, new_status=data.get("status", "paused"))
        except Exception as e:
            logger.exception("cronban_cancel failed")
            return _err("CRONBAN_CANCEL_FAILED", str(e))

    @mcp.tool()
    def cronban_status(entry_id: str) -> dict:
        """Get detailed status of a Cronban entry.

        Returns the full entry metadata including the last fire result and
        (for gate entries) the last gate evaluation outcome. Eval text is
        never included.

        Args:
            entry_id: UUID of the entry.

        Returns:
            {status, entry, last_fire_log, last_gate_log}
        """
        from vlt.mcp import _ok, _err

        try:
            import httpx
            from vlt.config import get_settings
            settings = get_settings()
            daemon_url = settings.daemon_url

            # Fetch entry
            r_entry = httpx.get(
                f"{daemon_url}/api/cronban/entries/{entry_id}",
                timeout=5.0,
            )
            if r_entry.status_code == 404:
                return _err("ENTRY_NOT_FOUND", f"Entry {entry_id} not found")
            r_entry.raise_for_status()
            entry = r_entry.json()

            # Fetch recent fire log (limit 1)
            last_fire = None
            try:
                r_fire = httpx.get(
                    f"{daemon_url}/api/cronban/logs/{entry_id}?limit=1",
                    timeout=5.0,
                )
                if r_fire.status_code == 200:
                    logs = r_fire.json()
                    last_fire = logs[0] if logs else None
            except Exception:
                pass

            # Fetch recent gate log (limit 1, for gate entries)
            last_gate = None
            if entry.get("entry_type") == "gate":
                try:
                    r_gate = httpx.get(
                        f"{daemon_url}/api/cronban/gate-logs/{entry_id}?limit=1",
                        timeout=5.0,
                    )
                    if r_gate.status_code == 200:
                        glogs = r_gate.json()
                        last_gate = glogs[0] if glogs else None
                except Exception:
                    pass

            return _ok(
                entry=entry,
                last_fire_log=last_fire,
                last_gate_log=last_gate,
            )
        except Exception as e:
            logger.exception("cronban_status failed")
            return _err("CRONBAN_STATUS_FAILED", str(e))
