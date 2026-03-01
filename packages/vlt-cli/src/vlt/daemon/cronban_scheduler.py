"""
Cronban background scheduler — cron tick + gate tick loops.

Runs two independent asyncio loops inside the daemon lifespan:
  - _cron_loop():  every 15 s — fires CRON entries whose next_fire_at is due
  - _gate_loop():  every 60 s — fires gate-check for GATE entries whose
                   check interval has elapsed

Both loops delegate actual execution to the existing API endpoints via
self-calls to localhost:8765, keeping fire logic in one place and
avoiding circular imports.

Public API
----------
    start_scheduler()  — call from lifespan startup
    stop_scheduler()   — call from lifespan shutdown
    fire_entry(entry_id, trigger_type, prompt_override)
                       — async helper used by cronban_routes.py webhook handler
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_running: bool = False
_task: Optional[asyncio.Task] = None

# Port the daemon listens on — kept in sync with server.py default
_DAEMON_PORT: int = 8765


def _daemon_url(path: str) -> str:
    return f"http://127.0.0.1:{_DAEMON_PORT}{path}"


# ---------------------------------------------------------------------------
# Core fire helper — used by webhook handler in cronban_routes.py too
# ---------------------------------------------------------------------------

async def fire_entry(
    entry_id: str,
    trigger_type: str = "cron",
    prompt_override: Optional[str] = None,
) -> dict:
    """
    Fire a CronbanEntry by resolving its prompt and injecting it into a
    session (or spawning a new one).

    This function contains the actual fire logic so that both the scheduler
    and the webhook route can share it without duplicating code.

    Steps:
      1. Load entry from DB.
      2. Resolve the effective prompt (skill > inline > override).
      3. Route to the right session mechanism:
         a. target_session_id set  → inject via /api/sessions/live/{id}/inject
         b. create_new_session     → POST /api/sessions/spawn
         c. fallback               → log error, no session available
      4. Write CronbanFireLog.
      5. Update entry last_fired_at, fire_count, and recompute next_fire_at (cron).

    Returns dict: {fired, session_id, log_id}
    """
    from vlt.db import engine
    from vlt.core.models import CronbanEntry, CronbanFireLog, CronbanSkill
    from sqlmodel import Session

    now_iso = datetime.utcnow().isoformat()
    log_id = str(uuid.uuid4())
    result_status = "success"
    error_msg: Optional[str] = None
    used_session_id: Optional[str] = None

    with Session(engine) as db:
        entry = db.get(CronbanEntry, entry_id)
        if not entry:
            logger.warning(f"fire_entry: entry {entry_id} not found")
            return {"fired": False, "session_id": None, "log_id": None}

        if entry.status != "active":
            logger.info(f"fire_entry: entry {entry_id} is {entry.status}, skipping")
            return {"fired": False, "session_id": None, "log_id": None}

        # ── Resolve prompt ────────────────────────────────────────────────
        prompt_text: Optional[str] = prompt_override

        if not prompt_text and entry.skill_id:
            skill = db.get(CronbanSkill, entry.skill_id)
            if skill:
                prompt_text = skill.prompt_markdown

        if not prompt_text:
            prompt_text = entry.prompt_text

        if not prompt_text:
            logger.warning(f"fire_entry: entry {entry_id} has no prompt — skipping")
            return {"fired": False, "session_id": None, "log_id": None}

        target_sid = entry.target_session_id
        cwd = entry.target_cwd
        create_new = entry.create_new_session

        # ── Dispatch ──────────────────────────────────────────────────────
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if target_sid:
                    # Try injecting into an existing managed session first
                    try:
                        from vlt.daemon.server import (
                            _managed_sessions,
                            _injection_queues,
                            _managed_message_worker,
                        )
                        mgd = _managed_sessions.get(target_sid)
                        if mgd:
                            mgd["queue"].append({"text": prompt_text})
                            if not mgd["processing"]:
                                asyncio.create_task(_managed_message_worker(target_sid))
                            used_session_id = target_sid
                        else:
                            # Session not in managed dict — try injection queue
                            _injection_queues.setdefault(target_sid, []).append(prompt_text)
                            used_session_id = target_sid
                    except ImportError:
                        # Fall back to HTTP injection endpoint
                        resp = await client.post(
                            _daemon_url(f"/api/sessions/{target_sid}/inject"),
                            json={"message": prompt_text},
                        )
                        if resp.status_code == 200:
                            used_session_id = target_sid
                        else:
                            raise RuntimeError(f"inject returned {resp.status_code}: {resp.text}")

                elif create_new and cwd:
                    # Spawn a fresh managed session
                    resp = await client.post(
                        _daemon_url("/api/sessions/spawn"),
                        json={"cwd": cwd, "prompt": prompt_text},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    used_session_id = data.get("session_id")

                else:
                    error_msg = "No target_session_id or create_new_session+cwd; cannot fire"
                    result_status = "error"
                    logger.warning(f"fire_entry {entry_id}: {error_msg}")

        except Exception as exc:
            result_status = "error"
            error_msg = str(exc)
            logger.error(f"fire_entry {entry_id} dispatch error: {exc}")

        # ── Write fire log ────────────────────────────────────────────────
        db.add(CronbanFireLog(
            id=log_id,
            entry_id=entry_id,
            fired_at=now_iso,
            trigger_type=trigger_type,
            target_session_id=used_session_id,
            prompt_sent=prompt_text[:2000] if prompt_text else None,
            result=result_status,
            error_message=error_msg,
        ))

        # ── Update entry stats ────────────────────────────────────────────
        entry.fire_count = (entry.fire_count or 0) + 1
        entry.last_fired_at = now_iso
        entry.updated_at = now_iso

        # Recompute next_fire_at for recurring cron entries
        if entry.entry_type == "cron" and (entry.cron_expression or entry.rrule_str):
            try:
                _next = _compute_next_fire(entry.cron_expression, entry.rrule_str, entry.timezone)
                entry.next_fire_at = _next
            except Exception as e:
                logger.warning(f"Could not recompute next_fire_at for {entry_id}: {e}")

        db.add(entry)
        db.commit()

    return {
        "fired": result_status == "success",
        "session_id": used_session_id,
        "log_id": log_id,
    }


def _compute_next_fire(cron_expr: Optional[str], rrule_str: Optional[str], tz: str = "UTC") -> Optional[str]:
    """Compute next fire time from cron or rrule. Returns ISO UTC string."""
    try:
        now = datetime.now(timezone.utc)
        if rrule_str:
            from dateutil.rrule import rrulestr
            rule = rrulestr(rrule_str, dtstart=now, ignoretz=False)
            nxt = rule.after(now)
            return nxt.astimezone(timezone.utc).isoformat() if nxt else None
        if cron_expr:
            from croniter import croniter
            it = croniter(cron_expr, now)
            return it.get_next(datetime).isoformat()
    except Exception as e:
        logger.warning(f"Could not compute next_fire_at: {e}")
    return None


# ---------------------------------------------------------------------------
# Cron tick — fire entries whose next_fire_at <= now
# ---------------------------------------------------------------------------

async def _cron_tick() -> None:
    """Query active CRON entries that are due and fire each one."""
    from vlt.db import engine
    from vlt.core.models import CronbanEntry
    from sqlmodel import Session, select

    now_iso = datetime.utcnow().isoformat()

    with Session(engine) as db:
        due = db.exec(
            select(CronbanEntry).where(
                CronbanEntry.entry_type == "cron",
                CronbanEntry.status == "active",
                CronbanEntry.next_fire_at != None,  # noqa: E711
                CronbanEntry.next_fire_at <= now_iso,
            )
        ).all()

    if not due:
        return

    logger.info(f"cron_tick: {len(due)} entries due to fire")
    for entry in due:
        try:
            result = await fire_entry(entry.id, trigger_type="cron")
            logger.info(
                f"cron_tick: fired {entry.id!r} ({entry.title!r}) "
                f"→ fired={result['fired']} session={result['session_id']}"
            )
        except Exception as e:
            logger.error(f"cron_tick: error firing {entry.id}: {e}")


# ---------------------------------------------------------------------------
# Gate tick — trigger gate-check for GATE entries past their interval
# ---------------------------------------------------------------------------

async def _gate_tick() -> None:
    """Query active GATE entries whose check interval has elapsed and run gate-check."""
    from vlt.db import engine
    from vlt.core.models import CronbanEntry
    from sqlmodel import Session, select

    now = datetime.utcnow()
    now_iso = now.isoformat()

    with Session(engine) as db:
        gate_entries = db.exec(
            select(CronbanEntry).where(
                CronbanEntry.entry_type == "gate",
                CronbanEntry.status == "active",
            )
        ).all()

    if not gate_entries:
        return

    for entry in gate_entries:
        try:
            interval_min = entry.gate_check_interval_minutes or 30
            last_checked = entry.gate_last_checked_at

            if last_checked:
                # Parse and compare timestamps
                last_dt = datetime.fromisoformat(last_checked.rstrip("Z"))
                elapsed_min = (now - last_dt).total_seconds() / 60.0
                if elapsed_min < interval_min:
                    continue  # Not yet due

            # Call the gate-check endpoint via self-call to avoid duplication
            logger.info(f"gate_tick: running gate-check for {entry.id!r} ({entry.title!r})")
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    _daemon_url(f"/api/cronban/entries/{entry.id}/gate-check")
                )
                data = resp.json()
                logger.info(
                    f"gate_tick: {entry.id!r} phase={data.get('phase')} met={data.get('met')}"
                )
        except Exception as e:
            logger.error(f"gate_tick: error checking {entry.id}: {e}")


# ---------------------------------------------------------------------------
# Loop wrappers
# ---------------------------------------------------------------------------

async def _cron_loop() -> None:
    """Run cron_tick every 15 seconds while _running."""
    while _running:
        try:
            await _cron_tick()
        except Exception as e:
            logger.error(f"cron_tick unhandled error: {e}")
        await asyncio.sleep(15)


async def _gate_loop() -> None:
    """Run gate_tick every 60 seconds while _running."""
    while _running:
        try:
            await _gate_tick()
        except Exception as e:
            logger.error(f"gate_tick unhandled error: {e}")
        await asyncio.sleep(60)


async def _scheduler_loop() -> None:
    """Run both loops concurrently until _running is False."""
    global _running
    _running = True
    cron_task = asyncio.create_task(_cron_loop())
    gate_task = asyncio.create_task(_gate_loop())
    try:
        await asyncio.gather(cron_task, gate_task)
    except asyncio.CancelledError:
        cron_task.cancel()
        gate_task.cancel()
        logger.info("Cronban scheduler loops cancelled")


# ---------------------------------------------------------------------------
# Public lifecycle API
# ---------------------------------------------------------------------------

def start_scheduler() -> None:
    """Start the scheduler as a background asyncio task. Call from lifespan startup."""
    global _task, _running
    _running = True
    _task = asyncio.ensure_future(_scheduler_loop())
    logger.info("Cronban scheduler started (cron=15s, gate=60s)")


def stop_scheduler() -> None:
    """Stop the scheduler gracefully. Call from lifespan shutdown."""
    global _running, _task
    _running = False
    if _task and not _task.done():
        _task.cancel()
        logger.info("Cronban scheduler stopped")
    _task = None
