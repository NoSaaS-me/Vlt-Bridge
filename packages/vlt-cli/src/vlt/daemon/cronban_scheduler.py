"""
Cronban background scheduler — cron tick + gate tick loops.

Runs two independent asyncio loops inside the daemon lifespan:
  - _cron_loop():  every 15 s — fires CronTriggers whose next_fire_at is due
  - _gate_loop():  every 60 s — runs gate-check for PipelineCards whose
                   check interval has elapsed (fallback; primary trigger is
                   _trigger_gate_evaluations() in server.py after each turn)

Both loops use fire helpers from cronban_routes.py to keep logic in one place.

Public API
----------
    start_scheduler()  — call from lifespan startup
    stop_scheduler()   — call from lifespan shutdown
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_running: bool = False
_task: Optional[asyncio.Task] = None

_GATE_CHECK_INTERVAL_MIN: int = 30  # Default gate-check interval for cards


# ---------------------------------------------------------------------------
# Cron tick — fire CronTriggers whose next_fire_at <= now
# ---------------------------------------------------------------------------

async def _cron_tick() -> None:
    """Query active CronTriggers that are due and fire each one."""
    from vlt.db import engine
    from vlt.core.models import CronTrigger
    from sqlmodel import Session, select

    now_iso = datetime.utcnow().isoformat()

    with Session(engine) as db:
        due = db.exec(
            select(CronTrigger).where(
                CronTrigger.status == "active",
                CronTrigger.next_fire_at != None,  # noqa: E711
                CronTrigger.next_fire_at <= now_iso,
            )
        ).all()

    if not due:
        return

    logger.info(f"cron_tick: {len(due)} triggers due to fire")

    from vlt.daemon.cronban_routes import fire_cron_trigger

    for trigger in due:
        try:
            result = await fire_cron_trigger(trigger.id, trigger_type="cron")
            logger.info(
                f"cron_tick: fired {trigger.id!r} ({trigger.title!r}) "
                f"→ fired={result['fired']} session={result['session_id']}"
            )
        except Exception as e:
            logger.error(f"cron_tick: error firing {trigger.id}: {e}")


# ---------------------------------------------------------------------------
# Gate tick — trigger evaluations for PipelineCards past their check interval
# ---------------------------------------------------------------------------

async def _gate_tick() -> None:
    """
    Fallback gate evaluation for active PipelineCards whose current stage has a
    gate and whose last check was more than _GATE_CHECK_INTERVAL_MIN minutes ago.

    The primary trigger for gate evaluations is _trigger_gate_evaluations() in
    server.py (fires after each managed session turn). This tick handles cards
    whose sessions aren't managed or whose agents have gone quiet.
    """
    from vlt.db import engine
    from vlt.core.models import PipelineCard, PipelineStage
    from sqlmodel import Session, select

    now = datetime.utcnow()
    now_iso = now.isoformat()

    with Session(engine) as db:
        active_cards = db.exec(
            select(PipelineCard).where(
                PipelineCard.status == "active",
                PipelineCard.gate_eval_pending == False,  # noqa: E712
            )
        ).all()

    if not active_cards:
        return

    for card in active_cards:
        try:
            # Check if current stage has a gate
            with Session(engine) as db:
                stage = db.get(PipelineStage, card.current_stage_id)
            if not stage or not stage.gate_id:
                continue  # No gate on this stage

            # Check if interval has elapsed
            last_checked = card.gate_last_checked_at
            if last_checked:
                last_dt = datetime.fromisoformat(last_checked.rstrip("Z"))
                elapsed_min = (now - last_dt).total_seconds() / 60.0
                if elapsed_min < _GATE_CHECK_INTERVAL_MIN:
                    continue  # Not yet due

            if not card.target_session_id:
                continue  # No session to evaluate against

            logger.info(
                f"gate_tick: scheduling evaluation for card={card.id!r} "
                f"stage={stage.name!r} gate={stage.gate_id!r}"
            )
            from vlt.daemon.cronban_evaluator import run_gate_evaluation
            asyncio.create_task(run_gate_evaluation(card.id, card.target_session_id))

        except Exception as e:
            logger.error(f"gate_tick: error processing card {card.id}: {e}")


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
