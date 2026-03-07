"""
Cronban API Routes — Pipeline-based scheduling and workflow automation.

Architecture:
  Skills   — reusable prompt templates (what Claude sees)
  Gates    — reusable eval criteria (what helper Claude checks, never seen by agent)
  Pipelines — ordered stage definitions; cards move through them
  Cards     — work items progressing through a pipeline
  Crons     — time-based triggers that create cards or fire skills directly
  Webhooks  — HTTP triggers that create cards or fire skills on external signal
  Settings  — gate verifier model configuration
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from sqlmodel import Session, select

from vlt.db import engine
from vlt.core.models import (
    AgentSession,
    CronbanFireLog,
    CronbanGate,
    CronbanGateLog,
    CronbanSettings,
    CronbanSkill,
    CronTrigger,
    Pipeline,
    PipelineCard,
    PipelineStage,
    WebhookListener,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cronban", tags=["cronban"])

_now = lambda: datetime.utcnow().isoformat()  # noqa: E731


# =============================================================================
# Serialisers
# =============================================================================

def _skill_to_dict(s: CronbanSkill) -> dict:
    return {
        "id": s.id,
        "project_id": s.project_id,
        "name": s.name,
        "description": s.description,
        "prompt_markdown": s.prompt_markdown,
        "tags": json.loads(s.tags_json) if s.tags_json else [],
        "created_at": s.created_at,
        "updated_at": s.updated_at,
    }


def _gate_to_dict(g: CronbanGate) -> dict:
    return {
        "id": g.id,
        "project_id": g.project_id,
        "name": g.name,
        "description": g.description,
        "prompt_markdown": g.prompt_markdown,
        "tags": json.loads(g.tags_json) if g.tags_json else [],
        "created_at": g.created_at,
        "updated_at": g.updated_at,
    }


def _stage_to_dict(s: PipelineStage) -> dict:
    return {
        "id": s.id,
        "pipeline_id": s.pipeline_id,
        "name": s.name,
        "stage_order": s.stage_order,
        "skill_id": s.skill_id,
        "prompt_text": s.prompt_text,
        "gate_id": s.gate_id,
        "eval_model": s.eval_model,
        "auto_advance": bool(s.auto_advance),
        "is_terminal": bool(s.is_terminal),
        "created_at": s.created_at,
        "updated_at": s.updated_at,
    }


def _pipeline_to_dict(p: Pipeline, stages: list[PipelineStage]) -> dict:
    return {
        "id": p.id,
        "project_id": p.project_id,
        "name": p.name,
        "description": p.description,
        "stages": [_stage_to_dict(s) for s in sorted(stages, key=lambda x: x.stage_order)],
        "created_at": p.created_at,
        "updated_at": p.updated_at,
    }


def _card_to_dict(c: PipelineCard) -> dict:
    return {
        "id": c.id,
        "pipeline_id": c.pipeline_id,
        "project_id": c.project_id,
        "title": c.title,
        "color": c.color,
        "current_stage_id": c.current_stage_id,
        "status": c.status,
        "skill_id": c.skill_id,
        "has_prompt": bool(c.prompt_text),
        "gate_id": c.gate_id,
        "target_session_id": c.target_session_id,
        "target_cwd": c.target_cwd,
        "use_helper_session": bool(c.use_helper_session),
        "gate_eval_pending": bool(c.gate_eval_pending),
        "gate_last_result": json.loads(c.gate_last_result) if c.gate_last_result else None,
        "gate_last_checked_at": c.gate_last_checked_at,
        "gate_consecutive_not_met": c.gate_consecutive_not_met,
        "fire_count": c.fire_count,
        "last_fired_at": c.last_fired_at,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
    }


def _cron_to_dict(c: CronTrigger) -> dict:
    return {
        "id": c.id,
        "project_id": c.project_id,
        "title": c.title,
        "status": c.status,
        "color": c.color,
        "pipeline_id": c.pipeline_id,
        "skill_id": c.skill_id,
        "has_prompt": bool(c.prompt_text),
        "gate_id": c.gate_id,
        "cron_expression": c.cron_expression,
        "rrule_str": c.rrule_str,
        "next_fire_at": c.next_fire_at,
        "timezone": c.timezone,
        "target_session_id": c.target_session_id,
        "target_cwd": c.target_cwd,
        "create_new_session": bool(c.create_new_session),
        "fire_once": bool(c.fire_once),
        "fire_count": c.fire_count,
        "last_fired_at": c.last_fired_at,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
    }


def _webhook_to_dict(w: WebhookListener) -> dict:
    return {
        "id": w.id,
        "project_id": w.project_id,
        "name": w.name,
        "status": w.status,
        "pipeline_id": w.pipeline_id,
        "skill_id": w.skill_id,
        "prompt_text": w.prompt_text,
        "has_prompt": bool(w.prompt_text),
        "has_secret": bool(w.webhook_secret),
        "target_session_id": w.target_session_id,
        "target_cwd": w.target_cwd,
        "create_new_session": bool(w.create_new_session),
        "connector_name": w.connector_name,
        "pattern_filter_json": w.pattern_filter_json,
        "backend_user_id": w.backend_user_id,
        "fire_count": w.fire_count,
        "last_fired_at": w.last_fired_at,
        "created_at": w.created_at,
        "updated_at": w.updated_at,
    }


# =============================================================================
# Fire helpers
# =============================================================================

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


def _parse_fire_in(s: str) -> "timedelta":
    """Parse 'hh:mm:ss', 'hh:mm', or 'mm:ss' offset into a timedelta."""
    from datetime import timedelta
    parts = [p.strip() for p in s.strip().split(':')]
    if len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, sec = int(parts[0]), int(parts[1]), 0
    else:
        h, m, sec = 0, 0, int(parts[0])
    return timedelta(hours=h, minutes=m, seconds=sec)


async def _dispatch_to_session(
    prompt: str,
    target_session_id: Optional[str],
    target_cwd: Optional[str],
    create_new: bool,
) -> Optional[str]:
    """
    Send a prompt to a session.

    Routing:
      0. Relay session active → inject via _session_inject_queues (PTY stdin).
      1. SDK session already running → write to stdin directly.
      2. Session exists in DB → spawn SDK session with --resume.
      3. create_new=True → spawn fresh session via /api/sessions/spawn.
    """
    try:
        import sys as _sys
        from pathlib import Path as _Path

        # The daemon runs as __main__ (python -m vlt.daemon.server), so
        # vlt.daemon.server is NOT in sys.modules — importing it would create
        # a second module object with a separate _sdk_sessions dict.
        # Always access live state through __main__ to share the same dicts.
        _srv = _sys.modules['__main__']
        _sdk_sessions = _srv._sdk_sessions
        _spawn_sdk_session = _srv._spawn_sdk_session

        if target_session_id:
            # 0. Relay session — inject text into the PTY queue
            _relay_queues = getattr(_srv, '_session_inject_queues', {})
            if target_session_id in _relay_queues:
                await _relay_queues[target_session_id].put((prompt + "\n").encode())
                logger.info(f"_dispatch: relay inject for {target_session_id}")
                return target_session_id

            # 1. Already running — write to stdin directly
            sdk = _sdk_sessions.get(target_session_id)
            if sdk and sdk["proc"].returncode is None:
                user_msg = json.dumps({
                    "type": "user",
                    "message": {"role": "user", "content": prompt},
                    "session_id": "default",
                }) + "\n"
                sdk["proc"].stdin.write(user_msg.encode())
                asyncio.create_task(sdk["proc"].stdin.drain())
                logger.info(f"_dispatch: SDK write for {target_session_id}")
                return target_session_id

            # 2. Session not running — look up DB and spawn with --resume (persistent)
            from vlt.db import engine as _engine  # lazy: avoids stale module-cache issues
            with Session(_engine) as db:
                sess = db.get(AgentSession, target_session_id)

            if sess and sess.cwd:
                cwd_key = str(_Path(sess.cwd).resolve())
                ok = await _spawn_sdk_session(target_session_id, cwd_key, is_new=False)
                if ok:
                    sdk = _sdk_sessions[target_session_id]
                    user_msg = json.dumps({
                        "type": "user",
                        "message": {"role": "user", "content": prompt},
                        "session_id": "default",
                    }) + "\n"
                    sdk["proc"].stdin.write(user_msg.encode())
                    await sdk["proc"].stdin.drain()
                    logger.info(f"_dispatch: SDK resume for {target_session_id} (status={sess.status})")
                    return target_session_id

        if create_new and target_cwd:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "http://127.0.0.1:8765/api/sessions/spawn",
                    json={"cwd": target_cwd, "prompt": prompt},
                )
                resp.raise_for_status()
                return resp.json().get("session_id")

    except Exception as exc:
        logger.error(f"_dispatch_to_session error: {exc}")

    return None


async def _resolve_prompt(
    skill_id: Optional[str],
    prompt_text: Optional[str],
    append_message: Optional[str] = None,
) -> Optional[str]:
    """Resolve effective prompt from skill or inline text, with optional append."""
    text: Optional[str] = None
    if skill_id:
        with Session(engine) as db:
            skill = db.get(CronbanSkill, skill_id)
            if skill:
                text = skill.prompt_markdown
    if not text:
        text = prompt_text
    if text and append_message:
        text = f"{text}\n\n---\n\n{append_message.strip()}"
    return text


async def fire_cron_trigger(trigger_id: str, trigger_type: str = "cron") -> dict:
    """
    Fire a CronTrigger. Either creates a PipelineCard (pipeline_id set)
    or fires the skill/prompt directly at a session.
    """
    log_id = str(uuid.uuid4())
    now_iso = _now()
    result_status = "success"
    error_msg: Optional[str] = None
    used_session_id: Optional[str] = None

    with Session(engine) as db:
        trigger = db.get(CronTrigger, trigger_id)
        if not trigger:
            return {"fired": False, "session_id": None, "log_id": None}
        if trigger.status != "active":
            return {"fired": False, "session_id": None, "log_id": None}

        project_id = trigger.project_id
        pipeline_id = trigger.pipeline_id
        skill_id = trigger.skill_id
        prompt_text = trigger.prompt_text
        target_sid = trigger.target_session_id
        target_cwd = trigger.target_cwd
        create_new = bool(trigger.create_new_session)
        cron_expr = trigger.cron_expression
        rrule = trigger.rrule_str
        tz = trigger.timezone or "UTC"
        title = trigger.title

    try:
        if pipeline_id:
            # Create a pipeline card at the first non-terminal stage
            with Session(engine) as db:
                first_stage = db.exec(
                    select(PipelineStage).where(
                        PipelineStage.pipeline_id == pipeline_id,
                        PipelineStage.is_terminal == False,  # noqa: E712
                    ).order_by(PipelineStage.stage_order)
                ).first()

            if not first_stage:
                raise RuntimeError(f"Pipeline {pipeline_id} has no non-terminal stages")

            card_id = str(uuid.uuid4())
            with Session(engine) as db:
                card = PipelineCard(
                    id=card_id,
                    pipeline_id=pipeline_id,
                    project_id=project_id,
                    title=title,
                    current_stage_id=first_stage.id,
                    target_session_id=target_sid,
                    target_cwd=target_cwd,
                    status="active",
                )
                db.add(card)
                db.commit()

            # Fire the stage's skill if set
            stage_prompt = await _resolve_prompt(first_stage.skill_id, first_stage.prompt_text)
            if stage_prompt:
                used_session_id = await _dispatch_to_session(
                    stage_prompt, target_sid, target_cwd, create_new
                )
                if used_session_id:
                    with Session(engine) as db:
                        c = db.get(PipelineCard, card_id)
                        if c:
                            c.target_session_id = used_session_id
                            c.fire_count = 1
                            c.last_fired_at = now_iso
                            db.add(c)
                            db.commit()
        else:
            # Standalone skill fire
            prompt = await _resolve_prompt(skill_id, prompt_text)
            if not prompt:
                raise RuntimeError("No prompt resolved for standalone fire")
            used_session_id = await _dispatch_to_session(prompt, target_sid, target_cwd, create_new)
            if not used_session_id:
                raise RuntimeError("No session available")

    except Exception as exc:
        result_status = "error"
        error_msg = str(exc)
        logger.error(f"fire_cron_trigger {trigger_id}: {exc}")

    with Session(engine) as db:
        db.add(CronbanFireLog(
            id=log_id,
            entry_id=trigger_id,
            fired_at=now_iso,
            trigger_type=trigger_type,
            target_session_id=used_session_id,
            prompt_sent=None,
            result=result_status,
            error_message=error_msg,
        ))
        trigger = db.get(CronTrigger, trigger_id)
        if trigger:
            trigger.fire_count = (trigger.fire_count or 0) + 1
            trigger.last_fired_at = now_iso
            trigger.updated_at = now_iso
            if getattr(trigger, 'fire_once', False):
                trigger.next_fire_at = None
                trigger.status = "completed"
            elif trigger.cron_expression or trigger.rrule_str:
                try:
                    trigger.next_fire_at = _compute_next_fire(trigger.cron_expression, trigger.rrule_str, trigger.timezone)
                except Exception:
                    pass
            db.add(trigger)
        db.commit()

    return {"fired": result_status == "success", "session_id": used_session_id, "log_id": log_id}


async def fire_webhook_listener(
    listener_id: str,
    append_message: Optional[str] = None,
) -> dict:
    """Fire a WebhookListener — creates a PipelineCard or fires a skill directly."""
    log_id = str(uuid.uuid4())
    now_iso = _now()
    result_status = "success"
    error_msg: Optional[str] = None
    used_session_id: Optional[str] = None

    with Session(engine) as db:
        listener = db.get(WebhookListener, listener_id)
        if not listener:
            return {"fired": False, "session_id": None, "log_id": None}
        if listener.status != "active":
            return {"fired": False, "session_id": None, "log_id": None}

        project_id = listener.project_id
        pipeline_id = listener.pipeline_id
        skill_id = listener.skill_id
        prompt_text = listener.prompt_text
        target_sid = listener.target_session_id
        target_cwd = listener.target_cwd
        create_new = bool(listener.create_new_session)
        name = listener.name

    try:
        if pipeline_id:
            with Session(engine) as db:
                first_stage = db.exec(
                    select(PipelineStage).where(
                        PipelineStage.pipeline_id == pipeline_id,
                        PipelineStage.is_terminal == False,  # noqa: E712
                    ).order_by(PipelineStage.stage_order)
                ).first()

            if not first_stage:
                raise RuntimeError(f"Pipeline {pipeline_id} has no non-terminal stages")

            card_id = str(uuid.uuid4())
            with Session(engine) as db:
                card = PipelineCard(
                    id=card_id,
                    pipeline_id=pipeline_id,
                    project_id=project_id,
                    title=f"{name} — {now_iso[:10]}",
                    current_stage_id=first_stage.id,
                    target_session_id=target_sid,
                    target_cwd=target_cwd,
                    status="active",
                )
                db.add(card)
                db.commit()

            stage_prompt = await _resolve_prompt(
                first_stage.skill_id, first_stage.prompt_text, append_message
            )
            if stage_prompt:
                used_session_id = await _dispatch_to_session(
                    stage_prompt, target_sid, target_cwd, create_new
                )
        else:
            prompt = await _resolve_prompt(skill_id, prompt_text, append_message)
            if not prompt:
                raise RuntimeError("No prompt resolved")
            used_session_id = await _dispatch_to_session(prompt, target_sid, target_cwd, create_new)
            if not used_session_id:
                raise RuntimeError("No session available")

    except Exception as exc:
        result_status = "error"
        error_msg = str(exc)
        logger.error(f"fire_webhook_listener {listener_id}: {exc}")

    with Session(engine) as db:
        db.add(CronbanFireLog(
            id=log_id,
            entry_id=listener_id,
            fired_at=now_iso,
            trigger_type="webhook",
            target_session_id=used_session_id,
            prompt_sent=None,
            result=result_status,
            error_message=error_msg,
        ))
        listener = db.get(WebhookListener, listener_id)
        if listener:
            listener.fire_count = (listener.fire_count or 0) + 1
            listener.last_fired_at = now_iso
            listener.updated_at = now_iso
            db.add(listener)
        db.commit()

    return {"fired": result_status == "success", "session_id": used_session_id, "log_id": log_id}


# =============================================================================
# Skills
# =============================================================================

@router.get("/skills")
async def list_skills(project_id: Optional[str] = None):
    with Session(engine) as db:
        q = select(CronbanSkill)
        if project_id:
            q = q.where(
                (CronbanSkill.project_id == project_id) | (CronbanSkill.project_id == None)  # noqa: E711
            )
        return [_skill_to_dict(s) for s in db.exec(q).all()]


@router.post("/skills")
async def create_skill(request: Request):
    body = await request.json()
    with Session(engine) as db:
        skill = CronbanSkill(
            id=str(uuid.uuid4()),
            project_id=body.get("project_id"),
            name=body.get("name", "New Skill"),
            description=body.get("description"),
            prompt_markdown=body.get("prompt_markdown", ""),
            tags_json=json.dumps(body.get("tags", [])),
        )
        db.add(skill)
        db.commit()
        db.refresh(skill)
        return _skill_to_dict(skill)


@router.get("/skills/{skill_id}")
async def get_skill(skill_id: str):
    with Session(engine) as db:
        s = db.get(CronbanSkill, skill_id)
        if not s:
            raise HTTPException(404, "Skill not found")
        return _skill_to_dict(s)


@router.patch("/skills/{skill_id}")
async def update_skill(skill_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        s = db.get(CronbanSkill, skill_id)
        if not s:
            raise HTTPException(404, "Skill not found")
        if "name" in body:
            s.name = body["name"]
        if "description" in body:
            s.description = body["description"]
        if "prompt_markdown" in body:
            s.prompt_markdown = body["prompt_markdown"]
        if "tags" in body:
            s.tags_json = json.dumps(body["tags"])
        s.updated_at = _now()
        db.add(s)
        db.commit()
        db.refresh(s)
        return _skill_to_dict(s)


@router.delete("/skills/{skill_id}")
async def delete_skill(skill_id: str):
    with Session(engine) as db:
        s = db.get(CronbanSkill, skill_id)
        if not s:
            raise HTTPException(404, "Skill not found")
        db.delete(s)
        db.commit()
    return {"ok": True}


# =============================================================================
# Gates
# =============================================================================

@router.get("/gates")
async def list_gates(project_id: Optional[str] = None):
    with Session(engine) as db:
        q = select(CronbanGate)
        if project_id:
            q = q.where(
                (CronbanGate.project_id == project_id) | (CronbanGate.project_id == None)  # noqa: E711
            )
        return [_gate_to_dict(g) for g in db.exec(q).all()]


@router.post("/gates")
async def create_gate(request: Request):
    body = await request.json()
    with Session(engine) as db:
        gate = CronbanGate(
            id=str(uuid.uuid4()),
            project_id=body.get("project_id"),
            name=body.get("name", "New Gate"),
            description=body.get("description"),
            prompt_markdown=body.get("prompt_markdown", ""),
            tags_json=json.dumps(body.get("tags", [])),
        )
        db.add(gate)
        db.commit()
        db.refresh(gate)
        return _gate_to_dict(gate)


@router.get("/gates/{gate_id}")
async def get_gate(gate_id: str):
    with Session(engine) as db:
        g = db.get(CronbanGate, gate_id)
        if not g:
            raise HTTPException(404, "Gate not found")
        return _gate_to_dict(g)


@router.patch("/gates/{gate_id}")
async def update_gate(gate_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        g = db.get(CronbanGate, gate_id)
        if not g:
            raise HTTPException(404, "Gate not found")
        if "name" in body:
            g.name = body["name"]
        if "description" in body:
            g.description = body["description"]
        if "prompt_markdown" in body:
            g.prompt_markdown = body["prompt_markdown"]
        if "tags" in body:
            g.tags_json = json.dumps(body["tags"])
        g.updated_at = _now()
        db.add(g)
        db.commit()
        db.refresh(g)
        return _gate_to_dict(g)


@router.delete("/gates/{gate_id}")
async def delete_gate(gate_id: str):
    with Session(engine) as db:
        g = db.get(CronbanGate, gate_id)
        if not g:
            raise HTTPException(404, "Gate not found")
        db.delete(g)
        db.commit()
    return {"ok": True}


# =============================================================================
# Pipelines
# =============================================================================

@router.get("/pipelines")
async def list_pipelines(project_id: Optional[str] = None):
    with Session(engine) as db:
        q = select(Pipeline)
        if project_id:
            q = q.where(
                (Pipeline.project_id == project_id) | (Pipeline.project_id == None)  # noqa: E711
            )
        pipelines = db.exec(q).all()
        result = []
        for p in pipelines:
            stages = db.exec(select(PipelineStage).where(PipelineStage.pipeline_id == p.id)).all()
            result.append(_pipeline_to_dict(p, stages))
        return result


@router.post("/pipelines")
async def create_pipeline(request: Request):
    body = await request.json()
    with Session(engine) as db:
        p = Pipeline(
            id=str(uuid.uuid4()),
            project_id=body.get("project_id"),
            name=body.get("name", "New Pipeline"),
            description=body.get("description"),
        )
        db.add(p)
        db.commit()
        db.refresh(p)
        return _pipeline_to_dict(p, [])


@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    with Session(engine) as db:
        p = db.get(Pipeline, pipeline_id)
        if not p:
            raise HTTPException(404, "Pipeline not found")
        stages = db.exec(select(PipelineStage).where(PipelineStage.pipeline_id == pipeline_id)).all()
        return _pipeline_to_dict(p, stages)


@router.patch("/pipelines/{pipeline_id}")
async def update_pipeline(pipeline_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        p = db.get(Pipeline, pipeline_id)
        if not p:
            raise HTTPException(404, "Pipeline not found")
        if "name" in body:
            p.name = body["name"]
        if "description" in body:
            p.description = body["description"]
        p.updated_at = _now()
        db.add(p)
        db.commit()
        db.refresh(p)
        stages = db.exec(select(PipelineStage).where(PipelineStage.pipeline_id == pipeline_id)).all()
        return _pipeline_to_dict(p, stages)


@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    with Session(engine) as db:
        p = db.get(Pipeline, pipeline_id)
        if not p:
            raise HTTPException(404, "Pipeline not found")
        # Delete cards and stages first
        cards = db.exec(select(PipelineCard).where(PipelineCard.pipeline_id == pipeline_id)).all()
        for c in cards:
            db.delete(c)
        stages = db.exec(select(PipelineStage).where(PipelineStage.pipeline_id == pipeline_id)).all()
        for s in stages:
            db.delete(s)
        db.delete(p)
        db.commit()
    return {"ok": True}


# =============================================================================
# Pipeline Stages
# =============================================================================

@router.post("/pipelines/{pipeline_id}/stages")
async def add_stage(pipeline_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        p = db.get(Pipeline, pipeline_id)
        if not p:
            raise HTTPException(404, "Pipeline not found")
        stage = PipelineStage(
            id=str(uuid.uuid4()),
            pipeline_id=pipeline_id,
            name=body.get("name", "New Stage"),
            stage_order=body.get("stage_order", 0),
            skill_id=body.get("skill_id"),
            prompt_text=body.get("prompt_text"),
            gate_id=body.get("gate_id"),
            eval_model=body.get("eval_model", "haiku"),
            auto_advance=bool(body.get("auto_advance", True)),
            is_terminal=bool(body.get("is_terminal", False)),
        )
        db.add(stage)
        p.updated_at = _now()
        db.add(p)
        db.commit()
        db.refresh(stage)
        return _stage_to_dict(stage)


@router.patch("/stages/{stage_id}")
async def update_stage(stage_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        s = db.get(PipelineStage, stage_id)
        if not s:
            raise HTTPException(404, "Stage not found")
        for field in ("name", "stage_order", "skill_id", "prompt_text", "gate_id", "eval_model"):
            if field in body:
                setattr(s, field, body[field])
        if "auto_advance" in body:
            s.auto_advance = bool(body["auto_advance"])
        if "is_terminal" in body:
            s.is_terminal = bool(body["is_terminal"])
        s.updated_at = _now()
        db.add(s)
        db.commit()
        db.refresh(s)
        return _stage_to_dict(s)


@router.delete("/stages/{stage_id}")
async def delete_stage(stage_id: str):
    with Session(engine) as db:
        s = db.get(PipelineStage, stage_id)
        if not s:
            raise HTTPException(404, "Stage not found")
        db.delete(s)
        db.commit()
    return {"ok": True}


@router.post("/pipelines/{pipeline_id}/stages/reorder")
async def reorder_stages(pipeline_id: str, request: Request):
    """Body: [{id, stage_order}]"""
    body = await request.json()
    with Session(engine) as db:
        for item in body:
            s = db.get(PipelineStage, item["id"])
            if s and s.pipeline_id == pipeline_id:
                s.stage_order = item["stage_order"]
                s.updated_at = _now()
                db.add(s)
        db.commit()
    return {"ok": True}


# =============================================================================
# Pipeline Cards
# =============================================================================

@router.get("/cards")
async def list_cards(project_id: Optional[str] = None, pipeline_id: Optional[str] = None, status: Optional[str] = None):
    with Session(engine) as db:
        q = select(PipelineCard)
        if project_id:
            q = q.where(PipelineCard.project_id == project_id)
        if pipeline_id:
            q = q.where(PipelineCard.pipeline_id == pipeline_id)
        if status:
            q = q.where(PipelineCard.status == status)
        return [_card_to_dict(c) for c in db.exec(q).all()]


@router.post("/cards")
async def create_card(request: Request):
    body = await request.json()
    pipeline_id = body.get("pipeline_id")
    if not pipeline_id:
        raise HTTPException(400, "pipeline_id is required")

    with Session(engine) as db:
        p = db.get(Pipeline, pipeline_id)
        if not p:
            raise HTTPException(404, "Pipeline not found")

        # Use provided stage or find first non-terminal stage
        stage_id = body.get("stage_id")
        if not stage_id:
            first = db.exec(
                select(PipelineStage).where(
                    PipelineStage.pipeline_id == pipeline_id,
                    PipelineStage.is_terminal == False,  # noqa: E712
                ).order_by(PipelineStage.stage_order)
            ).first()
            if not first:
                raise HTTPException(400, "Pipeline has no non-terminal stages")
            stage_id = first.id

        card = PipelineCard(
            id=str(uuid.uuid4()),
            pipeline_id=pipeline_id,
            project_id=body.get("project_id"),
            title=body.get("title", "New Card"),
            color=body.get("color"),
            current_stage_id=stage_id,
            status="active",
            skill_id=body.get("skill_id"),
            prompt_text=body.get("prompt_text"),
            gate_id=body.get("gate_id"),
            target_session_id=body.get("target_session_id"),
            target_cwd=body.get("target_cwd"),
            use_helper_session=bool(body.get("use_helper_session", False)),
        )
        db.add(card)
        db.commit()
        db.refresh(card)
        return _card_to_dict(card)


@router.get("/cards/{card_id}")
async def get_card(card_id: str):
    with Session(engine) as db:
        c = db.get(PipelineCard, card_id)
        if not c:
            raise HTTPException(404, "Card not found")
        return _card_to_dict(c)


@router.patch("/cards/{card_id}")
async def update_card(card_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        c = db.get(PipelineCard, card_id)
        if not c:
            raise HTTPException(404, "Card not found")
        for field in ("title", "color", "status", "skill_id", "prompt_text", "gate_id",
                      "target_session_id", "target_cwd", "use_helper_session"):
            if field in body:
                setattr(c, field, body[field])
        c.updated_at = _now()
        db.add(c)
        db.commit()
        db.refresh(c)
        return _card_to_dict(c)


@router.delete("/cards/{card_id}")
async def delete_card(card_id: str):
    with Session(engine) as db:
        c = db.get(PipelineCard, card_id)
        if not c:
            raise HTTPException(404, "Card not found")
        db.delete(c)
        db.commit()
    return {"ok": True}


@router.post("/cards/{card_id}/fire")
async def fire_card(card_id: str):
    """
    Manually fire a PipelineCard — resolves its prompt (card-level overrides stage)
    and dispatches it to the card's target session (or spawns one if needed).
    """
    log_id = str(uuid.uuid4())
    now_iso = _now()
    result_status = "success"
    error_msg: Optional[str] = None
    used_session_id: Optional[str] = None

    with Session(engine) as db:
        card = db.get(PipelineCard, card_id)
        if not card:
            raise HTTPException(404, "Card not found")

        stage = db.get(PipelineStage, card.current_stage_id) if card.current_stage_id else None

        # Card-level overrides take priority over stage values
        effective_skill_id = card.skill_id or (stage.skill_id if stage else None)
        effective_prompt_text = card.prompt_text or (stage.prompt_text if stage else None)
        target_sid = card.target_session_id
        target_cwd = card.target_cwd
        project_id = card.project_id or "default"
        use_helper = bool(card.use_helper_session)
        title = card.title

    try:
        prompt = await _resolve_prompt(effective_skill_id, effective_prompt_text)
        if not prompt:
            raise RuntimeError("No prompt resolved for card fire")

        if use_helper:
            # Route through the helper session pool (LIFO, creates if all busy)
            from vlt.daemon.cronban_evaluator import _get_available_helper, _dispatch_helper_and_wait
            import os
            cwd = target_cwd or os.path.expanduser("~")
            helper_id, is_new = _get_available_helper(project_id, cwd, "sonnet")
            success = await _dispatch_helper_and_wait(
                helper_id=helper_id, is_new=is_new, cwd=cwd,
                model_id="claude-sonnet-4-6", prompt=prompt,
            )
            used_session_id = helper_id if success else None
            if not success:
                raise RuntimeError("Helper session dispatch failed")
        else:
            used_session_id = await _dispatch_to_session(prompt, target_sid, target_cwd, bool(target_cwd))
            if not used_session_id:
                raise RuntimeError("No session available")

        with Session(engine) as db:
            c = db.get(PipelineCard, card_id)
            if c:
                c.fire_count = (c.fire_count or 0) + 1
                c.last_fired_at = now_iso
                c.target_session_id = used_session_id
                c.updated_at = now_iso
                db.add(c)
                db.commit()

    except Exception as exc:
        result_status = "error"
        error_msg = str(exc)
        logger.error(f"fire_card {card_id}: {exc}")

    with Session(engine) as db:
        db.add(CronbanFireLog(
            id=log_id,
            entry_id=card_id,
            fired_at=now_iso,
            trigger_type="card_manual",
            target_session_id=used_session_id,
            prompt_sent=None,
            result=result_status,
            error_message=error_msg,
        ))
        db.commit()

    return {"fired": result_status == "success", "session_id": used_session_id, "log_id": log_id}


@router.post("/cards/{card_id}/advance")
async def advance_card(card_id: str, request: Request):
    """
    Advance a card to the next stage (by stage_order) or a specific stage.
    Resets gate evaluation state. Sets status=completed if new stage is terminal.
    Body (optional): {stage_id: str}
    """
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    with Session(engine) as db:
        c = db.get(PipelineCard, card_id)
        if not c:
            raise HTTPException(404, "Card not found")

        target_stage_id = body.get("stage_id")

        if target_stage_id:
            new_stage = db.get(PipelineStage, target_stage_id)
            if not new_stage or new_stage.pipeline_id != c.pipeline_id:
                raise HTTPException(400, "Invalid target stage")
        else:
            # Find the next stage by order
            current = db.get(PipelineStage, c.current_stage_id)
            if not current:
                raise HTTPException(400, "Current stage not found")
            new_stage = db.exec(
                select(PipelineStage).where(
                    PipelineStage.pipeline_id == c.pipeline_id,
                    PipelineStage.stage_order > current.stage_order,
                ).order_by(PipelineStage.stage_order)
            ).first()
            if not new_stage:
                raise HTTPException(400, "No next stage available")

        # Move card
        c.current_stage_id = new_stage.id
        c.gate_eval_pending = False
        c.gate_last_result = None
        c.gate_last_checked_at = None
        c.gate_consecutive_not_met = 0
        if new_stage.is_terminal:
            c.status = "completed"
        c.updated_at = _now()
        db.add(c)
        db.commit()
        db.refresh(c)
        return _card_to_dict(c)


# =============================================================================
# Cron Triggers
# =============================================================================

@router.get("/crons")
async def list_crons(project_id: Optional[str] = None):
    with Session(engine) as db:
        q = select(CronTrigger)
        if project_id:
            q = q.where(
                (CronTrigger.project_id == project_id) | (CronTrigger.project_id == None)  # noqa: E711
            )
        return [_cron_to_dict(c) for c in db.exec(q).all()]


@router.post("/crons")
async def create_cron(request: Request):
    body = await request.json()
    cron_expr = body.get("cron_expression")
    rrule = body.get("rrule_str")
    tz = body.get("timezone", "UTC")
    fire_once = False
    next_fire = None

    fire_at = body.get("fire_at")
    fire_in = body.get("fire_in")
    if fire_at:
        from dateutil import parser as dtparser
        dt = dtparser.parse(fire_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        next_fire = dt.astimezone(timezone.utc).isoformat()
        fire_once = True
    elif fire_in:
        next_fire = (datetime.now(timezone.utc) + _parse_fire_in(fire_in)).isoformat()
        fire_once = True
    elif cron_expr or rrule:
        next_fire = _compute_next_fire(cron_expr, rrule, tz)

    with Session(engine) as db:
        t = CronTrigger(
            id=str(uuid.uuid4()),
            project_id=body.get("project_id"),
            title=body.get("title", "New Cron"),
            status=body.get("status", "active"),
            color=body.get("color"),
            pipeline_id=body.get("pipeline_id"),
            skill_id=body.get("skill_id"),
            prompt_text=body.get("prompt_text"),
            gate_id=body.get("gate_id"),
            cron_expression=cron_expr,
            rrule_str=rrule,
            next_fire_at=next_fire,
            timezone=tz,
            target_session_id=body.get("target_session_id"),
            target_cwd=body.get("target_cwd"),
            create_new_session=bool(body.get("create_new_session", False)),
            fire_once=fire_once,
        )
        db.add(t)
        db.commit()
        db.refresh(t)
        return _cron_to_dict(t)


@router.get("/crons/{cron_id}")
async def get_cron(cron_id: str):
    with Session(engine) as db:
        t = db.get(CronTrigger, cron_id)
        if not t:
            raise HTTPException(404, "Cron trigger not found")
        return _cron_to_dict(t)


@router.patch("/crons/{cron_id}")
async def update_cron(cron_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        t = db.get(CronTrigger, cron_id)
        if not t:
            raise HTTPException(404, "Cron trigger not found")
        for field in ("title", "status", "color", "pipeline_id", "skill_id", "prompt_text",
                      "gate_id", "cron_expression", "rrule_str", "timezone",
                      "target_session_id", "target_cwd"):
            if field in body:
                setattr(t, field, body[field])
        if "create_new_session" in body:
            t.create_new_session = bool(body["create_new_session"])
        if "fire_once" in body:
            t.fire_once = bool(body["fire_once"])
        # One-shot: fire_at or fire_in override cron recalculation
        fire_at = body.get("fire_at")
        fire_in = body.get("fire_in")
        if fire_at:
            from dateutil import parser as dtparser
            dt = dtparser.parse(fire_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            t.next_fire_at = dt.astimezone(timezone.utc).isoformat()
            t.fire_once = True
        elif fire_in:
            t.next_fire_at = (datetime.now(timezone.utc) + _parse_fire_in(fire_in)).isoformat()
            t.fire_once = True
        elif "cron_expression" in body or "rrule_str" in body:
            t.next_fire_at = _compute_next_fire(t.cron_expression, t.rrule_str, t.timezone or "UTC")
            t.fire_once = False
        t.updated_at = _now()
        db.add(t)
        db.commit()
        db.refresh(t)
        return _cron_to_dict(t)


@router.delete("/crons/{cron_id}")
async def delete_cron(cron_id: str):
    with Session(engine) as db:
        t = db.get(CronTrigger, cron_id)
        if not t:
            raise HTTPException(404, "Cron trigger not found")
        db.delete(t)
        db.commit()
    return {"ok": True}


@router.post("/crons/{cron_id}/fire")
async def manual_fire_cron(cron_id: str):
    """Manually trigger a cron now regardless of schedule."""
    result = await fire_cron_trigger(cron_id, trigger_type="manual")
    if not result["fired"]:
        raise HTTPException(500, "Fire failed")
    return result


# =============================================================================
# Webhook Listeners
# =============================================================================

@router.get("/webhooks")
async def list_webhooks(project_id: Optional[str] = None):
    with Session(engine) as db:
        q = select(WebhookListener)
        if project_id:
            q = q.where(
                (WebhookListener.project_id == project_id) | (WebhookListener.project_id == None)  # noqa: E711
            )
        return [_webhook_to_dict(w) for w in db.exec(q).all()]


@router.post("/webhooks")
async def create_webhook(request: Request):
    body = await request.json()
    with Session(engine) as db:
        w = WebhookListener(
            id=str(uuid.uuid4()),
            project_id=body.get("project_id"),
            name=body.get("name", "New Webhook"),
            status=body.get("status", "active"),
            pipeline_id=body.get("pipeline_id"),
            skill_id=body.get("skill_id"),
            prompt_text=body.get("prompt_text"),
            webhook_secret=body.get("webhook_secret"),
            target_session_id=body.get("target_session_id"),
            target_cwd=body.get("target_cwd"),
            create_new_session=bool(body.get("create_new_session", False)),
            connector_name=body.get("connector_name"),
            pattern_filter_json=body.get("pattern_filter_json"),
            backend_user_id=body.get("backend_user_id"),
        )
        db.add(w)
        db.commit()
        db.refresh(w)
        return _webhook_to_dict(w)


@router.get("/webhooks/{webhook_id}")
async def get_webhook(webhook_id: str):
    with Session(engine) as db:
        w = db.get(WebhookListener, webhook_id)
        if not w:
            raise HTTPException(404, "Webhook listener not found")
        return _webhook_to_dict(w)


@router.patch("/webhooks/{webhook_id}")
async def update_webhook(webhook_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        w = db.get(WebhookListener, webhook_id)
        if not w:
            raise HTTPException(404, "Webhook listener not found")
        for field in ("name", "status", "pipeline_id", "skill_id", "prompt_text",
                      "webhook_secret", "target_session_id", "target_cwd",
                      "connector_name", "pattern_filter_json", "backend_user_id"):
            if field in body:
                setattr(w, field, body[field])
        if "create_new_session" in body:
            w.create_new_session = bool(body["create_new_session"])
        w.updated_at = _now()
        db.add(w)
        db.commit()
        db.refresh(w)
        return _webhook_to_dict(w)


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(webhook_id: str):
    with Session(engine) as db:
        w = db.get(WebhookListener, webhook_id)
        if not w:
            raise HTTPException(404, "Webhook listener not found")
        db.delete(w)
        db.commit()
    return {"ok": True}


@router.post("/webhooks/{webhook_id}/fire")
async def fire_webhook(webhook_id: str, request: Request):
    """
    External HTTP trigger. Validates HMAC if webhook_secret is set.
    Body (optional): {append_message: str, signature: str}
    """
    body_bytes = await request.body()
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    with Session(engine) as db:
        w = db.get(WebhookListener, webhook_id)
        if not w:
            raise HTTPException(404, "Webhook listener not found")
        if w.status != "active":
            raise HTTPException(403, "Webhook listener is paused")

        if w.webhook_secret:
            sig = request.headers.get("X-Vlt-Signature") or body.get("signature", "")
            expected = "sha256=" + hmac.new(
                w.webhook_secret.encode(), body_bytes, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(sig, expected):
                raise HTTPException(401, "Invalid signature")

    append_message = body.get("append_message")
    result = await fire_webhook_listener(webhook_id, append_message=append_message)
    return result


# =============================================================================
# Logs
# =============================================================================

@router.get("/logs/{source_id}")
async def get_fire_logs(source_id: str, limit: int = 20):
    with Session(engine) as db:
        logs = db.exec(
            select(CronbanFireLog)
            .where(CronbanFireLog.entry_id == source_id)
            .order_by(CronbanFireLog.fired_at.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": l.id,
                "source_id": l.entry_id,
                "fired_at": l.fired_at,
                "trigger_type": l.trigger_type,
                "target_session_id": l.target_session_id,
                "result": l.result,
                "error_message": l.error_message,
            }
            for l in logs
        ]


@router.get("/gate-logs/{card_id}")
async def get_gate_logs(card_id: str, limit: int = 20):
    with Session(engine) as db:
        logs = db.exec(
            select(CronbanGateLog)
            .where(CronbanGateLog.entry_id == card_id)
            .order_by(CronbanGateLog.checked_at.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": l.id,
                "card_id": l.entry_id,
                "checked_at": l.checked_at,
                "gate_met": l.gate_met,
                "reasoning": l.reasoning,
                "model_used": l.model_used,
            }
            for l in logs
        ]


# =============================================================================
# Settings
# =============================================================================

@router.get("/settings/gate-model")
async def get_gate_settings():
    with Session(engine) as db:
        s = db.get(CronbanSettings, "default")
        if not s:
            s = CronbanSettings(id="default")
            db.add(s)
            db.commit()
            db.refresh(s)
        # claude_code never needs an API key — it's always "configured"
        has_key = s.gate_provider == "claude_code" or bool(s.gate_api_key)
        return {
            "provider": s.gate_provider,
            "model": s.gate_model,
            "has_api_key": has_key,
            "base_url": s.gate_base_url,
        }


@router.put("/settings/gate-model")
async def save_gate_settings(request: Request):
    body = await request.json()
    with Session(engine) as db:
        s = db.get(CronbanSettings, "default")
        if not s:
            s = CronbanSettings(id="default")
        if "provider" in body:
            s.gate_provider = body["provider"]
        if "model" in body:
            s.gate_model = body["model"]
        if "api_key" in body and body["api_key"]:
            s.gate_api_key = body["api_key"]
        if "base_url" in body:
            s.gate_base_url = body["base_url"] or None
        s.updated_at = _now()
        db.add(s)
        db.commit()
    return {"ok": True}


@router.post("/settings/gate-model/test")
async def test_gate_model():
    """Quick connectivity test — runs a minimal eval prompt against the configured provider."""
    with Session(engine) as db:
        s = db.get(CronbanSettings, "default")
        provider = s.gate_provider if s else "claude_code"
        model = s.gate_model if s else "sonnet"
        api_key = s.gate_api_key if s else None
        base_url = s.gate_base_url if s else None

    if provider == "claude_code":
        # Claude Code helper sessions are always available — no remote ping needed
        return {
            "ok": True,
            "result": {
                "met": True,
                "reasoning": "Claude Code helper sessions are managed locally — no API key required.",
            },
        }

    # External provider — try a minimal OpenAI-compatible chat completion
    _OPENAI_COMPAT_BASE = {
        "openrouter": "https://openrouter.ai/api/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    }
    if not api_key:
        return {"ok": False, "result": {"met": False, "reasoning": "No API key configured."}}

    endpoint_base = base_url if base_url else _OPENAI_COMPAT_BASE.get(provider, "")
    if not endpoint_base:
        return {"ok": False, "result": {"met": False, "reasoning": f"Unknown provider '{provider}' and no base URL set."}}

    url = endpoint_base.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                url,
                json={"model": model, "messages": [{"role": "user", "content": "Reply with exactly: GATE_RESULT: PASS"}], "max_tokens": 20},
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
        if resp.status_code == 200:
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "result": {"met": True, "reasoning": f"Provider responded OK. Model reply: {content[:100]!r}"}}
        else:
            return {"ok": False, "result": {"met": False, "reasoning": f"Provider returned HTTP {resp.status_code}: {resp.text[:200]}"}}
    except Exception as e:
        return {"ok": False, "result": {"met": False, "reasoning": str(e)[:300]}}
