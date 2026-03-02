"""
Cronban API Routes — Cron + Kanban + Calendar scheduling.

Three-part job structure:
  1. Schedule  — when to fire (cron expression / RRULE / one-shot)
  2. Skill     — prompt Claude sees (skill_id or inline prompt_text)
  3. Eval      — hidden criterion Claude NEVER sees; verifier model only

Endpoints:
  Skills  — CRUD for reusable prompt templates
  Entries — CRUD for scheduled jobs / kanban cards / webhook triggers
  Columns — Kanban column management
  Gate    — Verifier tool calls (transcript read, file check, shell verify)
  Settings — Gate model provider configuration
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from sqlmodel import Session, select

from vlt.db import engine
from vlt.core.models import (
    AgentSession,
    CronbanEntry,
    CronbanFireLog,
    CronbanGate,
    CronbanGateLog,
    CronbanSettings,
    CronbanSkill,
    KanbanColumn,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cronban", tags=["cronban"])

_now = lambda: datetime.utcnow().isoformat()


# =============================================================================
# Helpers
# =============================================================================

def _get_or_create_settings(db: Session) -> CronbanSettings:
    s = db.get(CronbanSettings, "default")
    if not s:
        s = CronbanSettings(id="default")
        db.add(s)
        db.commit()
        db.refresh(s)
    return s


def _entry_to_dict(e: CronbanEntry) -> dict:
    return {
        "id": e.id,
        "project_id": e.project_id,
        "title": e.title,
        "entry_type": e.entry_type,
        "status": e.status,
        "color": e.color,
        "skill_id": e.skill_id,
        "prompt_text": e.prompt_text,
        # eval_text and gate prompt intentionally omitted — hidden from Claude
        "gate_id": e.gate_id,
        "has_eval": bool(e.gate_id or e.eval_text),  # True if any eval is configured
        "eval_model": e.eval_model or "haiku",
        "gate_eval_pending": bool(e.gate_eval_pending),
        "target_session_id": e.target_session_id,
        "target_cwd": e.target_cwd,
        "create_new_session": e.create_new_session,
        "cron_expression": e.cron_expression,
        "rrule_str": e.rrule_str,
        "next_fire_at": e.next_fire_at,
        "timezone": e.timezone,
        "kanban_column_id": e.kanban_column_id,
        "kanban_order": e.kanban_order,
        "gate_check_interval_minutes": e.gate_check_interval_minutes,
        "gate_last_checked_at": e.gate_last_checked_at,
        "gate_last_result": json.loads(e.gate_last_result) if e.gate_last_result else None,
        "gate_consecutive_not_met": e.gate_consecutive_not_met,
        "gate_question_injected_at": e.gate_question_injected_at,
        "has_webhook_secret": bool(e.webhook_secret),
        "fire_count": e.fire_count,
        "last_fired_at": e.last_fired_at,
        "created_at": e.created_at,
        "updated_at": e.updated_at,
    }


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


def _column_to_dict(c: KanbanColumn) -> dict:
    return {
        "id": c.id,
        "project_id": c.project_id,
        "name": c.name,
        "col_order": c.col_order,
        "color": c.color,
        "is_terminal": c.is_terminal,
        "auto_graduate": c.auto_graduate,
        "graduation_column_id": c.graduation_column_id,
        "created_at": c.created_at,
    }


def _compute_next_fire(cron_expr: Optional[str], rrule_str: Optional[str], tz: str = "UTC") -> Optional[str]:
    """Compute next fire time from cron or rrule. Returns ISO UTC string."""
    try:
        now = datetime.now(timezone.utc)
        if rrule_str:
            from dateutil.rrule import rrulestr
            from dateutil import tz as dtz
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


# =============================================================================
# Skills CRUD
# =============================================================================

@router.get("/skills")
async def list_skills(project_id: Optional[str] = None):
    with Session(engine) as db:
        q = select(CronbanSkill)
        if project_id:
            q = q.where(
                (CronbanSkill.project_id == project_id) | (CronbanSkill.project_id == None)
            )
        skills = db.exec(q).all()
        return [_skill_to_dict(s) for s in skills]


@router.post("/skills")
async def create_skill(request: Request):
    body = await request.json()
    if not body.get("name") or not body.get("prompt_markdown"):
        raise HTTPException(400, "name and prompt_markdown required")
    with Session(engine) as db:
        skill = CronbanSkill(
            id=str(uuid.uuid4()),
            project_id=body.get("project_id"),
            name=body["name"],
            description=body.get("description"),
            prompt_markdown=body["prompt_markdown"],
            tags_json=json.dumps(body.get("tags", [])),
        )
        db.add(skill)
        db.commit()
        db.refresh(skill)
        return _skill_to_dict(skill)


@router.get("/skills/{skill_id}")
async def get_skill(skill_id: str):
    with Session(engine) as db:
        skill = db.get(CronbanSkill, skill_id)
        if not skill:
            raise HTTPException(404, "Skill not found")
        return _skill_to_dict(skill)


@router.patch("/skills/{skill_id}")
async def update_skill(skill_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        skill = db.get(CronbanSkill, skill_id)
        if not skill:
            raise HTTPException(404, "Skill not found")
        if "name" in body:
            skill.name = body["name"]
        if "description" in body:
            skill.description = body["description"]
        if "prompt_markdown" in body:
            skill.prompt_markdown = body["prompt_markdown"]
        if "tags" in body:
            skill.tags_json = json.dumps(body["tags"])
        skill.updated_at = _now()
        db.add(skill)
        db.commit()
        db.refresh(skill)
        return _skill_to_dict(skill)


@router.delete("/skills/{skill_id}")
async def delete_skill(skill_id: str):
    with Session(engine) as db:
        skill = db.get(CronbanSkill, skill_id)
        if not skill:
            raise HTTPException(404, "Skill not found")
        db.delete(skill)
        db.commit()
    return {"ok": True}


# =============================================================================
# Gates CRUD — reusable helper-Claude evaluation definitions
# =============================================================================

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


@router.get("/gates")
async def list_gates(project_id: Optional[str] = None):
    with Session(engine) as db:
        q = select(CronbanGate)
        if project_id:
            q = q.where(
                (CronbanGate.project_id == project_id) | (CronbanGate.project_id == None)  # noqa: E711
            )
        gates = db.exec(q).all()
        return [_gate_to_dict(g) for g in gates]


@router.post("/gates")
async def create_gate(request: Request):
    body = await request.json()
    if not body.get("name") or not body.get("prompt_markdown"):
        raise HTTPException(400, "name and prompt_markdown required")
    with Session(engine) as db:
        gate = CronbanGate(
            id=str(uuid.uuid4()),
            project_id=body.get("project_id"),
            name=body["name"],
            description=body.get("description"),
            prompt_markdown=body["prompt_markdown"],
            tags_json=json.dumps(body.get("tags", [])),
        )
        db.add(gate)
        db.commit()
        db.refresh(gate)
        return _gate_to_dict(gate)


@router.get("/gates/{gate_id}")
async def get_gate(gate_id: str):
    with Session(engine) as db:
        gate = db.get(CronbanGate, gate_id)
        if not gate:
            raise HTTPException(404, "Gate not found")
        return _gate_to_dict(gate)


@router.patch("/gates/{gate_id}")
async def update_gate(gate_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        gate = db.get(CronbanGate, gate_id)
        if not gate:
            raise HTTPException(404, "Gate not found")
        if "name" in body:
            gate.name = body["name"]
        if "description" in body:
            gate.description = body["description"]
        if "prompt_markdown" in body:
            gate.prompt_markdown = body["prompt_markdown"]
        if "tags" in body:
            gate.tags_json = json.dumps(body["tags"])
        gate.updated_at = _now()
        db.add(gate)
        db.commit()
        db.refresh(gate)
        return _gate_to_dict(gate)


@router.delete("/gates/{gate_id}")
async def delete_gate(gate_id: str):
    with Session(engine) as db:
        gate = db.get(CronbanGate, gate_id)
        if not gate:
            raise HTTPException(404, "Gate not found")
        db.delete(gate)
        db.commit()
    return {"ok": True}


# =============================================================================
# Kanban Columns CRUD
# =============================================================================

@router.get("/columns")
async def list_columns(project_id: Optional[str] = None):
    with Session(engine) as db:
        q = select(KanbanColumn)
        if project_id:
            q = q.where(KanbanColumn.project_id == project_id)
        cols = db.exec(q.order_by(KanbanColumn.col_order)).all()
        return [_column_to_dict(c) for c in cols]


@router.post("/columns")
async def create_column(request: Request):
    body = await request.json()
    if not body.get("project_id") or not body.get("name"):
        raise HTTPException(400, "project_id and name required")
    with Session(engine) as db:
        col = KanbanColumn(
            id=str(uuid.uuid4()),
            project_id=body["project_id"],
            name=body["name"],
            col_order=body.get("col_order", 0),
            color=body.get("color"),
            is_terminal=body.get("is_terminal", False),
            auto_graduate=body.get("auto_graduate", True),
            graduation_column_id=body.get("graduation_column_id"),
        )
        db.add(col)
        db.commit()
        db.refresh(col)
        return _column_to_dict(col)


@router.patch("/columns/{col_id}")
async def update_column(col_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        col = db.get(KanbanColumn, col_id)
        if not col:
            raise HTTPException(404, "Column not found")
        for field in ("name", "col_order", "color", "is_terminal", "auto_graduate", "graduation_column_id"):
            if field in body:
                setattr(col, field, body[field])
        db.add(col)
        db.commit()
        db.refresh(col)
        return _column_to_dict(col)


@router.delete("/columns/{col_id}")
async def delete_column(col_id: str, move_to: Optional[str] = None):
    with Session(engine) as db:
        col = db.get(KanbanColumn, col_id)
        if not col:
            raise HTTPException(404, "Column not found")
        # Move cards to another column if specified
        if move_to:
            cards = db.exec(
                select(CronbanEntry).where(CronbanEntry.kanban_column_id == col_id)
            ).all()
            for card in cards:
                card.kanban_column_id = move_to
                db.add(card)
        db.delete(col)
        db.commit()
    return {"ok": True}


# =============================================================================
# Entries CRUD
# =============================================================================

@router.get("/entries")
async def list_entries(
    project_id: Optional[str] = None,
    entry_type: Optional[str] = None,
    status: Optional[str] = None,
):
    with Session(engine) as db:
        q = select(CronbanEntry)
        if project_id:
            q = q.where(CronbanEntry.project_id == project_id)
        if entry_type:
            q = q.where(CronbanEntry.entry_type == entry_type)
        if status:
            q = q.where(CronbanEntry.status == status)
        entries = db.exec(q.order_by(CronbanEntry.created_at.desc())).all()
        return [_entry_to_dict(e) for e in entries]


@router.post("/entries")
async def create_entry(request: Request):
    body = await request.json()
    if not body.get("title"):
        raise HTTPException(400, "title required")

    entry_type = body.get("entry_type", "cron")
    if entry_type not in ("cron", "gate", "webhook"):
        raise HTTPException(400, "entry_type must be cron|gate|webhook")

    # Compute next_fire_at for cron entries
    next_fire = None
    if entry_type == "cron":
        next_fire = _compute_next_fire(
            body.get("cron_expression"),
            body.get("rrule_str"),
            body.get("timezone", "UTC"),
        )

    with Session(engine) as db:
        entry = CronbanEntry(
            id=str(uuid.uuid4()),
            project_id=body.get("project_id"),
            title=body["title"],
            entry_type=entry_type,
            status=body.get("status", "active"),
            color=body.get("color"),
            skill_id=body.get("skill_id"),
            prompt_text=body.get("prompt_text"),
            gate_id=body.get("gate_id"),               # Preferred: gate library reference
            eval_text=body.get("eval_text"),           # Legacy inline eval — gate_id takes priority
            target_session_id=body.get("target_session_id"),
            target_cwd=body.get("target_cwd"),
            create_new_session=body.get("create_new_session", False),
            cron_expression=body.get("cron_expression"),
            rrule_str=body.get("rrule_str"),
            next_fire_at=next_fire,
            timezone=body.get("timezone", "UTC"),
            kanban_column_id=body.get("kanban_column_id"),
            kanban_order=body.get("kanban_order", 0),
            gate_check_interval_minutes=body.get("gate_check_interval_minutes", 30),
            webhook_secret=body.get("webhook_secret"),
            eval_model=body.get("eval_model", "haiku"),
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return _entry_to_dict(entry)


@router.get("/entries/{entry_id}")
async def get_entry(entry_id: str):
    with Session(engine) as db:
        entry = db.get(CronbanEntry, entry_id)
        if not entry:
            raise HTTPException(404, "Entry not found")
        return _entry_to_dict(entry)


@router.patch("/entries/{entry_id}")
async def update_entry(entry_id: str, request: Request):
    body = await request.json()
    with Session(engine) as db:
        entry = db.get(CronbanEntry, entry_id)
        if not entry:
            raise HTTPException(404, "Entry not found")
        updatable = (
            "title", "status", "color", "skill_id", "prompt_text", "eval_text",
            "gate_id", "eval_model",
            "target_session_id", "target_cwd", "create_new_session",
            "cron_expression", "rrule_str", "timezone",
            "kanban_column_id", "kanban_order", "gate_check_interval_minutes",
            "webhook_secret",
        )
        for field in updatable:
            if field in body:
                setattr(entry, field, body[field])
        # Recompute next_fire_at if schedule fields changed
        if any(f in body for f in ("cron_expression", "rrule_str", "timezone")):
            entry.next_fire_at = _compute_next_fire(
                entry.cron_expression, entry.rrule_str, entry.timezone
            )
        entry.updated_at = _now()
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return _entry_to_dict(entry)


@router.delete("/entries/{entry_id}")
async def delete_entry(entry_id: str):
    with Session(engine) as db:
        entry = db.get(CronbanEntry, entry_id)
        if not entry:
            raise HTTPException(404, "Entry not found")
        db.delete(entry)
        db.commit()
    return {"ok": True}


@router.post("/entries/{entry_id}/move")
async def move_kanban_card(entry_id: str, request: Request):
    """Move a kanban card to a new column. If card has eval_text, triggers gate check first."""
    body = await request.json()
    new_col_id = body.get("column_id")
    force = body.get("force", False)

    with Session(engine) as db:
        entry = db.get(CronbanEntry, entry_id)
        if not entry:
            raise HTTPException(404, "Entry not found")

        if not force and entry.eval_text:
            # Client should call /gate-check first; this is a guard
            return {
                "moved": False,
                "requires_gate_check": True,
                "message": "This card has an eval. Run gate-check first, or pass force=true to override.",
            }

        entry.kanban_column_id = new_col_id
        entry.updated_at = _now()
        db.add(entry)
        db.commit()
    return {"moved": True, "column_id": new_col_id}


# =============================================================================
# Gate Check — verifier tool calls
# =============================================================================

@router.post("/entries/{entry_id}/gate-check")
async def gate_check(entry_id: str):
    """
    Run the two-phase gate check:
    Phase 1: Inject generic summary-request into Claude session.
    Phase 2: Read response, call verifier model with hidden eval_text.

    Returns {phase, met, reasoning} — clients poll this to see progress.
    """
    with Session(engine) as db:
        entry = db.get(CronbanEntry, entry_id)
        if not entry:
            raise HTTPException(404, "Entry not found")
        if not entry.eval_text:
            return {"phase": "skipped", "met": True, "reasoning": "No eval defined — auto-pass"}

        # Phase 1: inject generic question if not yet done
        if not entry.gate_question_injected_at:
            sid = entry.target_session_id
            if sid:
                sess = db.get(AgentSession, sid)
                if sess and sess.status != "dead":
                    # Import injection queue (daemon in-process)
                    try:
                        from vlt.daemon.server import _injection_queues, _managed_sessions, _managed_message_worker
                        import asyncio
                        question = (
                            "[Cronban gate check] Please briefly summarize what you have "
                            "accomplished on this task so far. What is done, and what (if anything) remains?"
                        )
                        _injection_queues.setdefault(sid, []).append(question)
                        mgd = _managed_sessions.get(sid)
                        if mgd:
                            mgd["queue"].append({"text": question})
                            if not mgd["processing"]:
                                asyncio.create_task(_managed_message_worker(sid))
                        entry.gate_question_injected_at = _now()
                        db.add(entry)
                        db.commit()
                        return {"phase": "1_injected", "met": None,
                                "reasoning": "Gate question sent to Claude. Poll again to get verdict."}
                    except Exception as e:
                        logger.warning(f"Gate inject failed: {e}")
                        return {"phase": "error", "met": False, "reasoning": str(e)}
            return {"phase": "no_session", "met": False, "reasoning": "No live session to query"}

        # Phase 2: read response and run verifier
        injected_at = datetime.fromisoformat(entry.gate_question_injected_at.rstrip("Z"))
        response_text = _read_transcript_after(entry.target_session_id, injected_at)
        if not response_text:
            return {"phase": "2_waiting", "met": None,
                    "reasoning": "Waiting for Claude to respond to gate question."}

        # Call verifier model
        start = datetime.utcnow()
        result = await _run_verifier(entry.eval_text, response_text, db)
        duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)

        # Log result
        db.add(CronbanGateLog(
            id=str(uuid.uuid4()),
            entry_id=entry_id,
            gate_met=result["met"],
            reasoning=result.get("reasoning", ""),
            model_used=result.get("model_used", "unknown"),
            tokens_used=result.get("tokens_used", 0),
            duration_ms=duration_ms,
        ))

        entry.gate_last_checked_at = _now()
        entry.gate_last_result = json.dumps({"met": result["met"], "reasoning": result.get("reasoning", "")})
        entry.gate_question_injected_at = None  # Reset for next cycle

        if result["met"]:
            entry.gate_consecutive_not_met = 0
        else:
            entry.gate_consecutive_not_met += 1
            if entry.gate_consecutive_not_met >= 10:
                entry.status = "paused"
            else:
                backoff = min(30 * (2 ** (entry.gate_consecutive_not_met // 2)), 1440)
                entry.gate_check_interval_minutes = backoff

        db.add(entry)
        db.commit()
        return {
            "phase": "2_complete",
            "met": result["met"],
            "reasoning": result.get("reasoning", ""),
        }


def _read_transcript_after(session_id: Optional[str], after_ts: datetime) -> Optional[str]:
    """Read first assistant message after after_ts from the JSONL transcript."""
    if not session_id:
        return None
    with Session(engine) as db:
        sess = db.get(AgentSession, session_id)
    if not sess or not sess.transcript_path:
        return None
    path = Path(sess.transcript_path)
    if not path.exists():
        return None
    texts = []
    try:
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "assistant":
                continue
            ts_str = entry.get("timestamp", "")
            if not ts_str:
                continue
            ts = datetime.fromisoformat(ts_str.rstrip("Z"))
            if ts <= after_ts:
                continue
            msg = entry.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
    except Exception as e:
        logger.warning(f"Transcript read error: {e}")
    return "\n\n".join(texts) if texts else None


async def _run_verifier(eval_text: str, agent_response: str, db: Session) -> dict:
    """Call the configured gate verifier model to classify agent_response against eval_text."""
    settings = _get_or_create_settings(db)

    prompt = (
        f"You are a completion verifier. A Claude agent was given a task and asked to summarize its progress.\n\n"
        f"Hidden success criterion (the agent never saw this):\n{eval_text}\n\n"
        f"Agent's self-report:\n{agent_response[:3000]}\n\n"
        f"Does the agent's report demonstrate that the success criterion is satisfied?\n"
        f"Respond JSON only: {{\"met\": true/false, \"reasoning\": \"one concise sentence\"}}\n"
        f"Only return met=true if clearly demonstrated. Uncertainty = false."
    )

    if settings.gate_provider == "gemini":
        return await _call_gemini(settings.gate_model, settings.gate_api_key, prompt)
    else:
        return await _call_openrouter(settings.gate_model, settings.gate_api_key, prompt)


async def _call_openrouter(model: str, api_key: Optional[str], prompt: str) -> dict:
    key = api_key or os.environ.get("VLT_OPENROUTER_API_KEY", "")
    if not key:
        return {"met": False, "reasoning": "No OpenRouter API key configured", "model_used": model, "tokens_used": 0}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "HTTP-Referer": "https://vlt.cli"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200},
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            return {**result, "model_used": model, "tokens_used": data.get("usage", {}).get("total_tokens", 0)}
    except Exception as e:
        logger.warning(f"OpenRouter verifier error: {e}")
        return {"met": False, "reasoning": f"Verifier error: {e}", "model_used": model, "tokens_used": 0}


async def _call_gemini(model: str, api_key: Optional[str], prompt: str) -> dict:
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        return {"met": False, "reasoning": "No Gemini API key configured", "model_used": model, "tokens_used": 0}
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 200}},
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            result = json.loads(content)
            tokens = (data.get("usageMetadata", {}).get("totalTokenCount", 0))
            return {**result, "model_used": model, "tokens_used": tokens}
    except Exception as e:
        logger.warning(f"Gemini verifier error: {e}")
        return {"met": False, "reasoning": f"Verifier error: {e}", "model_used": model, "tokens_used": 0}


# =============================================================================
# Verifier tool calls — exposed so the verifier agent can call them actively
# =============================================================================

@router.get("/gate/session-transcript/{session_id}")
async def gate_tool_get_transcript(session_id: str, limit_lines: int = 50):
    """Verifier tool: read recent assistant messages from a session transcript."""
    with Session(engine) as db:
        sess = db.get(AgentSession, session_id)
    if not sess or not sess.transcript_path:
        return {"content": "", "found": False}
    path = Path(sess.transcript_path) if sess.transcript_path else None
    if not path or not path.exists():
        return {"content": "", "found": False}
    lines = []
    try:
        for raw in path.read_text(errors="replace").splitlines()[-200:]:
            if not raw.strip():
                continue
            try:
                e = json.loads(raw)
            except Exception:
                continue
            if e.get("type") not in ("user", "assistant"):
                continue
            role = e.get("type")
            msg = e.get("message", {})
            content = msg.get("content", "")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        text += b.get("text", "")
            if text:
                lines.append(f"[{role.upper()}] {text[:500]}")
    except Exception as e:
        return {"content": "", "found": False, "error": str(e)}
    return {"content": "\n\n".join(lines[-limit_lines:]), "found": True}


@router.post("/gate/check-file")
async def gate_tool_check_file(request: Request):
    """Verifier tool: check if a file exists and return a snippet of its content."""
    body = await request.json()
    path_str = body.get("path", "")
    if not path_str:
        return {"exists": False}
    path = Path(path_str).expanduser()
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        snippet = path.read_text(errors="replace")[:500] if path.is_file() else None
        return {"exists": True, "path": str(path), "is_file": path.is_file(), "snippet": snippet}
    except Exception as e:
        return {"exists": True, "path": str(path), "error": str(e)}


@router.post("/gate/run-check")
async def gate_tool_run_check(request: Request):
    """
    Verifier tool: run a read-only verification command (tests, git log, etc.).
    ONLY safe read-only commands are allowed. Destructive commands are rejected.
    """
    body = await request.json()
    command = body.get("command", "").strip()
    cwd = body.get("cwd", os.path.expanduser("~"))

    # Safety: reject obviously destructive commands
    BLOCKED = ("rm ", "rm\t", "rmdir", "mv ", "dd ", "> ", "| rm", "sudo", "chmod", "chown",
                "mkfs", "fdisk", "truncate", "shred", "kill", "pkill")
    cmd_lower = command.lower()
    if any(b in cmd_lower for b in BLOCKED):
        raise HTTPException(403, f"Command blocked for safety: {command!r}")

    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=15,
        )
        return {
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:500],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Command timed out after 15s", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


# =============================================================================
# Logs
# =============================================================================

@router.get("/logs/{entry_id}")
async def get_fire_logs(entry_id: str, limit: int = 20):
    with Session(engine) as db:
        logs = db.exec(
            select(CronbanFireLog)
            .where(CronbanFireLog.entry_id == entry_id)
            .order_by(CronbanFireLog.fired_at.desc())
            .limit(limit)
        ).all()
    return [
        {
            "id": l.id, "entry_id": l.entry_id, "fired_at": l.fired_at,
            "trigger_type": l.trigger_type, "target_session_id": l.target_session_id,
            "result": l.result, "error_message": l.error_message,
        }
        for l in logs
    ]


@router.get("/gate-logs/{entry_id}")
async def get_gate_logs(entry_id: str, limit: int = 20):
    with Session(engine) as db:
        logs = db.exec(
            select(CronbanGateLog)
            .where(CronbanGateLog.entry_id == entry_id)
            .order_by(CronbanGateLog.checked_at.desc())
            .limit(limit)
        ).all()
    return [
        {
            "id": l.id, "entry_id": l.entry_id, "checked_at": l.checked_at,
            "gate_met": l.gate_met, "reasoning": l.reasoning,
            "model_used": l.model_used, "tokens_used": l.tokens_used, "duration_ms": l.duration_ms,
        }
        for l in logs
    ]


# =============================================================================
# Webhook trigger
# =============================================================================

@router.post("/entries/{entry_id}/fire")
async def fire_entry_endpoint(entry_id: str, request: Request):
    """
    Manually (or scheduler-initiated) fire a CronbanEntry.

    The scheduler calls this endpoint to keep fire logic in one place.
    The route delegates to cronban_scheduler.fire_entry() which handles:
      - Prompt resolution (skill > inline prompt)
      - Session injection or new session spawning
      - CronbanFireLog creation
      - Entry stats update (last_fired_at, fire_count, next_fire_at)

    Body (all optional):
      trigger_type   : "cron" | "gate" | "webhook" | "manual"  (default "manual")
      prompt_override: str — override the entry's prompt for this fire only

    Returns: {fired: bool, session_id: str | null, log_id: str | null}
    """
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    trigger_type = body.get("trigger_type", "manual")
    prompt_override = body.get("prompt_override")

    try:
        from vlt.daemon.cronban_scheduler import fire_entry
        result = await fire_entry(entry_id, trigger_type=trigger_type, prompt_override=prompt_override)
        return result
    except Exception as e:
        logger.error(f"fire_entry_endpoint {entry_id}: {e}")
        raise HTTPException(500, f"Fire failed: {e}")


@router.post("/webhook/{entry_id}")
async def trigger_webhook(entry_id: str, request: Request):
    """External HTTP trigger for webhook-type entries."""
    with Session(engine) as db:
        entry = db.get(CronbanEntry, entry_id)
    if not entry or entry.entry_type != "webhook":
        raise HTTPException(404, "Webhook entry not found")
    if entry.status != "active":
        raise HTTPException(409, f"Entry is {entry.status}, not active")

    # Verify HMAC if secret is set
    if entry.webhook_secret:
        sig = request.headers.get("X-Cronban-Signature", "")
        body = await request.body()
        expected = hmac.new(entry.webhook_secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(403, "Invalid signature")

    payload = {}
    try:
        payload = await request.json()
    except Exception:
        pass

    prompt_override = payload.get("prompt")
    # Fire via scheduler (fire_entry imported at call time to avoid circular)
    try:
        from vlt.daemon.cronban_scheduler import fire_entry
        import asyncio
        asyncio.create_task(fire_entry(entry_id, trigger_type="webhook", prompt_override=prompt_override))
    except Exception as e:
        raise HTTPException(500, f"Fire failed: {e}")

    return {"ok": True, "entry_id": entry_id}


# =============================================================================
# Settings — gate verifier model
# =============================================================================

@router.get("/settings/gate-model")
async def get_gate_settings():
    with Session(engine) as db:
        s = _get_or_create_settings(db)
        return {
            "provider": s.gate_provider,
            "model": s.gate_model,
            "has_api_key": bool(s.gate_api_key),
        }


@router.put("/settings/gate-model")
async def update_gate_settings(request: Request):
    body = await request.json()
    with Session(engine) as db:
        s = _get_or_create_settings(db)
        if "provider" in body:
            if body["provider"] not in ("openrouter", "gemini"):
                raise HTTPException(400, "provider must be openrouter|gemini")
            s.gate_provider = body["provider"]
        if "model" in body:
            s.gate_model = body["model"]
        if "api_key" in body and body["api_key"]:
            s.gate_api_key = body["api_key"]
        s.updated_at = _now()
        db.add(s)
        db.commit()
    return {"ok": True}


@router.post("/settings/gate-model/test")
async def test_gate_model():
    """Test the configured verifier model with a dummy classify call."""
    with Session(engine) as db:
        s = _get_or_create_settings(db)
        result = await _run_verifier(
            eval_text="The agent said hello.",
            agent_response="I said hello to the user as requested.",
            db=db,
        )
    return {"ok": True, "result": result}
