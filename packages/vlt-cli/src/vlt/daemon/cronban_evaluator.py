"""
cronban_evaluator.py — Helper-Claude gate evaluation engine.

Flow (triggered when an agent's turn ends):
  1. Find all active PipelineCards targeting the session that just went idle.
  2. For each card, load the current PipelineStage to get the gate_id.
  3. Build an evaluation prompt:
       - Original task (stage skill prompt / card info)
       - Agent's most recent output (last N assistant messages from transcript)
       - Gate eval criteria (from CronbanGate.prompt_markdown)
  4. Spawn / resume the project's dedicated "helper" Claude session.
  5. Wait for helper's response (direct subprocess call via _run_claude_message).
  6. Parse GATE_RESULT: PASS / GATE_RESULT: FAIL from the response.
  7. Update card gate_last_result; if PASS + stage.auto_advance → advance to next stage.

Helper session design:
  - One per project (keyed by project_id).
  - Marked with is_cronban_helper=True in agent_sessions.
  - Persists across evaluations via --resume, building up project context.
  - Does NOT trigger further gate evaluations when it goes idle (checked in
    server.py's _trigger_gate_evaluations).
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model name resolution
# ---------------------------------------------------------------------------
_MODEL_IDS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

_OPENAI_COMPAT_BASE = {
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


def _resolve_model(name: str) -> str:
    """Resolve short name (haiku/sonnet/opus) or pass through full model ID."""
    return _MODEL_IDS.get(name.lower(), name)


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

def _read_last_assistant_messages(session_id: str, n: int = 4) -> Optional[str]:
    """
    Return the last N assistant messages from the session's JSONL transcript,
    concatenated in order. Returns None if transcript unavailable or empty.
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession
    from sqlmodel import Session

    with Session(engine) as db:
        sess = db.get(AgentSession, session_id)
    if not sess or not sess.transcript_path:
        return None
    path = Path(sess.transcript_path)
    if not path.exists():
        return None

    messages: list[str] = []
    try:
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "assistant":
                continue
            msg = entry.get("message", {})
            content = msg.get("content", "")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                text = "\n".join(parts)
            if text.strip():
                messages.append(text.strip())
    except Exception as e:
        logger.warning(f"Transcript read error for {session_id}: {e}")
        return None

    if not messages:
        return None
    # Return last n messages
    return "\n\n---\n\n".join(messages[-n:])


# ---------------------------------------------------------------------------
# Eval prompt builder
# ---------------------------------------------------------------------------

def _build_eval_prompt(
    title: str,
    prompt_text: Optional[str],
    agent_output: str,
    eval_text: Optional[str],
) -> str:
    """
    Build the structured evaluation prompt sent to the helper Claude session.

    The helper must reply with a line containing exactly:
      GATE_RESULT: PASS
    or:
      GATE_RESULT: FAIL
    followed by 1–3 sentences of reasoning.
    """
    lines = [
        "=== GATE EVALUATION REQUEST ===",
        "",
        f"Task title: {title}",
    ]

    if prompt_text:
        lines += [
            "",
            "Original task given to the agent:",
            "```",
            prompt_text.strip()[:2000],
            "```",
        ]

    lines += [
        "",
        "Agent's most recent output:",
        "```",
        agent_output[:4000],
        "```",
    ]

    if eval_text:
        lines += [
            "",
            "Evaluation criteria (you may run scripts/checks as instructed):",
            eval_text.strip(),
        ]
        lines += [
            "",
            "Execute any instructions above. Use the results to assess whether",
            "the agent has successfully completed the task.",
        ]
    else:
        lines += [
            "",
            "Assess whether the agent has successfully completed the task.",
        ]

    lines += [
        "",
        "When done evaluating, output EXACTLY one of these lines:",
        "  GATE_RESULT: PASS",
        "  GATE_RESULT: FAIL",
        "followed by 1–3 sentences of reasoning.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GATE_RESULT parser
# ---------------------------------------------------------------------------

_GATE_RE = re.compile(r"GATE_RESULT:\s*(PASS|FAIL)", re.IGNORECASE)


def _parse_gate_result(text: str) -> dict:
    """Parse GATE_RESULT: PASS/FAIL from helper output. Returns {met, reasoning}."""
    m = _GATE_RE.search(text)
    if not m:
        logger.warning("Helper response missing GATE_RESULT marker — treating as FAIL")
        return {"met": False, "reasoning": "Helper did not return a clear GATE_RESULT verdict."}

    met = m.group(1).upper() == "PASS"
    # Reasoning: everything after the GATE_RESULT line
    after = text[m.end():].strip()
    reasoning = after[:500] if after else ("Task completed." if met else "Task not yet complete.")
    return {"met": met, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Helper session management
# ---------------------------------------------------------------------------

def _get_available_helper(project_id: str, cwd: str, eval_model: str) -> tuple[str, bool]:
    """
    Find an IDLE helper session for this project and model, or create a new one.

    Pool strategy: reuse any idle helper; if all are busy, spawn a new one.
    Returns (session_id, is_new).
    is_new=True  → caller must use --session-id (first spawn).
    is_new=False → caller must use --resume.
    """
    from vlt.db import engine
    from vlt.core.models import AgentSession
    from sqlmodel import Session, select

    model_id = _resolve_model(eval_model)

    with Session(engine) as db:
        # Find an idle helper for this project
        idle = db.exec(
            select(AgentSession).where(
                AgentSession.project_id == project_id,
                AgentSession.is_cronban_helper == True,  # noqa: E712
                AgentSession.status == "idle",
            ).order_by(AgentSession.last_activity.desc())
        ).first()

        if idle:
            logger.info(f"Reusing idle helper {idle.id} for project {project_id}")
            return idle.id, False

        # No idle helper — spawn a new one
        sid = str(uuid.uuid4())
        helper = AgentSession(
            id=sid,
            project_id=project_id,
            cwd=cwd,
            status="idle",
            source="managed",
            is_cronban_helper=True,
            model=model_id,
        )
        db.add(helper)
        db.commit()
        logger.info(f"Spawned new helper {sid} for project {project_id} model={model_id}")
        return sid, True


# ---------------------------------------------------------------------------
# Card advancement helper
# ---------------------------------------------------------------------------

def _advance_card(card_id: str, current_stage_id: str, pipeline_id: str) -> None:
    """
    Advance a PipelineCard to the next stage. If the next stage is terminal (or
    there is no next stage), mark the card as completed.
    """
    from vlt.db import engine
    from vlt.core.models import PipelineCard, PipelineStage
    from sqlmodel import Session, select

    with Session(engine) as db:
        current_stage = db.get(PipelineStage, current_stage_id)
        if not current_stage:
            logger.warning(f"advance_card: stage {current_stage_id} not found")
            return

        next_stage = db.exec(
            select(PipelineStage).where(
                PipelineStage.pipeline_id == pipeline_id,
                PipelineStage.stage_order > current_stage.stage_order,
            ).order_by(PipelineStage.stage_order)
        ).first()

        card = db.get(PipelineCard, card_id)
        if not card:
            return

        now_iso = datetime.utcnow().isoformat()

        if next_stage and not next_stage.is_terminal:
            # Move to next stage, reset gate state
            card.current_stage_id = next_stage.id
            card.gate_eval_pending = False
            card.gate_last_result = None
            card.gate_last_checked_at = None
            card.gate_consecutive_not_met = 0
            card.updated_at = now_iso
            db.add(card)
            db.commit()
            logger.info(
                f"advance_card: {card_id!r} advanced from {current_stage.name!r} "
                f"to {next_stage.name!r}"
            )
        else:
            # No next stage or next stage is terminal — mark completed
            card.status = "completed"
            if next_stage and next_stage.is_terminal:
                card.current_stage_id = next_stage.id
            card.gate_eval_pending = False
            card.updated_at = now_iso
            db.add(card)
            db.commit()
            logger.info(f"advance_card: {card_id!r} marked completed (no more non-terminal stages)")


# ---------------------------------------------------------------------------
# External API evaluation (z.ai / OpenRouter / Gemini — OpenAI-compat)
# ---------------------------------------------------------------------------

async def _call_external_api(
    prompt: str,
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str],
) -> Optional[str]:
    """
    Call an external OpenAI-compatible LLM for gate evaluation.

    Returns the assistant message text, or None on failure.
    """
    import httpx

    endpoint_base = base_url if base_url else _OPENAI_COMPAT_BASE.get(provider, "")
    if not endpoint_base:
        logger.error(f"External eval: unknown provider '{provider}' with no base_url")
        return None

    url = endpoint_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.0,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"External API eval failed ({provider}): {e}")
        return None


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

async def run_gate_evaluation(card_id: str, agent_session_id: str) -> None:
    """
    Run a full gate evaluation for one PipelineCard.

    Called as a background asyncio task after the agent's turn ends
    (from server.py _trigger_gate_evaluations) or by the gate_tick scheduler.
    """
    from vlt.db import engine
    from vlt.core.models import PipelineCard, PipelineStage, AgentSession
    from sqlmodel import Session

    logger.info(f"Gate evaluation starting: card={card_id} agent_session={agent_session_id}")

    # ── Load card + stage + gate prompt ──────────────────────────────────────
    with Session(engine) as db:
        card = db.get(PipelineCard, card_id)
        if not card or card.status != "active":
            logger.info(f"Gate eval skipped: card {card_id} not active")
            return

        stage = db.get(PipelineStage, card.current_stage_id)
        if not stage:
            logger.warning(f"Gate eval: stage {card.current_stage_id!r} not found for card {card_id}")
            return

        if not stage.gate_id:
            logger.info(f"Gate eval: stage {stage.name!r} has no gate — auto-passing")
            if stage.auto_advance:
                _advance_card(card_id, stage.id, card.pipeline_id)
            return

        # Resolve gate eval criteria
        from vlt.core.models import CronbanGate
        gate = db.get(CronbanGate, stage.gate_id)
        if not gate:
            logger.warning(f"Gate eval: gate {stage.gate_id!r} not found — skipping")
            return

        eval_instructions = gate.prompt_markdown

        # Resolve task prompt (what the working agent was told to do)
        prompt_text: Optional[str] = stage.prompt_text
        if not prompt_text and stage.skill_id:
            from vlt.core.models import CronbanSkill
            skill = db.get(CronbanSkill, stage.skill_id)
            if skill:
                prompt_text = skill.prompt_markdown

        project_id = card.project_id or "default"
        eval_model = stage.eval_model or "haiku"
        title = card.title
        pipeline_id = card.pipeline_id
        auto_advance = bool(stage.auto_advance)

        # Mark as evaluating
        card.gate_eval_pending = True
        db.add(card)
        db.commit()

    # ── Load gate settings (provider dispatch) ────────────────────────────
    from vlt.core.models import CronbanSettings
    with Session(engine) as db:
        gs = db.get(CronbanSettings, "default")
        gate_provider = gs.gate_provider if gs else "claude_code"
        gate_api_key = gs.gate_api_key if gs else None
        gate_base_url = gs.gate_base_url if gs else None
        # Per-stage eval_model overrides gate settings model for claude_code;
        # for external providers use the configured model.
        gate_cfg_model = gs.gate_model if gs else "sonnet"

    # ── Read agent's recent output ────────────────────────────────────────
    agent_output = _read_last_assistant_messages(agent_session_id, n=4)
    if not agent_output:
        logger.warning(f"Gate eval: no transcript output for session {agent_session_id}, skipping")
        with Session(engine) as db:
            c = db.get(PipelineCard, card_id)
            if c:
                c.gate_eval_pending = False
                db.add(c)
                db.commit()
        return

    # ── Build eval prompt ─────────────────────────────────────────────────
    eval_prompt = _build_eval_prompt(title, prompt_text, agent_output, eval_instructions)

    # ── Dispatch to configured provider ───────────────────────────────────
    raw_output: Optional[str] = None
    model_id: str

    if gate_provider == "claude_code":
        # Use the helper session pool (persists project context via --resume)
        with Session(engine) as db:
            agent_sess = db.get(AgentSession, agent_session_id)
            c = db.get(PipelineCard, card_id)
            cwd = (agent_sess.cwd if agent_sess else None) or (c.target_cwd if c else None) or "~"

        cwd = str(cwd).replace("~", __import__("os").path.expanduser("~"))
        model_id = _resolve_model(eval_model)

        helper_id, is_new = _get_available_helper(project_id, cwd, eval_model)
        logger.info(
            f"Gate eval [claude_code]: helper_session={helper_id} is_new={is_new} "
            f"model={model_id} cwd={cwd}"
        )

        try:
            from vlt.daemon.server import _run_claude_message

            await _run_claude_message(
                cwd=cwd,
                message=eval_prompt,
                session_id=helper_id,
                is_first=is_new,
                model=model_id,
            )

            if is_new:
                with Session(engine) as db:
                    h = db.get(AgentSession, helper_id)
                    if h:
                        h.status = "idle"
                        db.add(h)
                        db.commit()

        except Exception as e:
            logger.error(f"Gate eval: helper invocation failed: {e}")
            with Session(engine) as db:
                c = db.get(PipelineCard, card_id)
                if c:
                    c.gate_eval_pending = False
                    db.add(c)
                    db.commit()
            return

        raw_output = _read_last_assistant_messages(helper_id, n=1)

    else:
        # External OpenAI-compatible provider (zai, openrouter, gemini)
        model_id = gate_cfg_model
        if not gate_api_key:
            logger.error(f"Gate eval [{gate_provider}]: no API key configured — skipping")
            with Session(engine) as db:
                c = db.get(PipelineCard, card_id)
                if c:
                    c.gate_eval_pending = False
                    db.add(c)
                    db.commit()
            return

        logger.info(f"Gate eval [{gate_provider}]: model={model_id}")
        raw_output = await _call_external_api(
            prompt=eval_prompt,
            provider=gate_provider,
            model=model_id,
            api_key=gate_api_key,
            base_url=gate_base_url,
        )

    if not raw_output:
        logger.warning(f"Gate eval: no response from provider '{gate_provider}'")
        with Session(engine) as db:
            c = db.get(PipelineCard, card_id)
            if c:
                c.gate_eval_pending = False
                db.add(c)
                db.commit()
        return

    # ── Parse result ──────────────────────────────────────────────────────
    result = _parse_gate_result(raw_output)
    logger.info(f"Gate eval result: card={card_id} met={result['met']} reasoning={result['reasoning'][:100]}")

    # ── Persist result + optionally advance ───────────────────────────────
    with Session(engine) as db:
        c = db.get(PipelineCard, card_id)
        if not c:
            return

        now_iso = datetime.utcnow().isoformat()
        c.gate_last_result = json.dumps(result)
        c.gate_last_checked_at = now_iso
        c.gate_eval_pending = False

        if result["met"]:
            c.gate_consecutive_not_met = 0
        else:
            c.gate_consecutive_not_met = (c.gate_consecutive_not_met or 0) + 1

        db.add(c)

        # Log to gate logs
        from vlt.core.models import CronbanGateLog
        db.add(CronbanGateLog(
            id=str(uuid.uuid4()),
            entry_id=card_id,
            gate_met=result["met"],
            reasoning=result["reasoning"],
            model_used=model_id,
            tokens_used=0,
            duration_ms=0,
        ))
        db.commit()

    # Auto-advance if gate passed
    if result["met"] and auto_advance:
        with Session(engine) as db:
            c = db.get(PipelineCard, card_id)
            if c:
                _advance_card(card_id, c.current_stage_id, c.pipeline_id)
