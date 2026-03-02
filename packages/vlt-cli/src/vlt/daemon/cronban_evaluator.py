"""
cronban_evaluator.py — Helper-Claude gate evaluation engine.

Flow (triggered when an agent's turn ends):
  1. Find all active GATE entries targeting the session that just went idle.
  2. For each entry, build an evaluation prompt:
       - Original task (prompt_text / skill)
       - Agent's most recent output (last N assistant messages from transcript)
       - Optional held-out eval_text (instructions the agent never saw)
  3. Spawn / resume the project's dedicated "helper" Claude session.
  4. Wait for helper's response (direct subprocess call via _run_claude_message).
  5. Parse GATE_RESULT: PASS / GATE_RESULT: FAIL from the response.
  6. Update entry gate_last_result; if PASS + auto_graduate → move card.

Helper session design:
  - One per project (keyed by project_id).
  - Marked with is_cronban_helper=True in agent_sessions.
  - Persists across evaluations via --resume, building up project context.
  - Model is set at spawn time; eval_model on the entry influences spawning a
    new helper when the existing one dies.
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
# Main evaluation entry point
# ---------------------------------------------------------------------------

async def run_gate_evaluation(entry_id: str, agent_session_id: str) -> None:
    """
    Run a full gate evaluation for one CronbanEntry.

    Called as a background asyncio task after the agent's turn ends.
    """
    from vlt.db import engine
    from vlt.core.models import CronbanEntry, AgentSession, KanbanColumn
    from sqlmodel import Session, select

    logger.info(f"Gate evaluation starting: entry={entry_id} agent_session={agent_session_id}")

    # ── Load entry + gate prompt ───────────────────────────────────────────
    with Session(engine) as db:
        entry = db.get(CronbanEntry, entry_id)
        if not entry or entry.status != "active":
            logger.info(f"Gate eval skipped: entry {entry_id} not active")
            return
        if entry.entry_type != "gate":
            return

        # Resolve task prompt (what the working agent was told to do)
        prompt_text = entry.prompt_text
        if not prompt_text and entry.skill_id:
            from vlt.core.models import CronbanSkill
            skill = db.get(CronbanSkill, entry.skill_id)
            if skill:
                prompt_text = skill.prompt_markdown

        # Resolve gate eval instructions: gate_id takes priority over legacy eval_text
        eval_instructions: Optional[str] = None
        if entry.gate_id:
            from vlt.core.models import CronbanGate
            gate = db.get(CronbanGate, entry.gate_id)
            if gate:
                eval_instructions = gate.prompt_markdown
            else:
                logger.warning(f"Gate eval: gate_id {entry.gate_id!r} not found, skipping")
                return
        elif entry.eval_text:
            # Legacy inline eval
            eval_instructions = entry.eval_text
        else:
            logger.info(f"Gate eval: entry {entry_id} has no gate or eval_text, auto-passing")
            # No gate configured — treat as pass
            with Session(engine) as db2:
                e = db2.get(CronbanEntry, entry_id)
                if e:
                    e.gate_last_result = '{"met": true, "reasoning": "No gate configured — auto-pass."}'
                    e.gate_last_checked_at = datetime.utcnow().isoformat()
                    db2.add(e)
                    db2.commit()
            return

        project_id = entry.project_id or "default"
        eval_model = entry.eval_model or "haiku"
        title = entry.title
        kanban_column_id = entry.kanban_column_id

        # Mark as evaluating
        entry.gate_eval_pending = True
        db.add(entry)
        db.commit()

    # ── Read agent's recent output ────────────────────────────────────────
    agent_output = _read_last_assistant_messages(agent_session_id, n=4)
    if not agent_output:
        logger.warning(f"Gate eval: no transcript output for session {agent_session_id}, skipping")
        with Session(engine) as db:
            entry = db.get(CronbanEntry, entry_id)
            if entry:
                entry.gate_eval_pending = False
                db.add(entry)
                db.commit()
        return

    # ── Build eval prompt ─────────────────────────────────────────────────
    eval_prompt = _build_eval_prompt(title, prompt_text, agent_output, eval_instructions)

    # ── Find / create helper session ──────────────────────────────────────
    # Get project CWD from agent session or entry's target_cwd
    with Session(engine) as db:
        agent_sess = db.get(AgentSession, agent_session_id)
        entry = db.get(CronbanEntry, entry_id)
        cwd = (agent_sess.cwd if agent_sess else None) or (entry.target_cwd if entry else None) or "~"

    cwd = str(cwd).replace("~", __import__("os").path.expanduser("~"))

    helper_id, is_new = _get_available_helper(project_id, cwd, eval_model)
    model_id = _resolve_model(eval_model)

    logger.info(
        f"Gate eval: helper_session={helper_id} is_new={is_new} "
        f"model={model_id} cwd={cwd}"
    )

    # ── Invoke helper (direct subprocess, wait for result) ────────────────
    # We import _run_claude_message from server.py at call time to avoid
    # circular import at module load.
    try:
        from vlt.daemon.server import _run_claude_message

        await _run_claude_message(
            cwd=cwd,
            message=eval_prompt,
            session_id=helper_id,
            is_first=is_new,
            model=model_id,
        )

        # Mark helper session as not-new after first message
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
            entry = db.get(CronbanEntry, entry_id)
            if entry:
                entry.gate_eval_pending = False
                db.add(entry)
                db.commit()
        return

    # ── Read helper's response ────────────────────────────────────────────
    helper_output = _read_last_assistant_messages(helper_id, n=1)
    if not helper_output:
        logger.warning(f"Gate eval: no response from helper session {helper_id}")
        with Session(engine) as db:
            entry = db.get(CronbanEntry, entry_id)
            if entry:
                entry.gate_eval_pending = False
                db.add(entry)
                db.commit()
        return

    # ── Parse result ──────────────────────────────────────────────────────
    result = _parse_gate_result(helper_output)
    logger.info(f"Gate eval result: entry={entry_id} met={result['met']} reasoning={result['reasoning'][:100]}")

    # ── Persist result + optionally graduate ──────────────────────────────
    with Session(engine) as db:
        entry = db.get(CronbanEntry, entry_id)
        if not entry:
            return

        now_iso = datetime.utcnow().isoformat()
        entry.gate_last_result = json.dumps(result)
        entry.gate_last_checked_at = now_iso
        entry.gate_eval_pending = False
        entry.gate_question_injected_at = None  # Reset for next cycle

        if result["met"]:
            entry.gate_consecutive_not_met = 0
        else:
            entry.gate_consecutive_not_met += 1

        db.add(entry)

        # Log to gate logs
        from vlt.core.models import CronbanGateLog
        db.add(CronbanGateLog(
            id=str(uuid.uuid4()),
            entry_id=entry_id,
            gate_met=result["met"],
            reasoning=result["reasoning"],
            model_used=model_id,
            tokens_used=0,
            duration_ms=0,
        ))
        db.commit()

        # Auto-graduation
        if result["met"] and kanban_column_id:
            col = db.get(KanbanColumn, kanban_column_id)
            if col and col.auto_graduate:
                # Find target column
                if col.graduation_column_id:
                    target_id = col.graduation_column_id
                else:
                    # Next column by col_order
                    next_col = db.exec(
                        select(KanbanColumn).where(
                            KanbanColumn.project_id == col.project_id,
                            KanbanColumn.col_order > col.col_order,
                        ).order_by(KanbanColumn.col_order)
                    ).first()
                    target_id = next_col.id if next_col else None

                if target_id and target_id != kanban_column_id:
                    entry = db.get(CronbanEntry, entry_id)
                    if entry:
                        entry.kanban_column_id = target_id
                        db.add(entry)
                        db.commit()
                        logger.info(
                            f"Gate eval: auto-graduated {entry_id!r} "
                            f"from {kanban_column_id!r} to {target_id!r}"
                        )
