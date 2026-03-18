"""Content Factory backend — handler dispatcher.

Called by the artifact harness via stdin/stdout JSON protocol.
Uses harness_call() from artifact_harness to invoke connectors.

Artifact structure (relative to backend/):
  ../manifest.json              — artifact manifest (pipeline key)
  ../.vlt/storage/              — persistent storage root
  ../.vlt/storage/pipeline_config.json  — pipeline config override
  ../.vlt/storage/content/      — content queue items
  ../.vlt/storage/runbooks/     — saved runbooks
  ../.vlt/storage/test_history/ — test run history
  ../.vlt/hot_state.json        — hot-reload state (consumed on load)
  ../prompts/                   — prompt template files
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to this file's directory)
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent
_ARTIFACT_DIR = _BACKEND_DIR.parent
_MANIFEST_PATH = _ARTIFACT_DIR / "manifest.json"
_STORAGE_DIR = _ARTIFACT_DIR / ".vlt" / "storage"
_PIPELINE_CONFIG_PATH = _STORAGE_DIR / "pipeline_config.json"
_CONTENT_DIR = _STORAGE_DIR / "content"
_RUNBOOKS_DIR = _STORAGE_DIR / "runbooks"
_HISTORY_DIR = _STORAGE_DIR / "test_history"


def _ensure_dirs() -> None:
    for d in (_STORAGE_DIR, _CONTENT_DIR, _RUNBOOKS_DIR, _HISTORY_DIR):
        d.mkdir(parents=True, exist_ok=True)


_ensure_dirs()

# ---------------------------------------------------------------------------
# In-memory state (survives within a single process lifetime)
# ---------------------------------------------------------------------------
_state: dict = {
    "pipeline_config": None,  # Loaded lazily
}


# ---------------------------------------------------------------------------
# harness_call import — use the real one if available, else noop stub
# ---------------------------------------------------------------------------
try:
    from artifact_harness import harness_call  # type: ignore[import]
except ImportError:
    # Fallback during unit tests / offline runs
    def harness_call(call_type: str, payload: dict) -> dict:  # type: ignore[misc]
        raise RuntimeError("harness_call not available (artifact_harness not loaded)")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _default_pipeline_config() -> dict:
    return {
        "version": 1,
        "stages": [],
        "cost_limits": {
            "per_item_usd": 0.10,
            "daily_usd": 5.00,
        },
        "auto_approve_threshold": 7.0,
    }


def _load_pipeline_config() -> dict:
    """Load pipeline config: override file > manifest > defaults."""
    if _PIPELINE_CONFIG_PATH.exists():
        try:
            return json.loads(_PIPELINE_CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            pass

    if _MANIFEST_PATH.exists():
        try:
            manifest = json.loads(_MANIFEST_PATH.read_text())
            if "pipeline" in manifest:
                return manifest["pipeline"]
        except json.JSONDecodeError:
            pass

    return _default_pipeline_config()


def _save_pipeline_config(config: dict) -> None:
    """Persist config to the override file."""
    _PIPELINE_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def _get_config() -> dict:
    if _state["pipeline_config"] is None:
        _state["pipeline_config"] = _load_pipeline_config()
    return _state["pipeline_config"]


# ---------------------------------------------------------------------------
# Hot-reload state
# ---------------------------------------------------------------------------

def save_state() -> dict:
    """Serialise in-memory state for hot reload."""
    return {
        "pipeline_config": _state.get("pipeline_config"),
    }


def load_state(state: dict) -> None:
    """Restore in-memory state after hot reload."""
    if "pipeline_config" in state and state["pipeline_config"] is not None:
        _state["pipeline_config"] = state["pipeline_config"]


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _handle_get_config(params: dict) -> dict:
    return {"config": _get_config()}


def _handle_update_config(params: dict) -> dict:
    config = _get_config()
    if "stages" in params:
        config["stages"] = params["stages"]
    if "cost_limits" in params:
        config["cost_limits"].update(params["cost_limits"])
    if "auto_approve_threshold" in params:
        config["auto_approve_threshold"] = params["auto_approve_threshold"]
    config["version"] = config.get("version", 1) + 1
    _save_pipeline_config(config)
    _state["pipeline_config"] = config
    return {"config": config, "saved": True}


def _handle_test(params: dict) -> dict:
    """Run pipeline for one topic; save to test history but NOT the queue.

    Returns a flat result for the frontend:
      {text, score, feedback, cost_usd, run_id, stages}
    """
    from pipeline import execute_pipeline  # type: ignore[import]

    config = _get_config()
    topic = params.get("topic", "")
    stages = params.get("stages") or config.get("stages", [])

    stage_results = execute_pipeline(stages, topic=topic)

    # Save to test history
    run_id = f"test_{uuid.uuid4().hex[:8]}"
    record = {
        "id": run_id,
        "topic": topic,
        "stages": stage_results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (_HISTORY_DIR / f"{run_id}.json").write_text(json.dumps(record, indent=2))

    # Extract frontend-friendly fields from stage results
    text = ""
    score = None
    feedback = ""
    total_tokens = 0

    for stage_id, result in stage_results.items():
        if not isinstance(result, dict):
            continue
        # Last text output wins
        if "output" in result and result.get("score") is None:
            text = result["output"]
        # Last review score wins
        if "score" in result:
            score = result["score"]
            feedback = result.get("feedback", "")
        total_tokens += result.get("tokens", 0)

    cost_usd = round(total_tokens * 0.000002, 6)

    return {
        "run_id": run_id,
        "text": text,
        "score": score,
        "feedback": feedback,
        "cost_usd": cost_usd,
        "stages": stage_results,
    }


def _handle_generate(params: dict) -> dict:
    """Run pipeline and add result to content queue."""
    from pipeline import execute_pipeline  # type: ignore[import]
    from queue import add_item  # type: ignore[import]

    config = _get_config()
    topic = params.get("topic", "")
    stages = params.get("stages") or config.get("stages", [])

    stage_results = execute_pipeline(stages, topic=topic)
    item = add_item(stage_results)

    return {"item": item, "stages": stage_results}


def _handle_approve(params: dict) -> dict:
    from queue import approve_item  # type: ignore[import]
    item_id = params.get("item_id", "")
    if not item_id:
        raise ValueError("item_id is required")
    return {"item": approve_item(item_id)}


def _handle_reject(params: dict) -> dict:
    from queue import reject_item  # type: ignore[import]
    item_id = params.get("item_id", "")
    reason = params.get("reason", "")
    if not item_id:
        raise ValueError("item_id is required")
    return {"item": reject_item(item_id, reason)}


def _handle_get_queue(params: dict) -> dict:
    from queue import list_items  # type: ignore[import]
    status = params.get("status")
    return {"items": list_items(status=status)}


def _handle_get_history(params: dict) -> dict:
    limit = params.get("limit", 50)
    records = []
    for path in sorted(_HISTORY_DIR.iterdir(), reverse=True):
        if path.suffix == ".json":
            try:
                records.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                continue
            if len(records) >= limit:
                break
    return {"history": records}


def _handle_save_runbook(params: dict) -> dict:
    name = params.get("name", "")
    if not name:
        raise ValueError("name is required")
    config = params.get("config") or _get_config()
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    runbook_path = _RUNBOOKS_DIR / f"{safe_name}.json"
    record = {
        "name": name,
        "config": config,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    runbook_path.write_text(json.dumps(record, indent=2))
    return {"saved": True, "name": name}


def _handle_list_runbooks(params: dict) -> dict:
    runbooks = []
    for path in sorted(_RUNBOOKS_DIR.iterdir()):
        if path.suffix == ".json":
            try:
                rb = json.loads(path.read_text())
                runbooks.append({"name": rb["name"], "saved_at": rb.get("saved_at")})
            except (json.JSONDecodeError, KeyError):
                continue
    return {"runbooks": runbooks}


def _handle_run_runbook(params: dict) -> dict:
    name = params.get("name", "")
    topic = params.get("topic", "")
    if not name:
        raise ValueError("name is required")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    runbook_path = _RUNBOOKS_DIR / f"{safe_name}.json"
    if not runbook_path.exists():
        raise FileNotFoundError(f"Runbook '{name}' not found")
    record = json.loads(runbook_path.read_text())
    runbook_config = record.get("config", {})
    stages = runbook_config.get("stages", [])

    return _handle_generate({"topic": topic, "stages": stages})


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, callable] = {
    "get_config": _handle_get_config,
    "update_config": _handle_update_config,
    "test": _handle_test,
    "generate": _handle_generate,
    "approve": _handle_approve,
    "reject": _handle_reject,
    "get_queue": _handle_get_queue,
    "get_history": _handle_get_history,
    "save_runbook": _handle_save_runbook,
    "list_runbooks": _handle_list_runbooks,
    "run_runbook": _handle_run_runbook,
}


def handle(action: str, params: dict) -> dict:
    """Entry point called by the artifact harness."""
    handler = _HANDLERS.get(action)
    if handler is None:
        raise ValueError(f"Unknown action: '{action}'. Available: {sorted(_HANDLERS)}")
    return handler(params)
