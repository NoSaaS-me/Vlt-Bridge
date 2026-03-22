"""Content Factory — content queue management.

Queue items stored as individual JSON files under:
  ../artifacts/{artifact_id}/.vlt/storage/content/{item_id}.json

Item lifecycle:
  qc_pending   → human or auto review required
  auto_approved → all QC stages passed threshold; no human needed
  approved     → human approved
  rejected     → human rejected (reason stored)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_DIR = Path(__file__).parent
_CONTENT_DIR = _BACKEND_DIR.parent / ".vlt" / "storage" / "content"

# Valid status values
_VALID_STATUSES = {"qc_pending", "auto_approved", "approved", "rejected"}


def _ensure_dir() -> None:
    _CONTENT_DIR.mkdir(parents=True, exist_ok=True)


def _item_path(item_id: str) -> Path:
    return _CONTENT_DIR / f"{item_id}.json"


def _read_item(item_id: str) -> dict:
    path = _item_path(item_id)
    if not path.exists():
        raise FileNotFoundError(f"Content item not found: {item_id}")
    return json.loads(path.read_text())


def _write_item(item: dict) -> None:
    _ensure_dir()
    _item_path(item["id"]).write_text(json.dumps(item, indent=2))


def _detect_auto_approve(stages: dict) -> bool:
    """Return True if the last QC stage has auto_approved=True."""
    qc_types = {"text_review", "vision_review"}
    # Walk stage results in reverse order to find the last QC stage
    for stage_result in reversed(list(stages.values())):
        if isinstance(stage_result, dict) and "auto_approved" in stage_result:
            return bool(stage_result["auto_approved"])
    return False


def _estimate_cost(stages: dict) -> float:
    """Very rough token-based cost estimate ($0.000002 per token as floor)."""
    total_tokens = 0
    for result in stages.values():
        if isinstance(result, dict):
            total_tokens += result.get("tokens", 0)
    return round(total_tokens * 0.000002, 6)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_item(stages_result: dict, topic: str = "") -> dict:
    """Create a new content item from pipeline stage results.

    Status is set to 'auto_approved' if the last QC stage auto-approved,
    otherwise 'qc_pending'.

    Args:
        stages_result: Dict of stage_id → stage result from execute_pipeline().
        topic:         Original topic string (preserved for output routing).

    Returns:
        The created content item dict.
    """
    _ensure_dir()
    item_id = f"cnt_{uuid.uuid4().hex[:8]}"
    auto_approved = _detect_auto_approve(stages_result)
    status = "auto_approved" if auto_approved else "qc_pending"

    item: dict = {
        "id": item_id,
        "topic": topic,
        "pipeline_version": 1,
        "stages": stages_result,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "approved_at": None,
        "rejected_at": None,
        "reject_reason": None,
        "output_results": None,
        "total_cost_usd": _estimate_cost(stages_result),
    }
    _write_item(item)
    return item


def approve_item(item_id: str) -> dict:
    """Set item status to 'approved'.

    Args:
        item_id: Content item ID (e.g. "cnt_abc12345").

    Returns:
        Updated item dict.

    Raises:
        FileNotFoundError: If item does not exist.
    """
    item = _read_item(item_id)
    item["status"] = "approved"
    item["approved_at"] = datetime.now(timezone.utc).isoformat()
    item["rejected_at"] = None
    item["reject_reason"] = None
    _write_item(item)
    return item


def reject_item(item_id: str, reason: str = "") -> dict:
    """Set item status to 'rejected' with an optional reason.

    Args:
        item_id: Content item ID.
        reason:  Human-readable rejection reason.

    Returns:
        Updated item dict.

    Raises:
        FileNotFoundError: If item does not exist.
    """
    item = _read_item(item_id)
    item["status"] = "rejected"
    item["rejected_at"] = datetime.now(timezone.utc).isoformat()
    item["reject_reason"] = reason
    item["approved_at"] = None
    _write_item(item)
    return item


def list_items(status: str | None = None) -> list[dict]:
    """List content queue items, optionally filtered by status.

    Args:
        status: One of the valid statuses, or None for all items.

    Returns:
        List of item dicts sorted by created_at descending (newest first).

    Raises:
        ValueError: If status is not a recognised value.
    """
    if status is not None and status not in _VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Valid values: {sorted(_VALID_STATUSES)}"
        )

    _ensure_dir()
    items: list[dict] = []
    for path in _CONTENT_DIR.glob("*.json"):
        try:
            item = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if status is None or item.get("status") == status:
            items.append(item)

    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


def get_item(item_id: str) -> dict:
    """Fetch a single content item by ID.

    Args:
        item_id: Content item ID.

    Returns:
        Item dict.

    Raises:
        FileNotFoundError: If item does not exist.
    """
    return _read_item(item_id)
