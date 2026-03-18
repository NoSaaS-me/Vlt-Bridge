"""Per-connector daily cost tracking for artifact sandbox.

Costs are stored as JSON files in the artifact's .vlt/costs/ directory,
one file per calendar day (UTC). The manifest defines optional per-connector
daily limits; 0 means unlimited.

File layout:
    {artifact_disk_path}/.vlt/costs/{YYYY-MM-DD}.json

File schema:
    {
        "openrouter": {"total_usd": 0.45, "calls": 23},
        "huggingface": {"total_usd": 0.12, "calls": 5}
    }
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def _today() -> str:
    """Return today's date string (UTC, YYYY-MM-DD)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _costs_path(disk_path: str, date: str | None = None) -> Path:
    """Return the path to today's (or given day's) cost file."""
    day = date or _today()
    return Path(disk_path) / ".vlt" / "costs" / f"{day}.json"


def _read_costs(disk_path: str, date: str | None = None) -> dict:
    """Read the cost file for a given day. Returns {} if not found."""
    path = _costs_path(disk_path, date)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to read cost file {path}: {e}")
        return {}


def _write_costs(disk_path: str, data: dict, date: str | None = None) -> None:
    """Persist the cost file for a given day."""
    path = _costs_path(disk_path, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, indent=2))
    except OSError as e:
        log.error(f"Failed to write cost file {path}: {e}")


def _get_disk_path(artifact_id: str) -> str | None:
    """Resolve the artifact's disk path from the service layer."""
    try:
        from vlt.daemon.artifact_service import get_artifact
        artifact = get_artifact(artifact_id)
        if artifact:
            return artifact["disk_path"]
    except Exception as e:
        log.error(f"Failed to resolve disk path for artifact {artifact_id}: {e}")
    return None


def _get_manifest(artifact_id: str) -> dict:
    """Fetch the artifact's manifest dict (or empty dict on failure)."""
    try:
        from vlt.daemon.artifact_service import get_artifact
        artifact = get_artifact(artifact_id)
        if artifact:
            return artifact.get("manifest") or {}
    except Exception as e:
        log.error(f"Failed to fetch manifest for artifact {artifact_id}: {e}")
    return {}


def check_cost_limit(
    artifact_id: str,
    connector: str,
    estimated_cost: float = 0.0,
) -> tuple[bool, str]:
    """Check whether a connector call would exceed the daily limit.

    Args:
        artifact_id: The artifact making the call.
        connector: Connector name (e.g. "openrouter").
        estimated_cost: Estimated cost of the upcoming call in USD.

    Returns:
        (allowed, reason) — allowed=True means the call may proceed.
        If the daily limit is 0 (or absent), the call is always allowed.
    """
    manifest = _get_manifest(artifact_id)
    cost_limits = manifest.get("cost_limits", {})
    connector_limit = cost_limits.get(connector, {})
    daily_limit = connector_limit.get("daily_usd", 0)

    if daily_limit == 0:
        # 0 = unlimited
        return True, "unlimited"

    disk_path = _get_disk_path(artifact_id)
    if not disk_path:
        # Can't verify — allow but log
        log.warning(f"Cannot check cost limit: disk_path unknown for artifact {artifact_id}")
        return True, "disk_path_unknown"

    costs = _read_costs(disk_path)
    connector_costs = costs.get(connector, {"total_usd": 0.0, "calls": 0})
    current_total = connector_costs.get("total_usd", 0.0)

    if current_total + estimated_cost > daily_limit:
        reason = (
            f"Daily limit of ${daily_limit:.4f} for connector '{connector}' exceeded. "
            f"Current spend: ${current_total:.4f}, estimated: ${estimated_cost:.4f}."
        )
        return False, reason

    return True, "ok"


def record_cost(artifact_id: str, connector: str, cost_usd: float) -> None:
    """Record the cost of a completed connector call.

    Args:
        artifact_id: The artifact that made the call.
        connector: Connector name.
        cost_usd: Actual cost in USD (use 0.0 if the connector doesn't report cost).
    """
    if cost_usd < 0:
        log.warning(f"Negative cost recorded for {artifact_id}/{connector}: {cost_usd}")
        cost_usd = 0.0

    disk_path = _get_disk_path(artifact_id)
    if not disk_path:
        log.error(f"Cannot record cost: disk_path unknown for artifact {artifact_id}")
        return

    costs = _read_costs(disk_path)
    entry = costs.get(connector, {"total_usd": 0.0, "calls": 0})
    entry["total_usd"] = round(entry.get("total_usd", 0.0) + cost_usd, 6)
    entry["calls"] = entry.get("calls", 0) + 1
    costs[connector] = entry

    _write_costs(disk_path, costs)
    log.debug(
        f"Recorded ${cost_usd:.6f} for {artifact_id}/{connector}. "
        f"Daily total: ${entry['total_usd']:.6f} over {entry['calls']} calls."
    )


def get_daily_costs(artifact_id: str) -> dict:
    """Return today's cost summary for all connectors.

    Returns a dict like:
        {
            "date": "2026-03-17",
            "connectors": {
                "openrouter": {"total_usd": 0.45, "calls": 23},
                ...
            }
        }
    """
    disk_path = _get_disk_path(artifact_id)
    if not disk_path:
        return {"date": _today(), "connectors": {}}

    costs = _read_costs(disk_path)
    return {"date": _today(), "connectors": costs}
