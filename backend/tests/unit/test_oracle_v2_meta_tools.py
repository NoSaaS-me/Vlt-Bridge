"""Unit tests for meta_tools: list_tools and update_plan.

Tests:
    1. test_update_plan_no_plan         — returns "no plan active" when plan_ref is None
    2. test_update_plan_empty_plan      — returns "no plan active" when plan_ref is []
    3. test_update_plan_done            — marks a step as done, returns plan view
    4. test_update_plan_with_notes      — notes appear in plan view
    5. test_update_plan_invalid_status  — returns error string for bad status
    6. test_update_plan_index_out_of_range — returns error string
    7. test_update_plan_view_formatting — verifies status symbols (✓, →, ✗, –)
    8. test_list_tools_empty            — returns "No tools available."
    9. test_list_tools_with_tools       — lists tool name and first doc line
"""

from __future__ import annotations

import pytest

from backend.src.services.oracle_v2.tools.meta_tools import make_meta_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(*descriptions: str) -> list[dict]:
    return [{"description": d, "status": "pending", "notes": ""} for d in descriptions]


# ---------------------------------------------------------------------------
# T1: update_plan — no plan (None)
# ---------------------------------------------------------------------------

def test_update_plan_no_plan():
    tools = make_meta_tools(all_tools=[], plan_ref=None)
    update_plan = tools[1]
    result = update_plan(0, "done")
    assert "no plan" in result.lower()


# ---------------------------------------------------------------------------
# T2: update_plan — empty list
# ---------------------------------------------------------------------------

def test_update_plan_empty_plan():
    tools = make_meta_tools(all_tools=[], plan_ref=[])
    update_plan = tools[1]
    result = update_plan(0, "done")
    assert "no plan" in result.lower()


# ---------------------------------------------------------------------------
# T3: update_plan — mark step done
# ---------------------------------------------------------------------------

def test_update_plan_done():
    plan = _make_plan("Search for auth middleware", "Read auth_middleware.py")
    tools = make_meta_tools(all_tools=[], plan_ref=plan)
    update_plan = tools[1]

    result = update_plan(0, "done")

    assert "done" in result
    assert plan[0]["status"] == "done"
    # Step 1 still pending
    assert plan[1]["status"] == "pending"
    assert "[0]" in result
    assert "[1]" in result


# ---------------------------------------------------------------------------
# T4: update_plan — notes appear in output
# ---------------------------------------------------------------------------

def test_update_plan_with_notes():
    plan = _make_plan("Trace token verification flow")
    tools = make_meta_tools(all_tools=[], plan_ref=plan)
    update_plan = tools[1]

    result = update_plan(0, "done", notes="JWT_EXPIRY defaults to 90 days")

    assert "JWT_EXPIRY" in result
    assert plan[0]["notes"] == "JWT_EXPIRY defaults to 90 days"


# ---------------------------------------------------------------------------
# T5: update_plan — invalid status
# ---------------------------------------------------------------------------

def test_update_plan_invalid_status():
    plan = _make_plan("Search for auth middleware")
    tools = make_meta_tools(all_tools=[], plan_ref=plan)
    update_plan = tools[1]

    result = update_plan(0, "completed")

    assert "error" in result.lower() or "invalid" in result.lower()
    # Plan should not have been mutated
    assert plan[0]["status"] == "pending"


# ---------------------------------------------------------------------------
# T6: update_plan — index out of range
# ---------------------------------------------------------------------------

def test_update_plan_index_out_of_range():
    plan = _make_plan("Only step")
    tools = make_meta_tools(all_tools=[], plan_ref=plan)
    update_plan = tools[1]

    result_high = update_plan(5, "done")
    result_neg = update_plan(-1, "done")

    assert "error" in result_high.lower() or "out of range" in result_high.lower()
    assert "error" in result_neg.lower() or "out of range" in result_neg.lower()


# ---------------------------------------------------------------------------
# T7: update_plan — status symbols
# ---------------------------------------------------------------------------

def test_update_plan_view_formatting():
    plan = _make_plan("step A", "step B", "step C", "step D")
    tools = make_meta_tools(all_tools=[], plan_ref=plan)
    update_plan = tools[1]

    # Mark each step with a different status
    update_plan(0, "done")
    update_plan(1, "in_progress")
    update_plan(2, "blocked")
    result = update_plan(3, "skipped")

    assert "✓" in result   # done
    assert "→" in result   # in_progress
    assert "✗" in result   # blocked
    assert "–" in result   # skipped


# ---------------------------------------------------------------------------
# T8: list_tools — empty tool list
# ---------------------------------------------------------------------------

def test_list_tools_empty():
    tools = make_meta_tools(all_tools=[], plan_ref=None)
    list_tools = tools[0]
    result = list_tools()
    assert "no tools" in result.lower()


# ---------------------------------------------------------------------------
# T9: list_tools — with tools present
# ---------------------------------------------------------------------------

def test_list_tools_with_tools():
    def fake_search(query: str) -> str:
        """Search the codebase for relevant code."""
        return ""

    tools = make_meta_tools(all_tools=[fake_search], plan_ref=None)
    list_tools = tools[0]
    result = list_tools()

    assert "fake_search" in result
    assert "Search the codebase" in result
