"""Unit tests for oracle_v2 code_tools — symbol_lookup validation.

Tests:
    1. test_symbol_lookup_invalid_kind — returns error string for unknown kind
    2. test_symbol_lookup_valid_kinds  — all valid kinds do not trigger a kind error
    3. test_make_code_tools_returns_four — factory returns exactly 4 tools including symbol_lookup

Note: actual vlt subprocess calls are skipped here (integration-test territory).
"""

from __future__ import annotations

import pytest

from backend.src.services.oracle_v2.tools.code_tools import make_code_tools


# ---------------------------------------------------------------------------
# T1: symbol_lookup — invalid kind
# ---------------------------------------------------------------------------

def test_symbol_lookup_invalid_kind():
    tools = make_code_tools(project_id="test-project")
    # symbol_lookup is the 4th tool (index 3)
    symbol_lookup = tools[3]

    result = symbol_lookup("MyClass", kind="module")

    assert "error" in result.lower() or "invalid" in result.lower()
    assert "module" in result


# ---------------------------------------------------------------------------
# T2: symbol_lookup — valid kinds pass kind validation (no kind-error prefix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["function", "class", "method", "variable", ""])
def test_symbol_lookup_valid_kinds_no_validation_error(kind, monkeypatch):
    """Valid kinds should NOT return a kind-validation error.

    We monkeypatch _run_vlt_coderag to return an empty result so the test
    doesn't need a real vlt installation.
    """
    import backend.src.services.oracle_v2.tools.code_tools as ct

    monkeypatch.setattr(ct, "_run_vlt_coderag", lambda *a, **kw: {"results": []})

    tools = make_code_tools(project_id="test-project")
    symbol_lookup = tools[3]

    result = symbol_lookup("SomeSymbol", kind=kind)

    assert "Invalid kind" not in result


# ---------------------------------------------------------------------------
# T3: make_code_tools returns 4 callables
# ---------------------------------------------------------------------------

def test_make_code_tools_returns_four():
    tools = make_code_tools(project_id="test-project")
    assert len(tools) == 4
    names = [t.__name__ for t in tools]
    assert "search_code" in names
    assert "read_file" in names
    assert "get_repo_map" in names
    assert "symbol_lookup" in names


# ---------------------------------------------------------------------------
# T4: symbol_lookup — no results returns helpful message
# ---------------------------------------------------------------------------

def test_symbol_lookup_no_results(monkeypatch):
    import backend.src.services.oracle_v2.tools.code_tools as ct

    monkeypatch.setattr(ct, "_run_vlt_coderag", lambda *a, **kw: {"results": []})

    tools = make_code_tools(project_id="test-project")
    symbol_lookup = tools[3]

    result = symbol_lookup("NonExistentSymbol")

    assert "NonExistentSymbol" in result
    assert "search_code" in result  # hint to use broader search


# ---------------------------------------------------------------------------
# T5: symbol_lookup — formats result correctly when results present
# ---------------------------------------------------------------------------

def test_symbol_lookup_formats_result(monkeypatch):
    import backend.src.services.oracle_v2.tools.code_tools as ct

    fake_results = {
        "results": [
            {
                "file_path": "backend/src/services/oracle_v2/sandbox.py",
                "start_line": 197,
                "kind": "function",
                "content": "def make_sandbox_eval(tools: list[Callable]) -> Callable:",
            }
        ]
    }
    monkeypatch.setattr(ct, "_run_vlt_coderag", lambda *a, **kw: fake_results)

    tools = make_code_tools(project_id="test-project")
    symbol_lookup = tools[3]

    result = symbol_lookup("make_sandbox_eval", kind="function")

    assert "sandbox.py" in result
    assert "197" in result
    assert "make_sandbox_eval" in result
