# Implementation Plan: RLM Oracle

**Branch**: `022-rlm-oracle` | **Date**: 2026-02-22 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/022-rlm-oracle/spec.md`

## Summary

Replace the Behavior Tree Oracle with a Recursive Language Model (RLM) inference-time harness. The LLM receives a Python REPL environment where the entire project lives as addressable variables (`project`, `sub_oracle`, `Final`). It writes Python code to explore, slice, and synthesize answers. The root context window stays small and constant regardless of project size; all heavy lifting runs in REPL-executed code, with sub-oracle calls made programmatically inside loops (implementing O(|P|) work with O(1) root context).

The BT runtime, XML signal parser, query classifier, and prompt composer are removed entirely. The oracle REST API and MCP tool interface are unchanged — callers see no difference.

## Technical Context

**Language/Version**: Python 3.11+ (backend only; no frontend changes)
**Primary Dependencies**:
- `RestrictedPython` (new — REPL sandbox)
- `tree-sitter` + `tree-sitter-language-pack` (existing — symbol extraction)
- `FastAPI` + `sse-starlette` (existing — SSE streaming)
- `httpx` (existing — OpenRouter calls)
- `asyncio` (stdlib — `run_in_executor` + `Queue` bridge)

**Storage**: No new persistence. Ephemeral `RLMSession` per query. `OracleBridge` (existing) handles conversation history via existing `context_nodes` table.
**Testing**: pytest (existing backend test suite)
**Target Platform**: Linux server (FastAPI backend)
**Project Type**: Web application (backend-only change)
**Performance Goals**:
- Focused queries < 20s end-to-end (SC-003)
- Root LLM context < 4,000 tokens at all times (SC-002)
- Handle ≥500K token project corpus without truncation (SC-004)

**Constraints**:
- Max 25 root REPL iterations, max 8 sub-oracle iterations (FR-017)
- REPL step timeout: 30s (FR-016)
- Sub-oracle recursion cap: depth 2 (FR-017 + Assumptions)
- Files > 1MB: metadata only (FR-014)
- Oracle API contract: UNCHANGED (FR-019, FR-020)

**Scale/Scope**: Single-user, single backend instance. Concurrent oracle queries each get isolated REPL sessions.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Brownfield Integration | ✅ PASS | BT removal is spec-mandated (FR-018), not opportunistic refactoring. OpenRouter client moved, not rewritten. Oracle route wiring is a one-line swap. |
| II. Test-Backed Development | ✅ PASS | Unit tests required for `rlm_oracle.py`, `project_context.py`, `repl_executor.py`. Existing tests must continue passing. |
| III. Incremental Delivery | ✅ PASS | RLM wrapper implemented alongside BT; oracle route updated in final step; BT deleted last. |
| IV. Specification-Driven | ✅ PASS | All 23 FRs traced from spec.md. |

**Post-design re-check**: No new violations introduced. The `ProjectContext` design reuses existing `repomap.py` extraction code (constitution I: no rewrite). Three new service files match existing `services/` layer pattern.

## Project Structure

### Documentation (this feature)

```text
specs/022-rlm-oracle/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── oracle-api.yaml  # OpenAPI (existing endpoints, unchanged)
│   └── rlm-internal.md  # Python class contracts
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── services/
│   │   ├── rlm_oracle.py          # NEW: RLMOracleWrapper, RLMSession, RLMPromptBuilder
│   │   ├── project_context.py     # NEW: ProjectContext, TextHandle, FileManifest
│   │   ├── repl_executor.py       # NEW: REPLExecutor, REPLNamespace, QueuedStringIO
│   │   └── openrouter_client.py   # MOVED from bt/services/openrouter_client.py
│   ├── api/routes/
│   │   └── oracle.py              # MODIFIED: swap OracleBTWrapper → RLMOracleWrapper
│   └── models/
│       └── oracle.py              # UNCHANGED
├── tests/
│   └── unit/
│       ├── test_rlm_oracle.py     # NEW
│       ├── test_project_context.py # NEW
│       └── test_repl_executor.py  # NEW
│
└── [DELETED]:
    ├── src/bt/                    # Entire BT directory tree
    ├── src/models/signals.py
    ├── src/services/signal_parser.py
    ├── src/services/query_classifier.py
    └── src/services/prompt_composer.py

packages/vlt-cli/src/vlt/mcp/oracle_tools.py   # UNCHANGED (FR-020)
```

**Structure Decision**: Web application (existing `backend/` + `frontend/`). This feature is backend-only. Three new service modules in `backend/src/services/` follow the established pattern for service layer additions. No new routes — the oracle route file is patched in-place.

## Complexity Tracking

No constitution violations. No justification table needed.
