# Implementation Plan: Oracle & Librarian CodeAct Rework

**Branch**: `023-oracle-codeact-rework` | **Date**: 2026-03-11 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/023-oracle-codeact-rework/spec.md`

## Summary

Replace the ephemeral RLM Oracle (per-request `RLMSession`) with a LangGraph CodeAct agent backed by `AsyncSqliteSaver` for durable multi-turn state, Graphiti for cross-session temporal memory, an expanded tool registry including shell access, and a planner node for complex task decomposition. The existing `/api/oracle/stream` endpoint signature is preserved unchanged; the frontend requires zero modifications.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript / React 19 (frontend — no new frontend code for core Oracle)
**Primary Dependencies**:
- `langgraph-codeact>=0.1.3` — CodeAct graph builder
- `langgraph-checkpoint-sqlite` — `AsyncSqliteSaver` for thread state persistence
- `graphiti-core[anthropic]>=0.28.1` — temporal knowledge graph (memory layer)
- FalkorDB Docker container — Graphiti graph DB backend (~256MB RAM)
- Existing: `sse-starlette`, `fastapi`, `openrouter` client, `TavilySearchService`, `OpenRouterSearchService`

**Storage**:
- `data/checkpoints.db` — new SQLite file for LangGraph thread state (separate from `data/index.db`)
- `oracle_threads` table — added to `data/index.db` or new `data/oracle_threads.db` for thread listing index
- Graphiti data — persisted in FalkorDB Docker volume
- Existing `data/index.db` — unchanged (vault, FTS, tags, links)

**Testing**: `pytest` — unit tests for new `oracle_v2` services; integration test for multi-turn state persistence

**Target Platform**: Linux server / local dev (CachyOS); Docker for Graphiti + FalkorDB sidecar

**Project Type**: Web application (backend FastAPI + frontend React; only backend changes for core feature)

**Performance Goals**:
- First response chunk ≤3s for simple queries (SC-004)
- Multi-turn follow-up: no repeated retrieval — measurably faster than cold-start

**Constraints**:
- `/api/oracle/stream` request/response format 100% backward-compatible (SC-005)
- All existing SSE chunk types emitted (FR-019)
- `AsyncSqliteSaver` opened once in FastAPI lifespan — not per-request
- Shell commands restricted to allowlist only (SC-007)
- API keys never stored in checkpointed state

**Scale/Scope**: Local dev tool, primarily single user; architecture supports multi-user (scoped state per `user_id`)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| **I. Brownfield Integration** | ✅ PASS | New code isolated in `backend/src/services/oracle_v2/`. Existing `rlm_oracle.py`, `repl_executor.py`, `project_context.py` frozen (not deleted). `oracle.py` route file updated minimally — wrapper swap only. |
| **II. Test-Backed Development** | ✅ PASS | New `oracle_v2` services require pytest unit tests. Multi-turn persistence requires integration test. Existing tests untouched. |
| **III. Incremental Delivery** | ✅ PASS | 5-phase migration: spike first (3 tools only), full migration only after spike validates architecture. Old code path stays runnable until Phase 3. |
| **IV. Specification-Driven** | ✅ PASS | Spec `023-oracle-codeact-rework/spec.md` exists. Research in `Ai-notes/2026-03-11/Oracle-Rework/SPEC-FINAL.md`. |

**Complexity Tracking** — no violations, table omitted.

## Project Structure

### Documentation (this feature)

```text
specs/023-oracle-codeact-rework/
├── plan.md              ← this file
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
├── quickstart.md        ← Phase 1 output
├── contracts/
│   ├── oracle-stream.yaml     ← existing endpoint (confirmed backward compat)
│   └── oracle-threads.yaml    ← new thread management endpoints
└── tasks.md             ← Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── api/
│   │   ├── main.py                    ← ADD lifespan for AsyncSqliteSaver + Graphiti
│   │   └── routes/
│   │       └── oracle.py              ← SWAP wrapper; ADD /threads endpoints
│   ├── models/
│   │   └── oracle.py                  ← ADD thread list response models (OracleRequest unchanged)
│   └── services/
│       ├── oracle_v2/                 ← NEW package
│       │   ├── __init__.py
│       │   ├── graph.py               ← build_oracle_graph(), lifespan helper
│       │   ├── state.py               ← OracleState (extends CodeActState)
│       │   ├── nodes.py               ← memory_loader_node, planner_node, memory_writer_node
│       │   ├── sandbox.py             ← make_sandbox_eval(), ALLOWED_IMPORTS, run_shell()
│       │   ├── streaming.py           ← oracle_to_sse() LangGraph→OracleStreamChunk adapter
│       │   └── tools/
│       │       ├── __init__.py
│       │       ├── code_tools.py      ← wrap existing ToolExecutor code tools
│       │       ├── vault_tools.py     ← wrap existing vault tools
│       │       ├── thread_tools.py    ← wrap existing thread tools
│       │       ├── web_tools.py       ← web_search, web_fetch, deep_research
│       │       ├── shell_tools.py     ← run_shell(), git_log(), diff_file()
│       │       ├── memory_tools.py    ← remember(), recall()
│       │       └── meta_tools.py      ← list_tools(), update_plan(), delegate_librarian()
│       ├── memory/                    ← NEW package
│       │   ├── __init__.py
│       │   ├── client.py              ← Graphiti singleton
│       │   ├── loader.py              ← memory_loader_node impl
│       │   └── writer.py              ← memory_writer_node + background episode add
│       │
│       ├── rlm_oracle.py              ← FROZEN (no changes)
│       ├── repl_executor.py           ← FROZEN (no changes)
│       └── project_context.py         ← FROZEN (no changes)
│
├── tests/
│   ├── unit/
│   │   ├── test_oracle_v2_state.py
│   │   ├── test_oracle_v2_sandbox.py
│   │   ├── test_oracle_v2_tools.py
│   │   └── test_oracle_v2_streaming.py
│   └── integration/
│       └── test_oracle_v2_multiturn.py
│
└── pyproject.toml                     ← ADD new deps

docker/
└── graphiti-compose.yml               ← NEW: Graphiti + FalkorDB
```

**Structure Decision**: Web application (Option 2). Backend-only changes. All new Oracle code isolated in `services/oracle_v2/` and `services/memory/`. Route file `oracle.py` updated minimally. No frontend changes required.

---

## Phase 0: Research

*Research completed before this plan was written. Key findings from `Ai-notes/2026-03-11/Oracle-Rework/SPEC-FINAL.md` (4-agent research pass, March 2026):*

See [research.md](./research.md) for full consolidated findings.

---

## Phase 1: Design & Contracts

See [data-model.md](./data-model.md) and [contracts/](./contracts/).

### Entities

**New entities** (see `data-model.md` for full schemas):
1. `OracleState` — LangGraph TypedDict extending `CodeActState`; adds `user_id`, `project_id`, `plan`, `plan_step`, `recalled_facts`. API keys excluded (passed via `configurable`).
2. `oracle_threads` — SQLite table in `data/index.db` for thread listing index (LangGraph has no list_threads API)
3. `MemoryFact` — Graphiti episode/edge; stored in FalkorDB; scoped by `group_id=f"{user_id}:{project_id}"`
4. `OracleThreadSummary` / `OracleThreadListResponse` — new Pydantic response models

**Unchanged entities**:
- `OracleRequest` — request fields unchanged; `context_id` now maps to LangGraph `thread_id`
- `OracleStreamChunk` — all existing types preserved; `done` chunk emits `context_id=thread_id`

### API Contracts

| Contract | Description |
|---|---|
| [oracle-stream.yaml](./contracts/oracle-stream.yaml) | Existing `/api/oracle/stream` — unchanged request/response; backward compatible |
| [oracle-threads.yaml](./contracts/oracle-threads.yaml) | New `/api/oracle/threads` endpoints: list, get, delete, patch, history |

**New endpoints added**:
- `GET /api/oracle/threads` — list threads for authenticated user (filterable by project_id)
- `GET /api/oracle/threads/{thread_id}` — get thread metadata
- `DELETE /api/oracle/threads/{thread_id}` — delete thread from index
- `PATCH /api/oracle/threads/{thread_id}` — update thread title
- `GET /api/oracle/threads/{thread_id}/history` — get message history from LangGraph checkpoint

### quickstart.md

See [quickstart.md](./quickstart.md) for local dev setup including FalkorDB Docker container.
