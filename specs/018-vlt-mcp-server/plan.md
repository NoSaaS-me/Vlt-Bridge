# Implementation Plan: Vlt Unified MCP Server

**Branch**: `018-vlt-mcp-server` | **Date**: 2026-02-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/018-vlt-mcp-server/spec.md`

## Summary

Add a `vlt-mcp` STDIO command to the vlt-cli package that exposes all vlt capabilities (thread memory, code intelligence, vault notes, oracle) as a single unified MCP server. Agents configure it once in their global MCP settings and it auto-starts on demand. An oracle enable/disable toggle is added to the Document-MCP web UI Settings page, persisted per user in the backend database.

**Scope**: 5 new Python modules in vlt-cli, 1 new backend API route, 1 database migration, 1 frontend settings section. No existing code is rewritten.

---

## Technical Context

**Language/Version**: Python 3.11+ (vlt-cli), TypeScript 5.x + React 19 (frontend)
**Primary Dependencies**:
- vlt-cli: FastMCP (new), SQLAlchemy 2.0 (existing), httpx (existing), pydantic-settings (existing)
- Backend: FastAPI (existing), sqlite3 stdlib (existing per constitution)
- Frontend: React 19, shadcn/ui Switch + Tabs (existing imports in Settings.tsx)

**Storage**:
- Thread/code data: `~/.vlt/profiles/{profile}/vault.db` (SQLAlchemy, existing)
- Oracle toggle: `backend/data/index.db` → `user_settings` table, new `oracle_mcp_enabled` column
- Vault notes: Document-MCP backend filesystem + index (no change)

**Testing**: pytest (vlt-cli), pytest (backend)
**Target Platform**: Linux (primary), macOS compatible; STDIO process model
**Performance Goals**: Thread push <50ms, server cold-start <2s, code search <500ms for indexed project
**Constraints**: No port binding required; no daemon dependency for thread/meta tools; no breaking changes to existing `vlt` CLI

---

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Brownfield Integration | PASS | All changes are additive: new files, one migration, two UI additions |
| II. Test-Backed Development | PASS | pytest tests required per phase; existing tests must not break |
| III. Incremental Delivery | PASS | Each phase (threads → code → vault → oracle → UI) is independently testable |
| IV. Specification-Driven | PASS | All 32 FRs traced to implementation tasks |
| No Magic | PASS | Direct SQLAlchemy calls; explicit oracle check; no middleware hooks |
| Single Source of Truth | PASS | vault.db for thread/code state; backend DB for oracle toggle |
| Error Handling | PASS | All tools return structured error responses; no unhandled exceptions |

**No violations. No Complexity Tracking table required.**

---

## Project Structure

### Documentation (this feature)

```text
specs/018-vlt-mcp-server/
├── plan.md              ← this file
├── spec.md
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
├── quickstart.md        ← Phase 1 output
├── contracts/
│   ├── mcp-tools.yaml   ← MCP tool schemas
│   └── settings-api.yaml ← Oracle settings REST API
└── tasks.md             ← Phase 2 output (from /speckit.tasks)
```

### Source Code

```text
packages/vlt-cli/
├── pyproject.toml                          MODIFIED (add fastmcp dep + vlt-mcp script)
└── src/vlt/
    ├── mcp_server.py                       NEW — entry point
    ├── mcp/
    │   ├── __init__.py                     NEW
    │   ├── thread_tools.py                 NEW — vlt_thread_create/push/read/seek/list
    │   ├── code_tools.py                   NEW — vlt_code_init/search/map/status/lookup
    │   ├── vault_tools.py                  NEW — vlt_note_write/read/search/list/backlinks
    │   ├── oracle_tools.py                 NEW — vlt_oracle_query/status
    │   └── meta_tools.py                   NEW — vlt_status/vlt_project_detect
    └── tests/
        └── unit/
            ├── test_thread_tools.py        NEW
            ├── test_code_tools.py          NEW
            ├── test_oracle_tools.py        NEW
            └── test_meta_tools.py         NEW

backend/
├── src/
│   ├── api/
│   │   ├── main.py                        MODIFIED (register settings router)
│   │   └── routes/
│   │       └── settings.py                NEW — GET/PUT /api/settings/oracle
│   └── services/
│       └── database.py                    MODIFIED (oracle_mcp_enabled migration)
└── tests/
    └── unit/
        └── test_settings_routes.py        NEW

frontend/
└── src/
    └── pages/
        └── Settings.tsx                   MODIFIED (add Oracle tab section)
```

**Structure Decision**: Web application (backend + frontend + CLI package). The MCP server lives in the vlt-cli package, not the backend, because it needs direct SQLAlchemy access to the local vlt database and must run as a standalone STDIO process.

---

## Phase 0: Research (COMPLETE)

See [research.md](./research.md). All unknowns resolved:

| Question | Resolution |
|----------|------------|
| FastMCP STDIO API | Use existing backend pattern (`@mcp.tool`, `mcp.run(transport="stdio")`) |
| Thread seek without embeddings | Semantic first, fall back to SQLAlchemy LIKE; report `search_mode` in response |
| Oracle toggle persistence | New `oracle_mcp_enabled` column in backend `user_settings` table |
| Oracle toggle MCP check | HTTP GET to backend at tool call time; local env var fallback |
| Vault tools architecture | Proxy to backend HTTP API via httpx (auth via sync_token) |
| Code init in MCP context | DaemonClient if daemon running, else inline thread; return immediately |

---

## Phase 1: Design & Contracts (COMPLETE)

Artifacts produced:
- [data-model.md](./data-model.md) — entity schemas, error codes, state transitions
- [contracts/mcp-tools.yaml](./contracts/mcp-tools.yaml) — all 19 MCP tool schemas
- [contracts/settings-api.yaml](./contracts/settings-api.yaml) — oracle settings REST API
- [quickstart.md](./quickstart.md) — Claude Desktop/Code config guide

---

## Phase 2: Implementation Tasks

> Detailed tasks generated by `/speckit.tasks`. Summary of implementation order below.

### Milestone A: Core MCP Server + Thread Tools (P1 — highest value)

**A1**: Add `fastmcp>=2.0.0` to `packages/vlt-cli/pyproject.toml` dependencies. Add `vlt-mcp = "vlt.mcp_server:main"` to `[project.scripts]`.

**A2**: Create `packages/vlt-cli/src/vlt/mcp_server.py` — `create_server()` + `main()` with lazy tool module imports and STDIO transport.

**A3**: Create `packages/vlt-cli/src/vlt/mcp/__init__.py` (empty).

**A4**: Implement `thread_tools.py`:
- `vlt_thread_create` → `SqliteVaultService.create_project()` (if needed) + `create_thread()` + `add_thought()`
- `vlt_thread_push` → `SqliteVaultService.add_thought()` + measure wall time
- `vlt_thread_read` → `SqliteVaultService.get_thread_state()`
- `vlt_thread_seek` → `SqliteVaultService.search()` with LIKE fallback
- `vlt_thread_list` → SQLAlchemy `select(Thread)` filtered by project

**A5**: Implement `meta_tools.py`:
- `vlt_status` → list projects from DB, check DaemonClient, check backend ping, check settings
- `vlt_project_detect` → walk up from `path` (default: `cwd`) looking for `vlt.toml`

**A6**: Write `tests/unit/test_thread_tools.py` and `test_meta_tools.py` (mock SQLAlchemy session).

**A7**: Manual smoke test — add to Claude Code global MCP config, run `vlt_status`, then a full thread round-trip.

---

### Milestone B: Code Intelligence Tools (P2)

**B1**: Implement `code_tools.py`:
- `vlt_code_init` → check `CodeRAGStore` for existing index, check DaemonClient status, submit job or run `CodeRAGIndexer` in background thread. Return immediately with job status.
- `vlt_code_search` → `CodeRAGStore.search_chunks()` (hybrid retrieval already implemented)
- `vlt_code_map` → `generate_repo_map()` from `core/coderag/repomap.py`
- `vlt_code_status` → `CodeRAGStore.get_job_status(project_id)`
- `vlt_code_lookup` → `CodeRAGStore.find_symbols(name, kind)`

**B2**: Add `oracle` optional install to `vlt-mcp` entry point — tree-sitter, llama-index needed for indexer. Document in quickstart.

**B3**: Write `tests/unit/test_code_tools.py` — mock `CodeRAGStore`, test init idempotency, index-not-found guidance message.

---

### Milestone C: Vault Tools (P5)

**C1**: Implement `vault_tools.py`:
- All 5 tools proxy to backend HTTP API using `httpx.Client(base_url=settings.vault_url)`
- Auth header: `Authorization: Bearer {settings.sync_token}`
- On connection error: return `VAULT_UNAVAILABLE` structured error
- Map backend response fields to MCP tool output schema from `contracts/mcp-tools.yaml`

**C2**: No new tests required beyond error path coverage (backend API is tested separately).

---

### Milestone D: Oracle Tools (P4)

**D1**: Implement `oracle_tools.py`:
- `vlt_oracle_status` → HTTP GET `/api/settings/oracle` (if backend configured), then check local `VLT_ORACLE_ENABLED`; return three-state status
- `vlt_oracle_query` → check oracle enabled (same as status check), then proxy to `OracleClient` (backend thin-client) or `OracleOrchestrator` (local fallback)

**D2**: Add `oracle_enabled: bool = True` field to vlt-cli `Settings` class (reads `VLT_ORACLE_ENABLED` env var). Used as fallback when backend unavailable.

**D3**: Write `tests/unit/test_oracle_tools.py` — test disabled state, not-configured state, enabled + working state.

---

### Milestone E: Backend Oracle Toggle (P4 web UI)

**E1**: Add migration to `backend/src/services/database.py`:
```python
"ALTER TABLE user_settings ADD COLUMN oracle_mcp_enabled INTEGER NOT NULL DEFAULT 1"
```
Place after the last existing migration in the list.

**E2**: Create `backend/src/api/routes/settings.py`:
```python
router = APIRouter(prefix="/api/settings", tags=["settings"])

@router.get("/oracle", response_model=OracleSettings)
async def get_oracle_settings(auth: AuthContext = Depends(require_auth_context)):
    ...

@router.put("/oracle", response_model=OracleSettings)
async def update_oracle_settings(body: OracleSettingsUpdate, auth: AuthContext = Depends(require_auth_context)):
    ...
```
Uses `sqlite3` directly per constitution (same pattern as existing route handlers).

**E3**: Register the new router in `backend/src/api/main.py`:
```python
from .routes.settings import router as settings_router
app.include_router(settings_router)
```

**E4**: Write `backend/tests/unit/test_settings_routes.py` covering GET and PUT with auth.

---

### Milestone F: Frontend Settings UI (P4 web UI)

**F1**: Add Oracle section to `Settings.tsx`.
- Add new tab "Oracle" to the existing `Tabs` component
- Tab content: description, `Switch` component (already imported) bound to `oracle_mcp_enabled`
- On toggle change: call `PUT /api/settings/oracle`
- On load: call `GET /api/settings/oracle` to populate initial state

**F2**: Add `getOracleSettings` and `updateOracleSettings` to `frontend/src/services/api.ts` (or a new `services/settings.ts` if it doesn't already exist).

**F3**: Manual verification — toggle off, confirm disabled error from MCP oracle call, toggle on, confirm oracle works.

---

## Key Design Decisions

### Why vlt-cli, not backend
The MCP server lives in vlt-cli because it needs direct SQLAlchemy access to the local SQLite database at `~/.vlt/`. The backend runs on a server; vlt-mcp runs on the developer's machine as a subprocess. They share vault notes (via HTTP) but not thread/code data.

### Why STDIO, not HTTP
STDIO is spawn-on-demand. The AI assistant manages the process lifecycle. No ports, no daemon, no manual startup. The auto-start requirement is satisfied by nature of the STDIO transport protocol.

### Why proxy for vault, direct for threads
Thread and code data live in a local SQLite database that the vlt-mcp process can access directly. Vault notes live in the backend's filesystem + index — going direct would bypass the search indexer and require duplicating `VaultService` + `IndexerService` into vlt-cli. The HTTP proxy adds ~1-5ms latency per vault call, which is acceptable.

### Oracle check at tool-call time, not startup
The STDIO server starts fresh each session. Checking oracle status at startup is fine, but checking at call time means a toggle change takes effect on the current session's next oracle call, not just the next session. Marginally better UX at negligible cost.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| FastMCP version incompatibility between vlt-cli and backend | Low | Pin same version in both pyproject.toml files |
| Thread seek FTS fallback produces poor results | Medium | Label results with `search_mode: "keyword"`; agents can make informed decisions |
| `vlt_code_init` blocks MCP session for large repos | Low | Always runs inline in a background thread; MCP tool returns immediately with job ID |
| Oracle check HTTP call adds latency per oracle tool call | Low | Single GET to local backend (usually <5ms); acceptable |
| Settings.tsx Oracle tab conflicts with existing tab order | Low | Add as last tab; doesn't affect existing tabs |
