# Tasks: Vlt Unified MCP Server

**Input**: Design documents from `/specs/018-vlt-mcp-server/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓

**Tests**: Included — backend routes require pytest per constitution. MCP tool unit tests included per plan.md.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no shared dependencies)
- **[Story]**: User story this task belongs to (US1–US5)
- All paths absolute from repo root `/mnt/sda1/Projects/00Tooling/Vlt-Bridge/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add the `fastmcp` dependency, create the directory structure, and wire up the `vlt-mcp` console script entry point. Nothing here touches existing code.

- [X] T001 Add `fastmcp>=2.0.0` to `dependencies` and `vlt-mcp = "vlt.mcp_server:main"` to `[project.scripts]` in `packages/vlt-cli/pyproject.toml`
- [X] T002 Create directory `packages/vlt-cli/src/vlt/mcp/` and add empty `packages/vlt-cli/src/vlt/mcp/__init__.py`
- [X] T003 [P] Reinstall vlt-cli in editable mode to register the new console script: `cd packages/vlt-cli && pip install -e ".[oracle]"`

**Checkpoint**: `vlt-mcp --help` exits with usage (will fail until Phase 2, but the script entry must exist)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: MCP server skeleton, shared response helpers, and the oracle toggle settings field. Must be complete before any tool module can be tested end-to-end.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 Create `packages/vlt-cli/src/vlt/mcp_server.py` with `create_server()` (registers tool groups via lazy imports) and `main()` (sets up stderr logging, calls `mcp.run(transport="stdio")`)
- [X] T005 [P] Add `oracle_enabled: bool = True` field (reads `VLT_ORACLE_ENABLED` env var, prefix `VLT_`) to `Settings` class in `packages/vlt-cli/src/vlt/config.py`
- [X] T006 [P] Add `_ok(data: dict) -> dict` and `_err(code: str, message: str) -> dict` helpers to `packages/vlt-cli/src/vlt/mcp/__init__.py` — returns `{"status": "ok", **data}` and `{"status": "error", "code": code, "message": message}` respectively

**Checkpoint**: `vlt-mcp --help` shows FastMCP usage. `python -c "from vlt.mcp_server import create_server; create_server()"` runs without import errors.

---

## Phase 3: User Story 1 — Thread Memory via MCP (Priority: P1) 🎯 MVP

**Goal**: Full thread round-trip (create → push → read → seek → list) accessible via MCP tools with no CLI subprocess calls.

**Independent Test**: Configure `vlt-mcp` in Claude Code global MCP settings. Ask agent to: (1) create a thread, (2) push a thought, (3) read the thread back, (4) seek across threads. Verify all succeed with no CLI calls.

### Implementation for User Story 1

- [X] T007 [US1] Implement `vlt_thread_create` in `packages/vlt-cli/src/vlt/mcp/thread_tools.py` — calls `SqliteVaultService.create_project()` then `create_thread()` then `add_thought()`; auto-creates project if not found; returns `_ok({"thread_id", "project_id", "node_id", "sequence_id"})`
- [X] T008 [US1] Implement `vlt_thread_push` in `packages/vlt-cli/src/vlt/mcp/thread_tools.py` — calls `SqliteVaultService.add_thought()`; measures wall time; returns `_ok({"thread_id", "node_id", "sequence_id", "duration_ms"})`; returns `THREAD_NOT_FOUND` error if thread doesn't exist
- [X] T009 [US1] Implement `vlt_thread_read` in `packages/vlt-cli/src/vlt/mcp/thread_tools.py` — calls `SqliteVaultService.get_thread_state(thread_id, limit)`; returns `_ok({"thread_id", "project_id", "summary", "recent_nodes", "node_count"})` per schema in `contracts/mcp-tools.yaml`
- [X] T010 [US1] Implement `vlt_thread_seek` in `packages/vlt-cli/src/vlt/mcp/thread_tools.py` — primary path: `SqliteVaultService.search(query, project_id)`; on `VaultError`/missing embedding config, fall back to SQLAlchemy `LIKE '%query%'` on `Node.content`; always include `search_mode: "semantic" | "keyword"` in response
- [X] T011 [US1] Implement `vlt_thread_list` in `packages/vlt-cli/src/vlt/mcp/thread_tools.py` — SQLAlchemy `select(Thread)` filtered by `project_id` if provided; returns `_ok({"project_id", "threads": [{id, project_id, status, created_at, node_count}]})`
- [X] T012 [P] [US1] Implement `vlt_status` in `packages/vlt-cli/src/vlt/mcp/meta_tools.py` — lists all projects from DB, thread count per project, checks `DaemonClient.is_running()`, checks if `settings.sync_token` set (backend configured), checks `settings.oracle_enabled`; returns full health dict per schema in `contracts/mcp-tools.yaml`
- [X] T013 [P] [US1] Implement `vlt_project_detect` in `packages/vlt-cli/src/vlt/mcp/meta_tools.py` — walks up directory tree from `path` (defaults to `os.getcwd()`) looking for `vlt.toml`; if found, reads `project_id`; checks `CodeRAGStore` for existing index; returns `_ok({"found", "project_id", "project_name", "project_path", "has_coderag_index"})`
- [X] T014 [US1] Register `thread_tools` and `meta_tools` in `packages/vlt-cli/src/vlt/mcp_server.py` `create_server()` — import and call `register_thread_tools(mcp)` and `register_meta_tools(mcp)` (each module exposes a `register_*_tools(mcp: FastMCP)` function that decorates all tools onto the server)
- [X] T015 [P] [US1] Write `packages/vlt-cli/src/vlt/tests/unit/test_thread_tools.py` — mock `SqliteVaultService` with `unittest.mock.MagicMock`; test: create thread returns thread_id, push returns duration_ms ≤ measured wall time, read returns summary + nodes, seek returns results with search_mode field, list scoped by project; test THREAD_NOT_FOUND error on push to nonexistent thread
- [X] T016 [P] [US1] Write `packages/vlt-cli/src/vlt/tests/unit/test_meta_tools.py` — mock SQLAlchemy session and DaemonClient; test: vlt_status returns projects list + daemon_running bool; vlt_project_detect returns found=True when vlt.toml present in cwd, found=False when absent

**Checkpoint**: `pytest packages/vlt-cli/src/vlt/tests/unit/test_thread_tools.py packages/vlt-cli/src/vlt/tests/unit/test_meta_tools.py` passes. `vlt-mcp` starts and `vlt_status` returns a valid response.

---

## Phase 4: User Story 2 — Code Search and Repo Map via MCP (Priority: P2)

**Goal**: Full code intelligence surface via MCP — init, search, map, status, symbol lookup — without CLI subprocess calls.

**Independent Test**: Call `vlt_code_init` on a fresh project via MCP, poll `vlt_code_status` until `completed`, then `vlt_code_search "how does authentication work?"`. Verify results contain file_path and lineno.

### Implementation for User Story 2

- [X] T017 [US2] Implement `vlt_code_init` in `packages/vlt-cli/src/vlt/mcp/code_tools.py` — check `CodeRAGStore.get_job_status(project_id)` for existing index (if exists and not `force`, return current status); check `DaemonClient.is_running()` and submit via `DaemonClient.submit_job()` if daemon running; otherwise run `CodeRAGIndexer` in `threading.Thread` (daemon=True); return immediately with `_ok({"project_id", "job_status": "started" | "already_running", "job_id"})`
- [X] T018 [US2] Implement `vlt_code_search` in `packages/vlt-cli/src/vlt/mcp/code_tools.py` — call `CodeRAGStore.search_chunks(query, project_id, limit, language, file_pattern)` (hybrid vector + BM25 retrieval already implemented); if project not indexed, return `_err("INDEX_NOT_FOUND", "No code index for project '{project_id}'. Call vlt_code_init first.")`; return `_ok({"results": [...]})` per schema in `contracts/mcp-tools.yaml`
- [X] T019 [P] [US2] Implement `vlt_code_map` in `packages/vlt-cli/src/vlt/mcp/code_tools.py` — call `generate_repo_map(project_id, scope, max_tokens, include_signatures, include_docstrings)` from `vlt.core.coderag.repomap`; return `_ok({"map_text", "token_count", "files_included", "symbols_included", "symbols_total"})`; if no index, return `INDEX_NOT_FOUND`
- [X] T020 [P] [US2] Implement `vlt_code_status` in `packages/vlt-cli/src/vlt/mcp/code_tools.py` — call `CodeRAGStore.get_job_status(project_id)` and `CodeRAGStore.get_index_stats(project_id)` (files_count, chunks_count, symbols_count, graph_nodes, graph_edges, last_indexed); return full status dict per schema
- [X] T021 [US2] Implement `vlt_code_lookup` in `packages/vlt-cli/src/vlt/mcp/code_tools.py` — SQLAlchemy `select(SymbolDefinition).where(SymbolDefinition.name == symbol, project_id == ...)` with optional `kind` filter; return `_ok({"found": bool, "definitions": [...]})` per schema
- [X] T022 [US2] Register `code_tools` in `packages/vlt-cli/src/vlt/mcp_server.py` — add `from vlt.mcp.code_tools import register_code_tools` and call in `create_server()`
- [X] T023 [P] [US2] Write `packages/vlt-cli/src/vlt/tests/unit/test_code_tools.py` — mock `CodeRAGStore` and `DaemonClient`; test: init with no existing index starts job; init with existing index + no force returns current status; search on non-indexed project returns INDEX_NOT_FOUND; search returns results with required fields; status returns indexed=True after job completes

**Checkpoint**: `pytest packages/vlt-cli/src/vlt/tests/unit/test_code_tools.py` passes. Call `vlt_code_init` on a real project via MCP, then `vlt_code_status`, then `vlt_code_search`.

---

## Phase 5: User Story 3 — Global MCP Auto-Start (Priority: P3)

**Goal**: `vlt-mcp` can be added to Claude Code/Desktop global MCP config and starts automatically on session launch with first tool call responding within 2 seconds.

**Independent Test**: Add `vlt-mcp` to `~/.claude/settings.json` mcpServers block. Restart Claude Code. Issue `vlt_status` call — verify it responds without any manual server startup.

### Implementation for User Story 3

- [X] T024 [US3] Add `--check` flag to `packages/vlt-cli/src/vlt/mcp_server.py` `main()` — if `--check` in `sys.argv`, perform startup validation (DB connection, settings load, oracle_enabled check) and print JSON status to stdout, then exit 0. Used for cold-start testing and health verification.
- [X] T025 [P] [US3] Verify STDIO startup time: run `time echo '{}' | vlt-mcp` and confirm process initializes and accepts input within 2 seconds on the development machine. Document actual measured time in `specs/018-vlt-mcp-server/quickstart.md`. (Measured: 164ms on 9950x3d)
- [X] T026 [P] [US3] Add global MCP config via `claude mcp add --scope user vlt vlt-mcp` (stored in ~/.claude.json). Verified with `claude mcp list` — shows "vlt: vlt-mcp ✓ Connected".

**Checkpoint**: Claude Code lists all vlt tools in its tool panel without any manual server startup. Cold-start time measured and ≤2s.

---

## Phase 6: User Story 4 — Oracle Toggle in Web Settings (Priority: P4)

**Goal**: Users can enable/disable oracle MCP tools from the Settings page. Disabled state returns structured, actionable error. Enabled state proxies to oracle.

**Independent Test**: Open Settings, toggle Oracle off, save. Then ask an agent to call `vlt_oracle_status` — verify response contains `enabled: false` and a guidance message. Toggle back on, restart MCP session, call `vlt_oracle_query "what does this codebase do?"` — verify response.

### Implementation for User Story 4

- [X] T027 [US4] Add `oracle_mcp_enabled INTEGER NOT NULL DEFAULT 1` migration to the migration list in `backend/src/services/database.py` (append after the last existing `ALTER TABLE user_settings` entry)
- [X] T028 [P] [US4] Create Pydantic models `OracleSettings` and `OracleSettingsUpdate` in `backend/src/models/settings.py` — `OracleSettings` has `oracle_mcp_enabled: bool`; `OracleSettingsUpdate` has `oracle_mcp_enabled: bool`; added to existing settings.py
- [X] T029 [US4] Create `backend/src/api/routes/settings.py` with `GET /api/settings/oracle` and `PUT /api/settings/oracle` — uses `UserSettingsService.get/set_oracle_mcp_enabled()` for authenticated user
- [X] T030 [US4] Register settings router in `backend/src/api/main.py`
- [X] T031 [US4] Implement `vlt_oracle_status` in `packages/vlt-cli/src/vlt/mcp/oracle_tools.py` — checks backend /api/settings/oracle if configured, falls back to settings.oracle_enabled; returns three-state dict
- [X] T032 [US4] Implement `vlt_oracle_query` in `packages/vlt-cli/src/vlt/mcp/oracle_tools.py` — checks oracle_status; proxies to backend /api/oracle; returns ORACLE_DISABLED/ORACLE_NOT_CONFIGURED on bad state
- [X] T033 [US4] Register `oracle_tools` in `packages/vlt-cli/src/vlt/mcp_server.py` (already done via try/except ImportError in Phase 2)
- [X] T034 [P] [US4] Add `getOracleSettings()` and `updateOracleSettings(enabled: boolean)` to `frontend/src/services/api.ts`
- [X] T035 [US4] Add Oracle section to `frontend/src/pages/Settings.tsx` — added `TabsContent value="oracle"` with Oracle `TabsTrigger` (grid-cols-6); Switch bound to oracle_mcp_enabled; loads on mount; saves on toggle
- [X] T036 [P] [US4] Write `backend/tests/unit/test_settings_routes.py` — 6 tests passing (GET default, GET disabled, PUT false, PUT true, 401×2)
- [X] T037 [P] [US4] Write `packages/vlt-cli/src/vlt/tests/unit/test_oracle_tools.py` — 8 tests passing (enabled/disabled/configured/backend states, ORACLE_DISABLED, ORACLE_NOT_CONFIGURED, proxy)

**Checkpoint**: `pytest backend/tests/unit/test_settings_routes.py packages/vlt-cli/src/vlt/tests/unit/test_oracle_tools.py` passes. Toggle in UI changes oracle tool behavior in next MCP session.

---

## Phase 7: User Story 5 — Vault Notes via Unified MCP (Priority: P5)

**Goal**: All five vault note tools accessible through the unified vlt-mcp server, with structured errors when backend is unreachable.

**Independent Test**: With backend running, call `vlt_note_write`, `vlt_note_read`, `vlt_note_search`, `vlt_note_list`, and `vlt_note_backlinks` via MCP. Verify results. Then stop backend and verify `VAULT_UNAVAILABLE` error — not a stack trace.

### Implementation for User Story 5

- [X] T038 [US5] Implement `vlt_note_write` and `vlt_note_read` in `packages/vlt-cli/src/vlt/mcp/vault_tools.py` — httpx.Client with Bearer auth; VAULT_UNAVAILABLE on ConnectError/Timeout; NOTE_NOT_FOUND on 404 for read
- [X] T039 [P] [US5] Implement `vlt_note_search` and `vlt_note_list` in `packages/vlt-cli/src/vlt/mcp/vault_tools.py` — maps to GET /api/search and GET /api/notes
- [X] T040 [US5] Implement `vlt_note_backlinks` in `packages/vlt-cli/src/vlt/mcp/vault_tools.py` — maps to GET /api/backlinks/{path}; returns empty list on 404
- [X] T041 [US5] Register `vault_tools` in `packages/vlt-cli/src/vlt/mcp_server.py` (already done via try/except ImportError in Phase 2)

**Checkpoint**: All five vault tools return correct data when backend is running. All five return `VAULT_UNAVAILABLE` (not an exception) when backend is stopped.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Consistency pass, documentation, and end-to-end validation across all stories.

- [X] T042 Audit all MCP tool implementations — all 17 tools return structured `_err()` for all error conditions; no bare exceptions propagate; codes: INDEX_NOT_FOUND, INVALID_PATH, THREAD_NOT_FOUND, THREAD_CREATE_FAILED, PROJECT_NOT_FOUND, PROJECT_ERROR, VAULT_UNAVAILABLE, VAULT_NOT_CONFIGURED, VAULT_ERROR, NOTE_NOT_FOUND, ORACLE_DISABLED, ORACLE_NOT_CONFIGURED, ORACLE_ERROR, DB_ERROR, INTERNAL_ERROR
- [X] T043 [P] Updated `README.md` MCP Client Configuration section — added vlt-mcp recommended setup with `claude mcp add --scope user` and link to quickstart.md
- [X] T044 [P] Updated `CLAUDE.md` Recent Changes — 018-vlt-mcp-server entry describes 17 tools, 5 modules, cold-start time, and registration method; Active Technologies includes FastMCP 3.0+
- [X] T045 Full tool surface validation — `vlt_mcp --check` passes; `asyncio.run(mcp.list_tools())` confirms all 19 tools registered: 5 thread, 2 meta, 5 code, 2 oracle, 5 vault; `claude mcp list` shows "vlt: vlt-mcp ✓ Connected"; 39 unit tests passing

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Requires Phase 1 complete — blocks all user story phases
- **US1 Thread Memory (Phase 3)**: Requires Phase 2 — highest priority, start here
- **US2 Code Search (Phase 4)**: Requires Phase 2 — can run in parallel with US1 if staffed
- **US3 Auto-Start (Phase 5)**: Requires Phase 3 complete (needs working tools to validate auto-start)
- **US4 Oracle Toggle (Phase 6)**: Requires Phase 2 — backend and vlt-cli parts can run in parallel with US1/US2
- **US5 Vault Notes (Phase 7)**: Requires Phase 2 — can run in parallel with any other story
- **Polish (Phase 8)**: Requires all desired stories complete

### User Story Dependencies

- **US1 (P1)**: Independent after Phase 2 ✓
- **US2 (P2)**: Independent after Phase 2 ✓
- **US3 (P3)**: Depends on US1 completing (validates auto-start with working thread tools)
- **US4 (P4)**: Independent after Phase 2 for vlt-cli work; depends on backend migration (T027) for frontend work
- **US5 (P5)**: Independent after Phase 2 ✓

### Within Each User Story

- Tool implementations within a story can be parallelized (different functions in the same file are marked [P] where safe)
- Register call in mcp_server.py (T014, T022, T033, T041) depends on that story's tool module being complete
- Tests can be written in parallel with implementation (mocked interfaces)

---

## Parallel Execution Examples

### Phase 3 (US1 — Thread Tools) Parallel Opportunities

```bash
# These can run simultaneously (different functions in thread_tools.py):
Task: T007 — vlt_thread_create
Task: T008 — vlt_thread_push

# After T007 and T008 land:
Task: T009 — vlt_thread_read
Task: T010 — vlt_thread_seek

# These can run simultaneously (different files):
Task: T012 — vlt_status in meta_tools.py
Task: T013 — vlt_project_detect in meta_tools.py

# Tests can run in parallel with implementation:
Task: T015 — test_thread_tools.py
Task: T016 — test_meta_tools.py
```

### Phase 6 (US4 — Oracle Toggle) Parallel Opportunities

```bash
# Backend and vlt-cli work are in completely different packages:
Task: T027 + T028 + T029 + T030 — backend migration + route (sequential within backend)
Task: T031 + T032 — oracle_tools.py (vlt-cli, independent)
Task: T034 + T035 — frontend services + Settings.tsx (frontend, independent)

# Tests parallel with each other:
Task: T036 — backend test
Task: T037 — vlt-cli test
```

---

## Implementation Strategy

### MVP (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: US1 Thread Memory
4. **STOP and VALIDATE**: Thread round-trip via MCP, no CLI calls. Add to Claude Code config.
5. This alone eliminates the 200–500ms CLI subprocess overhead for thread operations.

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. US1 (Thread Memory) → **MVP: Agents can log/retrieve reasoning via MCP**
3. US2 (Code Search) → Agents can search codebases via MCP
4. US3 (Auto-Start) → Validation that global config works end-to-end
5. US4 (Oracle Toggle) → Backend UI control + oracle MCP tools
6. US5 (Vault Notes) → Unified surface — single MCP server for everything
7. Polish → Production ready

### Single-Developer Sequential Order

```
Phase 1 → Phase 2 → Phase 3 (US1) → Phase 5 (US3 validation) → Phase 4 (US2) → Phase 7 (US5) → Phase 6 (US4) → Phase 8 (Polish)
```

Rationale: Validate the STDIO auto-start with US1 tools before building more tools. US4 (oracle toggle) is most complex (backend + frontend + vlt-cli) so defer to near end. US5 (vault) is simple HTTP proxy, save for last.

---

## Summary

| Phase | User Story | Tasks | Key Deliverable |
|-------|-----------|-------|----------------|
| 1 | Setup | T001–T003 | fastmcp dep + vlt-mcp script |
| 2 | Foundational | T004–T006 | MCP server skeleton + helpers |
| 3 | US1 (P1) | T007–T016 | Thread memory via MCP ← **MVP** |
| 4 | US2 (P2) | T017–T023 | Code search via MCP |
| 5 | US3 (P3) | T024–T026 | Auto-start validation |
| 6 | US4 (P4) | T027–T037 | Oracle toggle + web UI |
| 7 | US5 (P5) | T038–T041 | Vault notes via MCP |
| 8 | Polish | T042–T045 | Error audit + docs |

**Total**: 45 tasks across 8 phases
**Parallel opportunities**: 18 tasks marked [P]
**MVP scope**: T001–T016 (Phases 1–3), delivers the highest-value capability
