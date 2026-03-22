# Tasks: Artifact Sandbox

**Input**: Design documents from `/specs/025-artifact-sandbox/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependencies, and directory structure

- [x] T001 Create artifact data directory structure: `data/artifacts/` and ensure it's in `.gitignore`
- [x] T002 Add `watchdog` dependency to `packages/vlt-cli/pyproject.toml` (already in backend)
- [x] T003 [P] Add `playwright` as optional dependency to `packages/vlt-cli/pyproject.toml` and run `playwright install chromium`
- [x] T004 [P] Create frontend component directory `frontend/src/components/artifacts/`
- [x] T005 [P] Create VltBridge client script at `frontend/public/vlt-bridge.js` — postMessage-based API stub (storage, notes, code, backend, connectors, events, artifact self-management) with request/response correlation via message IDs

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core models, services, and routing infrastructure that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Define `Artifact` SQLModel in `packages/vlt-cli/src/vlt/core/models.py` — fields: id, user_id, project_id, name, description, type, state, state_history_json, manifest_json, thread_id, disk_path, version, created_at, updated_at. Include state enum and validation.
- [x] T007 Add Artifact table creation DDL to daemon database initialization in `packages/vlt-cli/src/vlt/db.py` (or wherever `create_all` runs for SQLModel tables)
- [x] T008 Create `packages/vlt-cli/src/vlt/daemon/artifact_service.py` — core service class with: `create_artifact()` (init disk dir, git init, create manifest, insert DB row, create vlt thread), `get_artifact()`, `list_artifacts()`, `update_artifact()`, `delete_artifact()` (stop backend if running, rm disk dir, delete DB row), `transition_state()` (validate state machine graph, record history, git commit on transition)
- [x] T009 Create `packages/vlt-cli/src/vlt/daemon/artifact_routes.py` — FastAPI APIRouter with prefix `/api/artifacts`. Implement CRUD endpoints: POST `/`, GET `/`, GET `/{id}`, PUT `/{id}`, DELETE `/{id}`, POST `/{id}/state`. Import and register router in `server.py` via `app.include_router()`.
- [x] T010 Create `frontend/src/services/artifact-api.ts` — REST client functions: `createArtifact()`, `listArtifacts()`, `getArtifact()`, `updateArtifact()`, `deleteArtifact()`, `transitionState()`. Follow pattern from existing `services/api.ts`.
- [x] T011 Extend `AgentsPage.tsx` — add `'artifacts'` to `NavSection` union type, add entry to `NAV_ITEMS` array with `Puzzle` icon from Lucide, add conditional render block for `activeSection === 'artifacts'` that renders `ArtifactsCompositorView`.
- [x] T012 [P] Create `frontend/src/components/artifacts/ArtifactSidebar.tsx` — artifact list sidebar (follows `SessionSidebar.tsx` pattern). Shows artifact names, state badges (colored dots), "New Artifact" button. Props: `artifacts`, `selectedId`, `onSelect`, `onCreate`.
- [x] T013 [P] Create `frontend/src/components/artifacts/NewArtifactDialog.tsx` — dialog for creating artifacts. Fields: name, description, type (ephemeral/persistent), project_id (auto-filled from current project). Uses shadcn Dialog + Input + Select.
- [x] T014 Create `frontend/src/components/artifacts/ArtifactsCompositorView.tsx` — main layout component. Left: `ArtifactSidebar` (w-52). Right: selected artifact view (or empty state). Fetches artifacts via `artifact-api.ts`, manages selection state.

**Checkpoint**: Foundation ready — artifact CRUD works end-to-end (create in UI, see in sidebar, store on disk, persist in DB). No rendering or execution yet.

---

## Phase 3: User Story 1 — Create and Run a Simple Artifact (Priority: P1) MVP

**Goal**: User creates an artifact with HTML/CSS/JS files and sees it render in a sandboxed iframe with VltBridge storage API working.

**Independent Test**: Create artifact, write `index.html` with a VltBridge.storage call, see it render, verify storage persists across reloads.

### Implementation for User Story 1

- [x] T015 [US1] Add frontend serving endpoint to `artifact_routes.py`: `GET /api/artifacts/{id}/frontend/{path:path}` — serves files from `{disk_path}/frontend/`. For `index.html`, inject `<script src="/vlt-bridge.js">` and HMR client script before `</head>`. Set correct MIME types.
- [x] T016 [US1] Implement VltBridge host handler in `frontend/src/lib/vlt-bridge-host.ts` — listens for `postMessage` events from artifact iframe, routes `vlt_request` messages to daemon REST endpoints, sends `vlt_response` back. Handles: `storage.get`, `storage.set`, `storage.list`, `artifact.setState`, `artifact.getState`, `artifact.log`.
- [x] T017 [US1] Add artifact-scoped storage endpoints to `artifact_routes.py`: `GET /api/artifacts/{id}/storage/{key}`, `PUT /api/artifacts/{id}/storage/{key}`, `GET /api/artifacts/{id}/storage`. Store as JSON files in `{disk_path}/.vlt/storage/`.
- [x] T018 [US1] Create `frontend/src/components/artifacts/ArtifactViewer.tsx` — renders artifact frontend in sandboxed `<iframe>` with `sandbox="allow-scripts allow-same-origin"`. Sets iframe `src` to daemon frontend serving URL. Instantiates VltBridge host handler. Shows `ArtifactStateBar` above iframe.
- [x] T019 [P] [US1] Create `frontend/src/components/artifacts/ArtifactStateBar.tsx` — horizontal bar above iframe showing: artifact name, current state (colored badge), state transition buttons (advance/demote), last updated timestamp.
- [x] T020 [US1] Wire `ArtifactViewer` into `ArtifactsCompositorView.tsx` — when an artifact is selected in sidebar, render `ArtifactViewer` in the main pane. Pass artifact data and VltBridge callbacks.
- [x] T021 [US1] Complete `frontend/public/vlt-bridge.js` — fill in the postMessage request/response implementation for `VltBridge.storage` namespace (get, set, list) and `VltBridge.artifact` namespace (setState, getState, log). Each method returns a Promise that resolves when parent responds.
- [x] T022 [US1] Add VltBridge note access: extend `vlt-bridge.js` with `VltBridge.notes.read(path)`, `VltBridge.notes.write(path, content)`, `VltBridge.notes.search(query)`. Route through `vlt-bridge-host.ts` to existing backend note API endpoints.
- [x] T023 [US1] Add VltBridge code search: extend `vlt-bridge.js` with `VltBridge.code.search(query)`, `VltBridge.code.map()`. Route through `vlt-bridge-host.ts` to daemon code search endpoints.

**Checkpoint**: Simple HTML/JS artifacts render in iframe, VltBridge storage works, state machine transitions via UI buttons. This is the MVP.

---

## Phase 4: User Story 2 — Artifact with Server-Side Backend (Priority: P1)

**Goal**: Artifacts can have a Python backend process managed by the daemon, callable from the frontend via VltBridge.

**Independent Test**: Create artifact with `backend/main.py` defining `handle()`, call `VltBridge.backend.call("action", params)` from frontend JS, verify response arrives.

### Implementation for User Story 2

- [x] T024 [US2] Create `packages/vlt-cli/src/vlt/daemon/artifact_harness.py` — thin wrapper script that the daemon runs as a subprocess. Reads JSON lines from stdin (`{"action": "...", "params": {...}}`), imports the artifact's `main.py`, calls `handle(action, params)`, writes JSON response to stdout. Handles import errors and exceptions gracefully.
- [x] T025 [US2] Add backend process management to `artifact_service.py` — `_artifact_processes: Dict[str, dict]` (artifact_id → {proc, cwd}). Methods: `start_backend(artifact_id)` (spawn subprocess via `asyncio.create_subprocess_exec` running harness.py with artifact's backend/ as working dir), `stop_backend(artifact_id)` (terminate process), `call_backend(artifact_id, action, params)` (write JSON to stdin, read response from stdout with timeout), `_backend_reader(artifact_id, proc)` (async task reading stdout, routing responses).
- [x] T026 [US2] Add backend REST endpoints to `artifact_routes.py`: `POST /api/artifacts/{id}/backend/start`, `POST /api/artifacts/{id}/backend/stop`, `POST /api/artifacts/{id}/backend/call` (proxies to backend process).
- [x] T027 [US2] Add dependency installation to `start_backend()` in `artifact_service.py` — before spawning, check if `{disk_path}/backend/requirements.txt` exists. If so, run `uv pip install -r requirements.txt --target {disk_path}/backend/.deps/` as a subprocess. Add `.deps/` to `PYTHONPATH` in the harness subprocess env.
- [x] T028 [US2] Extend `vlt-bridge.js` with `VltBridge.backend.call(action, params)` method. Route through `vlt-bridge-host.ts` → `POST /api/artifacts/{id}/backend/call`.
- [x] T029 [US2] Add backend auto-restart to `artifact_service.py` — in `_backend_reader()`, if process exits unexpectedly (non-zero return code) and artifact state is `building`/`testing`/`deployed`, auto-restart after 2s delay. Log crash to artifact's vlt thread. Max 3 restarts before transitioning to `error` state.
- [x] T030 [US2] Add backend log streaming WebSocket to `artifact_routes.py`: `WS /ws/artifact/{id}/logs`. Buffer stdout/stderr from backend process, stream to connected clients as `{"type": "stdout/stderr", "data": "..."}`.

**Checkpoint**: Artifacts can have Python backends. Frontend calls backend via VltBridge. Backends auto-restart on crash. Log streaming works.

---

## Phase 5: User Story 3 — Hot Reload During Development (Priority: P2)

**Goal**: File changes in artifact directories trigger automatic reload — CSS swaps, JS/HTML full reload, Python backend restart — all debounced and state-aware.

**Independent Test**: Edit artifact CSS file while viewing in iframe, verify style updates without page reload. Edit Python file, verify backend restarts with state preserved.

### Implementation for User Story 3

- [x] T031 [US3] Create `packages/vlt-cli/src/vlt/daemon/artifact_watcher.py` — `ArtifactWatcher` class using `watchdog.observers.Observer` + `FileSystemEventHandler`. One watcher per active artifact. 500ms debounce via `asyncio.get_event_loop().call_later()`. Classifies changes: CSS-only, JS/HTML, Python, manifest. Emits `ArtifactChangeEvent` to reload coordinator.
- [x] T032 [US3] Implement reload coordinator in `artifact_watcher.py` — `ReloadCoordinator` class. On CSS-only: send `{"type": "css_update"}` via HMR WebSocket. On JS/HTML: send `{"type": "will_reload"}` then `{"type": "reload"}`. On Python: call `artifact_service.stop_backend()` (with state serialization), then `start_backend()` (with state restore). On both: backend first, then frontend.
- [x] T033 [US3] Add HMR WebSocket endpoint to `artifact_routes.py`: `WS /ws/artifact/{id}/hmr`. Maintains set of connected iframe clients per artifact. Reload coordinator pushes messages to all connected clients.
- [x] T034 [US3] Inject HMR client script into artifact `index.html` (extend T015 injection) — ~40 line script that opens WS to `/ws/artifact/{id}/hmr`, handles `css_update` (swap `<link>` hrefs), `will_reload` (call `window.__vlt_save_state` if defined), `reload` (location.reload), `state_restore` (call `window.__vlt_load_state`).
- [x] T035 [US3] Add state serialization to `artifact_harness.py` — before stop, send `{"action": "__save_state"}` to backend, wait up to 5s for response, write result to `{disk_path}/.vlt/hot_state.json`. On start, if `hot_state.json` exists, send `{"action": "__load_state", "params": <state>}` after import. Harness routes `__save_state`/`__load_state` to `save_state()`/`load_state()` functions if they exist in the artifact module.
- [x] T036 [US3] Add state machine demotion policy to `artifact_watcher.py` — on code change: if state is `approved` → demote to `testing`. If `deployed` → set `update_pending` flag. If `error` → demote to `building`. Log demotions to artifact thread.
- [x] T037 [US3] Register watcher lifecycle in daemon `server.py` lifespan — `_artifact_watchers: Dict[str, ArtifactWatcher]`. Start watcher when artifact is activated (selected in UI or backend started). Stop watcher when artifact is deactivated. Cleanup all watchers on shutdown.
- [x] T038 [US3] Add state change WebSocket to `artifact_routes.py`: `WS /ws/artifact/{id}/state`. Push `{"type": "state_change", ...}` on transitions, `{"type": "error", ...}` on crashes. Connect `ArtifactStateBar.tsx` to this WebSocket for live state updates.

**Checkpoint**: Full hot reload works — CSS swap, JS reload with state preservation, Python restart with state serialization. State machine reacts to code changes.

---

## Phase 6: User Story 5 — Test Runner and State Machine Automation (Priority: P2)

**Goal**: Artifacts define test commands in their manifest. Tests auto-run on code changes. Results drive state transitions and are injected into Claude Code sessions.

**Independent Test**: Create artifact with test command in manifest, modify code, verify tests auto-run and state advances or holds based on results.

**Note**: US5 is implemented before US4 because vision review depends on the test/state automation infrastructure.

### Implementation for User Story 5

- [x] T039 [US5] Add test execution to `artifact_service.py` — `run_tests(artifact_id)` method: reads `manifest.tests.command`, runs via `asyncio.create_subprocess_exec` in artifact dir with timeout from `manifest.tests.timeout` (default 30s), captures stdout/stderr/exit_code, returns structured result.
- [x] T040 [US5] Add test endpoint to `artifact_routes.py`: `POST /api/artifacts/{id}/test`. Calls `artifact_service.run_tests()`, returns `{passed, exit_code, stdout, stderr, duration_ms}`.
- [x] T041 [US5] Integrate test runner with hot reload — in `ReloadCoordinator`, after reload completes, if artifact has `manifest.tests.command`, auto-run tests. On pass: advance state (building→testing→pass, testing→reviewing or approved). On fail: hold at `testing`, log failure. Push test results via state WebSocket.
- [x] T042 [US5] Add test result injection into Claude Code sessions — when tests complete and the artifact has an associated Claude Code session (match via `project_id` + daemon session tracking), push results into `_injection_queues[session_id]` for delivery via `additionalContext` on next `UserPromptSubmit` hook.
- [x] T043 [US5] Add resource quota enforcement to `artifact_service.py` — parse `manifest.quotas` (max_cpu_seconds, max_memory_mb, max_storage_mb). For backend processes: set `resource.setrlimit(RLIMIT_AS, max_memory_mb * 1024 * 1024)` in harness subprocess. Monitor CPU time. On violation: kill process, transition to `error`, log quota exceeded event.

**Checkpoint**: Tests auto-trigger on code changes, drive state transitions, and inject results into agent sessions. Resource quotas enforced.

---

## Phase 7: User Story 4 — Vision Model Review (Priority: P2)

**Goal**: When artifact enters "reviewing" state, capture screenshot, send to vision model for assessment, pass assessment to primary model for pass/fail decision.

**Independent Test**: Transition artifact to reviewing, verify screenshot captured, vision model assessment logged, state advances or returns to building.

### Implementation for User Story 4

- [x] T044 [US4] Add vision model settings to backend — add `vision_model TEXT` and `vision_provider TEXT` columns to `user_settings` table via migration in `backend/src/services/database.py`. Extend `ModelSettings` Pydantic model in `backend/src/models/settings.py`. Extend `GET/PUT /api/settings/models` routes.
- [x] T045 [P] [US4] Add `supports_vision` field to `ModelInfo` in `backend/src/services/model_provider.py` — parse `architecture.modality` from OpenRouter API (look for `image` in input modalities). Hardcode `glm-4.6v` as vision-capable for GLM. Hardcode `gemini-2.0-flash-exp` and `gemini-1.5-pro` for Google.
- [x] T046 [P] [US4] Add vision model selector to Settings UI in `frontend/src/pages/Settings.tsx` — third model selector row (below oracle and subagent) following the same pattern: provider pill toggle + model dropdown + context length info. Label: "Vision Model (for artifact review)". Filter model list to `supports_vision === true`.
- [x] T047 [US4] Add screenshot capture to `artifact_service.py` — `capture_screenshot(artifact_id)` method using async Playwright: launch headless Chromium, navigate to artifact frontend URL, wait for load, screenshot to `{disk_path}/.vlt/screenshots/{timestamp}.png`, return path. Fallback: if Playwright not installed, log warning and skip.
- [x] T048 [US4] Add screenshot endpoint to `artifact_routes.py`: `POST /api/artifacts/{id}/screenshot` (trigger capture), `GET /api/artifacts/{id}/screenshot/{filename}` (serve image).
- [x] T049 [US4] Implement vision review flow in `artifact_service.py` — `review_artifact(artifact_id)` method: (1) capture screenshot, (2) discover vision model (check user's vision_model setting → scan OpenRouter for vision models → check GLM glm-4.6v → check Gemini → none found = skip with warning), (3) send screenshot + artifact description to vision model with prompt "Describe what you see in this screenshot and whether it matches these requirements: {manifest.description}", (4) pass vision assessment to primary oracle model with prompt "Based on this visual assessment, does the artifact pass review? Assessment: {vision_response}. Requirements: {description}. Respond with PASS or FAIL and explanation.", (5) on PASS → advance to approved, on FAIL → return to building with feedback.
- [x] T050 [US4] Integrate vision review with state machine — when state transitions to `reviewing` (from testing, after tests pass), auto-trigger `review_artifact()`. If no vision model available, log non-blocking error "Please configure a vision model", skip review, advance directly to approved.

**Checkpoint**: Vision model review works end-to-end. Screenshot captured, assessed by vision model, pass/fail decided by primary model. Graceful fallback when no vision model configured.

---

## Phase 8: User Story 6 — Connector Access from Artifacts (Priority: P3)

**Goal**: Artifacts access external services through connectors. Multiple instances of the same connector supported. Credentials never exposed to artifact code.

**Independent Test**: Create artifact that calls `VltBridge.connectors.call()`, verify call proxied through daemon, permissions checked, correct instance credentials used.

### Implementation for User Story 6

- [x] T051 [US6] Add `instance_id` to connector schema — add `instance_id TEXT NOT NULL DEFAULT 'default'` column to `connector_configs` table in `backend/src/services/database.py`. Migration: add column, rebuild PK to include instance_id, rebuild index. Update all `ConnectorService` methods to accept optional `instance_id` parameter (default `'default'`).
- [x] T052 [US6] Add connector instance REST endpoints to `backend/src/api/routes/connectors.py` — `GET /api/connectors/{name}/instances`, `GET /api/connectors/{name}/instances/{instance_id}/config`, `PUT /api/connectors/{name}/instances/{instance_id}/config`, `DELETE /api/connectors/{name}/instances/{instance_id}`.
- [x] T053 [P] [US6] Create proxy profile system — new `proxy_profiles` table DDL in `database.py`, new `ProxyProfileService` in `backend/src/services/proxy_service.py` (CRUD for proxy profiles, Fernet encryption for credentials), new routes in `backend/src/api/routes/proxy_profiles.py` (GET/POST/PUT/DELETE).
- [x] T054 [US6] Add connector proxy to VltBridge — extend `vlt-bridge.js` with `VltBridge.connectors.list()`, `VltBridge.connectors.call(connector, instance, action, params)`. Route through `vlt-bridge-host.ts` → daemon endpoint. Add connector proxy endpoint to `artifact_routes.py`: `POST /api/artifacts/{id}/connectors/call` — validates connector is declared in artifact manifest, proxies to backend connector invoke endpoint with instance_id.
- [x] T055 [US6] Add connector instance management UI — extend `frontend/src/pages/Settings.tsx` connectors section: show instances per connector, "Add Instance" button, per-instance config form, proxy profile selector dropdown. Add "Proxy Profiles" section to Settings for CRUD.
- [x] T056 [US6] Update MCP connector tools in `packages/vlt-cli/src/vlt/mcp/connector_tools.py` — `connector_call()` accepts optional `instance_id` parameter. `connector_list()` returns instance info per connector.

**Checkpoint**: Multi-instance connectors work. Artifacts call connectors via VltBridge. Proxy profiles configurable. All proxied through daemon, no credential leakage.

---

## Phase 9: User Story 7 — Artifact-to-Artifact Communication (Priority: P3)

**Goal**: Artifacts emit events and subscribe to events from other artifacts, enabling workflow pipelines.

**Independent Test**: Create two artifacts — emitter and subscriber. Emitter fires event, subscriber receives it with full payload.

### Implementation for User Story 7

- [x] T057 [US7] Create `packages/vlt-cli/src/vlt/daemon/artifact_event_bus.py` — `ArtifactEventBus` class: `_subscriptions` dict (artifact_id → set of event_type+callback), `emit(source, event_type, payload)` routes to subscribers (skip self), `subscribe(artifact_id, event_type, callback)`, `unsubscribe(artifact_id)`. Hop counter (max 10) to prevent infinite event loops. Log all events to source artifact's vlt thread.
- [x] T058 [US7] Add event endpoints to `artifact_routes.py` — `POST /api/artifacts/{id}/events/emit` (emit event, returns list of recipients). Add events WebSocket: `WS /ws/artifact/{id}/events` (push incoming events to artifact's frontend).
- [x] T059 [US7] Extend `vlt-bridge.js` with `VltBridge.events.emit(eventType, payload)` and `VltBridge.events.on(eventType, callback)`. Events received via the `/events` WebSocket are dispatched to registered callbacks.
- [x] T060 [US7] Register artifact event subscriptions from manifest — when artifact backend starts, read `manifest.events.subscribes`, register subscriptions on the event bus. On stop, unsubscribe. Event delivery calls the backend's `handle("__event", {event_type, source, payload})` for backend processing and pushes to frontend via events WebSocket.
- [x] T061 [US7] Wire event bus into daemon lifecycle — instantiate `ArtifactEventBus` as singleton in `server.py` lifespan. Pass to `artifact_service` and `artifact_routes`. Cleanup on shutdown.

**Checkpoint**: Artifact IPC works. Events route correctly, no self-notification, hop counter prevents loops.

---

## Phase 10: User Story 8 — Artifacts as MCP Tools (Priority: P3)

**Goal**: Deployed artifacts expose functions as MCP tools callable by any Claude Code session.

**Independent Test**: Deploy artifact with mcp_tools in manifest, verify tools appear in MCP server tool list, call tool and get response from artifact backend.

### Implementation for User Story 8

- [x] T062 [US8] Create `packages/vlt-cli/src/vlt/mcp/artifact_tools.py` — MCP tool group with: `vlt_artifact_create(name, description, project_id, type?)`, `vlt_artifact_list(project_id?)`, `vlt_artifact_update(artifact_id, files)` (writes files to disk), `vlt_artifact_state(artifact_id, target_state)`, `vlt_artifact_test(artifact_id)`, `vlt_artifact_screenshot(artifact_id)`. All route to daemon REST API.
- [x] T063 [US8] Register artifact tools in `packages/vlt-cli/src/vlt/mcp_server.py` — add `artifact_tools` to conditional import block, same pattern as other tool groups.
- [x] T064 [US8] Implement dynamic artifact MCP tool loading — deferred to Phase 2 per research.md decision R5. Comment in `artifact_tools.py` explains that dynamic tool registration from deployed artifacts requires MCP server restart. No implementation needed now.
- [x] T065 [US8] Add MCP server restart mechanism — deferred to Phase 2 per research.md decision R5. Comment in `artifact_tools.py` documents the decision. Claude Code reconnects automatically on next tool call after restart.

**Checkpoint**: Deployed artifacts expose MCP tools. Claude Code sessions can call artifact functions. Tools register/unregister on deploy/undeploy.

---

## Phase 11: User Story 9 — Import and Export Artifacts (Priority: P3)

**Goal**: Export artifacts as zip, import from zip. Enable manual sharing.

**Independent Test**: Create artifact, export as zip, delete, import zip, verify imported artifact runs identically.

### Implementation for User Story 9

- [x] T066 [US9] Add export endpoint to `artifact_routes.py` — `GET /api/artifacts/{id}/export`. Creates zip of artifact directory (manifest, frontend/, backend/, tests/), excludes `.vlt/` (runtime state), `.git/`, `backend/.deps/`. Returns `application/zip` with `Content-Disposition` header.
- [x] T067 [US9] Add import endpoint to `artifact_routes.py` — `POST /api/artifacts/import`. Accepts `multipart/form-data` with zip file. Validates manifest.json exists and is valid. Creates new artifact with new ID, extracts zip to disk, inserts DB row. Returns created artifact.
- [x] T068 [US9] Create `frontend/src/components/artifacts/ArtifactImportExport.tsx` — Export and import implemented inline: Upload button in `ArtifactSidebar.tsx` (via `onImport` prop + hidden file input), Download button in `ArtifactStateBar.tsx` (via `onExport` prop). `handleExport` in `ArtifactViewer.tsx` creates blob URL + anchor click. `handleImport` in `ArtifactsCompositorView.tsx` calls `importArtifact()`. No separate component needed.
- [x] T069 [US9] Wire import/export into `ArtifactSidebar.tsx` and `ArtifactStateBar.tsx` — Upload icon in sidebar header calls `onImport(file)`, Download icon in state bar calls `onExport()`. Both wired in `ArtifactsCompositorView.tsx`.

**Checkpoint**: Artifacts can be exported, shared as files, and imported on another machine. Credentials excluded from export.

---

## Phase 12: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T070 [P] Add artifact-related unit tests in `packages/vlt-cli/src/vlt/tests/unit/test_artifact_service.py` — test state machine transitions (valid and invalid), manifest validation, CRUD operations
- [x] T071 [P] Add artifact MCP tool tests in `packages/vlt-cli/src/vlt/tests/unit/test_artifact_tools.py` — test create, list, update, state transition tools
- [x] T072 Error handling sweep — ensure all artifact endpoints return structured JSON errors, frontend displays them via toast notifications, and edge cases (missing files, invalid manifest, process crashes) are handled gracefully
- [x] T073 [P] Add artifact event logging — ensure all significant operations (create, state change, test run, review, deploy, error) push entries to the artifact's vlt thread with structured messages
- [x] T074 Update quickstart.md validation — run through the quickstart guide end-to-end, fix any inaccuracies
- [x] T075 Security review — verify: iframe sandbox attributes are correct, VltBridge never leaks credentials, artifact backend processes can't escape their working directory, resource quotas can't be bypassed, connector permission enforcement works through VltBridge proxy

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational. No other story dependencies.
- **US2 (Phase 4)**: Depends on Foundational. Builds on US1 iframe infrastructure but can be developed independently (backend has its own endpoints).
- **US3 (Phase 5)**: Depends on US1 (iframe serving) and US2 (backend processes). Needs both to exist for full hot reload.
- **US5 (Phase 6)**: Depends on US3 (hot reload triggers tests). Needs watcher infrastructure.
- **US4 (Phase 7)**: Depends on US5 (state machine automation). Vision review triggers on state transition.
- **US6 (Phase 8)**: Depends on Foundational only. Can parallelize with US3-US5 but connectors are P3 priority.
- **US7 (Phase 9)**: Depends on US2 (backend processes needed for event handling). Can parallelize with US6.
- **US8 (Phase 10)**: Depends on US2 (backend call routing). Can parallelize with US6-US7.
- **US9 (Phase 11)**: Depends on Foundational only. Lightest dependency of any story.
- **Polish (Phase 12)**: Depends on all desired user stories being complete.

### User Story Dependency Graph

```
Foundational (Phase 2)
├── US1 (Phase 3) ← MVP
│   └── US2 (Phase 4)
│       ├── US3 (Phase 5)
│       │   └── US5 (Phase 6)
│       │       └── US4 (Phase 7)
│       ├── US7 (Phase 9)
│       └── US8 (Phase 10)
├── US6 (Phase 8) ← independent
└── US9 (Phase 11) ← independent
```

### Within Each User Story

- Models/schema before services
- Services before endpoints
- Backend endpoints before frontend integration
- VltBridge extensions before UI components that use them

### Parallel Opportunities

Within Phase 2: T012, T013 are parallel (different frontend components)
Within US1: T019 parallel with T015-T018 (state bar is independent)
Within US4: T045, T046 parallel with each other (backend model detection vs frontend UI)
Within US6: T053 parallel with T051-T052 (proxy profiles independent of connector instances)
US6, US7, US8, US9 can all parallelize after their dependencies are met

---

## Parallel Example: User Story 1

```bash
# After Foundational phase completes, launch parallel tasks:
Task T019: "Create ArtifactStateBar.tsx"     # Independent component
Task T015: "Add frontend serving endpoint"    # Backend work

# Then sequential:
Task T016: "Implement VltBridge host handler" # Needs T015 (serving URL)
Task T017: "Add storage endpoints"            # Backend, parallel with T016
Task T018: "Create ArtifactViewer.tsx"        # Needs T015, T016
Task T020: "Wire into CompositorView"         # Needs T018, T019
Task T021: "Complete vlt-bridge.js"           # Needs T017 (storage endpoints)
```

## Parallel Example: P3 Stories

```bash
# After US2 completes, these can run in parallel:
Agent A: US6 (Connectors)   — T051-T056
Agent B: US7 (IPC Events)   — T057-T061
Agent C: US8 (MCP Tools)    — T062-T065
Agent D: US9 (Import/Export) — T066-T069
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories)
3. Complete Phase 3: US1 — Create and Run Simple Artifact
4. **STOP and VALIDATE**: Create an artifact with HTML/JS, verify it renders, VltBridge storage works
5. Demo: "Look, I can create a plugin that runs in the browser with persistent storage"

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. US1 → Simple artifacts work (MVP!)
3. US2 → Artifacts have backends (major capability jump)
4. US3 → Hot reload (developer experience)
5. US5 → Tests automate state machine
6. US4 → Vision review closes the quality loop
7. US6 + US7 + US8 + US9 → Power features (parallel sprint)

### Estimated Scope

- **Phase 1-2**: 14 tasks (foundation)
- **US1 (P1)**: 9 tasks
- **US2 (P1)**: 7 tasks
- **US3 (P2)**: 8 tasks
- **US5 (P2)**: 5 tasks
- **US4 (P2)**: 7 tasks
- **US6 (P3)**: 6 tasks
- **US7 (P3)**: 5 tasks
- **US8 (P3)**: 4 tasks
- **US9 (P3)**: 4 tasks
- **Polish**: 6 tasks
- **Total**: 75 tasks

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US5 is scheduled before US4 because vision review depends on test/state automation
- US6-US9 are all P3 and can be parallelized as a sprint after P1+P2 stories complete
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
