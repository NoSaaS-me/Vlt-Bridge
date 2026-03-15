# Tasks: Composio Connection Vault

**Input**: Design documents from `/specs/024-composio-connection-vault/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks grouped by user story. Each story is independently testable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1, US2, US3)
- Exact file paths included in descriptions

---

## Phase 1: Setup

**Purpose**: Fix critical bugs and add shared infrastructure

- [ ] T001 Fix `disconnect()` — replace `connected_accounts.delete()` with raw HTTP `DELETE /v1/connectedAccounts/{id}` via `toolset.client.http.delete()` in `packages/vlt-connectors/src/vlt_connectors/service/composio.py`
- [ ] T002 [P] Create Pydantic request/response models (`ConnectRequest`, `ConnectResponse`, `AuthFieldInfo`, `AuthSchemeInfo`, `AppAuthInfo`, `ConnectionInfo`) in `backend/src/models/composio.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Service-layer methods that all user stories depend on

- [ ] T003 Add `app_auth_info(app_name) -> dict` method to `ComposioService` — query `app.testConnectors` and `auth_schemes.fields` to return `{has_managed_auth, primary_auth_mode, auth_schemes}` in `packages/vlt-connectors/src/vlt_connectors/service/composio.py`
- [ ] T004 Update `initiate_connection()` signature to accept `label`, `auth_mode`, `auth_config`, `connected_account_params`, `redirect_url` — set `use_composio_auth=False` when `auth_config` provided, return `{connection_id, redirect_url, status}` dict in `packages/vlt-connectors/src/vlt_connectors/service/composio.py`
- [ ] T005 Update `execute()` method to accept optional `connected_account_id` parameter and pass it to `toolset.execute_action()` in `packages/vlt-connectors/src/vlt_connectors/service/composio.py`

**Checkpoint**: Service layer supports all auth flows and connection routing. User story work can begin.

---

## Phase 3: User Story 1 — Managed + Custom OAuth Connection (Priority: P0)

**Goal**: Gmail (managed) connects as before. Twitter (custom) prompts for client_id/client_secret instead of 502 error.

**Independent Test**: Click Connect on Gmail — goes to OAuth redirect. Click Connect on Twitter — shows credential form, then goes to OAuth redirect. Both create working connections.

### Implementation for User Story 1

- [ ] T006 [US1] Add `GET /api/composio/{app_name}/auth-info` endpoint using `svc.app_auth_info()` in `backend/src/api/routes/composio_hub.py`
- [ ] T007 [US1] Update `POST /api/composio/connect/{app_name}` to accept `ConnectRequest` body — pass `auth_config`, `connected_account_params`, `label`, `auth_mode` to `svc.initiate_connection()`, return `ConnectResponse` in `backend/src/api/routes/composio_hub.py`
- [ ] T008 [P] [US1] Add `getAuthInfo(appName)` API function returning `AppAuthInfo` in `frontend/src/services/composio-hub.ts`
- [ ] T009 [P] [US1] Update `connectApp(appName)` to accept optional `ConnectRequest` body params in `frontend/src/services/composio-hub.ts`
- [ ] T010 [US1] Create `ComposioConnectDialog` component — on "Connect" click: calls `getAuthInfo`, if `has_managed_auth=false` shows credential form (label + integration_fields), if managed proceeds directly with optional label; submits to `connectApp` with body in `frontend/src/pages/ConnectorsPage.tsx`
- [ ] T011 [US1] Wire `ComposioConnectDialog` into `HubTab` — replace direct `handleConnect` call with dialog open, pass `onConnect` callback in `frontend/src/pages/ConnectorsPage.tsx`

**Checkpoint**: Managed OAuth apps (Gmail) work as before. Custom OAuth apps (Twitter) no longer 502 — user provides credentials in a form.

---

## Phase 4: User Story 2 — Connection Registry + Multi-Account (Priority: P1)

**Goal**: Users can have N connections per app with labels. Connections stored locally. Disconnect targets specific connection. Invoke routes to correct connection.

**Independent Test**: Connect Gmail twice (different accounts). Both connections show with labels. Disconnect one — the other stays. Invoke with `connection_id` routes correctly. Invoke without `connection_id` when 1 connection auto-selects; when >1 returns error listing connections.

### Implementation for User Story 2

- [ ] T012 [US2] Add `composio_connections` table DDL to both `DDL_STATEMENTS` and `MIGRATION_STATEMENTS` tuples in `backend/src/services/database.py`
- [ ] T013 [US2] Create `ComposioConnectionService` class with methods: `upsert()`, `list_for_user()`, `list_for_user_app()`, `get()`, `mark_revoked()`, `reconcile()` following `ConnectorService` pattern (raw sqlite3, `db.connect()`, upsert via ON CONFLICT) in `backend/src/services/composio_connections.py`
- [ ] T014 [US2] Update `POST /connect/{app_name}` to insert connection into local registry on success (using `ComposioConnectionService.upsert()`) in `backend/src/api/routes/composio_hub.py`
- [ ] T015 [US2] Update `GET /connected` to read from local registry with Composio reconciliation — call `svc.connected()` then `conn_svc.reconcile()` to sync, return enriched `ConnectionInfo` list in `backend/src/api/routes/composio_hub.py`
- [ ] T016 [US2] Add `DELETE /api/composio/connections/{connection_id}` endpoint — calls `svc.disconnect_by_id()` (raw HTTP) + `conn_svc.mark_revoked()` in `backend/src/api/routes/composio_hub.py`
- [ ] T017 [US2] Add `disconnect_by_id(connection_id)` method to `ComposioService` using raw HTTP `DELETE /v1/connectedAccounts/{id}` in `packages/vlt-connectors/src/vlt_connectors/service/composio.py`
- [ ] T018 [US2] Update `DELETE /api/composio/{app_name}` to disconnect ALL connections for the app (iterate local registry) and return `{disconnected_count}` in `backend/src/api/routes/composio_hub.py`
- [ ] T019 [US2] Update `POST /api/composio/{app_name}/invoke` to accept optional `connection_id` in `InvokeRequest` — if provided pass to `svc.execute()`, if omitted + 1 connection auto-select, if omitted + >1 return 400 with connection list in `backend/src/api/routes/composio_hub.py`
- [ ] T020 [P] [US2] Add `listConnections()`, `disconnectConnection(connectionId)` API functions in `frontend/src/services/composio-hub.ts`
- [ ] T021 [US2] Update `ComposioApp` type to include `connections: {connection_id, label, status}[]` array, update `listApps` response handling to merge connection data in `frontend/src/services/composio-hub.ts`
- [ ] T022 [US2] Update `AppCard` to show connection list for connected apps — each connection shows label + truncated ID + individual [x] disconnect button; add "+ Add Connection" button that opens `ComposioConnectDialog` in `frontend/src/pages/ConnectorsPage.tsx`
- [ ] T023 [US2] Update `HubTab` to pass connections data through and handle per-connection disconnect via `disconnectConnection(connectionId)` in `frontend/src/pages/ConnectorsPage.tsx`

**Checkpoint**: Users can connect multiple accounts per app. Each connection has a label. Individual connections can be disconnected. Invoke routes to correct connection.

---

## Phase 5: User Story 3 — MCP/CLI Connection Routing (Priority: P2)

**Goal**: AI agents can discover and select specific connections via MCP tools and CLI.

**Independent Test**: Call `connector_list` — Composio connectors show `connections` array. Call `connector_call` with `connection_id` — routes to correct account. Call without `connection_id` when >1 connection — returns MULTIPLE_CONNECTIONS error with options.

### Implementation for User Story 3

- [ ] T024 [US3] Add `connection_id: Optional[str] = None` parameter to `connector_call` MCP tool — pass through to invoke endpoint in `packages/vlt-cli/src/vlt/mcp/connector_tools.py`
- [ ] T025 [US3] Update `connector_list` to include `connections` array per Composio connector — fetch from `/api/composio/connected` and group by app_name in `packages/vlt-cli/src/vlt/mcp/connector_tools.py`
- [ ] T026 [US3] Update `connector_call` docstring — document `connection_id` parameter, auto-select behavior, MULTIPLE_CONNECTIONS error response in `packages/vlt-cli/src/vlt/mcp/connector_tools.py`
- [ ] T027 [P] [US3] Update `connector_list` docstring — document `connections` array in return schema in `packages/vlt-cli/src/vlt/mcp/connector_tools.py`
- [ ] T028 [US3] Add `--connection-id` option to `vlt connectors call` command — pass through to invoke endpoint in `packages/vlt-cli/src/vlt/main.py`
- [ ] T029 [US3] Update CONNECTORS section in MCP server instructions — document connection routing, `connection_id` parameter, multi-account workflow in `packages/vlt-cli/src/vlt/mcp_server.py`

**Checkpoint**: Agents can discover which connections exist, select specific ones, and get helpful errors when disambiguation is needed.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Quality, validation, and cleanup

- [ ] T030 [P] Update `GET /api/composio/apps` to include `has_managed_auth` and `primary_auth_mode` per app so frontend can show auth type badge in `backend/src/api/routes/composio_hub.py`
- [ ] T031 [P] Add unit tests for `ComposioConnectionService` — test upsert, list, reconcile, mark_revoked in `backend/tests/unit/test_composio_connections.py`
- [ ] T032 Verify all 6 user scenarios from spec.md pass manual testing — document results
- [ ] T033 Update CLAUDE.md with 024-composio-connection-vault in Recent Changes section in `/mnt/sda1/Projects/00Tooling/Vlt-Bridge/CLAUDE.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on T001 (disconnect fix). T003-T005 can start after T001.
- **US1 (Phase 3)**: Depends on T003, T004 (service methods). Backend (T006-T007) before frontend (T008-T011).
- **US2 (Phase 4)**: Depends on T004, T005 (service methods), T006-T007 (backend routes from US1). Can start backend (T012-T019) in parallel with US1 frontend work.
- **US3 (Phase 5)**: Depends on T019 (invoke with connection_id). Can start after US2 backend is done.
- **Polish (Phase 6)**: Depends on US1+US2 at minimum.

### User Story Dependencies

- **US1 (P0)**: Depends on Foundational only — no dependency on other stories
- **US2 (P1)**: Depends on US1 backend (connect endpoint returns connection_id)
- **US3 (P2)**: Depends on US2 backend (invoke accepts connection_id, connected endpoint returns connections)

### Within Each User Story

- Backend models/services before routes
- Routes before frontend API client
- API client before UI components
- Docstrings after implementation

### Parallel Opportunities

- T002 (models) runs in parallel with T001 (bug fix)
- T008-T009 (frontend API client) run in parallel with T006-T007 (backend routes)
- T020 (frontend API) runs in parallel with T012-T019 (backend registry)
- T024-T027 (MCP tools) can be split across files but connector_tools.py is a single file — serialize
- T030-T031 (polish tasks) run in parallel with each other

---

## Parallel Example: User Story 2

```bash
# Backend tasks can be split:
#   T012 (DDL) → T013 (service) → T14-T19 (routes) — sequential
# Frontend tasks start once T15 (GET /connected) is done:
#   T020 (API client) → T021-T023 (components) — sequential

# So the parallelism is:
#   Backend: T12 → T13 → T14-T19
#   Frontend: ----wait for T15---- → T20 → T21 → T22 → T23
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: Foundational (T003-T005)
3. Complete Phase 3: US1 (T006-T011)
4. **STOP and VALIDATE**: Gmail connects as before. Twitter shows credential form instead of 502.
5. Ship — all existing users unblocked.

### Incremental Delivery

1. Setup + Foundational → disconnect bug fixed, service ready
2. US1 → custom auth works → **Ship MVP**
3. US2 → multi-account + registry → Ship
4. US3 → MCP/CLI routing → Ship
5. Each story adds value without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- The delete bug (T001) is the most critical — it blocks all disconnect flows
- US1 can ship independently as the MVP — fixes the 502 error
- US2 is the heaviest phase — consider splitting backend (T012-T019) and frontend (T020-T023) across sessions
- US3 is lightweight — mostly docstring and parameter additions
