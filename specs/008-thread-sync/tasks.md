# Tasks: Thread Sync from CLI to Server

**Input**: Design documents from `/specs/008-thread-sync/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/threads-api.yaml

**Tests**: Included for backend components per constitution (Test-Backed Development)

**Organization**: Tasks grouped by user story. Note: User Story 2 (Web UI) is deferred to a future UI refactor.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `backend/src/`, `backend/tests/`
- **CLI**: `packages/vlt-cli/src/vlt/`

---

## Phase 1: Setup

**Purpose**: Database schema and shared infrastructure

- [ ] T001 Add thread tables DDL to backend/src/services/database.py (threads, thread_entries, thread_sync_status)
- [ ] T002 Add FTS5 virtual table for thread_entries_fts in backend/src/services/database.py
- [ ] T003 [P] Create Pydantic models in backend/src/models/thread.py (Thread, ThreadEntry, SyncRequest, SyncResponse, SyncStatus)
- [ ] T004 Verify database migration runs successfully on backend startup

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core services that MUST be complete before user stories

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 Create ThreadService class in backend/src/services/thread_service.py with database connection
- [ ] T006 Implement ThreadService.create_or_update_thread() in backend/src/services/thread_service.py
- [ ] T007 Implement ThreadService.add_entries() in backend/src/services/thread_service.py
- [ ] T008 Implement ThreadService.get_thread() in backend/src/services/thread_service.py
- [ ] T009 Implement ThreadService.list_threads() in backend/src/services/thread_service.py
- [ ] T010 Implement ThreadService.get_sync_status() in backend/src/services/thread_service.py
- [ ] T011 Implement ThreadService.search_threads() for FTS5 queries in backend/src/services/thread_service.py
- [ ] T012 [P] Write pytest tests for ThreadService in backend/tests/unit/test_thread_service.py

**Checkpoint**: ThreadService ready - user story implementation can now begin

---

## Phase 3: User Story 1 - AI Agent Syncs Work Context (Priority: P1)

**Goal**: CLI automatically syncs thread entries to backend after `vlt thread push`

**Independent Test**: Run `vlt thread push "test"` and verify entry appears via `GET /api/threads/{id}`

### Backend API (US1)

- [ ] T013 Create threads router in backend/src/api/routes/threads.py with APIRouter prefix="/api/threads"
- [ ] T014 [US1] Implement POST /api/threads/sync endpoint in backend/src/api/routes/threads.py
- [ ] T015 [US1] Implement GET /api/threads endpoint (list) in backend/src/api/routes/threads.py
- [ ] T016 [US1] Implement GET /api/threads/{thread_id} endpoint in backend/src/api/routes/threads.py
- [ ] T017 [US1] Implement GET /api/threads/{thread_id}/status endpoint in backend/src/api/routes/threads.py
- [ ] T018 [US1] Implement DELETE /api/threads/{thread_id} endpoint in backend/src/api/routes/threads.py
- [ ] T019 [US1] Register threads router in backend/src/api/main.py
- [ ] T020 [P] [US1] Write pytest tests for threads API in backend/tests/unit/test_threads_api.py

### CLI Sync Client (US1)

- [ ] T021 [US1] Add sync_token field to Settings in packages/vlt-cli/src/vlt/config.py
- [ ] T022 [US1] Create ThreadSyncClient class in packages/vlt-cli/src/vlt/core/sync.py with httpx async client
- [ ] T023 [US1] Implement ThreadSyncClient.sync_entries() HTTP POST to vault_url/api/threads/sync
- [ ] T024 [US1] Implement ThreadSyncClient.get_sync_status() HTTP GET from vault_url/api/threads/{id}/status
- [ ] T025 [US1] Add SyncQueueItem model and queue file handling in packages/vlt-cli/src/vlt/core/sync.py
- [ ] T026 [US1] Implement sync queue retry logic in packages/vlt-cli/src/vlt/core/sync.py
- [ ] T027 [US1] Hook sync into thread push command in packages/vlt-cli/src/vlt/main.py (after add_thought)
- [ ] T028 [US1] Add `vlt sync status` command in packages/vlt-cli/src/vlt/main.py
- [ ] T029 [US1] Add `vlt sync retry` command in packages/vlt-cli/src/vlt/main.py
- [ ] T030 [US1] Handle network errors gracefully with queue fallback in packages/vlt-cli/src/vlt/core/sync.py

**Checkpoint**: CLI can sync threads to backend. Test with `vlt thread push` and verify via API.

---

## Phase 4: User Story 3 - Oracle Answers Using Thread Context (Priority: P3)

**Goal**: Backend Oracle queries synced threads directly instead of subprocess

**Independent Test**: Ask Oracle "What did we discuss about X?" and verify thread citations in response

**Dependencies**: Requires US1 complete (threads must be synced to backend first)

### Thread Search API (US3)

- [ ] T031 [US3] Implement GET /api/threads/search endpoint in backend/src/api/routes/threads.py
- [ ] T032 [P] [US3] Write pytest test for search endpoint in backend/tests/unit/test_threads_api.py

### Oracle Integration (US3)

- [ ] T033 [US3] Create ThreadRetriever class in backend/src/services/thread_retriever.py
- [ ] T034 [US3] Implement ThreadRetriever.search() using ThreadService.search_threads() in backend/src/services/thread_retriever.py
- [ ] T035 [US3] Implement ThreadRetriever.format_citations() for source references in backend/src/services/thread_retriever.py
- [ ] T036 [US3] Remove subprocess vlt oracle calls from backend/src/services/oracle_bridge.py
- [ ] T037 [US3] Integrate ThreadRetriever into OracleBridge context assembly in backend/src/services/oracle_bridge.py
- [ ] T038 [US3] Update OracleBridge.ask_oracle_stream() to query local threads in backend/src/services/oracle_bridge.py
- [ ] T039 [US3] Add thread source type to SourceReference model in backend/src/models/oracle.py
- [ ] T040 [P] [US3] Write pytest tests for ThreadRetriever in backend/tests/unit/test_thread_retriever.py

**Checkpoint**: Oracle queries include thread context. Test by syncing threads then asking Oracle about thread content.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Error handling, validation, documentation

- [ ] T041 Add request validation (size limits, rate limiting) to /api/threads/sync in backend/src/api/routes/threads.py
- [ ] T042 Add concurrent write handling (row locking) to ThreadService in backend/src/services/thread_service.py
- [ ] T043 [P] Update quickstart.md with actual tested commands in specs/008-thread-sync/quickstart.md
- [ ] T044 [P] Add logging for sync operations in packages/vlt-cli/src/vlt/core/sync.py
- [ ] T045 Handle token expiry gracefully in CLI sync in packages/vlt-cli/src/vlt/core/sync.py
- [ ] T046 Run full integration test: push thread via CLI, query via Oracle, verify citations

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational
- **User Story 3 (Phase 4)**: Depends on User Story 1 (needs synced threads to query)
- **Polish (Phase 5)**: Depends on US1 and US3 complete

### Within Each Phase

- Models/DDL before services
- Services before API endpoints
- Core implementation before integration
- Tests can run in parallel with [P] marker

### Parallel Opportunities

**Setup Phase:**
```
T003 [P] can run in parallel with T001-T002
```

**Foundational Phase:**
```
T012 [P] (tests) can run in parallel with T005-T011 (but tests should fail initially)
```

**User Story 1:**
```
# Backend API tasks can run in parallel:
T014-T018 are sequential (same file)
T020 [P] (tests) can run in parallel

# CLI tasks can run in parallel after T021:
T022-T030 are mostly sequential (same files)
```

**User Story 3:**
```
T032 [P] and T040 [P] (tests) can run in parallel with implementation
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T012)
3. Complete Phase 3: User Story 1 (T013-T030)
4. **STOP and VALIDATE**: Test CLI sync works end-to-end
5. Deploy if thread sync alone provides value

### Full Feature (US1 + US3)

1. Complete Setup → Foundational → US1
2. Verify US1 with manual testing
3. Complete US3 (Oracle integration)
4. Run integration test (T046)
5. Complete Polish phase

### Suggested Order

```
Day 1: T001-T012 (Setup + Foundational)
Day 2: T013-T020 (Backend API)
Day 3: T021-T030 (CLI sync)
Day 4: T031-T040 (Oracle integration)
Day 5: T041-T046 (Polish + integration test)
```

---

## Notes

- User Story 2 (Web UI threads view) is **deferred** to future UI refactor
- [P] tasks can run in parallel if working on different files
- [US1] and [US3] labels track which story each task serves
- Backend uses raw SQLite (no ORM) per constitution
- CLI uses existing SQLAlchemy models - sync.py is new code
- Test with local backend first before remote deployment
