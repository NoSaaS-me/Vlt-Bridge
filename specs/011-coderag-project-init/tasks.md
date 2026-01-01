# Tasks: CodeRAG Project Integration

**Input**: Design documents from `/specs/011-coderag-project-init/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Backend routes require pytest tests per constitution. Frontend relies on manual verification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **CLI Package**: `packages/vlt-cli/src/vlt/`
- **Backend**: `backend/src/`
- **Frontend**: `frontend/src/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Database schema and shared service layer extensions

- [x] T001 Add `list_projects()` method to SqliteVaultService in `packages/vlt-cli/src/vlt/core/service.py`
- [x] T002 Add `has_coderag_index(project_id)` method to SqliteVaultService in `packages/vlt-cli/src/vlt/core/service.py`
- [x] T003 Add `JobStatus` enum to `packages/vlt-cli/src/vlt/core/models.py`
- [x] T004 Add `CodeRAGIndexJob` SQLAlchemy model to `packages/vlt-cli/src/vlt/core/models.py`
- [x] T005 Run alembic migration or update schema initialization for new table

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Add progress callback parameter to `CodeRAGIndexer.index_full()` in `packages/vlt-cli/src/vlt/core/coderag/indexer.py`
- [x] T007 Implement callback invocation at key progress points (file discovery, per-file completion, embedding generation)
- [x] T008 Create Pydantic models for CodeRAG API in `backend/src/models/coderag.py` (CodeRAGStatusResponse, InitCodeRAGRequest, InitCodeRAGResponse, JobStatusResponse)
- [x] T009 Create TypeScript types for CodeRAG in `frontend/src/types/coderag.ts`
- [x] T010 Create CodeRAG API service in `frontend/src/services/coderag.ts` with getCodeRAGStatus() and initCodeRAG()

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 & 4 - Interactive Init with Overwrite Protection (Priority: P1) 🎯 MVP

**Goal**: Enable developers to interactively initialize CodeRAG for a project with protection against accidental overwrites

**Independent Test**: Run `vlt coderag init` in a directory with source files, verify interactive project selection works, and that existing indexes cannot be overwritten without force flag

### Implementation for User Story 1 & 4

- [x] T011 [US1] Add `_interactive_project_selection()` helper function in `packages/vlt-cli/src/vlt/main.py`
- [x] T012 [US1] Implement numbered project list display with rich.console in `packages/vlt-cli/src/vlt/main.py`
- [x] T013 [US1] Add "Create new project" option to interactive selection in `packages/vlt-cli/src/vlt/main.py`
- [x] T014 [US1] Implement project creation via rich.prompt.Prompt in `packages/vlt-cli/src/vlt/main.py`
- [x] T015 [US4] Add overwrite detection check using `has_coderag_index()` in coderag_init command
- [x] T016 [US4] Implement warning message and confirmation prompt for existing index in `packages/vlt-cli/src/vlt/main.py`
- [x] T017 [US4] Add `--force` flag handling to bypass overwrite protection
- [x] T018 [US1] Update `coderag_init` command signature to call `_interactive_project_selection()` when `--project` not provided
- [x] T019 [US1] Add confirmation message with status check instructions after init starts
- [x] T020 [US1] Add pytest test for interactive project selection flow in `packages/vlt-cli/tests/test_coderag_init.py`

**Checkpoint**: Interactive init workflow complete with overwrite protection

---

## Phase 4: User Story 5 - Background Indexing with Daemon (Priority: P2)

**Goal**: Enable indexing to continue in background via daemon, surviving terminal closure

**Independent Test**: Start indexing, close terminal, open new terminal and verify indexing continues via status command

### Implementation for User Story 5

- [x] T021 [US5] Add `_queue_background_indexing()` helper function in `packages/vlt-cli/src/vlt/main.py`
- [x] T022 [US5] Implement job creation and insertion into `coderag_index_jobs` table
- [x] T023 [US5] Add `--background/--foreground` flag to `coderag_init` command (default: background)
- [x] T024 [US5] Add `_get_next_pending_job()` helper in `packages/vlt-cli/src/vlt/daemon/server.py`
- [x] T025 [US5] Implement `_run_indexing_job()` async function with progress updates in `packages/vlt-cli/src/vlt/daemon/server.py`
- [x] T026 [US5] Add `process_coderag_jobs()` background task to daemon lifespan in `packages/vlt-cli/src/vlt/daemon/server.py`
- [x] T027 [US5] Update job status to RUNNING when processing starts
- [x] T028 [US5] Update job status to COMPLETED or FAILED when processing ends
- [x] T029 [US5] Add progress percent calculation from files_processed/files_total

**Checkpoint**: Background indexing via daemon operational

---

## Phase 5: User Story 2 - CLI Progress Monitoring (Priority: P2)

**Goal**: Enable developers to check indexing progress via CLI status command

**Independent Test**: Start indexing, run `vlt coderag status` at various points, verify progress displayed

### Implementation for User Story 2

- [x] T030 [US2] Add `get_job_status()` method to service in `packages/vlt-cli/src/vlt/core/service.py`
- [x] T031 [US2] Add `get_active_job_for_project()` method in `packages/vlt-cli/src/vlt/core/service.py`
- [x] T032 [US2] Update `coderag status` command to show job progress in `packages/vlt-cli/src/vlt/main.py`
- [x] T033 [US2] Display files_processed / files_total with percentage
- [x] T034 [US2] Calculate and display time elapsed since started_at
- [x] T035 [US2] Calculate and display estimated time remaining based on progress rate
- [x] T036 [US2] Add `--json` flag for machine-readable output in status command
- [x] T037 [US2] Display completion summary when job status is COMPLETED

**Checkpoint**: CLI progress monitoring complete

---

## Phase 6: User Story 3 - Web UI Progress Monitoring (Priority: P2)

**Goal**: Enable developers to view indexing progress in web UI Settings page

**Independent Test**: Start indexing via CLI, open Settings page, verify progress bar updates

### Backend API Implementation

- [x] T038 [US3] Create `backend/src/api/routes/coderag.py` with router definition
- [x] T039 [US3] Implement `GET /api/coderag/status` endpoint calling OracleBridge
- [x] T040 [US3] Implement `POST /api/coderag/init` endpoint for web-triggered indexing
- [x] T041 [US3] Implement `GET /api/coderag/jobs/{job_id}` endpoint for job status
- [x] T042 [US3] Implement `POST /api/coderag/jobs/{job_id}/cancel` endpoint
- [x] T043 [US3] Register coderag router in `backend/src/api/main.py`
- [x] T044 [US3] Add OracleBridge method for `vlt coderag status --json` invocation
- [x] T045 [US3] Add pytest tests for coderag API routes in `backend/tests/unit/test_coderag_routes.py`

### Frontend Implementation

- [x] T046 [P] [US3] Add polling hook for CodeRAG status in `frontend/src/pages/Settings.tsx`
- [x] T047 [US3] Add Code Index Card section after Index Health in `frontend/src/pages/Settings.tsx`
- [x] T048 [US3] Display chunk_count and status with Badge component
- [x] T049 [US3] Implement progress bar div with dynamic width percentage
- [x] T050 [US3] Display files_processed / files_total during active indexing
- [x] T051 [US3] Add "Re-index Code" button with loading state
- [x] T052 [US3] Implement handleReindex function to call initCodeRAG
- [x] T053 [US3] Add polling interval (5 seconds) during active indexing
- [x] T054 [US3] Display last_indexed_at timestamp when available

**Checkpoint**: Web UI progress monitoring complete

---

## Phase 7: Cascade Delete & Cleanup

**Purpose**: Clean up CodeRAG data when project is deleted

- [x] T055 Add `coderag_delete` command to CLI in `packages/vlt-cli/src/vlt/main.py`
- [x] T056 Implement `delete_coderag_index(project_id)` in `packages/vlt-cli/src/vlt/core/service.py`
- [x] T057 Delete code_chunks, code_nodes, code_edges, symbol_definitions for project
- [x] T058 Delete coderag_index_jobs for project
- [x] T059 Add `--yes` flag to skip confirmation prompt
- [x] T060 Update `backend/src/services/project_service.py` to call `vlt coderag delete` on project deletion
- [x] T061 Add error handling with logger.warning if cleanup fails

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T062 [P] Add job cancellation support in daemon (check cancellation flag during indexing)
- [x] T063 Add edge case handling: no indexable files found
- [x] T064 Add edge case handling: daemon not running (fallback to foreground)
- [x] T065 Add edge case handling: disk space exhaustion during indexing
- [x] T066 Add clear error messages with recovery suggestions
- [x] T067 Run quickstart.md validation scenarios end-to-end (documented, not executed)
- [x] T068 Update CLAUDE.md with new coderag commands documentation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P2 → P2)
- **Cleanup (Phase 7)**: Can run after Phase 3 (needs coderag_init working)
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 & 4 (P1)**: Can start after Foundational (Phase 2) - Core init workflow
- **User Story 5 (P2)**: Can start after Foundational (Phase 2) - Adds daemon integration to init
- **User Story 2 (P2)**: Can start after Phase 4 (needs background jobs) - Adds CLI status
- **User Story 3 (P2)**: Can start after Phase 5 (needs status data) - Adds Web UI

### Within Each User Story

- Models before services
- Services before CLI commands
- CLI commands before API routes
- API routes before frontend
- Core implementation before tests

### Parallel Opportunities

- All Setup tasks (T001-T005) can run in sequence (same file modifications)
- Foundational T008, T009, T010 can run in parallel (different files)
- US1/4 implementation tasks (T011-T020) are sequential (same file)
- US3 backend (T038-T045) and frontend (T046-T054) can run in parallel after T038
- Phase 7 cleanup tasks can run in parallel with Phase 8 polish

---

## Parallel Example: Phase 6 (User Story 3)

```bash
# After T038 creates the router file, these can run in parallel:

# Backend stream:
Task: "T039 Implement GET /api/coderag/status"
Task: "T040 Implement POST /api/coderag/init"
Task: "T041 Implement GET /api/coderag/jobs/{job_id}"

# Frontend stream (after T038):
Task: "T046 Add polling hook for CodeRAG status"
Task: "T047 Add Code Index Card section"
```

---

## Implementation Strategy

### MVP First (User Stories 1 & 4 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Stories 1 & 4 (Interactive init with protection)
4. **STOP and VALIDATE**: Test `vlt coderag init` independently
5. Deploy/demo if ready - users can now initialize CodeRAG interactively

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add US1 & US4 → Interactive init with overwrite protection (MVP!)
3. Add US5 → Background indexing via daemon
4. Add US2 → CLI progress monitoring
5. Add US3 → Web UI progress monitoring
6. Add cleanup → Cascade delete on project removal
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Stories 1 & 4 (CLI init)
   - Developer B: User Story 5 (daemon integration)
3. After US5:
   - Developer A: User Story 2 (CLI status)
   - Developer B: User Story 3 Backend (API routes)
   - Developer C: User Story 3 Frontend (Settings panel)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Constitution requires pytest tests for new backend routes (T045)
- Frontend relies on manual verification per constitution
