# Tasks: Gemini Vault Chat Agent

**Input**: Design documents from `/specs/004-gemini-vault-chat/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Unit tests for RAG service included per Constitution (Test-Backed Development).

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `backend/src/`, `backend/tests/`
- **Frontend**: `frontend/src/`
- **Data**: `data/llamaindex/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add dependencies and create type definitions

- [ ] T001 Add LlamaIndex dependencies to `backend/requirements.txt`: llama-index, llama-index-llms-google-genai, llama-index-embeddings-google-genai
- [ ] T002 [P] Create TypeScript types in `frontend/src/types/rag.ts`: ChatMessage, SourceReference, NoteWritten, ChatRequest, ChatResponse
- [ ] T003 [P] Add GOOGLE_API_KEY and LLAMAINDEX_PERSIST_DIR to environment configuration in `backend/src/services/config.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core backend infrastructure for RAG that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Create Pydantic models in `backend/src/models/rag.py`: ChatMessage, SourceReference, NoteWritten, ChatRequest, ChatResponse, StatusResponse, ErrorResponse
- [ ] T005 Create RAG index service skeleton in `backend/src/services/rag_index.py` with `get_or_build_index()` singleton pattern
- [ ] T006 Implement index persistence: load from `data/llamaindex/` if exists, otherwise build and persist in `backend/src/services/rag_index.py`
- [ ] T007 Create `backend/tests/unit/test_rag_service.py` with test stubs for index loading, query execution, and error handling
- [ ] T008 Register RAG routes in `backend/src/api/main.py` (import and include rag router)

**Checkpoint**: Foundation ready - RAG service can load/build index on startup

---

## Phase 3: User Story 1 & 2 - Ask Questions + View Sources (Priority: P1) 🎯 MVP

**Goal**: Users can ask questions and receive AI-synthesized answers with source attribution

**Independent Test**: Type a question in the chat panel, verify response includes answer text and clickable source references

### Backend Implementation (US1+US2)

- [ ] T009 [US1] Implement `rag_chat()` function in `backend/src/services/rag_index.py` that queries index and returns answer with sources
- [ ] T010 [US1] Extract source metadata from LlamaIndex response nodes (path, title, snippet, score) in `backend/src/services/rag_index.py`
- [ ] T011 [US1] Create POST `/api/rag/chat` endpoint in `backend/src/api/routes/rag.py` wrapping `rag_chat()`
- [ ] T012 [P] [US1] Create GET `/api/rag/status` endpoint in `backend/src/api/routes/rag.py` returning index status
- [ ] T013 [US1] Implement unit tests for `rag_chat()` in `backend/tests/unit/test_rag_service.py`: happy path, no results, error handling

### Frontend Implementation (US1+US2)

- [ ] T014 [P] [US2] Create RAG API client in `frontend/src/services/rag.ts` with `sendMessage()` and `getStatus()` functions
- [ ] T015 [P] [US2] Create ChatMessage component in `frontend/src/components/ChatMessage.tsx` rendering user/assistant messages
- [ ] T016 [P] [US2] Create SourceList component in `frontend/src/components/SourceList.tsx` with collapsible source references
- [ ] T017 [US1] Create ChatPanel component in `frontend/src/components/ChatPanel.tsx` with message list and composer textarea
- [ ] T018 [US1] Integrate ChatPanel into MainApp layout in `frontend/src/pages/MainApp.tsx` as new panel/tab
- [ ] T019 [US2] Wire SourceList click handler to open note in document viewer via existing navigation

**Checkpoint**: User can ask a question, see AI answer with sources, and click source to view note

---

## Phase 4: User Story 3 - Multi-Turn Conversation (Priority: P2)

**Goal**: Users can have context-aware follow-up conversations

**Independent Test**: Ask "What is authentication?", then ask "How do I configure it?" - verify agent understands "it" refers to authentication

### Implementation (US3)

- [ ] T020 [US3] Add message history state management in `frontend/src/components/ChatPanel.tsx` using React useState
- [ ] T021 [US3] Pass full message history array to `POST /api/rag/chat` in `frontend/src/services/rag.ts`
- [ ] T022 [US3] Update `rag_chat()` in `backend/src/services/rag_index.py` to construct context from message history
- [ ] T023 [US3] Add conversation reset button in `frontend/src/components/ChatPanel.tsx` to clear history
- [ ] T024 [US3] Add unit test for multi-turn context handling in `backend/tests/unit/test_rag_service.py`

**Checkpoint**: Multi-turn conversation maintains context; page refresh clears history

---

## Phase 5: User Story 4 - Agent Creates Notes (Priority: P3, Optional)

**Goal**: Agent can create/append notes in a designated folder

**Independent Test**: Ask "create a summary note about authentication" - verify note appears in `agent-notes/` folder

### Implementation (US4)

- [ ] T025 [US4] Create `create_note()` helper in `backend/src/services/rag_index.py` constrained to `agent-notes/` folder
- [ ] T026 [US4] Create `append_to_note()` helper in `backend/src/services/rag_index.py` for updating existing notes
- [ ] T027 [US4] Register helpers as LlamaIndex FunctionTools in `backend/src/services/rag_index.py`
- [ ] T028 [US4] Update `rag_chat()` to use agent mode with tools when write intent detected
- [ ] T029 [US4] Add `notes_written` to ChatResponse and include in API response from `backend/src/api/routes/rag.py`
- [ ] T030 [P] [US4] Add NoteWritten badge component in `frontend/src/components/ChatMessage.tsx` showing created note path
- [ ] T031 [US4] Wire badge click to navigate to created note in vault viewer
- [ ] T032 [US4] Add unit tests for constrained write operations in `backend/tests/unit/test_rag_service.py`

**Checkpoint**: Agent can create notes; writes constrained to `agent-notes/` folder

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Error handling, edge cases, and validation

- [ ] T033 [P] Implement error handling for missing GOOGLE_API_KEY with 503 response in `backend/src/services/rag_index.py`
- [ ] T034 [P] Implement error handling for API rate limits with 429 response in `backend/src/api/routes/rag.py`
- [ ] T035 [P] Add loading state and error display in `frontend/src/components/ChatPanel.tsx`
- [ ] T036 [P] Add empty vault message when no documents indexed in `backend/src/services/rag_index.py`
- [ ] T037 Run quickstart.md validation: verify all setup steps work
- [ ] T038 Manual E2E test: full user journey through all implemented stories

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - US1+US2 (Phase 3) must complete before US3 (Phase 4)
  - US3 can complete before US4 (Phase 5 is optional)
- **Polish (Phase 6)**: Can run after Phase 3 minimum

### User Story Dependencies

- **User Story 1+2 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P2)**: Depends on US1+US2 for chat panel and history structure
- **User Story 4 (P3)**: Depends on US1+US2 for basic chat flow; optional feature

### Within Each Phase

- Backend models before services
- Services before routes
- Backend before frontend integration
- Core implementation before error handling

### Parallel Opportunities

**Phase 1:**
- T002 (TS types) and T003 (config) can run in parallel

**Phase 2:**
- T004 (models) must complete before T005-T008

**Phase 3:**
- T014, T015, T016 (frontend components) can run in parallel
- T012 (/status endpoint) can run in parallel with other backend work

**Phase 4:**
- T020-T024 are sequential (frontend then backend integration)

**Phase 5:**
- T025, T026 (helpers) sequential
- T030 (badge) can run parallel with backend once T029 complete

**Phase 6:**
- T033, T034, T035, T036 all parallel (different files)

---

## Parallel Example: Phase 3 Frontend

```bash
# Launch all independent frontend components together:
Task: "Create RAG API client in frontend/src/services/rag.ts"
Task: "Create ChatMessage component in frontend/src/components/ChatMessage.tsx"
Task: "Create SourceList component in frontend/src/components/SourceList.tsx"
```

---

## Implementation Strategy

### MVP First (Phase 1-3 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T008)
3. Complete Phase 3: User Story 1+2 (T009-T019)
4. **STOP and VALIDATE**: Test RAG query and source display independently
5. Deploy/demo if ready - this is the MVP!

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add US1+US2 → Test independently → **Deploy/Demo (MVP!)**
3. Add US3 → Test multi-turn → Deploy/Demo
4. Add US4 (optional) → Test note creation → Deploy/Demo
5. Each story adds value without breaking previous stories

### Estimated Effort

| Phase | Tasks | Estimated Hours |
|-------|-------|-----------------|
| Setup | 3 | 0.5 |
| Foundational | 5 | 2 |
| US1+US2 (MVP) | 11 | 4 |
| US3 | 5 | 2 |
| US4 (optional) | 8 | 3 |
| Polish | 6 | 1.5 |
| **Total** | **38** | **13** |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US1 and US2 combined in Phase 3 since they're tightly coupled (source display is part of query response)
- US4 is optional per spec - can skip if time-constrained
- Constitution requires pytest tests for backend features
- Frontend testing is manual verification per Constitution

