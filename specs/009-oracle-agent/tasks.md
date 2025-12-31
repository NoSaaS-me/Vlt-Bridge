# Tasks: Oracle Agent Architecture

**Input**: Design documents from `/specs/009-oracle-agent/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Tests included as specified in constitution (pytest for backend).

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, etc.)
- Paths use web app structure: `backend/src/`, `frontend/src/`, `packages/vlt-cli/src/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create project structure and external prompts directory

- [x] T001 Create backend/prompts/ directory structure per plan.md
- [x] T002 [P] Create backend/prompts/oracle/system.md with Oracle system prompt
- [x] T003 [P] Create backend/prompts/oracle/synthesis.md with answer generation prompt
- [x] T004 [P] Create backend/prompts/oracle/compression.md with context compression prompt
- [x] T005 [P] Create backend/prompts/oracle/no_context.md with fallback response
- [x] T006 [P] Create backend/prompts/librarian/system.md with Librarian system prompt
- [x] T007 [P] Create backend/prompts/librarian/organize.md with organization task prompt
- [x] T008 Copy specs/009-oracle-agent/contracts/tools.json to backend/prompts/tools.json

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core services that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T009 Create OracleContext Pydantic models in backend/src/models/oracle_context.py
- [x] T010 Add oracle_contexts table DDL to backend/src/services/database.py
- [x] T011 Create PromptLoader service in backend/src/services/prompt_loader.py
- [x] T012 Create ToolExecutor service in backend/src/services/tool_executor.py
- [x] T013 Create OracleAgent base class in backend/src/services/oracle_agent.py
- [x] T014 [P] Unit test for PromptLoader in backend/tests/unit/test_prompt_loader.py
- [x] T015 [P] Unit test for ToolExecutor in backend/tests/unit/test_tool_executor.py

**Checkpoint**: Foundation ready - user story implementation can begin

---

## Phase 3: User Story 1 - Ask Oracle via Web UI (Priority: P1) 🎯 MVP

**Goal**: Developer asks Oracle a question via Web UI, receives cited answer with streaming

**Independent Test**: Open AI Chat panel, type question, verify Oracle responds with sources

### Tests for User Story 1

- [x] T016 [P] [US1] Integration test for oracle streaming in backend/tests/integration/test_oracle_stream.py

### Implementation for User Story 1

- [x] T017 [US1] Implement agent loop with OpenRouter in backend/src/services/oracle_agent.py
- [x] T018 [US1] Implement streaming response generator in backend/src/services/oracle_agent.py
- [x] T019 [US1] Wire existing code tools (search_code, find_definition, find_references) in backend/src/services/tool_executor.py
- [x] T020 [US1] Wire existing vault tools (vault_read, vault_search, vault_list) in backend/src/services/tool_executor.py
- [x] T021 [US1] Update POST /api/oracle/stream route to use OracleAgent in backend/src/api/routes/oracle.py
- [x] T022 [US1] Update POST /api/oracle route to use OracleAgent in backend/src/api/routes/oracle.py
- [x] T023 [US1] Add in-memory context tracking (session state) in backend/src/services/oracle_agent.py
- [x] T024 [US1] Add source citation formatting to streaming chunks in backend/src/services/oracle_agent.py

**Checkpoint**: Web UI Oracle Q&A works with tool calling and citations

---

## Phase 4: User Story 2 - Oracle Uses Tools Autonomously (Priority: P1)

**Goal**: Oracle autonomously uses multiple tools to research before answering

**Independent Test**: Ask multi-source question, verify multiple tools invoked before answer

### Tests for User Story 2

- [x] T025 [P] [US2] Unit test for parallel tool execution in backend/tests/unit/test_tool_executor.py

### Implementation for User Story 2

- [x] T026 [US2] Implement parallel tool call handling in backend/src/services/oracle_agent.py
- [x] T027 [US2] Add tool call error handling and fallback in backend/src/services/oracle_agent.py
- [x] T028 [US2] Add tool call timeout handling in backend/src/services/tool_executor.py
- [x] T029 [US2] Implement web_search tool in backend/src/services/tool_executor.py
- [x] T030 [US2] Implement web_fetch tool in backend/src/services/tool_executor.py
- [x] T031 [US2] Add get_repo_map tool to ToolExecutor in backend/src/services/tool_executor.py

**Checkpoint**: Oracle autonomously chains tools and handles failures gracefully

---

## Phase 5: User Story 3 - Oracle Saves Research to Memory (Priority: P2)

**Goal**: Oracle can save findings to threads and vault notes for future reference

**Independent Test**: Ask Oracle to save a decision, verify it appears in thread/vault

### Tests for User Story 3

- [x] T032 [P] [US3] Integration test for thread_push via Oracle in backend/tests/integration/test_oracle_memory.py

### Implementation for User Story 3

- [x] T033 [US3] Implement thread_push tool in backend/src/services/tool_executor.py
- [x] T034 [US3] Implement thread_read tool in backend/src/services/tool_executor.py
- [x] T035 [US3] Implement thread_seek tool in backend/src/services/tool_executor.py
- [x] T036 [US3] Implement thread_list tool in backend/src/services/tool_executor.py
- [x] T037 [US3] Add POST /api/threads endpoint in backend/src/api/routes/threads.py
- [x] T038 [US3] Add POST /api/threads/{id}/entries endpoint in backend/src/api/routes/threads.py
- [x] T039 [US3] Add GET /api/threads/seek endpoint in backend/src/api/routes/threads.py
- [x] T040 [US3] Implement vault_write tool for Oracle notes in backend/src/services/tool_executor.py

**Checkpoint**: Oracle can save and retrieve research from long-term memory

---

## Phase 6: User Story 4 - Ask Oracle via CLI (Priority: P2)

**Goal**: `vlt oracle "question"` sends request to backend, streams response

**Independent Test**: Run `vlt oracle "test"` from terminal, verify streamed response

### Implementation for User Story 4

- [ ] T041 [US4] Create HTTP client for Oracle API in packages/vlt-cli/src/vlt/core/oracle_client.py
- [ ] T042 [US4] Implement streaming response handler in packages/vlt-cli/src/vlt/core/oracle_client.py
- [ ] T043 [US4] Update oracle command to use thin client in packages/vlt-cli/src/vlt/main.py
- [ ] T044 [US4] Add --json output format to oracle command in packages/vlt-cli/src/vlt/main.py
- [ ] T045 [US4] Add error handling for backend unavailable in packages/vlt-cli/src/vlt/core/oracle_client.py
- [ ] T046 [US4] Deprecate local OracleOrchestrator usage in packages/vlt-cli/src/vlt/main.py

**Checkpoint**: CLI oracle command works as thin client to backend

---

## Phase 7: User Story 5 - Librarian Subagent Delegation (Priority: P3)

**Goal**: Oracle delegates vault organization tasks to specialized Librarian agent

**Independent Test**: Ask Oracle to organize a folder, verify Librarian invoked

### Tests for User Story 5

- [x] T047 [P] [US5] Unit test for LibrarianAgent in backend/tests/unit/test_librarian_agent.py

### Implementation for User Story 5

- [x] T048 [US5] Create LibrarianAgent class in backend/src/services/librarian_agent.py
- [x] T049 [US5] Implement scoped tool access for Librarian in backend/src/services/librarian_agent.py
- [ ] T050 [US5] Implement vault_move tool in backend/src/services/tool_executor.py
- [ ] T051 [US5] Implement vault_create_index tool in backend/src/services/tool_executor.py
- [x] T052 [US5] Implement delegate_librarian tool in backend/src/services/tool_executor.py
- [x] T053 [US5] Wire Librarian into Oracle agent loop in backend/src/services/oracle_agent.py

**Checkpoint**: Oracle successfully delegates organization tasks to Librarian

---

## Phase 8: User Story 6 - Context Persistence Across Sessions (Priority: P3)

**Goal**: Conversation context survives browser refresh, model changes, multi-day gaps

**Independent Test**: Start conversation, refresh browser, verify context restored

### Tests for User Story 6

- [ ] T054 [P] [US6] Integration test for context persistence in backend/tests/integration/test_oracle_context.py

### Implementation for User Story 6

- [ ] T055 [US6] Implement OracleContextService in backend/src/services/oracle_context_service.py
- [ ] T056 [US6] Implement context save on each exchange in backend/src/services/oracle_agent.py
- [ ] T057 [US6] Implement context load on session resume in backend/src/services/oracle_agent.py
- [ ] T058 [US6] Implement context compression at 80% token budget in backend/src/services/oracle_context_service.py
- [ ] T059 [US6] Implement model change detection and re-summarization in backend/src/services/oracle_context_service.py
- [ ] T060 [US6] Add GET /api/oracle/context endpoint in backend/src/api/routes/oracle.py
- [ ] T061 [US6] Add DELETE /api/oracle/context endpoint in backend/src/api/routes/oracle.py
- [ ] T062 [US6] Implement key decisions preservation during compression in backend/src/services/oracle_context_service.py

**Checkpoint**: Context survives sessions, model changes, and compression

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements affecting multiple user stories

- [ ] T063 [P] Deprecate oracle_bridge.py with fallback warning in backend/src/services/oracle_bridge.py
- [ ] T064 [P] Add tool call visibility to streaming chunks in backend/src/services/oracle_agent.py
- [ ] T065 [P] Add context token usage to streaming chunks in backend/src/services/oracle_agent.py
- [ ] T066 [P] Add rate limiting to oracle endpoints in backend/src/api/routes/oracle.py
- [ ] T067 [P] Add logging for all tool executions in backend/src/services/tool_executor.py
- [ ] T068 Validate quickstart.md steps work end-to-end
- [ ] T069 Update CLAUDE.md with Oracle Agent documentation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - create prompts first
- **Foundational (Phase 2)**: Depends on Setup - creates core services
- **User Story 1 (Phase 3)**: Depends on Foundational - MVP Oracle
- **User Story 2 (Phase 4)**: Depends on US1 - extends tool usage
- **User Story 3 (Phase 5)**: Can start after Foundational - memory tools
- **User Story 4 (Phase 6)**: Can start after US1 - CLI client
- **User Story 5 (Phase 7)**: Can start after US1 - subagent
- **User Story 6 (Phase 8)**: Depends on US1 - context persistence
- **Polish (Phase 9)**: After all desired stories complete

### User Story Dependencies

```
Foundational ──┬──► US1 (P1) ──┬──► US2 (P1)
               │               │
               │               ├──► US4 (P2)
               │               │
               │               └──► US6 (P3)
               │
               ├──► US3 (P2) ◄──── (independent)
               │
               └──► US5 (P3) ◄──── (after US1)
```

### Parallel Opportunities

**Phase 1** (all in parallel):
- T002, T003, T004, T005, T006, T007 - prompt files

**Phase 2** (after sequential T009-T013):
- T014, T015 - unit tests

**User Stories** (after Phase 2):
- US3 can start in parallel with US1
- US4, US5, US6 can start after US1

---

## Parallel Example: Phase 1 Setup

```bash
# Launch all prompt file creation in parallel:
Task: "Create backend/prompts/oracle/system.md"
Task: "Create backend/prompts/oracle/synthesis.md"
Task: "Create backend/prompts/oracle/compression.md"
Task: "Create backend/prompts/oracle/no_context.md"
Task: "Create backend/prompts/librarian/system.md"
Task: "Create backend/prompts/librarian/organize.md"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2)

1. Complete Phase 1: Setup (prompts)
2. Complete Phase 2: Foundational (core services)
3. Complete Phase 3: User Story 1 (Web UI Oracle)
4. Complete Phase 4: User Story 2 (Autonomous tools)
5. **STOP and VALIDATE**: Test Oracle Q&A with tool calling
6. Deploy/demo MVP

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 + US2 → Test Oracle → Deploy (MVP!)
3. Add US3 → Memory works → Deploy
4. Add US4 → CLI works → Deploy
5. Add US5 → Librarian works → Deploy
6. Add US6 → Context persists → Deploy

### Suggested MVP Scope

**Minimum Viable Product**: Phase 1 + 2 + 3 + 4 (Setup, Foundation, US1, US2)
- Oracle answers questions via Web UI
- Oracle uses tools autonomously
- Streaming responses with citations
- ~35 tasks to MVP

---

## Notes

- [P] tasks = different files, no dependencies
- [US#] label maps task to user story
- Commit after each task
- Stop at checkpoints to validate
- oracle_bridge.py preserved as fallback during transition
- Tests follow TDD: write before implementation

---

## Summary

| Phase | Story | Task Count | Parallel |
|-------|-------|------------|----------|
| Setup | - | 8 | 6 |
| Foundational | - | 7 | 2 |
| US1 | Web UI Q&A | 9 | 1 |
| US2 | Tool Autonomy | 7 | 1 |
| US3 | Memory | 9 | 1 |
| US4 | CLI | 6 | 0 |
| US5 | Librarian | 7 | 1 |
| US6 | Context | 9 | 1 |
| Polish | - | 7 | 5 |
| **Total** | | **69** | **18** |
