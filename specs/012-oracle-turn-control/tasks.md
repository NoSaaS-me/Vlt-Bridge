# Tasks: Oracle Agent Turn Control

**Input**: Design documents from `/specs/012-oracle-turn-control/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Unit tests included per constitution ("Test-Backed Development" principle)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Web app**: `backend/src/`, `frontend/src/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and module structure

- [ ] T001 Create decision_tree module directory at backend/src/services/decision_tree/
- [ ] T002 Create decision_tree __init__.py with exports at backend/src/services/decision_tree/__init__.py
- [ ] T003 [P] Create test directories at backend/tests/unit/ if not exists

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core models and protocols that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### AgentConfig Model (FR-001, FR-003)

- [ ] T004 [P] Add AgentConfig fields to ModelSettings in backend/src/models/settings.py (7 new fields with validation bounds)
- [ ] T005 [P] Add AgentConfig Pydantic model for API responses in backend/src/models/settings.py

### AgentState Model (FR-015)

- [ ] T006 [P] Create AgentState dataclass with frozen=True, kw_only=True in backend/src/models/agent_state.py
- [ ] T007 Add derived properties (is_terminal, elapsed_seconds, iteration_percent, token_percent) to AgentState in backend/src/models/agent_state.py

### DecisionTree Protocol (FR-012)

- [ ] T008 Create DecisionTree Protocol with @runtime_checkable in backend/src/services/decision_tree/protocol.py
- [ ] T009 [P] Create decorator-based registry in backend/src/services/decision_tree/registry.py

### OracleStreamChunk Extension (FR-008)

- [ ] T010 [P] Add "system" to chunk type Literal in backend/src/models/oracle.py
- [ ] T011 Add system_type and system_message optional fields to OracleStreamChunk in backend/src/models/oracle.py

### Unit Tests for Foundational Models

- [ ] T012 [P] Create test_agent_state.py with immutability and property tests at backend/tests/unit/test_agent_state.py
- [ ] T013 [P] Create test_decision_tree.py with protocol conformance tests at backend/tests/unit/test_decision_tree.py

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Configure Agent Behavior (Priority: P1) 🎯 MVP

**Goal**: Users can configure agent limits in Settings, persisted across sessions

**Independent Test**: Open Settings, modify Max Iterations slider, verify persists after page refresh

### Backend Implementation for US1

- [ ] T014 [US1] Add get_agent_config() method to UserSettingsService in backend/src/services/user_settings.py
- [ ] T015 [US1] Add update_agent_config() method with validation to UserSettingsService in backend/src/services/user_settings.py
- [ ] T016 [US1] Add GET /api/settings/agent-config endpoint in backend/src/api/routes/models.py
- [ ] T017 [US1] Add PUT /api/settings/agent-config endpoint with validation in backend/src/api/routes/models.py
- [ ] T018 [P] [US1] Add POST /api/settings/agent-config/reset endpoint in backend/src/api/routes/models.py

### Frontend Implementation for US1

- [ ] T019 [P] [US1] Add AgentConfig interface to frontend/src/types/oracle.ts
- [ ] T020 [P] [US1] Add getAgentConfig() and updateAgentConfig() API calls in frontend/src/services/api.ts
- [ ] T021 [US1] Create AgentConfigPanel component with sliders/inputs in frontend/src/components/AgentConfigPanel.tsx
- [ ] T022 [US1] Add AgentConfig section to Settings page in frontend/src/pages/Settings.tsx
- [ ] T023 [US1] Implement auto-save on change with validation feedback in frontend/src/components/AgentConfigPanel.tsx

### Unit Test for US1

- [ ] T024 [P] [US1] Add tests for AgentConfig validation bounds in backend/tests/unit/test_settings.py

**Checkpoint**: User Story 1 complete - users can configure and persist agent settings

---

## Phase 4: User Story 2 - Graceful Limit Notifications (Priority: P1)

**Goal**: System messages appear in chat when approaching or hitting limits

**Independent Test**: Set low max iterations (3), ask complex question, observe system notifications

### Backend Implementation for US2

- [ ] T025 [US2] Create DefaultDecisionTree class implementing Protocol in backend/src/services/decision_tree/default.py
- [ ] T026 [US2] Implement should_continue() with all termination conditions (FR-004) in backend/src/services/decision_tree/default.py
- [ ] T027 [US2] Implement on_turn_start() with soft warning emission in backend/src/services/decision_tree/default.py
- [ ] T028 [US2] Implement on_tool_result() with action tracking in backend/src/services/decision_tree/default.py
- [ ] T029 [US2] Load user's AgentConfig in OracleAgent.query() from user_settings in backend/src/services/oracle_agent.py
- [ ] T030 [US2] Create AgentState at query start in OracleAgent.query() in backend/src/services/oracle_agent.py
- [ ] T031 [US2] Replace hardcoded MAX_TURNS with config.max_iterations in backend/src/services/oracle_agent.py
- [ ] T032 [US2] Add DecisionTree.on_turn_start() call before each turn in backend/src/services/oracle_agent.py
- [ ] T033 [US2] Add yield for system chunks when warnings triggered in backend/src/services/oracle_agent.py
- [ ] T034 [US2] Implement token tracking (estimate) per turn in backend/src/services/oracle_agent.py
- [ ] T035 [US2] Add timeout checking using start_time in backend/src/services/oracle_agent.py
- [ ] T036 [US2] Ensure main-chain content saved on any termination (FR-005) in backend/src/services/oracle_agent.py
- [ ] T037 [US2] Add termination reason to final response (FR-007) in backend/src/services/oracle_agent.py

### Frontend Implementation for US2

- [ ] T038 [P] [US2] Add 'system' to role union type in frontend/src/types/oracle.ts
- [ ] T039 [P] [US2] Add system_type and metadata fields to OracleMessage in frontend/src/types/oracle.ts
- [ ] T040 [US2] Handle 'system' chunk type in ChatPanel SSE handler in frontend/src/components/ChatPanel.tsx
- [ ] T041 [US2] Create system message in conversation when chunk received in frontend/src/components/ChatPanel.tsx

### Unit Tests for US2

- [ ] T042 [P] [US2] Test termination conditions in backend/tests/unit/test_termination.py
- [ ] T043 [P] [US2] Test soft warning thresholds (70%, 80%) in backend/tests/unit/test_termination.py

**Checkpoint**: User Story 2 complete - system notifications appear when approaching limits

---

## Phase 5: User Story 3 - Intelligent Termination (Priority: P2)

**Goal**: Agent stops on goal completion, detects no-progress loops

**Independent Test**: Ask simple question, verify agent stops without extra tool calls; trigger 3x same tool call

### Backend Implementation for US3

- [ ] T044 [US3] Add no-progress detection (3 consecutive identical actions) to DefaultDecisionTree in backend/src/services/decision_tree/default.py
- [ ] T045 [US3] Add action_signature() helper for comparing tool calls in backend/src/services/decision_tree/default.py
- [ ] T046 [US3] Update AgentState.recent_actions tracking in on_tool_result() in backend/src/services/decision_tree/default.py
- [ ] T047 [US3] Add consecutive error tracking (3 errors = terminate) in backend/src/services/decision_tree/default.py
- [ ] T048 [US3] Emit no_progress system chunk when detected in backend/src/services/oracle_agent.py
- [ ] T049 [US3] Emit error_limit system chunk when 3 consecutive errors in backend/src/services/oracle_agent.py

### Unit Tests for US3

- [ ] T050 [P] [US3] Test no-progress detection algorithm in backend/tests/unit/test_termination.py
- [ ] T051 [P] [US3] Test action_signature equality in backend/tests/unit/test_termination.py

**Checkpoint**: User Story 3 complete - agent terminates intelligently on goal or no-progress

---

## Phase 6: User Story 4 - DecisionTree Protocol for Extensions (Priority: P2)

**Goal**: Architecture supports custom decision trees via decorator registration

**Independent Test**: Register mock decision tree with decorator, verify it can be retrieved

### Backend Implementation for US4

- [ ] T052 [US4] Add @decision_tree("name") decorator to registry in backend/src/services/decision_tree/registry.py
- [ ] T053 [US4] Add get_decision_tree(name) function to registry in backend/src/services/decision_tree/registry.py
- [ ] T054 [US4] Register DefaultDecisionTree with @decision_tree("default") decorator in backend/src/services/decision_tree/default.py
- [ ] T055 [US4] Add extensions field usage example/documentation in backend/src/models/agent_state.py
- [ ] T056 [US4] Wire OracleAgent to use registry for tree lookup in backend/src/services/oracle_agent.py

### Unit Tests for US4

- [ ] T057 [P] [US4] Test decorator registration in backend/tests/unit/test_decision_tree.py
- [ ] T058 [P] [US4] Test get_decision_tree() retrieval in backend/tests/unit/test_decision_tree.py

**Checkpoint**: User Story 4 complete - DecisionTree is pluggable for future skills

---

## Phase 7: User Story 5 - System User in Chat Flow (Priority: P3)

**Goal**: System messages have distinct visual treatment in chat UI

**Independent Test**: Trigger system notification, verify distinct styling (amber warning)

### Frontend Implementation for US5

- [ ] T059 [US5] Add system message rendering branch to ChatMessage in frontend/src/components/ChatMessage.tsx
- [ ] T060 [US5] Style system messages with amber warning treatment in frontend/src/components/ChatMessage.tsx
- [ ] T061 [US5] Disable reply/edit actions for system messages in frontend/src/components/ChatMessage.tsx
- [ ] T062 [US5] Add metadata display (current_value/limit_value) to system messages in frontend/src/components/ChatMessage.tsx

**Checkpoint**: User Story 5 complete - system messages are visually distinct

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T063 [P] Add logging for all termination events in backend/src/services/oracle_agent.py
- [ ] T064 [P] Update quickstart.md with testing scenarios at specs/012-oracle-turn-control/quickstart.md
- [ ] T065 Run all unit tests and fix any failures
- [ ] T066 Manual verification of all acceptance scenarios from spec.md
- [ ] T067 [P] Code cleanup: remove any TODO comments, unused imports

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 and US2 are both P1 priority - do US1 first (enables Settings UI)
  - US3 and US4 are both P2 priority - can proceed in parallel
  - US5 is P3 - lowest priority, can wait
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Depends only on Foundational - No dependencies on other stories
- **User Story 2 (P1)**: Depends on Foundational + uses AgentConfig from US1 backend (T014-T015)
- **User Story 3 (P2)**: Depends on US2 (uses DefaultDecisionTree from T025-T028)
- **User Story 4 (P2)**: Depends on Foundational - No dependencies on other stories (registry is independent)
- **User Story 5 (P3)**: Depends on US2 (needs system chunk handling from T040-T041)

### Within Each User Story

- Backend before frontend (API must exist before UI calls it)
- Models before services
- Services before API routes
- Unit tests can run in parallel with implementation

### Parallel Opportunities

- T003-T007: All foundational models can be created in parallel
- T010-T013: Chunk extension and tests in parallel
- T019-T020: Frontend types and API calls in parallel
- T038-T039: Frontend type extensions in parallel
- T042-T043, T050-T051, T057-T058: Tests within each story in parallel

---

## Parallel Example: Foundational Phase

```bash
# Launch all foundational models together:
Task: "Add AgentConfig fields to ModelSettings in backend/src/models/settings.py"
Task: "Create AgentState dataclass in backend/src/models/agent_state.py"
Task: "Add 'system' to chunk type Literal in backend/src/models/oracle.py"
Task: "Create test_agent_state.py at backend/tests/unit/test_agent_state.py"
Task: "Create test_decision_tree.py at backend/tests/unit/test_decision_tree.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Configure Agent Behavior)
4. **STOP and VALIDATE**: Settings UI works, values persist
5. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add User Story 1 → Settings UI works → Deploy (MVP!)
3. Add User Story 2 → Limit notifications work → Deploy
4. Add User Story 3 → No-progress detection works → Deploy
5. Add User Story 4 → DecisionTree extensible → Deploy
6. Add User Story 5 → System messages styled → Deploy
7. Each story adds value without breaking previous stories

### Recommended Order (Solo Developer)

1. Phase 1: Setup (T001-T003)
2. Phase 2: Foundational (T004-T013) - critical path
3. Phase 3: US1 (T014-T024) - enables configuration
4. Phase 4: US2 (T025-T043) - core functionality
5. Phase 5: US3 (T044-T051) - intelligent termination
6. Phase 6: US4 (T052-T058) - extensibility
7. Phase 7: US5 (T059-T062) - polish
8. Phase 8: Final polish (T063-T067)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Tests included per constitution "Test-Backed Development" principle
