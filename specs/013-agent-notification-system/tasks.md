# Tasks: Agent Notification System

**Input**: Design documents from `/specs/013-agent-notification-system/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Unit tests are included per constitution (Test-Backed Development principle).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `backend/src/` for source, `backend/tests/` for tests
- **Frontend**: `frontend/src/` for source

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependencies, and ANS package structure

- [ ] T001 Add python-toon dependency to backend/pyproject.toml
- [ ] T002 [P] Create ANS package structure: backend/src/services/ans/__init__.py
- [ ] T003 [P] Create subscribers directory: backend/src/services/ans/subscribers/
- [ ] T004 [P] Create templates directory: backend/src/services/ans/templates/
- [ ] T005 [P] Add shadcn/ui tabs component to frontend via `npx shadcn@latest add tabs`
- [ ] T006 Run database migration to add disabled_subscribers_json column to user_settings in backend/src/services/database.py
- [ ] T007 Run database migration to add system_messages_json column to context_nodes in backend/src/services/database.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core ANS infrastructure that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

### Event System Foundation

- [ ] T008 [P] Create Event dataclass with Severity enum in backend/src/services/ans/event.py
- [ ] T009 [P] Create EventType constants (tool.*, budget.*, agent.*) in backend/src/services/ans/event.py
- [ ] T010 Implement EventBus (pub/sub) in backend/src/services/ans/bus.py
- [ ] T011 Create unit tests for EventBus in backend/tests/unit/test_ans_bus.py

### Subscriber System Foundation

- [ ] T012 [P] Create Subscriber dataclass and SubscriberConfig model in backend/src/services/ans/subscriber.py
- [ ] T013 [P] Create Priority and InjectionPoint enums in backend/src/services/ans/subscriber.py
- [ ] T014 Implement SubscriberLoader (TOML discovery from subscribers/ directory) in backend/src/services/ans/subscriber.py
- [ ] T015 Create unit tests for SubscriberLoader in backend/tests/unit/test_ans_subscriber.py

### Notification Generation Foundation

- [ ] T016 [P] Create Notification dataclass in backend/src/services/ans/accumulator.py
- [ ] T017 Implement NotificationAccumulator (batching, deduplication) in backend/src/services/ans/accumulator.py
- [ ] T018 Create unit tests for NotificationAccumulator in backend/tests/unit/test_ans_accumulator.py
- [ ] T019 Implement ToonFormatter (Jinja2 + TOON) in backend/src/services/ans/toon_formatter.py
- [ ] T020 Create unit tests for ToonFormatter in backend/tests/unit/test_toon_formatter.py

### Stream Extension

- [ ] T021 Add StreamEventType.SYSTEM to enum in backend/src/models/oracle.py
- [ ] T022 [P] Add ExchangeRole.SYSTEM to enum in backend/src/models/oracle_context.py

### Frontend Types Foundation

- [ ] T023 [P] Add 'system' to OracleMessage role union in frontend/src/types/oracle.ts
- [ ] T024 [P] Add 'system' to Role type in frontend/src/types/rag.ts

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Tool Failure Notifications (Priority: P1)

**Goal**: Agent receives tool failure/timeout notifications in its context window

**Independent Test**: Trigger a tool timeout during agent execution, verify notification appears in agent's next context and in chat UI as system message

### Subscriber Configuration for US1

- [ ] T025 [P] [US1] Create tool_failure.toml subscriber config in backend/src/services/ans/subscribers/tool_failure.toml
- [ ] T026 [P] [US1] Create tool_failure.toon.j2 template in backend/src/services/ans/templates/tool_failure.toon.j2

### Event Emission for US1

- [ ] T027 [US1] Emit tool.call.failure event from oracle_agent.py at line ~1043-1051 in backend/src/services/oracle_agent.py
- [ ] T028 [US1] Emit tool.call.timeout event from tool_executor.py at line ~240-261 in backend/src/services/tool_executor.py
- [ ] T029 [P] [US1] Emit tool.call.success event from oracle_agent.py at line ~964 (for future use) in backend/src/services/oracle_agent.py

### Notification Injection for US1

- [ ] T030 [US1] Initialize ANS accumulator in OracleAgent constructor in backend/src/services/oracle_agent.py
- [ ] T031 [US1] Inject notifications at "after_tool" point in run_turn_loop() at line ~964 in backend/src/services/oracle_agent.py
- [ ] T032 [US1] Yield OracleStreamChunk(type="system", content=...) for notifications in backend/src/services/oracle_agent.py

### Integration Test for US1

- [ ] T033 [US1] Create integration test: tool failure → notification → SSE chunk in backend/tests/unit/test_ans_integration.py

**Checkpoint**: Tool failure notifications working end-to-end

---

## Phase 4: User Story 2 - System Messages in Chat UI (Priority: P1)

**Goal**: Chat UI displays system messages with distinct visual styling

**Independent Test**: View chat panel after system event, verify system message appears with yellow/amber styling and "System" attribution

### Frontend SSE Handling for US2

- [ ] T034 [US2] Handle 'system' chunk type in streamOracle callback in frontend/src/components/ChatPanel.tsx
- [ ] T035 [US2] Create system message with role='system' when chunk received in frontend/src/components/ChatPanel.tsx

### System Message Styling for US2

- [ ] T036 [US2] Add isSystem check (message.role === 'system') in frontend/src/components/ChatMessage.tsx
- [ ] T037 [US2] Add system message avatar styling (yellow/amber with AlertCircle icon) in frontend/src/components/ChatMessage.tsx
- [ ] T038 [US2] Add system message container styling (yellow border-l-2, background) in frontend/src/components/ChatMessage.tsx
- [ ] T039 [P] [US2] Import AlertCircle icon from lucide-react in frontend/src/components/ChatMessage.tsx

### Persistence for US2

- [ ] T040 [US2] Store system messages in context_nodes.system_messages_json on save in backend/src/services/context_tree_service.py
- [ ] T041 [US2] Load system messages from context_nodes when building history in backend/src/services/oracle_agent.py

**Checkpoint**: System messages display correctly with distinct styling

---

## Phase 5: User Story 3 - Budget Warning Notifications (Priority: P2)

**Goal**: Agent receives budget warning notifications (token, iteration, timeout)

**Independent Test**: Configure low token budget, run agent until 80% consumption, verify warning notification appears

### Subscriber Configuration for US3

- [ ] T042 [P] [US3] Create budget_warning.toml subscriber config in backend/src/services/ans/subscribers/budget_warning.toml
- [ ] T043 [P] [US3] Create budget_warning.toon.j2 template in backend/src/services/ans/templates/budget_warning.toon.j2
- [ ] T044 [P] [US3] Create budget_exceeded.toml subscriber config in backend/src/services/ans/subscribers/budget_exceeded.toml
- [ ] T045 [P] [US3] Create budget_exceeded.toon.j2 template in backend/src/services/ans/templates/budget_exceeded.toon.j2

### Event Emission for US3

- [ ] T046 [US3] Emit budget.token.warning event when 80% threshold crossed in oracle_agent.py at line ~491-499 in backend/src/services/oracle_agent.py
- [ ] T047 [US3] Emit budget.iteration.warning event when 70% threshold crossed in backend/src/services/oracle_agent.py
- [ ] T048 [US3] Emit budget.token.exceeded and budget.iteration.exceeded events on limit hit in backend/src/services/oracle_agent.py

### Injection Point for US3

- [ ] T049 [US3] Inject notifications at "turn_start" point for budget warnings in backend/src/services/oracle_agent.py
- [ ] T050 [US3] Inject notifications at "immediate" point for budget exceeded (critical) in backend/src/services/oracle_agent.py

**Checkpoint**: Budget warnings appear before agent continues, exceeded notifications immediate

---

## Phase 6: User Story 4 - Loop Detection Notifications (Priority: P2)

**Goal**: Agent receives notification when stuck in repetitive pattern

**Independent Test**: Craft prompt causing repetitive behavior, verify loop detection notification appears

### Subscriber Configuration for US4

- [ ] T051 [P] [US4] Create loop_detected.toml subscriber config in backend/src/services/ans/subscribers/loop_detected.toml
- [ ] T052 [P] [US4] Create loop_detected.toon.j2 template in backend/src/services/ans/templates/loop_detected.toon.j2

### Event Emission for US4

- [ ] T053 [US4] Emit agent.loop.detected event when loop detection triggers in oracle_agent.py in backend/src/services/oracle_agent.py
- [ ] T054 [US4] Include pattern description and repetition count in event payload in backend/src/services/oracle_agent.py

**Checkpoint**: Loop detection notifications working

---

## Phase 7: User Story 5 - Subscriber Management Settings (Priority: P3)

**Goal**: Users can enable/disable non-core subscribers via Settings UI

**Independent Test**: Toggle a subscriber off in settings, trigger event, verify no notification generated

### Backend API for US5

- [ ] T055 [P] [US5] Create NotificationSettings Pydantic model in backend/src/models/notifications.py
- [ ] T056 [P] [US5] Create SubscriberInfo response model in backend/src/models/notifications.py
- [ ] T057 [US5] Implement GET /api/notifications/subscribers endpoint in backend/src/api/routes/notifications.py
- [ ] T058 [US5] Implement POST /api/notifications/subscribers/{id}/toggle endpoint in backend/src/api/routes/notifications.py
- [ ] T059 [US5] Implement GET /api/settings/notifications endpoint in backend/src/api/routes/models.py
- [ ] T060 [US5] Implement PUT /api/settings/notifications endpoint in backend/src/api/routes/models.py
- [ ] T061 [US5] Add disabled_subscribers to UserSettingsService in backend/src/services/user_settings.py

### Frontend Types for US5

- [ ] T062 [P] [US5] Create SubscriberInfo and NotificationSettings types in frontend/src/types/notifications.ts
- [ ] T063 [P] [US5] Create notifications API client in frontend/src/services/notifications.ts

### Frontend Settings UI for US5

- [ ] T064 [US5] Create NotificationSettings component with subscriber toggles in frontend/src/components/NotificationSettings.tsx
- [ ] T065 [US5] Add Tabs wrapper to Settings.tsx with Account, Models, Context, Notifications tabs in frontend/src/pages/Settings.tsx
- [ ] T066 [US5] Render NotificationSettings in Notifications tab in frontend/src/pages/Settings.tsx
- [ ] T067 [US5] Display core subscribers as always-enabled with tooltip explanation in frontend/src/components/NotificationSettings.tsx

### Integration for US5

- [ ] T068 [US5] Check disabled_subscribers before generating notification in accumulator in backend/src/services/ans/accumulator.py

**Checkpoint**: Subscribers can be toggled on/off via Settings UI

---

## Phase 8: User Story 6 - File-Based Subscriber Discovery (Priority: P3)

**Goal**: New subscribers added by config file, discovered at startup

**Independent Test**: Add new subscriber TOML file, restart service, verify subscriber in list

### Discovery Implementation for US6

- [ ] T069 [US6] Implement glob-based discovery of *.toml files in subscribers/ directory in backend/src/services/ans/subscriber.py
- [ ] T070 [US6] Validate TOML against subscriber schema on load in backend/src/services/ans/subscriber.py
- [ ] T071 [US6] Log warning and skip invalid config files in backend/src/services/ans/subscriber.py
- [ ] T072 [US6] Verify Jinja2 template exists for each subscriber on load in backend/src/services/ans/subscriber.py

### Unit Tests for US6

- [ ] T073 [US6] Test valid subscriber discovery in backend/tests/unit/test_ans_subscriber.py
- [ ] T074 [US6] Test invalid subscriber handling (graceful skip) in backend/tests/unit/test_ans_subscriber.py
- [ ] T075 [US6] Test missing template handling in backend/tests/unit/test_ans_subscriber.py

**Checkpoint**: File-based subscriber discovery working

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

### Error Handling

- [ ] T076 Implement template rendering fallback (generic error message) in backend/src/services/ans/toon_formatter.py
- [ ] T077 Implement event queue overflow handling (drop oldest low-priority) in backend/src/services/ans/bus.py
- [ ] T078 [P] Add TOON parsing error handling in UI (show raw content) in frontend/src/components/ChatMessage.tsx

### Performance

- [ ] T079 Add event processing timing logs for performance monitoring in backend/src/services/ans/accumulator.py
- [ ] T080 Verify <100ms event processing goal in integration tests in backend/tests/unit/test_ans_integration.py

### Documentation

- [ ] T081 [P] Update CLAUDE.md with ANS technology additions
- [ ] T082 [P] Validate quickstart.md instructions work end-to-end

### Collapsible UI (FR-026)

- [ ] T083 Add collapse/expand toggle for verbose system notifications in frontend/src/components/ChatMessage.tsx

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)
    │
    ▼
Phase 2 (Foundational) ◄── BLOCKS all user stories
    │
    ├───────────────────┬───────────────────┬───────────────────┐
    ▼                   ▼                   ▼                   ▼
Phase 3 (US1: P1)   Phase 4 (US2: P1)   Phase 5 (US3: P2)   Phase 6 (US4: P2)
Tool Failure        System Messages     Budget Warnings     Loop Detection
    │                   │                   │                   │
    └───────────────────┼───────────────────┼───────────────────┘
                        │                   │
                        ▼                   ▼
                Phase 7 (US5: P3)    Phase 8 (US6: P3)
                Subscriber Settings  File Discovery
                        │                   │
                        └───────────────────┘
                                │
                                ▼
                        Phase 9 (Polish)
```

### User Story Dependencies

- **US1 (Tool Failure)**: Depends on Phase 2 - No dependencies on other stories
- **US2 (System Messages)**: Depends on Phase 2, best after US1 (for testing) but independently testable
- **US3 (Budget Warnings)**: Depends on Phase 2 - Independent of US1/US2
- **US4 (Loop Detection)**: Depends on Phase 2 - Independent of US1/US2/US3
- **US5 (Settings UI)**: Depends on Phase 2 + at least one subscriber (US1 or US3/US4)
- **US6 (File Discovery)**: Can start after Phase 2, validates subscriber loading

### Within Each User Story

- Subscriber config TOML before template
- Event emission before injection
- Backend before frontend (where applicable)

### Parallel Opportunities

**Phase 1**:
- T002, T003, T004, T005 can run in parallel

**Phase 2**:
- T008, T009 (Event models) in parallel
- T012, T013 (Subscriber models) in parallel
- T016 after T008
- T021, T022, T023, T024 (type additions) all in parallel

**User Stories**:
- US1 and US2 are both P1 but can run in parallel (different files)
- US3 and US4 are both P2 and can run in parallel
- US5 and US6 are both P3 and can run in parallel

---

## Parallel Example: Phase 2 Foundation

```bash
# Batch 1: Model definitions (all parallel)
Task: "Create Event dataclass in backend/src/services/ans/event.py"
Task: "Create EventType constants in backend/src/services/ans/event.py"
Task: "Create Subscriber dataclass in backend/src/services/ans/subscriber.py"
Task: "Create Priority enum in backend/src/services/ans/subscriber.py"
Task: "Add StreamEventType.SYSTEM in backend/src/models/oracle.py"
Task: "Add ExchangeRole.SYSTEM in backend/src/models/oracle_context.py"
Task: "Add 'system' to OracleMessage in frontend/src/types/oracle.ts"
Task: "Add 'system' to Role type in frontend/src/types/rag.ts"

# Batch 2: Core implementations (after Batch 1)
Task: "Implement EventBus in backend/src/services/ans/bus.py"
Task: "Implement SubscriberLoader in backend/src/services/ans/subscriber.py"

# Batch 3: Accumulator and formatter (after Batch 1)
Task: "Implement NotificationAccumulator in backend/src/services/ans/accumulator.py"
Task: "Implement ToonFormatter in backend/src/services/ans/toon_formatter.py"

# Batch 4: Tests (after respective implementations)
Task: "Test EventBus in backend/tests/unit/test_ans_bus.py"
Task: "Test SubscriberLoader in backend/tests/unit/test_ans_subscriber.py"
Task: "Test NotificationAccumulator in backend/tests/unit/test_ans_accumulator.py"
Task: "Test ToonFormatter in backend/tests/unit/test_toon_formatter.py"
```

---

## Implementation Strategy

### MVP First (US1 + US2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Tool Failure)
4. Complete Phase 4: User Story 2 (System Messages)
5. **STOP and VALIDATE**:
   - Trigger a tool timeout
   - Verify notification appears in agent context
   - Verify system message appears in chat UI
6. Deploy/demo MVP

### Incremental Delivery

1. **MVP**: Setup + Foundational + US1 + US2 → Tool failures visible in UI
2. **v1.1**: Add US3 (Budget Warnings) → Graceful degradation
3. **v1.2**: Add US4 (Loop Detection) → Better agent self-awareness
4. **v1.3**: Add US5 (Settings UI) → User control
5. **v1.4**: Add US6 (File Discovery) → Extensibility
6. **v1.5**: Polish phase → Production-ready

### Suggested Task Count per Day (Solo Developer)

- **Day 1**: Phase 1 (7 tasks) + Phase 2 start (T008-T015)
- **Day 2**: Phase 2 complete (T016-T024)
- **Day 3**: Phase 3 US1 (9 tasks)
- **Day 4**: Phase 4 US2 (8 tasks)
- **Day 5**: Phase 5 US3 (9 tasks)
- **Day 6**: Phase 6 US4 (4 tasks) + Phase 7 US5 start
- **Day 7**: Phase 7 US5 complete (14 tasks)
- **Day 8**: Phase 8 US6 (7 tasks) + Phase 9 Polish

---

## Task Summary

| Phase | User Story | Task Count | Parallel Tasks |
|-------|------------|------------|----------------|
| 1 | Setup | 7 | 4 |
| 2 | Foundational | 17 | 8 |
| 3 | US1: Tool Failure (P1) | 9 | 3 |
| 4 | US2: System Messages (P1) | 8 | 1 |
| 5 | US3: Budget Warnings (P2) | 9 | 4 |
| 6 | US4: Loop Detection (P2) | 4 | 2 |
| 7 | US5: Subscriber Settings (P3) | 14 | 4 |
| 8 | US6: File Discovery (P3) | 7 | 0 |
| 9 | Polish | 8 | 3 |
| **Total** | | **83** | **29** |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- The constitution requires pytest unit tests for backend features
