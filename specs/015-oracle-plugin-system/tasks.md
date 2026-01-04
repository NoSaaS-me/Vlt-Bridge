# Tasks: Oracle Plugin System

**Input**: Design documents from `/specs/015-oracle-plugin-system/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/rules-api.yaml

**Tests**: Included per constitution requirement (Test-Backed Development)

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US6)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependencies, and directory structure

- [X] T001 Add lupa and simpleeval dependencies to backend/pyproject.toml
- [X] T002 [P] Create plugins service directory structure in backend/src/services/plugins/
- [X] T003 [P] Create built-in rules directory in backend/src/services/plugins/rules/
- [X] T004 [P] Create scripts directory with README in backend/src/services/plugins/scripts/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Add QUERY_START and SESSION_END event types to backend/src/services/ans/event.py
- [X] T006 Create plugin_state table migration in backend/src/services/database.py
- [X] T007 Add disabled_rules_json column to user_settings table in backend/src/services/database.py
- [X] T008 [P] Create HookPoint enum in backend/src/services/plugins/rule.py
- [X] T009 [P] Create ActionType enum in backend/src/services/plugins/rule.py
- [X] T010 Create Rule dataclass with validation in backend/src/services/plugins/rule.py
- [X] T011 Create RuleAction dataclass in backend/src/services/plugins/rule.py
- [X] T012 [P] Create TurnState dataclass in backend/src/services/plugins/context.py
- [X] T013 [P] Create HistoryState dataclass in backend/src/services/plugins/context.py
- [X] T014 [P] Create UserState dataclass in backend/src/services/plugins/context.py
- [X] T015 [P] Create ProjectState dataclass in backend/src/services/plugins/context.py
- [X] T016 Create PluginState dataclass with get() method in backend/src/services/plugins/context.py
- [X] T017 Create RuleContext dataclass combining all state in backend/src/services/plugins/context.py
- [X] T018 Create plugins __init__.py with public exports in backend/src/services/plugins/__init__.py

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Simple Threshold Rules (Priority: P1) MVP

**Goal**: Operators can create rules with threshold conditions without code

**Independent Test**: Create a threshold rule TOML, trigger the condition, verify notification appears

### Tests for User Story 1

- [X] T019 [P] [US1] Unit test for RuleLoader TOML parsing in backend/tests/unit/services/plugins/test_loader.py
- [X] T020 [P] [US1] Unit test for ExpressionEvaluator in backend/tests/unit/services/plugins/test_expression.py
- [X] T021 [P] [US1] Unit test for Rule validation in backend/tests/unit/services/plugins/test_loader.py

### Implementation for User Story 1

- [X] T022 [US1] Implement ExpressionEvaluator using simpleeval in backend/src/services/plugins/expression.py
- [X] T023 [US1] Implement RuleLoader with TOML discovery and validation in backend/src/services/plugins/loader.py
- [X] T024 [P] [US1] Create token-budget-warning.toml built-in rule in backend/src/services/plugins/rules/token_budget.toml
- [X] T025 [P] [US1] Create iteration-budget-warning.toml built-in rule in backend/src/services/plugins/rules/iteration_budget.toml
- [X] T026 [P] [US1] Create large-result-hint.toml built-in rule in backend/src/services/plugins/rules/large_result.toml
- [X] T027 [P] [US1] Create repeated-failure-warning.toml built-in rule in backend/src/services/plugins/rules/repeated_failure.toml
- [X] T028 [US1] Implement ActionDispatcher for notify_self action in backend/src/services/plugins/actions.py
- [X] T029 [US1] Add log and set_state actions to ActionDispatcher in backend/src/services/plugins/actions.py
- [X] T030 [US1] Add emit_event action to ActionDispatcher in backend/src/services/plugins/actions.py

**Checkpoint**: User Story 1 complete - rules can be defined and expressions evaluated

---

## Phase 4: User Story 2 - Hook Point Integration (Priority: P1)

**Goal**: Rules attach to agent lifecycle events and fire at precise moments

**Independent Test**: Create rules on different hook points, run agent session, verify rules fire at expected moments

### Tests for User Story 2

- [X] T031 [P] [US2] Unit test for RuleEngine event subscription in backend/tests/unit/services/plugins/test_engine.py
- [X] T032 [P] [US2] Integration test for hook point firing in backend/tests/integration/test_rules_hooks.py

### Implementation for User Story 2

- [X] T033 [US2] Implement RuleEngine with EventBus subscription in backend/src/services/plugins/engine.py
- [X] T034 [US2] Add rule evaluation logic to RuleEngine in backend/src/services/plugins/engine.py
- [X] T035 [US2] Add priority-ordered execution to RuleEngine in backend/src/services/plugins/engine.py
- [X] T036 [US2] Emit QUERY_START event in oracle_agent.py query() method
- [X] T037 [US2] Emit SESSION_END event in oracle_agent.py finally block
- [X] T038 [US2] Wire RuleEngine initialization in oracle_agent.py __init__
- [X] T039 [US2] Add rule context building from agent state in oracle_agent.py

**Checkpoint**: User Story 2 complete - rules fire on lifecycle events

---

## Phase 5: User Story 3 - Context API Access (Priority: P1)

**Goal**: Rules can read agent state through a defined context API

**Independent Test**: Create rule referencing context.turn.token_usage, trigger rule, verify correct value

### Tests for User Story 3

- [X] T040 [P] [US3] Unit test for RuleContextBuilder in backend/tests/unit/services/plugins/test_context.py
- [X] T041 [P] [US3] Unit test for PluginState persistence in backend/tests/unit/services/plugins/test_context.py

### Implementation for User Story 3

- [X] T042 [US3] Implement RuleContextBuilder in backend/src/services/plugins/context.py
- [X] T043 [US3] Add TurnState population from agent state in context.py
- [X] T044 [US3] Add HistoryState population from collected tool calls in context.py
- [X] T045 [US3] Add UserState population from user_settings in context.py
- [X] T046 [US3] Add ProjectState population from project settings in context.py
- [X] T047 [US3] Implement PluginStateService for SQLite persistence in backend/src/services/plugins/state.py
- [X] T048 [US3] Wire PluginState with database service in context.py

**Checkpoint**: User Story 3 complete - rules have full context access

---

## Phase 6: User Story 4 - Script Escape Hatch (Priority: P2)

**Goal**: Complex logic can use Lua scripts with full context access

**Independent Test**: Create rule with script reference, trigger it, verify script executes with context

### Tests for User Story 4

- [X] T049 [P] [US4] Unit test for LuaSandbox execution in backend/tests/unit/services/plugins/test_lua_sandbox.py
- [X] T050 [P] [US4] Unit test for LuaSandbox timeout in backend/tests/unit/services/plugins/test_lua_sandbox.py
- [X] T051 [P] [US4] Unit test for LuaSandbox sandboxing in backend/tests/unit/services/plugins/test_lua_sandbox.py

### Implementation for User Story 4

- [X] T052 [US4] Implement LuaSandbox with lupa wrapper in backend/src/services/plugins/lua_sandbox.py
- [X] T053 [US4] Add environment whitelisting to LuaSandbox in lua_sandbox.py
- [X] T054 [US4] Add threading-based timeout enforcement in lua_sandbox.py
- [X] T055 [US4] Add memory limit enforcement (max_memory) in lua_sandbox.py
- [X] T056 [US4] Implement context exposure to Lua scripts in lua_sandbox.py
- [X] T057 [US4] Add script execution path to RuleEngine in engine.py
- [X] T058 [US4] Add error handling and propagation for Lua errors in engine.py

**Checkpoint**: User Story 4 complete - complex logic via Lua scripts

---

## Phase 7: User Story 5 - Rule Management UI (Priority: P2)

**Goal**: Users can view, enable/disable rules through Settings interface

**Independent Test**: Open Settings, navigate to Rules, toggle rule, verify it no longer fires

### Tests for User Story 5

- [X] T059 [P] [US5] Integration test for Rules API endpoints in backend/tests/integration/test_rules_api.py

### Implementation for User Story 5

- [X] T060 [US5] Create Pydantic models for Rules API in backend/src/models/rule.py
- [X] T061 [US5] Implement GET /api/rules endpoint in backend/src/api/routes/rules.py
- [X] T062 [US5] Implement GET /api/rules/{rule_id} endpoint in backend/src/api/routes/rules.py
- [X] T063 [US5] Implement POST /api/rules/{rule_id}/toggle endpoint in backend/src/api/routes/rules.py
- [X] T064 [US5] Implement POST /api/rules/{rule_id}/test endpoint in backend/src/api/routes/rules.py
- [X] T065 [US5] Register rules router in backend/src/api/main.py
- [X] T066 [US5] Add disabled_rules handling to RuleEngine in engine.py
- [X] T067 [P] [US5] Create TypeScript types for Rules in frontend/src/types/rules.ts
- [X] T068 [P] [US5] Create Rules API client functions in frontend/src/services/rules.ts
- [X] T069 [US5] Create RuleSettings component with rule list in frontend/src/components/RuleSettings.tsx
- [X] T070 [US5] Add rule toggle functionality to RuleSettings.tsx
- [X] T071 [US5] Add test button for demo users in RuleSettings.tsx
- [X] T072 [US5] Add Rules tab to Settings page in frontend/src/pages/Settings.tsx

**Checkpoint**: User Story 5 complete - rules manageable via UI

---

## Phase 8: User Story 6 - Plugin Manifest and Discovery (Priority: P3)

**Goal**: Plugins packaged with manifest declaring rules, capabilities, and dependencies

**Independent Test**: Add plugin directory with manifest, restart service, verify plugin's rules appear

### Tests for User Story 6

- [X] T073 [P] [US6] Unit test for PluginLoader manifest parsing in backend/tests/unit/services/plugins/test_plugin_loader.py
- [X] T074 [P] [US6] Unit test for Plugin dependency validation in backend/tests/unit/services/plugins/test_plugin_loader.py

### Implementation for User Story 6

- [X] T075 [US6] Create Plugin dataclass in backend/src/services/plugins/plugin.py
- [X] T076 [US6] Create PluginSetting dataclass in backend/src/services/plugins/plugin.py
- [X] T077 [US6] Implement PluginLoader with manifest parsing in backend/src/services/plugins/plugin_loader.py
- [X] T078 [US6] Add dependency validation to PluginLoader in plugin_loader.py
- [X] T079 [US6] Add plugin directory scanning to PluginLoader in plugin_loader.py
- [X] T080 [US6] Create plugins directory in backend/src/services/plugins/plugins/
- [X] T081 [US6] Implement GET /api/plugins endpoint in backend/src/api/routes/rules.py
- [X] T082 [US6] Implement GET /api/plugins/{plugin_id} endpoint in rules.py
- [X] T083 [US6] Implement GET/PUT /api/plugins/{plugin_id}/settings endpoints in rules.py
- [X] T084 [US6] Implement GET/DELETE /api/plugins/{plugin_id}/state endpoints in rules.py
- [X] T085 [US6] Add plugin_settings_json column handling in user_settings.py
- [X] T086 [P] [US6] Add Plugin types to frontend/src/types/rules.ts
- [X] T087 [P] [US6] Add Plugin API functions to frontend/src/services/rules.ts
- [X] T088 [US6] Add plugin section to RuleSettings component

**Checkpoint**: User Story 6 complete - full plugin system

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, optimization, and refinements

- [X] T089 [P] Create docs/plugin-api/README.md overview documentation
- [X] T090 [P] Create docs/plugin-api/architecture/overview.md
- [X] T091 [P] Create docs/plugin-api/architecture/performance.md
- [X] T092 [P] Create docs/plugin-api/architecture/roadmap.md with stretch goals
- [X] T093 [P] Create docs/plugin-api/rules/format.md TOML schema documentation
- [X] T094 [P] Create docs/plugin-api/rules/conditions.md expression language guide
- [X] T095 [P] Create docs/plugin-api/rules/actions.md action types documentation
- [X] T096 [P] Create docs/plugin-api/rules/examples.md common patterns
- [X] T097 [P] Create docs/plugin-api/context-api/reference.md full API reference
- [X] T098 [P] Create docs/plugin-api/hooks/lifecycle.md hook point documentation
- [X] T099 [P] Create docs/plugin-api/scripting/lua-guide.md Lua scripting guide
- [X] T100 [P] Create docs/plugin-api/scripting/sandbox.md security model
- [X] T101 [P] Create docs/plugin-api/built-ins/token-budget.md
- [X] T102 [P] Create docs/plugin-api/built-ins/iteration-budget.md
- [X] T103 [P] Create docs/plugin-api/built-ins/large-result.md
- [X] T104 [P] Create docs/plugin-api/built-ins/repeated-failure.md
- [X] T105 Run quickstart.md validation and fix any issues
- [X] T106 Performance optimization: rule evaluation caching
- [X] T107 Add logging for rule evaluation timing

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-8)**: All depend on Foundational phase completion
  - US1-US3 are all P1 priority and build the core engine
  - US4-US5 are P2 priority and add scripting and UI
  - US6 is P3 priority for the full plugin system
- **Polish (Phase 9)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Foundation only - standalone threshold rules
- **User Story 2 (P1)**: Foundation + US1 (needs rules to evaluate)
- **User Story 3 (P1)**: Foundation + US2 (needs hooks to provide context)
- **User Story 4 (P2)**: Foundation + US1-3 (Lua scripts need full context)
- **User Story 5 (P2)**: Foundation + US1-3 (UI manages existing rules)
- **User Story 6 (P3)**: Foundation + US1-5 (plugins organize rules)

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Models/dataclasses before services
- Services before API endpoints
- Backend before frontend (API must exist)
- Core implementation before integration

### Parallel Opportunities

**Setup (Phase 1)**:
```
T002, T003, T004 can run in parallel
```

**Foundational (Phase 2)**:
```
T008, T009 can run in parallel (enums)
T012, T013, T014, T015 can run in parallel (state dataclasses)
```

**User Story 1**:
```
T019, T020, T021 can run in parallel (tests)
T024, T025, T026, T027 can run in parallel (built-in rules)
```

**User Story 5**:
```
T067, T068 can run in parallel (frontend types/services)
```

**Polish (Phase 9)**:
```
T089-T104 can all run in parallel (documentation)
```

---

## Implementation Strategy

### MVP First (User Stories 1-3)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: User Story 1 - Threshold Rules
4. Complete Phase 4: User Story 2 - Hook Integration
5. Complete Phase 5: User Story 3 - Context API
6. **STOP and VALIDATE**: Test all P1 stories independently
7. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 (Threshold Rules) → Test → Deploy/Demo (MVP!)
3. Add US2 (Hook Points) → Test → Deploy/Demo
4. Add US3 (Context API) → Test → Deploy/Demo
5. Add US4 (Lua Scripts) → Test → Deploy/Demo
6. Add US5 (Management UI) → Test → Deploy/Demo
7. Add US6 (Plugin System) → Test → Deploy/Demo

### Suggested MVP Scope

**Minimum Viable Product**: User Stories 1, 2, 3 (P1 priority)
- TOML-based threshold rules
- Hook point integration
- Context API access
- 4 built-in rules

This delivers the core rule engine without Lua scripting or UI, testable via API/config.

---

## Summary

| Phase | User Story | Tasks | Parallel |
|-------|------------|-------|----------|
| 1 | Setup | 4 | 3 |
| 2 | Foundational | 14 | 6 |
| 3 | US1 - Threshold Rules | 12 | 7 |
| 4 | US2 - Hook Points | 9 | 2 |
| 5 | US3 - Context API | 9 | 2 |
| 6 | US4 - Lua Scripts | 10 | 3 |
| 7 | US5 - Rule Management UI | 14 | 3 |
| 8 | US6 - Plugin System | 16 | 4 |
| 9 | Polish | 19 | 16 |
| **Total** | | **107** | **46** |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- Each user story is independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
