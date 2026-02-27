# Tasks: BT-Controlled Oracle Agent

**Input**: Design documents from `/specs/020-bt-oracle-agent/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Included as specified in Constitution (Test-Backed Development)

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1-US5)
- All paths relative to repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, copy prompt templates, verify dependencies

- [x] T001 Copy prompt templates from specs/020-bt-oracle-agent/prompts/oracle/ to backend/src/prompts/oracle/
- [x] T002 [P] Create backend/src/models/signals.py with Signal, SignalType enum, and field dataclasses per data-model.md
- [x] T003 [P] Create backend/src/models/query_classification.py with QueryType enum and QueryClassification dataclass per data-model.md
- [x] T004 [P] Create backend/src/bt/conditions/__init__.py to expose conditions module
- [x] T005 Verify BT runtime (019) tests pass: `cd backend && uv run pytest tests/unit/bt/ -q`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core services that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Create backend/src/services/signal_parser.py with parse_signal() and strip_signal() functions per research.md
- [x] T007 [P] Write backend/tests/unit/test_signal_parser.py with tests for all 6 signal types, malformed XML, and edge cases
- [x] T008 Create backend/src/services/query_classifier.py with classify_query() function using keyword heuristics per research.md
- [x] T009 [P] Write backend/tests/unit/test_query_classifier.py with tests for code/docs/research/conversational/action classification
- [x] T010 Create backend/src/services/prompt_composer.py with compose_prompt() function and segment registry per data-model.md
- [x] T011 [P] Write backend/tests/unit/test_prompt_composer.py with tests for segment composition rules
- [x] T012 Run all foundational tests: `cd backend && uv run pytest tests/unit/test_signal_parser.py tests/unit/test_query_classifier.py tests/unit/test_prompt_composer.py -v`

**Checkpoint**: Foundation ready - all 3 core services implemented and tested

---

## Phase 3: User Story 1 - Intelligent Context Selection (Priority: P1)

**Goal**: Agent intelligently decides what context sources to use based on query classification

**Independent Test**: Ask "What's the weather in Paris?" - agent should use web_search only, not code/vault

### Implementation for User Story 1

- [x] T013 [US1] Add query_classification field to blackboard schema in backend/src/bt/state/blackboard.py
- [x] T014 [US1] Create backend/src/bt/actions/query_analysis.py with analyze_query() action that calls classify_query()
- [x] T015 [US1] Create backend/src/bt/conditions/context_needs.py with needs_code_context(), needs_vault_context(), needs_web_context() conditions
- [x] T016 [US1] Modify backend/src/bt/trees/oracle-agent.lua to add Context Assessment Phase after initialization (before context loading)
- [x] T017 [US1] Update backend/src/bt/actions/oracle.py build_system_prompt() to call prompt_composer instead of loading monolithic prompt
- [x] T018 [P] [US1] Write backend/tests/unit/bt/test_query_analysis.py with tests for query→classification→context needs flow
- [x] T019 [US1] Integration test: verify "weather in Paris" routes to web_search only in backend/tests/integration/test_oracle_bt_integration.py

**Checkpoint**: Query classification determines context strategy - agent no longer searches everything

---

## Phase 4: User Story 2 - Agent Self-Reflection via Signals (Priority: P1)

**Goal**: Agent emits XML signals that BT parses to manage conversation flow

**Independent Test**: Tool failure scenario triggers `need_turn` signal emission and parsing

### Implementation for User Story 2

- [x] T020 [US2] Create backend/src/bt/conditions/signals.py with check_signal(), has_signal(), signal_type_is() conditions
- [x] T021 [US2] Create backend/src/bt/actions/signal_actions.py with parse_response_signal(), log_signal(), strip_signal_from_response() actions
- [x] T022 [US2] Modify backend/src/bt/wrappers/oracle_wrapper.py to call signal parser after each LLM response
- [x] T023 [US2] Add signal state tracking to blackboard: last_signal, signals_emitted list, consecutive_same_reason counter
- [x] T024 [US2] Update backend/src/bt/trees/oracle-agent.lua agent loop to check signals after llm_call and route accordingly
- [x] T025 [P] [US2] Write backend/tests/unit/bt/test_signal_conditions.py with tests for signal-based BT conditions
- [x] T026 [US2] Integration test: verify tool failure triggers need_turn signal in backend/tests/integration/test_oracle_bt_integration.py

**Checkpoint**: Agent self-reflection works - signals are emitted, parsed, and drive BT decisions

---

## Phase 5: User Story 3 - Budget and Loop Enforcement (Priority: P2)

**Goal**: BT enforces turn budgets and detects infinite loops from signal patterns

**Independent Test**: Set max_turns=5 and verify agent is forced to respond after 5 turns

### Implementation for User Story 3

- [x] T027 [US3] Create backend/src/bt/conditions/budget.py with turns_remaining(), is_at_budget_limit(), is_over_budget() conditions
- [x] T028 [US3] Create backend/src/bt/conditions/loop_detection.py with is_stuck_loop() condition checking consecutive_same_reason >= 3
- [x] T029 [US3] Create backend/src/bt/actions/budget_actions.py with force_completion(), emit_budget_warning() actions
- [x] T030 [US3] Update backend/src/bt/trees/oracle-agent.lua agent loop with budget guards and loop detection
- [x] T031 [US3] Add turn budget configuration to backend/src/config.py (ORACLE_MAX_TURNS, default 30)
- [x] T032 [P] [US3] Write backend/tests/unit/bt/test_budget_conditions.py with tests for budget limits
- [x] T033 [P] [US3] Write backend/tests/unit/bt/test_loop_detection.py with tests for stuck loop detection
- [x] T034 [US3] Integration test: verify max_turns=5 forces completion in backend/tests/integration/test_oracle_bt_integration.py

**Checkpoint**: Safety guardrails work - budget enforced, loops detected, agent can't spin forever

---

## Phase 6: User Story 4 - Dynamic Prompt Composition (Priority: P2)

**Goal**: System prompts are composed from segments based on query type

**Independent Test**: Code question includes code-analysis.md segment, excludes research.md segment

### Implementation for User Story 4

- [x] T035 [US4] Ensure all prompt segments exist in backend/src/prompts/oracle/ (base.md, signals.md, tools-reference.md, code-analysis.md, documentation.md, research.md, conversation.md)
- [x] T036 [US4] Add segment loader to prompt_composer.py that reads from backend/src/prompts/oracle/
- [x] T037 [US4] Implement segment priority ordering per data-model.md (base=0, signals=1, tools=2, context-specific=10)
- [x] T038 [US4] Add token budget tracking to prompt_composer.py (max 8000 tokens)
- [x] T039 [US4] Verify signals.md is ALWAYS included regardless of query type
- [x] T040 [P] [US4] Write backend/tests/unit/test_prompt_composition_segments.py with tests for segment inclusion/exclusion logic
- [x] T041 [US4] Integration test: verify code query prompt includes code-analysis.md in backend/tests/integration/test_oracle_bt_integration.py

**Checkpoint**: Prompts are dynamic - different queries get different prompt compositions

---

## Phase 7: User Story 5 - BERT Fallback for Edge Cases (Priority: P3)

**Goal**: Heuristic fallback when agent signals are absent or weak (BERT placeholder)

**Independent Test**: 3 turns without signals triggers fallback classification

### Implementation for User Story 5

- [x] T042 [US5] Create backend/src/services/fallback_classifier.py with heuristic_classify() function (BERT placeholder)
- [x] T043 [US5] Create backend/src/bt/conditions/fallback.py with needs_fallback() condition (no signal for 3+ turns OR confidence < 0.3 OR stuck signal)
- [x] T044 [US5] Create backend/src/bt/actions/fallback_actions.py with trigger_fallback(), apply_heuristic_classification() actions
- [x] T045 [US5] Update backend/src/bt/trees/oracle-agent.lua with fallback selector after signal check
- [x] T046 [US5] Add fallback logging to track when and why fallback activates
- [x] T047 [P] [US5] Write backend/tests/unit/test_fallback_classifier.py with tests for heuristic classification
- [x] T048 [P] [US5] Write backend/tests/unit/bt/test_fallback_conditions.py with tests for fallback trigger conditions
- [x] T049 [US5] Integration test: verify 3 turns without signals triggers fallback in backend/tests/integration/test_oracle_bt_integration.py

**Checkpoint**: Fallback works - system gracefully handles missing/weak signals

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Shadow mode, documentation, final validation

- [x] T050 Update backend/src/bt/wrappers/shadow_mode.py to compare signal emission between legacy and BT implementations
- [x] T051 Add signal discrepancy tracking to shadow mode comparison report
- [x] T052 [P] Update backend/src/config.py to add ORACLE_USE_BT environment variable documentation
- [x] T053 [P] Update CLAUDE.md with 020-bt-oracle-agent technology additions
- [x] T054 Run full test suite: `cd backend && uv run pytest tests/unit/bt/ tests/unit/test_signal_parser.py tests/unit/test_query_classifier.py tests/unit/test_prompt_composer.py tests/integration/test_oracle_bt_integration.py -v`
- [x] T055 Run quickstart.md validation: follow steps and verify shadow mode works
- [x] T056 Manual E2E test: Enable shadow mode, run 10 diverse queries, analyze comparison logs

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 and US2 are both P1 and can run in parallel
  - US3 and US4 are both P2 and can run in parallel (after P1 complete)
  - US5 is P3 and runs after P2 complete
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

| Story | Priority | Can Start After | Depends On |
|-------|----------|-----------------|------------|
| US1 | P1 | Phase 2 complete | Foundational only |
| US2 | P1 | Phase 2 complete | Foundational only |
| US3 | P2 | Phase 2 complete | US2 (needs signals for loop detection) |
| US4 | P2 | Phase 2 complete | US1 (needs classification for segment selection) |
| US5 | P3 | Phase 2 complete | US2 (needs signal absence detection) |

### Within Each User Story

1. Conditions before actions (conditions are checked, actions execute)
2. BT modifications after Python implementations
3. Tests can be written in parallel with implementation
4. Integration test last (needs everything wired up)

### Parallel Opportunities

**Phase 1 (Setup)**:
- T002, T003, T004 can run in parallel (different model files)

**Phase 2 (Foundational)**:
- T007, T009, T011 can run in parallel (test files)
- T006, T008, T010 must be sequential (services depend on each other conceptually but different files)

**Phase 3-7 (User Stories)**:
- US1 and US2 can run completely in parallel (both P1)
- US3 and US4 can start in parallel after P1 complete
- Within each story, [P] tasks can run in parallel

---

## Parallel Example: User Story 1 + User Story 2 (P1 stories)

```bash
# After Phase 2 completes, launch both P1 stories in parallel:

# Stream 1: User Story 1 (Context Selection)
Task: T013 [US1] Add query_classification to blackboard
Task: T014 [US1] Create query_analysis.py actions
Task: T015 [US1] Create context_needs.py conditions
Task: T016 [US1] Modify oracle-agent.lua Context Assessment
...

# Stream 2: User Story 2 (Signal Self-Reflection)
Task: T020 [US2] Create signals.py conditions
Task: T021 [US2] Create signal_actions.py
Task: T022 [US2] Modify oracle_wrapper.py
...
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (signal parser, classifier, composer)
3. Complete Phase 3: User Story 1 (intelligent context selection)
4. Complete Phase 4: User Story 2 (signal self-reflection)
5. **STOP and VALIDATE**: Test query routing and signal emission
6. Enable shadow mode and monitor

### Incremental Delivery

| Milestone | Stories | Value Delivered |
|-----------|---------|-----------------|
| Foundation | Setup + Phase 2 | Core services ready |
| MVP | + US1 + US2 | Context selection + signals working |
| Safety | + US3 | Budget and loop protection |
| Efficiency | + US4 | Optimized prompts |
| Robustness | + US5 | Fallback for edge cases |
| Production | + Polish | Shadow mode validated |

### Risk Mitigation

- Shadow mode allows rollback if BT implementation degrades quality
- Each story is independently testable
- Heuristics work without BERT (can add later)

---

## Task Summary

| Phase | Tasks | Parallel Opportunities |
|-------|-------|------------------------|
| Setup | 5 | 3 |
| Foundational | 7 | 3 |
| US1 (P1) | 7 | 1 |
| US2 (P1) | 7 | 1 |
| US3 (P2) | 8 | 2 |
| US4 (P2) | 7 | 1 |
| US5 (P3) | 8 | 2 |
| Polish | 7 | 2 |
| **Total** | **56** | **15** |

---

## Notes

- All [P] tasks can run in parallel within their phase
- [Story] labels enable filtering: `grep "[US1]" tasks.md`
- Signal parsing is <10ms (research.md) - no performance concerns
- All 53 oracle actions already implemented (research.md) - mostly wiring
- Prompt templates already exist in spec - just copy to backend
- Shadow mode infrastructure exists - just add signal comparison
