# Implementation Plan: Oracle Agent Turn Control

**Branch**: `012-oracle-turn-control` | **Date**: 2026-01-02 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/012-oracle-turn-control/spec.md`

## Summary

Refactor the Oracle agent's turn-taking logic to introduce enterprise-grade termination conditions (iteration limits, token budgets, timeouts, no-progress detection), create a pluggable DecisionTree protocol for future skill extensions, expose agent configuration in the Settings UI, and add system notifications visible in chat when limits are approached or reached.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript 5.x (frontend)
**Primary Dependencies**: FastAPI, Pydantic, React 18+, shadcn/ui
**Storage**: SQLite (existing user_settings table)
**Testing**: pytest (backend), manual verification (frontend per constitution)
**Target Platform**: Linux server (backend), Web browser (frontend)
**Project Type**: Web application (backend/ + frontend/)
**Performance Goals**: <500ms termination detection, <100ms settings save
**Constraints**: No breaking changes to existing SSE stream consumers
**Scale/Scope**: Single user settings table, ~7 new config fields

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Brownfield Integration** | PASS | Refactoring existing oracle_agent.py, matching existing patterns |
| **II. Test-Backed Development** | PASS | Unit tests for DecisionTree, AgentState, termination logic |
| **III. Incremental Delivery** | PASS | Can deploy config without UI first, then add UI |
| **IV. Specification-Driven** | PASS | All work traced to spec.md FR-001 through FR-019 |
| **No Magic** | PASS | Explicit Protocol interface, no metaclass tricks |
| **Single Source of Truth** | PASS | Settings in user_settings table, state in AgentState |
| **Error Handling** | PASS | Structured system messages for limit notifications |

## Project Structure

### Documentation (this feature)

```text
specs/012-oracle-turn-control/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 research findings
├── data-model.md        # Entity definitions
├── quickstart.md        # Developer quick reference
├── contracts/           # API contracts
│   ├── settings-api.yaml    # Settings endpoints
│   └── oracle-stream.md     # SSE chunk types
└── tasks.md             # Implementation tasks (Phase 2)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── settings.py          # MODIFY: Add AgentConfig fields
│   │   ├── oracle.py            # MODIFY: Add system chunk type
│   │   └── agent_state.py       # NEW: AgentState dataclass
│   ├── services/
│   │   ├── oracle_agent.py      # MODIFY: Integrate DecisionTree
│   │   ├── user_settings.py     # MODIFY: AgentConfig persistence
│   │   └── decision_tree/       # NEW: DecisionTree module
│   │       ├── __init__.py
│   │       ├── protocol.py      # DecisionTree Protocol
│   │       ├── default.py       # DefaultDecisionTree
│   │       └── registry.py      # Decorator-based registration
│   └── api/
│       └── routes/
│           └── models.py        # MODIFY: AgentConfig endpoints
└── tests/
    └── unit/
        ├── test_agent_state.py      # NEW
        ├── test_decision_tree.py    # NEW
        └── test_termination.py      # NEW

frontend/
├── src/
│   ├── types/
│   │   └── oracle.ts            # MODIFY: Add system role, AgentConfig
│   ├── components/
│   │   ├── ChatPanel.tsx        # MODIFY: Handle system chunks
│   │   ├── ChatMessage.tsx      # MODIFY: System message styling
│   │   └── AgentConfigPanel.tsx # NEW: Settings UI for AgentConfig
│   ├── pages/
│   │   └── Settings.tsx         # MODIFY: Add AgentConfig section
│   └── services/
│       └── api.ts               # MODIFY: AgentConfig API calls
└── tests/
    # Manual verification per constitution
```

**Structure Decision**: Web application structure (Option 2) - modifications to existing backend/frontend directories following established patterns.

## Complexity Tracking

No constitution violations requiring justification.

## Implementation Phases

### Phase 1: Backend Core (Priority: P1)

1. **AgentState dataclass** (`backend/src/models/agent_state.py`)
   - Immutable state with `frozen=True, kw_only=True`
   - Track: turn, tokens_used, start_time, recent_actions, termination_reason
   - Extension field for future modules

2. **DecisionTree Protocol** (`backend/src/services/decision_tree/protocol.py`)
   - `should_continue(state) -> (bool, reason)`
   - `on_turn_start(state) -> state`
   - `on_tool_result(state, result) -> state`
   - `get_config() -> AgentConfig`

3. **DefaultDecisionTree** (`backend/src/services/decision_tree/default.py`)
   - Implement termination conditions per FR-004
   - No-progress detection (3 consecutive identical actions)
   - Token/iteration/timeout tracking

4. **AgentConfig in settings** (`backend/src/models/settings.py`)
   - Add 7 new fields to ModelSettings
   - Validation bounds per FR-003

### Phase 2: Oracle Agent Integration (Priority: P1)

1. **Refactor query() method** (`backend/src/services/oracle_agent.py`)
   - Replace hardcoded MAX_TURNS with config
   - Create AgentState at query start
   - Call DecisionTree hooks at appropriate points
   - Emit system chunks for warnings

2. **System chunk type** (`backend/src/models/oracle.py`)
   - Add "system" to chunk type literal
   - Add system_type and system_message fields

3. **Settings API** (`backend/src/api/routes/models.py`)
   - GET/PUT endpoints for AgentConfig
   - Validation per FR-003

### Phase 3: Frontend (Priority: P1-P3)

1. **TypeScript types** (`frontend/src/types/oracle.ts`)
   - Add 'system' to role union
   - Add AgentConfig interface

2. **ChatPanel system handling** (`frontend/src/components/ChatPanel.tsx`)
   - Handle 'system' chunk type
   - Create system message in conversation

3. **ChatMessage system styling** (`frontend/src/components/ChatMessage.tsx`)
   - Distinct visual treatment (amber warning style)
   - No action buttons (read-only)

4. **Settings UI** (`frontend/src/pages/Settings.tsx`)
   - AgentConfig section with sliders/inputs
   - Auto-save on change

### Phase 4: Testing (Priority: P2)

1. **Unit tests for AgentState**
   - Immutability verification
   - Extension pattern

2. **Unit tests for DecisionTree**
   - Each termination condition
   - No-progress detection algorithm

3. **Integration tests**
   - End-to-end limit enforcement
   - System message delivery

## Key Decisions from Research

| Decision | Implementation |
|----------|---------------|
| Protocol over ABC | `typing.Protocol` with `@runtime_checkable` |
| Immutable state | `@dataclass(frozen=True, kw_only=True)` |
| Decorator registration | `_decision_trees` dict with `@decision_tree` decorator |
| Settings persistence | Extend existing ModelSettings, no new table |
| System messages | New SSE chunk type, new message role |
| No-progress detection | Compare stringified tool calls, 3-action window |

## Dependencies

- **Existing**: FastAPI, Pydantic, React, shadcn/ui, SQLite
- **No new dependencies required**

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking SSE consumers | High | Add chunk type, don't change existing types |
| Performance impact from state tracking | Medium | Use lightweight dataclass, minimal overhead |
| User confusion from system messages | Medium | Clear, actionable message content |
| Token counting inaccuracy | Low | Document as approximate, use estimates |

## Next Steps

Run `/speckit.tasks` to generate implementation task checklist.
