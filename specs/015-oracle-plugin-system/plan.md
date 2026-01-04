# Implementation Plan: Oracle Plugin System

**Branch**: `015-oracle-plugin-system` | **Date**: 2026-01-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/015-oracle-plugin-system/spec.md`

## Summary

Implement a rule engine and plugin architecture built on ANS (Agent Notification System) that enables reactive and proactive agent behaviors. The system uses a tiered complexity model: 80% of use cases handled by TOML rule definitions with `simpleeval` expressions, 20% by Lua scripts via `lupa`. Hook points integrate with the existing ANS EventBus for agent lifecycle events.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript 5.x (frontend)
**Primary Dependencies**:
- Backend: FastAPI, Pydantic, lupa (Lua), simpleeval (expressions)
- Frontend: React 18+, shadcn/ui

**Storage**: SQLite (extend existing schema for plugin_state table)
**Testing**: pytest (backend), Vitest (frontend)
**Target Platform**: Linux server (development), Docker (production)
**Project Type**: Web application (frontend + backend)

**Performance Goals**:
- Rule evaluation: <50ms per rule
- Condition parsing: <1ms (simpleeval)
- Lua script execution: <5s timeout

**Constraints**:
- Memory: <100MB for Lua sandbox (lupa max_memory)
- Sandboxing: No filesystem/network access in Lua scripts
- Core rules cannot be disabled

**Scale/Scope**:
- MVP: 4-6 built-in rules, unlimited custom rules
- Target: <100 rules per user/project

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Brownfield Integration | ✅ PASS | Extends ANS, follows existing patterns |
| II. Test-Backed Development | ✅ PASS | pytest for rule engine, expressions, Lua sandbox |
| III. Incremental Delivery | ✅ PASS | 6 user stories with clear priorities |
| IV. Specification-Driven | ✅ PASS | Full spec with acceptance criteria |
| Technology Standards (Backend) | ✅ PASS | Python 3.11+, FastAPI, Pydantic, SQLite |
| Technology Standards (Frontend) | ✅ PASS | React 18+, TypeScript, shadcn/ui |
| No Magic | ✅ PASS | TOML + explicit expressions, no hidden behavior |
| Single Source of Truth | ✅ PASS | Rules as files, state in SQLite |
| Error Handling | ✅ PASS | Structured errors for invalid rules/expressions |

## Project Structure

### Documentation (this feature)

```text
specs/015-oracle-plugin-system/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 research output
├── data-model.md        # Entity definitions
├── quickstart.md        # Getting started guide
├── contracts/
│   └── rules-api.yaml   # OpenAPI spec for Rules API
└── checklists/
    └── requirements.md  # Spec validation checklist
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   └── rule.py              # Rule, RuleAction, RuleContext Pydantic models
│   ├── services/
│   │   └── plugins/
│   │       ├── __init__.py
│   │       ├── rule.py          # Rule dataclass
│   │       ├── loader.py        # RuleLoader (TOML discovery)
│   │       ├── engine.py        # RuleEngine (evaluation, dispatch)
│   │       ├── lua_sandbox.py   # LuaSandbox (lupa integration)
│   │       ├── expression.py    # ExpressionEvaluator (simpleeval)
│   │       ├── actions.py       # ActionDispatcher
│   │       ├── context.py       # RuleContext builder
│   │       ├── rules/           # Built-in rule TOML files
│   │       │   ├── token_budget.toml
│   │       │   ├── iteration_budget.toml
│   │       │   ├── large_result.toml
│   │       │   └── repeated_failure.toml
│   │       └── scripts/         # Built-in Lua scripts
│   │           └── README.md
│   ├── api/
│   │   └── routes/
│   │       └── rules.py         # Rules API endpoints
│   └── services/
│       └── ans/
│           └── event.py         # Add QUERY_START, SESSION_END event types
└── tests/
    ├── unit/
    │   └── services/
    │       └── plugins/
    │           ├── test_loader.py
    │           ├── test_engine.py
    │           ├── test_lua_sandbox.py
    │           └── test_expression.py
    └── integration/
        └── test_rules_api.py

frontend/
├── src/
│   ├── types/
│   │   └── rules.ts             # TypeScript interfaces
│   ├── services/
│   │   └── rules.ts             # API client functions
│   └── components/
│       └── RuleSettings.tsx     # Settings UI for rules
└── tests/
    └── unit/
        └── rules.test.ts

docs/
└── plugin-api/
    ├── README.md
    ├── architecture/
    │   ├── overview.md
    │   ├── performance.md
    │   └── roadmap.md
    ├── rules/
    │   ├── format.md
    │   ├── conditions.md
    │   ├── actions.md
    │   └── examples.md
    ├── context-api/
    │   ├── reference.md
    │   ├── turn.md
    │   ├── history.md
    │   └── state.md
    ├── hooks/
    │   ├── lifecycle.md
    │   └── events.md
    ├── scripting/
    │   ├── lua-guide.md
    │   ├── sandbox.md
    │   └── examples.md
    └── built-ins/
        ├── token-budget.md
        ├── iteration-budget.md
        ├── large-result.md
        └── repeated-failure.md
```

**Structure Decision**: Web application pattern with backend/frontend separation. Plugin system lives in `backend/src/services/plugins/` as a new service module parallel to `ans/`. Documentation in `docs/plugin-api/` at project root.

## Complexity Tracking

No constitution violations requiring justification.

## Key Technical Decisions

### 1. Expression Evaluation (simpleeval)

Use `simpleeval` library for TOML condition expressions:
- Safe AST-based evaluation (no `eval()`)
- Supports boolean composition (`and`, `or`, `not`)
- Configurable function whitelist

### 2. Lua Embedding (lupa)

Use `lupa` for Lua script execution:
- LuaJIT provides 20-30x speedup over Python
- In-process (no subprocess overhead)
- Environment whitelisting for sandboxing
- Threading-based timeout enforcement

### 3. Hook Integration

Extend existing ANS EventBus:
- Add `QUERY_START` and `SESSION_END` event types
- Rules subscribe as specialized handlers
- Reuse existing event emission patterns

### 4. State Storage

Plugin state stored in SQLite:
```sql
CREATE TABLE plugin_state (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    plugin_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    UNIQUE(user_id, project_id, plugin_id, key)
);
```

## Implementation Phases

### Phase 1: Core Rule Engine (P1 Stories)

**US1: Simple Threshold Rules**
- [ ] Create `Rule` dataclass and Pydantic models
- [ ] Implement `RuleLoader` (TOML discovery, validation)
- [ ] Implement `ExpressionEvaluator` (simpleeval wrapper)
- [ ] Create built-in rules (4 TOMLs)

**US2: Hook Point Integration**
- [ ] Add `QUERY_START`, `SESSION_END` to EventType enum
- [ ] Emit events at hook points in oracle_agent.py
- [ ] Implement `RuleEngine` (subscribe to events, evaluate rules)

**US3: Context API Access**
- [ ] Create `RuleContext` builder
- [ ] Expose turn, history, user, project state
- [ ] Add `PluginState` with get() method

### Phase 2: Scripting & Actions (P2 Stories)

**US4: Script Escape Hatch**
- [ ] Implement `LuaSandbox` (lupa wrapper)
- [ ] Environment whitelisting
- [ ] Timeout enforcement (threading)
- [ ] Error handling and propagation

**US5: Rule Management UI**
- [ ] Create Rules API endpoints
- [ ] Add RuleSettings component to Settings page
- [ ] Implement rule toggle functionality

### Phase 3: Plugin System (P3 Stories)

**US6: Plugin Manifest and Discovery**
- [ ] Define manifest.toml schema
- [ ] Implement plugin directory scanning
- [ ] Plugin settings in user_settings

## Dependencies

- **ANS EventBus**: Rules subscribe to ANS events
- **Oracle Agent Loop**: Hook point emissions
- **User Settings Service**: Disabled rules storage
- **Database Service**: Plugin state persistence

## Generated Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| Research | specs/015-oracle-plugin-system/research.md | ✅ Complete |
| Data Model | specs/015-oracle-plugin-system/data-model.md | ✅ Complete |
| API Contract | specs/015-oracle-plugin-system/contracts/rules-api.yaml | ✅ Complete |
| Quickstart | specs/015-oracle-plugin-system/quickstart.md | ✅ Complete |
| Docs Structure | docs/plugin-api/ | ✅ Created |
| Tasks | specs/015-oracle-plugin-system/tasks.md | 🟡 Next: /speckit.tasks |
