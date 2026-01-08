# Implementation Plan: BT-Controlled Oracle Agent

**Branch**: `020-bt-oracle-agent` | **Date**: 2026-01-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/020-bt-oracle-agent/spec.md`

## Summary

Migrate the Oracle agent from a monolithic Python control loop to a Behavior Tree (BT) controlled architecture. The LLM (Claude) remains the primary decision-maker for tool calls, but now emits atomic XML signals that the BT parses to manage turn budgets, detect loops, and trigger fallbacks. Prompts are composed dynamically based on query classification.

**Key Technical Approach:**
- BT as "rubber ducky" - reads agent signals, enforces constraints
- XML signal protocol embedded in agent responses
- Composable prompt segments loaded based on query type
- Shadow mode for safe parallel testing
- BERT fallback (heuristic-only for MVP) when signals absent/weak

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript 5.x (frontend - no changes)
**Primary Dependencies**: FastAPI, Pydantic, lupa (Lua), existing BT runtime (019)
**Storage**: SQLite (existing index.db for state persistence)
**Testing**: pytest (unit + integration), manual E2E for agent behavior
**Target Platform**: Linux server (local mode), Hugging Face Spaces (space mode)
**Project Type**: web (backend + frontend monorepo)
**Performance Goals**: Signal parsing <50ms, BT overhead <10% latency increase
**Constraints**: Max 30 turns per conversation, prompt <8000 tokens
**Scale/Scope**: Single Oracle agent, ~10 prompt files, ~20 new/modified Python files

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Brownfield Integration | PASS | Extends existing oracle_agent.py via wrapper, doesn't rewrite. Matches existing FastAPI/Pydantic patterns. |
| II. Test-Backed Development | PASS | Will include pytest tests for signal parsing, BT conditions, prompt composition. |
| III. Incremental Delivery | PASS | Shadow mode allows old and new implementations to run in parallel. Feature flag controls cutover. |
| IV. Specification-Driven | PASS | Full spec at specs/020-bt-oracle-agent/spec.md with prompts as deliverables. |

**Technology Standards Check:**
- Backend: Python 3.11+, FastAPI, Pydantic, SQLite - all match
- Frontend: No changes required
- No Magic: BT control flow is explicit, signals are parseable XML
- Single Source of Truth: BT state in blackboard, signals logged for audit
- Error Handling: Signal parse failures handled gracefully with BERT fallback

**GATE PASSED** - Proceed to Phase 0

## Project Structure

### Documentation (this feature)

```text
specs/020-bt-oracle-agent/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
├── checklists/          # Validation checklists
│   └── requirements.md
└── prompts/oracle/      # Prompt deliverables (created in spec phase)
    ├── base.md
    ├── signals.md
    ├── tools-reference.md
    ├── code-analysis.md
    ├── documentation.md
    ├── research.md
    └── conversation.md
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   └── signals.py           # NEW: Signal dataclasses
│   ├── services/
│   │   ├── oracle_agent.py      # MODIFY: Add signal parsing hook
│   │   ├── signal_parser.py     # NEW: XML signal extraction
│   │   ├── prompt_composer.py   # NEW: Dynamic prompt assembly
│   │   └── query_classifier.py  # NEW: Query type classification
│   ├── bt/
│   │   ├── actions/
│   │   │   └── oracle.py        # MODIFY: Complete action implementations
│   │   ├── conditions/
│   │   │   └── signals.py       # NEW: Signal-based conditions
│   │   └── wrappers/
│   │       └── oracle_wrapper.py # MODIFY: Integrate signal parsing
│   └── prompts/oracle/          # NEW: Prompt templates (copy from spec)
└── tests/
    ├── unit/
    │   ├── test_signal_parser.py
    │   ├── test_prompt_composer.py
    │   └── test_query_classifier.py
    └── integration/
        └── test_oracle_bt_integration.py
```

**Structure Decision**: Web application structure (backend/ + frontend/). All changes are backend-only. Frontend continues to use existing ChatPanel with no modifications.

## Complexity Tracking

No constitution violations - table not needed.

---

## Phase 0: Research

### Research Tasks

1. **Signal Parsing Performance** - Validate regex-based XML extraction is <50ms
2. **Query Classification Heuristics** - Define keyword-based classification before BERT
3. **Prompt Token Budgets** - Measure actual token counts for composed prompts
4. **Shadow Mode Integration** - Review existing shadow_mode.py patterns
5. **BT Action Completion** - Audit which oracle.py actions are stubs vs implemented

### Research Findings

See [research.md](./research.md) for detailed findings.

---

## Phase 1: Design

### Data Model

See [data-model.md](./data-model.md) for entity definitions.

### API Contracts

See [contracts/](./contracts/) for OpenAPI schemas.

### Quick Start

See [quickstart.md](./quickstart.md) for developer onboarding.
