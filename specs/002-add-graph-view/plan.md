# Implementation Plan: Interactive Graph View

**Branch**: `002-add-graph-view` | **Date**: 2025-11-25 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-add-graph-view/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Add an interactive force-directed graph visualization to the frontend (`react-force-graph-2d`) backed by a new API endpoint (`GET /api/graph`) that serves node and link data derived from the existing SQLite index.

## Technical Context

**Language/Version**: Python 3.11+ (Backend), TypeScript/React 18 (Frontend)
**Primary Dependencies**: `react-force-graph-2d` (New), `FastAPI`, `sqlite3`
**Storage**: SQLite (`note_metadata`, `note_links` tables)
**Testing**: `pytest` (Backend), Manual/E2E (Frontend)
**Target Platform**: Web Browser (Canvas/WebGL support required)
**Project Type**: Web Application (Full Stack)
**Performance Goals**: Render <1000 nodes in <2 seconds.
**Constraints**: Must support Light/Dark mode dynamically.
**Scale/Scope**: Personal knowledge base scale (hundreds to thousands of notes).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

-   **Library-First**: N/A (Application feature).
-   **CLI Interface**: N/A (UI feature).
-   **Test-First**: Backend endpoint will be tested via `pytest`.
-   **Simplicity**: Using a proven library (`react-force-graph`) to avoid complex custom D3 implementations.

## Project Structure

### Documentation (this feature)

```text
specs/002-add-graph-view/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── graph-api.yaml
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── api/
│   │   └── routes/
│   │       └── graph.py       # New endpoint
│   ├── models/
│   │   └── graph.py           # New Pydantic models
│   └── services/
│       └── indexer.py         # Update: add get_graph_data()
└── tests/
    └── unit/
        └── test_graph_api.py  # New tests

frontend/
├── src/
│   ├── components/
│   │   └── GraphView.tsx      # New component
│   ├── pages/
│   │   └── MainApp.tsx        # Update: Add toggle
│   ├── services/
│   │   └── api.ts             # Update: Add getGraphData()
│   └── types/
│       └── graph.ts           # New types
```

**Structure Decision**: Standard Full-Stack layout.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A       |            |                                     |