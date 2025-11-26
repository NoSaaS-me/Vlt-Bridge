# Tasks: Interactive Graph View

**Feature**: 002-add-graph-view
**Status**: Pending
**Spec**: [spec.md](spec.md)

## Phase 1: Setup (Project Initialization)

*Goal: Install dependencies and define core data structures.*

- [x] T001 [P] Install `react-force-graph-2d` dependency in frontend/package.json
- [x] T002 [P] Define Pydantic models (`GraphNode`, `GraphLink`, `GraphData`) in backend/src/models/graph.py
- [x] T003 [P] Define TypeScript interfaces (`GraphNode`, `GraphLink`, `GraphData`) in frontend/src/types/graph.ts

## Phase 2: Foundational Tasks

*Goal: Establish core backend logic required for all user stories.*

- [x] T004 Update `IndexerService` with `get_graph_data()` method in backend/src/services/indexer.py
- [x] T005 Implement `GET /api/graph` endpoint in backend/src/api/routes/graph.py
- [x] T006 Register graph router in backend/src/api/main.py
- [x] T007 [P] Add `getGraphData` function to frontend/src/services/api.ts
- [x] T008 Create unit tests for graph API in backend/tests/unit/test_graph_api.py

## Phase 3: User Story 1 - Structural Overview

*Goal: Visualize the note vault structure as an interactive graph.*

- [x] T009 [US1] Create `GraphView` component with basic force-directed graph in frontend/src/components/GraphView.tsx
- [x] T010 [US1] Integrate `getGraphData` hook into `GraphView` to load real data in frontend/src/components/GraphView.tsx
- [x] T011 [US1] Add "Graph View" toggle button and conditional rendering logic in frontend/src/pages/MainApp.tsx
- [x] T012 [US1] Style the graph container to occupy the full center panel in frontend/src/pages/MainApp.tsx

## Phase 4: User Story 2 - Visual Navigation

*Goal: Enable navigation from the graph to specific notes.*

- [x] T013 [US2] Implement `onNodeClick` handler to trigger `onSelectNote` callback in frontend/src/components/GraphView.tsx
- [x] T014 [US2] Verify node hover tooltips display note titles correctly in frontend/src/components/GraphView.tsx
- [x] T015 [US2] Ensure switching back from Graph View restores the standard Note View in frontend/src/pages/MainApp.tsx

## Phase 5: User Story 3 - Orphan Identification & Visuals

*Goal: Highlight node connectivity and grouping.*

- [x] T016 [US3] Implement node sizing logic (`val` based on link count) in backend/src/services/indexer.py
- [x] T017 [US3] [P] Implement theme support (Dynamic Light/Dark colors) in frontend/src/components/GraphView.tsx
- [x] T018 [US3] [P] Implement categorical node coloring based on `group` (folder) in frontend/src/components/GraphView.tsx

## Final Phase: Polish

*Goal: Ensure stability and good UX.*

- [x] T019 Implement loading state spinner in `GraphView` while fetching data in frontend/src/components/GraphView.tsx
- [x] T020 Implement error handling banner in `GraphView` in frontend/src/components/GraphView.tsx
- [x] T021 Implement graph state persistence (zoom/pan) to retain view when toggling in frontend/src/components/GraphView.tsx

## Dependencies

1.  **Setup (T001-T003)**: Must be done first.
2.  **Foundational (T004-T008)**: Depends on Setup. Required for all US phases.
3.  **US1 (T009-T012)**: Depends on Foundational.
4.  **US2 (T013-T015)**: Depends on US1 (needs GraphView component).
5.  **US3 (T016-T018)**: Partially parallel with US2, but T016 (Backend sizing) is backend-only. T017/T018 modify GraphView.
6.  **Polish (T019-T020)**: Can be done anytime after US1.

## Parallel Execution Strategy

-   **Backend vs Frontend**: Once models (T002/T003) are agreed upon, T004-T006 (Backend) can run parallel to T007-T009 (Frontend).
-   **Within US3**: T017 (Theme) and T018 (Group Colors) are independent UI tasks.

## Implementation Strategy

1.  **MVP**: Complete Phases 1, 2, and 3 (US1). This delivers the core value: seeing the graph.
2.  **Interaction**: Phase 4 adds the critical navigation workflow.
3.  **Visuals**: Phase 5 enhances utility (orphans, importance).
