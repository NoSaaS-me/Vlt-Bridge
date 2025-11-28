# Tasks: ChatGPT App Integration

**Feature**: 003-chatgpt-app-integration
**Status**: Pending
**Spec**: [spec.md](spec.md)

## Phase 1: Setup (Project Initialization)

*Goal: Configure build pipelines and backend settings to support the new widget and auth modes.*

- [x] T001 Update `backend/src/services/config.py` to include `chatgpt_service_token` and `chatgpt_cors_origin` fields
- [x] T002 Update `frontend/vite.config.ts` to support multi-page build (`index.html` and `widget.html`)
- [x] T003 Create `frontend/widget.html` as the entry point for the ChatGPT widget
- [x] T004 Create `frontend/src/widget.tsx` as the root React component for the widget

## Phase 2: Foundational Tasks

*Goal: Establish authentication and infrastructure required for the integration.*

- [x] T005 Refactor `backend/src/services/auth.py` to implement Strategy pattern (`JWTValidator` and `StaticTokenValidator`)
- [x] T006 [P] Create unit tests for new Auth Strategy in `backend/tests/unit/test_auth_strategy.py`
- [x] T007 Update `backend/src/api/middleware/auth_middleware.py` to use the new Auth Service strategies
- [x] T008 Update `backend/src/api/main.py` to configure CORS for `https://chatgpt.com`
- [x] T009 [P] Update `backend/src/api/main.py` to serve `widget.html` on `/widget` route with `text/html+skybridge` MIME type

## Phase 3: User Story 1 - The Recall Loop

*Goal: Enable searching and viewing notes within ChatGPT.*

- [x] T010 [US1] Extract `NoteViewer` component logic into a reusable pure component (if not already) in `frontend/src/components/NoteViewer.tsx`
- [x] T011 [P] [US1] Create `SearchWidget` component in `frontend/src/components/SearchWidget.tsx` for the widget view
- [x] T012 [US1] Implement `WidgetApp` component in `frontend/src/widget.tsx` to handle routing between Note and Search views based on props/URL
- [x] T013 [US1] Update `backend/src/mcp/server.py` `read_note` tool to return `CallToolResult` with `_meta.openai.outputTemplate`
- [x] T014 [US1] Update `backend/src/mcp/server.py` `search_notes` tool to return `CallToolResult` with `_meta.openai.outputTemplate`

## Phase 4: User Story 2 - In-Context Editing

*Goal: Enable editing notes from ChatGPT interactions.*

- [x] T015 [US2] Verify `write_note` tool functionality with Service Token auth (no code change expected if T007 is correct, but validation needed)
- [x] T016 [US2] Implement auto-refresh or status indication in `WidgetApp` when a note is updated externally (by ChatGPT)

## Final Phase: Polish

*Goal: Ensure seamless experience and robustness.*

- [x] T017 Verify widget load performance and optimize bundle size if needed
- [x] T018 Add error boundary to `WidgetApp` to handle load failures gracefully inside the iframe

## Dependencies

1.  **Setup (T001-T004)**: Must be done first to enable widget development.
2.  **Foundational (T005-T009)**: Auth and serving infrastructure required before widgets can be loaded by ChatGPT.
3.  **US1 (T010-T014)**: Depends on Foundational. T013/T014 depend on T009 (widget serving).
4.  **US2 (T015-T016)**: Depends on US1.

## Parallel Execution Strategy

-   **Frontend vs Backend**: T002-T004 (Frontend Setup) can run parallel to T001 & T005-T008 (Backend Auth).
-   **Within US1**: T011 (Search Widget UI) and T013/T014 (MCP Tool Updates) can be done in parallel.

## Implementation Strategy

1.  **Infrastructure First**: Get the backend serving the widget HTML and accepting the service token.
2.  **Widget Shell**: Build the minimal widget shell that can display *something*.
3.  **Tool Connection**: Wire up the MCP tools to point to the widget.
4.  **Features**: Flesh out the Viewer and Search capabilities.
