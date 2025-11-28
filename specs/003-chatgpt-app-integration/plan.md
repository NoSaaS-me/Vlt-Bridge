# Implementation Plan: ChatGPT App Integration

**Branch**: `003-chatgpt-app-integration` | **Date**: 2025-11-26 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-chatgpt-app-integration/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Adapt the backend to support ChatGPT Apps SDK by adding a static service token auth strategy, configuring CORS for ChatGPT, and updating MCP tools to return metadata for widget rendering. Create a standalone `widget.html` entry point in the frontend.

## Technical Context

**Language/Version**: Python 3.11+, TypeScript/React 18
**Primary Dependencies**: `fastmcp`, `fastapi`, `vite`
**Storage**: No schema changes; relies on existing vault.
**Testing**: `pytest` for auth/tools; manual verification for widgets.
**Target Platform**: Hugging Face Spaces (Docker) + ChatGPT UI.
**Project Type**: Full Stack (Backend API + Frontend Widget).
**Performance Goals**: Widget load < 500ms.
**Constraints**: Must work alongside existing "local dev" and "HF OAuth" modes.
**Scale/Scope**: Demo scale (single tenant impersonation via service token).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

-   **Brownfield Integration**: Respects existing auth structure (adding a strategy, not rewriting). Reuses `NoteViewer` component.
-   **Test-Backed**: New auth strategy will be unit tested.
-   **Simplicity**: Using static token instead of full OIDC.

## Project Structure

### Documentation (this feature)

```text
specs/003-chatgpt-app-integration/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── api/
│   │   ├── main.py            # Update: CORS, widget route
│   │   └── middleware/
│   │       └── auth_middleware.py # Update: Strategy usage
│   ├── services/
│   │   ├── auth.py            # Update: Refactor to Strategy pattern
│   │   └── config.py          # Update: New config fields
│   └── mcp/
│       └── server.py          # Update: Tool return types
└── tests/
    └── unit/
        └── test_auth_strategy.py # New test

frontend/
├── vite.config.ts             # Update: Multi-page build
├── widget.html                # New: Widget entry point
└── src/
    └── widget.tsx             # New: Widget root component
```

**Structure Decision**: Multi-page build for Frontend; Strategy pattern for Backend Auth.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multi-page Vite | To isolate widget styles/scripts from main app | Iframing the full app is too heavy and leaky for ChatGPT widgets. |