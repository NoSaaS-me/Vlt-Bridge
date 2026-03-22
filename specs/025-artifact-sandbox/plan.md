# Implementation Plan: Artifact Sandbox

**Branch**: `025-artifact-sandbox` | **Date**: 2026-03-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/025-artifact-sandbox/spec.md`

## Summary

Artifact Sandbox is a plugin system for the Vlt platform that allows users and AI agents to create, develop, test, and deploy executable bundles (JS/CSS/HTML frontend + Python backend) within a sandboxed environment. Each artifact has a state machine enforcing quality (draft → building → testing → reviewing → approved → deployed), hot reload for rapid iteration, vision model review for visual QA, artifact-to-artifact IPC, connector integration with multi-instance support, and the ability to expose MCP tools to other agents. Artifacts are embedded in the agents view as a new tab, served through the daemon, and managed via both UI and MCP tools.

## Technical Context

**Language/Version**: Python 3.11+ (daemon/backend), TypeScript (frontend, React 19)
**Primary Dependencies**: FastAPI (daemon routes), watchdog 4.0+ (file watching), Playwright (screenshots), Pydantic (models), SQLModel/SQLAlchemy (DB), shadcn/ui (frontend components)
**Storage**: SQLite (artifacts table in daemon vault.db + connector/proxy/vision in backend index.db) + filesystem (artifact source files)
**Testing**: pytest (backend/daemon), manual + Playwright assertions (frontend)
**Target Platform**: Linux server (daemon host)
**Project Type**: Web application (monorepo: daemon + backend + frontend)
**Performance Goals**: Hot reload <2s, backend restart <5s, screenshot capture <10s, event delivery <500ms, MCP tool response <2s
**Constraints**: Artifact frontend isolation via iframe sandbox, backend process isolation via subprocess, no direct credential access from artifact code
**Scale/Scope**: Dozens of artifacts per user, up to 5 concurrent backend processes, individual artifact quotas (CPU, memory, storage)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Brownfield Integration | PASS | Extends existing patterns (NavSection, daemon routes, connector schema). No rewrites of working code. |
| II. Test-Backed Development | PASS | Backend artifact service will have pytest unit tests. Frontend relies on manual verification per constitution. |
| III. Incremental Delivery | PASS | 9 user stories with P1/P2/P3 priorities enable incremental implementation. New tab + routes don't destabilize existing features. |
| IV. Specification-Driven | PASS | Full spec at specs/025-artifact-sandbox/spec.md with 35 FRs and 10 SCs. |
| No Magic | PASS | Explicit JSON-lines protocol, explicit state machine transitions, no hidden registration. |
| Single Source of Truth | PASS | Artifact source on disk, metadata in SQLite. Disk is authoritative for code, DB for state. |
| Error Handling | PASS | All endpoints return structured errors. Vision model absence is non-blocking. |

**Post-Phase 1 re-check**: Constitution check remains PASS. Data model follows existing SQLite patterns. API contracts use standard REST. No ORM beyond existing SQLModel usage.

## Project Structure

### Documentation (this feature)

```text
specs/025-artifact-sandbox/
├── spec.md
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── api.md           # Phase 1 output
└── checklists/
    └── requirements.md
```

### Source Code (repository root)

```text
# Daemon (artifact runtime)
packages/vlt-cli/src/vlt/daemon/
├── server.py                      # app.include_router(artifact_router)
├── artifact_routes.py             # NEW: REST + WebSocket endpoints
├── artifact_service.py            # NEW: Artifact lifecycle, process mgmt
├── artifact_watcher.py            # NEW: File watching + hot reload
├── artifact_event_bus.py          # NEW: IPC event routing
└── artifact_harness.py            # NEW: Backend process wrapper script

packages/vlt-cli/src/vlt/mcp/
└── artifact_tools.py              # NEW: MCP tools for artifact CRUD

packages/vlt-cli/src/vlt/core/
└── models.py                      # EXTEND: Artifact SQLModel

# Backend (connector multi-instance, proxy profiles, vision settings)
backend/src/services/
├── database.py                    # EXTEND: migrations for instance_id, proxy_profiles
├── connector_service.py           # EXTEND: instance_id parameter
└── model_provider.py              # EXTEND: supports_vision field

backend/src/api/routes/
├── connectors.py                  # EXTEND: instance endpoints
├── models.py                      # EXTEND: vision model settings
└── proxy_profiles.py              # NEW: proxy profile CRUD

backend/src/models/
└── settings.py                    # EXTEND: vision_model, vision_provider

# Frontend
frontend/src/pages/
└── AgentsPage.tsx                 # EXTEND: artifacts NavSection

frontend/src/components/artifacts/
├── ArtifactsCompositorView.tsx    # NEW: Main artifacts layout
├── ArtifactSidebar.tsx            # NEW: Artifact list sidebar
├── ArtifactViewer.tsx             # NEW: Iframe container + VltBridge handler
├── ArtifactStateBar.tsx           # NEW: State machine display
├── ArtifactImportExport.tsx       # NEW: Zip import/export UI
└── NewArtifactDialog.tsx          # NEW: Creation dialog

frontend/src/services/
└── artifact-api.ts                # NEW: Artifact REST/WS client

frontend/src/lib/
└── vlt-bridge-host.ts             # NEW: postMessage handler (parent side)

# Artifact template (injected into iframe)
frontend/public/
└── vlt-bridge.js                  # NEW: VltBridge client (iframe side)

# Data
data/artifacts/                    # Artifact source files (per user)
```

**Structure Decision**: Web application pattern. Artifact runtime lives in the daemon (packages/vlt-cli) since it manages processes and file watching. Connector and model extensions live in the backend. UI components live in the frontend. This follows the existing monorepo boundaries.

## Complexity Tracking

No constitution violations requiring justification. The feature adds new files following existing patterns without exceeding complexity gates.
