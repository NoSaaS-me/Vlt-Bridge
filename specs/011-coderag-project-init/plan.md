# Implementation Plan: CodeRAG Project Integration

**Branch**: `011-coderag-project-init` | **Date**: 2026-01-01 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/011-coderag-project-init/spec.md`

## Summary

Implement interactive project selection for CodeRAG initialization, with background indexing via the vlt daemon and progress visibility in both CLI and web UI. Key deliverables:

1. **CLI**: Interactive `vlt coderag init` with project selection/creation
2. **Daemon**: Background indexing job processing with progress tracking
3. **Backend API**: New `/api/coderag/*` endpoints for status and job management
4. **Frontend**: Code Index status panel in Settings page with progress bar

## Technical Context

**Language/Version**: Python 3.11+ (CLI, backend), TypeScript 5.x (frontend)
**Primary Dependencies**:
- CLI: typer, rich, SQLAlchemy
- Backend: FastAPI, Pydantic, httpx
- Frontend: React 19, shadcn/ui, Tailwind CSS
**Storage**: SQLite (`~/.vlt/vault.db` for CLI, `data/index.db` for backend)
**Testing**: pytest (backend), manual verification (frontend)
**Target Platform**: Linux/macOS servers, modern browsers
**Project Type**: Web application (monorepo with CLI package)
**Performance Goals**:
- Init workflow < 60 seconds (excluding actual indexing)
- Progress updates visible within 5 seconds
- Status API response < 200ms
**Constraints**:
- Background indexing must survive terminal closure
- No silent data loss on project deletion
**Scale/Scope**: Single user per instance, codebases up to 100k files

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Brownfield Integration | ✅ Pass | Extends existing coderag module, adds new routes |
| II. Test-Backed Development | ✅ Pass | New backend routes require pytest tests |
| III. Incremental Delivery | ✅ Pass | CLI, daemon, API, frontend can ship independently |
| IV. Specification-Driven | ✅ Pass | Full spec in 011-coderag-project-init/spec.md |
| No Magic | ✅ Pass | Explicit job model, no hidden state |
| Single Source of Truth | ✅ Pass | CLI DB is authoritative for job state |
| Error Handling | ✅ Pass | API returns structured errors, CLI shows clear messages |

**Post-Design Re-check**: All gates still pass. No violations to track.

## Project Structure

### Documentation (this feature)

```text
specs/011-coderag-project-init/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Technical research findings
├── data-model.md        # Entity definitions
├── quickstart.md        # Implementation guide
├── contracts/           # API contracts
│   └── coderag-api.yaml # OpenAPI 3.0 spec
├── checklists/          # Quality checklists
│   └── requirements.md  # Spec validation
└── tasks.md             # Task breakdown (via /speckit.tasks)
```

### Source Code (repository root)

```text
# CLI Package
packages/vlt-cli/src/vlt/
├── core/
│   ├── models.py        # Add CodeRAGIndexJob model
│   └── service.py       # Add list_projects(), has_coderag_index()
├── daemon/
│   └── server.py        # Add process_coderag_jobs() background task
└── main.py              # Update coderag_init command, add coderag_delete

# Backend
backend/src/
├── api/
│   ├── routes/
│   │   └── coderag.py   # NEW: /api/coderag/* endpoints
│   └── main.py          # Register coderag router
├── models/
│   └── coderag.py       # NEW: Pydantic models for API
└── services/
    ├── oracle_bridge.py # Add coderag status methods
    └── project_service.py # Add cascade delete for coderag

# Frontend
frontend/src/
├── pages/
│   └── Settings.tsx     # Add Code Index status section
├── services/
│   └── coderag.ts       # NEW: API client for coderag endpoints
└── types/
    └── coderag.ts       # NEW: TypeScript types
```

**Structure Decision**: Web application structure with additional CLI package. All three layers (CLI, backend, frontend) are modified to implement the full feature.

## Complexity Tracking

No constitution violations. Feature uses established patterns:
- CLI: Same typer/rich patterns as existing commands
- Daemon: Same asyncio background task pattern as sync queue
- Backend: Same FastAPI router pattern as other routes
- Frontend: Same Card-based Settings section pattern

## Implementation Phases

### Phase 1: CLI Interactive Init (P1 stories)
- Add `list_projects()` and `has_coderag_index()` to service
- Update `coderag_init` with rich.prompt interactive selection
- Add project overwrite protection with force flag

### Phase 2: Background Job Infrastructure (P2 stories)
- Add `CodeRAGIndexJob` model to CLI models.py
- Add `process_coderag_jobs()` to daemon server.py
- Add progress callback to `CodeRAGIndexer.index_full()`

### Phase 3: Backend API Routes
- Create `backend/src/api/routes/coderag.py`
- Create `backend/src/models/coderag.py`
- Register router in main.py

### Phase 4: Frontend Status Panel
- Create `frontend/src/services/coderag.ts`
- Add Code Index section to Settings.tsx
- Implement progress bar with polling

### Phase 5: Cascade Delete & Cleanup
- Add `coderag_delete` CLI command
- Update `project_service.py` to call cleanup on project delete

## Artifacts Generated

| Artifact | Path | Status |
|----------|------|--------|
| Research | `specs/011-coderag-project-init/research.md` | ✅ Complete |
| Data Model | `specs/011-coderag-project-init/data-model.md` | ✅ Complete |
| API Contract | `specs/011-coderag-project-init/contracts/coderag-api.yaml` | ✅ Complete |
| Quickstart | `specs/011-coderag-project-init/quickstart.md` | ✅ Complete |
| Tasks | `specs/011-coderag-project-init/tasks.md` | Pending `/speckit.tasks` |
