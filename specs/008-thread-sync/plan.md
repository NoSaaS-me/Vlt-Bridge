# Implementation Plan: Thread Sync from CLI to Server

**Branch**: `008-thread-sync` | **Date**: 2025-12-31 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/008-thread-sync/spec.md`

## Summary

Enable vlt CLI to sync thread data to the Document-MCP backend server, replacing the broken subprocess-based Oracle approach. The CLI will POST thread entries to the backend after `vlt thread push`, and the backend will store threads locally for Oracle queries and Web UI display.

## Technical Context

**Language/Version**: Python 3.11+ (vlt-cli & backend), TypeScript 5.x (frontend)
**Primary Dependencies**: FastAPI, Pydantic, SQLAlchemy (vlt-cli), httpx, SQLite
**Storage**: SQLite (vlt-cli ~/.vlt/vault.db) + Backend SQLite (data/index.db)
**Testing**: pytest (backend), manual verification (frontend)
**Target Platform**: Linux server (backend), local machine (CLI)
**Project Type**: web (frontend + backend + CLI package)
**Performance Goals**: <5s sync latency, <2s UI load, 100 concurrent CLI sessions
**Constraints**: Incremental sync only (not full history), offline queue for failed syncs
**Scale/Scope**: Thousands of threads per user, 10KB average entry size

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Brownfield Integration | PASS | Extends existing vlt-cli thread commands, adds new backend routes following existing patterns |
| II. Test-Backed Development | PASS | Will add pytest tests for backend thread service and API endpoints |
| III. Incremental Delivery | PASS | Three phases: P1 sync infrastructure, P2 UI display, P3 Oracle integration |
| IV. Specification-Driven | PASS | All work traced to spec.md FR-001 through FR-016 |
| Technology: Python 3.11+ | PASS | Both vlt-cli and backend use Python 3.11+ |
| Technology: FastAPI/Pydantic | PASS | Backend follows FastAPI router pattern |
| Technology: SQLite | PASS | Both stores use SQLite (no ORM in backend, SQLAlchemy in vlt-cli) |
| Technology: React/TypeScript/Tailwind/shadcn | PASS | Frontend follows existing component patterns |
| Governance: No Magic | PASS | Explicit sync calls, no hidden synchronization |
| Governance: Single Source of Truth | PASS | CLI is source, backend is replicated view |
| Governance: Error Handling | PASS | Structured errors with retry queue |

**Gate Status**: PASS - No violations

## Project Structure

### Documentation (this feature)

```text
specs/008-thread-sync/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── threads-api.yaml # OpenAPI spec
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
# Web application structure (frontend + backend + CLI package)

packages/vlt-cli/src/vlt/
├── core/
│   ├── models.py              # Existing: Thread, Node, State ORM models
│   ├── service.py             # Existing: SqliteVaultService CRUD
│   ├── sync.py                # NEW: ThreadSyncClient for HTTP sync
│   └── identity.py            # Existing: OracleConfig with vault_url
├── config.py                  # Existing: Settings (add sync_token)
└── main.py                    # Existing: CLI commands (add sync hooks)

backend/
├── src/
│   ├── models/
│   │   ├── oracle.py          # Existing: OracleRequest, SourceReference
│   │   └── thread.py          # NEW: Thread, ThreadEntry, SyncStatus models
│   ├── services/
│   │   ├── database.py        # Existing: Add thread tables DDL
│   │   ├── oracle_bridge.py   # Existing: Replace subprocess with local queries
│   │   └── thread_service.py  # NEW: ThreadService for CRUD
│   └── api/
│       └── routes/
│           ├── oracle.py      # Existing: Modify to use ThreadService
│           └── threads.py     # NEW: /api/threads endpoints
└── tests/
    └── unit/
        └── test_thread_service.py  # NEW: pytest tests

frontend/
├── src/
│   ├── components/
│   │   └── ThreadsView.tsx    # NEW: Thread list display
│   ├── pages/
│   │   └── Threads.tsx        # NEW: Threads page
│   └── services/
│       └── api.ts             # Existing: Add thread API methods
└── tests/                     # Manual verification

data/
└── index.db                   # Backend SQLite (threads added here)
```

**Structure Decision**: Web application with three components:
1. **vlt-cli package** - Source of truth, pushes threads to backend
2. **backend** - Receives synced threads, serves API for Oracle and UI
3. **frontend** - Displays thread history, integrates with Oracle

## Complexity Tracking

> No violations - table not needed
