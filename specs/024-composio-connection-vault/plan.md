# Implementation Plan: Composio Connection Vault

**Branch**: `024-composio-connection-vault` | **Date**: 2026-03-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/024-composio-connection-vault/spec.md`

## Summary

The Composio integration assumes single-connection-per-app with managed OAuth for all apps. This fails for apps like Twitter (no managed auth) and prevents multi-account usage (personal + work Gmail). The Connection Vault adds a local connection registry, adaptive auth detection, and explicit connection routing — fixing 3 bugs and enabling multi-account support across the full stack (service, backend, frontend, MCP, CLI).

## Technical Context

**Language/Version**: Python 3.11+ (backend, service, CLI), TypeScript / React 19 (frontend)
**Primary Dependencies**: FastAPI, Composio SDK (`composio-core`), Pydantic, shadcn/ui
**Storage**: SQLite (raw `sqlite3`, no ORM — matches existing codebase pattern)
**Testing**: pytest (backend unit tests), manual verification (frontend)
**Target Platform**: Linux server + web browser
**Project Type**: Web application (backend + frontend + CLI/MCP)
**Performance Goals**: All new endpoints <500ms, auth-info cached per session
**Constraints**: Must not break existing single-connection flows, must work offline for local registry reads
**Scale/Scope**: Max 10 connections per app per user, ~200 Composio apps in catalog

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate | Status | Notes |
|------|--------|-------|
| Brownfield Integration | PASS | Extends existing `composio.py` service, `composio_hub.py` routes, `ConnectorsPage.tsx`. No rewrites. |
| Test-Backed Development | PASS | New service methods get pytest unit tests. Frontend is manual verification. |
| Incremental Delivery | PASS | 3 phases: fix bugs (P0) -> registry (P1) -> MCP routing (P2). Each independently shippable. |
| Specification-Driven | PASS | Full spec at `specs/024-composio-connection-vault/spec.md` |
| No Magic | PASS | Explicit auth detection via `testConnectors`, explicit connection routing via `connection_id` |
| Single Source of Truth | PASS | Composio stores tokens, local SQLite stores metadata (labels, routing). No duplication. |
| Error Handling | PASS | All endpoints return structured errors. Frontend handles via Alert component. |
| SQLite for persistence | PASS | Uses existing raw sqlite3 pattern with DDL_STATEMENTS + MIGRATION_STATEMENTS |
| Pydantic for validation | PASS | New request/response models use Pydantic BaseModel |

**Post-design re-check**: All gates still PASS. No new abstractions, no ORM, no migration framework. New table follows exact DDL pattern from `database.py`.

## Project Structure

### Documentation (this feature)

```text
specs/024-composio-connection-vault/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 research findings
├── data-model.md        # Entity schemas and DDL
├── quickstart.md        # Implementation guide with code snippets
├── contracts/
│   ├── api.yaml         # OpenAPI contract for REST endpoints
│   └── mcp.yaml         # MCP tool contract changes
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
# Service layer (vlt-connectors package)
packages/vlt-connectors/src/vlt_connectors/service/
└── composio.py                    # Updated: auth_info, initiate_connection, disconnect, execute

# Backend (FastAPI)
backend/src/
├── services/
│   ├── database.py                # Updated: composio_connections DDL
│   └── composio_connections.py    # NEW: connection registry service
├── api/routes/
│   └── composio_hub.py            # Updated: auth-info, connect body, connection routing
└── models/
    └── composio.py                # NEW: Pydantic models for request/response

# Frontend (React)
frontend/src/
├── pages/
│   └── ConnectorsPage.tsx         # Updated: connection list, auth form dialog
└── services/
    └── composio-hub.ts            # Updated: new API client functions

# MCP tools (vlt-cli)
packages/vlt-cli/src/vlt/mcp/
└── connector_tools.py             # Updated: connection_id param

# CLI
packages/vlt-cli/src/vlt/
└── main.py                        # Updated: --connection-id flag

# Tests
backend/tests/unit/
└── test_composio_connections.py   # NEW: connection registry tests
packages/vlt-connectors/tests/
└── test_composio_service.py       # NEW: service method tests
```

**Structure Decision**: Web application pattern. Changes span all 3 tiers (service, backend, frontend) plus MCP/CLI. No new packages — extends existing files and adds 2 new service files.

## Complexity Tracking

No constitution violations. All patterns match existing codebase conventions.
