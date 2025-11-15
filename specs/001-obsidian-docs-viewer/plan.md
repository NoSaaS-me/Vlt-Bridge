# Implementation Plan: Multi-Tenant Obsidian-Like Docs Viewer

**Branch**: `001-obsidian-docs-viewer` | **Date**: 2025-11-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-obsidian-docs-viewer/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a multi-tenant Obsidian-like documentation viewer with AI-first workflow (AI writes via MCP, humans read/edit via web UI). System provides per-user vaults with Markdown notes, full-text search, wikilink resolution, tag indexing, and backlink tracking. Backend exposes FastMCP server (STDIO + HTTP transports) and HTTP API with Bearer auth (JWT). Frontend is React SPA with shadcn/ui, featuring directory tree navigation and split-pane editor. Deployment targets: local PoC (single-user, STDIO) and Hugging Face Space (multi-tenant, OAuth).

**Technical Approach**: Python backend with FastAPI + FastMCP, SQLite per-user indices, filesystem-based vault storage, JWT authentication. React + Vite frontend with react-markdown rendering. Incremental index updates on writes, optimistic concurrency for UI, last-write-wins for MCP.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: FastAPI, FastMCP, python-frontmatter, PyJWT, huggingface_hub, SQLite (stdlib)
**Storage**: Filesystem (per-user vault directories), SQLite (per-user indices)
**Testing**: pytest (backend unit/integration), Vitest (frontend unit), Playwright (E2E)
**Target Platform**: Linux server (HF Space), local dev (Windows/macOS/Linux)
**Project Type**: Web application (Python backend + React frontend)
**Performance Goals**:
- MCP operations: <500ms (read/write/search) for vaults with 1,000 notes
- UI rendering: <2s directory tree load, <1s note render, <1s search results
- Index rebuild: <30s for 1,000 notes
**Constraints**:
- 1 MiB max note size
- 5,000 notes max per vault
- 256 char max path length
- 100% tenant isolation (security requirement)
- 409 Conflict on concurrent edits (UI only)
**Scale/Scope**:
- MVP: 10 concurrent users (HF Space), 5,000 notes per user
- Local PoC: single user, unlimited notes (within filesystem limits)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: No project constitution file found at `.specify/memory/constitution.md`. Constitution check skipped.

**Justification**: This is a new project without established architectural principles. Design decisions will be documented in research.md and can form the basis of a future constitution if needed.

## Project Structure

### Documentation (this feature)

```text
specs/001-obsidian-docs-viewer/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   ├── http-api.yaml    # OpenAPI 3.1 spec for HTTP API
│   └── mcp-tools.json   # MCP tool schemas (JSON Schema)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# Web application structure (Python backend + React frontend)

backend/
├── src/
│   ├── models/          # Pydantic models (Note, User, Index, Config)
│   ├── services/
│   │   ├── vault.py     # Filesystem vault operations
│   │   ├── indexer.py   # Full-text search, tags, link graph
│   │   ├── auth.py      # JWT + HF OAuth integration
│   │   └── config.py    # Configuration management
│   ├── api/
│   │   ├── main.py      # FastAPI app + middleware
│   │   ├── routes/      # API endpoints (notes, search, auth, index)
│   │   └── middleware/  # Auth middleware, error handlers
│   └── mcp/
│       └── server.py    # FastMCP server (STDIO + HTTP)
├── tests/
│   ├── unit/            # Service-level tests
│   ├── integration/     # API + MCP integration tests
│   └── contract/        # Contract tests for MCP tools + HTTP API
└── pyproject.toml       # Python dependencies (Poetry/pip)

frontend/
├── src/
│   ├── components/
│   │   ├── ui/          # shadcn/ui components
│   │   ├── DirectoryTree.tsx
│   │   ├── NoteViewer.tsx
│   │   ├── NoteEditor.tsx
│   │   ├── SearchBar.tsx
│   │   └── AuthFlow.tsx
│   ├── pages/
│   │   ├── App.tsx      # Main app layout
│   │   ├── Login.tsx    # HF OAuth landing
│   │   └── Settings.tsx # User profile + token management
│   ├── services/
│   │   ├── api.ts       # HTTP API client (fetch wrapper)
│   │   └── auth.ts      # Token management, OAuth helpers
│   ├── lib/
│   │   ├── wikilink.ts  # Wikilink parsing + resolution
│   │   └── markdown.ts  # react-markdown config
│   └── types/           # TypeScript types (Note, User, SearchResult)
├── tests/
│   ├── unit/            # Component tests (Vitest + Testing Library)
│   └── e2e/             # Playwright E2E tests
├── package.json         # Node dependencies
└── vite.config.ts       # Vite build config

data/                    # Runtime data (gitignored)
└── vaults/
    └── <user_id>/       # Per-user vault directories

.env.example             # Environment template (JWT_SECRET, HF OAuth, etc.)
README.md                # Setup instructions, MCP client config examples
```

**Structure Decision**: Web application structure selected based on spec requirements for Python backend (FastAPI + FastMCP) and React frontend (shadcn/ui). Backend and frontend are separate codebases to support independent development/testing cycles, with backend serving frontend as static files in production (HF Space). Data directory is runtime-only (vaults + SQLite indices), not version controlled.

## Complexity Tracking

> **No constitution violations to track** (no project constitution exists yet)

## Phase 0: Research & Technical Decisions

**Status**: Research required for technology integration patterns and best practices.

**Research Topics**:
1. FastMCP HTTP transport authentication patterns (Bearer token validation)
2. Hugging Face Space OAuth integration best practices (attach/parse helpers)
3. SQLite schema design for per-user multi-index storage (full-text + tags + links)
4. Wikilink normalization and resolution algorithms (slug matching, ambiguity handling)
5. React + shadcn/ui directory tree component patterns (collapsible, virtualization)
6. Optimistic concurrency implementation patterns (ETags vs version counters)
7. Markdown frontmatter parsing with fallback strategies (malformed YAML handling)
8. JWT token management in React (localStorage vs memory, refresh strategies)

**Output**: See `research.md` for detailed findings and decisions.

## Phase 1: Data Model & Contracts

**Prerequisites**: `research.md` complete

### Data Model

**Entities** (see `data-model.md` for full schemas):

1. **User**: `user_id`, `hf_profile` (optional), `vault_path`, `created_at`
2. **Note**: `path`, `title`, `metadata`, `body`, `version`, `created`, `updated`
3. **Wikilink**: `source_path`, `link_text`, `target_path` (nullable), `is_resolved`
4. **Tag**: `tag_name`, `note_paths[]`
5. **Index**: `user_id`, `note_count`, `last_full_rebuild`, `last_incremental_update`
6. **Token**: `jwt` (claims: `sub`, `exp`, `iat`)

### API Contracts

**HTTP API** (see `contracts/http-api.yaml`):
- Authentication: `POST /api/tokens`, `GET /api/me`
- Notes CRUD: `GET /api/notes`, `GET /api/notes/{path}`, `PUT /api/notes/{path}`, `DELETE /api/notes/{path}`
- Search: `GET /api/search?q=<query>`
- Navigation: `GET /api/backlinks/{path}`, `GET /api/tags`
- Index: `GET /api/index/health`, `POST /api/index/rebuild`

**MCP Tools** (see `contracts/mcp-tools.json`):
- `list_notes`: `{folder?: string}` → `[{path, title, last_modified}]`
- `read_note`: `{path: string}` → `{path, title, metadata, body}`
- `write_note`: `{path, title?, metadata?, body}` → `{status, path}`
- `delete_note`: `{path: string}` → `{status}`
- `search_notes`: `{query: string}` → `[{path, title, snippet}]`
- `get_backlinks`: `{path: string}` → `[{path, title}]`
- `get_tags`: `{}` → `[{tag, count}]`

### Quickstart

**Output**: See `quickstart.md` for:
- Local development setup (Python venv, Node install, env config)
- Running backend (STDIO MCP + HTTP API)
- Running frontend (Vite dev server)
- MCP client configuration (Claude Code STDIO example)
- Testing workflows (unit, integration, E2E)

## Phase 2: Task Generation

**Not included in this command** - run `/speckit.tasks` to generate dependency-ordered implementation tasks based on this plan and the data model.

---

## Execution Summary

### Phase 0: Research & Technical Decisions ✅ COMPLETE

**Generated**: `research.md` (1,407 lines)

**Researched and Resolved**:
1. ✅ FastMCP HTTP transport authentication patterns → Bearer token with `BearerAuth`, JWT validation
2. ✅ Hugging Face Space OAuth integration → `attach_huggingface_oauth` + `parse_huggingface_oauth` helpers
3. ✅ SQLite schema design → FTS5 contentless + separate tags/links tables with per-user isolation
4. ✅ Wikilink normalization → Case-insensitive normalized slug matching, same-folder preference
5. ✅ React + shadcn/ui directory tree → shadcn-extension Tree View with TanStack Virtual virtualization
6. ✅ Optimistic concurrency → Version counter in SQLite + `if_version` parameter, 409 on mismatch
7. ✅ Markdown frontmatter parsing → `python-frontmatter` with try-except fallback for malformed YAML
8. ✅ JWT token management → Memory storage (MVP) or memory + HttpOnly cookie (production)

**Key Decisions**:
- FTS5 with porter tokenizer and `prefix='2 3'` for autocomplete
- shadcn-extension tree handles 5,000+ notes with <200ms render
- Version counter is simpler and faster than content hashing
- Hybrid JWT approach (memory + HttpOnly) is 2025 security best practice

### Phase 1: Data Model & Contracts ✅ COMPLETE

**Generated**:
- `data-model.md` - Complete entity definitions, Pydantic models, TypeScript types, SQLite schema, validation rules, state transitions
- `contracts/http-api.yaml` - OpenAPI 3.1 spec with 11 endpoints, auth, all error responses, examples
- `contracts/mcp-tools.json` - 7 MCP tool schemas with JSON Schema validation, usage examples
- `quickstart.md` - Complete setup, run, test, deploy guide for local PoC and HF Space

**Data Model Highlights**:
- 7 core entities (User, Vault, Note, Wikilink, Tag, Index, Token)
- Complete Pydantic models with validators
- TypeScript types for frontend
- SQLite DDL with 5 tables (metadata, FTS, tags, links, health)
- State machines for note lifecycle and index updates

**API Contract Highlights**:
- HTTP API: 11 endpoints (auth, CRUD, search, navigation, index)
- MCP Tools: 7 tools (list, read, write, delete, search, backlinks, tags)
- All error codes documented (400, 401, 403, 409, 413, 500)
- Request/response examples for all operations

**Agent Context Updated**:
- ✅ Updated `/home/wolfe/Projects/Document-MCP/CLAUDE.md` with Python 3.11+, FastAPI, FastMCP, SQLite tech stack

### Constitution Re-Check ✅ N/A

**Status**: No project constitution exists. Design decisions documented in research.md can form future constitution baseline.

---

## Next Steps

Run `/speckit.tasks` to generate dependency-ordered implementation tasks based on this plan, data model, and contracts.

**Planning Phase Complete**: 2025-11-15
