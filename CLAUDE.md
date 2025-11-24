# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Document-MCP** is a multi-tenant Obsidian-like documentation viewer with AI-first workflow. AI agents write/update documentation via MCP (Model Context Protocol), while humans read and edit through a web UI. The system provides per-user vaults with Markdown notes, full-text search (SQLite FTS5), wikilink resolution, tag indexing, and backlink tracking.

**Architecture**: Python backend (FastAPI + FastMCP) + React frontend (Vite + shadcn/ui)

**Key Concepts**:
- **Vault**: Per-user filesystem directory containing .md files
- **MCP Server**: Exposes tools for AI agents (STDIO for local, HTTP for remote with JWT)
- **Indexer**: SQLite FTS5 for full-text search + separate tables for tags/links/metadata
- **Wikilinks**: `[[Note Name]]` resolved via case-insensitive slug matching (prefers same folder, then lexicographic)
- **Optimistic Concurrency**: Version counter in SQLite (not frontmatter); UI sends `if_version`, MCP uses last-write-wins

## Development Commands

### Backend (Python 3.11+)

```bash
cd backend

# Setup (first time)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Install dev dependencies
uv pip install -e ".[dev]"

# Run FastAPI HTTP server (for UI)
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run MCP STDIO server (for Claude Desktop/Code)
uv run python src/mcp/server.py

# Run MCP HTTP server (for remote clients with JWT)
uv run python src/mcp/server.py --http --port 8001

# Tests
uv run pytest                          # All tests
uv run pytest tests/unit               # Unit tests only
uv run pytest tests/integration        # Integration tests
uv run pytest -k test_vault_write      # Single test pattern
uv run pytest -v                       # Verbose output
uv run pytest --lf                     # Last failed tests
```

### Frontend (Node 18+, React + Vite)

```bash
cd frontend

# Setup (first time)
npm install

# Development server
npm run dev                   # Start Vite dev server (http://localhost:5173)

# Build
npm run build                 # TypeScript compile + Vite build to dist/

# Lint
npm run lint                  # ESLint check

# Preview production build
npm run preview               # Serve dist/ (after npm run build)
```

### Database Initialization

```bash
# Backend database is auto-initialized on first run
# Manual reset (WARNING: destroys all data)
cd backend
rm -f ../data/index.db
uv run python -c "from src.services.database import DatabaseService; DatabaseService().init_schema()"
```

## Architecture Deep Dive

### Backend Service Layers

**3-tier architecture**:

1. **Models** (`backend/src/models/`): Pydantic schemas for validation
   - `note.py`: Note, NoteMetadata, NoteSummary
   - `user.py`: User, UserProfile
   - `search.py`: SearchResult, SearchQuery
   - `index.py`: IndexHealth
   - `auth.py`: TokenRequest, TokenResponse

2. **Services** (`backend/src/services/`): Business logic
   - `vault.py`: Filesystem operations (read/write/list/delete notes)
     - `validate_note_path()`: Path security (no `..`, max 256 chars, Unix separators)
     - `sanitize_path()`: Resolves and enforces vault root boundary
   - `indexer.py`: SQLite FTS5 + metadata tracking
     - `index_note()`: Updates metadata, FTS, tags, links (synchronous on every write)
     - `search_notes()`: BM25 ranking with title 3x weight, body 1x, recency bonus
     - `get_backlinks()`: Follows link graph (note → sources that reference it)
   - `auth.py`: JWT + HF OAuth integration
     - `create_access_token()`: Issues JWT with sub=user_id, exp=90days
     - `verify_token()`: Validates JWT and extracts user_id
   - `config.py`: Env var management (MODE, JWT_SECRET_KEY, VAULT_BASE_DIR, etc.)
   - `database.py`: SQLite connection manager + schema DDL

3. **API/MCP** (`backend/src/api/` and `backend/src/mcp/`):
   - `api/routes/`: FastAPI endpoints (11 routes: auth, notes CRUD, search, backlinks, tags, index health/rebuild)
   - `api/middleware/auth_middleware.py`: JWT Bearer token validation
   - `mcp/server.py`: FastMCP tools (7 tools: list, read, write, delete, search, backlinks, tags)

**Critical Path Validation** (in `vault.py`):
- All note paths MUST pass `validate_note_path()` (returns `(bool, str)` tuple)
- Then `sanitize_path()` resolves and ensures no vault escape
- Failure = 400 Bad Request with specific error message

### SQLite Index Schema

5 tables (see `backend/src/services/database.py`):

1. **note_metadata**: Version tracking, size, timestamps (per note)
2. **note_fts**: Contentless FTS5 with porter tokenizer, `prefix='2 3'` for autocomplete
3. **note_tags**: Many-to-many (user_id, note_path, tag)
4. **note_links**: Link graph (source_path → target_path, is_resolved flag)
5. **index_health**: Aggregate stats (note_count, last_full_rebuild, last_incremental_update)

**Indexer Update Flow** (in `indexer.py`):
```
write_note() → vault.write_note() → indexer.index_note()
                                  ↓
                            [metadata table: version++]
                            [FTS table: re-insert title+body]
                            [tags table: clear + re-insert]
                            [links table: extract wikilinks, resolve, update backlinks]
                            [health table: note_count++, last_incremental_update=now]
```

### Wikilink Resolution Algorithm

In `indexer.py` (`resolve_wikilink` logic):

1. Normalize link text to slug: `normalize_slug("API Design")` → `"api-design"`
2. Find all notes where slug matches `normalize_slug(title)` or `normalize_slug(filename_stem)`
3. If multiple matches:
   - Prefer same folder as source note
   - Else lexicographically smallest path (ASCII sort)
4. Store in `note_links` table with `is_resolved=1` (or `0` if no match)

**Broken links** are tracked (is_resolved=0) and can be queried for UI "Create note" affordance.

### MCP Server Modes

**STDIO** (`python src/mcp/server.py`):
- For Claude Desktop/Code local integration
- Uses `LOCAL_USER_ID` from env (default: "local-dev")
- No authentication

**HTTP** (`python src/mcp/server.py --http --port 8001`):
- For remote clients (HF Space deployment)
- Requires `Authorization: Bearer <jwt>` header
- JWT validated → user_id extracted → scoped to that user's vault

**Endpoint**: Tools defined in `mcp/server.py` with FastMCP decorators (`@mcp.tool`)

### Frontend Architecture

**Component Hierarchy**:
```
App.tsx (main layout)
├── DirectoryTree.tsx (left sidebar: vault explorer with virtualization)
├── NoteViewer.tsx (right pane: read mode, react-markdown rendering)
├── NoteEditor.tsx (right pane: edit mode, split view with live preview)
├── SearchBar.tsx (debounced search with dropdown results)
└── AuthFlow.tsx (HF OAuth login, token management)
```

**Key Libraries**:
- `react-markdown`: Markdown rendering with wikilink custom renderer
- `shadcn/ui`: UI components (Tree, ScrollArea, Button, Textarea, Dialog)
- `lib/wikilink.ts`: Parse `[[...]]` + resolve via GET /api/backlinks
- `services/api.ts`: Fetch wrapper with Bearer token injection

**Wikilink Rendering** (in `NoteViewer.tsx`):
- Custom `react-markdown` renderer for links
- Detect `[[Note Name]]` pattern → fetch backlinks → resolve to path → make clickable
- Broken links styled differently (e.g., red/dashed underline)

### Version Conflict Flow (Optimistic Concurrency)

**UI Edit Scenario**:
1. User opens note → GET /api/notes/{path} → receives `{..., version: 5}`
2. User edits → clicks Save → PUT /api/notes/{path} with `{"if_version": 5, ...}`
3. Backend checks: if current version != 5 → return 409 Conflict
4. UI shows "Note changed, please reload" message

**MCP Write**: No version check, always succeeds (last-write-wins).

## Environment Configuration

See `.env.example` for all variables. Key settings:

- **MODE**: `local` (single-user, no OAuth) or `space` (HF multi-tenant)
- **JWT_SECRET_KEY**: Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- **VAULT_BASE_DIR**: Where vaults are stored (e.g., `./data/vaults`)
- **DB_PATH**: SQLite database file (e.g., `./data/index.db`)
- **LOCAL_USER_ID**: Default user for local mode (default: `local-dev`)

**HF Space variables** (only needed when MODE=space):
- HF_OAUTH_CLIENT_ID, HF_OAUTH_CLIENT_SECRET, HF_SPACE_HOST

## Constraints & Limits

- **Note size**: 1 MiB max (enforced in vault.py)
- **Vault limit**: 5,000 notes per user (configurable in indexer.py)
- **Path length**: 256 chars max (validated in vault.py)
- **Wikilink syntax**: Only `[[wikilink]]` supported (no aliases like `[[link|alias]]`)

## Performance Targets

- MCP operations: <500ms for 1,000-note vaults
- UI directory load: <2s
- Note render: <1s
- Search: <1s for 5,000 notes
- Index rebuild: <30s for 1,000 notes

## SpecKit Workflow (in .specify/)

This repo uses the SpecKit methodology for feature planning:

- **specs/###-feature-name/**: Feature documentation
  - `spec.md`: User stories, requirements, success criteria
  - `plan.md`: Tech stack, architecture, structure
  - `data-model.md`: Entities, schemas, validation
  - `contracts/`: OpenAPI + MCP tool schemas
  - `tasks.md`: Implementation task checklist
- **Slash commands**: `/speckit.specify`, `/speckit.plan`, `/speckit.tasks`, `/speckit.implement`
- **Scripts**: `.specify/scripts/bash/` (feature scaffolding, context updates)

Current active feature: `001-obsidian-docs-viewer`

## MCP Client Configuration

**Claude Desktop** (STDIO, local mode):
```json
{
  "mcpServers": {
    "document-mcp": {
      "command": "uv",
      "args": ["run", "python", "src/mcp/server.py"],
      "cwd": "/absolute/path/to/Document-MCP/backend"
    }
  }
}
```

**Remote HTTP** (HF Space with JWT):
```json
{
  "mcpServers": {
    "document-mcp": {
      "url": "https://your-space.hf.space/mcp",
      "transport": "http",
      "headers": {
        "Authorization": "Bearer YOUR_JWT_TOKEN"
      }
    }
  }
}
```

Obtain JWT: `POST /api/tokens` after HF OAuth login.
