# Implementation Tasks: Multi-Tenant Obsidian-Like Docs Viewer

**Feature Branch**: `001-obsidian-docs-viewer`
**Created**: 2025-11-15
**Status**: Ready for Implementation

## Implementation Strategy

**MVP = User Story 1 (AI Agent Writes) + User Story 2 (Human Reads UI)**

The MVP delivers immediate value:
- AI agents (via MCP STDIO) can create and maintain documentation
- Humans can browse, search, and read documentation in the web UI
- Full-text search, wikilinks, tags, and backlinks work end-to-end

**Post-MVP enhancements**:
- User Story 3: Human editing with version conflict detection
- User Story 4: Multi-tenant HF OAuth for production deployment
- User Story 5: Advanced search ranking and index health monitoring

**Progress Update (2025-11-16)**:
- Phase 1 setup complete except T017 (create `data/vaults/`) and T021 (`.env.example`); T024 remains as an outstanding run-step (schema init script prepared but not executed).
- Phase 2 foundational backend/models/types complete.
- Phase 3 MCP backend + prompt tightening complete; see `backend/src/mcp/server.py` for the updated tool contracts.

---

## Phase 1: Setup

**Goal**: Initialize project structure, dependencies, and database schema.

- [x] [T001] Create project directory structure at /home/wolfe/Projects/Document-MCP
- [x] [T002] [P] Create backend/src/models/ directory and __init__.py
- [x] [T003] [P] Create backend/src/services/ directory and __init__.py
- [x] [T004] [P] Create backend/src/api/routes/ directory and __init__.py
- [x] [T005] [P] Create backend/src/api/middleware/ directory and __init__.py
- [x] [T006] [P] Create backend/src/mcp/ directory and __init__.py
- [x] [T007] [P] Create backend/tests/unit/ directory and __init__.py
- [x] [T008] [P] Create backend/tests/integration/ directory and __init__.py
- [x] [T009] [P] Create backend/tests/contract/ directory and __init__.py
- [x] [T010] [P] Create frontend/src/components/ui/ directory
- [x] [T011] [P] Create frontend/src/pages/ directory
- [x] [T012] [P] Create frontend/src/services/ directory
- [x] [T013] [P] Create frontend/src/lib/ directory
- [x] [T014] [P] Create frontend/src/types/ directory
- [x] [T015] [P] Create frontend/tests/unit/ directory
- [x] [T016] [P] Create frontend/tests/e2e/ directory
- [ ] [T017] [P] Create data/vaults/ directory for runtime vault storage
- [x] [T018] Create backend/pyproject.toml with dependencies: fastapi, fastmcp, python-frontmatter, pyjwt, huggingface_hub, uvicorn
- [x] [T019] Create frontend/package.json with dependencies: react, vite, typescript, shadcn/ui, react-markdown
- [x] [T020] Create frontend/vite.config.ts with proxy to backend API
- [ ] [T021] Create .env.example with JWT_SECRET_KEY, HF_OAUTH_CLIENT_ID, HF_OAUTH_CLIENT_SECRET, VAULT_BASE_PATH
- [x] [T022] Create .gitignore to exclude data/, .env, node_modules/, __pycache__, dist/
- [x] [T023] Create backend/src/services/database.py with SQLite initialization DDL from data-model.md
- [ ] [T024] Execute SQLite schema initialization (note_metadata, note_fts, note_tags, note_links, index_health tables)

---

## Phase 2: Foundational

**Goal**: Build core infrastructure required by all user stories.

- [x] [T025] Create backend/src/services/config.py to load env vars: JWT_SECRET_KEY, VAULT_BASE_PATH, HF_OAUTH_CLIENT_ID, HF_OAUTH_CLIENT_SECRET
- [x] [T026] [P] Create backend/src/models/user.py with User and HFProfile Pydantic models from data-model.md
- [x] [T027] [P] Create backend/src/models/note.py with Note, NoteMetadata, NoteCreate, NoteUpdate, NoteSummary Pydantic models from data-model.md
- [x] [T028] [P] Create backend/src/models/index.py with Wikilink, Tag, IndexHealth Pydantic models from data-model.md
- [x] [T029] [P] Create backend/src/models/search.py with SearchResult, SearchRequest Pydantic models from data-model.md
- [x] [T030] [P] Create backend/src/models/auth.py with TokenResponse, JWTPayload Pydantic models from data-model.md
- [x] [T031] [P] Create frontend/src/types/user.ts with User and HFProfile TypeScript types from data-model.md
- [x] [T032] [P] Create frontend/src/types/note.ts with Note, NoteMetadata, NoteSummary, NoteCreateRequest, NoteUpdateRequest TypeScript types from data-model.md
- [x] [T033] [P] Create frontend/src/types/search.ts with SearchResult, Tag, IndexHealth TypeScript types from data-model.md
- [x] [T034] [P] Create frontend/src/types/auth.ts with TokenResponse, APIError TypeScript types from data-model.md
- [x] [T035] Create backend/src/services/vault.py with VaultService class: path validation, sanitization (sanitize_path function from data-model.md), vault directory initialization
- [x] [T036] Create backend/src/services/auth.py with AuthService class: JWT creation (create_jwt), validation (validate_jwt), placeholder for HF OAuth
- [x] [T037] Create backend/src/api/middleware/auth_middleware.py with extract_user_id_from_jwt function to validate Authorization: Bearer header
- [x] [T038] Create backend/src/api/middleware/error_handlers.py with FastAPI exception handlers for 400, 401, 403, 404, 409, 413, 500 from http-api.yaml

---

## Phase 3: User Story 1 - AI Agent Writes (P1)

**Goal**: Enable AI agents to write/update docs via MCP STDIO, with automatic indexing.

- [x] [T039] [US1] Create backend/src/services/vault.py VaultService.read_note method: read file, parse frontmatter with python-frontmatter, extract title (priority: frontmatter > H1 > filename stem)
- [x] [T040] [US1] Create backend/src/services/vault.py VaultService.write_note method: validate path/content, create parent dirs, write frontmatter + body, return absolute path
- [x] [T041] [US1] Create backend/src/services/vault.py VaultService.delete_note method: validate path, remove file, handle FileNotFoundError
- [x] [T042] [US1] Create backend/src/services/vault.py VaultService.list_notes method: walk vault tree, filter by folder param, return paths and titles
- [x] [T043] [US1] Create backend/src/services/indexer.py IndexerService class with db connection management
- [x] [T044] [US1] Create backend/src/services/indexer.py IndexerService.index_note method: delete old rows for (user_id, note_path), insert into note_metadata, note_fts, note_tags, note_links
- [x] [T045] [US1] Create backend/src/services/indexer.py IndexerService.extract_wikilinks method: regex pattern \[\[([^\]]+)\]\] to extract link_text from body
- [x] [T046] [US1] Create backend/src/services/indexer.py IndexerService.resolve_wikilinks method: normalize slug (data-model.md algorithm), match against normalized_title_slug and normalized_path_slug, prefer same-folder, update is_resolved
- [x] [T047] [US1] Create backend/src/services/indexer.py IndexerService.increment_version method: get current version or default to 1, increment, return new version
- [x] [T048] [US1] Create backend/src/services/indexer.py IndexerService.update_index_health method: update note_count, last_incremental_update timestamp
- [x] [T049] [US1] Create backend/src/services/indexer.py IndexerService.delete_note_index method: delete rows from all index tables, update backlinks to set is_resolved=false
- [x] [T050] [US1] Create backend/src/mcp/server.py FastMCP server initialization with name="obsidian-docs-viewer"
- [x] [T051] [US1] Create backend/src/mcp/server.py list_notes MCP tool: call VaultService.list_notes, return [{path, title, last_modified}]
- [x] [T052] [US1] Create backend/src/mcp/server.py read_note MCP tool: call VaultService.read_note, return {path, title, metadata, body}
- [x] [T053] [US1] Create backend/src/mcp/server.py write_note MCP tool: call VaultService.write_note, then IndexerService.index_note, return {status: "ok", path}
- [x] [T054] [US1] Create backend/src/mcp/server.py delete_note MCP tool: call VaultService.delete_note, then IndexerService.delete_note_index, return {status: "ok"}
- [x] [T055] [US1] Create backend/src/mcp/server.py search_notes MCP tool: query note_fts with bm25 ranking (3.0 title weight, 1.0 body weight), add recency bonus, return [{path, title, snippet}]
- [x] [T056] [US1] Create backend/src/mcp/server.py get_backlinks MCP tool: query note_links WHERE target_path=?, join note_metadata, return [{path, title}]
- [x] [T057] [US1] Create backend/src/mcp/server.py get_tags MCP tool: query note_tags GROUP BY tag, return [{tag, count}]
- [x] [T058] [US1] Create backend/src/mcp/server.py STDIO transport mode: if __name__ == "__main__", run FastMCP with stdio transport for local development
- [x] [T059] [US1] Add recency bonus calculation to search_notes: +1.0 for updated in last 7 days, +0.5 for last 30 days, 0 otherwise

---

## Phase 4: User Story 2 - Human Reads UI (P1)

**Goal**: Web UI for browsing, searching, and reading notes with wikilinks and backlinks.

- [x] [T060] [US2] Create backend/src/api/routes/notes.py with GET /api/notes endpoint: call VaultService.list_notes, return NoteSummary[] from http-api.yaml
- [x] [T061] [US2] Create backend/src/api/routes/notes.py with GET /api/notes/{path} endpoint: URL-decode path, call VaultService.read_note, return Note from http-api.yaml
- [x] [T062] [US2] Create backend/src/api/routes/search.py with GET /api/search endpoint: call IndexerService search with query param, return SearchResult[] from http-api.yaml
- [x] [T063] [US2] Create backend/src/api/routes/search.py with GET /api/backlinks/{path} endpoint: URL-decode path, query note_links, return BacklinkResult[] from http-api.yaml
- [x] [T064] [US2] Create backend/src/api/routes/search.py with GET /api/tags endpoint: query note_tags, return Tag[] from http-api.yaml
- [x] [T065] [US2] Create backend/src/api/main.py FastAPI app with CORS middleware, mount routes, include error handlers
- [x] [T066] [US2] Create frontend/src/services/api.ts API client with fetch wrapper: add Authorization: Bearer header, handle JSON responses, throw APIError on non-200
- [x] [T067] [US2] Create frontend/src/services/api.ts listNotes function: GET /api/notes?folder=, return NoteSummary[]
- [x] [T068] [US2] Create frontend/src/services/api.ts getNote function: GET /api/notes/{encodeURIComponent(path)}, return Note
- [x] [T069] [US2] Create frontend/src/services/api.ts searchNotes function: GET /api/search?q=, return SearchResult[]
- [x] [T070] [US2] Create frontend/src/services/api.ts getBacklinks function: GET /api/backlinks/{encodeURIComponent(path)}, return BacklinkResult[]
- [x] [T071] [US2] Create frontend/src/services/api.ts getTags function: GET /api/tags, return Tag[]
- [x] [T072] [US2] Create frontend/src/lib/wikilink.ts with extractWikilinks function: regex /\[\[([^\]]+)\]\]/g
- [x] [T073] [US2] Create frontend/src/lib/wikilink.ts with normalizeSlug function: lowercase, replace spaces/underscores with dash, strip non-alphanumeric
- [x] [T074] [US2] Create frontend/src/lib/markdown.tsx with react-markdown config: code highlighting, wikilink custom renderer
- [x] [T075] [US2] Initialize shadcn/ui in frontend/: run npx shadcn@latest init, select default theme
- [x] [T076] [US2] Install shadcn/ui components: ScrollArea, Button, Input, Card, Badge, Resizable, Collapsible, Dialog, Alert, Textarea, Dropdown-Menu, Avatar, Command, Tooltip, Popover
- [x] [T077] [US2] Create frontend/src/components/DirectoryTree.tsx: recursive tree view with collapsible folders, leaf items for notes, onClick handler to load note
- [x] [T078] [US2] Create frontend/src/components/NoteViewer.tsx: render note title, metadata (tags as badges, timestamps), react-markdown body with wikilink links, backlinks section in footer
- [x] [T079] [US2] Create frontend/src/components/SearchBar.tsx: Input with debounced onChange (300ms), dropdown results with onClick to navigate to note
- [x] [T080] [US2] Create frontend/src/pages/MainApp.tsx: two-pane layout (left: DirectoryTree + SearchBar in ScrollArea, right: NoteViewer), state management for selected note path
- [x] [T081] [US2] Add wikilink click handler in NoteViewer: onClick [[link]] → normalizeSlug → API lookup → navigate to resolved note
- [x] [T082] [US2] Add broken wikilink styling in NoteViewer: render unresolved [[links]] with distinct color/style
- [x] [T083] [US2] Create frontend/src/pages/MainApp.tsx useEffect to load directory tree on mount: call listNotes()
- [x] [T084] [US2] Create frontend/src/pages/MainApp.tsx useEffect to load note when path changes: call getNote(path) and getBacklinks(path)

---

## Phase 5: User Story 3 - Human Edits UI (P2)

**Goal**: Split-pane editor with optimistic concurrency protection.

- [x] [T085] [US3] Create backend/src/api/routes/notes.py with PUT /api/notes/{path} endpoint: URL-decode path, validate request body (NoteUpdate), check if_version, call VaultService.write_note, return NoteResponse from http-api.yaml
- [x] [T086] [US3] Add optimistic concurrency check in IndexerService.increment_version: if if_version provided and != current version, raise ConflictError
- [x] [T087] [US3] Create ConflictError exception in backend/src/models/errors.py, map to 409 Conflict in error_handlers.py
- [x] [T088] [US3] Create frontend/src/services/api.ts updateNote function: PUT /api/notes/{encodeURIComponent(path)} with {title?, metadata?, body, if_version?}, handle 409 response
- [x] [T089] [US3] Create frontend/src/components/NoteEditor.tsx: split-pane layout (left: textarea for markdown source, right: live preview with react-markdown)
- [x] [T090] [US3] Create frontend/src/components/NoteEditor.tsx with Save button: onClick → call updateNote with if_version from initial note load, handle success → switch to read mode
- [x] [T091] [US3] Create frontend/src/components/NoteEditor.tsx with Cancel button: onClick → discard changes, switch to read mode
- [x] [T092] [US3] Add 409 Conflict error handling in NoteEditor: display alert "Note changed since you opened it, please reload before saving"
- [x] [T093] [US3] Add Edit button to NoteViewer: onClick → switch main pane to NoteEditor mode, pass current note version
- [x] [T094] [US3] Update frontend/src/pages/MainApp.tsx to toggle between NoteViewer and NoteEditor based on edit mode state

---

## Phase 6: User Story 4 - Multi-Tenant OAuth (P2)

**Goal**: HF OAuth login, per-user vaults, JWT tokens for API and MCP HTTP.

- [ ] [T095] [US4] Create backend/src/services/auth.py HF OAuth integration: use huggingface_hub.attach_huggingface_oauth and parse_huggingface_oauth helpers
- [ ] [T096] [US4] Create backend/src/api/routes/auth.py with GET /auth/login endpoint: redirect to HF OAuth authorize URL
- [ ] [T097] [US4] Create backend/src/api/routes/auth.py with GET /auth/callback endpoint: parse_huggingface_oauth, map HF username to user_id, create vault if new user, set session cookie
- [ ] [T098] [US4] Create backend/src/api/routes/auth.py with POST /api/tokens endpoint: validate authenticated user, call AuthService.create_jwt, return TokenResponse from http-api.yaml
- [ ] [T099] [US4] Create backend/src/api/routes/auth.py with GET /api/me endpoint: validate Bearer token, return User from http-api.yaml
- [ ] [T100] [US4] Update backend/src/api/middleware/auth_middleware.py to extract user_id from JWT sub claim, attach to request.state.user_id
- [ ] [T101] [US4] Update backend/src/services/vault.py to scope all operations by user_id: vault path = VAULT_BASE_PATH / user_id
- [ ] [T102] [US4] Update backend/src/services/indexer.py to scope all queries by user_id: WHERE user_id = ?
- [ ] [T103] [US4] Initialize vault and index on first user login: create vault dir, insert initial index_health row
- [ ] [T104] [US4] Create backend/src/mcp/server.py HTTP transport mode: FastMCP with http transport, BearerAuth validation, extract user_id from JWT
- [x] [T105] [US4] Create frontend/src/services/auth.ts with login function: redirect to /auth/login
- [x] [T106] [US4] Create frontend/src/services/auth.ts with getCurrentUser function: GET /api/me, return User
- [x] [T107] [US4] Create frontend/src/services/auth.ts with getToken function: POST /api/tokens, return TokenResponse, store token in memory
- [x] [T108] [US4] Create frontend/src/pages/Login.tsx: "Sign in with Hugging Face" button → onClick call auth.login()
- [x] [T109] [US4] Create frontend/src/pages/Settings.tsx: display user profile (user_id, HF avatar), API token with copy button for MCP config
- [x] [T110] [US4] Update frontend/src/pages/App.tsx to call getCurrentUser on mount, redirect to Login if 401
- [x] [T111] [US4] Update frontend/src/services/api.ts to include token from localStorage in Authorization header

---

## Phase 7: User Story 5 - Advanced Search (P3)

**Goal**: Enhanced search ranking and index health monitoring.

- [x] [T112] [US5] Update backend/src/services/indexer.py search_notes to calculate recency bonus: +1.0 for updated in last 7 days, +0.5 for last 30 days, 0 otherwise
- [x] [T113] [US5] Update backend/src/services/indexer.py search_notes to calculate final score: (3 * title_bm25) + (1 * body_bm25) + recency_bonus
- [x] [T114] [US5] Create backend/src/api/routes/index.py with GET /api/index/health endpoint: query index_health, return IndexHealth from http-api.yaml
- [x] [T115] [US5] Create backend/src/api/routes/index.py with POST /api/index/rebuild endpoint: call IndexerService.rebuild_index, return RebuildResponse from http-api.yaml
- [x] [T116] [US5] Create backend/src/services/indexer.py IndexerService.rebuild_index method: delete all user rows, walk vault, parse all notes, re-insert into all index tables, update index_health
- [x] [T117] [US5] Create frontend/src/services/api.ts getIndexHealth function: GET /api/index/health, return IndexHealth
- [x] [T118] [US5] Create frontend/src/services/api.ts rebuildIndex function: POST /api/index/rebuild, return RebuildResponse
- [x] [T119] [US5] Add index health indicator to frontend/src/pages/MainApp.tsx: display note count and last updated timestamp in footer
- [x] [T120] [US5] Add "Rebuild Index" button to frontend/src/pages/Settings.tsx: onClick → call rebuildIndex, show progress/completion message

---

## Phase 8: Polish & Cross-Cutting

**Goal**: Documentation, configuration, logging, error handling improvements.

- [x] [T121] Create README.md with project overview, tech stack, local setup instructions (backend venv + npm install)
- [x] [T122] Add README.md section: "Running Backend" with uvicorn command for HTTP API and python -m backend.src.mcp.server for MCP STDIO
- [x] [T123] Add README.md section: "Running Frontend" with npm run dev command
- [x] [T124] Add README.md section: "MCP Client Configuration" with Claude Code/Desktop STDIO example from mcp-tools.json
- [ ] [T125] Add README.md section: "Deploying to Hugging Face Space" with environment variables and OAuth setup
- [x] [T126] Update .env.example with all variables: JWT_SECRET_KEY, VAULT_BASE_PATH, HF_OAUTH_CLIENT_ID, HF_OAUTH_CLIENT_SECRET, DATABASE_PATH
- [x] [T127] Add structured logging to backend/src/services/vault.py: log file operations with user_id, note_path, operation type
- [x] [T128] Add structured logging to backend/src/services/indexer.py: log index updates with user_id, note_path, duration_ms
- [x] [T129] Add structured logging to backend/src/mcp/server.py: log MCP tool calls with tool_name, user_id, duration_ms
- [x] [T130] Improve error messages in backend/src/api/middleware/error_handlers.py: include detail objects with field names and reasons
- [x] [T131] Add input validation to all HTTP API routes: validate path format, content size, required fields
- [x] [T132] Add input validation to all MCP tools: validate path format, content size via Pydantic models
- [x] [T133] Add rate limiting consideration to README.md: note potential need for per-user rate limits in production
- [x] [T134] Add performance optimization notes to README.md: FTS5 prefix indexes, SQLite WAL mode for concurrency

---

## Dependencies

### Story Completion Order

**Must complete in this order**:
1. **Phase 1** (Setup) → **Phase 2** (Foundational) → **Phase 3** (US1) + **Phase 4** (US2) in parallel
2. **Phase 5** (US3) depends on Phase 4 (needs HTTP API routes from US2)
3. **Phase 6** (US4) depends on Phase 2 (needs auth foundation) and Phase 3 (needs vault/indexer)
4. **Phase 7** (US5) depends on Phase 3 (needs indexer) and Phase 4 (needs HTTP API)
5. **Phase 8** (Polish) can run anytime after Phase 4 (MVP complete)

### Task Dependencies

**Critical path** (must be sequential):
- T023 (SQLite schema) → T043 (IndexerService) → T044 (index_note)
- T035 (VaultService foundation) → T039 (read_note) → T040 (write_note)
- T050 (FastMCP init) → T051-T057 (MCP tools)
- T065 (FastAPI app) → T060-T064 (HTTP routes)

**Parallelizable within phases** (marked with [P]):
- All directory creation tasks (T002-T017)
- All Pydantic model tasks (T026-T030)
- All TypeScript type tasks (T031-T034)
- All frontend component tasks within US2 (T077-T079)

---

## Parallel Execution Examples

### User Story 1 (AI Agent Writes) - Parallel Work

**Team A**: VaultService implementation (T039-T042)
**Team B**: IndexerService implementation (T043-T049)
**Team C**: MCP tools implementation (T051-T057)

After T050 (FastMCP init) completes, Team C can implement all 7 MCP tools in parallel since they're independent endpoints.

### User Story 2 (Human Reads UI) - Parallel Work

**Team A**: Backend HTTP API routes (T060-T064)
**Team B**: Frontend API client (T066-T071)
**Team C**: Frontend components (T077-T079)

After T065 (FastAPI app) and T075-T076 (shadcn/ui setup) complete, all three teams can work in parallel.

### User Story 4 (Multi-Tenant OAuth) - Parallel Work

**Team A**: Backend OAuth integration (T095-T104)
**Team B**: Frontend auth flow (T105-T111)

Both teams can work in parallel after Phase 2 (Foundational) completes.

---

## Summary

**Total Tasks**: 134
- **Phase 1 (Setup)**: 24 tasks
- **Phase 2 (Foundational)**: 14 tasks
- **Phase 3 (US1 - AI Agent Writes)**: 21 tasks
- **Phase 4 (US2 - Human Reads UI)**: 25 tasks
- **Phase 5 (US3 - Human Edits UI)**: 10 tasks
- **Phase 6 (US4 - Multi-Tenant OAuth)**: 17 tasks
- **Phase 7 (US5 - Advanced Search)**: 9 tasks
- **Phase 8 (Polish)**: 14 tasks

**MVP Tasks** (US1 + US2): 84 tasks (Phases 1-4)
**Post-MVP Tasks** (US3 + US4 + US5): 36 tasks (Phases 5-7)
**Polish Tasks**: 14 tasks (Phase 8)

**Estimated Effort**:
- MVP (US1 + US2): ~2-3 weeks (1-2 developers)
- Post-MVP (US3 + US4 + US5): ~1-2 weeks
- Polish: ~3-5 days
- **Total**: ~4-6 weeks for complete implementation

**Key Milestones**:
1. **Week 1**: Complete Phase 1-2 (Setup + Foundational)
2. **Week 2**: Complete Phase 3 (US1 - MCP STDIO working)
3. **Week 3**: Complete Phase 4 (US2 - Web UI working, MVP delivered)
4. **Week 4**: Complete Phase 5-6 (US3 editing + US4 multi-tenant)
5. **Week 5**: Complete Phase 7-8 (US5 advanced search + Polish)

**Next Steps**:
1. Review this task breakdown with stakeholders
2. Assign initial tasks (Phase 1 Setup) to team
3. Create GitHub issues from tasks using `/speckit.taskstoissues`
4. Begin implementation with T001 (project structure)
