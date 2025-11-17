# Feature Specification: Multi-Tenant Obsidian-Like Docs Viewer

**Feature Branch**: `001-obsidian-docs-viewer`
**Created**: 2025-11-15
**Status**: Draft
**Input**: User description: "Build a multi-tenant Obsidian-like docs viewer with FastMCP server (Python) exposing tools over MCP, HTTP API with Bearer auth for UI and MCP, multi-tenant vaults (per-user Markdown directories), indexing (full-text search, backlinks, tags), and a React + shadcn/ui frontend hosted in a Hugging Face Space with Obsidian-style UI (left: directory pane with vault explorer + search, right: main note pane with live-rendered Markdown and light editing). Primary workflow: AI (via MCP) writes/updates docs, humans read + occasionally tweak in the UI."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - AI Agent Writes and Updates Documentation (Priority: P1)

An AI agent (Claude via MCP) needs to create and maintain structured documentation within a user's vault. The agent discovers existing notes, creates new notes with proper frontmatter and wikilinks, updates existing content, and automatically maintains the index for searchability.

**Why this priority**: This is the primary workflow. Without AI write capability via MCP, the core value proposition doesn't exist. This enables the "AI writes, humans read" paradigm.

**Independent Test**: Can be fully tested by configuring an MCP client (Claude Code/Desktop) with STDIO transport, issuing `write_note` commands, and verifying files are created with correct frontmatter, body content, and automatic index updates. Delivers immediate value for AI-driven documentation workflows without requiring the UI.

**Acceptance Scenarios**:

1. **Given** an MCP client connected via STDIO transport, **When** the agent calls `write_note` with path "api/design.md", title "API Design", and markdown body, **Then** the file is created with YAML frontmatter containing title, created/updated timestamps, and the body content
2. **Given** a note already exists at "api/design.md" with version 3, **When** the agent calls `write_note` with updated content, **Then** the note is updated, version increments to 4, updated timestamp is set to now, and the full-text index is updated
3. **Given** a note containing wikilinks `[[Authentication Flow]]` and `[[Database Schema]]`, **When** the note is written, **Then** the link graph index records outgoing links and updates backlinks for target notes
4. **Given** a note with frontmatter tags `[backend, api]`, **When** the note is written, **Then** the tag index is updated to include this note under both tags
5. **Given** an MCP client requests `search_notes` with query "authentication", **When** notes containing "authentication" in title or body exist, **Then** results are returned ranked by title matches first, then body matches, with recency bonus

---

### User Story 2 - Human Reads Documentation in Web UI (Priority: P1)

A human user logs into the web UI, browses their vault's directory tree, searches for notes, and reads rendered Markdown with working wikilinks and backlinks visible.

**Why this priority**: The read-first UI is the primary human interaction mode. Without this, humans can't consume the AI-generated documentation effectively. This is equally critical to P1 as the write capability.

**Independent Test**: Can be fully tested by opening the web UI, authenticating with a static token (local mode), clicking through directory tree items, and verifying rendered Markdown displays correctly with clickable wikilinks. Delivers value for documentation consumption without requiring MCP or editing features.

**Acceptance Scenarios**:

1. **Given** a user is on the login page in local mode, **When** they access the UI with a valid static Bearer token in local storage, **Then** they see the directory pane with all vault notes organized by folder structure
2. **Given** the directory pane shows nested folders, **When** the user clicks on a note "api/design.md", **Then** the main pane displays the rendered Markdown with title "API Design", body content with proper formatting, and metadata footer showing tags and timestamps
3. **Given** a rendered note contains a wikilink `[[Authentication Flow]]`, **When** the user clicks the link, **Then** the UI navigates to and renders "authentication-flow.md" (resolved via normalized slug matching)
4. **Given** a note "auth.md" is referenced by 3 other notes, **When** the note is displayed, **Then** the backlinks section in the footer shows all 3 referring notes as clickable links
5. **Given** the user types "database" into the search bar, **When** the search debounces and executes, **Then** matching notes appear in a dropdown with snippets, ranked by title matches (3x weight) then body matches, with recency bonus

---

### User Story 3 - Human Edits Documentation in Web UI (Priority: P2)

A human user needs to make minor corrections or additions to AI-generated documentation. They click "Edit" on a note, see a split view with Markdown editor on the left and live preview on the right, make changes, and save with optimistic concurrency protection.

**Why this priority**: Enables human refinement of AI content. Important for quality but secondary to read/write workflows. Users can still achieve primary value (AI writes, humans read) without editing.

**Independent Test**: Can be tested by opening a note, clicking "Edit", modifying content in the textarea, clicking "Save", and verifying the file is updated with version conflict detection. Delivers value for collaborative human-AI documentation without requiring full MCP or advanced features.

**Acceptance Scenarios**:

1. **Given** a user is viewing a rendered note with version 5, **When** they click the "Edit" button, **Then** the main pane switches to split view: left side shows markdown source in a textarea, right side shows live-rendered preview
2. **Given** the user is in edit mode, **When** they modify the markdown body and click "Save", **Then** a PUT request is sent with `if_version: 5`, the note is updated to version 6, updated timestamp is set to now, and the UI switches back to read mode
3. **Given** the user opened a note with version 5 and another user/agent updated it to version 6, **When** the first user clicks "Save", **Then** the server returns 409 Conflict and the UI displays "This note changed since you opened it; please reload before saving"
4. **Given** the user edits the note title in frontmatter from "API Design" to "API Architecture", **When** they save, **Then** the file is updated with new title, directory tree reflects the change, and the title-based link resolution index is updated
5. **Given** the user adds a new wikilink `[[New Feature]]` that doesn't exist, **When** they save and view the rendered note, **Then** the wikilink is rendered as a "broken link" style with a "Create note" affordance

---

### User Story 4 - Multi-Tenant Access via Hugging Face OAuth (Priority: P2)

Multiple users can sign in to the Hugging Face Space using "Sign in with HF", each getting isolated vaults, personalized API tokens for MCP access, and per-user indices.

**Why this priority**: Enables the production deployment model. Critical for multi-user scenarios but not needed for local PoC or single-user hackathon demos.

**Independent Test**: Can be tested by deploying to HF Space with OAuth enabled, signing in with two different HF accounts, creating notes in each vault, and verifying complete data isolation. Delivers value for hosted multi-tenant scenarios without requiring advanced features.

**Acceptance Scenarios**:

1. **Given** a user visits the HF Space and is not authenticated, **When** they land on the app, **Then** they see a "Sign in with Hugging Face" button
2. **Given** a user clicks "Sign in with Hugging Face", **When** HF OAuth flow completes successfully, **Then** the backend maps their HF username to an internal user_id, creates a vault directory at `/data/vaults/<user_id>/`, initializes an empty index, and redirects to the main app UI
3. **Given** a user is authenticated via HF OAuth, **When** they call `POST /api/tokens`, **Then** the server issues a JWT with `sub=user_id` and `exp=now+90days`, returning `{"token": "<jwt>"}`
4. **Given** an MCP client configures the HTTP transport with `Authorization: Bearer <jwt>`, **When** the client calls `list_notes`, **Then** the server validates the JWT, extracts user_id, and returns notes only from that user's vault
5. **Given** two users (Alice and Bob) each create notes in their vaults, **When** Alice searches for notes, **Then** she sees only her own notes, never Bob's (complete data isolation)

---

### User Story 5 - Full-Text Search with Index Health Monitoring (Priority: P3)

Users and AI agents can search across all notes using full-text queries, with results ranked by relevance (title matches weighted higher, recency bonus). Users can manually trigger index rebuilds if needed and view index health status.

**Why this priority**: Enhances discoverability but is a supporting feature. The basic search in P1/P2 stories is sufficient for MVP. Index rebuild is primarily a maintenance/troubleshooting tool.

**Independent Test**: Can be tested by creating notes with specific keywords, calling `search_notes` MCP tool or `/api/search` endpoint, and verifying ranking (title 3x weight, body 1x, recency bonus). Can verify rebuild by calling `POST /api/index/rebuild` and checking updated counts. Delivers value for large vaults without requiring other features.

**Acceptance Scenarios**:

1. **Given** a vault with 50 notes, 10 containing "authentication" in body, and 2 containing "authentication" in title, **When** a search query "authentication" is executed, **Then** the 2 title-match notes are ranked first, followed by body-match notes, with notes updated in last 7 days receiving a +1.0 recency bonus
2. **Given** a note contains tokens in both title and body matching the query, **When** the search is executed, **Then** the score is `(3 * title_hits) + (1 * body_hits) + recency_bonus` and results are sorted by descending score
3. **Given** a user calls `GET /api/index/health`, **When** the index exists, **Then** the response includes `note_count`, `last_full_rebuild` timestamp, and `last_incremental_update` timestamp
4. **Given** a user has made many manual file changes outside the app, **When** they call `POST /api/index/rebuild`, **Then** the server drops existing index rows for their user_id, re-scans all .md files, rebuilds full-text index, tag index, and link graph, and updates `last_full_rebuild` timestamp
5. **Given** the index shows `note_count: 100` and `last_incremental_update` is 2 minutes ago, **When** a new note is written via MCP, **Then** `note_count` increments to 101, `last_incremental_update` is set to now, and the new note is immediately searchable

---

### Edge Cases

- **Wikilink ambiguity**: When `[[Note Name]]` matches multiple files (e.g., `docs/setup.md` and `guides/setup.md`), the system resolves deterministically by preferring same-folder match first, then lexicographically smallest path. Ambiguous links may be flagged in backlinks view.
- **Concurrent edits**: When Claude (via MCP, last-write-wins) and a human (via UI, optimistic concurrency) edit the same note simultaneously, the human's save will fail with 409 Conflict if the version changed, preventing silent data loss from the human perspective.
- **Broken wikilinks**: When a note contains `[[Non Existent Note]]`, it is rendered as a visually distinct "broken link" style. The index tracks unresolved links. UI offers "Create note" affordance on click.
- **Large note uploads**: When a note exceeds 1 MiB UTF-8 text, the server returns `413 Payload Too Large` with a clear error message.
- **Vault limit exceeded**: When a user attempts to create a note that would exceed 5,000 notes in their vault, the server returns `403 Forbidden` with error code "vault_note_limit_exceeded".
- **Malformed frontmatter**: When a note has invalid YAML frontmatter, the system treats it as a note without frontmatter, using the first `# Heading` as title or filename stem as fallback.
- **Path traversal attempts**: When a path contains `..` or absolute path components, the vault module normalizes and validates against the user's vault root, rejecting any escape attempts with `400 Bad Request`.
- **Token expiration**: When a JWT expires (after 90 days), API/MCP requests return `401 Unauthorized`. User must re-authenticate and issue a new token via `POST /api/tokens`.
- **Case-insensitive wikilink resolution**: When `[[api design]]` and `[[API Design]]` both exist as notes, resolution uses case-insensitive normalized slug matching, with exact case matches preferred, then any case variation.
- **Empty search query**: When search is called with an empty or whitespace-only query, the API returns an empty result set without error.

## Requirements *(mandatory)*

### Functional Requirements

#### Core Vault Operations

- **FR-001**: System MUST provide isolated vault directories per user under a configurable base path (e.g., `/data/vaults/<user_id>/`)
- **FR-002**: System MUST support arbitrary nested folder structures within each vault, containing Markdown (.md) files
- **FR-003**: System MUST enforce path normalization and validation to prevent directory traversal attacks (no `..` escapes, all paths relative to vault root)
- **FR-004**: System MUST parse Markdown files with optional YAML frontmatter containing metadata fields (title, tags, created, updated, project, etc.)
- **FR-005**: System MUST use `python-frontmatter` library (or equivalent) to load and serialize frontmatter + body
- **FR-006**: System MUST auto-manage `created` timestamp (set once on creation if not provided) and `updated` timestamp (always set to now on writes)
- **FR-007**: System MUST reject notes exceeding 1 MiB UTF-8 text with `413 Payload Too Large`
- **FR-008**: System MUST reject vault operations that would exceed 5,000 notes per user with `403 Forbidden` and error code "vault_note_limit_exceeded"
- **FR-009**: System MUST limit relative path strings to 256 characters maximum

#### Indexing and Search

- **FR-010**: System MUST maintain per-user indices for: (a) full-text search (token → note paths), (b) tag index (tag → note paths), (c) link graph (note → outgoing wikilinks, note → backlinks)
- **FR-011**: System MUST store indices in SQLite database with per-user isolation
- **FR-012**: System MUST support full-text search with simple tokenization (split on non-alphanumeric, case-insensitive)
- **FR-013**: System MUST rank search results using scoring formula: `(3 * title_hits) + (1 * body_hits) + recency_bonus`, where recency_bonus is 1.0 for updates in last 7 days, 0.5 for last 30 days, 0 otherwise
- **FR-014**: System MUST extract wikilinks from note bodies using regex pattern `\[\[([^\]]+)\]\]`
- **FR-015**: System MUST resolve wikilinks via case-insensitive normalized slug matching: normalize(link_text) matches normalize(filename_stem) or normalize(frontmatter_title)
- **FR-016**: System MUST handle ambiguous wikilinks deterministically: prefer same-folder match, then lexicographically smallest full path
- **FR-017**: System MUST track unresolved wikilinks (links with no matching note) in the index for UI display
- **FR-018**: System MUST update indices incrementally on every write/delete operation (synchronous, blocking)
- **FR-019**: System MUST provide manual full index rebuild capability that re-scans all vault files and reconstructs indices from scratch

#### Versioning and Concurrency

- **FR-020**: System MUST maintain a version counter (integer) per note, stored in the index (not in frontmatter)
- **FR-021**: System MUST increment version by 1 on every successful write operation
- **FR-022**: System MUST support optimistic concurrency for HTTP API writes: if `if_version` parameter is provided and does not match current version, return `409 Conflict`
- **FR-023**: System MUST implement last-write-wins for MCP tool writes (no version checking)

#### Authentication and Authorization

- **FR-024**: System MUST support two authentication modes: (a) Local mode with static user_id "local-dev" and optional static Bearer token, (b) HF Space mode with OAuth-based per-user identity
- **FR-025**: System MUST use JWT tokens for API and MCP HTTP authentication, containing claims: `sub=user_id`, `exp=now+90days`, signed with configurable secret
- **FR-026**: System MUST validate Bearer tokens via `Authorization: Bearer <token>` header on all protected endpoints
- **FR-027**: System MUST extract user_id from validated JWT and scope all vault/index operations to that user
- **FR-028**: System MUST integrate with Hugging Face OAuth in Space mode, using `huggingface_hub.attach_huggingface_oauth` and `parse_huggingface_oauth` helpers
- **FR-029**: System MUST map HF OAuth identity (username or ID) to internal user_id
- **FR-030**: System MUST create vault directory and initialize empty index on first login for new HF users

#### HTTP API

- **FR-031**: System MUST expose HTTP API using FastAPI (or equivalent) with JSON request/response format
- **FR-032**: System MUST provide endpoint `GET /api/me` returning user info (`user_id`, HF profile if applicable, authentication status)
- **FR-033**: System MUST provide endpoint `POST /api/tokens` to issue new JWT tokens for authenticated users
- **FR-034**: System MUST provide endpoint `GET /api/notes` to list notes with optional folder filtering, returning array of `{path, title, last_modified}`
- **FR-035**: System MUST provide endpoint `GET /api/notes/{path}` (where path is URL-encoded, includes `.md`) returning full note: `{path, title, metadata, body, version, created, updated}`
- **FR-036**: System MUST provide endpoint `PUT /api/notes/{path}` accepting `{title, metadata, body, if_version?}` to create/update notes
- **FR-037**: System MUST provide endpoint `DELETE /api/notes/{path}` to delete notes
- **FR-038**: System MUST provide endpoint `GET /api/search?q=<query>` returning ranked search results with snippets
- **FR-039**: System MUST provide endpoint `GET /api/backlinks/{path}` returning array of notes that reference the target note
- **FR-040**: System MUST provide endpoint `GET /api/tags` returning list of `{tag, count}` across all user notes
- **FR-041**: System MUST provide endpoint `GET /api/index/health` returning `{note_count, last_full_rebuild, last_incremental_update}`
- **FR-042**: System MUST provide endpoint `POST /api/index/rebuild` to trigger manual full index rebuild

#### MCP Server (FastMCP)

- **FR-043**: System MUST expose MCP server using FastMCP library with two transport modes: (a) STDIO for local development, (b) HTTP for remote/HF Space access
- **FR-044**: System MUST configure MCP HTTP transport to require `Authorization: Bearer <token>` header and validate JWT
- **FR-045**: System MUST provide MCP tool `list_notes` with input `{folder?: string}` returning `[{path, title, last_modified}]`
- **FR-046**: System MUST provide MCP tool `read_note` with input `{path: string}` returning `{path, title, metadata, body}`
- **FR-047**: System MUST provide MCP tool `write_note` with input `{path: string, title?: string, metadata?: object, body: string}` returning `{status: "ok", path}`
- **FR-048**: System MUST provide MCP tool `delete_note` with input `{path: string}` returning `{status: "ok"}`
- **FR-049**: System MUST provide MCP tool `search_notes` with input `{query: string}` returning `[{path, title, snippet}]`
- **FR-050**: System MUST provide MCP tool `get_backlinks` with input `{path: string}` returning `[{path, title}]`
- **FR-051**: System MUST provide MCP tool `get_tags` with input `{}` returning `[{tag, count}]`
- **FR-052**: System MUST define all MCP tool inputs/outputs with JSON Schema using FastMCP/Pydantic models

#### Frontend (React + shadcn/ui)

- **FR-053**: System MUST provide a single-page React application built with Vite or Next.js, using shadcn/ui components
- **FR-054**: System MUST implement Obsidian-style layout: left sidebar (directory pane + search), right main pane (note viewer/editor)
- **FR-055**: System MUST display directory tree in left sidebar with collapsible folders and note leaf items, using shadcn `ScrollArea`
- **FR-056**: System MUST provide search input in left sidebar with debounced queries calling `GET /api/search`, displaying results in dropdown
- **FR-057**: System MUST render selected note in main pane as read-only Markdown by default using `react-markdown` with plugins for code blocks and links
- **FR-058**: System MUST display note metadata in footer: tags (chips), created/updated timestamps, backlinks (clickable)
- **FR-059**: System MUST provide "Edit" button to switch main pane to edit mode: left side textarea with markdown source, right side live preview
- **FR-060**: System MUST provide "Save" button in edit mode that calls `PUT /api/notes/{path}` with `if_version`, handling 409 Conflict by showing "Note changed, please reload" message
- **FR-061**: System MUST render wikilinks as clickable links, resolving to target notes on click
- **FR-062**: System MUST render unresolved wikilinks as distinct "broken link" style with "Create note" affordance
- **FR-063**: System MUST provide "New note" button in left sidebar to create new notes with auto-generated frontmatter template
- **FR-064**: System MUST implement authentication flow for HF Space mode: landing page with "Sign in with Hugging Face" button, redirecting to main app after OAuth callback
- **FR-065**: System MUST call `GET /api/me` on startup to detect authentication status and `POST /api/tokens` to obtain Bearer token for API calls
- **FR-066**: System MUST display user profile/settings view showing user_id and API token(s) with "copy" button for MCP configuration
- **FR-067**: System MUST store Bearer token in memory or localStorage and include in `Authorization` header for all API requests
- **FR-068**: System MUST display small index health indicator showing note count and last updated timestamp

### Key Entities

- **User**: Represents an authenticated user (local-dev or HF OAuth identity). Has a unique `user_id`, optional HF profile data (username, avatar), and vault directory path.

- **Vault**: A user-specific directory tree containing Markdown notes. Has a root path (`/data/vaults/<user_id>/`), arbitrary nested folders, and .md files.

- **Note**: A Markdown file with optional YAML frontmatter and body content. Key attributes: `path` (relative to vault root, includes .md), `title` (from frontmatter or first H1 or filename stem), `metadata` (frontmatter key-value pairs), `body` (markdown content), `version` (integer counter), `created` (ISO timestamp), `updated` (ISO timestamp).

- **Wikilink**: A reference from one note to another using `[[link text]]` syntax. Has `source_note_path`, `link_text`, `target_note_path` (resolved via normalized slug matching, may be null if unresolved).

- **Tag**: A metadata label applied to notes via frontmatter `tags: [tag1, tag2]`. Has `tag_name` and count of associated notes.

- **Index**: Per-user data structures for efficient search and navigation. Contains: full-text inverted index (token → note paths), tag index (tag → note paths), link graph (note → outgoing wikilinks, note → backlinks).

- **Token (JWT)**: A signed JSON Web Token used for API and MCP authentication. Contains claims: `sub` (user_id), `exp` (expiration timestamp), signed with server secret.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: AI agents can create, read, update, and delete notes via MCP STDIO transport in under 500ms per operation for vaults with up to 1,000 notes
- **SC-002**: Human users can browse directory tree, select a note, and view rendered Markdown in under 2 seconds from click to full render
- **SC-003**: Search queries return ranked results in under 1 second for vaults with up to 5,000 notes
- **SC-004**: Wikilink navigation (click to target note) completes in under 1 second, with ambiguous links resolved deterministically without user intervention
- **SC-005**: Concurrent edit conflicts (human vs AI) are detected and prevented 100% of the time via optimistic concurrency for UI writes
- **SC-006**: Index health accurately reflects vault state: `note_count` matches actual file count within 1 second of any write operation (incremental update)
- **SC-007**: Multi-tenant isolation is complete: users never see or access other users' vaults or notes in API, MCP, or UI
- **SC-008**: OAuth authentication flow (HF Space mode) completes in under 10 seconds from "Sign in" click to main app view
- **SC-009**: Users can issue API tokens and configure MCP clients with documented steps, successfully making MCP HTTP requests within 5 minutes of first login
- **SC-010**: Manual index rebuild completes in under 30 seconds for vaults with up to 1,000 notes
- **SC-011**: System handles 10 concurrent users (each making read/write/search operations) without response time degradation beyond 2x baseline
- **SC-012**: Broken wikilinks are visually distinct and offer "Create note" affordance, with 90% of test users successfully creating target notes on first attempt
- **SC-013**: Note edit-save cycle with version conflict detection prevents silent overwrites in 100% of conflict scenarios
- **SC-014**: All API endpoints return appropriate HTTP status codes (200, 201, 400, 401, 403, 409, 413, 500) with clear error messages in JSON format

## Assumptions

1. **Deployment target**: Primary deployment is Hugging Face Space with OAuth. Local PoC uses static token or no auth.
2. **Note format**: All notes are UTF-8 encoded Markdown with optional YAML frontmatter. No binary files or non-.md formats in vaults.
3. **Wikilink syntax**: Only `[[wikilink]]` syntax is supported (Obsidian-style). No `[[link|alias]]` or other variants in initial version.
4. **Search sophistication**: Simple tokenization (split on non-alphanum) is sufficient. No stemming, synonyms, or advanced NLP.
5. **Concurrency model**: SQLite is acceptable for per-user indices given hackathon scale. Future versions may need distributed DB for higher concurrency.
6. **Frontend deployment**: Frontend is served from the same Python process as backend (static files or integrated framework), not a separate service.
7. **MCP client types**: Primary MCP clients are Claude Code, Claude Desktop (STDIO), and potentially other MCP-compatible tools via HTTP transport.
8. **Storage**: Filesystem-based vault storage is sufficient. No object storage or cloud sync in initial version.
9. **Performance targets**: Designed for individual user vaults (hundreds to low thousands of notes), not enterprise-scale knowledge bases.
10. **Security**: HTTPS is assumed for HF Space deployment. JWT secret management follows standard env-var configuration practices.

## Scope Boundaries

### In Scope

- Multi-tenant vault storage with per-user directories
- Full-text search, tag index, and bidirectional link graph
- Wikilink resolution with ambiguity handling and broken link detection
- HTTP API with Bearer auth (JWT)
- FastMCP server with STDIO (local) and HTTP (remote) transports
- React + shadcn/ui frontend with Obsidian-style layout
- Read-first UI with secondary editing capability
- Optimistic concurrency for UI, last-write-wins for MCP
- HF OAuth integration for multi-user Space deployment
- Manual index rebuild and health monitoring

### Out of Scope

- AI-driven re-organization or "smart" refactors of documentation structure
- Complex agentic flows or autonomous planning by AI
- Wikilink aliases (`[[link|display text]]`)
- Advanced Markdown features (footnotes, math rendering, diagrams) beyond basic code/link support
- Real-time collaborative editing (operational transforms or CRDTs)
- Version history or rollback beyond current version conflict detection
- Export to non-Markdown formats (PDF, HTML, etc.)
- Import from other note systems (Notion, Evernote, etc.)
- Mobile-optimized UI (desktop-first only)
- Offline support or PWA capabilities
- Fine-grained RBAC or multi-user permissions within a single vault (each user has their own isolated vault)
- Auto-save or draft states in UI editor
