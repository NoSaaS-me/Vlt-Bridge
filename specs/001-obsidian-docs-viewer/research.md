# Research: Multi-Tenant Obsidian-Like Docs Viewer

**Branch**: `001-obsidian-docs-viewer` | **Date**: 2025-11-15 | **Plan**: [plan.md](./plan.md)

## Overview

This document captures technical research and decisions for the implementation of a multi-tenant Obsidian-like documentation viewer. Each section addresses a specific research topic from Phase 0 of the implementation plan.

---

## 1. FastMCP HTTP Transport Authentication (Bearer Token)

### Decision

Use FastMCP's built-in `BearerAuth` mechanism with JWT token validation for HTTP transport authentication.

**Implementation approach**:
- Server: Configure FastMCP HTTP transport to accept `Authorization: Bearer <token>` header
- Client: Pass JWT token as string to `auth` parameter (FastMCP adds "Bearer" prefix automatically)
- Token format: JWT with claims `sub=user_id`, `exp=now+90days`, signed with `HS256` and server secret

### Rationale

1. **Native FastMCP support**: FastMCP provides first-class Bearer token authentication via `BearerAuth` class and string token shortcuts
2. **Minimal configuration**: Client code is as simple as `Client("https://...", auth="<token>")`
3. **Standard compliance**: Uses industry-standard `Authorization: Bearer` header pattern
4. **Transport flexibility**: Works seamlessly with both HTTP and SSE (Server-Sent Events) transports
5. **Non-interactive workflow**: Perfect for AI agents and service accounts that need programmatic access

### Alternatives Considered

**Alternative 1: Custom header authentication**
- **Rejected**: FastMCP supports custom headers but requires manual implementation of auth logic
- **Why rejected**: More complex, loses benefit of FastMCP's built-in token handling and validation

**Alternative 2: OAuth flow for MCP clients**
- **Rejected**: FastMCP supports full OAuth 2.1 flows with browser-based authentication
- **Why rejected**: Overly complex for AI agent use case; requires interactive browser flow which doesn't suit MCP STDIO or programmatic access patterns

**Alternative 3: API key-based authentication**
- **Rejected**: Could use simple API keys instead of JWTs
- **Why rejected**: JWTs provide expiration, claims, and stateless validation; better security posture for multi-tenant system

### Implementation Notes

**Server-side setup**:
```python
from fastmcp import FastMCP
from fastmcp.server.auth import BearerAuthProvider
import jwt

# For token validation (if using external issuer)
auth_provider = BearerAuthProvider(
    public_key="<RSA_PUBLIC_KEY>",
    issuer="https://your-issuer.com",
    audience="your-api"
)

# For internal JWT validation (our use case)
# Validate manually in middleware/dependency injection
def validate_jwt(token: str) -> str:
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return payload["sub"]  # user_id
```

**Client-side setup**:
```python
from fastmcp import Client

# Simplest approach - pass token as string
async with Client(
    "https://fastmcp.cloud/mcp",
    auth="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
) as client:
    await client.call_tool("list_notes", {})

# Explicit approach - use BearerAuth class
from fastmcp.client.auth import BearerAuth

async with Client(
    "https://fastmcp.cloud/mcp",
    auth=BearerAuth(token="eyJhbGci...")
) as client:
    await client.call_tool("list_notes", {})
```

**Key points**:
- Do NOT include "Bearer" prefix when passing token - FastMCP adds it automatically
- Token validation happens on every MCP tool call via HTTP transport
- STDIO transport bypasses authentication (local development only)
- For HF Space deployment, combine with HF OAuth to issue user-specific JWTs

**References**:
- FastMCP Bearer Auth docs: https://gofastmcp.com/clients/auth/bearer
- FastMCP authentication patterns: https://gyliu513.github.io/jekyll/update/2025/08/12/fastmcp-auth-patterns.html

---

## 2. Hugging Face Space OAuth Integration

### Decision

Use `huggingface_hub` library's built-in OAuth helpers (`attach_huggingface_oauth`, `parse_huggingface_oauth`) for zero-configuration OAuth integration in HF Spaces.

**Implementation approach**:
- Add `hf_oauth: true` to Space metadata in README.md
- Call `attach_huggingface_oauth(app)` to auto-register OAuth endpoints (`/oauth/huggingface/login`, `/oauth/huggingface/logout`, `/oauth/huggingface/callback`)
- Call `parse_huggingface_oauth(request)` in route handlers to extract authenticated user info
- Map HF username/ID to internal `user_id` for vault scoping

### Rationale

1. **Zero-configuration**: HF Spaces automatically injects OAuth environment variables (`OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`, `OAUTH_SCOPES`) when `hf_oauth: true` is set
2. **Local dev friendly**: `parse_huggingface_oauth` returns mock user in local mode, enabling seamless development without OAuth setup
3. **Minimal code**: Two function calls provide complete OAuth flow (login redirect, callback handling, session management)
4. **First-class support**: Official HF library with guaranteed compatibility with Spaces platform
5. **Standard OAuth 2.0**: Under the hood, implements industry-standard OAuth with PKCE

### Alternatives Considered

**Alternative 1: Manual OAuth implementation**
- **Rejected**: Implement OAuth flow manually using `authlib` or `requests-oauthlib`
- **Why rejected**: Significantly more code, requires manual handling of PKCE, state validation, and token exchange; error-prone and loses HF Spaces auto-configuration

**Alternative 2: Third-party auth provider (Auth0, WorkOS)**
- **Rejected**: Use external auth service and connect HF as identity provider
- **Why rejected**: Adds unnecessary complexity and external dependencies for a system designed specifically for HF Spaces deployment

**Alternative 3: Session-based auth without OAuth**
- **Rejected**: Use simple username/password with cookie sessions
- **Why rejected**: Poor UX (users already have HF accounts), requires password management, doesn't leverage HF ecosystem integration

### Implementation Notes

**Space configuration** (README.md frontmatter):
```yaml
---
title: Obsidian Docs Viewer
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
hf_oauth: true  # <-- Enable OAuth
---
```

**Backend integration** (FastAPI):
```python
from fastapi import FastAPI, Request
from huggingface_hub import attach_huggingface_oauth, parse_huggingface_oauth

app = FastAPI()

# Auto-register OAuth endpoints
attach_huggingface_oauth(app)

@app.get("/")
def index(request: Request):
    oauth_info = parse_huggingface_oauth(request)

    if oauth_info is None:
        return {"message": "Not logged in", "login_url": "/oauth/huggingface/login"}

    # Extract user info
    user_id = oauth_info.user_info.preferred_username  # or use 'sub' for UUID
    display_name = oauth_info.user_info.name
    avatar = oauth_info.user_info.picture

    return {
        "user_id": user_id,
        "display_name": display_name,
        "avatar": avatar
    }

@app.get("/api/me")
def get_current_user(request: Request):
    oauth_info = parse_huggingface_oauth(request)
    if oauth_info is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Map to internal user model
    user_id = oauth_info.user_info.preferred_username

    # Initialize vault on first login if needed
    vault_service.ensure_vault_exists(user_id)

    return {
        "user_id": user_id,
        "hf_profile": {
            "username": oauth_info.user_info.preferred_username,
            "name": oauth_info.user_info.name,
            "avatar": oauth_info.user_info.picture
        }
    }
```

**Frontend integration** (React):
```typescript
// Check auth status on app load
useEffect(() => {
  fetch('/api/me')
    .then(res => {
      if (res.ok) return res.json();
      throw new Error('Not authenticated');
    })
    .then(user => setCurrentUser(user))
    .catch(() => window.location.href = '/oauth/huggingface/login');
}, []);
```

**Key points**:
- `attach_huggingface_oauth` must be called BEFORE defining routes that need auth
- `parse_huggingface_oauth` returns `None` if not authenticated (check before accessing user_info)
- In local development, returns mocked user with deterministic username (e.g., "local-user")
- OAuth tokens/sessions are managed by `huggingface_hub` (stored in cookies)
- For API/MCP access, issue separate JWT after OAuth login via `POST /api/tokens`

**Environment variables** (auto-injected in HF Space):
- `OAUTH_CLIENT_ID`: Public client identifier
- `OAUTH_CLIENT_SECRET`: Secret for token exchange
- `OAUTH_SCOPES`: Space-specific scopes (typically `openid profile`)

**References**:
- HF OAuth docs: https://huggingface.co/docs/hub/spaces-oauth
- huggingface_hub API: https://huggingface.co/docs/huggingface_hub/en/package_reference/oauth

---

## 3. SQLite Schema Design for Multi-Index Storage

### Decision

Use SQLite with FTS5 (Full-Text Search 5) for full-text indexing, plus separate regular tables for tags and link graph. Implement per-user isolation via `user_id` column in all tables.

**Schema approach**:
- **Full-text index**: FTS5 virtual table with `title` and `body` columns, using `porter` tokenizer for stemming
- **Tag index**: Regular table with `user_id`, `tag`, `note_path` (many-to-many relationship)
- **Link graph**: Regular table with `user_id`, `source_path`, `target_path`, `link_text`, `is_resolved`
- **Metadata index**: Regular table with `user_id`, `note_path`, `version`, `created`, `updated`, `title` for fast lookups
- **Index health**: Regular table with `user_id`, `note_count`, `last_full_rebuild`, `last_incremental_update`

### Rationale

1. **FTS5 performance**: Native full-text search with inverted index, sub-100ms query times for thousands of documents
2. **Separate concerns**: Full-text (FTS5), tags (simple lookup), and links (graph traversal) have different query patterns; separate tables optimize each
3. **Per-user isolation**: `user_id` column in all tables enables simple WHERE filtering without complex row-level security
4. **External content pattern**: FTS5 with `content=''` (contentless) avoids storing document text twice (already in filesystem)
5. **Version tracking**: Metadata table stores version counter for optimistic concurrency without polluting frontmatter
6. **Prefix indexes**: FTS5 `prefix='2 3'` option enables fast autocomplete/prefix search

### Alternatives Considered

**Alternative 1: Single FTS5 table for everything**
- **Rejected**: Store tags and links as UNINDEXED columns in FTS5 table
- **Why rejected**: FTS5 is optimized for full-text, not structured data; complex queries (e.g., "all notes with tag X") would require scanning all rows; tags/links don't benefit from tokenization

**Alternative 2: Separate SQLite database per user**
- **Rejected**: One `.db` file per user instead of `user_id` column
- **Why rejected**: Increases file I/O overhead, complicates connection pooling, harder to implement global admin queries (e.g., total user count)

**Alternative 3: PostgreSQL with pg_trgm or RUM indexes**
- **Rejected**: Use full Postgres instead of SQLite
- **Why rejected**: Overkill for single-server deployment, adds deployment complexity, SQLite is sufficient for target scale (5,000 notes/user, 10 concurrent users)

**Alternative 4: In-memory index only**
- **Rejected**: Build inverted index in Python dict, no persistence
- **Why rejected**: Slow startup (rebuild on every restart), no durability, doesn't scale beyond single process

### Implementation Notes

**Schema definition**:
```sql
-- Metadata index (fast lookups, version tracking)
CREATE TABLE IF NOT EXISTS note_metadata (
    user_id TEXT NOT NULL,
    note_path TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    title TEXT NOT NULL,
    created TEXT NOT NULL,  -- ISO 8601 timestamp
    updated TEXT NOT NULL,  -- ISO 8601 timestamp
    PRIMARY KEY (user_id, note_path)
);
CREATE INDEX idx_metadata_user ON note_metadata(user_id);
CREATE INDEX idx_metadata_updated ON note_metadata(user_id, updated DESC);

-- Full-text search index (FTS5, contentless)
CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
    user_id UNINDEXED,
    note_path UNINDEXED,
    title,
    body,
    content='',  -- Contentless (we don't store the actual text)
    tokenize='porter unicode61',  -- Stemming + Unicode support
    prefix='2 3'  -- Prefix indexes for autocomplete
);

-- Tag index (many-to-many: notes <-> tags)
CREATE TABLE IF NOT EXISTS note_tags (
    user_id TEXT NOT NULL,
    note_path TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (user_id, note_path, tag)
);
CREATE INDEX idx_tags_user_tag ON note_tags(user_id, tag);
CREATE INDEX idx_tags_user_path ON note_tags(user_id, note_path);

-- Link graph (outgoing links from notes)
CREATE TABLE IF NOT EXISTS note_links (
    user_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    target_path TEXT,  -- NULL if unresolved
    link_text TEXT NOT NULL,  -- Original [[link text]]
    is_resolved INTEGER NOT NULL DEFAULT 0,  -- Boolean: 0=broken, 1=resolved
    PRIMARY KEY (user_id, source_path, link_text)
);
CREATE INDEX idx_links_user_source ON note_links(user_id, source_path);
CREATE INDEX idx_links_user_target ON note_links(user_id, target_path);
CREATE INDEX idx_links_unresolved ON note_links(user_id, is_resolved);

-- Index health tracking
CREATE TABLE IF NOT EXISTS index_health (
    user_id TEXT PRIMARY KEY,
    note_count INTEGER NOT NULL DEFAULT 0,
    last_full_rebuild TEXT,  -- ISO 8601 timestamp
    last_incremental_update TEXT  -- ISO 8601 timestamp
);
```

**Query patterns**:
```python
# Full-text search with ranking
cursor.execute("""
    SELECT
        note_path,
        title,
        bm25(note_fts, 3.0, 1.0) AS rank  -- Title weight=3, body weight=1
    FROM note_fts
    WHERE user_id = ? AND note_fts MATCH ?
    ORDER BY rank DESC
    LIMIT 50
""", (user_id, query))

# Get all notes with a specific tag
cursor.execute("""
    SELECT DISTINCT note_path, title
    FROM note_tags t
    JOIN note_metadata m USING (user_id, note_path)
    WHERE t.user_id = ? AND t.tag = ?
    ORDER BY m.updated DESC
""", (user_id, tag))

# Get backlinks for a note
cursor.execute("""
    SELECT DISTINCT l.source_path, m.title
    FROM note_links l
    JOIN note_metadata m ON l.user_id = m.user_id AND l.source_path = m.note_path
    WHERE l.user_id = ? AND l.target_path = ?
    ORDER BY m.updated DESC
""", (user_id, target_path))

# Get all unresolved links for UI display
cursor.execute("""
    SELECT source_path, link_text
    FROM note_links
    WHERE user_id = ? AND is_resolved = 0
""", (user_id,))
```

**Incremental update strategy**:
1. On `write_note`: Delete all existing rows for `(user_id, note_path)`, then insert new rows
2. Use transactions to ensure atomicity (delete old + insert new = single atomic operation)
3. Update `index_health.last_incremental_update` on every write

**Full rebuild strategy**:
1. Delete all index rows for `user_id`
2. Scan all `.md` files in vault directory
3. Parse each file and insert into all indexes
4. Update `index_health.note_count` and `last_full_rebuild`

**Key points**:
- FTS5 with `content=''` is contentless - we must manually INSERT/DELETE rows (no automatic synchronization)
- Use `porter` tokenizer for English stemming (search "running" matches "run")
- `bm25()` function provides relevance ranking (better than simple MATCH count)
- Prefix indexes (`prefix='2 3'`) enable fast `MATCH 'prefix*'` queries
- `UNINDEXED` columns in FTS5 are retrievable but not searchable (good for IDs)

**References**:
- SQLite FTS5 docs: https://www.sqlite.org/fts5.html
- FTS5 structure deep dive: https://darksi.de/13.sqlite-fts5-structure/

---

## 4. Wikilink Normalization and Resolution

### Decision

Implement case-insensitive normalized slug matching with deterministic ambiguity resolution based on Obsidian's behavior.

**Normalization algorithm**:
1. Extract link text from `[[link text]]`
2. Normalize: lowercase, replace spaces/hyphens/underscores with single dash, remove non-alphanumeric except dashes
3. Match normalized slug against normalized filename stems AND normalized frontmatter titles
4. If multiple matches: prefer same-folder match, then lexicographically smallest path

**Slug normalization function**:
```python
import re

def normalize_slug(text: str) -> str:
    """Normalize text to slug for case-insensitive matching."""
    text = text.lower()
    text = re.sub(r'[\s_]+', '-', text)  # Spaces/underscores → dash
    text = re.sub(r'[^a-z0-9-]', '', text)  # Keep only alphanumeric + dash
    text = re.sub(r'-+', '-', text)  # Collapse multiple dashes
    return text.strip('-')
```

### Rationale

1. **Obsidian compatibility**: Matches Obsidian's link resolution behavior (case-insensitive, flexible matching)
2. **User-friendly**: Users don't need to remember exact case or spacing (e.g., `[[API Design]]` matches `api-design.md`)
3. **Deterministic**: Same-folder preference + lexicographic tiebreaker ensures consistent resolution
4. **Efficient indexing**: Normalized slugs can be pre-computed and indexed for O(1) lookup
5. **Graceful fallback**: Broken links are tracked and displayed distinctly in UI

### Alternatives Considered

**Alternative 1: Exact case-sensitive matching**
- **Rejected**: Require `[[exact-filename]]` to match `exact-filename.md`
- **Why rejected**: Brittle user experience, doesn't match Obsidian behavior, forces users to remember exact capitalization

**Alternative 2: Fuzzy matching (Levenshtein distance)**
- **Rejected**: Use string similarity to find "close enough" matches
- **Why rejected**: Non-deterministic, slower, can match wrong notes ("Setup" matches "Startup"), confusing UX

**Alternative 3: Path-based links only**
- **Rejected**: Require full paths like `[[guides/setup]]` instead of `[[Setup]]`
- **Why rejected**: Verbose, doesn't match Obsidian's short-link paradigm, poor UX for large vaults

**Alternative 4: UUID-based links**
- **Rejected**: Use unique IDs like `[[#uuid-123]]` for stable references
- **Why rejected**: Not human-readable, requires additional metadata, doesn't match Obsidian convention

### Implementation Notes

**Resolution algorithm** (priority order):
```python
def resolve_wikilink(user_id: str, link_text: str, current_note_folder: str) -> str | None:
    """Resolve wikilink to note path, or None if unresolved."""
    normalized = normalize_slug(link_text)

    # Build candidate index: normalized_slug -> [note_paths]
    candidates = defaultdict(list)

    # Scan all notes for this user
    for note in list_all_notes(user_id):
        # Match against filename stem
        stem = Path(note.path).stem
        if normalize_slug(stem) == normalized:
            candidates[note.path].append(note.path)

        # Match against frontmatter title
        if note.title and normalize_slug(note.title) == normalized:
            candidates[note.path].append(note.path)

    if not candidates:
        return None  # Unresolved link

    paths = list(set(candidates.keys()))  # Deduplicate

    if len(paths) == 1:
        return paths[0]  # Unique match

    # Ambiguity resolution
    # 1. Prefer same-folder match
    same_folder = [p for p in paths if Path(p).parent == current_note_folder]
    if same_folder:
        return sorted(same_folder)[0]  # Lexicographic tiebreaker

    # 2. Lexicographically smallest path
    return sorted(paths)[0]
```

**Index optimization**:
Pre-compute normalized slugs for all notes and store in `note_metadata` table:
```sql
ALTER TABLE note_metadata ADD COLUMN normalized_title_slug TEXT;
ALTER TABLE note_metadata ADD COLUMN normalized_path_slug TEXT;
CREATE INDEX idx_metadata_title_slug ON note_metadata(user_id, normalized_title_slug);
CREATE INDEX idx_metadata_path_slug ON note_metadata(user_id, normalized_path_slug);
```

**Link extraction from Markdown**:
```python
import re

def extract_wikilinks(markdown_body: str) -> list[str]:
    """Extract all wikilink texts from markdown body."""
    pattern = r'\[\[([^\]]+)\]\]'
    return re.findall(pattern, markdown_body)
```

**Update link graph on write**:
```python
def update_link_graph(user_id: str, note_path: str, body: str):
    """Update outgoing links and backlinks for a note."""
    current_folder = str(Path(note_path).parent)

    # Extract wikilinks from body
    link_texts = extract_wikilinks(body)

    # Delete old links from this note
    db.execute("DELETE FROM note_links WHERE user_id=? AND source_path=?",
               (user_id, note_path))

    # Insert new links
    for link_text in link_texts:
        target_path = resolve_wikilink(user_id, link_text, current_folder)
        is_resolved = 1 if target_path else 0

        db.execute("""
            INSERT INTO note_links (user_id, source_path, target_path, link_text, is_resolved)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, note_path, target_path, link_text, is_resolved))
```

**UI rendering**:
```typescript
// Transform wikilinks to clickable links in rendered Markdown
function transformWikilinks(markdown: string, linkIndex: Record<string, string>): string {
  return markdown.replace(/\[\[([^\]]+)\]\]/g, (match, linkText) => {
    const targetPath = linkIndex[linkText];

    if (targetPath) {
      // Resolved link
      return `<a href="#/note/${encodeURIComponent(targetPath)}" class="wikilink">${linkText}</a>`;
    } else {
      // Broken link
      return `<a href="#/create/${encodeURIComponent(linkText)}" class="wikilink broken">${linkText}</a>`;
    }
  });
}
```

**Key points**:
- Pre-compute and cache slug mappings for performance (avoid re-scanning on every link resolution)
- Same-folder preference matches Obsidian's behavior (local references are intuitive)
- Lexicographic tiebreaker ensures determinism (same input always resolves to same output)
- Track `is_resolved` flag to identify broken links for UI warnings/affordances
- Update entire link graph on every note write (incremental update, not rebuild)

**Edge cases**:
- Empty link text `[[]]` - ignore/skip
- Nested brackets `[[foo [[bar]]]]` - naive regex fails; use proper parser or limit to non-nested pattern
- Link with pipe `[[link|display]]` - out of scope for MVP; treat entire string as link text

---

## 5. React + shadcn/ui Directory Tree Component

### Decision

Use **shadcn-extension Tree View** component with built-in virtualization via `@tanstack/react-virtual` for directory tree rendering.

**Component choice**: `shadcn-extension` Tree View
- **Installation**: Available at https://shadcn-extension.vercel.app/docs/tree-view
- **Features**: Virtualization, accordion-based expand/collapse, keyboard navigation, selection, custom icons
- **Why this one**: Only shadcn tree component with native virtualization support; critical for large vaults (5,000 notes)

### Rationale

1. **Virtualization required**: 5,000 notes would create 5,000+ DOM nodes without virtualization; TanStack Virtual renders only visible rows (~20-50 nodes)
2. **Performance**: Virtualization reduces initial render from ~2s to <100ms for large trees
3. **shadcn ecosystem**: Consistent styling with other shadcn/ui components (Button, ScrollArea, etc.)
4. **Accessibility**: Built on Radix UI primitives with keyboard navigation and ARIA support
5. **Customizable**: Supports custom icons per node, expand/collapse callbacks, and selection handling

### Alternatives Considered

**Alternative 1: MrLightful's shadcn Tree View**
- **Rejected**: Feature-rich component with drag-and-drop, custom icons
- **Why rejected**: No virtualization support; would cause performance issues with 1,000+ notes

**Alternative 2: Neigebaie's shadcn Tree View**
- **Rejected**: Advanced features (multi-select, checkboxes, context menus)
- **Why rejected**: No virtualization; overkill for simple directory browsing

**Alternative 3: react-arborist**
- **Rejected**: Powerful tree view library with virtualization and drag-and-drop
- **Why rejected**: Not part of shadcn ecosystem; requires custom styling to match UI; heavier dependency

**Alternative 4: Custom implementation with react-window**
- **Rejected**: Build tree view from scratch using `react-window` or `react-virtual`
- **Why rejected**: Significant development effort; reinventing the wheel; shadcn-extension already provides this

### Implementation Notes

**Installation**:
```bash
npx shadcn add https://shadcn-extension.vercel.app/registry/tree-view.json
```

**Component usage**:
```tsx
import { Tree, TreeNode } from "@/components/ui/tree-view";

interface FileTreeNode {
  id: string;
  name: string;
  path: string;
  isFolder: boolean;
  children?: FileTreeNode[];
}

function DirectoryTree({ vault, onSelectNote }: Props) {
  // Transform vault notes into tree structure
  const treeData = useMemo(() => buildTree(vault.notes), [vault.notes]);

  return (
    <Tree
      data={treeData}
      onSelectChange={(nodeId) => {
        const node = findNode(treeData, nodeId);
        if (!node.isFolder) {
          onSelectNote(node.path);
        }
      }}
      // Virtualization is enabled by default
      className="w-full h-full"
    />
  );
}

// Transform flat list of note paths into hierarchical tree
function buildTree(notes: Note[]): TreeNode[] {
  const root: Map<string, TreeNode> = new Map();

  for (const note of notes) {
    const parts = note.path.split('/');
    let currentLevel = root;

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const isFile = i === parts.length - 1;
      const id = parts.slice(0, i + 1).join('/');

      if (!currentLevel.has(part)) {
        currentLevel.set(part, {
          id,
          name: isFile ? note.title : part,
          path: id,
          isFolder: !isFile,
          children: isFile ? undefined : new Map()
        });
      }

      if (!isFile) {
        currentLevel = currentLevel.get(part)!.children!;
      }
    }
  }

  return Array.from(root.values());
}
```

**Styling for Obsidian-like appearance**:
```css
/* Custom styles for file tree */
.tree-view-node {
  @apply py-1 px-2 rounded hover:bg-accent transition-colors;
}

.tree-view-node.selected {
  @apply bg-accent text-accent-foreground font-medium;
}

.tree-view-folder {
  @apply flex items-center gap-2;
}

.tree-view-file {
  @apply flex items-center gap-2 text-sm;
}

/* Icons */
.folder-icon {
  @apply text-yellow-500;
}

.file-icon {
  @apply text-gray-500;
}
```

**Collapsible behavior**:
```tsx
// Track expanded folders in state
const [expanded, setExpanded] = useState<Set<string>>(new Set(['root']));

<Tree
  data={treeData}
  expanded={expanded}
  onExpandedChange={setExpanded}
  // Auto-expand to selected note's folder
  onSelectChange={(nodeId) => {
    const path = nodeId.split('/');
    const folders = path.slice(0, -1);
    setExpanded(new Set([...expanded, ...folders]));
  }}
/>
```

**Key points**:
- Virtualization is automatic with shadcn-extension Tree View (uses TanStack Virtual internally)
- Must transform flat note list into nested tree structure (use `buildTree` utility)
- Track expanded/collapsed state separately from tree data
- Custom icons per node type (folder vs file) via `icon` prop
- Use `ScrollArea` component from shadcn to wrap tree for custom scrollbars

**Performance targets**:
- Initial render: <200ms for 5,000 notes
- Expand/collapse: <50ms per folder
- Search filter: <100ms to re-render filtered tree

**Accessibility**:
- Keyboard navigation: Arrow keys to navigate, Enter to select, Space to expand/collapse
- Screen reader support: ARIA labels for folders/files, expand/collapse state
- Focus management: Visible focus indicators, focus restoration after selection

---

## 6. Optimistic Concurrency Implementation

### Decision

Use **version counter** (integer) stored in SQLite index with `if_version` parameter for UI writes. Implement **ETag-like validation** via `If-Match` header in HTTP API.

**Approach**:
- Version counter: Integer field in `note_metadata` table, incremented on every write
- UI writes: Include `if_version: N` in `PUT /api/notes/{path}` body
- Server validation: Compare `if_version` with current version; return `409 Conflict` if mismatch
- MCP writes: No version checking (last-write-wins)
- ETag header: Return `ETag: "<version>"` in `GET /api/notes/{path}` response for HTTP compliance

### Rationale

1. **Simple implementation**: Integer counter is trivial to increment and compare
2. **Explicit versioning**: Version in request body makes intent clear ("I'm updating version 5")
3. **Database-backed**: Version persists in index, not frontmatter (keeps note content clean)
4. **HTTP-friendly**: Can expose as ETag header for standards compliance
5. **Performance**: Integer comparison is O(1), no hash computation needed

### Alternatives Considered

**Alternative 1: ETag with content hash**
- **Rejected**: Compute MD5/SHA hash of note content, return as ETag header
- **Why rejected**: Hash computation on every read adds latency; version counter is sufficient and faster

**Alternative 2: Last-Modified timestamps**
- **Rejected**: Use `updated` timestamp + `If-Unmodified-Since` header
- **Why rejected**: Timestamp precision issues (SQLite stores ISO strings, not microsecond precision); race conditions if multiple updates within same second

**Alternative 3: Version in frontmatter**
- **Rejected**: Store `version: 5` in YAML frontmatter
- **Why rejected**: Pollutes user-facing metadata; incrementing version requires parsing/re-serializing frontmatter; harder to manage

**Alternative 4: MVCC (Multi-Version Concurrency Control)**
- **Rejected**: Store multiple versions of each note, allow rollback
- **Why rejected**: Complex implementation; storage overhead; out of scope for MVP (no version history requirement)

### Implementation Notes

**Schema addition**:
```sql
-- Version counter in note_metadata table
ALTER TABLE note_metadata ADD COLUMN version INTEGER NOT NULL DEFAULT 1;
```

**API endpoint implementation**:
```python
from fastapi import HTTPException, Header
from typing import Optional

@app.put("/api/notes/{path}")
async def update_note(
    path: str,
    body: NoteUpdateRequest,
    user_id: str = Depends(get_current_user),
    if_match: Optional[str] = Header(None)  # ETag header support
):
    # Get current version
    current = get_note_metadata(user_id, path)

    # Check if_version in body OR If-Match header
    expected_version = body.if_version or (int(if_match.strip('"')) if if_match else None)

    if expected_version is not None and current.version != expected_version:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "version_conflict",
                "message": "Note was updated by another process",
                "current_version": current.version,
                "provided_version": expected_version
            }
        )

    # Update note and increment version
    new_version = current.version + 1
    save_note(user_id, path, body.content)
    update_metadata(user_id, path, version=new_version, updated=now())

    return {
        "status": "ok",
        "version": new_version
    }

@app.get("/api/notes/{path}")
async def get_note(
    path: str,
    user_id: str = Depends(get_current_user)
):
    note = load_note(user_id, path)

    return JSONResponse(
        content={
            "path": note.path,
            "title": note.title,
            "metadata": note.metadata,
            "body": note.body,
            "version": note.version,
            "created": note.created,
            "updated": note.updated
        },
        headers={
            "ETag": f'"{note.version}"',  # Expose version as ETag
            "Cache-Control": "no-cache"   # Prevent stale reads
        }
    )
```

**Frontend implementation** (React):
```typescript
interface Note {
  path: string;
  title: string;
  body: string;
  version: number;
  // ...
}

async function saveNote(note: Note, newBody: string) {
  try {
    const response = await fetch(`/api/notes/${encodeURIComponent(note.path)}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
        // Option 1: Version in body
      },
      body: JSON.stringify({
        body: newBody,
        if_version: note.version  // Optimistic concurrency check
      })
    });

    if (response.status === 409) {
      const error = await response.json();
      alert(`Conflict: Note was updated (current version: ${error.current_version}). Please reload and try again.`);
      return;
    }

    const updated = await response.json();
    // Update local state with new version
    setNote({ ...note, body: newBody, version: updated.version });

  } catch (error) {
    console.error('Save failed:', error);
  }
}
```

**MCP tool implementation** (last-write-wins):
```python
@mcp.tool()
async def write_note(path: str, body: str, title: str = None) -> dict:
    """Write note via MCP (no version checking)."""
    user_id = get_user_from_context()

    # Load existing note to get current version (if exists)
    try:
        current = get_note_metadata(user_id, path)
        new_version = current.version + 1
    except NotFoundError:
        new_version = 1  # New note

    # Write without version check (last-write-wins)
    save_note(user_id, path, body, title)
    update_metadata(user_id, path, version=new_version, updated=now())

    return {"status": "ok", "path": path, "version": new_version}
```

**Conflict resolution UI**:
```tsx
function ConflictDialog({ currentVersion, serverVersion }: Props) {
  return (
    <Alert variant="destructive">
      <AlertTitle>Version Conflict</AlertTitle>
      <AlertDescription>
        This note was updated while you were editing (version {currentVersion} → {serverVersion}).
        <div className="mt-4 space-x-2">
          <Button onClick={reload}>Reload and Discard Changes</Button>
          <Button variant="outline" onClick={saveAsCopy}>Save as Copy</Button>
        </div>
      </AlertDescription>
    </Alert>
  );
}
```

**Key points**:
- Version counter starts at 1 for new notes, increments on every write
- HTTP API returns `409 Conflict` with detailed error message (current vs provided version)
- ETag header is optional but recommended for HTTP standards compliance
- MCP writes skip version check (AI agents don't need optimistic concurrency)
- Frontend shows clear error message with options: reload, save as copy, or manual merge

**Performance considerations**:
- Version check is single integer comparison (O(1))
- No need to read entire note content for validation
- Version update is atomic (SQLite transaction)

**References**:
- Optimistic concurrency patterns: https://event-driven.io/en/how_to_use_etag_header_for_optimistic_concurrency/
- HTTP conditional requests: https://developer.mozilla.org/en-US/docs/Web/HTTP/Conditional_requests

---

## 7. Markdown Frontmatter Parsing with Fallback

### Decision

Use `python-frontmatter` library for YAML parsing with try-except wrapper to handle malformed frontmatter gracefully. Implement fallback strategy: malformed YAML → treat as no frontmatter, extract title from first `# Heading` or filename stem.

**Parsing approach**:
```python
import frontmatter
from pathlib import Path

def parse_note(file_path: str) -> dict:
    """Parse note with frontmatter fallback."""
    try:
        # Attempt to parse frontmatter
        post = frontmatter.load(file_path)
        metadata = dict(post.metadata)
        body = post.content

    except (yaml.scanner.ScannerError, yaml.parser.ParserError) as e:
        # Malformed YAML - treat entire file as body
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        metadata = {}
        body = full_text

        # Log warning for debugging
        logger.warning(f"Malformed frontmatter in {file_path}: {e}")

    # Extract title (priority: frontmatter > first H1 > filename)
    title = (
        metadata.get('title') or
        extract_first_heading(body) or
        Path(file_path).stem
    )

    return {
        'title': title,
        'metadata': metadata,
        'body': body
    }

def extract_first_heading(markdown: str) -> str | None:
    """Extract first # Heading from markdown body."""
    match = re.match(r'^#\s+(.+)$', markdown, re.MULTILINE)
    return match.group(1).strip() if match else None
```

### Rationale

1. **Graceful degradation**: Malformed YAML doesn't break the system; note is still readable
2. **User-friendly**: Non-technical users may create invalid YAML; system should be forgiving
3. **Simple implementation**: Try-except wrapper is minimal code; `python-frontmatter` handles valid cases
4. **Fallback chain**: Title extraction has clear priority order (explicit > inferred > default)
5. **Debugging support**: Log warnings for malformed YAML so admins can fix source files

### Alternatives Considered

**Alternative 1: Strict parsing (fail on malformed YAML)**
- **Rejected**: Raise error and reject note with invalid frontmatter
- **Why rejected**: Poor UX; users may accidentally create invalid YAML (e.g., unquoted colons); breaks read-first workflow

**Alternative 2: TOML or JSON frontmatter**
- **Rejected**: Use `+++` TOML or `{{{ }}}` JSON delimiters instead of YAML
- **Why rejected**: Obsidian uses YAML exclusively; compatibility is critical

**Alternative 3: Lenient YAML parser**
- **Rejected**: Use `ruamel.yaml` with error recovery instead of PyYAML
- **Why rejected**: Adds complexity; `python-frontmatter` uses PyYAML internally; fallback strategy is simpler

**Alternative 4: Partial frontmatter extraction**
- **Rejected**: Parse valid keys, ignore malformed keys
- **Why rejected**: Difficult to implement; unclear semantics (which keys are valid?); safer to treat all as invalid

### Implementation Notes

**Error types to catch**:
```python
import yaml

try:
    post = frontmatter.load(file_path)
except (
    yaml.scanner.ScannerError,  # Invalid YAML syntax (e.g., unmatched quotes)
    yaml.parser.ParserError,    # Invalid YAML structure
    UnicodeDecodeError          # Non-UTF8 file encoding
) as e:
    # Fallback to no frontmatter
    pass
```

**Common malformed YAML examples**:
```yaml
---
title: API Design: Overview  # Unquoted colon - INVALID
tags: [backend, api]
---

---
title: "Setup Guide
description: Installation steps  # Unclosed quote - INVALID
---

---
  title: Indented incorrectly  # Bad indentation - INVALID
tags:
- frontend
---
```

**Auto-fix on write** (optional enhancement):
```python
def save_note(user_id: str, path: str, title: str, metadata: dict, body: str):
    """Save note with valid frontmatter (auto-fix on write)."""
    # Merge title into metadata
    metadata['title'] = title

    # Create Post object with validated metadata
    post = frontmatter.Post(body, **metadata)

    # Serialize with valid YAML
    file_content = frontmatter.dumps(post)

    # Write to file
    full_path = get_vault_path(user_id) / path
    full_path.write_text(file_content, encoding='utf-8')
```

**Title extraction regex**:
```python
def extract_first_heading(markdown: str) -> str | None:
    """Extract first # Heading (must be H1, not H2/H3)."""
    # Match # Heading (H1 only, not ## or ###)
    pattern = r'^#\s+(.+?)(?:\s+\{[^}]+\})?\s*$'
    match = re.search(pattern, markdown, re.MULTILINE)

    if match:
        heading = match.group(1).strip()
        # Remove Markdown formatting (e.g., **bold**, *italic*)
        heading = re.sub(r'[*_`]', '', heading)
        return heading

    return None
```

**Fallback priority**:
1. `metadata.get('title')` - Explicit frontmatter title
2. `extract_first_heading(body)` - First `# Heading` in body
3. `Path(file_path).stem` - Filename without `.md` extension

**Validation warnings**:
```python
# Add validation warnings to API response
if malformed_frontmatter:
    warnings.append({
        "type": "malformed_frontmatter",
        "message": "YAML frontmatter is invalid and was ignored",
        "line": error.problem_mark.line if hasattr(error, 'problem_mark') else None
    })
```

**UI display for warnings**:
```tsx
function NoteViewer({ note, warnings }: Props) {
  return (
    <div>
      {warnings.map(w => (
        <Alert key={w.type} variant="warning">
          <AlertTitle>Warning</AlertTitle>
          <AlertDescription>{w.message}</AlertDescription>
        </Alert>
      ))}
      <Markdown>{note.body}</Markdown>
    </div>
  );
}
```

**Key points**:
- Always catch `yaml.scanner.ScannerError` and `yaml.parser.ParserError` from PyYAML
- Log warnings with file path and error details for admin debugging
- Prefer graceful fallback over strict validation (read-first workflow)
- Auto-fix on write ensures newly saved notes have valid frontmatter
- Extract title from first `# Heading`, not `## Subheading` (H1 only)

**References**:
- python-frontmatter docs: https://python-frontmatter.readthedocs.io/
- PyYAML error handling: https://pyyaml.org/wiki/PyYAMLDocumentation

---

## 8. JWT Token Management in React

### Decision

Use **hybrid approach**: Store short-lived access token (JWT) in **memory** (React state/context), store long-lived refresh token in **HttpOnly cookie** (server-managed). For MVP without refresh tokens, store JWT in **memory only** with 90-day expiration.

**MVP approach** (no refresh tokens):
- Store JWT in React Context (memory)
- Token expires after 90 days (long-lived)
- On app load, check if token exists in memory → if not, redirect to login
- No localStorage (XSS vulnerability mitigation)
- No refresh flow (acceptable for MVP scale)

**Production approach** (with refresh tokens):
- Access token: 15-minute expiration, stored in memory
- Refresh token: 90-day expiration, stored in HttpOnly cookie
- Automatic refresh before access token expires
- Refresh endpoint: `POST /api/auth/refresh` (validates cookie, issues new access token)

### Rationale

1. **XSS protection**: Memory storage prevents JavaScript-based token theft (localStorage is vulnerable to XSS)
2. **CSRF protection**: HttpOnly cookies can't be accessed by JS, mitigating CSRF (when combined with SameSite attribute)
3. **Industry best practice (2025)**: Hybrid approach is current security standard for React SPAs
4. **Acceptable UX**: User logs in once per 90 days (or once per session if memory-only)
5. **No additional dependencies**: Built-in React Context API handles memory storage

### Alternatives Considered

**Alternative 1: localStorage for JWT**
- **Rejected**: Store JWT in `localStorage.setItem('token', jwt)`
- **Why rejected**: Vulnerable to XSS attacks (malicious scripts can read localStorage); still in OWASP Top 10; unacceptable security risk for multi-tenant system

**Alternative 2: sessionStorage for JWT**
- **Rejected**: Store JWT in `sessionStorage` (cleared on tab close)
- **Why rejected**: Poor UX (re-login on every new tab); still vulnerable to XSS

**Alternative 3: Cookies for both access and refresh tokens**
- **Rejected**: Store JWT in regular cookies (not HttpOnly)
- **Why rejected**: Vulnerable to CSRF if not using HttpOnly; vulnerable to XSS if accessible to JS

**Alternative 4: No token storage (re-authenticate on every request)**
- **Rejected**: Use HF OAuth on every API call
- **Why rejected**: Unacceptable latency; OAuth flow is slow (~2-3s per request)

### Implementation Notes

**MVP implementation** (memory-only, 90-day JWT):

```typescript
// Auth context (memory storage)
import { createContext, useContext, useState, useEffect } from 'react';

interface AuthContextType {
  token: string | null;
  setToken: (token: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setTokenState] = useState<string | null>(null);

  const setToken = (newToken: string) => {
    setTokenState(newToken);
  };

  const logout = () => {
    setTokenState(null);
    window.location.href = '/oauth/huggingface/logout';
  };

  return (
    <AuthContext.Provider value={{ token, setToken, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
}
```

```typescript
// App initialization (fetch token after OAuth)
function App() {
  const { token, setToken } = useAuth();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if authenticated via HF OAuth
    fetch('/api/me')
      .then(res => {
        if (!res.ok) throw new Error('Not authenticated');
        return res.json();
      })
      .then(user => {
        // Issue JWT token for API access
        return fetch('/api/tokens', { method: 'POST' });
      })
      .then(res => res.json())
      .then(data => {
        setToken(data.token);
        setLoading(false);
      })
      .catch(() => {
        // Redirect to OAuth login
        window.location.href = '/oauth/huggingface/login';
      });
  }, []);

  if (loading) return <div>Loading...</div>;

  return <MainApp />;
}
```

```typescript
// API client (include token in headers)
async function apiRequest(endpoint: string, options: RequestInit = {}) {
  const { token } = useAuth();

  const response = await fetch(`/api${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
      ...options.headers
    }
  });

  if (response.status === 401) {
    // Token expired or invalid
    logout();
    throw new Error('Unauthorized');
  }

  return response;
}
```

**Production implementation** (with refresh tokens):

```typescript
// Token refresh logic
let refreshPromise: Promise<string> | null = null;

async function refreshAccessToken(): Promise<string> {
  // Prevent multiple concurrent refresh calls
  if (refreshPromise) return refreshPromise;

  refreshPromise = fetch('/api/auth/refresh', {
    method: 'POST',
    credentials: 'include'  // Send HttpOnly cookie
  })
    .then(res => {
      if (!res.ok) throw new Error('Refresh failed');
      return res.json();
    })
    .then(data => {
      setToken(data.access_token);
      refreshPromise = null;
      return data.access_token;
    })
    .catch(err => {
      refreshPromise = null;
      logout();
      throw err;
    });

  return refreshPromise;
}

// Automatic refresh before token expires
useEffect(() => {
  if (!token) return;

  // Parse token to get expiration
  const payload = JSON.parse(atob(token.split('.')[1]));
  const expiresAt = payload.exp * 1000;
  const now = Date.now();
  const refreshAt = expiresAt - (5 * 60 * 1000);  // 5 minutes before expiry

  const timeoutId = setTimeout(() => {
    refreshAccessToken();
  }, refreshAt - now);

  return () => clearTimeout(timeoutId);
}, [token]);
```

**Backend refresh endpoint**:
```python
from fastapi import Cookie, HTTPException

@app.post("/api/auth/refresh")
async def refresh_token(
    refresh_token: str = Cookie(None, httponly=True, samesite='strict')
):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token")

    # Validate refresh token
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # Issue new access token (15-minute expiry)
    access_token = create_jwt(user_id, expiration_minutes=15)

    return {"access_token": access_token, "token_type": "bearer"}
```

**Key points**:
- Memory storage = token lost on page refresh (re-login required) → acceptable for MVP
- HttpOnly cookies cannot be accessed by JavaScript (XSS protection)
- Set `SameSite=strict` on refresh token cookie (CSRF protection)
- Refresh token rotation: issue new refresh token on each refresh (advanced security)
- Use `credentials: 'include'` in fetch to send HttpOnly cookies
- Parse JWT client-side to schedule refresh (or use server-sent expiry hint)

**Security checklist**:
- ✅ Access token in memory (XSS-resistant)
- ✅ Refresh token in HttpOnly cookie (XSS-resistant)
- ✅ SameSite=strict on cookies (CSRF-resistant)
- ✅ HTTPS required (prevent MITM)
- ✅ Short access token expiry (limit blast radius)
- ✅ Token refresh before expiry (seamless UX)
- ✅ Logout clears both tokens

**MVP vs Production tradeoff**:
- **MVP**: 90-day JWT in memory → simpler, acceptable for hackathon/PoC
- **Production**: 15-min access + 90-day refresh → better security, more complex

**References**:
- JWT storage best practices: https://www.descope.com/blog/post/developer-guide-jwt-storage
- HttpOnly cookies vs localStorage: https://www.wisp.blog/blog/understanding-token-storage-local-storage-vs-httponly-cookies
- React authentication patterns: https://marmelab.com/blog/2020/07/02/manage-your-jwt-react-admin-authentication-in-memory.html

---

## Summary of Key Decisions

| Topic | Decision | Primary Rationale |
|-------|----------|-------------------|
| **FastMCP Auth** | Bearer token with JWT validation | Native FastMCP support, minimal config, standard-compliant |
| **HF OAuth** | `attach_huggingface_oauth` + `parse_huggingface_oauth` | Zero-config, local dev friendly, official HF support |
| **SQLite Schema** | FTS5 for full-text + separate tables for tags/links | Performance, per-user isolation, optimized query patterns |
| **Wikilink Resolution** | Case-insensitive slug matching + same-folder preference | Obsidian compatibility, user-friendly, deterministic |
| **Directory Tree** | shadcn-extension Tree View with virtualization | Only shadcn option with virtualization for 5K+ notes |
| **Optimistic Concurrency** | Version counter in SQLite + `if_version` param | Simple, fast, HTTP-friendly, no content hashing overhead |
| **Frontmatter Parsing** | `python-frontmatter` + fallback to no frontmatter | Graceful degradation, user-friendly error handling |
| **JWT Management** | Memory storage (MVP) or memory + HttpOnly cookie (prod) | XSS protection, industry best practice (2025) |

---

## Next Steps

With research complete, proceed to **Phase 1: Data Model & Contracts**:

1. Create `data-model.md` with detailed Pydantic models and SQLite schemas
2. Create `contracts/http-api.yaml` with OpenAPI 3.1 specification
3. Create `contracts/mcp-tools.json` with MCP tool schemas (JSON Schema format)
4. Create `quickstart.md` with setup instructions and testing workflows

After Phase 1, run `/speckit.tasks` to generate dependency-ordered implementation tasks.
