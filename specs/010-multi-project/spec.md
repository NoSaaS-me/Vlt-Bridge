# 010: Multi-Project Architecture

## Overview

Transform Vlt-Bridge from a single-vault-per-user architecture to a multi-project-per-user architecture where each project has its own:
- Markdown vault (documentation)
- Vector store for CodeRAG (code intelligence)
- Threads from vlt CLI (development history/memory)
- Oracle context trees (conversation history)

## User Stories

### US-1: Project Selection
**As a** user
**I want to** select different projects from a dropdown
**So that** I can switch between different codebases/documentation sets

### US-2: Project Creation
**As a** user
**I want to** create new projects
**So that** I can track separate codebases independently

### US-3: Project-Scoped Data
**As a** user
**I want** my notes, threads, and code indexes isolated per project
**So that** queries don't mix context from different codebases

### US-4: vlt CLI Integration
**As a** developer using vlt CLI
**I want** my local project to sync with the web UI
**So that** I can access my threads and context from either interface

## Current Architecture Analysis

### One User = One Vault (Current)
```
User (user_id)
├── Vault: data/vaults/{user_id}/
├── RAG Index: llamaindex/{user_id}/
├── Threads: threads WHERE user_id = ?
├── Context Trees: context_trees WHERE user_id = ? AND project_id = ?
└── Settings: user_settings WHERE user_id = ?
```

### One User = Multiple Projects (Target)
```
User (user_id)
├── Project "vlt-bridge"
│   ├── Vault: data/vaults/{user_id}/vlt-bridge/
│   ├── RAG Index: llamaindex/{user_id}/vlt-bridge/
│   ├── Threads: threads WHERE user_id = ? AND project_id = 'vlt-bridge'
│   ├── Context Trees: context_trees WHERE ... AND project_id = 'vlt-bridge'
│   └── Settings: user_settings WHERE ... AND project_id = 'vlt-bridge'
│
├── Project "document-mcp"
│   └── ... (same structure)
│
└── Project "my-codebase"
    └── ... (same structure)
```

## Key Findings from Codebase Analysis

### Backend
1. **project_id already exists** in threads, oracle_contexts, context_trees tables
2. **NOT in vault-related tables**: note_metadata, note_fts, note_tags, note_links, index_health
3. **Vault paths hardcoded**: `vault_root / user_id / note_path` (no project layer)
4. **RAG index paths hardcoded**: `llamaindex_persist_dir / user_id /`

### vlt CLI
1. **Already project-scoped**: All tables have project_id foreign keys
2. **vlt.toml anchors projects**: `vlt init -p "Project Name"` creates config
3. **CodeRAG is project-scoped**: Embeddings, FTS, graph all filter by project_id
4. **ThreadSyncClient exists**: Can push threads to backend

### Frontend
1. **No project concept**: Single vault loaded on mount
2. **State in MainApp.tsx**: ~25 useState calls, all vault-centric
3. **DEMO banner at lines 595-600**: Unconditionally rendered
4. **ResizablePanelGroup layout**: Left sidebar | Main content | Chat flyout

## UI Changes (Per Mockup)

### Current Layout
```
┌─────────────────────────────────────────────────────────────┐
│ DEMO ONLY - ALL DATA IS TEMPORARY...                        │ ← Remove
├─────────────────────────────────────────────────────────────┤
│                      VAULT.MCP                    [C][G][S] │
├─────────────┬───────────────────────────────────────────────┤
│ [+New Note] │ Main Content                                  │
│ [+Folder]   │ - Note Viewer / Editor                       │
│ Search...   │ - Graph View                                 │
│             │ - Chat (center mode)                         │
│ Tree        │                                              │
└─────────────┴───────────────────────────────────────────────┘
```

### Target Layout
```
┌─────────────────────────────────────────────────────────────────────────┐
│ [Project ▼]              [T] [I] [C] [G] [S]              [Model] [...] │
├─────────────┬───────────────────────────────────────────────┬───────────┤
│ Navigation  │ Header: Oracle/Document/Issue/Thread          │ Flyout    │
│ Controls    │─────────────────────────────────────────────  │ Pane      │
│             │                                               │           │
│ Tree of     │ Main Content                                 │ Threads   │
│ Vlt Docs    │ - Note Viewer / Editor                       │ Issues    │
│ and Dirs    │ - Graph View                                 │ (dynamic) │
│             │ - AI Chat (full screen option)               │           │
│             │                                               │           │
├─────────────┼───────────────────────────────────────────────┴───────────┤
│             │ AI CHAT                                        [Send]     │
└─────────────┴───────────────────────────────────────────────────────────┘

Icon Legend: T=Threads | I=Issues | C=Chat | G=Graph | S=Settings
```

### UI Components Needed

1. **ProjectDropdown** (top-left)
   - List user's projects
   - Create new project option
   - Switch triggers data reload

2. **NavigationBar** (top-center)
   - T: Threads flyout toggle
   - I: Issue Tracking flyout toggle (future scope)
   - C: Chat toggle (existing)
   - G: Graph view toggle (existing)
   - S: Settings navigation (existing)

3. **ThreadsFlyout** (right side)
   - List threads for current project
   - Click to view thread details
   - Selecting populates main view

4. **IssuesFlyout** (right side, future scope)
   - Integration with `bd` (beads) CLI
   - Out of scope for initial implementation

## Technical Requirements

### Backend Changes

#### Database Schema Migration
Add `project_id` to vault-related tables:

```sql
-- note_metadata: Add project_id to composite PK
ALTER TABLE note_metadata ADD COLUMN project_id TEXT NOT NULL DEFAULT 'default';
-- Update PK to (user_id, project_id, note_path)

-- note_fts: Rebuild with project_id
-- note_tags: Add project_id
-- note_links: Add project_id
-- index_health: Add project_id (per-project health)
-- user_settings: Optional per-project settings
```

#### New Endpoints
```
GET  /api/projects                    # List user's projects
POST /api/projects                    # Create project
GET  /api/projects/{id}               # Get project details
PUT  /api/projects/{id}               # Update project
DELETE /api/projects/{id}             # Delete project (and all data)
```

#### Vault Service Changes
```python
# Current: vault_root / user_id / note_path
# Target:  vault_root / user_id / project_id / note_path

def sanitize_path(user_id: str, project_id: str, vault_root: Path, note_path: str):
    project_vault = vault_root / user_id / project_id
    # Ensure project_vault exists
    # Resolve note_path relative to project_vault
```

#### RAG Index Changes
```python
# Current: llamaindex_persist_dir / user_id /
# Target:  llamaindex_persist_dir / user_id / project_id /

def get_persist_dir(user_id: str, project_id: str) -> Path:
    return Path(config.LLAMAINDEX_PERSIST_DIR) / user_id / project_id
```

### Frontend Changes

#### New State in MainApp.tsx
```typescript
// Project state
const [projects, setProjects] = useState<Project[]>([]);
const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
const [selectedProject, setSelectedProject] = useState<Project | null>(null);

// Flyout state
const [isThreadsOpen, setIsThreadsOpen] = useState(false);
const [isIssuesOpen, setIsIssuesOpen] = useState(false);
```

#### New Types
```typescript
interface Project {
  id: string;           // slug: "vlt-bridge"
  name: string;         // display: "Vlt Bridge"
  description?: string;
  created_at: string;
  note_count?: number;
  thread_count?: number;
}
```

#### API Service Updates
```typescript
// All note/search/rag APIs need project_id
export async function fetchNotes(projectId: string): Promise<NoteSummary[]>
export async function searchNotes(projectId: string, query: string): Promise<SearchResult[]>
export async function fetchProjects(): Promise<Project[]>
export async function createProject(name: string): Promise<Project>
```

### vlt CLI Integration

#### Sync Flow
```
vlt CLI (local machine)
    │
    ├── vlt thread push <id> "thought"
    │       ↓
    │   ThreadSyncClient.sync_thread()
    │       ↓
    └── POST /api/threads/sync
            │
            ↓
        Backend (Document-MCP)
            │
            ├── Store in threads table
            │   (user_id, project_id, thread_id, ...)
            │
            └── Web UI sees thread
```

#### vlt.toml Integration
```toml
[project]
name = "My Project"
id = "my-project"

[oracle]
vault_url = "http://localhost:8000"   # Points to Document-MCP
# Web UI creates project with this ID
```

## Migration Strategy

### Phase 1: Database Schema (Non-Breaking)
1. Add `project_id` columns with DEFAULT 'default'
2. Backfill existing data to project_id='default'
3. Update indexes and constraints
4. All existing functionality continues working

### Phase 2: Backend Services
1. Add project_id parameter to VaultService methods
2. Add project_id parameter to IndexerService methods
3. Add project_id parameter to RAGService methods
4. Create ProjectService for CRUD

### Phase 3: API Layer
1. Add /api/projects endpoints
2. Update all note/search/rag endpoints to accept project_id
3. Add project_id to auth context (header or JWT claim)

### Phase 4: Frontend UI
1. Remove DEMO banner (or make conditional)
2. Add ProjectDropdown component
3. Add NavigationBar with new icons
4. Add ThreadsFlyout component
5. Update state management for project scoping
6. Update all API calls to include project_id

### Phase 5: vlt CLI Integration
1. Verify ThreadSyncClient works with new schema
2. Add project creation from CLI
3. Test bidirectional sync

## Success Criteria

1. User can create multiple projects
2. Each project has isolated vault, threads, RAG index
3. Project dropdown switches context cleanly
4. vlt CLI threads sync to correct project
5. No data leakage between projects
6. Backward compatible: existing single-user data migrates to 'default' project

## Out of Scope (This Phase)

1. Issue tracking integration (bd/beads CLI)
2. Project sharing between users
3. Project templates/cloning
4. Per-project settings (use user defaults)
5. Project deletion confirmation/archival

## References

- Current specs: 007-vlt-oracle, 008-thread-sync, 009-oracle-agent
- vlt CLI: packages/vlt-cli/
- UI mockup: User-provided sketch
