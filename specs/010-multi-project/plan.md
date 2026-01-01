# 010: Multi-Project Implementation Plan

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React 19)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  App.tsx                                                                    │
│    └── MainApp.tsx                                                          │
│          ├── ProjectDropdown.tsx (NEW)                                      │
│          ├── NavigationBar.tsx (NEW)                                        │
│          ├── DirectoryTree.tsx (project-scoped)                             │
│          ├── NoteViewer.tsx                                                 │
│          ├── ChatPanel.tsx                                                  │
│          ├── ThreadsFlyout.tsx (NEW)                                        │
│          └── GraphView.tsx                                                  │
│                                                                             │
│  State: projects[], selectedProjectId, isThreadsOpen, etc.                  │
│  Context: ProjectContext (NEW) - provides current project to children       │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ HTTP API
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BACKEND (FastAPI)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  API Routes                                                                 │
│    ├── /api/projects/*      (NEW: CRUD)                                     │
│    ├── /api/notes/*         (modified: project_id param)                    │
│    ├── /api/search/*        (modified: project_id param)                    │
│    ├── /api/oracle/*        (existing: already has project_id)              │
│    └── /api/threads/*       (existing: already has project_id)              │
│                                                                             │
│  Services                                                                   │
│    ├── ProjectService (NEW)                                                 │
│    ├── VaultService (modified: project_id)                                  │
│    ├── IndexerService (modified: project_id)                                │
│    └── RAGIndexService (modified: project_id)                               │
│                                                                             │
│  Database: SQLite with project-scoped tables                                │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ Thread Sync
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            VLT CLI                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Already project-scoped:                                                    │
│    - threads (project_id FK)                                                │
│    - coderag (project_id FK)                                                │
│    - oracle conversations                                                   │
│                                                                             │
│  ThreadSyncClient → POST /api/threads/sync → Backend                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Database Migration (Backend)

**Duration estimate**: Foundation work

**Files to modify:**
- `backend/src/services/database.py`
- `backend/src/models/project.py` (NEW)

**Tasks:**
1. Create migration script for schema changes
2. Add `projects` table
3. Add `project_id` to note_metadata, note_tags, note_links, index_health
4. Rebuild note_fts with project_id column
5. Add `default_project_id` to user_settings
6. Create ProjectService for CRUD operations

**Migration approach:**
- All existing data gets `project_id = 'default'`
- Auto-create 'default' project for existing users
- No breaking changes to existing functionality

---

### Phase 2: Backend Services (Backend)

**Files to modify:**
- `backend/src/services/vault.py`
- `backend/src/services/indexer.py`
- `backend/src/services/rag_index.py`
- `backend/src/services/project_service.py` (NEW)

**VaultService changes:**
```python
# Before
def sanitize_path(user_id: str, vault_root: Path, note_path: str) -> Path:
    return vault_root / user_id / note_path

# After
def sanitize_path(user_id: str, project_id: str, vault_root: Path, note_path: str) -> Path:
    return vault_root / user_id / project_id / note_path
```

**IndexerService changes:**
- Add `project_id` parameter to all public methods
- Update all SQL queries to include `AND project_id = ?`
- Wikilink resolution scoped to project only

**RAGIndexService changes:**
```python
# Before
def get_persist_dir(user_id: str) -> Path:
    return Path(config.LLAMAINDEX_PERSIST_DIR) / user_id

# After
def get_persist_dir(user_id: str, project_id: str) -> Path:
    return Path(config.LLAMAINDEX_PERSIST_DIR) / user_id / project_id
```

**New ProjectService:**
```python
class ProjectService:
    def list_projects(self, user_id: str) -> list[Project]
    def get_project(self, user_id: str, project_id: str) -> Project | None
    def create_project(self, user_id: str, data: ProjectCreate) -> Project
    def update_project(self, user_id: str, project_id: str, data: ProjectUpdate) -> Project
    def delete_project(self, user_id: str, project_id: str) -> bool
    def get_or_create_default(self, user_id: str) -> Project
```

---

### Phase 3: API Layer (Backend)

**Files to modify:**
- `backend/src/api/routes/projects.py` (NEW)
- `backend/src/api/routes/notes.py`
- `backend/src/api/routes/search.py`
- `backend/src/api/routes/graph.py`
- `backend/src/api/routes/rag.py`
- `backend/src/api/routes/index.py`

**New endpoints:**
```
GET    /api/projects                 → List user's projects
POST   /api/projects                 → Create project
GET    /api/projects/{id}            → Get project details
PUT    /api/projects/{id}            → Update project
DELETE /api/projects/{id}            → Delete project
GET    /api/projects/{id}/stats      → Get note/thread counts
```

**Modified endpoints (add project_id path param):**
```
GET    /api/projects/{project_id}/notes
POST   /api/projects/{project_id}/notes
GET    /api/projects/{project_id}/notes/{path}
PUT    /api/projects/{project_id}/notes/{path}
DELETE /api/projects/{project_id}/notes/{path}
GET    /api/projects/{project_id}/search
GET    /api/projects/{project_id}/graph
POST   /api/projects/{project_id}/rag/query
POST   /api/projects/{project_id}/index/rebuild
```

**Backward compatibility:**
- Keep existing `/api/notes/*` routes working with `project_id='default'`
- Add deprecation warnings for old routes

---

### Phase 4: Frontend - Project Infrastructure

**Files to create:**
- `frontend/src/types/project.ts`
- `frontend/src/services/projectApi.ts`
- `frontend/src/contexts/ProjectContext.tsx`
- `frontend/src/hooks/useProject.ts`

**Project types:**
```typescript
// types/project.ts
export interface Project {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  note_count?: number;
  thread_count?: number;
}
```

**API service:**
```typescript
// services/projectApi.ts
export async function fetchProjects(): Promise<Project[]>
export async function createProject(data: ProjectCreate): Promise<Project>
export async function updateProject(id: string, data: ProjectUpdate): Promise<Project>
export async function deleteProject(id: string): Promise<void>
```

**Project context:**
```typescript
// contexts/ProjectContext.tsx
interface ProjectContextValue {
  projects: Project[];
  selectedProject: Project | null;
  selectedProjectId: string | null;
  setSelectedProjectId: (id: string) => void;
  isLoading: boolean;
  refreshProjects: () => Promise<void>;
}

export const ProjectProvider: React.FC<{children: ReactNode}>
export const useProjectContext: () => ProjectContextValue
```

---

### Phase 5: Frontend - UI Components

**Files to create:**
- `frontend/src/components/ProjectDropdown.tsx`
- `frontend/src/components/NavigationBar.tsx`
- `frontend/src/components/ThreadsFlyout.tsx`
- `frontend/src/components/CreateProjectDialog.tsx`

**Files to modify:**
- `frontend/src/pages/MainApp.tsx` (major refactor)
- `frontend/src/services/api.ts` (add project_id to all calls)

#### ProjectDropdown Component

```typescript
// components/ProjectDropdown.tsx
interface ProjectDropdownProps {
  projects: Project[];
  selectedProject: Project | null;
  onSelectProject: (projectId: string) => void;
  onCreateProject: () => void;
}

export function ProjectDropdown({...}: ProjectDropdownProps) {
  return (
    <Select value={selectedProject?.id} onValueChange={onSelectProject}>
      <SelectTrigger className="w-48">
        <SelectValue placeholder="Select project" />
      </SelectTrigger>
      <SelectContent>
        {projects.map(p => (
          <SelectItem key={p.id} value={p.id}>
            {p.name}
          </SelectItem>
        ))}
        <SelectSeparator />
        <Button variant="ghost" onClick={onCreateProject}>
          <Plus className="h-4 w-4 mr-2" /> New Project
        </Button>
      </SelectContent>
    </Select>
  );
}
```

#### NavigationBar Component

```typescript
// components/NavigationBar.tsx
interface NavigationBarProps {
  onToggleThreads: () => void;
  onToggleChat: () => void;
  onToggleGraph: () => void;
  onNavigateSettings: () => void;
  isThreadsOpen: boolean;
  isChatOpen: boolean;
  isGraphView: boolean;
}

export function NavigationBar({...}: NavigationBarProps) {
  return (
    <div className="flex items-center gap-2">
      <TooltipButton
        icon={<List className="h-4 w-4" />}
        tooltip="Threads (T)"
        isActive={isThreadsOpen}
        onClick={onToggleThreads}
      />
      <TooltipButton
        icon={<AlertCircle className="h-4 w-4" />}
        tooltip="Issues (I) - Coming Soon"
        disabled
      />
      <TooltipButton
        icon={<MessageCircle className="h-4 w-4" />}
        tooltip="Chat (C)"
        isActive={isChatOpen}
        onClick={onToggleChat}
      />
      <TooltipButton
        icon={<Network className="h-4 w-4" />}
        tooltip="Graph (G)"
        isActive={isGraphView}
        onClick={onToggleGraph}
      />
      <TooltipButton
        icon={<Settings className="h-4 w-4" />}
        tooltip="Settings (S)"
        onClick={onNavigateSettings}
      />
    </div>
  );
}
```

#### ThreadsFlyout Component

```typescript
// components/ThreadsFlyout.tsx
interface ThreadsFlyoutProps {
  projectId: string;
  onSelectThread: (threadId: string) => void;
  onClose: () => void;
}

export function ThreadsFlyout({ projectId, onSelectThread, onClose }: ThreadsFlyoutProps) {
  const [threads, setThreads] = useState<Thread[]>([]);

  useEffect(() => {
    fetchThreads(projectId).then(setThreads);
  }, [projectId]);

  return (
    <div className="w-80 h-full border-l bg-background">
      <div className="flex items-center justify-between p-4 border-b">
        <h3 className="font-semibold">Threads</h3>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>
      <ScrollArea className="h-[calc(100%-56px)]">
        {threads.map(thread => (
          <ThreadItem
            key={thread.id}
            thread={thread}
            onClick={() => onSelectThread(thread.id)}
          />
        ))}
      </ScrollArea>
    </div>
  );
}
```

---

### Phase 6: MainApp.tsx Refactor

**Key changes:**

1. **Remove DEMO banner** (lines 595-600)
```typescript
// Remove this block entirely or make conditional
{isDemoMode && (
  <Alert variant="destructive">
    <AlertDescription>DEMO ONLY...</AlertDescription>
  </Alert>
)}
```

2. **Add ProjectProvider wrapper**
```typescript
// App.tsx
<ProjectProvider>
  <MainApp />
</ProjectProvider>
```

3. **New header layout**
```typescript
<header className="flex items-center justify-between px-4 py-3 border-b">
  {/* Left: Project dropdown */}
  <ProjectDropdown
    projects={projects}
    selectedProject={selectedProject}
    onSelectProject={handleSelectProject}
    onCreateProject={() => setIsCreateProjectOpen(true)}
  />

  {/* Center: Navigation icons */}
  <NavigationBar
    onToggleThreads={() => setIsThreadsOpen(prev => !prev)}
    onToggleChat={handleToggleChat}
    onToggleGraph={() => setIsGraphView(prev => !prev)}
    onNavigateSettings={() => navigate('/settings')}
    isThreadsOpen={isThreadsOpen}
    isChatOpen={isChatOpen}
    isGraphView={isGraphView}
  />

  {/* Right: Model selector, user menu */}
  <div className="flex items-center gap-2">
    <ModelSelector />
    <UserMenu />
  </div>
</header>
```

4. **Panel layout with flyouts**
```typescript
<ResizablePanelGroup direction="horizontal">
  {/* Left: Sidebar */}
  <ResizablePanel defaultSize={20} minSize={15} maxSize={30}>
    <Sidebar ... />
  </ResizablePanel>

  <ResizableHandle />

  {/* Center: Main content */}
  <ResizablePanel defaultSize={isThreadsOpen ? 55 : 80}>
    <MainContent ... />
  </ResizablePanel>

  {/* Right: Flyouts (conditional) */}
  {isThreadsOpen && (
    <>
      <ResizableHandle />
      <ResizablePanel defaultSize={25} minSize={20} maxSize={35}>
        <ThreadsFlyout
          projectId={selectedProjectId}
          onSelectThread={handleSelectThread}
          onClose={() => setIsThreadsOpen(false)}
        />
      </ResizablePanel>
    </>
  )}
</ResizablePanelGroup>
```

5. **Update API calls to include project_id**
```typescript
// Before
const notes = await fetchNotes();

// After
const notes = await fetchNotes(selectedProjectId);
```

---

### Phase 7: API Service Updates

**Files to modify:**
- `frontend/src/services/api.ts`

**Pattern:**
```typescript
// Before
export async function fetchNotes(): Promise<NoteSummary[]> {
  return apiFetch('/api/notes');
}

// After
export async function fetchNotes(projectId: string): Promise<NoteSummary[]> {
  return apiFetch(`/api/projects/${projectId}/notes`);
}

// Or with query param
export async function fetchNotes(projectId: string): Promise<NoteSummary[]> {
  return apiFetch(`/api/notes?project_id=${projectId}`);
}
```

**All updated functions:**
- `fetchNotes(projectId)`
- `fetchNote(projectId, path)`
- `saveNote(projectId, path, content, metadata)`
- `deleteNote(projectId, path)`
- `searchNotes(projectId, query)`
- `fetchBacklinks(projectId, path)`
- `fetchGraph(projectId)`
- `queryRag(projectId, query)`
- `rebuildIndex(projectId)`

---

### Phase 8: vlt CLI Integration

**Files to verify/modify:**
- `packages/vlt-cli/src/vlt/core/sync.py`
- `backend/src/api/routes/threads.py`

**Verification tasks:**
1. Confirm ThreadSyncClient sends correct project_id
2. Test thread sync from CLI to backend
3. Verify project creation from CLI (`vlt init -p "name"`)
4. Test Oracle queries use correct project context

**CLI → Web sync flow:**
```
vlt thread push oracle-agent "Implemented X feature"
    ↓
ThreadSyncClient.sync_thread()
    ↓
POST /api/threads/sync
{
  "thread_id": "oracle-agent",
  "project_id": "vlt-bridge",  # From vlt.toml
  "name": "Oracle Agent Development",
  "entries": [...]
}
    ↓
Backend stores in threads table
    ↓
Web UI shows in ThreadsFlyout
```

---

## File Change Summary

### New Files
```
backend/src/models/project.py
backend/src/services/project_service.py
backend/src/api/routes/projects.py
frontend/src/types/project.ts
frontend/src/services/projectApi.ts
frontend/src/contexts/ProjectContext.tsx
frontend/src/hooks/useProject.ts
frontend/src/components/ProjectDropdown.tsx
frontend/src/components/NavigationBar.tsx
frontend/src/components/ThreadsFlyout.tsx
frontend/src/components/CreateProjectDialog.tsx
```

### Modified Files
```
backend/src/services/database.py       # Migration script
backend/src/services/vault.py          # Add project_id param
backend/src/services/indexer.py        # Add project_id param
backend/src/services/rag_index.py      # Add project_id param
backend/src/api/routes/notes.py        # Add project_id param
backend/src/api/routes/search.py       # Add project_id param
backend/src/api/routes/graph.py        # Add project_id param
backend/src/api/routes/rag.py          # Add project_id param
backend/src/api/routes/index.py        # Add project_id param
backend/src/api/main.py                # Register projects router
frontend/src/services/api.ts           # Add project_id to all calls
frontend/src/pages/MainApp.tsx         # Major refactor
frontend/src/App.tsx                   # Add ProjectProvider
```

---

## Testing Strategy

### Unit Tests
- ProjectService CRUD operations
- VaultService with project isolation
- IndexerService with project scoping

### Integration Tests
- Create project → Create note → Search note → Delete note → Delete project
- Project isolation: Notes in project A not visible in project B
- Default project migration for existing users

### E2E Tests
- Project dropdown selection
- Thread flyout with project threads
- vlt CLI sync to web UI

---

## Rollback Plan

If issues arise:
1. Database migration is reversible (keep backup)
2. Old API routes still work with `project_id='default'`
3. Frontend can fall back to single-project mode
4. vlt CLI continues working (already project-scoped)

---

## Dependencies

- React 19 (existing)
- shadcn/ui Select component (existing)
- Lucide icons (existing)
- FastAPI (existing)
- SQLite (existing)

No new external dependencies required.
