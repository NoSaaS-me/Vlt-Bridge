# 010: Multi-Project Tasks

## Phase 1: Database Migration

- [ ] **T1.1** Create `backend/src/models/project.py` with Project, ProjectCreate, ProjectUpdate, ProjectList Pydantic models
- [ ] **T1.2** Add migration script to `backend/src/services/database.py`:
  - Create `projects` table
  - Add `project_id` column to `note_metadata` (with DEFAULT 'default')
  - Add `project_id` column to `note_tags`
  - Add `project_id` column to `note_links`
  - Add `project_id` column to `index_health`
  - Recreate `note_fts` with `project_id`
  - Add `default_project_id` to `user_settings`
- [ ] **T1.3** Create migration runner that:
  - Backs up existing database
  - Runs schema changes
  - Creates 'default' project for all existing users
  - Migrates existing data to project_id='default'
- [ ] **T1.4** Test migration with existing data
- [ ] **T1.5** Update `DatabaseService.initialize()` to include new schema

## Phase 2: Backend Services

- [ ] **T2.1** Create `backend/src/services/project_service.py`:
  - `list_projects(user_id)` → List[Project]
  - `get_project(user_id, project_id)` → Project | None
  - `create_project(user_id, data)` → Project
  - `update_project(user_id, project_id, data)` → Project
  - `delete_project(user_id, project_id)` → bool
  - `get_or_create_default(user_id)` → Project
  - `generate_project_slug(name)` → str
- [ ] **T2.2** Update `backend/src/services/vault.py`:
  - Add `project_id` parameter to `sanitize_path()`
  - Add `project_id` parameter to `read_note()`, `write_note()`, `delete_note()`, `move_note()`, `list_notes()`
  - Update vault path construction: `vault_root / user_id / project_id / note_path`
  - Add `ensure_project_vault_exists(user_id, project_id)` helper
- [ ] **T2.3** Update `backend/src/services/indexer.py`:
  - Add `project_id` parameter to all public methods
  - Update all SQL queries to filter by `project_id`
  - Update wikilink resolution to scope within project
  - Update index_health queries to be per-project
- [ ] **T2.4** Update `backend/src/services/rag_index.py`:
  - Modify `get_persist_dir(user_id, project_id)` to include project_id
  - Update `index_vault()`, `query()`, `clear_index()` with project_id
- [ ] **T2.5** Update `backend/src/services/user_settings.py`:
  - Add `get_default_project_id(user_id)` method
  - Add `set_default_project_id(user_id, project_id)` method
- [ ] **T2.6** Write unit tests for ProjectService
- [ ] **T2.7** Write unit tests for updated VaultService
- [ ] **T2.8** Write unit tests for updated IndexerService

## Phase 3: API Layer

- [ ] **T3.1** Create `backend/src/api/routes/projects.py`:
  - `GET /api/projects` → List projects
  - `POST /api/projects` → Create project
  - `GET /api/projects/{project_id}` → Get project
  - `PUT /api/projects/{project_id}` → Update project
  - `DELETE /api/projects/{project_id}` → Delete project
  - `GET /api/projects/{project_id}/stats` → Get counts
- [ ] **T3.2** Update `backend/src/api/routes/notes.py`:
  - Add project-scoped routes: `/api/projects/{project_id}/notes/*`
  - Keep legacy routes working with project_id='default'
  - Add deprecation warning to legacy routes
- [ ] **T3.3** Update `backend/src/api/routes/search.py`:
  - Add `project_id` query parameter
  - Filter search results by project
- [ ] **T3.4** Update `backend/src/api/routes/graph.py`:
  - Add `project_id` parameter to graph endpoint
  - Filter nodes/edges by project
- [ ] **T3.5** Update `backend/src/api/routes/rag.py`:
  - Add `project_id` to RAG query endpoint
  - Scope vector search to project
- [ ] **T3.6** Update `backend/src/api/routes/index.py`:
  - Add `project_id` to rebuild endpoint
  - Add `project_id` to health endpoint
- [ ] **T3.7** Register projects router in `backend/src/api/main.py`
- [ ] **T3.8** Write integration tests for projects API
- [ ] **T3.9** Write integration tests for project-scoped notes API

## Phase 4: Frontend - Project Infrastructure

- [ ] **T4.1** Create `frontend/src/types/project.ts`:
  - Project interface
  - ProjectCreate interface
  - ProjectUpdate interface
- [ ] **T4.2** Create `frontend/src/services/projectApi.ts`:
  - `fetchProjects()`
  - `createProject(data)`
  - `updateProject(id, data)`
  - `deleteProject(id)`
  - `fetchProjectStats(id)`
- [ ] **T4.3** Create `frontend/src/contexts/ProjectContext.tsx`:
  - ProjectProvider component
  - useProjectContext hook
  - State: projects, selectedProject, isLoading
  - Persist selectedProjectId to localStorage
- [ ] **T4.4** Create `frontend/src/hooks/useProject.ts`:
  - useProjects() - fetch and cache projects
  - useCurrentProject() - get selected project
- [ ] **T4.5** Update `frontend/src/services/api.ts`:
  - Add `projectId` parameter to `fetchNotes()`
  - Add `projectId` parameter to `fetchNote()`
  - Add `projectId` parameter to `saveNote()`
  - Add `projectId` parameter to `deleteNote()`
  - Add `projectId` parameter to `searchNotes()`
  - Add `projectId` parameter to `fetchBacklinks()`
  - Add `projectId` parameter to `fetchGraph()`
  - Add `projectId` parameter to `queryRag()`
  - Add `projectId` parameter to `rebuildIndex()`
- [ ] **T4.6** Update TypeScript types for project_id in note models

## Phase 5: Frontend - UI Components

- [ ] **T5.1** Create `frontend/src/components/ProjectDropdown.tsx`:
  - Select component with project list
  - "New Project" option at bottom
  - Project name display in trigger
  - Search/filter projects (if many)
- [ ] **T5.2** Create `frontend/src/components/CreateProjectDialog.tsx`:
  - Dialog with name input
  - Optional description
  - Slug preview/edit
  - Create button
- [ ] **T5.3** Create `frontend/src/components/NavigationBar.tsx`:
  - Icon buttons: T, I, C, G, S
  - Active state highlighting
  - Tooltips with labels
  - Keyboard shortcuts (future)
- [ ] **T5.4** Create `frontend/src/components/ThreadsFlyout.tsx`:
  - Thread list for current project
  - Thread item with status badge
  - Click to select/view thread
  - Empty state for no threads
  - Close button
- [ ] **T5.5** Create `frontend/src/components/ThreadItem.tsx`:
  - Thread name and status
  - Last updated timestamp
  - Entry count badge
  - Hover/selected state

## Phase 6: MainApp Refactor

- [ ] **T6.1** Remove DEMO banner (lines 595-600 in MainApp.tsx)
  - Or make conditional on `isDemoMode`
- [ ] **T6.2** Wrap App with ProjectProvider in App.tsx
- [ ] **T6.3** Add project state to MainApp:
  - `const { projects, selectedProject, setSelectedProjectId } = useProjectContext()`
- [ ] **T6.4** Refactor header layout:
  - Left: ProjectDropdown
  - Center: NavigationBar
  - Right: ModelSelector, UserMenu
- [ ] **T6.5** Add flyout state:
  - `const [isThreadsOpen, setIsThreadsOpen] = useState(false)`
- [ ] **T6.6** Update ResizablePanelGroup for flyouts:
  - Conditionally render ThreadsFlyout panel
  - Adjust main panel size when flyout open
- [ ] **T6.7** Update all API calls to include selectedProjectId:
  - `fetchNotes(selectedProjectId)`
  - `searchNotes(selectedProjectId, query)`
  - etc.
- [ ] **T6.8** Add project change handler:
  - Clear current note selection
  - Reload notes for new project
  - Reset UI state
- [ ] **T6.9** Add keyboard shortcuts (optional):
  - `Cmd+1` through `Cmd+5` for nav icons
  - `Cmd+P` for project switcher

## Phase 7: Integration & Testing

- [ ] **T7.1** Test project creation flow end-to-end
- [ ] **T7.2** Test project switching clears/reloads state
- [ ] **T7.3** Test note CRUD within project
- [ ] **T7.4** Test search scoped to project
- [ ] **T7.5** Test threads flyout shows correct threads
- [ ] **T7.6** Test graph view shows project notes only
- [ ] **T7.7** Test Oracle chat uses project context
- [ ] **T7.8** Test RAG queries scoped to project
- [ ] **T7.9** Test vlt CLI thread sync to correct project
- [ ] **T7.10** Test migration with existing user data

## Phase 8: vlt CLI Verification

- [ ] **T8.1** Verify ThreadSyncClient sends project_id from vlt.toml
- [ ] **T8.2** Test `vlt thread push` syncs to web UI
- [ ] **T8.3** Test `vlt init -p "name"` creates project in backend
- [ ] **T8.4** Test Oracle queries from CLI use project context
- [ ] **T8.5** Document CLI ↔ Web sync flow

## Phase 9: Polish & Documentation

- [ ] **T9.1** Add loading states for project operations
- [ ] **T9.2** Add error handling for project operations
- [ ] **T9.3** Add success toasts for project CRUD
- [ ] **T9.4** Update CLAUDE.md with project architecture
- [ ] **T9.5** Add API documentation for project endpoints
- [ ] **T9.6** Create user-facing documentation for multi-project

---

## Task Dependencies

```
Phase 1 (DB) ─────────────────────────────────────┐
                                                   │
Phase 2 (Services) ────────────────────────────────┤
           │                                       │
           └─── depends on Phase 1                 │
                                                   │
Phase 3 (API) ─────────────────────────────────────┤
           │                                       │
           └─── depends on Phase 2                 │
                                                   ├──→ Phase 7 (Testing)
Phase 4 (FE Infra) ────────────────────────────────┤      │
           │                                       │      └─── depends on all
           └─── depends on Phase 3                 │
                                                   │
Phase 5 (FE Components) ───────────────────────────┤
           │                                       │
           └─── depends on Phase 4                 │
                                                   │
Phase 6 (MainApp) ─────────────────────────────────┤
           │                                       │
           └─── depends on Phase 5                 │
                                                   │
Phase 8 (vlt CLI) ─────────────────────────────────┘
           │
           └─── depends on Phase 3 (API ready)
```

---

## Priority Order

**Critical Path (must complete in order):**
1. T1.1-T1.5 (Database migration)
2. T2.1-T2.4 (Core services)
3. T3.1-T3.7 (API routes)
4. T4.1-T4.6 (Frontend infra)
5. T5.1-T5.4 (UI components)
6. T6.1-T6.8 (MainApp refactor)

**Parallel Tasks (can work on while waiting):**
- T2.6-T2.8 (Tests) - after T2.1-T2.4
- T3.8-T3.9 (Tests) - after T3.1-T3.7
- T5.5 (ThreadItem) - after T5.4
- T8.1-T8.5 (vlt CLI) - after T3.1

**Nice-to-Have (can defer):**
- T6.9 (Keyboard shortcuts)
- T9.1-T9.6 (Polish & docs)
