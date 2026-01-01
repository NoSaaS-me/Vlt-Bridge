# Feature Specification: CodeRAG Project Integration

**Feature Branch**: `011-coderag-project-init`
**Created**: 2026-01-01
**Status**: Draft
**Input**: User description: "CodeRAG project integration: Interactive init workflow that marries code indexes to projects, with visibility into indexing status in both CLI and web UI. Init should allow selecting existing project or creating new one, refuse to overwrite existing indexes, support background indexing via daemon, and provide progress tracking."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Initialize CodeRAG for a New Project (Priority: P1)

A developer wants to set up code search capabilities for their project. They run an initialization command which guides them through selecting or creating a project, then begins indexing their codebase in the background so they can continue working while the index is built.

**Why this priority**: This is the foundational workflow - without interactive initialization that properly links code indexes to projects, users cannot use CodeRAG at all.

**Independent Test**: Can be fully tested by running the init command in a directory with source code files, selecting project options, and verifying the indexing process starts. Delivers immediate value by creating a searchable code index.

**Acceptance Scenarios**:

1. **Given** a directory containing source code files, **When** user runs the init command, **Then** they are presented with options to select an existing project or create a new one
2. **Given** user selects "create new project", **When** they provide a project name, **Then** a new project is created and associated with the code index
3. **Given** user selects an existing project without a code index, **When** they confirm the selection, **Then** the code index is associated with that project
4. **Given** initialization is confirmed, **When** indexing begins, **Then** the process runs in the background and user receives a confirmation message with status check instructions

---

### User Story 2 - Monitor Indexing Progress in CLI (Priority: P2)

A developer who has started indexing wants to check how far along the process is. They run a status command that shows progress information including files processed, time elapsed, and estimated time remaining.

**Why this priority**: Users need feedback that indexing is working, especially for large codebases where indexing may take several minutes.

**Independent Test**: Can be tested by starting an indexing process and running the status command at various points. Delivers value by providing transparency into the indexing process.

**Acceptance Scenarios**:

1. **Given** an indexing process is running, **When** user runs the status command, **Then** they see current progress (files indexed, total files, percentage complete)
2. **Given** an indexing process is running, **When** user runs the status command, **Then** they see time elapsed and estimated time remaining
3. **Given** indexing has completed, **When** user runs the status command, **Then** they see completion confirmation with summary statistics

---

### User Story 3 - Monitor Indexing Progress in Web UI (Priority: P2)

A developer using the web application wants to see the status of their code index. They navigate to a settings or status page where they can view indexing progress and trigger re-indexing if needed.

**Why this priority**: Web UI users need the same visibility as CLI users, maintaining feature parity across interfaces.

**Independent Test**: Can be tested by starting indexing via CLI and viewing status in web UI, or triggering indexing from web UI and observing progress updates.

**Acceptance Scenarios**:

1. **Given** user is on the settings/status page, **When** a code index exists for the current project, **Then** they see the index status (last indexed time, chunk count, index size)
2. **Given** indexing is in progress, **When** user views the status panel, **Then** they see a progress indicator with percentage and file count
3. **Given** user has appropriate permissions, **When** they click "Re-index Code", **Then** a new indexing process is triggered and progress is displayed

---

### User Story 4 - Prevent Accidental Index Overwrites (Priority: P1)

A developer accidentally runs the init command on a project that already has a code index. The system prevents data loss by refusing to overwrite the existing index and offers alternative actions.

**Why this priority**: Data protection is critical - accidental overwrites could destroy hours of indexing work and cause confusion about index state.

**Independent Test**: Can be tested by running init on a project with an existing index and verifying the appropriate warnings and options are presented.

**Acceptance Scenarios**:

1. **Given** a project already has a code index, **When** user attempts to run init for that project, **Then** they receive a warning that an index already exists
2. **Given** warning is displayed, **When** user is presented with options, **Then** they can choose to: skip (keep existing), force re-index (with confirmation), or cancel
3. **Given** user selects force re-index, **When** they confirm the action, **Then** the existing index is replaced with a fresh index

---

### User Story 5 - Background Indexing with Daemon (Priority: P2)

A developer starts indexing and wants to close their terminal or work on other tasks. The indexing continues in the background via a daemon process, surviving terminal closure.

**Why this priority**: Large codebases require significant indexing time; blocking the terminal is poor UX for developers.

**Independent Test**: Can be tested by starting indexing, closing the terminal, and verifying via status command (in new terminal) that indexing continues.

**Acceptance Scenarios**:

1. **Given** user confirms initialization, **When** indexing begins, **Then** the process is handed off to a background daemon
2. **Given** indexing is running in daemon, **When** user closes their terminal, **Then** indexing continues uninterrupted
3. **Given** daemon is running indexing, **When** user opens a new terminal and checks status, **Then** they see accurate progress information

---

### Edge Cases

- What happens when the target directory has no indexable files? System should report "No supported files found" with guidance on supported file types.
- What happens when a project is deleted while indexing is in progress? Indexing should be cancelled and resources cleaned up.
- What happens when disk space runs out during indexing? Indexing should fail gracefully with a clear error message and partial results preserved if possible.
- What happens when the daemon process crashes mid-indexing? Next status check should detect the failure and allow user to resume or restart.
- What happens when user tries to search before indexing completes? Partial results should be returned from indexed files, with a note that indexing is in progress.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST present an interactive prompt when init is run, allowing selection of existing projects or creation of a new project
- **FR-002**: System MUST associate each code index with exactly one project (one-to-one relationship)
- **FR-003**: System MUST refuse to initialize a code index for a project that already has one, unless user explicitly forces re-indexing
- **FR-004**: System MUST run indexing as a background process that survives terminal closure
- **FR-005**: System MUST track indexing progress including: files processed, total files, percentage complete, time elapsed
- **FR-006**: System MUST provide a CLI command to check indexing status for the current project
- **FR-007**: System MUST expose indexing status via the web application settings/status interface
- **FR-008**: System MUST clean up orphaned code index data when a project is deleted
- **FR-009**: System MUST support cancellation of in-progress indexing operations
- **FR-010**: System MUST provide clear error messages when indexing fails, with actionable recovery steps
- **FR-011**: System MUST allow partial search results from already-indexed files while indexing is in progress

### Key Entities

- **Project**: The container entity that owns notes, threads, and now a code index. Each project can have at most one code index.
- **Code Index**: The collection of indexed code artifacts (chunks, embeddings, graphs) for a single project's codebase.
- **Indexing Job**: A background task that tracks the progress of code indexing, including status, progress metrics, and error state.
- **Index Status**: The current state of a code index (not_initialized, indexing, ready, failed, stale).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can complete the interactive init workflow in under 60 seconds (excluding actual indexing time)
- **SC-002**: Indexing progress updates are visible within 5 seconds of any change in both CLI and web UI
- **SC-003**: 100% of accidental overwrite attempts are blocked with clear warnings (no silent data loss)
- **SC-004**: Background indexing survives terminal closure in 100% of cases when daemon is operational
- **SC-005**: Users can retrieve partial search results within 10 seconds of first files being indexed (no waiting for full completion)
- **SC-006**: Index status is accurately reported with less than 5% deviation from actual progress
- **SC-007**: Failed indexing operations provide actionable error messages in 100% of failure cases

## Assumptions

- The daemon infrastructure already exists and can be extended to handle indexing jobs
- Projects are already implemented with CRUD operations available
- The code indexing engine (chunker, embedder, storage) already exists and works correctly
- Users have appropriate permissions to manage projects they own
- The web UI has a settings or status page where the indexing panel can be added
- File system watching for incremental updates is out of scope for this feature (manual re-index only)

## Scope Boundaries

**In Scope**:
- Interactive CLI init workflow with project selection/creation
- Background indexing via daemon
- Progress tracking in CLI and web UI
- Protection against accidental overwrites
- Cleanup on project deletion

**Out of Scope**:
- Automatic file watching and incremental re-indexing (future feature)
- Multi-project shared code indexes
- Code index migration between projects
- Scheduled/periodic re-indexing
