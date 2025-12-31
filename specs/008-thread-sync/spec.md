# Feature Specification: Thread Sync from CLI to Server

**Feature Branch**: `008-thread-sync`
**Created**: 2025-12-31
**Status**: Draft
**Input**: User description: "Thread sync from vlt CLI to backend server for Oracle queries"

## Problem Statement

The vlt Oracle feature is currently broken because the backend attempts to run `vlt oracle` as a subprocess. This approach fails because:

1. The backend server has no `vlt.toml` configuration (it lives where the AI agent works)
2. The backend has no thread history (threads are stored locally in the CLI's SQLite database)
3. The subprocess has no context about the project the agent is working on

The correct architecture requires the vlt CLI (running on the agent's machine) to **sync thread data TO the server**, so the backend can query its own local copy of synced data.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - AI Agent Syncs Work Context (Priority: P1)

An AI coding agent uses `vlt thread push` to record decisions, discoveries, and progress during development. After pushing a thread entry, the CLI automatically syncs this information to the remote server so that:
- The agent's work history is preserved even if the local machine is reset
- Other tools (like the web UI) can access the agent's reasoning and progress
- The Oracle can reference thread history when answering questions

**Why this priority**: Without thread sync, the entire Oracle feature cannot function. This is the foundational capability that enables all other features.

**Independent Test**: Can be tested by running `vlt thread push "test message"` and verifying the message appears in the server's thread storage via API.

**Acceptance Scenarios**:

1. **Given** a configured vault_url in vlt.toml, **When** the agent runs `vlt thread push "Discovered auth bug in login.py"`, **Then** the thread entry is stored both locally AND synced to the remote server within 5 seconds.

2. **Given** the remote server is temporarily unavailable, **When** the agent pushes a thread entry, **Then** the entry is stored locally and queued for sync when connectivity is restored.

3. **Given** an existing thread with 10 entries, **When** a new entry is pushed, **Then** only the new entry is synced (not the entire thread history).

---

### User Story 2 - Web UI User Views Agent Work History (Priority: P2)

A developer opens the Vlt-Bridge web UI to review what an AI agent has been working on. They can see the thread history, including timestamped entries showing the agent's reasoning, decisions, and discoveries.

**Why this priority**: Viewing thread history provides visibility into AI agent work, building trust and enabling oversight. This depends on P1 (sync) being complete.

**Independent Test**: Can be tested by logging into the web UI and viewing the threads list, verifying thread entries appear with correct timestamps and content.

**Acceptance Scenarios**:

1. **Given** a user is authenticated in the web UI, **When** they navigate to the threads view, **Then** they see all threads synced from CLI sessions associated with their account.

2. **Given** a thread has 50 entries, **When** the user opens the thread, **Then** entries are displayed in chronological order with timestamps.

3. **Given** a thread entry was pushed 30 seconds ago, **When** the user refreshes the threads view, **Then** the new entry appears.

---

### User Story 3 - Oracle Answers Questions Using Thread Context (Priority: P3)

A user asks the Oracle a question like "What did we discover about the authentication bug?" The Oracle searches synced thread history alongside vault documents and code to provide a contextual answer with citations.

**Why this priority**: This is the end goal that enables intelligent code assistance, but depends on P1 (sync) and benefits from P2 (visibility).

**Independent Test**: Can be tested by asking the Oracle a question about a previously synced thread topic and verifying the answer references thread content.

**Acceptance Scenarios**:

1. **Given** a thread containing "Discovered rate limiting issue in API gateway", **When** the user asks "What issues did we find in the API?", **Then** the Oracle response references the thread entry with source citation.

2. **Given** multiple threads across different projects, **When** the user asks a question, **Then** the Oracle searches only threads relevant to the current project context.

3. **Given** a question with no relevant thread matches, **When** the Oracle responds, **Then** it gracefully indicates no thread history is relevant while still searching other sources (vault, code).

---

### Edge Cases

- What happens when the CLI pushes a thread but the server rejects it (invalid token, quota exceeded)?
- How does the system handle duplicate sync attempts (idempotency)?
- What happens when a thread is deleted locally but already synced to server?
- How does sync behave when the user's token expires mid-session?
- What happens if the same user runs multiple CLI sessions simultaneously pushing to the same thread?

## Requirements *(mandatory)*

### Functional Requirements

**CLI Sync Mechanism:**
- **FR-001**: CLI MUST sync thread entries to the remote server after each `vlt thread push` command
- **FR-002**: CLI MUST read the server URL from `vault_url` in vlt.toml configuration
- **FR-003**: CLI MUST authenticate sync requests using a stored API token
- **FR-004**: CLI MUST queue failed sync attempts and retry when connectivity is restored
- **FR-005**: CLI MUST support incremental sync (only new entries, not full history)

**Backend Storage:**
- **FR-006**: Backend MUST store synced threads in persistent storage (database)
- **FR-007**: Backend MUST associate threads with the authenticated user
- **FR-008**: Backend MUST expose thread CRUD operations via REST API
- **FR-009**: Backend MUST validate incoming thread data (schema validation, size limits)
- **FR-010**: Backend MUST handle concurrent writes to the same thread safely

**Backend Oracle Integration:**
- **FR-011**: Backend Oracle MUST query synced threads directly from local storage (no subprocess)
- **FR-012**: Backend Oracle MUST include thread content in context assembly for LLM queries
- **FR-013**: Backend Oracle MUST provide thread source citations in responses

**Web UI:**
- **FR-014**: Web UI MUST display list of synced threads
- **FR-015**: Web UI MUST display thread entries with timestamps
- **FR-016**: Web UI MUST support filtering threads by project

### Key Entities

- **Thread**: A named sequence of entries representing a conversation or work session. Key attributes: id, name, project, user_id, created_at, updated_at.
- **ThreadEntry**: A single message within a thread. Key attributes: id, thread_id, role (user/assistant/system), content, timestamp.
- **SyncStatus**: Tracks sync state for each thread entry. Key attributes: entry_id, synced_at, sync_attempts, last_error.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Thread entries sync from CLI to server within 5 seconds under normal network conditions
- **SC-002**: Failed sync attempts are automatically retried and succeed within 1 minute of connectivity restoration
- **SC-003**: Users can view synced thread history in the web UI within 2 seconds of page load
- **SC-004**: Oracle queries that reference thread content return relevant citations 90% of the time when matching content exists
- **SC-005**: System handles 100 concurrent CLI sessions syncing threads without data loss or corruption
- **SC-006**: Synced threads persist across server restarts without data loss

## Assumptions

- Users will have valid API tokens configured in the CLI (obtained via existing authentication flow)
- The backend database (SQLite) can handle the thread storage volume for the expected user base
- Network latency between CLI and server is typically under 500ms
- Thread entries are typically under 10KB each (text content)
- The CLI already has a local thread storage mechanism that this feature extends

## Out of Scope

- Real-time collaborative thread editing (multiple users editing same thread)
- Thread encryption at rest (beyond standard database security)
- Thread export/import functionality
- Thread versioning or edit history
- Conflict resolution for simultaneous edits (last-write-wins is acceptable)
