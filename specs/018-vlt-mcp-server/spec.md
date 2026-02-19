# Feature Specification: Vlt Unified MCP Server

**Feature Branch**: `018-vlt-mcp-server`
**Created**: 2026-02-18
**Status**: Draft
**Input**: User description: "expose vlt threads, vault notes, CodeRAG code indexing, and oracle through a single auto-starting STDIO MCP server with oracle toggle in web UI settings"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Thread Memory via MCP (Priority: P1)

An AI agent configured with vlt as a global MCP server can persist and retrieve reasoning chains without shelling out to the CLI. The agent creates a thread, appends thoughts throughout a work session, reads the compressed state to resume context, and searches across past reasoning when encountering a familiar problem.

**Why this priority**: Thread memory is the highest-value capability in vlt and currently the most fragmented — agents must invoke CLI subprocesses (~200–500ms each), which is slow, brittle, and unavailable in sandboxed environments. Moving this to direct MCP tools is the single biggest improvement in agent ergonomics.

**Independent Test**: Configure vlt as a global MCP, ask an agent to log a decision and retrieve it via MCP tools only — no CLI invocation. Agent should complete a create → push → read → seek round-trip and retrieve the thought accurately.

**Acceptance Scenarios**:

1. **Given** vlt is configured as a global MCP server, **When** an agent calls `vlt_thread_create` with a project name and initial thought, **Then** the thread is created and persisted, and the agent receives a thread ID it can reference in future calls.

2. **Given** an existing thread, **When** an agent calls `vlt_thread_push` with a thought and optional author attribution, **Then** the thought is appended to the thread and the call completes in under 50 milliseconds.

3. **Given** an existing thread with multiple nodes, **When** an agent calls `vlt_thread_read`, **Then** the agent receives the compressed state summary plus recent nodes, sufficient to restore reasoning context without re-reading every raw entry.

4. **Given** past threads across multiple projects, **When** an agent calls `vlt_thread_seek` with a natural language query, **Then** semantically relevant nodes are returned with thread attribution, allowing the agent to find past solutions without knowing which thread to look in.

5. **Given** a thread push call that returns success, **When** the MCP server subsequently crashes, **Then** the pushed thought is still present on the next read — the response MUST NOT be returned before the data is durable.

---

### User Story 2 — Code Search and Repo Map via MCP (Priority: P2)

An AI agent dropped into an unfamiliar codebase can initialize a code index via MCP, check indexing progress, and then issue semantic searches and request a repository structure overview — all without CLI access or manual setup.

**Why this priority**: CodeRAG is already built and working via CLI. Exposing it through MCP makes it accessible to agents that don't have shell access and removes the overhead of subprocess coordination. High-value for any agent doing code understanding work.

**Independent Test**: On a fresh project, call the code initialization tool via MCP, poll status until complete, then run a search query. Verify results contain relevant code chunks with file paths and line numbers.

**Acceptance Scenarios**:

1. **Given** a project directory path, **When** an agent calls the code initialization tool, **Then** indexing begins and the agent receives status information indicating the job has started.

2. **Given** a completed code index, **When** an agent queries for a concept in natural language, **Then** the agent receives ranked code chunks with file path, line range, symbol name, and relevance score.

3. **Given** a completed code index, **When** an agent requests a repository map, **Then** the agent receives a structured summary of files, classes, and functions within a specified token budget, ordered by structural importance.

4. **Given** a completed code index, **When** an agent looks up a specific symbol name, **Then** the agent receives the definition location(s) with file path, line number, and kind (function, class, method).

5. **Given** a project with no code index, **When** an agent calls a code search tool, **Then** the agent receives a clear message that indexing has not been run, with guidance on how to start it — not an opaque failure.

6. **Given** a code initialization call while indexing is already running, **When** the agent calls init again without forcing a re-index, **Then** no duplicate job is created — the agent receives the current status of the running job.

---

### User Story 3 — Global MCP Auto-Start (Priority: P3)

An agent or user adds vlt to their global AI assistant MCP configuration once. From that point forward, the MCP server starts automatically whenever the AI assistant launches — no manual daemon startup, no port management, no checking if a server is running.

**Why this priority**: Without auto-start, agents find vlt configured but the server not running and receive cryptic errors. The design must make global config fire-and-forget.

**Independent Test**: Add vlt to global MCP config, restart the AI assistant, and immediately invoke a thread push tool call — no manual server startup required.

**Acceptance Scenarios**:

1. **Given** vlt is listed in the AI assistant's global MCP config, **When** the AI assistant launches, **Then** the vlt MCP server starts automatically without any additional user action.

2. **Given** the vlt MCP server is running, **When** the AI assistant session ends, **Then** the server terminates cleanly without leaving orphaned processes.

3. **Given** a cold start with no prior state, **When** the vlt MCP server is spawned, **Then** the first tool call is responsive within 2 seconds of server spawn.

4. **Given** multiple concurrent AI assistant sessions, **When** each session starts its own vlt server, **Then** sessions do not conflict — data is consistent across sessions.

---

### User Story 4 — Oracle Toggle in Web Settings (Priority: P4)

A user can enable or disable the oracle capability from the web UI settings page. When disabled, agents that call oracle tools receive a clear message explaining the tool exists but is currently disabled, rather than an opaque failure. When re-enabled, oracle tools become available on the next session.

**Why this priority**: The oracle requires external API credentials and has cost implications. Users need clear, GUI-accessible control over whether agents can invoke it, without requiring manual config file edits.

**Independent Test**: Disable oracle in settings, then verify an MCP oracle tool call returns a descriptive disabled message. Re-enable, restart MCP session, verify oracle calls succeed.

**Acceptance Scenarios**:

1. **Given** the settings page, **When** a user navigates to the Oracle section, **Then** they see the current oracle status (enabled/disabled) and a toggle control.

2. **Given** oracle is disabled in settings, **When** an agent calls an oracle tool via MCP, **Then** the agent receives a response that clearly states oracle is disabled and indicates how it can be re-enabled — not an empty error or stack trace.

3. **Given** oracle is enabled but not configured (no API credentials), **When** an agent calls the oracle status tool, **Then** the response distinguishes between "disabled by user" and "enabled but missing credentials" with specific guidance for each.

4. **Given** oracle is re-enabled in settings, **When** the agent starts a new MCP session, **Then** oracle tools function normally.

---

### User Story 5 — Vault Notes via Unified MCP (Priority: P5)

An agent that wants to write documentation or read notes can do so through the same vlt MCP server, using consistent tool naming and response patterns, rather than switching between a separate vault MCP server and the vlt server.

**Why this priority**: A single MCP server with a coherent surface is easier for agents to reason about than two separate servers with different naming conventions and connection modes.

**Independent Test**: Write a note, read it back, and search for it by content — all via vlt MCP tools without explicitly starting the backend server.

**Acceptance Scenarios**:

1. **Given** the vlt MCP server is running, **When** an agent calls the note write tool with a path and content, **Then** the note is saved and available for reading and search.

2. **Given** an existing note with wikilinks to other notes, **When** an agent calls the backlinks tool for that note, **Then** the agent receives the list of notes that reference it.

3. **Given** the vault backend is unreachable, **When** an agent calls a vault tool, **Then** the agent receives a clear message indicating the document server is unreachable with actionable guidance — not a silent failure.

---

### Edge Cases

- What happens when `vlt_thread_push` is called with an invalid or nonexistent thread ID?
- How does code search behave when the index is mid-rebuild (stale or partial data available)?
- What happens when two agents push to the same thread simultaneously?
- How does the oracle tool behave when the underlying model API is rate-limited or down?
- What happens when code initialization is called for a path that does not exist on disk?
- What happens when the oracle toggle is changed mid-session — does the current session see the change?
- What does `vlt_project_detect` return when called from a directory with no project config file?

---

## Requirements *(mandatory)*

### Functional Requirements

**Thread Memory**

- **FR-001**: The system MUST expose tools for creating, appending to, reading, and searching reasoning threads via MCP, with no dependency on CLI subprocess calls.
- **FR-002**: Thread append operations MUST complete and confirm durability within 50 milliseconds under normal conditions.
- **FR-003**: Thread read MUST return both a compressed state summary and the N most recent nodes, sufficient for an agent to resume context.
- **FR-004**: Thread search MUST support natural language queries and return results ranked by semantic relevance, optionally scoped by project.
- **FR-005**: Thread operations MUST support author attribution — callers can tag thoughts with a name for multi-agent traceability.
- **FR-006**: Thread data MUST be durably persisted — a server crash after a successful push response MUST NOT result in data loss.

**Code Intelligence**

- **FR-007**: The system MUST expose a tool to initialize or re-initialize a code index for a given project path.
- **FR-008**: Code initialization MUST be idempotent — calling it multiple times without a force flag MUST NOT duplicate work or corrupt the existing index.
- **FR-009**: The system MUST expose a tool to query the current indexing status, returning completion state, file counts, and any error conditions.
- **FR-010**: The system MUST expose a code search tool supporting semantic and keyword queries with optional filtering by language and file pattern.
- **FR-011**: Code search results MUST include file path, line range, symbol name, and relevance score at minimum.
- **FR-012**: The system MUST expose a repository map tool that returns a token-budgeted, importance-ordered overview of the codebase structure.
- **FR-013**: The system MUST expose a symbol lookup tool that returns definition location(s) for a given symbol name.
- **FR-014**: All code tools MUST return actionable guidance when called before indexing has been initialized.

**Oracle**

- **FR-015**: The system MUST expose oracle query and status tools via MCP.
- **FR-016**: Oracle tools MUST always be registered and discoverable by agents regardless of the enabled/disabled state.
- **FR-017**: When the oracle is disabled, oracle tool calls MUST return a structured response indicating the disabled state and how to re-enable it — not an opaque error.
- **FR-018**: The oracle status tool MUST distinguish between three states: disabled by user, enabled but misconfigured (missing credentials), and enabled and operational.

**Oracle Toggle**

- **FR-019**: The web UI settings page MUST include an oracle enable/disable toggle visible to authenticated users.
- **FR-020**: The toggle state MUST persist across sessions and server restarts.
- **FR-021**: Toggle changes MUST take effect for new MCP sessions without requiring a system restart.

**Auto-Start**

- **FR-022**: The vlt MCP server MUST function as a STDIO server that an AI assistant can spawn automatically from a single global MCP config entry.
- **FR-023**: The server MUST be installable as a named command available in PATH after installing the vlt-cli package, requiring no additional setup.
- **FR-024**: Server startup MUST NOT require the background indexing daemon — thread, vault, and meta tools MUST work without it.
- **FR-025**: The server MUST complete its own initialization (profile loading, storage connections) without requiring external setup steps beyond initial package installation.

**Vault Notes**

- **FR-026**: The system MUST expose vault note tools (write, read, search, list, backlinks) through the unified MCP server.
- **FR-027**: Vault tool responses MUST follow the same structural conventions as thread and code tools (consistent error format, consistent field naming).
- **FR-028**: When vault tools are unavailable (document server unreachable), the response MUST describe the failure with actionable guidance — not an unhandled exception.

**General**

- **FR-029**: All tools MUST follow a consistent naming convention that makes their capability domain obvious to a model reading the tool list.
- **FR-030**: All tools MUST return structured, parseable responses — no free-form text-only returns on success paths.
- **FR-031**: The system MUST expose a health/status tool that returns what capabilities are initialized, what projects exist, and what is available in the current session.
- **FR-032**: The system MUST expose a project detection tool that infers the current project from the working directory, eliminating the need for agents to pass explicit project IDs on every call.

### Key Entities

- **Thread**: A named, persistent reasoning chain belonging to a project. Has a state (compressed summary) and an ordered log of nodes.
- **Node**: An atomic entry in a thread — a thought, decision, or observation. Has content, author attribution, timestamp, and a sequence ID.
- **Project**: A named workspace grouping threads and code indexes. Auto-detected from a project config file when possible.
- **Code Index**: A searchable index of a codebase, associated with a project. Has a status lifecycle (pending → running → completed / failed), file/chunk/symbol counts, and a last-indexed timestamp.
- **Oracle Toggle**: A per-profile preference controlling whether oracle tools are active. Persisted independently of runtime state.
- **MCP Session**: The lifetime of a single vlt MCP server process, spawned by the AI assistant and terminated when the session ends.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Agents can log a thought via MCP in under 50ms — eliminating the 200–500ms CLI subprocess overhead present today.
- **SC-002**: Agents can complete a full thread round-trip (create → push → read → seek) using only MCP tool calls, with zero CLI invocations required.
- **SC-003**: The MCP server is responsive to its first tool call within 2 seconds of being spawned from a cold start.
- **SC-004**: A code index initialized via MCP tool returns searchable results — no manual CLI steps required to make it searchable.
- **SC-005**: An agent with no prior knowledge of a codebase can obtain a useful repository overview via a single MCP tool call, within a caller-specified token budget.
- **SC-006**: A user can toggle oracle access on or off from the settings page in under 3 clicks, with the change reflected in new agent sessions without editing any config files.
- **SC-007**: When oracle is disabled, agents receive a response containing the word "disabled" and actionable re-enable instructions — not a stack trace, empty response, or generic error.
- **SC-008**: All MCP tools across all capability groups (thread, code, vault, oracle, meta) are discoverable from a single MCP server config entry.
- **SC-009**: Concurrent use by multiple agent sessions does not result in data corruption or cross-session data leakage.
- **SC-010**: After the one-time global MCP config is set, no further manual steps (daemon start, port management, environment setup) are required for agents to use vlt tools in any subsequent session.

---

## Assumptions

- The vlt-cli package is installed and the MCP server command is available in PATH. First-time package installation is out of scope.
- The user has completed initial profile setup (API key configuration). Profile bootstrap is out of scope.
- Vault note tools will route through the Document-MCP backend HTTP API rather than accessing the filesystem directly, to ensure the vault search index stays consistent. This means vault tools require the backend server to be reachable — this constraint must be clearly communicated in tool descriptions and error messages.
- Oracle toggle state persists at the profile level. There is one toggle per profile, not per-user-session.
- Streaming oracle responses are out of scope for v1 — oracle queries return a single structured response when complete.
- The oracle toggle in the web UI applies to the MCP surface only. Direct CLI oracle usage is unaffected by the UI toggle.
- Multi-profile MCP support (switching profiles mid-session) is out of scope. Each MCP config entry targets one profile.
