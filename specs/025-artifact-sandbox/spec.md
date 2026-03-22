# Feature Specification: Artifact Sandbox

**Feature Branch**: `025-artifact-sandbox`
**Created**: 2026-03-15
**Status**: Draft
**Input**: Artifact plugin system — executable JS/CSS/HTML/Python bundles with sandboxed frontend + server-side backend, state machine enforcement, vision model review, hot reload, artifact-to-artifact IPC, connector integration, MCP tool exposure, and import/export as zip.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Create and Run a Simple Artifact (Priority: P1)

A user (or AI agent) creates a new artifact from the Artifacts tab in the agents view. They provide a name, description, and source files (HTML/CSS/JS for the frontend, optionally Python for the backend). The artifact renders in a sandboxed iframe embedded in the main agent pane. The frontend can interact with the platform through a structured bridge API (storage, notes, code search). The artifact progresses through a state machine from draft to approved.

**Why this priority**: This is the foundational flow — without it, nothing else works. A single artifact that renders and runs is the minimum viable product.

**Independent Test**: Can be fully tested by creating an artifact with a single `index.html`, viewing it render in the iframe, calling a VltBridge storage API, and verifying the state machine transitions from draft through approved.

**Acceptance Scenarios**:

1. **Given** the user is on the Artifacts tab, **When** they click "New Artifact" and provide a name and an `index.html` file, **Then** the artifact is created, appears in the sidebar list, and renders in the iframe pane.
2. **Given** an artifact exists in "draft" state, **When** the user or agent writes source files and triggers the "build" action, **Then** the state transitions to "building" and the artifact reloads with the new code.
3. **Given** an artifact frontend calls `VltBridge.storage.set("key", value)`, **When** the bridge processes the request, **Then** the value is persisted and retrievable via `VltBridge.storage.get("key")` across page reloads.
4. **Given** an artifact has passed its tests, **When** the state machine transitions to "approved", **Then** the artifact is marked complete and its approval is recorded with a timestamp.

---

### User Story 2 - Artifact with Server-Side Backend (Priority: P1)

A user creates an artifact that has both a frontend (HTML/JS/CSS in iframe) and a backend (Python process managed by the daemon). The backend runs as its own isolated process, communicates with the daemon, and exposes functionality to the frontend through the VltBridge API. The frontend never talks to the backend directly — all communication is proxied through the daemon.

**Why this priority**: Most non-trivial artifacts need server-side logic (data processing, API calls, connector access). Without this, artifacts are limited to static client-side rendering.

**Independent Test**: Can be tested by creating an artifact with a `backend/main.py` that defines a `handle(action, params)` function, calling it from the frontend via VltBridge, and verifying the response arrives correctly.

**Acceptance Scenarios**:

1. **Given** an artifact has a `backend/main.py` with a `handle` function, **When** the artifact is activated, **Then** the daemon starts the backend process and reports it as healthy.
2. **Given** the backend is running, **When** the frontend calls `VltBridge.backend.call("my_action", {param: "value"})`, **Then** the request is proxied through the daemon to the backend process and the response is returned to the frontend.
3. **Given** the backend process crashes, **When** the daemon detects the failure, **Then** it restarts the backend automatically and logs the crash to the artifact's thread.
4. **Given** the artifact has `requirements.txt` in its backend directory, **When** the backend is started for the first time, **Then** dependencies are installed in an isolated location before the process launches.

---

### User Story 3 - Hot Reload During Development (Priority: P2)

While an agent (Claude Code session) or human is iterating on an artifact's source code, file changes are detected and the artifact live-reloads without losing state. CSS changes swap stylesheets without a full reload. JS/HTML changes trigger a full iframe reload with state preservation. Python backend changes trigger a state-aware restart. The state machine reacts to code changes appropriately (approved code that changes demotes to testing).

**Why this priority**: The tight feedback loop between writing code and seeing results is what makes artifact development practical. Without hot reload, every change requires manual restart.

**Independent Test**: Can be tested by editing an artifact's CSS file while it's running, verifying the style updates without page reload, then editing a Python backend file and verifying the backend restarts with preserved state.

**Acceptance Scenarios**:

1. **Given** an artifact is rendered in the iframe, **When** a CSS file in the artifact's frontend directory changes, **Then** the stylesheet is swapped without a full page reload and DOM state is preserved.
2. **Given** an artifact is rendered in the iframe, **When** a JS or HTML file changes, **Then** the iframe fully reloads within 2 seconds of the last file change (debounced).
3. **Given** an artifact backend implements `save_state()`/`load_state()`, **When** a Python file changes, **Then** the backend serializes state, restarts with new code, and restores state — all within 5 seconds.
4. **Given** an artifact is in "approved" state, **When** any source file changes, **Then** the state is demoted to "testing" and the change is logged.
5. **Given** a Claude Code session is writing multiple files rapidly (within 500ms), **When** files are saved, **Then** only one reload cycle occurs after the debounce window expires.

---

### User Story 4 - Vision Model Review (Priority: P2)

When an artifact's state machine reaches the "reviewing" stage, the system captures a screenshot of the artifact's frontend, sends it to a vision-capable model with a description of what the artifact should look like/do, and receives a structured assessment. The assessment is passed to the primary agent model to decide pass/fail. If no vision model is configured, a non-blocking error is logged and displayed, and the review step is skipped.

**Why this priority**: Automated visual review closes the quality loop without requiring a human to manually inspect every iteration. It enables autonomous agent development workflows.

**Independent Test**: Can be tested by transitioning an artifact to "reviewing" state, verifying a screenshot is captured, sent to a vision model, and the assessment is recorded. Also test the fallback when no vision model is configured.

**Acceptance Scenarios**:

1. **Given** an artifact transitions to "reviewing" state and a vision model is configured, **When** the review runs, **Then** a screenshot is captured, sent to the vision model with the artifact's description and requirements, and a structured assessment is returned.
2. **Given** the vision model returns an assessment, **When** the primary model evaluates it, **Then** the artifact either advances to "approved" or returns to "building" with specific feedback.
3. **Given** no vision model is configured anywhere, **When** the review step is triggered, **Then** the review is skipped, a non-blocking error "Please configure a vision model" is logged to the artifact thread and displayed in the UI, and the artifact advances without visual review.
4. **Given** multiple model endpoints are configured (OpenRouter, z.ai, Gemini), **When** the system looks for a vision-capable model, **Then** it tries the user's configured vision model first, then falls back through available providers until one with vision capability is found.

---

### User Story 5 - Artifact Test Runner and State Machine Automation (Priority: P2)

An artifact defines test commands in its manifest. When code changes occur or the state machine evaluates, tests are automatically run. Test results drive state transitions: passing tests advance the state, failing tests hold it. The agent receives test feedback automatically through the session injection mechanism.

**Why this priority**: Automated testing is what makes the state machine meaningful — without it, state transitions are just manual checkpoints.

**Independent Test**: Can be tested by creating an artifact with a test command in its manifest, modifying code to break a test, and verifying the state machine holds at "testing" with the failure displayed.

**Acceptance Scenarios**:

1. **Given** an artifact manifest defines `tests.command`, **When** code changes are detected, **Then** the test command runs automatically and results are captured.
2. **Given** all tests pass, **When** the state machine evaluates, **Then** the artifact advances to the next state (testing → reviewing or approved).
3. **Given** tests fail, **When** the state machine evaluates, **Then** the artifact stays in "testing" state and failure details are logged.
4. **Given** a Claude Code session is building the artifact, **When** tests complete, **Then** the results are injected into the session's next turn via the hook mechanism so the agent can see them without polling.

---

### User Story 6 - Connector Access from Artifacts (Priority: P3)

An artifact's frontend or backend can access external services through the existing connector system. The artifact manifest declares which connectors and instances it needs. Multiple instances of the same connector are supported (e.g., three Reddit accounts). Credentials are never exposed to the artifact — all calls are proxied through the daemon. The user's per-action permission settings (allow/ask/off) are enforced.

**Why this priority**: Connectors enable artifacts to interact with the real world (post to Reddit, send emails, upload to YouTube). Without this, artifacts are isolated sandboxes with no external reach.

**Independent Test**: Can be tested by creating an artifact that calls `VltBridge.connectors.call("reddit", "bot-1", "post", {...})`, verifying the call is proxied through the daemon, permissions are checked, and credentials are never visible to the artifact code.

**Acceptance Scenarios**:

1. **Given** an artifact manifest declares connector instances, **When** the frontend calls `VltBridge.connectors.call(connector, instance, action, params)`, **Then** the request is proxied through the daemon using the specified instance's credentials.
2. **Given** a connector action is set to "ask" permission, **When** the artifact tries to invoke it, **Then** the call is blocked and the user is prompted for approval before proceeding.
3. **Given** a user has multiple instances of the same connector (e.g., three Reddit accounts), **When** the artifact references a specific instance by name, **Then** the correct credentials are used for that instance.
4. **Given** an artifact tries to call a connector not declared in its manifest, **Then** the call is rejected with an error explaining the connector must be declared.

---

### User Story 7 - Artifact-to-Artifact Communication (Priority: P3)

Artifacts can emit events and subscribe to events from other artifacts, enabling workflow pipelines. An artifact declares in its manifest which events it emits and subscribes to. Events are routed through the daemon's event bus. All events are logged to the emitting artifact's vlt thread for auditability.

**Why this priority**: This enables the "IFTTT/n8n on steroids" use case — chaining artifacts into complex automated workflows (Reddit bot → click funnel → video generator → QC pipeline).

**Independent Test**: Can be tested by creating two artifacts — one that emits "data_ready" events and one that subscribes to them — and verifying the subscriber receives the event and can act on it.

**Acceptance Scenarios**:

1. **Given** Artifact A emits event "post_created" with a payload, **When** Artifact B is subscribed to "post_created", **Then** Artifact B receives the event with the full payload.
2. **Given** an event is emitted, **Then** it is logged to the emitting artifact's vlt thread with timestamp and payload summary.
3. **Given** Artifact B subscribes to events but Artifact A is not running, **When** Artifact A starts and emits, **Then** Artifact B receives events from that point forward (no replay of missed events).
4. **Given** an artifact emits an event, **Then** it does not receive its own event (no self-notification).

---

### User Story 8 - Artifacts as MCP Tools (Priority: P3)

A deployed artifact can expose functions as MCP tools that other AI agents can call. The artifact manifest declares tool definitions (name, description, parameters). When the artifact is deployed, these tools are dynamically registered with the MCP server. Any Claude Code session or AI agent with the vlt MCP server can then call these tools, which route to the artifact's backend.

**Why this priority**: This turns artifacts into a tool-authoring system — users extend the platform's capabilities by building and deploying artifacts rather than writing MCP server code.

**Independent Test**: Can be tested by deploying an artifact with an MCP tool definition, then calling that tool from a Claude Code session and verifying the call reaches the artifact backend and returns the correct response.

**Acceptance Scenarios**:

1. **Given** an artifact manifest defines `mcp_tools` entries, **When** the artifact reaches "deployed" state, **Then** the tools are registered with the MCP server and appear in tool listings.
2. **Given** an MCP tool routes to an artifact backend, **When** the tool is called with parameters, **Then** the artifact backend's `handle` function receives the action and params and returns a response.
3. **Given** a deployed artifact is stopped or undeployed, **When** the MCP server is queried, **Then** the artifact's tools are no longer available and calls return an error explaining the artifact is not running.

---

### User Story 9 - Import and Export Artifacts (Priority: P3)

Users can export an artifact as a zip file containing all source files, manifest, and configuration. Users can import a zip file to create a new artifact from a previously exported one. This enables manual sharing of artifacts before a marketplace exists.

**Why this priority**: Without a marketplace (Phase 2), manual sharing via zip is the only way to distribute useful artifacts between users or machines.

**Independent Test**: Can be tested by creating an artifact, exporting it as zip, deleting the original, importing the zip, and verifying the imported artifact runs identically.

**Acceptance Scenarios**:

1. **Given** an artifact exists, **When** the user clicks "Export", **Then** a zip file is downloaded containing the complete artifact directory (manifest, frontend, backend, tests).
2. **Given** a valid artifact zip file, **When** the user clicks "Import" and selects the file, **Then** a new artifact is created with the contents of the zip, assigned a new ID, and appears in the sidebar.
3. **Given** an exported zip from a different user, **When** imported, **Then** connector instance references are preserved in the manifest but credentials are NOT included — the user must configure their own connector instances.
4. **Given** a zip file with an invalid or missing manifest, **When** import is attempted, **Then** the import fails with a clear error message explaining what's wrong.

---

### Edge Cases

- What happens when an artifact's backend process exceeds its memory quota? The daemon kills the process, logs the OOM event, transitions the artifact to "error" state, and displays the error in the UI.
- What happens when hot reload fires during an active test run? The test run is cancelled, the reload completes, and tests re-run from scratch.
- What happens when two artifacts subscribe to each other's events creating a loop? Event routing includes a hop counter (max 10). If exceeded, the event is dropped and a warning is logged.
- What happens when a Claude Code session writes files to an artifact directory that doesn't match the expected structure? Only files within recognized directories (frontend/, backend/, tests/) trigger hot reload. Other files are ignored by the watcher.
- What happens when the daemon restarts while persistent artifacts are running? On startup, the daemon scans for artifacts in "deployed" state and restarts their backend processes automatically.
- What happens when an artifact's backend takes longer than 5 seconds to serialize state during hot reload? The daemon kills the process after the timeout, logs a warning, and restarts without state restoration.
- What happens when an artifact is exported while its backend is running? The export captures source files only — runtime state and hot_state.json are excluded from the zip.
- What happens when multiple browser tabs have the same artifact open? All tabs receive hot reload signals via their own WebSocket connections. Each iframe operates independently.

## Requirements *(mandatory)*

### Functional Requirements

**Core Artifact Lifecycle**

- **FR-001**: System MUST allow creation of artifacts via the UI (Artifacts tab) and via MCP tool (`vlt_artifact_create`).
- **FR-002**: System MUST store artifact source files on disk in a structured directory layout (frontend/, backend/, tests/) with a manifest file.
- **FR-003**: System MUST render artifact frontends in a sandboxed iframe embedded in the agent pane, isolated from the host application.
- **FR-004**: System MUST serve artifact frontend files through the daemon, injecting the VltBridge script before serving `index.html`.
- **FR-005**: System MUST manage artifact backend processes as isolated server-side processes, one per artifact.
- **FR-006**: System MUST enforce a state machine for each artifact with states: draft, building, testing, reviewing, approved, deployed, error.
- **FR-007**: System MUST track state transition history with timestamps and actor (user, agent, harness).
- **FR-008**: System MUST initialize a git repository in each artifact directory on creation and auto-commit on state transitions.

**VltBridge API**

- **FR-009**: System MUST provide a `VltBridge` object inside the artifact iframe via an injected script that communicates with the parent page through postMessage.
- **FR-010**: VltBridge MUST expose scoped storage (get/set/list per artifact), vault note access (read/write/search), code search, and artifact self-management (setState, getState, log).
- **FR-011**: VltBridge MUST expose connector access that proxies all calls through the daemon — artifact code MUST never have direct access to credentials.
- **FR-012**: VltBridge MUST expose backend call routing for artifacts with server-side backends.
- **FR-013**: VltBridge MUST expose event emit/subscribe for artifact-to-artifact IPC.

**Hot Reload**

- **FR-014**: System MUST watch active artifact directories for file changes using filesystem notifications.
- **FR-015**: System MUST debounce file change events with a configurable window (default 500ms) to handle multi-file writes.
- **FR-016**: System MUST classify changes as CSS-only (hot swap), JS/HTML (full iframe reload), Python (backend restart), or manifest (config reload).
- **FR-017**: System MUST support optional state serialization/restoration for backend processes during hot reload via a `save_state()`/`load_state()` contract.
- **FR-018**: System MUST apply state machine demotion policies on code changes (approved → testing, deployed → update_pending).

**Testing & Review**

- **FR-019**: System MUST execute artifact test commands defined in the manifest when code changes are detected.
- **FR-020**: System MUST inject test results into active Claude Code sessions via the existing hook mechanism.
- **FR-021**: System MUST capture screenshots of artifact frontends for vision model review using a server-side browser automation tool.
- **FR-022**: System MUST discover vision-capable models by checking the user's configured vision model, then falling back through available providers.
- **FR-023**: System MUST handle absence of vision models gracefully with a non-blocking error logged and displayed to the user.

**Connectors & Multi-Instance**

- **FR-024**: System MUST support multiple named instances of the same connector per user (e.g., three Reddit accounts with different credentials).
- **FR-025**: System MUST support proxy profile configuration (named proxy configs that connector instances can reference).
- **FR-026**: Artifacts MUST declare required connector instances in their manifest; calls to undeclared connectors MUST be rejected.

**MCP Tool Exposure**

- **FR-027**: System MUST dynamically register MCP tools from deployed artifacts with the MCP server.
- **FR-028**: System MUST unregister artifact MCP tools when an artifact is stopped or undeployed.
- **FR-029**: MCP tool calls MUST route through the daemon to the artifact's backend process.

**Import/Export**

- **FR-030**: System MUST export artifacts as zip files containing all source files and manifest (excluding runtime state and credentials).
- **FR-031**: System MUST import zip files to create new artifacts with new IDs, validating the manifest structure before creation.

**Resource Management**

- **FR-032**: System MUST enforce per-artifact resource quotas defined in the manifest (CPU timeout, memory limit, storage limit).
- **FR-033**: System MUST kill backend processes that exceed resource quotas and transition the artifact to "error" state.

**Vision Model Configuration**

- **FR-034**: System MUST provide a vision model selector in the settings UI alongside the existing oracle and subagent model selectors.
- **FR-035**: Vision model review MUST use a two-step process: vision model describes the screenshot, then the primary model evaluates the description against requirements.

### Key Entities

- **Artifact**: An executable bundle with a unique ID, name, description, type (ephemeral/persistent), source files (frontend + backend), manifest, state machine position, and optional MCP tool definitions. Belongs to a user and project.
- **Artifact Manifest**: Configuration file declaring the artifact's metadata, frontend entry point, backend entry point and runtime, connector instance requirements, MCP tool definitions, event declarations, resource quotas, and test commands.
- **Artifact State**: The current position in the lifecycle state machine (draft → building → testing → reviewing → approved → deployed → error), with full transition history.
- **Connector Instance**: A named configuration of a connector with its own credentials, linked to a proxy profile. Multiple instances of the same connector can exist per user.
- **Proxy Profile**: A named proxy configuration (URL, credentials) that connector instances can reference for routing their traffic.
- **Artifact Event**: A typed message with a payload emitted by one artifact and received by subscribing artifacts, routed through the daemon's event bus.
- **Vision Model Setting**: A user-level configuration specifying which model and provider to use for artifact visual review, with automatic fallback discovery.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can create, edit, and run a simple HTML/JS artifact within 1 minute of opening the Artifacts tab.
- **SC-002**: Artifact frontend hot reload (CSS swap) completes within 200ms of file change detection; full reload within 2 seconds.
- **SC-003**: Artifact backend restart (with state serialization) completes within 5 seconds of Python file change detection.
- **SC-004**: 95% of artifact state machine transitions complete without manual intervention (driven by test results and automated review).
- **SC-005**: Vision model review produces an actionable pass/fail assessment within 30 seconds of entering the reviewing state.
- **SC-006**: Artifact-to-artifact event delivery completes within 500ms of emission.
- **SC-007**: Deployed artifact MCP tools respond within 2 seconds when called by external agents.
- **SC-008**: Artifact import from zip creates a runnable artifact within 10 seconds.
- **SC-009**: Artifacts with backend processes remain stable for 24+ hours in deployed state without memory leaks or crashes (under normal operating conditions).
- **SC-010**: Resource quota enforcement kills runaway processes within 5 seconds of exceeding limits.

## Assumptions

- The daemon is always running when artifacts are active (artifacts are server-dependent).
- Users have at least one model endpoint configured (OpenRouter, z.ai, or Gemini) for the oracle/agent workflows that interact with artifacts.
- Artifact frontends are single-page applications served as static files — no server-side rendering or build tooling beyond what the daemon provides.
- Python is the only supported backend runtime in Phase 1. Other runtimes (WASM, Go, Rust) are deferred.
- The existing connector permission model (allow/ask/off per action) is sufficient for artifact use — no artifact-specific permission layer is needed beyond manifest declaration.
- Marketplace/template sharing is out of scope for Phase 1.
- Elm compilation support is out of scope for Phase 1.
- Artifact collaboration beyond git-based workflows in standalone folders is out of scope.

## Dependencies

- Existing daemon session management (SDK sessions, hook system)
- Existing connector system (connector_service, connector_tools)
- Existing MCP server (mcp_server.py, dynamic tool registration)
- Existing vlt thread system (for artifact audit logging)
- Server-side browser automation tool (for screenshot capture)
- File system notification library (for hot reload watching)
