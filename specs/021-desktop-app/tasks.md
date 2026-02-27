# Tasks: Unified Agent Platform (Desktop App)

**Branch**: `021-desktop-app` | **Spec**: [Link](./spec.md) | **Plan**: [Link](./plan.md)

## Phase 1: Setup & Initialization

- [x] T001 Initialize `packages/vlt-daemon` structure with Mojo configuration
- [x] T002 Initialize `desktop-app` Tauri project with React/TypeScript template
- [x] T003 Configure `desktop-app` with dependencies (`xterm.js`, `xterm-addon-webgl`, `lucide-react`)
- [x] T004 Create `backend/src/services/daemon_service.py` skeleton for Server-side daemon tracking
- [x] T005 Create `backend/src/api/routes/daemon.py` endpoint skeleton for protocol handling

## Phase 2: Foundational (Daemon & Protocol)

- [x] T006 Implement `Daemon` entity in `backend/src/models/daemon.py` (SQLAlchemy/Pydantic)
- [x] T007 Implement `SessionLease` and `AgentSession` models in `backend/src/models/session.py`
- [x] T008 [P] Implement Mojo WebSocket server in `packages/vlt-daemon/src/server.mojo` (using Python interop if needed)
- [x] T009 [P] Implement JSON-RPC message parser in `packages/vlt-daemon/src/protocol.mojo`
- [x] T010 Implement `SessionService` in `backend/src/services/session_service.py` to manage leases
- [x] T011 Implement `DaemonConnect` WebSocket endpoint in `backend/src/api/routes/daemon.py` to register daemons

## Phase 3: User Story 1 (Remote Heavy Compute)

**Goal**: Connect Desktop App to Daemon and execute a basic command.

- [x] T012 [US1] Implement `ProcessPool` in `packages/vlt-daemon/src/process_pool.mojo` using Python `subprocess`
- [x] T013 [US1] Implement PTY manager in `packages/vlt-daemon/src/pty.mojo` using Python `pty`
- [x] T014 [US1] Create `Terminal` component in `desktop-app/src/components/Terminal.tsx` using `xterm.js`
- [x] T015 [US1] Implement `DaemonClient` in `desktop-app/src/services/daemon.ts` for WebSocket communication
- [x] T016 [US1] Implement `execute_task` handler in `packages/vlt-daemon/src/handlers.mojo`
- [x] T017 [US1] Implement `ClientConnect` endpoint in `backend/src/api/routes/client.py` to proxy commands to Daemon
- [x] T018 [US1] Connect Desktop App `Terminal` to `ClientConnect` WebSocket and render output stream

## Phase 4: User Story 2 (Instant Task Startup)

**Goal**: Optimize startup latency using warm process pool.

- [x] T019 [P] [US2] Enhance `ProcessPool` in `packages/vlt-daemon/src/process_pool.mojo` to pre-fork workers
- [x] T020 [US2] Implement `SessionManager` in `packages/vlt-daemon/src/session.mojo` to assign warm processes
- [x] T021 [US2] Add latency metrics logging to `packages/vlt-daemon/src/metrics.mojo`
- [x] T022 [US2] Update `desktop-app` UI to show "Starting..." vs "Ready" states in `desktop-app/src/components/SessionStatus.tsx`

## Phase 5: User Story 3 (Seamless Device Handoff)

**Goal**: Persist session history and support multi-client attach.

- [x] T023 [P] [US3] Implement circular buffer for history in `packages/vlt-daemon/src/history.mojo`
- [x] T024 [US3] Implement `task_update` sync from Daemon to Server in `backend/src/services/daemon_sync.py`
- [x] T025 [US3] Update `ClientConnect` endpoint to send full history on new connection
- [x] T026 [US3] Test detach/attach flow by connecting Web UI (existing React app) to `ClientConnect` endpoint

## Phase 6: Polish & Integration

- [x] T027 Bundle `OpenVSCode Server` binary with `desktop-app` (Tauri sidecar config)
- [x] T028 Implement "IDE" layout in `desktop-app/src/layouts/IdeLayout.tsx` (Split Editor/Terminal)
- [x] T029 Secure WebSocket connections with JWT validation in `packages/vlt-daemon/src/auth.mojo`
- [x] T030 Add "Remote Connection" settings screen in `desktop-app/src/pages/Settings.tsx`

## Dependencies

- **US1** requires Foundational Phase (Protocol & Models).
- **US2** extends US1 (Process Pool optimization).
- **US3** requires US1 (Basic execution) and Backend Session Service.

## Implementation Strategy

- **MVP**: Complete Phase 1-3 to prove the "Remote Terminal" concept.
- **Optimization**: Phase 4 ensures the performance goals (<500ms).
- **Production**: Phase 5 & 6 enable the actual workflow utility.
