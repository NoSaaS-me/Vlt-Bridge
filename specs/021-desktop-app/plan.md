# Implementation Plan: Unified Agent Platform (Desktop App)

**Branch**: `021-desktop-app` | **Date**: 2026-01-24 | **Spec**: [Link](./spec.md)
**Input**: Feature specification from `/specs/021-desktop-app/spec.md`

## Summary

Transform Vlt-Bridge into a client-daemon architecture. A high-performance **Mojo/Rust Daemon** orchestrates agent processes and PTYs, controlled by a centralized **FastAPI Server**. A new **Tauri Desktop App** provides a native "IDE 2.0" interface with GPU-accelerated terminal (Alacritty/xterm-webgl) and embedded code editing.

## Technical Context

**Language/Version**: Python 3.11+ (Server), Mojo (Daemon Shim), Rust 1.75+ (Tauri), TypeScript/React 18 (Frontend)
**Primary Dependencies**:
- **Backend**: FastAPI, Uvicorn, Websockets
- **Daemon**: Mojo SDK, Python `pty`/`subprocess` modules
- **Desktop**: Tauri v2, xterm.js (with WebGL addon), OpenVSCode Server (bundled)
**Storage**: SQLite (Session state), Filesystem (Logs)
**Testing**: pytest (Server), cargo test (Tauri), Vitest (Frontend)
**Target Platform**: Linux (Daemon/Server), macOS/Windows/Linux (Desktop Client)
**Project Type**: Hybrid (Web + Desktop + Daemon Microservice)
**Performance Goals**: <50ms terminal latency, <500ms session startup
**Constraints**: Must support remote daemon connection over SSH/Tunnel.
**Scale/Scope**: <100 concurrent sessions per daemon node.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Brownfield Integration**: Respects existing FastAPI backend as the authority. Reuses React frontend components in Tauri.
- [x] **Test-Backed**: Requires new integration tests for the Daemon-Server protocol.
- [x] **Incremental Delivery**: Phase 1 implements the Daemon shim without breaking the existing Web UI.
- [x] **Specification-Driven**: All work maps to `spec.md`.

## Project Structure

### Documentation (this feature)

```text
specs/021-desktop-app/
├── plan.md              # This file
├── research.md          # Technology decisions (Mojo, Tauri, Protocol)
├── data-model.md        # Entity definitions (Daemon, Session)
├── quickstart.md        # Setup guide
├── contracts/           # API schemas (daemon-api.yaml)
└── tasks.md             # Task breakdown
```

### Source Code (repository root)

```text
backend/                 # Existing FastAPI Server (The Authority)
├── src/
│   ├── api/routes/      # New endpoints for Daemon/Client sync
│   └── services/        # New SessionService, DaemonService

packages/
└── vlt-daemon/          # NEW: Mojo-based Daemon
    ├── src/             # Mojo source files
    └── mojo.toml        # Package config

desktop-app/             # NEW: Tauri Application
├── src-tauri/           # Rust backend for Tauri
│   ├── src/
│   └── tauri.conf.json
└── src/                 # React Frontend (shared + desktop specific)
```

**Structure Decision**: A monorepo approach adding `desktop-app` and `packages/vlt-daemon` alongside existing `backend` and `frontend`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Mojo Language | Fixes GIL/Process bottlenecks | Pure Python creates unacceptable UI freezes during agent tasks. Rust rewrite is too costly. |
| Tauri App | Native terminal performance | Browser-based terminal (Web only) cannot achieve <50ms latency or GPU acceleration reliably. |
| Daemon Microservice | Remote Compute | Monolithic web app cannot run heavy workloads on a remote server while UI stays local. |