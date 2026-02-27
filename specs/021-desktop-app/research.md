# Research: Unified Agent Platform (Desktop App)

**Feature**: Unified Agent Platform (Desktop App)
**Date**: 2026-01-24
**Status**: Complete

## Unknowns & Clarifications

### Technology Stack
- **Mojo Readiness**: Mojo has good interoperability with Python's `socket` and `subprocess` modules, making it a viable candidate for the daemon shim. It can handle PTYs via Python's `pty` module.
- **Tauri + Alacritty**: Embedding Alacritty directly is complex. The recommended approach is using `tauri-plugin-pty` which uses `xterm.js` on the frontend and a Rust PTY backend. For true GPU acceleration, we can explore `wgpu` integration but `xterm.js` with WebGL addon is a safer, proven fallback.
- **VS Code Embedding**: "OpenVSCode Server" as a sidecar process in Tauri is the most robust solution for a full IDE experience. Embedding Monaco is a lighter alternative if we only need an editor.

### Architecture
- **Daemon-Server Quorum**: The "Lease" mechanism uses a centralized Server authority model.
- **Session Persistence**: Serialization of agent state is critical.

### Dependencies
- `portable-pty` (Rust) is solid for the backend if we go with `tauri-plugin-pty`.
- Mojo can use Python's `asyncio` or its own emerging async features.

## Decision Log

### Decision 1: Daemon Language
- **Decision**: **Mojo** for the daemon orchestrator. It can leverage Python's ecosystem for PTY/Socket while offering better performance headroom than pure Python. We will use Python's `pty` and `socket` modules via Mojo's interop.

### Decision 2: Terminal Renderer
- **Decision**: **xterm.js (WebGL)** in the Tauri frontend. It's battle-tested (VS Code uses it), integrates easily with React, and offers near-native performance with the WebGL addon. Embedding a separate wgpu surface for Alacritty is high-risk/high-effort for Phase 1.

### Decision 3: Editor Integration
- **Decision**: **OpenVSCode Server as a Sidecar**. This provides the "IDE 2.0" experience with extensions and file tree out of the box. We will bundle it and spawn it from the Tauri app.

### Decision 4: Quorum Protocol
- **Mechanism**: The Web Server (FastAPI) is the authority.
- **Flow**:
    1.  Clients (Web/Desktop) send commands to the **Server**.
    2.  Server assigns a `Lease` to a specific Daemon (identified by ID).
    3.  Server pushes the command to the Daemon (via WebSocket).
    4.  Daemon executes and streams results back to Server.
    5.  Server broadcasts results to all subscribed Clients.
-   **Reasoning**: Centralizes state, simplifies multi-client sync, and prevents split-brain.

## Protocol Schema (JSON-RPC)

**Server -> Daemon:**
```json
{
  "jsonrpc": "2.0",
  "method": "execute_task",
  "params": {
    "lease_id": "uuid",
    "task": { "command": "train_model", "env": {...} }
  },
  "id": 1
}
```

**Daemon -> Server (Stream):**
```json
{
  "jsonrpc": "2.0",
  "method": "task_update",
  "params": {
    "lease_id": "uuid",
    "output": "chunk of stdout",
    "status": "running"
  }
}
```

**Daemon -> Server (Result):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "lease_id": "uuid",
    "exit_code": 0,
    "status": "completed"
  },
  "id": 1
}
```
