# Unified Agent Platform (Desktop App)

## Overview

### Problem Statement
The current Vlt-Bridge architecture relies on a monolithic Python backend where the UI and compute are tightly coupled. This leads to several issues:
- **Performance**: Spawning new Python processes for each agent task causes multi-second UI freezes and high latency.
- **Terminal Quality**: The web-based terminal (xterm.js) suffers from input latency and rendering issues compared to native terminals.
- **Local-Only Compute**: Users cannot easily offload heavy agent workloads (e.g., training, complex reasoning) to remote GPU servers while maintaining a local control interface.
- **Single-Client Lock**: Agent sessions are tied to the specific browser tab or process that spawned them, preventing seamless handoff between devices.

### Solution Summary
Transform Vlt-Bridge into a **Unified Agent Platform** with a Client-Daemon architecture.
- **Mojo/Rust Daemon**: A high-performance orchestrator that runs independently, managing agent processes and state. It can run locally or on a remote server.
- **Tauri Desktop App**: A native desktop client providing a professional "IDE 2.0" experience with GPU-accelerated terminal (Alacritty) and integrated code editing.
- **Web Client**: The existing Vlt-Bridge web UI becomes a lightweight client connecting to the same daemon.

### Scope
- **In Scope**:
    - **Daemon**: Implementation of the orchestrator (likely using a Mojo shim for GIL-free performance) handling sessions, process pooling, and PTYs.
    - **Desktop App**: Tauri-based application wrapping the frontend with native terminal integration.
    - **Protocol**: WebSocket/IPC-based JSON-RPC protocol for client-daemon communication.
    - **Session Management**: Server-side session persistence allowing attach/detach.
    - **Remote Support**: Secure tunneling/connection for remote daemon access.
- **Out of Scope**:
    - Re-implementing the core Agent logic (Oracle/Librarian remain in Python, managed by the Daemon).
    - Full replacement of VS Code (we integrate/embed, not replace).

## User Scenarios & Testing

### Scenario 1: Remote Heavy Compute
**User**: A developer training a model using an agent.
**Flow**:
1.  User starts the **Vlt Daemon** on a remote Linux server with H100 GPUs.
2.  User opens **Vlt Desktop** on their MacBook.
3.  User configures the connection to the remote daemon (via secure tunnel/SSH).
4.  User types "Train the model on the dataset" in the desktop terminal.
5.  The command executes on the remote server with full GPU access.
6.  Terminal output streams instantly to the laptop.

**Acceptance**:
-   Command executes on remote host.
-   Latency remains low (<100ms) despite network.
-   Session persists if laptop disconnects.

### Scenario 2: Instant Task Startup
**User**: A developer running frequent small tasks.
**Flow**:
1.  User types a command to "Fix the typo in README".
2.  Daemon instantly allocates a pre-warmed Python process from its pool.
3.  Agent starts execution immediately (<500ms latency).
4.  No UI freeze occurs during startup.

**Acceptance**:
-   Time from "Enter" to first log output is under 500ms.
-   UI remains responsive throughout.

### Scenario 3: Seamless Device Handoff
**User**: A developer moving from desk to mobile.
**Flow**:
1.  User starts a long-running refactoring task on **Desktop App**.
2.  User leaves desk and opens **Vlt Web UI** on their phone.
3.  User sees the *same* active session, including full scrollback history.
4.  User pauses the task from the phone.
5.  Desktop app reflects the "Paused" state immediately.

**Acceptance**:
-   Session state is synchronized across clients.
-   Scrollback history is preserved and delivered to new clients.

## Functional Requirements

### FR-1: Daemon Orchestrator
The system MUST provide a central daemon process.
-   **Process Management**: Maintain a "warm pool" of Python worker processes to minimize startup time.
-   **Session State**: act as the "Source of Truth" for active agent sessions, holding their PTY handles and history.
-   **Concurrency**: Handle multiple concurrent sessions without blocking (leveraging Mojo/Rust async I/O).

### FR-2: Native Desktop Client
The system MUST provide a desktop application via Tauri.
-   **Terminal**: Render terminal output using a native, GPU-accelerated backend (e.g., Alacritty) for <50ms latency.
-   **Webview**: Render the existing React UI for chat and visualization.
-   **Editor Integration**: Embed a code editor (Monaco or VS Code Web) for direct file manipulation.

### FR-3: Unified Communication Protocol
The system MUST use a standard protocol for all clients.
-   **Transport**: WebSocket for Web/Remote, Unix Domain Sockets for Local Desktop.
-   **Format**: JSON-RPC handling commands (`start_session`, `send_input`) and events (`output_chunk`, `state_change`).
-   **Multiplexing**: Support multiple clients listening to the same session stream.

### FR-4: Security & Remote Access
The system MUST secure connections.
-   **Authentication**: Token-based auth for remote connections.
-   **Tunneling**: Support for secure tunnels (like VS Code Remote) to expose the daemon over the internet safely.

## Success Criteria

### Quantitative
-   **Startup Latency**: New agent sessions start in **< 500ms**.
-   **Terminal Latency**: Key-to-pixel latency in Desktop App is **< 50ms**.
-   **Concurrency**: Daemon handles **10+ active sessions** without degradation.
-   **Recovery**: Clients can reconnect to a running session in **< 1s**.

### Qualitative
-   **"Native Feel"**: Terminal scrolling and interaction feels indistinguishable from iTerm2/Alacritty.
-   **Reliability**: Sessions never "die" just because the UI window was closed.

## Key Entities

### Session
Represents a persistent agent task.
-   `id`: UUID
-   `status`: RUNNING | PAUSED | COMPLETED | FAILED
-   `process_id`: ID of the worker process
-   `history`: Circular buffer of terminal output
-   `created_at`: Timestamp

### Client
Represents a connected UI.
-   `id`: Connection ID
-   `type`: DESKTOP | WEB
-   `subscribed_sessions`: List of Session IDs being watched

### ProcessPool
Manages worker resources.
-   `warm_count`: Number of ready-to-go processes
-   `max_processes`: Hard limit on concurrency

## Assumptions
-   The existing `OracleAgent` and `ToolExecutor` logic can be wrapped/imported by the Daemon without complete rewrite.
-   Tauri's `wgpu` integration is mature enough for Alacritty embedding (or we use a high-performance Canvas renderer as fallback).
-   Mojo is available and stable enough for the shim layer (or we fall back to Rust if Mojo is immature).