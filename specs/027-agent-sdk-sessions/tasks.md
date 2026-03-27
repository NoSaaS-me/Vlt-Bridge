# 027 — Agent SDK Sessions: Tasks

## Phase 1: SDK Foundation

### Task 1.1: Install claude-agent-sdk
- Add `claude-agent-sdk` to `packages/vlt-cli/pyproject.toml` dependencies
- Verify import works: `from claude_agent_sdk import query, ClaudeAgentOptions`
- **Files**: `packages/vlt-cli/pyproject.toml`

### Task 1.2: Add mode toggle to SessionSidebar
- Add a toggle button in SessionSidebar header (next to daemon status dot)
- Two modes: "Relay" and "SDK"
- Persist selection in localStorage (`vlt:session-mode`)
- Pass mode to NewSessionDialog and spawn calls
- Style: compact segmented control or icon toggle
- **Files**: `frontend/src/components/agents/SessionSidebar.tsx`

### Task 1.3: Update NewSessionDialog for mode
- Show selected mode in dialog
- Pass mode through to `onStartFresh()` callback
- **Files**: `frontend/src/components/agents/NewSessionDialog.tsx`

### Task 1.4: Update daemon-api.ts spawn flow
- Add `mode: 'agent-sdk'` to the mode union type
- Route `agent-sdk` mode to new spawn endpoint: `POST /api/sessions/spawn/sdk`
- **Files**: `frontend/src/services/daemon-api.ts`

### Task 1.5: Create SDKSessionManager
- New module: `packages/vlt-cli/src/vlt/daemon/sdk_manager.py`
- `SDKSessionManager` class with:
  - `spawn(session_id, cwd, prompt, options) -> SDKSession`
  - `resume(session_id, prompt) -> SDKSession`
  - `send_message(session_id, text)`
  - `interrupt(session_id)`
  - `dismiss(session_id)`
- Manages `_agent_sdk_sessions: dict[str, AgentSDKSession]`
- Each session holds: `ClaudeSDKClient` instance, `session_id`, `cwd`, `status`
- Uses `ClaudeAgentOptions` with:
  - `env={"ANTHROPIC_API_KEY": key}` from daemon settings
  - `setting_sources=[]`
  - `permission_mode="bypassPermissions"`
  - Explicit `mcp_servers` config for vlt tools
- **Files**: New `packages/vlt-cli/src/vlt/daemon/sdk_manager.py`

### Task 1.6: Create SDKEventBridge
- New module: `packages/vlt-cli/src/vlt/daemon/sdk_event_bridge.py`
- Translates Agent SDK stream events to daemon infrastructure:
  - `AssistantMessage` → synthesize JSONL entry, push to `_live_stream_queues`, status "thinking"
  - `ToolResultMessage` → synthesize JSONL entry, status "executing"
  - `ResultMessage` → status "idle", update DB (cost, turns), trigger gate evals
  - `SystemMessage` → log, optional status update
- Writes synthetic JSONL transcript file for session persistence
- **Files**: New `packages/vlt-cli/src/vlt/daemon/sdk_event_bridge.py`

### Task 1.7: Wire SDKSessionManager into daemon server.py
- Import and instantiate `SDKSessionManager` in lifespan
- New endpoint: `POST /api/sessions/spawn/sdk`
  - Accepts: `{"cwd": str, "prompt": str, "session_id"?: str}`
  - Creates AgentSession in DB with `source="agent-sdk"`
  - Calls `sdk_manager.spawn()`
  - Returns session info
- Update `session_live_ws()` recv_loop:
  - Add third routing path: if session is agent-sdk, call `sdk_manager.send_message()`
- Update `dismiss_session()`:
  - Add agent-sdk cleanup path: call `sdk_manager.dismiss()`
- Update lifespan shutdown:
  - Clean up agent-sdk sessions
- **Files**: `packages/vlt-cli/src/vlt/daemon/server.py`

### Task 1.8: SDK Settings UI
- Add "SDK" tab or section in Settings page
- Fields: API key input (masked), default model selector, permission mode
- Store via daemon settings API
- **Files**: Frontend settings component, daemon settings endpoints

### Task 1.9: AgentSession model update
- Add `engine` column: `"relay" | "subprocess-sdk" | "agent-sdk"`
- Add `cost_usd` column (Float, nullable)
- Add `turn_count` column (Integer, nullable)
- Migration for existing rows (default engine based on source)
- **Files**: `packages/vlt-cli/src/vlt/core/models.py`, migrations

## Phase 2: Workflow Engine (Future)

### Task 2.1: WorkflowEngine core
### Task 2.2: VltThreadStore for state persistence
### Task 2.3: TOML workflow graph parser
### Task 2.4: Conditional routing
### Task 2.5: Human-in-the-loop injection
### Task 2.6: Workflow UI components

## Phase 3: Advanced Controls (Future)

### Task 3.1: Model swap endpoint + UI
### Task 3.2: MCP status/reconnect endpoints
### Task 3.3: Budget controls
### Task 3.4: Session forking
