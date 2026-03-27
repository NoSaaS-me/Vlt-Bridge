# 027 — Agent SDK Sessions

## Overview

Replace the current subprocess-based SDK session management with the Claude Agent SDK (`claude-agent-sdk`) Python package. Add a mode toggle in the frontend to switch between Relay (PTY/terminal) and SDK (Agent SDK) sessions. SDK sessions use a separate Anthropic API key for isolation from the user's Claude Code subscription.

## Goals

1. **SDK Mode**: New session mode using `claude-agent-sdk` `query()` / `ClaudeSDKClient` for structured, programmatic Claude sessions
2. **Mode Toggle UI**: Visual toggle in SessionSidebar to switch between Relay and SDK modes
3. **Environment Isolation**: SDK sessions use a separate API key, isolated settings, and controlled MCP config — no bleed into the user's Claude Code environment
4. **Event Bridge**: Translate Agent SDK stream events into the existing daemon live-streaming infrastructure (WebSocket protocol, JSONL transcripts, status updates)
5. **Control Protocol Exposure**: Expose Agent SDK control capabilities (interrupt, model swap, MCP toggle) to the frontend
6. **Workflow Engine Foundation**: Lightweight stateful workflow engine (LangGraph-inspired, not LangGraph) for multi-session orchestration

## Non-Goals

- In-process MCP server (existing external MCP is fine, latency is not a concern)
- Replacing relay mode (it stays, just not the default for new work)
- LangGraph as a dependency (too heavyweight, impedance mismatch with Agent SDK)
- Modifying existing MCP tools

## Architecture

```
Frontend (React)
    │
    ├── SessionSidebar [Mode Toggle: Relay | SDK]
    ├── LiveSessionPanel (unchanged — transport-agnostic)
    └── NewSessionDialog (mode selector added)
    │
    │ WebSocket (same protocol)
    ▼
Daemon (FastAPI)
    │
    ├── SDKSessionManager (new)
    │   ├── ClaudeAgentOptions per session
    │   ├── Separate ANTHROPIC_API_KEY
    │   ├── setting_sources=[] (isolated)
    │   ├── Explicit MCP config (vlt tools)
    │   └── Control protocol (interrupt, model swap)
    │
    ├── SDKEventBridge (new)
    │   ├── Agent SDK events → JSONL transcript entries
    │   ├── Status updates → _push_status_to_live()
    │   └── Result events → gate evaluations
    │
    ├── WorkflowEngine (new, Phase 2)
    │   ├── TOML-defined workflow graphs
    │   ├── State checkpoints via vlt threads
    │   ├── Conditional routing
    │   └── Human-in-the-loop injection
    │
    └── [Legacy] Relay sessions (unchanged)
```

## Agent SDK Session Lifecycle

```
spawn(session_id, cwd, prompt, api_key)
  │
  ▼
ClaudeAgentOptions(
    env={"ANTHROPIC_API_KEY": api_key},
    setting_sources=[],
    cwd=cwd,
    mcp_servers=<explicit vlt MCP config>,
    permission_mode="bypassPermissions",
    allowed_tools=[...],
)
  │
  ▼
query(prompt, options) → async generator
  │
  ├── AssistantMessage → EventBridge → JSONL + WebSocket
  ├── ToolResultMessage → EventBridge → JSONL + WebSocket
  └── ResultMessage → EventBridge → status idle + gate eval
  │
  ▼
Session persists in _agent_sdk_sessions dict
Resume via ClaudeAgentOptions(resume=session_id)
```

## Environment Isolation

| Concern | Solution |
|---------|----------|
| Auth | `env={"ANTHROPIC_API_KEY": <separate key>}` |
| Settings | `setting_sources=[]` — no inheritance |
| Session storage | Controlled `cwd` per session |
| MCP servers | Explicit config, not inherited from user |
| Permissions | `permission_mode="bypassPermissions"` |

API key stored in daemon settings (Settings UI or env var `VLT_SDK_API_KEY`).

## WebSocket Protocol (unchanged)

Server → Client:
- `{"type": "initial", "entries": [...], "session": {...}}`
- `{"type": "entry", "entry": <JSONL object>}`
- `{"type": "status", "status": "thinking"|"idle"|"executing"}`
- `{"type": "ctx_pct", "ctx_pct": <int>}`

Client → Server:
- `{"type": "inject", "text": "user message"}`

The EventBridge synthesizes JSONL entries from Agent SDK events so the frontend sees identical message format regardless of session mode.

## Data Model Changes

### AgentSession model

Add `source` value: `"agent-sdk"` (alongside existing `"relay"`, `"managed"`, `"hook"`, `"discovery"`)

Consider adding columns:
- `engine`: `"relay" | "subprocess-sdk" | "agent-sdk"` (more explicit than overloading source)
- `cost_usd`: Accumulated cost from ResultMessage.total_cost_usd
- `turn_count`: From ResultMessage.num_turns

### Daemon Settings

New settings for SDK mode:
- `sdk_api_key`: Anthropic API key for SDK sessions
- `sdk_default_model`: Default model for SDK sessions
- `sdk_permission_mode`: Default permission mode
- `sdk_max_turns`: Default max turns per query
- `sdk_max_budget_usd`: Default budget limit
- `sdk_mcp_config`: MCP server config to pass to sessions

## Control Protocol Features

Exposed to frontend via new API endpoints:

| Endpoint | Agent SDK Control | Purpose |
|----------|-------------------|---------|
| `POST /api/sessions/{id}/interrupt` | `interrupt()` | Stop current task |
| `PUT /api/sessions/{id}/model` | `set_model()` | Swap model mid-session |
| `PUT /api/sessions/{id}/permissions` | `set_permission_mode()` | Change permissions |
| `GET /api/sessions/{id}/mcp-status` | `mcp_status()` | MCP server health |
| `POST /api/sessions/{id}/mcp-reconnect` | `mcp_reconnect()` | Reconnect failed MCP |

## Workflow Engine (Phase 2)

Lightweight state machine over Agent SDK sessions:

- **State**: Persisted in vlt threads (ACID-safe, survives restarts)
- **Graph**: TOML-defined nodes with prompt templates, tool sets, routing conditions
- **Routing**: Conditional edges based on result inspection
- **Human-in-the-loop**: Inject messages, override routing, pause/resume
- **Checkpointing**: State snapshot at each node boundary
- **Forking**: Branch workflow at any checkpoint

Not implementing LangGraph because the Agent SDK owns the agent loop. Our engine manages *between-session* state, not *within-session* execution.

## Dependencies

- `claude-agent-sdk` (Python package, added to vlt-cli pyproject.toml)
- No new frontend dependencies

## Phases

1. **SDK Foundation**: Install SDK, mode toggle UI, SDKSessionManager, EventBridge, wire into daemon
2. **Workflow Engine**: State machine, vlt thread persistence, TOML graphs, workflow UI
3. **Advanced Controls**: Model swap, MCP management, budget controls, forking
