# Vlt-Bridge

Persistent memory, code intelligence, and runtime platform for AI agents.

## What it does

**Memory & Retrieval**
- **Threads** — append-only reasoning logs per project. Librarian compresses nodes into semantic state. `push` is <50ms.
- **CodeRAG** — hybrid code index (vector + BM25 + ctags call-graph). Python, TypeScript, JavaScript, Go, Rust.
- **Oracle** — multi-source Q&A over threads, code, and vault. RLM REPL harness: LLM gets a Python sandbox, writes code to explore, sets `Final` to answer.

**Vault & Web UI**
- File vault with wikilinks, FTS5 search (BM25, title 3x weight), backlinks, graph view. Stores any file type — markdown, images, PDFs, audio, video — with OCR indexing for search.
- Split-pane editor with live preview. AI chat (RAG + Gemini). TTS (ElevenLabs). Asset upload with drag-and-drop, inline embedding via `[[embed:file.png]]`.

**Agent Sessions**
- Niri-style terminal compositor for managing multiple Claude Code sessions side by side.
- Live session view with bidirectional control (send prompts, see responses in real-time).
- Hook-based session tracking — Claude Code HTTP hooks (SessionStart, Stop, PreToolUse, etc.) report session state to the daemon automatically.
- SDK sessions: persistent Claude subprocess that stays alive between messages.

**Daemon (port 8765)**
- Central hub. Agent session management (Claude Code hooks + SDK subprocess control).
- Artifact sandbox lifecycle. Connector routing. Cronban scheduling. CodeRAG job queue.

**Artifact Sandboxes**
- Self-contained apps that run inside the platform. Each artifact has its own frontend (HTML/JS served in an iframe), backend (Python subprocess managed by the daemon), and working directory inside the vault.
- The daemon provides a bidirectional harness: the backend can initiate calls outward (connector invocations, storage, event emission) via stdout, and receives requests from the frontend via stdin. The frontend talks to the backend through a PostMessage bridge (`VltBridge`) injected into the iframe.
- Multi-stage pipelines with configurable stage types (text generation, text cleanup, text review, image generation, vision review). Stages chain outputs forward.
- Configurable output destinations fire on approval: write to vault as notes, save files locally, or publish through connectors.
- State machine lifecycle: draft → building → testing → reviewing → approved → deployed.
- Folder sync: `vlt artifact sync` pushes a local directory to an artifact like rsync. `vlt artifact pull` downloads it.
- Templates: `text_factory` ships a complete content generation pipeline (generate → cleanup → review → output).

**Connectors**
- Pluggable AI model connectors: OpenRouter, z.ai, z.ai Vision, custom OpenAI-compatible endpoints (vLLM, Ollama, self-hosted).
- Composio integrations for external services (Gmail, GitHub, etc).
- Used by artifact backends for LLM calls, and available directly via CLI/MCP.

**Cronban**
- Schedule recurring prompts into live Claude Code sessions. 5-field cron expressions or one-shot timers.
- Visual kanban board in the web UI for managing scheduled tasks.
- Can trigger artifact runbooks on a schedule for automated content pipelines.

## Install

```bash
# From source (dev)
cd packages/vlt-cli
pip install -e ".[oracle]"
vlt profile init
vlt config set-key <OPENROUTER_KEY>
vlt daemon start
```

CLI is the primary interface. MCP is also available for agents that prefer tool-use over shell:

```bash
claude mcp add --scope user vlt vlt-mcp
```

To run the web UI (vault browser, agent compositor, cronban board, artifact viewer):

```bash
./start-dev.sh    # backend :8000, frontend :5173
```

> **Future:** `pip install vlt-bridge && vlt setup` — single install that bundles CLI + daemon + connectors + web UI. Not yet published to PyPI.

## Architecture

```mermaid
flowchart TB
  CLI[vlt CLI]
  MCP[vlt-mcp]
  WebUI[Web UI :5173]

  subgraph Daemon ["Daemon :8765"]
    Sessions[Session Manager]
    Artifacts[Artifact Sandboxes]
    Cronban[Cronban Scheduler]
    CRAGd[CodeRAG Jobs]
  end

  Backend["Backend :8000\n(FastAPI)"]
  Connectors[Connectors]
  Vault[("Vault\n(files + SQLite FTS)")]
  ThreadDB[("Thread DB\n(SQLite)")]

  CLI --> ThreadDB
  CLI --> Daemon
  CLI --> Backend
  MCP --> ThreadDB
  MCP --> Daemon
  MCP --> Backend
  WebUI --> Backend
  WebUI --> Daemon
  Backend --> Vault
  Daemon --> Connectors
  Daemon --> Artifacts
```

## Project structure

```
Vlt-Bridge/
├── backend/                # FastAPI — vault CRUD, search, auth, Oracle, RAG, TTS, assets
│   └── src/
│       ├── api/            # REST routes + middleware
│       ├── mcp/            # MCP server (vault tools, STDIO/HTTP)
│       ├── models/         # Pydantic schemas
│       └── services/       # vault, asset_vault, indexer, rlm_oracle, ans, ...
├── frontend/               # React 19, Vite 7, shadcn/ui
│   └── src/
│       ├── pages/          # AgentsPage (compositor), ConnectorsPage, ...
│       └── components/     # agents/, artifacts/, cronban/, ...
├── packages/
│   ├── vlt-cli/            # CLI + daemon + vlt-mcp server
│   │   └── src/vlt/daemon/ # Daemon: sessions, artifacts, cronban, harness
│   └── vlt-connectors/     # Pluggable model connectors (OpenRouter, z.ai, custom)
├── specs/                  # SpecKit feature specs
├── data/                   # Runtime: vaults/, index.db (gitignored)
└── desktop-app/            # Optional Tauri wrapper
```

## CLI reference

### Threads

| Command | Description |
|---------|-------------|
| `vlt overview` | Active projects and thread states |
| `vlt thread new <project> <id> "goal"` | Create a thread |
| `vlt thread push <id> "thought"` | Append a node (<50ms) |
| `vlt thread read <id>` | Read thread state |
| `vlt thread read <id> --search "jwt"` | Filtered read |
| `vlt thread seek "query"` | Semantic search across all threads |
| `vlt thread list --project <id>` | List threads in a project |

Use `--author "AgentName"` on `push` for multi-agent attribution.

### CodeRAG

| Command | Description |
|---------|-------------|
| `vlt coderag init --project <id> --path <dir>` | Index a codebase |
| `vlt coderag status --project <id>` | Check indexing progress |
| `vlt coderag search "query" --project <id>` | Hybrid search |
| `vlt coderag map --project <id>` | Repo structure overview |

### Oracle

| Command | Description |
|---------|-------------|
| `vlt oracle "question"` | Query (uses backend if available) |
| `vlt oracle "question" --local` | Force local execution |
| `vlt oracle "question" --source code` | Restrict to code index |
| `vlt oracle "question" --thinking` | Show retrieval traces |

### Artifacts

| Command | Description |
|---------|-------------|
| `vlt artifact list` | List all artifacts |
| `vlt artifact create <name> --template text_factory` | Create from template |
| `vlt artifact get <id>` | Inspect artifact details |
| `vlt artifact files <id>` | List files with sizes and hashes |
| `vlt artifact sync <id> <folder>` | Push local folder to artifact (delta sync) |
| `vlt artifact pull <id> <folder>` | Download artifact to local folder |
| `vlt artifact call <id> <action> [params]` | Run backend action (test, generate, approve, ...) |
| `vlt artifact start/stop <id>` | Control backend process |
| `vlt artifact state <id> <state>` | Transition state machine |
| `vlt artifact templates` | List available templates |

### Connectors & Cronban

| Command | Description |
|---------|-------------|
| `vlt connectors list` | Available connectors |
| `vlt connectors actions <name>` | Connector action schemas |
| `vlt cron sessions` | List live Claude Code sessions |
| `vlt cron add <title> <expr> <prompt> --session <id>` | Schedule recurring prompt |
| `vlt cron list` | Show active schedules |
| `vlt cron pause/resume/delete <id>` | Manage schedules |

### System

| Command | Description |
|---------|-------------|
| `vlt daemon start/stop/status` | Daemon lifecycle |
| `vlt profile init` | Initialize storage profile |
| `vlt config set-key <key>` | Set OpenRouter API key |

## Environment variables

Source of truth: [backend/src/services/config.py](backend/src/services/config.py).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JWT_SECRET_KEY` | yes | -- | `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `VAULT_BASE_PATH` | no | `./data/vaults` | Per-user vault root |
| `ENABLE_LOCAL_MODE` | no | `true` | Single-user local mode |
| `LOCAL_USER_ID` | no | `local-dev` | User ID in local mode |
| `GOOGLE_API_KEY` | no | -- | Gemini for RAG / AI chat |
| `ELEVENLABS_API_KEY` | no | -- | TTS |
| `ELEVENLABS_VOICE_ID` | no | -- | ElevenLabs voice |
| `ORACLE_MAX_TURNS` | no | `30` | Max REPL iterations per Oracle query |
| `ORACLE_PROMPT_BUDGET` | no | `8000` | Token budget for Oracle system prompt |
| `BASE_URL` | no | `http://localhost:8000` | Production base URL |

## Deployment

```bash
docker build -t vlt-bridge .
docker run -p 7860:7860 -e JWT_SECRET_KEY="dev-secret" vlt-bridge
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup.
