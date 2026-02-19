---
title: Document Viewer
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: A bridge for all agents to pass markdown
tags:
  - building-mcp-track-consumer
  - building-mcp-track-enterprise
  - building-mcp-track-creative

---

# Vlt-Bridge

Two things in one repo: a persistent memory system for AI agents (`vlt` CLI) and a web-based Obsidian-style vault viewer with an MCP server (`Document-MCP`). They share the vault concept — markdown files on disk — but are otherwise independent systems that don't talk to each other at runtime. The CLI does not route through the backend; agents can shell out to `vlt` directly or hit the MCP server's vault tools over HTTP.

The web UI is a full-featured note editor and viewer aimed at humans. The MCP server exposes vault operations to AI agents. The CLI is aimed at agents and power users who want persistent thread memory, code indexing, and oracle queries without going through a web server.

## Monorepo Layout

```
Vlt-Bridge/
├── backend/              # FastAPI server (Document-MCP) + MCP server (FastMCP)
│   └── src/
│       ├── api/          # REST routes: notes, search, graph, RAG, TTS, auth
│       ├── mcp/          # FastMCP STDIO/HTTP server (vault tools)
│       ├── bt/           # Behavior Tree runtime for Oracle agent
│       └── services/     # vault, indexer, auth, database, ANS, signal_parser
├── frontend/             # React 19 + Vite 7 + shadcn/ui
├── packages/
│   └── vlt-cli/          # vlt CLI — threads, coderag, oracle, librarian daemon
├── specs/                # SpecKit feature specs
└── data/                 # Runtime data (vaults/, index.db) — gitignored
```

## vlt CLI

The CLI is installed from `packages/vlt-cli` and exposed as the `vlt` command. It's the primary interface for agent memory workflows.

**Requirements**: Python 3.11+, an [OpenRouter](https://openrouter.ai/) API key, and `universal-ctags` for Oracle (`apt install universal-ctags`).

**Install:**

```bash
cd packages/vlt-cli
python -m venv .venv && source .venv/bin/activate
pip install -e ".[oracle]"
vlt profile init
vlt config set-key <YOUR_OPENROUTER_KEY>
```

### Thread Memory

Threads are the core primitive. Each thread belongs to a project and is a persistent, append-only log of thoughts. A background "Librarian" daemon compresses raw nodes into semantic state objects using an LLM (Grok-4.1-Fast by default). `thread push` is optimized for speed — under 50ms — so agents can log freely without blocking.

```bash
vlt overview                                      # see active projects and thread states
vlt thread new <project> <thread-id> "goal"       # create a thread
vlt thread push <thread-id> "insight or decision" # append a node
vlt thread read <thread-id>                       # load semantic state (summarized)
vlt thread read <thread-id> --search "jwt"        # search within a thread
vlt thread seek "concept or question"             # semantic search across all threads
vlt tag <node-id> "#bug"                          # attach a semantic tag
vlt link <node-id> <thread-id>                    # cross-link thoughts
```

Multi-agent attribution: pass `--author "AgentName"` on every `push`. The author field is stored with the node and visible on read/seek.

### CodeRAG

Hybrid code indexing: vector similarity, BM25, and a graph derived from ctags call-graph data. Languages supported: Python, TypeScript, TSX, JavaScript, Go, Rust.

```bash
vlt coderag init --project <id> --path /path/to/repo   # index a codebase (background by default)
vlt coderag init --project <id> --foreground            # index with live progress output
vlt coderag init --project <id> --force                 # re-index (overwrite existing)
vlt coderag status --project <id>                       # check indexing status
vlt coderag search "authentication flow" --project <id> # hybrid retrieval query
vlt coderag map --project <id>                          # repo structure overview
vlt coderag map --project <id> --scope src/api/         # scoped map
```

Status values: `pending`, `running`, `completed`, `failed`, `cancelled`. Indexing runs via the background daemon by default; if the daemon isn't running, you'll be prompted to run in foreground.

Configure inclusions/exclusions per project with a `coderag.toml` in the project root:

```toml
[coderag]
include = ["**/*.py", "**/*.ts", "**/*.tsx"]
exclude = ["**/node_modules/**", "**/.venv/**", "**/dist/**"]
```

### Oracle

Multi-source query synthesis over code index, vault notes, and thread history. Uses a Behavior Tree-controlled agent loop (OpenRouter models) with loop detection, budget enforcement, and an XML signal protocol for self-reflection.

```bash
vlt oracle "How does authentication work?"
vlt oracle "Where is UserService defined?" --source code
vlt oracle "Why did we choose SQLite?" --source threads
vlt oracle "Explain the architecture" --local
vlt oracle "Hard question" --thinking --model anthropic/claude-sonnet-4
```

By default, Oracle checks for a running backend server and uses it in thin-client mode (sharing context with the web UI). `--local` forces local processing. `--explain` shows retrieval traces. `--source` accepts `code`, `vault`, or `threads` and can be used multiple times to filter retrieval.

### Daemon Management

```bash
vlt daemon start    # background indexing daemon
vlt daemon stop
vlt daemon status
vlt librarian start # background summarization daemon (for thread compression)
```

## Document-MCP

### Web UI

Browser-based vault viewer and editor. Talks to the FastAPI backend at port 8000.

- Obsidian-style wikilinks (`[[Note Name]]`) resolved via SQLite slug matching — same-folder match preferred, then lexicographic
- Full-text search: SQLite FTS5 with BM25 ranking, title weighted 3x, recency bonus
- Backlinks: automatic tracking of incoming references per note
- Graph view: force-directed note relationship visualization (`react-force-graph-2d`)
- Split-pane editor with live markdown preview
- AI chat panel: RAG over vault content (LlamaIndex + Gemini embeddings)
- TTS: ElevenLabs integration for reading notes aloud
- Optimistic concurrency: `if_version` field on saves, 409 on conflict

Authentication: HF OAuth in `space` mode, or no-auth local mode with a fixed `LOCAL_USER_ID`.

### MCP Server

Exposes 7 tools over STDIO or HTTP+JWT:

| Tool | Description |
|------|-------------|
| `list_notes` | List notes in vault (optionally filtered by path prefix) |
| `read_note` | Read note content and metadata |
| `write_note` | Create or overwrite a note (last-write-wins, no version check) |
| `delete_note` | Delete a note and remove from index |
| `search_notes` | FTS5 full-text search with BM25 ranking |
| `get_backlinks` | Return all notes that link to a given note |
| `get_tags` | List all tags in the vault |

Note: the MCP server covers vault notes only. Thread memory, CodeRAG, and Oracle are not exposed through MCP — use the CLI directly for those.

### Running Locally

```bash
# Full stack (recommended)
./start-dev.sh    # backend on :8000, frontend on :5173
./stop-dev.sh
./status-dev.sh

# Backend only
cd backend
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# MCP STDIO server (for Claude Desktop/Code, local mode)
uv run python src/mcp/server.py

# MCP HTTP server (JWT-authenticated, for remote clients)
uv run python src/mcp/server.py --http --port 8001

# Frontend only
cd frontend
npm install
npm run dev    # http://localhost:5173
```

Tests:

```bash
cd backend
uv run pytest                     # all tests
uv run pytest tests/unit          # unit only
uv run pytest tests/integration   # integration only
uv run pytest -k test_vault_write # single pattern
```

## MCP Client Configuration

### vlt-mcp Unified Server (Recommended)

The `vlt-mcp` server exposes threads, code intelligence, oracle, and vault notes through a single auto-starting MCP server. No ports or daemon required for core operations.

**One-time setup:**
```bash
cd packages/vlt-cli && pip install -e ".[oracle]"
claude mcp add --scope user vlt vlt-mcp    # Claude Code (user scope)
```

**Cold-start time**: 164ms. Claude Code spawns it automatically on first tool call.

See [`specs/018-vlt-mcp-server/quickstart.md`](specs/018-vlt-mcp-server/quickstart.md) for full setup including Claude Desktop, per-project override, and oracle toggle.

---

### Document-MCP Backend Server (Legacy STDIO)

**STDIO (local mode, Claude Desktop/Code):**

```json
{
  "mcpServers": {
    "document-mcp": {
      "command": "uv",
      "args": ["run", "python", "src/mcp/server.py"],
      "cwd": "/absolute/path/to/Vlt-Bridge/backend"
    }
  }
}
```

**HTTP (remote HF Space deployment, JWT):**

```json
{
  "mcpServers": {
    "document-mcp": {
      "transport": "http",
      "url": "https://YOUR_USERNAME-Document-MCP.hf.space/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_JWT_TOKEN"
      }
    }
  }
}
```

Get the JWT from Settings inside the web UI after logging in via HF OAuth.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODE` | yes | `local` | `local` (single-user) or `space` (HF multi-tenant OAuth) |
| `JWT_SECRET_KEY` | yes | — | Generate: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `VAULT_BASE_DIR` | no | `./data/vaults` | Filesystem root for per-user vaults |
| `DB_PATH` | no | `./data/index.db` | SQLite database |
| `LOCAL_USER_ID` | no | `local-dev` | User identity in local mode |
| `HF_OAUTH_CLIENT_ID` | space mode | — | HF OAuth app client ID |
| `HF_OAUTH_CLIENT_SECRET` | space mode | — | HF OAuth app client secret |
| `GOOGLE_API_KEY` | optional | — | Gemini API for RAG embeddings and AI chat |
| `ELEVENLABS_API_KEY` | optional | — | TTS integration |
| `ELEVENLABS_VOICE_ID` | optional | — | ElevenLabs voice ID |
| `ORACLE_MAX_TURNS` | optional | `30` | Oracle agent iteration limit |
| `ORACLE_PROMPT_BUDGET` | optional | `8000` | Oracle system prompt token limit |

See `.env.example` for the full list.

## Deployment

The backend and frontend are packaged together as a Docker image for HF Spaces deployment. See [DEPLOYMENT.md](./DEPLOYMENT.md) for step-by-step instructions covering HF Space creation, OAuth app setup, environment variable configuration, and push options.

```bash
# Local Docker build (mirrors HF Spaces)
docker build -t vlt-bridge .
docker run -p 7860:7860 -e JWT_SECRET_KEY="dev-secret" vlt-bridge
```
