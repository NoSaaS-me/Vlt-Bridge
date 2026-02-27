# Quickstart

> Get from zero to working in under 5 minutes. Pick your path.

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.11+ | `python --version` |
| Node.js | 18+ | `node --version` |
| `universal-ctags` | any | `ctags --version` (for Oracle symbol extraction) |

Install ctags if missing:
```bash
# Arch/CachyOS
sudo pacman -S ctags

# Debian/Ubuntu
sudo apt install universal-ctags

# macOS
brew install universal-ctags
```

---

## Path 1 — vlt CLI (agent memory + code search + Oracle)

This is the core tool. Install it first regardless of which other paths you take.

### Install

```bash
cd packages/vlt-cli
python -m venv .venv && source .venv/bin/activate
pip install -e ".[oracle]"
```

Verify:
```bash
vlt --version
vlt overview        # Should show empty project list
```

### Configure API keys

vlt uses [OpenRouter](https://openrouter.ai/) for CodeRAG embeddings. For LLM inference you can use OpenRouter (any model) or Z.AI (GLM models).

```bash
# Set your OpenRouter key (used for CodeRAG embeddings + non-GLM oracle calls)
vlt config set-key <YOUR_OPENROUTER_KEY>

# Optional: Z.AI GLM key (for glm-4.x models via oracle --model glm-4.7)
vlt config set-key --provider glm <YOUR_GLM_KEY>
```

Initialize a profile for your project:
```bash
# From your project root (creates vlt.toml if needed)
vlt profile init
```

### Thread memory (< 50ms writes)

```bash
# Create a thread
vlt thread new my-project auth-design "Design authentication flow"

# Log a decision
vlt thread push auth-design "Chose JWT HS256 — simpler than RSA for single-server setup"

# Read thread (summary + recent nodes)
vlt thread read auth-design

# Search across all threads
vlt thread seek "why did we choose jwt"
```

Multi-agent: use `--author "AgentName"` on every `push` for attribution.

### CodeRAG (index + search your codebase)

```bash
# Index your project (background daemon, or foreground if daemon not running)
vlt coderag init --project my-project --path /path/to/your/repo

# Check progress
vlt coderag status --project my-project

# Search code
vlt coderag search "authentication middleware" --project my-project

# Get a repo map (Aider-style overview)
vlt coderag map --project my-project
vlt coderag map --project my-project --scope src/api/
```

Optional `coderag.toml` in your project root to customize what gets indexed:
```toml
[coderag]
include = ["src/**/*.py", "frontend/src/**/*.ts"]
exclude = ["**/node_modules/**", "**/.venv/**"]
```

### Oracle (multi-source AI answers)

Oracle queries code index + vault notes + thread history. It uses an RLM harness: the LLM writes Python in a sandbox to explore your project, then sets `Final` when done.

```bash
# Ask about code
vlt oracle "How does authentication work?" --project my-project

# Restrict to code only
vlt oracle "Where is UserService defined?" --source code --project my-project

# Use a specific model (default uses your configured oracle_model)
vlt oracle "Explain the REPL harness" --model anthropic/claude-sonnet-4-6 --project my-project

# Use a GLM model (routes to Z.AI automatically when model starts with glm-)
vlt oracle "Summarize recent decisions" --model glm-4.7 --project my-project

# Extended thinking for hard questions
vlt oracle "Why do we use RestrictedPython over exec()?" --thinking --project my-project

# Show retrieval traces (debug)
vlt oracle "..." --explain --project my-project
```

---

## Path 2 — vlt-mcp (connect Claude Code/Desktop to everything)

One MCP server that gives Claude access to threads, CodeRAG, Oracle, and vault.

### Install (if you haven't done Path 1 yet)

```bash
cd packages/vlt-cli && pip install -e ".[oracle]"
```

### Register with Claude Code

```bash
claude mcp add --scope user vlt vlt-mcp
```

Verify in Claude Code:
```
/mcp
```
You should see `vlt` listed with 19 tools (vlt_thread_*, vlt_code_*, vlt_oracle_*, vlt_note_*, vlt_status, vlt_project_detect).

### First use in Claude

```
Hey Claude, call vlt_status to check what's available, then vlt_project_detect to find the project.
```

### Per-project scope

To limit Oracle to a specific project in a session, tell Claude to pass `project_id` to the oracle tools, or add a `vlt.toml` to your project root:

```toml
[project]
name = "my-project"
id = "my-project"
```

Then Claude will auto-detect it when you run Claude Code from that directory.

---

## Path 3 — Web UI + Oracle backend (Document-MCP)

The web app gives humans a vault viewer/editor and exposes the Oracle through a REST API that agents can call.

### 1. Backend setup

```bash
cd backend
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Create `.env` (copy from example and edit):
```bash
cp .env.example .env
```

Minimum required settings:
```bash
# .env
JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
VAULT_BASE_PATH=./data/vaults
```

Start the server:
```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://localhost:8000/api/health
# {"status":"ok","mode":"local"}
```

### 2. Frontend setup

```bash
cd frontend
npm install
npm run dev    # http://localhost:5173
```

### 3. Configure Oracle LLM keys in the UI

Open **Settings → Models** in the web UI:

| Setting | Key to provide | Used for |
|---------|---------------|----------|
| OpenRouter API key | OpenRouter key | CodeRAG embeddings + non-GLM oracle |
| GLM API key | Z.AI key | Oracle when model starts with `glm-` |
| Oracle model | e.g. `deepseek/deepseek-chat-v3-0324` | Default oracle model |

To use Z.AI GLM models for the Oracle, set the oracle model to `glm-4.7` (or any `glm-` prefixed model) and provide your Z.AI API key. OpenRouter key is still needed for CodeRAG embeddings.

### 4. Index your code from the UI

In the web UI go to **Settings → CodeRAG**, enter your project path, and click **Initialize**. Progress is visible in the status indicator.

Or from the CLI (shares the same index):
```bash
vlt coderag init --project my-project --path /path/to/repo
```

### 5. Automated startup

```bash
./start-dev.sh      # Starts backend :8000 + frontend :5173
./stop-dev.sh       # Stop both
./status-dev.sh     # Check running processes
```

---

## Path 4 — Docker (self-contained)

```bash
docker build -t vlt-bridge .
docker run -p 7860:7860 \
  -e JWT_SECRET_KEY="$(python -c "import secrets; print(secrets.token_urlsafe(32))")" \
  vlt-bridge
```

Access at `http://localhost:7860`.

---

## Models & providers

| Provider | Model examples | Key needed | Use for |
|----------|---------------|------------|---------|
| OpenRouter | `deepseek/deepseek-chat-v3-0324`, `anthropic/claude-sonnet-4-6` | OpenRouter key | Oracle + CodeRAG embeddings |
| Z.AI GLM | `glm-4.7`, `glm-4.7-flash`, `glm-4.5` | Z.AI key | Oracle LLM inference (model name must start with `glm-`) |

OpenRouter key is always needed for CodeRAG embeddings (`qwen/qwen3-embedding-8b`). GLM key is only needed when using GLM models for the Oracle.

---

## Quick reference

```bash
# vlt CLI
vlt overview                                        # See all projects/threads
vlt thread new <proj> <id> "goal"                   # New thread
vlt thread push <id> "insight"                      # Log a thought
vlt thread read <id>                                # Read thread
vlt thread seek "query"                             # Semantic search
vlt coderag init --project <id> --path <dir>        # Index codebase
vlt coderag status --project <id>                   # Check index status
vlt coderag search "query" --project <id>           # Search code
vlt coderag map --project <id>                      # Repo map
vlt oracle "question" --project <id>                # Ask Oracle
vlt daemon start                                    # Start background daemon
vlt daemon status                                   # Check daemon

# Backend tests
cd backend && uv run pytest                         # All tests
cd backend && uv run pytest tests/unit              # Unit only
cd backend && uv run pytest -k test_vault_write     # Single pattern
```

---

## Troubleshooting

**`vlt: command not found`** — activate the venv: `source packages/vlt-cli/.venv/bin/activate`

**`ctags not found`** — install `universal-ctags` (not legacy ctags). Symbol extraction degrades gracefully but repomap loses call-graph detail.

**CodeRAG indexing stuck at `pending`** — the daemon isn't running. Either `vlt daemon start` or re-run with `--foreground`.

**Oracle: `OpenRouter API key not configured`** — set your key with `vlt config set-key <KEY>` (CLI) or in Settings → Models (web UI).

**Oracle: `Z.AI GLM API key not configured`** — only required when oracle model starts with `glm-`. Set with `vlt config set-key --provider glm <KEY>` or in Settings → Models.

**`409 Conflict` on note save (web UI)** — the note was updated between open and save. Reload to get the latest version, re-apply your changes, and save again.

**Backend: `JWT secret is not configured`** — set `JWT_SECRET_KEY` in your `.env`. Generate one with:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
