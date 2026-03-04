# vlt — Quickstart Guide

Persistent cognitive state, session relay, and code intelligence for AI agents and humans.

---

## 1. Install

```bash
# From repo root
pip install --user --break-system-packages -e packages/vlt-cli
```

Verify:
```bash
vlt --help
vlt-claude --version   # should print claude's version
```

---

## 2. Run the Setup Wizard

The wizard checks every component and fixes what it can in one shot:

```bash
vlt setup
```

It covers:
- Daemon health (starts it if down)
- Claude Code hooks (`~/.claude/settings.json`)
- `vlt-claude` wrapper symlink
- Project `vlt.toml` (prompts to create if missing)

---

## 3. Wire `claude` to the Relay

After install, `vlt-claude` is the relay wrapper. Set up your shell so `claude` calls it and the original CLI stays reachable as `claude-bare`.

> **Why two commands?** `claude` goes through the relay (sessions visible in web UI, injectable from browser). `claude-bare` reaches the real binary directly — useful for scripts, one-off SDK calls, or when the daemon is down.

### fish

```fish
# Wire functions (run once)
function claude; vlt-claude $argv; end; funcsave claude
function claude-bare; ~/.local/bin/claude $argv; end; funcsave claude-bare
```

### zsh

Add to `~/.zshrc`:

```zsh
export PATH="$HOME/.local/share/vlt/bin:$PATH"

function claude() { vlt-claude "$@"; }
function claude-bare() { ~/.local/bin/claude "$@"; }
```

Then reload: `source ~/.zshrc`

### bash

Add to `~/.bashrc`:

```bash
export PATH="$HOME/.local/share/vlt/bin:$PATH"

function claude() { vlt-claude "$@"; }
function claude-bare() { ~/.local/bin/claude "$@"; }
```

Then reload: `source ~/.bashrc`

### macOS (zsh default since Catalina)

Same as zsh above. Note: if you installed Python via Homebrew or the python.org installer, `vlt-claude` may land in a non-standard location. Find it and adjust:

```zsh
which vlt-claude   # e.g. /opt/homebrew/bin/vlt-claude
                   #   or ~/Library/Python/3.x/bin/vlt-claude
```

Then use that path in your `claude-bare` function if `~/.local/bin/claude` doesn't exist:

```zsh
function claude-bare() { /usr/local/bin/claude "$@"; }   # adjust to match `which claude`
```

### Windows (PowerShell)

Add to your PowerShell profile (`$PROFILE`):

```powershell
function claude { vlt-claude @args }
function claude-bare { & "$env:LOCALAPPDATA\AnthropicClaude\claude.exe" @args }
```

Reload: `. $PROFILE`

> **Note:** On Windows, `vlt-claude` lands in `%APPDATA%\Python\PythonXX\Scripts\`. Make sure that folder is in your `$PATH`. Run `where.exe vlt-claude` to confirm.

> **Symlinks on Windows** require Developer Mode (`Settings → Developer Mode → on`). The setup wizard creates the symlink at `~\.local\share\vlt\bin\claude.exe` if Developer Mode is enabled; otherwise add `vlt-claude` to PATH directly.

---

## 4. Start the Daemon

```bash
vlt daemon start        # start in background
vlt daemon status       # check health
vlt daemon stop         # stop
```

The daemon runs at `http://localhost:8765` and manages session relay, hook events, cronban, and the web UI bridge.

---

## 5. Project Setup

In any project directory:

```bash
vlt setup               # creates vlt.toml, registers hooks, checks wrapper
```

Or manually:

```bash
# Creates vlt.toml in current directory
cat > vlt.toml <<EOF
[project]
id = "my-project"
name = "My Project"
path = "$(pwd)"
EOF
```

---

## 6. Core Workflow — Threads

Threads are persistent reasoning chains. Use them to survive context resets.

```bash
# Start a thread
vlt thread new my-project auth-design "Designing JWT auth flow"

# Log thoughts (fast, <50ms)
vlt thread push auth-design "Using RS256 over HS256 for multi-service support"
vlt thread push auth-design "Need refresh token rotation — stateless JWTs won't cut it"

# Read current state (summary + recent nodes)
vlt thread read auth-design

# Semantic search across all threads
vlt thread seek "refresh token"

# List threads for a project
vlt thread list my-project
```

---

## 7. MCP Server

The vlt MCP server exposes 19 tools to Claude Code:

```bash
# Register (one-time)
claude mcp add --scope user vlt vlt-mcp

# Verify
claude mcp list
```

Tools available: `vlt_thread_*`, `vlt_code_*`, `vlt_oracle_*`, `vlt_note_*`, `vlt_status`, `vlt_project_detect`.

---

## 8. CodeRAG — Code Intelligence

Index a codebase for hybrid semantic + BM25 search:

```bash
vlt coderag init --project my-project --path /path/to/codebase
vlt coderag status --project my-project
vlt coderag search "authentication middleware" --project my-project
vlt coderag map --project my-project
```

---

## 9. Cronban — Scheduled Agent Tasks

Schedule prompts to fire into running sessions:

```bash
vlt cron list                              # list scheduled triggers
vlt cron add --session <id> --cron "0 9 * * 1-5" "Daily standup reminder"
vlt cron fire <trigger-id>                 # fire immediately (test)
vlt cron delete <trigger-id>
```

---

## 10. Re-run Setup Anytime

```bash
vlt setup --check-only    # audit without changing anything
vlt setup                 # check and auto-fix
```
