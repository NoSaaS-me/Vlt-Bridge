# Quickstart: Vlt Unified MCP Server

**Branch**: `018-vlt-mcp-server`

This guide covers configuring `vlt-mcp` as a global MCP server in Claude Desktop and Claude Code.

---

## Prerequisites

1. vlt-cli installed with the MCP extra:
   ```bash
   cd packages/vlt-cli
   pip install -e ".[oracle]"
   ```

2. Profile initialized:
   ```bash
   vlt profile init
   vlt config set-key <YOUR_OPENROUTER_KEY>
   ```

3. Verify the command exists:
   ```bash
   vlt-mcp --help
   ```

---

## Claude Code (Global, Recommended)

Use the `claude mcp add` command (user scope):

```bash
claude mcp add --scope user vlt vlt-mcp
```

That's it. Claude Code will spawn `vlt-mcp` automatically at session start.

> **Note**: Do not add `mcpServers` directly to `~/.claude/settings.json` — that field is not
> supported at the user scope. The `claude mcp add --scope user` command stores it in the
> correct location.

**Cold-start time**: 164ms measured on 9950x3d. Well within the 2-second target.

Verify with:
```bash
vlt-mcp --check
```

---

## Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "vlt": {
      "command": "vlt-mcp",
      "env": {}
    }
  }
}
```

Restart Claude Desktop. The server will start automatically on next launch.

---

## Per-Project Override (optional)

Add `.mcp.json` to any project root:

```json
{
  "mcpServers": {
    "vlt": {
      "command": "vlt-mcp",
      "env": {
        "VLT_PROFILE": "work"
      }
    }
  }
}
```

This overrides the global config for that project directory, using the `work` profile.

---

## Disable Oracle (optional)

From the Document-MCP web UI: **Settings → Oracle → toggle off**.

Or, without the web UI:
```bash
echo "VLT_ORACLE_ENABLED=false" >> ~/.vlt/profiles/default/.env
```

---

## Verify It's Working

In Claude Code, ask the agent:

> "Run vlt_status to check what's available"

Expected response: a dict showing your profile, active projects, and available capabilities.

---

## Index a Project for Code Search

In any coding session:

> "Use vlt_code_init to index this project, then search for how authentication works"

The agent will call `vlt_code_init` with the project path, poll `vlt_code_status` until complete, then run `vlt_code_search`.

---

## Notes

- The MCP server runs once per AI assistant session. It starts when the session starts and exits when the session ends. No background daemon needed for thread and meta operations.
- Vault tools (`vlt_note_*`) require the Document-MCP backend to be running (`./start-dev.sh`). Thread and code tools work standalone.
- Code indexing background jobs require the vlt daemon: `vlt daemon start`. If the daemon isn't running, `vlt_code_init` runs indexing inline (blocking) and returns when complete.
