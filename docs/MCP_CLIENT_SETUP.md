# MCP Client Configuration Guide

This guide shows how to configure various MCP clients to connect to the Document-MCP server for Markdown note management.

## Overview

Document-MCP provides an MCP server that exposes tools for managing Markdown notes, similar to how [jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server) provides tools for Jupyter notebooks. Instead of notebook cells, we work with Markdown files in a vault structure.

## Available MCP Tools

| Tool Name | Description |
|-----------|-------------|
| `list_notes` | List all notes in the vault (optionally filtered by folder) |
| `read_note` | Read a Markdown note with metadata and body |
| `write_note` | Create or update a note (auto-updates timestamps and search index) |
| `delete_note` | Delete a note and remove it from the index |
| `search_notes` | Full-text search with BM25 ranking and recency bonus |
| `get_backlinks` | List notes that reference the target note |
| `get_tags` | List all tags with associated note counts |

## Transport Modes

### 1. STDIO Transport (Local Development)

For local development with Claude Desktop, Cursor, or other MCP clients.

#### Using Python Module

```json
{
  "mcpServers": {
    "obsidian-docs": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/absolute/path/to/Document-MCP/backend",
      "env": {
        "LOCAL_USER_ID": "local-dev",
        "JWT_SECRET_KEY": "local-dev-secret-key-123",
        "VAULT_BASE_PATH": "/absolute/path/to/Document-MCP/data/vaults",
        "DB_PATH": "/absolute/path/to/Document-MCP/data/index.db",
        "PYTHONPATH": "/absolute/path/to/Document-MCP/backend",
        "FASTMCP_SHOW_CLI_BANNER": "false"
      }
    }
  }
}
```

#### Using uv (Recommended)

```json
{
  "mcpServers": {
    "obsidian-docs": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp.server"],
      "cwd": "/absolute/path/to/Document-MCP/backend",
      "env": {
        "LOCAL_USER_ID": "local-dev",
        "JWT_SECRET_KEY": "local-dev-secret-key-123",
        "VAULT_BASE_PATH": "/absolute/path/to/Document-MCP/data/vaults",
        "DB_PATH": "/absolute/path/to/Document-MCP/data/index.db"
      }
    }
  }
}
```

### 2. HTTP Transport (Remote/Hosted)

For remote access via HTTP endpoint (e.g., Hugging Face Spaces deployment).

#### Configuration

```json
{
  "mcpServers": {
    "obsidian-docs": {
      "transport": "http",
      "url": "https://your-space.hf.space/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_JWT_TOKEN"
      }
    }
  }
}
```

#### Obtaining a JWT Token

1. **Via OAuth (Production)**: Login through Hugging Face OAuth at `/auth/login`
2. **Via API (Development)**: 
   ```bash
   curl -X POST http://localhost:8001/api/tokens \
     -H "Authorization: Bearer YOUR_EXISTING_TOKEN"
   ```

## Client-Specific Configuration

### Claude Desktop

**Location**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

**STDIO Example**:
```json
{
  "mcpServers": {
    "obsidian-docs": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp.server"],
      "cwd": "C:\\Workspace\\Document-MCP\\backend",
      "env": {
        "LOCAL_USER_ID": "local-dev",
        "JWT_SECRET_KEY": "local-dev-secret-key-123",
        "VAULT_BASE_PATH": "C:\\Workspace\\Document-MCP\\data\\vaults",
        "DB_PATH": "C:\\Workspace\\Document-MCP\\data\\index.db"
      }
    }
  }
}
```

**HTTP Example**:
```json
{
  "mcpServers": {
    "obsidian-docs": {
      "transport": "http",
      "url": "http://localhost:8001/mcp",
      "headers": {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
      }
    }
  }
}
```

### Cursor / VS Code with MCP Extension

**Location**: `.cursor/mcp.json` or `.vscode/mcp.json`

```json
{
  "mcpServers": {
    "obsidian-docs": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp.server"],
      "cwd": "${workspaceFolder}/backend",
      "env": {
        "LOCAL_USER_ID": "local-dev",
        "JWT_SECRET_KEY": "local-dev-secret-key-123",
        "VAULT_BASE_PATH": "${workspaceFolder}/data/vaults",
        "DB_PATH": "${workspaceFolder}/data/index.db"
      }
    }
  }
}
```

### Cline (VS Code Extension)

Similar to Cursor configuration, but may require restarting VS Code after changes.

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `LOCAL_USER_ID` | User ID for local development | No | `local-dev` |
| `JWT_SECRET_KEY` | Secret key for JWT token signing | Yes (HTTP mode) | - |
| `VAULT_BASE_PATH` | Base directory for user vaults | Yes | - |
| `DB_PATH` | Path to SQLite database | Yes | - |
| `FASTMCP_SHOW_CLI_BANNER` | Show FastMCP banner on startup | No | `true` |

## Testing the Connection

### Test STDIO Mode

```bash
cd backend
uv run python -m src.mcp.server
```

You should see:
```
MCP server starting in STDIO mode...
Listening for MCP requests on stdin/stdout...
```

### Test HTTP Mode

1. Start the FastAPI server (port 8000):
   ```bash
   cd backend
   uv run uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. Start the MCP server in HTTP mode (port 8001) in a **separate** terminal:
   ```bash
   cd backend
   MCP_TRANSPORT=http MCP_PORT=8001 python -m src.mcp.server
   ```

3. Test the MCP endpoint:
   ```bash
   curl -X POST http://localhost:8001/mcp \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "jsonrpc": "2.0",
       "id": 1,
       "method": "tools/list",
       "params": {}
     }'
   ```

## Troubleshooting

### "Module not found" errors

- Ensure `PYTHONPATH` includes the backend directory
- Install dependencies: `cd backend && uv pip install -e .`
- Use `uv run` to ensure the virtual environment is activated

### "Authorization required" errors

- For HTTP mode, ensure you're sending a valid JWT token
- Generate a token: `POST /api/tokens` (requires authentication)
- For local development, use STDIO mode with `LOCAL_USER_ID`

### Connection refused

- Ensure the server is running on the expected port
- Check firewall settings
- For HTTP mode, verify the URL is correct

### Tools not appearing in client

- Restart the MCP client after configuration changes
- Check the client logs for error messages
- Verify the `cwd` path is absolute and correct
- Ensure all environment variables are set

## Best Practices

1. **Use absolute paths** for `cwd` and file paths in configuration
2. **Separate environments**: Use different `LOCAL_USER_ID` values for different projects
3. **Token security**: Never commit JWT tokens to version control
4. **Path handling**: Use forward slashes in JSON config (even on Windows)
5. **Testing**: Test STDIO mode locally before deploying HTTP mode

## Advanced: Docker Deployment

For production deployments, you can containerize the MCP server:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY backend/ ./backend/
RUN cd backend && pip install uv && uv pip install -e .

ENV LOCAL_USER_ID=default
ENV VAULT_BASE_PATH=/data/vaults
ENV DB_PATH=/data/index.db

CMD ["python", "-m", "backend.src.mcp.server"]
```

Then configure clients to use Docker:

```json
{
  "mcpServers": {
    "obsidian-docs": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/path/to/data:/data",
        "-e", "LOCAL_USER_ID",
        "-e", "JWT_SECRET_KEY",
        "your-image:latest"
      ],
      "env": {
        "LOCAL_USER_ID": "local-dev",
        "JWT_SECRET_KEY": "your-secret"
      }
    }
  }
}
```

