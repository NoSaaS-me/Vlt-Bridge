# Quickstart Guide: Multi-Tenant Obsidian-Like Docs Viewer

**Feature Branch**: `001-obsidian-docs-viewer`
**Created**: 2025-11-15
**Status**: Draft

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Running the Backend](#running-the-backend)
4. [Running the Frontend](#running-the-frontend)
5. [MCP Client Configuration](#mcp-client-configuration)
6. [Testing Workflows](#testing-workflows)
7. [Development Workflows](#development-workflows)
8. [Hugging Face Space Deployment](#hugging-face-space-deployment)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have the following installed:

### Required Software

- **Python**: 3.11 or higher
  - Check version: `python --version` or `python3 --version`
  - Download: https://www.python.org/downloads/

- **Node.js**: 18 or higher
  - Check version: `node --version`
  - Download: https://nodejs.org/

- **Git**: Any recent version
  - Check version: `git --version`
  - Download: https://git-scm.com/downloads/

### Optional Tools

- **Poetry** (recommended for Python dependency management): `pip install poetry`
- **Claude Desktop** or **Claude Code** (for MCP STDIO integration)
- **Docker** (for containerized deployment testing)

---

## Local Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Document-MCP
```

### 2. Create Environment Configuration

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

**`.env.example` contents**:

```bash
# Application Mode
# Options: "local" (single-user development) or "space" (multi-tenant HF Space)
MODE=local

# JWT Secret Key
# Required for HTTP/JWT auth (space mode). Remove this line entirely in local STDIO mode.
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
# JWT_SECRET_KEY=your-secret-key-change-in-production

# Vault Storage
# Base directory for per-user vault storage
VAULT_BASE_DIR=./data/vaults

# SQLite Database
# Path to SQLite database file
DB_PATH=./data/index.db

# Local Mode Configuration
# Default user ID for local development (no authentication)
LOCAL_USER_ID=local-dev

# Optional: Static Bearer Token for Local Mode
# Leave empty to disable token requirement in local mode
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
LOCAL_STATIC_TOKEN=

# Hugging Face Space Configuration (only needed for HF deployment)
# HF_OAUTH_CLIENT_ID=your-hf-client-id
# HF_OAUTH_CLIENT_SECRET=your-hf-client-secret
# HF_SPACE_HOST=https://your-space.hf.space

# Backend Server
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Frontend Development
VITE_API_BASE_URL=http://localhost:8000
```

**Generate secure secrets**:

```bash
# Generate JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate optional local static token
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Update `.env` with generated values.

### 3. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

#### Option A: Using Poetry (Recommended)

```bash
# Install Poetry if not already installed
pip install poetry

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

#### Option B: Using pip with venv

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt` example**:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
fastmcp==0.3.0
python-frontmatter==1.0.1
PyJWT==2.8.0
huggingface-hub==0.19.4
pydantic==2.5.0
python-multipart==0.0.6
aiosqlite==0.19.0
```

### 4. Initialize Database Schema

Run the database initialization script:

```bash
# From backend/ directory
python -m src.services.init_db
```

**`src/services/init_db.py` (create this file)**:

```python
"""
Database initialization script.
Creates SQLite schema and indexes.
"""
import sqlite3
import os
from pathlib import Path

def init_database(db_path: str):
    """Initialize SQLite database with schema."""
    # Ensure directory exists
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Begin transaction
    cursor.execute("BEGIN TRANSACTION")

    # Core metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS note_metadata (
            user_id TEXT NOT NULL,
            note_path TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            title TEXT NOT NULL,
            created TEXT NOT NULL,
            updated TEXT NOT NULL,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            normalized_title_slug TEXT,
            normalized_path_slug TEXT,
            PRIMARY KEY (user_id, note_path)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_user ON note_metadata(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_updated ON note_metadata(user_id, updated DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_title_slug ON note_metadata(user_id, normalized_title_slug)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_path_slug ON note_metadata(user_id, normalized_path_slug)")

    # Full-text search index
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
            user_id UNINDEXED,
            note_path UNINDEXED,
            title,
            body,
            content='',
            tokenize='porter unicode61',
            prefix='2 3'
        )
    """)

    # Tag index
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS note_tags (
            user_id TEXT NOT NULL,
            note_path TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (user_id, note_path, tag)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_user_tag ON note_tags(user_id, tag)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_user_path ON note_tags(user_id, note_path)")

    # Link graph
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS note_links (
            user_id TEXT NOT NULL,
            source_path TEXT NOT NULL,
            target_path TEXT,
            link_text TEXT NOT NULL,
            is_resolved INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, source_path, link_text)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_user_source ON note_links(user_id, source_path)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_user_target ON note_links(user_id, target_path)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_unresolved ON note_links(user_id, is_resolved)")

    # Index health tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS index_health (
            user_id TEXT PRIMARY KEY,
            note_count INTEGER NOT NULL DEFAULT 0,
            last_full_rebuild TEXT,
            last_incremental_update TEXT
        )
    """)

    # Commit transaction
    cursor.execute("COMMIT")

    conn.close()
    print(f"Database initialized at: {db_path}")

if __name__ == "__main__":
    # Load DB path from environment or use default
    db_path = os.getenv("DB_PATH", "./data/index.db")
    init_database(db_path)
```

Run initialization:

```bash
python -m src.services.init_db
```

Expected output:
```
Database initialized at: ./data/index.db
```

### 5. Create Vault Directory Structure

```bash
# From project root
mkdir -p data/vaults/local-dev

# Create a test note
cat > data/vaults/local-dev/README.md << 'EOF'
---
title: Welcome to Your Vault
tags: [getting-started]
created: 2025-01-15T10:00:00Z
updated: 2025-01-15T10:00:00Z
---

# Welcome to Your Vault

This is your personal documentation vault.

## Features

- **AI-powered writing**: Use MCP tools to create and update notes
- **Full-text search**: Search across all notes instantly
- **Wikilinks**: Link notes together with `[[Note Name]]` syntax
- **Tags**: Organize notes with frontmatter tags

## Getting Started

Try creating your first note:
- Via MCP: Use the `write_note` tool
- Via UI: Click "New Note" in the sidebar

Check out [[Quick Start Guide]] for more details.
EOF
```

### 6. Frontend Setup

Open a new terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

**`package.json` example**:

```json
{
  "name": "obsidian-docs-viewer-frontend",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:e2e": "playwright test",
    "lint": "eslint src --ext ts,tsx"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "react-markdown": "^9.0.0",
    "remark-gfm": "^4.0.0",
    "remark-frontmatter": "^5.0.0",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "@radix-ui/react-scroll-area": "^1.0.5",
    "@radix-ui/react-separator": "^1.0.3",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.1.0",
    "lucide-react": "^0.295.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@typescript-eslint/eslint-plugin": "^6.14.0",
    "@typescript-eslint/parser": "^6.14.0",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.16",
    "eslint": "^8.55.0",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.3.6",
    "typescript": "^5.2.2",
    "vite": "^5.0.8",
    "vitest": "^1.0.4",
    "@playwright/test": "^1.40.0"
  }
}
```

Configure API base URL (if not already in `.env`):

```bash
# frontend/.env
VITE_API_BASE_URL=http://localhost:8000
```

---

## Running the Backend

The backend can be run in multiple modes depending on your use case.

### Mode 1: FastAPI HTTP Server (for UI access)

Start the FastAPI development server:

```bash
# From backend/ directory (with venv activated)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Test the API**:

```bash
# Health check
curl http://localhost:8000/api/health

# Expected response:
# {"status": "ok", "mode": "local"}

# Get user info (local mode)
curl http://localhost:8000/api/me

# Expected response:
# {"user_id": "local-dev", "vault_path": "./data/vaults/local-dev"}
```

### Mode 2: MCP STDIO Server (for Claude Code/Desktop)

Start the MCP server in STDIO mode:

```bash
# From backend/ directory
python -m src.mcp.server
```

Expected output:
```
MCP server starting in STDIO mode...
Listening for MCP requests on stdin/stdout...
```

This server communicates via standard input/output and is designed to be invoked by MCP clients like Claude Desktop.

### Mode 3: MCP HTTP Server (for remote MCP access)

Start the MCP server with HTTP transport:

```bash
# From backend/ directory
python -m src.mcp.server --http --port 8001
```

Expected output:
```
MCP server starting in HTTP mode...
Server running at: http://0.0.0.0:8001
Authentication: Bearer token required
```

**Test MCP HTTP endpoint**:

```bash
# Issue a token first (if using authentication)
curl -X POST http://localhost:8000/api/tokens

# Use token to call MCP tool
curl -X POST http://localhost:8001/mcp/call \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "list_notes",
    "arguments": {}
  }'
```

### Run All Backend Services (Production Mode)

For production, run both servers together:

```bash
# Terminal 1: FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: MCP HTTP server
python -m src.mcp.server --http --port 8001
```

Or use a process manager like `supervisord` or `pm2`.

---

## Running the Frontend

Start the Vite development server:

```bash
# From frontend/ directory
npm run dev
```

Expected output:
```
  VITE v5.0.8  ready in 523 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h to show help
```

Open your browser to **http://localhost:5173**

### Local Mode UI Flow

1. **Landing page**: In local mode, you'll be automatically authenticated as `local-dev`
2. **Directory pane** (left sidebar):
   - Shows vault folder structure
   - Click on folders to expand/collapse
   - Click on notes to view content
3. **Search bar** (top of left sidebar):
   - Type to search notes
   - Results appear in dropdown with snippets
4. **Main pane** (right side):
   - View rendered Markdown
   - Click "Edit" to enter edit mode
   - Click wikilinks to navigate between notes
5. **Footer** (below main pane):
   - View tags (clickable chips)
   - See created/updated timestamps
   - View backlinks (notes that reference current note)

### Configure Static Token (Optional)

If you set `LOCAL_STATIC_TOKEN` in `.env`, configure the frontend:

**In browser console** (http://localhost:5173):

```javascript
localStorage.setItem('auth_token', 'your-static-token-from-env');
location.reload();
```

Or update `frontend/src/services/auth.ts` to automatically use the token in local mode.

---

## MCP Client Configuration

### Claude Desktop (STDIO)

Configure Claude Desktop to use the MCP STDIO server.

**Config file location**:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

**`claude_desktop_config.json`**:

```json
{
  "mcpServers": {
    "obsidian-docs": {
      "command": "python",
      "args": [
        "-m",
        "src.mcp.server"
      ],
      "cwd": "/absolute/path/to/Document-MCP/backend",
      "env": {
        "MODE": "local",
        "VAULT_BASE_DIR": "/absolute/path/to/Document-MCP/data/vaults",
        "DB_PATH": "/absolute/path/to/Document-MCP/data/index.db",
        "LOCAL_USER_ID": "local-dev"
      }
    }
  }
}
```

**Important**: Replace `/absolute/path/to/Document-MCP` with your actual project path.

**Test in Claude Desktop**:

1. Restart Claude Desktop
2. Open a conversation
3. Try: "List all notes in my vault"
4. Claude should call the `list_notes` MCP tool

### Claude Code (STDIO)

Claude Code uses a similar configuration. Create or edit `~/.claude/mcp_config.json`:

```json
{
  "mcpServers": {
    "obsidian-docs": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/absolute/path/to/Document-MCP/backend",
      "env": {
        "MODE": "local",
        "VAULT_BASE_DIR": "/absolute/path/to/Document-MCP/data/vaults",
        "DB_PATH": "/absolute/path/to/Document-MCP/data/index.db"
      }
    }
  }
}
```

`JWT_SECRET_KEY` is optional for STDIO mode—omit it from the environment when running locally without HTTP authentication.

### MCP HTTP Transport (Remote Access)

For remote access (e.g., from HF Space or external tools), use HTTP transport.

**Setup**:

1. Start MCP HTTP server:
   ```bash
   python -m src.mcp.server --http --port 8001
   ```

2. Issue a JWT token:
   ```bash
   curl -X POST http://localhost:8000/api/tokens
   ```

   Response:
   ```json
   {
     "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
     "token_type": "bearer",
     "expires_at": "2025-04-15T10:30:00Z"
   }
   ```

3. Configure MCP client with HTTP endpoint:

**Example HTTP MCP client configuration** (pseudo-code):

```json
{
  "mcp_endpoint": "http://localhost:8001/mcp",
  "auth": {
    "type": "bearer",
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

**Manual HTTP MCP call**:

```bash
curl -X POST http://localhost:8001/mcp/call \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "write_note",
    "arguments": {
      "path": "test/hello.md",
      "title": "Hello World",
      "body": "# Hello World\n\nThis is a test note created via MCP HTTP."
    }
  }'
```

Expected response:
```json
{
  "status": "ok",
  "path": "test/hello.md"
}
```

---

## Testing Workflows

### Backend Unit Tests

Run pytest for unit tests:

```bash
# From backend/ directory
pytest tests/unit -v
```

**Example test structure**:

```
backend/tests/
├── unit/
│   ├── test_vault.py          # Vault service tests
│   ├── test_indexer.py        # Indexer service tests
│   ├── test_auth.py           # Auth service tests
│   └── test_wikilink.py       # Wikilink resolution tests
├── integration/
│   ├── test_api_notes.py      # API integration tests
│   ├── test_api_search.py     # Search API tests
│   └── test_mcp_tools.py      # MCP tool integration tests
└── contract/
    ├── test_http_api_contract.py   # OpenAPI contract validation
    └── test_mcp_contract.py        # MCP tool schema validation
```

**Run specific test file**:

```bash
pytest tests/unit/test_vault.py -v
```

**Expected output**:

```
tests/unit/test_vault.py::test_create_note PASSED
tests/unit/test_vault.py::test_read_note PASSED
tests/unit/test_vault.py::test_update_note PASSED
tests/unit/test_vault.py::test_delete_note PASSED
tests/unit/test_vault.py::test_path_validation PASSED
tests/unit/test_vault.py::test_path_traversal_blocked PASSED

====== 6 passed in 0.23s ======
```

### Backend Integration Tests

Test full API workflows:

```bash
# From backend/ directory
pytest tests/integration -v
```

Expected output:
```
tests/integration/test_api_notes.py::test_create_and_read_note PASSED
tests/integration/test_api_notes.py::test_update_note_with_version PASSED
tests/integration/test_api_notes.py::test_version_conflict_409 PASSED
tests/integration/test_api_search.py::test_full_text_search PASSED
tests/integration/test_api_search.py::test_search_ranking PASSED
tests/integration/test_mcp_tools.py::test_write_note_tool PASSED
tests/integration/test_mcp_tools.py::test_search_notes_tool PASSED

====== 7 passed in 1.45s ======
```

### Backend Contract Tests

Validate API contracts against OpenAPI spec:

```bash
pytest tests/contract -v
```

Expected output:
```
tests/contract/test_http_api_contract.py::test_openapi_spec_valid PASSED
tests/contract/test_http_api_contract.py::test_all_endpoints_match_spec PASSED
tests/contract/test_mcp_contract.py::test_mcp_tool_schemas_valid PASSED

====== 3 passed in 0.34s ======
```

### Frontend Unit Tests

Run Vitest for component tests:

```bash
# From frontend/ directory
npm test
```

**Example test structure**:

```
frontend/tests/
├── unit/
│   ├── DirectoryTree.test.tsx
│   ├── NoteViewer.test.tsx
│   ├── NoteEditor.test.tsx
│   ├── SearchBar.test.tsx
│   └── wikilink.test.ts
└── e2e/
    ├── note-crud.spec.ts
    ├── search.spec.ts
    └── navigation.spec.ts
```

**Run specific test**:

```bash
npm test -- DirectoryTree.test.tsx
```

Expected output:
```
✓ tests/unit/DirectoryTree.test.tsx (5)
  ✓ renders folder tree structure
  ✓ expands/collapses folders on click
  ✓ selects note on click
  ✓ highlights selected note
  ✓ updates tree when notes change

Test Files  1 passed (1)
Tests  5 passed (5)
```

### Frontend E2E Tests

Run Playwright for end-to-end tests:

```bash
# From frontend/ directory
npm run test:e2e
```

**Setup Playwright** (first time):

```bash
npx playwright install
```

Expected output:
```
Running 8 tests using 4 workers

  ✓ [chromium] › note-crud.spec.ts:3:1 › create new note (1.2s)
  ✓ [chromium] › note-crud.spec.ts:15:1 › edit existing note (0.9s)
  ✓ [chromium] › note-crud.spec.ts:28:1 › delete note (0.7s)
  ✓ [chromium] › search.spec.ts:3:1 › search notes by title (0.8s)
  ✓ [chromium] › search.spec.ts:12:1 › search notes by content (0.9s)
  ✓ [chromium] › navigation.spec.ts:3:1 › navigate via directory tree (0.6s)
  ✓ [chromium] › navigation.spec.ts:14:1 › navigate via wikilink click (1.1s)
  ✓ [chromium] › navigation.spec.ts:25:1 › view backlinks (0.8s)

  8 passed (6.0s)
```

---

## Development Workflows

### Workflow 1: AI Agent Writes a Note via MCP

**Scenario**: Use Claude Desktop to create a new design document.

1. **Open Claude Desktop** (with MCP configured)

2. **Prompt Claude**:
   ```
   Create a new note at "design/api-authentication.md" with the following:

   Title: API Authentication Design
   Tags: backend, security, api

   Content:
   # API Authentication Design

   ## Overview
   This document describes the authentication strategy for our API.

   ## JWT Token Flow
   1. User authenticates via HF OAuth
   2. Server issues JWT with 90-day expiration
   3. Client includes token in Authorization header

   ## Token Validation
   - Validate signature using HS256
   - Check expiration timestamp
   - Extract user_id from 'sub' claim

   See [[Security Best Practices]] for more details.
   ```

3. **Claude executes**:
   - Calls `write_note` MCP tool
   - Creates file at `data/vaults/local-dev/design/api-authentication.md`
   - Writes frontmatter with title, tags, timestamps
   - Writes markdown body
   - Updates index (full-text search, tag index, wikilink graph)

4. **Verify in UI**:
   - Refresh browser at http://localhost:5173
   - Expand "design" folder in directory tree
   - Click "api-authentication.md"
   - See rendered note with wikilink to "Security Best Practices"

### Workflow 2: Human Edits a Note in UI

**Scenario**: Fix a typo in the note created above.

1. **Navigate to note**:
   - Open http://localhost:5173
   - Click on "design/api-authentication.md" in directory tree

2. **Enter edit mode**:
   - Click "Edit" button in top-right
   - UI switches to split view: markdown editor (left) and live preview (right)

3. **Make changes**:
   - Fix typo: "authenication" → "authentication"
   - Add a new section:
     ```markdown
     ## Token Expiration
     Tokens expire after 90 days. Users must re-authenticate.
     ```

4. **Save changes**:
   - Click "Save" button
   - UI sends `PUT /api/notes/design/api-authentication.md` with `if_version: 1`
   - Backend increments version to 2, updates timestamp
   - UI switches back to read mode with updated content

5. **Verify version tracking**:
   - Check footer: "Updated: <timestamp>"
   - Try editing again (version is now 2)

### Workflow 3: Search for Notes

**Scenario**: Find all notes about "authentication".

1. **Use search bar**:
   - Type "authentication" in search bar (top of left sidebar)
   - Search debounces and calls `GET /api/search?q=authentication`

2. **View results**:
   - Results dropdown shows:
     - "API Authentication Design" (title match, high score)
     - "Security Best Practices" (body match, lower score)
   - Snippets highlight matching text: "...API <mark>authentication</mark> strategy..."

3. **Navigate to result**:
   - Click on "API Authentication Design"
   - Main pane renders the note

### Workflow 4: Follow Wikilinks and View Backlinks

**Scenario**: Navigate from "API Authentication Design" to "Security Best Practices" via wikilink.

1. **View note with wikilink**:
   - Open "design/api-authentication.md"
   - See wikilink: `[[Security Best Practices]]` rendered as clickable link

2. **Click wikilink**:
   - UI resolves slug: "security-best-practices"
   - Navigates to "security-best-practices.md" (or prompts to create if not exists)

3. **View backlinks**:
   - Footer shows "Referenced by:" section
   - Lists "design/api-authentication.md" as a backlink
   - Click backlink to navigate back

### Workflow 5: Rebuild Index

**Scenario**: Manually added notes outside the app, need to rebuild index.

1. **Add notes manually**:
   ```bash
   # From terminal
   cat > data/vaults/local-dev/notes/meeting-2025-01-16.md << 'EOF'
   ---
   title: Team Meeting Notes
   tags: [meetings]
   ---

   # Team Meeting - Jan 16, 2025

   ## Attendees
   - Alice, Bob, Charlie

   ## Topics
   - Reviewed [[API Authentication Design]]
   - Discussed deployment strategy
   EOF
   ```

2. **Rebuild index**:
   ```bash
   curl -X POST http://localhost:8000/api/index/rebuild
   ```

   Response:
   ```json
   {
     "status": "ok",
     "note_count": 3,
     "rebuilt_at": "2025-01-16T14:30:00Z"
   }
   ```

3. **Verify in UI**:
   - Refresh browser
   - See new note in directory tree
   - Note is searchable
   - Backlink from "API Authentication Design" now shows "Team Meeting Notes"

---

## Hugging Face Space Deployment

Deploy the application to Hugging Face Spaces for multi-tenant access.

### Prerequisites

1. **HuggingFace account**: https://huggingface.co/join
2. **Create a Space**:
   - Go to https://huggingface.co/new-space
   - Choose "Docker" as SDK
   - Enable "OAuth" in Space settings

### 1. Configure OAuth

In your Space settings, enable OAuth and note:
- **Client ID**: `hf_oauth_client_id_xxxxx`
- **Client Secret**: `hf_oauth_client_secret_xxxxx`

### 2. Set Environment Variables

In Space settings → Variables, add:

| Variable | Value | Visibility |
|----------|-------|------------|
| `MODE` | `space` | Public |
| `JWT_SECRET_KEY` | (generate secure secret) | Secret |
| `VAULT_BASE_DIR` | `/data/vaults` | Public |
| `DB_PATH` | `/data/index.db` | Public |
| `HF_OAUTH_CLIENT_ID` | (from OAuth settings) | Secret |
| `HF_OAUTH_CLIENT_SECRET` | (from OAuth settings) | Secret |
| `HF_SPACE_HOST` | `https://your-space.hf.space` | Public |

### 3. Create Dockerfile

**`Dockerfile`**:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Node.js for frontend build
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy backend
COPY backend/ /app/backend/
WORKDIR /app/backend
RUN pip install --no-cache-dir -r requirements.txt

# Initialize database
RUN python -m src.services.init_db

# Copy frontend
COPY frontend/ /app/frontend/
WORKDIR /app/frontend

# Build frontend
RUN npm install && npm run build

# Copy built frontend to backend static directory
RUN mkdir -p /app/backend/static && \
    cp -r /app/frontend/dist/* /app/backend/static/

# Set working directory back to backend
WORKDIR /app/backend

# Create data directories
RUN mkdir -p /data/vaults

# Expose ports
EXPOSE 7860 8001

# Start backend (FastAPI + MCP HTTP)
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 7860 & python -m src.mcp.server --http --port 8001 & wait"]
```

### 4. Configure Backend to Serve Frontend

**`backend/src/api/main.py`** (add static file serving):

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Obsidian Docs Viewer API")

# ... (existing API routes) ...

# Serve frontend static files
static_dir = os.path.join(os.path.dirname(__file__), "../../static")
if os.path.exists(static_dir):
    app.mount("/assets", StaticFiles(directory=f"{static_dir}/assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend SPA (fallback to index.html for client-side routing)."""
        if full_path.startswith("api/"):
            # API routes handled by existing endpoints
            return {"error": "Not found"}

        file_path = os.path.join(static_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)

        # Fallback to index.html for SPA routing
        return FileResponse(f"{static_dir}/index.html")
```

### 5. Build Frontend with HF Space URL

Update **`frontend/.env.production`**:

```bash
VITE_API_BASE_URL=https://your-space.hf.space
```

Build:

```bash
cd frontend
npm run build
```

### 6. Push to HF Space

```bash
# From project root
git init
git add .
git commit -m "Initial commit"

# Add HF Space remote
git remote add space https://huggingface.co/spaces/your-username/your-space-name
git push space main
```

### 7. Verify Deployment

1. Visit `https://your-space.hf.space`
2. Click "Sign in with Hugging Face"
3. Authorize the app
4. You should see the main UI with your isolated vault

### 8. Configure MCP HTTP Client for HF Space

1. Sign in to the Space and navigate to Settings
2. Click "Issue API Token"
3. Copy the JWT token
4. Configure your local MCP client:

**MCP HTTP client config**:

```json
{
  "mcp_endpoint": "https://your-space.hf.space:8001/mcp",
  "auth": {
    "type": "bearer",
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

5. Test from command line:

```bash
curl -X POST https://your-space.hf.space:8001/mcp/call \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "list_notes",
    "arguments": {}
  }'
```

---

## Troubleshooting

### Common Errors

#### Error: `401 Unauthorized - Invalid token`

**Cause**: JWT token is invalid, expired, or missing.

**Solution**:
1. Check token expiration:
   ```bash
   # Decode JWT (requires `pyjwt` library)
   python -c "import jwt; print(jwt.decode('your-token', options={'verify_signature': False}))"
   ```
2. Issue a new token:
   ```bash
   curl -X POST http://localhost:8000/api/tokens
   ```
3. Update token in client configuration

#### Error: `400 Bad Request - Path traversal blocked`

**Cause**: Note path contains `..` or absolute path components.

**Example**: `write_note(path="../etc/passwd")`

**Solution**: Use relative paths only, no `..` allowed
```python
# ✅ Correct
write_note(path="design/api.md")

# ❌ Incorrect
write_note(path="../design/api.md")
write_note(path="/design/api.md")
```

#### Error: `409 Conflict - Version mismatch`

**Cause**: Note was updated by another user/agent since you last read it.

**Example**:
1. You load note with version 5
2. Claude updates it via MCP (version → 6)
3. You click "Save" with `if_version: 5`
4. Server returns `409 Conflict`

**Solution**:
1. Reload the note to get latest version
2. Re-apply your changes
3. Save with new version number

**UI handling**:
```typescript
// In NoteEditor component
try {
  await updateNote(path, { body, if_version: currentVersion });
} catch (error) {
  if (error.status === 409) {
    alert("This note changed since you opened it. Please reload before saving.");
    // Optionally: reload note automatically
  }
}
```

#### Error: `413 Payload Too Large`

**Cause**: Note content exceeds 1 MiB (1,048,576 bytes).

**Solution**: Split large notes into smaller files
```bash
# Check note size
wc -c data/vaults/local-dev/large-note.md

# If > 1 MiB, split into multiple notes
```

#### Error: `403 Forbidden - Vault note limit exceeded`

**Cause**: Vault has reached 5,000 note limit.

**Solution**:
1. Delete unused notes
2. Archive old notes to external storage
3. Request limit increase (requires code change in `FR-008`)

#### Error: `SQLite database is locked`

**Cause**: Concurrent writes to SQLite database.

**Solution**:
1. Ensure only one backend process is running
2. Use connection pooling (in production)
3. Consider upgrading to PostgreSQL for high concurrency

**Quick fix**:
```bash
# Kill all backend processes
pkill -f uvicorn
pkill -f "python -m src.mcp.server"

# Restart backend
uvicorn src.api.main:app --reload
```

#### Error: `FTS5 extension not available`

**Cause**: SQLite was compiled without FTS5 support.

**Solution**:
1. Check if FTS5 is available:
   ```python
   import sqlite3
   conn = sqlite3.connect(":memory:")
   cursor = conn.cursor()
   cursor.execute("PRAGMA compile_options")
   print([row[0] for row in cursor.fetchall()])
   # Look for 'ENABLE_FTS5' in output
   ```

2. If not available, reinstall SQLite with FTS5:
   ```bash
   # macOS (Homebrew)
   brew reinstall sqlite3

   # Linux (Ubuntu/Debian)
   sudo apt-get install --reinstall libsqlite3-0

   # Python (rebuild pysqlite3 with FTS5)
   pip install pysqlite3-binary
   ```

### Debug Mode

Enable debug logging for troubleshooting:

**Backend** (`.env`):
```bash
LOG_LEVEL=DEBUG
```

**Frontend** (`frontend/.env`):
```bash
VITE_DEBUG=true
```

**View logs**:

```bash
# Backend logs (if using uvicorn)
tail -f backend.log

# Or run with verbose output
uvicorn src.api.main:app --reload --log-level debug
```

**Frontend logs**: Open browser DevTools → Console

### Health Checks

#### Check API Health

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "ok",
  "mode": "local",
  "vault_base_dir": "./data/vaults",
  "db_path": "./data/index.db"
}
```

#### Check Index Health

```bash
curl http://localhost:8000/api/index/health
```

Expected response:
```json
{
  "user_id": "local-dev",
  "note_count": 15,
  "last_full_rebuild": "2025-01-15T10:00:00Z",
  "last_incremental_update": "2025-01-16T14:30:00Z"
}
```

#### Verify MCP Server

```bash
# Test STDIO server (requires MCP client)
echo '{"tool": "list_notes", "arguments": {}}' | python -m src.mcp.server

# Test HTTP server
curl -X POST http://localhost:8001/mcp/call \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"tool": "list_notes", "arguments": {}}'
```

### Performance Profiling

#### Backend API Response Time

```bash
# Use curl with timing
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/notes

# curl-format.txt:
# time_namelookup:  %{time_namelookup}s\n
# time_connect:     %{time_connect}s\n
# time_appconnect:  %{time_appconnect}s\n
# time_total:       %{time_total}s\n
```

Expected: `time_total < 2s` for directory listing

#### Search Query Performance

```bash
time curl -s "http://localhost:8000/api/search?q=authentication" > /dev/null
```

Expected: `< 1s` for vaults with up to 5,000 notes

#### Index Rebuild Time

```bash
time curl -X POST http://localhost:8000/api/index/rebuild
```

Expected: `< 30s` for 1,000 notes

---

## Next Steps

1. **Read the full specification**: See `specs/001-obsidian-docs-viewer/spec.md`
2. **Review data model**: See `specs/001-obsidian-docs-viewer/data-model.md`
3. **Check API contracts**: See `specs/001-obsidian-docs-viewer/contracts/`
4. **Start implementation**: Run `/speckit.tasks` to generate implementation tasks

---

## Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **FastMCP Documentation**: https://github.com/jlowin/fastmcp
- **React + shadcn/ui**: https://ui.shadcn.com/
- **HuggingFace Spaces**: https://huggingface.co/docs/hub/spaces
- **MCP Protocol**: https://modelcontextprotocol.io/

---

**Last Updated**: 2025-11-15
**Status**: Ready for implementation
