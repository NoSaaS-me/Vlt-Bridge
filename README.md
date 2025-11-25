# Document Viewer

A multi-tenant Obsidian-like documentation system with AI agent integration via Model Context Protocol (MCP).

## 🎯 Overview

Document Viewer enables both humans and AI agents to create, browse, and search documentation with powerful features like:

- 📝 **Markdown Notes** with YAML frontmatter
- 🔗 **Wikilinks** - `[[Note Name]]` style internal linking with auto-resolution
- 🔍 **Full-Text Search** - BM25 ranking with recency bonus
- ↩️ **Backlinks** - Automatic tracking of which notes reference each other
- 🏷️ **Tags** - Organize notes with frontmatter tags
- ✏️ **Split-Pane Editor** - Live markdown preview with optimistic concurrency
- 🤖 **MCP Integration** - AI agents can read/write docs via FastMCP
- 👥 **Multi-Tenant** - Isolated vaults per user (production ready with HF OAuth)

## 🏗️ Tech Stack

### Backend
- **FastAPI** - HTTP API server
- **FastMCP** - MCP server for AI agent integration
- **SQLite FTS5** - Full-text search with BM25 ranking
- **python-frontmatter** - YAML frontmatter parsing
- **PyJWT** - Token-based authentication

### Frontend
- **React + Vite** - Modern web framework
- **shadcn/ui** - Beautiful UI components
- **Tailwind CSS** - Utility-first styling
- **react-markdown** - Markdown rendering with custom wikilink support
- **TypeScript** - Type-safe frontend code

## 📦 Local Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- `uv` (Python package manager) or `pip`

### 1. Clone Repository

```bash
git clone <repository-url>
cd Document-MCP
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
uv venv
# or: python -m venv .venv

# Install dependencies
uv pip install -e .
# or: .venv/bin/pip install -e .

# Initialize database
cd ..
VIRTUAL_ENV=backend/.venv backend/.venv/bin/python -c "from backend.src.services.database import init_database; init_database()"
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

### 4. Environment Configuration

The project includes development scripts that set environment variables automatically. For manual configuration, create a `.env` file in the backend directory:

```bash
# backend/.env
JWT_SECRET_KEY=your-secret-key-here
VAULT_BASE_PATH=/path/to/Document-MCP/data/vaults
```

See `.env.example` for all available options.

## 🚀 Running the Application

### Easy Start (Recommended)

Use the provided scripts to start both servers:

```bash
# Start frontend and backend
./start-dev.sh

# Check status
./status-dev.sh

# Stop servers
./stop-dev.sh

# View logs
tail -f backend.log frontend.log
```

### Manual Start

#### Running Backend

Start the HTTP API server:

```bash
cd backend
JWT_SECRET_KEY="local-dev-secret-key-123" \
VAULT_BASE_PATH="$(pwd)/../data/vaults" \
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at: `http://localhost:8000`

API docs (Swagger): `http://localhost:8000/docs`

#### Running MCP Server (STDIO Mode)

For AI agent integration via MCP:

```bash
cd backend
JWT_SECRET_KEY="local-dev-secret-key-123" \
VAULT_BASE_PATH="$(pwd)/../data/vaults" \
.venv/bin/python -m src.mcp.server
```

#### Running Frontend

```bash
cd frontend
npm run dev
```

Frontend will be available at: `http://localhost:5173`

## 🤖 MCP Client Configuration

To use the Document Viewer with AI agents (Claude Desktop, Cline, etc.), add this to your MCP configuration:

### Claude Desktop / Cline

Add to `~/.cursor/mcp.json` (or Claude Desktop settings):

```json
{
  "mcpServers": {
    "obsidian-docs": {
      "command": "python",
      "args": ["-m", "backend.src.mcp.server"],
      "cwd": "/path/to/Document-MCP",
      "env": {
        "BEARER_TOKEN": "local-dev-token",
        "FASTMCP_SHOW_CLI_BANNER": "false",
        "PYTHONPATH": "/path/to/Document-MCP",
        "JWT_SECRET_KEY": "local-dev-secret-key-123",
        "VAULT_BASE_PATH": "/path/to/Document-MCP/data/vaults"
      }
    }
  }
}
```

**Note:** In production, use actual JWT tokens instead of `local-dev-token`.

### Available MCP Tools

AI agents can use these tools:

- `list_notes` - List all notes in vault
- `read_note` - Read a specific note
- `write_note` - Create or update a note
- `delete_note` - Remove a note
- `search_notes` - Full-text search with BM25 ranking
- `get_backlinks` - Find notes linking to a target
- `get_tags` - List all tags with usage counts

## 🏛️ Architecture

### Data Model

**Note Structure:**
```yaml
---
title: My Note
tags: [guide, tutorial]
created: 2025-01-15T10:00:00Z
updated: 2025-01-15T14:30:00Z
---

# My Note

Content with [[Wikilinks]] to other notes.
```

**Vault Structure:**
```
data/vaults/
├── local-dev/           # Development user vault
│   ├── Getting Started.md
│   ├── API Documentation.md
│   └── ...
└── {user_id}/          # Production user vaults
    └── *.md
```

**Index Tables (SQLite):**
- `note_metadata` - Note versions, titles, timestamps
- `note_fts` - FTS5 full-text search index
- `note_tags` - Tag associations
- `note_links` - Wikilink graph (resolved/unresolved)
- `index_health` - Index statistics per user

### Key Features

**Wikilink Resolution:**
- Normalizes titles to slugs: `[[Getting Started]]` → `getting-started`
- Matches against both title and filename
- Prefers same-folder matches
- Tracks broken links for UI styling

**Search Ranking:**
- BM25 algorithm with title-weighted scoring (3x title, 1x body)
- Recency bonus: +1.0 for notes updated in last 7 days, +0.5 for last 30 days
- Returns highlighted snippets with `<mark>` tags

**Optimistic Concurrency:**
- Version-based conflict detection for note edits
- Prevents data loss from concurrent edits
- Returns 409 Conflict with helpful message

## 🔒 Authentication

### Local Development
Uses a static token: `local-dev-token`

### Production (Hugging Face OAuth)
- Multi-tenant with per-user isolated vaults
- JWT tokens with user_id claims
- Automatic vault initialization on first login

See deployment documentation for HF OAuth setup.

## 📊 Performance Considerations

**SQLite Optimizations:**
- FTS5 with prefix indexes (`prefix='2 3'`) for fast autocomplete and substring matching
- Recommended: Enable WAL mode for concurrent reads/writes:
  ```sql
  PRAGMA journal_mode=WAL;
  PRAGMA synchronous=NORMAL;
  ```
- Normalized slug indexes (`normalized_title_slug`, `normalized_path_slug`) for O(1) wikilink resolution
- BM25 ranking weights: 3.0 for title matches, 1.0 for body matches

**Rate Limiting:**
- ⚠️ **Production Recommendation**: Add per-user rate limits to prevent abuse
- API endpoints currently have no rate limiting
- Consider implementing:
  - `/api/notes` (POST): 100 requests/hour per user
  - `/api/index/rebuild` (POST): 10 requests/day per user
  - `/api/search`: 1000 requests/hour per user
- Use libraries like `slowapi` or Redis-based rate limiting

**Scaling:**
- **Single-server**: SQLite handles 100K+ notes efficiently
- **Multi-server**: Migrate to PostgreSQL with `pg_trgm` or `pgvector` for FTS
- **Caching**: Add Redis for:
  - Session tokens (reduce DB lookups)
  - Frequently accessed notes
  - Search result caching (TTL: 5 minutes)
- **CDN**: Serve frontend assets via CDN for global performance

## 🧪 Development

### Project Structure

```
Document-MCP/
├── backend/
│   ├── src/
│   │   ├── api/          # FastAPI routes & middleware
│   │   ├── mcp/          # FastMCP server
│   │   ├── models/       # Pydantic models
│   │   └── services/     # Business logic
│   └── tests/            # Backend tests
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components
│   │   ├── services/     # API client
│   │   └── types/        # TypeScript types
│   └── tests/            # Frontend tests
├── data/
│   ├── vaults/          # User markdown files
│   └── index.db         # SQLite database
├── specs/               # Feature specifications
└── start-dev.sh         # Development startup script
```

### Adding a New Note (via UI)

1. Click "New Note" button
2. Enter note name (`.md` extension optional)
3. Edit in split-pane editor
4. Save with Cmd/Ctrl+S

### Adding a New Note (via MCP)

```python
# AI agent writes a note
write_note(
    path="guides/my-guide.md",
    body="# My Guide\n\nContent here with [[links]]",
    title="My Guide",
    metadata={"tags": ["guide", "tutorial"]}
)
```

## 🐛 Troubleshooting

**Backend won't start:**
- Ensure virtual environment is activated
- Check environment variables are set
- Verify database is initialized

**Frontend shows connection errors:**
- Ensure backend is running on port 8000
- Check Vite proxy configuration in `frontend/vite.config.ts`

**Search returns no results:**
- Verify notes are indexed (check Settings → Index Health)
- Try rebuilding the index via Settings page

**MCP tools not showing in Claude:**
- Verify MCP configuration path is correct
- Check `PYTHONPATH` includes project root
- Restart Claude Desktop after config changes

## 📝 License

[Add license information]

## 🤝 Contributing

[Add contributing guidelines]

## 📧 Contact

[Add contact information]

