"""Seed demo vault with sample documentation."""

from __future__ import annotations

import logging

from .database import init_database
from .database import DatabaseService
from .indexer import IndexerService
from .vault import VaultService

logger = logging.getLogger(__name__)

# Demo notes with wikilinks and tags
DEMO_NOTES = [
    {
        "path": "Getting Started.md",
        "title": "Getting Started",
        "tags": ["guide", "intro"],
        "body": """# Getting Started

Welcome to the Document Viewer! This is an AI-powered documentation system with wikilinks, full-text search, and backlinks.

## Key Features

- **Wikilinks**: Link between notes using `[[Note Name]]` syntax
- **Full-Text Search**: Powered by SQLite FTS5 with BM25 ranking
- **Backlinks**: Automatically track which notes reference each other
- **MCP Integration**: AI agents can read and write docs via [[MCP Integration]]
- **Multi-Tenant**: Each user has an isolated vault

## Next Steps

1. Browse the [[API Documentation]]
2. Learn about [[Wikilink Examples]]
3. Understand the [[Architecture Overview]]
4. Check out [[Search Features]]

## Demo Mode

⚠️ This is a **demo instance** - all data is temporary and resets on server restart."""
    },
    {
        "path": "API Documentation.md",
        "title": "API Documentation",
        "tags": ["api", "reference"],
        "body": """# API Documentation

The Document Viewer exposes a REST API for managing notes and searching.

## Authentication

All API requests require a `Bearer` token in the `Authorization` header:

```
Authorization: Bearer <your-jwt-token>
```

Get your token from [[Settings]] after signing in with Hugging Face OAuth.

## Endpoints

### Notes

- `GET /api/notes` - List all notes in your vault
- `GET /api/notes/{path}` - Get a specific note
- `POST /api/notes` - Create a new note
- `PUT /api/notes/{path}` - Update an existing note

### Search

- `GET /api/search?q=query` - Full-text search with [[Search Features]]
- `GET /api/backlinks/{path}` - Get notes that link to this note
- `GET /api/tags` - List all tags with counts

### Index Management

- `GET /api/index/health` - Check index status
- `POST /api/index/rebuild` - Rebuild the search index

## Related

- [[MCP Integration]] - AI agent access via MCP
- [[Wikilink Examples]] - How to use wikilinks"""
    },
    {
        "path": "MCP Integration.md",
        "title": "MCP Integration",
        "tags": ["mcp", "ai", "integration"],
        "body": """# MCP Integration

The Model Context Protocol (MCP) allows AI agents like Claude to interact with your documentation vault.

## Available Tools

The MCP server exposes these tools:

- `list_notes` - List all notes in the vault
- `read_note` - Read a specific note with metadata
- `write_note` - Create or update a note
- `delete_note` - Remove a note
- `search_notes` - Full-text search with ranking
- `get_backlinks` - Find notes linking to a target
- `get_tags` - List all tags

## Configuration

For **local development** (STDIO mode), see [[Getting Started]].

For **HTTP mode** (HF Space), use:

```json
{
  "mcpServers": {
    "obsidian-docs": {
      "url": "https://huggingface.co/spaces/bigwolfe/Document-MCP/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_JWT_TOKEN"
      }
    }
  }
}
```

## Related

- [[API Documentation]] - REST API reference
- [[Architecture Overview]] - System design"""
    },
    {
        "path": "Wikilink Examples.md",
        "title": "Wikilink Examples",
        "tags": ["guide", "wikilinks"],
        "body": """# Wikilink Examples

Wikilinks are a powerful way to connect notes using simple `[[bracket]]` syntax.

## Basic Wikilinks

Link to other notes by title:

- [[Getting Started]]
- [[API Documentation]]
- [[MCP Integration]]

## How It Works

Wikilinks are resolved using **normalized slug matching**:

1. The link text is normalized (lowercase, spaces to dashes)
2. The system checks note titles and filenames
3. Same-folder matches are preferred
4. Broken links are styled differently

## Broken Links

If you create a link to a note that doesn't exist, like [[Nonexistent Note]], it will be styled as a broken link.

## Backlinks

When you link to a note, it automatically appears in that note's backlinks section. Try viewing [[Getting Started]] to see this note in its backlinks!

## Advanced Features

- Links are indexed for fast resolution
- The [[Search Features]] include wikilink graph analysis
- See [[Architecture Overview]] for implementation details"""
    },
    {
        "path": "Architecture Overview.md",
        "title": "Architecture Overview",
        "tags": ["architecture", "technical"],
        "body": """# Architecture Overview

The Document Viewer is built with a modern tech stack optimized for AI-human collaboration.

## Tech Stack

### Backend

- **FastAPI** - HTTP API server
- **FastMCP** - MCP server for AI agents
- **SQLite FTS5** - Full-text search engine
- **python-frontmatter** - YAML metadata parsing

### Frontend

- **React + Vite** - Modern web framework
- **shadcn/ui** - Beautiful UI components
- **Tailwind CSS** - Utility-first styling
- **react-markdown** - Markdown rendering

## Data Model

### Notes

Each note is a Markdown file with optional YAML frontmatter:

```yaml
---
title: My Note
tags: [guide, tutorial]
created: 2025-01-15T10:00:00Z
updated: 2025-01-15T14:30:00Z
---

# Note content here
```

### Indexing

The system maintains several indexes:

- **note_metadata** - Versions, titles, timestamps
- **note_fts** - Full-text search (SQLite FTS5)
- **note_tags** - Tag associations
- **note_links** - Wikilink graph

See [[Search Features]] for ranking details.

## Multi-Tenancy

Each user gets an isolated vault at `data/vaults/{user_id}/`. See [[API Documentation]] for authentication."""
    },
    {
        "path": "Search Features.md",
        "title": "Search Features",
        "tags": ["search", "features"],
        "body": """# Search Features

The Document Viewer includes powerful full-text search with intelligent ranking.

## BM25 Ranking

Search uses the BM25 algorithm with custom weights:

- **Title matches**: 3x weight
- **Body matches**: 1x weight
- **Recency bonus**: +1.0 for notes updated in last 7 days, +0.5 for last 30 days

## Search Syntax

Just type natural language queries:

- `authentication` - Find all notes mentioning authentication
- `api design` - Multiple terms are combined
- Searches are tokenized and case-insensitive

## Index Health

Check the footer of the main app to see:

- Total note count
- Last index update timestamp

Rebuild the index from [[Settings]] if needed.

## Related

- [[API Documentation]] - Search API endpoints
- [[Architecture Overview]] - Technical implementation
- [[Wikilink Examples]] - Linking between notes"""
    },
    {
        "path": "Settings.md",
        "title": "Settings",
        "tags": ["settings", "config"],
        "body": """# Settings

Access settings from the main app to manage your vault and API access.

## User Profile

View your authenticated user ID and account information.

## API Token

Your JWT token for API and MCP access:

- Copy the token to configure MCP clients
- Token expires after 7 days
- Re-authenticate to get a new token

See [[MCP Integration]] for configuration examples.

## Index Health

Monitor your vault's search index:

- **Note count**: Total indexed notes
- **Last rebuild**: Full index rebuild timestamp
- **Last update**: Most recent incremental update

Use the **Rebuild Index** button if:

- Search results seem outdated
- You manually edited files outside the app
- Index health looks unhealthy

## Related

- [[Getting Started]] - First steps
- [[API Documentation]] - Using the API"""
    },
    {
        "path": "guides/Quick Reference.md",
        "title": "Quick Reference",
        "tags": ["guide", "reference"],
        "body": """# Quick Reference

A cheat sheet for common tasks in the Document Viewer.

## Creating Notes

1. Click "New Note" button
2. Enter note name (`.md` extension optional)
3. Write content in split-pane editor
4. Click "Save"

## Editing Notes

1. Navigate to a note
2. Click "Edit" button
3. Modify in left pane, preview in right pane
4. Click "Save" (handles version conflicts automatically)

## Using Wikilinks

Link to other notes: `[[Note Title]]`

See [[Wikilink Examples]] for more details.

## Searching

Use the search bar at the top of the directory pane. Results are ranked by relevance and recency.

Learn more in [[Search Features]].

## MCP Access

Get your API token from [[Settings]] and configure your MCP client per [[MCP Integration]]."""
    },
    {
        "path": "guides/Troubleshooting.md",
        "title": "Troubleshooting",
        "tags": ["guide", "help"],
        "body": """# Troubleshooting

Common issues and solutions.

## Search Not Finding Notes

**Problem**: Search returns no results or outdated results.

**Solution**: Go to [[Settings]] and click "Rebuild Index". This re-scans all notes and updates the search index.

## Wikilink Not Working

**Problem**: Clicking a wikilink doesn't navigate, or shows as broken.

**Solution**: 
- Check the target note exists
- Wikilinks match on normalized slugs (case-insensitive, spaces→dashes)
- See [[Wikilink Examples]] for how resolution works

## Version Conflict on Save

**Problem**: "This note changed since you opened it" error when saving.

**Solution**: 
- Someone else (or an AI agent) edited the note while you were editing
- Reload the page to get the latest version
- Copy your changes and re-apply them
- This prevents data loss from concurrent edits

## Authentication Issues

**Problem**: Keep getting logged out or "401 Unauthorized" errors.

**Solution**:
- JWT tokens expire after 7 days
- Sign in again to get a new token
- Check that your [[MCP Integration]] config uses a valid token

## Data Disappeared

**Problem**: Notes or changes are missing after page reload.

**Solution**:
- This is a **DEMO instance** with ephemeral storage
- Data resets when the server restarts
- For permanent storage, deploy your own instance
- See [[Getting Started]] for demo disclaimer"""
    },
    {
        "path": "FAQ.md",
        "title": "FAQ",
        "tags": ["faq", "help"],
        "body": """# FAQ

Frequently asked questions about the Document Viewer.

## General

**Q: What is this?**

A: An AI-powered documentation system where AI agents (via [[MCP Integration]]) can write and update docs, and humans can read and refine them in a beautiful UI.

**Q: Is my data persistent?**

A: **No, this is a demo instance.** All data is ephemeral and resets on server restart. For permanent storage, deploy your own instance.

**Q: How do I sign in?**

A: Click "Sign in with Hugging Face" on the login page. You'll authenticate via HF OAuth and get isolated vault access.

## Features

**Q: What are wikilinks?**

A: Wikilinks let you link between notes using `[[Note Name]]` syntax. See [[Wikilink Examples]].

**Q: How does search work?**

A: Full-text search with BM25 ranking, weighted by title matches and recency. See [[Search Features]].

**Q: Can AI agents edit my docs?**

A: Yes! Configure [[MCP Integration]] to let AI agents read and write notes via MCP tools.

## Technical

**Q: What tech stack?**

A: FastAPI + SQLite FTS5 backend, React + shadcn/ui frontend. See [[Architecture Overview]].

**Q: Is it multi-tenant?**

A: Yes, each HF user gets an isolated vault with per-user search indexes.

**Q: Where's the source code?**

A: See [[Getting Started]] for links to the repository.

## Related

- [[Troubleshooting]] - Common issues
- [[guides/Quick Reference]] - Command cheat sheet"""
    },
]

WELCOME_NOTE_FILENAME = "Welcome.md"
WELCOME_NOTE_TITLE = "Welcome to Your Vault"
WELCOME_NOTE_TEMPLATE = """# Welcome to Your Vault

Thanks for signing in with Hugging Face! This workspace is private to **{user_id}**.

## Next steps

- Use the directory tree on the left to browse or create folders.
- Click **New Note** to create your first document.
- Use the **Search** bar to find content as your vault grows.

## Settings & API access

- Visit the Settings page to copy your API token for MCP or automation.
- Regenerate the token at any time if you accidentally share it.

## Tips

- Organize content with plain folders; each note is a Markdown file.
- Wikilink to other notes with `[[Example Note]]` once you create them.
- Data in this demo space is ephemeral and may reset when the app restarts.

Enjoy documenting!"""


def seed_demo_vault(user_id: str = "demo-user") -> int:
    """
    Create demo notes in the specified user's vault.
    
    Returns the number of notes created.
    """
    vault_service = VaultService()
    indexer_service = IndexerService()
    
    logger.info(f"Seeding demo vault for user: {user_id}")
    
    # Create demo notes
    notes_created = 0
    for note_data in DEMO_NOTES:
        try:
            path = note_data["path"]
            title = note_data["title"]
            tags = note_data.get("tags", [])
            body = note_data["body"]
            
            # Write note to vault
            note = vault_service.write_note(
                user_id,
                path,
                title=title,
                metadata={"tags": tags},
                body=body,
            )
            
            # Index the note
            indexer_service.index_note(user_id, note)
            notes_created += 1
            
            logger.info(f"Created demo note: {path}")
        
        except Exception as e:
            logger.error(f"Failed to create demo note {note_data['path']}: {e}")
    
    logger.info(f"Seeded {notes_created} demo notes for user: {user_id}")
    return notes_created


def ensure_welcome_note(user_id: str) -> bool:
    """
    Ensure an authenticated user's vault contains the welcome note.

    Returns True if a welcome note was created, False if it already existed
    or was skipped because the vault already has user content.
    """
    vault_service = VaultService()
    indexer_service = IndexerService()
    db_service = DatabaseService()

    logger.info("Ensuring welcome note for user", extra={"user_id": user_id})

    # Make sure the user's vault directory exists
    vault_service.initialize_vault(user_id)

    existing_notes = vault_service.list_notes(user_id)

    if any(note["path"] == WELCOME_NOTE_FILENAME for note in existing_notes):
        logger.info("Welcome note already present", extra={"user_id": user_id})
        return False

    if existing_notes:
        # User has already created content—do not overwrite their vault
        logger.info(
            "Vault already contains notes; skipping welcome note",
            extra={"user_id": user_id, "note_count": len(existing_notes)},
        )
        return False

    body = WELCOME_NOTE_TEMPLATE.format(user_id=user_id)
    note = vault_service.write_note(
        user_id,
        WELCOME_NOTE_FILENAME,
        title=WELCOME_NOTE_TITLE,
        body=body,
        metadata={"tags": ["welcome"]},
    )

    indexer_service.index_note(user_id, note)

    conn = db_service.connect()
    try:
        with conn:
            indexer_service.update_index_health(conn, user_id)
    finally:
        conn.close()

    logger.info("Welcome note created", extra={"user_id": user_id})
    return True


def init_and_seed(user_id: str = "demo-user") -> None:
    """
    Initialize database schema and seed demo vault.
    
    This is called on application startup to ensure the app always has
    valid demo content, even with ephemeral storage.
    """
    logger.info("Initializing database and seeding demo vault...")
    
    # Initialize database schema
    db_path = init_database()
    logger.info(f"Database initialized at: {db_path}")
    
    # Seed demo vault
    notes_created = seed_demo_vault(user_id)
    
    logger.info(f"Initialization complete. Created {notes_created} demo notes.")


__all__ = ["seed_demo_vault", "init_and_seed", "ensure_welcome_note"]

