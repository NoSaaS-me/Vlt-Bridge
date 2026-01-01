# 010: Multi-Project Data Model

## Database Schema Changes

### New Table: projects

```sql
CREATE TABLE IF NOT EXISTS projects (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings_json TEXT,  -- Optional per-project settings override
    PRIMARY KEY (user_id, project_id)
);

CREATE INDEX idx_projects_user_id ON projects(user_id);
```

### Modified Table: note_metadata

**Current:**
```sql
CREATE TABLE note_metadata (
    user_id TEXT NOT NULL,
    note_path TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    title TEXT,
    title_slug TEXT,
    filename_slug TEXT,
    size_bytes INTEGER,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (user_id, note_path)
);
```

**New:**
```sql
CREATE TABLE note_metadata (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'default',
    note_path TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    title TEXT,
    title_slug TEXT,
    filename_slug TEXT,
    size_bytes INTEGER,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (user_id, project_id, note_path),
    FOREIGN KEY (user_id, project_id) REFERENCES projects(user_id, project_id)
);

CREATE INDEX idx_note_metadata_project ON note_metadata(user_id, project_id);
CREATE INDEX idx_note_metadata_slug ON note_metadata(user_id, project_id, title_slug);
```

### Modified Table: note_fts (FTS5)

**Current:**
```sql
CREATE VIRTUAL TABLE note_fts USING fts5(
    user_id,
    note_path,
    title,
    body,
    tokenize='porter unicode61',
    prefix='2 3',
    content=''
);
```

**New:**
```sql
CREATE VIRTUAL TABLE note_fts USING fts5(
    user_id,
    project_id,
    note_path,
    title,
    body,
    tokenize='porter unicode61',
    prefix='2 3',
    content=''
);
```

### Modified Table: note_tags

**Current:**
```sql
CREATE TABLE note_tags (
    user_id TEXT NOT NULL,
    note_path TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (user_id, note_path, tag)
);
```

**New:**
```sql
CREATE TABLE note_tags (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'default',
    note_path TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (user_id, project_id, note_path, tag),
    FOREIGN KEY (user_id, project_id, note_path)
        REFERENCES note_metadata(user_id, project_id, note_path) ON DELETE CASCADE
);

CREATE INDEX idx_note_tags_project ON note_tags(user_id, project_id);
CREATE INDEX idx_note_tags_tag ON note_tags(user_id, project_id, tag);
```

### Modified Table: note_links

**Current:**
```sql
CREATE TABLE note_links (
    user_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    target_path TEXT NOT NULL,
    link_text TEXT,
    is_resolved BOOLEAN DEFAULT 0,
    PRIMARY KEY (user_id, source_path, target_path)
);
```

**New:**
```sql
CREATE TABLE note_links (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'default',
    source_path TEXT NOT NULL,
    target_path TEXT NOT NULL,
    link_text TEXT,
    is_resolved BOOLEAN DEFAULT 0,
    PRIMARY KEY (user_id, project_id, source_path, target_path),
    FOREIGN KEY (user_id, project_id, source_path)
        REFERENCES note_metadata(user_id, project_id, note_path) ON DELETE CASCADE
);

CREATE INDEX idx_note_links_target ON note_links(user_id, project_id, target_path);
```

### Modified Table: index_health

**Current:**
```sql
CREATE TABLE index_health (
    user_id TEXT PRIMARY KEY,
    note_count INTEGER DEFAULT 0,
    last_full_rebuild TIMESTAMP,
    last_incremental_update TIMESTAMP,
    errors TEXT
);
```

**New:**
```sql
CREATE TABLE index_health (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'default',
    note_count INTEGER DEFAULT 0,
    last_full_rebuild TIMESTAMP,
    last_incremental_update TIMESTAMP,
    errors TEXT,
    PRIMARY KEY (user_id, project_id),
    FOREIGN KEY (user_id, project_id) REFERENCES projects(user_id, project_id)
);
```

### Modified Table: user_settings (Optional Per-Project)

**Current:**
```sql
CREATE TABLE user_settings (
    user_id TEXT PRIMARY KEY,
    oracle_model TEXT,
    oracle_provider TEXT,
    thinking_enabled BOOLEAN DEFAULT 1,
    thinking_budget INTEGER DEFAULT 10000,
    max_tokens INTEGER DEFAULT 8000,
    timeout_seconds INTEGER DEFAULT 120,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**New (Global + Per-Project Override):**
```sql
-- Global user settings (fallback)
CREATE TABLE user_settings (
    user_id TEXT PRIMARY KEY,
    oracle_model TEXT,
    oracle_provider TEXT,
    thinking_enabled BOOLEAN DEFAULT 1,
    thinking_budget INTEGER DEFAULT 10000,
    max_tokens INTEGER DEFAULT 8000,
    timeout_seconds INTEGER DEFAULT 120,
    default_project_id TEXT,  -- User's default project
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Per-project settings override (nullable fields = use global)
CREATE TABLE project_settings (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    oracle_model TEXT,           -- NULL = use global
    oracle_provider TEXT,        -- NULL = use global
    thinking_enabled BOOLEAN,    -- NULL = use global
    thinking_budget INTEGER,     -- NULL = use global
    max_tokens INTEGER,          -- NULL = use global
    timeout_seconds INTEGER,     -- NULL = use global
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, project_id),
    FOREIGN KEY (user_id, project_id) REFERENCES projects(user_id, project_id)
);
```

## Existing Tables (Already Project-Scoped)

These tables already have `project_id` and need no schema changes:

### threads
```sql
CREATE TABLE threads (
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    name TEXT,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, thread_id)
);
```

### context_trees
```sql
CREATE TABLE context_trees (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    root_id TEXT NOT NULL,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active_node_id TEXT,
    PRIMARY KEY (user_id, project_id, root_id)
);
```

### context_nodes
```sql
CREATE TABLE context_nodes (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    root_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    parent_id TEXT,
    role TEXT NOT NULL,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, project_id, root_id, node_id)
);
```

## Migration Script

```sql
-- Migration: Add project support to vault tables
-- Version: 010
-- Date: 2026-01-01

BEGIN TRANSACTION;

-- 1. Create projects table
CREATE TABLE IF NOT EXISTS projects (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings_json TEXT,
    PRIMARY KEY (user_id, project_id)
);

-- 2. Create default project for all existing users
INSERT INTO projects (user_id, project_id, name, description)
SELECT DISTINCT user_id, 'default', 'Default Project', 'Migrated from single-vault'
FROM note_metadata
WHERE NOT EXISTS (
    SELECT 1 FROM projects WHERE projects.user_id = note_metadata.user_id
);

-- 3. Add project_id to note_metadata
-- SQLite doesn't support ADD COLUMN with composite PK change, so we recreate
CREATE TABLE note_metadata_new (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'default',
    note_path TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    title TEXT,
    title_slug TEXT,
    filename_slug TEXT,
    size_bytes INTEGER,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (user_id, project_id, note_path)
);

INSERT INTO note_metadata_new
SELECT user_id, 'default', note_path, version, title, title_slug,
       filename_slug, size_bytes, created_at, updated_at
FROM note_metadata;

DROP TABLE note_metadata;
ALTER TABLE note_metadata_new RENAME TO note_metadata;

-- 4. Create indexes
CREATE INDEX IF NOT EXISTS idx_note_metadata_project
    ON note_metadata(user_id, project_id);
CREATE INDEX IF NOT EXISTS idx_note_metadata_slug
    ON note_metadata(user_id, project_id, title_slug);

-- 5. Recreate note_tags with project_id
CREATE TABLE note_tags_new (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'default',
    note_path TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (user_id, project_id, note_path, tag)
);

INSERT INTO note_tags_new
SELECT user_id, 'default', note_path, tag FROM note_tags;

DROP TABLE note_tags;
ALTER TABLE note_tags_new RENAME TO note_tags;

CREATE INDEX IF NOT EXISTS idx_note_tags_project ON note_tags(user_id, project_id);
CREATE INDEX IF NOT EXISTS idx_note_tags_tag ON note_tags(user_id, project_id, tag);

-- 6. Recreate note_links with project_id
CREATE TABLE note_links_new (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'default',
    source_path TEXT NOT NULL,
    target_path TEXT NOT NULL,
    link_text TEXT,
    is_resolved BOOLEAN DEFAULT 0,
    PRIMARY KEY (user_id, project_id, source_path, target_path)
);

INSERT INTO note_links_new
SELECT user_id, 'default', source_path, target_path, link_text, is_resolved
FROM note_links;

DROP TABLE note_links;
ALTER TABLE note_links_new RENAME TO note_links;

CREATE INDEX IF NOT EXISTS idx_note_links_target
    ON note_links(user_id, project_id, target_path);

-- 7. Recreate index_health with project_id
CREATE TABLE index_health_new (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'default',
    note_count INTEGER DEFAULT 0,
    last_full_rebuild TIMESTAMP,
    last_incremental_update TIMESTAMP,
    errors TEXT,
    PRIMARY KEY (user_id, project_id)
);

INSERT INTO index_health_new
SELECT user_id, 'default', note_count, last_full_rebuild,
       last_incremental_update, errors
FROM index_health;

DROP TABLE index_health;
ALTER TABLE index_health_new RENAME TO index_health;

-- 8. Add default_project_id to user_settings
ALTER TABLE user_settings ADD COLUMN default_project_id TEXT DEFAULT 'default';

-- 9. Rebuild FTS index with project_id
DROP TABLE IF EXISTS note_fts;
CREATE VIRTUAL TABLE note_fts USING fts5(
    user_id,
    project_id,
    note_path,
    title,
    body,
    tokenize='porter unicode61',
    prefix='2 3',
    content=''
);

-- Re-populate FTS from note_metadata (body content needs separate process)
INSERT INTO note_fts(user_id, project_id, note_path, title, body)
SELECT user_id, project_id, note_path, title, ''
FROM note_metadata;

COMMIT;
```

## Filesystem Changes

### Vault Directory Structure

**Current:**
```
data/vaults/
├── user-alice/
│   ├── Getting Started.md
│   ├── docs/
│   │   └── API.md
│   └── ...
└── user-bob/
    └── ...
```

**New:**
```
data/vaults/
├── user-alice/
│   ├── default/           # Migrated existing vault
│   │   ├── Getting Started.md
│   │   ├── docs/
│   │   │   └── API.md
│   │   └── ...
│   └── vlt-bridge/        # New project
│       ├── README.md
│       └── ...
└── user-bob/
    └── default/
        └── ...
```

### RAG Index Structure

**Current:**
```
data/llamaindex/
├── user-alice/
│   ├── docstore.json
│   ├── index_store.json
│   └── vector_store.json
└── user-bob/
    └── ...
```

**New:**
```
data/llamaindex/
├── user-alice/
│   ├── default/
│   │   ├── docstore.json
│   │   ├── index_store.json
│   │   └── vector_store.json
│   └── vlt-bridge/
│       └── ...
└── user-bob/
    └── default/
        └── ...
```

## Pydantic Models

### Project Model

```python
# backend/src/models/project.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class ProjectCreate(ProjectBase):
    id: Optional[str] = Field(None, pattern=r'^[a-z0-9-]+$', max_length=50)
    # If not provided, generated from name slug

class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class Project(ProjectBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    note_count: Optional[int] = 0
    thread_count: Optional[int] = 0

    class Config:
        from_attributes = True

class ProjectList(BaseModel):
    projects: list[Project]
    total: int
```

### Updated Note Models

```python
# backend/src/models/note.py (additions)

class NoteMetadata(BaseModel):
    note_path: str
    project_id: str = 'default'  # NEW
    version: int = 1
    title: str
    # ... rest unchanged

class NoteSummary(BaseModel):
    note_path: str
    project_id: str = 'default'  # NEW
    title: str
    # ... rest unchanged
```

## TypeScript Types

```typescript
// frontend/src/types/project.ts

export interface Project {
  id: string;           // slug: "vlt-bridge"
  name: string;         // display: "Vlt Bridge"
  description?: string;
  created_at: string;   // ISO timestamp
  updated_at: string;
  note_count?: number;
  thread_count?: number;
}

export interface ProjectCreate {
  name: string;
  description?: string;
  id?: string;  // Optional, auto-generated if not provided
}

export interface ProjectUpdate {
  name?: string;
  description?: string;
}

// Update existing types
export interface NoteSummary {
  note_path: string;
  project_id: string;  // NEW
  title: string;
  // ... rest unchanged
}

export interface Note {
  note_path: string;
  project_id: string;  // NEW
  title: string;
  body: string;
  // ... rest unchanged
}
```

## Query Pattern Changes

### Before (User-Scoped)
```python
# Vault service
def get_notes(user_id: str) -> list[NoteSummary]:
    return db.execute(
        "SELECT * FROM note_metadata WHERE user_id = ?",
        (user_id,)
    )
```

### After (Project-Scoped)
```python
# Vault service
def get_notes(user_id: str, project_id: str) -> list[NoteSummary]:
    return db.execute(
        "SELECT * FROM note_metadata WHERE user_id = ? AND project_id = ?",
        (user_id, project_id)
    )
```

All vault-related queries add `AND project_id = ?` filter.
