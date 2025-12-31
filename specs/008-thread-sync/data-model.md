# Data Model: Thread Sync

**Feature**: 008-thread-sync
**Date**: 2025-12-31

## Overview

Thread sync involves three data stores:
1. **vlt-cli local** (~/.vlt/vault.db) - Source of truth, SQLAlchemy ORM
2. **Backend** (data/index.db) - Synced replica, raw SQLite
3. **CLI sync queue** (~/.vlt/sync_queue.json) - Offline retry buffer

## Entities

### Thread (Backend)

Represents a conversation thread synced from vlt-cli.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| user_id | TEXT | NOT NULL, PK part | Owner's user ID |
| thread_id | TEXT | NOT NULL, PK part | Thread identifier (slug) |
| project_id | TEXT | NOT NULL | Parent project ID |
| name | TEXT | NOT NULL | Thread display name |
| status | TEXT | NOT NULL, DEFAULT 'active' | One of: active, archived, blocked |
| created_at | TEXT | NOT NULL | ISO 8601 timestamp |
| updated_at | TEXT | NOT NULL | ISO 8601 timestamp |

**Primary Key**: (user_id, thread_id)

**Relationships**:
- One Thread has many ThreadEntries
- One Thread has one SyncStatus

---

### ThreadEntry (Backend)

Represents a single node/thought within a thread.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| user_id | TEXT | NOT NULL, PK part | Owner's user ID |
| entry_id | TEXT | NOT NULL, PK part | Entry UUID |
| thread_id | TEXT | NOT NULL, FK | Parent thread ID |
| sequence_id | INTEGER | NOT NULL | Order within thread (0-indexed) |
| content | TEXT | NOT NULL | Entry text content |
| author | TEXT | NOT NULL, DEFAULT 'user' | Who wrote it (user, claude, system) |
| timestamp | TEXT | NOT NULL | ISO 8601 timestamp |

**Primary Key**: (user_id, entry_id)

**Foreign Key**: (user_id, thread_id) → threads(user_id, thread_id)

**Index**: (user_id, thread_id, sequence_id) for ordered retrieval

---

### SyncStatus (Backend)

Tracks sync state for each thread to enable incremental sync.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| user_id | TEXT | NOT NULL, PK part | Owner's user ID |
| thread_id | TEXT | NOT NULL, PK part | Thread identifier |
| last_synced_sequence | INTEGER | NOT NULL, DEFAULT -1 | Highest synced sequence_id |
| last_sync_at | TEXT | NOT NULL | ISO 8601 timestamp of last sync |
| sync_error | TEXT | NULL | Last error message if failed |

**Primary Key**: (user_id, thread_id)

---

### SyncQueueItem (CLI - JSON file)

Queued entry for retry when sync fails.

| Field | Type | Description |
|-------|------|-------------|
| thread_id | TEXT | Target thread |
| entry | Object | Full entry data (entry_id, sequence_id, content, author, timestamp) |
| attempts | INTEGER | Number of sync attempts |
| last_attempt | TEXT | ISO 8601 timestamp |
| error | TEXT | Last error message |

**Storage**: `~/.vlt/sync_queue.json` (array of items)

---

## State Transitions

### Thread Status

```
          create
            ↓
        [active] ←────────┐
            │             │
    archive │             │ reactivate
            ↓             │
       [archived] ────────┘
            │
      block │
            ↓
       [blocked]
```

### Sync State

```
       push entry
            ↓
       [pending] ← network error
            │
      POST success
            ↓
       [synced]
            │
       new entry
            ↓
       [pending] ...
```

---

## Validation Rules

### Thread
- `thread_id`: 1-128 chars, alphanumeric + hyphens, lowercase
- `project_id`: 1-128 chars, alphanumeric + hyphens, lowercase
- `name`: 1-256 chars
- `status`: Must be one of: active, archived, blocked
- `created_at`, `updated_at`: Valid ISO 8601 timestamps

### ThreadEntry
- `entry_id`: Valid UUID format
- `sequence_id`: >= 0, monotonically increasing within thread
- `content`: 1-100,000 chars (10KB max after UTF-8 encoding)
- `author`: 1-64 chars (typically: user, claude, system)
- `timestamp`: Valid ISO 8601 timestamp

### SyncStatus
- `last_synced_sequence`: >= -1 (-1 means never synced)
- `last_sync_at`: Valid ISO 8601 timestamp

---

## Backend DDL

```sql
-- threads table
CREATE TABLE IF NOT EXISTS threads (
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'archived', 'blocked')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, thread_id)
);

CREATE INDEX IF NOT EXISTS idx_threads_user_project ON threads(user_id, project_id);
CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(user_id, status);

-- thread_entries table
CREATE TABLE IF NOT EXISTS thread_entries (
    user_id TEXT NOT NULL,
    entry_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    sequence_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT 'user',
    timestamp TEXT NOT NULL,
    PRIMARY KEY (user_id, entry_id),
    FOREIGN KEY (user_id, thread_id) REFERENCES threads(user_id, thread_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_entries_thread_seq ON thread_entries(user_id, thread_id, sequence_id);
CREATE INDEX IF NOT EXISTS idx_entries_timestamp ON thread_entries(user_id, timestamp);

-- thread_sync_status table
CREATE TABLE IF NOT EXISTS thread_sync_status (
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    last_synced_sequence INTEGER NOT NULL DEFAULT -1,
    last_sync_at TEXT NOT NULL,
    sync_error TEXT,
    PRIMARY KEY (user_id, thread_id),
    FOREIGN KEY (user_id, thread_id) REFERENCES threads(user_id, thread_id) ON DELETE CASCADE
);

-- Full-text search for thread content (optional, for Oracle)
CREATE VIRTUAL TABLE IF NOT EXISTS thread_entries_fts USING fts5(
    content,
    content=thread_entries,
    content_rowid=rowid
);
```

---

## Pydantic Models (Backend)

```python
# backend/src/models/thread.py

from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class ThreadEntry(BaseModel):
    """Single entry/node within a thread."""
    entry_id: str = Field(..., description="UUID of the entry")
    sequence_id: int = Field(..., ge=0, description="Order within thread")
    content: str = Field(..., min_length=1, max_length=100000)
    author: str = Field("user", max_length=64)
    timestamp: datetime

class Thread(BaseModel):
    """Thread synced from vlt-cli."""
    thread_id: str = Field(..., min_length=1, max_length=128)
    project_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=256)
    status: Literal["active", "archived", "blocked"] = "active"
    created_at: datetime
    updated_at: datetime
    entries: Optional[List[ThreadEntry]] = None

class SyncRequest(BaseModel):
    """Request to sync thread entries from CLI."""
    thread_id: str
    project_id: str
    name: str
    status: Literal["active", "archived", "blocked"] = "active"
    entries: List[ThreadEntry] = Field(..., min_length=1)

class SyncResponse(BaseModel):
    """Response after syncing entries."""
    thread_id: str
    synced_count: int
    last_synced_sequence: int

class ThreadListResponse(BaseModel):
    """List of threads for a user."""
    threads: List[Thread]
    total: int
```

---

## CLI Models (vlt-cli)

Existing models in `packages/vlt-cli/src/vlt/core/models.py` - no changes needed.

New sync models:

```python
# packages/vlt-cli/src/vlt/core/sync.py

from pydantic import BaseModel
from typing import List, Optional

class SyncQueueItem(BaseModel):
    """Queued entry for retry."""
    thread_id: str
    entry: dict  # entry_id, sequence_id, content, author, timestamp
    attempts: int = 0
    last_attempt: Optional[str] = None
    error: Optional[str] = None
```
