# Data Model: Vlt Unified MCP Server

**Branch**: `018-vlt-mcp-server` | **Date**: 2026-02-18

---

## Existing Entities (no changes to schema)

These entities already exist in `~/.vlt/profiles/{profile}/vault.db` (SQLAlchemy ORM via vlt-cli). The MCP tools read and write these via `SqliteVaultService` and `CodeRAGStore`.

### Project
Defined in `packages/vlt-cli/src/vlt/core/models.py`

| Field | Type | Notes |
|-------|------|-------|
| `id` | String PK | Slug (e.g., "vlt-bridge") |
| `name` | String | Display name |
| `description` | String | Optional |
| `created_at` | DateTime | Auto-set |

### Thread
| Field | Type | Notes |
|-------|------|-------|
| `id` | String PK | Slug (e.g., "auth-design") |
| `project_id` | String FK | References Project.id |
| `name` | String | Display name |
| `status` | String | "active" \| "archived" |
| `created_at` | DateTime | Auto-set |

### Node (thoughts in a thread)
| Field | Type | Notes |
|-------|------|-------|
| `id` | String PK | UUID |
| `thread_id` | String FK | References Thread.id |
| `content` | String | The thought text |
| `author` | String | Attribution (e.g., "agent", "Wolfe") |
| `timestamp` | DateTime | Auto-set |
| `sequence_id` | Integer | Monotonically increasing per thread |

### State (compressed thread summary)
| Field | Type | Notes |
|-------|------|-------|
| `thread_id` | String PK | References Thread.id (1:1) |
| `summary` | String | LLM-compressed state |
| `updated_at` | DateTime | When Librarian last ran |

### CodeChunk
Defined in `packages/vlt-cli/src/vlt/core/models.py`

| Field | Type | Notes |
|-------|------|-------|
| `id` | String PK | UUID |
| `project_id` | String | Project the chunk belongs to |
| `file_path` | String | Relative path from project root |
| `name` | String | Symbol/chunk name |
| `qualified_name` | String | Fully-qualified (e.g., `MyClass.my_method`) |
| `chunk_type` | Enum | ChunkType: function, class, method, module, etc. |
| `language` | String | "python", "typescript", etc. |
| `lineno` | Integer | Start line (1-indexed) |
| `end_lineno` | Integer | End line |
| `signature` | String | Function/class signature |
| `body` | String | Source code body |
| `docstring` | String | Docstring if present |
| `class_context` | String | Enclosing class name (for methods) |
| `imports` | String | File-level imports (context enrichment) |
| `content_hash` | String | MD5 of body (for incremental indexing) |
| `embedding` | Bytes | Serialized float vector |

### SymbolDefinition
| Field | Type | Notes |
|-------|------|-------|
| `id` | String PK | UUID |
| `project_id` | String | Project scope |
| `name` | String | Symbol name (e.g., "authenticate") |
| `kind` | String | "function", "class", "method", "variable" |
| `file_path` | String | Relative path |
| `lineno` | Integer | Definition line |
| `scope` | String | Enclosing scope |
| `signature` | String | Signature if applicable |
| `language` | String | Language |

### RepoMap
| Field | Type | Notes |
|-------|------|-------|
| `project_id` | String PK | 1:1 with project |
| `map_text` | String | Token-budgeted map text |
| `token_count` | Integer | Actual token count |
| `files_included` | Integer | Number of files in map |
| `symbols_included` | Integer | Symbols in map |
| `symbols_total` | Integer | Total symbols in project |
| `generated_at` | DateTime | Cache timestamp |

### CodeRAGIndexJob
| Field | Type | Notes |
|-------|------|-------|
| `id` | String PK | UUID |
| `project_id` | String | Project being indexed |
| `project_path` | String | Absolute filesystem path |
| `status` | String | "pending" \| "running" \| "completed" \| "failed" \| "cancelled" |
| `files_total` | Integer | Total files to index (nullable during pending) |
| `files_indexed` | Integer | Progress counter |
| `chunks_created` | Integer | Total chunks written |
| `symbols_created` | Integer | Total symbols written |
| `error_message` | String | Set on failure |
| `created_at` | DateTime | Job submission time |
| `started_at` | DateTime | When daemon picked up job |
| `completed_at` | DateTime | When indexing finished |

---

## New Entities

### oracle_mcp_enabled (backend user_settings migration)

**Location**: `backend/src/services/database.py` migration list
**Change**: One new `ALTER TABLE` migration

```sql
ALTER TABLE user_settings ADD COLUMN oracle_mcp_enabled INTEGER NOT NULL DEFAULT 1
```

This is stored per `user_id` in the backend's `user_settings` table. `1` = enabled (default), `0` = disabled.

**Stored in**: `backend/data/index.db` (the backend's SQLite DB, not vlt-cli's vault.db)

**Read by**:
- Backend `GET /api/settings/oracle` → returns current value
- MCP oracle tools → via HTTP GET to backend (if backend configured)

**Written by**:
- Backend `PUT /api/settings/oracle` → from web UI toggle

---

## MCP Response Schema Conventions

All MCP tools return a dict following this convention:

### Success
```json
{
  "status": "ok",
  "<domain>": { ... }
}
```

### Error
```json
{
  "status": "error",
  "code": "MACHINE_READABLE_CODE",
  "message": "Human-readable description with guidance"
}
```

### Error Codes
| Code | When |
|------|------|
| `PROJECT_NOT_FOUND` | Project slug doesn't exist, and auto-create is not applicable |
| `THREAD_NOT_FOUND` | Thread slug doesn't exist |
| `INDEX_NOT_FOUND` | CodeRAG index not initialized for project |
| `INDEX_RUNNING` | Indexing in progress (not an error, informational) |
| `ORACLE_DISABLED` | `oracle_mcp_enabled = false` in backend settings |
| `ORACLE_NOT_CONFIGURED` | Oracle enabled but no API key/backend credentials |
| `VAULT_UNAVAILABLE` | Backend HTTP API unreachable |
| `INVALID_PATH` | Provided path failed security validation |
| `DAEMON_NOT_RUNNING` | Daemon unavailable (background indexing fell back to inline) |

---

## State Transitions

### CodeRAGIndexJob Status
```
[SUBMIT] → pending
[DAEMON PICKUP] → running
[COMPLETE] → completed
[ERROR] → failed
[USER CANCEL] → cancelled
```

### Thread Status
```
active (default)
archived (manual, via CLI only — not exposed in MCP v1)
```

### Oracle MCP Enabled
```
true (default) ↔ false
Toggle via web UI. Takes effect on next MCP oracle tool call.
```
