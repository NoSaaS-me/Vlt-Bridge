# Data Model: CodeRAG Project Integration

**Feature Branch**: `011-coderag-project-init`
**Date**: 2026-01-01

## Entity Overview

```
┌─────────────┐     1:1     ┌──────────────────┐
│   Project   │─────────────│  CodeRAGIndex    │
└─────────────┘             └──────────────────┘
                                    │
                                    │ 1:N
                                    ▼
                            ┌──────────────────┐
                            │ CodeRAGIndexJob  │
                            └──────────────────┘
```

---

## 1. CodeRAGIndex (Metadata)

**Purpose**: Track the state of a project's code index.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| project_id | string | PK, FK(projects.id) | Owning project |
| status | enum | NOT NULL | Current index state |
| file_count | integer | DEFAULT 0 | Number of indexed files |
| chunk_count | integer | DEFAULT 0 | Number of code chunks |
| last_indexed_at | datetime | nullable | Last successful index time |
| last_indexed_path | string | nullable | Root path that was indexed |
| error_message | string | nullable | Error if status=failed |
| created_at | datetime | NOT NULL | First initialization time |
| updated_at | datetime | NOT NULL | Last state change |

**Status Enum Values**:
- `not_initialized` - No index exists
- `indexing` - Currently being indexed
- `ready` - Index complete and searchable
- `failed` - Last indexing attempt failed
- `stale` - Index exists but may be outdated

**Validation Rules**:
- `project_id` must reference existing project
- `status` transitions: not_initialized → indexing → ready|failed
- `file_count` and `chunk_count` must be ≥ 0
- `last_indexed_at` must be in the past

---

## 2. CodeRAGIndexJob

**Purpose**: Track background indexing jobs with progress.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | string (UUID) | PK | Job identifier |
| project_id | string | FK(projects.id), NOT NULL | Target project |
| status | enum | NOT NULL | Job state |
| target_path | string | NOT NULL | Directory to index |
| force | boolean | DEFAULT false | Ignore incremental cache |
| priority | integer | DEFAULT 0 | Job queue priority |
| files_total | integer | DEFAULT 0 | Total files discovered |
| files_processed | integer | DEFAULT 0 | Files completed |
| chunks_created | integer | DEFAULT 0 | Chunks generated |
| progress_percent | integer | DEFAULT 0 | Calculated percentage |
| created_at | datetime | NOT NULL | Job queued time |
| started_at | datetime | nullable | Processing start time |
| completed_at | datetime | nullable | Processing end time |
| error_message | string | nullable | Error details if failed |

**Status Enum Values**:
- `pending` - Queued, not yet started
- `running` - Currently processing
- `completed` - Finished successfully
- `failed` - Terminated with error
- `cancelled` - User-cancelled

**Validation Rules**:
- `id` must be valid UUID v4
- `files_processed` ≤ `files_total`
- `progress_percent` = (files_processed / files_total) * 100
- `started_at` must be after `created_at`
- `completed_at` must be after `started_at`
- Only one job per project can have status=running

---

## 3. Project (Existing - Extended)

**Existing entity** in both CLI and backend. No schema changes needed.

**New relationship**:
- `code_index: Optional[CodeRAGIndex]` - 0..1 relationship
- When project deleted → cascade delete CodeRAGIndex and related data

---

## State Transitions

### CodeRAGIndex Status Flow

```
              ┌─────────────────────────────────────┐
              │                                     │
              ▼                                     │
    ┌─────────────────┐    init     ┌──────────┐   │
    │ not_initialized │────────────>│ indexing │───┤ (on error)
    └─────────────────┘             └──────────┘   │
                                         │         │
                                         │ success │
                                         ▼         │
                                    ┌─────────┐    │
                       ┌───────────>│  ready  │    │
                       │            └─────────┘    │
                       │                 │         │
                       │                 │ re-index│
                       │                 ▼         │
                       │            ┌──────────┐   │
                       └────────────│ indexing │───┘
                                    └──────────┘
                                         │
                                         │ failure
                                         ▼
                                    ┌──────────┐
                                    │  failed  │
                                    └──────────┘
```

### CodeRAGIndexJob Status Flow

```
    ┌─────────┐    pick    ┌─────────┐
    │ pending │───────────>│ running │
    └─────────┘            └─────────┘
         │                      │
         │ cancel               │ success
         ▼                      ▼
    ┌───────────┐         ┌───────────┐
    │ cancelled │         │ completed │
    └───────────┘         └───────────┘
                               │
                               │ error
                               ▼
                          ┌──────────┐
                          │  failed  │
                          └──────────┘
```

---

## API Response Types

### CodeRAGStatusResponse

```typescript
interface CodeRAGStatusResponse {
  project_id: string;
  status: 'not_initialized' | 'indexing' | 'ready' | 'failed' | 'stale';
  file_count: number;
  chunk_count: number;
  last_indexed_at: string | null;  // ISO 8601
  error_message: string | null;
  active_job: JobSummary | null;   // If currently indexing
}

interface JobSummary {
  job_id: string;
  progress_percent: number;
  files_processed: number;
  files_total: number;
  started_at: string;
}
```

### InitCodeRAGRequest

```typescript
interface InitCodeRAGRequest {
  project_id: string;
  target_path: string;
  force?: boolean;       // Default: false
  background?: boolean;  // Default: true
}
```

### InitCodeRAGResponse

```typescript
interface InitCodeRAGResponse {
  job_id: string;
  status: 'queued' | 'started';
  message: string;
}
```

### JobStatusResponse

```typescript
interface JobStatusResponse {
  job_id: string;
  project_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress_percent: number;
  files_total: number;
  files_processed: number;
  chunks_created: number;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  duration_seconds: number | null;
}
```

---

## Database Location

| Component | Database | Table Location |
|-----------|----------|----------------|
| CLI (vlt) | `~/.vlt/vault.db` | `coderag_index_jobs` |
| Backend | `data/index.db` | `coderag_status` (cache) |

**Note**: The CLI owns the authoritative job data. The backend caches status for API responses and polls the CLI's database via OracleBridge when needed.

---

## Cascade Delete Behavior

When a project is deleted:

1. **Backend** (`project_service.py`):
   - Delete `coderag_status` cache row
   - Call `vlt coderag delete --project <id>` via OracleBridge

2. **CLI** (`vlt coderag delete`):
   - Delete all `code_chunks` where project_id = X
   - Delete all `code_nodes` where project_id = X
   - Delete all `code_edges` where project_id = X
   - Delete all `symbol_definitions` where project_id = X
   - Delete all `coderag_index_jobs` where project_id = X
   - Delete `coderag_indexes` row where project_id = X
