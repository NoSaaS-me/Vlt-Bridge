# Research: Thread Sync from CLI to Server

**Feature**: 008-thread-sync
**Date**: 2025-12-31
**Status**: Complete

## Research Tasks

### 1. Existing vlt-cli Thread Storage

**Question**: How are threads currently stored in vlt-cli?

**Decision**: Use existing SQLAlchemy ORM models as the source schema for sync

**Rationale**:
- vlt-cli uses SQLAlchemy ORM with well-defined models in `packages/vlt-cli/src/vlt/core/models.py`
- Thread structure: `Thread` → `Node` (linked list with `prev_node_id`)
- Already supports: embeddings, tags, references, states (summaries)
- Database location: `~/.vlt/vault.db`

**Key Entities Found**:
| Entity | Primary Key | Key Fields |
|--------|-------------|------------|
| Thread | id (string) | project_id, status, created_at |
| Node | id (UUID) | thread_id, sequence_id, content, author, timestamp, embedding |
| State | id (UUID) | target_id, target_type, summary, head_node_id |
| Tag | id (int) | name (unique) |
| Reference | id (UUID) | source_node_id, target_thread_id, note |

**Alternatives Considered**:
- File-based JSON export: Rejected - doesn't match existing architecture
- Custom schema for sync: Rejected - would require translation layer

---

### 2. Existing vault_url Configuration

**Question**: Is there existing infrastructure for CLI-to-server communication?

**Decision**: Extend existing `OracleConfig.vault_url` for thread sync

**Rationale**:
- `vault_url` already exists in `packages/vlt-cli/src/vlt/core/identity.py`
- Default: `http://localhost:8000`
- Currently used for vault note retrieval in Oracle
- httpx already used in codebase for HTTP calls

**Code Location**:
```python
# packages/vlt-cli/src/vlt/core/identity.py
class OracleConfig(BaseModel):
    vault_url: str = Field(
        default="http://localhost:8000",
        description="Document-MCP vault URL for markdown notes"
    )
```

**Alternatives Considered**:
- New config section: Rejected - vault_url already points to the correct server
- Environment variable: Rejected - vlt.toml is the established config location

---

### 3. Backend API Patterns

**Question**: How should thread sync endpoints be structured?

**Decision**: Follow existing FastAPI router patterns in `backend/src/api/routes/`

**Rationale**:
- Consistent with existing endpoints (oracle.py, notes.py, search.py)
- Use `APIRouter` with `/api/threads` prefix
- Dependency injection via `Depends(get_auth_context)`
- Pydantic models for request/response validation

**Pattern Reference** (from oracle.py):
```python
router = APIRouter(prefix="/api/oracle", tags=["oracle"])

@router.post("", response_model=OracleResponse)
async def query_oracle(
    request: OracleRequest,
    auth: AuthContext = Depends(get_auth_context),
    oracle: OracleBridge = Depends(get_oracle_bridge),
):
```

**Alternatives Considered**:
- WebSocket for real-time sync: Rejected - overkill for push-based sync, adds complexity
- GraphQL: Rejected - not used elsewhere in codebase, breaks constitution

---

### 4. Backend Database Schema

**Question**: How should threads be stored in the backend?

**Decision**: Add thread tables to existing SQLite database (data/index.db) using raw DDL

**Rationale**:
- Backend uses raw `sqlite3` (no ORM) per constitution
- Tables scoped by `user_id` like other backend tables
- Follow existing DDL pattern in `database.py`

**Schema Design**:
```sql
-- threads table (synced from CLI)
CREATE TABLE IF NOT EXISTS threads (
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, thread_id)
);

-- thread_entries table (synced from CLI nodes)
CREATE TABLE IF NOT EXISTS thread_entries (
    user_id TEXT NOT NULL,
    entry_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    sequence_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT 'user',
    timestamp TEXT NOT NULL,
    PRIMARY KEY (user_id, entry_id),
    FOREIGN KEY (user_id, thread_id) REFERENCES threads(user_id, thread_id)
);

-- sync_status table (tracks sync state)
CREATE TABLE IF NOT EXISTS thread_sync_status (
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    last_synced_sequence INTEGER NOT NULL DEFAULT -1,
    last_sync_at TEXT NOT NULL,
    sync_error TEXT,
    PRIMARY KEY (user_id, thread_id)
);
```

**Alternatives Considered**:
- Separate database file: Rejected - adds operational complexity
- SQLAlchemy ORM: Rejected - not used in backend per constitution

---

### 5. Sync Protocol Design

**Question**: How should incremental sync work?

**Decision**: Sequence-based incremental sync with queue for offline handling

**Rationale**:
- Nodes have `sequence_id` (0-indexed, ordered within thread)
- Backend tracks `last_synced_sequence` per thread
- CLI sends only entries with `sequence_id > last_synced_sequence`
- Failed syncs queued locally and retried

**Protocol Flow**:
```
1. CLI: vlt thread push "insight"
   ↓
2. CLI: Save node locally with sequence_id = N
   ↓
3. CLI: POST /api/threads/{thread_id}/entries
   Body: { entries: [{ sequence_id: N, content: "...", author: "...", timestamp: "..." }] }
   Headers: Authorization: Bearer <token>
   ↓
4. Backend: Validate & store entry
   ↓
5. Backend: Update thread_sync_status.last_synced_sequence = N
   ↓
6. Backend: Return { synced: [N], thread_id: "..." }

On failure (network error, 5xx):
- CLI stores in local queue (~/.vlt/sync_queue.json)
- Retry on next push or via `vlt sync` command
```

**Alternatives Considered**:
- Full thread sync on each push: Rejected - inefficient for large threads
- Timestamp-based sync: Rejected - sequence_id is more reliable for ordering
- Real-time WebSocket: Rejected - push-based is sufficient, simpler

---

### 6. Authentication for Sync

**Question**: How does CLI authenticate with backend?

**Decision**: Use existing JWT token stored in CLI config

**Rationale**:
- Backend already uses JWT authentication
- Add `sync_token` field to vlt config
- Token obtained from web UI Settings page (existing flow)
- Passed as `Authorization: Bearer <token>` header

**Implementation**:
```python
# packages/vlt-cli/src/vlt/config.py
class Settings(BaseSettings):
    sync_token: Optional[str] = Field(None, description="JWT for sync authentication")
```

**Alternatives Considered**:
- API key: Rejected - JWT already established in backend
- OAuth flow in CLI: Rejected - adds complexity, existing token flow works

---

### 7. Oracle Integration

**Question**: How does backend Oracle query synced threads?

**Decision**: Replace subprocess calls with direct database queries via ThreadService

**Rationale**:
- Current approach (subprocess `vlt oracle`) is broken - no vlt.toml on server
- Backend has synced thread data in its own database
- Create `ThreadRetriever` service that queries local threads table
- Integrate with existing Oracle context assembly

**Architecture Change**:
```
BEFORE (broken):
  Web UI → Backend → subprocess vlt oracle → FAILS

AFTER:
  Web UI → Backend Oracle → ThreadService.search_threads() → local SQLite
                         → VaultService.search_notes() → local vault
                         → CodeRAGService.search() → local code index
                         → LLM synthesis
```

**Alternatives Considered**:
- Keep subprocess with vlt.toml on server: Rejected - threads still not synced
- HTTP call to CLI: Rejected - CLI runs on agent machine, not accessible

---

## Summary

All NEEDS CLARIFICATION items resolved:

| Item | Decision |
|------|----------|
| Thread storage format | Use existing vlt-cli SQLAlchemy ORM models as source |
| Sync transport | HTTP POST to vault_url with JWT auth |
| Backend storage | Raw SQLite DDL in data/index.db |
| Sync protocol | Sequence-based incremental sync |
| Authentication | JWT token stored in CLI config |
| Oracle integration | Direct database queries replacing subprocess |

**Next Phase**: Generate data-model.md and API contracts
