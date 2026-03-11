# Quickstart: Oracle CodeAct Dev Setup

**Feature**: `023-oracle-codeact-rework`
**Updated**: 2026-03-11

## Prerequisites

- Docker (for FalkorDB)
- uv (Python package manager)
- Existing Vlt-Bridge backend dev environment

---

## 1. Start FalkorDB (Graphiti backend)

```bash
# From repo root
docker compose -f docker/graphiti-compose.yml up -d

# Verify it's running
docker compose -f docker/graphiti-compose.yml ps
# Expected: falkordb running on port 6379
```

FalkorDB uses the Redis protocol on port 6379. Graphiti connects to `bolt://localhost:6379` with empty credentials.

---

## 2. Install new Python dependencies

```bash
cd backend
uv pip install -e ".[oracle-v2]"
# or individually:
uv pip install \
  "langgraph-codeact>=0.1.3" \
  "langgraph-checkpoint-sqlite" \
  "graphiti-core[anthropic]>=0.28.1"
```

Add to `pyproject.toml` optional deps:
```toml
[project.optional-dependencies]
oracle-v2 = [
    "langgraph-codeact>=0.1.3",
    "langgraph-checkpoint-sqlite",
    "graphiti-core[anthropic]>=0.28.1",
]
```

---

## 3. Environment variables

No new required env vars. The following existing vars are used:
- `ANTHROPIC_API_KEY` — used by Graphiti for entity extraction (resolved from user settings at runtime)
- `DB_PATH` — existing SQLite DB path; `oracle_threads` table is added here automatically

New optional env vars (with defaults):
```bash
ORACLE_CHECKPOINT_DB=data/checkpoints.db   # LangGraph checkpoint DB path
FALKORDB_URL=bolt://localhost:6379          # FalkorDB connection
```

---

## 4. Database migration

`oracle_threads` table is auto-created on first startup by the lifespan handler. No manual migration needed.

`data/checkpoints.db` is created automatically by `AsyncSqliteSaver` on first use.

---

## 5. Run the backend

```bash
cd backend
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Watch for:
```
INFO: OracleV2: AsyncSqliteSaver initialized (data/checkpoints.db)
INFO: OracleV2: Graphiti client connected (bolt://localhost:6379)
INFO: Oracle V2 lifespan ready
```

---

## 6. Test multi-turn continuity

```bash
# Start a new thread
curl -X POST http://localhost:8000/api/oracle/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question": "Where is the authentication middleware?", "project_id": "vlt-bridge"}' \
  --no-buffer

# Grab the context_id from the done chunk, then:
curl -X POST http://localhost:8000/api/oracle/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question": "What does it import?", "context_id": "<thread_id_from_above>", "project_id": "vlt-bridge"}' \
  --no-buffer
```

The second response should use already-retrieved file content (no repeated `search_code` calls visible in `tool_call` chunks).

---

## 7. List threads

```bash
curl http://localhost:8000/api/oracle/threads?project_id=vlt-bridge \
  -H "Authorization: Bearer $TOKEN"
```

---

## 8. Verify FalkorDB data (optional)

```bash
docker exec -it falkordb redis-cli
> GRAPH.LIST
> GRAPH.QUERY graphiti "MATCH (n) RETURN n LIMIT 5"
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `bolt://localhost:6379 connection refused` | FalkorDB container not running — `docker compose -f docker/graphiti-compose.yml up -d` |
| `AsyncSqliteSaver: database is locked` | Only one backend instance can hold the checkpoint DB. Stop duplicate processes. |
| Graphiti entity extraction timeout | Check `ANTHROPIC_API_KEY` is set in user settings (resolved per-request). |
| `get_state_history()` hangs | Use `aget_state_history()` — the sync version hangs with async checkpointer (LangGraph Issue #2992). |
| Custom events not appearing in stream | `get_stream_writer()` silently drops events in async eval_fn (Issue #6447). Use sync `eval_fn` with `asyncio.run_coroutine_threadsafe()` for async tool calls. |
