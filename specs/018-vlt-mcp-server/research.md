# Research: Vlt Unified MCP Server

**Branch**: `018-vlt-mcp-server` | **Date**: 2026-02-18

---

## 1. FastMCP STDIO Transport

**Decision**: Use FastMCP in STDIO mode as the server framework for `vlt-mcp`.

**Rationale**: The existing backend MCP server (`backend/src/mcp/server.py`) already uses FastMCP and is proven working. The STDIO transport pattern is exactly what's needed — the AI assistant (Claude Desktop/Code) spawns the subprocess, communicates via stdin/stdout, and the process lifecycle is managed entirely by the client. No ports, no daemon, no health checks. Auto-start is a property of STDIO by design.

**Implementation pattern** (validated from existing backend server):
```python
from fastmcp import FastMCP

mcp = FastMCP("vlt", instructions="...")

@mcp.tool(name="vlt_thread_push")
def vlt_thread_push(thread_id: str, thought: str, author: str = "agent") -> dict:
    ...

mcp.run(transport="stdio")  # default transport
```

**fastmcp dependency**: Already present in `backend/pyproject.toml`. Must be added to `packages/vlt-cli/pyproject.toml` for the new `vlt-mcp` entry point.

**Alternatives considered**: HTTP transport — rejected for the primary use case. HTTP requires a running server, a known port, and manual startup. STDIO is spawn-on-demand. HTTP transport remains available for remote/authenticated scenarios (same `--http` flag pattern as backend server).

---

## 2. Thread Seek Without Vector Embeddings

**Decision**: Dual-mode seek — use vector similarity when embeddings are available, fall back to SQLite FTS on the `node_content` column when not.

**Rationale**: `IVaultService.search()` in `core/service.py` uses `VectorService` for semantic similarity. The vector service requires an embedding model (OpenRouter API). Agents that haven't configured an API key would get errors on every seek call, which is worse than a degraded but working FTS result.

**FTS fallback approach**: The `nodes` table in `vault.db` can be searched with `LIKE` queries via SQLAlchemy or by adding a lightweight FTS index. Since the existing codebase uses SQLAlchemy ORM, the cleanest path is a `text()` query with `LIKE '%query%'` as the fallback — not true FTS but functional for keyword matching.

**Implementation**: In `mcp/thread_tools.py`, wrap seek with:
1. Try `SqliteVaultService.search(query, project_id)` — uses vector similarity
2. On `VaultError` or missing embedding config, fall back to SQLAlchemy LIKE query on `Node.content`
3. Return results in same format regardless of which path was taken
4. Include a `search_mode: "semantic" | "keyword"` field in the response so agents know what they got

**Alternatives considered**: Mandating embedding config — rejected because it would make seek unusable without API key setup. SQLite FTS5 extension — better quality than LIKE but requires schema migration; deferred to a future enhancement.

---

## 3. Oracle Toggle Storage & MCP Integration

**Decision**: Store `oracle_mcp_enabled` as a new column in the existing backend `user_settings` table. MCP server queries the backend API at tool-call time (if backend is configured); falls back to `VLT_ORACLE_ENABLED` environment variable if backend is unavailable.

**Rationale**: The backend already has a `user_settings` table with an established migration pattern (via `ALTER TABLE` statements in `database.py`). Adding `oracle_mcp_enabled BOOLEAN DEFAULT TRUE` follows the exact same pattern used for `disabled_subscribers_json`, `disabled_rules_json`, etc. The migration is one line.

The MCP oracle tools check this setting at call time (not at startup) so the server doesn't need a restart when the toggle changes — the next oracle call will check and see the updated value.

**Two-tier fallback**:
1. If `settings.vault_url` is configured and backend is reachable: `GET /api/settings/oracle` → check `oracle_mcp_enabled`
2. If backend is unreachable: check `VLT_ORACLE_ENABLED` from local profile `.env` (defaults to `True`)

This means: web UI toggle takes effect on the next oracle MCP call. Local env var provides an override for offline/no-backend scenarios.

**API endpoints required**:
- `GET /api/settings/oracle` → `{"oracle_mcp_enabled": bool}`
- `PUT /api/settings/oracle` → body `{"oracle_mcp_enabled": bool}` → 200 OK

**Frontend**: New section in `Settings.tsx` Oracle tab using the existing `Switch` component (already imported) and the existing `Tabs` pattern (already in use).

**Alternatives considered**: Writing directly to vlt-cli profile `.env` from backend — rejected because the backend doesn't know the client machine's vlt profile path in multi-tenant deployments. Single source of truth in vlt-cli `.env` only — rejected because there's no way for the web UI to write to the client's filesystem.

---

## 4. Vault Tools Architecture

**Decision**: Vault tools proxy to the Document-MCP backend HTTP API (existing endpoints), not direct filesystem access.

**Rationale**: The vault has a dual-layer storage: filesystem markdown files + SQLite FTS index. Writing directly to the filesystem from the vlt-mcp server would bypass the indexer and produce stale search results. The backend API already handles the write-through correctly. Since `settings.vault_url` and `settings.sync_token` are already configured for oracle thin-client mode, the same credentials work for vault API calls.

**Implementation**: Use `httpx` (already a dependency) to make authenticated requests:
- `Authorization: Bearer {settings.sync_token}` header
- Backend URL from `settings.vault_url`

**When backend unavailable**: Return `{"status": "error", "code": "VAULT_UNAVAILABLE", "message": "Document-MCP backend unreachable at {url}. Start with ./start-dev.sh"}`. Never raise unhandled exceptions.

**Alternatives considered**: Direct filesystem access — rejected (bypass issue). Replicating VaultService + IndexerService into vlt-cli — rejected (massive duplication, maintenance burden, no clear win over HTTP proxy for low-throughput vault operations).

---

## 5. Code Initialization in MCP Context

**Decision**: `vlt_code_init` submits to daemon if running, falls back to synchronous inline indexing if daemon is unavailable. Returns immediately in both cases with a status handle.

**Rationale**: The existing `DaemonClient` (at `vlt/daemon/client.py`) already handles job submission. From MCP tool context, we can't block for a long-running index operation (could be minutes for large repos). The daemon handles background execution cleanly. If daemon isn't running, run the indexer in a background thread and return a synthetic job ID for status polling.

**Status polling**: `vlt_code_status` queries `CodeRAGStore` for the job status — same as `vlt coderag status` CLI. This works regardless of whether the daemon or inline thread did the indexing.

---

## 6. pyproject.toml Changes Required

**Add to `packages/vlt-cli/pyproject.toml`**:
- `fastmcp>=2.0.0` to `dependencies`
- `vlt-mcp = "vlt.mcp_server:main"` to `[project.scripts]`
- The `oracle` optional extra (`llama-index`, `tree-sitter`) should be a recommended install but not required for thread/meta tools to work

**FastMCP version**: The backend uses `fastmcp` — check `backend/pyproject.toml` for pinned version and use same to avoid incompatibilities.

---

## 7. Constitution Compliance

**Brownfield (I)**: Adding new files only. `config.py` gets `oracle_enabled` field (additive). `database.py` gets one ALTER TABLE migration (additive). `Settings.tsx` gets a new tab section (additive). No existing code rewritten.

**Test-backed (II)**: pytest tests required for all MCP tool modules in `packages/vlt-cli/src/vlt/tests/`. Backend settings endpoint requires test coverage in `backend/tests/`.

**Incremental (III)**: The MCP server is a new command (`vlt-mcp`) that doesn't affect `vlt` CLI. Phase 1 (threads + meta) is independently testable before phases 2–4.

**No Magic (IV)**: Direct SQLAlchemy calls to `SqliteVaultService`, `CodeRAGStore`. No dynamic tool registration, no reflection. Oracle check is an explicit if-statement, not a middleware hook.

**SQLite (V)**: The backend settings endpoint uses sqlite3 stdlib per constitution (backend already does this). vlt-cli uses SQLAlchemy ORM (pre-existing; not changing the storage layer).

**All unknowns resolved. No NEEDS CLARIFICATION items remain.**
