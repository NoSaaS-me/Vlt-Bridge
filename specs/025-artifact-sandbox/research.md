# Research: Artifact Sandbox

## R1: Frontend Tab Integration

**Decision**: Add `'artifacts'` to the `NavSection` union type in `AgentsPage.tsx` and render an `ArtifactsCompositorView` conditionally. Follow the existing pattern of `NAV_ITEMS` array + icon buttons.

**Rationale**: The agents page uses a simple union type + conditional rendering — no router, no complex tab system. Adding a new section is a one-line type change + one array entry + one conditional block. The Lucide `Puzzle` icon fits the "plugin/artifact" concept.

**Alternatives considered**: Separate route/page for artifacts (rejected — breaks the agent-centric workflow, artifacts embed alongside sessions). Shadcn Tabs (rejected — not what the existing UI uses).

## R2: Daemon Route Organization

**Decision**: Extract artifact routes into `packages/vlt-cli/src/vlt/daemon/artifact_routes.py` using `APIRouter(prefix="/api/artifacts")`, following the `cronban_routes.py` pattern. WebSocket endpoints for HMR and state streaming go in the same file.

**Rationale**: server.py is already large. The cronban extraction pattern proves the approach works. Artifact routes include CRUD, backend proxy, frontend serving, and multiple WebSocket endpoints — enough to warrant extraction.

**Alternatives considered**: Everything in server.py (rejected — too much code). Separate FastAPI sub-application (rejected — overkill, router is sufficient).

## R3: File Watching

**Decision**: Use `watchdog` (already installed in backend deps, v4.0+) with its async observer pattern. Add `watchdog` to `packages/vlt-cli/pyproject.toml` as well since the watcher runs in the daemon.

**Rationale**: `watchdog` is already a project dependency. It uses OS-native events (inotify on Linux). The daemon's existing file-monitoring (history.jsonl) uses polling — the artifact watcher should use proper inotify for lower latency. `watchfiles` is an alternative but adds a new dependency unnecessarily.

**Alternatives considered**: `watchfiles` (rejected — new dependency when watchdog is already available). Polling loop like history watcher (rejected — too high latency for HMR, need sub-100ms detection).

## R4: Screenshot Capture

**Decision**: Add `playwright` as an optional dependency to the daemon. Use async Playwright to launch a headless Chromium instance, navigate to the artifact's frontend URL served by the daemon, capture a screenshot, and save it to the artifact's `.vlt/screenshots/` directory.

**Rationale**: Playwright is the standard for headless browser automation. It's more reliable than html2canvas (which can't cross iframe boundaries). The server-side approach means we control the environment. The Playwright MCP tools already exist in the user's environment, but those are for the MCP client — we need Playwright as a library in the daemon for programmatic use.

**Alternatives considered**: html2canvas inside iframe via postMessage (rejected — unreliable with complex CSS, canvas elements, web fonts). Puppeteer (rejected — Playwright has better Python support). Screenshot service (rejected — unnecessary external dependency).

## R5: MCP Dynamic Tool Registration

**Decision**: For Phase 1, artifact MCP tools require an MCP server restart to register. The daemon exposes a `/api/mcp/restart` endpoint that the artifact deployment flow calls. Long-term (Phase 2+), explore FastMCP's internal tool registry for runtime manipulation.

**Rationale**: FastMCP 3.x doesn't expose a public runtime add/remove tool API. The MCP server is a STDIO process — Claude Code reconnects automatically on restart. The restart approach is simple, reliable, and covers the use case (tools only change when an artifact is deployed/undeployed, which is infrequent).

**Alternatives considered**: Runtime tool injection via FastMCP internals (rejected — private API, fragile). Proxy tool that routes to artifacts dynamically (considered — would work as a single `artifact_call(artifact_id, action, params)` tool, but loses the discoverability benefit of per-artifact tools). Restart is the pragmatic choice.

## R6: Multi-Instance Connector Schema

**Decision**: Add `instance_id TEXT NOT NULL DEFAULT 'default'` to the `connector_configs` table primary key. Migrate existing data by setting all rows to `instance_id='default'`. Update `ConnectorService` methods to accept an optional `instance_id` parameter (defaulting to `'default'` for backwards compatibility).

**Rationale**: The current schema uses `(user_id, connector_name, config_key)` as PK. Adding `instance_id` to the PK allows multiple credential sets per connector. The default value ensures existing single-instance connectors work unchanged.

**Alternatives considered**: Separate `connector_instances` table (rejected — adds a join for every config lookup). JSON blob per instance (rejected — breaks the existing key-value pattern). Prefixed keys like `bot-1__api_key` (rejected — hacky, breaks existing key lookup logic).

## R7: Vision Model Detection

**Decision**: Parse `architecture.modality` and `input_modalities` from OpenRouter's model API response. Add `supports_vision: bool` to `ModelInfo`. For GLM, hardcode `glm-4.6v` as vision-capable. For Gemini, hardcode `gemini-2.0-flash-exp` and `gemini-1.5-pro` as vision-capable.

**Rationale**: OpenRouter's API returns modality data that we currently discard. Parsing it is a one-line addition. Hardcoding for GLM and Gemini is acceptable since those model lists are already hardcoded.

**Alternatives considered**: Capability probing at runtime (rejected — slow, requires sending a test image to each model). User-only configuration (rejected — bad UX, users shouldn't need to know which models support vision).

## R8: Artifact Backend Process Model

**Decision**: Each artifact backend runs as an isolated Python subprocess managed by the daemon, similar to SDK sessions. The daemon communicates with the backend via stdin/stdout JSON lines (same protocol as SDK sessions). The backend process exposes a `handle(action, params) -> dict` function contract. The daemon wraps this in a thin harness script that reads JSON from stdin, calls `handle()`, and writes responses to stdout.

**Rationale**: Follows the proven SDK session pattern (`asyncio.create_subprocess_exec`, stdin/stdout pipes, reader task). Process isolation prevents one artifact's crash from affecting others or the daemon. The JSON-lines protocol is already well-tested in the codebase.

**Alternatives considered**: Shared REPL sandbox like OracleV2 (rejected — no isolation between artifacts). Unix domain sockets per artifact (rejected — more complex than stdin/stdout with no clear benefit). HTTP per artifact (rejected — port allocation complexity, the daemon should proxy).

## R9: Artifact Storage

**Decision**: Artifacts live on disk at `data/artifacts/{user_id}/{artifact_id}/` with metadata in a new SQLite `artifacts` table. Git init on creation, auto-commit on state transitions. Versioning is just git history — lightweight, no extra schema needed.

**Rationale**: Disk storage matches the vault pattern (files on disk, metadata in SQLite). Git is already in the stack. Auto-committing on state transitions gives free version history with meaningful commit messages ("state: building → testing"). No need for a custom versioning system.

**Alternatives considered**: SQLite-only with source in blob columns (rejected — can't edit files from Claude Code sessions). Separate git repos per artifact (rejected — overhead of remote management). Version table with full source snapshots (rejected — git does this better).

## R10: Proxy Profile System

**Decision**: New `proxy_profiles` table in backend SQLite: `(user_id, name, proxy_url, proxy_username, proxy_password)`. Connector instances reference a proxy profile by name. The connector service injects proxy settings into httpx/requests when making calls for that instance.

**Rationale**: Separating proxy config from connector config allows reuse (many connector instances can share one proxy). The settings UI gets a "Proxy Profiles" management section.

**Alternatives considered**: Per-instance proxy fields (rejected — duplicates config when multiple instances use the same proxy). System-wide proxy env vars (rejected — too coarse, different connectors need different proxies).
