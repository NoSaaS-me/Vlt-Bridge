# Tasks: Oracle & Librarian CodeAct Rework

**Input**: Design documents from `/specs/023-oracle-codeact-rework/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Format**: `[ID] [P?] [Story] Description`
- **[P]**: Parallelizable (different files, no shared state dependencies)
- **[Story]**: User story this task belongs to (US1–US4)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: New dependencies, Docker sidecar, package scaffolding. No existing code modified.

- [x] T001 Add `langgraph-codeact>=0.1.3`, `langgraph-checkpoint-sqlite`, `graphiti-core[anthropic]>=0.28.1` to `backend/pyproject.toml` optional deps group `oracle-v2`
- [x] T002 [P] Create `docker/graphiti-compose.yml` with FalkorDB service (port 6379, named volume `falkordb_data`) ← already exists, verify content
- [x] T003 [P] Create `backend/src/services/oracle_v2/__init__.py` (empty package marker)
- [x] T004 [P] Create `backend/src/services/oracle_v2/tools/__init__.py` (empty package marker)
- [x] T005 [P] Create `backend/src/services/memory/__init__.py` (empty package marker)

**Checkpoint**: `uv pip install -e ".[oracle-v2]"` succeeds; all new packages importable

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that ALL user stories depend on. Must be complete before any story work begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Create `backend/src/services/oracle_v2/state.py`: `OracleState(CodeActState)` TypedDict with `user_id: str`, `project_id: str`, `plan: Optional[list[str]]`, `plan_step: int`, `recalled_facts: list[str]`. Import from `langgraph_codeact`. Do NOT include `api_key` or `oracle_model` fields (security — passed via `config["configurable"]` only, never checkpointed).
- [x] T007 Create `backend/src/services/oracle_v2/sandbox.py`: `make_sandbox_eval(tools: list[Callable]) -> Callable[[str, dict], tuple[str, dict]]` — sync `eval_fn` compatible with `create_codeact`. Includes: `ALLOWED_MODULES` set (`re`, `json`, `math`, `datetime`, `collections`, `itertools`, `functools`), custom `__import__` restrictor, thread-based 30s timeout, `new_vars` extraction (pickle-safe: skip non-serializable values). Injects tool callables directly into `_locals` dict passed to `exec`.
- [x] T008 Add `run_shell()` to `backend/src/services/oracle_v2/sandbox.py`: shell command callable injected into `_locals` (NOT callable via exec string eval). `SHELL_ALLOWLIST = {"git", "grep", "find", "ls", "cat", "head", "tail", "wc", "diff", "rg"}`. `GIT_SUBCOMMAND_ALLOWLIST = {"log", "diff", "status", "show", "blame", "branch", "tag", "stash", "shortlog"}`. Reject shell chaining patterns (`;`, `&&`, `||`, `$()`, backticks). Return stdout as string, raise `ValueError` on blocked commands.
- [x] T009 Add `oracle_threads` table to `backend/src/services/database.py` `initialize()` method: `CREATE TABLE IF NOT EXISTS oracle_threads (thread_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, project_id TEXT NOT NULL, title TEXT, created_at TEXT NOT NULL, last_active_at TEXT NOT NULL); CREATE INDEX IF NOT EXISTS idx_oracle_threads_user ON oracle_threads(user_id, last_active_at DESC);`
- [x] T010 Add `OracleThreadSummary` and `OracleThreadListResponse` Pydantic models to `backend/src/models/oracle.py` (alongside existing models, no changes to existing models)
- [x] T011 Add `AsyncSqliteSaver` initialization to `backend/src/api/main.py` lifespan (lines 40-64): open `AsyncSqliteSaver.from_conn_string(settings.oracle_checkpoint_db)` as async context manager, store on `app.state.oracle_checkpointer`. Add `ORACLE_CHECKPOINT_DB` env var to `backend/src/services/config.py` (default: `"data/checkpoints.db"`).
- [x] T012 Add Graphiti client initialization to `backend/src/api/main.py` lifespan: import `Graphiti` from `graphiti_core`, init with `bolt://localhost:6379`, store on `app.state.graphiti`. Read `FALKORDB_URL` env var (default: `"bolt://localhost:6379"`). Wrap in try/except — if FalkorDB unreachable, log warning and set `app.state.graphiti = None` (graceful degradation per edge case spec).

**Checkpoint**: Server starts cleanly with FalkorDB running; `oracle_threads` table exists in `data/index.db`; `app.state.oracle_checkpointer` and `app.state.graphiti` set in lifespan

---

## Phase 3: User Story 1 — Multi-Turn Conversation (Priority: P1) 🎯 MVP

**Goal**: Replace ephemeral RLMOracleWrapper with OracleV2Wrapper backed by AsyncSqliteSaver. Same `/api/oracle/stream` signature. Context persists across turns using same `thread_id`.

**Independent Test**: Ask "Where is the authentication middleware?" then immediately ask "What does it import?" — second response must NOT show repeated `search_code` tool_call chunks.

### Implementation

- [x] T013 [US1] Create `backend/src/services/oracle_v2/tools/code_tools.py`: wrap existing `ToolExecutor` code search methods (`search_code`, `read_file`, `get_repo_map`) as Python callables with docstrings. Accept `project_id` from closure (bound at wrapper init time, not as arg). Return plain strings — no Pydantic models (CodeAct tools must return strings for REPL display).
- [x] T014 [P] [US1] Create `backend/src/services/oracle_v2/tools/vault_tools.py`: wrap existing vault service (`vault_search`, `vault_read`, `vault_write`) as Python callables with docstrings. Accept `user_id`, `project_id` from closure.
- [x] T015 [US1] Create `backend/src/services/oracle_v2/streaming.py`: `oracle_to_sse(events: AsyncIterator) -> AsyncGenerator[OracleStreamChunk, None]` adapter. Map `on_chat_model_stream` → `content`, `on_tool_start` → `tool_call`, `on_tool_end` → `tool_result`, `on_custom_event` name=`repl_stdout` → `progress`, graph end → `done` (set `context_id=thread_id`). On first chunk, emit `context_update` chunk with `context_id=thread_id` if this is a new thread.
- [x] T016 [US1] Create `backend/src/services/oracle_v2/graph.py`: `build_oracle_graph(tools: list[Callable], checkpointer) -> CompiledGraph`. Call `create_codeact(model, tools, eval_fn, state_schema=OracleState)` with `eval_fn` from `make_sandbox_eval(tools)`. Compile with `checkpointer`. Add `build_oracle_graph_from_request(request: OracleRequest, app_state) -> tuple[CompiledGraph, dict]` helper that assembles `config["configurable"]` with `thread_id`, `api_key` (from `UserSettingsService`), `oracle_model` (from `UserSettingsService` or `request.model`). Model constructed from `config["configurable"]["oracle_model"]` resolved inside graph, NOT stored in state.
- [x] T017 [US1] Create `OracleV2Wrapper` in `backend/src/services/oracle_v2/__init__.py`: `__init__(user_id, api_key, project_id, model, max_tokens)` constructor matching `RLMOracleWrapper` signature. `async def process_query(query: str, context_id: Optional[str]) -> AsyncGenerator[OracleStreamChunk, None]`: if `context_id` is None, generate new UUID thread_id + insert row into `oracle_threads`; stream via `graph.astream_events(version="v2", config=config)` piped through `oracle_to_sse()`. Store active task reference for cancellation.
- [x] T018 [US1] Swap wrapper in `backend/src/api/routes/oracle.py`: update import line to `OracleV2Wrapper`, update non-streaming instantiation (~line 162), update streaming instantiation (~line 298). No changes to route signatures, request parsing, or response handling.
- [x] T019 [US1] Update `oracle_threads` row `last_active_at` and auto-derive `title` (first 60 chars of first user message, trimmed) in `OracleV2Wrapper.process_query()` after a turn completes.
- [x] T020 [US1] Write unit tests: `backend/tests/unit/test_oracle_v2_state.py` — verify `OracleState` fields, no `api_key` field; `backend/tests/unit/test_oracle_v2_streaming.py` — verify chunk type mapping with mock events.
- [x] T021 [US1] Write integration test: `backend/tests/integration/test_oracle_v2_multiturn.py` — two-turn test using `MemorySaver` (not `AsyncSqliteSaver`) for speed; verify `context` dict carries variables from turn 1 into turn 2.

**Checkpoint**: `/api/oracle/stream` responds with SSE stream. Two sequential requests with same `context_id` show second turn uses context from first (verified via `tool_call` chunks — no repeated retrieval).

---

## Phase 4: User Story 2 — Cross-Session Memory (Priority: P2)

**Goal**: Graphiti knowledge graph stores facts per project. New sessions recall relevant facts at turn start via `recalled_facts` in `OracleState`. Old facts superseded by contradicting new facts.

**Independent Test**: Tell Oracle "the rate limiter is in src/api/middleware/rate_limit.py" in session A. Kill and restart server. Ask "where is rate limiting handled?" in new session B — Oracle answers correctly without searching.

### Implementation

- [x] T022 [US2] Create `backend/src/services/memory/client.py`: `get_graphiti(app_state) -> Optional[Graphiti]` — returns `app.state.graphiti` or `None` if unavailable. `group_id_for(user_id: str, project_id: str) -> str` helper: returns `f"{user_id}:{project_id}"`. `user_group_id(user_id: str) -> str`: returns `f"{user_id}:_user"`.
- [x] T023 [US2] Create `backend/src/services/memory/loader.py`: `async def load_recalled_facts(graphiti: Optional[Graphiti], query: str, user_id: str, project_id: str, limit: int = 10) -> list[str]`. If `graphiti is None`, return `[]` (graceful degradation). Call `graphiti.search(query=query, group_id=group_id_for(user_id, project_id), limit=limit)`. Return list of `edge.fact` strings from results where `edge.invalid_at is None` (currently valid only).
- [x] T024 [US2] Create `backend/src/services/memory/writer.py`: `async def write_episode(graphiti: Optional[Graphiti], episode_body: str, user_id: str, project_id: str, thread_id: str, turn_index: int) -> None`. If `graphiti is None`, return silently. Call `graphiti.add_episode(name=f"oracle-turn-{thread_id}-{turn_index}", episode_body=episode_body, group_id=group_id_for(user_id, project_id), source="message", reference_time=datetime.utcnow())`. Fire as `asyncio.create_task()` (background — do not block turn response).
- [x] T025 [US2] Add `memory_loader_node` and `memory_writer_node` to `backend/src/services/oracle_v2/nodes.py`: `memory_loader_node(state: OracleState, config) -> dict` — calls `load_recalled_facts()` with last user message as query; returns `{"recalled_facts": [...]}`. `memory_writer_node(state: OracleState, config) -> dict` — assembles `episode_body` from last assistant message; fires `write_episode()` as background task; returns `{}`.
- [x] T026 [US2] Create `backend/src/services/oracle_v2/tools/memory_tools.py`: `remember(fact: str) -> str` — stores explicit named fact via `graphiti.add_episode()` synchronously (called from REPL, use `asyncio.run_coroutine_threadsafe()`). `recall(query: str) -> str` — returns formatted fact list from `load_recalled_facts()`. Both accept `graphiti`, `user_id`, `project_id` from closure.
- [x] T027 [US2] Wire `memory_loader_node` and `memory_writer_node` into `graph.py`: insert `memory_loader_node` before the CodeAct agent node; insert `memory_writer_node` after the CodeAct agent node (in the turn-end path). Inject `recalled_facts` into agent system prompt prefix: `"Recalled facts from memory:\n{chr(10).join(recalled_facts)}\n\n"` when non-empty.
- [x] T028 [US2] Add `remember` and `recall` tools to default tool list in `graph.py` `build_oracle_graph()`.
- [x] T029 [US2] Write unit test `backend/tests/unit/test_oracle_v2_tools.py`: verify `remember()` and `recall()` with mock Graphiti; verify `None` graphiti returns empty gracefully.

**Checkpoint**: Fact stored in session A is recalled in session B (server restart between). `tool_call` chunk with `remember(...)` visible in stream when agent explicitly stores a fact.

---

## Phase 5: User Story 3 — General-Purpose Tools (Priority: P3)

**Goal**: Expand tool registry with shell commands (allowlisted), web search, vault write, thread tools, and meta tools. Agent can run `git log`, save vault notes, and search the web.

**Independent Test**: Ask "What changed in the auth module in the last 10 git commits?" — `tool_call` chunk shows `git_log(...)` call; response includes actual commit messages.

### Implementation

- [x] T030 [US3] Create `backend/src/services/oracle_v2/tools/shell_tools.py`: `git_log(path: str = "", n: int = 10) -> str`, `git_diff(ref1: str = "HEAD~1", ref2: str = "HEAD", path: str = "") -> str`, `git_blame(path: str, start_line: int = 1, end_line: int = 50) -> str`, `find_files(pattern: str, directory: str = ".") -> str`, `grep_files(pattern: str, path: str = ".", flags: str = "") -> str`. All delegate to `run_shell()` from `sandbox.py` with pre-validated args. Docstrings describe args and return format.
- [x] T031 [P] [US3] Create `backend/src/services/oracle_v2/tools/web_tools.py`: `web_search(query: str, num_results: int = 5) -> str` — calls existing `TavilySearchService` or `OpenRouterSearchService` based on user settings (resolved from closure). `web_fetch(url: str) -> str` — HTTP GET with 30s timeout, returns text content (truncated to 8000 chars). `deep_research(query: str) -> str` — delegates to existing `DeepResearchWrapper` if `deep_research` flag enabled on request; otherwise returns error string.
- [x] T032 [P] [US3] Create `backend/src/services/oracle_v2/tools/thread_tools.py`: `read_thread(thread_name: str) -> str`, `search_threads(query: str) -> str`, `list_threads() -> str` — wrap existing vlt thread service methods. Accept `user_id` from closure.
- [x] T033 [P] [US3] Create `backend/src/services/oracle_v2/tools/meta_tools.py`: `list_tools() -> str` — returns formatted list of all injected tool names and their docstring first lines. `update_plan(new_steps: list[str]) -> str` — updates `state["plan"]` via LangGraph `Command` (use `asyncio.run_coroutine_threadsafe()`). `delegate_librarian(task: str) -> str` — calls existing Librarian service synchronously as a callable; returns result string.
- [x] T034 [US3] Add all new tools to default tool list in `backend/src/services/oracle_v2/graph.py` `build_oracle_graph()`. Gate `web_search`/`web_fetch`/`deep_research` behind `deep_research` flag from `OracleRequest` (passed via configurable).
- [x] T035 [US3] Extend `backend/src/services/oracle_v2/sandbox.py` `ALLOWED_MODULES` to include `subprocess` (read-only — `run_shell()` uses subprocess but the module itself isn't exposed to the REPL). Update `make_sandbox_eval()` to expose `run_shell` in `_locals` alongside tool callables.
- [x] T036 [US3] Write unit test additions to `backend/tests/unit/test_oracle_v2_sandbox.py`: verify `run_shell()` blocks `;`, `&&`, `||`, `$()` patterns; verify allowlisted commands pass; verify git subcommand allowlist.

**Checkpoint**: `git_log("backend/src/api/routes/oracle.py", n=10)` called from Oracle REPL returns actual commit log. `web_search("langgraph codeact")` returns results when Tavily key configured.

---

## Phase 6: User Story 4 — Structured Planning (Priority: P4)

**Goal**: Rule-based classifier routes complex tasks through LLM planner (Sonnet model). Simple factual questions bypass planner. Plan tracked in `OracleState.plan` / `plan_step`.

**Independent Test**: Ask "Find all files that import from src/services/vault.py and tell me what functions they use" — stream shows planner generating multi-step plan before tool calls begin. Ask "What is the vault service?" — stream shows NO planner step (direct answer).

### Implementation

- [x] T037 [US4] Add `planner_node` to `backend/src/services/oracle_v2/nodes.py`: accepts `OracleState`, generates decomposed plan as `list[str]` using Sonnet model (resolved from `config["configurable"]["oracle_model"]`). Returns `{"plan": [...], "plan_step": 0}`. System prompt: "You are a task decomposer. Break the user's request into ordered, concrete steps for a code search agent. Output only a numbered list."
- [x] T038 [US4] Add rule-based classifier to `backend/src/services/oracle_v2/graph.py`: `classify_query(query: str) -> Literal["direct", "plan"]`. `DIRECT_PREFIXES = {"what is", "define", "when was", "who is", "list the", "show me", "where is"}`. `COMPLEX_KEYWORDS = {"refactor", "implement", "build", "analyze", "compare", "find all", "trace", "audit", "summarize all", "across the codebase"}`. Rules: token count < 15 AND starts with direct prefix → `"direct"`. Complex keyword present OR token count > 50 → `"plan"`. Default → `"direct"`.
- [x] T039 [US4] Wire conditional routing into `graph.py`: after `memory_loader_node`, add conditional edge: `classify_query(last_user_message)` → `"planner_node"` (if `"plan"`) or directly to CodeAct agent (if `"direct"`). After `planner_node`, edge goes to CodeAct agent.
- [x] T040 [US4] Add `update_plan` meta tool (from T033) wiring: when agent calls `update_plan(new_steps)`, update `OracleState.plan` and emit a `progress` chunk with plan text so frontend can display plan state.
- [x] T041 [US4] Emit `progress` chunk with plan content at planner_node completion: emit `OracleStreamChunk(type="progress", content=f"Plan:\n{chr(10).join(plan)}")` via `on_custom_event` in `streaming.py`.

**Checkpoint**: Complex task request shows `progress` chunk with numbered plan steps before any `tool_call` chunks. Simple "what is" question shows no plan in stream.

---

## Phase 7: Thread Management Endpoints & Polish

**Purpose**: Thread listing/delete/history API endpoints, integration test, error handling hardening.

- [x] T042 Add thread management endpoints to `backend/src/api/routes/oracle.py` per `contracts/oracle-threads.yaml`: `GET /api/oracle/threads`, `GET /api/oracle/threads/{thread_id}`, `DELETE /api/oracle/threads/{thread_id}`, `PATCH /api/oracle/threads/{thread_id}`, `GET /api/oracle/threads/{thread_id}/history`. Thread history loaded from `await app.state.oracle_checkpointer.aget_state(config)` (async — never use sync `get_state()`).
- [x] T043 [P] Add cancellation support to `OracleV2Wrapper`: `_active_tasks: dict[str, asyncio.Task]` keyed by `thread_id`. Set task in `process_query()` on stream start; clear on completion. Add `cancel(thread_id: str) -> bool` method. Wire `POST /api/oracle/cancel/{context_id}` in `oracle.py` to call `wrapper.cancel(thread_id)`.
- [x] T044 [P] Harden `oracle_to_sse()` in `streaming.py`: catch `asyncio.CancelledError` → emit `done` chunk (not `error`) so state remains resumable. Catch other exceptions → emit `error` chunk + log.
- [x] T045 [P] Harden `memory_loader_node` and `memory_writer_node` in `nodes.py`: wrap all Graphiti calls in try/except; on failure log warning and continue (graceful degradation per spec edge case).
- [x] T046 [P] Write unit test `backend/tests/unit/test_oracle_v2_sandbox.py`: verify non-serializable REPL vars are skipped in `new_vars` extraction; verify 30s timeout raises `TimeoutError`; verify blocked import raises `ImportError`.
- [x] T047 Run `.specify/scripts/bash/update-agent-context.sh claude` to add `oracle_v2` package paths and new technologies to `CLAUDE.md`

---

## Dependencies

```
Phase 1 (Setup)
    └── Phase 2 (Foundation: state, sandbox, DB schema, lifespan)
            └── Phase 3 (US1: graph, streaming, wrapper, route swap)   ← MVP
                    ├── Phase 4 (US2: memory nodes, Graphiti tools)
                    ├── Phase 5 (US3: shell/web/thread/meta tools)
                    └── Phase 6 (US4: planner node, classifier)
                            └── Phase 7 (Thread endpoints, cancellation, polish)
```

**US3 and US4 can begin in parallel once Phase 3 (US1) is complete.**
**US2 can also begin once Phase 2 foundational work is done (Graphiti client exists).**

---

## Parallel Execution Examples

### After Phase 2 completes (US1 in progress):
```
Worker A: T013 code_tools.py
Worker B: T014 vault_tools.py (independent file)
Worker C: T020 unit tests state + streaming
→ Main: T015 streaming.py → T016 graph.py → T017 OracleV2Wrapper → T018 route swap
```

### After Phase 3 completes (US2, US3, US4 can parallelize):
```
Worker A: T022-T029 (US2 — Graphiti memory)
Worker B: T030-T036 (US3 — shell/web/thread tools)
→ Main: T037-T041 (US4 — planner; depends on graph.py from Phase 3)
```

### Phase 7 (all polish tasks are independent):
```
Workers A-D in parallel: T042, T043, T044, T045, T046
→ Main: T047 context update
```

---

## Implementation Strategy

**MVP scope**: Phases 1–3 (US1 only)

After Phase 3 is complete:
- `/api/oracle/stream` works end-to-end with durable multi-turn state
- `context_id` is a real LangGraph `thread_id`
- Frontend requires zero changes
- Old `RLMOracleWrapper` still exists (frozen, not deleted) for rollback

**Rollback plan**: If Phase 3 fails validation, revert the 3-line wrapper swap in `oracle.py`. All oracle_v2 code is isolated — no existing code modified except `main.py` lifespan and `oracle.py` route file.

---

## Task Summary

| Phase | Story | Task Count | Parallelizable |
|---|---|---|---|
| 1: Setup | — | 5 | 4 |
| 2: Foundation | — | 7 | 0 |
| 3: US1 Multi-Turn | P1 | 9 | 2 |
| 4: US2 Memory | P2 | 8 | 0 |
| 5: US3 Tools | P3 | 7 | 4 |
| 6: US4 Planning | P4 | 5 | 0 |
| 7: Polish | — | 6 | 5 |
| **Total** | | **47** | **15** |
