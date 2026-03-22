# Research: Oracle & Librarian CodeAct Rework

**Sources**: 4 parallel deep-research agents, March 2026. Live package source verified.
**Full detailed findings**: `Ai-notes/2026-03-11/Oracle-Rework/SPEC-FINAL.md`

---

## Decision: CodeAct Framework

**Decision**: `langgraph-codeact` v0.1.3

**Rationale**: LLM writes Python as action (CodeAct pattern). Tools are Python callables injected directly into the REPL namespace — no JSON schema tool-calling needed. `CodeActState.context: dict[str, Any]` persists REPL variables automatically via the LangGraph checkpointer. This solves multi-turn state with zero custom code.

**Alternatives considered**:
- Hand-rolled REPL loop (current RLM Oracle) — multi-turn state broken by design (ephemeral session)
- Standard tool-calling loop — prior attempt (OracleAgent), too many failure modes for long tasks
- DeerFlow — web research pipeline only, not codebase exploration

**Key API**:
```python
create_codeact(model, tools, eval_fn, *, prompt=None, state_schema=CodeActState) -> StateGraph
# Returns uncommitted StateGraph — call .compile(checkpointer=...)
```
Tools as Python callables with docstrings. All tool signatures injected into system prompt. BYOS sandbox via `eval_fn: (code: str, locals: dict) -> tuple[str, dict]`.

---

## Decision: Thread State Persistence

**Decision**: `AsyncSqliteSaver` from `langgraph-checkpoint-sqlite`

**Rationale**: Async-native, FastAPI lifespan compatible, zero extra infrastructure (SQLite already in use). Drop-in upgrade path to `AsyncPostgresSaver` for future multi-process scale.

**Alternatives considered**:
- `MemorySaver` — in-process only, lost on restart
- `PostgresSaver` — adds infra dependency, unnecessary for local dev tool

**Key API**:
```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
async with AsyncSqliteSaver.from_conn_string("data/checkpoints.db") as cp:
    graph = build_oracle_graph(...).compile(checkpointer=cp)
```
Must open once in FastAPI `lifespan`, share across all requests.

**Critical bug**: Never call sync `get_state_history()` with async checkpointer (hangs, Issue #2992). Always use `aget_state_history()`.

**No list_threads() API**: LangGraph has no thread listing by user. Maintain own `oracle_threads` table.

---

## Decision: Cross-Session Memory

**Decision**: Graphiti (`graphiti-core[anthropic]` v0.28.1) + FalkorDB

**Rationale**: Bi-temporal knowledge graph — facts have `valid_at` and `invalid_at` timestamps. When code moves, old path fact becomes `invalid_at = now`, new path stored as current. LangMem has no temporal modeling. Zep Community was deprecated April 2025.

**Alternatives considered**:
- Zep Community — deprecated April 2025, no supported self-hosted option
- Zep Cloud — vendor lock-in, 1K episodes/month free tier too small
- LangMem — LangGraph-native but flat key-value, no temporal tracking

**FalkorDB vs Neo4j**: FalkorDB is ~256MB RAM (Redis-protocol); Neo4j requires JVM (~2-4GB). FalkorDB is Bolt-protocol compatible.

**Key API**:
```python
from graphiti_core import Graphiti
graphiti = Graphiti("bolt://localhost:6379", "", "")
# Add episode (entity extraction happens async via Anthropic)
await graphiti.add_episode(name=..., episode_body=..., group_id=f"{user_id}:{project_id}")
# Semantic search (returns edges with valid_at/invalid_at)
results = await graphiti.search(query=..., group_id=..., limit=5)
```
Multi-tenancy: `group_id=f"{user_id}:{project_id}"` for project scope, `f"{user_id}:_user"` for user scope.

---

## Decision: Sandbox

**Decision**: Expanded in-process exec with module allowlist

**Rationale**: Tool callables (search_code, vault_read, etc.) are Python function objects. They cannot cross E2B VM boundary or Docker subprocess without an HTTP microservice layer. In-process exec with custom `__import__` provides zero latency, direct tool invocation, and sufficient security for a trusted local tool.

**Alternatives considered**:
- E2B — tool callables can't be passed into microVM; requires HTTP proxy for tools ($0.001/task, 150ms cold start)
- Docker subprocess — same tool-callability problem
- RestrictedPython (current) — blocks numpy, no shell, no file I/O beyond injected objects; too limited for expanded tool set

**Shell security**: `run_shell()` exposed as an explicit Python callable in `_locals` — not callable via `exec`. Shell allowlist: `git` (subcommands: log, diff, status, show, blame, branch, tag, stash, shortlog), `grep`, `find`, `ls`, `cat`, `head`, `tail`, `wc`, `diff`, `rg`. Reject shell chaining (`;`, `&&`, `||`, `$()`).

**`async get_stream_writer()` bug**: Issue #6447 — custom events silently dropped in async eval_fn. Workaround: use sync `eval_fn` + run async tool calls via `asyncio.run_coroutine_threadsafe()`. Or accept per-iteration (not per-line) progress via `stream_mode="updates"`.

---

## Decision: Planner Node

**Decision**: Rule-based classifier first → LLM planner (Sonnet) only for complex tasks

**Rationale**: Rule-based pre-filter costs zero tokens for obvious cases (short factual questions). LLM planner uses the capable model (Sonnet-class) because planning quality matters — decomposing ambiguous multi-step tasks requires reasoning, not speed. Paper `arxiv:2503.09572` shows 3× success rate improvement with planning (9.85% → 29.63%).

**Alternatives considered**:
- Always plan with cheap model (haiku) — haiku makes planning errors on ambiguous requests; quality loss not worth latency savings
- Never plan — current Oracle behavior; root cause of losing track on complex tasks
- Embedding-based routing — overkill for <40 tool set

**Classifier**:
```python
DIRECT_PREFIXES = {"what is", "define", "when was", "who is", "list the", "show me", "where is"}
COMPLEX_KEYWORDS = {"refactor", "implement", "build", "analyze", "compare", "find all", ...}
# <15 tokens + direct prefix → skip planning
# complex keyword or >50 tokens → invoke LLM planner
```

---

## Decision: Streaming

**Decision**: `graph.astream_events(version="v2")` for SSE adapter

**Rationale**: `astream_events` gives named events (`on_chat_model_stream`, `on_tool_start`, `on_tool_end`, `on_custom_event`) that map cleanly to existing `OracleStreamChunk` types. `stream_mode="updates"` as fallback for REPL per-iteration progress.

**Chunk type mapping**:
| LangGraph event | OracleStreamChunk type |
|---|---|
| `on_chat_model_stream` | `content` |
| `on_tool_start` | `tool_call` |
| `on_tool_end` | `tool_result` |
| `on_custom_event` name=`repl_stdout` | `progress` |
| graph end | `done` + `context_id` |

---

## Decision: Cancellation

**Decision**: `asyncio.Task.cancel()` replaces `wrapper.cancel()`

**Rationale**: LangGraph `astream()` is an async generator; standard asyncio task cancellation propagates `CancelledError` cleanly. `LangGraph.interrupt()` is human-in-the-loop (graph pauses and awaits input), not request cancellation.

**Known issue**: Subgraph `CancelledError` propagation broken in FastAPI streaming (Issue #5682) — subgraphs may continue briefly after cancellation. Acceptable for this use case.

---

## Decision: Thread Listing

**Decision**: Maintain own `oracle_threads` SQLite table

**Rationale**: LangGraph has no `list_threads()` API (open issue #1320, unresolved in 1.1.0). Direct SQL on checkpoint DB requires `thread_id` naming conventions and is fragile. Own table is clean and fast.

**Schema**:
```sql
CREATE TABLE oracle_threads (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);
CREATE INDEX idx_oracle_threads_user ON oracle_threads(user_id, last_active_at DESC);
```

---

## Decision: API Key Handling

**Decision**: API keys passed via LangGraph `config["configurable"]`, never stored in `OracleState`

**Rationale**: `OracleState` is checkpointed to SQLite. Storing API keys there creates a plaintext credential store. LangGraph's `config["configurable"]` is not persisted by the checkpointer — safe for per-request secrets.

```python
config = {
    "configurable": {
        "thread_id": thread_id,
        "api_key": api_key,          # not checkpointed
        "oracle_model": oracle_model, # not checkpointed
    }
}
```

---

## Codebase Integration Findings (from brownfield agent)

**Lifespan**: Already exists at `main.py:40-64`. Add oracle_v2 init inside it before `session_manager.run()`.

**Wrapper interface**: `async def process_query(query: str, context_id: Optional[str]) -> AsyncGenerator[OracleStreamChunk, None]` — oracle.py calls this on both `RLMOracleWrapper` and `DeepResearchWrapper`. `OracleV2Wrapper` must implement the same signature for drop-in compatibility.

**Route swap**: 3 locations in `oracle.py` — import line, non-streaming instantiation (~line 162), streaming instantiation (~line 298). Route signatures unchanged.

**OracleRequest fields**: `question`, `sources`, `explain`, `model`, `thinking`, `max_tokens` (default 16000), `context_id`, `project_id`, `deep_research`.

**OracleStreamChunk types**: `thinking`, `content`, `source`, `tool_call`, `tool_result`, `done`, `error`, `system`, `context_update`, `progress`.

**UserSettingsService getters**: `get_openrouter_api_key`, `get_glm_api_key`, `get_tavily_api_key`, `get_search_provider`, `get_oracle_model`, `get_subagent_model`.

**Storage**: `data/index.db` (7.9MB) is the main DB containing `oracle_bridge_history`, `context_trees`, `context_nodes`. Add `oracle_threads` table here. LangGraph checkpoints go in a separate `data/checkpoints.db`.

**ContextTreeService**: Route calls `_save_oracle_turn_to_tree()` which reads the `context_id` from the `done` chunk. oracle_v2 wrapper must emit `OracleStreamChunk(type="done", context_id=thread_id)` so the existing tree persistence still works.

**New pyproject.toml deps**: `langgraph-codeact>=0.1.3`, `langgraph-checkpoint-sqlite`, `graphiti-core[anthropic]>=0.28.1`.
