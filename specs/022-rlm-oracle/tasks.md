# Tasks: RLM Oracle

**Input**: Design documents from `/specs/022-rlm-oracle/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: Included — unit tests required per plan.md constitution check.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- All paths are relative to repository root

---

## Phase 1: Setup & Migration

**Purpose**: Install new dependency, migrate OpenRouterClient out of BT, create file skeletons.

- [x] T001 Add `RestrictedPython>=8.0` to `backend/pyproject.toml` dependencies section
- [x] T002 Copy `backend/src/bt/services/openrouter_client.py` to `backend/src/services/openrouter_client.py` and update its import path (do NOT delete from bt/ yet — bt/ deletion happens in Phase 7)
- [x] T003 [P] Update all imports of `bt.services.openrouter_client` within `backend/src/bt/` to reference `services.openrouter_client` (no external absolute imports found — bt/ uses relative imports internally)
- [x] T004 [P] Create empty skeleton `backend/src/services/repl_executor.py` with module docstring referencing FR-015, FR-016, FR-017
- [x] T005 [P] Create empty skeleton `backend/src/services/project_context.py` with module docstring referencing FR-007 through FR-014
- [x] T006 [P] Create empty skeleton `backend/src/services/rlm_oracle.py` with module docstring referencing FR-001 through FR-006, FR-019 through FR-023

**Checkpoint**: `uv pip install -e .` succeeds in `backend/`; no import errors on existing oracle route

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core execution engine and data structures that ALL user stories depend on. No user story work begins until this phase is complete.

**⚠️ CRITICAL**: US1–US4 all require REPLExecutor, FileManifest, RLMSession, and RLMPromptBuilder.

- [x] T007 Implement `REPLNamespace` class in `backend/src/services/repl_executor.py`: restricted Python namespace using `RestrictedPython.compile_restricted`, `safer_getattr`, `safe_iter`, `guarded_iter_unpack_sequence`; approved `__builtins__` dict with `open=None`, `__import__=None`; approved stdlib modules (`re`, `json`, `math`, `datetime`, `collections`, `itertools`) injected as names; `Final` sentinel detection via `'Final' in namespace._variables`; `has_final()` and `get_final()` methods; `_variables` dict persists across iterations
- [x] T008 Implement `QueuedStringIO` and `REPLExecutor.execute()` in `backend/src/services/repl_executor.py`: `QueuedStringIO(io.StringIO)` subclass whose `write()` calls `asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)` with 256-byte buffering; `execute(code)` async method runs `exec(byte_code, namespace)` via `loop.run_in_executor(None, ...)` with `threading.Thread` timeout of 30s; `SENTINEL = object()` EOF signal; returns `ExecutionResult(success, stdout_full, stdout_preview≤200chars, stdout_total_chars, error, has_final, duration_ms)` (see `data-model.md`)
- [x] T009 [P] Implement `FileManifest` and `FileEntry` dataclasses in `backend/src/services/project_context.py`: `FileEntry(path, size_bytes, language, last_modified, is_binary)` where language detected from extension; `FileManifest(files: list[FileEntry])`; `build_manifest(project_path: Path) -> FileManifest` function that walks filesystem, skips `.git/`, `node_modules/`, `__pycache__/`, `.venv/`; marks `is_binary=True` for non-text extensions; max 10,000 files (truncate with log warning)
- [x] T010 [P] Implement `RLMSession` dataclass and `RLMPromptBuilder` in `backend/src/services/rlm_oracle.py`: `RLMSession` fields: `session_id` (UUID4), `user_id`, `project_id`, `query`, `context_id`, `recursion_depth` (0=root), `iteration_count`, `max_iterations` (25 root / 8 sub), `llm_history: list[dict]`, `status` (Literal["running","completed","exhausted","error"]), `final_value`, `partial_result`, `started_at`; `RLMPromptBuilder.build_system_prompt(project_context)` returns the 5-part RLM prompt (Environment+Namespace, Execution Protocol with "CONSTRAINT: max 3 sub_oracle calls per root session", Anti-patterns, Response Format, Task-specific guidance); `build_iteration_message(exec_result)` returns ONLY metadata: stdout_preview (≤200 chars), stdout_total_chars, error if any — never full stdout (FR-003)
- [x] T011 [P] Write unit tests for `REPLExecutor` in `backend/tests/unit/test_repl_executor.py`: test basic Python exec succeeds; test `import os` raises error; test `open('/etc/passwd')` raises error; test `__subclasses__()` traversal is blocked by `safer_getattr`; test infinite loop hits 30s timeout; test `Final = "done"` sets `has_final=True`; test stdout captured in `stdout_full`; test approved modules (`re`, `json`, `math`) work correctly

**Checkpoint**: `uv run pytest backend/tests/unit/test_repl_executor.py -v` passes; REPL sandbox blocks dangerous ops; Final detection works

---

## Phase 3: User Story 1 — Cross-Codebase Synthesis (P1) 🎯 MVP

**Goal**: LLM can write Python code to explore files across the entire project, call sub_oracle programmatically in loops, synthesize answers from ≥5 non-adjacent files.

**Independent Test**: POST `/api/oracle` with `"question": "How does the connection lifecycle flow from a vlt-mcp tool call all the way to SQLite?"` → answer correctly mentions ≥3 distinct files without hallucinating.

- [x] T012 [P] [US1] Implement `TextHandle` for files in `backend/src/services/project_context.py`: fields `path`, `size_bytes`, `language`, `resource_type="file"`; `read(start_line, end_line)` — returns string content or `{"notice": "file too large", "size_bytes": N}` if >1MB or binary; `symbols()` — calls `extract_symbols_from_ast` from `packages/vlt-cli/src/vlt/core/coderag/repomap.py` via subprocess-free import (add vlt-cli to backend dev deps or copy symbol extraction); returns `list[SymbolInfo]`; `grep(pattern)` — regex search returning `list[GrepMatch]`; `chunks(max_lines=200)` — splits by function/class boundaries using `extract_symbols_from_ast` line ranges, falls back to line-count chunking; `__repr__` returns `"TextHandle(path, N lines, language)"`
- [x] T013 [P] [US1] Implement `ProjectContext` file operations in `backend/src/services/project_context.py`: `__init__(project_id, user_id)` — loads vlt project record to get `project_path`, builds `FileManifest`; `get_manifest()` → `FileManifest`; `file_count()` → int; `file(path)` → `TextHandle`; `files(pattern="**/*")` → `list[TextHandle]` (glob-filtered, handles only, no content loaded); `search(query, limit=20)` → `list[SearchMatch]` using `vlt_code_search` BM25 (falls back to grep-all if no CodeRAG index); `grep(pattern)` → `list[GrepMatch]` across all files; `build_project_context(project_id, user_id)` factory function
- [x] T014 [US1] Implement `SubOracleCallable` in `backend/src/services/rlm_oracle.py`: `__call__(prompt: str) -> str` validates `parent_session.recursion_depth < 2` (raise `RecursionDepthExceeded` otherwise); creates child `RLMSession(recursion_depth=parent+1, max_iterations=8)`; runs full RLM loop synchronously (this callable is invoked from thread pool, so blocking is OK); returns child's `final_value` as string; logs sub_oracle call count per session; emits ANS `budget.iteration.warning` when `sub_oracle` called ≥3 times in same root session
- [x] T015 [US1] Implement `RLMOracleWrapper.process_query()` in `backend/src/services/rlm_oracle.py`: async generator yielding `OracleStreamChunk`; builds `ProjectContext` via `build_project_context()`; creates `RLMSession`; builds system prompt via `RLMPromptBuilder`; REPL loop: (1) call OpenRouterClient with `llm_history`, (2) extract Python code block from LLM response, (3) execute via `REPLExecutor.execute()`, (4) yield `OracleStreamChunk(type="progress", content=stdout)` for each REPL stdout chunk, (5) append `build_iteration_message(exec_result)` to `llm_history`, (6) check `has_final()` → if True yield Final as `type="content"` then `type="done"`; on budget exhaustion yield partial result + `type="done"` with `incomplete=True`; emit ANS events for errors and budget warnings (FR-021)
- [x] T016 [US1] Update `backend/src/api/routes/oracle.py`: replace `from backend.src.bt.wrappers import OracleBTWrapper` with `from backend.src.services.rlm_oracle import RLMOracleWrapper`; replace `OracleBTWrapper(...)` construction with `RLMOracleWrapper(...)`; all other route logic unchanged (FR-019)
- [x] T017 [P] [US1] Write unit tests for `ProjectContext` in `backend/tests/unit/test_project_context.py`: test manifest builds correctly from temp directory; test `file()` returns TextHandle with correct metadata; test `files("**/*.py")` filters by pattern; test `TextHandle.read()` returns content; test `TextHandle.read()` returns notice dict for >1MB file; test `TextHandle.symbols()` returns list for Python file; test `TextHandle.chunks()` returns list of smaller handles; test `grep()` returns matches with line numbers
- [x] T018 [P] [US1] Write unit tests for `RLMOracleWrapper` in `backend/tests/unit/test_rlm_oracle.py`: test `process_query` yields `OracleStreamChunk` types; test loop terminates when `Final` set; test loop terminates at iteration budget (mock LLM to never set Final); test `RecursionDepthExceeded` raised at depth 3; test sub_oracle returns string; test ANS events emitted on budget warning; test partial result returned when budget exhausted

**Checkpoint**: POST `/api/oracle` with a synthesis question returns an answer; existing oracle integration tests pass; REPL loop terminates correctly on `Final`

---

## Phase 4: User Story 2 — Focused Query Efficiency (P2)

**Goal**: Simple, targeted questions complete in ≤3 REPL iterations without scanning unrelated files.

**Independent Test**: POST `/api/oracle` with `"question": "What does vlt_code_lookup return when no index exists?"` completes in <20s and ≤3 iterations logged.

- [x] T019 [US2] Implement task-specific guidance routing in `RLMPromptBuilder.build_system_prompt()` in `backend/src/services/rlm_oracle.py`: add 4 guidance sections (Code Search / Symbol Lookup, Architecture Understanding, Bug Analysis / Root Cause, Long Document Analysis >50KB); add explicit instruction "If the answer can be found with 1 project.search() call, do so directly — never call sub_oracle for single-file lookups"; keep total prompt under 4,000 tokens (SC-002)
- [x] T020 [US2] Add `iteration_count` logging to `RLMOracleWrapper.process_query()` in `backend/src/services/rlm_oracle.py`: log session_id, query, iteration_count, final status at session end (DEBUG level); add `metadata.iteration_count` to the `type="done"` OracleStreamChunk payload so callers can observe
- [x] T021 [US2] Add focused-query efficiency unit test in `backend/tests/unit/test_rlm_oracle.py`: mock LLM to return Final on 2nd iteration for a focused single-file query; assert `OracleStreamChunk(type="done").metadata["iteration_count"] <= 3`

**Checkpoint**: Focused queries complete in ≤3 iterations in unit tests; `type="done"` chunk carries iteration metadata

---

## Phase 5: User Story 3 — Project History & Decision Reconstruction (P2)

**Goal**: Oracle can search vlt threads and vault notes to answer "Why did we switch from X?" questions.

**Independent Test**: Submit a question whose answer is in a vlt thread; oracle finds and quotes it. Submit a question referencing vault notes; oracle uses them.

- [ ] T022 [P] [US3] Implement `TextHandle` for vlt threads in `backend/src/services/project_context.py`: `resource_type="thread"`; metadata: `thread_id`, `node_count`, `project_id`; `read()` — calls `vlt thread read <thread_id> --limit 50` via `SqliteVaultService` (backend has access to vlt DB via shared `data/` directory); `search(query)` inside a thread using FTS; `__repr__` returns `"TextHandle(thread:<thread_id>, N nodes)"`
- [ ] T023 [P] [US3] Implement `TextHandle` for vault notes in `backend/src/services/project_context.py`: `resource_type="note"`; metadata: `path`, `size_bytes`, `title`; `read()` — calls Document-MCP backend GET `/api/notes/{path}` using `httpx` with `LOCAL_USER_ID`; if backend unavailable returns `{"notice": "Document-MCP backend not running", "path": path}` (from spec Assumptions)
- [ ] T024 [US3] Add `thread(thread_id)`, `threads(project_id=None)`, `note(path)`, `notes()` methods to `ProjectContext` in `backend/src/services/project_context.py`: `threads()` queries `SqliteVaultService.list_threads(project_id)` returning list of thread TextHandles; `notes()` calls GET `/api/notes` — gracefully returns empty list if backend unreachable; update `build_project_context()` factory to inject thread/note access
- [ ] T025 [US3] Update `RLMPromptBuilder.build_system_prompt()` in `backend/src/services/rlm_oracle.py`: add `project.thread(thread_id)`, `project.threads()`, `project.note(path)`, `project.notes()` to the Environment & Namespace section; add a "Project History / Decision Reconstruction" guidance section (search threads first, then code, then notes)
- [ ] T026 [US3] Write unit tests for thread/note handles in `backend/tests/unit/test_project_context.py`: test `thread()` returns TextHandle; test `thread.read()` returns node content; test `note()` returns TextHandle; test `note.read()` returns graceful notice when backend unreachable; test `threads()` returns list for a project; test `notes()` returns empty list (not error) when backend unreachable

**Checkpoint**: Unit tests pass; `project.threads()` and `project.notes()` accessible from REPL namespace

---

## Phase 6: User Story 4 — Streaming Progress Visibility (P3)

**Goal**: Users see meaningful REPL progress events before the final answer streams.

**Independent Test**: Submit a query that requires ≥5 REPL iterations; observe `type="progress"` SSE events appearing before `type="content"` events in the SSE stream.

- [ ] T027 [US4] Verify `QueuedStringIO.write()` immediately yields `OracleStreamChunk(type="progress", content=chunk)` in `RLMOracleWrapper.process_query()` in `backend/src/services/rlm_oracle.py`: each chunk from the asyncio Queue should be yielded as a progress event before the next iteration begins; confirm the SSE generator does not buffer progress events (FR-022)
- [ ] T028 [US4] Implement terminal sub-oracle streaming in `backend/src/services/rlm_oracle.py`: when `Final` is set, stream the Final string value token-by-token as `type="content"` events using `RLMOracleWrapper._stream_final(final_value)` generator; use `asyncio.Queue` pattern — the last sub-oracle call's output streams directly to SSE channel before full string assembled (FR-023); yield `type="done"` after all content streamed
- [ ] T029 [US4] Add streaming integration test in `backend/tests/unit/test_rlm_oracle.py`: mock LLM to take 5 iterations before setting Final; collect all `OracleStreamChunk` events; assert at least one `type="progress"` event appears before first `type="content"` event; assert exactly one `type="done"` event at end (SC-007)

**Checkpoint**: SSE stream shows `progress → content → done` ordering; no buffering of progress events

---

## Phase 7: BT Deletion & Validation

**Purpose**: Remove all BT artifacts after new implementation is verified. Do NOT start this phase until Phases 3–6 are complete and all tests pass.

- [ ] T030 Run full test suite `uv run pytest backend/tests/ -v` and confirm all tests pass before deletion
- [ ] T031 Delete `backend/src/bt/` directory entirely (SC-008: "no BT imports remaining in any non-test file")
- [ ] T032 [P] Delete `backend/src/models/signals.py`
- [ ] T033 [P] Delete `backend/src/services/signal_parser.py`
- [ ] T034 [P] Delete `backend/src/services/query_classifier.py`
- [ ] T035 [P] Delete `backend/src/services/prompt_composer.py`
- [ ] T036 Verify no BT imports remain: `grep -r "from.*bt\." backend/src/ --include="*.py" | grep -v test` must return empty; `grep -r "import.*bt\." backend/src/ --include="*.py" | grep -v test` must return empty (SC-008)
- [ ] T037 Run full test suite again: `uv run pytest backend/tests/ -v` — all must pass with BT deleted (SC-006)

**Checkpoint**: `bt/` directory gone; grep confirms zero BT imports; all tests green

---

## Phase 8: Polish & Cross-Cutting Concerns

- [ ] T038 [P] Add `end_line` field to `Symbol` dataclass in `packages/vlt-cli/src/vlt/core/coderag/repomap.py` using `node.end_point[0] + 1` from tree-sitter (enables accurate `TextHandle.chunks()` splitting)
- [ ] T039 [P] Implement `_extract_go_symbols()` in `packages/vlt-cli/src/vlt/core/coderag/repomap.py` following the Python extraction pattern: handle `function_declaration`, `method_declaration`, `type_declaration`; return `List[Symbol]` with name, qualified_name, lineno, end_line, signature
- [ ] T040 Run quickstart.md validation: execute each scenario in `specs/022-rlm-oracle/quickstart.md` and verify expected behavior (synthesis query, focused query, SSE stream ordering, BT grep check)
- [ ] T041 [P] Update `CLAUDE.md` `## Active Technologies` section: replace BT Oracle entry with RLM Oracle entry; update `## Recent Changes` with 022-rlm-oracle summary
- [ ] T042 [P] Update `backend/CLAUDE.md` (if it exists) or add RLM architecture note to `CLAUDE.md` `## BT Oracle` section: replace with RLM Oracle description, key files, environment variables

**Checkpoint**: All quickstart scenarios pass; CLAUDE.md reflects new architecture; Go symbol extraction works

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Requires T001 (RestrictedPython installed) and T004–T006 skeletons
- **US1 (Phase 3)**: Requires T007–T008 (REPLExecutor) + T009–T010 (FileManifest + ProjectContext skeleton) + T010 (RLMSession + RLMPromptBuilder) — all Phase 2
- **US2 (Phase 4)**: Requires Phase 3 complete (RLMPromptBuilder and process_query must exist)
- **US3 (Phase 5)**: Requires Phase 3 T013 (ProjectContext exists to add methods to)
- **US4 (Phase 6)**: Requires Phase 3 T015 (process_query exists to modify streaming in)
- **BT Deletion (Phase 7)**: Requires ALL of Phases 3–6 complete and tests passing
- **Polish (Phase 8)**: Requires Phase 7 complete

### User Story Dependencies

- **US1 (P1)**: Depends only on Foundational — no other story dependencies
- **US2 (P2)**: Depends on US1 (modifies RLMPromptBuilder and process_query created in US1)
- **US3 (P2)**: Depends on Phase 2 (ProjectContext skeleton) — can run in parallel with US2
- **US4 (P3)**: Depends on US1 (modifies streaming in process_query) — can run after US1, parallel with US2/US3

### Within Each User Story

- Foundational data structures (dataclasses) before services
- Services before route integration
- Route integration before end-to-end tests
- Unit tests can be written in parallel with implementation (TDD or post-write)

### Critical Path

```
T001 → T007 → T008 → T015 → T016 → T030 → T031 → T040
       ↑
T009 ──┤
T010 ──┘
```

---

## Parallel Execution Examples

### Phase 2: Foundational (parallel opportunities)

```bash
# These three can run simultaneously (different files):
Task: "T009 FileManifest + FileEntry in project_context.py"
Task: "T010 RLMSession + RLMPromptBuilder in rlm_oracle.py"
Task: "T011 REPLExecutor unit tests in test_repl_executor.py"
```

### Phase 3: US1 (parallel opportunities)

```bash
# TextHandle and ProjectContext are in same file but different class sections:
Task: "T012 TextHandle for files"
Task: "T013 ProjectContext file operations"  # depends on T012 for handle type

# Tests can be written while implementation is in review:
Task: "T017 ProjectContext unit tests"
Task: "T018 RLMOracleWrapper unit tests"
```

### Phase 5: US3 (parallel opportunities)

```bash
# Thread and note handles are independent:
Task: "T022 TextHandle for vlt threads"
Task: "T023 TextHandle for vault notes"
```

### Phase 7: BT Deletion (parallel opportunities)

```bash
# After T031 (bt/ deleted), these are independent:
Task: "T032 Delete signals.py"
Task: "T033 Delete signal_parser.py"
Task: "T034 Delete query_classifier.py"
Task: "T035 Delete prompt_composer.py"
```

---

## Implementation Strategy

### MVP First (US1 Only — Phases 1–3 + Phase 7)

1. Complete Phase 1: Setup & Migration
2. Complete Phase 2: Foundational
3. Complete Phase 3: US1 (cross-codebase synthesis)
4. **STOP and VALIDATE**: Test synthesis question, focused question, existing oracle API contract
5. Complete Phase 7: BT Deletion (once US1 is stable)
6. **DEMO**: Oracle works end-to-end with RLM harness, BT fully removed

### Incremental Delivery

1. Phases 1–2 → Foundation ready (REPL executor works, session management works)
2. Phase 3 → Oracle answers synthesis questions (MVP!)
3. Phase 4 → Focused queries are efficient (≤3 iterations)
4. Phase 5 → Oracle searches threads and notes
5. Phase 6 → Progress streaming visible in UI
6. Phase 7 → BT fully removed, codebase cleaned
7. Phase 8 → Polish (Go support, quickstart validation)

---

## Task Summary

| Phase | Tasks | Parallel Tasks | User Story |
|-------|-------|---------------|------------|
| Phase 1: Setup | T001–T006 | T003–T006 (4P) | — |
| Phase 2: Foundational | T007–T011 | T009–T011 (3P) | — |
| Phase 3: US1 (P1) 🎯 | T012–T018 | T012, T017, T018 (3P) | US1 |
| Phase 4: US2 (P2) | T019–T021 | — | US2 |
| Phase 5: US3 (P2) | T022–T026 | T022–T023 (2P) | US3 |
| Phase 6: US4 (P3) | T027–T029 | — | US4 |
| Phase 7: BT Deletion | T030–T037 | T032–T035 (4P) | — |
| Phase 8: Polish | T038–T042 | T038–T039, T041–T042 (4P) | — |

**Total tasks**: 42
**Parallelizable**: 21 ([P] marked)
**MVP scope**: Phases 1–3 + Phase 7 = T001–T018, T030–T037 (25 tasks)

---

## Notes

- [P] tasks = different files or independent sections, no blocking dependencies
- [US1]–[US4] labels map to user stories in spec.md
- Every task includes exact file path
- BT deletion (Phase 7) must NOT happen until Phases 3–6 are complete and tests pass
- `TextHandle.symbols()` in T012 requires vlt-cli's `repomap.py` — either add vlt-cli as backend dev dependency or extract the symbol extraction function; check `backend/pyproject.toml` for existing vlt-cli reference first
- `SubOracleCallable` (T014) runs synchronously from a thread pool — the child RLM session blocks its thread; this is intentional (it's already off the event loop via `run_in_executor`)
- Commit after each phase or logical task group; avoid committing while tests fail
