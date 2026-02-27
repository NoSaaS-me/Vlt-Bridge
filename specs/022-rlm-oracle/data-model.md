# Data Model: RLM Oracle

**Phase 1 Output** | **Branch**: `022-rlm-oracle` | **Date**: 2026-02-22

All entities are runtime-only. Nothing new is persisted to SQLite — the oracle already uses the existing `context_nodes` table (via `OracleBridge`) for conversation history.

---

## Entity: RLMSession

**Purpose**: Encapsulates all state for a single oracle query execution.
**Lifetime**: Created per query, destroyed when `Final` is set or budget exhausted. Not persisted.

```python
@dataclass
class RLMSession:
    session_id: str             # UUID, for logging and ANS events
    user_id: str
    project_id: str | None
    query: str                  # Original user question
    context_id: str | None      # Conversation thread ID (from OracleBridge)
    recursion_depth: int        # 0 = root, 1 = sub-oracle, cap at 2
    iteration_count: int        # Current REPL loop iteration (max 25 root / 8 sub)
    max_iterations: int         # Budget: 25 for root, 8 for sub-oracle
    llm_history: list[dict]     # [{role: "user"|"assistant", content: str}, ...]
    repl_namespace: REPLNamespace
    project_context: ProjectContext
    started_at: datetime
    status: Literal["running", "completed", "exhausted", "error"]
    final_value: str | None     # Set when LLM assigns Final variable
    partial_result: str | None  # Best partial result when budget exhausted
```

**State transitions**:
```
created → running → completed  (Final set)
                  → exhausted  (iteration budget reached)
                  → error      (unrecoverable exception)
```

**Validation rules**:
- `recursion_depth` must be ≤ 2 (root 0, sub 1, sub-sub 2 max)
- `max_iterations` is 25 at depth 0, 8 at depth ≥ 1
- `session_id` must be UUID4

---

## Entity: ProjectContext

**Purpose**: Root namespace object in the REPL. Wraps a vlt project's source files, threads, and vault notes without loading content until explicitly requested.
**Lifetime**: Created once per `RLMSession`. Stateless between REPL iterations (REPL variables hold results).

```python
class ProjectContext:
    project_id: str
    project_path: Path | None       # From vlt project record; None if not indexed
    manifest: FileManifest          # Preloaded metadata (no file I/O)

    # Manifest-only operations (no file I/O)
    def get_manifest(self) -> FileManifest: ...
    def file_count(self) -> int: ...
    def total_size_bytes(self) -> int: ...

    # Search operations (return match locations, no full content)
    def search(self, query: str, limit: int = 20) -> list[SearchMatch]: ...
    def grep(self, pattern: str, flags: int = 0) -> list[GrepMatch]: ...

    # Handle factory — lazy content loading
    def file(self, path: str) -> TextHandle: ...
    def files(self, pattern: str = "**/*") -> list[TextHandle]: ...

    # Thread access
    def thread(self, thread_id: str) -> TextHandle: ...
    def threads(self, project_id: str | None = None) -> list[TextHandle]: ...

    # Vault note access (requires Document-MCP backend)
    def note(self, path: str) -> TextHandle: ...
    def notes(self) -> list[TextHandle]: ...
```

**FileManifest** (precomputed, constant memory):
```python
@dataclass
class FileManifest:
    files: list[FileEntry]

@dataclass
class FileEntry:
    path: str               # Relative path from project root
    size_bytes: int
    language: str | None    # Detected from extension
    last_modified: datetime
    is_binary: bool
```

**Constraints**:
- Manifest built once at session start; never reloaded
- Files > 1MB: `file.read()` returns `{"notice": "file too large", "size_bytes": N}` instead of content
- Binary files: `file.read()` returns `{"notice": "binary file", "size_bytes": N}`
- If `project_path` is None: `files()` returns empty; threads/notes still available

---

## Entity: TextHandle

**Purpose**: Lazy reference to a text resource. Metadata always accessible; content loaded on demand. Supports slicing, symbol extraction, grep, and semantic chunking.
**Lifetime**: Created by `ProjectContext`; immutable reference.

```python
class TextHandle:
    # Always-available metadata (no I/O)
    path: str
    size_bytes: int
    language: str | None
    resource_type: Literal["file", "thread", "note", "chunk"]

    # Lazy content loading
    def read(self, start_line: int | None = None, end_line: int | None = None) -> str: ...
    # Returns content string, or {"notice": ..., "size_bytes": N} if too large/binary

    # Symbol extraction (tree-sitter, no full read required in metadata mode)
    def symbols(self) -> list[SymbolInfo]: ...
    # Returns [{name, kind, line_number, end_line, signature, qualified_name, docstring}]

    # Grep within this resource
    def grep(self, pattern: str) -> list[GrepMatch]: ...

    # Chunking — returns list of handles, each covering a semantic segment
    def chunks(self, max_lines: int = 200) -> list[TextHandle]: ...
    # Splits by function/class boundaries (tree-sitter) or by line count

    # String representation for quick LLM inspection
    def __repr__(self) -> str: ...
    # e.g., "TextHandle(backend/src/services/oracle.py, 312 lines, python)"
```

**SymbolInfo** (from Phase 0 research — reuses `Symbol` from `repomap.py`):
```python
@dataclass
class SymbolInfo:
    name: str
    kind: str              # "class" | "function" | "method"
    line_number: int       # 1-indexed
    end_line: int          # From tree-sitter node.end_point
    signature: str         # Full signature with type hints
    qualified_name: str    # e.g., "OracleService.process_query"
    docstring: str | None
    parent_class: str | None
```

**GrepMatch**:
```python
@dataclass
class GrepMatch:
    line_number: int
    line_content: str
    context_before: list[str]   # Up to 2 lines
    context_after: list[str]    # Up to 2 lines
```

**Validation rules**:
- `read()` with invalid line range → clamp silently (no exception)
- `chunks()` on binary or too-large file → returns `[self]` (single chunk, content notice on read)
- `symbols()` on non-parseable language → falls back to ctags, then regex, then `[]`

---

## Entity: REPLNamespace

**Purpose**: The restricted Python execution environment for one oracle session. Persists variable state across all iterations of one query.
**Lifetime**: Same as `RLMSession`. Cleared when session ends.

```python
class REPLNamespace:
    # Public REPL bindings (visible to LLM)
    project: ProjectContext         # The project wrapper
    sub_oracle: Callable            # Recursive LLM call function
    Final: Any                      # Sentinel — set by LLM to terminate loop

    # Approved stdlib modules (injected as names)
    # re, json, collections, itertools, math, datetime

    # Internal
    _variables: dict                # Accumulated LLM-written variables
    _iteration_outputs: list[str]   # Metadata summaries per iteration (not full stdout)
    _stdout_capture: QueuedStringIO # Captures print() during exec

    def execute(self, code: str) -> ExecutionResult: ...
    # Runs code in restricted namespace, captures stdout, checks for Final

    def has_final(self) -> bool: ...
    # Returns True if LLM has set the Final variable

    def get_final(self) -> Any: ...
    # Returns the Final value (str coercion if not already str)
```

**ExecutionResult**:
```python
@dataclass
class ExecutionResult:
    success: bool
    stdout_full: str        # Full captured stdout (for SSE streaming)
    stdout_preview: str     # First 200 chars of stdout (for LLM history)
    stdout_total_chars: int # Total stdout length (for LLM history)
    error: str | None       # Exception traceback if failed
    has_final: bool         # Whether Final was set in this iteration
    duration_ms: float
```

**Validation rules**:
- Code execution uses `compile_restricted()` (RestrictedPython); `SyntaxError` → `ExecutionResult(success=False, error=...)`
- Execution timeout: 30 seconds per step (threading.Thread + join(timeout=30))
- `__builtins__` set to dict (not module) to prevent import escape
- `_getattr_` set to `safer_getattr` to block `__subclasses__()` traversal

---

## Entity: SubOracleCall

**Purpose**: A fresh, bounded LLM invocation made programmatically from within REPL code.
**Lifetime**: Created on `sub_oracle(prompt)` call; returns string; destroyed.

```python
# Callable injected into REPL namespace as `sub_oracle`
class SubOracleCallable:
    parent_session: RLMSession
    max_depth: int = 2              # From spec assumptions

    def __call__(self, prompt: str) -> str:
        # Validates recursion depth
        # Creates child RLMSession(recursion_depth=parent+1, max_iterations=8)
        # Runs full RLM loop synchronously (called from thread pool)
        # Returns child session's Final value as string
        # Raises RecursionDepthExceeded if depth > max_depth
```

**SubOracleResult** (internal):
```python
@dataclass
class SubOracleResult:
    prompt: str
    result: str         # Final value from child session
    iterations_used: int
    duration_ms: float
    depth: int          # Recursion depth (1 or 2)
```

**Validation rules**:
- `recursion_depth` > 2 → raise `RecursionDepthExceeded` (LLM sees ImportError-style message)
- `sub_oracle` call count tracked per root session; after 3 calls per iteration → emit ANS budget warning
- Sub-oracle runs with `max_iterations=8` regardless of parent's remaining budget

---

## Deleted Entities (BT Oracle — FR-018)

The following entities from the BT oracle are **fully removed**:

| Entity | File | Replacement |
|--------|------|-------------|
| `OracleBTWrapper` | `bt/wrappers/oracle_wrapper.py` | `RLMOracleWrapper` in `services/rlm_oracle.py` |
| `SignalType`, `Signal` | `models/signals.py` | N/A (no signals in RLM) |
| `SignalParser` | `services/signal_parser.py` | N/A |
| `QueryClassifier` | `services/query_classifier.py` | N/A (LLM decides routing via prompt) |
| `PromptComposer` | `services/prompt_composer.py` | Replaced by `RLMPromptBuilder` (simpler) |
| `BTNode`, `BTCondition`, etc. | `bt/` tree | N/A |

**Preserved entities** (no change):
- `OracleStreamChunk` (`models/oracle.py`) — unchanged SSE wire format
- `OracleBridge` (`services/oracle_bridge.py`) — conversation history (get/clear)
- `EventBus`, ANS subscribers (`services/ans/`) — adapted for RLM events
- `OpenRouterClient` (`bt/services/openrouter_client.py`) → moved to `services/openrouter_client.py`
