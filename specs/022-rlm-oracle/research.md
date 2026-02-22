# Research: RLM Oracle Implementation

**Phase 0 Output** | **Branch**: `022-rlm-oracle` | **Date**: 2026-02-22

Four parallel research agents investigated the critical unknowns. Findings consolidated below.

---

## Topic 1: REPL Sandbox — Restricted Python Execution

**Decision**: RestrictedPython + threading timeout + custom builtins dict
**Rationale**: AST-based compile-time restrictions block dangerous operations before execution. Threading timeout (not `signal.alarm`) works in the multi-threaded FastAPI environment. The "trusted single user" trust model from the spec means we need protection against LLM footguns, not adversarial injection.
**Alternatives considered**:
- *Process/container isolation (gVisor, seccomp)*: Maximum security but out of scope — adds infrastructure complexity, unnecessary for single-user trust model.
- *Signal-based timeout*: Only works in main thread; incompatible with FastAPI's threaded executor.
- *Custom `__builtins__` dict alone*: Simpler but allows `__subclasses__()` traversal escape; RestrictedPython's `_getattr_` guard closes this.

**Implementation approach**:
```python
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import safe_builtins, safer_getattr, safe_iter

ALLOWED_MODULES = {
    're': re, 'json': json, 'math': math,
    'datetime': datetime, 'collections': collections, 'itertools': itertools
}

namespace = {
    '__builtins__': {**safe_builtins, '__import__': None, 'open': None},
    '_getattr_': safer_getattr,    # Blocks __subclasses__ traversal
    '_getiter_': safe_iter,
    **ALLOWED_MODULES,
}
byte_code = compile_restricted(source, '<repl>', 'exec')
thread = threading.Thread(target=exec, args=(byte_code, namespace), daemon=True)
thread.start(); thread.join(timeout=30)
```

**Key escape vectors defended**:
| Vector | Defense |
|---|---|
| `__subclasses__()` traversal | RestrictedPython `_getattr_` → `safer_getattr` |
| `__builtins__.__dict__['__import__']` | Set `__builtins__` to dict, not module |
| Runaway loops | threading timeout (30s, configurable) |
| Direct file I/O | `open: None` in builtins |

---

## Topic 2: Async SSE Streaming from Sync REPL

**Decision**: `asyncio.run_in_executor()` + `asyncio.Queue` + custom `QueuedStringIO`
**Rationale**: The sync REPL `exec()` runs in a `ThreadPoolExecutor` thread. A custom `StringIO` subclass pushes chunks via `asyncio.run_coroutine_threadsafe()` into an `asyncio.Queue` that the async SSE generator drains. This matches the existing `EventSourceResponse` pattern in `backend/src/api/routes/oracle.py` exactly.
**Alternatives considered**:
- *`anyio.to_thread.run_sync()`*: Equivalent ergonomics, but adds dependency and doesn't improve over asyncio for this project's existing pattern.
- *Subprocess with pipe*: Better isolation but high overhead, complex cleanup, unnecessary given RestrictedPython.
- *Lua sandbox (existing `bt/lua/sandbox.py`)*: Secure, but LLM-generated Lua code is not the RLM paradigm; Python is required for the `ProjectContext` API.

**Implementation pattern**:
```python
class QueuedStringIO(io.StringIO):
    def write(self, s: str) -> int:
        asyncio.run_coroutine_threadsafe(self._queue.put(s), self._loop).result(timeout=0.1)
        return len(s)

async def stream_exec(code: str) -> AsyncGenerator[str, None]:
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    io_bridge = QueuedStringIO(queue, loop)
    task = loop.run_in_executor(None, _sync_exec, code, io_bridge)
    while True:
        chunk = await asyncio.wait_for(queue.get(), timeout=30)
        if chunk is SENTINEL: break
        yield chunk
    await task  # Propagate exceptions
```

**Performance**: ~2-150ms end-to-end latency (98% network). Chunk size of 256 bytes balances overhead vs responsiveness.

---

## Topic 3: RLM System Prompt Design

**Decision**: Code-only paradigm with `Final` sentinel, metadata-only history contract, explicit recursion guardrail for DeepSeek
**Rationale**: The arxiv 2512.24601 paper validated this design on GPT-4o and Qwen3-Coder without fine-tuning. Tekta.ai production analysis revealed Qwen-family models (including DeepSeek) require explicit `sub_oracle` call limits to prevent runaway recursion costs. The 5-part prompt structure below maps directly to the spec's FRs.
**Alternatives considered**:
- *Natural language instructions instead of code-only*: Contradicts RLM paper's core insight — LLM must write code to achieve O(|P|) work with O(1) context.
- *No explicit recursion limit in prompt*: Paper found models self-limit on GPT-5 but NOT on Qwen/DeepSeek family; guardrail required.
- *JSON output format instead of Final variable*: Final variable is simpler and already used by paper's reference implementation.

**Prompt structure** (5 parts):
1. **Environment & Namespace**: Declare `project` (ProjectContext), `sub_oracle(prompt)`, approved stdlib modules, `Final` sentinel
2. **Execution Protocol**: Metadata-only history (LLM can't see prior print output), code-over-prose, recursion discipline (max 3 `sub_oracle` calls per root session)
3. **Anti-patterns**: Infinite recursion, printing large results, hallucinating file paths, forgetting `Final`
4. **Response Format**: `Final = <answer>` terminates loop; valid types: str, dict, list
5. **Task-specific guidance**: Routing sections for code search, architecture, bug analysis, long document synthesis

**Critical DeepSeek note**: Add explicit "CONSTRAINT: max 3 sub_oracle calls per turn" with error consequence stated. Without this, DeepSeek/Qwen models generate thousands of recursive calls (confirmed in production literature).

---

## Topic 4: Symbol Extraction for TextHandle.symbols()

**Decision**: Reuse `repomap.py:extract_symbols_from_ast()` (already in codebase); add Go support; add `end_line` from tree-sitter `node.end_point`
**Rationale**: The codebase already has complete tree-sitter symbol extraction for Python, TypeScript, and JavaScript in `packages/vlt-cli/src/vlt/core/coderag/repomap.py` (lines 57–224). Reinventing this would be waste. Python's `ast` module is Python-only; tree-sitter covers 23+ languages with existing `tree-sitter-language-pack` dependency.
**Alternatives considered**:
- *Python `ast` module*: Python-only, missing end-line on Python <3.10, not suitable for multi-language TextHandle.
- *ctags (Universal Ctags binary)*: Already a fallback in `ctags.py`; requires external binary, no end-line support, but fine as Tier 2 fallback.
- *Regex fallback*: Already planned as Tier 3; needed for unsupported languages only.

**Existing code to reuse**:
- `parser.py`: `parse_file(content, language)` → tree-sitter `Tree`
- `repomap.py`: `extract_symbols_from_ast(tree, source, path, language)` → `List[Symbol]`
- `Symbol` dataclass: `{name, qualified_name, file_path, symbol_type, signature, lineno, docstring}`

**Gaps to fill**:
1. Go extraction: `_extract_go_symbols()` function (tree-sitter grammar available, extraction not implemented)
2. `end_line` field: Add `node.end_point[0] + 1` to `Symbol` and extraction functions
3. Regex fallback for languages with no tree-sitter extraction implemented

**Performance**: ~0.5–2ms/file for ≤200KB files. 100 files ≈ 50–200ms — fits within <20s oracle target.

---

## Resolved Technical Context

All NEEDS CLARIFICATION items from plan template resolved:

| Item | Resolution |
|------|-----------|
| REPL sandbox library | RestrictedPython 8.x |
| Async/sync bridge | asyncio.Queue + run_in_executor |
| Symbol extraction | repomap.py (reuse existing) |
| System prompt paradigm | Code-only + Final sentinel |
| DeepSeek recursion guard | Explicit "max 3" in prompt text |
| Sub-oracle streaming | Terminal sub-oracle streams directly to SSE channel (see FR-023) |
