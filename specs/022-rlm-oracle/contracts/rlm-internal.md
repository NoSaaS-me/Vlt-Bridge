# RLM Internal Interfaces

**Internal Python contracts** — these are not HTTP APIs, they are Python class interfaces
that callers inside the backend depend on.

---

## RLMOracleWrapper

Replaces `OracleBTWrapper` as the oracle entry point. Same call contract.

```python
class RLMOracleWrapper:
    """Entry point for the RLM oracle. Drop-in replacement for OracleBTWrapper."""

    def __init__(
        self,
        user_id: str,
        api_key: str,
        project_id: str | None = None,
        model: str = "deepseek/deepseek-chat-v3",
        max_tokens: int = 4096,
    ): ...

    async def process_query(
        self,
        query: str,
        context_id: str | None = None,
    ) -> AsyncGenerator[OracleStreamChunk, None]:
        """
        Run RLM session. Yields OracleStreamChunk events:
        - type="progress": REPL stdout as produced (FR-022)
        - type="content": Final value tokens (FR-023)
        - type="done": Session metadata
        - type="system": ANS notifications (FR-021)
        - type="error": Error details
        """
        ...

    async def cancel(self) -> None:
        """Cancel the active session (best-effort)."""
        ...
```

**Callers**: `backend/src/api/routes/oracle.py` (no change to calling code)

---

## REPLExecutor

Handles restricted code execution with streaming stdout.

```python
class REPLExecutor:
    """Executes LLM-generated Python in a restricted namespace with streaming stdout."""

    def __init__(
        self,
        namespace: REPLNamespace,
        timeout_s: float = 30.0,
    ): ...

    async def execute(self, code: str) -> AsyncGenerator[ExecutionEvent, None]:
        """
        Execute code, yielding events:
        - ExecutionEvent(type="stdout", content=str)  — as printed
        - ExecutionEvent(type="result", result=ExecutionResult)  — final
        """
        ...
```

---

## ProjectContext Factory

```python
def build_project_context(
    project_id: str | None,
    user_id: str,
) -> ProjectContext:
    """
    Build ProjectContext for an oracle session.
    - Loads FileManifest from project's filesystem path (if available)
    - Falls back to threads+notes only if project_path is None
    - Notes are skipped (with structured notice) if Document-MCP backend unavailable
    """
    ...
```

---

## RLMPromptBuilder

```python
class RLMPromptBuilder:
    """Builds the root system prompt for the RLM loop."""

    def build_system_prompt(self, project_context: ProjectContext) -> str:
        """
        Returns a constant-size system prompt (SC-002: <4000 tokens).
        Includes:
        1. Environment & Namespace declaration
        2. Execution Protocol (metadata-only history, recursion limit)
        3. Anti-patterns
        4. Response Format (Final sentinel)
        5. Task-specific guidance segments
        """
        ...

    def build_iteration_message(
        self,
        exec_result: ExecutionResult,
    ) -> str:
        """
        Builds the user-turn message appended after each REPL iteration.
        Contains ONLY metadata: stdout_preview (≤200 chars), stdout_total_chars,
        error (if any), has_final.
        Never includes full stdout content (FR-003).
        """
        ...
```

---

## ANS Integration Points

The RLM loop emits these events to the existing EventBus (FR-021):

```python
# Iteration budget warning (≥70% consumed)
bus.emit(Event(
    type="budget.iteration.warning",
    source="rlm_oracle",
    severity=Severity.WARNING,
    payload={
        "session_id": session.session_id,
        "iterations_used": session.iteration_count,
        "iterations_max": session.max_iterations,
        "percent": session.iteration_count / session.max_iterations,
    }
))

# REPL execution error
bus.emit(Event(
    type="tool.call.failure",
    source="rlm_oracle",
    severity=Severity.WARNING,
    payload={
        "session_id": session.session_id,
        "error": exec_result.error,
        "iteration": session.iteration_count,
    }
))

# Budget exhausted (no Final set within limit)
bus.emit(Event(
    type="budget.iteration.exceeded",
    source="rlm_oracle",
    severity=Severity.CRITICAL,
    payload={"session_id": session.session_id}
))
```

---

## MCP Tool Interface (Unchanged)

`vlt_oracle_query` in `packages/vlt-cli/src/vlt/mcp/oracle_tools.py` — no changes (FR-020).

```python
def vlt_oracle_query(
    query: str,
    project_id: str | None = None,
    context_id: str | None = None,
) -> dict:
    """
    Returns {status, answer, context_id, sources_used}
    Calls /api/oracle (non-streaming) internally.
    """
    ...
```
