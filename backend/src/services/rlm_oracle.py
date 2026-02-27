"""RLM Oracle — Recursive Language Model inference-time harness.

Implements FR-001 (persistent REPL session), FR-002 (constant-size metadata),
FR-003 (metadata-only history), FR-004 (Final sentinel loop), FR-005 (sub_oracle),
FR-006 (programmatic sub-oracle), FR-019 (unchanged API), FR-021 (ANS events),
FR-022 (REPL stdout as SSE progress), FR-023 (Final streams as content).

Components:
    RLMSession         — Per-query session state (ephemeral, not persisted).
    RLMPromptBuilder   — Builds the 5-part system prompt and iteration messages.
    SubOracleCallable  — Callable injected as `sub_oracle` in REPL namespace.
    RLMOracleWrapper   — Drop-in replacement for OracleBTWrapper; drives the loop.

Part of 022-rlm-oracle (RLM Oracle replacement).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

# Default iteration budgets (FR-017)
ROOT_MAX_ITERATIONS = 25
SUB_MAX_ITERATIONS = 8


@dataclass
class RLMSession:
    """Per-query session state. Ephemeral — not persisted to any database.

    Created by RLMOracleWrapper.process_query(), destroyed when Final is set
    or iteration budget exhausted.
    """

    session_id: str                    # UUID4
    user_id: str
    project_id: Optional[str]
    query: str                         # Original user question
    context_id: Optional[str]          # Conversation thread ID (for multi-turn)
    recursion_depth: int               # 0=root, 1=sub-oracle, 2=sub-sub (max)
    max_iterations: int                # Budget: ROOT_MAX_ITERATIONS or SUB_MAX_ITERATIONS
    llm_history: list[dict]            # [{role: "user"|"assistant", content: str}, ...]
    started_at: datetime               # UTC
    status: Literal["running", "completed", "exhausted", "error"] = "running"
    iteration_count: int = 0
    final_value: Optional[str] = None  # Set when LLM assigns Final variable
    partial_result: Optional[str] = None  # Best partial result if budget exhausted
    sub_oracle_call_count: int = 0     # Tracks sub_oracle() calls for ANS warnings

    @classmethod
    def create_root(
        cls,
        user_id: str,
        query: str,
        project_id: Optional[str] = None,
        context_id: Optional[str] = None,
    ) -> "RLMSession":
        """Create a root-level session (depth=0)."""
        return cls(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            project_id=project_id,
            query=query,
            context_id=context_id,
            recursion_depth=0,
            max_iterations=ROOT_MAX_ITERATIONS,
            llm_history=[],
            started_at=datetime.now(timezone.utc),
        )

    @classmethod
    def create_sub(cls, parent: "RLMSession", prompt: str) -> "RLMSession":
        """Create a sub-oracle session (depth = parent.depth + 1)."""
        return cls(
            session_id=str(uuid.uuid4()),
            user_id=parent.user_id,
            project_id=parent.project_id,
            query=prompt,
            context_id=parent.context_id,
            recursion_depth=parent.recursion_depth + 1,
            max_iterations=SUB_MAX_ITERATIONS,
            llm_history=[],
            started_at=datetime.now(timezone.utc),
        )

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_iterations - self.iteration_count)

    @property
    def budget_percent_used(self) -> float:
        return self.iteration_count / self.max_iterations if self.max_iterations else 1.0

    def is_budget_exhausted(self) -> bool:
        return self.iteration_count >= self.max_iterations


# ---------------------------------------------------------------------------
# System prompt template (SC-002: stay under 4,000 tokens)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
## EXECUTION ENVIRONMENT

You are a code-writing AI agent in a Python 3.11 REPL environment. The project you're analyzing lives as variables in this environment — NOT in your context window.

### Project
{project_summary}

### Available Namespace

**`project`** (ProjectContext): Explore the codebase:
- `project.get_manifest()` → FileManifest (iterable of FileEntry; each has `.path`, `.size_bytes`, `.language`)
- `project.file(path)` → TextHandle (lazy; supports fuzzy: `"foo.py"` → `"src/services/foo.py"`)
- `project.files(pattern="**/*")` → list[TextHandle]
- `project.search(query, limit=20)` → list[SearchMatch] (each has `.path`, `.snippet`, `.score`, `.line_number`)
- `project.grep(pattern)` → list[GrepMatch] (each has `.path`, `.line_number`, `.line_content`)
- `project.thread(thread_id)` → TextHandle (vlt reasoning thread)
- `project.threads()` → list[TextHandle]
- `project.note(path)` → TextHandle (vault markdown note)
- `project.notes()` → list[TextHandle]

**TextHandle** supports:
- `.read(start_line=None, end_line=None)` → str (always string; errors return "[error] ..." prefix)
- `.symbols()` → list[SymbolInfo] (each has `.name`, `.kind`, `.line_number`, `.end_line`, `.signature`)
- `.grep(pattern)` → list[GrepMatch]
- `.chunks(max_lines=200)` → list[TextHandle] (semantic chunks)
- `repr(handle)` → "TextHandle(path, N lines, language)"

**`sub_oracle(prompt: str) -> str`**: Recursive LLM call for complex sub-tasks.
- CONSTRAINT: max 3 sub_oracle calls per root session (enforced)
- Each call costs ~1 API round-trip; use only for substantial context (>1KB)
- Returns the child session's Final value as a string

**`Final`**: Set this variable to terminate the loop.
- `Final = "your answer here"` → terminates and returns
- **Always set Final to a markdown-formatted string** — prose, headings, code blocks, bullet lists.
- Do NOT set Final to a raw dict or list; format your findings as readable markdown instead.

**Standard library** (pre-loaded, use directly OR `import`): `re`, `json`, `math`, `datetime`, `collections`, `itertools`, `string`
**NOT available**: `os`, `subprocess`, `open`, `requests`, `threading`, `socket`, `locals`, `globals`, `eval`, `exec`
**Tip**: Instead of `if 'x' in locals()`, initialize variables first: `x = None` then check `if x is not None`

---

## HOW TO BEHAVE

1. **Code over prose**: Always write Python to explore — not English explanations
2. **Metadata-only history**: You CANNOT see stdout from previous iterations
   - Store results in variables: `results = project.search("auth")`
   - DO NOT rely on printed output being available next iteration
3. **Recursion discipline** [CRITICAL]:
   - Ask: "Can I solve this with project.search/file/grep first?"
   - **If the answer can be found with 1 `project.search()` call, do so directly — never call `sub_oracle` for single-file lookups**
   - sub_oracle is expensive — use only for synthesizing >1KB of content across multiple sources
   - If you call sub_oracle more than 3 times, execution force-terminates
4. **Chunking large content**: `chunks = file.chunks(200)` then iterate
5. **Terminate decisively**: Set `Final` when you have sufficient evidence

---

## AVOID THESE PATTERNS

\u274c Calling sub_oracle when 1 project.search() call would answer the question
\u274c Calling sub_oracle for single-file lookups — use project.file(path).read() instead
\u274c Printing large results expecting to see them next iteration
\u274c Hallucinating file paths — always use `project.get_manifest()` or `project.search()` first
\u274c Over-verifying: calling sub_oracle when project.search already found the answer
\u274c Forgetting to set Final — always end with `Final = <answer>`

---

## RESPONSE FORMAT

Your response MUST end with:

    Final = "your markdown-formatted answer"

**Final MUST be a string.** Write your answer as markdown — prose, headings (`##`), bullet lists, code blocks (` ``` `), etc.
Do NOT assign a raw dict or list to Final; format findings as readable text instead.

---

## GUIDANCE BY QUERY TYPE

### Code search / symbol lookup
1. `project.search(query)` or `project.grep(pattern)` to locate
2. `handle.read(start_line, end_line)` to read relevant section
3. `Final = f"Found in **{{path}}** (line {{line}}):\\n\\n```python\\n{{snippet}}\\n```\\n\\n{{explanation}}"`

### Architecture / cross-cutting
1. `project.get_manifest()` to see all files
2. `project.search()` for patterns across codebase
3. sub_oracle on each module for analysis (max 3 calls)
4. `Final = f"## Architecture Summary\\n\\n{{summary}}\\n\\n**Key files:**\\n{{chr(10).join('- ' + f for f in key_files)}}"`

### Bug / root cause
1. Search for error patterns
2. Read relevant files
3. `Final = f"## Root Cause\\n\\n{{root_cause}}\\n\\n**Affected files:** {{', '.join(affected_files)}}"`

### Project history / decision reconstruction (why we did X)
1. **Threads first**: `threads = project.threads()` → scan `repr(t)` to find relevant ones → `t.read()` for each
2. **Then code**: `project.search(query)` for implementation evidence
3. **Then notes**: `notes = project.notes()` → scan for documentation
4. Synthesize: cite specific thread IDs and node sequences
5. `Final = f"## Decision Rationale\\n\\n{{rationale}}\\n\\n**Source:** thread `{{thread_id}}`, node {{seq}}"`

### Large document analysis (>50KB)
1. `chunks = handle.chunks(200)`
2. For each chunk: `sub_oracle("Extract X from this", chunk.read())`  [use max 3 calls]
3. Aggregate and synthesize locally
4. `Final = f"## Analysis\\n\\n{{synthesis}}"`
"""


class RLMPromptBuilder:
    """Builds the RLM system prompt and per-iteration history messages."""

    # Max chars for stdout preview in iteration message (FR-003)
    STDOUT_PREVIEW_MAX = 200

    def build_system_prompt(self, project_summary: str) -> str:
        """Build the complete system prompt for the RLM loop.

        Args:
            project_summary: Short manifest summary (e.g., "142 files, 850KB. Languages: python:45, typescript:23")

        Returns:
            The multi-part system prompt string. Should stay under 4,000 tokens.
        """
        return _SYSTEM_PROMPT_TEMPLATE.format(project_summary=project_summary)

    def build_iteration_message(
        self,
        stdout_full: str,
        stdout_total_chars: int,
        error: Optional[str],
        iteration_number: int,
        max_iterations: int = 0,
    ) -> str:
        """Build the user-turn message appended after each REPL iteration.

        CRITICAL: Contains ONLY metadata. Never includes full stdout content.
        The LLM cannot see what it printed in previous turns.

        Args:
            stdout_full: The captured stdout (used only to extract preview)
            stdout_total_chars: Total chars printed
            error: Exception message if the code failed
            iteration_number: Current iteration number (for context)
            max_iterations: Total iteration budget (for budget warnings)

        Returns:
            A compact metadata message the LLM sees as the "result" of its code.
        """
        preview = stdout_full[: self.STDOUT_PREVIEW_MAX] if stdout_full else ""
        truncated = stdout_total_chars > self.STDOUT_PREVIEW_MAX

        lines = [f"[Iteration {iteration_number} result]"]

        if stdout_full:
            lines.append(f"stdout preview ({stdout_total_chars} total chars):")
            lines.append(f"  {repr(preview)}" + (" [truncated]" if truncated else ""))
        else:
            lines.append("stdout: (empty)")

        if error:
            lines.append(f"error: {error}")

        # Budget warning at 80% usage
        if max_iterations > 0:
            remaining = max_iterations - iteration_number
            pct_used = iteration_number / max_iterations
            if pct_used >= 0.8:
                lines.append(
                    f"⚠ BUDGET WARNING: {remaining} iteration(s) remaining. "
                    "Set `Final = <answer>` NOW with your best answer."
                )

        lines.append("")
        lines.append(
            "Continue: write more Python code, or set `Final = <your_answer>` to terminate."
        )

        return "\n".join(lines)

    def build_initial_user_message(self, query: str, manifest_summary: str) -> str:
        """Build the first user message that starts the REPL loop."""
        return (
            f"Question: {query}\n\n"
            f"Project: {manifest_summary}\n\n"
            "Use `project` to explore the codebase. Set `Final = <answer>` when done."
        )


# ---------------------------------------------------------------------------
# Code extraction helper
# ---------------------------------------------------------------------------

def _extract_code(llm_response: str) -> str:
    """Extract Python code from an LLM response.

    Prefers ```python ... ``` fenced blocks. Falls back to any ``` ... ```
    block. If no fences found, detects unfenced code by looking for Python
    markers (Final, project., import, def, for/if with colon).
    """
    import re as _re

    # ```python ... ```
    match = _re.search(r"```python\s*\n(.*?)```", llm_response, _re.DOTALL)
    if match:
        return match.group(1).strip()

    # ``` ... ``` (bare or other language)
    match = _re.search(r"```\w*\s*\n(.*?)```", llm_response, _re.DOTALL)
    if match:
        return match.group(1).strip()

    # Unfenced code detection: if the response looks like Python code
    # (contains REPL namespace references or common Python patterns),
    # treat the whole response as code. This handles models like Grok
    # that sometimes skip fencing.
    text = llm_response.strip()
    code_markers = (
        "Final =", "Final=",
        "project.", "sub_oracle(",
        "print(", "for ", "if ",
        "import ", "results =", "results=",
    )
    # Must contain at least one REPL-specific marker AND look like code
    has_repl_marker = any(m in text for m in ("Final", "project.", "sub_oracle("))
    has_code_marker = any(m in text for m in code_markers)
    # Heuristic: mostly code if lines contain assignments, calls, or keywords
    lines = text.split("\n")
    non_empty = [ln for ln in lines if ln.strip()]
    code_lines = sum(
        1 for ln in non_empty
        if ("=" in ln and not ln.strip().endswith(":"))    # assignments
        or "(" in ln                                        # function calls
        or ln.strip().startswith(("#", "for ", "if ", "while ", "try:",
                                  "except", "with ", "import ", "from ",
                                  "def ", "class ", "return "))
    )
    looks_like_code = len(non_empty) > 0 and code_lines / len(non_empty) > 0.4

    if has_repl_marker and has_code_marker and looks_like_code:
        # Strip at most one leading prose line (e.g., "Here is the code:")
        # but preserve everything once code begins.
        if lines and not ("=" in lines[0] or "(" in lines[0] or
                          lines[0].strip().startswith(("#", "for ", "if ",
                              "while ", "try:", "import ", "from ",
                              "def ", "class "))):
            return "\n".join(lines[1:]).strip()
        return text

    return ""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RecursionDepthExceeded(Exception):
    """Raised when sub_oracle is called beyond the allowed recursion depth."""


# ---------------------------------------------------------------------------
# SubOracleCallable — T014
# ---------------------------------------------------------------------------

class SubOracleCallable:
    """Callable injected as ``sub_oracle`` in the REPL namespace.

    Called synchronously from the REPL thread (a daemon thread without a
    running event loop), so ``asyncio.run()`` is safe here.
    """

    MAX_SUB_ORACLE_CALLS = 3

    def __init__(
        self,
        parent_session: RLMSession,
        api_key: str,
        model: str,
        project_id: str,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
    ) -> None:
        self._parent = parent_session
        self._api_key = api_key
        self._model = model
        self._project_id = project_id
        self._max_tokens = max_tokens
        self._base_url = base_url

    def __call__(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        """Run a child RLM loop synchronously and return its Final value.

        Extra *args and **kwargs are accepted but ignored — LLMs sometimes
        pass additional arguments (e.g. ``sub_oracle("query", "context")``).
        """
        if args or kwargs:
            logger.debug("sub_oracle ignoring extra args=%r kwargs=%r", args, kwargs)
        if self._parent.recursion_depth >= 2:
            raise RecursionDepthExceeded(
                f"Max recursion depth (2) reached at depth "
                f"{self._parent.recursion_depth}. "
                "Cannot call sub_oracle recursively beyond depth 2."
            )
        if self._parent.sub_oracle_call_count >= self.MAX_SUB_ORACLE_CALLS:
            raise RecursionDepthExceeded(
                f"Max sub_oracle calls ({self.MAX_SUB_ORACLE_CALLS}) exhausted for this session. "
                "Synthesize the answer with what you already have."
            )

        self._parent.sub_oracle_call_count += 1
        logger.debug(
            "sub_oracle call #%d from session %s (depth=%d)",
            self._parent.sub_oracle_call_count,
            self._parent.session_id,
            self._parent.recursion_depth,
        )

        child_session = RLMSession.create_sub(self._parent, prompt)

        import asyncio as _asyncio
        return _asyncio.run(
            _run_rlm_child_loop(
                session=child_session,
                api_key=self._api_key,
                model=self._model,
                project_id=self._project_id,
                max_tokens=self._max_tokens,
                base_url=self._base_url,
            )
        )


# ---------------------------------------------------------------------------
# Core loop for child (sub-oracle) sessions — no streaming
# ---------------------------------------------------------------------------

async def _run_rlm_child_loop(
    session: RLMSession,
    api_key: str,
    model: str,
    project_id: str,
    max_tokens: int = 4096,
    base_url: Optional[str] = None,
) -> str:
    """Execute the RLM loop for a sub-oracle child session.

    Runs to completion without streaming; returns the final_value string.
    """
    # Deferred imports to avoid circular dependencies.
    from .project_context import build_project_context
    from .openrouter_client import OpenRouterClient
    from .repl_executor import REPLExecutor, REPLNamespace

    child_client_kwargs = {"api_key": api_key}
    if base_url:
        child_client_kwargs["base_url"] = base_url
    llm = OpenRouterClient(**child_client_kwargs)
    prompt_builder = RLMPromptBuilder()

    project_context = build_project_context(project_id, session.user_id)
    manifest_summary = project_context.get_manifest().summary()

    system_prompt = prompt_builder.build_system_prompt(manifest_summary)
    initial_msg = prompt_builder.build_initial_user_message(session.query, manifest_summary)

    session.llm_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_msg},
    ]

    sub_oracle_fn = SubOracleCallable(
        parent_session=session,
        api_key=api_key,
        model=model,
        project_id=project_id,
        max_tokens=max_tokens,
    )

    namespace = REPLNamespace()
    namespace.inject(project_context, sub_oracle_fn)
    executor = REPLExecutor(namespace=namespace, timeout_s=30.0)

    while not session.is_budget_exhausted():
        session.iteration_count += 1

        try:
            llm_response = await llm.complete(
                messages=session.llm_history,
                model=model,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error(
                "LLM call failed in child session %s (iter %d): %s",
                session.session_id, session.iteration_count, exc,
            )
            session.status = "error"
            return session.partial_result or f"(LLM error: {exc})"

        llm_content = llm_response.get("content", "")
        session.llm_history.append({"role": "assistant", "content": llm_content})

        code = _extract_code(llm_content)
        if not code:
            if llm_content.strip():
                if session.iteration_count == 1:
                    # Nudge back into REPL mode on first iteration
                    session.llm_history.append({
                        "role": "user",
                        "content": (
                            "You must write Python code to explore the project. "
                            "Use `project.search()`, `project.file()`, etc. "
                            "Set `Final = <answer>` when done."
                        ),
                    })
                    continue
                # Later iterations: treat prose as the answer
                session.status = "completed"
                return llm_content.strip()
            continue

        exec_result = None
        async for r in executor.execute(code):
            exec_result = r

        if exec_result is None:
            continue

        if exec_result.has_final:
            session.status = "completed"
            session.final_value = exec_result.final_value
            return session.final_value or "(empty answer)"

        if exec_result.stdout_full:
            session.partial_result = exec_result.stdout_full[:500]

        iteration_msg = prompt_builder.build_iteration_message(
            stdout_full=exec_result.stdout_full,
            stdout_total_chars=exec_result.stdout_total_chars,
            error=exec_result.error,
            iteration_number=session.iteration_count,
            max_iterations=session.max_iterations,
        )
        session.llm_history.append({"role": "user", "content": iteration_msg})

    session.status = "exhausted"
    return session.partial_result or "(budget exhausted without answer)"


# ---------------------------------------------------------------------------
# RLMOracleWrapper — T015 (drop-in replacement for OracleBTWrapper)
# ---------------------------------------------------------------------------

class RLMOracleWrapper:
    """Inference-time harness replacing OracleBTWrapper (FR-019).

    LLM writes Python code in a REPL loop to explore the project through
    the ``project`` object.  When the LLM assigns ``Final = <answer>``,
    the loop terminates and the answer streams out.
    """

    def __init__(
        self,
        user_id: str,
        api_key: str,
        project_id: str,
        model: str = "deepseek/deepseek-chat-v3-0324",
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
    ) -> None:
        self._user_id = user_id
        self._api_key = api_key
        self._project_id = project_id
        self._model = model
        self._max_tokens = max_tokens
        self._base_url = base_url  # None = use OpenRouterClient default (OpenRouter)
        self._cancelled = False

    def cancel(self) -> None:
        """Signal cancellation. The loop checks this flag between iterations."""
        self._cancelled = True

    async def _stream_final(self, final_value: str):
        """Stream the Final answer in chunks for progressive SSE delivery (FR-023).

        Yields sub-strings of *final_value* in ~50-character pieces so the
        frontend can start rendering before the full answer has been emitted.
        """
        if not final_value:
            yield ""
            return
        CHUNK_SIZE = 50
        for i in range(0, len(final_value), CHUNK_SIZE):
            yield final_value[i:i + CHUNK_SIZE]

    async def process_query(
        self,
        query: str,
        context_id: Optional[str] = None,
    ):
        """Run the RLM REPL loop and stream OracleStreamChunk events.

        Yields:
            OracleStreamChunk(type="progress") — REPL stdout per iteration
            OracleStreamChunk(type="content")  — Final answer text
            OracleStreamChunk(type="done")     — Termination marker
            OracleStreamChunk(type="error")    — On fatal failure
        """
        from .project_context import build_project_context
        from .openrouter_client import OpenRouterClient
        from .repl_executor import REPLExecutor, REPLNamespace
        from ..models.oracle import OracleStreamChunk

        session = RLMSession.create_root(
            user_id=self._user_id,
            query=query,
            project_id=self._project_id,
            context_id=context_id,
        )

        logger.info(
            "RLM session %s started: user=%s project=%s query=%r",
            session.session_id, self._user_id, self._project_id, query[:80],
        )

        try:
            project_context = build_project_context(self._project_id, self._user_id)
        except Exception as exc:
            logger.error("Failed to build ProjectContext: %s", exc)
            yield OracleStreamChunk(type="error", error=f"ProjectContext error: {exc}")
            return

        manifest_summary = project_context.get_manifest().summary()
        prompt_builder = RLMPromptBuilder()
        system_prompt = prompt_builder.build_system_prompt(manifest_summary)
        initial_msg = prompt_builder.build_initial_user_message(query, manifest_summary)

        session.llm_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_msg},
        ]

        sub_oracle_fn = SubOracleCallable(
            parent_session=session,
            api_key=self._api_key,
            model=self._model,
            project_id=self._project_id,
            max_tokens=self._max_tokens,
            base_url=self._base_url,
        )

        namespace = REPLNamespace()
        namespace.inject(project_context, sub_oracle_fn)
        executor = REPLExecutor(namespace=namespace, timeout_s=30.0)
        client_kwargs = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        llm = OpenRouterClient(**client_kwargs)

        while not session.is_budget_exhausted():
            if self._cancelled:
                logger.info("RLM session %s cancelled by user", session.session_id)
                yield OracleStreamChunk(
                    type="error",
                    error="Session cancelled.",
                    metadata={"iteration_count": session.iteration_count},
                )
                return

            session.iteration_count += 1
            logger.debug(
                "RLM iter %d/%d (session %s)",
                session.iteration_count, session.max_iterations, session.session_id,
            )

            # ----------------------------------------------------------
            # 1. LLM call
            # ----------------------------------------------------------
            try:
                llm_response = await llm.complete(
                    messages=session.llm_history,
                    model=self._model,
                    max_tokens=self._max_tokens,
                )
            except Exception as exc:
                logger.error(
                    "LLM call failed at iter %d: %s", session.iteration_count, exc
                )
                session.status = "error"
                yield OracleStreamChunk(type="error", error=f"LLM error: {exc}")
                return

            llm_content = llm_response.get("content", "")
            session.llm_history.append({"role": "assistant", "content": llm_content})

            logger.debug(
                "RLM iter %d LLM raw (%d chars): %s",
                session.iteration_count, len(llm_content), llm_content[:500],
            )

            # ----------------------------------------------------------
            # 2. Extract code
            # ----------------------------------------------------------
            code = _extract_code(llm_content)
            logger.debug(
                "RLM iter %d extracted code (%d chars)",
                session.iteration_count, len(code),
            )
            if not code:
                # Model returned prose without a code block.
                if llm_content.strip():
                    if session.iteration_count == 1:
                        # First iteration: LLM should write code, not prose.
                        # Nudge it back into REPL mode.
                        session.llm_history.append({
                            "role": "user",
                            "content": (
                                "You must write Python code to explore the project. "
                                "Use `project.search()`, `project.file()`, etc. "
                                "Set `Final = <answer>` when you have your answer."
                            ),
                        })
                        continue
                    # Later iterations: treat prose as the final answer.
                    session.status = "completed"
                    session.final_value = llm_content.strip()
                    yield OracleStreamChunk(
                        type="content", content=session.final_value
                    )
                    yield OracleStreamChunk(
                        type="done",
                        metadata={"iteration_count": session.iteration_count},
                    )
                    return
                continue

            # ----------------------------------------------------------
            # 3. Execute in REPL
            # ----------------------------------------------------------
            exec_result = None
            async for r in executor.execute(code):
                exec_result = r

            if exec_result is None:
                continue

            logger.debug(
                "RLM iter %d REPL: has_final=%s stdout=%d error=%s",
                session.iteration_count, exec_result.has_final,
                exec_result.stdout_total_chars,
                exec_result.error[:100] if exec_result.error else None,
            )

            # ----------------------------------------------------------
            # 4. Stream REPL stdout as progress event (FR-022)
            # ----------------------------------------------------------
            if exec_result.stdout_full:
                yield OracleStreamChunk(
                    type="progress",
                    content=exec_result.stdout_full,
                    metadata={"iteration": session.iteration_count},
                )

            # ----------------------------------------------------------
            # 5. Check for Final (FR-004)
            # ----------------------------------------------------------
            if exec_result.has_final:
                session.status = "completed"
                session.final_value = exec_result.final_value

                # Stream Final token-by-token as content (FR-023)
                async for chunk in self._stream_final(session.final_value or ""):
                    yield OracleStreamChunk(type="content", content=chunk)
                yield OracleStreamChunk(
                    type="done",
                    metadata={"iteration_count": session.iteration_count},
                )
                logger.debug(
                    "RLM session %s done: query=%r iter=%d status=%s",
                    session.session_id, query[:80], session.iteration_count, session.status,
                )
                return

            # ----------------------------------------------------------
            # 6. Store partial result, build iteration message (FR-003)
            # ----------------------------------------------------------
            if exec_result.stdout_full:
                session.partial_result = exec_result.stdout_full[:500]

            iteration_msg = prompt_builder.build_iteration_message(
                stdout_full=exec_result.stdout_full,
                stdout_total_chars=exec_result.stdout_total_chars,
                error=exec_result.error,
                iteration_number=session.iteration_count,
            )
            session.llm_history.append({"role": "user", "content": iteration_msg})

        # ------------------------------------------------------------------ #
        # Budget exhausted
        # ------------------------------------------------------------------ #
        session.status = "exhausted"
        logger.debug(
            "RLM session %s done: query=%r iter=%d status=%s",
            session.session_id, query[:80], session.iteration_count, session.status,
        )
        partial = session.partial_result or "(No answer within iteration budget.)"
        yield OracleStreamChunk(
            type="content",
            content=partial,
        )
        yield OracleStreamChunk(
            type="done",
            metadata={
                "iteration_count": session.iteration_count,
                "incomplete": True,
            },
        )
