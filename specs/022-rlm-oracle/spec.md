# Feature Specification: RLM Oracle — Recursive Language Model Inference Harness

**Feature Branch**: `022-rlm-oracle`
**Created**: 2026-02-22
**Status**: Draft

## Overview

Replace the Behavior Tree Oracle with a Recursive Language Model (RLM) inference-time harness. Rather than controlling agent workflow through XML signals, BT nodes, and a prompt composer, the oracle gives the LLM a persistent REPL environment containing the entire project as addressable variables and lets it write Python code to explore, slice, and synthesize answers. The LLM's context window stays small and fixed regardless of project size; all heavy lifting happens in REPL-executed code, with sub-oracle calls made programmatically inside loops.

This is a clean architectural replacement. The BT runtime, signal parser, query classifier, prompt composer, and BT wrapper are removed entirely. The oracle API surface (endpoint, SSE streaming contract, MCP tool interface) is unchanged.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Cross-Codebase Synthesis Query (Priority: P1)

A developer asks a question that requires reading and synthesizing information from multiple files, modules, or layers of the codebase that a keyword or vector search would not reliably surface. Examples: "How does the connection lifecycle flow from a vlt-mcp tool call all the way to SQLite?", "Where are all the places we call the backend oracle API?", "What architectural decisions led to the current BT design?"

The current BT Oracle fails or gives partial answers for these because it relies on retrieval to pre-select context. The RLM harness solves this by letting the LLM write code that iterates over all relevant files, calling sub-oracle on each slice, and synthesizing the results — guaranteed coverage regardless of embedding quality.

**Why this priority**: This is the primary motivation for the feature. P1 because everything else is secondary to fixing the fundamental retrieval bottleneck.

**Independent Test**: Can be fully tested by submitting a question whose correct answer requires synthesizing information from ≥5 files not closely related by embedding distance, verifying the answer is complete and accurate.

**Acceptance Scenarios**:

1. **Given** a project with source code, **When** a user asks "Explain the full database session lifecycle across all layers," **Then** the oracle produces an answer that correctly describes behavior in ≥3 different files without hallucinating connections not present in the code.

2. **Given** a query whose answer is spread across code, a vlt thread decision log, and a vault note, **When** the user submits the query, **Then** the oracle synthesizes all three sources into a coherent response.

3. **Given** a project corpus exceeding any single model context window, **When** the user asks a question requiring broad synthesis, **Then** the oracle returns a correct answer without truncating or randomly sampling the input.

---

### User Story 2 — Focused Query Handled Efficiently (Priority: P2)

A developer asks a focused, specific question: "What does `vlt_code_lookup` return when no index exists?" The oracle should not fan out across the whole project for simple targeted questions — it should recognize the query is narrow and handle it with minimal sub-oracle calls.

**Why this priority**: P2 because efficiency and latency matter for the majority of queries which are focused, not broad. An oracle that always does full fan-out would be too slow for daily use.

**Independent Test**: Submit a focused single-file question, verify the oracle answers it in ≤3 REPL iterations without scanning unrelated files.

**Acceptance Scenarios**:

1. **Given** a focused question about a specific function, **When** submitted, **Then** the oracle locates and reads only the relevant file(s) rather than scanning everything.

2. **Given** a conversational or definitional question ("What is a vlt thread?"), **When** submitted, **Then** the oracle answers in ≤2 iterations without invoking expensive file iteration loops.

---

### User Story 3 — Project History and Decision Reconstruction (Priority: P2)

A developer asks "Why did we switch from X to Y?" or "What was the reasoning behind the oracle architecture?" — questions whose answers live in vlt threads and vault notes, not in source code. The oracle should reach into the full thread history and note corpus to reconstruct the reasoning.

**Why this priority**: P2 because thread and note synthesis is a major use case for vlt-oracle and currently underserved by code-retrieval-focused approaches.

**Independent Test**: Submit a question whose correct answer is in a vlt thread, verify the oracle finds and quotes the relevant thread content.

**Acceptance Scenarios**:

1. **Given** a design decision documented in a vlt thread ≥10 pushes ago, **When** the user asks about it, **Then** the oracle finds and quotes the relevant thread content.

2. **Given** a question referencing both a code behavior and its documented rationale, **When** submitted, **Then** the oracle combines code analysis with thread/note context in the answer.

---

### User Story 4 — Streaming Progress Visibility (Priority: P3)

The oracle shows live progress while it works. As the REPL executes — scanning files, calling sub-oracle on slices, building up results — the user sees meaningful status updates before the final answer streams.

**Why this priority**: P3 because this is UX quality, not correctness. The oracle is useful without it; streaming makes long-running queries feel less opaque.

**Independent Test**: Submit a query that triggers ≥5 REPL iterations and observe that progress messages appear before the final answer.

**Acceptance Scenarios**:

1. **Given** a multi-iteration oracle query, **When** the query is running, **Then** the user sees incremental status messages (e.g., "Scanning 12 files…", "Found 3 relevant sections…") before the final answer begins streaming.

2. **Given** any oracle query, **When** the final answer is ready, **Then** it streams token-by-token consistent with current oracle behavior.

---

### Edge Cases

- What happens when the REPL code throws a Python exception? The exception traceback is included in the next LLM iteration's history so it can recover and retry with corrected code.
- What happens when sub-oracle is called with a prompt larger than the model's context window? The caller is responsible for slicing via the smart wrapper's chunking API; the sub-oracle receives only a bounded segment.
- What happens when the LLM never sets `Final` within the iteration budget? The oracle terminates, returns the best partial result found in REPL state, and marks the response as incomplete.
- What happens when the REPL code attempts to import unauthorized modules (`os`, `subprocess`)? The restricted namespace raises `ImportError` which the LLM sees in the next iteration and can work around.
- What happens when the project has no CodeRAG index? The project wrapper exposes files directly via the filesystem path stored in the vlt project; BM25 search degrades gracefully to grep.
- What happens when a file is binary or very large (>1MB)? The file handle exposes only metadata; attempting to read content returns a structured notice the LLM can handle programmatically.
- What happens on concurrent oracle queries from multiple users? Each query gets its own isolated REPL session with its own namespace and state.

---

## Requirements *(mandatory)*

### Functional Requirements

**REPL Loop**

- **FR-001**: The system MUST maintain a persistent REPL session per oracle query where all project content is accessible as named variables in a Python execution environment, not loaded into the LLM context window.
- **FR-002**: The LLM MUST receive only constant-size metadata about the project at the start of each query (file count, total size, short manifest excerpt) — never raw file content in the root context.
- **FR-003**: After each REPL code execution, the system MUST append to the LLM's history only a metadata summary of stdout (prefix of ≤200 characters plus total character count) — not the full stdout content.
- **FR-004**: The REPL loop MUST continue until the LLM sets the `Final` variable in REPL state, or the iteration budget is exhausted.
- **FR-005**: The system MUST provide a `sub_oracle(prompt: str) -> str` function within the REPL namespace that makes a fresh LLM call with its own bounded context window and returns the result as a string for storage in REPL variables.
- **FR-006**: Sub-oracle calls MUST be invokable programmatically from within REPL code, including inside loops, enabling the model to perform work proportional to the size of the project.

**ProjectContext Wrapper**

- **FR-007**: The system MUST provide a `ProjectContext` object in the REPL namespace that exposes the full project (source files, vlt threads, vault notes) without loading content into memory until explicitly requested.
- **FR-008**: `ProjectContext` MUST provide a file manifest (paths, sizes, languages, last-modified timestamps) that the LLM can inspect to decide what to read, without triggering any file I/O.
- **FR-009**: `ProjectContext` MUST provide text search (grep/regex) and semantic search across all files as standalone operations that return match locations without loading full file content.
- **FR-010**: File content access MUST be provided via handle objects that lazily load content only when explicitly read, and support line-range slicing to avoid loading entire large files unnecessarily.
- **FR-011**: File handles MUST expose symbol extraction (functions, classes, methods with line numbers) as a metadata operation that does not require the LLM to read the full file body.
- **FR-012**: File handles MUST provide a chunking method that splits large files into semantic segments (by function/class boundaries where parseable, by line count otherwise), each chunk itself being a handle supporting the same interface.
- **FR-013**: The system MUST expose vlt thread content through `ProjectContext` with the same lazy-loading contract — thread metadata available immediately, node content loaded on demand.
- **FR-014**: Files exceeding a configurable size threshold (default: 1MB) MUST return only metadata from content-read operations, preventing memory exhaustion.

**REPL Safety**

- **FR-015**: The REPL namespace MUST restrict available symbols to an explicit allowlist: the project context object, the sub-oracle function, approved standard library modules (`re`, `json`, `collections`, `itertools`, `math`, `datetime`), and the `Final` sentinel. Direct filesystem access and process execution MUST be excluded.
- **FR-016**: Each REPL code execution step MUST have a per-step timeout (default: 30 seconds) to prevent runaway loops from blocking the server indefinitely.
- **FR-017**: The system MUST enforce a maximum iteration count per oracle query (default: 25 root iterations; default: 8 sub-oracle iterations per call) and terminate gracefully with a partial result when the budget is reached.

**Migration from BT**

- **FR-018**: The BT runtime, BT nodes, BT conditions, BT wrapper, XML signal parser, query classifier, and dynamic prompt composer MUST be removed as part of this feature.
- **FR-019**: The oracle REST API (`/api/oracle`, `/api/oracle/stream`) and SSE streaming contract MUST remain unchanged — callers see no interface difference.
- **FR-020**: The vlt MCP `vlt_oracle_query` tool interface MUST remain unchanged.
- **FR-021**: The ANS (Agent Notification System) event emission MUST be adapted to fire from the RLM loop — iteration budget warnings and REPL execution errors — preserving the existing event bus and subscriber infrastructure.

**Streaming**

- **FR-022**: REPL stdout (progress messages printed by the LLM's code) MUST be emitted as SSE progress events to the client as they are produced, giving users visible work status.
- **FR-023**: When `Final` is set, its value MUST stream as SSE content events token-by-token, consistent with current oracle streaming behavior.

---

### Key Entities

- **RLM Session**: A single oracle query's execution context — contains the REPL state, LLM history, iteration count, project context binding, and recursion depth tracker. Scoped to one query, not persisted.
- **ProjectContext**: The root object in the REPL namespace. Wraps a vlt project (source files, threads, notes). Provides manifest, search, and lazy-loaded handle access. Stateless between REPL iterations; results are stored in LLM-written REPL variables.
- **TextHandle**: A lazy reference to a text resource (file, thread, note, or chunk). Carries metadata always; loads content only on explicit read. Supports slicing, symbol extraction, grep, and semantic chunking.
- **REPLNamespace**: The restricted Python execution environment for one oracle session. Contains the project context, sub-oracle function, approved modules, and the Final sentinel. Persists variable state across all iterations of one query.
- **SubOracleCall**: A fresh, bounded LLM invocation made programmatically from within REPL code. Has its own iteration budget and context window. Returns a string. Recursion depth is tracked and capped.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The oracle correctly answers questions requiring synthesis across ≥5 non-adjacent files at least 80% of the time on a representative 20-question test set, measured against ground-truth answers prepared in advance.
- **SC-002**: The root LLM context window at any iteration stays under 4,000 tokens regardless of project size, verifiable by logging token counts during test queries.
- **SC-003**: Simple focused queries (answer resides in ≤2 files) complete end-to-end in under 20 seconds on the current Vlt-Bridge codebase.
- **SC-004**: The oracle handles a synthesized project corpus of ≥500,000 tokens without truncation, random sampling, or out-of-memory errors.
- **SC-005**: REPL execution errors are self-corrected by the LLM in the next iteration at least 70% of the time, measurable by counting error/recovery pairs in execution logs.
- **SC-006**: The existing oracle API endpoints and the `vlt_oracle_query` MCP tool pass all current integration tests without modification to any calling code.
- **SC-007**: Any query taking longer than 5 seconds emits at least one SSE progress event before the final answer, giving users visible activity feedback.
- **SC-008**: All BT-related directories and files are removed, resulting in a net reduction of the backend source tree with no BT imports remaining in any non-test file.

---

## Assumptions

- The model used (OpenRouter/DeepSeek or equivalent) reliably follows REPL-style code-writing instructions when given a clear system prompt — validated by the RLM paper (arxiv 2512.24601) on both GPT-5 and Qwen3 without fine-tuning.
- The vlt project record contains a filesystem path (set during `vlt coderag init`) that ProjectContext uses to locate source files; projects without this path fall back to threads and notes only.
- Sub-oracle recursion is capped at depth 2 (root + one level of sub-calls) for this initial implementation; deeper recursion adds latency with marginal benefit at current project scale.
- Vault notes require the Document-MCP backend to be running; if unavailable, notes are omitted from ProjectContext with a structured notice in the REPL.
- The REPL trust model is "trusted single user" — namespace restriction guards against accidental footguns, not adversarial prompts.
- Streaming the Final value token-by-token is achievable because Final is generated by a terminal sub-oracle call whose output can be streamed directly to the SSE channel before the full string is assembled.
