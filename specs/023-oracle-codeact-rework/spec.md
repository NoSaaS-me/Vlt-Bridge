# Feature Specification: Oracle & Librarian CodeAct Rework

**Feature Branch**: `023-oracle-codeact-rework`
**Created**: 2026-03-11
**Status**: Draft
**Input**: User description: "Rework the Oracle and Librarian agents using LangGraph CodeAct for multi-turn state management, Graphiti for cross-session memory, an expanded tool registry with shell access, and a planner node for complex task decomposition."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Multi-Turn Conversation That Remembers Context (Priority: P1)

A developer asks the Oracle a question about a codebase, gets an answer, and then asks a follow-up question that builds on the first. The agent should remember what it looked up, what files it examined, and what conclusions it reached — without re-doing that work. The conversation should feel continuous, not like starting from scratch on every message.

**Why this priority**: This is the root cause of the current rewrite. Every prior Oracle version fails at multi-turn continuity. It is the core value proposition of the upgrade and blocks all other improvements.

**Independent Test**: Can be fully tested by asking "Where is the authentication middleware?" then immediately asking "What does it import?" — the agent should answer the second question using context already gathered, without re-searching for the auth file.

**Acceptance Scenarios**:

1. **Given** a user has an active Oracle conversation where the agent already retrieved information about a file, **When** they ask a follow-up question referencing that file, **Then** the agent answers using the context already in memory without repeating retrieval steps.
2. **Given** a user closes the Oracle panel and reopens it later, **When** they continue the same conversation thread, **Then** the agent resumes with full context of the previous session intact.
3. **Given** a user is on turn 5 of a conversation, **When** the server restarts, **Then** the conversation can be resumed from the last saved checkpoint.

---

### User Story 2 - Cross-Session Project Memory (Priority: P2)

A developer uses the Oracle across multiple sessions over days or weeks. Facts established in earlier sessions ("the auth module was refactored last week", "we use snake_case for all service functions") should be available in new sessions without the agent needing to re-discover them. The agent should remember project-specific architectural decisions and the developer's preferences.

**Why this priority**: Without cross-session memory, every new conversation starts cold. The agent becomes genuinely useful as a project collaborator only when it accumulates knowledge over time.

**Independent Test**: Can be tested by telling the Oracle a project fact in one session ("the rate limiter is in src/api/middleware/rate_limit.py"), ending the session entirely, starting a new one, and asking "where is rate limiting handled?" — the agent should know without searching.

**Acceptance Scenarios**:

1. **Given** a developer tells the Oracle a project fact in session A, **When** they start a new session B on a different day, **Then** the Oracle recalls that fact when asked a related question.
2. **Given** a project file has moved since a fact was stored, **When** the Oracle is told the new location, **Then** it updates its knowledge and no longer references the old path.
3. **Given** a developer has stated a preference ("I like concise answers"), **When** the Oracle responds in a future session, **Then** it respects that preference without being reminded again.

---

### User Story 3 - General-Purpose Tasks Beyond Codebase Search (Priority: P3)

A developer asks the Oracle to perform tasks that go beyond lookup: running a git command to see recent changes, calculating something based on retrieved data, writing a summary note to the vault, or researching an external library. The agent should handle these general-purpose requests without breaking down.

**Why this priority**: The current Oracle is too narrow — it can only search the codebase and vault. Expanding its action space makes it useful for a wider range of developer tasks. Builds on P1/P2 being stable first.

**Independent Test**: Can be tested by asking "What changed in the auth module in the last 10 git commits?" — the agent should retrieve git log output and summarize meaningful changes.

**Acceptance Scenarios**:

1. **Given** a developer asks about recent git history for a file, **When** the Oracle processes the request, **Then** it retrieves actual version history and provides a meaningful summary.
2. **Given** a developer asks the Oracle to save a finding to the vault, **When** the agent completes research, **Then** a new note appears in the vault with the correct content.
3. **Given** a developer asks the Oracle to research an external library, **When** it uses web search to gather information, **Then** it synthesizes a useful answer from multiple sources.

---

### User Story 4 - Structured Planning for Complex Multi-Step Tasks (Priority: P4)

A developer asks the Oracle to do something complex: "find all the places we're missing error handling across the codebase" or "trace all callers of this function and summarize what they do". The agent should decompose the request into a plan, execute it step by step, and remain on track.

**Why this priority**: Complex tasks are where agents most often fail. Planning improves reliability for the hardest cases. Depends on P1 (needs multi-turn state to track plan progress).

**Independent Test**: Can be tested by asking "Find all files that import from src/services/vault.py and tell me what functions they use" — the agent should plan a search strategy, execute it completely, and deliver a structured answer.

**Acceptance Scenarios**:

1. **Given** a complex multi-step request, **When** the agent processes it, **Then** it generates an internal plan and executes each step in order without losing track midway.
2. **Given** a step in the plan produces unexpected results, **When** the agent encounters the issue, **Then** it adapts its plan rather than returning an incomplete answer.
3. **Given** a simple factual question, **When** the agent processes it, **Then** it skips the planning step and answers directly without extra latency.

---

### Edge Cases

- What happens when a conversation thread becomes very long and the context window fills? Earlier context must be summarized gracefully without losing critical facts.
- What happens when the agent tries a shell command not in the allowlist? It receives a clear error and finds an alternative approach.
- What happens when the memory service is unavailable? The agent continues functioning with degraded memory — responses are still generated, just without recalled facts.
- What happens when the user cancels a long-running query mid-stream? The response stops cleanly and the conversation state remains consistent and resumable.
- What happens when two sessions send messages to the same conversation thread simultaneously? The second request cancels the first before starting.
- What happens when a stored memory fact conflicts with current code? The agent updates memory with the newer finding rather than contradicting itself.

---

## Requirements *(mandatory)*

### Functional Requirements

**Multi-Turn State**

- **FR-001**: The system MUST preserve the agent's working context (variables, intermediate findings) across multiple messages in the same conversation thread without the agent re-fetching already-retrieved information.
- **FR-002**: The system MUST allow a conversation to be resumed after a server restart or session timeout with full message history and working context restored.
- **FR-003**: The system MUST associate each conversation with a persistent thread identifier that the frontend can pass to continue an existing conversation.
- **FR-004**: The system MUST provide a list of past conversation threads for the authenticated user, browseable by the frontend.

**Cross-Session Memory**

- **FR-005**: The system MUST persist project facts, architectural decisions, and file locations discovered during conversations so they are available in future sessions for the same project.
- **FR-006**: The system MUST detect when a stored fact is contradicted by new information and mark the old fact as no longer valid.
- **FR-007**: The system MUST scope memory separately per user and per project so facts from one project do not appear in another project's conversations.
- **FR-008**: The agent MUST be able to explicitly store a named fact and retrieve facts by semantic similarity to a query.

**Tool Registry**

- **FR-009**: The system MUST provide the agent access to: semantic code search, file reading, vault read/write/search, thread read/write, web search, web fetch, and version history operations.
- **FR-010**: The system MUST restrict shell command execution to a fixed allowlist of read-only operations (version history, file listing, text search, file content retrieval, line counting, file comparison).
- **FR-011**: The agent MUST be able to discover what tools are available to it at runtime.
- **FR-012**: The system MUST prevent arbitrary code execution on the host operating system; only the approved tool set may interact with the filesystem or shell.

**Planner**

- **FR-013**: The system MUST generate a decomposed task plan before executing requests identified as complex multi-step tasks.
- **FR-014**: The system MUST bypass the planning step for simple factual questions to avoid unnecessary latency.
- **FR-015**: The agent MUST be able to revise its plan mid-task when new information requires a different approach.

**Streaming & API Compatibility**

- **FR-016**: The system MUST stream responses to the frontend in real time, including progress updates as the agent works through each step.
- **FR-017**: The existing `/api/oracle/stream` request format MUST be accepted without modification — current frontend requests work without changes.
- **FR-018**: The system MUST support cancellation of an active Oracle query, stopping the response stream within a reasonable time after cancellation is requested.
- **FR-019**: The existing Oracle SSE chunk types (content, progress, tool_call, tool_result, done, error) MUST continue to be emitted so the existing frontend renders correctly.

**Librarian**

- **FR-020**: The agent MUST be able to delegate content summarization and vault organization tasks to the Librarian as a synchronous callable operation during its execution.
- **FR-021**: The existing standalone thread summarization endpoint MUST continue to function without modification.

**Web Research**

- **FR-022**: The agent MUST be able to invoke multi-step web research when it determines external information synthesis is necessary, using the user's configured search provider.
- **FR-023**: The frontend MUST provide a control to grant or restrict the agent's permission to use web research for a given conversation.

### Key Entities

- **Conversation Thread**: A persistent multi-turn exchange identified by a thread ID. Contains full message history and agent working context. Scoped to a user and project.
- **Agent Working Context**: Variables and intermediate findings from the current execution session. Persisted across messages in the same thread.
- **Memory Fact**: A piece of project or user knowledge with a temporal validity window. Can become outdated; newer contradicting facts supersede older ones.
- **Task Plan**: A decomposed multi-step plan generated at the start of a complex request. Can be updated by the agent mid-execution.
- **Tool**: A callable operation available in the agent's execution environment. Categories: retrieval, mutation, compute, shell (allowlisted), memory, and meta.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Follow-up questions in an active conversation are answered correctly using context from earlier in the same thread in at least 95% of test cases, without repeating retrieval work already done.
- **SC-002**: A fact explicitly stored in memory in one session is correctly recalled in a new session at least 90% of the time when the query is semantically related.
- **SC-003**: The agent successfully completes complex multi-step tasks (3+ dependent steps) at a rate at least 3× higher than the current Oracle on the same benchmark task set.
- **SC-004**: Simple factual questions (single lookup, no multi-step dependency) receive a first response chunk within 3 seconds under normal load.
- **SC-005**: All existing frontend Oracle requests function correctly against the new backend without any frontend code changes — 100% backward compatibility.
- **SC-006**: A cancellation request stops the active response stream within 2 seconds and leaves the conversation in a consistent, resumable state.
- **SC-007**: Shell commands outside the approved allowlist are rejected 100% of the time with a clear error message; no arbitrary shell execution occurs.
- **SC-008**: A conversation thread can be resumed after a server restart with full context, verified by asking a follow-up question that references information from before the restart.

---

## Assumptions

- The frontend model selector for Oracle controls the main actor model. A separate setting (or fallback to the same model) controls the planner node.
- API keys are resolved per-request from user settings and are never stored in the persistent conversation state.
- The existing CodeRAG index is used for semantic code search without modification to indexing.
- The existing vault storage and thread storage systems are used as-is; this feature does not change how notes or vlt threads are stored.
- The memory service runs as a local Docker service alongside the backend — no cloud dependency.
- Existing conversation history from the previous Oracle implementation remains readable by the frontend in read-only mode; new conversations use the new persistence mechanism.
- The feature targets local/trusted-user deployment; full multi-tenant sandbox isolation is a future concern.
- The Librarian's standalone thread summarization capability is not changed by this feature.
