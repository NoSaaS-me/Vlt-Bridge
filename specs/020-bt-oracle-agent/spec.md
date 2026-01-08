# Feature Specification: BT-Controlled Oracle Agent

**Feature Branch**: `020-bt-oracle-agent`
**Created**: 2026-01-08
**Status**: Draft
**Input**: User description: "BT-Controlled Oracle Agent with Atomic Signal Contract - Migrate Oracle from monolithic Python to BT-controlled architecture with XML signal protocol for agent self-reflection, prompt templates for context assessment and tool selection, and BERT fallback for edge cases"

## Overview

The Oracle agent currently operates as a monolithic control loop where the LLM makes all decisions about tool calls, context gathering, and response generation. This works, but leads to:
- **Compulsive information gathering**: Agent reads every document when one would suffice
- **Unpredictable tool selection**: No structured decision-making about which tools to use
- **No self-awareness signals**: Agent can't communicate "I need more turns" or "I'm stuck"
- **Rigid prompt loading**: Same prompt regardless of query type

This feature introduces a **Behavior Tree (BT) control layer** where:
1. **Claude remains the primary decision-maker** for tool calls and reasoning
2. **Agent emits atomic XML signals** as explicit self-reflection
3. **BT reads signals** and enforces constraints (budget, loops, fallbacks)
4. **Prompts are composed dynamically** based on query type and context

The prompts define the agent's behavior contract - they ARE half the specification.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Intelligent Context Selection (Priority: P1)

A developer asks a question and the agent intelligently decides what context sources to use rather than searching everything.

**Why this priority**: This is the core problem - the agent wastes time and tokens gathering unnecessary context. Solving this makes every interaction more efficient.

**Independent Test**: Ask "What's the weather in Paris?" - agent should NOT search code or vault, should use web search only.

**Acceptance Scenarios**:

1. **Given** a user asks "What's the weather in Paris?", **When** the agent processes the query, **Then** it calls web_search without first searching code or vault
2. **Given** a user asks "How does the auth middleware work?", **When** the agent processes the query, **Then** it searches code first, not web
3. **Given** a user asks "What did we decide about caching?", **When** the agent processes the query, **Then** it searches vault/threads first, not code or web
4. **Given** a user asks "Thanks, that helps!", **When** the agent processes the query, **Then** it responds conversationally without any tool calls

---

### User Story 2 - Agent Self-Reflection via Signals (Priority: P1)

The agent can communicate its internal state through structured XML signals, enabling the system to grant more turns, detect stuck states, or trigger fallbacks.

**Why this priority**: Without signals, the system has no visibility into agent reasoning. Signals enable the "rubber ducky" pattern where the agent talks to itself and the BT listens.

**Independent Test**: Create a scenario where a tool fails and verify the agent emits a `need_turn` signal requesting retry.

**Acceptance Scenarios**:

1. **Given** the agent needs more iterations to complete a task, **When** it responds, **Then** it includes a `<signal type="need_turn">` with reason and confidence
2. **Given** the agent has gathered sufficient context, **When** it responds, **Then** it includes a `<signal type="context_sufficient">` with sources count
3. **Given** all attempted tools fail, **When** the agent responds, **Then** it includes a `<signal type="stuck">` with attempted tools and blocker
4. **Given** the agent emits a signal, **When** the BT parses the response, **Then** it extracts and acts on the signal appropriately

---

### User Story 3 - Budget and Loop Enforcement (Priority: P2)

The BT enforces turn budgets and detects infinite loops based on signal patterns.

**Why this priority**: Without enforcement, a malfunctioning agent could consume unlimited tokens or spin forever. This is a safety guardrail.

**Independent Test**: Set max turns to 5 and verify agent is forced to respond after 5 turns even if requesting more.

**Acceptance Scenarios**:

1. **Given** an agent has used 29 of 30 turns, **When** it requests another turn via signal, **Then** the BT grants one final turn
2. **Given** an agent has used 30 of 30 turns, **When** it requests another turn, **Then** the BT forces completion with partial answer
3. **Given** an agent emits the same `need_turn` reason 3 times consecutively, **When** the BT detects this pattern, **Then** it treats the agent as stuck and triggers fallback
4. **Given** an agent is stuck, **When** BERT fallback activates, **Then** an alternative strategy is attempted or the limitation is acknowledged

---

### User Story 4 - Dynamic Prompt Composition (Priority: P2)

System prompts are composed from segments based on query type, not loaded monolithically.

**Why this priority**: Different query types need different instructions. A code question needs different guidance than a research question.

**Independent Test**: Ask a code question and verify the prompt includes code-specific instructions but not research instructions.

**Acceptance Scenarios**:

1. **Given** a query classified as "code", **When** the system prompt is built, **Then** it includes the code analysis segment
2. **Given** a query classified as "research", **When** the system prompt is built, **Then** it includes the research methodology segment
3. **Given** any query, **When** the system prompt is built, **Then** it always includes the signal emission instructions
4. **Given** a query that doesn't need tools, **When** the system prompt is built, **Then** tool-heavy segments are omitted to save tokens

---

### User Story 5 - BERT Fallback for Edge Cases (Priority: P3)

When the agent doesn't emit clear signals, BERT classifiers provide fallback decisions.

**Why this priority**: BERT is a safety net, not the primary system. It activates when the agent fails to communicate clearly.

**Independent Test**: Simulate an agent response with no signal and verify BERT fallback provides a default classification.

**Acceptance Scenarios**:

1. **Given** an agent response with no signal, **When** 3 turns pass without signals, **Then** BERT fallback activates to classify intent
2. **Given** an agent emits a signal with confidence < 0.3, **When** the BT processes it, **Then** BERT fallback is consulted for confirmation
3. **Given** an explicit `stuck` signal, **When** BERT processes it, **Then** it attempts to identify an alternative strategy
4. **Given** BERT classifiers are unavailable, **When** fallback is needed, **Then** heuristic defaults are used instead

---

### Edge Cases

- What happens when the agent emits multiple signals in one response? (Only first is processed, log warning)
- What happens when signal XML is malformed? (Treat as no signal, trigger BERT fallback)
- What happens when agent requests a capability that doesn't exist? (Log capability gap, acknowledge limitation)
- What happens when all context sources return empty? (Agent should emit `stuck` or `partial_answer`)
- What happens during a topic shift mid-conversation? (Context assessment re-runs, previous context may be cleared)

---

## Requirements *(mandatory)*

### Functional Requirements

#### Signal Contract
- **FR-001**: Agent response format MUST support embedded XML signals at end of response
- **FR-002**: System MUST parse and extract signals from agent responses in real-time (streaming)
- **FR-003**: System MUST support these signal types: `need_turn`, `context_sufficient`, `stuck`, `need_capability`, `partial_answer`, `delegation_recommended`
- **FR-004**: Each signal MUST include a `confidence` score (0.0-1.0)
- **FR-005**: System MUST strip signal XML from user-visible response text

#### BT Control Layer
- **FR-006**: BT MUST control when the agent loop continues vs terminates
- **FR-007**: BT MUST enforce maximum turn limits (configurable, default 30)
- **FR-008**: BT MUST detect loop patterns (same reason signal 3+ times)
- **FR-009**: BT MUST log all signals for debugging and audit
- **FR-010**: BT MUST support shadow mode (run both old and new implementations)

#### Prompt Composition
- **FR-011**: System prompt MUST be composed from segments, not monolithic
- **FR-012**: Signal emission instructions MUST always be included in system prompt
- **FR-013**: Query-type-specific segments MUST be loaded based on context assessment
- **FR-014**: Prompt segments MUST be stored as separate files for maintainability

#### Context Assessment
- **FR-015**: System MUST classify query type before context gathering (code/documentation/research/conversational/action)
- **FR-016**: Context gathering tools MUST be selected based on query classification
- **FR-017**: System MUST track which context sources have been tried
- **FR-018**: System MUST have a "sufficient context" heuristic (replaceable by BERT later)

#### BERT Fallback
- **FR-019**: BERT fallback MUST activate when no signal for 3+ turns
- **FR-020**: BERT fallback MUST activate when signal confidence < 0.3
- **FR-021**: BERT fallback MUST activate on explicit `stuck` signal
- **FR-022**: System MUST function with heuristic defaults when BERT unavailable

### Key Entities

- **Signal**: Structured agent self-reflection (type, confidence, fields, raw_xml)
- **PromptSegment**: Reusable prompt component (id, content, conditions for inclusion)
- **QueryClassification**: Result of context assessment (type, needs_code, needs_vault, needs_web)
- **AgentState**: Current loop state (turn count, signals emitted, tools tried, context sources)

---

## Prompt Deliverables *(mandatory for this feature)*

The prompts define the agent's behavior contract. These are first-class deliverables:

### Core Prompts

| Prompt File | Purpose | Always Included |
|-------------|---------|-----------------|
| `oracle/base.md` | Core identity, capabilities, constraints | Yes |
| `oracle/signals.md` | XML signal format and when to emit each type | Yes |
| `oracle/tools-reference.md` | Available tools and when to use each | Yes |

### Context-Specific Prompts

| Prompt File | Purpose | Included When |
|-------------|---------|---------------|
| `oracle/code-analysis.md` | How to analyze code, cite sources | query_type = "code" |
| `oracle/documentation.md` | How to search and synthesize docs | query_type = "documentation" |
| `oracle/research.md` | How to conduct web research | query_type = "research" |
| `oracle/conversation.md` | How to handle follow-ups | query_type = "conversational" |
| `oracle/actions.md` | How to perform write operations | query_type = "action" |

### Signal Emission Prompt (signals.md)

This is the critical prompt that teaches the agent to emit signals. It MUST include:

1. **Signal format specification** - exact XML structure
2. **When to emit each signal type** - clear triggers
3. **Confidence calibration** - what 0.3 vs 0.7 vs 0.9 means
4. **Examples** - complete response with embedded signal
5. **Anti-patterns** - what NOT to do (multiple signals, inline signals)

### Prompt Composition Rules

1. Always include: `base.md` + `signals.md` + `tools-reference.md`
2. Add context-specific prompt based on query classification
3. If context is large, add `summarization.md` instructions
4. If previous errors occurred, add `error-recovery.md`
5. Total prompt should not exceed 8000 tokens (before tools)

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Agent correctly classifies query type 90% of the time (measured against labeled test set)
- **SC-002**: Agent emits appropriate signal in 95% of responses (measured by signal presence and correctness)
- **SC-003**: Unnecessary tool calls reduced by 50% compared to baseline (measured by tool call count for same queries)
- **SC-004**: Agent completes tasks within turn budget 99% of the time (no runaway loops)
- **SC-005**: Shadow mode shows equivalent or better response quality vs current implementation
- **SC-006**: System functions correctly with BERT disabled (heuristic-only mode)

### Quality Metrics

- **SC-007**: Agent response latency does not increase by more than 10% due to BT overhead
- **SC-008**: Signal parsing adds less than 50ms to response processing
- **SC-009**: All signals are logged and auditable
- **SC-010**: Prompt composition is deterministic (same inputs = same prompt)

---

## Assumptions

1. **Claude understands XML signals**: The LLM can reliably emit structured XML when instructed
2. **Signal parsing is reliable**: Regex-based XML extraction is sufficient for well-formed signals
3. **Query classification is tractable**: Simple heuristics can achieve 80%+ accuracy before BERT
4. **BT overhead is negligible**: Tree traversal adds minimal latency vs direct function calls
5. **Prompts fit in context**: Combined prompt segments stay under token limits
6. **Shadow mode is feasible**: Both implementations can run in parallel for comparison

---

## Dependencies

- **BT Universal Runtime (019)**: Must be complete and production-ready
- **Existing Oracle Agent**: Current implementation provides baseline behavior
- **ANS (Agent Notification System)**: For emitting system messages to frontend
- **Tool Executor**: Existing tool execution infrastructure

---

## Out of Scope

- **BERT model training**: Using pre-trained models or heuristics only
- **UI changes**: Frontend displays same chat interface
- **New tools**: No new tool implementations
- **Artifact generation**: Future feature, architecture supports it but not implemented
- **Multi-agent orchestration**: Single Oracle agent only

---

## References

- Architecture document: `Ai-notes/01-08-2026/BT-Agent-Integration/architecture.md`
- Signal contract: `Ai-notes/01-08-2026/BT-Agent-Integration/signal-contract.md`
- BERT signals wishlist: `Ai-notes/01-08-2026/BT-Agent-Integration/bert-signals.md`
- Test scenarios: `Ai-notes/01-08-2026/BT-Agent-Integration/test-scenarios.md`
- Implementation plan: `Ai-notes/01-08-2026/BT-Agent-Integration/implementation-plan.md`
