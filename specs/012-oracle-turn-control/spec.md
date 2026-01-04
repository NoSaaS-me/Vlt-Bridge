# Feature Specification: Oracle Agent Turn Control & Decision Tree Protocol

**Feature Branch**: `012-oracle-turn-control`
**Created**: 2026-01-02
**Status**: Draft
**Input**: Refactor Oracle agent turn-taking logic with pluggable decision trees, configurable agent limits, and proper termination conditions

## Overview

The Oracle agent currently has a rigid turn-taking system with a hard-coded 30-iteration limit and no intelligent termination conditions. This feature refactors the control flow to:

1. Introduce enterprise-grade termination conditions (goal achievement, token budgets, timeouts, no-progress detection)
2. Create a pluggable DecisionTree protocol for future skill/workflow extensions
3. Expose agent configuration in the user settings UI
4. Add system notifications visible in chat when approaching or hitting limits

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure Agent Behavior (Priority: P1)

A user opens the Settings page and adjusts the Oracle agent's behavior to match their preferences. They can set iteration limits, token budgets, and timeout values. These settings persist across sessions and immediately affect subsequent conversations.

**Why this priority**: Core functionality that enables all other features. Without user-configurable settings, the agent uses hardcoded defaults that may not suit different use cases.

**Independent Test**: Can be fully tested by opening Settings, modifying AgentConfig values, and verifying they persist after page refresh. Delivers immediate value by giving users control.

**Acceptance Scenarios**:

1. **Given** a user is on the Settings page, **When** they modify the "Max Iterations" slider from 15 to 10, **Then** the value is saved and displayed correctly on page reload
2. **Given** a user has set custom AgentConfig values, **When** they start a new Oracle conversation, **Then** the agent respects those configured limits
3. **Given** a user has invalid input (e.g., max iterations = 0), **When** they try to save, **Then** validation prevents saving with a helpful error message

---

### User Story 2 - Graceful Limit Notifications (Priority: P1)

When the Oracle agent approaches configured limits (70% of max iterations, 80% of token budget), a "System" message appears in the chat flow notifying the user. The agent also receives a hint to wrap up its response. When hard limits are hit, the agent saves its accumulated work and provides an inline explanation.

**Why this priority**: Critical for user experience - users need visibility into why the agent may produce truncated responses. Prevents confusion and frustration.

**Independent Test**: Can be tested by setting a low max iteration limit (e.g., 3), asking a complex question that requires many tool calls, and observing system notifications appear in chat.

**Acceptance Scenarios**:

1. **Given** an agent is at 70% of max iterations (e.g., iteration 7 of 10), **When** the next turn begins, **Then** a "System" message appears in chat: "Approaching iteration limit (7/10)"
2. **Given** an agent is at 80% of token budget, **When** the next turn begins, **Then** a "System" message appears in chat with token usage info
3. **Given** an agent hits the max iteration limit, **When** it cannot continue, **Then** it saves all accumulated content from the main chain and includes an inline explanation in its final response
4. **Given** an agent hits a limit, **When** it terminates, **Then** any subagent work-in-progress is NOT saved (only main chain content is preserved)

---

### User Story 3 - Intelligent Termination (Priority: P2)

The Oracle agent stops processing when it achieves its goal with high confidence, rather than continuing unnecessarily. It also detects when it's making no progress (repeating the same actions) and terminates gracefully with an explanation.

**Why this priority**: Prevents token waste and improves response times. Users get faster answers when the agent knows it's done.

**Independent Test**: Can be tested by asking a simple question and verifying the agent stops after providing the answer, rather than making additional unnecessary tool calls.

**Acceptance Scenarios**:

1. **Given** an agent has provided a complete answer with no pending tool calls, **When** the model returns finish_reason="stop", **Then** the agent terminates without additional iterations
2. **Given** an agent has made the same tool call with the same arguments 3 times consecutively, **When** the third identical result returns, **Then** the agent terminates with "no_progress" reason and explains the situation
3. **Given** an agent encounters 3 consecutive tool errors, **When** the third error occurs, **Then** the agent terminates gracefully with accumulated content and error summary

---

### User Story 4 - DecisionTree Protocol for Future Extensions (Priority: P2)

The agent's control flow is refactored to use a pluggable DecisionTree interface. The default implementation preserves current behavior but with improved termination conditions. Future skills (like DeepResearcher) can provide their own decision trees via decorator-based registration.

**Why this priority**: Architectural foundation for future extensibility. Without this, adding new workflows requires modifying core agent code.

**Independent Test**: Can be tested by verifying the default decision tree produces the same behavior as the current implementation (minus the removed 30-turn hard limit). Future testing will verify skill registration.

**Acceptance Scenarios**:

1. **Given** the refactored agent with default DecisionTree, **When** processing a standard query, **Then** behavior is equivalent to current implementation with improved termination
2. **Given** a skill declares a custom DecisionTree via decorator, **When** that skill is active, **Then** the agent uses the skill's decision tree for control flow
3. **Given** AgentState is extended with new fields for a workflow module, **When** the module runs, **Then** it can read/write its state without affecting core state

---

### User Story 5 - System User in Chat Flow (Priority: P3)

A new "System" participant appears in the chat UI when system-level notifications need to be displayed. This is visually distinct from user and assistant messages, providing clear separation of concerns.

**Why this priority**: UX improvement that enhances the primary notification functionality. Lower priority because notifications could technically appear as assistant messages, but this provides better clarity.

**Independent Test**: Can be tested by triggering a system notification (e.g., approaching limits) and verifying a distinctly styled "System" message appears in the conversation.

**Acceptance Scenarios**:

1. **Given** a system notification is triggered, **When** the chat UI renders, **Then** a "System" message appears with distinct styling (different from user/assistant)
2. **Given** multiple system notifications in a conversation, **When** viewing chat history, **Then** all system messages are preserved and displayed correctly
3. **Given** a system message, **When** the user hovers or interacts, **Then** no reply/edit actions are available (read-only)

---

### Edge Cases

- What happens when a user sets max iterations to 1? The agent completes one turn, saves content, and terminates with explanation
- What happens when token budget is exceeded mid-response? Current response chunk is completed, then agent terminates with partial content saved
- What happens when the user cancels while a system notification is pending? Cancellation takes priority; notification may or may not appear
- What happens when AgentConfig values conflict (e.g., timeout too short for token budget)? Whichever limit is hit first wins; no cross-validation required
- What happens when the connection drops during a limit notification? Finally block saves accumulated content as before
- What happens when no-progress detection triggers but the agent hasn't produced any content? Agent terminates with explanation of what it attempted

## Requirements *(mandatory)*

### Functional Requirements

**Agent Configuration**

- **FR-001**: System MUST provide an AgentConfig data structure with the following user-configurable fields:
  - `maxIterations`: Maximum number of agent turns (default: 15)
  - `softWarningPercent`: Percentage at which to warn (default: 70)
  - `tokenBudget`: Maximum tokens per session (default: 50,000)
  - `tokenWarningPercent`: Percentage at which to warn (default: 80)
  - `timeoutSeconds`: Overall query timeout (default: 120)
  - `maxToolCallsPerTurn`: Limit tool calls per turn (default: 100)
  - `maxParallelTools`: Concurrency limit (default: 3)

- **FR-002**: System MUST persist AgentConfig per user across sessions

- **FR-003**: System MUST validate AgentConfig values with sensible bounds:
  - `maxIterations`: 1-50
  - `tokenBudget`: 1,000-200,000
  - `timeoutSeconds`: 10-600
  - `maxToolCallsPerTurn`: 1-20
  - `maxParallelTools`: 1-10

**Termination Conditions**

- **FR-004**: System MUST terminate agent loop when any of these conditions occur (in priority order):
  1. User cancellation
  2. Model returns finish_reason="stop" with no tool calls
  3. Max iterations reached
  4. Token budget exceeded
  5. Timeout exceeded
  6. No-progress detected (3 consecutive identical actions)
  7. Error limit reached (3 consecutive tool errors)

- **FR-005**: System MUST save accumulated main-chain content when any termination condition triggers

- **FR-006**: System MUST NOT save subagent work-in-progress when termination occurs

- **FR-007**: System MUST include termination reason in the final response when limits are hit

**Notifications**

- **FR-008**: System MUST emit a "system" message type in the SSE stream for notifications

- **FR-009**: System MUST inject a hint to the model when soft warning thresholds are crossed

- **FR-010**: System MUST display system messages in the chat UI with distinct styling

- **FR-011**: System messages MUST include:
  - Current value (e.g., "iteration 7")
  - Limit value (e.g., "of 10")
  - Contextual advice (e.g., "Consider wrapping up your response")

**DecisionTree Protocol**

- **FR-012**: System MUST define a DecisionTree interface with these operations:
  - `should_continue(state) -> (bool, reason)`: Determine if loop continues
  - `on_turn_start(state) -> state`: Hook before each turn
  - `on_tool_result(state, result) -> state`: Process tool results
  - `get_config() -> AgentConfig`: Return configuration for this tree

- **FR-013**: System MUST provide a DefaultDecisionTree that implements current behavior with improved termination

- **FR-014**: Skills MUST be able to declare custom decision trees via decorator-based registration

- **FR-015**: AgentState MUST be extensible for workflow modules without breaking core functionality

**Settings UI**

- **FR-016**: Settings page MUST display all AgentConfig fields with appropriate input controls

- **FR-017**: Settings page MUST show current values loaded from persistence

- **FR-018**: Settings page MUST validate input and display errors inline

- **FR-019**: Settings page MUST save changes immediately (no separate save button)

### Key Entities

- **AgentConfig**: User-configurable limits and thresholds for agent behavior. Persisted per-user. Contains iteration limits, token budgets, timeouts, and tool constraints.

- **AgentState**: Runtime state during query execution. Tracks current iteration, tokens used, elapsed time, recent actions (for no-progress detection), and termination reason. Extended by workflow modules.

- **DecisionTree**: Interface defining control flow behavior. Contains methods for continuation decisions, state hooks, and configuration access. Implemented by default tree and future skill trees.

- **SystemMessage**: Chat message type for system notifications. Contains severity level, message content, and metadata (limits, current values).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can configure all AgentConfig values in under 30 seconds via the Settings UI

- **SC-002**: 100% of queries respect configured limits without exceeding them

- **SC-003**: Users see system notifications at least 2 iterations before hard limits are hit (based on configured soft warning percentages)

- **SC-004**: Agent terminates within 5 seconds of any limit being reached

- **SC-005**: Zero data loss when limits trigger - all main-chain content is preserved and accessible

- **SC-006**: Default configuration reduces average unnecessary iterations by 30% compared to current 30-turn limit (measured by comparing iterations-to-completion)

- **SC-007**: No-progress detection correctly identifies stuck loops with 95% accuracy (false positive rate under 5%)

- **SC-008**: DecisionTree interface allows new workflow implementations without modifying core agent code

## Assumptions

1. **Token counting is approximate**: Token budget enforcement uses estimates; exact counts may vary by a few percent
2. **No-progress detection uses action equality**: Two actions are "identical" if they have the same tool name and stringified arguments
3. **Soft warnings are advisory**: The model may or may not act on injected hints; this is acceptable
4. **Settings auto-save is acceptable**: Users expect immediate persistence without explicit save action
5. **System messages are ephemeral**: They appear in the current session but are not persisted to conversation history
6. **Default values are sensible for most users**: Power users can adjust; most users should not need to change defaults

## Out of Scope

- Skill plugin system and registration (separate spec)
- DeepResearcher workflow integration (separate spec)
- Cost tracking or billing based on token usage
- Per-project AgentConfig overrides (user-level only for this spec)
- Retry logic for failed tools (already implemented; not changing)
- Model fallback chains (not in scope)
