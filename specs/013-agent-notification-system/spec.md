# Feature Specification: Agent Notification System (ANS)

**Feature Branch**: `013-agent-notification-system`
**Created**: 2026-01-03
**Status**: Draft
**Input**: Agent Notification System with TOON format, event bus, file-based subscribers, and system role in chat UI

## Overview

The Agent Notification System (ANS) provides a modular, event-driven mechanism for injecting messages into the agent's conversation context. It enables the system to communicate important events (tool failures, budget warnings, loop detection, external triggers) to the agent in a token-efficient format called TOON (Token-Oriented Object Notation). This feature also introduces a "system" role in the chat UI alongside existing "user" and "agent" roles, enabling clear visual distinction of notification sources and laying groundwork for multi-participant collaboration.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Agent Receives Tool Failure Notifications (Priority: P1)

When a tool call fails during agent execution, the agent should receive a notification in its context so it can understand what went wrong and take corrective action without making blind retry attempts.

**Why this priority**: Tool failures are the most common operational event. Without notifications, agents retry blindly or halt without context. This directly impacts agent effectiveness and user experience.

**Independent Test**: Can be tested by triggering a tool timeout/failure during agent execution and verifying the notification appears in the agent's next context window and is visible in the chat UI as a system message.

**Acceptance Scenarios**:

1. **Given** the agent calls a tool that times out, **When** the timeout occurs, **Then** the agent receives a notification with the tool name, error type, and timestamp in its next context.
2. **Given** multiple tool failures occur within 5 seconds, **When** the agent's context is updated, **Then** the failures are batched into a single notification with all details preserved.
3. **Given** the same tool fails repeatedly, **When** 3+ identical failures occur within 5 seconds, **Then** they are deduplicated into a count-based notification (e.g., "vault_search failed 3 times").

---

### User Story 2 - System Messages Appear in Chat UI (Priority: P1)

The chat interface displays notifications as distinct "system" messages, visually differentiated from user and agent messages. Users can see what notifications the agent received.

**Why this priority**: Users need visibility into system events to understand agent behavior. Without this, the agent appears to act on invisible information.

**Independent Test**: Can be tested by viewing the chat panel after a system event occurs, verifying the system message appears with distinct styling and the source is identifiable.

**Acceptance Scenarios**:

1. **Given** a notification is injected into the conversation, **When** the user views the chat panel, **Then** the message appears with "System" attribution and visually distinct styling.
2. **Given** multiple notifications exist in a conversation, **When** the user scrolls through history, **Then** system messages are clearly distinguishable from user and agent messages.
3. **Given** a notification contains structured TOON data, **When** rendered in the UI, **Then** it displays in a human-readable format (not raw TOON syntax).

---

### User Story 3 - Budget Warning Notifications (Priority: P2)

When the agent approaches resource limits (token budget, iteration count, timeout), it receives early warning notifications allowing it to wrap up gracefully rather than being abruptly terminated.

**Why this priority**: Graceful degradation improves agent output quality. Without warnings, agents may be cut off mid-thought, losing work.

**Independent Test**: Can be tested by configuring a low token budget, running the agent until it hits 80% consumption, and verifying the warning notification appears.

**Acceptance Scenarios**:

1. **Given** the agent has consumed 80% of its token budget, **When** the next context update occurs, **Then** the agent receives a warning notification with remaining budget and percentage.
2. **Given** the agent has used 70% of its configured iteration limit, **When** this threshold is crossed, **Then** the agent receives a notification with iterations remaining.
3. **Given** multiple budget warnings would fire simultaneously, **When** the agent's context is updated, **Then** they are batched into a single comprehensive status notification.

---

### User Story 4 - Loop Detection Notifications (Priority: P2)

When the system detects the agent is stuck in a repetitive pattern, it notifies the agent so it can break the loop or request user intervention.

**Why this priority**: Loop detection prevents wasted compute and user frustration. The agent needs to know it's looping to take corrective action.

**Independent Test**: Can be tested by crafting a prompt that causes repetitive behavior, verifying the loop detection notification appears after the configured threshold.

**Acceptance Scenarios**:

1. **Given** the agent repeats the same action pattern 3+ times, **When** loop detection triggers, **Then** the agent receives a notification describing the detected pattern.
2. **Given** a loop is detected, **When** the notification is displayed in chat UI, **Then** the user can see the system identified problematic behavior.

---

### User Story 5 - Configurable Subscriber Management (Priority: P3)

Users can enable/disable specific notification types through a settings interface. Some core notifications remain always-on for system stability.

**Why this priority**: Power users need control over notification verbosity. However, this is lower priority than the notifications themselves working.

**Independent Test**: Can be tested by toggling a subscriber off in settings, triggering the associated event, and verifying no notification is generated.

**Acceptance Scenarios**:

1. **Given** a user opens the Settings page, **When** they navigate to the Notifications tab, **Then** they see a list of available subscribers with toggle controls.
2. **Given** a user disables the "tool success" subscriber, **When** a tool succeeds, **Then** no notification is generated for that event.
3. **Given** a core subscriber (like budget_exceeded), **When** the user attempts to disable it, **Then** it remains enabled with an explanation of why it cannot be disabled.

---

### User Story 6 - File-Based Subscriber Discovery (Priority: P3)

Administrators can add new notification subscribers by adding configuration files to a designated directory. The system discovers and loads these at startup.

**Why this priority**: Extensibility is important for long-term maintainability but not critical for initial launch.

**Independent Test**: Can be tested by adding a new subscriber config file, restarting the service, and verifying the subscriber appears in the list and processes events.

**Acceptance Scenarios**:

1. **Given** a new subscriber config file is added to the subscribers directory, **When** the system starts, **Then** the subscriber is loaded and active.
2. **Given** an invalid subscriber config file exists, **When** the system starts, **Then** it logs an error and continues loading other valid subscribers.
3. **Given** a subscriber is removed from the directory, **When** the system restarts, **Then** that subscriber no longer processes events.

---

### Edge Cases

- What happens when the event bus receives events faster than they can be processed?
  - Events are queued; if queue exceeds limit, oldest low-priority events are dropped with a summary notification.
- What happens when a subscriber template fails to render?
  - Fallback to a generic error notification; log the template error for debugging.
- What happens when notifications would exceed context limits?
  - Older notifications are summarized (e.g., "12 tool success events" instead of 12 individual notifications).
- What happens during conversation replay/history viewing?
  - Persisted system messages display as they occurred; no re-triggering of notifications.
- What happens if TOON parsing fails?
  - Display raw content with a parsing error indicator; never lose the notification.

## Requirements *(mandatory)*

### Functional Requirements

#### Event System

- **FR-001**: System MUST emit events for tool call outcomes (pending, success, failure, timeout).
- **FR-002**: System MUST emit events for budget threshold crossings (token, iteration, timeout at configurable percentages).
- **FR-003**: System MUST emit events when loop detection triggers.
- **FR-004**: System MUST emit events when agent turn starts and ends.
- **FR-005**: Events MUST include: unique ID, event type, timestamp, source identifier, severity level, and event-specific payload.

#### Subscriber System

- **FR-006**: System MUST discover subscribers from configuration files in a designated directory at startup.
- **FR-007**: Subscribers MUST specify which event types they listen to.
- **FR-008**: Subscribers MUST define filtering conditions (e.g., minimum severity level).
- **FR-009**: Subscribers MUST define a template for formatting events into notifications.
- **FR-010**: Subscribers MAY specify batching rules (time window, max batch size).
- **FR-011**: Subscribers MAY specify deduplication rules (dedupe key, time window).
- **FR-012**: System MUST support marking subscribers as "core" (cannot be disabled by users).

#### Notification Generation

- **FR-013**: System MUST use TOON format for notifications to minimize token usage.
- **FR-014**: System MUST support Jinja2-style templates for generating TOON output.
- **FR-015**: System MUST batch multiple events of the same type into tabular TOON format.
- **FR-016**: System MUST deduplicate identical events within the configured window.
- **FR-017**: System MUST prioritize notifications (critical, high, normal, low) with critical delivered immediately.
- **FR-018**: System MUST inject notifications at defined points: turn start, after tool results, turn end.

#### Persistence

- **FR-019**: Notifications MUST persist in the conversation context alongside user and agent messages.
- **FR-020**: System messages MUST be retrievable when loading conversation history.
- **FR-021**: System MUST maintain FIFO ordering for notifications within the same priority level.

#### Chat UI

- **FR-022**: Chat interface MUST display three distinct message roles: user, agent, and system.
- **FR-023**: System messages MUST be visually distinguished from user and agent messages.
- **FR-024**: System messages MUST display the notification source (e.g., "Tool Executor", "Budget Monitor").
- **FR-025**: TOON-formatted content MUST be rendered in human-readable form in the UI.
- **FR-026**: Users MUST be able to collapse/expand verbose system notifications.

#### Settings UI

- **FR-027**: Settings page MUST include a Notifications tab (tabbed settings interface).
- **FR-028**: Notifications tab MUST list all loaded subscribers with enable/disable toggles.
- **FR-029**: Settings MUST display subscriber name, description, subscribed event types, and current status.
- **FR-030**: Core subscribers MUST show as always-enabled with explanatory tooltip.
- **FR-031**: Settings changes MUST take effect immediately without requiring restart.

### Key Entities

- **Event**: An occurrence in the system with type, timestamp, source, severity, and payload. Events are the input to the notification system.
- **Subscriber**: A configuration that listens for specific event types, filters them, and transforms matching events into notifications using a template.
- **Notification**: A formatted message (in TOON) ready to be injected into the agent's conversation context. Has priority and injection point.
- **System Message**: A notification persisted in the conversation, displayed in the chat UI as coming from "system" role.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of tool failures result in a notification visible to the agent within its next context window.
- **SC-002**: Notifications consume 40% fewer tokens than equivalent JSON-formatted messages (leveraging TOON efficiency).
- **SC-003**: Users can identify system messages at a glance with 100% accuracy (distinct visual styling).
- **SC-004**: System handles 100 events per second without notification loss or significant delay (< 100ms processing time).
- **SC-005**: Users can enable/disable non-core subscribers within 3 clicks from any page.
- **SC-006**: Agent receives budget warnings with at least 20% remaining capacity, giving time to wrap up gracefully.
- **SC-007**: Adding a new subscriber requires only adding a config file and restarting (no code changes).
- **SC-008**: Conversation history correctly displays all historical system messages when loaded.

## Assumptions

- The TOON reference implementation (Python) is suitable for production use.
- Jinja2 templating provides sufficient flexibility for notification formatting.
- The existing conversation persistence mechanism can accommodate a new "system" role without major refactoring.
- File-based subscriber discovery is preferred over database storage for auditability and version control.
- Core subscribers (tool_failure, budget_warning, budget_exceeded, loop_detected) cannot be disabled to ensure system stability.
- Debounce windows of 1-5 seconds are appropriate defaults for most event types.
- Batch windows aligned with injection points (turn_start, after_tool, turn_end) provide good UX without excessive fragmentation.

## Out of Scope

- Agent tools for dynamically creating/modifying subscribers (future: requires Python sandbox)
- CLI event injection from external processes (future: vlt CLI integration)
- Async subagent completion notifications (future: requires subagent framework)
- Cross-session notification history/audit log (future: dedicated notifications table)
- Human-to-human collaboration via the system role (future: separate feature)
- Browser push notifications for system events (future: PWA enhancement)
