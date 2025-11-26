# Feature Specification: AI Chat Window

**Feature Branch**: `003-ai-chat-window`  
**Created**: 2025-11-26  
**Status**: Draft  
**Input**: User description: "Add an AI Chat Window using OpenRouter as the LLM provider. The system should reuse existing MCP tools (backend agent) to manage the vault. Include a 'Persona/Mode' selector to allow users to choose specialized system prompts for tasks like reindexing, cross-linking, and summarization. Chat history should be persisted to the vault."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - General Q&A with Vault Context (Priority: P1)

As a user, I want to ask questions about my notes so that I can quickly find information or synthesize concepts without manually searching.

**Why this priority**: This is the core value proposition—enabling natural language interaction with the knowledge base.

**Independent Test**: Can be tested by asking a question about a known note and verifying the answer cites the correct information.

**Acceptance Scenarios**:

1. **Given** the chat window is open, **When** I ask "What is the summary of project X?", **Then** the agent searches the vault and returns a summary based on the note content.
2. **Given** a specific note is open, **When** I ask "Summarize this", **Then** the agent reads the current note context and provides a summary.

---

### User Story 2 - Vault Management via Personas (Priority: P2)

As a power user, I want to select specialized "Personas" (e.g., Auto-Linker, Tag Gardener) so that I can perform complex maintenance tasks with optimized prompts.

**Why this priority**: Distinguishes this from a simple "chatbot" by adding workflow automation capabilities.

**Independent Test**: Select a persona, give a relevant command, and verify the specific tool (write/update) is called.

**Acceptance Scenarios**:

1. **Given** the "Auto-Linker" persona is selected, **When** I ask "Fix links in Note A", **Then** the agent identifies unlinked concepts and updates the note with `[[WikiLinks]]`.
2. **Given** the "Tag Gardener" persona is selected, **When** I ask "Clean up tags", **Then** the agent identifies synonymous tags and standardizes them across affected notes.

---

### User Story 3 - Chat History Persistence (Priority: P3)

As a user, I want my chat conversations to be saved in the vault so that I can reference past insights or continue working later.

**Why this priority**: Ensures work isn't lost and integrates chat logs as first-class citizens in the vault.

**Independent Test**: Refresh the browser and verify the previous conversation is still visible.

**Acceptance Scenarios**:

1. **Given** I have had a conversation, **When** I refresh the page, **Then** the chat history is restored.
2. **Given** a conversation is finished, **When** I look in the vault file explorer, **Then** I see a new Markdown file (e.g., in `Chat Logs/`) containing the transcript.

---

### Edge Cases

- **Network Failure**: What happens if OpenRouter is down? (System should show error and allow retry).
- **Large Context**: What happens if the vault search returns too much text? (Agent should truncate or summarize input).
- **Invalid Tool Use**: What happens if the agent tries to write a file with invalid characters? (System should catch error and ask agent to retry).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a UI interface for Chat (floating or sidebar) that persists across navigation.
- **FR-002**: System MUST allow users to configure an OpenRouter API Key (via env vars or UI settings).
- **FR-003**: System MUST expose existing internal MCP tools (`read_note`, `write_note`, `search_notes`, etc.) to the LLM.
- **FR-004**: System MUST support selecting "Personas" that inject specific system prompts (Auto-Linker, Tag Gardener, etc.) into the context.
- **FR-005**: Chat sessions MUST be automatically saved to the vault as Markdown files (e.g., in a `Chat Logs` folder).
- **FR-006**: System MUST stream LLM responses to the UI for real-time feedback.
- **FR-007**: System MUST support creating new chat sessions and switching between past sessions.

### Key Entities

- **Chat Session**: Represents a conversation thread. Properties: ID, Title, Created Date, Messages (User/Assistant/Tool).
- **Persona**: A preset configuration of System Prompt + available Tools.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Agent responses start streaming within 3 seconds of user input.
- **SC-002**: 95% of "Auto-Linker" requests result in valid WikiLinks being added without syntax errors.
- **SC-003**: Users can switch between active chat and past history (refresh/reload) with zero data loss.
- **SC-004**: System can handle a context window of at least 16k tokens (supporting moderate-sized note analysis).