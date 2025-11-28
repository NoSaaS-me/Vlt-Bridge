# Feature Specification: Gemini Vault Chat Agent

**Feature Branch**: `004-gemini-vault-chat`  
**Created**: 2025-11-28  
**Status**: Draft  
**Input**: User description: "Add a Gemini-powered planning chat agent using LlamaIndex for RAG over the Markdown vault. Use Gemini as both LLM and embedding model. Include a new chat panel in the HF Space frontend that calls a RAG backend endpoint, displays assistant responses with linked sources, and optionally allows the agent to write notes."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ask Questions About Vault Content (Priority: P1)

A user opens the Gemini Planning Agent panel and asks a question about content stored in their Markdown vault. The system searches the vault, retrieves relevant passages, and returns an AI-generated answer that synthesizes information from the relevant notes.

**Why this priority**: This is the core value proposition—enabling users to query their knowledge base conversationally and get AI-synthesized answers grounded in their own documents.

**Independent Test**: Can be fully tested by typing a question and verifying the response is relevant to vault content, with sources listed.

**Acceptance Scenarios**:

1. **Given** a vault containing notes about project architecture, **When** user asks "How does authentication work?", **Then** the system returns an answer citing relevant notes with snippets
2. **Given** a vault with multiple related notes, **When** user asks a question that spans multiple topics, **Then** the system synthesizes information from multiple sources and lists all referenced notes
3. **Given** a vault with no relevant content, **When** user asks an unrelated question, **Then** the system responds that no relevant information was found in the vault

---

### User Story 2 - View Source Notes (Priority: P1)

After receiving an answer from the chat agent, the user can see which notes were used to generate the response. They can click on a source to view the note in the existing document viewer or see an inline snippet.

**Why this priority**: Source attribution is essential for trust and verification. Users need to know where information comes from and validate AI responses against original content.

**Independent Test**: Can be tested by receiving an answer and clicking on a listed source to verify it opens the correct note.

**Acceptance Scenarios**:

1. **Given** an assistant response with sources, **When** user clicks a source link, **Then** the corresponding note opens in the document viewer
2. **Given** an assistant response with sources, **When** user expands a source, **Then** they see a snippet of the relevant passage
3. **Given** an assistant response, **When** sources are displayed, **Then** each source shows the note title and path

---

### User Story 3 - Multi-Turn Conversation (Priority: P2)

Users can have a multi-turn conversation with the agent, asking follow-up questions that build on previous context. The agent maintains conversation history for coherent responses.

**Why this priority**: Natural conversation flow improves user experience, but basic single-query functionality delivers core value first.

**Independent Test**: Can be tested by asking a question, then asking a follow-up that references "it" or "that", and verifying the agent understands the context.

**Acceptance Scenarios**:

1. **Given** a previous question about "authentication", **When** user asks "How do I configure it?", **Then** the agent understands "it" refers to authentication
2. **Given** an ongoing conversation, **When** user starts a new topic, **Then** the agent responds appropriately to the new context
3. **Given** a conversation session, **When** user refreshes the page, **Then** conversation history is cleared (new session starts)

---

### User Story 4 - Agent Creates Notes (Priority: P3)

Users can instruct the agent to create new notes based on the conversation. The agent writes notes to a dedicated folder in the vault and informs the user what was created.

**Why this priority**: Note creation adds significant value but requires more complex safety controls. Core reading/query functionality should be solid first.

**Independent Test**: Can be tested by asking the agent to "create a summary note about X" and verifying a new note appears in the designated folder.

**Acceptance Scenarios**:

1. **Given** a conversation about a topic, **When** user asks "create a summary note", **Then** the agent creates a new Markdown note in the agent folder
2. **Given** an agent-created note, **When** user views the response, **Then** a badge or link shows the created note path
3. **Given** an existing note, **When** user asks the agent to append content, **Then** the agent updates the existing note appropriately

---

### Edge Cases

- What happens when the vault is empty or has no indexed content? → System returns a friendly message indicating no documents are available
- How does the system handle very long user queries? → Query is truncated to reasonable limits with user notification
- What happens if the AI service is unavailable? → System shows an error message and suggests retrying
- How are malformed or non-Markdown files handled? → Non-Markdown files are ignored during indexing
- What if the agent tries to write outside the designated folder? → Write operations are constrained to the agent folder only

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a chat interface for users to ask natural language questions about vault content
- **FR-002**: System MUST search the vault and retrieve relevant passages to answer user queries
- **FR-003**: System MUST generate AI responses that synthesize information from retrieved content
- **FR-004**: System MUST display source notes for each response, including note title and path
- **FR-005**: System MUST allow users to navigate from a source reference to the full note
- **FR-006**: System MUST maintain conversation history within a session for multi-turn dialogue
- **FR-007**: System MUST build and persist a searchable index of vault content
- **FR-008**: System MUST load an existing index on startup if available
- **FR-009**: System MUST constrain agent write operations to a designated agent folder only
- **FR-010**: System MUST display a notification when the agent creates or updates a note
- **FR-011**: System MUST show an appropriate error message if the AI service is unavailable

### Key Entities

- **Chat Message**: Represents a single message in the conversation (role: user or assistant, content, timestamp)
- **Chat Session**: A collection of messages in a single conversation context (started when user opens panel, cleared on page refresh)
- **Source Reference**: Metadata about a note used to generate a response (note title, path, relevant snippet)
- **Agent Note**: A Markdown note created by the agent, stored in the designated agent folder

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users receive a relevant answer with sources within 5 seconds of submitting a query
- **SC-002**: 90% of responses include at least one source reference when relevant content exists
- **SC-003**: Users can navigate from a source reference to the full note in one click
- **SC-004**: Multi-turn conversations correctly reference previous context in 80% of follow-up questions
- **SC-005**: Agent-created notes appear in the designated folder and are visible in the vault viewer within 2 seconds
- **SC-006**: System gracefully handles AI service unavailability with a clear error message

## Assumptions

- Users have a Markdown vault with content they want to query
- The existing document viewer from the Docs Widget can be reused for viewing source notes
- Index rebuilds are acceptable on service restarts for the initial release
- Session history is ephemeral and not persisted across page refreshes
- Agent write operations are limited to creating and appending to notes (no deletion)
