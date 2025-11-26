# Tasks: AI Chat Window

**Feature Branch**: `003-ai-chat-window`
**Spec**: [specs/003-ai-chat-window/spec.md](spec.md)
**Plan**: [specs/003-ai-chat-window/plan.md](plan.md)

## Phase 1: Setup
*Goal: Initialize project structure and install dependencies.*

- [ ] T001 Create contracts directory and API spec at specs/003-ai-chat-window/contracts/chat-api.yaml
- [ ] T002 [P] Create directory backend/src/services/chat
- [ ] T003 [P] Create directory frontend/src/components/chat
- [ ] T004 Add openai dependency to backend/requirements.txt

## Phase 2: Foundational
*Goal: Core backend logic for Chat (Service Layer).*

- [ ] T005 [US1] Define ChatMessage and ChatRequest models in backend/src/models/chat.py
- [ ] T006 [US1] Define Persona and Prompt models in backend/src/models/chat.py
- [ ] T007 [US2] Implement prompt storage (dictionary of personas) in backend/src/services/prompts.py
- [ ] T008 [US1] Create ChatService class skeleton in backend/src/services/chat.py

## Phase 3: User Story 1 - General Q&A with Vault Context
*Goal: Enable basic chat interactions with streaming and tool use.*
*Test Criteria: Can ask a question and get a streaming response citing vault notes.*

- [ ] T009 [US1] Implement OpenAI client initialization in backend/src/services/chat.py
- [ ] T010 [US1] Implement tool registry (wrap VaultService/IndexerService) in backend/src/services/chat.py
- [ ] T011 [US1] Implement stream_chat method with SSE generator in backend/src/services/chat.py
- [ ] T012 [US1] Create unit tests for ChatService in backend/tests/unit/test_chat_service.py
- [ ] T013 [US1] Implement POST /api/chat endpoint in backend/src/api/routes/chat.py
- [ ] T014 [US1] Register chat router in backend/src/api/main.py
- [ ] T015 [P] [US1] Create ChatMessage component in frontend/src/components/chat/ChatMessage.tsx
- [ ] T016 [US1] Create ChatWindow component skeleton in frontend/src/components/chat/ChatWindow.tsx
- [ ] T017 [US1] Implement streaming fetch logic in frontend/src/services/api.ts
- [ ] T018 [US1] Connect ChatWindow to API and handle SSE stream in frontend/src/components/chat/ChatWindow.tsx

## Phase 4: User Story 2 - Vault Management via Personas
*Goal: Allow users to select specialized agents for maintenance tasks.*
*Test Criteria: Selecting "Auto-Linker" injects the correct system prompt and tools.*

- [ ] T019 [US2] Add GET /api/chat/personas endpoint to backend/src/api/routes/chat.py
- [ ] T020 [P] [US2] Create PersonaSelector component in frontend/src/components/chat/PersonaSelector.tsx
- [ ] T021 [US2] Add persona selection state to frontend/src/components/chat/ChatWindow.tsx
- [ ] T022 [US2] Update ChatService to accept and apply persona ID in backend/src/services/chat.py

## Phase 5: User Story 3 - Chat History Persistence
*Goal: Save conversation logs to the vault.*
*Test Criteria: Chat logs appear as Markdown files in "Chat Logs/" folder.*

- [ ] T023 [US3] Implement save_chat_log method in backend/src/services/chat.py (Markdown formatting)
- [ ] T024 [US3] Update POST /api/chat to auto-save on completion (or session end) in backend/src/api/routes/chat.py
- [ ] T025 [US3] Add logic to restore history from ChatRequest.history in backend/src/services/chat.py
- [ ] T026 [US3] Add "Clear History" or "New Chat" button in frontend/src/components/chat/ChatWindow.tsx

## Phase 6: Polish & Cross-Cutting Concerns
*Goal: Final UI touches and error handling.*

- [ ] T027 [P] Style ChatWindow with Tailwind (responsive sidebar/floating) in frontend/src/components/chat/ChatWindow.tsx
- [ ] T028 Implement error handling for OpenRouter failures in backend/src/services/chat.py
- [ ] T029 Add tool execution status messages to UI stream in frontend/src/components/chat/ChatMessage.tsx

## Dependencies

- **US1** depends on Setup & Foundational tasks.
- **US2** extends US1 (can be parallelized after US1 backend is stable).
- **US3** extends US1 backend logic.

## Implementation Strategy

1.  **MVP (US1)**: Get the chat bubble working with a hardcoded "Hello World" stream, then hook up OpenRouter.
2.  **Tools**: Enable `read_note` and `search_notes` so the agent isn't blind.
3.  **Personas (US2)**: Add the dropdown and the specialized prompts.
4.  **Persistence (US3)**: Add the file writing logic last.
