# Implementation Plan: AI Chat Window

**Branch**: `003-ai-chat-window` | **Date**: 2025-11-26 | **Spec**: [specs/003-ai-chat-window/spec.md](spec.md)
**Input**: Feature specification from `/specs/003-ai-chat-window/spec.md`

## Summary

Implement an integrated AI Chat Window powered by OpenRouter. This involves a new backend `POST /api/chat` endpoint that uses the `openai` client to communicate with LLMs, exposing internal `VaultService` methods as tools. The frontend will receive a new `ChatWindow` component with streaming support (SSE) and persona selection. Chat history will be persisted as Markdown files in the vault.

## Technical Context

**Language/Version**: Python 3.11+ (Backend), TypeScript/React 18 (Frontend)
**Primary Dependencies**:
- Backend: `openai` (for OpenRouter), `fastapi` (StreamingResponse)
- Frontend: `fetch` (Streaming body reading), Tailwind CSS
**Storage**:
- Active Session: In-memory (or transient SQLite)
- Persistence: Markdown files in `Chat Logs/` folder
**Testing**: `pytest` (Backend), Manual/E2E (Frontend)
**Target Platform**: Web Application (Linux Dev Environment)
**Project Type**: Full-stack (FastAPI + React)
**Performance Goals**: <3s time-to-first-token
**Constraints**: Must reuse existing `VaultService` logic; no new database services (keep it lightweight).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Brownfield Integration**: Reuses `VaultService` and `IndexerService`. Matches `backend/src` and `frontend/src` structure.
- [x] **Test-Backed Development**: Backend logic will be unit tested.
- [x] **Incremental Delivery**: New API route and independent UI component.
- [x] **Specification-Driven**: All features map to `spec.md` requirements.

## Project Structure

### Documentation (this feature)

```text
specs/003-ai-chat-window/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── api/
│   │   └── routes/
│   │       └── chat.py       # NEW: Chat endpoint logic
│   ├── services/
│   │   ├── chat.py           # NEW: Chat orchestration service (OpenAI wrapper)
│   │   └── prompts.py        # NEW: System prompts/personas definitions
│   └── models/
│       └── chat.py           # NEW: Pydantic models for Chat requests/responses
└── tests/
    └── unit/
        └── test_chat_service.py # NEW: Tests for chat logic

frontend/
├── src/
│   ├── components/
│   │   ├── chat/             # NEW: Chat UI Components
│   │   │   ├── ChatWindow.tsx
│   │   │   ├── ChatMessage.tsx
│   │   │   └── PersonaSelector.tsx
│   └── services/
│       └── api.ts            # UPDATE: Add chat endpoints
└── tests/
```

**Structure Decision**: Standard Full-stack separation. Backend adds a dedicated `chat` service and route to isolate LLM logic from core data services. Frontend adds a self-contained `chat/` directory for UI components.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A       |            |                                     |