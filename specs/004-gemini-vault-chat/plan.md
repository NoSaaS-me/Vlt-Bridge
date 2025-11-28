# Implementation Plan: Gemini Vault Chat Agent

**Branch**: `004-gemini-vault-chat` | **Date**: 2025-11-28 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/004-gemini-vault-chat/spec.md`

## Summary

Add a Gemini-powered RAG chat agent to the Document-MCP platform. Users can ask natural language questions about their Markdown vault and receive AI-synthesized answers grounded in their documents. The system uses LlamaIndex for document indexing and retrieval, with Gemini as both the LLM and embedding model. An optional Phase 2 adds constrained note-writing capabilities.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript (frontend)  
**Primary Dependencies**: FastAPI, LlamaIndex, llama-index-llms-google-genai, llama-index-embeddings-google-genai, React 18+, Tailwind CSS, Shadcn/UI  
**Storage**: Filesystem vault (existing), LlamaIndex persisted vector store (new, under `data/llamaindex/`)  
**Testing**: pytest (backend), manual verification (frontend)  
**Target Platform**: Hugging Face Spaces (Docker), Linux server  
**Project Type**: Web application (frontend + backend)  
**Performance Goals**: <5 seconds for RAG response (per SC-001)  
**Constraints**: Must not break existing MCP server or ChatGPT widget  
**Scale/Scope**: Hackathon scale—index rebuilds acceptable on restart

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Brownfield Integration | ✅ Pass | Uses existing VaultService, adds new routes/services alongside existing code |
| II. Test-Backed Development | ✅ Pass | Plan includes pytest tests for RAG service; frontend is manual verification |
| III. Incremental Delivery | ✅ Pass | P1 stories (read-only RAG) can ship before P3 (write tools) |
| IV. Specification-Driven | ✅ Pass | All work traced to spec.md; Phase 2 is optional per spec |
| No Magic | ✅ Pass | Direct LlamaIndex usage, no custom abstractions |
| Single Source of Truth | ✅ Pass | Vault remains source of truth; index is derived view |
| Error Handling | ✅ Pass | Spec requires FR-011 error messages for AI unavailability |

**Technology Stack Compliance**:
- Backend: Python 3.11+, FastAPI, Pydantic ✅
- Frontend: React 18+, TypeScript, Tailwind, Shadcn/UI ✅
- Storage: Filesystem-based (LlamaIndex persisted store) ✅

## Project Structure

### Documentation (this feature)

```text
specs/004-gemini-vault-chat/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── rag-api.yaml     # OpenAPI spec for RAG endpoints
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── api/
│   │   └── routes/
│   │       └── rag.py           # NEW: RAG chat endpoint
│   ├── models/
│   │   └── rag.py               # NEW: Pydantic models for RAG
│   └── services/
│       └── rag_index.py         # NEW: LlamaIndex service
└── tests/
    └── unit/
        └── test_rag_service.py  # NEW: RAG service tests

frontend/
├── src/
│   ├── components/
│   │   ├── ChatPanel.tsx        # NEW: Chat interface
│   │   ├── ChatMessage.tsx      # NEW: Message component
│   │   └── SourceList.tsx       # NEW: Source references
│   ├── services/
│   │   └── rag.ts               # NEW: RAG API client
│   └── types/
│       └── rag.ts               # NEW: TypeScript types

data/
└── llamaindex/                  # NEW: Persisted vector index
```

**Structure Decision**: Web application structure (Option 2). New files added alongside existing code per Constitution Principle I.

## Complexity Tracking

> No violations requiring justification.

## Implementation Phases

### Phase 1: Core RAG Query (P1 Stories)

Implements User Stories 1-2: Ask questions, view sources.

**Backend Tasks**:
1. Add LlamaIndex dependencies to `requirements.txt`
2. Create `rag_index.py` service with `get_or_build_index()` singleton
3. Create `rag.py` Pydantic models for request/response
4. Create `rag.py` route with `POST /api/rag/chat` endpoint
5. Add unit tests for RAG service

**Frontend Tasks**:
1. Create `ChatPanel.tsx` component with message list and composer
2. Create `ChatMessage.tsx` for rendering user/assistant messages
3. Create `SourceList.tsx` for collapsible source references
4. Add `rag.ts` API client service
5. Integrate ChatPanel into MainApp layout

### Phase 2: Multi-Turn Conversation (P2 Story)

Implements User Story 3: Context-aware follow-ups.

**Tasks**:
1. Maintain chat history in frontend state
2. Pass full message history to backend
3. Update RAG service to use chat history for context

### Phase 3: Agent Note Writing (P3 Story, Optional)

Implements User Story 4: Create/append notes via agent.

**Tasks**:
1. Create constrained write helpers (`create_note`, `append_to_note`)
2. Register as LlamaIndex agent tools
3. Add `notes_written` to response model
4. Show created notes badge in UI
