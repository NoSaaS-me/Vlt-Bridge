# Quickstart: Gemini Vault Chat Agent

**Feature**: 004-gemini-vault-chat  
**Date**: 2025-11-28

## Prerequisites

1. Python 3.11+ installed
2. Node.js 18+ installed
3. Google API key with Gemini access

## Setup

### 1. Install Backend Dependencies

```bash
cd backend
pip install llama-index llama-index-llms-google-genai llama-index-embeddings-google-genai
```

### 2. Configure Environment

Add to your `.env` file (or export in terminal):

```bash
GOOGLE_API_KEY=your-gemini-api-key-here
VAULT_DIR=data/vaults/demo-user
LLAMAINDEX_PERSIST_DIR=data/llamaindex
```

### 3. Start Backend

```bash
cd backend
uvicorn src.api.main:app --reload --port 8000
```

The RAG index will be built on first startup (may take a few seconds).

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

## Verify Installation

### Check RAG Status

```bash
curl http://localhost:8000/api/rag/status
```

Expected response:
```json
{
  "status": "ready",
  "index_ready": true,
  "documents_indexed": 15
}
```

### Test RAG Chat

```bash
curl -X POST http://localhost:8000/api/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is this project about?"}]}'
```

Expected response:
```json
{
  "answer": "This project is Document-MCP, a...",
  "sources": [
    {
      "path": "Getting Started.md",
      "title": "Getting Started",
      "snippet": "Document-MCP provides...",
      "score": 0.89
    }
  ],
  "notes_written": []
}
```

## Development Workflow

### Backend Changes

1. Edit files in `backend/src/services/rag_index.py` or `backend/src/api/routes/rag.py`
2. Server auto-reloads with `--reload` flag
3. Run tests: `cd backend && pytest tests/unit/test_rag_service.py -v`

### Frontend Changes

1. Edit files in `frontend/src/components/` (ChatPanel, ChatMessage, SourceList)
2. Vite auto-reloads on save
3. Open browser at `http://localhost:5173`

### Rebuilding the Index

Delete the persist directory and restart:

```bash
rm -rf data/llamaindex
# Restart backend
```

## File Locations

| Component | Path |
|-----------|------|
| RAG Service | `backend/src/services/rag_index.py` |
| RAG Routes | `backend/src/api/routes/rag.py` |
| RAG Models | `backend/src/models/rag.py` |
| Chat Panel | `frontend/src/components/ChatPanel.tsx` |
| API Client | `frontend/src/services/rag.ts` |
| Types | `frontend/src/types/rag.ts` |
| Index Storage | `data/llamaindex/` |

## Troubleshooting

### "GOOGLE_API_KEY not set"

Ensure the environment variable is exported:
```bash
export GOOGLE_API_KEY=your-key-here
```

### "Index not ready"

Wait a few seconds after startup for indexing to complete. Check logs for errors.

### "Rate limited"

Gemini API has rate limits. Wait and retry, or check your API quota.

### Empty sources in response

Check that your vault has Markdown files. Run `ls data/vaults/demo-user/` to verify.

