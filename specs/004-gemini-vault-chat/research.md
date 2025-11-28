# Research: Gemini Vault Chat Agent

**Feature**: 004-gemini-vault-chat  
**Date**: 2025-11-28

## LlamaIndex Integration

### Decision: Use LlamaIndex Core with Google GenAI Extensions

**Rationale**: LlamaIndex provides a mature, well-documented framework for building RAG applications. The `llama-index-llms-google-genai` and `llama-index-embeddings-google-genai` packages provide first-class Gemini support without requiring custom integration code.

**Alternatives Considered**:
- **LangChain**: More complex, larger dependency footprint. LlamaIndex is more focused on document retrieval use cases.
- **Direct Gemini API**: Would require implementing chunking, embedding, and retrieval logic manually. Higher development effort.
- **OpenAI + pgvector**: Requires PostgreSQL, conflicts with SQLite-only approach in constitution.

### Key LlamaIndex Patterns

```python
# Singleton index pattern (recommended)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import load_index_from_storage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

_index: VectorStoreIndex | None = None

def get_or_build_index(vault_path: Path, persist_dir: Path) -> VectorStoreIndex:
    global _index
    if _index is not None:
        return _index
    
    if persist_dir.exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        _index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(str(vault_path), recursive=True).load_data()
        _index = VectorStoreIndex.from_documents(documents)
        _index.storage_context.persist(persist_dir=str(persist_dir))
    
    return _index
```

## Gemini Model Selection

### Decision: gemini-1.5-flash for LLM, text-embedding-004 for Embeddings

**Rationale**: 
- `gemini-1.5-flash` offers good balance of speed and quality for interactive chat
- `text-embedding-004` is Google's latest text embedding model with 768 dimensions
- Both are cost-effective for hackathon/demo scale

**Alternatives Considered**:
- `gemini-1.5-pro`: Higher quality but slower and more expensive
- `gemini-2.0-flash-exp`: Experimental, may not be stable

### Environment Variables

```
GOOGLE_API_KEY=<api-key>
VAULT_DIR=data/vaults/demo-user  # Or dynamically per user
LLAMAINDEX_PERSIST_DIR=data/llamaindex
```

## Source Attribution Strategy

### Decision: Extract source metadata from LlamaIndex response nodes

**Rationale**: LlamaIndex query responses include source nodes with file paths and text chunks. We can map these back to vault note paths and extract snippets for display.

```python
response = query_engine.query(question)
sources = []
for node in response.source_nodes:
    sources.append({
        "path": node.metadata.get("file_path"),
        "title": derive_title_from_path(node.metadata.get("file_path")),
        "snippet": node.text[:200] + "..." if len(node.text) > 200 else node.text,
        "score": node.score
    })
```

## Multi-Turn Conversation

### Decision: Use LlamaIndex ChatEngine for conversation memory

**Rationale**: LlamaIndex provides `as_chat_engine()` which wraps the index with conversation memory. This handles context naturally without custom implementation.

```python
chat_engine = index.as_chat_engine(
    chat_mode="context",
    llm=GoogleGenAI(model="gemini-1.5-flash")
)
response = chat_engine.chat("Follow-up question here")
```

**Note**: For MVP, we'll use a simpler approach where the frontend passes full message history and we construct context in the query. This avoids server-side session state.

## Agent Tools (Phase 2)

### Decision: Use LlamaIndex FunctionTool with constrained paths

**Rationale**: LlamaIndex supports registering Python functions as tools for agentic use. We can constrain write operations to an `agent-notes/` subdirectory.

```python
from llama_index.core.tools import FunctionTool

def create_note(title: str, content: str) -> str:
    """Create a new note in the agent folder."""
    safe_filename = slugify(title)
    path = f"agent-notes/{safe_filename}.md"
    vault_service.write_note(user_id, path, title=title, body=content)
    return f"Created note: {path}"

create_note_tool = FunctionTool.from_defaults(fn=create_note)
```

## Error Handling

### Decision: Graceful degradation with user-friendly messages

**Patterns**:
1. API key missing → 503 "AI service not configured"
2. API rate limit → 429 "Please wait and try again"
3. Network error → 503 "AI service temporarily unavailable"
4. Empty vault → 200 with message "No documents indexed"

## Performance Considerations

### Index Persistence

- First indexing: ~1-5 seconds for small vaults (<100 notes)
- Subsequent loads: ~100ms from persisted storage
- Query latency: ~1-3 seconds depending on Gemini API response time

### Recommendations

1. Load index on startup (not on first request)
2. Use environment variable to configure persist directory
3. For large vaults, consider lazy loading or background indexing (post-MVP)

## Dependencies to Add

```
# requirements.txt additions
llama-index
llama-index-llms-google-genai
llama-index-embeddings-google-genai
```

**Note**: These packages have their own dependencies (e.g., `google-generativeai`). Tested compatible with Python 3.11+.

