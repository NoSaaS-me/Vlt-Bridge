# Phase 0: Research & Design Decisions

**Feature**: AI Chat Window (`003-ai-chat-window`)

## 1. OpenRouter Integration
**Question**: What is the best way to integrate OpenRouter in Python?
**Finding**: OpenRouter is API-compatible with OpenAI. The standard `openai` Python client library is recommended, configured with `base_url="https://openrouter.ai/api/v1"` and the OpenRouter API key.
**Decision**: Use `openai` Python package.
**Rationale**: Industry standard, robust, async support.
**Alternatives**: `requests` (too manual), `langchain` (too heavy/complex for this specific need).

## 2. Real-time Streaming
**Question**: How to stream LLM tokens from FastAPI to React?
**Finding**: Server-Sent Events (SSE) is the standard for unidirectional text streaming. FastAPI supports this via `StreamingResponse`.
**Decision**: Use `StreamingResponse` with a generator that yields SSE-formatted data (`data: ...\n\n`).
**Rationale**: Simpler than WebSockets, works well through proxies/firewalls, native support in modern browsers (`EventSource` or `fetch` with readable streams).

## 3. Tool Execution Strategy
**Question**: How to invoke existing MCP tools (`list_notes`, `read_note`) from the chat endpoint?
**Finding**: The tools are defined as decorated functions in `backend/src/mcp/server.py`. We can import them directly. However, `FastMCP` wraps them. We might need to access the underlying function or just call the wrapper if it allows direct invocation.
**Decision**: Import the `mcp` object from `backend/src/mcp/server.py`. Use `mcp.list_tools()` to dynamically get tool definitions for the system prompt. Call the underlying functions directly if exposed, or use the `mcp.call_tool()` API if available. *Fallback*: Re-import the service functions (`vault_service.read_note`) directly if the MCP wrapper adds too much overhead/complexity for internal calls.
**Refinement**: The `server.py` defines tools using `@mcp.tool`. The most robust way is to import the `vault_service` and `indexer_service` instances directly from `server.py` (or a shared module) and wrap them in a simple "Agent Tool" registry for the LLM, mirroring the MCP definitions. This avoids "fake" network calls to localhost.

## 4. Frontend UI Components
**Question**: What UI library to use for the chat interface?
**Finding**: Project uses Tailwind + generic React.
**Decision**: Build a custom `ChatWindow` component using Tailwind. Use a scrollable container for messages and a sticky footer for the input.
**Rationale**: Lightweight, full control over styling.

## 5. Chat History Persistence
**Question**: How to store chat history?
**Finding**: Spec requires saving to Markdown files in the vault.
**Decision**:
1.  **In-Memory/Database**: Use a simple `sqlite` table (or just in-memory if stateless) to hold the *active* conversation state for the UI.
2.  **Persistence**: On "End Session" or auto-save (debounced), dump the conversation to `Chat Logs/{timestamp}-{title}.md`.
**Rationale**: Markdown is the source of truth. The database is just for the "hot" state to avoid parsing MD files on every new message.

## 6. System Prompts & Personas
**Question**: How to manage prompts?
**Decision**: Store prompts in a simple dictionary or JSON file in `backend/src/services/prompts.py`.
**Structure**:
```python
PERSONAS = {
    "default": "You are a helpful assistant...",
    "auto-linker": "You are an expert editor. Your goal is to densely connect notes...",
}
```

