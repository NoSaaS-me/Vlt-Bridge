## Gemini LlamaIndex Vault Chat Agent Spec

Feature spec: Gemini + LlamaIndex Vault Chat Agent (HF Space)

    Objective

    Add a second planning chat interface to the Hugging Face Space.
    Use LlamaIndex for retrieval-augmented generation over the same Markdown vault used by Document-MCP.
    Use Gemini (via LlamaIndex) as both the LLM and embedding model.
    Optionally allow the agent to write new notes into the vault via constrained tools.

Non-goals

    Do not change the existing MCP server or ChatGPT App widget behavior.
    Do not introduce a new external database; rely on LlamaIndex storage or simple filesystem persistence for the hackathon.

    High-Level Architecture

    Vault: directory of Markdown notes already used by Document-MCP.
    Indexer: Python module using LlamaIndex to scan the vault, split notes into chunks, and build a VectorStoreIndex backed by a simple vector store.
    Chat backend: FastAPI endpoints that load the index, run RAG queries with Gemini, and return answers plus source notes.
    HF Space frontend: a new chat panel that calls the backend, shows the assistant response, and lists linked sources (note titles and paths).

    Backend details

    Dependencies: llama-index core, llama-index-llms-google-genai, llama-index-embeddings-google-genai.
    Env vars: GOOGLE_API_KEY, VAULT_DIR, LLAMAINDEX_PERSIST_DIR.
    On startup: if a persisted index exists, load it; otherwise, scan VAULT_DIR for markdown files, build a new index, and persist it under LLAMAINDEX_PERSIST_DIR.
    Provide a helper get_or_build_index that returns a singleton VectorStoreIndex.
    Implement a function rag_chat(messages) that:
        Takes a simple chat history array.
        Uses index.as_query_engine with Gemini as the LLM.
        Runs a query on the latest user message.
        Returns a dict with fields: answer (string), sources (list of title, path, snippet), notes_written (empty list for now).
    Expose POST /api/rag/chat in FastAPI that wraps rag_chat.

    Frontend details

    Add a new panel or tab labeled Gemini Planning Agent.
    Layout: left side may keep the existing docs UI; right side is a chat view.
    Chat view: list of messages and a composer textarea with a Send button.
    On send: push the user message into local history, POST to /api/rag/chat, then append the assistant answer and its sources when the response arrives.
    Under each assistant message, show a collapsible Sources section; clicking a source should either open the note in the existing viewer or show the snippet inline.

    Index refresh strategy

    On every backend startup, attempt to load an existing index; rebuild if missing or invalid.
    For hackathon scale, it is acceptable that index updates require a restart or redeploy.

    Phase 2 (optional write tools)

    Implement safe note-writing helpers (create_note, append_to_note, tag_note) that operate only in a dedicated agent folder inside the vault.
    Register these as tools for a LlamaIndex-based agent using Gemini as the reasoning model.
    Extend /api/rag/chat so that responses can include notes_written metadata when the agent creates or updates notes.
    In the UI, show a small badge when a new note is created, with a link into the vault viewer.

    Implementation order

    Wire dependencies and environment variables.
    Implement get_or_build_index and verify indexing works.
    Implement rag_chat and the /api/rag/chat endpoint.
    Build the frontend chat UI and hook it up to the endpoint.
    If time allows, add Phase 2 tools and surface created notes in the UI.

