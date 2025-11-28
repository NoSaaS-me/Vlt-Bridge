# Phase 0: Research & Technical Decisions

## 1. FastMCP Metadata Injection

**Decision**: Return `CallToolResult` with `_meta` for UI tools.

**Strategy**:
-   We will stop returning pure Pydantic models from tools that need to trigger widgets (`read_note`, `search_notes`).
-   Instead, these tools will instantiate the Pydantic model, dump it to a dictionary, and wrap it in a `CallToolResult` object.
-   The `_meta` field will contain `openai: { outputTemplate: "..." }`.
-   Non-UI tools (e.g., `list_notes`, `delete_note`) will continue to return Pydantic models or simple text to keep them lightweight.

**Rationale**: This aligns with the OpenAI Apps SDK pattern and allows us to trigger widgets without breaking the existing schema validation (since `structuredContent` will still match the Pydantic schema).

## 2. React Widget Strategy

**Decision**: Use a separate Vite entry point (`widget.html` + `widget.tsx`).

**Strategy**:
-   Create `frontend/widget.html` as a lightweight entry point.
-   Create `frontend/src/widget.tsx` to render the widget application.
-   Refactor `NoteViewer.tsx` into a "pure" component (if it isn't already) that can be imported by both `App.tsx` and `widget.tsx`.
-   Use `vite-plugin-html` or manual rollup config to output multiple HTML files.

**Rationale**:
-   **Isolation**: Prevents the main app's router, sidebar, and heavy layout styles from leaking into the iframe.
-   **Performance**: The widget bundle will be smaller.
-   **Simplicity**: Easier to reason about "widget state" when it's a fresh React mount rather than a route transition in a complex SPA.

## 3. Authentication

**Decision**: Use a configurable "Service Token" strategy.

**Strategy**:
-   Refactor `AuthService` to support a `TokenValidator` interface or strategy.
-   Implement `JWTValidator` (existing) and `StaticTokenValidator` (new).
-   Add `CHATGPT_SERVICE_TOKEN` to `AppConfig`.
-   If `CHATGPT_SERVICE_TOKEN` is set, the backend will accept it as a valid Bearer token for any user context (or a specific "chatgpt-bot" user).

**Rationale**:
-   Hugging Face OAuth is not compatible with the Apps SDK OIDC flow.
-   Implementing a full OIDC provider is out of scope for the hackathon.
-   A static service token is secure enough for a demo/hackathon submission and easy to configure in the OpenAI Developer Platform.

## 4. Infrastructure & Hosting

**Decision**: Serve `widget.html` via FastAPI static mount with Skybridge MIME type.

**Strategy**:
-   Update `backend/src/api/main.py` to serve `frontend/dist/widget.html` on a specific route (e.g., `/widget`).
-   Ensure the Content-Type header is `text/html+skybridge` (or whatever the specific requirement is, usually just serving it is enough, but we will double-check if OpenAI needs specific headers). *Correction: The expert mentioned `text/html+skybridge` media type for `FileResponse`.*
-   Update CORS to allow `https://chatgpt.com` (and `https://*.chatgpt.com`).

**Rationale**: Required for the widget to load inside the ChatGPT iframe.
