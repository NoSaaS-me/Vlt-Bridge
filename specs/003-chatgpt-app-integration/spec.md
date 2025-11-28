# Feature Specification: ChatGPT App Integration

**Feature**: 003-chatgpt-app-integration
**Status**: Draft
**Created**: 2025-11-26

## 1. Summary

Transform the Document-MCP project into a "ChatGPT App" compatible with the OpenAI Apps SDK. This integration enables ChatGPT users to interact with their document vault using native-feeling UI widgets (Note Viewer, Search Results) embedded directly in the chat interface, powered by the existing FastMCP server.

## 2. Problem Statement

**Context**: Currently, the Document-MCP project works as a standalone web app or a standard MCP server.
**Problem**: ChatGPT users accessing the vault via standard MCP tools receive raw markdown text in the chat stream, which is verbose and lacks interactivity. They cannot easily visualize the vault or navigate links without leaving the chat context.
**Impact**: Limits the "AI Knowledge Assistant" experience by forcing users to context-switch between ChatGPT and a separate tab, or suffer through poor readability of raw text responses.

## 3. Goals & Non-Goals

### Goals
-   **Seamless Integration**: Enable users to search, view, and edit notes entirely within the ChatGPT interface.
-   **Visual Widgets**: Replace raw text responses with interactive UI widgets for:
    -   Note Viewing (with Markdown rendering and Wikilink support)
    -   Search Results (clean list with snippets)
-   **Dual-Mode Operation**: Ensure the application continues to function as a standalone web app and standard MCP server while supporting the ChatGPT App mode.
-   **Hackathon Readiness**: Prioritize a functional integration. We will use "No Authentication" mode for the hackathon submission to bypass OAuth complexity, securing it via obscurity (hidden URL) and "demo-user" isolation.

### Non-Goals
-   **Full Obsidian UI in Chat**: We will not iframe the entire application (sidebar, graph view, settings) into ChatGPT.
-   **Production OAuth**: We will not implement a full OIDC provider. **Future Work**: Implement a "Headless OAuth" provider to secure the endpoint properly using Client Credentials semantics wrapped in Authorization Code flow.
-   **Complex Graph Viz**: The Graph View widget is out of scope for the initial V1 integration.

## 4. User Scenarios

### Scenario 1: The Recall Loop
**User**: A developer brainstorming in ChatGPT.
**Action**: User asks, "What did I note about the authentication API?"
**System**: ChatGPT calls `search_notes("authentication API")`.
**Result**: Instead of a JSON dump, a **Search Results Widget** appears in the chat, listing matching notes. The user clicks "API Documentation" in the widget.
**Follow-up**: The widget transitions to a **Note Viewer Widget**, displaying the rendered markdown of "API Documentation".

### Scenario 2: In-Context Editing
**User**: Reading the "API Documentation" note in the widget.
**Action**: User tells ChatGPT, "Update the Auth section to mention we use RS256 now."
**System**: ChatGPT calls `read_note` (invisible to user), generates the diff, calls `write_note`, and confirms.
**Result**: The Note Viewer widget refreshes (or a status widget appears) showing the updated content directly in the thread.

## 5. Functional Requirements

### 5.1 Backend & MCP
-   **Metadata Injection**: The `read_note` and `search_notes` tools must return a `CallToolResult` containing the `_meta.openai.outputTemplate` field to trigger widgets.
-   **CORS**: The API must allow requests from `https://chatgpt.com` to support iframe loading.
-   **No Auth Mode**: The backend must support a configured `ENABLE_NOAUTH_MCP` flag. When enabled, MCP tools will default to the "demo-user" context if no Authorization header is present, bypassing strict checks.

### 5.2 Frontend Widgets
-   **Widget Entry Point**: A new build target (`widget.html` + `widget.tsx`) must be created to serve simplified UI components.
-   **Note Viewer Widget**: A lightweight version of the `NoteViewer` component that:
    -   Renders Markdown.
    -   Handles Wikilink clicks (by requesting ChatGPT to navigate or loading the new note within the widget).
    -   Hides the sidebar and app chrome.
-   **Search Widget**: A simple list view for search results that triggers note navigation on click.

### 5.3 Infrastructure
-   **Static Serving**: The FastAPI server must serve `widget.html` with the correct `text/html+skybridge` MIME type when requested.
-   **Build Pipeline**: The Vite configuration must output both the main SPA (`index.html`) and the widget bundle (`widget.html`).

## 6. Success Criteria

1.  **Widget Rendering**: A `read_note` tool call successfully renders the custom HTML widget inside the ChatGPT Developer Mode interface.
2.  **Navigation**: Clicking a Wikilink in the widget successfully loads the target note (either by refreshing the widget or triggering a new tool call).
3.  **Zero Regression**: The existing standalone web app (`/`) continues to function normally for local development.
4.  **Performance**: Widget load time < 500ms (leveraging the lightweight bundle).

## 7. Assumptions & Dependencies

-   **Host**: Hugging Face Spaces or Localhost (tunneled) will be used for hosting.
-   **Apps SDK**: We rely on the OpenAI Apps SDK beta features; behavior may be subject to platform changes.
-   **Auth**: We assume a single-tenant or shared-tenant "demo" mode is acceptable for the hackathon submission.

## 8. Questions & Clarifications

1.  **Widget Navigation**: When a user clicks a link in the widget, should it trigger a client-side router push (staying in the same iframe) or ask ChatGPT to "open note X"?
    *   *Assumption*: Client-side navigation within the widget is smoother for "browsing", while asking ChatGPT is better for "contextualizing". We will prioritize **client-side navigation** for V1 to keep the UI snappy.

2.  **Auth Header**: Will ChatGPT send the service token in the `Authorization` header?
    *   *Assumption*: Yes, we will configure the Custom API Action or App definition with the static Bearer token.