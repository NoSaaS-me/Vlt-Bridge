# Document-MCP Constitution

## Core Principles

### I. Brownfield Integration
Respect the existing codebase. Do not rewrite or refactor existing working code unless explicitly required by the feature spec. Match the style, naming conventions, and architectural patterns of the `backend/` (FastAPI) and `frontend/` (React+Vite) directories.

### II. Test-Backed Development
Every new backend feature must include `pytest` unit tests. Frontend logic should be tested where feasible, but UI components rely on manual verification/E2E. Do not break existing tests.

### III. Incremental Delivery
Features should be implemented in small, safe increments. Use feature flags or parallel file structures (e.g., new routes/components) to avoid destabilizing the main application during development.

### IV. Specification-Driven
All work must be traced back to a `specs/` document. If a requirement changes during implementation, the spec must be updated to reflect reality. Do not implement "ghost features" not in the spec.

## Technology Standards

**Backend**:
-   Python 3.11+
-   FastAPI for API routes
-   Pydantic for data validation
-   SQLite for persistence (via `sqlite3` stdlib, no ORM overhead preferred)

**Frontend**:
-   React 18+
-   TypeScript
-   Tailwind CSS for styling
-   Shadcn/UI for components
-   Lucide React for icons

## Governance

-   **No Magic**: Avoid "clever" abstractions. Explicit is better than implicit.
-   **Single Source of Truth**: The `data/` directory is the state of record for the vault. The index is a derived view.
-   **Error Handling**: All API endpoints must return structured error responses. Frontend must handle these gracefully.

**Version**: 1.0.0 | **Ratified**: 2025-11-25