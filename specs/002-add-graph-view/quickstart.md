# Quickstart: Graph View

## Prerequisites
-   Existing backend running with indexed notes.
-   Frontend dependencies installed.

## Installation

1.  **Frontend Dependencies**:
    ```bash
    cd frontend
    npm install react-force-graph-2d
    ```

2.  **Backend Setup**:
    No extra packages needed (uses existing `sqlite3`).

## Verification Steps

1.  **Start Backend**:
    ```bash
    ./start-dev.sh
    ```

2.  **Open Application**:
    Navigate to `http://localhost:5173`.

3.  **Switch to Graph View**:
    -   Locate the "Graph" icon/button in the top toolbar (next to "New Note").
    -   Click it.
    -   **Verify**: The center panel should replace the note editor with a dark/light canvas showing nodes.

4.  **Interact**:
    -   **Hover**: Mouse over a node; see the tooltip with the note title.
    -   **Drag**: Click and drag a node; it should move and pull connected nodes.
    -   **Click**: Click a node; the view should switch back to "Note View" and open that specific note.
    -   **Zoom**: Scroll wheel to zoom in/out.

5.  **Check Unlinked Notes**:
    -   Create a new note with no links.
    -   Switch to Graph View.
    -   **Verify**: The new note appears as a standalone node.
