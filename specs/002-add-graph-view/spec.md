# Feature Specification: Interactive Graph View

**Feature**: 002-add-graph-view
**Status**: Draft
**Created**: 2025-11-25

## 1. Summary

An interactive, force-directed graph visualization of the note vault that displays notes as nodes and wikilinks as connections. This view provides users with a high-level understanding of their knowledge base's structure and an alternative method for navigation.

## 2. Problem Statement

**Context**: Currently, users browse notes via a linear directory tree or search results.
**Problem**: These views do not reveal the structural relationships, clusters, or connectivity density of the knowledge base. Users cannot easily see which notes are central "hubs" or which are isolated "orphans."
**Impact**: Reduces the ability to maintain a well-connected knowledge garden and limits discovery of related concepts.

## 3. Goals & Non-Goals

### Goals
-   Provide a visual representation of the vault's link structure.
-   Enable intuitive navigation by clicking on nodes in the graph.
-   Highlight important notes through visual properties (e.g., node size based on connectivity).
-   Identify unlinked (orphan) notes visually.

### Non-Goals
-   Full 3D visualization (2D is sufficient for V1).
-   Complex graph editing (e.g., creating links by dragging lines between nodes).
-   Advanced graph analytics (centrality metrics beyond simple degree).

## 4. User Scenarios

### Scenario 1: Structural Overview
**User**: A researcher managing a large knowledge base.
**Action**: Clicks the "Graph View" toggle button in the application toolbar.
**Result**: The note editor is replaced by a full-panel canvas showing a web of connected nodes. The user pans and zooms to explore different clusters of notes, identifying a dense cluster related to "Project X."

### Scenario 2: Visual Navigation
**User**: Looking for a specific core concept note.
**Action**: Identifies a large node in the center of the graph (indicating many connections). hovers over it to confirm the title "Core Concepts," and clicks it.
**Result**: The application switches back to the standard note view, displaying the "Core Concepts" note.

### Scenario 3: Orphan Identification
**User**: Wants to improve note connectivity.
**Action**: Opens the graph view and looks for small nodes floating unconnected at the periphery of the main cluster.
**Result**: Identifies an isolated note, clicks it to open, and adds links to connect it to the rest of the graph.

## 5. Functional Requirements

### 5.1 Graph Visualization
-   **Nodes**: Represent individual notes.
-   **Edges**: Represent resolved wikilinks between notes.
-   **Node Sizing**: Nodes must scale dynamically based on their number of connections (link degree); heavily linked notes appear larger.
-   **Unlinked Notes**: Notes with no connections must be visible, floating freely within the simulation (not hidden).
-   **Theme Compatibility**: The graph background and element colors must adapt to the application's current theme (Light/Dark mode).

### 5.2 Interaction
-   **Navigation**: Clicking a node must activate that note in the main view and switch away from the graph.
-   **Controls**: Users must be able to pan the canvas and zoom in/out.
-   **Hover Details**: Hovering over a node must display a tooltip with the note's title.
-   **Physics**: Nodes should naturally repel each other to reduce overlap, with links acting as springs to hold connected notes together.

### 5.3 UI Integration
-   **Access Control**: A toggle control (e.g., "Graph" vs. "Note") must be available in the main application toolbar.
-   **Persistence**: The graph view should retain its state (zoom level, position) transiently while the app is open, if possible, or reload quickly.

### 5.4 Data Source
-   **Graph Data**: The system must generate a graph payload containing:
    -   Nodes: ID (path), Label (title), Size metric (link count), Grouping (folder).
    -   Links: Source, Target.
-   **Folder Grouping**: Nodes should ideally be visually distinct (e.g., by color) based on their top-level folder or category.

## 6. Success Criteria

1.  **Load Time**: Graph renders with < 2 seconds latency for a vault of up to 1,000 notes.
2.  **Visual Clarity**: Unlinked notes are clearly distinguishable from the main connected component.
3.  **Navigation Accuracy**: Clicking a node opens the correct corresponding note 100% of the time.
4.  **Theme Consistency**: Switching between light and dark modes updates the graph colors immediately or upon next render without requiring a reload.

## 7. Assumptions & Dependencies

-   **Browser Support**: The user's environment supports WebGL or HTML5 Canvas for performant rendering.
-   **Data Volume**: The initial implementation target is for personal knowledge bases (hundreds to low thousands of notes), not enterprise scale (millions).
-   **Link Resolution**: Only "resolved" links (links pointing to existing notes) generate edges.

## 8. Questions & Clarifications

*(None required at this stage. Standard force-directed graph behavior is assumed for layout logic.)*