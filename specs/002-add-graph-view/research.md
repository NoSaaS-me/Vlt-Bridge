# Phase 0: Research & Technical Decisions

## 1. Graph Visualization Library

**Decision**: Use `react-force-graph-2d`.

**Rationale**:
-   **Performance**: Uses HTML5 Canvas/WebGL for rendering, capable of handling thousands of nodes (meeting the "personal knowledge base" scale requirement).
-   **React Integration**: Native React component wrapper around `force-graph`, managing the lifecycle and updates declarative.
-   **Feature Set**: Built-in zoom/pan, auto-centering, node/link interactions (hover/click), and flexible styling.
-   **Maintainability**: Widely used, active community.

**Alternatives Considered**:
-   `vis-network`: Good but heavier and imperative API is harder to integrate cleanly with modern React hooks.
-   `d3-force` (raw): Too low-level. Would require rebuilding canvas rendering, zoom/pan logic, and interaction handlers from scratch.
-   `cytoscape.js`: Powerful but focused more on graph theory analysis; visual customization is CSS-like but sometimes more complex for "floating particle" aesthetics.

## 2. Data Structure & API

**Decision**: Flat structure with `nodes` and `links` arrays.

**Schema**:
```json
{
  "nodes": [
    { "id": "path/to/note.md", "label": "Note Title", "val": 1, "group": "folder-name" }
  ],
  "links": [
    { "source": "path/to/source.md", "target": "path/to/target.md" }
  ]
}
```

**Rationale**:
-   Matches the expected input format of `react-force-graph`.
-   `val` property allows automatic node sizing based on degree (calculated on backend or frontend). Backend calculation is preferred for caching/performance.
-   `id` uses the file path to ensure uniqueness and easy mapping back to navigation events.

## 3. Theme Integration

**Decision**: Pass dynamic colors via React props, reading from CSS variables or a ThemeContext.

**Strategy**:
-   The `GraphView` component will hook into the current theme (light/dark).
-   Colors for background, nodes, and text will be passed to `<ForceGraph2D />`.
-   **Light Mode**: White background, dark grey nodes/links.
-   **Dark Mode**: `hsl(var(--background))` (usually dark), light grey nodes.
-   **Groups**: Use a categorical color scale for folders (e.g., D3 scale or a fixed palette).

## 4. Unlinked Notes

**Decision**: Include all notes in the `nodes` array, even if they have no entries in `links`.

**Physics**:
-   The force simulation will naturally push unconnected nodes away from the center cluster but keep them within the viewport if a bounding box or gravity is applied.
-   We will apply a weak `d3.forceManyBody` (repulsion) and a central `d3.forceCenter` to keep the "cloud" visible.

## 5. Backend Implementation

**Decision**: Add `get_graph_data` to `IndexerService`.

**Logic**:
1.  Fetch all notes from `note_metadata` (id, title).
2.  Fetch all links from `note_links` where `is_resolved=1`.
3.  Compute link counts for node sizing (optional optimization: do this in SQL or Python).
4.  Return JSON.
5.  **Caching**: Use a simple in-memory cache with a short TTL (e.g., 5 minutes) or invalidation on note create/update events to ensure sub-2s response times for large vaults. *For V1, direct SQL query is likely fast enough for <1000 notes.*
