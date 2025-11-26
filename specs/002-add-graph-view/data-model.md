# Data Model: Graph View

## Entities

### GraphNode
Represents a single note in the graph.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier (Note Path). |
| `label` | `str` | Display title of the note. |
| `val` | `int` | Weight/Size of the node (default: 1 + link_degree). |
| `group` | `str` | Grouping category (e.g., top-level folder). |

### GraphLink
Represents a directed connection between two notes.

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | ID of the source note. |
| `target` | `str` | ID of the target note. |

### GraphData
The top-level payload returned by the API.

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `List[GraphNode]` | All notes in the vault. |
| `links` | `List[GraphLink]` | All resolved connections. |

## Database Mapping

-   **Nodes**: Sourced from `note_metadata` table.
    -   `id` <- `note_path`
    -   `label` <- `title`
    -   `group` <- `note_path` (parsed parent directory)
-   **Links**: Sourced from `note_links` table.
    -   `source` <- `source_path`
    -   `target` <- `target_path`
    -   Filter: `is_resolved = 1`
