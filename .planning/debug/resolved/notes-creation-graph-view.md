---
status: resolved
trigger: "Notes creation fails with error; Graph view not loading"
created: 2026-02-26T00:00:00Z
updated: 2026-02-26T00:03:00Z
---

## Current Focus

hypothesis: Two confirmed bugs found. Fixing now.
test: Verified with Python REPL simulation
expecting: Fixes restore note creation and graph view
next_action: Apply fixes to graph.py (model) and indexer.py (get_graph_data return shape)

## Symptoms

expected: Notes can be created; Graph view loads
actual: Notes creation may fail (conditional); Graph view returns 500 error
errors: Pydantic ValidationError on /api/graph — missing 'group' field on GraphNode, missing 'links' key (got 'edges')
reproduction: GET /api/graph → 500 Internal Server Error
started: After recent security + model changes

## Eliminated

- hypothesis: sanitize_path os.path.commonpath breaks note creation for new notes
  evidence: Tested in Python REPL — commonpath works correctly for non-existent paths on Linux; .resolve() still produces correct absolute paths
  timestamp: 2026-02-26T00:01:00Z

- hypothesis: double-raise in sanitize_path breaks path validation
  evidence: The except ValueError catches the inner raise and re-raises — redundant but not broken; path validation still works
  timestamp: 2026-02-26T00:01:00Z

## Evidence

- timestamp: 2026-02-26T00:01:00Z
  checked: backend/src/models/graph.py
  found: GraphNode requires 'group: str = Field(...)' with NO default. GraphData uses 'links: List[GraphLink]'.
  implication: Any response from get_graph_data() will fail Pydantic validation

- timestamp: 2026-02-26T00:01:00Z
  checked: backend/src/services/indexer.py get_graph_data()
  found: Returns dict with 'edges' key (not 'links'). Nodes have only {id, label} — missing 'group' and 'val'.
  implication: Shape mismatch causes 500 on every GET /api/graph

- timestamp: 2026-02-26T00:01:00Z
  checked: frontend/src/types/graph.ts and GraphView.tsx
  found: Frontend expects {nodes: GraphNode[], links: GraphLink[]} where GraphNode has {id, label, val, group}. Uses 'links' (not 'edges').
  implication: Backend model matches frontend expectation; indexer.get_graph_data() is the source of the mismatch

- timestamp: 2026-02-26T00:01:00Z
  checked: backend/src/services/vault.py sanitize_path
  found: os.path.commonpath security fix is CORRECT. No bug here.
  implication: Bug 1 (notes creation) is actually Bug 2 (graph). Note creation path validation is fine.

## Resolution

root_cause: |
  Bug (graph view): indexer.get_graph_data() returns wrong shape:
  - Key 'edges' instead of 'links' (GraphData model expects 'links')
  - Node dicts missing 'group' field (required, no default in GraphNode)
  - Node dicts missing 'val' field (has default=1, so ok)
  Fix: Update get_graph_data() to return 'links' key and add 'group' field
  (group = top-level folder of note path, e.g. 'folder/' or '' for root)

fix: |
  1. indexer.py get_graph_data(): Changed 'edges' key to 'links', added 'group' and 'val'
     fields to each node dict. group = top-level folder or '' for root notes.
  2. vault.py sanitize_path(): Refactored try/except to not accidentally catch its own
     inner ValueError — now calls commonpath first, catches only commonpath's ValueError,
     then does the comparison check separately.

verification: |
  - Simulated GraphData(**fixed_output) in Python REPL — passes Pydantic validation
  - Tested sanitize_path logic for normal notes, subfolder notes, traversal attempts
  - All cases behave correctly

files_changed:
  - backend/src/services/indexer.py
  - backend/src/services/vault.py
