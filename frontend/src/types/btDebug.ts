/**
 * Types for BT Debug API (Phase 7: Observability)
 *
 * Part of the BT Universal Runtime (spec 019).
 */

// =============================================================================
// Enums
// =============================================================================

export type TreeStatus = 'idle' | 'running' | 'completed' | 'failed' | 'yielded';

export type NodeType = 'composite' | 'decorator' | 'leaf';

export type RunStatus = 'fresh' | 'success' | 'failure' | 'running';

// =============================================================================
// Tree List Types
// =============================================================================

export interface TreeSummary {
  id: string;
  name: string;
  status: TreeStatus;
  tick_count: number;
  node_count: number;
  source_path: string;
  loaded_at: string;
  last_tick_at: string | null;
}

export interface TreeListResponse {
  trees: TreeSummary[];
  total: number;
}

// =============================================================================
// Tree State Types
// =============================================================================

export interface NodeInfo {
  id: string;
  name: string;
  node_type: NodeType;
  status: RunStatus;
  tick_count: number;
  running_since: string | null;
  last_tick_duration_ms: number;
  is_active: boolean;
  children: NodeInfo[];
  metadata: Record<string, unknown>;
}

export interface TreeStateResponse {
  id: string;
  name: string;
  description: string;
  status: TreeStatus;
  tick_count: number;
  node_count: number;
  max_depth: number;
  max_tick_duration_ms: number;
  tick_budget: number;
  reload_pending: boolean;
  source_path: string;
  source_hash: string;
  loaded_at: string;
  last_tick_at: string | null;
  execution_start: string | null;
  blackboard_size_bytes: number;
  active_path: string[];
  root: NodeInfo;
}

// =============================================================================
// Blackboard Types
// =============================================================================

export interface BlackboardKeyInfo {
  key: string;
  schema_type: string;
  has_value: boolean;
  value_preview: string | null;
  size_bytes: number;
}

export interface BlackboardResponse {
  tree_id: string;
  scope_name: string;
  parent_scope: string | null;
  size_bytes: number;
  max_size_bytes: number;
  key_count: number;
  keys: BlackboardKeyInfo[];
  snapshot: Record<string, unknown>;
  reads_this_tick: string[];
  writes_this_tick: string[];
}

// =============================================================================
// Tick History Types
// =============================================================================

export interface TickHistoryEntry {
  tick_number: number;
  timestamp: string;
  node_id: string;
  node_path: string[];
  status: RunStatus;
  duration_ms: number;
  event_type: string | null;
  details: Record<string, unknown>;
}

export interface TickHistoryResponse {
  tree_id: string;
  total_entries: number;
  returned_entries: number;
  offset: number;
  limit: number;
  entries: TickHistoryEntry[];
}

// =============================================================================
// Breakpoint Types
// =============================================================================

export interface BreakpointInfo {
  node_id: string;
  enabled: boolean;
  condition: string | null;
  hit_count: number;
  created_at: string;
}

export interface SetBreakpointRequest {
  node_id: string;
  enabled?: boolean;
  condition?: string | null;
}

export interface DeleteBreakpointRequest {
  node_id: string;
}

export interface BreakpointResponse {
  tree_id: string;
  breakpoints: BreakpointInfo[];
  message: string;
}

// =============================================================================
// Visualization Types
// =============================================================================

export interface TreeVisualization {
  tree_id: string;
  ascii_tree: string;
  json_structure: Record<string, unknown>;
  dot_graph: string;
}

export type VisualizationFormat = 'json' | 'ascii' | 'dot' | 'all';
