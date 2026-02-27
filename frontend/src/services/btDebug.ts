/**
 * BT Debug API Service
 *
 * Part of the BT Universal Runtime (spec 019).
 * Provides frontend access to behavior tree debugging endpoints.
 */

import { apiFetch } from './api';
import type {
  TreeListResponse,
  TreeStateResponse,
  BlackboardResponse,
  TickHistoryResponse,
  BreakpointResponse,
  SetBreakpointRequest,
  DeleteBreakpointRequest,
  TreeVisualization,
  VisualizationFormat,
} from '@/types/btDebug';

// =============================================================================
// Tree List
// =============================================================================

/**
 * List all registered behavior trees.
 */
export async function listTrees(): Promise<TreeListResponse> {
  return apiFetch<TreeListResponse>('/api/bt/debug/trees');
}

// =============================================================================
// Tree State
// =============================================================================

/**
 * Get detailed state of a specific tree.
 */
export async function getTreeState(treeId: string): Promise<TreeStateResponse> {
  return apiFetch<TreeStateResponse>(`/api/bt/debug/tree/${encodeURIComponent(treeId)}`);
}

// =============================================================================
// Blackboard
// =============================================================================

/**
 * Get blackboard state for a tree.
 */
export async function getBlackboard(treeId: string): Promise<BlackboardResponse> {
  return apiFetch<BlackboardResponse>(`/api/bt/debug/tree/${encodeURIComponent(treeId)}/blackboard`);
}

// =============================================================================
// Tick History
// =============================================================================

interface HistoryOptions {
  limit?: number;
  offset?: number;
  nodeId?: string;
  status?: string;
}

/**
 * Get tick history for a tree.
 */
export async function getTickHistory(
  treeId: string,
  options: HistoryOptions = {}
): Promise<TickHistoryResponse> {
  const params = new URLSearchParams();

  if (options.limit !== undefined) {
    params.set('limit', String(options.limit));
  }
  if (options.offset !== undefined) {
    params.set('offset', String(options.offset));
  }
  if (options.nodeId) {
    params.set('node_id', options.nodeId);
  }
  if (options.status) {
    params.set('status', options.status);
  }

  const query = params.toString();
  const url = `/api/bt/debug/tree/${encodeURIComponent(treeId)}/history${query ? `?${query}` : ''}`;

  return apiFetch<TickHistoryResponse>(url);
}

// =============================================================================
// Breakpoints
// =============================================================================

/**
 * List all breakpoints for a tree.
 */
export async function listBreakpoints(treeId: string): Promise<BreakpointResponse> {
  return apiFetch<BreakpointResponse>(`/api/bt/debug/tree/${encodeURIComponent(treeId)}/breakpoint`);
}

/**
 * Set a breakpoint on a node.
 */
export async function setBreakpoint(
  treeId: string,
  request: SetBreakpointRequest
): Promise<BreakpointResponse> {
  return apiFetch<BreakpointResponse>(
    `/api/bt/debug/tree/${encodeURIComponent(treeId)}/breakpoint`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    }
  );
}

/**
 * Delete a breakpoint from a node.
 */
export async function deleteBreakpoint(
  treeId: string,
  request: DeleteBreakpointRequest
): Promise<BreakpointResponse> {
  return apiFetch<BreakpointResponse>(
    `/api/bt/debug/tree/${encodeURIComponent(treeId)}/breakpoint`,
    {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    }
  );
}

/**
 * Clear all breakpoints for a tree.
 */
export async function clearAllBreakpoints(treeId: string): Promise<BreakpointResponse> {
  return apiFetch<BreakpointResponse>(
    `/api/bt/debug/tree/${encodeURIComponent(treeId)}/breakpoint/all`,
    {
      method: 'DELETE',
    }
  );
}

// =============================================================================
// Visualization
// =============================================================================

interface VisualizationOptions {
  format?: VisualizationFormat;
  showStatus?: boolean;
  showTiming?: boolean;
}

/**
 * Get tree visualization in various formats.
 */
export async function getTreeVisualization(
  treeId: string,
  options: VisualizationOptions = {}
): Promise<TreeVisualization> {
  const params = new URLSearchParams();

  if (options.format) {
    params.set('format', options.format);
  }
  if (options.showStatus !== undefined) {
    params.set('show_status', String(options.showStatus));
  }
  if (options.showTiming !== undefined) {
    params.set('show_timing', String(options.showTiming));
  }

  const query = params.toString();
  const url = `/api/bt/debug/tree/${encodeURIComponent(treeId)}/visualization${query ? `?${query}` : ''}`;

  return apiFetch<TreeVisualization>(url);
}

// =============================================================================
// Polling Helper
// =============================================================================

/**
 * Poll tree state at regular intervals.
 * Returns a cleanup function to stop polling.
 */
export function pollTreeState(
  treeId: string,
  callback: (state: TreeStateResponse) => void,
  intervalMs: number = 1000,
  onError?: (error: Error) => void
): () => void {
  let active = true;

  const poll = async () => {
    if (!active) return;

    try {
      const state = await getTreeState(treeId);
      if (active) {
        callback(state);
      }
    } catch (error) {
      if (active && onError) {
        onError(error instanceof Error ? error : new Error(String(error)));
      }
    }

    if (active) {
      setTimeout(poll, intervalMs);
    }
  };

  // Start polling
  poll();

  // Return cleanup function
  return () => {
    active = false;
  };
}
