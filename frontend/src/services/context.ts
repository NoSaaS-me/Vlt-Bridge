/**
 * Context Tree API Service
 * Handles all context tree management operations
 */
import { apiFetch } from './api';
import type {
  ContextTree,
  ContextTreeData,
  ContextTreesResponse,
  ContextNode,
  PruneResponse,
  ContextSettings,
} from '@/types/context';

/**
 * Get all context trees for the current user
 */
export async function getContextTrees(projectId?: string): Promise<ContextTreesResponse> {
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  const query = params.toString();
  return apiFetch<ContextTreesResponse>(`/api/oracle/context/trees${query ? `?${query}` : ''}`);
}

/**
 * Get a specific context tree by root ID
 */
export async function getContextTree(rootId: string): Promise<ContextTreeData> {
  return apiFetch<ContextTreeData>(`/api/oracle/context/trees/${encodeURIComponent(rootId)}`);
}

/**
 * Create a new context tree
 */
export async function createTree(label?: string, projectId?: string): Promise<ContextTree> {
  return apiFetch<ContextTree>('/api/oracle/context/trees', {
    method: 'POST',
    body: JSON.stringify({ label, project_id: projectId }),
  });
}

/**
 * Delete a context tree
 */
export async function deleteTree(rootId: string): Promise<void> {
  await apiFetch<void>(`/api/oracle/context/trees/${encodeURIComponent(rootId)}`, {
    method: 'DELETE',
  });
}

/**
 * Set the active context tree
 */
export async function setActiveTree(rootId: string): Promise<void> {
  await apiFetch<void>(`/api/oracle/context/trees/${encodeURIComponent(rootId)}/activate`, {
    method: 'POST',
  });
}

/**
 * Checkout a specific node (set as current node)
 */
export async function checkoutNode(nodeId: string): Promise<ContextTree> {
  return apiFetch<ContextTree>(`/api/oracle/context/nodes/${encodeURIComponent(nodeId)}/checkout`, {
    method: 'POST',
  });
}

/**
 * Label a node
 */
export async function labelNode(nodeId: string, label: string): Promise<ContextNode> {
  return apiFetch<ContextNode>(`/api/oracle/context/nodes/${encodeURIComponent(nodeId)}/label`, {
    method: 'PUT',
    body: JSON.stringify({ label }),
  });
}

/**
 * Set checkpoint status for a node
 */
export async function setCheckpoint(nodeId: string, isCheckpoint: boolean): Promise<ContextNode> {
  return apiFetch<ContextNode>(`/api/oracle/context/nodes/${encodeURIComponent(nodeId)}/checkpoint`, {
    method: 'PUT',
    body: JSON.stringify({ is_checkpoint: isCheckpoint }),
  });
}

/**
 * Prune a tree (remove old non-checkpoint nodes)
 */
export async function pruneTree(rootId: string): Promise<PruneResponse> {
  return apiFetch<PruneResponse>(`/api/oracle/context/trees/${encodeURIComponent(rootId)}/prune`, {
    method: 'POST',
  });
}

/**
 * Get context settings for the current user
 */
export async function getContextSettings(): Promise<ContextSettings> {
  return apiFetch<ContextSettings>('/api/oracle/context/settings');
}

/**
 * Update context settings
 */
export async function updateContextSettings(settings: Partial<ContextSettings>): Promise<ContextSettings> {
  return apiFetch<ContextSettings>('/api/oracle/context/settings', {
    method: 'PUT',
    body: JSON.stringify(settings),
  });
}
