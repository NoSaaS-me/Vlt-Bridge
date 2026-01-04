/**
 * Rules API client functions (T068)
 *
 * Provides functions to interact with the Rules API endpoints
 * for managing Oracle Plugin System rules.
 */

import type {
  RuleInfo,
  RuleDetail,
  RuleListResponse,
  RuleToggleRequest,
  RuleTestRequest,
  RuleTestResponse,
  HookPoint,
  // Plugin types (PluginInfo used via PluginListResponse)
  PluginDetail,
  PluginListResponse,
  PluginSettingsUpdateRequest,
} from '@/types/rules';
import { getAuthToken } from './api';

/**
 * Get the base URL for API requests
 */
function getBaseUrl(): string {
  if (window.API_BASE_URL) {
    return window.API_BASE_URL;
  }
  return '';
}

/**
 * Get authorization headers
 */
function getHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  return headers;
}

/**
 * Fetch all rules with optional filtering
 *
 * @param options Filter options
 * @returns Promise resolving to RuleListResponse
 */
export async function fetchRules(options?: {
  pluginId?: string;
  trigger?: HookPoint;
  enabledOnly?: boolean;
}): Promise<RuleListResponse> {
  const params = new URLSearchParams();

  if (options?.pluginId) {
    params.set('plugin_id', options.pluginId);
  }
  if (options?.trigger) {
    params.set('trigger', options.trigger);
  }
  if (options?.enabledOnly) {
    params.set('enabled_only', 'true');
  }

  const queryString = params.toString();
  const url = `${getBaseUrl()}/api/rules${queryString ? `?${queryString}` : ''}`;

  const response = await fetch(url, {
    method: 'GET',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      error: 'unknown_error',
      detail: `HTTP ${response.status}`,
    }));
    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Fetch a single rule by ID
 *
 * @param ruleId Qualified rule ID
 * @returns Promise resolving to RuleDetail
 */
export async function fetchRule(ruleId: string): Promise<RuleDetail> {
  const url = `${getBaseUrl()}/api/rules/${encodeURIComponent(ruleId)}`;

  const response = await fetch(url, {
    method: 'GET',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      error: 'unknown_error',
      detail: `HTTP ${response.status}`,
    }));
    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Toggle a rule's enabled state
 *
 * @param ruleId Qualified rule ID
 * @param enabled Whether to enable or disable the rule
 * @returns Promise resolving to updated RuleInfo
 */
export async function toggleRule(ruleId: string, enabled: boolean): Promise<RuleInfo> {
  const url = `${getBaseUrl()}/api/rules/${encodeURIComponent(ruleId)}/toggle`;

  const body: RuleToggleRequest = { enabled };

  const response = await fetch(url, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      error: 'unknown_error',
      detail: `HTTP ${response.status}`,
    }));

    // Handle specific error cases
    if (errorData.error === 'cannot_disable_core_rule') {
      throw new Error('Cannot disable core rules');
    }

    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Test a rule's condition with optional context overrides
 *
 * @param ruleId Qualified rule ID
 * @param contextOverride Optional context values to override
 * @returns Promise resolving to RuleTestResponse
 */
export async function testRule(
  ruleId: string,
  contextOverride?: Record<string, unknown>
): Promise<RuleTestResponse> {
  const url = `${getBaseUrl()}/api/rules/${encodeURIComponent(ruleId)}/test`;

  const body: RuleTestRequest = contextOverride ? { context_override: contextOverride } : {};

  const response = await fetch(url, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      error: 'unknown_error',
      detail: `HTTP ${response.status}`,
    }));
    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
  }

  return response.json();
}

// ============================================================================
// Plugin API Functions (T087)
// ============================================================================

/**
 * Fetch all plugins
 *
 * @returns Promise resolving to PluginListResponse
 */
export async function fetchPlugins(): Promise<PluginListResponse> {
  const url = `${getBaseUrl()}/api/plugins`;

  const response = await fetch(url, {
    method: 'GET',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      error: 'unknown_error',
      detail: `HTTP ${response.status}`,
    }));
    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Fetch a single plugin by ID
 *
 * @param pluginId Plugin identifier
 * @returns Promise resolving to PluginDetail
 */
export async function fetchPlugin(pluginId: string): Promise<PluginDetail> {
  const url = `${getBaseUrl()}/api/plugins/${encodeURIComponent(pluginId)}`;

  const response = await fetch(url, {
    method: 'GET',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      error: 'unknown_error',
      detail: `HTTP ${response.status}`,
    }));
    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Fetch settings for a specific plugin
 *
 * @param pluginId Plugin identifier
 * @returns Promise resolving to effective settings
 */
export async function fetchPluginSettings(
  pluginId: string
): Promise<Record<string, number | string | boolean>> {
  const url = `${getBaseUrl()}/api/plugins/${encodeURIComponent(pluginId)}/settings`;

  const response = await fetch(url, {
    method: 'GET',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      error: 'unknown_error',
      detail: `HTTP ${response.status}`,
    }));
    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Update settings for a specific plugin
 *
 * @param pluginId Plugin identifier
 * @param settings Setting values to update
 * @returns Promise resolving to updated effective settings
 */
export async function updatePluginSettings(
  pluginId: string,
  settings: Record<string, number | string | boolean>
): Promise<Record<string, number | string | boolean>> {
  const url = `${getBaseUrl()}/api/plugins/${encodeURIComponent(pluginId)}/settings`;

  const body: PluginSettingsUpdateRequest = { settings };

  const response = await fetch(url, {
    method: 'PUT',
    headers: getHeaders(),
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      error: 'unknown_error',
      detail: `HTTP ${response.status}`,
    }));
    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Fetch state for a specific plugin
 *
 * @param pluginId Plugin identifier
 * @param projectId Project ID for state scoping (optional, defaults to "default")
 * @returns Promise resolving to plugin state
 */
export async function fetchPluginState(
  pluginId: string,
  projectId: string = 'default'
): Promise<Record<string, unknown>> {
  const params = new URLSearchParams();
  params.set('project_id', projectId);

  const url = `${getBaseUrl()}/api/plugins/${encodeURIComponent(pluginId)}/state?${params.toString()}`;

  const response = await fetch(url, {
    method: 'GET',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      error: 'unknown_error',
      detail: `HTTP ${response.status}`,
    }));
    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Clear state for a specific plugin
 *
 * @param pluginId Plugin identifier
 * @param projectId Project ID for state scoping (optional, defaults to "default")
 */
export async function clearPluginState(
  pluginId: string,
  projectId: string = 'default'
): Promise<void> {
  const params = new URLSearchParams();
  params.set('project_id', projectId);

  const url = `${getBaseUrl()}/api/plugins/${encodeURIComponent(pluginId)}/state?${params.toString()}`;

  const response = await fetch(url, {
    method: 'DELETE',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      error: 'unknown_error',
      detail: `HTTP ${response.status}`,
    }));
    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
  }
}
