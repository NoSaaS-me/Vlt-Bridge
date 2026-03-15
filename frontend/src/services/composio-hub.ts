/**
 * Composio Integration Hub API client.
 * All routes go through /api/composio
 */

export interface ComposioApp {
  name: string;
  display_name: string;
  description: string;
  categories: string[];
  connected: boolean;
}

export interface ComposioConnection {
  app_name: string;
  status: string;
  connection_id: string;
}

export interface ComposioAction {
  name: string;
  display_name: string;
  description: string;
  parameters: Record<string, unknown>;
}

// --- Auth info types (024-composio-connection-vault) ---

export interface AuthFieldInfo {
  name: string;
  display_name: string;
  description: string;
  type: string;
  required: boolean;
  expected_from_customer: boolean;
}

export interface AuthSchemeInfo {
  auth_mode: string;
  integration_fields: AuthFieldInfo[];
  user_fields: AuthFieldInfo[];
}

export interface AppAuthInfo {
  has_managed_auth: boolean;
  primary_auth_mode: string;
  auth_schemes: AuthSchemeInfo[];
}

export interface ConnectRequest {
  label?: string;
  auth_mode?: string;
  auth_config?: Record<string, string>;
  connected_account_params?: Record<string, string>;
  redirect_url?: string;
}

export interface ConnectResponse {
  app: string;
  connection_id: string;
  label: string;
  redirect_url: string | null;
  status: string;
}

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('auth_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`/api/composio${path}`, {
    ...init,
    headers: { ...authHeaders(), ...(init?.headers ?? {}) },
    signal: AbortSignal.timeout(15000),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Composio API ${resp.status}: ${text}`);
  }
  return resp.json();
}

export const getComposioStatus = () =>
  req<{ configured: boolean }>('/status');

export const listApps = () =>
  req<{ apps: ComposioApp[]; total: number }>('/apps');

export const listConnected = () =>
  req<{ connections: ComposioConnection[]; total: number }>('/connected');

export const getAuthInfo = (appName: string) =>
  req<AppAuthInfo>(`/${appName}/auth-info`);

export const connectApp = (appName: string, body?: ConnectRequest) =>
  req<ConnectResponse>(`/connect/${appName}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });

export const disconnectApp = (appName: string) =>
  req<{ app: string; disconnected: boolean }>(`/${appName}`, {
    method: 'DELETE',
  });

export const listAppActions = (appName: string) =>
  req<{ app: string; actions: ComposioAction[]; total: number }>(`/${appName}/actions`);

export const invokeComposioAction = (
  appName: string,
  action: string,
  params: Record<string, unknown> = {},
  connectionId?: string
) =>
  req<{ success: boolean; data: Record<string, unknown> }>(`/${appName}/invoke`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, params, connection_id: connectionId }),
  });

export const getComposioConfig = (appName: string) =>
  req<{ connector: string; config: Record<string, string> }>(`/${appName}/config`);

export const saveComposioConfig = (appName: string, config: Record<string, string>) =>
  req<{ connector: string; saved: boolean }>(`/${appName}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config }),
  });
