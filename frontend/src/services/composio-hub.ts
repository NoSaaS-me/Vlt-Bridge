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

export const connectApp = (appName: string) =>
  req<{ app: string; redirect_url: string }>(`/connect/${appName}`, {
    method: 'POST',
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
  params: Record<string, unknown> = {}
) =>
  req<{ success: boolean; data: Record<string, unknown> }>(`/${appName}/invoke`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, params }),
  });

export const getComposioConfig = (appName: string) =>
  req<{ connector: string; config: Record<string, string> }>(`/${appName}/config`);

export const saveComposioConfig = (appName: string, config: Record<string, string>) =>
  req<{ connector: string; saved: boolean }>(`/${appName}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config }),
  });
