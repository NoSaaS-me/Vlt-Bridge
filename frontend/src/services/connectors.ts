/**
 * API client for the connector system.
 */

export interface CredentialField {
  name: string;
  label: string;
  secret: boolean;
  placeholder: string;
}

export interface ConnectorParam {
  name: string;
  description: string;
  required: boolean;
  default: string | null;
}

export interface ConnectorAction {
  name: string;
  description: string;
  params: ConnectorParam[];
}

export interface ConnectorInfo {
  name: string;
  display_name: string;
  description: string;
  credential_fields: CredentialField[];
  actions: ConnectorAction[];
  enabled: boolean;
  configured: boolean;
}

export interface ConnectorInvokeResponse {
  success: boolean;
  result: Record<string, unknown>;
  error: string | null;
}

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('auth_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function listConnectors(): Promise<ConnectorInfo[]> {
  const resp = await fetch('/api/connectors', { headers: authHeaders() });
  if (!resp.ok) throw new Error(`Failed to load connectors (${resp.status})`);
  const data = await resp.json();
  return data.connectors;
}

export async function getConnectorConfig(
  connectorName: string
): Promise<Record<string, string>> {
  const resp = await fetch(`/api/connectors/${connectorName}/config`, {
    headers: authHeaders(),
  });
  if (!resp.ok) throw new Error(`Failed to load connector config (${resp.status})`);
  const data = await resp.json();
  return data.config;
}

export async function saveConnectorConfig(
  connectorName: string,
  config: Record<string, string>
): Promise<void> {
  const resp = await fetch(`/api/connectors/${connectorName}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ config }),
  });
  if (!resp.ok) throw new Error(`Failed to save connector config (${resp.status})`);
}

export async function invokeConnector(
  connectorName: string,
  action: string,
  params: Record<string, unknown>
): Promise<ConnectorInvokeResponse> {
  const resp = await fetch(`/api/connectors/${connectorName}/invoke`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ action, params }),
  });
  if (!resp.ok) throw new Error(`Connector invoke failed (${resp.status})`);
  return resp.json();
}
