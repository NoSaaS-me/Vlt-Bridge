/**
 * Artifact API — communicates with the artifact endpoints on the vlt daemon (localhost:8765).
 * The Vite dev proxy maps /vlt → http://localhost:8765 (stripping /vlt prefix).
 */

export interface ArtifactData {
  id: string;
  user_id: string;
  project_id: string;
  name: string;
  description: string | null;
  type: 'ephemeral' | 'persistent';
  state: 'draft' | 'building' | 'testing' | 'reviewing' | 'approved' | 'deployed' | 'error' | 'update_pending';
  state_history: Array<{ state: string; at: string; by: string }>;
  manifest: Record<string, unknown>;
  thread_id: string | null;
  disk_path: string;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface TestResult {
  passed: boolean;
  exit_code: number;
  stdout: string;
  stderr: string;
  duration_ms: number;
}

async function daemonFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`/vlt${path}`, {
    ...options,
    signal: options.signal ?? AbortSignal.timeout(10000),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail ?? body.message ?? detail;
    } catch {
      // ignore parse errors
    }
    throw new Error(detail);
  }

  if (response.status === 204) return {} as T;
  return response.json() as Promise<T>;
}

export interface ArtifactTemplate {
  name: string;
  description: string;
  has_backend: boolean;
  connectors: string[];
}

export async function listTemplates(): Promise<ArtifactTemplate[]> {
  return daemonFetch<ArtifactTemplate[]>('/api/artifacts/templates');
}

export async function createArtifact(data: {
  name: string;
  description?: string;
  type?: string;
  project_id?: string;
  template?: string;
}): Promise<ArtifactData> {
  return daemonFetch<ArtifactData>('/api/artifacts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

export async function listArtifacts(
  projectId?: string,
  state?: string,
): Promise<ArtifactData[]> {
  const params = new URLSearchParams();
  if (projectId) params.set('project_id', projectId);
  if (state) params.set('state', state);
  const qs = params.toString() ? `?${params}` : '';
  return daemonFetch<ArtifactData[]>(`/api/artifacts${qs}`);
}

export async function getArtifact(id: string): Promise<ArtifactData> {
  return daemonFetch<ArtifactData>(`/api/artifacts/${encodeURIComponent(id)}`);
}

export async function updateArtifact(
  id: string,
  data: Partial<Pick<ArtifactData, 'name' | 'description'> & { manifest: Record<string, unknown> }>,
): Promise<ArtifactData> {
  return daemonFetch<ArtifactData>(`/api/artifacts/${encodeURIComponent(id)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

export async function deleteArtifact(id: string): Promise<void> {
  await daemonFetch<void>(`/api/artifacts/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
}

export async function transitionState(
  id: string,
  targetState: string,
  actor = 'user',
): Promise<ArtifactData> {
  return daemonFetch<ArtifactData>(`/api/artifacts/${encodeURIComponent(id)}/state`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_state: targetState, actor }),
  });
}

export async function callBackend(
  id: string,
  action: string,
  params: Record<string, unknown> = {},
): Promise<unknown> {
  return daemonFetch<unknown>(`/api/artifacts/${encodeURIComponent(id)}/backend/call`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, params }),
  });
}

export async function startBackend(id: string): Promise<{ status: string; pid?: number }> {
  return daemonFetch<{ status: string; pid?: number }>(
    `/api/artifacts/${encodeURIComponent(id)}/backend/start`,
    { method: 'POST' },
  );
}

export async function stopBackend(id: string): Promise<{ status: string }> {
  return daemonFetch<{ status: string }>(
    `/api/artifacts/${encodeURIComponent(id)}/backend/stop`,
    { method: 'POST' },
  );
}

export async function runTests(id: string): Promise<TestResult> {
  return daemonFetch<TestResult>(`/api/artifacts/${encodeURIComponent(id)}/test`, {
    method: 'POST',
  });
}

export async function exportArtifact(id: string): Promise<Blob> {
  const response = await fetch(`/vlt/api/artifacts/${encodeURIComponent(id)}/export`, {
    signal: AbortSignal.timeout(30000),
  });
  if (!response.ok) throw new Error(`Export failed: ${response.status}`);
  return response.blob();
}

export async function importArtifact(file: File, projectId?: string): Promise<ArtifactData> {
  const formData = new FormData();
  formData.append('file', file);
  const params = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
  const response = await fetch(`/vlt/api/artifacts/import${params}`, {
    method: 'POST',
    body: formData,
    signal: AbortSignal.timeout(30000),
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return response.json() as Promise<ArtifactData>;
}

export async function storageGet(id: string, key: string): Promise<unknown> {
  const result = await daemonFetch<{ key: string; value: unknown }>(
    `/api/artifacts/${encodeURIComponent(id)}/storage/${encodeURIComponent(key)}`,
  );
  return result.value;
}

export async function storageSet(id: string, key: string, value: unknown): Promise<void> {
  await daemonFetch<void>(
    `/api/artifacts/${encodeURIComponent(id)}/storage/${encodeURIComponent(key)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value }),
    },
  );
}

export async function storageList(id: string): Promise<string[]> {
  const result = await daemonFetch<{ keys: string[] }>(
    `/api/artifacts/${encodeURIComponent(id)}/storage`,
  );
  return result.keys;
}

/**
 * Build the WebSocket URL for artifact state change streaming.
 */
export function getArtifactStateWsUrl(artifactId: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  // Route through Vite proxy (/vlt → daemon) for dev; direct to daemon for production
  return `${proto}//${window.location.host}/vlt/api/artifacts/ws/${artifactId}/state`;
}
