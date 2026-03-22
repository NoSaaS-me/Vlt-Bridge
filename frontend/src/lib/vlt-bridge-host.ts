/**
 * VltBridgeHost — PostMessage handler that runs in the parent page (not the iframe).
 *
 * Listens for `vlt_request` messages from an artifact iframe and routes them to
 * daemon REST endpoints, then posts the response back as `vlt_response`.
 *
 * Message protocol (from iframe):
 *   { type: 'vlt_request', id: string, route: string, params: unknown }
 *
 * Response protocol (to iframe):
 *   { type: 'vlt_response', id: string, ok: true, result: unknown }
 *   { type: 'vlt_response', id: string, ok: false, error: string }
 */

interface VltRequest {
  type: 'vlt_request';
  id: string;
  method?: string;
  route?: string;  // alias for method
  params?: Record<string, unknown>;
}

async function daemonFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  // Pipeline operations (backend/call with test/generate) need longer timeouts
  const isSlowPath = path.includes('/backend/call');
  const defaultTimeout = isSlowPath ? 120_000 : 10_000;
  const response = await fetch(`/vlt${path}`, {
    ...options,
    signal: options.signal ?? AbortSignal.timeout(defaultTimeout),
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail ?? body.message ?? detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  if (response.status === 204) return {} as T;
  return response.json() as Promise<T>;
}

async function backendFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = localStorage.getItem('auth_token') ?? '';
  const response = await fetch(path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers as Record<string, string> ?? {}),
    },
    signal: options.signal ?? AbortSignal.timeout(10000),
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail ?? body.message ?? detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  if (response.status === 204) return {} as T;
  return response.json() as Promise<T>;
}

async function routeRequest(
  artifactId: string,
  route: string,
  params: Record<string, unknown> = {},
): Promise<unknown> {
  switch (route) {
    case 'storage.get': {
      const { key } = params as { key: string };
      const result = await daemonFetch<{ value: unknown }>(
        `/api/artifacts/${encodeURIComponent(artifactId)}/storage/${encodeURIComponent(key)}`,
      );
      return result.value;
    }

    case 'storage.set': {
      const { key, value } = params as { key: string; value: unknown };
      await daemonFetch(
        `/api/artifacts/${encodeURIComponent(artifactId)}/storage/${encodeURIComponent(key)}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value }),
        },
      );
      return { ok: true };
    }

    case 'storage.list': {
      const result = await daemonFetch<{ keys: string[] }>(
        `/api/artifacts/${encodeURIComponent(artifactId)}/storage`,
      );
      return result.keys;
    }

    case 'backend.call': {
      const { action, params: callParams = {} } = params as {
        action: string;
        params?: Record<string, unknown>;
      };
      return daemonFetch(
        `/api/artifacts/${encodeURIComponent(artifactId)}/backend/call`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action, params: callParams }),
        },
      );
    }

    case 'artifact.getState': {
      const artifact = await daemonFetch<{ state: string }>(
        `/api/artifacts/${encodeURIComponent(artifactId)}`,
      );
      return artifact.state;
    }

    case 'artifact.setState': {
      const { state, actor = 'artifact' } = params as { state: string; actor?: string };
      return daemonFetch(
        `/api/artifacts/${encodeURIComponent(artifactId)}/state`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target_state: state, actor }),
        },
      );
    }

    case 'artifact.log': {
      console.log(`[artifact:${artifactId}]`, params.message ?? params);
      return { ok: true };
    }

    case 'notes.read': {
      const { path, projectId } = params as { path: string; projectId?: string };
      const qs = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
      return backendFetch(`/api/notes/${encodeURIComponent(path)}${qs}`);
    }

    case 'notes.write': {
      const { path, content, projectId } = params as {
        path: string;
        content: string;
        projectId?: string;
      };
      const qs = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
      return backendFetch(`/api/notes/${encodeURIComponent(path)}${qs}`, {
        method: 'PUT',
        body: JSON.stringify({ content }),
      });
    }

    case 'notes.search': {
      const { query, projectId } = params as { query: string; projectId?: string };
      const searchParams = new URLSearchParams({ q: query });
      if (projectId) searchParams.set('project_id', projectId);
      return backendFetch(`/api/search?${searchParams}`);
    }

    case 'code.search': {
      const { query, projectId, limit = 10 } = params as {
        query: string;
        projectId?: string;
        limit?: number;
      };
      const searchParams = new URLSearchParams({ q: query, limit: String(limit) });
      if (projectId) searchParams.set('project_id', projectId);
      return daemonFetch(`/api/coderag/search?${searchParams}`);
    }

    case 'code.map': {
      const { projectId, scope } = params as { projectId?: string; scope?: string };
      const searchParams = new URLSearchParams();
      if (projectId) searchParams.set('project_id', projectId);
      if (scope) searchParams.set('scope', scope);
      const qs = searchParams.toString() ? `?${searchParams}` : '';
      return daemonFetch(`/api/coderag/map${qs}`);
    }

    case 'connectors.list': {
      // Proxies through backend to respect per-user auth and connector registry
      return backendFetch('/api/connectors');
    }

    case 'connectors.call': {
      const { connector, instance, action, params: callParams = {} } = params as {
        connector: string;
        instance: string;
        action: string;
        params?: Record<string, unknown>;
      };
      // Daemon validates the connector is declared in the artifact manifest,
      // then forwards to the backend invoke endpoint
      return daemonFetch(
        `/api/artifacts/${encodeURIComponent(artifactId)}/connectors/call`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ connector, instance_id: instance ?? 'default', action, params: callParams }),
        },
      );
    }

    case 'events.emit': {
      const { event_type, payload = {} } = params as {
        event_type: string;
        payload?: Record<string, unknown>;
      };
      return daemonFetch(
        `/api/artifacts/${encodeURIComponent(artifactId)}/events/emit`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ event_type, payload }),
        },
      );
    }

    default:
      throw new Error(`Unknown route: ${route}`);
  }
}

/**
 * Attach a postMessage listener for vlt_request messages from the given iframe.
 * Returns a cleanup function that removes the listener.
 */
export function createBridgeHost(
  artifactId: string,
  iframeEl: HTMLIFrameElement,
): () => void {
  const handler = async (event: MessageEvent) => {
    // Only accept messages from the artifact iframe
    if (event.source !== iframeEl.contentWindow) return;

    const data = event.data as VltRequest;
    if (!data || data.type !== 'vlt_request') return;

    const { id, params = {} } = data;
    const route = data.method ?? data.route ?? '';

    try {
      const result = await routeRequest(artifactId, route, params);
      iframeEl.contentWindow?.postMessage(
        { type: 'vlt_response', id, ok: true, result },
        '*',
      );
    } catch (err) {
      iframeEl.contentWindow?.postMessage(
        {
          type: 'vlt_response',
          id,
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        },
        '*',
      );
    }
  };

  window.addEventListener('message', handler);
  return () => window.removeEventListener('message', handler);
}
