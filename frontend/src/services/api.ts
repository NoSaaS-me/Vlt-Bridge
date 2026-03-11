import type { Note, NoteSummary, NoteUpdateRequest, NoteCreateRequest, WikilinkResolution, NotePreview } from '@/types/note';
import type { SearchResult, Tag, IndexHealth } from '@/types/search';
import type { User } from '@/types/user';
import type { APIError } from '@/types/auth';
import type { AssetSummary, AssetMetadata, AssetUploadResponse } from '@/types/asset';
// Token refresh callback — set by auth.ts to avoid circular import.
// When a 401 occurs on a demo session, apiFetch calls this to get a fresh token.
let _tokenRefresher: (() => Promise<string | null>) | null = null;
let _isDemoSession: (() => boolean) | null = null;

export function registerTokenRefresher(
  refresher: () => Promise<string | null>,
  isDemoCheck: () => boolean,
): void {
  _tokenRefresher = refresher;
  _isDemoSession = isDemoCheck;
}

/**
 * Custom error class for API errors
 */
export class APIException extends Error {
  status: number;
  error: string;
  detail?: Record<string, unknown>;

  constructor(
    status: number,
    message: string,
    errorCode?: string,
    detail?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'APIException';
    this.status = status;
    this.error = errorCode || message;
    this.detail = detail;
  }
}

declare global {
  interface Window {
    API_BASE_URL?: string;
  }
}

/**
 * Get the current bearer token from localStorage
 */
export function getAuthToken(): string | null {
  return localStorage.getItem('auth_token');
}

/**
 * Set the bearer token in localStorage
 */
export function setAuthToken(token: string): void {
  localStorage.setItem('auth_token', token);
}

/**
 * Clear the bearer token from localStorage
 */
export function clearAuthToken(): void {
  localStorage.removeItem('auth_token');
}

/**
 * Base fetch wrapper with authentication and error handling
 */
export async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getAuthToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  // Handle absolute URL for widget embedding.
  // Security: Validate that the configured API_BASE_URL shares the same origin
  // as the current page before using it. An attacker-controlled parent frame
  // could set window.API_BASE_URL to an external origin, causing Bearer tokens
  // to be sent cross-origin (credential exfiltration).
  let url = endpoint;
  if (endpoint.startsWith('/') && window.API_BASE_URL) {
    try {
      const configuredBase = new URL(window.API_BASE_URL);
      const currentOrigin = window.location.origin;
      // Only use the configured base if it matches the current origin
      if (configuredBase.origin === currentOrigin) {
        url = `${window.API_BASE_URL}${endpoint}`;
      }
      // If origins differ, fall through to use the relative URL (same-origin fetch)
    } catch {
      // Malformed API_BASE_URL — ignore it and use relative path
    }
  }

  let response = await fetch(url, {
    ...options,
    headers,
  });

  // Auto-refresh stale demo tokens on 401 and retry once
  if (response.status === 401 && _isDemoSession?.()) {
    const newToken = await _tokenRefresher?.() ?? null;
    if (newToken) {
      headers['Authorization'] = `Bearer ${newToken}`;
      response = await fetch(url, { ...options, headers });
    }
  }

  if (!response.ok) {
    let errorData: APIError;
    try {
      errorData = await response.json();
    } catch {
      errorData = {
        error: 'Unknown error',
        message: `HTTP ${response.status}: ${response.statusText}`,
      };
    }
    const message = errorData.message || `HTTP ${response.status}`;
    const code = errorData.error || message;
    throw new APIException(response.status, message, code, errorData.detail);
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

/**
 * T066: List all notes with optional folder filtering
 */
import type { GraphData } from '@/types/graph';

export async function getGraphData(projectId?: string): Promise<GraphData> {
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  const query = params.toString();
  return apiFetch<GraphData>(query ? `/api/graph?${query}` : '/api/graph');
}

export async function listNotes(projectId?: string, folder?: string): Promise<NoteSummary[]> {
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  if (folder) {
    params.set('folder', folder);
  }
  const query = params.toString();
  const endpoint = query ? `/api/notes?${query}` : '/api/notes';
  return apiFetch<NoteSummary[]>(endpoint);
}

/**
 * T067: Get a single note by path
 */
export async function getNote(path: string, projectId?: string): Promise<Note> {
  const encodedPath = encodeURIComponent(path);
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  const query = params.toString();
  return apiFetch<Note>(`/api/notes/${encodedPath}${query ? `?${query}` : ''}`);
}

/**
 * T068: Search notes by query string
 */
export async function searchNotes(query: string, projectId?: string): Promise<SearchResult[]> {
  const params = new URLSearchParams({ q: query });
  if (projectId) {
    params.set('project_id', projectId);
  }
  return apiFetch<SearchResult[]>(`/api/search?${params.toString()}`);
}

/**
 * T069: Get backlinks for a note
 */
export interface BacklinkResult {
  note_path: string;
  title: string;
}

export async function getBacklinks(path: string, projectId?: string): Promise<BacklinkResult[]> {
  const encodedPath = encodeURIComponent(path);
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  const query = params.toString();
  return apiFetch<BacklinkResult[]>(`/api/backlinks/${encodedPath}${query ? `?${query}` : ''}`);
}

export interface DemoTokenResponse {
  token: string;
  token_type: string;
  expires_at: string;
  user_id: string;
}

export async function getDemoToken(): Promise<DemoTokenResponse> {
  return apiFetch<DemoTokenResponse>('/api/demo/token');
}

/**
 * T070: Get all tags with counts
 */
export async function getTags(): Promise<Tag[]> {
  return apiFetch<Tag[]>('/api/tags');
}

/**
 * T071: Create a note
 */
export async function createNote(data: NoteCreateRequest, projectId?: string): Promise<Note> {
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  const query = params.toString();
  return apiFetch<Note>(`/api/notes${query ? `?${query}` : ''}`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateNote(
  path: string,
  data: NoteUpdateRequest,
  projectId?: string
): Promise<Note> {
  const encodedPath = encodeURIComponent(path);
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  const query = params.toString();
  return apiFetch<Note>(`/api/notes/${encodedPath}${query ? `?${query}` : ''}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

/**
 * Get current user information
 */
export async function getCurrentUser(): Promise<User> {
  return apiFetch<User>('/api/me');
}

/**
 * Get index health information
 */
export async function getIndexHealth(projectId?: string): Promise<IndexHealth> {
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  const query = params.toString();
  return apiFetch<IndexHealth>(`/api/index/health${query ? `?${query}` : ''}`);
}

/**
 * Trigger a full index rebuild
 */
export interface RebuildResponse {
  status: string;
  notes_indexed: number;
  duration_ms: number;
}

export async function rebuildIndex(projectId?: string): Promise<RebuildResponse> {
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  const query = params.toString();
  return apiFetch<RebuildResponse>(`/api/index/rebuild${query ? `?${query}` : ''}`, {
    method: 'POST',
  });
}

/**
 * Retrieve system logs
 */
export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  extra: Record<string, any>;
}

export async function getLogs(): Promise<LogEntry[]> {
  return apiFetch<LogEntry[]>('/api/system/logs');
}

/**
 * Move or rename a note to a new path
 */
export async function moveNote(oldPath: string, newPath: string, projectId?: string): Promise<Note> {
  const encodedPath = encodeURIComponent(oldPath);
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  const query = params.toString();
  return apiFetch<Note>(`/api/notes/${encodedPath}${query ? `?${query}` : ''}`, {
    method: 'PATCH',
    body: JSON.stringify({ new_path: newPath }),
  });
}

/**
 * Delete a note
 */
export async function deleteNote(path: string, projectId?: string): Promise<void> {
  const encodedPath = encodeURIComponent(path);
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  const query = params.toString();
  await apiFetch<void>(`/api/notes/${encodedPath}${query ? `?${query}` : ''}`, {
    method: 'DELETE',
  });
}

/**
 * Resolve a wikilink to its target note path
 */
export async function resolveWikilink(
  linkText: string,
  contextPath?: string
): Promise<WikilinkResolution> {
  const params = new URLSearchParams({ link: linkText });
  if (contextPath) {
    params.set('context', contextPath);
  }
  return apiFetch<WikilinkResolution>(`/api/wikilinks/resolve?${params.toString()}`);
}

/**
 * Get lightweight preview data for a note
 */
export async function getNotePreview(path: string): Promise<NotePreview> {
  const encodedPath = encodeURIComponent(path);
  return apiFetch<NotePreview>(`/api/notes/${encodedPath}/preview`);
}

/**
 * Thread API functions
 */
import type { ThreadListResponse, Thread } from '@/types/thread';

/**
 * List threads for a project
 */
export async function listThreads(projectId?: string): Promise<ThreadListResponse> {
  const params = new URLSearchParams();
  if (projectId) {
    params.set('project_id', projectId);
  }
  const query = params.toString();
  return apiFetch<ThreadListResponse>(`/api/threads${query ? `?${query}` : ''}`);
}

/**
 * Get a single thread with entries
 */
export async function getThread(threadId: string): Promise<Thread> {
  return apiFetch<Thread>(`/api/threads/${encodeURIComponent(threadId)}`);
}

/**
 * Oracle settings API functions
 */
export interface OracleSettings {
  oracle_mcp_enabled: boolean;
}

export async function getOracleSettings(): Promise<OracleSettings> {
  return apiFetch<OracleSettings>('/api/settings/oracle');
}

export async function updateOracleSettings(enabled: boolean): Promise<OracleSettings> {
  return apiFetch<OracleSettings>('/api/settings/oracle', {
    method: 'PUT',
    body: JSON.stringify({ oracle_mcp_enabled: enabled }),
  });
}

// ============ Oracle V2 Thread & Memory API ============

export interface OracleThreadSummary {
  thread_id: string;
  project_id: string | null;
  title: string | null;
  created_at: string;
  last_active_at: string;
}

export interface OracleThreadListResponse {
  threads: OracleThreadSummary[];
  total: number;
}

export interface OracleThreadHistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface OracleThreadHistoryResponse {
  thread_id: string;
  messages: OracleThreadHistoryMessage[];
}

export interface OracleMemorySearchResponse {
  facts: string[];
  available: boolean;
  message?: string;
  error?: string;
}

export async function listOracleThreads(
  limit = 50,
  offset = 0,
): Promise<OracleThreadListResponse> {
  return apiFetch<OracleThreadListResponse>(
    `/api/oracle/threads?limit=${limit}&offset=${offset}`,
  );
}

export async function getOracleThreadHistory(
  threadId: string,
): Promise<OracleThreadHistoryResponse> {
  return apiFetch<OracleThreadHistoryResponse>(
    `/api/oracle/threads/${encodeURIComponent(threadId)}/history`,
  );
}

export async function deleteOracleThread(threadId: string): Promise<void> {
  await apiFetch<void>(`/api/oracle/threads/${encodeURIComponent(threadId)}`, {
    method: 'DELETE',
  });
}

export async function renameOracleThread(
  threadId: string,
  title: string,
): Promise<OracleThreadSummary> {
  return apiFetch<OracleThreadSummary>(
    `/api/oracle/threads/${encodeURIComponent(threadId)}`,
    {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    },
  );
}

export async function searchOracleMemory(
  query: string,
  projectId = 'default',
  limit = 20,
): Promise<OracleMemorySearchResponse> {
  const params = new URLSearchParams({ query, project_id: projectId, limit: String(limit) });
  return apiFetch<OracleMemorySearchResponse>(`/api/oracle/memory?${params}`);
}

// ============ Asset API ============

export async function listAssets(folder?: string, projectId?: string): Promise<AssetSummary[]> {
  const params = new URLSearchParams();
  if (folder) params.set('folder', folder);
  if (projectId) params.set('project_id', projectId);
  const qs = params.toString() ? `?${params}` : '';
  return apiFetch<AssetSummary[]>(`/api/assets${qs}`);
}

export async function getAssetMetadata(path: string, projectId?: string): Promise<AssetMetadata> {
  const params = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
  return apiFetch<AssetMetadata>(`/api/assets/${encodeURIComponent(path)}/metadata${params}`);
}

export async function deleteAsset(path: string, projectId?: string): Promise<void> {
  const params = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
  await apiFetch<void>(`/api/assets/${encodeURIComponent(path)}${params}`, { method: 'DELETE' });
}

export async function moveAsset(path: string, newPath: string, projectId?: string): Promise<void> {
  const params = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
  await apiFetch<void>(`/api/assets/${encodeURIComponent(path)}${params}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ new_path: newPath }),
  });
}

/**
 * Returns a URL for direct asset access (image/video/audio src attributes).
 * Uses ?token= query param instead of Authorization header because HTML media
 * elements (<img>, <video>, <audio>) cannot set custom request headers.
 * NOTE: The backend /api/assets/{path} route must accept auth via BOTH
 * the Authorization Bearer header AND the ?token= query param to support
 * this use-case.
 */
export function getAssetUrl(path: string, projectId?: string): string {
  // Use relative URL in dev (Vite proxies /api → backend); use API_BASE_URL in production.
  const base = window.API_BASE_URL || '';
  const token = localStorage.getItem('auth_token') || '';
  const params = new URLSearchParams({ token });
  if (projectId) params.set('project_id', projectId);
  return `${base}/api/assets/${encodeURIComponent(path)}?${params}`;
}

export async function uploadAsset(
  file: File,
  targetPath: string,
  projectId?: string,
  onProgress?: (pct: number) => void
): Promise<AssetUploadResponse> {
  // Use relative URL in dev (Vite proxies /api → backend); use API_BASE_URL in production.
  const base = window.API_BASE_URL || '';
  const token = localStorage.getItem('auth_token') || '';
  const formData = new FormData();
  formData.append('file', file);
  formData.append('path', targetPath);
  if (projectId) formData.append('project_id', projectId);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${base}/api/assets/upload`);
    xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
      };
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`));
      }
    };
    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.send(formData);
  });
}