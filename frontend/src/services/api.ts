import type { Note, NoteSummary, NoteUpdateRequest, NoteCreateRequest } from '@/types/note';
import type { SearchResult, Tag, IndexHealth } from '@/types/search';
import type { User } from '@/types/user';
import type { APIError } from '@/types/auth';

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

/**
 * Get the current bearer token from localStorage
 */
function getAuthToken(): string | null {
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
async function apiFetch<T>(
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

  const response = await fetch(endpoint, {
    ...options,
    headers,
  });

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
export async function listNotes(folder?: string): Promise<NoteSummary[]> {
  const params = new URLSearchParams();
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
export async function getNote(path: string): Promise<Note> {
  const encodedPath = encodeURIComponent(path);
  return apiFetch<Note>(`/api/notes/${encodedPath}`);
}

/**
 * T068: Search notes by query string
 */
export async function searchNotes(query: string): Promise<SearchResult[]> {
  const params = new URLSearchParams({ q: query });
  return apiFetch<SearchResult[]>(`/api/search?${params.toString()}`);
}

/**
 * T069: Get backlinks for a note
 */
export interface BacklinkResult {
  note_path: string;
  title: string;
}

export async function getBacklinks(path: string): Promise<BacklinkResult[]> {
  const encodedPath = encodeURIComponent(path);
  return apiFetch<BacklinkResult[]>(`/api/backlinks/${encodedPath}`);
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
export async function createNote(data: NoteCreateRequest): Promise<Note> {
  return apiFetch<Note>('/api/notes', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateNote(
  path: string,
  data: NoteUpdateRequest
): Promise<Note> {
  const encodedPath = encodeURIComponent(path);
  return apiFetch<Note>(`/api/notes/${encodedPath}`, {
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
export async function getIndexHealth(): Promise<IndexHealth> {
  return apiFetch<IndexHealth>('/api/index/health');
}

/**
 * Trigger a full index rebuild
 */
export interface RebuildResponse {
  status: string;
  notes_indexed: number;
  duration_ms: number;
}

export async function rebuildIndex(): Promise<RebuildResponse> {
  return apiFetch<RebuildResponse>('/api/index/rebuild', {
    method: 'POST',
  });
}

/**
 * Move or rename a note to a new path
 */
export async function moveNote(oldPath: string, newPath: string): Promise<Note> {
  const encodedPath = encodeURIComponent(oldPath);
  return apiFetch<Note>(`/api/notes/${encodedPath}`, {
    method: 'PATCH',
    body: JSON.stringify({ new_path: newPath }),
  });
}

