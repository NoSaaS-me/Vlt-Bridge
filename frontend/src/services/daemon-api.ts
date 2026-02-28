/**
 * Daemon API — communicates with the vlt daemon running on localhost:8765.
 * The Vite dev proxy maps /vlt → http://localhost:8765 (stripping /vlt prefix).
 */

export interface AgentSession {
  id: string;
  project_id: string | null;
  name: string;
  cwd: string;
  status: 'idle' | 'thinking' | 'executing' | 'dead';
  model: string | null;
  ctx_pct: number | null;
  pid: number;
  bypass_perms: boolean;
  source: string;
  created_at: string;
  last_activity: string;
}

export interface DaemonHealth {
  status: string;
  uptime_seconds: number;
  backend_connected: boolean;
  queue_size: number;
}

/**
 * List all active (non-dead) Claude Code agent sessions.
 */
export async function listSessions(): Promise<AgentSession[]> {
  const response = await fetch('/vlt/api/sessions', { signal: AbortSignal.timeout(4000) });
  if (!response.ok) throw new Error(`Daemon error: ${response.status} ${response.statusText}`);
  const data = await response.json();
  // API returns array directly
  return Array.isArray(data) ? data : (data.sessions ?? []);
}

/**
 * Inject text into a running PTY relay session.
 */
export async function injectToSession(
  sessionId: string,
  text: string,
  pressEnter = true,
): Promise<void> {
  const response = await fetch(`/vlt/api/sessions/${sessionId}/inject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, press_enter: pressEnter }),
    signal: AbortSignal.timeout(4000),
  });
  if (!response.ok) throw new Error(`Inject failed: ${response.status}`);
}

/**
 * Get daemon health status.
 */
export async function getDaemonHealth(): Promise<DaemonHealth> {
  const response = await fetch('/vlt/health', { signal: AbortSignal.timeout(2000) });
  if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
  return response.json();
}

export interface HookEvent {
  id: string;
  ts: string;        // naive UTC ISO (no Z) from daemon
  event: string;
  session_id: string;
  cwd: string;
  status: string | null;
}

/**
 * Fetch recent Claude Code hook events.
 */
export async function listHookEvents(limit = 100): Promise<HookEvent[]> {
  const response = await fetch(`/vlt/api/hooks/recent?limit=${limit}`, {
    signal: AbortSignal.timeout(4000),
  });
  if (!response.ok) throw new Error(`Events fetch failed: ${response.status}`);
  const data = await response.json();
  return data.events ?? [];
}

/**
 * Parse daemon timestamp strings (naive UTC ISO, no Z suffix) as UTC Date.
 */
export function parseDaemonTs(isoStr: string): Date {
  return new Date(isoStr.endsWith('Z') ? isoStr : isoStr + 'Z');
}

// ---------------------------------------------------------------------------
// Transcript API
// ---------------------------------------------------------------------------

export interface TranscriptEntry {
  lineIndex: number;
  type: string;
  message?: { role: string; content: string | ContentBlock[] };
  timestamp?: string;
  uuid?: string;
  sessionId?: string;
  cwd?: string;
  raw: Record<string, unknown>;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ContentBlock = { type: string; [key: string]: any };

export interface TranscriptResponse {
  path: string;
  entries: TranscriptEntry[];
  total_lines: number;
}

/**
 * Fetch parsed JSONL transcript for a session.
 */
export async function fetchTranscript(
  sessionId: string,
  types?: string[],
): Promise<TranscriptResponse> {
  const params = types?.length ? `?types=${types.join(',')}` : '';
  const response = await fetch(`/vlt/api/sessions/${sessionId}/transcript${params}`, {
    signal: AbortSignal.timeout(10000),
  });
  if (!response.ok) throw new Error(`Transcript fetch failed: ${response.status}`);
  return response.json();
}

/**
 * Write back a modified transcript (full JSONL replacement).
 */
export async function saveTranscript(
  sessionId: string,
  entries: Record<string, unknown>[],
): Promise<void> {
  const response = await fetch(`/vlt/api/sessions/${sessionId}/transcript`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entries }),
    signal: AbortSignal.timeout(15000),
  });
  if (!response.ok) throw new Error(`Transcript save failed: ${response.status}`);
}
