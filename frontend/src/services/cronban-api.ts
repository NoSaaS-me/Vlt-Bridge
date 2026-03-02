/**
 * Cronban API — scheduling, kanban, and skills.
 * All routes go through /vlt proxy → daemon localhost:8765
 */

export interface CronbanGate {
  id: string;
  project_id: string | null;
  name: string;
  description: string | null;
  prompt_markdown: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface CronbanSkill {
  id: string;
  project_id: string | null;
  name: string;
  description: string | null;
  prompt_markdown: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface KanbanColumn {
  id: string;
  project_id: string;
  name: string;
  col_order: number;
  color: string | null;
  is_terminal: boolean;
  auto_graduate: boolean;
  graduation_column_id: string | null;
  created_at: string;
}

export interface CronbanEntry {
  id: string;
  project_id: string | null;
  title: string;
  entry_type: 'cron' | 'gate' | 'webhook';
  status: 'active' | 'paused' | 'completed' | 'failed';
  color: string | null;
  // Skill
  skill_id: string | null;
  prompt_text: string | null;
  // Eval is never returned in full — just has_eval flag
  has_eval: boolean;
  gate_id: string | null;
  eval_model: 'haiku' | 'sonnet' | 'opus';
  gate_eval_pending: boolean;
  // Target
  target_session_id: string | null;
  target_cwd: string | null;
  create_new_session: boolean;
  // Schedule
  cron_expression: string | null;
  rrule_str: string | null;
  next_fire_at: string | null;
  timezone: string;
  // Gate
  kanban_column_id: string | null;
  kanban_order: number;
  gate_check_interval_minutes: number;
  gate_last_checked_at: string | null;
  gate_last_result: { met: boolean; reasoning: string } | null;
  gate_consecutive_not_met: number;
  gate_question_injected_at: string | null;
  // Webhook
  has_webhook_secret: boolean;
  // Stats
  fire_count: number;
  last_fired_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface GateCheckResult {
  phase: string;
  met: boolean | null;
  reasoning: string;
}

export interface CronbanGateSettings {
  provider: 'openrouter' | 'gemini';
  model: string;
  has_api_key: boolean;
}

const BASE = '/vlt/api/cronban';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { signal: AbortSignal.timeout(8000), ...init });
  if (!r.ok) throw new Error(`Cronban API ${r.status}: ${await r.text()}`);
  return r.json();
}

// ---------------------------------------------------------------------------
// Skills
// ---------------------------------------------------------------------------
export const listSkills = (projectId?: string) =>
  req<CronbanSkill[]>(`/skills${projectId ? `?project_id=${projectId}` : ''}`);

export const getSkill = (id: string) => req<CronbanSkill>(`/skills/${id}`);

export const createSkill = (data: Partial<CronbanSkill>) =>
  req<CronbanSkill>('/skills', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const updateSkill = (id: string, data: Partial<CronbanSkill>) =>
  req<CronbanSkill>(`/skills/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const deleteSkill = (id: string) =>
  req<{ ok: boolean }>(`/skills/${id}`, { method: 'DELETE' });

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------
export const listGates = (projectId?: string) =>
  req<CronbanGate[]>(`/gates${projectId ? `?project_id=${projectId}` : ''}`);

export const getGate = (id: string) => req<CronbanGate>(`/gates/${id}`);

export const createGate = (data: Partial<CronbanGate>) =>
  req<CronbanGate>('/gates', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const updateGate = (id: string, data: Partial<CronbanGate>) =>
  req<CronbanGate>(`/gates/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const deleteGate = (id: string) =>
  req<{ ok: boolean }>(`/gates/${id}`, { method: 'DELETE' });

// ---------------------------------------------------------------------------
// Columns
// ---------------------------------------------------------------------------
export const listColumns = (projectId?: string) =>
  req<KanbanColumn[]>(`/columns${projectId ? `?project_id=${projectId}` : ''}`);

export const createColumn = (data: Partial<KanbanColumn>) =>
  req<KanbanColumn>('/columns', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const updateColumn = (id: string, data: Partial<KanbanColumn>) =>
  req<KanbanColumn>(`/columns/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const deleteColumn = (id: string, moveTo?: string) =>
  req<{ ok: boolean }>(`/columns/${id}${moveTo ? `?move_to=${moveTo}` : ''}`, { method: 'DELETE' });

// ---------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------
export const listEntries = (params?: { project_id?: string; entry_type?: string; status?: string }) => {
  const p = new URLSearchParams();
  if (params?.project_id) p.set('project_id', params.project_id);
  if (params?.entry_type) p.set('entry_type', params.entry_type);
  if (params?.status) p.set('status', params.status);
  return req<CronbanEntry[]>(`/entries?${p}`);
};

export const getEntry = (id: string) => req<CronbanEntry>(`/entries/${id}`);

export const createEntry = (data: Partial<CronbanEntry> & { eval_text?: string; gate_id?: string }) =>
  req<CronbanEntry>('/entries', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const updateEntry = (id: string, data: Partial<CronbanEntry> & { eval_text?: string; gate_id?: string }) =>
  req<CronbanEntry>(`/entries/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const deleteEntry = (id: string) =>
  req<{ ok: boolean }>(`/entries/${id}`, { method: 'DELETE' });

export const moveKanbanCard = (entryId: string, columnId: string, force = false) =>
  req<{ moved: boolean; requires_gate_check?: boolean; message?: string }>(
    `/entries/${entryId}/move`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ column_id: columnId, force }),
    },
  );

// ---------------------------------------------------------------------------
// Gate check
// ---------------------------------------------------------------------------
export const runGateCheck = (entryId: string) =>
  req<GateCheckResult>(`/entries/${entryId}/gate-check`, { method: 'POST' });

// ---------------------------------------------------------------------------
// Logs
// ---------------------------------------------------------------------------
export const getFireLogs = (entryId: string, limit = 20) =>
  req<object[]>(`/logs/${entryId}?limit=${limit}`);

export const getGateLogs = (entryId: string, limit = 20) =>
  req<object[]>(`/gate-logs/${entryId}?limit=${limit}`);

// ---------------------------------------------------------------------------
// Gate model settings
// ---------------------------------------------------------------------------
export const getGateSettings = () => req<CronbanGateSettings>('/settings/gate-model');

export const saveGateSettings = (data: { provider?: string; model?: string; api_key?: string }) =>
  req<{ ok: boolean }>('/settings/gate-model', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const testGateModel = () =>
  req<{ ok: boolean; result: GateCheckResult }>('/settings/gate-model/test', { method: 'POST' });
