/**
 * Cronban API — Pipeline-based scheduling and workflow automation.
 * All routes go through /vlt proxy → daemon localhost:8765
 */

// ---------------------------------------------------------------------------
// Library entities
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Pipeline entities
// ---------------------------------------------------------------------------

export interface PipelineStage {
  id: string;
  pipeline_id: string;
  name: string;
  stage_order: number;
  skill_id: string | null;
  prompt_text: string | null;
  gate_id: string | null;
  eval_model: 'haiku' | 'sonnet' | 'opus';
  auto_advance: boolean;
  is_terminal: boolean;
  created_at: string;
  updated_at: string;
}

export interface Pipeline {
  id: string;
  project_id: string | null;
  name: string;
  description: string | null;
  stages: PipelineStage[];
  created_at: string;
  updated_at: string;
}

export interface PipelineCard {
  id: string;
  pipeline_id: string;
  project_id: string | null;
  title: string;
  color: string | null;
  current_stage_id: string;
  status: 'active' | 'completed' | 'paused' | 'failed';
  target_session_id: string | null;
  target_cwd: string | null;
  gate_eval_pending: boolean;
  gate_last_result: { met: boolean; reasoning: string } | null;
  gate_last_checked_at: string | null;
  gate_consecutive_not_met: number;
  fire_count: number;
  last_fired_at: string | null;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// Trigger entities
// ---------------------------------------------------------------------------

export interface CronTrigger {
  id: string;
  project_id: string | null;
  title: string;
  status: 'active' | 'paused' | 'completed';
  color: string | null;
  pipeline_id: string | null;
  skill_id: string | null;
  has_prompt: boolean;
  fire_once: boolean;
  cron_expression: string | null;
  rrule_str: string | null;
  next_fire_at: string | null;
  timezone: string;
  target_session_id: string | null;
  target_cwd: string | null;
  create_new_session: boolean;
  fire_count: number;
  last_fired_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface WebhookListener {
  id: string;
  project_id: string | null;
  name: string;
  status: 'active' | 'paused';
  pipeline_id: string | null;
  skill_id: string | null;
  has_prompt: boolean;
  has_secret: boolean;
  target_session_id: string | null;
  target_cwd: string | null;
  create_new_session: boolean;
  fire_count: number;
  last_fired_at: string | null;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

export interface FireResult {
  fired: boolean;
  session_id: string | null;
  log_id: string | null;
}

export interface GateCheckResult {
  phase: string;
  met: boolean | null;
  reasoning: string;
}

export interface CronbanGateSettings {
  provider: 'claude_code' | 'zai' | 'openrouter' | 'gemini';
  model: string;
  has_api_key: boolean;
  base_url?: string | null;
}

// ---------------------------------------------------------------------------
// HTTP helper
// ---------------------------------------------------------------------------

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
// Pipelines
// ---------------------------------------------------------------------------
export const listPipelines = (projectId?: string) =>
  req<Pipeline[]>(`/pipelines${projectId ? `?project_id=${projectId}` : ''}`);

export const getPipeline = (id: string) => req<Pipeline>(`/pipelines/${id}`);

export const createPipeline = (data: { project_id?: string; name: string; description?: string }) =>
  req<Pipeline>('/pipelines', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const updatePipeline = (id: string, data: Partial<Pick<Pipeline, 'name' | 'description'>>) =>
  req<Pipeline>(`/pipelines/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const deletePipeline = (id: string) =>
  req<{ ok: boolean }>(`/pipelines/${id}`, { method: 'DELETE' });

// ---------------------------------------------------------------------------
// Pipeline Stages
// ---------------------------------------------------------------------------
export const addStage = (pipelineId: string, data: Partial<PipelineStage>) =>
  req<PipelineStage>(`/pipelines/${pipelineId}/stages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const updateStage = (id: string, data: Partial<PipelineStage>) =>
  req<PipelineStage>(`/stages/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const deleteStage = (id: string) =>
  req<{ ok: boolean }>(`/stages/${id}`, { method: 'DELETE' });

export const reorderStages = (pipelineId: string, items: { id: string; stage_order: number }[]) =>
  req<{ ok: boolean }>(`/pipelines/${pipelineId}/stages/reorder`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(items),
  });

// ---------------------------------------------------------------------------
// Pipeline Cards
// ---------------------------------------------------------------------------
export const listCards = (params?: { project_id?: string; pipeline_id?: string; status?: string }) => {
  const p = new URLSearchParams();
  if (params?.project_id) p.set('project_id', params.project_id);
  if (params?.pipeline_id) p.set('pipeline_id', params.pipeline_id);
  if (params?.status) p.set('status', params.status);
  return req<PipelineCard[]>(`/cards?${p}`);
};

export const getCard = (id: string) => req<PipelineCard>(`/cards/${id}`);

export const createCard = (data: {
  pipeline_id: string;
  project_id?: string;
  title: string;
  color?: string;
  stage_id?: string;
  target_session_id?: string;
  target_cwd?: string;
}) =>
  req<PipelineCard>('/cards', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const updateCard = (id: string, data: Partial<PipelineCard>) =>
  req<PipelineCard>(`/cards/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const deleteCard = (id: string) =>
  req<{ ok: boolean }>(`/cards/${id}`, { method: 'DELETE' });

export const advanceCard = (id: string, stageId?: string) =>
  req<PipelineCard>(`/cards/${id}/advance`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(stageId ? { stage_id: stageId } : {}),
  });

// ---------------------------------------------------------------------------
// Cron Triggers
// ---------------------------------------------------------------------------
export const listCronTriggers = (projectId?: string) =>
  req<CronTrigger[]>(`/crons${projectId ? `?project_id=${projectId}` : ''}`);

export const getCronTrigger = (id: string) => req<CronTrigger>(`/crons/${id}`);

export const createCronTrigger = (data: Partial<CronTrigger> & { prompt_text?: string; fire_at?: string; fire_in?: string }) =>
  req<CronTrigger>('/crons', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const updateCronTrigger = (id: string, data: Partial<CronTrigger> & { prompt_text?: string; fire_at?: string; fire_in?: string }) =>
  req<CronTrigger>(`/crons/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const deleteCronTrigger = (id: string) =>
  req<{ ok: boolean }>(`/crons/${id}`, { method: 'DELETE' });

export const fireCronTrigger = (id: string) =>
  req<FireResult>(`/crons/${id}/fire`, { method: 'POST' });

// ---------------------------------------------------------------------------
// Webhook Listeners
// ---------------------------------------------------------------------------
export const listWebhooks = (projectId?: string) =>
  req<WebhookListener[]>(`/webhooks${projectId ? `?project_id=${projectId}` : ''}`);

export const getWebhook = (id: string) => req<WebhookListener>(`/webhooks/${id}`);

export const createWebhook = (data: Partial<WebhookListener> & { prompt_text?: string; webhook_secret?: string }) =>
  req<WebhookListener>('/webhooks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const updateWebhook = (id: string, data: Partial<WebhookListener> & { prompt_text?: string; webhook_secret?: string }) =>
  req<WebhookListener>(`/webhooks/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const deleteWebhook = (id: string) =>
  req<{ ok: boolean }>(`/webhooks/${id}`, { method: 'DELETE' });

export const fireWebhook = (id: string, appendMessage?: string) =>
  req<FireResult>(`/webhooks/${id}/fire`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(appendMessage ? { append_message: appendMessage } : {}),
  });

// ---------------------------------------------------------------------------
// Logs
// ---------------------------------------------------------------------------
export const getFireLogs = (sourceId: string, limit = 20) =>
  req<object[]>(`/logs/${sourceId}?limit=${limit}`);

export const getGateLogs = (cardId: string, limit = 20) =>
  req<object[]>(`/gate-logs/${cardId}?limit=${limit}`);

// ---------------------------------------------------------------------------
// Gate model settings
// ---------------------------------------------------------------------------
export const getGateSettings = () => req<CronbanGateSettings>('/settings/gate-model');

export const saveGateSettings = (data: { provider?: string; model?: string; api_key?: string; base_url?: string }) =>
  req<{ ok: boolean }>('/settings/gate-model', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

export const testGateModel = () =>
  req<{ ok: boolean; result: GateCheckResult }>('/settings/gate-model/test', { method: 'POST' });
