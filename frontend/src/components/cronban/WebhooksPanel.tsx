/**
 * WebhooksPanel — Manage WebhookListeners.
 *
 * Each webhook exposes a POST endpoint at /api/cronban/webhooks/{id}/fire
 * that can be triggered by external systems (CI/CD, error handlers, etc.).
 * Optional HMAC-SHA256 auth via webhook_secret.
 *
 * Connector-bound webhooks additionally expose a backend endpoint at
 * /api/webhooks/connector/{id} that verifies HMAC, applies pattern filters,
 * and injects event variables into the prompt before forwarding to the daemon.
 *
 * Layout:
 *   - Left: list of webhook listeners
 *   - Right: editor (name, skill/prompt, pipeline, secret, target session,
 *             connector binding, pattern filter)
 */
import { useState, useEffect, useCallback } from 'react';
import { Plus, Webhook, Trash2, Zap, Copy, Eye, EyeOff, Plug } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  type WebhookListener,
  type CronbanSkill,
  listWebhooks,
  createWebhook,
  updateWebhook,
  deleteWebhook,
  fireWebhook,
  listSkills,
} from '@/services/cronban-api';
import { type ConnectorInfo, listConnectors } from '@/services/connectors';

// ---------------------------------------------------------------------------
// Webhook list item
// ---------------------------------------------------------------------------
function WebhookListItem({
  webhook,
  isSelected,
  onSelect,
  onDelete,
  onFire,
}: {
  webhook: WebhookListener;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onFire: () => void;
}) {
  return (
    <div
      className={cn(
        'group relative flex items-start gap-2 rounded-md p-2.5 cursor-pointer transition-colors',
        isSelected ? 'bg-muted/60 text-foreground' : 'hover:bg-muted/30 text-muted-foreground',
      )}
      onClick={onSelect}
    >
      <Webhook className={cn('h-3.5 w-3.5 shrink-0 mt-0.5', webhook.status === 'active' ? 'text-blue-400' : 'text-muted-foreground/40')} />
      <div className="flex-1 min-w-0">
        <p className={cn('text-xs font-medium truncate', isSelected && 'text-foreground')}>
          {webhook.name}
        </p>
        <p className="text-[10px] text-muted-foreground mt-0.5">
          {webhook.fire_count > 0 ? `Fired ${webhook.fire_count}×` : 'Never fired'}
          {webhook.has_secret && ' · 🔒'}
          {webhook.connector_name && ` · ${webhook.connector_name}`}
          {webhook.pipeline_id && ' · pipeline'}
        </p>
      </div>
      <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
        <button
          onClick={(e) => { e.stopPropagation(); onFire(); }}
          className="p-1 rounded text-muted-foreground hover:text-blue-400 transition-colors"
          title="Fire now"
        >
          <Zap className="h-3 w-3" />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="p-1 rounded text-muted-foreground hover:text-destructive transition-colors"
          title="Delete"
        >
          <Trash2 className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pattern condition row
// ---------------------------------------------------------------------------
interface Condition {
  field: string;
  operator: string;
  value: string;
}

const OPERATORS = [
  'equals', 'contains', 'starts_with', 'ends_with',
  'glob', 'regex', 'not_equals', 'not_contains',
] as const;

function ConditionRow({
  condition,
  fieldHints,
  onChange,
  onRemove,
}: {
  condition: Condition;
  fieldHints: string[];
  onChange: (c: Condition) => void;
  onRemove: () => void;
}) {
  return (
    <div className="flex items-center gap-1.5 text-xs">
      <input
        list="field-hints"
        value={condition.field}
        onChange={(e) => onChange({ ...condition, field: e.target.value })}
        placeholder="field"
        className="flex-1 min-w-0 h-7 rounded border border-border bg-background px-2 text-xs"
      />
      <datalist id="field-hints">
        {fieldHints.map((f) => <option key={f} value={f} />)}
      </datalist>
      <select
        value={condition.operator}
        onChange={(e) => onChange({ ...condition, operator: e.target.value })}
        className="h-7 rounded border border-border bg-background px-1.5 text-xs"
      >
        {OPERATORS.map((op) => <option key={op} value={op}>{op}</option>)}
      </select>
      <Input
        value={condition.value}
        onChange={(e) => onChange({ ...condition, value: e.target.value })}
        placeholder="value"
        className="flex-1 min-w-0 h-7 text-xs"
      />
      <button
        onClick={onRemove}
        className="p-1 rounded text-muted-foreground hover:text-destructive transition-colors shrink-0"
        title="Remove condition"
      >
        ×
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Webhook editor
// ---------------------------------------------------------------------------
function WebhookEditor({
  webhook,
  skills,
  connectors,
  onSave,
}: {
  webhook: WebhookListener;
  skills: CronbanSkill[];
  connectors: ConnectorInfo[];
  onSave: (data: Partial<WebhookListener> & { webhook_secret?: string; prompt_text?: string }) => void;
}) {
  const [name, setName] = useState(webhook.name);
  const [status, setStatus] = useState<'active' | 'paused'>(webhook.status);
  const [skillId, setSkillId] = useState(webhook.skill_id ?? '');
  const [secretDraft, setSecretDraft] = useState('');
  const [showSecret, setShowSecret] = useState(false);
  const [connectorName, setConnectorName] = useState(webhook.connector_name ?? '');
  const [backendUserId, setBackendUserId] = useState(webhook.backend_user_id ?? '');
  const [conditions, setConditions] = useState<Condition[]>(() => {
    if (!webhook.pattern_filter_json) return [];
    try {
      const parsed = JSON.parse(webhook.pattern_filter_json);
      return parsed.conditions ?? [];
    } catch { return []; }
  });
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    setName(webhook.name);
    setStatus(webhook.status);
    setSkillId(webhook.skill_id ?? '');
    setSecretDraft('');
    setConnectorName(webhook.connector_name ?? '');
    setBackendUserId(webhook.backend_user_id ?? '');
    try {
      const parsed = webhook.pattern_filter_json ? JSON.parse(webhook.pattern_filter_json) : null;
      setConditions(parsed?.conditions ?? []);
    } catch { setConditions([]); }
    setDirty(false);
  }, [webhook.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const mark = () => setDirty(true);

  // Field hints from selected connector's webhook_events
  const selectedConnector = connectors.find((c) => c.name === connectorName);
  const fieldHints: string[] = (selectedConnector as any)?.webhook_events?.flatMap((e: any) => e.fields ?? []) ?? [];

  const handleSave = () => {
    const patternFilterJson = conditions.length > 0
      ? JSON.stringify({ conditions })
      : null;

    const data: Parameters<typeof onSave>[0] = {
      name,
      status,
      skill_id: skillId || undefined,
      connector_name: connectorName || null,
      backend_user_id: backendUserId || null,
      pattern_filter_json: patternFilterJson,
    };
    if (secretDraft) data.webhook_secret = secretDraft;
    onSave(data);
    setDirty(false);
  };

  // Fire endpoint — manual
  const manualEndpointPath = `/vlt/api/cronban/webhooks/${webhook.id}/fire`;
  // Connector webhook endpoint — hits the backend for HMAC verify + pattern filter
  const connectorEndpointPath = `/api/webhooks/connector/${webhook.id}`;

  return (
    <div className="flex-1 p-4 space-y-4 overflow-y-auto">
      <div className="space-y-3">
        {/* Name */}
        <div className="space-y-1">
          <label className="text-xs text-muted-foreground">Name</label>
          <Input
            value={name}
            onChange={(e) => { setName(e.target.value); mark(); }}
            className="h-8 text-sm"
          />
        </div>

        {/* Status toggle */}
        <div className="flex items-center gap-3">
          <label className="text-xs text-muted-foreground">Status</label>
          <div className="flex gap-2">
            {(['active', 'paused'] as const).map((s) => (
              <button
                key={s}
                onClick={() => { setStatus(s); mark(); }}
                className={cn(
                  'px-2.5 py-1 rounded text-xs border transition-colors',
                  status === s
                    ? s === 'active'
                      ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400'
                      : 'bg-amber-500/20 border-amber-500/40 text-amber-400'
                    : 'border-border text-muted-foreground hover:bg-muted/30',
                )}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Skill */}
        <div className="space-y-1">
          <label className="text-xs text-muted-foreground">Skill</label>
          <select
            value={skillId}
            onChange={(e) => { setSkillId(e.target.value); mark(); }}
            className="w-full h-8 rounded border border-border bg-background text-sm px-2"
          >
            <option value="">— none / use inline prompt —</option>
            {skills.map((s) => (
              <option key={s.id} value={s.id}>{s.name}</option>
            ))}
          </select>
        </div>

        {/* Secret */}
        <div className="space-y-1">
          <label className="text-xs text-muted-foreground flex items-center gap-1">
            Webhook secret{' '}
            {webhook.has_secret && (
              <span className="text-[9px] text-emerald-400 bg-emerald-500/15 px-1.5 py-0.5 rounded">
                set
              </span>
            )}
          </label>
          <div className="flex gap-1">
            <Input
              type={showSecret ? 'text' : 'password'}
              value={secretDraft}
              onChange={(e) => { setSecretDraft(e.target.value); mark(); }}
              placeholder={webhook.has_secret ? '(leave blank to keep existing)' : 'Enter to set…'}
              className="h-8 text-sm font-mono flex-1"
            />
            <Button
              size="sm"
              variant="ghost"
              className="h-8 w-8 p-0"
              onClick={() => setShowSecret((v) => !v)}
            >
              {showSecret ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
            </Button>
          </div>
        </div>

        {/* ── Connector Binding ─────────────────────────────────────── */}
        <div className="border border-border rounded-md p-3 space-y-3">
          <div className="flex items-center gap-1.5">
            <Plug className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Connector binding
            </span>
          </div>

          {/* Connector selector */}
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Connector</label>
            <select
              value={connectorName}
              onChange={(e) => { setConnectorName(e.target.value); setConditions([]); mark(); }}
              className="w-full h-8 rounded border border-border bg-background text-sm px-2"
            >
              <option value="">— none (generic webhook) —</option>
              {connectors
                .filter((c) => c.connector_type !== 'service')
                .map((c) => (
                  <option key={c.name} value={c.name}>{c.display_name}</option>
                ))}
            </select>
            {connectorName && (
              <p className="text-[10px] text-muted-foreground">
                Webhook URL shown below handles HMAC verification + pattern filtering.
              </p>
            )}
          </div>

          {/* Backend user ID (only shown when connector is set) */}
          {connectorName && (
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Backend user ID</label>
              <Input
                value={backendUserId}
                onChange={(e) => { setBackendUserId(e.target.value); mark(); }}
                placeholder="local-dev"
                className="h-8 text-sm font-mono"
              />
              <p className="text-[10px] text-muted-foreground">
                The user whose connector credentials are used for HMAC verification.
              </p>
            </div>
          )}

          {/* Pattern filter (only shown when connector is set) */}
          {connectorName && (
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground">
                Pattern filter{' '}
                <span className="text-muted-foreground/60">(all conditions AND-matched)</span>
              </label>
              <div className="space-y-1.5">
                {conditions.map((cond, i) => (
                  <ConditionRow
                    key={i}
                    condition={cond}
                    fieldHints={fieldHints}
                    onChange={(c) => { const next = [...conditions]; next[i] = c; setConditions(next); mark(); }}
                    onRemove={() => { setConditions(conditions.filter((_, j) => j !== i)); mark(); }}
                  />
                ))}
              </div>
              <Button
                size="sm"
                variant="outline"
                className="h-7 text-xs gap-1"
                onClick={() => { setConditions([...conditions, { field: '', operator: 'contains', value: '' }]); mark(); }}
              >
                <Plus className="h-3 w-3" />
                Add condition
              </Button>
              {fieldHints.length > 0 && (
                <p className="text-[10px] text-muted-foreground">
                  Available fields: {fieldHints.join(', ')}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Endpoint URLs */}
        <div className="space-y-2">
          {connectorName ? (
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Connector webhook URL</label>
              <div className="flex items-center gap-1.5 bg-blue-500/5 border border-blue-500/20 rounded px-2.5 py-1.5">
                <code className="text-[10px] font-mono text-blue-400 flex-1 truncate">
                  POST {connectorEndpointPath}
                </code>
                <button
                  onClick={() => navigator.clipboard.writeText(connectorEndpointPath)}
                  className="text-muted-foreground hover:text-foreground transition-colors"
                  title="Copy path"
                >
                  <Copy className="h-3 w-3" />
                </button>
              </div>
              <p className="text-[10px] text-muted-foreground">
                Configure this URL in your connector's webhook settings (e.g. Mailgun Routes).
                Handles HMAC verification, pattern filtering, and variable injection automatically.
              </p>
            </div>
          ) : null}

          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">
              {connectorName ? 'Manual fire endpoint' : 'Fire endpoint'}
            </label>
            <div className="flex items-center gap-1.5 bg-muted/30 rounded border border-border px-2.5 py-1.5">
              <code className="text-[10px] font-mono text-muted-foreground flex-1 truncate">
                POST {manualEndpointPath}
              </code>
              <button
                onClick={() => navigator.clipboard.writeText(manualEndpointPath)}
                className="text-muted-foreground hover:text-foreground transition-colors"
                title="Copy path"
              >
                <Copy className="h-3 w-3" />
              </button>
            </div>
            <p className="text-[10px] text-muted-foreground">
              Optional body: {'{ "append_message": "…" }'}
              {webhook.has_secret && ', "signature": "sha256=…"'}
            </p>
          </div>
        </div>
      </div>

      {dirty && (
        <Button size="sm" onClick={handleSave} className="w-full">
          Save changes
        </Button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main WebhooksPanel
// ---------------------------------------------------------------------------
export function WebhooksPanel({ projectId }: { projectId?: string }) {
  const [webhooks, setWebhooks] = useState<WebhookListener[]>([]);
  const [skills, setSkills] = useState<CronbanSkill[]>([]);
  const [connectors, setConnectors] = useState<ConnectorInfo[]>([]);
  const [selected, setSelected] = useState<WebhookListener | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [ws, ss, cs] = await Promise.all([
        listWebhooks(projectId),
        listSkills(projectId),
        listConnectors().catch(() => [] as ConnectorInfo[]),
      ]);
      setWebhooks(ws);
      setSkills(ss);
      setConnectors(cs);
      if (!selected && ws.length > 0) setSelected(ws[0]);
    } finally {
      setLoading(false);
    }
  }, [projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { load(); }, [load]);

  const handleCreate = async () => {
    try {
      const w = await createWebhook({
        project_id: projectId,
        name: 'New Webhook',
        status: 'active',
      });
      setWebhooks((prev) => [...prev, w]);
      setSelected(w);
    } catch (e) {
      console.error('Create webhook failed:', e);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteWebhook(id);
      setWebhooks((prev) => prev.filter((w) => w.id !== id));
      if (selected?.id === id) setSelected(webhooks.find((w) => w.id !== id) ?? null);
    } catch (e) {
      console.error('Delete webhook failed:', e);
    }
  };

  const handleFire = async (id: string) => {
    try {
      await fireWebhook(id);
      await load();
    } catch (e) {
      console.error('Fire webhook failed:', e);
    }
  };

  const handleSave = async (data: Parameters<typeof updateWebhook>[1]) => {
    if (!selected) return;
    try {
      const updated = await updateWebhook(selected.id, data);
      setWebhooks((prev) => prev.map((w) => (w.id === selected.id ? updated : w)));
      setSelected(updated);
    } catch (e) {
      console.error('Save webhook failed:', e);
    }
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
        Loading…
      </div>
    );
  }

  return (
    <div className="h-full flex overflow-hidden">
      {/* Sidebar */}
      <div className="w-56 shrink-0 border-r border-border flex flex-col">
        <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
          <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Webhooks
          </span>
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0"
            onClick={handleCreate}
            title="New webhook"
          >
            <Plus className="h-3.5 w-3.5" />
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto p-1.5 space-y-0.5">
          {webhooks.length === 0 ? (
            <div className="px-2 py-4 text-center text-xs text-muted-foreground">
              No webhooks yet
            </div>
          ) : (
            webhooks.map((w) => (
              <WebhookListItem
                key={w.id}
                webhook={w}
                isSelected={selected?.id === w.id}
                onSelect={() => setSelected(w)}
                onDelete={() => handleDelete(w.id)}
                onFire={() => handleFire(w.id)}
              />
            ))
          )}
        </div>
      </div>

      {/* Editor */}
      {selected ? (
        <WebhookEditor
          key={selected.id}
          webhook={selected}
          skills={skills}
          connectors={connectors}
          onSave={handleSave}
        />
      ) : (
        <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
          <div className="text-center space-y-2">
            <Webhook className="h-8 w-8 mx-auto opacity-20" />
            <p>Select a webhook to edit</p>
            <Button size="sm" variant="outline" onClick={handleCreate} className="gap-1.5">
              <Plus className="h-3.5 w-3.5" />
              New webhook
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
