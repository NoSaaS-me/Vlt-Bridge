/**
 * WebhooksPanel — Manage WebhookListeners.
 *
 * Each webhook exposes a POST endpoint at /api/cronban/webhooks/{id}/fire
 * that can be triggered by external systems (CI/CD, error handlers, etc.).
 * Optional HMAC-SHA256 auth via webhook_secret.
 *
 * Layout:
 *   - Left: list of webhook listeners
 *   - Right: editor (name, skill/prompt, pipeline, secret, target session)
 */
import { useState, useEffect, useCallback } from 'react';
import { Plus, Webhook, Trash2, Zap, Copy, Eye, EyeOff } from 'lucide-react';
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
// Webhook editor
// ---------------------------------------------------------------------------
function WebhookEditor({
  webhook,
  skills,
  onSave,
}: {
  webhook: WebhookListener;
  skills: CronbanSkill[];
  onSave: (data: Partial<WebhookListener> & { webhook_secret?: string; prompt_text?: string }) => void;
}) {
  const [name, setName] = useState(webhook.name);
  const [status, setStatus] = useState<'active' | 'paused'>(webhook.status);
  const [skillId, setSkillId] = useState(webhook.skill_id ?? '');
  const [secretDraft, setSecretDraft] = useState('');
  const [showSecret, setShowSecret] = useState(false);
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    setName(webhook.name);
    setStatus(webhook.status);
    setSkillId(webhook.skill_id ?? '');
    setSecretDraft('');
    setDirty(false);
  }, [webhook.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const mark = () => setDirty(true);

  const handleSave = () => {
    const data: Parameters<typeof onSave>[0] = {
      name,
      status,
      skill_id: skillId || undefined,
    };
    if (secretDraft) data.webhook_secret = secretDraft;
    onSave(data);
    setDirty(false);
  };

  // Endpoint URL (displayed, not functional from UI)
  const endpointPath = `/vlt/api/cronban/webhooks/${webhook.id}/fire`;

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

        {/* Endpoint */}
        <div className="space-y-1">
          <label className="text-xs text-muted-foreground">Fire endpoint</label>
          <div className="flex items-center gap-1.5 bg-muted/30 rounded border border-border px-2.5 py-1.5">
            <code className="text-[10px] font-mono text-muted-foreground flex-1 truncate">
              POST {endpointPath}
            </code>
            <button
              onClick={() => navigator.clipboard.writeText(endpointPath)}
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
  const [selected, setSelected] = useState<WebhookListener | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [ws, ss] = await Promise.all([
        listWebhooks(projectId),
        listSkills(projectId),
      ]);
      setWebhooks(ws);
      setSkills(ss);
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
