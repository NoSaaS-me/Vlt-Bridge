/**
 * GatesPanel — Verifier gate library.
 *
 * Left: scrollable list of gates with search + create.
 * Right: markdown editor for the selected gate's prompt.
 *
 * Gates are the "held-out" evaluation criterion that helper Claude
 * uses to judge whether the working agent completed its task.
 * They are NEVER shown to the working agent.
 */
import { useState, useEffect, useCallback } from 'react';
import { Plus, Trash2, Tag, Search, ShieldCheck } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  type CronbanGate,
  listGates,
  createGate,
  updateGate,
  deleteGate,
} from '@/services/cronban-api';

// ---------------------------------------------------------------------------
// Gate list item
// ---------------------------------------------------------------------------
function GateListItem({
  gate,
  selected,
  onClick,
}: {
  gate: CronbanGate;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left px-3 py-2.5 rounded-md transition-colors group',
        selected
          ? 'bg-violet-500/15 border border-violet-500/40'
          : 'hover:bg-muted/40 border border-transparent',
      )}
    >
      <div className="flex items-center gap-2 mb-0.5">
        <ShieldCheck className="h-3 w-3 shrink-0 text-violet-400" />
        <span className="text-sm font-medium truncate">{gate.name}</span>
      </div>
      {gate.description && (
        <p className="text-xs text-muted-foreground truncate pl-5">{gate.description}</p>
      )}
      {gate.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1 pl-5">
          {gate.tags.slice(0, 3).map((t) => (
            <span key={t} className="text-[9px] px-1 py-0.5 rounded bg-muted text-muted-foreground">
              {t}
            </span>
          ))}
        </div>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Gate editor (right pane)
// ---------------------------------------------------------------------------
function GateEditor({
  gate,
  onSave,
  onDelete,
}: {
  gate: CronbanGate | null;
  onSave: (updated: CronbanGate) => void;
  onDelete: (id: string) => void;
}) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [prompt, setPrompt] = useState('');
  const [tagsRaw, setTagsRaw] = useState('');
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    if (gate) {
      setName(gate.name);
      setDescription(gate.description ?? '');
      setPrompt(gate.prompt_markdown);
      setTagsRaw(gate.tags.join(', '));
      setDirty(false);
    }
  }, [gate?.id]);

  const handleSave = useCallback(async () => {
    if (!gate || !name.trim()) return;
    setSaving(true);
    try {
      const tags = tagsRaw.split(',').map((t) => t.trim()).filter(Boolean);
      const updated = await updateGate(gate.id, {
        name: name.trim(),
        description: description.trim() || null,
        prompt_markdown: prompt,
        tags,
      });
      setDirty(false);
      onSave(updated);
    } finally {
      setSaving(false);
    }
  }, [gate, name, description, prompt, tagsRaw, onSave]);

  if (!gate) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
        Select a gate to edit, or create a new one.
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col gap-3 p-4 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Edit Gate</h3>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="text-destructive hover:text-destructive"
            onClick={() => onDelete(gate.id)}
          >
            <Trash2 className="h-3.5 w-3.5 mr-1" />
            Delete
          </Button>
          <Button size="sm" onClick={handleSave} disabled={saving || !dirty}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </div>
      </div>

      {/* Name */}
      <div className="space-y-1">
        <label className="text-xs text-muted-foreground font-medium">Name</label>
        <Input
          value={name}
          onChange={(e) => { setName(e.target.value); setDirty(true); }}
          placeholder="All Tests Pass"
          className="h-8 text-sm"
        />
      </div>

      {/* Description */}
      <div className="space-y-1">
        <label className="text-xs text-muted-foreground font-medium">Description (optional)</label>
        <Input
          value={description}
          onChange={(e) => { setDescription(e.target.value); setDirty(true); }}
          placeholder="What this gate verifies"
          className="h-8 text-sm"
        />
      </div>

      {/* Eval prompt — the criterion sent to helper Claude */}
      <div className="space-y-1 flex-1 flex flex-col">
        <label className="text-xs text-muted-foreground font-medium flex items-center gap-1.5">
          Eval Prompt
          <span className="text-[9px] bg-orange-500/10 text-orange-400 border border-orange-500/20 px-1 py-0.5 rounded">
            helper Claude only
          </span>
        </label>
        <Textarea
          value={prompt}
          onChange={(e) => { setPrompt(e.target.value); setDirty(true); }}
          placeholder={
            'Run `npm test` and verify all tests pass.\n' +
            'Check that the PR description references the issue number.\n' +
            'The helper can run scripts and read files.'
          }
          className="flex-1 min-h-48 text-sm font-mono resize-none"
        />
        <p className="text-[10px] text-muted-foreground">
          Helper Claude executes this after the working agent's turn ends.
          The working agent never sees this prompt.
        </p>
      </div>

      {/* Tags */}
      <div className="space-y-1">
        <label className="text-xs text-muted-foreground font-medium flex items-center gap-1">
          <Tag className="h-3 w-3" />
          Tags (comma-separated)
        </label>
        <Input
          value={tagsRaw}
          onChange={(e) => { setTagsRaw(e.target.value); setDirty(true); }}
          placeholder="tests, ci, code-quality"
          className="h-8 text-sm"
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------
export function GatesPanel({ projectId }: { projectId?: string }) {
  const [gates, setGates] = useState<CronbanGate[]>([]);
  const [selected, setSelected] = useState<CronbanGate | null>(null);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const data = await listGates(projectId);
      setGates(data);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => { load(); }, [load]);

  const handleCreate = useCallback(async () => {
    const gate = await createGate({
      project_id: projectId ?? null,
      name: 'New Gate',
      prompt_markdown: '# Evaluation Criteria\n\nDescribe what helper Claude should check here.\nYou can include instructions to run scripts.',
      tags: [],
    });
    setGates((prev) => [gate, ...prev]);
    setSelected(gate);
  }, [projectId]);

  const handleSave = useCallback((updated: CronbanGate) => {
    setGates((prev) => prev.map((g) => (g.id === updated.id ? updated : g)));
    setSelected(updated);
  }, []);

  const handleDelete = useCallback(async (id: string) => {
    await deleteGate(id);
    setGates((prev) => prev.filter((g) => g.id !== id));
    if (selected?.id === id) setSelected(null);
  }, [selected]);

  const filtered = gates.filter((g) =>
    !search ||
    g.name.toLowerCase().includes(search.toLowerCase()) ||
    (g.description ?? '').toLowerCase().includes(search.toLowerCase()) ||
    g.tags.some((t) => t.toLowerCase().includes(search.toLowerCase())),
  );

  return (
    <div className="h-full flex">
      {/* Left: list */}
      <div className="w-64 shrink-0 border-r border-border flex flex-col">
        <div className="p-3 border-b border-border flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search gates…"
              className="h-7 pl-7 text-xs"
            />
          </div>
          <Button size="sm" variant="outline" onClick={handleCreate} className="h-7 px-2">
            <Plus className="h-3.5 w-3.5" />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {loading && (
            <p className="text-xs text-muted-foreground px-2 py-4 text-center">Loading…</p>
          )}
          {!loading && filtered.length === 0 && (
            <p className="text-xs text-muted-foreground px-2 py-4 text-center">
              {search ? 'No matches.' : 'No gates yet. Create one →'}
            </p>
          )}
          {filtered.map((g) => (
            <GateListItem
              key={g.id}
              gate={g}
              selected={selected?.id === g.id}
              onClick={() => setSelected(g)}
            />
          ))}
        </div>
      </div>

      {/* Right: editor */}
      <GateEditor gate={selected} onSave={handleSave} onDelete={handleDelete} />
    </div>
  );
}
