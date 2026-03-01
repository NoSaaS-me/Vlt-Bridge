/**
 * SkillsPanel — Prompt template library.
 *
 * Left: scrollable list of skills with search + create.
 * Right: markdown editor for the selected skill.
 */
import { useState, useEffect, useCallback } from 'react';
import { Plus, Trash2, Tag, Search, FileText } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  type CronbanSkill,
  listSkills,
  createSkill,
  updateSkill,
  deleteSkill,
} from '@/services/cronban-api';

// ---------------------------------------------------------------------------
// Skill list item
// ---------------------------------------------------------------------------
function SkillListItem({
  skill,
  selected,
  onClick,
}: {
  skill: CronbanSkill;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left px-3 py-2.5 rounded-md transition-colors group',
        selected
          ? 'bg-blue-500/15 border border-blue-500/40'
          : 'hover:bg-muted/40 border border-transparent',
      )}
    >
      <div className="flex items-center gap-2 mb-0.5">
        <FileText className="h-3 w-3 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium truncate">{skill.name}</span>
      </div>
      {skill.description && (
        <p className="text-xs text-muted-foreground truncate pl-5">{skill.description}</p>
      )}
      {skill.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1 pl-5">
          {skill.tags.slice(0, 3).map((t) => (
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
// Skill editor (right pane)
// ---------------------------------------------------------------------------
function SkillEditor({
  skill,
  onSave,
  onDelete,
}: {
  skill: CronbanSkill | null;
  onSave: (updated: CronbanSkill) => void;
  onDelete: (id: string) => void;
}) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [prompt, setPrompt] = useState('');
  const [tagsRaw, setTagsRaw] = useState('');
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    if (skill) {
      setName(skill.name);
      setDescription(skill.description ?? '');
      setPrompt(skill.prompt_markdown);
      setTagsRaw(skill.tags.join(', '));
      setDirty(false);
    }
  }, [skill?.id]);

  const handleSave = useCallback(async () => {
    if (!skill || !name.trim()) return;
    setSaving(true);
    try {
      const tags = tagsRaw.split(',').map((t) => t.trim()).filter(Boolean);
      const updated = await updateSkill(skill.id, {
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
  }, [skill, name, description, prompt, tagsRaw, onSave]);

  if (!skill) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
        Select a skill to edit, or create a new one.
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col gap-3 p-4 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Edit Skill</h3>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="text-destructive hover:text-destructive"
            onClick={() => onDelete(skill.id)}
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
          placeholder="Daily Code Review"
          className="h-8 text-sm"
        />
      </div>

      {/* Description */}
      <div className="space-y-1">
        <label className="text-xs text-muted-foreground font-medium">Description (optional)</label>
        <Input
          value={description}
          onChange={(e) => { setDescription(e.target.value); setDirty(true); }}
          placeholder="Short summary of what this skill does"
          className="h-8 text-sm"
        />
      </div>

      {/* Prompt — the actual skill content Claude sees */}
      <div className="space-y-1 flex-1 flex flex-col">
        <label className="text-xs text-muted-foreground font-medium flex items-center gap-1.5">
          Prompt
          <span className="text-[9px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-1 py-0.5 rounded">
            Claude sees this
          </span>
        </label>
        <Textarea
          value={prompt}
          onChange={(e) => { setPrompt(e.target.value); setDirty(true); }}
          placeholder="Write the task prompt in markdown. This is what Claude will be asked to do."
          className="flex-1 min-h-48 text-sm font-mono resize-none"
        />
        <p className="text-[10px] text-muted-foreground">
          Supports markdown. Use this to describe the task. The eval (hidden from Claude) is set separately per entry.
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
          placeholder="review, daily, code-quality"
          className="h-8 text-sm"
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------
export function SkillsPanel({ projectId }: { projectId?: string }) {
  const [skills, setSkills] = useState<CronbanSkill[]>([]);
  const [selected, setSelected] = useState<CronbanSkill | null>(null);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const data = await listSkills(projectId);
      setSkills(data);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => { load(); }, [load]);

  const handleCreate = useCallback(async () => {
    const skill = await createSkill({
      project_id: projectId ?? null,
      name: 'New Skill',
      prompt_markdown: '# Task\n\nDescribe what Claude should do here.',
      tags: [],
    });
    setSkills((prev) => [skill, ...prev]);
    setSelected(skill);
  }, [projectId]);

  const handleSave = useCallback((updated: CronbanSkill) => {
    setSkills((prev) => prev.map((s) => (s.id === updated.id ? updated : s)));
    setSelected(updated);
  }, []);

  const handleDelete = useCallback(async (id: string) => {
    await deleteSkill(id);
    setSkills((prev) => prev.filter((s) => s.id !== id));
    if (selected?.id === id) setSelected(null);
  }, [selected]);

  const filtered = skills.filter((s) =>
    !search ||
    s.name.toLowerCase().includes(search.toLowerCase()) ||
    (s.description ?? '').toLowerCase().includes(search.toLowerCase()) ||
    s.tags.some((t) => t.toLowerCase().includes(search.toLowerCase())),
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
              placeholder="Search skills…"
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
              {search ? 'No matches.' : 'No skills yet. Create one →'}
            </p>
          )}
          {filtered.map((s) => (
            <SkillListItem
              key={s.id}
              skill={s}
              selected={selected?.id === s.id}
              onClick={() => setSelected(s)}
            />
          ))}
        </div>
      </div>

      {/* Right: editor */}
      <SkillEditor skill={selected} onSave={handleSave} onDelete={handleDelete} />
    </div>
  );
}
