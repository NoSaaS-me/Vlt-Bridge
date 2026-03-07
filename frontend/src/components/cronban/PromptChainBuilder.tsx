/**
 * PromptChainBuilder — shared component for building ordered chains of
 * prompt blocks and skill references that assemble into a single prompt string.
 *
 * Used by CalendarView (CronTriggerDialog) and KanbanBoard (NewCardDialog).
 */
import { ArrowUp, ArrowDown, X, Plus } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { CronbanSkill } from '@/services/cronban-api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ChainItem =
  | { id: string; type: 'text'; content: string }
  | { id: string; type: 'skill'; skillId: string };

// ---------------------------------------------------------------------------
// assembleChain — flatten chain items into a single prompt string
// ---------------------------------------------------------------------------

export function assembleChain(items: ChainItem[], skills: CronbanSkill[]): string {
  const parts: string[] = [];
  for (const item of items) {
    if (item.type === 'text') {
      if (item.content.trim()) parts.push(item.content.trim());
    } else {
      const skill = skills.find((s) => s.id === item.skillId);
      if (skill) parts.push(`## ${skill.name}\n\n${skill.prompt_markdown.trim()}`);
    }
  }
  return parts.join('\n\n---\n\n');
}

// ---------------------------------------------------------------------------
// PromptChainBuilder component
// ---------------------------------------------------------------------------

export function PromptChainBuilder({
  items,
  skills,
  onChange,
}: {
  items: ChainItem[];
  skills: CronbanSkill[];
  onChange: (items: ChainItem[]) => void;
}) {
  const addText = () =>
    onChange([...items, { id: crypto.randomUUID(), type: 'text', content: '' }]);

  const addSkill = () => {
    const first = skills[0];
    if (!first) return;
    onChange([...items, { id: crypto.randomUUID(), type: 'skill', skillId: first.id }]);
  };

  const update = (id: string, patch: Partial<ChainItem>) =>
    onChange(items.map((item) => (item.id === id ? { ...item, ...patch } as ChainItem : item)));

  const remove = (id: string) => onChange(items.filter((item) => item.id !== id));

  const move = (id: string, dir: -1 | 1) => {
    const idx = items.findIndex((item) => item.id === id);
    if (idx + dir < 0 || idx + dir >= items.length) return;
    const next = [...items];
    [next[idx], next[idx + dir]] = [next[idx + dir], next[idx]];
    onChange(next);
  };

  return (
    <div className="space-y-1.5">
      {items.map((item, idx) => (
        <div key={item.id} className="rounded border border-border bg-muted/20 p-2 space-y-1.5">
          <div className="flex items-center justify-between gap-1">
            <span className={cn(
              'text-[10px] font-medium px-1.5 py-0.5 rounded border',
              item.type === 'text'
                ? 'text-blue-400 bg-blue-500/10 border-blue-500/30'
                : 'text-purple-400 bg-purple-500/10 border-purple-500/30',
            )}>
              {item.type === 'text' ? '📝 Prompt' : '🔧 Skill'}
            </span>
            <div className="flex items-center gap-0.5">
              <button
                onClick={() => move(item.id, -1)}
                disabled={idx === 0}
                className="p-0.5 rounded text-muted-foreground hover:text-foreground disabled:opacity-30 transition-colors"
              >
                <ArrowUp className="h-3 w-3" />
              </button>
              <button
                onClick={() => move(item.id, 1)}
                disabled={idx === items.length - 1}
                className="p-0.5 rounded text-muted-foreground hover:text-foreground disabled:opacity-30 transition-colors"
              >
                <ArrowDown className="h-3 w-3" />
              </button>
              <button
                onClick={() => remove(item.id)}
                className="p-0.5 rounded text-muted-foreground hover:text-destructive transition-colors"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          </div>

          {item.type === 'text' ? (
            <textarea
              value={item.content}
              onChange={(e) => update(item.id, { content: e.target.value })}
              placeholder="Write prompt text here..."
              rows={3}
              className="w-full text-xs bg-background border border-border rounded px-2 py-1.5 resize-y font-mono text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            />
          ) : (
            <select
              value={item.skillId}
              onChange={(e) => update(item.id, { skillId: e.target.value })}
              className="w-full h-7 text-xs bg-background border border-border rounded px-2 text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              {skills.length === 0 ? (
                <option value="">No skills — create some in the Skills tab</option>
              ) : (
                skills.map((s) => (
                  <option key={s.id} value={s.id}>{s.name}</option>
                ))
              )}
            </select>
          )}
        </div>
      ))}

      <div className="flex gap-1.5 pt-0.5">
        <button
          onClick={addText}
          className="flex items-center gap-1 px-2 py-1 rounded border border-dashed border-border text-[10px] text-muted-foreground hover:border-blue-500/50 hover:text-blue-400 transition-colors"
        >
          <Plus className="h-3 w-3" /> Prompt Block
        </button>
        <button
          onClick={addSkill}
          disabled={skills.length === 0}
          className="flex items-center gap-1 px-2 py-1 rounded border border-dashed border-border text-[10px] text-muted-foreground hover:border-purple-500/50 hover:text-purple-400 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Plus className="h-3 w-3" /> Skill
        </button>
      </div>
    </div>
  );
}
