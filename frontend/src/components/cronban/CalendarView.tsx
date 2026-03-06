/**
 * CalendarView — Full calendar for CronTriggers using FullCalendar.
 *
 * Views: month, week (timeGrid), day, agenda list.
 * Click a day/time slot → opens CronTriggerDialog.
 * Click an event → shows detail popover (title, cron, next fire, last fire).
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import FullCalendar from '@fullcalendar/react';
import dayGridPlugin from '@fullcalendar/daygrid';
import timeGridPlugin from '@fullcalendar/timegrid';
import interactionPlugin from '@fullcalendar/interaction';
import listPlugin from '@fullcalendar/list';
import type { EventClickArg, DateSelectArg, EventContentArg } from '@fullcalendar/core';
import { Plus, X, Clock, Zap, RotateCcw, Pencil, Trash2, ArrowUp, ArrowDown, ChevronDown, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  type CronTrigger,
  type CronbanSkill,
  type CronbanGate,
  listCronTriggers,
  createCronTrigger,
  updateCronTrigger,
  deleteCronTrigger,
  listSkills,
  listGates,
} from '@/services/cronban-api';
import { type AgentSession, listSessions } from '@/services/daemon-api';

// ---------------------------------------------------------------------------
// Color map → FullCalendar eventColor
// ---------------------------------------------------------------------------
const COLOR_MAP: Record<string, string> = {
  blue:    '#3b82f6',
  emerald: '#10b981',
  amber:   '#f59e0b',
  rose:    '#f43f5e',
  purple:  '#a855f7',
  slate:   '#64748b',
};

function triggerColor(color: string | null) {
  return COLOR_MAP[color ?? 'blue'] ?? COLOR_MAP.blue;
}

// ---------------------------------------------------------------------------
// Event detail popover
// ---------------------------------------------------------------------------
interface PopoverTrigger {
  trigger: CronTrigger;
  x: number;
  y: number;
}

function EventPopover({
  data,
  onClose,
  onEdit,
  onDelete,
}: {
  data: PopoverTrigger;
  onClose: () => void;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const { trigger, x, y } = data;
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  const fmt = (iso: string | null) =>
    iso ? new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—';

  return (
    <div
      ref={ref}
      style={{ position: 'fixed', left: x, top: y, zIndex: 9999 }}
      className="w-72 rounded-lg border border-border bg-popover shadow-xl p-4 space-y-3"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="h-3 w-3 rounded-full shrink-0 mt-0.5"
            style={{ background: triggerColor(trigger.color) }}
          />
          <span className="font-semibold text-sm text-foreground leading-tight">{trigger.title}</span>
        </div>
        <div className="flex items-center gap-0.5 shrink-0">
          <button
            onClick={onEdit}
            title="Edit"
            className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
          >
            <Pencil className="h-3 w-3" />
          </button>
          <button
            onClick={onDelete}
            title="Delete"
            className="p-1 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
          >
            <Trash2 className="h-3 w-3" />
          </button>
          <button
            onClick={onClose}
            className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {trigger.cron_expression && (
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-mono bg-muted/30 rounded px-2 py-1">
          <Clock className="h-3 w-3 shrink-0" />
          {trigger.cron_expression}
        </div>
      )}
      {trigger.fire_once && !trigger.cron_expression && (
        <div className="flex items-center gap-1.5 text-xs text-amber-400/80 bg-amber-500/10 rounded px-2 py-1">
          <Zap className="h-3 w-3 shrink-0" />
          One-off
        </div>
      )}

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="space-y-0.5">
          <p className="text-muted-foreground">Next fire</p>
          <p className="text-foreground font-medium">{fmt(trigger.next_fire_at)}</p>
        </div>
        <div className="space-y-0.5">
          <p className="text-muted-foreground">Last fired</p>
          <p className="text-foreground font-medium">{fmt(trigger.last_fired_at)}</p>
        </div>
        <div className="space-y-0.5">
          <p className="text-muted-foreground">Total fires</p>
          <p className="text-foreground font-medium flex items-center gap-1">
            <Zap className="h-3 w-3" />
            {trigger.fire_count}
          </p>
        </div>
        <div className="space-y-0.5">
          <p className="text-muted-foreground">Status</p>
          <span
            className={cn(
              'inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border',
              trigger.status === 'active'
                ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
                : trigger.status === 'completed'
                ? 'text-blue-400 bg-blue-500/10 border-blue-500/30'
                : 'text-muted-foreground bg-muted/50 border-border',
            )}
          >
            {trigger.status === 'active' && <RotateCcw className="h-2.5 w-2.5" />}
            {trigger.status}
          </span>
        </div>
      </div>

      {trigger.pipeline_id && (
        <p className="text-[10px] text-blue-400/80 flex items-center gap-1">
          🔗 Feeds pipeline
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// CronBuilder — visual schedule builder (frequency + time pickers)
// ---------------------------------------------------------------------------
type CronFreq = 'every-minute' | 'every-n-min' | 'hourly' | 'daily' | 'weekdays' | 'weekly' | 'monthly' | 'custom';

const DOW_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const FREQ_OPTIONS: { value: CronFreq; label: string }[] = [
  { value: 'every-minute', label: 'Every minute' },
  { value: 'every-n-min',  label: 'Every N minutes' },
  { value: 'hourly',       label: 'Every hour' },
  { value: 'daily',        label: 'Every day' },
  { value: 'weekdays',     label: 'Weekdays (Mon–Fri)' },
  { value: 'weekly',       label: 'Every week' },
  { value: 'monthly',      label: 'Every month' },
  { value: 'custom',       label: 'Custom expression' },
];

function CronBuilder({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [freq, setFreq] = useState<CronFreq>('daily');
  const [hour, setHour] = useState('9');
  const [minute, setMinute] = useState('0');
  const [dow, setDow] = useState<number[]>([1]); // 0=Sun … 6=Sat
  const [dom, setDom] = useState('1');
  const [everyN, setEveryN] = useState('30');

  // Build and emit cron expression whenever any setting changes
  useEffect(() => {
    if (freq === 'custom') return;
    const mm = minute;
    const hh = hour;
    const d = dow.length > 0 ? dow.slice().sort((a, b) => a - b).join(',') : '1';
    const exprs: Record<Exclude<CronFreq, 'custom'>, string> = {
      'every-minute': '* * * * *',
      'every-n-min':  `*/${everyN} * * * *`,
      'hourly':       `${mm} * * * *`,
      'daily':        `${mm} ${hh} * * *`,
      'weekdays':     `${mm} ${hh} * * 1-5`,
      'weekly':       `${mm} ${hh} * * ${d}`,
      'monthly':      `${mm} ${hh} ${dom} * *`,
    };
    onChange(exprs[freq as Exclude<CronFreq, 'custom'>]);
  }, [freq, hour, minute, dow, dom, everyN]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggleDow = (d: number) =>
    setDow((prev) => prev.includes(d) ? prev.filter((x) => x !== d) : [...prev, d]);

  const needsTime   = ['daily', 'weekdays', 'weekly', 'monthly'].includes(freq);
  const needsDow    = freq === 'weekly';
  const needsDom    = freq === 'monthly';
  const needsN      = freq === 'every-n-min';
  const needsMinute = freq === 'hourly';

  const selectCls = 'h-7 text-xs bg-background border border-border rounded px-2 text-foreground focus:outline-none focus:ring-1 focus:ring-ring';

  return (
    <div className="space-y-2">
      {/* Frequency */}
      <select
        value={freq}
        onChange={(e) => setFreq(e.target.value as CronFreq)}
        className={cn(selectCls, 'w-full h-8')}
      >
        {FREQ_OPTIONS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
      </select>

      {/* Every N minutes */}
      {needsN && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-14 shrink-0">Every</span>
          <Input type="number" min="1" max="59" value={everyN}
            onChange={(e) => setEveryN(e.target.value)} className="h-7 text-xs w-20" />
          <span className="text-xs text-muted-foreground">minutes</span>
        </div>
      )}

      {/* Hourly — just minute offset */}
      {needsMinute && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-14 shrink-0">At minute</span>
          <select value={minute} onChange={(e) => setMinute(e.target.value)} className={selectCls}>
            {Array.from({ length: 60 }, (_, i) => (
              <option key={i} value={String(i)}>{String(i).padStart(2, '0')}</option>
            ))}
          </select>
        </div>
      )}

      {/* Time of day */}
      {needsTime && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-14 shrink-0">At</span>
          <select value={hour} onChange={(e) => setHour(e.target.value)} className={selectCls}>
            {Array.from({ length: 24 }, (_, i) => (
              <option key={i} value={String(i)}>{String(i).padStart(2, '0')}</option>
            ))}
          </select>
          <span className="text-xs text-muted-foreground">:</span>
          <select value={minute} onChange={(e) => setMinute(e.target.value)} className={selectCls}>
            {Array.from({ length: 60 }, (_, i) => (
              <option key={i} value={String(i)}>{String(i).padStart(2, '0')}</option>
            ))}
          </select>
        </div>
      )}

      {/* Day of week (weekly) */}
      {needsDow && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-14 shrink-0">On</span>
          <div className="flex gap-1">
            {DOW_LABELS.map((label, i) => (
              <button
                key={i}
                type="button"
                onClick={() => toggleDow(i)}
                className={cn(
                  'h-6 w-8 text-[10px] rounded border transition-colors',
                  dow.includes(i)
                    ? 'bg-primary/20 border-primary/50 text-primary'
                    : 'border-border text-muted-foreground hover:bg-muted/30',
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Day of month (monthly) */}
      {needsDom && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-14 shrink-0">On day</span>
          <Input type="number" min="1" max="31" value={dom}
            onChange={(e) => setDom(e.target.value)} className="h-7 text-xs w-20" />
          <span className="text-xs text-muted-foreground">of month</span>
        </div>
      )}

      {/* Custom: raw cron field */}
      {freq === 'custom' && (
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="0 9 * * *"
          className="h-8 text-sm font-mono"
        />
      )}

      {/* Preview */}
      {freq !== 'custom' && value && (
        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground font-mono bg-muted/30 rounded px-2 py-1">
          <Clock className="h-3 w-3 shrink-0" />
          {value}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Prompt chain builder types + helpers
// ---------------------------------------------------------------------------
type ChainItem =
  | { id: string; type: 'text'; content: string }
  | { id: string; type: 'skill'; skillId: string };

function assembleChain(items: ChainItem[], skills: CronbanSkill[]): string {
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

function PromptChainBuilder({
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

// ---------------------------------------------------------------------------
// CronTriggerDialog — create / edit form with recurring and one-off support
// ---------------------------------------------------------------------------
const COLORS = ['blue', 'emerald', 'amber', 'rose', 'purple', 'slate'] as const;

function CronTriggerDialog({
  open,
  onOpenChange,
  projectId,
  initial,
  onSaved,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId?: string;
  initial?: CronTrigger;
  onSaved: (trigger: CronTrigger) => void;
}) {
  const [title, setTitle] = useState(initial?.title ?? '');
  const [cron, setCron] = useState(initial?.cron_expression ?? '');
  const [color, setColor] = useState<string>(initial?.color ?? 'blue');
  const [scheduleType, setScheduleType] = useState<'recurring' | 'once'>(
    initial?.fire_once ? 'once' : 'recurring',
  );
  const [onceMode, setOnceMode] = useState<'in' | 'at'>('in');
  const [fireIn, setFireIn] = useState('');
  const [fireAt, setFireAt] = useState('');
  const [saving, setSaving] = useState(false);
  const [promptText, setPromptText] = useState('');
  const [gateId, setGateId] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [chainItems, setChainItems] = useState<ChainItem[]>([]);
  const [skills, setSkills] = useState<CronbanSkill[]>([]);
  const [gates, setGates] = useState<CronbanGate[]>([]);
  const [loadingSkills, setLoadingSkills] = useState(false);
  const [targetSessionId, setTargetSessionId] = useState<string | null>(null);
  const [targetCwd, setTargetCwd] = useState('');
  const [createNew, setCreateNew] = useState(false);
  const [availSessions, setAvailSessions] = useState<AgentSession[]>([]);

  useEffect(() => {
    if (open) {
      setTitle(initial?.title ?? '');
      setCron(initial?.cron_expression ?? '');
      setColor(initial?.color ?? 'blue');
      setScheduleType(initial?.fire_once ? 'once' : 'recurring');
      setOnceMode('in');
      setFireIn('');
      setFireAt('');
      setPromptText('');
      setGateId(initial?.gate_id ?? null);
      setTargetSessionId(initial?.target_session_id ?? null);
      setTargetCwd(initial?.target_cwd ?? '');
      setCreateNew(initial?.create_new_session ?? false);
      setAdvancedOpen(false);
      setChainItems([]);
    }
  }, [open, initial]);

  // Load skills when advanced section is opened; load gates + sessions on open
  useEffect(() => {
    if (!advancedOpen || skills.length > 0) return;
    setLoadingSkills(true);
    listSkills(projectId).then(setSkills).catch(console.error).finally(() => setLoadingSkills(false));
  }, [advancedOpen, projectId, skills.length]);

  useEffect(() => {
    if (!open) return;
    if (gates.length === 0) listGates(projectId).then(setGates).catch(console.error);
    listSessions().then((s) => setAvailSessions(s.filter((s) => s.status !== 'dead'))).catch(console.error);
  }, [open, projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSave = async () => {
    if (!title.trim()) return;
    setSaving(true);
    try {
      const payload: Parameters<typeof createCronTrigger>[0] & { fire_at?: string; fire_in?: string } = {
        project_id: projectId,
        title: title.trim(),
        color,
        status: 'active',
      };
      if (scheduleType === 'recurring') {
        payload.cron_expression = cron.trim() || undefined;
        payload.fire_once = false;
      } else if (onceMode === 'in' && fireIn.trim()) {
        payload.fire_in = fireIn.trim();
        payload.fire_once = true;
      } else if (onceMode === 'at' && fireAt) {
        payload.fire_at = new Date(fireAt).toISOString();
        payload.fire_once = true;
      }

      // Prompt: chain builder takes priority over simple textarea
      if (advancedOpen && chainItems.length > 0) {
        const assembled = assembleChain(chainItems, skills);
        if (assembled) payload.prompt_text = assembled;
      } else if (promptText.trim()) {
        payload.prompt_text = promptText.trim();
      }

      // Gate
      if (gateId) payload.gate_id = gateId;

      // Session targeting
      if (createNew && targetCwd.trim()) {
        payload.target_cwd = targetCwd.trim();
        payload.create_new_session = true;
      } else if (targetSessionId) {
        payload.target_session_id = targetSessionId;
        payload.create_new_session = false;
      }

      let saved: CronTrigger;
      if (initial) {
        saved = await updateCronTrigger(initial.id, payload);
      } else {
        saved = await createCronTrigger(payload);
      }
      onSaved(saved);
      onOpenChange(false);
    } finally {
      setSaving(false);
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-md max-h-[90vh] flex flex-col rounded-lg border border-border bg-popover shadow-2xl">
        {/* Fixed header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-4 shrink-0">
          <h2 className="text-sm font-semibold">
            {initial ? 'Edit Trigger' : 'New Trigger'}
          </h2>
          <button
            onClick={() => onOpenChange(false)}
            className="text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto px-6 space-y-3">
          {/* Title */}
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Title</label>
            <Input
              autoFocus
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSave()}
              placeholder="e.g. Daily standup report"
              className="h-8 text-sm"
            />
          </div>

          {/* Schedule type toggle */}
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Schedule</label>
            <div className="flex gap-1.5">
              {(['recurring', 'once'] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setScheduleType(t)}
                  className={cn(
                    'px-3 py-1 rounded text-xs border transition-colors capitalize',
                    scheduleType === t
                      ? 'bg-primary/10 border-primary/40 text-primary'
                      : 'border-border text-muted-foreground hover:bg-muted/30',
                  )}
                >
                  {t === 'once' ? 'One-off' : 'Recurring'}
                </button>
              ))}
            </div>
          </div>

          {/* Recurring: visual cron builder */}
          {scheduleType === 'recurring' && (
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Schedule</label>
              <CronBuilder value={cron} onChange={setCron} />
            </div>
          )}

          {/* One-off: in / at sub-selector */}
          {scheduleType === 'once' && (
            <div className="space-y-2">
              <div className="flex gap-1.5">
                {(['in', 'at'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setOnceMode(m)}
                    className={cn(
                      'px-2.5 py-0.5 rounded text-xs border transition-colors',
                      onceMode === m
                        ? 'bg-muted border-border text-foreground'
                        : 'border-transparent text-muted-foreground hover:bg-muted/30',
                    )}
                  >
                    {m === 'in' ? 'In (offset)' : 'At (date/time)'}
                  </button>
                ))}
              </div>

              {onceMode === 'in' && (
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">
                    Offset from now — hh:mm or hh:mm:ss
                  </label>
                  <Input
                    value={fireIn}
                    onChange={(e) => setFireIn(e.target.value)}
                    placeholder="1:30  or  0:45:00"
                    className="h-8 text-sm font-mono"
                  />
                </div>
              )}

              {onceMode === 'at' && (
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Date and time</label>
                  <input
                    type="datetime-local"
                    value={fireAt}
                    onChange={(e) => setFireAt(e.target.value)}
                    className="w-full h-8 rounded border border-border bg-background text-sm px-2 text-foreground [color-scheme:dark]"
                  />
                </div>
              )}
            </div>
          )}

          {/* Color */}
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Color</label>
            <div className="flex gap-1.5">
              {COLORS.map((c) => (
                <button
                  key={c}
                  onClick={() => setColor(c)}
                  className={cn(
                    'h-6 w-6 rounded-full border-2 transition-transform',
                    color === c ? 'border-foreground scale-110' : 'border-transparent',
                  )}
                  style={{ background: COLOR_MAP[c] }}
                  title={c}
                />
              ))}
            </div>
          </div>

          {/* Session target */}
          <div className="space-y-1.5">
            <label className="text-xs text-muted-foreground">Fire at</label>
            <div className="flex gap-1.5">
              {(['session', 'cwd', 'none'] as const).map((mode) => {
                const active = mode === 'session'
                  ? !createNew && !!targetSessionId
                  : mode === 'cwd'
                  ? createNew
                  : !createNew && !targetSessionId;
                return (
                  <button
                    key={mode}
                    onClick={() => {
                      if (mode === 'session') { setCreateNew(false); }
                      else if (mode === 'cwd') { setCreateNew(true); setTargetSessionId(null); }
                      else { setCreateNew(false); setTargetSessionId(null); setTargetCwd(''); }
                    }}
                    className={cn(
                      'px-2.5 py-0.5 rounded text-xs border transition-colors',
                      active
                        ? 'bg-primary/10 border-primary/40 text-primary'
                        : 'border-border text-muted-foreground hover:bg-muted/30',
                    )}
                  >
                    {mode === 'session' ? 'Session' : mode === 'cwd' ? 'New in dir' : 'Any'}
                  </button>
                );
              })}
            </div>

            {/* Session picker or cwd input */}
            {!createNew && availSessions.length > 0 && (
              <select
                value={targetSessionId ?? ''}
                onChange={(e) => setTargetSessionId(e.target.value || null)}
                className="w-full h-8 text-xs bg-background border border-border rounded px-2 text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              >
                <option value="">— Any available session —</option>
                {availSessions.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name || s.id.slice(0, 8)} [{s.source}]
                  </option>
                ))}
              </select>
            )}
            {createNew && (
              <Input
                value={targetCwd}
                onChange={(e) => setTargetCwd(e.target.value)}
                placeholder="/path/to/project"
                className="h-8 text-xs font-mono"
              />
            )}
          </div>

          {/* Prompt */}
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">
              Prompt <span className="text-muted-foreground/60">(what Claude should do when this fires)</span>
            </label>
            <textarea
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              placeholder="Describe the task..."
              rows={3}
              className="w-full text-xs bg-background border border-border rounded px-2 py-1.5 resize-y text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </div>

          {/* Gate — completion criterion */}
          {gates.length > 0 && (
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">
                Gate <span className="text-muted-foreground/60">(completion criterion, checked after each turn)</span>
              </label>
              <select
                value={gateId ?? ''}
                onChange={(e) => setGateId(e.target.value || null)}
                className="w-full h-8 text-xs bg-background border border-border rounded px-2 text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              >
                <option value="">— None (run indefinitely) —</option>
                {gates.map((g) => (
                  <option key={g.id} value={g.id}>{g.name}</option>
                ))}
              </select>
            </div>
          )}

          {/* Advanced prompt chain builder */}
          <div className="border-t border-border/50 pt-3 pb-1">
            <button
              onClick={() => setAdvancedOpen((v) => !v)}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors w-full"
            >
              {advancedOpen
                ? <ChevronDown className="h-3.5 w-3.5" />
                : <ChevronRight className="h-3.5 w-3.5" />}
              <span className="font-medium">Advanced</span>
              {chainItems.length > 0 && (
                <span className="ml-auto text-[10px] text-blue-400 bg-blue-500/10 border border-blue-500/30 px-1.5 py-0.5 rounded">
                  {chainItems.length} block{chainItems.length !== 1 ? 's' : ''}
                </span>
              )}
            </button>

            {advancedOpen && (
              <div className="mt-2.5 space-y-1.5">
                <p className="text-[10px] text-muted-foreground/70">
                  Chain prompt blocks and skills — assembled in order when this trigger fires.
                </p>
                {loadingSkills ? (
                  <p className="text-[10px] text-muted-foreground animate-pulse">Loading skills…</p>
                ) : (
                  <PromptChainBuilder
                    items={chainItems}
                    skills={skills}
                    onChange={setChainItems}
                  />
                )}
              </div>
            )}
          </div>
        </div>

        {/* Fixed footer */}
        <div className="flex justify-end gap-2 px-6 py-4 border-t border-border/50 shrink-0">
          <Button
            size="sm"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={saving}
          >
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={handleSave}
            disabled={saving || !title.trim()}
          >
            {saving ? 'Saving…' : initial ? 'Save' : 'Create'}
          </Button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Custom event pill renderer
// ---------------------------------------------------------------------------
function EventPill({ info }: { info: EventContentArg }) {
  const { event } = info;
  const isDayGrid = info.view.type.startsWith('dayGrid');
  return (
    <div
      className={cn(
        'flex items-center gap-1 w-full overflow-hidden',
        isDayGrid ? 'px-1 py-0.5' : 'px-1.5 py-1',
      )}
    >
      <span
        className="h-1.5 w-1.5 rounded-full shrink-0"
        style={{ background: event.borderColor ?? event.backgroundColor }}
      />
      <span className={cn('truncate leading-none', isDayGrid ? 'text-[11px]' : 'text-xs')}>
        {event.title}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function CalendarView({ projectId }: { projectId?: string }) {
  const [triggers, setTriggers] = useState<CronTrigger[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editTrigger, setEditTrigger] = useState<CronTrigger | null>(null);
  const [popover, setPopover] = useState<PopoverTrigger | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listCronTriggers(projectId);
      setTriggers(data);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => { load(); }, [load]);

  // Build FullCalendar events
  const fcEvents = triggers
    .filter((t) => t.next_fire_at)
    .map((t) => ({
      id: t.id,
      title: t.title,
      start: t.next_fire_at!,
      allDay: false,
      backgroundColor: triggerColor(t.color) + '22',
      borderColor: triggerColor(t.color),
      textColor: triggerColor(t.color),
      extendedProps: { trigger: t },
    }));

  const firedEvents = triggers
    .filter((t) => t.last_fired_at && t.fire_count > 0)
    .map((t) => ({
      id: `fired-${t.id}`,
      title: `✓ ${t.title}`,
      start: t.last_fired_at!,
      allDay: false,
      backgroundColor: 'transparent',
      borderColor: triggerColor(t.color) + '55',
      textColor: triggerColor(t.color) + '99',
      extendedProps: { trigger: t, isFired: true },
    }));

  const handleEventClick = useCallback((arg: EventClickArg) => {
    const trigger: CronTrigger = arg.event.extendedProps.trigger;
    const rect = arg.el.getBoundingClientRect();
    const x = rect.right + 8 > window.innerWidth - 300 ? rect.left - 288 : rect.right + 8;
    const y = Math.min(rect.top, window.innerHeight - 280);
    setPopover({ trigger, x, y });
  }, []);

  const handleDateSelect = useCallback((_arg: DateSelectArg) => {
    setDialogOpen(true);
  }, []);

  const handleSaved = useCallback((trigger: CronTrigger) => {
    setTriggers((prev) => {
      const exists = prev.some((t) => t.id === trigger.id);
      return exists ? prev.map((t) => (t.id === trigger.id ? trigger : t)) : [trigger, ...prev];
    });
  }, []);

  const handleEdit = useCallback(() => {
    if (popover) {
      setEditTrigger(popover.trigger);
      setPopover(null);
    }
  }, [popover]);

  const handleDelete = useCallback(async () => {
    if (!popover) return;
    const id = popover.trigger.id;
    setPopover(null);
    try {
      await deleteCronTrigger(id);
      setTriggers((prev) => prev.filter((t) => t.id !== id));
    } catch (e) {
      console.error('Delete trigger failed:', e);
    }
  }, [popover]);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="shrink-0 flex items-center justify-between px-4 py-2.5 border-b border-border">
        <span className="text-sm font-semibold text-foreground">Scheduled Triggers</span>
        <div className="flex items-center gap-2">
          {loading && (
            <span className="text-xs text-muted-foreground animate-pulse">Loading…</span>
          )}
          <Button size="sm" onClick={() => setDialogOpen(true)} className="h-7 gap-1 text-xs">
            <Plus className="h-3.5 w-3.5" />
            New Trigger
          </Button>
        </div>
      </div>

      {/* FullCalendar */}
      <div className="flex-1 overflow-hidden [&_.fc]:h-full [&_.fc-view-harness]:flex-1 fc-dark">
        <FullCalendar
          plugins={[dayGridPlugin, timeGridPlugin, interactionPlugin, listPlugin]}
          initialView="dayGridMonth"
          headerToolbar={{
            left: 'prev,next today',
            center: 'title',
            right: 'dayGridMonth,timeGridWeek,timeGridDay,listWeek',
          }}
          buttonText={{
            today: 'Today',
            month: 'Month',
            week: 'Week',
            day: 'Day',
            list: 'Agenda',
          }}
          height="100%"
          events={[...fcEvents, ...firedEvents]}
          selectable
          selectMirror
          select={handleDateSelect}
          eventClick={handleEventClick}
          eventContent={(info) => <EventPill info={info} />}
          eventDisplay="block"
          dayMaxEvents={4}
          nowIndicator
          weekends
          slotMinTime="06:00:00"
          slotMaxTime="22:00:00"
        />
      </div>

      {/* Event detail popover */}
      {popover && (
        <EventPopover
          data={popover}
          onClose={() => setPopover(null)}
          onEdit={handleEdit}
          onDelete={handleDelete}
        />
      )}

      {/* Create dialog */}
      <CronTriggerDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        projectId={projectId}
        onSaved={handleSaved}
      />

      {/* Edit dialog */}
      <CronTriggerDialog
        open={editTrigger !== null}
        onOpenChange={(open) => { if (!open) setEditTrigger(null); }}
        projectId={projectId}
        initial={editTrigger ?? undefined}
        onSaved={handleSaved}
      />
    </div>
  );
}
