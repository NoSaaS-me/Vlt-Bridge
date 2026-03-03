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
import { Plus, X, Clock, Zap, RotateCcw, Pencil, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  type CronTrigger,
  listCronTriggers,
  createCronTrigger,
  updateCronTrigger,
  deleteCronTrigger,
} from '@/services/cronban-api';

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

  useEffect(() => {
    if (open) {
      setTitle(initial?.title ?? '');
      setCron(initial?.cron_expression ?? '');
      setColor(initial?.color ?? 'blue');
      setScheduleType(initial?.fire_once ? 'once' : 'recurring');
      setOnceMode('in');
      setFireIn('');
      setFireAt('');
    }
  }, [open, initial]);

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
      <div className="w-full max-w-md rounded-lg border border-border bg-popover shadow-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
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

        <div className="space-y-3">
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

          {/* Recurring: cron expression */}
          {scheduleType === 'recurring' && (
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">
                Cron expression{' '}
                <span className="text-muted-foreground/60">(e.g. 0 9 * * 1-5 for weekdays 9am)</span>
              </label>
              <Input
                value={cron}
                onChange={(e) => setCron(e.target.value)}
                placeholder="0 9 * * *"
                className="h-8 text-sm font-mono"
              />
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
        </div>

        <div className="flex justify-end gap-2 pt-2">
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
