/**
 * CalendarView — Full calendar for cron-type entries using FullCalendar.
 *
 * Views: month, week (timeGrid), day, agenda list.
 * Click a day/time slot → opens EntryWizard pre-filled with a sensible cron.
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
import { type CronbanEntry, listEntries, deleteEntry } from '@/services/cronban-api';
import { EntryWizard } from './EntryWizard';

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

function entryColor(color: string | null) {
  return COLOR_MAP[color ?? 'blue'] ?? COLOR_MAP.blue;
}

// ---------------------------------------------------------------------------
// Event detail popover
// ---------------------------------------------------------------------------
interface PopoverEntry {
  entry: CronbanEntry;
  x: number;
  y: number;
}

function EventPopover({
  data,
  onClose,
  onEdit,
  onDelete,
}: {
  data: PopoverEntry;
  onClose: () => void;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const { entry, x, y } = data;
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
            style={{ background: entryColor(entry.color) }}
          />
          <span className="font-semibold text-sm text-foreground leading-tight">{entry.title}</span>
        </div>
        <div className="flex items-center gap-0.5 shrink-0">
          <button
            onClick={onEdit}
            title="Edit entry"
            className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
          >
            <Pencil className="h-3 w-3" />
          </button>
          <button
            onClick={onDelete}
            title="Delete entry"
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

      {entry.cron_expression && (
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-mono bg-muted/30 rounded px-2 py-1">
          <Clock className="h-3 w-3 shrink-0" />
          {entry.cron_expression}
        </div>
      )}

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="space-y-0.5">
          <p className="text-muted-foreground">Next fire</p>
          <p className="text-foreground font-medium">{fmt(entry.next_fire_at)}</p>
        </div>
        <div className="space-y-0.5">
          <p className="text-muted-foreground">Last fired</p>
          <p className="text-foreground font-medium">{fmt(entry.last_fired_at)}</p>
        </div>
        <div className="space-y-0.5">
          <p className="text-muted-foreground">Total fires</p>
          <p className="text-foreground font-medium flex items-center gap-1">
            <Zap className="h-3 w-3" />
            {entry.fire_count}
          </p>
        </div>
        <div className="space-y-0.5">
          <p className="text-muted-foreground">Status</p>
          <span
            className={cn(
              'inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border',
              entry.status === 'active'
                ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
                : 'text-muted-foreground bg-muted/50 border-border',
            )}
          >
            {entry.status === 'active' && <RotateCcw className="h-2.5 w-2.5" />}
            {entry.status}
          </span>
        </div>
      </div>

      {entry.has_eval && (
        <p className="text-[10px] text-amber-400/80 flex items-center gap-1">
          🔒 Has hidden eval criterion
        </p>
      )}
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
  const [entries, setEntries] = useState<CronbanEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [editEntry, setEditEntry] = useState<CronbanEntry | null>(null);
  const [popover, setPopover] = useState<PopoverEntry | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listEntries({ entry_type: 'cron', project_id: projectId });
      setEntries(data);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => { load(); }, [load]);

  // Build FullCalendar events array
  const fcEvents = entries
    .filter((e) => e.next_fire_at)
    .map((e) => ({
      id: e.id,
      title: e.title,
      start: e.next_fire_at!,
      allDay: false,
      backgroundColor: entryColor(e.color) + '22',
      borderColor: entryColor(e.color),
      textColor: entryColor(e.color),
      extendedProps: { entry: e },
    }));

  // Also show last_fired_at as a dimmer "fired" event for context
  const firedEvents = entries
    .filter((e) => e.last_fired_at && e.fire_count > 0)
    .map((e) => ({
      id: `fired-${e.id}`,
      title: `✓ ${e.title}`,
      start: e.last_fired_at!,
      allDay: false,
      backgroundColor: 'transparent',
      borderColor: entryColor(e.color) + '55',
      textColor: entryColor(e.color) + '99',
      extendedProps: { entry: e, isFired: true },
    }));

  const handleEventClick = useCallback((arg: EventClickArg) => {
    const entry: CronbanEntry = arg.event.extendedProps.entry;
    const rect = arg.el.getBoundingClientRect();
    // Position popover to the right of the event, or left if near right edge
    const x = rect.right + 8 > window.innerWidth - 300
      ? rect.left - 288
      : rect.right + 8;
    const y = Math.min(rect.top, window.innerHeight - 280);
    setPopover({ entry, x, y });
  }, []);

  const handleDateSelect = useCallback((_arg: DateSelectArg) => {
    setWizardOpen(true);
  }, []);

  const handleCreated = useCallback((entry: CronbanEntry) => {
    setEntries((prev) => {
      const exists = prev.some((e) => e.id === entry.id);
      return exists ? prev.map((e) => (e.id === entry.id ? entry : e)) : [entry, ...prev];
    });
  }, []);

  const handleEdit = useCallback(() => {
    if (popover) {
      setEditEntry(popover.entry);
      setPopover(null);
    }
  }, [popover]);

  const handleDelete = useCallback(async () => {
    if (!popover) return;
    const id = popover.entry.id;
    setPopover(null);
    try {
      await deleteEntry(id);
      setEntries((prev) => prev.filter((e) => e.id !== id));
    } catch (e) {
      console.error('Delete entry failed:', e);
    }
  }, [popover]);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="shrink-0 flex items-center justify-between px-4 py-2.5 border-b border-border">
        <span className="text-sm font-semibold text-foreground">Scheduled Jobs</span>
        <div className="flex items-center gap-2">
          {loading && (
            <span className="text-xs text-muted-foreground animate-pulse">Loading…</span>
          )}
          <Button size="sm" onClick={() => setWizardOpen(true)} className="h-7 gap-1 text-xs">
            <Plus className="h-3.5 w-3.5" />
            New Entry
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

      {/* Create wizard */}
      <EntryWizard
        open={wizardOpen}
        onOpenChange={setWizardOpen}
        projectId={projectId}
        initialType="cron"
        onCreated={handleCreated}
      />

      {/* Edit wizard */}
      <EntryWizard
        open={editEntry !== null}
        onOpenChange={(open) => { if (!open) setEditEntry(null); }}
        projectId={projectId}
        initialEntry={editEntry ?? undefined}
        onCreated={handleCreated}
      />
    </div>
  );
}
