/**
 * CalendarView — month/week calendar grid for cron-type entries.
 *
 * Month view: 7-column CSS grid, each cell = one day, up to 3 chips + "+N more".
 * Week view:  7 columns × 24 rows time grid (Google Calendar style).
 * Unscheduled entries: shown in a right sidebar panel.
 *
 * Data: listEntries({ entry_type: 'cron', project_id })
 * Opens EntryWizard on "New Entry" button.
 */
import { useState, useEffect, useCallback, useMemo } from 'react';
import { ChevronLeft, ChevronRight, Plus, Clock, CalendarDays } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { type CronbanEntry, listEntries } from '@/services/cronban-api';
import { EntryWizard } from './EntryWizard';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseDate(iso: string | null): Date | null {
  if (!iso) return null;
  const d = new Date(iso);
  return isNaN(d.getTime()) ? null : d;
}

/** Returns YYYY-MM-DD for a Date in local time */
function toDateKey(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

/** Returns the Date for the first day of a month view grid (Sunday) */
function getGridStart(year: number, month: number): Date {
  const first = new Date(year, month, 1);
  const dow = first.getDay(); // 0=Sun
  const start = new Date(year, month, 1 - dow);
  return start;
}

/** Adds `days` days to a date (returns new Date) */
function addDays(date: Date, days: number): Date {
  const d = new Date(date);
  d.setDate(d.getDate() + days);
  return d;
}

/** Returns the Monday of the week containing `date` */
function getWeekStart(date: Date): Date {
  const d = new Date(date);
  const dow = d.getDay(); // 0=Sun
  d.setDate(d.getDate() - dow);
  return d;
}

const WEEKDAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

function colorChipClasses(color: string | null) {
  switch (color) {
    case 'emerald': return 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300';
    case 'amber':   return 'bg-amber-500/15 border-amber-500/40 text-amber-300';
    case 'rose':    return 'bg-rose-500/15 border-rose-500/40 text-rose-300';
    case 'purple':  return 'bg-purple-500/15 border-purple-500/40 text-purple-300';
    case 'slate':   return 'bg-slate-500/15 border-slate-500/40 text-slate-300';
    default:        return 'bg-blue-500/15 border-blue-500/40 text-blue-300';
  }
}

function colorDotClass(color: string | null) {
  switch (color) {
    case 'emerald': return 'bg-emerald-400';
    case 'amber':   return 'bg-amber-400';
    case 'rose':    return 'bg-rose-400';
    case 'purple':  return 'bg-purple-400';
    case 'slate':   return 'bg-slate-400';
    default:        return 'bg-blue-400';
  }
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

function EntryTooltip({ entry }: { entry: CronbanEntry }) {
  const fired = entry.last_fired_at ? new Date(entry.last_fired_at).toLocaleString() : 'Never';
  const next = entry.next_fire_at ? new Date(entry.next_fire_at).toLocaleString() : '—';
  return (
    <div className="absolute bottom-full left-0 mb-1.5 z-50 min-w-[200px] max-w-[280px] rounded-md border border-border bg-popover p-2.5 shadow-xl text-xs space-y-1 pointer-events-none">
      <p className="font-semibold text-foreground truncate">{entry.title}</p>
      {entry.cron_expression && (
        <p className="font-mono text-muted-foreground">{entry.cron_expression}</p>
      )}
      <div className="text-muted-foreground space-y-0.5 pt-0.5">
        <p>Next: {next}</p>
        <p>Last fired: {fired}</p>
        {entry.fire_count > 0 && <p>Total fires: {entry.fire_count}</p>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Entry chip (used in month + unscheduled)
// ---------------------------------------------------------------------------

function EntryChip({ entry }: { entry: CronbanEntry }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div className="relative">
      <div
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        className={cn(
          'flex items-center gap-1 px-1.5 py-0.5 rounded border text-[10px] font-medium truncate cursor-default',
          colorChipClasses(entry.color),
        )}
      >
        <span className={cn('h-1.5 w-1.5 rounded-full shrink-0', colorDotClass(entry.color))} />
        <span className="truncate">{entry.title}</span>
      </div>
      {hovered && <EntryTooltip entry={entry} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Month view
// ---------------------------------------------------------------------------

interface MonthViewProps {
  year: number;
  month: number;
  entriesByDate: Map<string, CronbanEntry[]>;
  today: string;
}

function MonthView({ year, month, entriesByDate, today }: MonthViewProps) {
  const gridStart = getGridStart(year, month);
  const cells: Date[] = [];
  for (let i = 0; i < 42; i++) {
    cells.push(addDays(gridStart, i));
  }

  // Trim trailing blank weeks
  const lastWithContent = cells.reduce((last, cell, idx) => {
    const key = toDateKey(cell);
    if (cell.getMonth() === month || (entriesByDate.get(key)?.length ?? 0) > 0) {
      return idx;
    }
    return last;
  }, 0);
  const rowsNeeded = Math.ceil((lastWithContent + 1) / 7);
  const visibleCells = cells.slice(0, rowsNeeded * 7);

  return (
    <div className="flex-1 overflow-auto">
      {/* Day labels */}
      <div className="grid grid-cols-7 border-b border-border">
        {WEEKDAY_LABELS.map((d) => (
          <div key={d} className="px-2 py-1.5 text-[10px] font-medium text-muted-foreground text-center">
            {d}
          </div>
        ))}
      </div>

      {/* Grid */}
      <div
        className="grid grid-cols-7"
        style={{ gridTemplateRows: `repeat(${rowsNeeded}, minmax(80px, 1fr))` }}
      >
        {visibleCells.map((cell) => {
          const key = toDateKey(cell);
          const isCurrentMonth = cell.getMonth() === month;
          const isToday = key === today;
          const entries = entriesByDate.get(key) ?? [];
          const visible = entries.slice(0, 3);
          const overflow = entries.length - 3;

          return (
            <div
              key={key}
              className={cn(
                'border-r border-b border-border p-1.5 min-h-[80px] flex flex-col gap-0.5',
                !isCurrentMonth && 'opacity-30',
              )}
            >
              {/* Day number */}
              <div className="flex items-center justify-end mb-0.5">
                <span
                  className={cn(
                    'text-xs font-medium w-5 h-5 flex items-center justify-center rounded-full',
                    isToday
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground',
                  )}
                >
                  {cell.getDate()}
                </span>
              </div>

              {/* Entry chips */}
              <div className="flex flex-col gap-0.5 overflow-hidden">
                {visible.map((e) => (
                  <EntryChip key={e.id} entry={e} />
                ))}
                {overflow > 0 && (
                  <span className="text-[9px] text-muted-foreground px-1">
                    +{overflow} more
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Week view
// ---------------------------------------------------------------------------

interface WeekViewProps {
  weekStart: Date;
  entriesByDate: Map<string, CronbanEntry[]>;
  today: string;
}

function WeekView({ weekStart, entriesByDate, today }: WeekViewProps) {
  const days: Date[] = Array.from({ length: 7 }, (_, i) => addDays(weekStart, i));
  const hours = Array.from({ length: 24 }, (_, i) => i);

  return (
    <div className="flex-1 overflow-auto">
      {/* Header row */}
      <div className="sticky top-0 z-10 bg-background grid grid-cols-[48px_repeat(7,1fr)] border-b border-border">
        <div className="h-10" />
        {days.map((day) => {
          const key = toDateKey(day);
          const isToday = key === today;
          return (
            <div key={key} className="flex flex-col items-center justify-center py-1.5 border-l border-border first:border-0">
              <span className="text-[10px] text-muted-foreground">{WEEKDAY_LABELS[day.getDay()]}</span>
              <span
                className={cn(
                  'text-sm font-medium w-7 h-7 flex items-center justify-center rounded-full',
                  isToday ? 'bg-primary text-primary-foreground' : 'text-foreground',
                )}
              >
                {day.getDate()}
              </span>
            </div>
          );
        })}
      </div>

      {/* Hour rows */}
      <div className="grid grid-cols-[48px_repeat(7,1fr)]">
        {hours.map((h) => (
          <>
            {/* Time label */}
            <div
              key={`label-${h}`}
              className="border-b border-border px-1.5 pt-0.5 text-[9px] text-muted-foreground text-right leading-none"
              style={{ height: '3rem' }}
            >
              {h === 0 ? '' : `${h}:00`}
            </div>

            {/* Day cells */}
            {days.map((day) => {
              const key = toDateKey(day);
              const entries = (entriesByDate.get(key) ?? []).filter((e) => {
                const d = parseDate(e.next_fire_at);
                return d ? d.getHours() === h : false;
              });
              return (
                <div
                  key={`${key}-${h}`}
                  className="border-b border-l border-border relative"
                  style={{ height: '3rem' }}
                >
                  {entries.map((e) => (
                    <div key={e.id} className="absolute inset-x-0.5 top-0.5">
                      <EntryChip entry={e} />
                    </div>
                  ))}
                </div>
              );
            })}
          </>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Unscheduled sidebar
// ---------------------------------------------------------------------------

function UnscheduledSidebar({ entries }: { entries: CronbanEntry[] }) {
  if (entries.length === 0) return null;

  return (
    <div className="w-52 shrink-0 border-l border-border flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b border-border">
        <p className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
          <Clock className="h-3 w-3" />
          Unscheduled ({entries.length})
        </p>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
        {entries.map((e) => (
          <div
            key={e.id}
            className={cn(
              'px-2 py-1.5 rounded border text-xs',
              colorChipClasses(e.color),
            )}
          >
            <div className="flex items-center gap-1.5 mb-0.5">
              <span className={cn('h-1.5 w-1.5 rounded-full shrink-0', colorDotClass(e.color))} />
              <span className="font-medium truncate">{e.title}</span>
            </div>
            {e.cron_expression && (
              <p className="text-[9px] font-mono opacity-70 pl-3">{e.cron_expression}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main CalendarView
// ---------------------------------------------------------------------------

export function CalendarView({ projectId }: { projectId?: string }) {
  const [viewMode, setViewMode] = useState<'month' | 'week'>('month');
  const [entries, setEntries] = useState<CronbanEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [wizardOpen, setWizardOpen] = useState(false);

  // Navigation state: for month = {year, month}, for week = weekStart Date
  const now = new Date();
  const [navYear, setNavYear] = useState(now.getFullYear());
  const [navMonth, setNavMonth] = useState(now.getMonth());
  const [navWeekStart, setNavWeekStart] = useState(() => getWeekStart(now));

  const today = toDateKey(now);

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

  // Build a map: YYYY-MM-DD → entries with next_fire_at on that day
  const entriesByDate = useMemo(() => {
    const map = new Map<string, CronbanEntry[]>();
    for (const e of entries) {
      const d = parseDate(e.next_fire_at);
      if (!d) continue;
      const key = toDateKey(d);
      const bucket = map.get(key) ?? [];
      bucket.push(e);
      map.set(key, bucket);
    }
    return map;
  }, [entries]);

  const unscheduled = useMemo(
    () => entries.filter((e) => !e.next_fire_at),
    [entries],
  );

  // Navigation
  const goToday = useCallback(() => {
    const n = new Date();
    setNavYear(n.getFullYear());
    setNavMonth(n.getMonth());
    setNavWeekStart(getWeekStart(n));
  }, []);

  const goBack = useCallback(() => {
    if (viewMode === 'month') {
      setNavMonth((m) => {
        if (m === 0) { setNavYear((y) => y - 1); return 11; }
        return m - 1;
      });
    } else {
      setNavWeekStart((w) => addDays(w, -7));
    }
  }, [viewMode]);

  const goForward = useCallback(() => {
    if (viewMode === 'month') {
      setNavMonth((m) => {
        if (m === 11) { setNavYear((y) => y + 1); return 0; }
        return m + 1;
      });
    } else {
      setNavWeekStart((w) => addDays(w, 7));
    }
  }, [viewMode]);

  const headerLabel = useMemo(() => {
    if (viewMode === 'month') return `${MONTH_NAMES[navMonth]} ${navYear}`;
    const end = addDays(navWeekStart, 6);
    if (navWeekStart.getMonth() === end.getMonth()) {
      return `${MONTH_NAMES[navWeekStart.getMonth()]} ${navWeekStart.getDate()}–${end.getDate()}, ${navWeekStart.getFullYear()}`;
    }
    return `${MONTH_NAMES[navWeekStart.getMonth()]} ${navWeekStart.getDate()} – ${MONTH_NAMES[end.getMonth()]} ${end.getDate()}, ${end.getFullYear()}`;
  }, [viewMode, navYear, navMonth, navWeekStart]);

  const handleCreated = useCallback((entry: CronbanEntry) => {
    setEntries((prev) => [entry, ...prev]);
  }, []);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Top bar */}
      <div className="shrink-0 flex items-center gap-2 px-4 py-2.5 border-b border-border">
        {/* View toggle */}
        <div className="flex rounded-md border border-border overflow-hidden">
          <button
            onClick={() => setViewMode('month')}
            className={cn(
              'flex items-center gap-1 px-2.5 py-1 text-xs font-medium transition-colors',
              viewMode === 'month'
                ? 'bg-muted text-foreground'
                : 'text-muted-foreground hover:bg-muted/40',
            )}
          >
            <CalendarDays className="h-3 w-3" />
            Month
          </button>
          <button
            onClick={() => setViewMode('week')}
            className={cn(
              'flex items-center gap-1 px-2.5 py-1 text-xs font-medium border-l border-border transition-colors',
              viewMode === 'week'
                ? 'bg-muted text-foreground'
                : 'text-muted-foreground hover:bg-muted/40',
            )}
          >
            <Clock className="h-3 w-3" />
            Week
          </button>
        </div>

        {/* Navigation */}
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={goBack} className="h-7 w-7 p-0">
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <span className="text-sm font-medium min-w-[180px] text-center">{headerLabel}</span>
          <Button variant="ghost" size="sm" onClick={goForward} className="h-7 w-7 p-0">
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>

        <Button variant="outline" size="sm" onClick={goToday} className="h-7 text-xs">
          Today
        </Button>

        <div className="flex-1" />

        {/* Loading indicator */}
        {loading && (
          <span className="text-xs text-muted-foreground animate-pulse">Loading…</span>
        )}

        {/* New Entry */}
        <Button
          size="sm"
          onClick={() => setWizardOpen(true)}
          className="h-7 gap-1 text-xs"
        >
          <Plus className="h-3.5 w-3.5" />
          New Entry
        </Button>
      </div>

      {/* Main content + unscheduled sidebar */}
      <div className="flex-1 flex overflow-hidden">
        {/* Calendar body */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {!loading && entries.length === 0 ? (
            <div className="flex-1 flex items-center justify-center text-muted-foreground">
              <div className="text-center space-y-2">
                <CalendarDays className="h-8 w-8 mx-auto opacity-30" />
                <p className="text-sm">No scheduled cron entries yet.</p>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setWizardOpen(true)}
                  className="gap-1"
                >
                  <Plus className="h-3.5 w-3.5" />
                  Create your first entry
                </Button>
              </div>
            </div>
          ) : viewMode === 'month' ? (
            <MonthView
              year={navYear}
              month={navMonth}
              entriesByDate={entriesByDate}
              today={today}
            />
          ) : (
            <WeekView
              weekStart={navWeekStart}
              entriesByDate={entriesByDate}
              today={today}
            />
          )}
        </div>

        {/* Unscheduled sidebar */}
        <UnscheduledSidebar entries={unscheduled} />
      </div>

      {/* Entry wizard */}
      <EntryWizard
        open={wizardOpen}
        onOpenChange={setWizardOpen}
        projectId={projectId}
        initialType="cron"
        onCreated={handleCreated}
      />
    </div>
  );
}
