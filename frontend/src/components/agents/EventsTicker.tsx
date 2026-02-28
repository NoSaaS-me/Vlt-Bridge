/**
 * EventsTicker — Compact horizontal event strip at the bottom of the compositor.
 *
 * Replaces the old full-size Events quadrant. Shows recent hook events
 * in a scrolling horizontal list with auto-scroll to latest.
 */
import { useRef, useEffect } from 'react';
import { Activity } from 'lucide-react';
import { cn } from '@/lib/utils';
import { parseDaemonTs, type HookEvent } from '@/services/daemon-api';
import { EVENT_LABEL, shortPath } from './utils';

export function EventsTicker({ events }: { events: HookEvent[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to right (latest events) when new events arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollLeft = scrollRef.current.scrollWidth;
    }
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 border-t border-border/50 shrink-0">
        <Activity className="h-3 w-3 text-muted-foreground/30" />
        <span className="text-[10px] text-muted-foreground/40">No events</span>
      </div>
    );
  }

  // Show last 30 events in ticker
  const recent = events.slice(-30);

  return (
    <div className="flex items-center gap-2 border-t border-border/50 shrink-0 overflow-hidden">
      <div className="px-2 py-1.5 shrink-0 border-r border-border/30">
        <Activity className="h-3 w-3 text-muted-foreground/50" />
      </div>
      <div
        ref={scrollRef}
        className="flex-1 flex items-center gap-1.5 overflow-x-auto py-1 pr-2"
        style={{ scrollbarWidth: 'none' }}
      >
        {recent.map((ev) => {
          const d = parseDaemonTs(ev.ts);
          const t = d.toLocaleTimeString('en', {
            hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
          });
          const label = EVENT_LABEL[ev.event] ?? ev.event.toLowerCase().replace('_', '-');
          return (
            <div
              key={ev.id}
              className="flex items-center gap-1 shrink-0 text-[9px] font-mono"
              title={`${ev.event} at ${t} — ${ev.cwd || ev.session_id}`}
            >
              <span className="text-muted-foreground/40 tabular-nums">{t}</span>
              <span className={cn(
                'px-1 rounded font-bold uppercase',
                ev.event === 'UserPromptSubmit' ? 'bg-blue-500/20 text-blue-400' :
                ev.event === 'PostToolUse' ? 'bg-amber-500/20 text-amber-400' :
                ev.event === 'Stop' ? 'bg-red-500/20 text-red-400' :
                ev.event === 'SessionStart' ? 'bg-emerald-500/20 text-emerald-400' :
                'bg-muted text-muted-foreground',
              )}>
                {label}
              </span>
              <span className="text-muted-foreground/40">
                {ev.cwd ? shortPath(ev.cwd) : ev.session_id.slice(0, 6)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
