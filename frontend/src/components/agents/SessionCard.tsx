/**
 * SessionCard — Compact session display for sidebar list.
 */
import { Bot, Folder, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type AgentSession } from '@/services/daemon-api';
import { statusColor, isPulsing, shortPath, timeAgo } from './utils';

export function SessionCard({
  session,
  selected,
  onSelect,
  onDismiss,
  compact = false,
}: {
  session: AgentSession;
  selected: boolean;
  onSelect: () => void;
  onDismiss?: () => void;
  compact?: boolean;
}) {
  if (compact) {
    return (
      <div className="group relative">
        <button
          onClick={onSelect}
          className={cn(
            'w-full text-left rounded px-2.5 py-1.5 flex items-center gap-2 transition-all duration-150',
            selected
              ? 'bg-blue-500/15 text-blue-300 border border-blue-500/40'
              : 'hover:bg-muted/40 border border-transparent',
            session.status === 'dead' && 'opacity-40',
          )}
        >
          <span className={cn(
            'h-2 w-2 rounded-full shrink-0',
            isPulsing(session.status) ? 'bg-amber-400 animate-pulse' :
            session.status === 'idle' ? 'bg-emerald-400' : 'bg-muted-foreground/40',
          )} />
          <span className="text-xs font-mono truncate flex-1">
            {session.name || `pid:${session.pid}`}
          </span>
          <span className="text-[9px] text-muted-foreground shrink-0 tabular-nums">
            {timeAgo(session.last_activity)}
          </span>
        </button>
        {onDismiss && (
          <button
            onClick={(e) => { e.stopPropagation(); onDismiss(); }}
            className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:bg-destructive/20 hover:text-destructive text-muted-foreground"
            title="Dismiss session"
          >
            <X className="h-3 w-3" />
          </button>
        )}
      </div>
    );
  }

  return (
    <button
      onClick={onSelect}
      className={cn(
        'w-full text-left rounded-md border p-3 space-y-2 transition-all duration-200',
        selected
          ? 'border-blue-500/60 bg-blue-500/8 ring-1 ring-blue-500/30'
          : 'border-border hover:border-border/80 hover:bg-muted/30',
        session.status === 'dead' && 'opacity-50',
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Bot className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="text-sm font-medium truncate">{session.name || `pid:${session.pid}`}</span>
        </div>
        <div className={cn(
          'flex items-center gap-1 text-xs px-1.5 py-0.5 rounded border font-medium shrink-0',
          statusColor(session.status),
        )}>
          {isPulsing(session.status) && (
            <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
          )}
          {session.status.charAt(0).toUpperCase() + session.status.slice(1)}
        </div>
      </div>

      <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-mono">
        <Folder className="h-3 w-3 shrink-0" />
        <span className="truncate" title={session.cwd}>{shortPath(session.cwd)}</span>
      </div>

      <div className="flex items-center gap-2">
        {session.model && (
          <span className="text-[10px] text-muted-foreground font-mono truncate flex-1">
            {session.model.split('/').pop()}
          </span>
        )}
        {session.ctx_pct != null && (
          <div className="flex items-center gap-1 shrink-0">
            <div className="h-1 w-16 rounded-full bg-muted overflow-hidden">
              <div
                className={cn(
                  'h-full rounded-full transition-all duration-500',
                  session.ctx_pct > 80 ? 'bg-red-500' :
                  session.ctx_pct > 60 ? 'bg-amber-500' : 'bg-blue-500',
                )}
                style={{ width: `${Math.min(100, session.ctx_pct)}%` }}
              />
            </div>
            <span className="text-[10px] text-muted-foreground tabular-nums">
              {Math.round(session.ctx_pct)}%
            </span>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span className="font-mono">⬡ {session.source}</span>
        <span>{timeAgo(session.last_activity)}</span>
      </div>
    </button>
  );
}
