/**
 * SessionCard — Compact session display for sidebar list.
 */
import { useState, useCallback, useRef, useEffect } from 'react';
import { Bot, Folder, X, Skull, Pencil } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type AgentSession } from '@/services/daemon-api';
import { statusColor, isPulsing, shortPath, timeAgo } from './utils';

export function SessionCard({
  session,
  selected,
  onSelect,
  onDismiss,
  onRename,
  compact = false,
}: {
  session: AgentSession;
  selected: boolean;
  onSelect: () => void;
  onDismiss?: () => void;
  onRename?: (id: string, name: string) => void;
  compact?: boolean;
}) {
  // Inline confirmation: first click arms → second click kills (within 2s)
  const [armed, setArmed] = useState(false);
  const armTimer = useRef<ReturnType<typeof setTimeout>>();

  // Inline rename state
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleKillClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (!onDismiss) return;
      if (armed) {
        clearTimeout(armTimer.current);
        setArmed(false);
        onDismiss();
      } else {
        setArmed(true);
        armTimer.current = setTimeout(() => setArmed(false), 2000);
      }
    },
    [armed, onDismiss],
  );

  const handleRenameStart = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      setEditValue(session.name || '');
      setEditing(true);
    },
    [session.name],
  );

  const handleRenameSave = useCallback(() => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== session.name && onRename) {
      onRename(session.id, trimmed);
    }
    setEditing(false);
  }, [editValue, session.name, session.id, onRename]);

  const handleRenameKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') handleRenameSave();
      if (e.key === 'Escape') setEditing(false);
    },
    [handleRenameSave],
  );

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  useEffect(() => () => clearTimeout(armTimer.current), []);

  const statusDot = cn(
    'h-2 w-2 rounded-full shrink-0',
    isPulsing(session.status)
      ? 'bg-amber-400 animate-pulse'
      : session.status === 'idle'
        ? 'bg-emerald-400'
        : 'bg-muted-foreground/40',
  );

  if (compact) {
    return (
      <div className="group relative">
        {editing ? (
          <div
            className={cn(
              'w-full rounded px-2.5 py-1.5 flex items-center gap-2',
              'bg-blue-500/10 border border-blue-500/40',
            )}
          >
            <span className={statusDot} />
            <input
              ref={inputRef}
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onKeyDown={handleRenameKeyDown}
              onBlur={handleRenameSave}
              className="flex-1 text-xs font-mono bg-transparent outline-none text-foreground min-w-0"
            />
          </div>
        ) : (
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
            <span className={statusDot} />
            <span className="text-xs font-mono truncate flex-1">
              {session.name || `pid:${session.pid}`}
            </span>
            <span className="text-[9px] text-muted-foreground shrink-0 tabular-nums">
              {timeAgo(session.last_activity)}
            </span>
          </button>
        )}

        {/* Pencil — rename, shows on hover when not editing */}
        {onRename && !editing && (
          <button
            onClick={handleRenameStart}
            className={cn(
              'absolute top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100',
              'transition-all p-0.5 rounded text-muted-foreground hover:text-foreground hover:bg-muted/60',
              onDismiss ? 'right-7' : 'right-1',
            )}
            title="Rename session"
          >
            <Pencil className="h-2.5 w-2.5" />
          </button>
        )}

        {/* Kill button */}
        {onDismiss && !editing && (
          <button
            onClick={handleKillClick}
            className={cn(
              'absolute right-1 top-1/2 -translate-y-1/2 transition-all p-0.5 rounded text-[9px] flex items-center gap-0.5',
              armed
                ? 'opacity-100 bg-destructive/20 text-destructive border border-destructive/40 px-1'
                : 'opacity-0 group-hover:opacity-100 text-muted-foreground hover:bg-destructive/10 hover:text-destructive',
            )}
            title={armed ? 'Click again to kill' : 'Kill session'}
          >
            {armed ? (
              <>
                <Skull className="h-2.5 w-2.5" />
                <span>Sure?</span>
              </>
            ) : (
              <X className="h-3 w-3" />
            )}
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
