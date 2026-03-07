/**
 * HelperSessionsPanel — shows the pool of cronban helper sessions.
 *
 * Helper sessions are persistent Claude Code SDK subprocesses managed by
 * the cronban evaluator. They are reused (LIFO) for gate evaluations and
 * card fire dispatches when `use_helper_session=true`.
 *
 * Shows: status, project, model, last activity, and a kill button.
 */
import { useState, useEffect, useCallback } from 'react';
import { Trash2, RefreshCw, Bot, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { type AgentSession, listHelperSessions, terminateSession } from '@/services/daemon-api';

const STATUS_COLORS: Record<string, string> = {
  idle:      'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  thinking:  'bg-amber-500/20 text-amber-400 border-amber-500/30',
  executing: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  dead:      'bg-muted/20 text-muted-foreground border-border',
};

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.floor(m / 60)}h ago`;
}

export function HelperSessionsPanel() {
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [showDead, setShowDead] = useState(false);
  const [killing, setKilling] = useState<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    try {
      const data = await listHelperSessions(showDead);
      setSessions(data);
    } catch (e) {
      console.error('Helper sessions fetch failed:', e);
    } finally {
      setLoading(false);
    }
  }, [showDead]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  const handleKill = async (sessionId: string) => {
    setKilling((prev) => new Set(prev).add(sessionId));
    try {
      await terminateSession(sessionId);
      await refresh();
    } catch (e) {
      console.error('Kill failed:', e);
    } finally {
      setKilling((prev) => {
        const next = new Set(prev);
        next.delete(sessionId);
        return next;
      });
    }
  };

  const alive = sessions.filter((s) => s.status !== 'dead');
  const busy  = alive.filter((s) => s.status !== 'idle');

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          <Bot className="h-4 w-4 text-purple-400" />
          <h2 className="text-sm font-semibold">Helper Sessions</h2>
          {alive.length > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
              {alive.length} alive · {busy.length} busy
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-[10px] text-muted-foreground cursor-pointer">
            <input
              type="checkbox"
              checked={showDead}
              onChange={(e) => setShowDead(e.target.checked)}
              className="h-3 w-3"
            />
            Show terminated
          </label>
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0"
            onClick={refresh}
            title="Refresh"
          >
            <RefreshCw className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {loading && (
          <div className="flex items-center justify-center py-12 text-muted-foreground gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm">Loading…</span>
          </div>
        )}

        {!loading && sessions.length === 0 && (
          <div className="text-center py-12 space-y-2">
            <Bot className="h-8 w-8 mx-auto text-muted-foreground/40" />
            <p className="text-sm text-muted-foreground">No helper sessions yet.</p>
            <p className="text-[10px] text-muted-foreground/60 max-w-xs mx-auto">
              Helper sessions are created automatically when gate evaluations run
              or when a card fires with "Helper Session" mode.
            </p>
          </div>
        )}

        {sessions.map((s) => (
          <div
            key={s.id}
            className={cn(
              'rounded-lg border bg-muted/10 p-3 space-y-2 transition-opacity',
              s.status === 'dead' && 'opacity-50',
            )}
          >
            {/* Top row: status + id + kill */}
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 min-w-0">
                <span className={cn(
                  'shrink-0 text-[10px] px-1.5 py-0.5 rounded border font-medium',
                  STATUS_COLORS[s.status] ?? STATUS_COLORS.dead,
                )}>
                  {s.status}
                </span>
                <span className="text-[10px] font-mono text-muted-foreground truncate">
                  {s.id.slice(0, 16)}…
                </span>
              </div>
              {s.status !== 'dead' && (
                <button
                  onClick={() => handleKill(s.id)}
                  disabled={killing.has(s.id)}
                  className="shrink-0 p-0.5 text-muted-foreground hover:text-destructive transition-colors disabled:opacity-40"
                  title="Terminate helper session"
                >
                  {killing.has(s.id)
                    ? <Loader2 className="h-3 w-3 animate-spin" />
                    : <Trash2 className="h-3 w-3" />}
                </button>
              )}
            </div>

            {/* Details row */}
            <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-[10px] text-muted-foreground">
              {s.project_id && (
                <span>
                  <span className="opacity-60">project </span>
                  <span className="font-mono">{s.project_id.slice(0, 20)}</span>
                </span>
              )}
              {s.model && (
                <span>
                  <span className="opacity-60">model </span>
                  <span>{s.model.replace('claude-', '').replace('-20251001', '')}</span>
                </span>
              )}
              {s.cwd && (
                <span className="truncate max-w-[200px]">
                  <span className="opacity-60">cwd </span>
                  <span className="font-mono">{s.cwd}</span>
                </span>
              )}
              {s.last_activity && (
                <span className="ml-auto opacity-60">{relativeTime(s.last_activity)}</span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Footer info */}
      <div className="px-4 py-2 border-t border-border shrink-0">
        <p className="text-[9px] text-muted-foreground/50">
          Helper sessions are reused LIFO — idle ones are picked first, new ones spawned when all busy.
          They maintain project context across gate evaluations via <code>--resume</code>.
        </p>
      </div>
    </div>
  );
}
