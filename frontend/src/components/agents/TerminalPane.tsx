/**
 * TerminalPane — Single terminal session with header/footer chrome.
 *
 * ┌─ header: session name │ status │ model │ ctx% ─┐
 * │                                                 │
 * │  xterm.js terminal                              │
 * │                                                 │
 * └─ footer: cwd path ─────────────────────────────┘
 */
import { useRef } from 'react';
import { Terminal, Wifi, WifiOff } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type AgentSession } from '@/services/daemon-api';
import { statusColor, isPulsing, shortPath } from './utils';
import { useTerminalSession } from '@/hooks/useTerminalSession';

export function TerminalPane({
  session,
  isVisible,
  isFocused,
  onFocus,
}: {
  session: AgentSession;
  isVisible: boolean;
  isFocused: boolean;
  onFocus: () => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  const { isConnected } = useTerminalSession({
    sessionId: session.id,
    containerRef,
    isVisible,
    isFocused,
  });

  return (
    <div
      className={cn(
        'h-full flex flex-col rounded-lg border overflow-hidden transition-all duration-200',
        isFocused
          ? 'border-blue-500/50 ring-1 ring-blue-500/20 shadow-lg shadow-blue-500/5'
          : 'border-border/60 hover:border-border',
      )}
      onClick={onFocus}
    >
      {/* Header */}
      <div className="px-3 py-2 flex items-center justify-between shrink-0 bg-background/80 backdrop-blur-sm border-b border-border/50">
        <div className="flex items-center gap-2 min-w-0">
          <Terminal className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          <span className="text-xs font-medium truncate">
            {session.name || session.id.slice(0, 8)}
          </span>
          <div className={cn(
            'flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border font-medium shrink-0',
            statusColor(session.status),
          )}>
            {isPulsing(session.status) && (
              <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
            )}
            {session.status}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {session.model && (
            <span className="text-[10px] text-muted-foreground font-mono hidden sm:inline">
              {session.model.split('/').pop()}
            </span>
          )}
          {session.ctx_pct != null && (
            <div className="flex items-center gap-1">
              <div className="h-1 w-10 rounded-full bg-muted overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-500',
                    session.ctx_pct > 80 ? 'bg-red-500' :
                    session.ctx_pct > 60 ? 'bg-amber-500' : 'bg-blue-500',
                  )}
                  style={{ width: `${Math.min(100, session.ctx_pct)}%` }}
                />
              </div>
              <span className="text-[9px] text-muted-foreground tabular-nums">
                {Math.round(session.ctx_pct)}%
              </span>
            </div>
          )}
          {isConnected ? (
            <Wifi className="h-3 w-3 text-emerald-400" />
          ) : (
            <WifiOff className="h-3 w-3 text-muted-foreground/40" />
          )}
        </div>
      </div>

      {/* Terminal — overflow-x:auto so wide content scrolls rather than wraps */}
      <div className="flex-1 relative overflow-x-auto overflow-y-hidden bg-[#0a0f1a]">
        <div ref={containerRef} className="absolute inset-0 p-1" />
      </div>

      {/* Footer */}
      <div className="px-3 py-1 flex items-center gap-2 text-[10px] text-muted-foreground border-t border-border/30 bg-background/60 shrink-0">
        <span className="font-mono truncate">{shortPath(session.cwd)}</span>
        <span className="ml-auto shrink-0 font-mono">⬡ {session.source}</span>
      </div>
    </div>
  );
}
