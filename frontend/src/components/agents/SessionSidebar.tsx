/**
 * SessionSidebar — Left panel listing relay + discovery sessions.
 *
 * Relay sessions on top (sorted by activity), discovery sessions below in
 * a collapsible section. Clicking a session scrolls the compositor to it.
 */
import { useState } from 'react';
import { Bot, ChevronDown, ChevronRight, WifiOff, Search } from 'lucide-react';
import { type AgentSession } from '@/services/daemon-api';
import { Separator } from '@/components/ui/separator';
import { SessionCard } from './SessionCard';

export function SessionSidebar({
  sessions,
  isLoading,
  daemonOnline,
  focusedSessionId,
  onSelectSession,
  onSelectDiscoveredSession,
}: {
  sessions: AgentSession[];
  isLoading: boolean;
  daemonOnline: boolean;
  focusedSessionId: string | null;
  onSelectSession: (id: string) => void;
  onSelectDiscoveredSession?: (session: AgentSession) => void;
}) {
  const [discoveryOpen, setDiscoveryOpen] = useState(false);

  const relaySessions = sessions
    .filter((s) => s.source === 'relay' && s.status !== 'dead')
    .sort((a, b) => new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime());

  const discoverySessions = sessions
    .filter((s) => s.source !== 'relay' && s.status !== 'dead')
    .sort((a, b) => new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime());

  return (
    <div className="h-full flex flex-col bg-background/50 border-r border-border">
      {/* Header */}
      <div className="px-3 pt-3 pb-2 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <Bot className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Sessions
          </span>
        </div>
        <div className="flex items-center gap-2">
          {daemonOnline ? (
            <span className="flex items-center gap-1 text-[10px] text-emerald-400">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              live
            </span>
          ) : (
            <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
              <WifiOff className="h-3 w-3" />
            </span>
          )}
        </div>
      </div>
      <Separator />

      {/* Session lists */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-3 space-y-2">
            {[0, 1].map((i) => (
              <div key={i} className="h-10 rounded border border-border bg-muted/30 animate-pulse" />
            ))}
          </div>
        ) : !daemonOnline ? (
          <div className="flex flex-col items-center justify-center py-10 px-3 text-center">
            <WifiOff className="h-6 w-6 text-muted-foreground/30 mb-2" />
            <p className="text-[10px] text-muted-foreground">Daemon offline</p>
            <code className="mt-1 text-[9px] font-mono bg-muted px-1.5 py-0.5 rounded text-foreground/60">
              vlt daemon start
            </code>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {/* Relay sessions — always visible */}
            {relaySessions.length === 0 ? (
              <div className="py-6 text-center">
                <p className="text-[10px] text-muted-foreground/60">No relay sessions</p>
                <code className="mt-1 text-[9px] font-mono bg-muted px-1.5 py-0.5 rounded text-foreground/60 inline-block">
                  vlt session-relay
                </code>
              </div>
            ) : (
              relaySessions.map((s) => (
                <SessionCard
                  key={s.id}
                  session={s}
                  selected={s.id === focusedSessionId}
                  onSelect={() => onSelectSession(s.id)}
                  compact
                />
              ))
            )}

            {/* Discovery sessions — collapsible */}
            {discoverySessions.length > 0 && (
              <>
                <Separator className="my-2" />
                <button
                  onClick={() => setDiscoveryOpen(!discoveryOpen)}
                  className="w-full flex items-center gap-1.5 px-2 py-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                >
                  {discoveryOpen ? (
                    <ChevronDown className="h-3 w-3" />
                  ) : (
                    <ChevronRight className="h-3 w-3" />
                  )}
                  <Search className="h-3 w-3" />
                  <span className="font-medium uppercase tracking-wider">
                    Discovered ({discoverySessions.length})
                  </span>
                </button>
                {discoveryOpen && (
                  <div className="space-y-1 mt-1">
                    {discoverySessions.map((s) => (
                      <SessionCard
                        key={s.id}
                        session={s}
                        selected={s.id === focusedSessionId}
                        onSelect={() =>
                          onSelectDiscoveredSession
                            ? onSelectDiscoveredSession(s)
                            : onSelectSession(s.id)
                        }
                        compact
                      />
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
