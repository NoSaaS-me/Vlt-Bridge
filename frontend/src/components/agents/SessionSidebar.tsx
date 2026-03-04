/**
 * SessionSidebar — Left panel listing Claude Code sessions for the current project.
 *
 * Sessions are scoped to the selected project via project_id filtering.
 * The "+" button opens a popover with options to find existing sessions
 * or spawn a new one.
 */
import { useState, useEffect } from 'react';
import { Bot, Plus, WifiOff, Search, Sparkles } from 'lucide-react';
import { type AgentSession } from '@/services/daemon-api';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { SessionCard } from './SessionCard';
import { FindSessionDialog } from './FindSessionDialog';
import { NewSessionDialog } from './NewSessionDialog';

export function SessionSidebar({
  sessions,
  isLoading,
  daemonOnline,
  focusedSessionId,
  currentProjectId,
  spawnCwd,
  onSelectSession,
  onSelectDiscoveredSession,
  onDismissSession,
  onStartFresh,
  onResumeSession,
  onRenameSession,
  onRefresh,
}: {
  sessions: AgentSession[];
  isLoading: boolean;
  daemonOnline: boolean;
  focusedSessionId: string | null;
  currentProjectId: string | null;
  spawnCwd: string;
  onSelectSession: (id: string) => void;
  onSelectDiscoveredSession?: (session: AgentSession) => void;
  onDismissSession?: (id: string) => void;
  onStartFresh?: (prompt: string) => void;
  onResumeSession?: (session: AgentSession) => void;
  onRenameSession?: (id: string, name: string) => void;
  onRefresh?: () => void;
}) {
  const [popoverOpen, setPopoverOpen] = useState(false);
  const [findDialogOpen, setFindDialogOpen] = useState(false);
  const [newSessionOpen, setNewSessionOpen] = useState(false);
  const [showHistory, setShowHistory] = useState(() => {
    try { return localStorage.getItem('vlt:livechat:showHistory') === 'true'; } catch { return false; }
  });

  // Sync with Settings page changes via storage event
  useEffect(() => {
    const handler = (e: StorageEvent) => {
      if (e.key === 'vlt:livechat:showHistory') setShowHistory(e.newValue === 'true');
    };
    window.addEventListener('storage', handler);
    return () => window.removeEventListener('storage', handler);
  }, []);

  // All non-dead sessions sorted by activity.
  const allSessions = sessions
    .filter((s) => s.status !== 'dead')
    .sort((a, b) => new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime());

  // When history is hidden, only show actively running sessions.
  const liveSessions = showHistory
    ? allSessions
    : allSessions.filter((s) => s.status === 'thinking' || s.status === 'executing');

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
        <div className="flex items-center gap-1.5">
          {onStartFresh && daemonOnline && currentProjectId && (
            <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-5 w-5 p-0"
                  title="Add session"
                >
                  <Plus className="h-3 w-3" />
                </Button>
              </PopoverTrigger>
              <PopoverContent
                align="end"
                className="w-44 p-1"
                sideOffset={4}
              >
                <button
                  className="w-full flex items-center gap-2 rounded px-2 py-1.5 text-xs hover:bg-muted/80 transition-colors"
                  onClick={() => {
                    setPopoverOpen(false);
                    setNewSessionOpen(true);
                  }}
                >
                  <Sparkles className="h-3 w-3 text-muted-foreground" />
                  New session
                </button>
                <button
                  className="w-full flex items-center gap-2 rounded px-2 py-1.5 text-xs hover:bg-muted/80 transition-colors"
                  onClick={() => {
                    setPopoverOpen(false);
                    setFindDialogOpen(true);
                  }}
                >
                  <Search className="h-3 w-3 text-muted-foreground" />
                  Find & add session
                </button>
              </PopoverContent>
            </Popover>
          )}
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

      {/* Session list */}
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
        ) : liveSessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 px-3 text-center">
            <Bot className="h-6 w-6 text-muted-foreground/30 mb-2" />
            {!showHistory && allSessions.length > 0 ? (
              <>
                <p className="text-[10px] text-muted-foreground">No active sessions</p>
                <p className="mt-1 text-[9px] text-muted-foreground/60">
                  Enable history in Settings → Agents
                </p>
              </>
            ) : (
              <>
                <p className="text-[10px] text-muted-foreground">No sessions for this project</p>
                <p className="mt-1 text-[9px] text-muted-foreground/60">
                  Use + to find or start a session
                </p>
              </>
            )}
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {liveSessions.map((s) => (
              <SessionCard
                key={s.id}
                session={s}
                selected={s.id === focusedSessionId}
                onSelect={() =>
                  s.source === 'relay'
                    ? onSelectSession(s.id)
                    : onSelectDiscoveredSession
                      ? onSelectDiscoveredSession(s)
                      : onSelectSession(s.id)
                }
                onDismiss={onDismissSession ? () => onDismissSession(s.id) : undefined}
                onRename={onRenameSession}
                compact
              />
            ))}
          </div>
        )}
      </div>

      {/* Find dialog */}
      {currentProjectId && (
        <FindSessionDialog
          open={findDialogOpen}
          onOpenChange={setFindDialogOpen}
          currentProjectId={currentProjectId}
          onAssigned={() => onRefresh?.()}
        />
      )}

      {/* New session dialog */}
      {currentProjectId && onStartFresh && onResumeSession && (
        <NewSessionDialog
          open={newSessionOpen}
          onOpenChange={setNewSessionOpen}
          projectId={currentProjectId}
          spawnCwd={spawnCwd}
          onStartFresh={onStartFresh}
          onResume={onResumeSession}
        />
      )}
    </div>
  );
}
