/**
 * AgentsPage — Niri-style terminal compositor for Claude Code sessions.
 *
 * Layout:
 *   Left nav (48px) — icons (Agents, Cronban, Connectors)
 *   Session sidebar   — relay + discovery session list
 *   Compositor        — horizontal scrolling terminal panes
 *   Events ticker     — compact event strip at bottom
 */
import { useState, useCallback, useMemo } from 'react';
import { Bot, Clock, Plug, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type AgentSession, dismissSession, spawnSession, renameSession } from '@/services/daemon-api';
import { useSessionPolling } from '@/hooks/useSessionPolling';
import { useProjectContext } from '@/contexts/ProjectContext';
import { SessionSidebar } from '@/components/agents/SessionSidebar';
import { TerminalCompositor } from '@/components/agents/TerminalCompositor';
import { EventsTicker } from '@/components/agents/EventsTicker';
import { LiveSessionPanel } from '@/components/agents/LiveSessionPanel';
import { CronbanView as CronbanViewReal } from '@/components/cronban/CronbanView';
import { ConnectorsPage } from '@/pages/ConnectorsPage';

// ---------------------------------------------------------------------------
// Nav
// ---------------------------------------------------------------------------

type NavSection = 'agents' | 'cronban' | 'connectors';

const NAV_ITEMS: { id: NavSection; icon: React.ElementType; label: string }[] = [
  { id: 'agents', icon: Bot, label: 'Agents' },
  { id: 'cronban', icon: Clock, label: 'Cronban' },
  { id: 'connectors', icon: Plug, label: 'Connectors' },
];

// ---------------------------------------------------------------------------
// Placeholder sections (unchanged from original)
// ---------------------------------------------------------------------------

function CronbanView({ projectId, onViewSession }: { projectId?: string; onViewSession?: (sessionId: string) => void }) {
  return <CronbanViewReal projectId={projectId} onViewSession={onViewSession} />;
}

function ConnectorsView() {
  return <ConnectorsPage />;
}

// ---------------------------------------------------------------------------
// Agents Compositor Layout
// ---------------------------------------------------------------------------

function AgentsCompositorView({
  polling,
  focusedSessionId,
  liveSession,
  selectedProjectId,
  spawnCwd,
  onSelectSession,
  onSelectDiscoveredSession,
  onDismissSession,
  onCloseLiveSession,
  onStartFresh,
  onResumeSession,
  onRenameSession,
}: {
  polling: ReturnType<typeof useSessionPolling>;
  focusedSessionId: string | null;
  liveSession: AgentSession | null;
  selectedProjectId: string | null;
  spawnCwd: string;
  onSelectSession: (id: string) => void;
  onSelectDiscoveredSession: (session: AgentSession) => void;
  onDismissSession: (id: string) => void;
  onCloseLiveSession: () => void;
  onStartFresh: (prompt: string) => void;
  onResumeSession: (session: AgentSession) => void;
  onRenameSession: (id: string, name: string) => void;
}) {
  return (
    <div className="h-full flex">
      {/* Session sidebar */}
      <div className="w-52 shrink-0">
        <SessionSidebar
          sessions={polling.sessions}
          isLoading={polling.isLoading}
          daemonOnline={polling.daemonOnline}
          focusedSessionId={liveSession?.id ?? focusedSessionId}
          currentProjectId={selectedProjectId}
          spawnCwd={spawnCwd}
          onSelectSession={onSelectSession}
          onSelectDiscoveredSession={onSelectDiscoveredSession}
          onDismissSession={onDismissSession}
          onStartFresh={onStartFresh}
          onResumeSession={onResumeSession}
          onRenameSession={onRenameSession}
          onRefresh={polling.refresh}
        />
      </div>

      {/* Main content: LiveSessionPanel (hook sessions) or TerminalCompositor (relay) */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-hidden">
          {liveSession ? (
            <LiveSessionPanel
              session={liveSession}
              onClose={onCloseLiveSession}
            />
          ) : (
            <TerminalCompositor
              sessions={polling.sessions}
              focusedSessionId={focusedSessionId}
              onFocusSession={onSelectSession}
            />
          )}
        </div>
        <EventsTicker events={polling.events} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main AgentsPage
// ---------------------------------------------------------------------------

export function AgentsPage() {
  const [activeSection, setActiveSection] = useState<NavSection>('agents');
  const { selectedProjectId } = useProjectContext();
  const polling = useSessionPolling(selectedProjectId);
  const [focusedSessionId, setFocusedSessionId] = useState<string | null>(null);
  const [liveSession, setLiveSession] = useState<AgentSession | null>(null);

  // Derive a CWD for spawning: use the most recent session's CWD for this project,
  // or fall back to a sensible default.
  const spawnCwd = useMemo(() => {
    const projectSession = polling.sessions.find((s) => s.cwd);
    return projectSession?.cwd ?? '/mnt/sda1/Projects/00Tooling/Vlt-Bridge';
  }, [polling.sessions]);

  const handleSelectSession = useCallback((id: string) => {
    setLiveSession(null); // close live panel when selecting a relay session
    setFocusedSessionId(id);
  }, []);

  const handleSelectDiscoveredSession = useCallback((session: AgentSession) => {
    setLiveSession(session);
  }, []);

  const handleDismissSession = useCallback((id: string) => {
    dismissSession(id).then(() => polling.refresh()).catch(console.error);
  }, [polling]);

  const handleStartFresh = useCallback(
    (prompt: string) => {
      spawnSession(spawnCwd, { prompt, mode: 'relay' })
        .then((result) => {
          setTimeout(() => polling.refresh(), 2000);
          if (result.session_id) {
            setFocusedSessionId(result.session_id);
          }
        })
        .catch((err) => console.error('Spawn failed:', err));
    },
    [polling, spawnCwd],
  );

  const handleResumeSession = useCallback((session: AgentSession) => {
    setLiveSession(session);
  }, []);

  const handleRenameSession = useCallback((id: string, name: string) => {
    renameSession(id, name).then(() => polling.refresh()).catch(console.error);
  }, [polling]);

  return (
    <div className="h-full flex">
      {/* Left nav sidebar */}
      <aside className="w-16 border-r border-border flex flex-col items-center pt-3 pb-2 gap-0.5 shrink-0">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveSection(item.id)}
            className={cn(
              'w-14 rounded flex flex-col items-center justify-center py-1.5 gap-0.5 transition-colors duration-150',
              activeSection === item.id
                ? 'bg-blue-500/20 text-blue-400'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
            )}
          >
            <item.icon className="h-4 w-4" />
            <span className="text-[9px] font-medium uppercase tracking-wider leading-none">
              {item.label}
            </span>
          </button>
        ))}
        <div className="flex-1" />
        <button
          onClick={polling.refresh}
          className="w-14 rounded flex flex-col items-center justify-center py-1.5 gap-0.5 text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
          title={`Last refresh: ${polling.lastRefresh.toLocaleTimeString()}`}
        >
          <RefreshCw className="h-3.5 w-3.5" />
          <span className="text-[9px] font-medium uppercase tracking-wider leading-none">Refresh</span>
        </button>
      </aside>

      {/* Content */}
      <div className={cn('flex-1', activeSection === 'connectors' ? 'overflow-y-auto' : 'overflow-hidden')}>
        {activeSection === 'agents' && (
          <AgentsCompositorView
            polling={polling}
            focusedSessionId={focusedSessionId}
            liveSession={liveSession}
            selectedProjectId={selectedProjectId}
            spawnCwd={spawnCwd}
            onSelectSession={handleSelectSession}
            onSelectDiscoveredSession={handleSelectDiscoveredSession}
            onDismissSession={handleDismissSession}
            onCloseLiveSession={() => setLiveSession(null)}
            onStartFresh={handleStartFresh}
            onResumeSession={handleResumeSession}
            onRenameSession={handleRenameSession}
          />
        )}
        {activeSection === 'cronban' && (
          <CronbanView
            projectId={selectedProjectId ?? undefined}
            onViewSession={(sessionId) => {
              // Switch to agents tab and open the session
              const session = polling.sessions.find((s) => s.id === sessionId);
              if (session?.source === 'relay') {
                // Relay sessions render in the TerminalCompositor (xterm.js)
                setLiveSession(null);
                setFocusedSessionId(sessionId);
              } else if (session) {
                // Non-relay sessions open in the LiveSessionPanel (JSONL view)
                setLiveSession(session);
              } else {
                // Session might not be in current polling data — create minimal ref
                // Default to LiveSessionPanel; it will auto-upgrade if relay
                setLiveSession({
                  id: sessionId,
                  project_id: selectedProjectId,
                  name: sessionId.slice(0, 8),
                  cwd: '',
                  status: 'idle',
                  model: null,
                  ctx_pct: null,
                  pid: 0,
                  bypass_perms: false,
                  source: 'managed',
                  is_cronban_helper: false,
                  created_at: new Date().toISOString(),
                  last_activity: new Date().toISOString(),
                });
              }
              setActiveSection('agents');
            }}
          />
        )}
        {activeSection === 'connectors' && <ConnectorsView />}
      </div>
    </div>
  );
}
